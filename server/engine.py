"""Continuous-batching driver around one DeepInfer LLMEngine (single GPU, DP=1), for the OpenAI server.

DeepInfer has no async/online entry point; its offline `LLM.generate` is `add_request` for every prompt and then
`step()` until `has_unfinished_requests()` is False, followed by one `flush_inflight_reqs()` (with the zero-overhead
scheduler, ZOS, `step()` returns the outputs of the previous step, so the last step's outputs are only released by the
flush). `LLMEngine.add_request` / `abort_request` only touch the C++ scheduler's queues and may be called between any
two `step()` calls; requests added while others decode are prefilled in the next schedule. So the clean hook is to own
the engine on one thread and replace `LLM._run_engine` by a loop that drains an inbox between steps:

    loop:
        apply queued add / abort commands          (engine.add_request / engine.abort_request)
        if engine.has_unfinished_requests():  outs = engine.step()
        elif steps since the last flush:      outs = engine.flush_inflight_reqs()
        else:                                 block on the inbox
        clone each RequestOutput (the pool slot is recycled by the next step) and route it by request_id

Every DeepInfer call happens on that one thread. HTTP handler threads only talk to it through queues. DeepInfer's
source is not modified; the LLM is built exactly like tools/run_deepinfer.py (shims + runtime patches).
"""

from __future__ import annotations

import itertools
import logging
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("di_server.engine")


def _prof_rng(name: str):
    try:
        import sm120_prof
        return sm120_prof.rng(name)
    except ImportError:
        import contextlib
        return contextlib.nullcontext()


def _set_prof_active(on: bool) -> None:
    try:
        import sm120_prof
        sm120_prof.ACTIVE = on
    except ImportError:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOOLS = os.path.join(ROOT, "tools")


@dataclass
class GenRequest:
    """One generation request as the engine thread sees it."""

    prompt_ids: list[int]
    max_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    seed: int | None = None
    ignore_eos: bool = False
    logprobs: int = 0  # top-K raw-logit logprobs per output token (logprob_tap.py), 0 = off
    so_args: dict | None = None  # structured output (DeepInfer SamplingParams.so_args; tools.py builds it)
    multi_modal_data: dict | None = None  # images: {"image_bytes": [...], "image_sizes": [(w, h)]} (images.py)
    request_id: str = ""
    # filled by the engine thread: ("token", token_id, finished, finish_reason, (top_ids, top_logprobs) | None)
    #                              | ("error", message)
    out: "queue.Queue[tuple]" = field(default_factory=queue.Queue)
    submitted: float = field(default_factory=time.perf_counter)
    first_token_t: float | None = None
    n_out: int = 0
    done: bool = False


# Prefill attempt interval used by "auto" with DP > 1 (decode steps between two prefill chunks when several ranks
# share the step-type vote). 16 was the best of 1 / 8 / 16 on 4x RTX PRO 6000 (docs/SERVER.md).
DP_AUTO_INTERVAL = 16


def resolve_prefill_interval(value, dp_size: int) -> int:
    """--prefill-interval / --prefill-any-interval: an int >= 1, or "auto" (1 for dp 1, DP_AUTO_INTERVAL otherwise)."""
    if value is None or str(value).strip().lower() == "auto":
        return 1 if int(dp_size) <= 1 else DP_AUTO_INTERVAL
    v = int(value)
    if v < 1:
        raise ValueError(f"prefill interval must be >= 1 or 'auto', got {value!r}")
    return v


class EngineLoop:
    """Owns the DeepInfer LLM on a dedicated thread; thread-safe submit()/abort()."""

    def __init__(self, ckpt: str, *, max_num_seqs: int = 8, max_model_len: int = 16384,
                 max_num_batched_tokens: int = 4096, gpu_memory_utilization: float = 0.90, max_blocks: int = 256,
                 nextn: int = 0, graphs: bool = False, zos: str = "default", chunked_prefill: bool = True,
                 prefill_interval: int | str = "auto", prefill_any_interval: int | str = "auto",
                 deterministic: bool = False,
                 prefix_cache: dict | None = None,
                 dp_size: int = 1, dp_rank: int = 0, ep_size: int | None = None, mtp_varlen: bool = False):
        self.ckpt = ckpt
        self.cfg = dict(max_num_seqs=max_num_seqs, max_model_len=max_model_len,
                        max_num_batched_tokens=max_num_batched_tokens, gpu_memory_utilization=gpu_memory_utilization,
                        max_blocks=max_blocks, nextn=nextn, graphs=graphs, zos=zos, chunked_prefill=chunked_prefill,
                        prefill_interval=prefill_interval, prefill_any_interval=prefill_any_interval,
                        deterministic=deterministic, prefix_cache=prefix_cache,
                        dp_size=dp_size, dp_rank=dp_rank, ep_size=ep_size or dp_size, mtp_varlen=mtp_varlen)
        # has_unfinished_requests() is an all-reduce over the DP group: with several ranks, an idle rank must keep
        # polling quickly or it delays the other ranks' steps (docs/MULTI_GPU.md)
        self.idle_poll = 0.5 if dp_size == 1 else 0.005
        self.prefix_cache = None
        self.max_model_len = max_model_len
        self._inbox: "queue.Queue[tuple]" = queue.Queue()
        self._live: dict[str, GenRequest] = {}
        self._ids = itertools.count(1)
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._error: BaseException | None = None
        # tokens: generated tokens routed to requests (with MTP several per step); emit_steps: summed over requests, the
        # engine steps in which a request received tokens, so (tokens - requests) / (emit_steps - requests) is the
        # tokens per decode step of a sequence (1 + accepted drafts). steps: engine.step() calls, which with DP > 1
        # every rank makes in lockstep (idle ranks run placeholder steps): a per-rank count, not per request
        self.stats = dict(steps=0, flushes=0, added=0, aborted=0, finished=0, max_running=0, busy_s=0.0, tokens=0,
                          emit_steps=0)
        self.eos_id: int | None = None
        self._thread = threading.Thread(target=self._run, name="deepinfer-engine", daemon=True)

    # ------------------------------------------------------------------ public (any thread)
    def start(self, timeout: float = 1800.0) -> None:
        self._thread.start()
        if not self._ready.wait(timeout):
            raise TimeoutError("engine did not come up")
        if self._error is not None:
            raise RuntimeError(f"engine start failed: {self._error!r}") from self._error

    def submit(self, req: GenRequest) -> GenRequest:
        req.request_id = str(next(self._ids))  # numeric strings: LLM._run_engine sorts ids with int()
        if not self.alive:  # crashed engine: fail now instead of after --request-timeout
            req.out.put(("error", f"engine is not running: {self._error!r}"))
            return req
        self._inbox.put(("add", req))
        eng = sys.modules.get("sm120_engram")  # Engram rows of the prompt into the page cache while it waits for a slot
        if eng is not None and req.prompt_ids is not None:
            try:
                eng.prefetch_prompt(req.request_id, req.prompt_ids)
            except Exception:  # noqa: BLE001 - advisory only
                pass
        return req

    def abort(self, request_id: str) -> None:
        self._inbox.put(("abort", request_id))
        eng = sys.modules.get("sm120_engram")
        if eng is not None:
            eng.prefetch_cancel(request_id)

    def shutdown(self) -> None:
        self._stop.set()
        self._inbox.put(("noop", None))
        self._thread.join(timeout=60)

    @property
    def vision(self) -> bool:
        """The engine has DeepInfer's multimodal (image) processor."""
        return getattr(getattr(self, "engine", None), "multi_modal_processor", None) is not None

    def image_tokens(self, width: int, height: int) -> int:
        """Prompt tokens one image of this size expands to (DeepInfer's own estimate; pure arithmetic, any thread)."""
        from deepinfer.ve.processor.common import ImageInput

        mmp = self.engine.multi_modal_processor
        return int(mmp.mm_processor.image_processor.estimate_num_tokens(ImageInput(width=width, height=height)))

    @property
    def alive(self) -> bool:
        return self._thread.is_alive() and self._error is None

    # ------------------------------------------------------------------ engine thread
    def _build(self):
        c = self.cfg
        if TOOLS not in sys.path:
            sys.path.insert(0, TOOLS)
        import argparse

        import run_deepinfer  # tools/run_deepinfer.py: same env defaults, shims and patches as the eval runs

        # The NORMAL scheduler only tries a prefill every DEEPINFER_PREFILL_ATTEMPT_INTERVAL (default 70) decode steps,
        # idle/placeholder steps included (csrc/engine/scheduler.hpp:296-330): ~2.3 s of TTFT at 32 ms/step on the
        # RTX 5090 test machine. 1 = prefill whenever a prompt waits (one decode step stalls per prefill; chunked prefill caps it).
        os.environ.setdefault("DEEPINFER_PREFILL_ATTEMPT_INTERVAL",
                              str(resolve_prefill_interval(c["prefill_interval"], c["dp_size"])))
        # With DP > 1 that interval only marks "ALL" points, where a prefill step happens only if EVERY DP rank has a
        # waiting prompt (MIN all-reduce of the ranks' votes). A rank alone with a prompt waits for an "ANY" point,
        # every DEEPINFER_PREFILL_ANY_ATTEMPT_INTERVAL decode steps (default 100; scheduler.hpp:296-340). With
        # --dp 4 a single request waited up to 100 decode steps (~2 s TTFT at 20 ms per step). 1 = an ANY point at
        # every step except the one right after a prefill step. Cost: while any rank has a waiting prompt, every
        # rank's decode pauses for that prefill chunk (the other ranks run an empty prompt step).
        # "auto" (default): 1 / 1 with one rank, where the vote is trivial; with DP > 1 an interval of DP_AUTO_INTERVAL
        # for both, which guarantees that many decode steps between two prefill chunks (with 1, every step could be a
        # prefill step and the decoders got ~1 token per chunk; docs/SERVER.md). An explicit -e DEEPINFER_* wins.
        os.environ.setdefault("DEEPINFER_PREFILL_ANY_ATTEMPT_INTERVAL",
                              str(resolve_prefill_interval(c["prefill_any_interval"], c["dp_size"])))
        log.info("prefill intervals: DEEPINFER_PREFILL_ATTEMPT_INTERVAL=%s DEEPINFER_PREFILL_ANY_ATTEMPT_INTERVAL=%s "
                 "(--prefill-interval %s, --prefill-any-interval %s, dp %s)",
                 os.environ["DEEPINFER_PREFILL_ATTEMPT_INTERVAL"], os.environ["DEEPINFER_PREFILL_ANY_ATTEMPT_INTERVAL"],
                 c["prefill_interval"], c["prefill_any_interval"], c["dp_size"])
        if c["mtp_varlen"]:
            # DeepInfer's adaptive speculative decoding (csrc/engine/mtp_varlen.hpp): after graph capture the worker
            # replays every decode graph to get a (graph size -> ms) curve (model_runner.profile_decode_perf, cached in
            # DEEPINFER_DECODE_PERF_CACHE_DIR for DEEPINFER_DECODE_PERF_CACHE_TTL_SEC); each step the scheduler picks
            # the total verify size that maximizes expected tokens / ms from the DFlash confidence head and gives the
            # draft slots to the most confident sequences (per-request K <= nextn). Needs CUDA graphs and nextn > 0.
            os.environ["DEEPINFER_ENABLE_MTP_VARLEN"] = "1"
            root = os.environ.get("SM120_CACHE_ROOT", "/cache")
            if os.path.isdir(root) and os.access(root, os.W_OK):
                os.environ.setdefault("DEEPINFER_DECODE_PERF_CACHE_DIR", os.path.join(root, "decode_perf"))
        if c["deterministic"]:
            # b12x fused_moe is not run-to-run deterministic (diag_repeat.py); the torch MoE reference is
            os.environ["SM120_MOE_IMPL"] = "torch"

        run_deepinfer.setup_env(argparse.Namespace(
            max_blocks=c["max_blocks"], zos=c["zos"], batch_invariant=False, nextn=c["nextn"], graphs=c["graphs"],
            force_topk=None, deepinfer_root=os.environ.get("DEEPINFER_ROOT", "/opt/deepinfer")))
        import sm120_patches

        sm120_patches.install()
        from _config import create_llm  # DeepInfer examples/_config.py

        nextn = int(os.environ.get("DEEPINFER_USE_NEXTN_DECODE", "0") or 0)
        pcfg = c["prefix_cache"]
        if pcfg:
            sys.path.insert(0, os.path.join(ROOT, "prefix_cache"))
            import prefix_cache

            prefix_cache.configure_env(c["max_num_seqs"], pcfg["max_entries"])  # reserve state slots for snapshots
        llm = create_llm(self.ckpt, c["dp_rank"], ep_size=c["ep_size"], cp_size=1, data_parallel_size=c["dp_size"],
                         max_num_batched_tokens=c["max_num_batched_tokens"], max_model_len=c["max_model_len"],
                         max_num_seqs=c["max_num_seqs"], gpu_memory_utilization=c["gpu_memory_utilization"],
                         lookahead_tokens=min(1, nextn), chunked_prefill=c["chunked_prefill"], disable_log_stats=True)
        if pcfg:
            self.prefix_cache = prefix_cache.install(llm.llm_engine, nextn=nextn, **pcfg)
        from logprob_tap import LogprobTap

        self.tap = LogprobTap()
        self.tap.install()  # inert until a request asks for logprobs
        self.nextn = nextn
        return llm

    def _run(self) -> None:
        try:
            self.llm = self._build()
            self.engine = self.llm.llm_engine
            self.eos_id = self.engine.eos_id
            from deepinfer._C.engine import RequestOutput  # pyright: ignore[reportMissingModuleSource]
            from deepinfer.sampling_params import SamplingParams

            self._RequestOutput, self._SamplingParams = RequestOutput, SamplingParams
        except BaseException as exc:  # noqa: BLE001
            log.exception("engine build failed")
            self._error = exc
            self._ready.set()
            return
        self._ready.set()
        log.info("engine ready: %s", self.cfg)
        pending_flush = False
        try:
            while not self._stop.is_set():
                unfinished = self.engine.has_unfinished_requests()
                self._drain(block=not unfinished and not pending_flush)
                unfinished = self.engine.has_unfinished_requests()
                if unfinished:
                    t0 = time.perf_counter()
                    with _prof_rng("sm120:engine.step"):
                        outs = self.engine.step()
                    self.stats["busy_s"] += time.perf_counter() - t0
                    self.stats["steps"] += 1
                    pending_flush = True
                elif pending_flush:
                    with _prof_rng("sm120:engine.flush"):
                        outs = self.engine.flush_inflight_reqs()
                    self.stats["flushes"] += 1
                    pending_flush = False
                else:
                    continue
                self._route(outs)
                self.stats["max_running"] = max(self.stats["max_running"], len(self._live))
        except BaseException as exc:  # noqa: BLE001
            log.exception("engine loop crashed")
            self._error = exc
            for r in list(self._live.values()):
                r.out.put(("error", f"engine crashed: {exc!r}"))
            self._live.clear()
            return
        # Orderly shutdown: close the engine here, on its own thread, while the interpreter is intact. Left to
        # LLMEngine.__del__ at interpreter exit, unpinning the multimodal pool fails (ImportError: sys.meta_path
        # is None) and the still exported pool mapping raises BufferError in SharedMemory.__del__.
        try:
            self.llm.close()
        except Exception:  # noqa: BLE001
            log.exception("engine close failed")

    def _drain(self, block: bool) -> None:
        """Apply queued commands; block (with a timeout, so shutdown is seen) only when the engine is idle."""
        try:
            item = self._inbox.get(timeout=self.idle_poll) if block else self._inbox.get_nowait()
        except queue.Empty:
            return
        while True:
            kind, arg = item
            if kind == "add":
                self._add(arg)
            elif kind == "abort":
                r = self._live.pop(arg, None)
                self.tap.close(arg)
                if r is not None:
                    self.engine.abort_request(arg)
                    self.stats["aborted"] += 1
                    r.done = True
            try:
                item = self._inbox.get_nowait()
            except queue.Empty:
                return

    def _add(self, r: GenRequest) -> None:
        import numpy as np

        try:
            extra = {"so_args": r.so_args} if r.so_args is not None else {}
            sp = self._SamplingParams(temperature=r.temperature, top_p=r.top_p, top_k=r.top_k, max_tokens=r.max_tokens,
                                      ignore_eos=r.ignore_eos, seq_seed=r.seed, **extra)
            mmd = r.multi_modal_data
            if mmd is None and self.engine.multi_modal_processor is not None:
                # as LLM.generate does: with a multimodal processor every prompt carries multi_modal_data, else a
                # text-only prompt batched with image prompts breaks the batch's mm_token_type (deepseek_vl.py)
                mmd = {"image_bytes": [], "image_sizes": []}
            mm = {"multi_modal_data": mmd} if mmd is not None else {}
            seq = self.engine.add_request(user_id="0", request_id=r.request_id, sampling_params=sp,
                                          prompt_token_ids=np.asarray(r.prompt_ids, dtype=np.int32), **mm)
        except Exception as exc:  # noqa: BLE001  (bad sampling params, empty prompt, ...)
            r.out.put(("error", f"{type(exc).__name__}: {exc}"))
            return
        if seq is None:  # add_request drops requests with out-of-range token ids
            r.out.put(("error", "request dropped by the engine (invalid token ids)"))
            return
        self._live[r.request_id] = r
        if r.logprobs:
            self.tap.add(r.request_id, r.logprobs)
        self.stats["added"] += 1
        self._profile_maybe_start(r)

    # ------------------------------------------------------------------ SM120_PROFILE_PREFILL (diagnostics)
    _prof = None

    def _profile_maybe_start(self, r) -> None:
        """SM120_PROFILE_PREFILL=N: profile (torch profiler, CPU + CUDA) from the N-th request this engine process
        admits until its first token, i.e. its whole prefill, and write a trace plus a per-category table to
        SM120_PROFILE_DIR (default /work). One shot per process; with --dp every rank counts its own requests."""
        n = int(os.environ.get("SM120_PROFILE_PREFILL", "0") or 0)
        if n <= 0 or self.stats["added"] != n or self._prof is not None:
            return
        try:
            import torch
            from torch.profiler import ProfilerActivity, profile
            torch.cuda.synchronize()
            p = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False)
            _set_prof_active(True)
            p.__enter__()
            self._prof_tid = threading.get_native_id()
            self._prof = (r.request_id, p, time.perf_counter(), self.stats["steps"], len(r.prompt_ids))
            log.info("SM120_PROFILE_PREFILL: profiling request %d (%s, %d prompt tokens)", n, r.request_id,
                     len(r.prompt_ids))
        except Exception as exc:  # noqa: BLE001
            log.warning("SM120_PROFILE_PREFILL: could not start the profiler: %r", exc)
            self._prof = ("", None, 0.0, 0, 0)

    def _profile_stop(self, r) -> None:
        rid, p, t0, s0, plen = self._prof
        self._prof = ("", None, 0.0, 0, 0)  # one shot
        try:
            import torch
            torch.cuda.synchronize()
            wall = (time.perf_counter() - t0) * 1000
            p.__exit__(None, None, None)
            _set_prof_active(False)
            out = os.environ.get("SM120_PROFILE_DIR", "/work")
            tag = f"prefill_r{self.cfg.get('dp_rank', 0)}_p{os.getpid()}"
            path = os.path.join(out, tag)
            p.export_chrome_trace(path + ".trace.json")
            sys.path.insert(0, os.path.join(ROOT, "tests"))
            import contextlib
            import io
            import prof_e2e
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                prof_e2e.summarize(f"prefill of {plen} tokens ({rid})", prof_e2e.kernels(p), wall,
                                   self.stats["steps"] - s0)
                self._profile_cpu_table(p, wall)
            open(path + ".txt", "w").write(buf.getvalue())
            log.info("SM120_PROFILE_PREFILL: wrote %s.trace.json and %s.txt (wall %.0f ms)\n%s", path, path, wall,
                     buf.getvalue())
        except Exception as exc:  # noqa: BLE001
            log.warning("SM120_PROFILE_PREFILL: could not write the profile: %r", exc)

    def _profile_cpu_table(self, p, wall: float) -> None:
        """Host side of the profiled prefill: named ranges (sm120:* ours, deepinfer:* DeepInfer's) and the CUDA /
        collective calls that block, by total time; then the engine thread's timeline of those ranges."""
        import collections
        evs = [ev for ev in p.events() if ev.device_type.name == "CPU"]
        # the engine thread = the one that ran engine.step (profiler thread ids need not be native ids)
        tid = next((ev.thread for ev in evs if ev.name == "sm120:engine.step"), getattr(self, "_prof_tid", None))
        tot, cnt = collections.Counter(), collections.Counter()
        timeline = []
        t_first = None
        block = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize", "cudaMemcpy",
                 "cudaMemcpyAsync", "aten::_local_scalar_dense", "aten::item")
        for ev in evs:
            n = ev.name
            ours = n.startswith(("sm120:", "deepinfer:"))
            if not (ours or n in block or n.startswith(("nccl:", "c10d::", "gloo:", "record_param_comms"))):
                continue
            d = (ev.time_range.end - ev.time_range.start) / 1000.0
            key = n if ev.thread == tid or tid is None else f"{n} [other thread]"
            tot[key] += d
            cnt[key] += 1
            if ours and ev.thread == tid:
                t_first = ev.time_range.start if t_first is None else min(t_first, ev.time_range.start)
                timeline.append((ev.time_range.start, d, n))
        print(f"\n  host side (ms; wall {wall:.1f} ms; 'other thread' = gather / prefetch / comm threads):")
        for k, v in tot.most_common(30):
            print(f"    {v:10.2f} ms  x{cnt[k]:<6d} {k}")
        if timeline:
            print("  engine thread timeline (named ranges, start offset ms / duration ms):")
            for t, d, n in sorted(timeline)[:80]:
                if d >= 0.5:
                    print(f"    +{(t - t_first) / 1000.0:9.2f}  {d:9.2f}  {n}")

    def _route(self, outs) -> None:
        seen = set()  # one call = the outputs of one engine step (ZOS: of the previous one)
        for ptr in outs:
            o = self._RequestOutput.clone(ptr)  # the pool slot is recycled by the next step
            r = self._live.get(o.request_id)
            if r is None:  # aborted, or a placeholder / in-flight output of a request already closed
                continue
            if r.first_token_t is None:
                r.first_token_t = time.perf_counter()
                if self._prof is not None and self._prof[0] == r.request_id:
                    self._profile_stop(r)
            r.n_out += 1
            self.stats["tokens"] += 1
            if o.request_id not in seen:  # steps in which this request got tokens (speculative decoding: >= 1 each)
                seen.add(o.request_id)
                self.stats["emit_steps"] += 1
            row = self.tap.pop_row(o.request_id) if r.logprobs else None
            r.out.put(("token", int(o.token_id), bool(o.finished), str(o.finish_reason or ""), row))
            if o.finished:
                r.done = True
                self._live.pop(o.request_id, None)
                self.tap.close(o.request_id)
                self.stats["finished"] += 1
