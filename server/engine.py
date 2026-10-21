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
    request_id: str = ""
    # filled by the engine thread: ("token", token_id, finished, finish_reason, (top_ids, top_logprobs) | None)
    #                              | ("error", message)
    out: "queue.Queue[tuple]" = field(default_factory=queue.Queue)
    submitted: float = field(default_factory=time.perf_counter)
    first_token_t: float | None = None
    n_out: int = 0
    done: bool = False


class EngineLoop:
    """Owns the DeepInfer LLM on a dedicated thread; thread-safe submit()/abort()."""

    def __init__(self, ckpt: str, *, max_num_seqs: int = 8, max_model_len: int = 16384,
                 max_num_batched_tokens: int = 4096, gpu_memory_utilization: float = 0.90, max_blocks: int = 256,
                 nextn: int = 0, graphs: bool = False, zos: str = "default", chunked_prefill: bool = True,
                 prefill_interval: int = 1, deterministic: bool = False, prefix_cache: dict | None = None,
                 dp_size: int = 1, dp_rank: int = 0, ep_size: int | None = None):
        self.ckpt = ckpt
        self.cfg = dict(max_num_seqs=max_num_seqs, max_model_len=max_model_len,
                        max_num_batched_tokens=max_num_batched_tokens, gpu_memory_utilization=gpu_memory_utilization,
                        max_blocks=max_blocks, nextn=nextn, graphs=graphs, zos=zos, chunked_prefill=chunked_prefill,
                        prefill_interval=prefill_interval, deterministic=deterministic, prefix_cache=prefix_cache,
                        dp_size=dp_size, dp_rank=dp_rank, ep_size=ep_size or dp_size)
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
        self.stats = dict(steps=0, flushes=0, added=0, aborted=0, finished=0, max_running=0, busy_s=0.0)
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
        self._inbox.put(("add", req))
        return req

    def abort(self, request_id: str) -> None:
        self._inbox.put(("abort", request_id))

    def shutdown(self) -> None:
        self._stop.set()
        self._inbox.put(("noop", None))
        self._thread.join(timeout=60)

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
        os.environ.setdefault("DEEPINFER_PREFILL_ATTEMPT_INTERVAL", str(c["prefill_interval"]))
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
                    outs = self.engine.step()
                    self.stats["busy_s"] += time.perf_counter() - t0
                    self.stats["steps"] += 1
                    pending_flush = True
                elif pending_flush:
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
            sp = self._SamplingParams(temperature=r.temperature, top_p=r.top_p, top_k=r.top_k, max_tokens=r.max_tokens,
                                      ignore_eos=r.ignore_eos, seq_seed=r.seed)
            seq = self.engine.add_request(user_id="0", request_id=r.request_id, sampling_params=sp,
                                          prompt_token_ids=np.asarray(r.prompt_ids, dtype=np.int32))
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

    def _route(self, outs) -> None:
        for ptr in outs:
            o = self._RequestOutput.clone(ptr)  # the pool slot is recycled by the next step
            r = self._live.get(o.request_id)
            if r is None:  # aborted, or a placeholder / in-flight output of a request already closed
                continue
            if r.first_token_t is None:
                r.first_token_t = time.perf_counter()
            r.n_out += 1
            row = self.tap.pop_row(o.request_id) if r.logprobs else None
            r.out.put(("token", int(o.token_id), bool(o.finished), str(o.finish_reason or ""), row))
            if o.finished:
                r.done = True
                self._live.pop(o.request_id, None)
                self.tap.close(o.request_id)
                self.stats["finished"] += 1
