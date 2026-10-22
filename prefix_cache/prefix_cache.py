"""Prefix caching for DeepInfer V4.1 (single engine), as Python wrappers around the public LLMEngine API.

DeepInfer's C++ scheduler already has the *mechanics* of a prefix hit, used by its (unreleased) KV store: a sequence
created by `Scheduler.fork_seq(parent, blocks, "_<branch>", ...)` with `cache.num_tokens = k` keeps the first
k // 128 KV blocks, gets a copy-on-write clone of the parent's per-sequence **state slot** (`_clone_state`: SWA rings of
every layer + compressor / indexer partial groups + DFlash SWA), a copy of a partially filled tail block, and its
prefill continues at token k like a later chunk of a chunked prefill (scheduler.hpp:398-433, 1060-1120). Missing in
the open-source release is the cache itself. This module adds it:

* **Snapshots.** Every request gets `chunk_checkpoints` at multiples of `checkpoint_stride` (default: the larger of
  `grain` and max_num_batched_tokens, so a prefill that runs alone is not split into more engine steps than chunked
  prefill makes anyway; a smaller stride, down to `grain`, gives finer hits at the cost of one engine step per
  checkpoint), so its prefill chunks end there. After a step whose prefill chunk ended at k (a checkpoint, any other
  chunk end at a multiple of `grain` (default 128 = one KV block, the kvstore grain `KVSTORE_GRAIN`), or the end of
  the prompt), the k-token prefix is recorded: its first ceil(k / 128) KV blocks are pinned
  (`BlockManager.pin_blocks`) and the sequence's state slot is copied into a cache-owned slot (the engine's own
  `clone_and_zero_states_and_blocks_on_default_stream`, stream-ordered after the step's forward and before the
  sequence's next step). Prefix KV is never rewritten (paged, append-only); the state at exactly k is what makes a hit
  exact: SWA window, compressor partial group (k odd at a prompt end), indexer state.
* **Index.** key = (k, blake2b of the k token ids); entries in an LRU (OrderedDict). Eviction unpins the blocks and
  frees the state slot; limits: `max_entries` snapshots (state slots are reserved for them at engine start via
  DEEPINFER_NUM_STATE_SLOTS_RATIO) and `max_block_frac` of the KV blocks pinned by the cache.
* **Hit on add_request.** The longest cached k <= len(prompt) - 2 whose hash matches (k < len - 1 so that at least
  the last prompt token is computed, which also keeps fork_seq on its "prompt incomplete" path): the request's Sequence
  is built by the engine (`enqueue=False`), given `num_cached_tokens = k` and the snapshot's `state_id`, and handed to
  `fork_seq` with the cached block ids; the fork (same request id, prompt and sampling params) is what the scheduler
  runs. Engram needs nothing: n-gram hashes are computed from the sequence's token ids, which the fork carries.

Needs one C++ binding, `Sequence.block_table` (read-only): patches/deepinfer-sequence-block-table.patch.
Generation end: when a request finishes normally, the prefix prompt + answer is snapshotted too (k = num_cached_tokens
at that moment, read together with the state), so the next chat turn does not recompute the previous answer.
Not cached: multimodal prompts, prompt_logprobs (fork_seq refuses), structured output.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger("prefix_cache")

BLOCK = 128  # DeepInfer KV block size (cache_config.block_size; KVSTORE_GRAIN)


def configure_env(max_num_seqs: int, max_entries: int) -> None:
    """Reserve state slots for the snapshots. Call before the engine is built.

    num_state_slots = int(max_num_seqs * DEEPINFER_NUM_STATE_SLOTS_RATIO) + 1 (cache_engine.get_num_state_slots)."""
    base = float(os.environ.get("DEEPINFER_NUM_STATE_SLOTS_RATIO", "1.2"))
    os.environ["DEEPINFER_NUM_STATE_SLOTS_RATIO"] = str(base + (max_entries + 1) / max(1, max_num_seqs))


def _rng(name: str):
    """Profiler range for SM120_PROFILE_PREFILL (sm120_prof), a no-op otherwise or without sm120 on the path."""
    try:
        import sm120_prof
        return sm120_prof.rng(name)
    except ImportError:
        import contextlib
        return contextlib.nullcontext()


def _digest(ids: np.ndarray, k: int) -> bytes:
    return hashlib.blake2b(np.ascontiguousarray(ids[:k], dtype=np.int32).tobytes(), digest_size=16).digest()


@dataclass
class Entry:
    k: int
    digest: bytes
    blocks: list[int]
    state_id: int
    hits: int = 0
    created: float = field(default_factory=time.perf_counter)


class _SchedulerProxy:
    """engine.scheduler stand-in: records which sequences ran a prefill chunk in the step being scheduled."""

    def __init__(self, real, cache: "PrefixCache"):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_cache", cache)

    def __getattr__(self, name):
        return getattr(self._real, name)

    def __setattr__(self, name, value):
        setattr(self._real, name, value)

    def schedule(self, *a, **kw):
        out = self._real.schedule(*a, **kw)
        try:
            self._cache._on_schedule(*out)
        except Exception:  # noqa: BLE001  (the cache must never break serving)
            log.exception("prefix cache: schedule hook failed")
        return out


class PrefixCache:
    def __init__(self, engine, *, grain: int = BLOCK, max_entries: int = 64, max_block_frac: float = 0.5,
                 snapshot_prompt_end: bool | None = None, nextn: int = 0, checkpoint_stride: int | None = None):
        if grain % BLOCK:
            raise ValueError(f"grain must be a multiple of {BLOCK}")
        if not checkpoint_stride:  # None / 0: do not split prefills beyond the engine's own chunking
            mnbt = getattr(getattr(engine, "scheduler_config", None), "max_num_batched_tokens", None) or grain
            checkpoint_stride = max(grain, int(mnbt))
        checkpoint_stride = -(-int(checkpoint_stride) // grain) * grain  # a multiple of the grain
        from deepinfer._C.engine import Sequence

        if not hasattr(Sequence, "block_table"):
            raise RuntimeError("prefix cache needs the Sequence.block_table binding: build _C with "
                               "patches/deepinfer-sequence-block-table.patch (prefix_cache/build_patched_c.sh)")
        self.engine = engine
        self.grain = grain
        self.checkpoint_stride = checkpoint_stride
        self.max_entries = max_entries
        self.max_blocks = int(max_block_frac * engine.cache_config.num_gpu_blocks) if hasattr(engine, "cache_config") \
            else None
        self.block_size = getattr(getattr(engine, "cache_config", None), "block_size", BLOCK)
        assert self.block_size == BLOCK, self.block_size
        # prompt-end snapshots with MTP on were verified bit-exact (nextn 5, ckpt_full): on by default in every mode
        self.snapshot_prompt_end = True if snapshot_prompt_end is None else snapshot_prompt_end
        self.entries: "OrderedDict[tuple[int, bytes], Entry]" = OrderedDict()
        # switches (tests): lookup = serve hits, snapshot = record prefixes, checkpoints = split prefills at the grain
        self.lookup = self.snapshot = self.checkpoints = True
        self.extra_checkpoints: list[int] = []  # tests: also split every prefill here (chunk-aligned baselines)
        self._pending: list[tuple] = []  # (Sequence, k) whose prefill chunk ends at k in the step being executed
        self._live: dict[str, object] = {}  # request_id -> Sequence, for generation-end snapshots
        self.snapshot_gen_end = True
        self.stats = dict(lookups=0, hits=0, hit_tokens=0, prompt_tokens=0, snapshots=0, gen_end_snapshots=0,
                          evictions=0, snapshot_skipped=0)
        from deepinfer._C.engine import RequestOutput

        self._RO = RequestOutput
        self._orig_add = engine.add_request
        self._orig_step = engine.step
        self._orig_flush = engine.flush_inflight_reqs
        engine.scheduler = _SchedulerProxy(engine.scheduler, self)
        engine.add_request = self.add_request
        engine.step = self.step
        engine.flush_inflight_reqs = self.flush_inflight_reqs
        log.info("prefix cache on: grain %d, checkpoint stride %d, max_entries %d, max_blocks %s, prompt-end snapshots "
                 "%s, state slots %d", grain, checkpoint_stride, max_entries, self.max_blocks, self.snapshot_prompt_end,
                 self._num_state_slots())

    # ------------------------------------------------------------------ helpers
    def _num_state_slots(self) -> int:
        cc = getattr(self.engine, "cache_config", None)
        return int(getattr(cc, "num_state_seqs", -1)) if cc is not None else -1

    def _pinned_blocks(self) -> int:
        # list(...) copies the values in one C call: safe against the engine thread inserting / evicting while
        # summary() runs on an HTTP thread (iterating the OrderedDict itself can raise "mutated during iteration")
        return len({b for e in list(self.entries.values()) for b in e.blocks})

    def _evict_one(self) -> bool:
        if not self.entries:
            return False
        _, e = self.entries.popitem(last=False)
        self.engine.block_manager.unpin_blocks(e.blocks)
        self.engine.state_manager.unpin_blocks([e.state_id])
        self.stats["evictions"] += 1
        return True

    def clear(self) -> None:
        while self._evict_one():
            pass
        self._live.clear()

    def _alloc_state(self) -> int | None:
        sm = self.engine.state_manager
        # keep at least max_num_seqs free slots for live requests (the scheduler allocates one per sequence)
        reserve = int(getattr(self.engine.scheduler_config, "max_num_seqs", 1)) + 1 \
            if hasattr(self.engine, "scheduler_config") else 2
        while self.entries and (sm.get_num_free_blocks() <= reserve or len(self.entries) >= self.max_entries):
            self._evict_one()
        if sm.get_num_free_blocks() <= reserve:
            return None
        sm.lock()
        try:
            return int(sm.allocate_nolock(1)[0])
        finally:
            sm.unlock()

    # ------------------------------------------------------------------ engine wrappers
    def add_request(self, *args, **kw):
        names = ("user_id", "request_id", "sampling_params", "prompt_token_ids", "multi_modal_data", "arrival_time",
                 "extra", "enqueue", "chunk_checkpoints")
        a = dict(zip(names, args))
        a.update(kw)
        ids = np.asarray(a["prompt_token_ids"], dtype=np.int32)
        sp = a.get("sampling_params")
        mm = a.get("multi_modal_data")
        # LLM._add_request passes an empty image list for every text prompt of a vision model
        text_only = mm is None or not (mm.get("image_bytes") or mm.get("images") or mm.get("video_bytes"))
        usable = ((self.lookup or self.snapshot or self.checkpoints) and a.get("enqueue", True) and text_only
                  and not getattr(sp, "prompt_logprobs", False) and getattr(sp, "so_args", None) is None
                  and not (isinstance(sp, dict) and (sp.get("prompt_logprobs") or sp.get("so_args"))))
        if not usable:
            return self._orig_add(**a)
        L = int(ids.size)
        st = self.checkpoint_stride
        ckpts = list(range(st, L, st)) if (self.checkpoints or self.snapshot) else []
        ckpts = sorted(set(ckpts) | {c for c in self.extra_checkpoints if 0 < c < L})
        if a.get("chunk_checkpoints"):
            ckpts = sorted(set(ckpts) | set(a["chunk_checkpoints"]))
        self.stats["lookups"] += 1
        self.stats["prompt_tokens"] += L
        with _rng("sm120:prefix_cache.lookup"):
            hit = self._lookup(ids) if self.lookup else None
        if hit is None:
            a["chunk_checkpoints"] = ckpts
            return self._orig_add(**a)
        a["enqueue"] = False
        a["chunk_checkpoints"] = None
        seq = self._orig_add(**a)
        if seq is None:
            return None
        e = hit
        k = e.k
        seq.num_cached_tokens = k
        seq.state_id = e.state_id
        seq.cached_hit_tokens = k
        nblk = -(-k // BLOCK)
        child = self.engine.scheduler._real.fork_seq(
            seq=seq, blocks=np.asarray(e.blocks[:nblk], dtype=np.int32), target_branch_id="_pc",
            sampling_params=seq.sampling_params, so_matcher=None, sp_token=0, replace_last_token_id=-1,
            seq_id=next(self.engine.seq_counter), arrival_time=seq.arrival_time)
        child.chunk_checkpoints = [c for c in ckpts if c > k]
        mmp = self.engine.multi_modal_processor
        if mmp is not None and mm is not None:
            # as LLMEngine.add_request's _enqueue: the processor fills the (shared) multimodal dict asynchronously and
            # fetch() calls scheduler.set_schedulable(id); fork_seq marks the fork schedulable at once, so undo that.
            # num_cached_tokens is the processor's own KV-hit argument (multimodal_processor.add).
            child.schedulable = False
            mmp.add(a["request_id"], mm, num_cached_tokens=k)
        e.hits += 1
        self.entries.move_to_end((e.k, e.digest))
        self.stats["hits"] += 1
        self.stats["hit_tokens"] += k
        log.debug("prefix hit: request %s, %d/%d tokens cached", a["request_id"], k, L)
        return child

    def _lookup(self, ids: np.ndarray) -> Entry | None:
        L = int(ids.size)
        ks = sorted({k for k, _ in self.entries if k <= L - 2}, reverse=True)
        for k in ks:
            e = self.entries.get((k, _digest(ids, k)))
            if e is not None:
                return e
        return None

    def _on_schedule(self, seq_metadata_list, scheduler_outputs) -> None:
        if not self.snapshot:
            return
        if self.snapshot_gen_end:
            for seq in scheduler_outputs.scheduled_seqs:
                self._live[seq.request_id] = seq
            if len(self._live) > 4096:  # aborted requests never report a normal finish
                for rid in list(self._live)[:2048]:
                    del self._live[rid]
        if not scheduler_outputs.prompt_run:
            return
        for seq, md in zip(scheduler_outputs.scheduled_seqs, seq_metadata_list):
            k = int(md.seqlen_k)
            L = seq.get_prompt_len()
            if k <= 0 or k > L:
                continue
            if k == L and not self.snapshot_prompt_end:
                continue
            if k != L and k % self.grain:
                continue
            self._pending.append((seq, k))

    def _finished(self, outs) -> None:
        """Queue a generation-end snapshot for every request that finished normally in these outputs.

        The scheduler frees a finished sequence's blocks and state slot in the next schedule(), so the snapshot is
        taken right after this step() returns. With ZOS the engine post-processes step N inside step N+1, after
        N+1's forward was launched (worker.execute_model_step), so a finished sequence may have run one more forward:
        its KV / state then cover one more token. num_cached_tokens and the state are read together at copy time,
        i.e. after everything launched so far, so (k, state) stay consistent either way."""
        if not self.snapshot or not self.snapshot_gen_end:
            return
        RO = self._RO
        for ptr in outs:
            if not RO.get_finished(ptr):
                continue
            rid = RO.get_request_id(ptr)
            seq = self._live.pop(rid, None)
            if seq is not None and RO.get_finish_reason(ptr) in ("stop", "length"):
                self._pending.append((seq, None))  # k = num_cached_tokens at copy time

    def _take_snapshots(self) -> None:
        pending, self._pending = self._pending, []
        for seq, k in pending:
            gen_end = k is None
            if gen_end:
                k = int(seq.num_cached_tokens)
                if k <= 0 or k > len(seq.data.get_token_ids()):
                    self.stats["snapshot_skipped"] += 1
                    continue
            ids = np.asarray(seq.data.get_token_ids()[:k], dtype=np.int32)
            key = (k, _digest(ids, k))
            if key in self.entries:
                self.entries.move_to_end(key)
                continue
            nblk = -(-k // BLOCK)
            blocks = list(seq.block_table)[:nblk]
            if len(blocks) < nblk or seq.state_id < 0:
                self.stats["snapshot_skipped"] += 1
                continue
            if self.max_blocks is not None:
                while self.entries and self._pinned_blocks() + nblk > self.max_blocks:
                    self._evict_one()
                if nblk > self.max_blocks:
                    self.stats["snapshot_skipped"] += 1
                    continue
            sid = self._alloc_state()
            if sid is None:
                self.stats["snapshot_skipped"] += 1
                continue
            self.engine.block_manager.pin_blocks(blocks)
            self.engine._run_workers(
                "clone_and_zero_states_and_blocks_on_default_stream",
                clone_states=np.asarray([[int(seq.state_id), sid]], dtype=np.int32),
                clone_blocks=np.zeros((0, 2), dtype=np.int32), zero_states=np.zeros((0,), dtype=np.int32))
            self.entries[key] = Entry(k=k, digest=key[1], blocks=blocks, state_id=sid)
            self.stats["snapshots"] += 1
            self.stats["gen_end_snapshots"] += gen_end

    def step(self):
        out = self._orig_step()
        with _rng("sm120:prefix_cache.snapshots"):
            self._finished(out)
            # the step's forward is queued on the stream; the copy is ordered after it and before the next step
            self._take_snapshots()
        return out

    def flush_inflight_reqs(self):
        out = self._orig_flush()
        self._finished(out)
        self._take_snapshots()
        return out

    def summary(self) -> dict:
        """Counters for /health; called from HTTP threads while the engine thread updates the cache."""
        s = dict(self.stats)
        s.update(entries=len(self.entries), pinned_blocks=self._pinned_blocks(), grain=self.grain,
                 checkpoint_stride=self.checkpoint_stride,
                 hit_rate=s["hits"] / s["lookups"] if s["lookups"] else 0.0,
                 token_hit_rate=s["hit_tokens"] / s["prompt_tokens"] if s["prompt_tokens"] else 0.0)
        return s


def install(engine, **kw) -> PrefixCache:
    return PrefixCache(engine, **kw)
