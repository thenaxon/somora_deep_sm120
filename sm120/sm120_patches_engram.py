"""SM120 runtime patches that let DeepInfer run with Engram enabled (no RDMA, tables not in GPU memory).

Activation: call `install()` once at start-up, after `sm120_patches.install()` (it registers lazy
post-import hooks through `sm120_patches.when_imported`, imports nothing heavy itself). The hook line
for sm120_patches.py is in docs/ENGRAM.md ("Requests"). `SM120_ENGRAM_PATCHES=0` disables all
of them; `sm120_patches_engram.APPLIED` lists what ran.

    E1  models/deepseek.py:477 materialize_hf_parameter -> tiny placeholder for `is_engram` tables
        (stock: a CPU tensor of the shard, ~98 GB per table, plus a CUDA fp32 sf, ~12 GB per table).
    E2  hugging_face/weights.py:331 _load_engram_tables -> record (shard path, tensor names) per table
        instead of copying 100 GB per table; other tables (no placeholder) still load the stock way.
    E3  v4_modules/engram.py:69 NgramEmbedding.__init__ keeps that record; engram.py:195
        EngramRuntime.initialize opens the tables with sm120_engram (mmap of the shard, SM120_ENGRAM_MODE)
        and hands them to the deep_ep.EngramBuffer shim (attach_tables) instead of write()+gather_sf().
    E5  hugging_face/loading.py:48,98 safe_open (shard validation, shape checks) -> header-only fallback when
        safetensors cannot map a shard: it maps files privately, which the kernel refuses (ENOMEM) for the
        101 GB Engram shard on a machine with less RAM+swap. Only keys()/get_slice().get_shape()/get_dtype()
        are used there.
    E6  CUDA-graph decode: models/deepseek.py DeepseekModel.__init__ registers models with engram;
        worker/cuda_graph_runner.py:150 CudaGraphRunner.replay first hashes the step's n-gram windows (already
        written into the graph's input_metadata_source by _prepare_input_tensors, incl. ZOS) and calls
        EngramBuffer.prestage(), which fills the static row buffers the captured hooks read. Inside the graph
        the fetch is only a view + sf transposition of those buffers (sm120_engram.EngramFetcher._graph_hooks).
    E7  engine/llm_engine.py LLMEngine.add_request / abort_request report every admitted prompt to the engram
        prefetcher (sm120_engram.EngramPrefetcher; E3 starts it, E6 hands it the hash metadata): its rows are read
        into the page cache ahead of the prefill. Advisory only: the gather is unchanged.
    E9  engine/llm_engine.py LLMEngine.step / flush_inflight_reqs: after the step (its outputs synchronized, not yet
        returned) sm120_engram.check_graph_fetch() raises if an in-graph engram fetch failed or timed out, so no
        output of that step is used (host / uring tables with CUDA graphs).
    E8  engine/llm_engine.py LLMEngine._init_cache: in SM120_ENGRAM_MODE=pinned, the KV sizing errors ("No available
        memory for the cache blocks", max seq len too large) get a hint with the GPU memory the registration took.
    E4  (only SM120_ENGRAM_TK=torch) tile_kernels.engram.{engram_hash,fused_weight,engram_gate_fwd} ->
        sm120_engram torch versions (for bisecting the TileLang kernels on SM120).

    SM120_ENGRAM_LAZY=1|0   E1/E2 (default 1; 0 = stock loader, needs RAM for the full tables)
"""

from __future__ import annotations

import dataclasses
import os

APPLIED: list[str] = []
_INSTALLED = False


def _on(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "no", "off")


def _note(msg: str) -> None:
    APPLIED.append(msg)
    try:
        import sm120_patches

        sm120_patches.APPLIED.append(msg)
    except Exception:  # noqa: BLE001
        pass
    if _on("SM120_PATCH_VERBOSE"):
        print(f"[sm120_patches_engram] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ E1
def _placeholder(param):
    """Shape/dtype-correct stand-in that owns 1 byte: expand() of a 1x1 tensor. Every attribute the
    builder attached (is_engram, layer_id, row_start, num_per_part, head_vocab_sizes, ...) is kept."""
    import torch

    data = torch.zeros((1, 1), dtype=param.dtype, device="cpu").expand(*param.shape)
    result = torch.nn.Parameter(data, requires_grad=False)
    for k, v in param.__dict__.items():
        if k != "sf":
            setattr(result, k, v)
    if hasattr(param, "sf"):
        result.sf = torch.ones((1, 1), dtype=param.sf.dtype, device="cpu").expand(*param.sf.shape)
    result.sm120_engram_lazy = True
    return result


def _patch_materialize(mod):
    """E1: deepseek.py imports materialize_hf_parameter by name (deepseek.py:32), so patch it there."""
    orig = mod.materialize_hf_parameter
    if getattr(orig, "_sm120_engram", False):
        return

    def materialize_hf_parameter(param, *, device, packed_fp4=False):
        if getattr(param, "is_engram", False):
            return _placeholder(param)
        return orig(param, device=device, packed_fp4=packed_fp4)

    materialize_hf_parameter._sm120_engram = True
    mod.materialize_hf_parameter = materialize_hf_parameter
    _note("E1 deepseek.materialize_hf_parameter: Engram tables -> 1-byte placeholders (read later from the shard)")


# ------------------------------------------------------------------------------------------------ E2
def _patch_load_tables(mod):
    orig = mod._load_engram_tables
    if getattr(orig, "_sm120_engram", False):
        return

    def _load_engram_tables(params, coverage, hf_dir):
        import sm120_engram

        lazy, rest = {}, {}
        for model_name, weight_name in coverage.engram_tables.items():
            p = params.get(model_name)
            (lazy if p is not None and getattr(p, "sm120_engram_lazy", False) else rest)[model_name] = weight_name
        stats = {"engram_tables": 0, "engram_rows": 0}
        if rest:
            stats = orig(params, dataclasses.replace(coverage, engram_tables=rest), hf_dir)
        ckpt_dir = os.environ.get("SM120_ENGRAM_CKPT") or str(hf_dir)
        for model_name, weight_name in sorted(lazy.items()):
            p = params[model_name]
            shard = mod._shard_of(coverage.weight_map, weight_name)
            path = os.path.join(ckpt_dir, shard)
            header, _ = sm120_engram.read_safetensors_header(path)
            keys = set(header) - {"__metadata__"}
            wkey = mod.resolve_hf_name(weight_name, keys)
            skey = mod.resolve_hf_name(mod.scale_name(weight_name), keys)
            wloc, sloc = sm120_engram.locate_tensor(path, wkey), sm120_engram.locate_tensor(path, skey)
            assert wloc.shape[1] == p.shape[1], f"{model_name}: HF table width {wloc.shape[1]} != runtime {p.shape[1]}"
            assert sloc.shape[0] == wloc.shape[0], f"{model_name}: scale rows {sloc.shape} != table rows {wloc.shape}"
            p.sm120_engram_source = (wloc, sloc)
            stats["engram_tables"] += 1
            stats["engram_rows"] += wloc.shape[0]
            _note(f"E2 {model_name}: {wloc.shape[0]} rows mapped from {os.path.basename(path)} "
                  f"(@{wloc.offset}, scale @{sloc.offset}), not copied")
        return stats

    _load_engram_tables._sm120_engram = True
    mod._load_engram_tables = _load_engram_tables
    _note("E2 hugging_face.weights._load_engram_tables: record shard locations instead of copying")


# ------------------------------------------------------------------------------------------------ E3
def _patch_runtime(mod):
    NgramEmbedding, EngramRuntime = mod.NgramEmbedding, mod.EngramRuntime
    if getattr(EngramRuntime.initialize, "_sm120_engram", False):
        return
    orig_init = NgramEmbedding.__init__

    def __init__(self, cfg, source_embedding):
        orig_init(self, cfg, source_embedding)
        p = source_embedding.embedding  # the Parameter: `.data` (self.weight) drops custom attributes
        self.sm120_source = getattr(p, "sm120_engram_source", None)
        self.sm120_layer_id = getattr(p, "layer_id", None)
        self.sm120_row_start = int(getattr(p, "row_start", 0))

    def initialize(self) -> None:
        import torch

        import sm120_engram

        embeddings = [layer.ngram_embedding for layer in self._layers]
        if self._buffer is None:
            self._buffer = self.allocate(embeddings)
        buf = self._buffer
        if not hasattr(buf, "attach_tables"):
            raise RuntimeError("SM120 Engram: deep_ep.EngramBuffer is not the SM120 shim (check PYTHONPATH)")
        mode = os.environ.get("SM120_ENGRAM_MODE", "auto")
        max_lookups = embeddings[0].max_num_tokens * embeddings[0].num_entries_per_token
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        tables, offsets = [], []
        for e in embeddings:
            if e.sm120_source is not None:
                wloc, sloc = e.sm120_source
                t = sm120_engram.open_table(None, e.sm120_layer_id, wloc=wloc, sloc=sloc, mode=mode, device=device,
                                            max_lookups=max_lookups)
                tables.append(t)
                offsets.append(0)  # the whole global table, whatever this rank's shard was
            else:
                # stock-loaded shard (SM120_ENGRAM_LAZY=0): the CPU tensor itself, never the relabelled view
                if e.engram_group.size() > 1:
                    raise NotImplementedError("SM120 Engram: stock-loaded shards need engram_parallel_size=1")
                sf = e.gather_sf() if e.sf is not None else None
                tables.append(sm120_engram.EngramHostTable.from_tensors(e.weight, sf, name=f"layer{e.sm120_layer_id}"))
                offsets.append(e.sm120_row_start)
        buf.attach_tables(tables, offsets)
        for e in embeddings:
            e.release_storage()
        desc = ", ".join(f"{t.name}:{t.kind}:{t.rows}rows" for t in tables)
        _note(f"E3 EngramRuntime.initialize: attached {desc}")
        if sm120_engram.start_prefetcher(tables) is not None:
            _note("E3 engram prefetch at request admission on (SM120_ENGRAM_PREFETCH=0 turns it off)")
        if os.environ.get("SM120_ENGRAM_RESIDENCY", "1") != "0":
            _log_residency(sm120_engram, tables)

    def _log_residency(sm120_engram, tables) -> None:
        """How much of the host-mode tables the page cache holds: rows that are not resident are random NVMe reads in
        every prefill (about 0.5 s per 4k-token prompt on the RTX 5090 test machine with a cold cache)."""
        host = [t for t in tables if t.kind == "host" and getattr(t, "_uring", None) is None and t.source is not None]
        if not host:
            return
        res = [t.resident() for t in host]
        if any(r is None for r in res):
            return
        size = sum(t.nbytes for t in host)
        avail = sm120_engram._mem_available()
        msg = (f"engram tables resident in the page cache: {sum(res) / 2**30:.1f} / {size / 2**30:.1f} GiB "
               f"({100 * sum(res) / max(size, 1):.0f} %); MemAvailable "
               f"{'?' if avail is None else f'{avail / 2**30:.1f} GiB'}")
        if sum(res) < 0.9 * size:
            msg += (" - lookups of non-resident rows read from disk in every prefill (SM120_ENGRAM_PREFAULT=1 if RAM "
                    "holds the tables, else SM120_ENGRAM_MODE=uring; docs/ENGRAM.md)")
        print(f"[sm120_engram] {msg}", flush=True)

    initialize._sm120_engram = True
    NgramEmbedding.__init__ = __init__
    EngramRuntime.initialize = initialize
    _note("E3 NgramEmbedding.__init__ / EngramRuntime.initialize: tables via sm120_engram")


# ------------------------------------------------------------------------------------------------ E5
class _HeaderSlice:
    def __init__(self, info):
        self._info = info

    def get_shape(self):
        return list(self._info["shape"])

    def get_dtype(self):
        return self._info["dtype"]


class _HeaderHandle:
    """The metadata part of a safetensors handle, from the header alone (no mapping)."""

    def __init__(self, path):
        import sm120_engram

        header, _ = sm120_engram.read_safetensors_header(path)
        self._header = {k: v for k, v in header.items() if k != "__metadata__"}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def keys(self):
        return list(self._header)

    def get_slice(self, name):
        return _HeaderSlice(self._header[name])

    def get_tensor(self, name):
        raise RuntimeError("SM120 header-only safetensors handle: tensor data is read elsewhere")


def _patch_safe_open(mod):
    orig = mod.safe_open
    if getattr(orig, "_sm120_engram", False):
        return

    def safe_open(path, *args, **kwargs):
        try:
            handle = orig(path, *args, **kwargs)
            handle.__enter__()
        except Exception as exc:  # noqa: BLE001
            if "mmap" not in str(exc) and "allocate memory" not in str(exc):
                raise
            _note(f"E5 safe_open({os.path.basename(str(path))}) failed ({exc}); using its header only")
            return _HeaderHandle(str(path))
        return _Entered(handle)

    safe_open._sm120_engram = True
    mod.safe_open = safe_open
    _note("E5 hugging_face.loading.safe_open: header-only fallback for unmappable shards")


class _Entered:
    """Wraps an already entered safetensors handle so `with` works once more."""

    def __init__(self, handle):
        self._h = handle

    def __enter__(self):
        return self._h

    def __exit__(self, *exc):
        return self._h.__exit__(*exc)


# ------------------------------------------------------------------------------------------------ E6
_ENGRAM_MODELS: list = []


def _patch_model_registry(mod):
    cls = mod.DeepseekModel
    if getattr(cls.__init__, "_sm120_engram", False):
        return
    orig = cls.__init__

    def __init__(self, *args, **kwargs):
        orig(self, *args, **kwargs)
        if getattr(self, "engram_runtime", None) is not None:
            _ENGRAM_MODELS.append(self)
            import sm120_engram  # hash metadata for the admission-time prefetch (E7)

            sm120_engram.set_hash_metadata(self.ngram_multipliers, self.ngram_vocab_sizes, self.ngram_offsets)

    orig_fwd = cls.forward

    def forward(self, kv_caches, input_metadata, *args, **kwargs):
        if getattr(self, "engram_runtime", None) is not None:  # E7: only prefill tokens count against the read-ahead
            import sm120_engram

            sm120_engram.set_step_is_prompt(bool(getattr(input_metadata, "is_prompt", True)))
        return orig_fwd(self, kv_caches, input_metadata, *args, **kwargs)

    __init__._sm120_engram = True
    cls.__init__, cls.forward = __init__, forward
    _note("E6 DeepseekModel.__init__ / forward: register engram models for graph prestaging, step kind for E7")


def prestage_for_replay(input_metadata) -> None:
    """Hash this step's windows and fill the engram row buffers before a graph replay (eager, current stream)."""
    if not _ENGRAM_MODELS or not _on("SM120_ENGRAM_PRESTAGE"):  # =0 only as a negative control (stale rows)
        return
    import tile_kernels
    import torch

    ids = getattr(input_metadata, "ngram_token_ids", None)
    if ids is None:
        return
    ids = ids.to("cuda", non_blocking=True) if not ids.is_cuda else ids
    mask = getattr(input_metadata, "images_seq_mask", None)
    if mask is not None and not mask.is_cuda:
        mask = mask.to("cuda", non_blocking=True)
    for m in _ENGRAM_MODELS:
        buf = m.engram_runtime._buffer
        if buf is None or getattr(buf.fetcher, "graph_mode", None) != "prestage":
            continue  # nothing captured yet, or the in-graph ("async") fetch serves itself
        hashes = tile_kernels.engram.engram_hash(ids.contiguous(), m.ngram_multipliers, m.ngram_vocab_sizes,
                                                 m.ngram_offsets, mask)
        buf.prestage(hashes)
        n = buf.fetcher.stats.get("prestages", 0)
        if n in (1, 10, 100, 1000):
            _note(f"E6 prestaged engram rows before graph replay #{n} (T={hashes.shape[1]})")


def _patch_graph_replay(mod):
    cls = mod.CudaGraphRunner
    if getattr(cls.replay, "_sm120_engram", False):
        return
    orig = cls.replay

    def replay(self, cuda_graph_data):
        prestage_for_replay(cuda_graph_data.input_metadata_source)
        return orig(self, cuda_graph_data)

    replay._sm120_engram = True
    cls.replay = replay
    _note("E6 CudaGraphRunner.replay: engram rows prestaged before every replay")


# ------------------------------------------------------------------------------------------------ E7
def _patch_admission(mod):
    """E7: LLMEngine.add_request / abort_request report prompts to the Engram prefetcher (sm120_engram.EngramPrefetcher):
    every entry point (offline LLM, server) gets the prefetch; the server also calls it earlier, at submit."""
    cls = mod.LLMEngine
    if getattr(cls.add_request, "_sm120_engram", False):
        return
    orig_add, orig_abort = cls.add_request, cls.abort_request

    def add_request(self, /, user_id, request_id, sampling_params, prompt_token_ids, *args, **kwargs):
        seq = orig_add(self, user_id, request_id, sampling_params, prompt_token_ids, *args, **kwargs)
        if seq is not None:
            import sm120_engram

            sm120_engram.prefetch_prompt(request_id, prompt_token_ids, admitted=True)
        return seq

    def abort_request(self, request_id):
        import sm120_engram

        for rid in [request_id] if isinstance(request_id, str) else list(request_id):
            sm120_engram.prefetch_cancel(rid)
        return orig_abort(self, request_id)

    add_request._sm120_engram = True
    cls.add_request, cls.abort_request = add_request, abort_request
    _note("E7 LLMEngine.add_request / abort_request: prompts reported to the engram prefetcher")


# ------------------------------------------------------------------------------------------------ E9
def _patch_step_check(mod):
    """E9: a failed in-graph engram fetch fails the engine step instead of routing its outputs."""
    cls = mod.LLMEngine
    if getattr(cls.step, "_sm120_engram", False):
        return
    import sys

    def wrap(orig):
        def run(self, *args, **kwargs):
            out = orig(self, *args, **kwargs)
            eng = sys.modules.get("sm120_engram")
            if eng is not None and eng._FETCHERS:
                eng.check_graph_fetch()
            return out

        run._sm120_engram = True
        return run

    cls.step = wrap(cls.step)
    cls.flush_inflight_reqs = wrap(cls.flush_inflight_reqs)
    _note("E9 LLMEngine.step / flush_inflight_reqs: a failed in-graph engram fetch fails the step")


# ------------------------------------------------------------------------------------------------ E8
def _patch_init_cache(mod):
    """E8: DeepInfer sizes the KV cache from the free GPU memory after profiling. In pinned mode the registration of the
    shared segment took some of it (the driver's page tables, ~2 MiB per GiB), so a budget that just fitted in host mode
    can fail; say so instead of only the stock "Try increasing `gpu_memory_utilization`"."""
    cls = mod.LLMEngine
    if getattr(cls._init_cache, "_sm120_engram", False):
        return
    orig = cls._init_cache

    def _init_cache(self, *args, **kwargs):
        try:
            return orig(self, *args, **kwargs)
        except ValueError as e:
            import sys

            sh = getattr(sys.modules.get("sm120_engram"), "_SHM", None) or {}
            if not sh.get("registered") or "gpu_memory_utilization" not in str(e):
                raise
            util = getattr(getattr(self, "cache_config", None), "gpu_memory_utilization", None)
            cur = f" (now {util:.2f})" if isinstance(util, float) else ""
            raise ValueError(
                f"{e}\nSM120_ENGRAM_MODE=pinned: registering the {sh['size'] / 2**30:.1f} GiB engram segment took "
                f"{sh.get('gpu_cost', 0) / 2**20:.0f} MiB of GPU memory on this rank (the driver's page tables for the "
                f"mapping), which the KV cache does not get. Raise --gpu-memory-utilization{cur}, e.g. to 0.95, or lower "
                f"--max-model-len (docs/MULTI_GPU_RUNBOOK.md §3).") from e

    _init_cache._sm120_engram = True
    cls._init_cache = _init_cache
    _note("E8 LLMEngine._init_cache: pinned-mode hint on KV sizing errors")


# ------------------------------------------------------------------------------------------------ E4
def _patch_tile_kernels(mod):
    import sm120_engram

    mod.engram_hash = sm120_engram.engram_hash
    mod.fused_weight = sm120_engram.fused_weight
    mod.engram_gate_fwd = sm120_engram.engram_gate_fwd
    _note("E4 tile_kernels.engram.{engram_hash,fused_weight,engram_gate_fwd} -> torch (SM120_ENGRAM_TK=torch)")


# ------------------------------------------------------------------------------------------------ install
def install() -> None:
    """Register the hooks (idempotent). Requires sm120_patches (the hook machinery)."""
    global _INSTALLED
    if _INSTALLED or not _on("SM120_ENGRAM_PATCHES"):
        return
    import sm120_patches

    if not sm120_patches.enabled():
        return
    _INSTALLED = True
    when = sm120_patches.when_imported
    if _on("SM120_ENGRAM_LAZY"):
        when("deepinfer.model_executor.models.deepseek")(_patch_materialize)
        when("deepinfer.hugging_face.weights")(_patch_load_tables)
        when("deepinfer.hugging_face.loading")(_patch_safe_open)
    when("deepinfer.model_executor.models.v4_modules.engram")(_patch_runtime)
    when("deepinfer.model_executor.models.deepseek")(_patch_model_registry)
    when("deepinfer.worker.cuda_graph_runner")(_patch_graph_replay)
    when("deepinfer.engine.llm_engine")(_patch_admission)
    when("deepinfer.engine.llm_engine")(_patch_init_cache)
    when("deepinfer.engine.llm_engine")(_patch_step_check)
    if os.environ.get("SM120_ENGRAM_TK", "").strip().lower() == "torch":
        when("tile_kernels.engram")(_patch_tile_kernels)
