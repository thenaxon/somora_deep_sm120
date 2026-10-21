"""b12x 1.5.1 compressed_sparse_mla (cache_format="deepseek_v41") backend for FlashMLA sparse decode.

Calling convention copied from the validated feasibility test
tests/test_attn_b12x_compat.py (run_mode, "DeepInfer
physical slots" variant; PASSED on RTX 5090, bf16 compute cos 0.999999):

  * main cache  (k_cache, 528 B/token V4.1 FP8)  -> b12x "swa" cache, page = k_cache.shape[1]
  * extra cache (extra_k_cache, 288 B/token FP4) -> b12x "indexed" cache, page = extra.shape[1]
  * caches are passed as zero-copy 2-D [num_pages, page_bytes] views with the allocator page stride
  * indices are DeepInfer physical slots (block * page + offset) passed as given (-1 = invalid, which
    b12x skips) with per-row lengths = topk_length (the earlier prefix compaction is still
    available with SM120_ATTN_DECODE_COMPACT=1).

Plans are cached per (heads, row bucket, widths, page sizes, cache descriptors, sink, compute
mode). q / indices / lengths / out live in persistent per-plan buffers padded to the row bucket so
every call presents the identical storage ABI to the prepared plan (padded rows: length 0).

CUDA graphs: a call issues no host syncs and allocates nothing data-dependent, so it can
be captured once the plan exists. Plans are created (and compiled) on the first *eager* call of a
shape; DeepInfer's CudaGraphRunner always runs one eager warm-up per batch size before capturing,
which is exactly that call. Creating a plan while the stream is capturing raises (b12x refuses to
compile under capture), which the dispatcher turns into a torch fallback (or an error with
SM120_ATTN_STRICT=1). The staging buffers are persistent per plan entry, so graphs of different
batch sizes that share a row bucket share them; that is safe because graphs replay in stream order.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, replace

import torch

from . import _config as cfg
from .quant import V41_FP4_BYTES, V41_FP8_BYTES

_LOCK = threading.Lock()
_MODS: dict = {}
_ENTRIES: dict = {}
_FAILED: set = set()


def _modules():
    if not _MODS:
        from b12x import preparation as prep
        from b12x.attention import compressed_sparse_mla as mla
        from b12x.attention.compressed_sparse_mla import _tuning as tuning
        _MODS.update(mla=mla, prep=prep, tuning=tuning)
    return _MODS


_SUPPORTED_CACHE: dict = {}


def _device_supported(device: torch.device) -> bool:
    key = str(device)
    if key not in _SUPPORTED_CACHE:
        try:
            _SUPPORTED_CACHE[key] = bool(_modules()["mla"].is_supported(device))
        except Exception as exc:  # noqa: BLE001
            cfg.log_once(("b12x_import", str(exc)[:200]), f"b12x compressed_sparse_mla unavailable: {exc}")
            _SUPPORTED_CACHE[key] = False
    return _SUPPORTED_CACHE[key]


def unsupported_reason(q: torch.Tensor, k_cache: torch.Tensor, extra_k_cache: torch.Tensor | None,
                       d_v: int) -> str | None:
    if q.device.type != "cuda":
        return "not on CUDA"
    if q.dtype != torch.bfloat16 or q.dim() != 3 or q.shape[-1] != 512 or d_v != 512:
        return f"q must be bf16 [rows, heads, 512] with d_v 512 (got {q.dtype} {tuple(q.shape)}, d_v={d_v})"
    if q.shape[1] % 16 != 0:
        return f"num heads {q.shape[1]} not a multiple of 16"
    if k_cache.dim() != 4 or k_cache.shape[2] != 1 or k_cache.shape[3] != V41_FP8_BYTES:
        return f"main cache must be [pages, page, 1, 528] (got {tuple(k_cache.shape)})"
    if extra_k_cache is not None and (extra_k_cache.dim() != 4 or extra_k_cache.shape[2] != 1
                                      or extra_k_cache.shape[3] != V41_FP4_BYTES):
        return f"extra cache must be [pages, page, 1, 288] (got {tuple(extra_k_cache.shape)})"
    for c in (k_cache, extra_k_cache):
        if c is not None and (c.stride(3) != 1 or c.stride(1) != c.shape[3]):
            return "cache records must be packed within a page"
    if not _device_supported(q.device):
        return "b12x compressed_sparse_mla not supported on this device / not importable"
    return None


def _page_view(c: torch.Tensor) -> torch.Tensor:
    u = c if c.dtype == torch.uint8 else c.view(torch.uint8)
    return u.as_strided((u.shape[0], u.shape[1] * u.shape[3]), (u.stride(0), 1), u.storage_offset())


def _alignment(t: torch.Tensor) -> int:
    p = t.data_ptr()
    return min(16, p & -p) if p else 16


def _desc(t: torch.Tensor) -> tuple:
    return (tuple(t.shape), tuple(t.stride()), _alignment(t))


def _compact(idx: torch.Tensor, length: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Move valid entries (>= 0 and pos < length) to a prefix (stable), -1 elsewhere."""
    valid = idx >= 0
    if length is not None:
        pos = torch.arange(idx.shape[-1], device=idx.device).view(1, -1)
        valid &= pos < length.long().view(-1, 1)
    order = torch.sort((~valid).to(torch.int32), dim=-1, stable=True).indices
    out = torch.where(valid, idx.to(torch.int32), torch.full_like(idx, -1, dtype=torch.int32)).gather(-1, order)
    return out.contiguous(), valid.sum(-1).to(torch.int32).contiguous()


_CONST: dict = {}


def _const(key, make):
    """Cached read-only constant tensor (created eagerly - DeepInfer's warm-up precedes every capture)."""
    t = _CONST.get(key)
    if t is None:
        t = make()
        if not cfg.capturing():  # a tensor made under capture is only filled on replay: do not cache it
            _CONST[key] = t
    return t


def _raw(idx: torch.Tensor, length: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Indices as given (-1 holes kept), length = topk_length or the full width. No copies when the inputs
    already are contiguous int32 (b12x clamps the lengths to the width itself)."""
    i = idx if (idx.dtype == torch.int32 and idx.is_contiguous()) else idx.to(torch.int32).contiguous()
    R, W = i.shape
    if length is None:
        return i, _const(("len", str(i.device), R, W),
                         lambda: torch.full((R,), W, dtype=torch.int32, device=i.device))
    ln = length if (length.dtype == torch.int32 and length.is_contiguous()) else length.to(torch.int32).contiguous()
    return i, ln


def _bucket(rows: int) -> int:
    return cfg.next_pow2(rows, 8)


_ARENA: dict = {}


def _shared_scratch(device, specs) -> torch.Tensor:
    """One decode scratch shared by every plan entry (calls and graph replays are stream ordered).

    Grow-only and never freed: a larger requirement allocates a new arena for the new entry while the
    old buffer stays alive for the entries (and captured CUDA graphs) that use it. DeepInfer warms up and
    captures the largest batch first, so normally one arena serves all row buckets (MTP verify batches
    need up to max_num_seqs * (1 + nextn) rows, which made per-entry scratch run out of memory)."""
    if len(specs) != 1 or specs[0].dtype != torch.uint8:
        return torch.empty(specs[0].shape, dtype=specs[0].dtype, device=device)
    need = int(specs[0].shape[0])
    key = str(device)
    buf = _ARENA.get(key)
    if buf is None or buf.numel() < need:
        buf = torch.empty((need,), dtype=torch.uint8, device=device)
        _ARENA[key] = buf
    return buf


def prepare_plan(plan, name: str, device) -> tuple:
    """Prepare (compile / load) a b12x plan with its default config; returns its scratch specs.

    Must not run under CUDA graph capture (b12x raises there)."""
    if cfg.capturing():
        raise RuntimeError(f"b12x plan {name!r} is not prepared and the stream is capturing a CUDA graph; "
                           "run the same shapes eagerly once before capture")
    prep = _modules()["prep"]
    holder: dict = {}

    def prepare_call(state, holder=holder):
        holder["state"] = state
        return prep.PreparedCall(run=lambda: None)

    req = plan.request(name=name, prepare_call=prepare_call)
    prepare_default = getattr(prep, "prepare_default", None)
    if prepare_default is not None:
        prepare_default(req)
    else:
        session = prep.PreparationSession(device=device, autotune=False)
        session.__enter__()
        _MODS.setdefault("sessions", []).append(session)
        session.prepare((req,))
    try:
        return tuple(plan.memory_requirements().scratch)
    except Exception:  # noqa: BLE001
        state = holder.get("state")
        if state is not None and hasattr(state, "scratch_plan"):
            return tuple(state.scratch_plan.scratch_specs())
        if state is not None and hasattr(state, "layout"):
            return tuple(state.layout.scratch_specs())
        raise


@dataclass
class _Entry:
    plan: object
    binding: object
    q: torch.Tensor
    out: torch.Tensor
    swa_idx: torch.Tensor
    swa_len: torch.Tensor
    idx_idx: torch.Tensor | None
    idx_len: torch.Tensor | None
    lse: bool
    scratch: list


def _build_entry(device, H: int, bucket: int, T1: int, T2: int, page1: int, page2: int | None,
                 swa2d: torch.Tensor, idx2d: torch.Tensor | None, sink: torch.Tensor | None, compute: str,
                 want_lse: bool) -> _Entry:
    m = _modules()
    mla, tuning = m["mla"], m["tuning"]
    q = torch.zeros((bucket, H, 512), dtype=torch.bfloat16, device=device)
    out = torch.empty((bucket, H, 512), dtype=torch.bfloat16, device=device)
    swa_idx = torch.full((bucket, T1), -1, dtype=torch.int32, device=device)
    swa_len = torch.zeros((bucket,), dtype=torch.int32, device=device)
    idx_idx = torch.full((bucket, T2), -1, dtype=torch.int32, device=device) if idx2d is not None else None
    idx_len = torch.zeros((bucket,), dtype=torch.int32, device=device) if idx2d is not None else None
    caps = mla.Caps(
        device=device, num_q_heads=H, max_q_rows=bucket, max_width=T1 + T2,
        swa_width=T1, indexed_width=T2, swa_page_size=page1,
        indexed_page_size=page2 if idx2d is not None else None,
        max_page_table_width=1, mode="decode", decode_row_capacity=bucket,
        cache_format="deepseek_v41", use_cuda_graph=False,
    )
    inv_kwargs = dict(q=q, swa_k_cache=swa2d, indexed_k_cache=idx2d, attn_sink=sink, out=out)
    if want_lse:
        inv_kwargs.update(return_lse=True, lse_scale="natural")
    invocation = mla.invocation_from_tensors(**inv_kwargs)
    plan0 = mla.plan(caps, invocation=invocation)
    default_cfg = tuning._default_config(plan0.query, None)
    plan = plan0
    if compute != default_cfg.v41_compute_mode:
        override = replace(default_cfg, v41_compute_mode=compute, v41_heads_per_block=16)
        plan = mla.plan(caps, invocation=invocation, override=override)
    specs = prepare_plan(plan, f"sm120_attn.decode.h{H}.r{bucket}.w{T1}+{T2}.{compute}", device)
    scratch = [_shared_scratch(device, specs)]
    bkw = dict(q=q, swa_indices=swa_idx, swa_lengths=swa_len)
    if idx2d is not None:
        bkw.update(indexed_indices=idx_idx, indexed_lengths=idx_len)
    binding = mla.bind(plan, scratch=scratch, **bkw)
    cfg.log_once(("b12x_plan", H, bucket, T1, T2, page1, page2, compute),
                 f"b12x compressed_sparse_mla plan: heads={H} rows<={bucket} swa_width={T1} indexed_width={T2} "
                 f"pages=({page1},{page2}) compute={compute} default_cfg={default_cfg}")
    return _Entry(plan=plan, binding=binding, q=q, out=out, swa_idx=swa_idx, swa_len=swa_len,
                  idx_idx=idx_idx, idx_len=idx_len, lse=want_lse, scratch=scratch)


def sparse_decode_b12x(q: torch.Tensor, k_cache: torch.Tensor, indices: torch.Tensor,
                       topk_length: torch.Tensor | None, sm_scale: float, attn_sink: torch.Tensor | None,
                       extra_k_cache: torch.Tensor | None, extra_indices: torch.Tensor | None,
                       extra_topk_length: torch.Tensor | None):
    """q [R,H,512] bf16 -> (out bf16 [R,H,512] (fresh tensor), lse [R,H] fp32 (NaN unless enabled))."""
    mla = _modules()["mla"]
    R, H, _ = q.shape
    device = q.device
    has_extra = extra_k_cache is not None and extra_indices is not None
    # b12x skips -1 entries itself (verified on SM120 by every decode test incl. the CUDA-graph one), so the
    # index lists are passed as given with length = topk_length; SM120_ATTN_DECODE_COMPACT=1 restores the
    # Fallback compaction (valid entries moved to a prefix by a stable sort).
    if os.environ.get("SM120_ATTN_DECODE_COMPACT", "0") != "1":
        i1, l1 = _raw(indices, topk_length)
        if has_extra:
            i2, l2 = _raw(extra_indices, extra_topk_length)
    else:
        i1, l1 = _compact(indices, topk_length)
        if has_extra:
            i2, l2 = _compact(extra_indices, extra_topk_length)
    T1 = int(i1.shape[1])
    T2 = int(i2.shape[1]) if has_extra else 0
    swa2d = _page_view(k_cache)
    idx2d = _page_view(extra_k_cache) if has_extra else None
    page1 = int(k_cache.shape[1])
    page2 = int(extra_k_cache.shape[1]) if has_extra else None
    bucket = _bucket(R)
    compute = cfg.compute_mode()
    want_lse = cfg.b12x_lse()
    sink = None if attn_sink is None else attn_sink.float().contiguous()
    key = (str(device), H, bucket, T1, T2, page1, page2, _desc(swa2d), None if idx2d is None else _desc(idx2d),
           sink is not None, compute, want_lse)
    if key in _FAILED:
        raise RuntimeError("b12x plan previously failed for this configuration")
    with _LOCK:
        entry = _ENTRIES.get(key)
        if entry is None:
            try:
                entry = _build_entry(device, H, bucket, T1, T2, page1, page2, swa2d, idx2d, sink, compute, want_lse)
            except Exception:
                if not cfg.capturing():  # a capture-time miss may still succeed eagerly later
                    _FAILED.add(key)
                raise
            _ENTRIES[key] = entry
    run_kw = dict(swa_k_cache=swa2d, indexed_k_cache=idx2d, swa_page_size=page1, sm_scale=float(sm_scale),
                  attn_sink=sink, cache_format="deepseek_v41")
    if has_extra:
        run_kw["indexed_page_size"] = page2
    if want_lse:
        run_kw.update(return_lse=True, lse_scale="natural")
    direct = (os.environ.get("SM120_ATTN_DECODE_STAGE", "0") != "1" and q.is_contiguous()
              and q.data_ptr() % 16 == 0 and not want_lse)
    if direct:
        # Bind the caller's tensors (as LIL vLLM does per call) and let b12x write a fresh output: no staging
        # copies, no tail clearing, no clone - about 10 tiny kernels per layer and decode step less.
        out = torch.empty((R, H, 512), dtype=torch.bfloat16, device=device)
        bkw = dict(q=q, swa_indices=i1, swa_lengths=l1)
        if has_extra:
            bkw.update(indexed_indices=i2, indexed_lengths=l2)
        binding = mla.bind(entry.plan, scratch=entry.scratch, **bkw)
        mla.run(plan=entry.plan, binding=binding, out=out, **run_kw)
        lse = _const(("nan_lse", str(device), R, H),
                     lambda: torch.full((R, H), float("nan"), dtype=torch.float32, device=device))
        return out, lse
    # stage inputs into the plan-owned buffers
    entry.q[:R].copy_(q)
    entry.swa_idx[:R].copy_(i1)
    entry.swa_len[:R].copy_(l1)
    if has_extra:
        entry.idx_idx[:R].copy_(i2)
        entry.idx_len[:R].copy_(l2)
    if R < entry.swa_len.shape[0]:
        # Padded rows attend to nothing. Always cleared (not only after a larger call) so that a
        # captured CUDA graph does not depend on which call ran before it.
        entry.swa_len[R:].zero_()
        if has_extra:
            entry.idx_len[R:].zero_()
    res = mla.run(plan=entry.plan, binding=entry.binding, out=entry.out, **run_kw)
    lse = None
    if isinstance(res, tuple):
        o = res[0]
        if want_lse and len(res) > 1 and res[1] is not None:
            lse = res[1]
    else:
        o = res
    out = o[:R].clone()
    if lse is not None:
        lse = lse.reshape(-1, H)[:R].float().clone()
        lse = torch.where(torch.isneginf(lse), torch.full_like(lse, float("inf")), lse)
    else:
        lse = torch.full((R, H), float("nan"), dtype=torch.float32, device=device)
    return out, lse


def clear() -> None:
    with _LOCK:
        _ENTRIES.clear()
        _FAILED.clear()
