"""b12x 1.5.1 compressed_sparse_mla (cache_format="deepseek_v41", mode="extend") backend for
FlashMLA sparse prefill (flash_mla_sparse_fwd and the attention core of the fused prefill).

FlashMLA's prefill contract is a bf16 KV buffer [s_kv, 512] plus flat indices into it. b12x has
no bf16-KV sparse kernel; its prefill ("extend" mode, the single-pass MG kernel that LIL vLLM
uses for V4.1 prefill) reads the paged V4.1 caches. DeepInfer assembles the prefill KV as
[loaded SWA cache | current SWA kv | loaded compressed cache] (attn.py:813-857) and:

  * SWA tokens (cached and current) are FP8-exact: the current-chunk kv goes through
    CastArgs("e4m3", ue8m0, cast_back=True) (attn.py:412-421) and cached ones are loaded from the
    528 B FP8 records;
  * compressed tokens are FP4-exact: they are dequantized from 288 B FP4 records.

So we re-encode the buffer into two transient caches (528 B FP8 "swa" and 288 B FP4 "indexed",
page 64, slot = kv row) and route every selected index to the cache whose encoding reproduces
that token exactly. With SM120_ATTN_PREFILL_KV=auto (default) a token goes to the FP4 cache iff
its FP4 round trip is bit-exact, else to the FP8 cache. For tokens that are FP8- or FP4-exact
(everything DeepInfer produces) the KV the kernel sees is bit-identical to FlashMLA's bf16 KV;
any other token is FP8-quantized (logged with SM120_ATTN_DEBUG=1). SM120_ATTN_PREFILL_KV=fp8
puts everything into the FP8 cache (single index list; lossy for FP4-origin tokens).

Remaining numerical differences vs FlashMLA (see docs/ATTENTION.md):
  * the fused prefill folds the RMS factor into q in bf16 (FlashMLA applies it to the fp32 logits);
  * max_logits is not produced (NaN); lse is NaN unless SM120_ATTN_B12X_LSE=1. DeepInfer
    discards both at every call site;
  * SM120_ATTN_COMPUTE=fp8 switches the kernel to FP8-internal math (faster, less accurate).

Eager only (prefill is never graph captured in DeepInfer): one host sync per call to size the
index lists. Plans are cached per (heads, list widths, cache pages, sink, compute, lse) with a
fixed row capacity (SM120_ATTN_PREFILL_ROWS; longer prefills loop over row chunks); widths and
cache sizes are bucketed to powers of two so the number of compiled plans stays small. One growing scratch arena is shared by all prefill plans (calls are stream ordered).
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, replace

import torch

from . import _config as cfg
from .b12x_mla import _alignment, _device_supported, _modules, prepare_plan
from .quant import (V41_FP4_BYTES, V41_FP8_BYTES, dequant_v41_fp4_records, quantize_v41_fp4_records,
                    quantize_v41_fp8_records)

PAGE = 64
_LOCK = threading.Lock()
_PLANS: dict = {}
_FAILED: set = set()
_ARENA: dict = {}
_CACHE_PAGES: dict = {}


def unsupported_reason(q: torch.Tensor, kv: torch.Tensor, d_v: int) -> str | None:
    if q.device.type != "cuda":
        return "not on CUDA"
    if q.dtype != torch.bfloat16 or q.dim() != 3 or q.shape[-1] != 512 or d_v != 512:
        return f"q must be bf16 [s_q, heads, 512] with d_v 512 (got {q.dtype} {tuple(q.shape)}, d_v={d_v})"
    if q.shape[1] % 16 != 0:
        return f"num heads {q.shape[1]} not a multiple of 16"
    if kv.dim() != 2 or kv.shape[-1] != 512:
        return f"kv must be [s_kv, 512] (got {tuple(kv.shape)})"
    if kv.shape[0] >= 2 ** 31 // PAGE:
        return "kv too long for int32 slots"
    if not _device_supported(q.device):
        return "b12x compressed_sparse_mla not supported on this device / not importable"
    return None


def _arena(device, nbytes: int) -> torch.Tensor:
    key = str(device)
    buf = _ARENA.get(key)
    if buf is None or buf.numel() < nbytes:
        buf = torch.empty((cfg.align(max(nbytes, 1), 1 << 20),), dtype=torch.uint8, device=device)
        _ARENA[key] = buf
    return buf


# ---------------------------------------------------------------------------------------------
# transient caches
# ---------------------------------------------------------------------------------------------
def _encode_caches(kv: torch.Tensor, mode: str):
    """kv [s_kv, 512] -> (fp8 cache [P, 64*528], fp4 cache [P, 64*288] | None, is_fp4 bool [s_kv] | None).

    Slot of kv row i is i in both caches. P is grow-only (power of two, per device) because the
    cache shape is part of the plan's compile identity."""
    n = kv.shape[0]
    dev = kv.device
    pages = max(_CACHE_PAGES.get(str(dev), 0), cfg.next_pow2(max(1, (n + PAGE - 1) // PAGE), 64))
    _CACHE_PAGES[str(dev)] = pages
    fp8 = torch.empty((pages, PAGE * V41_FP8_BYTES), dtype=torch.uint8, device=dev)
    fp4 = None if mode == "fp8" else torch.empty((pages, PAGE * V41_FP4_BYTES), dtype=torch.uint8, device=dev)
    is_fp4 = None
    done = False
    if cfg.fused_impl() == "triton":
        from . import fused_triton
        if fused_triton.available():
            try:  # one Triton pass: both records + the FP4-exact flag
                is_fp4 = fused_triton.encode_v41(kv, fp8.view(-1, V41_FP8_BYTES),
                                                 None if fp4 is None else fp4.view(-1, V41_FP4_BYTES))
                done = True
            except Exception as exc:  # noqa: BLE001
                if cfg.strict():
                    raise
                cfg.log_once(("encode_v41_fail", type(exc).__name__), f"triton KV encoder failed ({exc}); using torch")
    if not done:
        fp8.view(-1, V41_FP8_BYTES)[:n] = quantize_v41_fp8_records(kv)
        if fp4 is not None:
            rec4 = quantize_v41_fp4_records(kv)
            is_fp4 = (dequant_v41_fp4_records(rec4) == kv.float()).all(-1)
            fp4.view(-1, V41_FP4_BYTES)[:n] = rec4
    if fp4 is None:
        return fp8, None, None
    if cfg.debug():
        from .quant import dequant_v41_fp8_records
        fp8_exact = (dequant_v41_fp8_records(fp8.view(-1, V41_FP8_BYTES)[:n]) == kv.float()).all(-1)
        lossy = int((~(fp8_exact | is_fp4)).sum().item())
        cfg.log_once(("prefill_kv_classes", n, lossy),
                     f"b12x prefill KV: {n} tokens, {int(is_fp4.sum().item())} FP4-exact, "
                     f"{lossy} neither FP8- nor FP4-exact (FP8-quantized)")
    return fp8, fp4, is_fp4


def _compact_mask(idx: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    order = torch.sort((~valid).to(torch.int32), dim=-1, stable=True).indices
    out = torch.where(valid, idx, torch.full_like(idx, -1)).gather(-1, order)
    return out, valid.sum(-1, dtype=torch.int32)


def _width(n: int, cap: int) -> int:
    return min(cfg.next_pow2(max(n, 1), 64), cfg.next_pow2(max(cap, 1), 64))


# ---------------------------------------------------------------------------------------------
# plans
# ---------------------------------------------------------------------------------------------
@dataclass
class _Plan:
    plan: object
    specs: tuple


def _get_plan(device, H: int, rows: int, T1: int, T2: int, fp8_cache: torch.Tensor,
              fp4_cache: torch.Tensor | None, has_sink: bool, compute: str, want_lse: bool) -> _Plan:
    key = (str(device), H, rows, T1, T2, tuple(fp8_cache.shape), None if fp4_cache is None else tuple(fp4_cache.shape),
           has_sink, compute, want_lse)
    if key in _FAILED:
        raise RuntimeError("b12x prefill plan previously failed for this configuration")
    with _LOCK:
        p = _PLANS.get(key)
        if p is not None:
            return p
        try:
            m = _modules()
            mla, tuning = m["mla"], m["tuning"]
            caps = mla.Caps(
                device=device, num_q_heads=H, max_q_rows=rows, max_width=T1 + T2,
                swa_width=T1, indexed_width=T2, swa_page_size=PAGE,
                indexed_page_size=PAGE if fp4_cache is not None else None,
                max_page_table_width=1, mode="extend", cache_format="deepseek_v41", use_cuda_graph=False,
            )

            def desc(t):
                return dict(shape=tuple(t.shape), stride=tuple(t.stride()), alignment=_alignment(t),
                            dtype=str(t.dtype).removeprefix("torch."))

            invocation = mla.invocation_from_descriptors(
                q=dict(shape=(rows, H, 512), stride=(H * 512, 512, 1), alignment=16, dtype="bfloat16"),
                swa_cache=desc(fp8_cache), indexed_cache=None if fp4_cache is None else desc(fp4_cache),
                attn_sink_present=has_sink, return_lse=want_lse, lse_scale="natural" if want_lse else "base2",
                output_mode="provided")
            plan = mla.plan(caps, invocation=invocation)
            default_cfg = tuning._default_config(plan.query, None)
            if compute != default_cfg.v41_compute_mode:
                plan = mla.plan(caps, invocation=invocation,
                                override=replace(default_cfg, v41_compute_mode=compute, v41_heads_per_block=16))
            specs = prepare_plan(plan, f"sm120_attn.prefill.h{H}.r{rows}.w{T1}+{T2}.{compute}", device)
        except Exception:
            _FAILED.add(key)
            raise
        cfg.log_once(("b12x_prefill_plan",) + key[1:],
                     f"b12x compressed_sparse_mla extend plan: heads={H} rows<={rows} swa_width={T1} "
                     f"indexed_width={T2} pages=({fp8_cache.shape[0]},{None if fp4_cache is None else fp4_cache.shape[0]}) "
                     f"compute={compute} default_cfg={default_cfg}")
        p = _Plan(plan=plan, specs=specs)
        _PLANS[key] = p
        return p


# ---------------------------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------------------------
def sparse_prefill_b12x(q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, sm_scale: float,
                        attn_sink: torch.Tensor | None, topk_length: torch.Tensor | None):
    """q [s_q,H,512] bf16 (any q scale already folded in), kv [s_kv,512] bf16, indices [s_q,topk].

    Returns (out bf16 [s_q,H,512], max_logits [s_q,H] (NaN), lse [s_q,H] fp32 (NaN unless enabled)).
    """
    mla = _modules()["mla"]
    s_q, H, _ = q.shape
    s_kv = kv.shape[0]
    topk = indices.shape[-1]
    dev = q.device
    compute = cfg.compute_mode()
    want_lse = cfg.b12x_lse()
    out = torch.empty((s_q, H, 512), dtype=torch.bfloat16, device=dev)
    max_logits = torch.full((s_q, H), float("nan"), dtype=torch.float32, device=dev)
    lse = torch.full((s_q, H), float("nan"), dtype=torch.float32, device=dev)
    if s_q == 0 or s_kv == 0 or topk == 0:
        out.zero_()
        lse.fill_(float("inf"))
        return out, max_logits.fill_(float("-inf")), lse
    q = q.contiguous()
    kv = kv.contiguous()
    fp8_cache, fp4_cache, is_fp4 = _encode_caches(kv, cfg.prefill_kv_mode())

    idx = indices.to(torch.int32)
    valid = (idx >= 0) & (idx < s_kv)
    if topk_length is not None:
        pos = torch.arange(topk, device=dev).view(1, -1)
        valid &= pos < topk_length.to(torch.long).view(-1, 1)
    # Widths of the two index lists: the longest list is read back (one host sync per layer, 40 per prefill step)
    # and the plan narrowed to it. SM120_ATTN_PREFILL_SYNC=0 uses the static bound (top-k) instead: no sync, but the
    # extend kernel's cost follows the plan width, not the row lengths (RTX 5090, ckpt_full, 4096-token step: GPU
    # time 2x, TTFT 0.97 -> 1.8 s), so it only pays where the host, not the GPU, bounds the step.
    sync = os.environ.get("SM120_ATTN_PREFILL_SYNC", "1") != "0"
    if fp4_cache is not None:
        in_fp4 = is_fp4[idx.clamp(0, s_kv - 1).long()] & valid
        i1, l1 = _compact_mask(idx, valid & ~in_fp4)
        i2, l2 = _compact_mask(idx, in_fp4)
        if sync:
            m1, m2 = (int(v) for v in torch.stack((l1.max(), l2.max())).tolist())
            if m2 == 0:  # no FP4-exact token selected: single-cache plan
                fp4_cache = None
        else:
            m1 = m2 = topk
    else:
        i1, l1 = _compact_mask(idx, valid)
        m1, m2 = (int(l1.max().item()) if sync else topk), 0
    T1 = _width(m1, topk)
    T2 = _width(m2, topk) if fp4_cache is not None else 0
    i1 = i1[:, :T1] if T1 <= topk else torch.nn.functional.pad(i1, (0, T1 - topk), value=-1)
    i1 = i1.contiguous()
    if fp4_cache is not None:
        i2 = (i2[:, :T2] if T2 <= topk else torch.nn.functional.pad(i2, (0, T2 - topk), value=-1)).contiguous()
    rows_cap = cfg.next_pow2(cfg.prefill_rows(), 16)  # fixed capacity: one compile per width pair
    p = _get_plan(dev, H, rows_cap, T1, T2, fp8_cache, fp4_cache, attn_sink is not None, compute, want_lse)
    nbytes = sum(int(s.shape[0]) for s in p.specs)
    scratch = _arena(dev, nbytes)
    sink = None if attn_sink is None else attn_sink.float().contiguous()
    for a in range(0, s_q, rows_cap):
        b = min(a + rows_cap, s_q)
        bkw = dict(q=q[a:b], swa_indices=i1[a:b], swa_lengths=l1[a:b].contiguous())
        if fp4_cache is not None:
            bkw.update(indexed_indices=i2[a:b], indexed_lengths=l2[a:b].contiguous())
        binding = mla.bind(p.plan, scratch=scratch, **bkw)
        run_kw = dict(swa_k_cache=fp8_cache, indexed_k_cache=fp4_cache, swa_page_size=PAGE, sm_scale=float(sm_scale),
                      attn_sink=sink, out=out[a:b], cache_format="deepseek_v41")
        if fp4_cache is not None:
            run_kw["indexed_page_size"] = PAGE
        if want_lse:
            run_kw.update(return_lse=True, lse_scale="natural")
        res = mla.run(plan=p.plan, binding=binding, **run_kw)
        if want_lse and isinstance(res, tuple) and len(res) > 1 and res[1] is not None:
            lse[a:b] = res[1].reshape(b - a, H).float()
    # FlashMLA contract for queries without any valid key: out 0, lse +inf, max -inf
    lonely = (l1 + (l2 if fp4_cache is not None else 0)) == 0
    # Queries without any valid key: b12x already writes out = 0 (checked with and without sink on SM120,
    # test_prefill_b12x_*); only the small LSE / max tensors need FlashMLA's +inf / -inf.
    # SM120_ATTN_PREFILL_LONELY_FIX=1 also forces out = 0 (one extra pass over out).
    if os.environ.get("SM120_ATTN_PREFILL_LONELY_FIX", "0") == "1":
        out.masked_fill_(lonely.view(-1, 1, 1), 0)
    lse.masked_fill_(lonely.view(-1, 1), float("inf"))
    max_logits.masked_fill_(lonely.view(-1, 1), float("-inf"))
    return out, max_logits, lse


def clear() -> None:
    with _LOCK:
        _PLANS.clear()
        _FAILED.clear()
        _ARENA.clear()
        _CACHE_PAGES.clear()
