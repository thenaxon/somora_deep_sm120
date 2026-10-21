"""Sparse MLA attention (FlashMLA sparse prefill / sparse decode semantics) for SM120.

Reference semantics: DeepInfer/vendor/FlashMLA/tests/ref.py (ref_sparse_attn_fwd,
ref_sparse_attn_decode) and the docstrings in vendor/FlashMLA/flash_mla/flash_mla_interface.py.

    P      = (q . k) * sm_scale [* rms_scale(q)]     over the selected tokens, invalid -> -inf
    lse    = logsumexp(P)          (+inf when a query has no valid token; the sink is NOT included)
    max    = max(P)                (-inf when no valid token)
    out    = softmax(P) @ v * 1 / (1 + exp(sink - lse))   (0 when no valid token)
    v      = k[..., :d_v]  (MQA, K == V latent)

Validity: prefill: index < 0 or >= s_kv is invalid; decode: index == -1 invalid (we also treat
any negative as invalid). topk_length[i] masks positions >= topk_length[i].
"""

from __future__ import annotations

import logging
import os
import sys

import torch

from . import _config as cfg
from .quant import dequant_v41_records, gather_paged_records


_P_BF16 = os.environ.get("SM120_ATTN_P_BF16", "1") != "0"


def _attend_rows(q: torch.Tensor, kv: torch.Tensor, valid: torch.Tensor, sm_scale: float, d_v: int,
                 attn_sink: torch.Tensor | None, q_scale: torch.Tensor | None):
    """q [R,H,D] (any float dtype), kv fp32 [R,T,D], valid bool [R,T] ->
    out fp32 [R,H,d_v], max_logits [R,H], lse [R,H] (+inf for lonely rows)."""
    qf = q.float()
    P = torch.bmm(qf, kv.transpose(1, 2))  # [R,H,T]
    if q_scale is not None:
        P.mul_(q_scale.float().unsqueeze(-1) * sm_scale)
    else:
        P.mul_(sm_scale)
    P.masked_fill_(~valid.unsqueeze(1), float("-inf"))
    max_logits = P.amax(dim=-1)
    lse = torch.logsumexp(P, dim=-1)
    if attn_sink is not None:
        lse_o = torch.logaddexp(lse, attn_sink.float().view(1, -1).expand_as(lse))
    else:
        lse_o = lse
    lse_o = torch.where(lse_o == float("-inf"), torch.full_like(lse_o, float("inf")), lse_o)
    P.sub_(lse_o.unsqueeze(-1)).exp_()
    if _P_BF16:
        # FlashMLA and DeepSeek's reference kernel feed P to the P@V MMA as bf16.
        P = P.to(torch.bfloat16).float()
    out = torch.bmm(P, kv[..., :d_v])
    lse = torch.where(lse == float("-inf"), torch.full_like(lse, float("inf")), lse)
    return out, max_logits, lse


# ---------------------------------------------------------------------------------------------
# Prefill: bf16 KV, flat indices into kv
# ---------------------------------------------------------------------------------------------
def sparse_prefill_torch(q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, sm_scale: float,
                         d_v: int = 512, attn_sink: torch.Tensor | None = None,
                         topk_length: torch.Tensor | None = None, q_scale: torch.Tensor | None = None,
                         out_dtype: torch.dtype | None = torch.bfloat16):
    """q [s_q,H,D] bf16, kv [s_kv,D] (bf16), indices [s_q,topk] int.

    Returns (out [s_q,H,d_v] out_dtype (fp32 if out_dtype is None), max_logits [s_q,H], lse [s_q,H]).
    """
    s_q, H, D = q.shape
    s_kv = kv.shape[0]
    topk = indices.shape[-1]
    dev = q.device
    out = torch.empty((s_q, H, d_v), dtype=out_dtype or torch.float32, device=dev)
    max_logits = torch.empty((s_q, H), dtype=torch.float32, device=dev)
    lse = torch.empty((s_q, H), dtype=torch.float32, device=dev)
    if s_q == 0:
        return out, max_logits, lse
    per_row = topk * D * 4 + 2 * H * topk * 4 + topk * D * kv.element_size()
    step = cfg.rows_per_chunk(per_row, s_q)
    pos = torch.arange(topk, device=dev)
    with cfg.matmul_precision(cfg.allow_tf32()):
        for a in range(0, s_q, step):
            b = min(a + step, s_q)
            idx = indices[a:b].long()
            valid = (idx >= 0) & (idx < s_kv)
            if topk_length is not None:
                valid &= pos.view(1, -1) < topk_length[a:b].long().view(-1, 1)
            if s_kv == 0:
                g = torch.zeros((b - a, topk, D), dtype=torch.float32, device=dev)
            else:
                g = kv[idx.clamp(0, s_kv - 1)].float()  # [r,T,D]
            g.masked_fill_(~valid.unsqueeze(-1), 0.0)
            o, m, l = _attend_rows(q[a:b], g, valid, sm_scale, d_v, attn_sink,
                                   None if q_scale is None else q_scale[a:b])
            out[a:b] = o
            max_logits[a:b] = m
            lse[a:b] = l
    return out, max_logits, lse


# ---------------------------------------------------------------------------------------------
# Decode: paged quantized caches (528 B fp8 / 288 B fp4 per token), optional extra cache
# ---------------------------------------------------------------------------------------------
def _valid_mask(idx: torch.Tensor, length: torch.Tensor | None) -> torch.Tensor:
    valid = idx >= 0
    if length is not None:
        pos = torch.arange(idx.shape[-1], device=idx.device).view(1, -1)
        valid &= pos < length.long().view(-1, 1)
    return valid


def _gather_dequant(k_cache: torch.Tensor, idx: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    g = dequant_v41_records(gather_paged_records(k_cache, idx.clamp_min(0)))
    g = torch.nan_to_num(g, nan=0.0)  # FlashMLA ref: NaN in gathered KV -> 0
    g.masked_fill_(~valid.unsqueeze(-1), 0.0)
    return g


def sparse_decode_torch(q: torch.Tensor, k_cache: torch.Tensor, indices: torch.Tensor,
                        topk_length: torch.Tensor | None, sm_scale: float, d_v: int = 512,
                        attn_sink: torch.Tensor | None = None, extra_k_cache: torch.Tensor | None = None,
                        extra_indices: torch.Tensor | None = None, extra_topk_length: torch.Tensor | None = None,
                        q_scale: torch.Tensor | None = None, out_dtype: torch.dtype | None = torch.bfloat16):
    """q [R,H,D] bf16; indices [R,topk] (flat block*bs+off, -1 invalid); lengths [R] or None.

    Returns (out [R,H,d_v] out_dtype (fp32 if None), lse [R,H] float32).
    """
    R, H, D = q.shape
    dev = q.device
    T1 = indices.shape[-1]
    T2 = 0 if extra_indices is None else extra_indices.shape[-1]
    out = torch.empty((R, H, d_v), dtype=out_dtype or torch.float32, device=dev)
    lse = torch.empty((R, H), dtype=torch.float32, device=dev)
    if R == 0:
        return out, lse
    T = T1 + T2
    per_row = T * (D * 4 * 2 + 2 * H * 4 + 600)
    step = cfg.rows_per_chunk(per_row, R)
    for a in range(0, R, step):
        b = min(a + step, R)
        idx1 = indices[a:b]
        v1 = _valid_mask(idx1, None if topk_length is None else topk_length[a:b])
        g = _gather_dequant(k_cache, idx1, v1)
        valid = v1
        if extra_indices is not None:
            idx2 = extra_indices[a:b]
            v2 = _valid_mask(idx2, None if extra_topk_length is None else extra_topk_length[a:b])
            g = torch.cat((g, _gather_dequant(extra_k_cache, idx2, v2)), dim=1)
            valid = torch.cat((v1, v2), dim=1)
        o, _m, l = _attend_rows(q[a:b], g, valid, sm_scale, d_v, attn_sink,
                                None if q_scale is None else q_scale[a:b])
        out[a:b] = o
        lse[a:b] = l
    return out, lse


def sparse_decode(q: torch.Tensor, k_cache: torch.Tensor, indices: torch.Tensor,
                  topk_length: torch.Tensor | None, sm_scale: float, d_v: int = 512,
                  attn_sink: torch.Tensor | None = None, extra_k_cache: torch.Tensor | None = None,
                  extra_indices: torch.Tensor | None = None, extra_topk_length: torch.Tensor | None = None,
                  q_scale: torch.Tensor | None = None, out_dtype=torch.bfloat16,
                  impl: str | None = None, q_folded: torch.Tensor | None = None):
    """Dispatcher: b12x compressed_sparse_mla when possible, else torch. Same contract as
    sparse_decode_torch. q_scale (RMS factor) is folded into q for b12x (bf16 rounding); q_folded
    is a precomputed bf16(q * q_scale) for that. out_dtype="native": bf16 from b12x, fp32 from torch."""
    impl = impl or cfg.decode_impl()
    if impl == "b12x":
        from . import b12x_mla
        reason = b12x_mla.unsupported_reason(q, k_cache, extra_k_cache, d_v)
        if reason is None:
            try:
                qq = q if q_scale is None else (q_folded if q_folded is not None else
                                                (q.float() * q_scale.float().unsqueeze(-1)).to(torch.bfloat16))
                o, l = b12x_mla.sparse_decode_b12x(qq, k_cache, indices, topk_length, sm_scale, attn_sink,
                                                   extra_k_cache, extra_indices, extra_topk_length)
                if out_dtype == "native":
                    pass
                elif out_dtype is None:
                    o = o.float()
                elif o.dtype != out_dtype:
                    o = o.to(out_dtype)
                return o, l
            except Exception as exc:  # noqa: BLE001
                if cfg.strict():
                    raise
                cfg.log_once(("b12x_decode_fail", type(exc).__name__, str(exc)[:200]),
                             f"b12x sparse decode failed ({type(exc).__name__}: {exc}); falling back to torch"
                             + (" (inside CUDA graph capture: the graph records the torch path)"
                                if cfg.capturing() else ""),
                             logging.WARNING)
        else:
            if cfg.strict():
                raise RuntimeError(f"b12x sparse decode unsupported: {reason}")
            cfg.log_once(("b12x_decode_unsupported", reason), f"b12x sparse decode unsupported ({reason}); using torch")
    return sparse_decode_torch(q, k_cache, indices, topk_length, sm_scale, d_v, attn_sink, extra_k_cache,
                               extra_indices, extra_topk_length, q_scale, None if out_dtype == "native" else out_dtype)


def sparse_prefill(q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, sm_scale: float, d_v: int = 512,
                   attn_sink: torch.Tensor | None = None, topk_length: torch.Tensor | None = None,
                   q_scale: torch.Tensor | None = None, out_dtype=torch.bfloat16,
                   impl: str | None = None, q_folded: torch.Tensor | None = None):
    """Dispatcher: b12x compressed_sparse_mla extend mode when possible, else torch. Same contract as
    sparse_prefill_torch except that the b12x path returns NaN max_logits (and NaN lse unless
    SM120_ATTN_B12X_LSE=1). q_scale (RMS factor) is folded into q for b12x (bf16 rounding)."""
    if "sm120_capture" in sys.modules:  # docs/MTP_YOCO.md: compressed positions this layer attends to
        from .capture import record_prefill_indices
        record_prefill_indices(kv, indices, topk_length)
    impl = impl or cfg.prefill_impl()
    if impl == "b12x":
        from . import b12x_prefill
        reason = b12x_prefill.unsupported_reason(q, kv, d_v)
        if reason is None:
            try:
                qq = q if q_scale is None else (q_folded if q_folded is not None else
                                                (q.float() * q_scale.float().unsqueeze(-1)).to(torch.bfloat16))
                o, m, l = b12x_prefill.sparse_prefill_b12x(qq, kv, indices, sm_scale, attn_sink, topk_length)
                if out_dtype == "native":
                    pass
                elif out_dtype is None:
                    o = o.float()
                elif o.dtype != out_dtype:
                    o = o.to(out_dtype)
                return o, m, l
            except Exception as exc:  # noqa: BLE001
                if cfg.strict():
                    raise
                cfg.log_once(("b12x_prefill_fail", type(exc).__name__, str(exc)[:200]),
                             f"b12x sparse prefill failed ({type(exc).__name__}: {exc}); falling back to torch",
                             logging.WARNING)
        else:
            if cfg.strict():
                raise RuntimeError(f"b12x sparse prefill unsupported: {reason}")
            cfg.log_once(("b12x_prefill_unsupported", reason),
                         f"b12x sparse prefill unsupported ({reason}); using torch")
    else:
        cfg.log_once("prefill_impl_torch", "sparse prefill: torch implementation")
    return sparse_prefill_torch(q, kv, indices, sm_scale, d_v, attn_sink, topk_length, q_scale,
                                None if out_dtype == "native" else out_dtype)
