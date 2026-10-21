"""Unfused implementation of FlashMLA's fused_norm_rope_attn_rope_cast (DeepSeek V4.1).

Original: DeepInfer/vendor/FlashMLA/flash_mla/fused_norm_rope_attn_rope_cast.py (+ csrc
api/fused_norm_rope_attn_rope_cast_fwd.cpp, permute_q_b_proj/kernel.cu, permute_wv_proj/kernel.cu);
reference: vendor/FlashMLA/tests/test-fused-norm-rope-attn-rope-cast.py.

Pipeline (per query token):
  1. q arrives in the permute_q_b_proj layout (rows of q_b_proj were permuted, so the per-token
     h*d vector is ordered [(d//16), h, d%16]); undo it -> natural [h, d].
  2. optional Q RMSNorm (no weight): scale r = rsqrt(mean(q^2) + eps), applied to the logits
     (exactly like the reference: P = (rope(q).bf16 . k) * sm_scale * r).
  3. Q RoPE on the last rope_dim channels (fp32, then rounded to bf16).
  4. sparse attention (prefill: bf16 kv; decode: paged V4.1 caches + optional extra cache); both
     dispatch to b12x when SM120_ATTN_{PREFILL,DECODE}_IMPL resolve to b12x.
  5. O inverse RoPE (conjugate) on the last rope_dim channels of the fp32 output.
  6. cast to FP8 e4m3 per 32 elements with ue8m0 scales, in the permute_wv_proj layout: per
     wv group the [wv_group_size, d_v] block is stored as [(d_v//32), wv_group_size, 32].
     out_fp8 [s_q, n_wv_group, wv_group_size*d_v]; out_sf int32 [s_q, n_wv_group,
     wv_group_size*d_v/128] (4 packed ue8m0), stride(0) == 1 (TMA-aligned MN-major).

With SM120_ATTN_PERMUTE=0, steps 1 and 6 use natural layouts and both permute functions are identity.
Steps 1-3 and 5-6 run as two Triton kernels (fused_triton.py) unless SM120_ATTN_FUSED_IMPL=torch.
"""

from __future__ import annotations

import logging

import torch

from . import _config as cfg
from .quant import alloc_tma_colmajor_sf, as_u8, pack_ue8m0, per_token_cast_fp8_ue8m0
from .rope import apply_rope_, rms_scale
from .sparse_attn import sparse_decode, sparse_prefill


# ---------------------------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------------------------
def q_b_proj_row_permutation(h_q: int, d_q: int, device=None) -> torch.Tensor:
    """out_row[row] for row = head * d_q + dim (permute_q_b_proj/kernel.cu)."""
    r = torch.arange(h_q * d_q, device=device)
    h = r // d_q
    d = r % d_q
    return d % 16 + h * 16 + (d // 16) * (h_q * 16)


def q_from_fused_layout(q: torch.Tensor) -> torch.Tensor:
    """[s, h, d] tensor holding the permuted per-token vector -> natural [s, h, d] (contiguous)."""
    s, h, d = q.shape
    if not cfg.permute_enabled():
        return q.contiguous()
    assert d % 16 == 0
    assert q.stride(2) == 1 and q.stride(1) == d, "fused q must be contiguous within each token"
    flat = q.reshape(s, h * d)
    return flat.view(s, d // 16, h, 16).permute(0, 2, 1, 3).reshape(s, h, d)


def q_to_fused_layout(q: torch.Tensor) -> torch.Tensor:
    """Inverse of q_from_fused_layout (natural [s,h,d] -> permuted, as produced by a permuted q_b_proj)."""
    s, h, d = q.shape
    if not cfg.permute_enabled():
        return q.contiguous()
    return q.reshape(s, h, d // 16, 16).permute(0, 2, 1, 3).reshape(s, h, d)


def wv_chunk_permutation(wv_group_size: int, d_o: int, device=None) -> torch.Tensor:
    """src chunk index for every destination 32-chunk of a wv group row (permute_wv_proj/kernel.cu):
    source chunk i = head * (d_o/32) + c lands at c * wv_group_size + head."""
    C = d_o // 32
    i = torch.arange(wv_group_size * C, device=device)
    dst = (i % C) * wv_group_size + i // C
    src = torch.empty_like(i)
    src[dst] = i
    return src


def cast_output(out: torch.Tensor, n_wv_group: int) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 [s, h, d_v] -> (fp8 [s, G, gs*d_v], int32 sf [s, G, gs*d_v/128] stride(0)==1)."""
    s, h, dv = out.shape
    assert h % n_wv_group == 0 and dv % 128 == 0
    gs = h // n_wv_group
    if cfg.permute_enabled():
        x = out.view(s, n_wv_group, gs, dv // 32, 32).permute(0, 1, 3, 2, 4).reshape(s, n_wv_group, gs * dv)
    else:
        x = out.reshape(s, n_wv_group, gs * dv)
    fp8, sf = per_token_cast_fp8_ue8m0(x, 32)
    sf_out = alloc_tma_colmajor_sf(s, n_wv_group, gs * dv // 128, out.device)
    if s > 0:
        sf_out.copy_(pack_ue8m0(sf))
    return fp8.contiguous(), sf_out


def _check_cast_args(num_per_channels, use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0) -> None:
    if num_per_channels != 32:
        raise ValueError(f"num_per_channels must be 32, got {num_per_channels}")
    if not (use_tma_aligned_col_major_sf and round_sf and use_packed_ue8m0):
        raise NotImplementedError("only use_tma_aligned_col_major_sf=round_sf=use_packed_ue8m0=True is implemented "
                                  "(the only configuration documented by FlashMLA)")


def _triton_failed(where: str, exc: Exception) -> None:
    if cfg.strict():
        raise exc
    cfg.log_once(("fused_triton_fail", where, type(exc).__name__, str(exc)[:200]),
                 f"triton fused {where} failed ({type(exc).__name__}: {exc}); using torch", logging.WARNING)


def _use_triton() -> bool:
    if cfg.fused_impl() != "triton":
        return False
    from . import fused_triton
    return fused_triton.available()


def _prep(q, enable_q_norm, rms_norm_eps, token_positions, is_rope_neox_style, rope_dim, cos_sin_cache):
    """-> (q_r bf16 natural, q_scale fp32 | None, q_folded bf16 | None)."""
    if _use_triton():
        from . import fused_triton
        try:
            return fused_triton.prepare_q(q, enable_q_norm, rms_norm_eps, token_positions, is_rope_neox_style,
                                          rope_dim, cos_sin_cache, cfg.permute_enabled(), fold=True)
        except Exception as exc:  # noqa: BLE001
            _triton_failed("prepare_q", exc)
    q_r, q_scale = _prepare_q(q, enable_q_norm, rms_norm_eps, token_positions, is_rope_neox_style, rope_dim,
                              cos_sin_cache)
    return q_r, q_scale, None


def _finish(out, token_positions, cos_sin_cache, rope_dim, is_rope_neox_style, n_wv_group):
    """attention output (bf16 or fp32) -> inverse RoPE -> (fp8, sf) in the wv layout."""
    if _use_triton():
        from . import fused_triton
        try:
            s, h, dv = out.shape
            sf = alloc_tma_colmajor_sf(s, n_wv_group, (h // n_wv_group) * dv // 128, out.device)
            return fused_triton.rope_cast(out, token_positions, cos_sin_cache, rope_dim, is_rope_neox_style,
                                          n_wv_group, cfg.permute_enabled(), sf)
        except Exception as exc:  # noqa: BLE001
            _triton_failed("rope_cast", exc)
    out = out.float() if out.dtype != torch.float32 else out
    apply_rope_(out, token_positions, cos_sin_cache, rope_dim, is_rope_neox_style, conjugate=True)
    return cast_output(out, n_wv_group)


def _prepare_q(q, enable_q_norm, rms_norm_eps, token_positions, is_rope_neox_style, rope_dim, cos_sin_cache):
    q_nat = q_from_fused_layout(q)
    q_scale = rms_scale(q_nat, rms_norm_eps) if enable_q_norm else None
    q_r = apply_rope_(q_nat.float(), token_positions, cos_sin_cache, rope_dim, is_rope_neox_style,
                      conjugate=False).to(torch.bfloat16)
    return q_r, q_scale


# ---------------------------------------------------------------------------------------------
# Public (FlashMLA-compatible) entry points
# ---------------------------------------------------------------------------------------------
def prefill(enable_q_norm: bool, rms_norm_eps: float, token_positions: torch.Tensor, is_rope_neox_style: bool,
            rope_dim: int, cos_sin_cache: torch.Tensor, n_wv_group: int, num_per_channels: int,
            use_tma_aligned_col_major_sf: bool, round_sf: bool, use_packed_ue8m0: bool,
            q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, sm_scale: float, d_v: int = 512,
            attn_sink: torch.Tensor | None = None, topk_length: torch.Tensor | None = None):
    """Returns (out_fp8, out_sf, max_logits [s_q,h_q], lse [s_q,h_q])."""
    _check_cast_args(num_per_channels, use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0)
    assert kv.dim() == 3 and kv.shape[1] == 1, "only h_kv == 1 (MQA) is supported"
    assert indices.dim() == 3 and indices.shape[1] == 1
    q_r, q_scale, q_fold = _prep(q, enable_q_norm, rms_norm_eps, token_positions, is_rope_neox_style, rope_dim,
                                 cos_sin_cache)
    out, max_logits, lse = sparse_prefill(q_r, kv[:, 0], indices[:, 0], sm_scale, d_v, attn_sink,
                                          topk_length, q_scale, out_dtype="native", q_folded=q_fold)
    out_fp8, out_sf = _finish(out, token_positions, cos_sin_cache, rope_dim, is_rope_neox_style, n_wv_group)
    return out_fp8, out_sf, max_logits, lse


def decode(enable_q_norm: bool, rms_norm_eps: float, token_positions: torch.Tensor, is_rope_neox_style: bool,
           rope_dim: int, cos_sin_cache: torch.Tensor, n_wv_group: int, num_per_channels: int,
           use_tma_aligned_col_major_sf: bool, round_sf: bool, use_packed_ue8m0: bool,
           q: torch.Tensor, k_cache: torch.Tensor, indices_in_kvcache: torch.Tensor, sm_scale: float,
           d_v: int = 512, attn_sink: torch.Tensor | None = None, topk_length: torch.Tensor | None = None,
           extra_k_cache: torch.Tensor | None = None, extra_indices_in_kvcache: torch.Tensor | None = None,
           extra_topk_length: torch.Tensor | None = None):
    """Returns (out_fp8, out_sf, lse [s_q,h_q])."""
    _check_cast_args(num_per_channels, use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0)
    if (extra_k_cache is None) != (extra_indices_in_kvcache is None):
        raise ValueError("extra_k_cache and extra_indices_in_kvcache must be given together")
    q_r, q_scale, q_fold = _prep(q, enable_q_norm, rms_norm_eps, token_positions, is_rope_neox_style, rope_dim,
                                 cos_sin_cache)
    out, lse = sparse_decode(q_r, k_cache, indices_in_kvcache, topk_length, sm_scale, d_v, attn_sink,
                             extra_k_cache, extra_indices_in_kvcache, extra_topk_length, q_scale, out_dtype="native",
                             q_folded=q_fold)
    out_fp8, out_sf = _finish(out, token_positions, cos_sin_cache, rope_dim, is_rope_neox_style, n_wv_group)
    return out_fp8, out_sf, lse


def permute_q_b_proj(weight_and_sf, h_q: int, d_q: int):
    """Permute q_b_proj rows (weight [h_q*d_q, K] fp8 and its row-indexed scale factors).

    Exactly the row mapping of permute_q_b_proj/kernel.cu. The scale-factor tensor only has to be
    indexed by output row on dim 0 (any dtype / trailing layout); a block-scaled [N/128, ...] sf
    cannot be row-permuted and is rejected.
    """
    weight, sf = weight_and_sf
    rows = h_q * d_q
    assert weight.dim() == 2 and weight.shape[0] == rows, (tuple(weight.shape), h_q, d_q)
    if sf.shape[0] != rows:
        raise ValueError(f"permute_q_b_proj needs per-row scale factors (sf.shape[0] == {rows}), got "
                         f"{tuple(sf.shape)}; use SM120_ATTN_PERMUTE=0 if the GEMM shim keeps block scales")
    if not cfg.permute_enabled():
        return weight.clone(), sf.clone()
    out_row = q_b_proj_row_permutation(h_q, d_q, weight.device)
    src = torch.empty_like(out_row)
    src[out_row] = torch.arange(rows, device=weight.device)
    w_perm = as_u8(weight).index_select(0, src).view(weight.dtype) if weight.element_size() == 1 \
        else weight.index_select(0, src)
    sf_src = src.to(sf.device)
    sf_perm = as_u8(sf).index_select(0, sf_src).view(sf.dtype) if sf.element_size() == 1 \
        else sf.index_select(0, sf_src)
    return w_perm, sf_perm


def permute_wv_proj(weight_and_sf, wv_group_size: int, d_o: int):
    """Permute the input channels of wv_proj [G, N, gs*d_o] in 32-chunks (permute_wv_proj/kernel.cu)
    and its per-32 scale factors (int32 packed ue8m0 [G, N, gs*d_o/128] or unpacked [G, N, gs*d_o/32]).
    Output scale tensors keep the input's strides (DeepGEMM MN-major layout)."""
    weight, sf = weight_and_sf
    G, N, K = weight.shape
    assert K == wv_group_size * d_o and d_o % 32 == 0, (tuple(weight.shape), wv_group_size, d_o)
    if not cfg.permute_enabled():
        return weight.clone(), sf.clone()
    nchunks = K // 32
    src = wv_chunk_permutation(wv_group_size, d_o, weight.device)
    w8 = as_u8(weight).contiguous().view(G, N, nchunks, 32)
    w_perm = w8.index_select(2, src).reshape(G, N, K).view(weight.dtype)

    def _restride(t: torch.Tensor) -> torch.Tensor:
        res = torch.empty_strided(sf.shape, sf.stride(), dtype=sf.dtype, device=sf.device)
        res.copy_(t)
        return res

    sf_src = src.to(sf.device)
    if sf.dtype == torch.int32 and sf.shape[-1] * 4 == nchunks:
        b = sf.contiguous().view(torch.uint8)  # [G, N, nchunks], byte j of int k = scale 4k+j
        assert b.shape[-1] == nchunks
        sf_perm = b.index_select(-1, sf_src).contiguous().view(torch.int32)
        return w_perm, _restride(sf_perm)
    if sf.shape[-1] == nchunks:
        base = as_u8(sf) if sf.element_size() == 1 else sf
        sf_perm = base.index_select(-1, sf_src)
        if sf.element_size() == 1:
            sf_perm = sf_perm.view(sf.dtype)
        return w_perm, _restride(sf_perm)
    raise ValueError(f"permute_wv_proj: unsupported scale-factor shape {tuple(sf.shape)} dtype {sf.dtype} "
                     f"(expected [..., {nchunks // 4}] int32 packed or [..., {nchunks}] per-32)")
