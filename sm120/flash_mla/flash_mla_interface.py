"""SM120 shim of flash_mla.flash_mla_interface (vendor/FlashMLA/flash_mla/flash_mla_interface.py).

Same names, signatures, shapes and dtypes. Backends live in sm120_attn:
  * flash_mla_with_kvcache  -> b12x compressed_sparse_mla (SM120_ATTN_IMPL=b12x, when the cache
                               formats map: main 528 B, extra 288 B) else pure torch
  * flash_mla_sparse_fwd    -> b12x compressed_sparse_mla extend mode over transient V4.1 caches
                               re-encoded from the bf16 KV (SM120_ATTN_PREFILL_IMPL=b12x), else
                               pure torch (bf16 KV gather + fp32 softmax), chunked
  * flash_attn_varlen_*     -> pure torch dense attention (forward only; not used by DeepInfer)
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

import torch

from sm120_attn.sparse_attn import sparse_decode, sparse_prefill


@dataclasses.dataclass
class FlashMLASchedMeta:
    """
    A class that stores the tile scheduler metadata of FlashMLA (kept for API compatibility; the
    SM120 kernels do not need it, but the config consistency checks are reproduced).
    """

    @dataclasses.dataclass
    class Config:
        b: int
        s_q: int
        h_q: int
        page_block_size: int
        h_k: int

        causal: bool
        is_fp8_kvcache: bool
        topk: Optional[int]

        extra_page_block_size: Optional[int]
        extra_topk: Optional[int]

        enable_batch_invariant: bool

    have_initialized: bool = False

    config: Optional[Config] = None

    tile_scheduler_metadata: Optional[torch.Tensor] = None
    num_splits: Optional[torch.Tensor] = None


def get_mla_metadata(*args, **kwargs) -> Tuple[FlashMLASchedMeta, None]:
    """Returns an empty FlashMLASchedMeta (same as the original)."""
    return FlashMLASchedMeta(), None


def _check_or_init_sched_meta(sched_meta: FlashMLASchedMeta, q, k_cache, causal, is_fp8_kvcache, topk,
                              extra_k_page_block_size, extra_topk, enable_batch_invariant) -> None:
    if not sched_meta.have_initialized:
        sched_meta.have_initialized = True
        sched_meta.config = FlashMLASchedMeta.Config(
            b=q.shape[0], s_q=q.shape[1], h_q=q.shape[2], page_block_size=k_cache.shape[1], h_k=k_cache.shape[2],
            causal=causal, is_fp8_kvcache=is_fp8_kvcache, topk=topk,
            extra_page_block_size=extra_k_page_block_size, extra_topk=extra_topk,
            enable_batch_invariant=enable_batch_invariant,
        )
        return
    c = sched_meta.config
    msg = (" Your input arguments are inconsistent with sched_meta. Please make sure the input arguments are "
           "consistent across different invocations of flash_mla_with_kvcache on the same sched_meta.")
    assert c is not None
    assert c.b == q.shape[0], "sched_meta.config.b must be equal to batch_size." + msg
    assert c.s_q == q.shape[1], "sched_meta.config.s_q must be equal to seq_len_q." + msg
    assert c.h_q == q.shape[2], "sched_meta.config.h_q must be equal to num_heads_q." + msg
    assert c.page_block_size == k_cache.shape[1], "sched_meta.config.page_block_size must be equal to page_block_size." + msg
    assert c.h_k == k_cache.shape[2], "sched_meta.config.h_k must be equal to num_heads_k." + msg
    assert c.causal == causal, "sched_meta.config.causal must be equal to causal." + msg
    assert c.is_fp8_kvcache == is_fp8_kvcache, "sched_meta.config.is_fp8_kvcache must be equal to is_fp8_kvcache." + msg
    assert c.enable_batch_invariant == enable_batch_invariant, "sched_meta.config.enable_batch_invariant must be equal to enable_batch_invariant." + msg
    assert c.topk == topk, "sched_meta.config.topk must be equal to the last dim of indices_in_kvcache." + msg
    assert c.extra_page_block_size == extra_k_page_block_size, "sched_meta.config.extra_page_block_size must be equal to the page_block_size of extra_k_cache." + msg
    assert c.extra_topk == extra_topk, "sched_meta.config.extra_topk must be equal to the last dim of extra_indices_in_kvcache." + msg


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
    head_dim_v: int,
    tile_scheduler_metadata: FlashMLASchedMeta,
    num_splits: None = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    is_fp8_kvcache: bool = True,
    indices: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    enable_batch_invariant: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparse decode over paged V4.1 caches (528 B fp8 or 288 B fp4 per token).

    q: [b, s_q, h_q, 512] bf16; k_cache: [blocks, page, 1, bytes]; indices: [b, s_q, topk] int32
    (block*page + offset, -1 invalid); topk_length / extra_topk_length: [b] int32.
    Returns out [b, s_q, h_q, head_dim_v] bf16, softmax_lse [b, h_q, s_q] fp32 (+inf for queries
    without any valid key; NaN on the b12x path unless SM120_ATTN_B12X_LSE=1).
    """
    sched_meta = tile_scheduler_metadata
    assert isinstance(sched_meta, FlashMLASchedMeta), "tile_scheduler_metadata must be of type FlashMLASchedMeta"
    assert num_splits is None, "num_splits must be None"
    assert indices is not None, "Sparse attention is required: `indices` must be provided"
    assert not causal, "causal must be False when sparse attention is enabled"
    assert is_fp8_kvcache, "is_fp8_kvcache must be True, since only sparse attention with a quantized KV cache is supported"
    if (extra_k_cache is None) != (extra_indices_in_kvcache is None):
        raise ValueError("extra_k_cache and extra_indices_in_kvcache must be given together")

    topk = indices.shape[-1]
    extra_k_page_block_size = extra_k_cache.shape[1] if extra_k_cache is not None else None
    extra_topk = extra_indices_in_kvcache.shape[-1] if extra_indices_in_kvcache is not None else None
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    _check_or_init_sched_meta(sched_meta, q, k_cache, causal, is_fp8_kvcache, topk, extra_k_page_block_size,
                              extra_topk, enable_batch_invariant)

    b, s_q, h_q, d = q.shape
    rows = b * s_q
    q3 = q.reshape(rows, h_q, d)
    idx = indices.reshape(rows, topk)
    rep = (lambda t: t.reshape(b)) if s_q == 1 else (lambda t: t.reshape(b).repeat_interleave(s_q))  # no copy at s_q 1
    tl = None if topk_length is None else rep(topk_length)
    eidx = None if extra_indices_in_kvcache is None else extra_indices_in_kvcache.reshape(rows, extra_topk)
    etl = None if extra_topk_length is None else rep(extra_topk_length)
    out, lse = sparse_decode(q3, k_cache, idx, tl, float(softmax_scale), int(head_dim_v), attn_sink,
                             extra_k_cache, eidx, etl, None, torch.bfloat16)
    out = out.view(b, s_q, h_q, head_dim_v)
    lse = lse.view(b, s_q, h_q).transpose(1, 2)
    return out, lse


def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sparse prefill. q [s_q,h_q,d_qk] bf16, kv [s_kv,h_kv=1,d_qk] bf16, indices [s_q,h_kv,topk] int32
    (invalid: < 0 or >= s_kv). Returns (out [s_q,h_q,d_v] bf16, max_logits [s_q,h_q], lse [s_q,h_q]).
    On the b12x path max_logits is NaN and lse is NaN unless SM120_ATTN_B12X_LSE=1 (DeepInfer
    discards both)."""
    assert kv.dim() == 3 and kv.shape[1] == 1, "only h_kv == 1 (MQA) is supported"
    assert indices.dim() == 3 and indices.shape[1] == 1, "indices must be [s_q, h_kv=1, topk]"
    out, max_logits, lse = sparse_prefill(q, kv[:, 0], indices[:, 0], float(sm_scale), int(d_v), attn_sink,
                                          topk_length, None, torch.bfloat16)
    return out, max_logits, lse


# ---------------------------------------------------------------------------------------------
# Dense varlen attention (FlashMLA sm100 dense prefill). Forward only, pure torch.
# ---------------------------------------------------------------------------------------------
def _flash_attn_varlen_forward(q, k, v, cu_seqlens_qo, cu_seqlens_kv, max_seqlen_qo, max_seqlen_kv,
                               causal=False, softmax_scale=None, is_varlen=True):
    total_q, hq, dqk = q.shape
    hk = k.shape[1]
    dv = v.shape[-1]
    if softmax_scale is None:
        softmax_scale = dqk ** (-0.5)
    out = torch.zeros((total_q, hq, dv), dtype=q.dtype, device=q.device)
    lse = torch.full((hq, total_q), float("-inf"), dtype=torch.float32, device=q.device).T
    cq = cu_seqlens_qo.tolist()
    ck = cu_seqlens_kv.tolist()
    rep = hq // hk
    for i in range(len(cq) - 1):
        qs, qe, ks, ke = cq[i], cq[i + 1], ck[i], ck[i + 1]
        if qe <= qs:
            continue
        qi = q[qs:qe].float().transpose(0, 1)  # [h, Lq, d]
        ki = k[ks:ke].float().repeat_interleave(rep, dim=1).transpose(0, 1)
        vi = v[ks:ke].float().repeat_interleave(rep, dim=1).transpose(0, 1)
        s = torch.matmul(qi, ki.transpose(1, 2)) * softmax_scale
        if causal:
            lq, lk = qe - qs, ke - ks
            row = torch.arange(lq, device=q.device).view(-1, 1)
            col = torch.arange(lk, device=q.device).view(1, -1)
            s.masked_fill_(col > row + (lk - lq), float("-inf"))  # bottom-right aligned
        l_ = torch.logsumexp(s, dim=-1)  # [h, Lq]
        p = torch.exp(s - torch.where(torch.isneginf(l_), torch.zeros_like(l_), l_).unsqueeze(-1))
        out[qs:qe] = torch.matmul(p, vi).transpose(0, 1).to(q.dtype)
        lse[qs:qe] = l_.transpose(0, 1)
    return out, lse


def flash_attn_varlen_func(q, k, v, cu_seqlens_qo, cu_seqlens_kv, max_seqlen_qo, max_seqlen_kv, dropout_p=0.0,
                           softmax_scale=None, causal=False, deterministic=False, is_varlen=True):
    assert dropout_p == 0.0, "dropout is not supported, `dropout_p` must be 0.0"
    assert not deterministic, "the `deterministic` flag is not supported"
    with torch.no_grad():
        return _flash_attn_varlen_forward(q, k, v, cu_seqlens_qo, cu_seqlens_kv, max_seqlen_qo, max_seqlen_kv,
                                          causal=causal, softmax_scale=softmax_scale, is_varlen=is_varlen)


def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, head_dim_qk, dropout_p=0.0, softmax_scale=None,
                                     causal=False, deterministic=False, is_varlen=True):
    return flash_attn_varlen_func(qkv[:, :, :head_dim_qk], qkv[:, :, head_dim_qk:head_dim_qk * 2],
                                  qkv[:, :, head_dim_qk * 2:], cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
                                  dropout_p, softmax_scale, causal, deterministic, is_varlen)


def flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_qo, cu_seqlens_kv, max_seqlen_qo, max_seqlen_kv, head_dim_qk,
                                    dropout_p=0.0, softmax_scale=None, causal=False, deterministic=False,
                                    is_varlen=True):
    return flash_attn_varlen_func(q, kv[:, :, :head_dim_qk], kv[:, :, head_dim_qk:], cu_seqlens_qo, cu_seqlens_kv,
                                  max_seqlen_qo, max_seqlen_kv, dropout_p, softmax_scale, causal, deterministic,
                                  is_varlen)
