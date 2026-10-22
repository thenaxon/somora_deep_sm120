"""DeepGEMM attention-indexer ("MQA logits") replacements for SM120.

Backends: b12x dsa_indexer MXFP4 score (b12x_indexer.py) when SM120_INDEXER_IMPL (default:
SM120_ATTN_IMPL) is b12x and the operands are MXFP4 (DeepInfer's FP4_INDEX_KV_FORMAT), otherwise
the pure-torch reference below (fp32, chunked). b12x failures fall back to torch with a warning
(SM120_ATTN_STRICT=1 raises).

Originals: DeepInfer/vendor/DeepGEMM/csrc/apis/attention.hpp (+ tests/test_attention.py refs).
Call sites: DeepInfer/deepinfer/model_executor/models/v4_modules/index_attn.py.

    logits[q, t] = sum_h relu(q_h . k_t) * weights[q, h]     (no softmax scale; fp32 accumulate)

Operand formats (DeepGEMM SM100): q = (fp, sf) with fp int8 packed e2m1 [.., D/2] (MXFP4) or
float8_e4m3fn [.., D] (MXFP8), sf int32 packed ue8m0 (gran 32); weights bf16 -> logits bf16
(float32 weights -> float32 logits, SM90 flavour). Contiguous kv = (fp [N, D/2|D], sf [N] int32).
Paged kv cache = uint8 [pages, page, 1, D/2|D + 4] with BLOCK-wide [values | scales] sections.

Output geometry (only the valid region is meaningful; we fill the rest with -inf):
  * fp8_fp4_mqa_logits:             [s_q, max_seqlen_k], column j <-> kv token ks[q] + j, valid j < ke-ks
  * fp8_fp4_paged_mqa_logits:       [B*next_n, max_context_len], column j <-> logical token j, valid j < ctx
  * fp8_fp4_sparse_mqa_logits:      [s_q, nblk*sbk], column t <-> token blocks[q, t//sbk]*sbk + off + t%sbk
                                    (off = ks % sbk with use_unaligned_ks), valid for the first
                                    ceil((ke - base)/sbk) blocks and tokens in [ks, ke)
  * fp8_fp4_paged_sparse_mqa_logits: [s_q, nblk*sbk], logical token blocks[q, t//sbk]*sbk + t%sbk < ctx
Logits rows are allocated with a 1024-byte aligned stride like DeepGEMM.

The *metadata functions return lightweight objects/tensors; the torch kernels do not need them
(only the sparse ones carry the arguments the original kernels read from metadata).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from . import _config as cfg
from .quant import dequant_index_values, split_index_cache, ue8m0_to_float, unpack_e2m1

_NEG_INF = float("-inf")


def _b12x(name: str, fn, *args):
    """Run a b12x_indexer entry point. Returns logits, or None to use the torch path."""
    if cfg.indexer_impl() != "b12x":
        return None
    from . import b12x_indexer
    try:
        res = getattr(b12x_indexer, fn)(*args)
    except Exception as exc:  # noqa: BLE001
        if cfg.strict() or cfg.capturing():  # a graph must not record the (slower, fp32) torch path
            raise
        cfg.log_once(("b12x_idx_fail", name, type(exc).__name__, str(exc)[:200]),
                     f"b12x {name} failed ({type(exc).__name__}: {exc}); falling back to torch", logging.WARNING)
        return None
    if isinstance(res, torch.Tensor):
        cfg.log_once(("b12x_idx_ok", name), f"{name}: b12x dsa_indexer MXFP4 score")
        return res
    if res is None:
        return None
    if cfg.strict():
        raise RuntimeError(f"b12x {name} unsupported: {res}")
    cfg.log_once(("b12x_idx_unsupported", name, res), f"b12x {name} unsupported ({res}); using torch")
    return None


def _out_dtype(weights: torch.Tensor) -> torch.dtype:
    return torch.float32 if weights.dtype == torch.float32 else torch.bfloat16


def _alloc_logits(rows: int, cols: int, dtype: torch.dtype, device, col_align: int = 1) -> torch.Tensor:
    per = 1024 // torch.empty((), dtype=dtype).element_size()
    stride = cfg.align(cfg.align(max(cols, 1), col_align), per)
    buf = torch.empty((max(rows, 1), stride), dtype=dtype, device=device)
    return buf[:rows, :cols]


def _split_q(q):
    if isinstance(q, (tuple, list)):
        q_fp, q_sf = q
    else:
        q_fp, q_sf = q, None
    return q_fp, q_sf


def _head_dim(q_fp: torch.Tensor) -> int:
    return q_fp.shape[-1] * (2 if q_fp.dtype in (torch.int8, torch.uint8) else 1)


def _weighted_relu_scores(qf: torch.Tensor, kf: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """qf [m,H,D], kf [n,D] (shared) or [m,n,D] (per row), w [m,H] -> [m,n] float32."""
    if kf.dim() == 2:
        m, H, D = qf.shape
        s = (qf.reshape(m * H, D) @ kf.t()).view(m, H, -1)
    else:
        s = torch.bmm(qf, kf.transpose(1, 2))
    s.relu_().mul_(w.unsqueeze(-1))
    return s.sum(1)


def _dequant_paged_tokens(kv_cache: torch.Tensor, slots: torch.Tensor, is_fp4: bool, head_dim: int,
                          mx_sf: bool) -> torch.Tensor:
    """Gather + dequantize indexer K tokens from a block-layout paged cache. slots >= 0."""
    vbytes = head_dim // 2 if is_fp4 else head_dim
    vals, sfs = split_index_cache(kv_cache, vbytes, 4)
    bs = kv_cache.shape[1]
    s = slots.long()
    blk, off = s // bs, s % bs
    v = vals[blk, off]  # [..., vbytes]
    sb = sfs[blk, off]  # [..., 4]
    x = unpack_e2m1(v) if is_fp4 else v.contiguous().view(torch.float8_e4m3fn).float()
    if mx_sf:
        sc = ue8m0_to_float(sb)[..., : head_dim // 32]
        return (x.view(*x.shape[:-1], head_dim // 32, 32) * sc.unsqueeze(-1)).view(x.shape)
    return x * sb.contiguous().view(torch.float32)


# ---------------------------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------------------------
def get_mqa_logits_metadata(cu_seq_len_k_start, cu_seq_len_k_end, num_kv_tokens, num_heads):
    return torch.zeros((1,), dtype=torch.int32, device=cu_seq_len_k_start.device)


def get_paged_mqa_logits_metadata(context_lens, block_kv, num_sms=1, indices=None):
    n = int(num_sms) if isinstance(num_sms, int) and num_sms > 0 else 1
    return torch.zeros((n + 1, 2), dtype=torch.int32, device=context_lens.device)


@dataclass
class SparseMQAMetadata:
    ks: torch.Tensor
    ke: torch.Tensor
    num_kv_tokens: int
    blocks: torch.Tensor  # absolute block indices [s_q, nblk]
    sparse_block_kv: int
    use_unaligned_ks: bool


@dataclass
class PagedSparseMQAMetadata:
    context_lens: torch.Tensor  # [s_q]
    block_table: torch.Tensor  # [s_q, max_pages]
    page_kv: int
    blocks: torch.Tensor  # logical block indices [s_q, nblk]
    sparse_block_kv: int


def get_sparse_mqa_logits_metadata(cu_seq_len_k_start, cu_seq_len_k_end, num_kv_tokens, sparse_kv_block_indices,
                                   qk_dtype=None, sparse_block_kv=16, use_unaligned_ks=False):
    assert sparse_block_kv in (8, 16)
    return SparseMQAMetadata(cu_seq_len_k_start.clone(), cu_seq_len_k_end.clone(), int(num_kv_tokens),
                             sparse_kv_block_indices.clone(), int(sparse_block_kv), bool(use_unaligned_ks))


def get_paged_sparse_mqa_logits_metadata(context_lens, block_table, indices, page_kv, sparse_kv_block_indices,
                                         qk_dtype=None, sparse_block_kv=16):
    assert sparse_block_kv in (8, 16) and page_kv % sparse_block_kv == 0
    return PagedSparseMQAMetadata(context_lens.reshape(-1).clone(), block_table, int(page_kv),
                                  sparse_kv_block_indices.clone(), int(sparse_block_kv))


# ---------------------------------------------------------------------------------------------
# contiguous (prefill) logits
# ---------------------------------------------------------------------------------------------
def fp8_fp4_mqa_logits(q, kv, weights, cu_seq_len_k_start, cu_seq_len_k_end, max_seqlen_k, schedule_meta=None,
                       **_unused):
    res = _b12x("fp8_fp4_mqa_logits", "mqa_logits", q, kv, weights, cu_seq_len_k_start, cu_seq_len_k_end,
                max_seqlen_k)
    if res is not None:
        return res
    q_fp, q_sf = _split_q(q)
    kv_fp, kv_sf = kv
    S, H = q_fp.shape[0], q_fp.shape[1]
    D = _head_dim(q_fp)
    N = kv_fp.shape[0]
    dev = q_fp.device
    max_seqlen_k = int(max_seqlen_k)
    logits = _alloc_logits(S, max_seqlen_k, _out_dtype(weights), dev, col_align=128)
    logits.fill_(_NEG_INF)
    if S == 0 or N == 0 or max_seqlen_k == 0:
        return logits
    ks = cu_seq_len_k_start.long()
    ke = cu_seq_len_k_end.long()
    w = weights.float()
    budget = cfg.chunk_bytes()
    # window [m, <=N] fp32 + gathered [m, max_seqlen_k] fp32 + output: bound by the budget
    q_rows = max(1, min(S, 4096, budget // (16 * max(1, min(N, max_seqlen_k) + max_seqlen_k))))
    cols = torch.arange(max_seqlen_k, device=dev)
    for a in range(0, S, q_rows):
        b = min(a + q_rows, S)
        lo = int(ks[a:b].min().item())
        hi = int(torch.minimum(ke[a:b], ks[a:b] + max_seqlen_k).max().item())
        hi = min(max(hi, lo), N)
        if hi <= lo:
            continue
        qf = dequant_index_values(q_fp[a:b], None if q_sf is None else q_sf[a:b], D)  # [m,H,D]
        m = b - a
        window = torch.full((m, hi - lo), _NEG_INF, dtype=torch.float32, device=dev)
        nk = max(1, budget // max(1, m * H * 4 + D * 4))
        for c0 in range(lo, hi, nk):
            c1 = min(c0 + nk, hi)
            kf = dequant_index_values(kv_fp[c0:c1], None if kv_sf is None else kv_sf[c0:c1], D)
            window[:, c0 - lo:c1 - lo] = _weighted_relu_scores(qf, kf, w[a:b])
        src = ks[a:b].view(-1, 1) - lo + cols.view(1, -1)  # [m, max_seqlen_k]
        valid = (cols.view(1, -1) < (ke[a:b] - ks[a:b]).view(-1, 1)) & (src < hi - lo) & (src >= 0)
        vals = window.gather(1, src.clamp(0, hi - lo - 1))
        logits[a:b] = torch.where(valid, vals, torch.full_like(vals, _NEG_INF)).to(logits.dtype)
    return logits


# ---------------------------------------------------------------------------------------------
# paged (decode) logits
# ---------------------------------------------------------------------------------------------
def _paged_scores(qf, w, kv_cache, block_table_rows, tok, ctx, is_fp4, D, mx_sf):
    """qf [m,H,D], tok [m,T] logical token ids, ctx [m] -> scores [m,T] float32 (-inf invalid)."""
    page = kv_cache.shape[1]
    nb = kv_cache.shape[0]
    blk_logical = (tok // page).clamp(0, block_table_rows.shape[1] - 1)
    phys_blk = block_table_rows.long().gather(1, blk_logical).clamp(0, nb - 1)
    slots = phys_blk * page + tok % page
    valid = (tok >= 0) & (tok < ctx.view(-1, 1))
    kf = _dequant_paged_tokens(kv_cache, slots, is_fp4, D, mx_sf)  # [m,T,D]
    kf.masked_fill_(~valid.unsqueeze(-1), 0.0)
    kf = torch.nan_to_num(kf, nan=0.0)
    s = _weighted_relu_scores(qf, kf, w)
    return s.masked_fill_(~valid, _NEG_INF)


def fp8_fp4_paged_mqa_logits(q, kv_cache, weights, context_lens, block_table, schedule_meta, max_context_len,
                             indices=None, **_unused):
    res = _b12x("fp8_fp4_paged_mqa_logits", "paged_mqa_logits", q, kv_cache, weights, context_lens, block_table,
                max_context_len)
    if res is not None:
        return res
    q_fp, q_sf = _split_q(q)
    assert q_fp.dim() == 4, "q must be [batch, next_n, heads, dim]"
    B, nn, H = q_fp.shape[:3]
    D = _head_dim(q_fp)
    is_fp4 = q_fp.dtype in (torch.int8, torch.uint8)
    mx_sf = q_sf is not None
    dev = q_fp.device
    rows = B * nn
    max_context_len = int(max_context_len)
    logits = _alloc_logits(rows, max_context_len, _out_dtype(weights), dev)
    logits.fill_(_NEG_INF)
    if rows == 0 or max_context_len == 0:
        return logits
    qf_all = q_fp.reshape(rows, *q_fp.shape[2:])
    qs_all = None if q_sf is None else q_sf.reshape(rows, *q_sf.shape[2:])
    ctx = context_lens.reshape(rows).long().clamp(0, max_context_len)
    bt = block_table
    if bt.shape[0] != rows:  # [B, max_blocks] shared by the next_n tokens of a request
        bt = bt.repeat_interleave(nn, dim=0)
    w = weights.float()
    budget = cfg.chunk_bytes()
    capturing = cfg.capturing()  # no host syncs under CUDA graph capture: use the static bound
    L = max_context_len if capturing else int(ctx.max().item())
    if L == 0:
        return logits
    per_tok = D * 4 * 2 + H * 4 + 64
    row_step = max(1, min(rows, budget // max(1, per_tok * min(L, 8192))))
    for a in range(0, rows, row_step):
        b = min(a + row_step, rows)
        m = b - a
        La = L if capturing else int(ctx[a:b].max().item())
        if La == 0:
            continue
        qf = dequant_index_values(qf_all[a:b], None if qs_all is None else qs_all[a:b], D)
        tstep = max(1, budget // max(1, per_tok * m))
        for t0 in range(0, La, tstep):
            t1 = min(t0 + tstep, La)
            tok = torch.arange(t0, t1, device=dev).view(1, -1).expand(m, -1)
            s = _paged_scores(qf, w[a:b], kv_cache, bt[a:b], tok, ctx[a:b], is_fp4, D, mx_sf)
            logits[a:b, t0:t1] = s.to(logits.dtype)
    return logits


# ---------------------------------------------------------------------------------------------
# sparse (candidate-block) logits
# ---------------------------------------------------------------------------------------------
def fp8_fp4_sparse_mqa_logits(q, kv, weights, metadata: SparseMQAMetadata, num_max_sparse_blocks,
                              sparse_block_kv, use_unaligned_ks=False, **_unused):
    q_fp, q_sf = _split_q(q)
    kv_fp, kv_sf = kv
    S, H = q_fp.shape[0], q_fp.shape[1]
    D = _head_dim(q_fp)
    N = kv_fp.shape[0]
    sbk = int(sparse_block_kv)
    nblk = int(num_max_sparse_blocks)
    T = nblk * sbk
    dev = q_fp.device
    if S == 0 or N == 0:
        return _alloc_logits(S, T, _out_dtype(weights), dev).fill_(_NEG_INF)
    ks = metadata.ks.long().view(-1, 1)
    ke = metadata.ke.long().view(-1, 1)
    blocks = metadata.blocks.long()[:, :nblk]
    off = (ks % sbk) if (use_unaligned_ks or metadata.use_unaligned_ks) else torch.zeros_like(ks)
    base = (ks // sbk) * sbk + off
    nvalid = ((ke - base).clamp_min(0) + sbk - 1) // sbk  # [S,1]
    t = torch.arange(T, device=dev).view(1, -1)
    slot = t // sbk
    tok = blocks.gather(1, slot.expand(S, -1)) * sbk + off + t % sbk  # [S,T]
    valid = (slot < nvalid.clamp(max=nblk)) & (tok >= ks) & (tok < ke) & (tok < N) & (blocks.gather(
        1, slot.expand(S, -1)) >= 0)
    res = _b12x("fp8_fp4_sparse_mqa_logits", "sparse_mqa_logits", q, kv, weights,
                torch.where(valid, tok, torch.full_like(tok, -1)), metadata.ke)
    if res is not None:
        return res
    logits = _alloc_logits(S, T, _out_dtype(weights), dev).fill_(_NEG_INF)
    w = weights.float()
    step = cfg.rows_per_chunk(T * (D * 4 * 2 + H * 4 + 32), S)
    for a in range(0, S, step):
        b = min(a + step, S)
        qf = dequant_index_values(q_fp[a:b], None if q_sf is None else q_sf[a:b], D)
        tk = tok[a:b].clamp(0, N - 1)
        kf = dequant_index_values(kv_fp[tk], None if kv_sf is None else kv_sf[tk], D)
        kf.masked_fill_(~valid[a:b].unsqueeze(-1), 0.0)
        s = _weighted_relu_scores(qf, kf, w[a:b]).masked_fill_(~valid[a:b], _NEG_INF)
        logits[a:b] = s.to(logits.dtype)
    return logits


def fp8_fp4_paged_sparse_mqa_logits(q, kv_cache, weights, metadata: PagedSparseMQAMetadata, num_max_sparse_blocks,
                                    sparse_block_kv, **_unused):
    q_fp, q_sf = _split_q(q)
    assert q_fp.dim() == 4 and q_fp.shape[1] == 1
    S, _, H = q_fp.shape[:3]
    D = _head_dim(q_fp)
    is_fp4 = q_fp.dtype in (torch.int8, torch.uint8)
    sbk = int(sparse_block_kv)
    nblk = int(num_max_sparse_blocks)
    T = nblk * sbk
    dev = q_fp.device
    if S == 0:
        return _alloc_logits(S, T, _out_dtype(weights), dev).fill_(_NEG_INF)
    ctx = metadata.context_lens.long()
    blocks = metadata.blocks.long()[:, :nblk]
    nvalid = ((ctx + sbk - 1) // sbk).clamp(max=nblk).view(-1, 1)
    t = torch.arange(T, device=dev).view(1, -1)
    slot = t // sbk
    tok = blocks.gather(1, slot.expand(S, -1)) * sbk + t % sbk
    tok = torch.where(slot < nvalid, tok, torch.full_like(tok, -1))
    res = _b12x("fp8_fp4_paged_sparse_mqa_logits", "paged_sparse_mqa_logits", q, kv_cache, weights, metadata, tok)
    if res is not None:
        return res
    logits = _alloc_logits(S, T, _out_dtype(weights), dev).fill_(_NEG_INF)
    qf_all = q_fp[:, 0]
    qs_all = None if q_sf is None else q_sf[:, 0]
    w = weights.float()
    step = cfg.rows_per_chunk(T * (D * 4 * 2 + H * 4 + 64), S)
    for a in range(0, S, step):
        b = min(a + step, S)
        qf = dequant_index_values(qf_all[a:b], None if qs_all is None else qs_all[a:b], D)
        s = _paged_scores(qf, w[a:b], kv_cache, metadata.block_table[a:b], tok[a:b], ctx[a:b], is_fp4, D,
                          q_sf is not None)
        logits[a:b] = s.to(logits.dtype)
    return logits


DEEP_GEMM_EXPORTS = {
    "get_mqa_logits_metadata": get_mqa_logits_metadata,
    "fp8_fp4_mqa_logits": fp8_fp4_mqa_logits,
    "get_paged_mqa_logits_metadata": get_paged_mqa_logits_metadata,
    "fp8_fp4_paged_mqa_logits": fp8_fp4_paged_mqa_logits,
    "get_sparse_mqa_logits_metadata": get_sparse_mqa_logits_metadata,
    "get_paged_sparse_mqa_logits_metadata": get_paged_sparse_mqa_logits_metadata,
    "fp8_fp4_sparse_mqa_logits": fp8_fp4_sparse_mqa_logits,
    "fp8_fp4_paged_sparse_mqa_logits": fp8_fp4_paged_sparse_mqa_logits,
}

__all__ = list(DEEP_GEMM_EXPORTS) + ["DEEP_GEMM_EXPORTS", "SparseMQAMetadata", "PagedSparseMQAMetadata"]
