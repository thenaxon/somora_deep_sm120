"""b12x 1.5.1 dsa_indexer (cache_format="mxfp4") backend for the four DeepGEMM MQA-logits functions.

Byte formats match DeepInfer's FP4_INDEX_KV_FORMAT exactly (checked against
b12x/attention/dsa_indexer/mxfp4.py):

  * index-K page (b12x ``index_mxfp4_page_bytes``): ``[page, 64]`` packed e2m1 (even element in
    the low nibble) followed by ``[page, 4]`` ue8m0 (one per 32 values) - DeepInfer's block-wide
    ``[values | scales]`` layout with 68 B/token. Any page size that is a multiple of 8 works, so
    DeepInfer's paged ``ica_cache`` [pages, page, 1, 68] is used in place (2-D page view);
  * q: packed e2m1 [rows, H, 64] + 4 ue8m0 bytes per head (DeepGEMM's int32 packed sf viewed as
    bytes); weights bf16 [rows, H].

We only use the staged ``score`` entry point, which writes dense logits
``scores[row, col] = bf16(sum_h bf16(relu(bf16(q_h . k)) * w_h))`` with -inf for every invalid
column. Column semantics:

  * no candidates: column = logical token, valid for token < min(cache_lengths[row], active),
    physical page via page_table[row, token // page]  -> fp8_fp4_paged_mqa_logits as is;
  * candidates: column t scores token candidates[row, t] (-1 = invalid), valid for
    t < candidate_lengths[row] and token < cache_lengths[row]  -> the two sparse variants and the
    contiguous prefill variant (candidates = ks + column), with a transient page pool built
    from the contiguous K for the prefill variants.

Numerics vs DeepGEMM: b12x rounds q.k, relu(q.k)*w and the head sum to bf16 (this mirrors the
published V4.1 model.py scorer); DeepGEMM accumulates in fp32 and rounds once. Expect ~1e-2
relative differences per logit and occasional swaps of near-tied top-k entries.

Constraints (else the caller falls back to torch): MXFP4 q with ue8m0 sf (int8/uint8 [.., 64] +
int32), head dim 128, H <= 32 and 32 % H == 0, bf16 weights, candidate widths <= 16384,
16-byte aligned pages.

CUDA graphs: the paged (decode) variants issue no host syncs; plans are created on the first eager
call per shape (DeepInfer's graph warm-up). Every call allocates its own scratch (from the graph
pool under capture), so the returned logits never alias a later call.
"""

from __future__ import annotations

import threading

import torch

from . import _config as cfg
from .b12x_mla import prepare_plan

MAX_CANDIDATES = 16384
_POOL_PAGE = 64
_LOCK = threading.Lock()
_PLANS: dict = {}
_FAILED: set = set()
_MODS: dict = {}
_SUPPORTED: dict = {}


def _idx():
    if not _MODS:
        from b12x.attention import dsa_indexer
        _MODS["idx"] = dsa_indexer
    return _MODS["idx"]


def _device_supported(device) -> bool:
    key = str(device)
    if key not in _SUPPORTED:
        try:
            _SUPPORTED[key] = bool(_idx().is_supported(device))
        except Exception as exc:  # noqa: BLE001
            cfg.log_once(("b12x_idx_import", str(exc)[:200]), f"b12x dsa_indexer unavailable: {exc}")
            _SUPPORTED[key] = False
    return _SUPPORTED[key]


# ---------------------------------------------------------------------------------------------
# operand checks / views
# ---------------------------------------------------------------------------------------------
def _split(q):
    if isinstance(q, (tuple, list)):
        return q[0], q[1]
    return q, None


def q_reason(q_fp: torch.Tensor, q_sf, weights: torch.Tensor, heads_dim: int) -> str | None:
    if q_fp.device.type != "cuda":
        return "not on CUDA"
    if q_fp.dtype not in (torch.int8, torch.uint8) or q_fp.shape[-1] != 64:
        return f"q must be packed MXFP4 [.., 64] (got {q_fp.dtype} {tuple(q_fp.shape)})"
    if q_sf is None or q_sf.dtype != torch.int32:
        return "q scales must be int32 packed ue8m0"
    H = q_fp.shape[heads_dim]
    if H > 32 or 32 % H:
        return f"b12x MXFP4 indexer needs index heads dividing 32 (got {H})"
    if weights.dtype != torch.bfloat16:
        return f"weights must be bf16 (got {weights.dtype}; fp32 weights keep the fp32-logits torch path)"
    if not _device_supported(q_fp.device):
        return "b12x dsa_indexer not supported on this device / not importable"
    return None


def _q_views(q_fp: torch.Tensor, q_sf: torch.Tensor, rows: int, H: int):
    qv = q_fp.reshape(rows, H, 64)
    qv = (qv if qv.dtype == torch.uint8 else qv.view(torch.uint8)).contiguous()
    sv = q_sf.reshape(rows, H).contiguous().view(torch.uint8).view(rows, H, 4)
    return qv, sv


def _pool_view(kv_cache: torch.Tensor) -> torch.Tensor | None:
    """[pages, page, 1, 68] uint8 (any page stride) -> [pages, page*68] view, or None if unusable."""
    if kv_cache.dim() != 4 or kv_cache.shape[2] != 1 or kv_cache.shape[3] != 68 or kv_cache.element_size() != 1:
        return None
    if kv_cache.stride(3) != 1 or kv_cache.stride(1) != 68 or kv_cache.shape[1] % 8:
        return None
    u = kv_cache if kv_cache.dtype == torch.uint8 else kv_cache.view(torch.uint8)
    pool = u.as_strided((u.shape[0], u.shape[1] * 68), (u.stride(0), 1), u.storage_offset())
    if pool.stride(0) % 16 or pool.data_ptr() % 16:
        return None
    return pool


def _pool_from_contiguous(k_fp: torch.Tensor, k_sf: torch.Tensor):
    """Contiguous K (packed e2m1 [N,64], int32 ue8m0 [N]) -> page pool [P, 64*68] + shared page row."""
    n = k_fp.shape[0]
    P = max(1, (n + _POOL_PAGE - 1) // _POOL_PAGE)
    pool = torch.zeros((P, _POOL_PAGE * 68), dtype=torch.uint8, device=k_fp.device)
    vals = pool[:, : _POOL_PAGE * 64].view(P, _POOL_PAGE, 64)
    sfs = pool[:, _POOL_PAGE * 64:].view(P, _POOL_PAGE, 4)
    kv = k_fp if k_fp.dtype == torch.uint8 else k_fp.view(torch.uint8)
    ks = k_sf.reshape(n).contiguous().view(torch.uint8).view(n, 4)
    full = n // _POOL_PAGE
    if full:
        vals[:full].copy_(kv[: full * _POOL_PAGE].reshape(full, _POOL_PAGE, 64))
        sfs[:full].copy_(ks[: full * _POOL_PAGE].reshape(full, _POOL_PAGE, 4))
    rem = n - full * _POOL_PAGE
    if rem:
        vals[full, :rem].copy_(kv[full * _POOL_PAGE:])
        sfs[full, :rem].copy_(ks[full * _POOL_PAGE:])
    table = torch.arange(P, dtype=torch.int32, device=k_fp.device).view(1, P)
    return pool, table


# ---------------------------------------------------------------------------------------------
# plans + score
# ---------------------------------------------------------------------------------------------
def _get_plan(device, H: int, rows_cap: int, ptw_cap: int, page: int, mode: str, cand_cap: int):
    key = (str(device), H, rows_cap, ptw_cap, page, mode, cand_cap)
    if key in _FAILED:
        raise RuntimeError("b12x indexer plan previously failed for this configuration")
    with _LOCK:
        hit = _PLANS.get(key)
        if hit is not None:
            return hit
        try:
            idx = _idx()
            caps = idx.Caps(device=device, num_q_heads=H, max_q_rows=rows_cap, max_page_table_width=ptw_cap,
                            topk=512, mode=mode, page_size=page, cache_format="mxfp4",
                            output_index_space="logical", max_candidates=cand_cap)
            plan = idx.plan(caps)
            specs = prepare_plan(plan, f"sm120_attn.indexer.{mode}.h{H}.r{rows_cap}.p{ptw_cap}x{page}.c{cand_cap}",
                                 device)
        except Exception:
            if not cfg.capturing():
                _FAILED.add(key)
            raise
        nbytes = sum(int(s.shape[0]) for s in specs)
        cfg.log_once(("b12x_idx_plan",) + key[1:],
                     f"b12x dsa_indexer mxfp4 score plan: mode={mode} heads={H} rows<={rows_cap} "
                     f"pages<={ptw_cap}x{page} candidates={cand_cap} scratch={nbytes >> 20} MiB")
        _PLANS[key] = (plan, nbytes)
        return _PLANS[key]


def _rows_cap(rows: int, width: int, extra_per_row: int = 0) -> int:
    """Row capacity (power of two) such that the per-call scratch fits the budget."""
    per_row = width * 6 + 512 * 12 + extra_per_row
    lim = max(1, cfg.indexer_scratch_bytes() // max(1, per_row))
    cap = 1
    while cap * 2 <= lim and cap < rows:
        cap *= 2
    return cap


def _score(plan_nbytes, qv, sv, w, pool, page_table, lengths, active, cand, cand_len, score_width):
    """One b12x score launch. Returns scores bf16 [rows, score_width] (owns a fresh scratch)."""
    idx = _idx()
    plan, nbytes = plan_nbytes
    rows = qv.shape[0]
    dev = qv.device
    scratch = torch.empty((nbytes,), dtype=torch.uint8, device=dev)
    dummy = torch.empty((rows, 512), dtype=torch.int32, device=dev)
    kw = dict(scratch=scratch, q_mxfp4=qv, q_scales=sv, query_weights=w, index_k_cache=pool, page_table=page_table,
              cache_lengths=lengths, active_width=active, output_indices=dummy)
    if cand is not None:
        kw.update(candidate_indices=cand, candidate_lengths=cand_len)
    else:
        kw.update(score_width=int(score_width))
    binding = idx.bind(plan, **kw)
    return idx.score(binding)


_CONST: dict = {}


def _const_full(shape, value: int, dev) -> torch.Tensor:
    """Cached read-only int32 constant (no fill kernel per call; made eagerly before any graph capture)."""
    key = (tuple(shape), int(value), str(dev))
    t = _CONST.get(key)
    if t is None:
        t = torch.full(shape, int(value), dtype=torch.int32, device=dev)
        if not cfg.capturing():
            _CONST[key] = t
    return t


def _active(value: int, dev) -> torch.Tensor:
    return _const_full((1,), min(int(value), 2 ** 31 - 1), dev)


def _alloc_logits(rows: int, cols: int, device) -> torch.Tensor:
    stride = cfg.align(max(cols, 1), 512)
    return torch.empty((max(rows, 1), stride), dtype=torch.bfloat16, device=device)[:rows, :cols]


# ---------------------------------------------------------------------------------------------
# DeepGEMM entry points (return None if this backend does not apply; the caller logs + falls back)
# ---------------------------------------------------------------------------------------------
def paged_mqa_logits(q, kv_cache, weights, context_lens, block_table, max_context_len):
    q_fp, q_sf = _split(q)
    if q_fp.dim() != 4:
        return "q must be [batch, next_n, heads, 64]"
    r = q_reason(q_fp, q_sf, weights, 2)
    if r:
        return r
    pool = _pool_view(kv_cache)
    if pool is None:
        return f"index cache {tuple(kv_cache.shape)} / strides {tuple(kv_cache.stride())} not usable as a b12x pool"
    B, nn, H = q_fp.shape[:3]
    rows = B * nn
    W = int(max_context_len)
    dev = q_fp.device
    if rows == 0 or W == 0:
        return _alloc_logits(rows, W, dev).fill_(float("-inf"))
    page = kv_cache.shape[1]
    bt = block_table
    if bt.shape[0] != rows:
        bt = bt.repeat_interleave(nn, dim=0)
    if bt.dtype != torch.int32 or bt.stride(1) != 1:
        bt = bt.to(torch.int32).contiguous()
    SW = cfg.align(W, 512)
    ptw = max(int(bt.shape[1]), (SW + page - 1) // page)
    if 2 ** 31 - 1 < SW:
        return "context too long"
    qv, sv = _q_views(q_fp, q_sf, rows, H)
    w = weights.reshape(rows, H).contiguous()
    lens = context_lens.reshape(rows).to(torch.int32).contiguous()
    active = _active(W, dev)
    cap = _rows_cap(rows, SW)
    plan = _get_plan(dev, H, cap, ptw, page, "decode", 0)
    if cap >= rows:
        return _score(plan, qv, sv, w, pool, bt, lens, active, None, None, SW)[:, :W]
    out = _alloc_logits(rows, W, dev)
    for a in range(0, rows, cap):
        b = min(a + cap, rows)
        out[a:b] = _score(plan, qv[a:b], sv[a:b], w[a:b], pool, bt[a:b], lens[a:b], active, None, None, SW)[:, :W]
    return out


def _candidate_scores(plan_args, qv, sv, w, pool, table, lens, cand, T, dev):
    """Score candidates [rows, T] (-1 = invalid) -> logits [rows, T] with a 1024 B-aligned stride.

    Widths above 16384 (b12x's candidate limit) are scored in column chunks and copied."""
    rows = qv.shape[0]
    cap_c = min(cfg.align(T, 512), MAX_CANDIDATES)
    cand = cand.to(torch.int32)
    active = _active(2 ** 31 - 1, dev)
    H, ptw, page, mode = plan_args
    shared = table.shape[0] == 1
    cap = _rows_cap(rows, cap_c, cap_c * 4)
    plan = _get_plan(dev, H, cap, ptw, page, mode, cap_c)
    single = cap >= rows and T <= cap_c
    out = None if single else _alloc_logits(rows, T, dev)
    for c0 in range(0, T, cap_c):
        c1 = min(c0 + cap_c, T)
        cc = cand[:, c0:c1]
        if c1 - c0 != cap_c:
            cc = torch.nn.functional.pad(cc, (0, cap_c - (c1 - c0)), value=-1)
        cc = cc.contiguous()
        cl = _const_full((rows,), c1 - c0, dev)
        for a in range(0, rows, cap):
            b = min(a + cap, rows)
            sc = _score(plan, qv[a:b], sv[a:b], w[a:b], pool, table if shared else table[a:b], lens[a:b], active,
                        cc[a:b], cl[a:b], cap_c)
            if single:
                return sc[:, :T]
            out[a:b, c0:c1] = sc[:, : c1 - c0]
    return out


def paged_sparse_mqa_logits(q, kv_cache, weights, metadata, tok: torch.Tensor):
    """tok [S, T]: logical token per column (-1 = invalid slot), computed by the caller."""
    q_fp, q_sf = _split(q)
    if q_fp.dim() != 4 or q_fp.shape[1] != 1:
        return "q must be [s_q, 1, heads, 64]"
    r = q_reason(q_fp, q_sf, weights, 2)
    if r:
        return r
    pool = _pool_view(kv_cache)
    if pool is None:
        return f"index cache {tuple(kv_cache.shape)} not usable as a b12x pool"
    S, _, H = q_fp.shape[:3]
    T = tok.shape[1]
    dev = q_fp.device
    if S == 0 or T == 0:
        return _alloc_logits(S, T, dev).fill_(float("-inf"))
    bt = metadata.block_table
    if bt.dtype != torch.int32 or bt.stride(1) != 1:
        bt = bt.to(torch.int32).contiguous()
    qv, sv = _q_views(q_fp, q_sf, S, H)
    w = weights.reshape(S, H).contiguous()
    lens = metadata.context_lens.reshape(S).to(torch.int32).contiguous()
    return _candidate_scores((H, int(bt.shape[1]), kv_cache.shape[1], "decode"), qv, sv, w, pool, bt, lens, tok, T,
                             dev)


def _contig_k(kv):
    k_fp, k_sf = kv
    if k_fp.dtype not in (torch.int8, torch.uint8) or k_fp.dim() != 2 or k_fp.shape[1] != 64:
        return None
    if k_sf is None or k_sf.dtype != torch.int32 or k_sf.numel() != k_fp.shape[0]:
        return None
    return k_fp, k_sf


def sparse_mqa_logits(q, kv, weights, tok: torch.Tensor, ke: torch.Tensor):
    """Contiguous K, candidates tok [S, T] (absolute token ids, -1 invalid; already masked to [ks, ke))."""
    q_fp, q_sf = _split(q)
    if q_fp.dim() != 3:
        return "q must be [s_q, heads, 64]"
    r = q_reason(q_fp, q_sf, weights, 1)
    if r:
        return r
    k = _contig_k(kv)
    if k is None:
        return "kv must be (packed e2m1 [N, 64], int32 ue8m0 [N])"
    S, H = q_fp.shape[:2]
    T = tok.shape[1]
    dev = q_fp.device
    if S == 0 or T == 0 or k[0].shape[0] == 0:
        return _alloc_logits(S, T, dev).fill_(float("-inf"))
    pool, table = _pool_from_contiguous(*k)
    qv, sv = _q_views(q_fp, q_sf, S, H)
    w = weights.reshape(S, H).contiguous()
    lens = ke.reshape(S).to(torch.int32).contiguous()
    ptw = cfg.next_pow2(table.shape[1], 16)
    return _candidate_scores((H, ptw, _POOL_PAGE, "prefill"), qv, sv, w, pool, table, lens, tok, T, dev)


def mqa_logits(q, kv, weights, ks: torch.Tensor, ke: torch.Tensor, max_seqlen_k: int):
    """Contiguous (prefill) logits [S, max_seqlen_k]: column j <-> token ks + j, valid j < ke - ks.

    Scored in candidate chunks of <= 16384 columns. Columns past the longest row's valid range are
    left uninitialized (as DeepGEMM does; DeepInfer passes `end` to top-k) unless
    SM120_INDEXER_FILL_TAIL=1."""
    q_fp, q_sf = _split(q)
    if q_fp.dim() != 3:
        return "q must be [s_q, heads, 64]"
    r = q_reason(q_fp, q_sf, weights, 1)
    if r:
        return r
    k = _contig_k(kv)
    if k is None:
        return "kv must be (packed e2m1 [N, 64], int32 ue8m0 [N])"
    S, H = q_fp.shape[:2]
    W = int(max_seqlen_k)
    N = k[0].shape[0]
    dev = q_fp.device
    if S == 0 or W == 0 or N == 0:
        return _alloc_logits(S, W, dev).fill_(float("-inf"))
    pool, table = _pool_from_contiguous(*k)
    ptw = cfg.next_pow2(table.shape[1], 16)
    qv, sv = _q_views(q_fp, q_sf, S, H)
    w = weights.reshape(S, H).contiguous()
    ks32 = ks.reshape(S).to(torch.int32)
    span = (ke.reshape(S).to(torch.int32) - ks32).clamp(0, W)
    lens = (ks32 + span).contiguous()
    CW = min(MAX_CANDIDATES, cfg.align(W, 512))
    # skip column chunks past the longest valid row (needs one host sync; only worth it above one chunk)
    ext = W if (W <= CW or cfg.capturing()) else int(span.max().item())
    cols = torch.arange(CW, dtype=torch.int32, device=dev).view(1, -1)
    active = _active(N, dev)
    cap = _rows_cap(S, CW, CW * 4)
    plan = _get_plan(dev, H, cap, ptw, _POOL_PAGE, "prefill", CW)
    single = W <= CW and cap >= S
    out = None if single else _alloc_logits(S, W, dev)
    for c0 in range(0, max(ext, 1), CW):
        c1 = min(c0 + CW, W)
        for a in range(0, S, cap):
            b = min(a + cap, S)
            cand = (ks32[a:b].view(-1, 1) + c0 + cols).contiguous()
            cand_len = (span[a:b] - c0).clamp(0, CW).contiguous()
            sc = _score(plan, qv[a:b], sv[a:b], w[a:b], pool, table, lens[a:b], active, cand, cand_len, CW)
            if single:
                return sc[:, :W]
            out[a:b, c0:c1] = sc[:, : c1 - c0]
    covered = min(cfg.align(max(ext, 1), CW), W)
    if cfg.indexer_fill_tail() and covered < W:
        out[:, covered:].fill_(float("-inf"))
    return out


def clear() -> None:
    with _LOCK:
        _PLANS.clear()
        _FAILED.clear()
