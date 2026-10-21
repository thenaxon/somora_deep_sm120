"""deep_select.topk replacement (torch.topk based).

Original: DeepInfer/vendor/DeepSelect/deep_select/interface.py (semantics checked in
vendor/DeepSelect/tests/test.py). Contract reproduced here:

  * per row i only columns [0, end[i]) are candidates (end None -> vocab_size)
  * the first c_i = min(end[i], topk) output slots hold the selected elements; slots >= c_i get
    idx_oob_fill_value (NOT offset) and value_oob_fill_value
  * selected indices get += output_idx_offset[i] (if given)
  * sorted=True: values descending; sorted_index=True: indices ascending (takes precedence when
    both are set; the original only tests one of them at a time)
  * neither: order unspecified (we return value-descending)
  * output_idx (if given) is written in place and returned; otherwise outputs are allocated with a
    32-byte aligned row stride like the original
  * values are gathered from the input (dtype preserved); return_value=False -> first item None
  * NaN handling differs: DeepSelect aborts / writes 0x3F3F3F3F into output_idx[i][0]; here NaN
    ranks below every finite value (never selected before a real candidate)
  * ties are broken arbitrarily (as in the original)

Selection key (valid -inf/NaN -> lowest finite, columns >= end -> -inf, fp32) is built by one Triton pass (SM120_TOPK_IMPL=triton, default unless SM120_ATTN_IMPL=torch), else by torch ops
on an fp32 copy. Both are CUDA-graph safe (no host syncs).
"""

from __future__ import annotations

from typing import Optional, Tuple

import os
import sys

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001
    triton = None

_STRIDE_BYTES = (16, 32)


def _impl() -> str:
    v = os.environ.get("SM120_TOPK_IMPL", "").strip().lower()
    if not v:
        v = "torch" if os.environ.get("SM120_ATTN_IMPL", "b12x").strip().lower() == "torch" else "triton"
    return v


if triton is not None:

    @triton.jit
    def _key_kernel(x_ptr, x_stride, end_ptr, key_ptr, V, LOWEST, HAS_END: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        inb = cols < V
        x = tl.load(x_ptr + row.to(tl.int64) * x_stride + cols, mask=inb, other=0.0)
        bad = (x != x) | (x == float("-inf"))
        k = tl.where(bad, LOWEST, x)
        if HAS_END:
            e = tl.load(end_ptr + row)
            k = tl.where(cols < e, k, float("-inf"))
        tl.store(key_ptr + row.to(tl.int64) * V + cols, k.to(key_ptr.dtype.element_ty), mask=inb)


if triton is not None:

    @triton.jit
    def _topk_index_kernel(x_ptr, x_s, end_ptr, off_ptr, out_ptr, out_s, V, K, fill, LOWEST,
                           HAS_END: tl.constexpr, HAS_OFF: tl.constexpr, BLOCK: tl.constexpr, KP: tl.constexpr):
        """Index-ordered top-k of one row (deep_select sorted_index semantics), radix select on fp32 keys.

        key = x (valid NaN/-inf -> lowest finite), columns >= end -> -inf; select the K largest keys with ties
        at the threshold taken by lowest index (torch gatherTopK order); write them in ascending index order;
        slots >= min(end, K) get `fill`; selected indices get + off[row]."""
        r = tl.program_id(0)
        r64 = r.to(tl.int64)
        cols = tl.arange(0, BLOCK)
        inb = cols < V
        x = tl.load(x_ptr + r64 * x_s + cols, mask=inb, other=0.0).to(tl.float32)
        if HAS_END:
            e = tl.minimum(tl.maximum(tl.load(end_ptr + r).to(tl.int32), 0), V)
        else:
            e = V
        bad = (x != x) | (x == float("-inf"))
        key = tl.where(bad, LOWEST, x)
        key = tl.where(cols < e, key, float("-inf"))
        bits = key.to(tl.int32, bitcast=True)
        ordered = bits ^ ((bits >> 31) & 0x7FFFFFFF)  # monotone signed int
        u = ordered.to(tl.int64) + 2147483648  # monotone in [0, 2^32)
        u = tl.where(inb, u, -1)  # padding columns below everything
        kk = tl.minimum(K, V)
        prefix = tl.full((), 0, tl.int64)
        remaining = kk.to(tl.int64)
        bins = tl.arange(0, 256)
        for p in tl.static_range(4):
            shift: tl.constexpr = 24 - 8 * p
            if p == 0:
                match = u >= 0
            else:
                match = (u >= 0) & ((u >> (shift + 8)) == (prefix >> (shift + 8)))
            digit = tl.where(match, (u >> shift) & 255, 0).to(tl.int32)
            hist = tl.histogram(digit, 256)
            nonmatch = BLOCK - tl.sum(match.to(tl.int32), axis=0)
            hist = hist - tl.where(bins == 0, nonmatch, 0)
            incl = tl.cumsum(hist, axis=0)
            total = tl.sum(hist, axis=0)
            suffix = (total - incl + hist).to(tl.int64)  # count of matching digits >= d
            d_sel = tl.max(tl.where(suffix >= remaining, bins, -1), axis=0)
            above = tl.sum(tl.where(bins > d_sel, hist, 0), axis=0).to(tl.int64)
            remaining = remaining - above
            prefix = prefix | (d_sel.to(tl.int64) << shift)
        gt = u > prefix
        eq = u == prefix
        need_eq = kk.to(tl.int64) - tl.sum(gt.to(tl.int64), axis=0)
        eq_rank = tl.cumsum(eq.to(tl.int32), axis=0)
        sel = gt | (eq & (eq_rank.to(tl.int64) <= need_eq))
        pos = tl.cumsum(sel.to(tl.int32), axis=0) - 1
        count = tl.minimum(e, K)
        idx = cols
        if HAS_OFF:
            idx = idx + tl.load(off_ptr + r).to(tl.int32)
        dst = out_ptr + r64 * out_s
        tl.store(dst + pos, idx.to(dst.dtype.element_ty), mask=sel & (pos < count))
        slots = tl.arange(0, KP)
        tl.store(dst + slots, tl.full((KP,), fill, tl.int32).to(dst.dtype.element_ty),
                 mask=(slots >= count) & (slots < K))


_FUSED_MAX_V = 16384


def _topk_index_triton(x: torch.Tensor, k: int, end, offset, out: torch.Tensor, fill: int) -> bool:
    N, V = x.shape
    if V > _FUSED_MAX_V or V == 0 or x.stride(1) != 1 or out.stride(1) != 1:
        return False
    BLOCK = max(256, triton.next_power_of_2(V))
    KP = triton.next_power_of_2(max(k, 1))
    if KP > BLOCK:
        return False
    e = None if end is None else end.reshape(N)
    o = None if offset is None else offset.reshape(N)
    if e is not None and not e.is_contiguous():
        e = e.contiguous()
    if o is not None and not o.is_contiguous():
        o = o.contiguous()
    lowest = float(torch.finfo(torch.float32).min)
    _topk_index_kernel[(N,)](x, x.stride(0), e if e is not None else x, o if o is not None else x, out, out.stride(0),
                             V, int(k), int(fill), lowest, HAS_END=e is not None, HAS_OFF=o is not None,
                             BLOCK=BLOCK, KP=KP, num_warps=16 if BLOCK >= 8192 else (8 if BLOCK >= 2048 else 4))
    return True


def _key_triton(x: torch.Tensor, end_i32: torch.Tensor | None) -> torch.Tensor:
    N, V = x.shape
    if x.stride(1) != 1:
        x = x.contiguous()
    # fp32 key (like the torch path): torch.topk's bf16 and fp32 kernels break ties differently, and the
    # indexer's relu scores tie a lot (exact zeros, repeated tokens); fp32 keeps the torch/DeepSeek-ref choice
    key = torch.empty((N, V), dtype=torch.float32, device=x.device)
    BLOCK = 2048
    _key_kernel[(N, triton.cdiv(V, BLOCK))](x, x.stride(0), end_i32 if end_i32 is not None else x, key, V,
                                            float(torch.finfo(torch.float32).min), HAS_END=end_i32 is not None,
                                            BLOCK=BLOCK, num_warps=4)
    return key


def get_stride_requirement() -> Tuple[int, int]:
    """(input row-stride bytes, output row-stride bytes); torch has no requirement, keep originals' values."""
    return _STRIDE_BYTES


def _aligned_empty(rows: int, cols: int, device, dtype) -> torch.Tensor:
    per = max(1, _STRIDE_BYTES[1] // torch.empty((), dtype=dtype).element_size())
    c = (cols + per - 1) // per * per
    return torch.empty((rows, c), device=device, dtype=dtype)[:, :cols]


def topk(
    input: torch.Tensor,
    topk: int,
    sorted: bool = False,
    begin: Optional[torch.Tensor] = None,
    end: Optional[torch.Tensor] = None,
    indices_type: torch.dtype = torch.int64,
    sorted_index: bool = False,
    hint: Optional[torch.Tensor] = None,
    output_idx: Optional[torch.Tensor] = None,
    output_idx_offset: Optional[torch.Tensor] = None,
    idx_oob_fill_value: int = 2147483647,
    value_oob_fill_value: float = float("-inf"),
    return_value: bool = True,
    abort_when_nan_found: bool = True,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    assert begin is None, "`begin` is not supported now"
    assert hint is None, "`hint` is not supported now"
    assert input.dim() == 2, input.shape
    assert indices_type in (torch.int32, torch.int64)
    N, V = input.shape
    k = int(topk)
    dev = input.device

    out_val = _aligned_empty(N, k, dev, input.dtype) if return_value else None
    if output_idx is None:
        output_idx = _aligned_empty(N, k, dev, indices_type)
    else:
        assert output_idx.dtype == indices_type
        assert tuple(output_idx.shape) == (N, k), (tuple(output_idx.shape), (N, k))
    if N == 0 or k == 0:
        return out_val, output_idx

    # fused index-ordered top-k (DeepInfer's indexer calls: sorted_index, no values) for rows up to 16384 wide.
    # Opt-in (SM120_TOPK_FUSED=1): bit-identical to the torch path incl. tie order, but ~10.8 ms per call on
    # SM120 (tl.histogram / full-row cumsum in one CTA per row are pathological) vs ~0.33 ms for torch.topk.
    if (_impl() == "triton" and triton is not None and dev.type == "cuda" and not return_value
            and (sorted_index or not sorted) and input.dtype in (torch.float32, torch.bfloat16, torch.float16)
            and os.environ.get("SM120_TOPK_FUSED", "0") == "1"):  # opt-in: correct but slow (see docs/ATTENTION.md 3f)
        try:
            if _topk_index_triton(input, k, end, output_idx_offset, output_idx, int(idx_oob_fill_value)):
                if "sm120_capture" in sys.modules:
                    from .capture import record_candidate_blocks
                    record_candidate_blocks(output_idx)
                return out_val, output_idx
        except Exception:  # noqa: BLE001
            pass
    # key: valid elements > -inf (valid -inf / NaN mapped to the lowest finite value), OOB -> -inf
    lens = end.reshape(N).long().clamp(0, V) if end is not None else torch.full((N,), V, dtype=torch.long, device=dev)
    key = None
    if (_impl() == "triton" and triton is not None and dev.type == "cuda"
            and input.dtype in (torch.float32, torch.bfloat16, torch.float16)):
        try:
            key = _key_triton(input, None if end is None else lens.to(torch.int32))
        except Exception:  # noqa: BLE001
            key = None
    if key is None:
        x = input.float()
        lowest = torch.finfo(torch.float32).min
        key = torch.where(torch.isnan(x) | torch.isneginf(x), torch.full_like(x, lowest), x)
        if end is not None:
            cols = torch.arange(V, device=dev).view(1, -1)
            key = key.masked_fill(cols >= lens.view(-1, 1), float("-inf"))
    kk = min(k, V)
    count = lens.clamp(max=k)  # number of meaningful output slots per row
    slot = torch.arange(kk, device=dev).view(1, -1)
    valid = slot < count.view(-1, 1)
    if sorted_index:
        # every valid column (< end, key > -inf) outranks every OOB column, and OOB columns have larger
        # indices, so after an index sort the first `count` slots are exactly the valid selections:
        # no value-sorted top-k needed
        _, idx = torch.topk(key, kk, dim=1, largest=True, sorted=False)
        idx = torch.sort(idx, dim=1).values
        idx = torch.where(valid, idx, torch.zeros_like(idx))
    else:
        _, idx = torch.topk(key, kk, dim=1, largest=True, sorted=True)  # valid entries first
    vals = input.gather(1, idx) if out_val is not None else None

    res_idx = idx
    if output_idx_offset is not None:
        res_idx = res_idx + output_idx_offset.reshape(N, 1).long()
    res_idx = torch.where(valid, res_idx, torch.full_like(res_idx, int(idx_oob_fill_value)))
    if kk < k:
        pad = torch.full((N, k - kk), int(idx_oob_fill_value), dtype=res_idx.dtype, device=dev)
        res_idx = torch.cat((res_idx, pad), dim=1)
    output_idx.copy_(res_idx.to(indices_type))
    if "sm120_capture" in sys.modules:  # docs/MTP_YOCO.md: candidate blocks of the candidate source
        from .capture import record_candidate_blocks
        record_candidate_blocks(output_idx)

    if out_val is not None:
        v = torch.where(valid, vals, torch.full_like(vals, value_oob_fill_value))
        if kk < k:
            v = torch.cat((v, torch.full((N, k - kk), value_oob_fill_value, dtype=v.dtype, device=dev)), dim=1)
        out_val.copy_(v)
    return out_val, output_idx
