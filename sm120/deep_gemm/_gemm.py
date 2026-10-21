"""GEMM / einsum entry points of the SM120 deep_gemm shim.

Semantics follow vendor/DeepGEMM/csrc/apis/{gemm,einsum,hyperconnection}.hpp:
    *_nt : d[m, n] = alpha * sum_k a[m, k] * b[n, k] (+ c[m, n]),   a: [M, K], b: [N, K]
    *_nn : b given as [K, N];  *_tn : a given as [K, M];  *_tt : both transposed.
    FP8 operands are (e4m3, sf) pairs, packed-FP4 operands are (int8 e2m1 x2, sf) pairs.
    d is bf16 (or fp32 for the bf16 GEMMs) and may be a strided view (row stride padded).

Torch reference = exact dequantisation to bf16 (e4m3 * 2^e and e2m1 * 2^e are exactly representable)
followed by a cuBLAS bf16 GEMM with fp32 accumulation, i.e. the same math as DeepGEMM's tensor-core
kernels up to fp32 summation order. fp32 outputs use a bf16 x bf16 -> fp32 GEMM (no bf16 rounding).
"""

from __future__ import annotations

import contextlib

import torch

from . import _b12x, _env, _sf

_DQ_CACHE: dict = {}
_MM_OUT_DTYPE_OK: bool | None = None


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
@contextlib.contextmanager
def _no_tf32():
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


def mm_fp32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """bf16 a [m, k] x bf16 b [n, k]^T -> fp32 [m, n] with fp32 accumulation and no bf16 rounding."""
    global _MM_OUT_DTYPE_OK
    if a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16 and _MM_OUT_DTYPE_OK is not False:
        try:
            r = torch.mm(a, b.t(), out_dtype=torch.float32)
            _MM_OUT_DTYPE_OK = True
            return r
        except (TypeError, RuntimeError, NotImplementedError):
            _MM_OUT_DTYPE_OK = False
    m, k = a.shape
    n = b.shape[0]
    out = torch.empty((m, n), dtype=torch.float32, device=a.device)
    af = a.float()
    step = _sf.chunk_rows(k * 4, n, _env.chunk_bytes())
    with _no_tf32():
        for s in range(0, n, step):
            e = min(n, s + step)
            out[:, s:e] = af @ b[s:e].float().t()
    return out


def mm_bf16(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """bf16 [m, k] x bf16 [n, k]^T -> bf16 [m, n] (cuBLAS, fp32 accumulate, one rounding)."""
    return torch.mm(a, b.t())


def _store(d: torch.Tensor, r: torch.Tensor, c: torch.Tensor | None, alpha: float | None) -> None:
    """d <- alpha * r (+ c), with r computed before d is touched (d may alias c)."""
    if alpha is not None and alpha != 1.0:
        r = r.float() * float(alpha)
    if c is not None:
        r = r.float() + c.float()
    d.copy_(r.view(d.shape) if r.shape != d.shape and r.numel() == d.numel() else r)


def _grans(recipe, recipe_a, recipe_b) -> tuple[int | None, int | None]:
    if recipe is not None:
        return int(recipe[2]), int(recipe[2])
    if recipe_a is not None or recipe_b is not None:
        ga = int(recipe_a[1]) if recipe_a is not None else None
        gb = int(recipe_b[1]) if recipe_b is not None else None
        return ga, gb
    return None, None


def _cache_key(t: torch.Tensor, sf: torch.Tensor | None, g) -> tuple:
    try:
        ver = (t._version, None if sf is None else sf._version)
    except Exception:  # inference tensors have no version counter
        ver = None
    return (t.data_ptr(), tuple(t.shape), t.dtype, None if sf is None else (sf.data_ptr(), tuple(sf.shape)), g, ver)


def dequant_weight(w: torch.Tensor, sf: torch.Tensor | None, gran_k: int | None, kind: str = "linear") -> torch.Tensor:
    """Static weight -> bf16 (optionally cached, see SM120_DEQUANT_CACHE)."""
    mode = _env.dequant_cache_mode()
    use_cache = mode == "all" or (mode == "einsum" and kind == "einsum")
    if not use_cache:
        return _sf.dequant_any(w, sf, gran_k, torch.bfloat16)
    key = _cache_key(w, sf, gran_k)
    hit = _DQ_CACHE.get(key)
    if hit is None:
        # a cache entry filled inside a CUDA graph capture would live in the graph pool and only hold data
        # after a replay; DeepInfer's eager warm-up fills it before any capture
        _b12x._not_in_capture(f"dequant cache fill for weight {tuple(w.shape)}")
        hit = _sf.dequant_any(w, sf, gran_k, torch.bfloat16)
        _DQ_CACHE[key] = hit
    return hit


def clear_dequant_cache() -> None:
    _DQ_CACHE.clear()


def _check_no_epilogue(epilogue, name: str) -> None:
    if epilogue is not None:
        raise NotImplementedError(f"deep_gemm.{name}: custom epilogues are not supported by the SM120 shim")


# ----------------------------------------------------------------------------------------------
# dense FP8 / FP4
# ----------------------------------------------------------------------------------------------
def fp8_fp4_gemm_nt(a, b, d, c=None, recipe=None, recipe_a=None, recipe_b=None, compiled_dims="nk",
                    disable_ue8m0_cast=False, alpha=None, epilogue=None):
    del compiled_dims, disable_ue8m0_cast
    _check_no_epilogue(epilogue, "fp8_fp4_gemm_nt")
    a_t, sfa = _sf.unpack_operand(a)
    b_t, sfb = _sf.unpack_operand(b)
    ga, gb = _grans(recipe, recipe_a, recipe_b)
    if a_t.shape[0] == 0:
        return
    if _env.gemm_impl() == "b12x":
        r = _b12x.try_linear(a_t, sfa, ga, b_t, sfb, gb, d, alpha, epilogue)
        if r is not None:
            _env.log_once(("fp8_nt", "b12x"), "fp8_fp4_gemm_nt -> b12x block_fp8_linear")
            _store(d, r, c, None)
            return
    _env.log_once(("fp8_nt", "torch"), "fp8_fp4_gemm_nt -> torch reference")
    A = _sf.dequant_any(a_t, sfa, ga, torch.bfloat16)
    B = dequant_weight(b_t, sfb, gb)
    if c is None and alpha is None and d.dtype == torch.bfloat16:
        d.copy_(mm_bf16(A, B))
    else:
        _store(d, mm_fp32(A, B), c, alpha)


def _not_nt(name):
    def fn(*args, **kwargs):
        raise NotImplementedError(
            f"deep_gemm.{name} is not used by DeepInfer and not implemented by the SM120 shim "
            "(the scale-factor granularity semantics of transposed FP8 operands differ from the NT case)"
        )

    fn.__name__ = name
    return fn


fp8_fp4_gemm_nn = _not_nt("fp8_fp4_gemm_nn")
fp8_fp4_gemm_tn = _not_nt("fp8_fp4_gemm_tn")
fp8_fp4_gemm_tt = _not_nt("fp8_fp4_gemm_tt")
fp8_gemm_nt = fp8_fp4_gemm_nt
fp4_gemm_nt = fp8_fp4_gemm_nt
fp8_gemm_nn = fp8_fp4_gemm_nn
fp8_gemm_tn = fp8_fp4_gemm_tn
fp8_gemm_tt = fp8_fp4_gemm_tt


# ----------------------------------------------------------------------------------------------
# dense BF16 / cuBLASLt
# ----------------------------------------------------------------------------------------------
def bf16_gemm_nt(a, b, d, c=None, compiled_dims="nk", alpha=None, epilogue=None):
    del compiled_dims
    _check_no_epilogue(epilogue, "bf16_gemm_nt")
    if a.shape[0] == 0:
        return
    a2 = a if a.dtype == torch.bfloat16 else a.to(torch.bfloat16)
    b2 = b if b.dtype == torch.bfloat16 else b.to(torch.bfloat16)
    if d.dtype == torch.bfloat16 and c is None and alpha is None:
        d.copy_(mm_bf16(a2, b2))
    else:
        _store(d, mm_fp32(a2, b2), c, alpha)


def bf16_gemm_nn(a, b, d, c=None, compiled_dims="nk", alpha=None, epilogue=None):
    bf16_gemm_nt(a, b.transpose(0, 1), d, c, compiled_dims, alpha, epilogue)


def bf16_gemm_tn(a, b, d, c=None, compiled_dims="mn", alpha=None, epilogue=None):
    bf16_gemm_nt(a.transpose(0, 1), b.transpose(0, 1), d, c, compiled_dims, alpha, epilogue)


def bf16_gemm_tt(a, b, d, c=None, compiled_dims="mn", alpha=None, epilogue=None):
    bf16_gemm_nt(a.transpose(0, 1), b, d, c, compiled_dims, alpha, epilogue)


def cublaslt_gemm_nt(a, b, d, c=None):
    bf16_gemm_nt(a, b, d, c)


def cublaslt_gemm_nn(a, b, d, c=None):
    bf16_gemm_nn(a, b, d, c)


def cublaslt_gemm_tn(a, b, d, c=None):
    bf16_gemm_tn(a, b, d, c)


def cublaslt_gemm_tt(a, b, d, c=None):
    bf16_gemm_tt(a, b, d, c)


def cublaslt_nvfp4_gemm_nt(a, b, d, c=None):
    raise NotImplementedError("deep_gemm.cublaslt_nvfp4_gemm_nt is not implemented by the SM120 shim")


def batched_syrk(a, d):
    # d[b] = a[b] @ a[b]^T
    d.copy_(torch.bmm(a.float(), a.float().transpose(1, 2)).to(d.dtype))


def batched_symm(a, b, d):
    d.copy_(torch.bmm(a.float(), b.float()).to(d.dtype))


# ----------------------------------------------------------------------------------------------
# M-grouped (MoE) GEMMs
# ----------------------------------------------------------------------------------------------
def _group_rows(grouped_layout: torch.Tensor, num_groups: int, m: int, use_psum_layout: bool):
    """-> list of (group, row_start, row_end) for contiguous groups, or (group, index_tensor)."""
    if use_psum_layout:
        alignment = _layout_alignment()
        ends = [int(v) for v in grouped_layout.tolist()]
        assert len(ends) == num_groups, (len(ends), num_groups)
        out, prev = [], 0
        for g, end in enumerate(ends):
            start = 0 if g == 0 else _sf.align(prev, alignment)
            end = min(end, m)
            if end > start:
                out.append((g, start, end))
            prev = ends[g]
        return out
    lay = grouped_layout.to(torch.int64)
    out = []
    for g in torch.unique(lay[lay >= 0]).tolist():
        out.append((int(g), (lay == g).nonzero(as_tuple=True)[0]))
    return out


def _take_rows(t: torch.Tensor, rows):
    if isinstance(rows, slice) or t.element_size() != 1 or t.dtype == torch.uint8:
        return t[rows]
    return t.contiguous().view(torch.uint8)[rows].view(t.dtype)


def _layout_alignment() -> int:
    from . import _config

    return _config.get_mk_alignment_for_contiguous_layout()


def _grouped_impl(a_t, sfa, ga, b_t, sfb, gb, d, grouped_layout, use_psum_layout, ensure_zero_padding):
    m = int(a_t.shape[0])
    if m == 0:
        return
    G = int(b_t.shape[0])
    groups = _group_rows(grouped_layout, G, m, use_psum_layout)
    covered = torch.zeros(m, dtype=torch.bool, device=d.device) if ensure_zero_padding else None
    for item in groups:
        if len(item) == 3:
            g, s, e = item
            rows = slice(s, e)
        else:
            g, rows = item
        A = _sf.dequant_any(_take_rows(a_t, rows), None if sfa is None else sfa[rows], ga, torch.bfloat16)
        B = dequant_weight(b_t[g], None if sfb is None else sfb[g], gb, kind="grouped")
        d[rows] = mm_bf16(A, B)
        if covered is not None:
            covered[rows] = True
    if covered is not None:
        d[~covered] = 0


def m_grouped_fp8_fp4_gemm_nt_contiguous(a, b, d, grouped_layout, recipe=None, recipe_a=None, recipe_b=None,
                                         compiled_dims="nk", disable_ue8m0_cast=False, use_psum_layout=False,
                                         ensure_zero_padding=True, expected_m_for_psum_layout=None, epilogue=None):
    del compiled_dims, disable_ue8m0_cast, expected_m_for_psum_layout
    _check_no_epilogue(epilogue, "m_grouped_fp8_fp4_gemm_nt_contiguous")
    a_t, sfa = _sf.unpack_operand(a)
    b_t, sfb = _sf.unpack_operand(b)
    ga, gb = _grans(recipe, recipe_a, recipe_b)
    _env.log_once("grouped_fp8", "m_grouped_fp8_fp4_gemm_nt_contiguous -> torch reference")
    _grouped_impl(a_t, sfa, ga, b_t, sfb, gb, d, grouped_layout, use_psum_layout, ensure_zero_padding)


def m_grouped_bf16_gemm_nt_contiguous(a, b, d, grouped_layout, compiled_dims="nk", use_psum_layout=False,
                                      ensure_zero_padding=True, expected_m_for_psum_layout=None, epilogue=None):
    del compiled_dims, expected_m_for_psum_layout
    _check_no_epilogue(epilogue, "m_grouped_bf16_gemm_nt_contiguous")
    _grouped_impl(a, None, None, b, None, None, d, grouped_layout, use_psum_layout, ensure_zero_padding)


def m_grouped_fp8_fp4_gemm_nn_contiguous(a, b, d, grouped_layout, recipe=None, recipe_a=None, recipe_b=None,
                                         compiled_dims="nk", disable_ue8m0_cast=False, use_psum_layout=False,
                                         ensure_zero_padding=True, epilogue=None):
    raise NotImplementedError("deep_gemm.m_grouped_fp8_fp4_gemm_nn_contiguous is not implemented by the SM120 shim")


def m_grouped_bf16_gemm_nn_contiguous(a, b, d, grouped_layout, compiled_dims="nk", use_psum_layout=False,
                                      ensure_zero_padding=True, epilogue=None):
    m_grouped_bf16_gemm_nt_contiguous(a, b.transpose(1, 2), d, grouped_layout, compiled_dims, use_psum_layout,
                                      ensure_zero_padding, None, epilogue)


def _masked_impl(a_t, sfa, ga, b_t, sfb, gb, d, masked_m):
    counts = [int(v) for v in masked_m.tolist()]
    for g, cnt in enumerate(counts):
        if cnt <= 0:
            continue
        A = _sf.dequant_any(a_t[g, :cnt], None if sfa is None else sfa[g, :cnt], ga, torch.bfloat16)
        B = dequant_weight(b_t[g], None if sfb is None else sfb[g], gb, kind="grouped")
        d[g, :cnt] = mm_bf16(A, B)


def m_grouped_fp8_fp4_gemm_nt_masked(a, b, d, masked_m, expected_m, recipe=None, recipe_a=None, recipe_b=None,
                                     compiled_dims="nk", disable_ue8m0_cast=False, epilogue=None):
    del expected_m, compiled_dims, disable_ue8m0_cast
    _check_no_epilogue(epilogue, "m_grouped_fp8_fp4_gemm_nt_masked")
    a_t, sfa = _sf.unpack_operand(a)
    b_t, sfb = _sf.unpack_operand(b)
    ga, gb = _grans(recipe, recipe_a, recipe_b)
    _masked_impl(a_t, sfa, ga, b_t, sfb, gb, d, masked_m)


def m_grouped_bf16_gemm_nt_masked(a, b, d, masked_m, expected_m, compiled_dims="nk", epilogue=None):
    del expected_m, compiled_dims
    _check_no_epilogue(epilogue, "m_grouped_bf16_gemm_nt_masked")
    _masked_impl(a, None, None, b, None, None, d, masked_m)


m_grouped_fp8_gemm_nt_contiguous = m_grouped_fp8_fp4_gemm_nt_contiguous
m_grouped_fp4_gemm_nt_contiguous = m_grouped_fp8_fp4_gemm_nt_contiguous
m_grouped_fp8_gemm_nn_contiguous = m_grouped_fp8_fp4_gemm_nn_contiguous
m_grouped_fp8_gemm_nt_masked = m_grouped_fp8_fp4_gemm_nt_masked
m_grouped_fp4_gemm_nt_masked = m_grouped_fp8_fp4_gemm_nt_masked


def _k_grouped(name):
    def fn(*args, **kwargs):
        raise NotImplementedError(f"deep_gemm.{name} (K-grouped, training-only) is not implemented by the SM120 shim")

    fn.__name__ = name
    return fn


k_grouped_fp8_gemm_tn_contiguous = _k_grouped("k_grouped_fp8_gemm_tn_contiguous")
k_grouped_fp8_gemm_nt_contiguous = _k_grouped("k_grouped_fp8_gemm_nt_contiguous")
k_grouped_fp4_gemm_nt_contiguous = _k_grouped("k_grouped_fp4_gemm_nt_contiguous")
k_grouped_bf16_gemm_tn_contiguous = _k_grouped("k_grouped_bf16_gemm_tn_contiguous")


# ----------------------------------------------------------------------------------------------
# einsum
# ----------------------------------------------------------------------------------------------
def _write_fp8_out(dst, val_bf16_bhd: torch.Tensor) -> None:
    """DeepGEMM FP8 epilogue: per_token_cast_to_fp8(d_bf16.view(b, h*d), gran 32, packed UE8M0)."""
    d_fp8, d_sf = dst
    bsz = val_bf16_bhd.shape[0]
    flat = val_bf16_bhd.reshape(bsz, -1)
    q, sf = _sf.quant_fp8_rows(flat, 32)
    d_fp8.copy_(q.view(d_fp8.shape))
    groups = sf.shape[-1]
    b = _sf.float_to_ue8m0_bytes(sf)
    g4 = _sf.align(groups, 4)
    if g4 != groups:
        b = torch.cat([b, torch.full((bsz, g4 - groups), 127, dtype=torch.uint8, device=b.device)], dim=-1)
    packed = b.contiguous().view(torch.int32)
    d_sf.copy_(packed.view(d_sf.shape))


def _bmm_bhr_hdr_bhd(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # A [b, h, r], B [h, d, r] -> [b, h, d] (bf16, fp32 accumulate)
    return torch.bmm(A.transpose(0, 1), B.transpose(1, 2)).transpose(0, 1)


def fp8_einsum(expr, a, b, d, c=None, recipe=(1, 128, 128), epilogue=None):
    _check_no_epilogue(epilogue, "fp8_einsum")
    a_t, sfa = _sf.unpack_operand(a)
    b_t, sfb = _sf.unpack_operand(b)
    gk = int(recipe[2]) if recipe is not None else None
    if expr == "bhr,hdr->bhd" and c is None:
        import sm120_einsum  # Triton kernel (SM120_EINSUM_IMPL=torch keeps the code below)

        if sm120_einsum.fp8_einsum_bhr_hdr(a_t, sfa, b_t, sfb, d, gk):
            _env.log_once(("fp8_einsum_triton", expr), f"fp8_einsum('{expr}') -> triton (sm120_einsum)")
            return
    _env.log_once(("fp8_einsum", expr), f"fp8_einsum('{expr}') -> torch (dequant bf16 + bmm)")
    if expr == "bhr,hdr->bhd":
        A = _sf.dequant_any(a_t, sfa, gk, torch.bfloat16)
        B = dequant_weight(b_t, sfb, gk, kind="einsum")
        r = _bmm_bhr_hdr_bhd(A, B)
    else:
        # "bhd,hdr->bhr" / "bhd,bhr->hdr" are SM100-only in DeepGEMM and unused by DeepInfer.
        raise NotImplementedError(f"deep_gemm.fp8_einsum('{expr}') is not implemented by the SM120 shim")
    if isinstance(d, (tuple, list)):
        if c is not None:
            r = (r.float() + c.float()).to(torch.bfloat16)
        _write_fp8_out(d, r.to(torch.bfloat16))
        return
    if c is not None:
        d.copy_((r.float() + c.float()).to(d.dtype))
    else:
        d.copy_(r)


def einsum(expr, a, b, d, c=None, epilogue=None):
    _check_no_epilogue(epilogue, "einsum")
    hp = d.dtype == torch.float32
    af = a.float() if hp else a
    bf = b.float() if hp else b
    with _no_tf32():
        if expr == "bmk,bnk->mn":
            r = torch.einsum("bmk,bnk->mn", af, bf)
        elif expr in ("bhr,hdr->bhd", "bhd,hdr->bhr", "bhd,bhr->hdr"):
            r = torch.einsum(expr, af, bf)
        else:
            raise NotImplementedError(f"deep_gemm.einsum('{expr}') is not implemented by the SM120 shim")
    if c is not None:
        r = r.float() + c.float()
    d.copy_(r.to(d.dtype))


# ----------------------------------------------------------------------------------------------
# hyper-connection prenorm GEMM
# ----------------------------------------------------------------------------------------------
def tf32_hc_prenorm_gemm(a, b, d, sqr_sum, num_splits=None):
    """d = a.float() @ b^T and sqr_sum = sum(a.float()^2, -1), optionally as split-K partials.

    a: bf16 [m, k], b: fp32 [n, k]; d: fp32 [m, n] or [S, m, n]; sqr_sum: fp32 [m] or [S, m].
    The partials are reduced by the consumer (sum over S), so the split of K is free: we compute the
    exact K-chunks DeepGEMM would (ceil(k / S) per split) in full fp32 (more accurate than TF32).
    """
    m, k = a.shape
    if m == 0:
        return
    import sm120_mhc  # one Triton launch for all split-K partials (SM120_MHC_IMPL=torch keeps the code below)

    if sm120_mhc.tf32_hc_prenorm_gemm(a, b, d, sqr_sum, num_splits):
        _env.log_once("hc_prenorm_triton", "tf32_hc_prenorm_gemm -> triton (sm120_mhc, fp32 FMA)")
        return
    with _no_tf32():
        if num_splits is None:
            af = a.float()
            d.copy_(af @ b.float().t())
            sqr_sum.copy_(af.square().sum(-1))
            return
        S = int(num_splits)
        step = _sf.ceil_div(k, S)
        bf = b.float()
        for s in range(S):
            k0, k1 = s * step, min(k, (s + 1) * step)
            if k1 <= k0:
                d[s].zero_()
                sqr_sum[s].zero_()
                continue
            af = a[:, k0:k1].float()
            d[s].copy_(af @ bf[:, k0:k1].t())
            sqr_sum[s].copy_(af.square().sum(-1))
