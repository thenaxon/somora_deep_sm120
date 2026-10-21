"""SM120 kernels for DeepInfer's mHC (hyper-connection) path: deep_gemm.tf32_hc_prenorm_gemm.

DeepInfer (v4_modules/mhc_ops.py mhc_prenorm_gemm) calls, per sub-layer,
    tf32_hc_prenorm_gemm(x [m, hc*hidden] bf16, fn [n=(2+hc)*hc, hc*hidden] fp32,
                         d [S, m, n] fp32, sqr_sum [S, m] fp32, S=16)
and reduces the S split-K partials itself (tile_kernels mhc_reduce_partials_and_rmsnorm_fwd):
    d[s] = x[:, Ks] @ fn[:, Ks]^T,  sqr_sum[s] = sum(x[:, Ks]^2),  Ks = s-th ceil(k/S) chunk of K.
The torch fallback in deep_gemm/_gemm.py loops over S (one mm + square + sum + copies per split:
256 tiny kernels per decode step on the 8-layer test model). Here one Triton launch computes every
partial: grid (row blocks, S x P), fp32 FMA dot (input_precision="ieee", i.e. full fp32 - at least as
accurate as DeepGEMM's TF32), square sum in the same pass. For few rows (decode) each split is cut into
P sub-splits that are summed in a fixed order afterwards (deterministic). With S=None, K is one split.

Switch: SM120_MHC_IMPL = triton (default) | torch. Failures fall back to torch (logged once).
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001
    triton = None

_LOGGED: set = set()


def impl() -> str:
    v = os.environ.get("SM120_MHC_IMPL", "triton").strip().lower()
    if v not in ("triton", "torch"):
        raise ValueError(f"SM120_MHC_IMPL must be 'triton' or 'torch', got {v!r}")
    return v


def precision() -> str:
    """tl.dot input precision: ieee (default; full fp32) or tf32 (DeepGEMM's own tf32_hc_prenorm_gemm contract)."""
    v = os.environ.get("SM120_MHC_PRECISION", "ieee").strip().lower()
    if v not in ("ieee", "tf32"):
        raise ValueError(f"SM120_MHC_PRECISION must be ieee|tf32, got {v!r}")
    return v


def _log_once(key, msg):
    if key not in _LOGGED:
        _LOGGED.add(key)
        print(f"[sm120_mhc] {msg}", flush=True)


if triton is not None:

    @triton.jit
    def _hc_prenorm_kernel(a_ptr, a_stride, b_ptr, b_stride, d_ptr, d_ss, d_sm, sq_ptr, sq_ss, M, N, K, step, sub,
                           P: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                           PREC: tl.constexpr):
        pm = tl.program_id(0)
        s = tl.program_id(1)  # output slot = split * P + sub-split
        rm = pm * BM + tl.arange(0, BM)
        rn = tl.arange(0, BN)
        split_end = tl.minimum((s // P + 1) * step, K)
        k0 = (s // P) * step + (s % P) * sub
        k1 = tl.minimum(k0 + sub, split_end)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        sq = tl.zeros((BM,), dtype=tl.float32)
        for kb in range(k0, k1, BK):
            rk = kb + tl.arange(0, BK)
            km = rk < k1
            a = tl.load(a_ptr + rm[:, None].to(tl.int64) * a_stride + rk[None, :],
                        mask=(rm[:, None] < M) & km[None, :], other=0.0).to(tl.float32)
            b = tl.load(b_ptr + rn[:, None] * b_stride + rk[None, :], mask=(rn[:, None] < N) & km[None, :],
                        other=0.0)
            acc = tl.dot(a, tl.trans(b), acc, input_precision=PREC)
            sq += tl.sum(a * a, axis=1)
        tl.store(d_ptr + s.to(tl.int64) * d_ss + rm[:, None].to(tl.int64) * d_sm + rn[None, :], acc,
                 mask=(rm[:, None] < M) & (rn[None, :] < N))
        tl.store(sq_ptr + s.to(tl.int64) * sq_ss + rm, sq, mask=rm < M)

    @triton.jit
    def _reduce_parts_kernel(dp_ptr, sp_ptr, d_ptr, d_ss, sq_ptr, sq_ss, MN, M, P: tl.constexpr, BLOCK: tl.constexpr):
        """d[s] = sum_p dp[s*P+p] (flat [M*N]) and sq[s] = sum_p sp[s*P+p] ([M]), fixed order (deterministic)."""
        s = tl.program_id(0)
        e = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        in_d = e < MN
        in_s = (e >= MN) & (e < MN + M)
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for p in range(P):
            base = (s * P + p).to(tl.int64)
            acc += tl.load(dp_ptr + base * MN + e, mask=in_d, other=0.0)
            acc += tl.load(sp_ptr + base * M + (e - MN), mask=in_s, other=0.0)
        tl.store(d_ptr + s.to(tl.int64) * d_ss + e, acc, mask=in_d)
        tl.store(sq_ptr + s.to(tl.int64) * sq_ss + (e - MN), acc, mask=in_s)


_SMS: dict = {}


def _num_sms(device) -> int:
    key = str(device)
    if key not in _SMS:
        _SMS[key] = torch.cuda.get_device_properties(device).multi_processor_count
    return _SMS[key]


def tf32_hc_prenorm_gemm(a, b, d, sqr_sum, num_splits=None) -> bool:
    """Triton implementation; returns False if it does not apply (caller uses the torch path)."""
    if triton is None or impl() != "triton" or a.device.type != "cuda":
        return False
    try:
        m, k = a.shape
        n = b.shape[0]
        if m == 0:
            return True
        if a.stride(1) != 1 or b.stride(1) != 1 or b.dtype != torch.float32 or n > 64:
            return False
        S = 1 if num_splits is None else int(num_splits)
        d3 = d.view(1, *d.shape) if num_splits is None else d
        sq2 = sqr_sum.view(1, -1) if num_splits is None else sqr_sum
        if d3.stride(2) != 1 or tuple(d3.shape) != (S, m, n) or tuple(sq2.shape) != (S, m) or sq2.stride(1) != 1:
            return False
        step = -(-k // S)
        BN = max(16, triton.next_power_of_2(n))
        BM = 16 if m <= 16 else (32 if m <= 32 else 64)
        BK = 64
        rb = triton.cdiv(m, BM)
        # few rows (decode): split every K chunk again over P CTAs and reduce the P sub-partials in a fixed
        # order afterwards (deterministic, no atomics); many rows: the (row block, split) grid is enough
        P = 1
        while P < 16 and rb * S * P < 2 * _num_sms(a.device) and step // (2 * P) >= BK:
            P *= 2
        sub = -(-(-(-step // P)) // BK) * BK
        if P == 1:
            _hc_prenorm_kernel[(rb, S)](a, a.stride(0), b, b.stride(0), d3, d3.stride(0), d3.stride(1), sq2,
                                        sq2.stride(0), m, n, k, step, step, P=1, BM=BM, BN=BN, BK=BK,
                                        PREC=precision(), num_warps=4)
            return True
        dp = torch.empty((S * P, m, n), dtype=torch.float32, device=a.device)
        sp = torch.empty((S * P, m), dtype=torch.float32, device=a.device)
        _hc_prenorm_kernel[(rb, S * P)](a, a.stride(0), b, b.stride(0), dp, dp.stride(0), dp.stride(1), sp,
                                        sp.stride(0), m, n, k, step, sub, P=P, BM=BM, BN=BN, BK=BK,
                                        PREC=precision(), num_warps=4)
        if d3.stride(1) == n and d3.stride(2) == 1:  # one fused, ordered reduce for both partial sets
            RB = 256
            _reduce_parts_kernel[(S, triton.cdiv(m * n + m, RB))](dp, sp, d3, d3.stride(0), sq2, sq2.stride(0),
                                                                  m * n, m, P=P, BLOCK=RB, num_warps=4)
        else:
            torch.sum(dp.view(S, P, m, n), dim=1, out=d3)
            torch.sum(sp.view(S, P, m), dim=1, out=sq2)
        return True
    except Exception as exc:  # noqa: BLE001
        if os.environ.get("SM120_ATTN_STRICT", "0") == "1":
            raise
        _log_once(("hc_fail", type(exc).__name__), f"triton tf32_hc_prenorm_gemm failed ({exc}); using torch")
        return False
