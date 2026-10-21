"""SM120 Triton kernel for deep_gemm.fp8_einsum("bhr,hdr->bhd", recipe=(1, 1, 32)) (V4.1 wo_a projection).

DeepInfer (v4_modules/attn.py _wv_and_o_proj):
    fp8_einsum("bhr,hdr->bhd", (ctx_fp8 [T,G,K] e4m3, ctx_sf int32 [T,G,K/128] packed ue8m0),
               (wv [G,N,K] e4m3, wv_sf int32 [G,N,K/128] packed ue8m0), d, recipe=(1,1,32))
with d either bf16 [T,G,N] or the FP8 output pair (e4m3 [T,G,N], int32 [T, G*N/128] packed ue8m0, any strides,
DeepGEMM per_token_cast_to_fp8(gran 32) of the bf16 result).

Kernel: grid (G, T tiles, N tiles). Operands are dequantised in registers (e4m3 * 2^e is exact in bf16),
bf16 tensor-core dot with fp32 accumulation (= DeepGEMM's FP8 MMA on exact products), bf16 rounding of
the result (as the torch path's bmm), then either a bf16 store or the fused per-32 FP8 cast with the
shim's exact quantiser semantics (deep_gemm/_sf.quant_fp8_rows: amax clamp 1e-4, ceil-ue8m0 scale).
The weight is read as FP8 from the parameter itself (no bf16 dequant cache needed).

Switch: SM120_EINSUM_IMPL = triton (default) | torch. Unsupported layouts / failures -> torch.
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
    v = os.environ.get("SM120_EINSUM_IMPL", "triton").strip().lower()
    if v not in ("triton", "torch"):
        raise ValueError(f"SM120_EINSUM_IMPL must be 'triton' or 'torch', got {v!r}")
    return v


def _log_once(key, msg):
    if key not in _LOGGED:
        _LOGGED.add(key)
        print(f"[sm120_einsum] {msg}", flush=True)


if triton is not None:

    @triton.jit
    def _ue8m0(b):
        return (b.to(tl.int32) << 23).to(tl.float32, bitcast=True)

    @triton.jit
    def _einsum_kernel(a_ptr, a_s0, a_s1, sa_ptr, sa_s0, sa_s1, sa_s2,
                       w_ptr, w_s0, w_s1, sw_ptr, sw_s0, sw_s1, sw_s2,
                       o_ptr, o_s0, o_s1, so_ptr, so_s0, so_s1, T, N, K,
                       OUT_FP8: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        g = tl.program_id(0)
        pm = tl.program_id(1)
        pn = tl.program_id(2)
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        mm = rm < T
        nm = rn < N
        rm64 = rm.to(tl.int64)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        NG: tl.constexpr = BK // 32
        for k0 in range(0, K, BK):
            rk = k0 + tl.arange(0, BK)
            kg = k0 // 32 + tl.arange(0, NG)  # per-32 scale groups of this K block
            a = tl.load(a_ptr + rm64[:, None] * a_s0 + g * a_s1 + rk[None, :], mask=mm[:, None], other=0.0)
            sa = tl.load(sa_ptr + (rm64[:, None] * sa_s0 + g * sa_s1 + (kg // 4)[None, :] * sa_s2) * 4
                         + (kg % 4)[None, :], mask=mm[:, None], other=127)
            af = tl.reshape(tl.reshape(a.to(tl.float32), (BM, NG, 32)) * _ue8m0(sa)[:, :, None], (BM, BK))
            w = tl.load(w_ptr + g * w_s0 + rn[:, None].to(tl.int64) * w_s1 + rk[None, :], mask=nm[:, None], other=0.0)
            sw = tl.load(sw_ptr + (g * sw_s0 + rn[:, None].to(tl.int64) * sw_s1 + (kg // 4)[None, :] * sw_s2) * 4
                         + (kg % 4)[None, :], mask=nm[:, None], other=127)
            wf = tl.reshape(tl.reshape(w.to(tl.float32), (BN, NG, 32)) * _ue8m0(sw)[:, :, None], (BN, BK))
            acc = tl.dot(af.to(tl.bfloat16), tl.trans(wf.to(tl.bfloat16)), acc)
        r = acc.to(tl.bfloat16)
        if OUT_FP8:
            # bf16 result -> per-32 e4m3 (columns of the flattened [G*N] row), DeepGEMM per_token_cast_to_fp8
            rf = tl.reshape(r.to(tl.float32), (BM, BN // 32, 32))
            amax = tl.maximum(tl.max(tl.abs(rf), axis=2), 1e-4)
            t = tl.math.div_rn(amax, 448.0)
            bits = t.to(tl.int32, bitcast=True)
            e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
            e = tl.minimum(tl.maximum(e, 1), 254)
            inv = ((254 - e) << 23).to(tl.float32, bitcast=True)  # exactly 1 / 2^(e-127)
            q = tl.reshape(rf * inv[:, :, None], (BM, BN)).to(tl.float8e4nv)
            col = g * N + rn
            tl.store(o_ptr + rm64[:, None] * o_s0 + col[None, :], q, mask=mm[:, None] & nm[None, :])
            cg = (g * N + pn * BN) // 32 + tl.arange(0, BN // 32)
            gm = (pn * BN + tl.arange(0, BN // 32) * 32) < N
            tl.store(so_ptr + (rm64[:, None] * so_s0 + (cg // 4)[None, :] * so_s1) * 4 + (cg % 4)[None, :],
                     e.to(tl.uint8), mask=mm[:, None] & gm[None, :])
        else:
            tl.store(o_ptr + rm64[:, None] * o_s0 + g * o_s1 + rn[None, :], r, mask=mm[:, None] & nm[None, :])


def _u8(t: torch.Tensor) -> torch.Tensor:
    """Byte view starting at t's first element (strides are applied in the kernel in int32 units)."""
    return t.as_strided((1,), (1,), t.storage_offset()).view(torch.uint8)


def fp8_einsum_bhr_hdr(a_t, sfa, b_t, sfb, d, gran_k) -> bool:
    """Returns False if this kernel does not apply (caller uses the torch path)."""
    if triton is None or impl() != "triton":
        return False
    try:
        if a_t.dtype != torch.float8_e4m3fn or b_t.dtype != torch.float8_e4m3fn or gran_k != 32:
            return False
        if sfa is None or sfb is None or sfa.dtype != torch.int32 or sfb.dtype != torch.int32:
            return False
        if a_t.dim() != 3 or b_t.dim() != 3 or a_t.device.type != "cuda":
            return False
        T, G, K = a_t.shape
        G2, N, K2 = b_t.shape
        if G2 != G or K2 != K or K % 64 or a_t.stride(2) != 1 or b_t.stride(2) != 1:
            return False
        if tuple(sfa.shape) != (T, G, K // 128) or tuple(sfb.shape) != (G, N, K // 128):
            return False
        out_fp8 = isinstance(d, (tuple, list))
        if out_fp8:
            d_fp8, d_sf = d
            if (d_fp8.dtype != torch.float8_e4m3fn or tuple(d_fp8.shape) != (T, G, N) or not d_fp8.is_contiguous()
                    or d_sf.dtype != torch.int32 or tuple(d_sf.shape) != (T, G * N // 128) or N % 32):
                return False
            o, o_s0, o_s1 = d_fp8, G * N, N
            so, so_s0, so_s1 = _u8(d_sf), d_sf.stride(0), d_sf.stride(1)
        else:
            if d.dtype != torch.bfloat16 or tuple(d.shape) != (T, G, N) or d.stride(2) != 1:
                return False
            o, o_s0, o_s1 = d, d.stride(0), d.stride(1)
            so, so_s0, so_s1 = o, 0, 0
        if T == 0:
            return True
        # decode: G * N/BN CTAs must cover the SMs (the weight stream is the cost) -> narrow N tiles. Prefill
        # (RTX 5090 sweep, G=8 N=1024 K=4096, bit-identical across configs): 64x64 tiles up to T=2048 (T=512 211 vs
        # 276 us, T=2048 721 vs 893 us), 128x128 above; 4 warps everywhere (T=4096 1452 vs 1749 us with 8 warps)
        BM = 16 if T <= 16 else (32 if T <= 32 else (64 if T <= 2048 else 128))
        BN = int(os.environ.get("SM120_EINSUM_BN", "0")) or (32 if T <= 32 else (64 if T <= 2048 else 128))
        grid = (G, triton.cdiv(T, BM), triton.cdiv(N, BN))
        _einsum_kernel[grid](a_t, a_t.stride(0), a_t.stride(1), _u8(sfa), sfa.stride(0), sfa.stride(1), sfa.stride(2),
                             b_t, b_t.stride(0), b_t.stride(1), _u8(sfb), sfb.stride(0), sfb.stride(1), sfb.stride(2),
                             o, o_s0, o_s1, so, so_s0, so_s1, T, N, K, OUT_FP8=out_fp8, BM=BM, BN=BN, BK=64,
                             num_warps=4, num_stages=3)
        return True
    except Exception as exc:  # noqa: BLE001
        if os.environ.get("SM120_ATTN_STRICT", "0") == "1":
            raise
        _log_once(("einsum_fail", type(exc).__name__), f"triton fp8_einsum failed ({exc}); using torch")
        return False
