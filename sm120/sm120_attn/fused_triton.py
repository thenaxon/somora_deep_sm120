"""Triton kernels for the non-attention parts of fused_norm_rope_attn_rope_cast (SM120).

Same math and layouts as the torch reference in fused.py (one program per (token, head)):

  prepare_q:  q (permute_q_b_proj layout or natural) -> natural q_r = bf16(rope(q)), the RMS factor
              r = rsqrt(mean(q^2) + eps) of the pre-RoPE q, and optionally q_fold = bf16(q_r * r)
              (what the b12x attention core consumes).
  rope_cast:  attention output [s, h, 512] (bf16 or fp32) -> inverse RoPE in fp32 -> per-32 FP8
              e4m3 with ue8m0 scale ceil_pow2(max(amax, 1e-4) / 448), written in the wv layout
              (permuted or natural) and the int32 TMA-aligned col-major packed scale tensor.

Differences vs torch are at the last-ulp level (FMA contraction in the RoPE, reduction order of the
RMS sum); FP8 codes can differ by one step on rounding ties. Selected with SM120_ATTN_FUSED_IMPL
(triton | torch, default triton); any failure falls back to torch.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001
    triton = None
    tl = None

D_HEAD = 512
ROPE = 64


def available() -> bool:
    return triton is not None


if triton is not None:

    @triton.jit
    def _rope_vals(x, xp, d, pos, cs_ptr, cs_stride, NOPE: tl.constexpr, NEOX: tl.constexpr, CONJ: tl.constexpr):
        """x: values at channels d, xp: values at the RoPE partner channel. Returns rotated values."""
        is_rope = d >= NOPE
        r = d - NOPE
        if NEOX:
            i = r % 32
            first = r < 32
        else:
            i = r // 2
            first = (r % 2) == 0
        i = tl.where(is_rope, i, 0)
        cos = tl.load(cs_ptr + pos * cs_stride + i, mask=is_rope, other=1.0)
        sin = tl.load(cs_ptr + pos * cs_stride + 32 + i, mask=is_rope, other=0.0)
        if CONJ:
            sin = -sin
        # torch reference: r0 = x0*cos - x1*sin ; r1 = x0*sin + x1*cos
        y = tl.where(first, x * cos - xp * sin, xp * sin + x * cos)
        return tl.where(is_rope, y, x)

    @triton.jit
    def _partner(d, NOPE: tl.constexpr, NEOX: tl.constexpr):
        r = d - NOPE
        if NEOX:
            p = tl.where(r < 32, d + 32, d - 32)
        else:
            p = d ^ 1
        return tl.where(d >= NOPE, p, d)

    @triton.jit
    def _prepare_q_kernel(q_ptr, q_stride, pos_ptr, cs_ptr, cs_stride, qr_ptr, qf_ptr, sc_ptr, eps,
                          H: tl.constexpr, D: tl.constexpr, NOPE: tl.constexpr, PERMUTE: tl.constexpr,
                          NORM: tl.constexpr, NEOX: tl.constexpr, FOLD: tl.constexpr):
        s = tl.program_id(0)
        h = tl.program_id(1)
        d = tl.arange(0, D)
        dp = _partner(d, NOPE, NEOX)
        base = q_ptr + s.to(tl.int64) * q_stride
        if PERMUTE:
            src = (d // 16) * (H * 16) + h * 16 + (d % 16)
            srcp = (dp // 16) * (H * 16) + h * 16 + (dp % 16)
        else:
            src = h * D + d
            srcp = h * D + dp
        x = tl.load(base + src).to(tl.float32)
        xp = tl.load(base + srcp).to(tl.float32)
        pos = tl.load(pos_ptr + s).to(tl.int64)
        y = _rope_vals(x, xp, d, pos, cs_ptr, cs_stride, NOPE, NEOX, False).to(tl.bfloat16)
        out_off = (s.to(tl.int64) * H + h) * D + d
        tl.store(qr_ptr + out_off, y)
        if NORM:
            r = 1.0 / tl.sqrt_rn(tl.sum(x * x, axis=0) / D + eps)
            tl.store(sc_ptr + s.to(tl.int64) * H + h, r)
            if FOLD:
                tl.store(qf_ptr + out_off, (y.to(tl.float32) * r).to(tl.bfloat16))

    @triton.jit
    def _rope_cast_kernel(o_ptr, o_stride_s, o_stride_h, pos_ptr, cs_ptr, cs_stride, fp8_ptr, sf_ptr, a_s,
                          H: tl.constexpr, GS: tl.constexpr, NOPE: tl.constexpr, PERMUTE: tl.constexpr,
                          NEOX: tl.constexpr):
        s = tl.program_id(0)
        h = tl.program_id(1)
        c = tl.arange(0, 16)[:, None]
        j = tl.arange(0, 32)[None, :]
        d = c * 32 + j
        dp = _partner(d, NOPE, NEOX)
        base = o_ptr + s.to(tl.int64) * o_stride_s + h * o_stride_h
        x = tl.load(base + d).to(tl.float32)
        xp = tl.load(base + dp).to(tl.float32)
        pos = tl.load(pos_ptr + s).to(tl.int64)
        y = _rope_vals(x, xp, d, pos, cs_ptr, cs_stride, NOPE, NEOX, True)
        amax = tl.max(tl.abs(y), axis=1)  # [16]
        v = tl.math.div_rn(tl.maximum(amax, 1e-4), 448.0)
        bits = v.to(tl.int32, bitcast=True)
        e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
        e = tl.minimum(tl.maximum(e, 1), 254)
        sf = (e << 23).to(tl.float32, bitcast=True)
        qv = tl.minimum(tl.maximum(tl.math.div_rn(y, sf[:, None]), -448.0), 448.0).to(tl.float8e4nv)
        g = h // GS
        hg = h % GS
        cc = tl.arange(0, 16)
        if PERMUTE:
            k = cc * GS + hg
        else:
            k = hg * 16 + cc
        G = H // GS
        dst = s.to(tl.int64) * (G * GS * 512) + g * (GS * 512) + k[:, None] * 32 + j
        tl.store(fp8_ptr + dst, qv)
        k4 = GS * 4
        sf_off = ((g * k4 + k // 4).to(tl.int64) * a_s + s) * 4 + (k % 4)
        tl.store(sf_ptr + sf_off, e.to(tl.uint8))


def _check(t: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor, rope_dim: int):
    if t.device.type != "cuda" or t.dim() != 3 or t.shape[-1] != D_HEAD or rope_dim != ROPE:
        raise ValueError(f"triton fused path needs CUDA [s, h, 512] with rope_dim 64 (got {tuple(t.shape)}, {rope_dim})")
    if cos_sin_cache.dtype != torch.float32 or cos_sin_cache.stride(-1) != 1 or cos_sin_cache.shape[-1] < ROPE:
        raise ValueError("cos_sin_cache must be float32 [n, >=64] with unit inner stride")
    if positions.numel() != t.shape[0]:
        raise ValueError("positions must have one entry per token")


def prepare_q(q: torch.Tensor, enable_q_norm: bool, eps: float, positions: torch.Tensor, is_neox: bool,
              rope_dim: int, cos_sin_cache: torch.Tensor, permute: bool, fold: bool):
    """Returns (q_r bf16 [s,h,512], q_scale fp32 [s,h] | None, q_fold bf16 [s,h,512] | None)."""
    _check(q, positions, cos_sin_cache, rope_dim)
    s, h, d = q.shape
    if q.stride(2) != 1 or q.stride(1) != d:
        raise ValueError("q must be contiguous within each token")
    dev = q.device
    q_r = torch.empty((s, h, d), dtype=torch.bfloat16, device=dev)
    sc = torch.empty((s, h), dtype=torch.float32, device=dev) if enable_q_norm else None
    qf = torch.empty((s, h, d), dtype=torch.bfloat16, device=dev) if (enable_q_norm and fold) else None
    if s:
        pos = positions.reshape(s)
        if not pos.is_contiguous():
            pos = pos.contiguous()
        _prepare_q_kernel[(s, h)](q, q.stride(0), pos, cos_sin_cache, cos_sin_cache.stride(0), q_r,
                                  qf if qf is not None else q_r, sc if sc is not None else q_r, float(eps),
                                  H=h, D=d, NOPE=d - ROPE, PERMUTE=bool(permute), NORM=bool(enable_q_norm),
                                  NEOX=bool(is_neox), FOLD=qf is not None, num_warps=4)
    return q_r, sc, (qf if qf is not None else (q_r if fold else None))


def rope_cast(out: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor, rope_dim: int,
              is_neox: bool, n_wv_group: int, permute: bool, sf_out: torch.Tensor):
    """out [s,h,512] bf16/fp32 -> (fp8 [s,G,gs*512], sf_out) ; sf_out from quant.alloc_tma_colmajor_sf."""
    _check(out, positions, cos_sin_cache, rope_dim)
    s, h, d = out.shape
    if out.stride(2) != 1:
        out = out.contiguous()
    gs = h // n_wv_group
    fp8 = torch.empty((s, n_wv_group, gs * d), dtype=torch.float8_e4m3fn, device=out.device)
    if s:
        pos = positions.reshape(s)
        if not pos.is_contiguous():
            pos = pos.contiguous()
        a_s = sf_out.stride(2)  # aligned s of the [G, K4, align4(s)] base storage
        assert sf_out.stride(0) == 1 and sf_out.stride(1) == sf_out.shape[2] * a_s
        base = sf_out.as_strided((1,), (1,), sf_out.storage_offset()).view(torch.uint8)
        _rope_cast_kernel[(s, h)](out, out.stride(0), out.stride(1), pos, cos_sin_cache, cos_sin_cache.stride(0),
                                  fp8, base, a_s, H=h, GS=gs, NOPE=d - ROPE, PERMUTE=bool(permute),
                                  NEOX=bool(is_neox), num_warps=4)
    return fp8, sf_out


# ---------------------------------------------------------------------------------------------
# V4.1 KV record encoder for the b12x prefill (one pass per token):
#   fp8 record [528] = quant.quantize_v41_fp8_records, fp4 record [288] = quant.quantize_v41_fp4_records,
#   flag = 1 iff dequant(fp4 record) == x exactly (the token came from an FP4 cache record).
# ---------------------------------------------------------------------------------------------
if triton is not None:

    @triton.jit
    def _encode_v41_kernel(kv_ptr, kv_stride, fp8_ptr, fp4_ptr, flag_ptr, WRITE_FP4: tl.constexpr):
        row = tl.program_id(0)
        r64 = row.to(tl.int64)
        # ---- FP8: 16 groups of 32, ue8m0 scale 2^ceil(log2(max(amax/448, 1e-4)))
        g8 = tl.arange(0, 16)[:, None]
        j32 = tl.arange(0, 32)[None, :]
        x8 = tl.load(kv_ptr + r64 * kv_stride + g8 * 32 + j32).to(tl.float32)
        t = tl.maximum(tl.math.div_rn(tl.max(tl.abs(x8), axis=1), 448.0), 1e-4)
        bits = t.to(tl.int32, bitcast=True)
        e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
        sf8 = (e << 23).to(tl.float32, bitcast=True)
        q8 = tl.minimum(tl.maximum(tl.math.div_rn(x8, sf8[:, None]), -448.0), 448.0).to(tl.float8e4nv)
        base8 = fp8_ptr + r64 * 528
        tl.store(base8 + g8 * 32 + j32, q8.to(tl.uint8, bitcast=True))
        tl.store(base8 + 512 + tl.arange(0, 16), e.to(tl.uint8))
        if WRITE_FP4:
            # ---- FP4: 32 groups of 16, e4m3 scale = e4m3(clamp(amax/6, 2^-9, 448)), e2m1 RNE codes
            g4 = tl.arange(0, 32)[:, None]
            j16 = tl.arange(0, 16)[None, :]
            x4 = tl.load(kv_ptr + r64 * kv_stride + g4 * 16 + j16).to(tl.float32)
            s = tl.minimum(tl.maximum(tl.math.div_rn(tl.max(tl.abs(x4), axis=1), 6.0), 0.001953125),
                           448.0).to(tl.float8e4nv)
            sf4 = s.to(tl.float32)
            norm = tl.math.div_rn(x4, sf4[:, None])  # IEEE division: e2m1 rounding ties must be exact
            a = tl.abs(norm)
            a = tl.where(a != a, 0.0, a)
            a = tl.minimum(a, 6.0)
            c_gt = ((a > 0.25).to(tl.int32) + (a > 0.75).to(tl.int32) + (a > 1.25).to(tl.int32)
                    + (a > 1.75).to(tl.int32) + (a > 2.5).to(tl.int32) + (a > 3.5).to(tl.int32)
                    + (a > 5.0).to(tl.int32))
            tie = ((a == 0.25) | (a == 0.75) | (a == 1.25) | (a == 1.75) | (a == 2.5) | (a == 3.5)
                   | (a == 5.0))
            code = c_gt + (tie.to(tl.int32) & (c_gt & 1))
            sign = (norm.to(tl.int32, bitcast=True) >> 31) & 1
            c = (sign << 3) | code
            lo, hi = tl.split(tl.reshape(c, [32, 8, 2]))
            packed = (lo | (hi << 4)).to(tl.uint8)
            base4 = fp4_ptr + r64 * 288
            tl.store(base4 + tl.arange(0, 32)[:, None] * 8 + tl.arange(0, 8)[None, :], packed)
            tl.store(base4 + 256 + tl.arange(0, 32), s.to(tl.uint8, bitcast=True))
            mag = tl.where(code < 4, code.to(tl.float32) * 0.5,
                           tl.where(code == 4, 2.0, tl.where(code == 5, 3.0, tl.where(code == 6, 4.0, 6.0))))
            val = tl.where(sign == 1, -mag, mag) * sf4[:, None]
            ok = tl.min(tl.min((val == x4).to(tl.int32), axis=1), axis=0)
            tl.store(flag_ptr + row, ok.to(tl.int8))


def encode_v41(kv: torch.Tensor, fp8_records: torch.Tensor, fp4_records: torch.Tensor | None):
    """kv [n, 512] bf16/fp32 -> fills fp8_records [>=n, 528] and fp4_records [>=n, 288] (uint8, contiguous rows);
    returns the bool FP4-exact flag [n] (None without fp4_records)."""
    n = kv.shape[0]
    if kv.dim() != 2 or kv.shape[1] != D_HEAD or kv.stride(1) != 1:
        raise ValueError("kv must be [n, 512] with unit inner stride")
    flag = torch.empty((n,), dtype=torch.int8, device=kv.device) if fp4_records is not None else None
    if n:
        _encode_v41_kernel[(n,)](kv, kv.stride(0), fp8_records, fp4_records if fp4_records is not None else fp8_records,
                                 flag if flag is not None else fp8_records, WRITE_FP4=fp4_records is not None,
                                 num_warps=4)
    return None if flag is None else flag.bool()
