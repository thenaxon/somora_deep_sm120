"""Scale-factor layouts, (de)quantisation helpers shared by every deep_gemm shim function.

Layout contract (identical to DeepGEMM on SM100, so FlashMLA-style weight permutations, DeepInfer's
own `.view()`s and `tile_kernels` outputs all stay valid):

* "packed UE8M0" SF: torch.int32 tensor ``[..., MN, ceil(K / gran_k / 4)]``; every int32 holds four
  UE8M0 exponent bytes (little endian, byte j <-> K-group 4*i + j). Padding groups hold 127 (= 1.0).
* MN-major, TMA aligned: stride(-2) == 1, stride(-1) == align(MN, 4) (``get_tma_aligned_size``),
  for 3D: stride(-3) == stride(-1) * size(-1).
* Weight SFs are always *per row* after ``transform_sf_into_required_layout`` (block scales are
  broadcast along MN by ``gran_mn``), activations are per row by construction.

All readers below accept any stride (they call ``.contiguous()``), so callers may pass the
MN-major tensors directly, views of them (DeepInfer's ``context_sf.view(T, g, -1)``) or row-major
copies. FP32 SFs (SM90 style / un-transformed) are accepted as well.
"""

from __future__ import annotations

import math

import torch

FP8_MAX = 448.0
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
_LUT: dict = {}


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def get_tma_aligned_size(x: int, element_size: int) -> int:
    assert 16 % element_size == 0, element_size
    return align(int(x), 16 // int(element_size))


def is_packed_fp4(t: torch.Tensor) -> bool:
    return t.dtype in (torch.int8, torch.uint8)


def _lut(device) -> torch.Tensor:
    key = str(device)
    t = _LUT.get(key)
    if t is None:
        t = torch.tensor(_E2M1, dtype=torch.float32, device=device)
        _LUT[key] = t
    return t


# ----------------------------------------------------------------------------------------------
# UE8M0 <-> float
# ----------------------------------------------------------------------------------------------
def ue8m0_bytes_to_float(b: torch.Tensor) -> torch.Tensor:
    """uint8 exponent bytes -> float32 2^(b-127) (b == 0 gives 2^-127 as a denormal-free 0-ish)."""
    return (b.to(torch.int32) << 23).view(torch.float32)


def float_to_ue8m0_bytes(sf: torch.Tensor) -> torch.Tensor:
    """Power-of-two float32 scales -> exponent bytes (mantissa is dropped, as DeepGEMM does)."""
    return ((sf.float().contiguous().view(torch.int32) >> 23) & 0xFF).to(torch.uint8)


def ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    bits = x.abs().float().contiguous().view(torch.int32)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).ne(0).to(torch.int32)
    return (exp.clamp(1, 254) << 23).view(torch.float32)


def packed_sf_bytes(sf: torch.Tensor) -> torch.Tensor:
    """int32 packed UE8M0 [..., mn, kp] (any strides) -> uint8 [..., mn, 4*kp] contiguous."""
    assert sf.dtype == torch.int32, sf.dtype
    return sf.contiguous().view(torch.uint8)


# ----------------------------------------------------------------------------------------------
# MN-major TMA-aligned constructors (DeepGEMM utils.layout semantics)
# ----------------------------------------------------------------------------------------------
def empty_mn_major(shape: tuple[int, ...], dtype: torch.dtype, device) -> torch.Tensor:
    """Allocate ``shape`` = [..., mn, k] with DeepGEMM's MN-major TMA-aligned strides."""
    shape = tuple(int(s) for s in shape)
    assert len(shape) in (2, 3), shape
    mn, k = shape[-2], shape[-1]
    tma = get_tma_aligned_size(mn, torch.empty((), dtype=dtype).element_size())
    if len(shape) == 2:
        return torch.empty_strided((mn, k), (1, tma), dtype=dtype, device=device)
    g = shape[0]
    return torch.empty_strided((g, mn, k), (tma * k, 1, tma), dtype=dtype, device=device)


def get_mn_major_tma_aligned_tensor(sf: torch.Tensor) -> torch.Tensor:
    out = empty_mn_major(tuple(sf.shape), sf.dtype, sf.device)
    out.copy_(sf)
    return out


def pack_bytes_mn_major(b: torch.Tensor) -> torch.Tensor:
    """uint8 exponent bytes [..., mn, groups] -> int32 packed [..., mn, ceil(groups/4)], MN-major."""
    squeeze = b.dim() == 2
    if squeeze:
        b = b.unsqueeze(0)
    g, mn, k = b.shape
    k4 = align(k, 4)
    if k4 != k:
        pad = torch.full((g, mn, k4 - k), 127, dtype=torch.uint8, device=b.device)
        b = torch.cat([b, pad], dim=-1)
    packed = b.contiguous().view(torch.int32)  # [g, mn, k4/4]
    out = empty_mn_major((g, mn, k4 // 4), torch.int32, b.device)
    out.copy_(packed)
    return out.squeeze(0) if squeeze else out


def get_mn_major_tma_aligned_packed_ue8m0_tensor(sf: torch.Tensor, psum_layout=None) -> torch.Tensor:
    """FP32 power-of-two SF [..., mn, groups] -> packed int32 MN-major (DeepGEMM semantics)."""
    del psum_layout  # only lets the CUDA packer skip gap rows; the values are identical
    assert sf.dtype == torch.float32, sf.dtype
    return pack_bytes_mn_major(float_to_ue8m0_bytes(sf))


# ----------------------------------------------------------------------------------------------
# Per-element scale expansion
# ----------------------------------------------------------------------------------------------
def infer_gran_k(sf: torch.Tensor, k: int) -> int:
    """Guess the K granularity of an SF tensor from its last dimension."""
    last = int(sf.shape[-1])
    per = 4 if sf.dtype == torch.int32 else 1
    for g in (32, 128, 16, 64):
        if ceil_div(ceil_div(k, g), per) == last:
            return g
    raise ValueError(f"cannot infer SF K-granularity: sf.shape={tuple(sf.shape)} dtype={sf.dtype} k={k}")


def sf_to_group_scales(sf: torch.Tensor, mn: int, k: int, gran_k: int | None = None) -> tuple[torch.Tensor, int]:
    """Any supported SF -> (float32 [..., mn, ceil(k/gran_k)] per-row group scales, gran_k).

    Accepts packed int32 UE8M0 (per row or per MN-block) and FP32 scales (per row or per block).
    """
    if gran_k is None:
        gran_k = infer_gran_k(sf, k)
    groups = ceil_div(k, gran_k)
    if sf.dtype == torch.int32:
        s = ue8m0_bytes_to_float(packed_sf_bytes(sf))[..., :groups]
    elif sf.dtype == torch.float8_e8m0fnu or sf.dtype == torch.uint8:
        s = ue8m0_bytes_to_float(sf.contiguous().view(torch.uint8))[..., :groups]
    elif sf.is_floating_point():
        s = sf.float()
        if s.shape[-1] != groups:
            raise ValueError(f"fp32 SF last dim {s.shape[-1]} != ceil(k/gran_k)={groups}")
    else:
        raise TypeError(f"unsupported SF dtype {sf.dtype}")
    rows = int(s.shape[-2])
    if rows != mn:
        gran_mn = ceil_div(mn, rows) if rows > 0 else 1
        # block scales: broadcast along MN
        cand = [g for g in (32, 128, gran_mn) if ceil_div(mn, g) == rows]
        if not cand:
            raise ValueError(f"SF rows {rows} incompatible with mn={mn}")
        s = s.repeat_interleave(cand[0], dim=-2)[..., :mn, :]
    return s, gran_k


def _apply_scales(vals: torch.Tensor, scales: torch.Tensor, gran_k: int, out_dtype: torch.dtype) -> torch.Tensor:
    """vals float32 [..., mn, k] * group scales [..., mn, ceil(k/g)] -> out_dtype [..., mn, k]."""
    k = vals.shape[-1]
    groups = scales.shape[-1]
    kp = groups * gran_k
    if kp != k:
        vals = torch.nn.functional.pad(vals, (0, kp - k))
    shp = vals.shape[:-1]
    out = (vals.view(*shp, groups, gran_k) * scales.unsqueeze(-1)).view(*shp, kp)
    if kp != k:
        out = out[..., :k]
    return out.to(out_dtype)


def dequant_fp8(x: torch.Tensor, sf: torch.Tensor, gran_k: int | None = None,
                out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """E4M3 [..., mn, k] + SF -> out_dtype. Exact in bf16 (e4m3 * 2^e has 4 significant bits)."""
    assert x.dtype == torch.float8_e4m3fn, x.dtype
    mn, k = int(x.shape[-2]), int(x.shape[-1])
    scales, g = sf_to_group_scales(sf, mn, k, gran_k)
    return _apply_scales(x.float(), scales, g, out_dtype)


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """[..., k/2] bytes (even element = low nibble, DeepGEMM/HF convention) -> float32 [..., k]."""
    p = packed.contiguous().view(torch.uint8)
    lut = _lut(p.device)
    lo = lut[(p & 0x0F).long()]
    hi = lut[(p >> 4).long()]
    return torch.stack((lo, hi), dim=-1).flatten(-2)


def dequant_fp4(x: torch.Tensor, sf: torch.Tensor, gran_k: int | None = 32,
                out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Packed e2m1 int8 [..., mn, k/2] + SF -> out_dtype [..., mn, k] (exact in bf16)."""
    mn, k = int(x.shape[-2]), int(x.shape[-1]) * 2
    scales, g = sf_to_group_scales(sf, mn, k, gran_k)
    return _apply_scales(unpack_e2m1(x), scales, g, out_dtype)


def dequant_any(x: torch.Tensor, sf: torch.Tensor | None, gran_k: int | None = None,
                out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    if x.dtype == torch.float8_e4m3fn:
        assert sf is not None
        return dequant_fp8(x, sf, gran_k, out_dtype)
    if is_packed_fp4(x):
        assert sf is not None
        return dequant_fp4(x, sf, gran_k if gran_k is not None else 32, out_dtype)
    return x.to(out_dtype)


def logical_k(x: torch.Tensor) -> int:
    return int(x.shape[-1]) * (2 if is_packed_fp4(x) else 1)


def unpack_operand(op) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(op, (tuple, list)):
        assert len(op) == 2, len(op)
        return op[0], op[1]
    return op, None


# ----------------------------------------------------------------------------------------------
# Quantisation (DeepGEMM per_token_cast_to_fp8 semantics: amax clamp 1e-4, ceil-UE8M0 scale)
# ----------------------------------------------------------------------------------------------
def quant_fp8_rows(x: torch.Tensor, gran_k: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """[m, n] -> (e4m3 [m, n], float32 power-of-two scales [m, ceil(n/gran_k)])."""
    m, n = x.shape
    npad = align(n, gran_k)
    xf = x.float()
    if npad != n:
        xf = torch.nn.functional.pad(xf, (0, npad - n))
    v = xf.view(m, npad // gran_k, gran_k)
    amax = v.abs().amax(dim=-1).clamp(min=1e-4)
    sf = ceil_to_ue8m0(amax / FP8_MAX)
    q = (v * (1.0 / sf.unsqueeze(-1))).to(torch.float8_e4m3fn).view(m, npad)[:, :n].contiguous()
    return q, sf


def quant_dequant_mxfp8(x: torch.Tensor, gran_k: int = 32, out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Emulate an MXFP8 activation cast (per-row 1x32 UE8M0) and return the dequantised values."""
    shp = x.shape
    x2 = x.reshape(-1, shp[-1])
    if x2.shape[0] == 0:
        return x.to(out_dtype)
    q, sf = quant_fp8_rows(x2, gran_k)
    return _apply_scales(q.float(), sf, gran_k, out_dtype).view(shp)


def chunk_rows(bytes_per_row: int, total: int, budget: int) -> int:
    if total <= 0:
        return 1
    return max(1, min(total, budget // max(1, bytes_per_row)))


def prod(xs) -> int:
    return int(math.prod(int(v) for v in xs))
