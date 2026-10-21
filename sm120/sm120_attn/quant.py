"""Byte-exact (de)quantization helpers for the DeepSeek V4.1 cache / activation formats.

Formats (all little-endian, values first then scales, see
DeepInfer/vendor/FlashMLA/tests/quant.py and DeepInfer tilelang_ops/cache_common.py):

* V4.1 FP8 main/SWA token, 528 B:   512 x float8_e4m3fn | 16 x ue8m0 (one per 32 values)
* V4.1 FP4 main token, 288 B:       256 B packed e2m1 (even element = low nibble) | 32 x e4m3
                                    (one per 16 values, plain e4m3 scale, no global scale)
* Indexer K (FP4_INDEX_KV_FORMAT):  per *block* [block_size x 64 B e2m1 | block_size x 4 B ue8m0]
                                    (one ue8m0 per 32 values, 4 packed into an int32)
* Indexer K (MXFP8):                per block [block_size x 128 B e4m3 | block_size x 4 B ue8m0]
* Activation per-32 cast:           e4m3 values + ue8m0 scales packed 4-per-int32
                                    (DeepGEMM "use_packed_ue8m0" format)
"""

from __future__ import annotations

import torch

FP8_MAX = 448.0
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
_LUT_CACHE: dict = {}

V41_FP8_BYTES = 528
V41_FP4_BYTES = 288


def _e2m1_lut(device) -> torch.Tensor:
    key = str(device)
    lut = _LUT_CACHE.get(key)
    if lut is None:
        lut = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=device)
        _LUT_CACHE[key] = lut
    return lut


def as_u8(t: torch.Tensor) -> torch.Tensor:
    """Reinterpret a 1-byte tensor (uint8/int8/fp8) as uint8 without copying."""
    if t.dtype == torch.uint8:
        return t
    assert t.element_size() == 1, t.dtype
    return t.view(torch.uint8)


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """[..., n/2] bytes (even element in the low nibble) -> [..., n] float32."""
    p = as_u8(packed)
    lut = _e2m1_lut(p.device)
    lo = (p & 0x0F).long()
    hi = (p >> 4).long()
    codes = torch.stack((lo, hi), dim=-1).flatten(-2)
    return lut[codes]


def ue8m0_to_float(b: torch.Tensor) -> torch.Tensor:
    """ue8m0 exponent bytes -> float32 2^(b-127)."""
    return (as_u8(b).to(torch.int32) << 23).view(torch.float32)


def ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    """Round a positive float32 scale up to a power of two (DeepGEMM ceil_to_ue8m0)."""
    bits = x.abs().float().contiguous().view(torch.int32)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).ne(0).to(torch.int32)
    return (exp.clamp(1, 254) << 23).view(torch.float32)


def pack_ue8m0(sf: torch.Tensor) -> torch.Tensor:
    """float32 power-of-two scales [..., k] (k % 4 == 0) -> int32 [..., k/4] (4 exponent bytes each)."""
    assert sf.shape[-1] % 4 == 0, sf.shape
    e = ((sf.float().contiguous().view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
    return e.contiguous().view(torch.int32)


def unpack_ue8m0_int32(sf: torch.Tensor) -> torch.Tensor:
    """int32 [...] or [..., k] packed ue8m0 -> float32 [..., 4] / [..., 4k]."""
    s = sf.contiguous()
    if s.dim() == 0:
        s = s.view(1)
    return ue8m0_to_float(s.unsqueeze(-1).view(torch.uint8).flatten(-2))


# ---------------------------------------------------------------------------------------------
# V4.1 main / SWA KV token records
# ---------------------------------------------------------------------------------------------
def dequant_v41_fp8_records(rec: torch.Tensor) -> torch.Tensor:
    """[..., 528] uint8 -> [..., 512] float32 (exact)."""
    rec = as_u8(rec)
    lead = rec.shape[:-1]
    r = rec.reshape(-1, V41_FP8_BYTES)
    vals = r[:, :512].contiguous().view(torch.float8_e4m3fn).float().view(-1, 16, 32)
    sf = ue8m0_to_float(r[:, 512:528]).view(-1, 16, 1)
    return (vals * sf).view(*lead, 512)


def dequant_v41_fp4_records(rec: torch.Tensor) -> torch.Tensor:
    """[..., 288] uint8 -> [..., 512] float32 (exact: e2m1 x e4m3 fits in bf16/fp32)."""
    rec = as_u8(rec)
    lead = rec.shape[:-1]
    r = rec.reshape(-1, V41_FP4_BYTES)
    vals = unpack_e2m1(r[:, :256]).view(-1, 32, 16)
    sf = r[:, 256:288].contiguous().view(torch.float8_e4m3fn).float().view(-1, 32, 1)
    return (vals * sf).view(*lead, 512)


def dequant_v41_records(rec: torch.Tensor) -> torch.Tensor:
    nbytes = rec.shape[-1]
    if nbytes == V41_FP8_BYTES:
        return dequant_v41_fp8_records(rec)
    if nbytes == V41_FP4_BYTES:
        return dequant_v41_fp4_records(rec)
    raise ValueError(f"unsupported V4.1 KV record size {nbytes} B/token (expected 528 or 288)")


def gather_paged_records(k_cache: torch.Tensor, flat_idx: torch.Tensor) -> torch.Tensor:
    """Gather token records from a paged cache [num_blocks, block_size, 1, B] (any block stride).

    flat_idx (any shape, must be >= 0 and in range) = block * block_size + offset.
    Returns uint8 [*flat_idx.shape, B].
    """
    assert k_cache.dim() == 4 and k_cache.shape[2] == 1, k_cache.shape
    kc = as_u8(k_cache)
    bs = kc.shape[1]
    idx = flat_idx.long()
    return kc[idx // bs, idx % bs, 0]


def quantize_v41_fp8_records(x: torch.Tensor) -> torch.Tensor:
    """Reference encoder [..., 512] -> [..., 528] uint8 (FlashMLA tests/quant.py semantics)."""
    lead = x.shape[:-1]
    xf = x.float().reshape(-1, 16, 32)
    sf_inv = xf.abs().amax(-1) / FP8_MAX
    sf = torch.pow(2.0, torch.clamp_min(sf_inv, 1e-4).log2().ceil())
    q = (xf / sf.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    out = torch.empty((xf.shape[0], V41_FP8_BYTES), dtype=torch.uint8, device=x.device)
    out[:, :512] = q.view(-1, 512).view(torch.uint8)
    out[:, 512:] = ((sf.contiguous().view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
    return out.view(*lead, V41_FP8_BYTES)


def _quantize_e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even e2m1 codes (uint8, sign in bit 3), saturating at +-6."""
    mags = torch.tensor(_E2M1_VALUES[:8], dtype=torch.float32, device=x.device)
    sign = torch.signbit(x).to(torch.uint8) << 3
    a = torch.nan_to_num(x.abs(), nan=0.0, posinf=6.0).clamp_max(6.0)
    mids = (mags[:-1] + mags[1:]) / 2
    code = torch.bucketize(a, mids, right=True)
    on_tie = (a.unsqueeze(-1) == mids).any(dim=-1)
    tie_code = torch.bucketize(a, mids, right=False)
    code = torch.where(on_tie, tie_code + (tie_code & 1), code)
    return sign | code.to(torch.uint8)


def pack_e2m1(x: torch.Tensor) -> torch.Tensor:
    """[..., n] float (already scaled) -> [..., n/2] uint8 (even element in the low nibble)."""
    c = _quantize_e2m1_codes(x.float())
    return c[..., 0::2] | (c[..., 1::2] << 4)


def quantize_v41_fp4_records(x: torch.Tensor) -> torch.Tensor:
    """Reference encoder [..., 512] -> [..., 288] uint8 (FlashMLA tests/quant.py semantics)."""
    lead = x.shape[:-1]
    xf = x.float().reshape(-1, 32, 16)
    amax = xf.abs().amax(-1)
    sf = torch.clamp(amax / 6.0, 2.0 ** -9, 448.0).to(torch.float8_e4m3fn)
    norm = xf / sf.float().unsqueeze(-1)
    out = torch.empty((xf.shape[0], V41_FP4_BYTES), dtype=torch.uint8, device=x.device)
    out[:, :256] = pack_e2m1(norm.view(-1, 512))
    out[:, 256:] = sf.view(torch.uint8)
    return out.view(*lead, V41_FP4_BYTES)


# ---------------------------------------------------------------------------------------------
# Activation casts (per-token, per-32 ue8m0 FP8; DeepGEMM / tile_kernels packed format)
# ---------------------------------------------------------------------------------------------
def per_token_cast_fp8_ue8m0(x: torch.Tensor, gran: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """x [..., n] -> (fp8 [..., n], float32 power-of-two scales [..., n/gran]).

    sf = ceil_ue8m0(max(amax, 1e-4) / 448)  (DeepGEMM per_token_cast_to_fp8(use_ue8m0=True)).
    """
    n = x.shape[-1]
    assert n % gran == 0
    xv = x.float().reshape(*x.shape[:-1], n // gran, gran)
    amax = xv.abs().amax(-1).clamp_min(1e-4)
    sf = ceil_to_ue8m0(amax / FP8_MAX)
    q = (xv / sf.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.reshape(x.shape), sf


def alloc_tma_colmajor_sf(s: int, n_groups: int, k_packs: int, device) -> torch.Tensor:
    """int32 [s, n_groups, k_packs] with stride(0) == 1 (DeepGEMM TMA-aligned MN-major SF).

    Mirrors FlashMLA csrc/api/fused_norm_rope_attn_rope_cast_fwd.cpp allocate_scale_factor(..)
    followed by .transpose(0, 1).
    """
    aligned_s = (s + 3) // 4 * 4
    base = torch.empty((n_groups, k_packs, aligned_s), dtype=torch.int32, device=device)
    return base.transpose(1, 2)[:, :s].transpose(0, 1)


# ---------------------------------------------------------------------------------------------
# Indexer (MQA logits) operands
# ---------------------------------------------------------------------------------------------
def dequant_index_values(fp: torch.Tensor, sf: torch.Tensor | None, head_dim: int | None = None) -> torch.Tensor:
    """Dequantize indexer Q/K values.

    fp: int8 (packed e2m1, [..., D/2]) or float8_e4m3fn ([..., D]).
    sf: int32 packed ue8m0 ([...] or [..., k]; gran 32, padded to multiples of 4 scales),
        float32 per-token scale ([...], SM90 style), or None (unscaled fp8).
    Returns float32 [..., D].
    """
    if fp.dtype == torch.int8 or fp.dtype == torch.uint8:
        vals = unpack_e2m1(fp)
    elif fp.dtype == torch.float8_e4m3fn:
        vals = fp.float()
    else:
        raise TypeError(f"unsupported indexer operand dtype {fp.dtype}")
    d = vals.shape[-1] if head_dim is None else head_dim
    if sf is None:
        return vals
    if sf.dtype == torch.int32:
        scales = unpack_ue8m0_int32(sf)
        if scales.shape[:-1] != vals.shape[:-1]:
            scales = scales.reshape(*vals.shape[:-1], -1)
        nsf = d // 32
        scales = scales[..., :nsf]
        return (vals.view(*vals.shape[:-1], nsf, 32) * scales.unsqueeze(-1)).view(vals.shape)
    if sf.dtype == torch.float32:
        return vals * sf.reshape(vals.shape[:-1]).unsqueeze(-1)
    raise TypeError(f"unsupported indexer scale dtype {sf.dtype}")


def split_index_cache(kv_cache: torch.Tensor, value_bytes: int, sf_bytes: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-wide [values | scales] index cache [nb, bs, 1, value_bytes+sf_bytes] -> views.

    Returns (values uint8 [nb, bs, value_bytes], scales uint8 [nb, bs, sf_bytes]); works for any
    block stride (the cache may be a region inside a multi-layer block buffer).
    """
    nb, bs, nh, bpt = kv_cache.shape
    assert nh == 1 and bpt == value_bytes + sf_bytes, (kv_cache.shape, value_bytes, sf_bytes)
    kc = as_u8(kv_cache)
    raw = kc.as_strided((nb, bs * bpt), (kc.stride(0), 1), kc.storage_offset())
    vals = raw[:, : bs * value_bytes].view(nb, bs, value_bytes)
    sfs = raw[:, bs * value_bytes: bs * bpt].view(nb, bs, sf_bytes)
    return vals, sfs
