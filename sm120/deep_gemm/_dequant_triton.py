"""One-kernel E4M3 + per-row group scale -> bf16 dequantisation for the b12x block_fp8_linear input.

`_sf.dequant_fp8` does it with torch ops: the e4m3 tensor is widened to fp32, the scales are expanded, multiplied and
cast back (about 6 kernels, an fp32 temporary of M x K x 4 bytes, and about 0.1 ms of Python per call; 200 calls
per prefill step on the full-depth model). This kernel reads the e4m3 bytes and the scale bytes once and writes
bf16. It is bit-identical: e4m3 * 2^e is exact in fp32 and in bf16 (4 significant bits), the scale is the same
2^(byte - 127) (byte 0 gives +0.0 in both paths).

Covers the layout DeepInfer passes: x [M, K] e4m3 with unit stride along K, SF either int32 packed UE8M0
[M, K/g/4] or float32 [M, K/g] (any strides), one scale row per x row, g in {32, 128}, K a multiple of g. Returns None
otherwise (caller uses `_sf.dequant_fp8`). SM120_DEQUANT_IMPL=torch disables it.
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001
    triton = None

if triton is not None:

    @triton.jit
    def _dequant_kernel(x_ptr, xs0, sf_ptr, ss0, ss1, out_ptr, os0, K, G: tl.constexpr, PACKED: tl.constexpr,
                        BLOCK: tl.constexpr):
        m = tl.program_id(0).to(tl.int64)
        k = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = k < K
        x = tl.load(x_ptr + m * xs0 + k, mask=mask, other=0.0).to(tl.float32)
        grp = k // G
        if PACKED:
            word = tl.load(sf_ptr + m * ss0 + (grp // 4) * ss1, mask=mask, other=0)
            byte = (word >> ((grp % 4) * 8)) & 0xFF
            scale = (byte << 23).to(tl.float32, bitcast=True)
        else:
            scale = tl.load(sf_ptr + m * ss0 + grp * ss1, mask=mask, other=1.0)
        tl.store(out_ptr + m * os0 + k, (x * scale).to(tl.bfloat16), mask=mask)


def dequant_fp8_rows(x: torch.Tensor, sf: torch.Tensor, gran_k: int | None) -> torch.Tensor | None:
    if triton is None or os.environ.get("SM120_DEQUANT_IMPL", "triton") == "torch":
        return None
    if x.dim() != 2 or x.dtype != torch.float8_e4m3fn or not x.is_cuda or x.stride(1) != 1 or sf.dim() != 2:
        return None
    m, k = map(int, x.shape)
    if int(sf.shape[0]) != m or m == 0:
        return None
    if sf.dtype == torch.int32:
        packed = True
        g = gran_k or next((c for c in (128, 32) if -(-(-(-k // c)) // 4) == int(sf.shape[1])), None)
    elif sf.dtype == torch.float32:
        packed = False
        g = gran_k or next((c for c in (128, 32) if -(-k // c) == int(sf.shape[1])), None)
    else:
        return None
    if g not in (32, 128) or k % g:  # partial last group: keep the torch path (DeepInfer's K are multiples of 128)
        return None
    groups = -(-k // g)
    if (packed and int(sf.shape[1]) * 4 < groups) or (not packed and int(sf.shape[1]) < groups):
        return None
    out = torch.empty((m, k), dtype=torch.bfloat16, device=x.device)
    BLOCK = 1024
    _dequant_kernel[(m, triton.cdiv(k, BLOCK))](x, x.stride(0), sf, sf.stride(0), sf.stride(1), out, out.stride(0), k,
                                                G=g, PACKED=packed, BLOCK=BLOCK, num_warps=4)
    return out
