"""The one-kernel E4M3 dequant (deep_gemm/_dequant_triton.py) is bit-identical to _sf.dequant_fp8.

    PYTHONPATH=/opt/sm120/sm120:/opt/deepinfer pytest -q /opt/sm120/tests/test_dequant_triton.py
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sm120"))

from deep_gemm import _dequant_triton, _sf  # noqa: E402

DEV = "cuda"


def _case(m, k, g, packed, mn_major, seed=0):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    x = (torch.randn((m, k), generator=gen, device=DEV) * 3).to(torch.float8_e4m3fn)
    groups = -(-k // g)
    e = torch.randint(110, 140, (m, groups), generator=gen, device=DEV, dtype=torch.int32).to(torch.uint8)
    e[0, 0] = 0  # byte 0: +0.0 scale in both paths
    if packed:
        sf = _sf.pack_bytes_mn_major(e) if mn_major else _sf.pack_bytes_mn_major(e).contiguous()
    else:
        sf = _sf.ue8m0_bytes_to_float(e)
        if mn_major:
            sf = _sf.get_mn_major_tma_aligned_tensor(sf)
    return x, sf


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("m,k", [(1, 7168), (7, 4096), (50, 1536), (513, 7168), (4096, 2048), (3, 1024)])
@pytest.mark.parametrize("g", [128, 32])
@pytest.mark.parametrize("packed", [True, False])
@pytest.mark.parametrize("mn_major", [True, False])
def test_bit_identical(m, k, g, packed, mn_major):
    x, sf = _case(m, k, g, packed, mn_major)
    ref = _sf.dequant_fp8(x, sf, g, torch.bfloat16)
    got = _dequant_triton.dequant_fp8_rows(x, sf, g)
    assert got is not None
    assert torch.equal(got.view(torch.int16), ref.contiguous().view(torch.int16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_declines_other_layouts():
    x, sf = _case(64, 1024, 128, True, False)
    assert _dequant_triton.dequant_fp8_rows(x, sf[:32], 128) is None  # block (MN-grouped) scales
    assert _dequant_triton.dequant_fp8_rows(x.float(), sf, 128) is None
    x2, sf2 = _case(3, 200, 128, True, False)
    assert _dequant_triton.dequant_fp8_rows(x2, sf2, 128) is None  # partial last group
