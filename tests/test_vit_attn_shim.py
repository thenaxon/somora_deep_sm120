"""ViT attention shim (sm120/flash_attn/cute/interface.py, used by DeepInfer's vision encoder via flash_attn).

    PYTHONPATH=/opt/sm120/sm120:/opt/deepinfer pytest -q -s /opt/sm120/tests/test_vit_attn_shim.py

The shim runs F.scaled_dot_product_attention per cu_seqlens segment. With 3-D [H, T, D] inputs torch falls back to
the math kernel and materialises the fp32 [H, T, T] scores (about 10 GiB for one full-size V4.1 image: 1024 image
tokens = 93x93 ViT patches with downsample 3 -> T = 8649..8748, H=16, D=64); the shim now passes [1, H, T, D] so a
fused kernel runs. Checks: output == the old 3-D path (bf16 rounding), close to an fp32 reference, peak extra memory
small, for one full-size image and for variable-length batches of several images (cu_seqlens, list or tensor).
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
SHIMS = os.path.join(os.path.dirname(HERE), "sm120")
if SHIMS not in sys.path:
    sys.path.insert(0, SHIMS)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
H, D = 16, 64


def _old_3d(q, k, v, bounds, scale):
    """The previous implementation: SDPA on [H, T, D] per segment."""
    out = torch.empty_like(q)
    for s, e in zip(bounds[:-1], bounds[1:]):
        o = F.scaled_dot_product_attention(q[s:e].transpose(0, 1), k[s:e].transpose(0, 1), v[s:e].transpose(0, 1),
                                           scale=scale)
        out[s:e] = o.transpose(0, 1)
    return out


def _ref_fp32(q, k, v, bounds, scale):
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    for s, e in zip(bounds[:-1], bounds[1:]):
        qs, ks, vs = (t[s:e].float().transpose(0, 1) for t in (q, k, v))
        for h in range(H):  # one head at a time keeps the reference itself small
            out[s:e, h] = torch.softmax(qs[h] @ ks[h].T * scale, -1) @ vs[h]
    return out


def _peak_extra_mib(fn) -> tuple[torch.Tensor, int]:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = fn()
    torch.cuda.synchronize()
    return out, (torch.cuda.max_memory_allocated() - base) >> 20


@pytest.mark.parametrize("seqlens,as_tensor", [
    ([8748], False),                 # one full-size image (31 x (31 + 1) llm tokens -> 93 x 94 patches)
    ([8748], True),
    ([8649, 1024, 2916], True),      # several images of different sizes in one ViT batch
    ([3025, 8748], False),
])
def test_vit_attn_shim(seqlens, as_tensor):
    from flash_attn.cute.interface import _flash_attn_fwd

    torch.manual_seed(0)
    T = sum(seqlens)
    q, k, v = (torch.randn(T, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    bounds = [0]
    for n in seqlens:
        bounds.append(bounds[-1] + n)
    cu = torch.tensor(bounds, dtype=torch.int32, device="cuda") if as_tensor else bounds
    scale = 1.0 / math.sqrt(D)

    new, new_mib = _peak_extra_mib(lambda: _flash_attn_fwd(q, k, v, cu_seqlens_q=cu)[0])
    old, old_mib = _peak_extra_mib(lambda: _old_3d(q, k, v, bounds, scale))
    ref = _ref_fp32(q, k, v, bounds, scale)
    d_old = (new.float() - old.float()).abs().max().item()
    d_ref = (new.float() - ref).abs().max().item()
    d_old_ref = (old.float() - ref).abs().max().item()
    print(f"seqlens={seqlens}: peak extra new {new_mib} MiB, old 3-D {old_mib} MiB; max|new-old|={d_old:.2e} "
          f"max|new-ref|={d_ref:.2e} (old vs ref {d_old_ref:.2e})")
    assert new.shape == q.shape and new.dtype == q.dtype
    assert torch.isfinite(new.float()).all()
    assert d_ref <= max(2 * d_old_ref, 4e-3), "new path further from the fp32 reference than the old one"
    assert d_old <= 8e-3, "new path differs from the old 3-D path beyond bf16 rounding"
    assert new_mib < 256, f"peak extra {new_mib} MiB: SDPA did not take a fused kernel"


def test_vit_attn_shim_no_cu_seqlens():
    from flash_attn.cute.interface import _flash_attn_fwd

    q, k, v = (torch.randn(1024, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    new = _flash_attn_fwd(q, k, v)[0]
    old = _old_3d(q, k, v, [0, 1024], 1.0 / math.sqrt(D))
    assert (new.float() - old.float()).abs().max().item() <= 8e-3
