"""RoPE / RMS helpers matching FlashMLA's fused_norm_rope_attn_rope_cast reference
(DeepInfer/vendor/FlashMLA/tests/test-fused-norm-rope-attn-rope-cast.py:_rope_inplace).

cos_sin_cache: [num_positions, rope_dim] float32, first half cos, second half sin
(DeepInfer layers/rotary_embedding.py:_build_cos_sin_cache). The rotated part is the LAST
rope_dim channels of each head. Interleaved (is_neox_style=False, the V4.1 default) rotates
pairs (2i, 2i+1); NeoX rotates (i, i + rope_dim/2).
"""

from __future__ import annotations

import torch


def apply_rope_(x: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor, rope_dim: int,
                is_neox_style: bool, conjugate: bool) -> torch.Tensor:
    """In-place RoPE on float32 x [s, h, d] (last rope_dim channels). Returns x."""
    assert x.dtype == torch.float32 and x.dim() == 3
    half = rope_dim // 2
    cs = cos_sin_cache.float()[positions.long()]  # [s, rope_dim]
    cos = cs[:, :half].unsqueeze(1)
    sin = cs[:, half:rope_dim].unsqueeze(1)
    if conjugate:
        sin = -sin
    r = x[..., x.shape[-1] - rope_dim:]
    if is_neox_style:
        x0 = r[..., :half].clone()
        x1 = r[..., half:].clone()
        r[..., :half] = x0 * cos - x1 * sin
        r[..., half:] = x0 * sin + x1 * cos
    else:
        x0 = r[..., 0::2].clone()
        x1 = r[..., 1::2].clone()
        r[..., 0::2] = x0 * cos - x1 * sin
        r[..., 1::2] = x0 * sin + x1 * cos
    return x


def rms_scale(q: torch.Tensor, eps: float) -> torch.Tensor:
    """rsqrt(mean(q^2, -1) + eps) in fp32 -> [s, h] (no weight; V4.1 q_b_norm has no weight)."""
    qf = q.float()
    return torch.rsqrt(qf.pow(2).sum(-1) / q.shape[-1] + eps)
