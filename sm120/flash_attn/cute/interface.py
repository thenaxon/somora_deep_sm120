"""Torch SDPA replacement for flash_attn.cute.interface._flash_attn_fwd (ViT varlen, non-causal).

Mirrors deepinfer/npu_ops/vit_attn.py: q/k/v are THD, cu_seqlens_q marks segment ends; full
attention within each segment. Only the first return value is used by vit.py.
"""

import math
from typing import Any

import torch
import torch.nn.functional as F


def _flash_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor | list[int] | None = None,
    softmax_scale: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q.shape[-1])
    if cu_seqlens_q is None:
        bounds = [0, q.shape[0]]
    else:
        bounds = [int(x) for x in (cu_seqlens_q.tolist() if torch.is_tensor(cu_seqlens_q) else cu_seqlens_q)]
        if bounds[0] != 0:
            bounds = [0] + bounds
    out = torch.empty_like(q)
    for s, e in zip(bounds[:-1], bounds[1:]):
        if e <= s:
            continue
        # [T, H, D] -> [1, H, T, D]: SDPA's flash/efficient kernels need a batch dim; with 3-D inputs torch falls back
        # to the math kernel, which materialises the fp32 [H, T, T] scores (~10 GiB for a full-size image, T=8748)
        o = F.scaled_dot_product_attention(
            q[s:e].transpose(0, 1)[None], k[s:e].transpose(0, 1)[None], v[s:e].transpose(0, 1)[None], scale=scale
        )
        out[s:e] = o[0].transpose(0, 1)
    return out, None
