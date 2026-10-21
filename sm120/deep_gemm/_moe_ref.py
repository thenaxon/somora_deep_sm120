"""Pure-torch MoE reference shared by the mega-MoE shims and sm120_moe's ``torch`` backend.

Numerics mirror DeepInfer's non-mega FP8xFP4 path (mlp.py routed_experts_forward):
    h   = X @ W13_e^T                       (X: MXFP8 activations, W13: FP4/FP8/BF16 weights, bf16 out)
    g,u = h[:, :n], h[:, n:]                (W13 = cat(w1 gate, w3 up), checkpoint order)
    act = silu(clamp(g, max=L)) * clamp(u, -L, L) * topk_weight     (swiglu_forward_and_per_token_cast)
    act = MXFP8(act)  if fp8 activations     (1x32 UE8M0, DeepGEMM per_token_cast semantics)
    y   = act @ W2_e^T                      (bf16 out)
    out = sum_k y_k (fp32, slot order)      (DeepEP combine) [+ shared expert bias]
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from . import _sf


def _w(w: torch.Tensor, sf: torch.Tensor | None, e: int) -> torch.Tensor:
    if w.dtype == torch.bfloat16:
        return w[e]
    return _sf.dequant_any(w[e], None if sf is None else sf[e], 32, torch.bfloat16)


def swiglu(h: torch.Tensor, n: int, clamp: float | None) -> torch.Tensor:
    g = h[:, :n].float()
    u = h[:, n:].float()
    if clamp is not None:
        g = g.clamp(max=clamp)
        u = u.clamp(min=-clamp, max=clamp)
    return F.silu(g) * u


def moe_reference(x: torch.Tensor, topk_idx: torch.Tensor, topk_weights: torch.Tensor,
                  w13: torch.Tensor, s13: torch.Tensor | None, w2: torch.Tensor, s2: torch.Tensor | None,
                  clamp: float | None, fp8_act: bool, expert_offset: int = 0) -> torch.Tensor:
    """x: bf16 [T, K] (already dequantised activations); returns fp32 [T, K_out] routed sum.

    topk_idx entries outside [expert_offset, expert_offset + E) (or -1) are skipped.
    """
    T = x.shape[0]
    E = int(w13.shape[0])
    n2 = int(w13.shape[1])
    n = n2 // 2
    k_out = int(w2.shape[1])
    out = torch.zeros((T, k_out), dtype=torch.float32, device=x.device)
    if T == 0:
        return out
    ids = topk_idx.to(torch.int64) - expert_offset
    valid = (ids >= 0) & (ids < E)
    if not bool(valid.any()):
        return out
    experts = torch.unique(ids[valid]).tolist()
    # fp32 accumulation (DeepEP combine also reduces in fp32; only the summation order differs)
    for e in experts:
        tok, slot = ((ids == e) & valid).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        W13 = _w(w13, s13, e)
        W2 = _w(w2, s2, e)
        h = torch.mm(x[tok], W13.t())  # bf16 out, fp32 accumulate
        act = swiglu(h, n, clamp) * topk_weights[tok, slot].float().unsqueeze(1)
        act = _sf.quant_dequant_mxfp8(act, 32, torch.bfloat16) if fp8_act else act.to(torch.bfloat16)
        y = torch.mm(act, W2.t())
        out.index_add_(0, tok, y.float())
    return out


def shared_reference(x: torch.Tensor, w1: torch.Tensor, s1, w2: torch.Tensor, s2, clamp: float | None,
                     fp8_act: bool) -> torch.Tensor:
    """Shared expert (no routing weight) -> bf16 [T, K]."""
    W1 = w1 if w1.dtype == torch.bfloat16 else _sf.dequant_any(w1, s1, 32, torch.bfloat16)
    W2 = w2 if w2.dtype == torch.bfloat16 else _sf.dequant_any(w2, s2, 32, torch.bfloat16)
    h = torch.mm(x, W1.t())
    act = swiglu(h, h.shape[1] // 2, clamp)
    act = _sf.quant_dequant_mxfp8(act, 32, torch.bfloat16) if fp8_act else act.to(torch.bfloat16)
    return torch.mm(act, W2.t())
