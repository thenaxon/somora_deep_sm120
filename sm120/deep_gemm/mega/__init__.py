"""Mega-MoE shim (single rank, pure torch). DeepInfer is steered to its NON-mega MoE path on SM120
(sm120_patches: ParallelProcessGroup.is_within_nvl_domain -> False), so this is a correctness
fallback only.

Layout of this shim (it owns the layout, consistently between producer and consumer):
  * transform_weights_for_mega_moe is the identity (no gate/up interleave, no UTCCP SF transpose):
    weights stay in DeepInfer's grouped layout [E, 2n, K/2] (FP4) / [E, 2n, K] + per-row packed SF.
  * SymmBuffer exposes plain tensors with the views DeepInfer touches:
      x            [num_max_tokens, hidden]  e4m3 (fp8xfp4/fp8xfp8) or bf16 (bf16xbf16)
      x_sf         [num_max_tokens, hidden/128] int32, row-major (DeepInfer casts into it with
                   use_tma_aligned_col_major_sf=False, "mega_moe" extra layout)
      topk_idx     [num_max_tokens, topk] int64, topk_weights [num_max_tokens, topk] fp32
      shared_l1_acts_sf  extra-SF storage (written by the fused norm cast, unused by the reference)
"""

from __future__ import annotations

import types
from typing import Optional

import torch

from .. import _sf
from .._moe_ref import moe_reference, shared_reference

_TOKEN_ALIGNMENT = 128  # LCM of the candidate BLOCK_M values


def get_token_alignment_for_mega_moe() -> int:
    return _TOKEN_ALIGNMENT


def get_block_m_for_mega_moe(num_ranks, num_experts, num_max_tokens_per_rank, num_tokens, num_topk, mma_type) -> int:
    del num_ranks, num_experts, num_max_tokens_per_rank, num_tokens, num_topk, mma_type
    return 128


class SymmBuffer:
    def __init__(self, group, num_experts: int, num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int, num_shared_experts: int = 0,
                 mma_type: str = "fp8xfp4", activation: str = "swiglu", base: Optional["SymmBuffer"] = None):
        assert activation == "swiglu", activation
        if group is not None and group.size() != 1:
            raise NotImplementedError(
                "SM120 deep_gemm.SymmBuffer supports a single rank only (no NVLink symmetric memory on "
                "SM120/PCIe); DeepInfer should take the non-mega MoE path (see patches/PATCHES.md)")
        num_max_tokens_per_rank = _sf.align(int(num_max_tokens_per_rank), _TOKEN_ALIGNMENT)
        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.num_shared_experts = num_shared_experts
        self.mma_type = mma_type
        self.activation = activation
        dev = torch.device("cuda", torch.cuda.current_device())
        T = num_max_tokens_per_rank
        fp8 = mma_type.split("x")[0] == "fp8"
        if base is not None and base.x is not None and base.x.shape[0] >= T and base.mma_type == mma_type \
                and base.hidden == hidden and base.num_topk == num_topk:
            src = base
            self.x, self.x_sf = src.x, src.x_sf
            self.topk_idx, self.topk_weights = src.topk_idx, src.topk_weights
            self.shared_l1_acts_sf = src.shared_l1_acts_sf
        else:
            self.x = torch.zeros((T, hidden), dtype=torch.float8_e4m3fn if fp8 else torch.bfloat16, device=dev)
            self.x_sf = torch.zeros((T, _sf.ceil_div(hidden, 128)), dtype=torch.int32, device=dev)
            self.topk_idx = torch.full((T, num_topk), -1, dtype=torch.int64, device=dev)
            self.topk_weights = torch.zeros((T, num_topk), dtype=torch.float32, device=dev)
            rows = _sf.ceil_div(T, 128) * 128
            self.shared_l1_acts_sf = torch.empty_strided(
                (rows, _sf.ceil_div(hidden, 128)), (1, rows), dtype=torch.int32, device=dev)
        self.shared_l1_acts = self.shared_l2_acts = self.shared_l2_acts_sf = None
        self.l1_acts = self.l1_acts_sf = self.l2_acts = self.l2_acts_sf = None
        self.buffer = self.x  # DeepInfer only logs buffer.nbytes
        self.handle = types.SimpleNamespace(buffer_ptrs=[self.x.data_ptr()])

    def destroy(self) -> None:
        self.handle = None
        self.buffer = None
        self.group = None
        self.x = None
        self.x_sf = None


def get_symm_buffer_for_mega_moe(group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden,
                                 num_shared_experts=0, use_fp8_dispatch=None, mma_type="fp8xfp4",
                                 activation="swiglu") -> SymmBuffer:
    if use_fp8_dispatch is not None:
        assert use_fp8_dispatch == (mma_type.split("x")[0] == "fp8")
    return SymmBuffer(group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden,
                      num_shared_experts, mma_type=mma_type, activation=activation)


def transform_weights_for_mega_moe(l1_weights, l2_weights, activation: str = "swiglu"):
    """Identity: this shim's mega reference consumes DeepInfer's grouped layout directly."""
    assert activation == "swiglu", activation
    return l1_weights, l2_weights


def _split(w):
    return (w[0], w[1]) if isinstance(w, (tuple, list)) else (w, None)


def _mega_ref(y, l1_weights, l2_weights, sym_buffer, shared_l1_weights, shared_l2_weights,
              cumulative_local_expert_recv_stats, activation_clamp, fp8):
    T = y.shape[0]
    if T == 0:
        return
    x = sym_buffer.x[:T]
    if fp8:
        x = _sf.dequant_fp8(x, sym_buffer.x_sf[:T], 32, torch.bfloat16)
    idx = sym_buffer.topk_idx[:T]
    wts = sym_buffer.topk_weights[:T]
    w13, s13 = _split(l1_weights)
    w2, s2 = _split(l2_weights)
    out = moe_reference(x, idx, wts, w13, s13, w2, s2, activation_clamp, fp8)
    if cumulative_local_expert_recv_stats is not None:
        E = int(w13.shape[0])
        ids = idx[idx >= 0].to(torch.int64)
        cumulative_local_expert_recv_stats += torch.bincount(ids, minlength=E)[:E].to(
            cumulative_local_expert_recv_stats.dtype)
    if shared_l1_weights is not None:
        sw1, ss1 = _split(shared_l1_weights)
        sw2, ss2 = _split(shared_l2_weights)
        out += shared_reference(x, sw1, ss1, sw2, ss2, activation_clamp, fp8).float()
    y.copy_(out.to(y.dtype))


def fp8_fp4_mega_moe(y, l1_weights, l2_weights, sym_buffer, shared_l1_weights=None, shared_l2_weights=None,
                     cumulative_local_expert_recv_stats=None, recipe=(1, 1, 32), activation="swiglu",
                     activation_clamp=None, fast_math=True):
    del recipe, fast_math
    assert activation == "swiglu", activation
    _mega_ref(y, l1_weights, l2_weights, sym_buffer, shared_l1_weights, shared_l2_weights,
              cumulative_local_expert_recv_stats, activation_clamp, fp8=True)


def bf16_mega_moe(y, l1_weights, l2_weights, sym_buffer, shared_l1_weights=None, shared_l2_weights=None,
                  cumulative_local_expert_recv_stats=None, activation="swiglu", activation_clamp=None,
                  fast_math=True):
    del fast_math
    assert activation == "swiglu", activation
    _mega_ref(y, l1_weights, l2_weights, sym_buffer, shared_l1_weights, shared_l2_weights,
              cumulative_local_expert_recv_stats, activation_clamp, fp8=False)


__all__ = [
    "SymmBuffer",
    "get_symm_buffer_for_mega_moe",
    "transform_weights_for_mega_moe",
    "fp8_fp4_mega_moe",
    "bf16_mega_moe",
    "get_block_m_for_mega_moe",
    "get_token_alignment_for_mega_moe",
]
