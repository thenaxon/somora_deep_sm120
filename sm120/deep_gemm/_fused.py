"""Torch references of DeepGEMM's fused "mega" helpers that DeepInfer can call outside mega-MoE:

* mega_mhc       : previous sub-layer mHC post + this sub-layer mHC pre (mixes, sinkhorn, apply)
                   + RMSNorm (+ MXFP8 cast). Port of vendor/DeepGEMM/tests/test_mega_mhc.py
                   ``mhc_reference`` (the bit-exactness oracle of the SM100 kernel).
* bf16_mega_gate : bf16 gate GEMM (fp32 logits) fused with the noaux_tc top-k gate; reference =
                   fp32 logits + tile_kernels.moe.moe_topk_gate with the identical kwargs (DeepInfer
                   calls exactly that pair when use_mega_gate is False, mlp.py:400).

On SM120 sm120_patches disables both fusions (use_mega_mhc / use_mega_gate = False) so DeepInfer
runs its own unfused tile_kernels paths; these are fallbacks if something calls them anyway.
"""

from __future__ import annotations

import torch

from . import _sf
from ._gemm import _no_tf32, mm_fp32


def _sinkhorn_reference(value: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    value = value.softmax(dim=-1) + eps
    value = value / (value.sum(dim=-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        value = value / (value.sum(dim=-1, keepdim=True) + eps)
        value = value / (value.sum(dim=-2, keepdim=True) + eps)
    return value


def _extra_sf_rows(num_tokens: int, block_m: int, device) -> torch.Tensor:
    token_idx = torch.arange(num_tokens, device=device)
    index_in_block = token_idx % block_m
    return (token_idx // block_m * _sf.align(block_m, 128) + index_in_block // 128 * 128
            + index_in_block % 32 * 4 + index_in_block % 128 // 32)


@torch.no_grad()
def mega_mhc(x, residual, shifted_prev_mix, post_mix, comb_res_mix, fn, mix_scales, mix_bases, hc_mult,
             hc_norm_eps, hc_pre_eps, hc_post_scale, sinkhorn_eps, num_sinkhorn_iters, rmsnorm_weight,
             rmsnorm_eps, rmsnorm_scale, new_residual, new_prev_mix, new_post_mix, new_comb_res_mix,
             y_bf16=None, y_fp8=None, y_gemm_sf=None, y_routed_sf=None, y_shared_sf=None, shared_sf_block_m=0):
    T, hidden = x.shape
    if T == 0:
        return
    nr = x.float().unsqueeze(1) * post_mix
    nr += torch.einsum("tij,tih->tjh", comb_res_mix.float(), residual.float())
    nr = nr.bfloat16()
    norm_input = None if shifted_prev_mix is None else (nr.float() * shifted_prev_mix).sum(dim=1).bfloat16()
    with _no_tf32():
        mixes = nr.flatten(1).float() @ fn.float().mT
    sq = nr.float().square().sum((1, 2))
    mixes = mixes * torch.rsqrt(sq / (hc_mult * hidden) + hc_norm_eps).unsqueeze(1)
    scales = torch.cat((mix_scales[0].expand(hc_mult), mix_scales[1].expand(hc_mult),
                        mix_scales[2].expand(hc_mult * hc_mult)))
    mixes = mixes * scales + mix_bases
    prev_mix = mixes[:, :hc_mult].sigmoid().unsqueeze(2) + hc_pre_eps
    p_mix = (mixes[:, hc_mult:2 * hc_mult].sigmoid() * hc_post_scale).unsqueeze(2)
    c_mix = _sinkhorn_reference(mixes[:, 2 * hc_mult:].view(-1, hc_mult, hc_mult), num_sinkhorn_iters, sinkhorn_eps)
    if norm_input is None:
        norm_input = (nr.float() * prev_mix).sum(dim=1).bfloat16()
    nf = norm_input.float()
    scale = 1.0 if rmsnorm_scale is None else float(rmsnorm_scale)
    yb = (nf * torch.rsqrt(nf.square().mean(1) + rmsnorm_eps).unsqueeze(1)
          * (rmsnorm_weight.float() * scale)).bfloat16()

    new_residual.copy_(nr)
    new_post_mix.copy_(p_mix)
    new_comb_res_mix.copy_(c_mix)
    if new_prev_mix is not None:
        new_prev_mix.copy_(prev_mix)
    if y_bf16 is not None:
        y_bf16.copy_(yb)
    if y_fp8 is None:
        return
    q, sf = _sf.quant_fp8_rows(yb, 32)
    y_fp8.copy_(q)
    packed = _sf.float_to_ue8m0_bytes(sf)
    g4 = _sf.align(packed.shape[-1], 4)
    if g4 != packed.shape[-1]:
        packed = torch.cat([packed, torch.full((T, g4 - packed.shape[-1]), 127, dtype=torch.uint8,
                                               device=packed.device)], dim=-1)
    packed = packed.contiguous().view(torch.int32)  # [T, hidden/128]
    if y_gemm_sf is not None:
        y_gemm_sf.copy_(packed)
    if y_routed_sf is not None:
        y_routed_sf.copy_(packed)
    if y_shared_sf is not None:
        bm = int(shared_sf_block_m)
        rows = _sf.ceil_div(T, bm) * _sf.align(bm, 128)
        storage = torch.as_strided(y_shared_sf, (rows, y_shared_sf.shape[1]), y_shared_sf.stride())
        storage[_extra_sf_rows(T, bm, packed.device)] = packed


def bf16_mega_gate(x, weight, num_topk, use_shared_as_routed, num_shared_experts, routed_scaling_factor, ep_rank,
                   scoring_func="sqrtsoftplus", mask=None, bias=None, image_bias=None, image_token_mask=None,
                   fix_routing_mask=None, to_physical_map=None, logical_count=None, unmapped_topk_idx=None,
                   force_random=None, out=None):
    from tile_kernels.moe import moe_topk_gate

    logits = mm_fp32(x if x.dtype == torch.bfloat16 else x.bfloat16(), weight.bfloat16())
    kwargs = dict(
        num_topk=num_topk, use_shared_as_routed=use_shared_as_routed, num_shared_experts=num_shared_experts,
        routed_scaling_factor=routed_scaling_factor, ep_rank=ep_rank, mask=mask, bias=bias, image_bias=image_bias,
        image_token_mask=image_token_mask, force_random=force_random, to_physical_map=to_physical_map,
        logical_count=logical_count, scoring_func=scoring_func, out=out,
    )
    if fix_routing_mask is not None:
        kwargs["fix_routing_mask"] = fix_routing_mask
    if unmapped_topk_idx is not None:
        kwargs["unmapped_topk_idx"] = unmapped_topk_idx
    return moe_topk_gate(logits, **kwargs)
