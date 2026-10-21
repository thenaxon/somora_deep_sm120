"""Scale-factor layout transforms (vendor csrc/apis/layout.hpp, SM100 branch semantics).

``transform_sf_into_required_layout`` defines the weight-SF layout every GEMM of this shim consumes:

    FP32 SF [G?, ceil(MN/gran_mn), ceil(K/gran_k)]   (power-of-two values, e.g. restored UE8M0)
      -> broadcast along MN by gran_mn (row r uses block r // gran_mn)
      -> UE8M0 exponent bytes (``bits >> 23``, exactly what DeepGEMM does), padded with 127 to a
         multiple of 4 groups, packed 4-per-int32
      -> int32 [G?, MN, ceil(K/gran_k/4)], MN-major, TMA-aligned stride align(MN, 4)

    INT32 SF already in (1, gran_k) form -> returned unchanged (DeepGEMM only checks strides).

This is bit-for-bit the SM100 DeepGEMM layout, so code that permutes/views weight+SF pairs
(FlashMLA ``permute_q_b_proj`` / ``permute_wv_proj``, DeepInfer ``.view()``s, EPLB ``sf.mT``) behaves
exactly as upstream. The GEMM shims decode it with ``_sf.sf_to_group_scales`` (stride agnostic).
"""

from __future__ import annotations

import torch

from . import _sf


def _grans(recipe, is_sfa):
    recipe = tuple(int(v) for v in recipe)
    if len(recipe) == 3:
        if is_sfa is None:
            raise ValueError("transform_sf_into_required_layout: 3-tuple recipe requires is_sfa")
        return (recipe[0] if is_sfa else recipe[1]), recipe[2]
    if len(recipe) == 2:
        if is_sfa is not None:
            raise ValueError("transform_sf_into_required_layout: 2-tuple recipe must not set is_sfa")
        return recipe[0], recipe[1]
    raise ValueError(f"invalid recipe {recipe}")


def _check(sf: torch.Tensor, mn: int, k: int, gran_mn: int, gran_k: int, num_groups) -> None:
    want_dim = 2 + (num_groups is not None)
    if sf.dim() != want_dim:
        raise ValueError(f"SF dim {sf.dim()} != {want_dim} (num_groups={num_groups})")
    if num_groups is not None and sf.size(-3) != num_groups:
        raise ValueError(f"SF groups {sf.size(-3)} != {num_groups}")
    if sf.size(-2) != _sf.ceil_div(mn, gran_mn):
        raise ValueError(f"SF rows {sf.size(-2)} != ceil({mn}/{gran_mn})")
    per = 1 if sf.dtype == torch.float32 else 4
    if sf.size(-1) != _sf.ceil_div(k, gran_k * per):
        raise ValueError(f"SF cols {sf.size(-1)} != ceil({k}/({gran_k}*{per}))")


def transform_sf_into_required_layout(sf, mn, k, recipe, num_groups=None, is_sfa=None,
                                      disable_ue8m0_cast=False, psum_layout=None):
    del psum_layout
    mn, k = int(mn), int(k)
    gran_mn, gran_k = _grans(recipe, is_sfa)
    if sf.dtype not in (torch.float32, torch.int32):
        raise TypeError(f"transform_sf_into_required_layout: unsupported SF dtype {sf.dtype}")
    _check(sf, mn, k, gran_mn, gran_k, num_groups)
    if sf.dtype == torch.float32:
        if disable_ue8m0_cast:
            # SM90-style FP32 SF: keep FP32, MN-major for per-row SFs, unchanged for block SFs.
            return _sf.get_mn_major_tma_aligned_tensor(sf) if gran_mn == 1 else sf
        if gran_k not in (32, 128):
            raise ValueError(f"unsupported gran_k {gran_k}")
        if gran_mn != 1:
            sf = sf.repeat_interleave(gran_mn, dim=-2)[..., :mn, :]
        return _sf.pack_bytes_mn_major(_sf.float_to_ue8m0_bytes(sf))
    # int32 packed UE8M0
    if gran_mn != 1:
        raise ValueError("packed int32 SF must be per row (gran_mn == 1)")
    return sf


def get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(sf, grouped_layout, ks_cpu, gran_k, k_alignment,
                                                          use_psum_layout=False):
    raise NotImplementedError("K-grouped SF layouts (training-only) are not implemented by the SM120 shim")
