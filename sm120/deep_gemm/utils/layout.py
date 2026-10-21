"""deep_gemm.utils.layout (SM120 shim): TMA alignment / SF layout helpers and the MK alignment knob."""
from .._config import (
    get_mk_alignment_for_contiguous_layout,
    get_theoretical_mk_alignment_for_contiguous_layout,
    set_mk_alignment_for_contiguous_layout,
)
from .._layout import get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor
from .._sf import (
    get_mn_major_tma_aligned_packed_ue8m0_tensor,
    get_mn_major_tma_aligned_tensor,
    get_tma_aligned_size,
)

get_m_alignment_for_contiguous_layout = get_mk_alignment_for_contiguous_layout
get_k_alignment_for_contiguous_layout = get_mk_alignment_for_contiguous_layout

__all__ = [
    "get_tma_aligned_size",
    "get_mn_major_tma_aligned_tensor",
    "get_mn_major_tma_aligned_packed_ue8m0_tensor",
    "get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor",
    "set_mk_alignment_for_contiguous_layout",
    "get_mk_alignment_for_contiguous_layout",
    "get_theoretical_mk_alignment_for_contiguous_layout",
    "get_m_alignment_for_contiguous_layout",
    "get_k_alignment_for_contiguous_layout",
]
