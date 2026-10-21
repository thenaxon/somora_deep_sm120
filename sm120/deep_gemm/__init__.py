"""SM120 drop-in replacement for DeepSeek's DeepGEMM (vendor/DeepGEMM, v2.8.2 API surface).

DeepGEMM's kernels are SM90/SM100-only (tcgen05/TMEM, 2-CTA MMA, >200 KB smem). This package keeps
the import path, names and signatures DeepInfer uses and implements them with
  * b12x 1.5.1 where a validated mapping exists (SM120_GEMM_IMPL=b12x, default):
      fp8_fp4_gemm_nt (FP8 x FP8, UE8M0 1x32 / 32x32)  -> b12x.gemm.block_fp8_linear
  * an exact pure-torch reference everywhere else (and for everything with SM120_GEMM_IMPL=torch).
The DeepGEMM indexer kernels (fp8_fp4_*mqa_logits + metadata) are provided by the attention shim
(``sm120_attn.indexer.DEEP_GEMM_EXPORTS``) and re-exported here.

See docs/MOE_GEMM.md for the mapping table, call sites and known gaps, and _env.py for
the runtime switches.
"""

from __future__ import annotations

import types as _types

# ---------------------------------------------------------------- configs
from ._config import (  # noqa: F401
    get_mk_alignment_for_contiguous_layout,
    get_num_sms,
    get_pdl,
    get_tc_util,
    get_theoretical_mk_alignment_for_contiguous_layout,
    init,
    is_deterministic,
    set_block_size_multiple_of,
    set_ignore_compile_dims,
    set_mk_alignment_for_contiguous_layout,
    set_num_sms,
    set_pdl,
    set_tc_util,
    shutdown,
    use_deterministic_algorithms,
)

# ---------------------------------------------------------------- GEMMs
from ._gemm import (  # noqa: F401
    batched_symm,
    batched_syrk,
    bf16_gemm_nn,
    bf16_gemm_nt,
    bf16_gemm_tn,
    bf16_gemm_tt,
    cublaslt_gemm_nn,
    cublaslt_gemm_nt,
    cublaslt_gemm_tn,
    cublaslt_gemm_tt,
    cublaslt_nvfp4_gemm_nt,
    einsum,
    fp4_gemm_nt,
    fp8_einsum,
    fp8_fp4_gemm_nn,
    fp8_fp4_gemm_nt,
    fp8_fp4_gemm_tn,
    fp8_fp4_gemm_tt,
    fp8_gemm_nn,
    fp8_gemm_nt,
    fp8_gemm_tn,
    fp8_gemm_tt,
    k_grouped_bf16_gemm_tn_contiguous,
    k_grouped_fp4_gemm_nt_contiguous,
    k_grouped_fp8_gemm_nt_contiguous,
    k_grouped_fp8_gemm_tn_contiguous,
    m_grouped_bf16_gemm_nn_contiguous,
    m_grouped_bf16_gemm_nt_contiguous,
    m_grouped_bf16_gemm_nt_masked,
    m_grouped_fp4_gemm_nt_contiguous,
    m_grouped_fp4_gemm_nt_masked,
    m_grouped_fp8_fp4_gemm_nn_contiguous,
    m_grouped_fp8_fp4_gemm_nt_contiguous,
    m_grouped_fp8_fp4_gemm_nt_masked,
    m_grouped_fp8_gemm_nn_contiguous,
    m_grouped_fp8_gemm_nt_contiguous,
    m_grouped_fp8_gemm_nt_masked,
    tf32_hc_prenorm_gemm,
)
from ._fused import bf16_mega_gate, mega_mhc  # noqa: F401

# ---------------------------------------------------------------- layout
from ._layout import (  # noqa: F401
    get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor,
    transform_sf_into_required_layout,
)
from ._sf import (  # noqa: F401
    get_mn_major_tma_aligned_packed_ue8m0_tensor,
    get_mn_major_tma_aligned_tensor,
    get_tma_aligned_size,
)

# ---------------------------------------------------------------- mega
from . import mega  # noqa: F401
from .mega import (  # noqa: F401
    SymmBuffer,
    bf16_mega_moe,
    fp8_fp4_mega_moe,
    get_block_m_for_mega_moe,
    get_symm_buffer_for_mega_moe,
    get_token_alignment_for_mega_moe,
    transform_weights_for_mega_moe,
)

# ---------------------------------------------------------------- misc modules
from . import locality_domain, testing, utils  # noqa: F401,E402
from .locality_domain import (  # noqa: F401,E402
    destroy_localizer,
    get_num_locality_domains,
    is_localization_available,
    is_localized,
    localize,
)
from .utils import *  # noqa: F401,F403,E402
from .utils import get_k_alignment_for_contiguous_layout, get_m_alignment_for_contiguous_layout  # noqa: F401,E402


class _FP8CastWithClamp:  # deep_gemm.epilogue.FP8CastWithClamp (only constructed, never run here)
    def __init__(self, sfd=None, clamp_value=None, clamped_count=None):
        self.sfd, self.clamp_value, self.clamped_count = sfd, clamp_value, clamped_count


epilogue = _types.SimpleNamespace(FP8CastWithClamp=_FP8CastWithClamp)

# ---------------------------------------------------------------- indexer 
_INDEXER_NAMES = (
    "get_mqa_logits_metadata",
    "fp8_fp4_mqa_logits",
    "get_paged_mqa_logits_metadata",
    "fp8_fp4_paged_mqa_logits",
    "get_sparse_mqa_logits_metadata",
    "get_paged_sparse_mqa_logits_metadata",
    "fp8_fp4_sparse_mqa_logits",
    "fp8_fp4_paged_sparse_mqa_logits",
)
_INDEXER_ERROR: BaseException | None = None


def _load_indexer() -> dict:
    """Import sm120_attn.indexer.DEEP_GEMM_EXPORTS (dict name->callable, or an iterable of names)."""
    global _INDEXER_ERROR
    try:
        import sm120_attn.indexer as _idx

        exports = _idx.DEEP_GEMM_EXPORTS
    except ImportError as exc:  # attention shim not present yet
        _INDEXER_ERROR = exc
        return {}
    except AttributeError as exc:
        _INDEXER_ERROR = exc
        return {}
    if isinstance(exports, dict):
        found = dict(exports)
    else:
        found = {name: getattr(_idx, name) for name in exports}
    _INDEXER_ERROR = None
    return found


def _install_indexer() -> None:
    found = _load_indexer()
    g = globals()
    for name, fn in found.items():
        g[name] = fn


def _missing_indexer(name: str):
    def fn(*args, **kwargs):
        # retry: sm120_attn may have become importable after deep_gemm was imported
        _install_indexer()
        real = globals().get(name)
        if real is not None and getattr(real, "_sm120_missing", False) is False:
            return real(*args, **kwargs)
        raise RuntimeError(
            f"deep_gemm.{name} is provided by the SM120 attention shim (sm120_attn.indexer.DEEP_GEMM_EXPORTS), "
            f"which could not be imported: {_INDEXER_ERROR!r}"
        )

    fn.__name__ = name
    fn._sm120_missing = True
    return fn


for _n in _INDEXER_NAMES:
    globals()[_n] = _missing_indexer(_n)
_install_indexer()  # last, so a circular `import deep_gemm` inside sm120_attn sees everything else

__version__ = "2.8.2+sm120shim"
