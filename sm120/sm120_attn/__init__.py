"""SM120 (RTX 5090 / RTX PRO 6000) replacements for DeepInfer's SM100-only attention stack.

Backs the drop-in packages ``flash_mla`` and ``deep_select`` in this shim tree and provides the
DeepGEMM indexer functions (``indexer.DEEP_GEMM_EXPORTS``) re-exported by the ``deep_gemm`` shim.
See docs/ATTENTION.md and _config.py for the switches (SM120_ATTN_IMPL=b12x|torch, ...).
"""

from . import _config as config  # noqa: F401
from .indexer import DEEP_GEMM_EXPORTS  # noqa: F401
from .sparse_attn import sparse_decode, sparse_decode_torch, sparse_prefill, sparse_prefill_torch  # noqa: F401
from .topk import topk  # noqa: F401

__all__ = ["config", "DEEP_GEMM_EXPORTS", "sparse_decode", "sparse_decode_torch", "sparse_prefill", "sparse_prefill_torch",
           "topk"]
