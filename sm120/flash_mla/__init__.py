"""SM120 shim of FlashMLA (vendor/FlashMLA/flash_mla/__init__.py): same public surface."""
__version__ = "1.0.0+sm120shim"

from flash_mla.flash_mla_interface import (  # noqa: F401
    FlashMLASchedMeta,
    get_mla_metadata,
    flash_mla_with_kvcache,
    flash_attn_varlen_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_varlen_kvpacked_func,
    flash_mla_sparse_fwd,
)

from . import fused_norm_rope_attn_rope_cast  # noqa: F401

__all__ = [
    "get_mla_metadata",
    "flash_mla_with_kvcache",
    "flash_attn_varlen_func",
    "flash_attn_varlen_qkvpacked_func",
    "flash_attn_varlen_kvpacked_func",
    "flash_mla_sparse_fwd",
    "fused_norm_rope_attn_rope_cast",
]
