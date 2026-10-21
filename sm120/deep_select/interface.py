"""SM120 shim of deep_select.interface (same names/signatures as vendor/DeepSelect)."""
from sm120_attn.topk import get_stride_requirement, topk  # noqa: F401

__all__ = ["get_stride_requirement", "topk"]
