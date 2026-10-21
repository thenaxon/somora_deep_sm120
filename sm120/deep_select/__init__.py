"""SM120 shim of DeepSelect (torch.topk based). See sm120_attn/topk.py."""
from .__version__ import __version__  # noqa: F401
from .interface import get_stride_requirement, topk  # noqa: F401
