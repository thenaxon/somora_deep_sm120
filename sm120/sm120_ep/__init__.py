"""Option-B multi-GPU expert parallelism for DeepInfer on PCIe-only SM120 machines (see docs/MULTI_GPU.md).

    import sm120_ep
    moe.sm120_routed = sm120_ep.wrap(moe)        # in sm120_moe.setup, after the b12x routed experts exist
"""

from .moe import EPRoutedExperts, ep_impl, wrap  # noqa: F401
from .transport import GlooTransport, NcclTransport, make_transport  # noqa: F401

__all__ = ["EPRoutedExperts", "wrap", "ep_impl", "make_transport", "NcclTransport", "GlooTransport"]
