"""deep_gemm.utils (SM120 shim)."""
from . import layout, math  # noqa: F401
from .layout import *  # noqa: F401,F403
from .math import *  # noqa: F401,F403


def init_dist(local_rank: int, num_local_ranks: int):
    raise NotImplementedError("deep_gemm.utils.init_dist is a test helper and not provided by the SM120 shim")


def uneven_all_gather(tensor, dim: int = 0, group=None):
    import torch
    import torch.distributed as dist

    if group is None or dist.get_world_size(group) == 1:
        return tensor
    raise NotImplementedError("deep_gemm.utils.uneven_all_gather (multi-rank) is not provided by the SM120 shim")
