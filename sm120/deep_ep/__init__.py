"""SM120 drop-in replacement for DeepSeek's DeepEP v2 (vendor/DeepEP, 2.5.0 API surface).

DeepEP's kernels need NVLink symmetric memory / RDMA (NVSHMEM-style GIN), which SM120 PCIe boxes do not
have. The single-GPU path runs (EP=1, CP=1, engram off), so:
  * EPBuffer.dispatch / combine are exact single-rank implementations with DeepEP's tensor layouts
    (see _ep.py) -- used only by DeepInfer's original MoE path (SM120_MOE_IMPL=deepep); the default
    b12x MoE path does not dispatch at all.
  * get_physical_domain_size reports an NVLink domain of 1 GPU (PCIe only), so DeepInfer never picks
    the NVLink-only mega-MoE path for EP > 1.
  * BucketBuffer / PPBuffer raise a clear NotImplementedError when constructed.
  * EngramBuffer (_engram.py) is a process-local implementation: rows come from host RAM / an mmap of
    the HF shard / NVMe / GPU memory via sm120_engram, not from RDMA peers (docs/ENGRAM.md).
"""

from __future__ import annotations

from . import comm  # noqa: F401
from ._ep import EPBuffer, EPHandle, topk_idx_t  # noqa: F401
from ._event import EventHandle, EventOverlap  # noqa: F401
from ._engram import EngramBuffer  # noqa: F401
from ._stubs import BucketBuffer, BucketSession, BufferAllocator, BufferBase, PPBuffer  # noqa: F401
from .comm import destroy_all_managed_nccl_comm, get_logical_domain_size, get_physical_domain_size  # noqa: F401


def get_num_allocation_alignment() -> int:
    return 2 << 20


def get_num_rdma_alignment() -> int:
    return 128


def get_num_tma_alignment() -> int:
    return 16


def init_jit() -> None:
    pass


def check_nccl_so() -> None:
    pass


__version__ = "2.5.0+sm120shim"
