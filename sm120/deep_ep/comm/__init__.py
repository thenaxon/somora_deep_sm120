"""deep_ep.comm (SM120 shim)."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch


def _size(group_or_buffer: Any) -> int:
    for attr in ("num_ranks",):
        if hasattr(group_or_buffer, attr):
            return int(getattr(group_or_buffer, attr))
    try:
        return int(group_or_buffer.size())
    except Exception:  # noqa: BLE001
        return 1


def get_physical_domain_size(group_or_buffer: Any) -> Tuple[int, int]:
    """(RDMA-domain size, NVLink-domain size). SM120 boxes have no NVLink: every GPU is its own
    NVLink domain, so the NVLink domain size is 1 and all peers are "remote" (PCIe/NCCL).

    DeepInfer's ParallelProcessGroup.is_within_nvl_domain(group) is
        group.size() == 1 or group.size() <= get_physical_domain_size(group)[1]
    -> for EP > 1 this returns False (non-mega MoE path). For EP == 1 the size()==1 short-circuit makes it
    True regardless, which is why sm120_patches also patches is_within_nvl_domain.
    """
    return _size(group_or_buffer), 1


def get_logical_domain_size(group_or_buffer: Any, allow_hybrid_mode: Optional[bool] = None) -> Tuple[int, int]:
    return _size(group_or_buffer), 1


def barrier(group=None, *args, **kwargs) -> None:
    if group is not None and _size(group) > 1:
        import torch.distributed as dist

        dist.barrier(group=group)


def get_comm_stream(*args, **kwargs):
    return torch.cuda.current_stream()


class NCCLCommHandle:
    def __init__(self, nccl_comm: int = 0, managed: bool = False) -> None:
        self.nccl_comm = nccl_comm
        self.managed = managed

    def get(self) -> int:
        return self.nccl_comm


def get_nccl_comm_handle(group, force_new_comm: bool = False) -> NCCLCommHandle:
    return NCCLCommHandle(0, False)


def destroy_all_managed_nccl_comm() -> None:
    pass


__all__ = [
    "NCCLCommHandle",
    "barrier",
    "destroy_all_managed_nccl_comm",
    "get_comm_stream",
    "get_logical_domain_size",
    "get_nccl_comm_handle",
    "get_physical_domain_size",
]
