"""Collectives for the Option-B expert-parallel MoE (all-gather tokens, reduce-scatter outputs).

    SM120_EP_TRANSPORT = nccl (default) | gloo
      nccl  torch.distributed all_gather_into_tensor / reduce_scatter_tensor on DeepInfer's EP process group
            (NCCL over PCIe on 4x RTX PRO 6000; CUDA-graph capturable). Needs one GPU per rank.
      gloo  CPU staging through a gloo group: all_gather + all_reduce(SUM) then slice. Eager only, slow;
            exists so that world_size > 1 can be tested with several processes on ONE GPU (NCCL refuses
            duplicate devices).
    (b12x.comm.pcie.PCIeTwoShotBF16 - CUDA-IPC, fp32-accumulating reduce-scatter, world size 4 only - is the
     planned PCIe alternative to NCCL; see docs/MULTI_GPU.md. It needs 4 distinct GPUs, so it is not wired
     here yet.)
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def transport_name() -> str:
    v = os.environ.get("SM120_EP_TRANSPORT", "nccl").strip().lower()
    if v not in ("nccl", "gloo"):
        raise ValueError(f"SM120_EP_TRANSPORT must be nccl|gloo, got {v!r}")
    return v


class NcclTransport:
    name = "nccl"
    graph_capturable = True

    def __init__(self, group):
        self.group = group
        self.world = dist.get_world_size(group)
        self.rank = dist.get_rank(group)

    def all_gather(self, out: torch.Tensor, inp: torch.Tensor) -> None:
        """inp [n, ...] -> out [world*n, ...] (rank-major)."""
        dist.all_gather_into_tensor(out, inp, group=self.group)

    def reduce_scatter_sum(self, out: torch.Tensor, inp: torch.Tensor) -> None:
        """inp [world*n, ...] -> out [n, ...] = sum over ranks of their inp[rank*n:(rank+1)*n]."""
        dist.reduce_scatter_tensor(out, inp, op=dist.ReduceOp.SUM, group=self.group)

    def all_gather_counts(self, n: int, device) -> list[int]:
        t = torch.tensor([n], dtype=torch.int64, device=device)
        out = torch.empty((self.world,), dtype=torch.int64, device=device)
        dist.all_gather_into_tensor(out, t, group=self.group)
        return [int(v) for v in out.tolist()]  # host sync: prefill (eager) only


class GlooTransport:
    """CPU staging (tests on one GPU). Reduce-scatter = all_reduce(SUM) on CPU then the local slice; the sum is
    done in fp32 to mimic a single rounding."""

    name = "gloo"
    graph_capturable = False

    def __init__(self, group):
        self.group = group
        self.world = dist.get_world_size(group)
        self.rank = dist.get_rank(group)

    def all_gather(self, out: torch.Tensor, inp: torch.Tensor) -> None:
        parts = [torch.empty_like(inp, device="cpu") for _ in range(self.world)]
        dist.all_gather(parts, inp.cpu(), group=self.group)
        out.copy_(torch.cat(parts, 0))

    def reduce_scatter_sum(self, out: torch.Tensor, inp: torch.Tensor) -> None:
        full = inp.float().cpu()
        dist.all_reduce(full, op=dist.ReduceOp.SUM, group=self.group)
        n = out.shape[0]
        out.copy_(full[self.rank * n:(self.rank + 1) * n].to(out.dtype))

    def all_gather_counts(self, n: int, device) -> list[int]:
        parts = [torch.zeros((1,), dtype=torch.int64) for _ in range(self.world)]
        dist.all_gather(parts, torch.tensor([n], dtype=torch.int64), group=self.group)
        return [int(p.item()) for p in parts]


def make_transport(group, name: str | None = None):
    name = name or transport_name()
    if name == "gloo":
        return GlooTransport(group)
    return NcclTransport(group)
