"""Collectives for the Option-B expert-parallel MoE (all-gather tokens, reduce-scatter outputs).

    SM120_EP_TRANSPORT = nccl (default) | gloo | b12x
      nccl  torch.distributed all_gather_into_tensor / reduce_scatter_tensor on DeepInfer's EP process group
            (NCCL over PCIe on 4x RTX PRO 6000; CUDA-graph capturable). Needs one GPU per rank.
      gloo  CPU staging through a gloo group: all_gather + all_reduce(SUM) then slice. Eager only, slow;
            exists so that world_size > 1 can be tested with several processes on ONE GPU (NCCL refuses
            duplicate devices).
      b12x  eager prefill collectives through b12x.comm.pcie.PCIeTwoShotBF16 (CUDA-IPC peer writes, no NCCL;
            fp32-accumulating single-rounding reduce-scatter; world size 4 only), NCCL for everything else
            (decode under CUDA graphs, counts, shapes it does not take). For hosts where NCCL does not use P2P.
            SM120_EP_B12X_MAX_ROWS (default 4096 = --max-num-batched-tokens) rows per rank; prefill rows are
            padded to multiples of 256 (one prepared plan per size). UNTESTED on 4 GPUs so far: run
            tests/bench_ep_comm.py there first (docs/MULTI_GPU.md).
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def transport_name() -> str:
    v = os.environ.get("SM120_EP_TRANSPORT", "nccl").strip().lower()
    if v not in ("nccl", "gloo", "b12x"):
        raise ValueError(f"SM120_EP_TRANSPORT must be nccl|gloo|b12x, got {v!r}")
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


_B12X: dict = {}  # shared by all layers: {"cpu": gloo group, "ok": bool, (row_elems): runtime, plans}

_BUCKET = 256


class B12xTransport(NcclTransport):
    """NCCL transport whose eager prefill all-gather / reduce-scatter go through b12x PCIeTwoShotBF16.

    Every decision that selects b12x or NCCL depends only on values that are equal on all ranks (row count,
    shapes, capture state, and an all-reduced success flag when a runtime or plan is created), so the ranks never
    mix transports inside one collective."""

    name = "b12x"

    def __init__(self, group, cpu_group=None):
        super().__init__(group)
        self.max_rows = int(os.environ.get("SM120_EP_B12X_MAX_ROWS", "4096"))
        if cpu_group is not None:  # tests: a gloo group with the EP ranks (default: DeepInfer's CPU DP group)
            _B12X["cpu"] = cpu_group

    # ---- shared state
    def _cpu_group(self):
        if "cpu" not in _B12X:
            cpu = None
            try:
                from deepinfer.model_executor.parallel_utils.parallel_state import PG
                cand = PG.cpu_data_parallel_group()
                if sorted(dist.get_process_group_ranks(cand)) == sorted(dist.get_process_group_ranks(self.group)):
                    cpu = cand
            except Exception:  # noqa: BLE001
                cpu = None
            if cpu is None:
                print("[sm120_ep] b12x transport: no CPU group with the EP ranks; using NCCL", flush=True)
            _B12X["cpu"] = cpu
        return _B12X["cpu"]

    def _agree(self, ok: bool) -> bool:
        t = torch.tensor([1 if ok else 0], dtype=torch.int64)
        dist.all_reduce(t, op=dist.ReduceOp.MIN, group=_B12X["cpu"])
        return bool(t.item())

    def _runtime(self, row_elems: int, device):
        key = ("rt", row_elems)
        if key not in _B12X:
            rt, err = None, None
            if self.world != 4 or self._cpu_group() is None:
                _B12X[key] = None
                return None
            try:
                from b12x.comm import pcie
                rt = pcie.PCIeTwoShotBF16.from_exchange_group(exchange_group=_B12X["cpu"], device=device,
                                                              max_rows=self.world * self.max_rows, row_elems=row_elems)
            except Exception as exc:  # noqa: BLE001
                err = exc
            if not self._agree(rt is not None):
                print(f"[sm120_ep] b12x PCIeTwoShotBF16 (row_elems {row_elems}) unavailable on some rank "
                      f"({type(err).__name__ if err else 'peer'}: {err}); using NCCL", flush=True)
                rt = None
            else:
                print(f"[sm120_ep] b12x PCIeTwoShotBF16 runtime: {self.world} ranks, {self.max_rows} rows/rank, "
                      f"row {row_elems} bf16", flush=True)
            _B12X[key] = rt
        return _B12X[key]

    def _plan(self, surface: str, rt, rows_in: int, rows_out: int, device):
        """(plan, owned payload, owned out) for this surface and shape, or None (all ranks agree)."""
        key = (surface, rt.row_elems, rows_in)
        if key not in _B12X:
            res, err = None, None
            try:
                from b12x import preparation as prep
                from b12x.comm import pcie
                from b12x.comm.pcie import _twoshot_preparation as tsp
                pay, out = self._buffers(surface, rt, device)
                pay, out = pay[:rows_in], out[:rows_out]  # views of the max-size buffers: same strides, same base
                q = pcie.query_from_runtime(rt, surface=surface, call={"payload": pay, "out": out})
                plan = pcie.plan(q, runtime=rt)
                req = plan.request(name=f"sm120_ep.{surface}.r{rows_in}",
                                   prepare_call=lambda state: tsp.prepared_call(state, payload=pay, out=out))
                prep.prepare_default(req)
                res = (plan, pay, out)
            except Exception as exc:  # noqa: BLE001
                err = exc
            if not self._agree(res is not None):
                print(f"[sm120_ep] b12x {surface} plan for {rows_in} rows failed ({type(err).__name__ if err else 'peer'}:"
                      f" {err}); NCCL for this size", flush=True)
                res = None
            _B12X[key] = res
        return _B12X[key]

    def _buffers(self, surface: str, rt, device):
        """Owned payload / output buffers of the largest size, shared by every row bucket of this surface."""
        key = ("buf", surface, rt.row_elems)
        if key not in _B12X:
            big, small = self.world * self.max_rows, self.max_rows
            rin, rout = (small, big) if surface.endswith("all_gather") else (big, small)
            _B12X[key] = (torch.empty((rin, rt.row_elems), dtype=torch.bfloat16, device=device),
                          torch.empty((rout, rt.row_elems), dtype=torch.bfloat16, device=device))
        return _B12X[key]

    def _eligible(self, rows: int) -> bool:
        return 0 < rows <= self.max_rows and rows % _BUCKET == 0 and not torch.cuda.is_current_stream_capturing()

    # ---- EP interface
    def round_rows(self, rows: int) -> int:
        """Prefill rows per rank, padded to the plan bucket (equal on all ranks, as `rows` is)."""
        r = -(-rows // _BUCKET) * _BUCKET
        return r if r <= self.max_rows else rows

    def all_gather(self, out: torch.Tensor, inp: torch.Tensor) -> None:
        n = inp.shape[0]
        if self._eligible(n) and inp.dtype == torch.uint8 and inp.dim() == 2 and inp.shape[1] % 16 == 0:
            rt = self._runtime(inp.shape[1] // 2, inp.device)
            p = None if rt is None else self._plan("PCIeTwoShotBF16.all_gather", rt, n, self.world * n, inp.device)
            if p is not None:
                plan, pay, res = p
                pay.view(torch.uint8).copy_(inp)
                rt.all_gather(pay, res, plan=plan)
                out.copy_(res.view(torch.uint8).view(out.shape))
                return
        super().all_gather(out, inp)

    def reduce_scatter_sum(self, out: torch.Tensor, inp: torch.Tensor) -> None:
        n = out.shape[0]
        if self._eligible(n) and inp.dtype == torch.bfloat16 and inp.dim() == 2:
            rt = self._runtime(inp.shape[1], inp.device)
            p = None if rt is None else self._plan("PCIeTwoShotBF16.reduce_scatter", rt, self.world * n, n, inp.device)
            if p is not None:
                plan, pay, res = p
                pay.copy_(inp)
                rt.reduce_scatter(pay, res, plan=plan)
                out.copy_(res)
                return
        super().reduce_scatter_sum(out, inp)


def make_transport(group, name: str | None = None):
    name = name or transport_name()
    if name == "gloo":
        return GlooTransport(group)
    if name == "b12x":
        return B12xTransport(group)
    return NcclTransport(group)
