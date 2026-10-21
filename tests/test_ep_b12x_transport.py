"""EP prefill over the b12x PCIe transport vs NCCL, on a 4-GPU PCIe host (SM120_EP_TRANSPORT=b12x).

    torchrun --nproc_per_node 4 /opt/sm120/tests/test_ep_b12x_transport.py

Every rank builds the same synthetic MXFP4 experts (test_ep_moe sizes), keeps its quarter, and runs
EPRoutedExperts in prefill mode (is_prompt=True, per-layer count exchange) with different token counts per
rank (including an idle rank), once over NcclTransport and once over B12xTransport. Checks: the b12x path was
taken (a PCIeTwoShotBF16 runtime exists), the outputs agree with NCCL (b12x reduce-scatter accumulates in fp32,
NCCL in bf16: cos > 0.9999) and with a full-expert run of the rank's own tokens. Prints the time per layer.
"""

from __future__ import annotations

import os
import sys
import time

import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "sm120"))
sys.path.insert(0, HERE)
os.environ.setdefault("B12X_AUTOTUNE", "0")


def main() -> int:
    import sm120_moe
    import test_ep_moe as T
    from sm120_ep import transport as tr
    from sm120_ep.moe import EPRoutedExperts
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)
    gloo = dist.new_group(backend="gloo")
    wts = T.make_weights(4)
    nl = T.E // world
    local = sm120_moe.B12xRoutedExperts(T.make_moe(wts, range(rank * nl, (rank + 1) * nl), dev))
    full = sm120_moe.B12xRoutedExperts(T.make_moe(wts, range(0, T.E), dev))
    nccl = EPRoutedExperts(local, group=None, num_local_experts=nl, ep_rank=rank, hidden=T.K, topk=T.TOPK,
                           decode_rows=64, transport=tr.NcclTransport(dist.group.WORLD))
    b12x = EPRoutedExperts(local, group=None, num_local_experts=nl, ep_rank=rank, hidden=T.K, topk=T.TOPK,
                           decode_rows=64, transport=tr.B12xTransport(dist.group.WORLD, cpu_group=gloo))
    ok = True
    for step, counts in enumerate(([300, 20, 0, 1], [1, 1, 1, 1], [4096, 1, 512, 77], [256, 256, 256, 256])):
        n = counts[rank]
        x, ids, w = T.make_tokens(n, 1000 * step + rank, dev)
        ref = full(x, ids, w) if n else None
        a = nccl(x, ids, w, is_prompt=True)
        b = b12x(x, ids, w, is_prompt=True)
        torch.cuda.synchronize()
        tn = tb = 0.0
        for _ in range(5):
            t0 = time.perf_counter(); nccl(x, ids, w, is_prompt=True); torch.cuda.synchronize(); tn += time.perf_counter() - t0
            t0 = time.perf_counter(); b12x(x, ids, w, is_prompt=True); torch.cuda.synchronize(); tb += time.perf_counter() - t0
        used = any(k[0] == "rt" and v is not None for k, v in tr._B12X.items() if isinstance(k, tuple))
        c_ab = T.cos(a, b) if n else 1.0
        c_ref = T.cos(b, ref) if n else 1.0
        good = used and c_ab > 0.9999 and c_ref > 0.9999
        ok &= good
        print(f"[b12x-ep] rank {rank} step {step} counts {counts}: b12x runtime {used} cos(b12x, nccl) {c_ab:.6f} "
              f"cos(b12x, full) {c_ref:.6f}  nccl {tn / 5 * 1e3:.2f} ms  b12x {tb / 5 * 1e3:.2f} ms  "
              f"{'OK' if good else 'FAIL'}", flush=True)
    t = torch.tensor([1 if ok else 0], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    if rank == 0:
        print("ALL OK" if int(t.item()) else "FAILED", flush=True)
    dist.destroy_process_group()
    return 0 if int(t.item()) else 1


if __name__ == "__main__":
    sys.exit(main())
