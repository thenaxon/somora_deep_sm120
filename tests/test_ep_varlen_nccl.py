"""Exact-size EP prefill collectives (SM120_EP_VARLEN) on NCCL, one GPU per rank (the multi-GPU host).

    torchrun --nproc_per_node 4 /opt/sm120/tests/test_ep_varlen_nccl.py

The single-GPU test (test_ep_varlen.py, gloo) cannot show whether NCCL's per-rank reduce (mode 1) sums the bf16
partials in the same order as its reduce-scatter. Here every rank runs one EP prefill layer over a deterministic
per-row stand-in for the experts (zero for rows without a local route) for several count patterns and modes 0 / ag /
1 / force, and checks bit-identity against mode 0 (padded all-gather / reduce-scatter). It also times the layer's
collectives per mode (CUDA events, eager) at one busy rank with 2048 and 4096 tokens.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "sm120"))
sys.path.insert(0, HERE)

PATTERNS = [(300, 1, 1, 1), (300, 200, 1, 1), (0, 50, 0, 7), (64, 64, 64, 64), (2048, 1, 1, 1), (4096, 1, 1, 1)]
MODES = ("0", "ag", "1", "force")
H, TOPK, E = 5120, 6, 384


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dev = torch.device("cuda")
    from sm120_ep import dp_sync
    from sm120_ep.moe import EPRoutedExperts
    from sm120_ep.transport import NcclTransport
    nl = E // world

    def row_fn(xg, idl, wg):
        s = (wg.float() * (idl >= 0) * (idl.float() + 2.0)).sum(-1, keepdim=True)
        return (xg.float() * s).to(torch.bfloat16)

    ep = EPRoutedExperts(row_fn, group=None, num_local_experts=nl, ep_rank=rank, hidden=H, topk=TOPK, decode_rows=64,
                         transport=NcclTransport(dist.group.WORLD))
    dp_sync._STATE["active"] = True
    ok = True
    for pat in PATTERNS:
        n = pat[rank]
        g = torch.Generator(device="cpu").manual_seed(1000 * rank + n)
        x = (torch.randn((n, H), generator=g) * 0.5).to(torch.bfloat16).to(dev)
        ids = torch.rand((n, E), generator=g).topk(TOPK, -1).indices.to(dev)
        w = torch.softmax(torch.randn((n, TOPK), generator=g), -1).float().to(dev)
        dp_sync._STATE["prefill_counts"] = list(pat)
        outs, ms = {}, {}
        for m in MODES:
            os.environ["SM120_EP_VARLEN"] = m
            outs[m] = ep(x, ids, w, is_prompt=True)
            torch.cuda.synchronize()
            dist.barrier()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(5):
                ep(x, ids, w, is_prompt=True)
            b.record()
            b.synchronize()
            ms[m] = a.elapsed_time(b) / 5
            dist.barrier()
        os.environ.pop("SM120_EP_VARLEN", None)
        same = {m: torch.equal(outs["0"].view(torch.int16), outs[m].view(torch.int16)) for m in MODES}
        flags = torch.tensor([int(all(same.values()))], device=dev)
        dist.all_reduce(flags, op=dist.ReduceOp.MIN)
        t = torch.tensor([ms[m] for m in MODES], device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        if rank == 0:
            print(f"counts {pat}: bit-identical to padded on all ranks {bool(flags.item())} (rank 0: {same}); "
                  f"layer ms (max over ranks) " + " ".join(f"{m}={v:.3f}" for m, v in zip(MODES, t.tolist())),
                  flush=True)
        ok &= bool(flags.item())
    if rank == 0:
        print("ALL OK" if ok else "FAILED", flush=True)
    dist.destroy_process_group()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
