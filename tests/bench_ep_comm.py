#!/usr/bin/env python3
"""Option-B EP collectives on a multi-GPU PCIe machine (e.g. 4x RTX PRO 6000 Blackwell Max-Q): NCCL vs b12x PCIeTwoShotBF16, plus a full EP MoE layer.

One command on the 4-GPU machine (inside the deepinfer-sm120 image):

    torchrun --nproc_per_node 4 /opt/sm120/tests/bench_ep_comm.py --out /work/bench_ep_comm.json [--moe]

Per rows-per-rank R it times, as CUDA-graph replays (the decode path) and eager calls:
  all-gather       packed EP rows [R, 2H + 8k bytes] (hidden bf16 | int32 ids | fp32 weights) -> [4R, ...]
  reduce-scatter   bf16 partial outputs [4R, H] -> [R, H] (sum)
with NCCL (torch.distributed on the default NCCL group) and b12x.comm.pcie.PCIeTwoShotBF16 (CUDA-IPC, world 4,
fp32-accumulated single-rounding reduce-scatter). Checks that both transports give the same all-gather bytes and
the same reduce-scatter (within bf16 rounding). With --moe it also times one EP MoE layer end to end
(sm120_ep.EPRoutedExperts over synthetic MXFP4 experts: --experts global, --intermediate, top-k 6) at the same R,
against the single-GPU full-expert layer on rank 0. Prints a table on rank 0 and writes JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "sm120"))
sys.path.insert(0, HERE)


def graph_time(fn, iters=50):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    dist.barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        g.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / iters * 1000.0  # us


def eager_time(fn, iters=30):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(ts)


class B12xTwoShot:
    """PCIeTwoShotBF16 with prepared all-gather / reduce-scatter plans for fixed shapes."""

    def __init__(self, group, device, max_rows, row_elems):
        from b12x.comm import pcie
        from b12x.comm.pcie import _twoshot_preparation as tsp
        from b12x import preparation as prep
        self.pcie, self.tsp, self.prep = pcie, tsp, prep
        self.rt = pcie.PCIeTwoShotBF16.from_exchange_group(exchange_group=group, device=device, max_rows=max_rows,
                                                            row_elems=row_elems)

    def plan(self, surface, payload, out, name):
        q = self.pcie.query_from_runtime(self.rt, surface=surface, call={"payload": payload, "out": out})
        p = self.pcie.plan(q, runtime=self.rt)
        req = p.request(name=name, prepare_call=lambda state: self.tsp.prepared_call(state, payload=payload, out=out))
        self.prep.prepare_default(req)
        return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=5120)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--rows", default="1,4,8,16,32,64,192,512,2048")
    ap.add_argument("--moe", action="store_true", help="also time one EP MoE layer (synthetic MXFP4 experts)")
    ap.add_argument("--experts", type=int, default=384)
    ap.add_argument("--intermediate", type=int, default=2048)
    ap.add_argument("--no-b12x", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=dev)
    world, rank = dist.get_world_size(), dist.get_rank()
    gloo = dist.new_group(backend="gloo")
    H, k = a.hidden, a.topk
    row_bytes = 2 * H + 8 * k
    assert row_bytes % 16 == 0, "packed row must be a multiple of 8 bf16 elements"
    rows_list = [int(r) for r in a.rows.split(",") if r]
    Rmax = max(rows_list)
    info = dict(world=world, hidden=H, topk=k, row_bytes=row_bytes, gpu=torch.cuda.get_device_name(),
                p2p=[torch.cuda.can_device_access_peer(local_rank, j) for j in range(torch.cuda.device_count())
                     if j != local_rank],
                nccl=".".join(map(str, torch.cuda.nccl.version())), torch=torch.__version__)
    if rank == 0:
        print(json.dumps(info), flush=True)

    b12x_ag = b12x_rs = None
    if not a.no_b12x:
        try:
            b12x_ag = B12xTwoShot(gloo, dev, world * Rmax, row_bytes // 2)
            b12x_rs = B12xTwoShot(gloo, dev, world * Rmax, H)
        except Exception as exc:  # noqa: BLE001
            if rank == 0:
                print(f"b12x PCIeTwoShotBF16 unavailable: {type(exc).__name__}: {exc}", flush=True)
    results = []
    for R in rows_list:
        g = torch.Generator(device=dev).manual_seed(1000 * rank + R)
        send = torch.randint(0, 255, (R, row_bytes), dtype=torch.uint8, device=dev, generator=g)
        recv = torch.empty((world * R, row_bytes), dtype=torch.uint8, device=dev)
        part = (torch.randn((world * R, H), device=dev, generator=g) * 0.1).to(torch.bfloat16)
        red = torch.empty((R, H), dtype=torch.bfloat16, device=dev)
        r = dict(rows=R, ag_bytes_per_rank=R * row_bytes, rs_bytes_per_rank=world * R * H * 2)

        def nccl_ag():
            dist.all_gather_into_tensor(recv, send)

        def nccl_rs():
            dist.reduce_scatter_tensor(red, part, op=dist.ReduceOp.SUM)

        r["nccl_ag_graph_us"] = graph_time(nccl_ag)
        r["nccl_rs_graph_us"] = graph_time(nccl_rs)
        r["nccl_ag_eager_us"] = eager_time(nccl_ag)
        r["nccl_rs_eager_us"] = eager_time(nccl_rs)
        ref_recv, ref_red = recv.clone(), red.clone()
        if b12x_ag is not None:
            try:
                s16, r16 = send.view(torch.bfloat16), recv.view(torch.bfloat16)
                p_ag = b12x_ag.plan("PCIeTwoShotBF16.all_gather", s16, r16, f"ep.ag.r{R}")
                p_rs = b12x_rs.plan("PCIeTwoShotBF16.reduce_scatter", part, red, f"ep.rs.r{R}")

                def b_ag():
                    b12x_ag.rt.all_gather(s16, r16, plan=p_ag)

                def b_rs():
                    b12x_rs.rt.reduce_scatter(part, red, plan=p_rs)

                b_ag()
                b_rs()
                torch.cuda.synchronize()
                r["b12x_ag_matches_nccl"] = bool(torch.equal(recv, ref_recv))
                r["b12x_rs_max_abs_vs_nccl"] = (red.float() - ref_red.float()).abs().max().item()
                r["b12x_ag_eager_us"] = eager_time(b_ag)
                r["b12x_rs_eager_us"] = eager_time(b_rs)
                with b12x_ag.rt.capture(plan=p_ag):
                    r["b12x_ag_graph_us"] = graph_time(b_ag)
                with b12x_rs.rt.capture(plan=p_rs):
                    r["b12x_rs_graph_us"] = graph_time(b_rs)
            except Exception as exc:  # noqa: BLE001
                r["b12x_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        results.append(r)
        if rank == 0:
            print(json.dumps(r), flush=True)

    moe_results = []
    if a.moe:
        moe_results = bench_moe(a, dev, world, rank, rows_list)

    if rank == 0:
        print("\nrows/rank | AG bytes/rank | NCCL AG graph us | b12x AG graph us | RS bytes/rank | NCCL RS graph us | "
              "b12x RS graph us | NCCL AG+RS eager us")
        for r in results:
            print(f"{r['rows']:9d} | {r['ag_bytes_per_rank']:13d} | {r['nccl_ag_graph_us']:16.1f} | "
                  f"{r.get('b12x_ag_graph_us', float('nan')):16.1f} | {r['rs_bytes_per_rank']:13d} | "
                  f"{r['nccl_rs_graph_us']:16.1f} | {r.get('b12x_rs_graph_us', float('nan')):16.1f} | "
                  f"{r['nccl_ag_eager_us'] + r['nccl_rs_eager_us']:10.1f}" + (f"  {r['b12x_error']}" if "b12x_error" in r else ""))
        for m in moe_results:
            print(json.dumps(m))
        if a.out:
            with open(a.out, "w") as f:
                json.dump(dict(info=info, collectives=results, moe=moe_results), f, indent=1)
    dist.barrier()
    dist.destroy_process_group()


def bench_moe(a, dev, world, rank, rows_list):
    """One EP MoE layer (sm120_ep.EPRoutedExperts, NCCL) vs the full-expert layer on one GPU (rank 0)."""
    import sm120_moe
    import test_ep_moe as T
    from sm120_ep.moe import EPRoutedExperts
    from sm120_ep.transport import NcclTransport
    T.E, T.K, T.N, T.TOPK = a.experts, a.hidden, a.intermediate, a.topk
    wts = T.make_weights(7)  # identical on all ranks (seeded)
    nl = T.E // world
    local = sm120_moe.B12xRoutedExperts(T.make_moe(wts, range(rank * nl, (rank + 1) * nl), dev))
    full = sm120_moe.B12xRoutedExperts(T.make_moe(wts, range(0, T.E), dev)) if rank == 0 else None
    del wts
    ep = EPRoutedExperts(local, group=None, num_local_experts=nl, ep_rank=rank, hidden=T.K, topk=T.TOPK,
                         decode_rows=max(rows_list), transport=NcclTransport(dist.group.WORLD))
    out = []
    for R in rows_list:
        x, ids, w = T.make_tokens(R, 31 * rank + R, dev)
        res = dict(moe_rows_per_rank=R)
        from sm120_ep import dp_sync
        dp_sync._STATE.update(active=True, synced=R)  # DP graph-size sync: EP rows = the step's rows
        res["ep_layer_graph_us"] = graph_time(lambda: ep(x, ids, w, is_prompt=False), iters=20)
        dp_sync._STATE.update(active=False, synced=None)  # fixed capacity = max(--rows) rows per rank
        res["ep_layer_fixed_capacity_graph_us"] = graph_time(lambda: ep(x, ids, w, is_prompt=False), iters=20)
        dp_sync._STATE.update(active=True, synced=R)
        if rank == 0:
            res["single_gpu_full_layer_graph_us"] = _local_graph_time(lambda: full(x, ids, w))
            got, ref = ep(x, ids, w, is_prompt=False), full(x, ids, w)
            res["ep_vs_full_cos"] = T.cos(got, ref)
        else:
            ep(x, ids, w, is_prompt=False)
        dp_sync._STATE.update(active=False, synced=None)
        dist.barrier()
        out.append(res)
    return out


def _local_graph_time(fn, iters=20):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        fn()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        g.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / iters * 1000.0


if __name__ == "__main__":
    main()
