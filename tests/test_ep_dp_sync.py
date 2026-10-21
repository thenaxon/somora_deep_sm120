"""DP-wide decode graph-size max-sync (sm120_ep.dp_sync) with 2 gloo processes on ONE GPU.

    PYTHONPATH=/opt/sm120/sm120:/opt/deepinfer python3 /opt/sm120/tests/test_ep_dp_sync.py

Each process patches DeepInfer's real CudaGraphRunner.select plus a minimal worker whose execute_model_step
follows DeepInfer's order: select() for the ZOS correction, select() again in input preparation (padding the
batch with dummy tokens up to the graph size), then one EP MoE layer (sm120_ep.EPRoutedExperts over b12x local
experts, gloo transport) on graph-size rows with is_prompt=False. Ranks get different batch sizes, including
placeholder steps. Checks: both ranks pick the same graph size every step (= select(max of local sizes)),
both selects inside a step agree, and the EP output of the real rows matches a full-expert run.
"""

from __future__ import annotations

import os
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "sm120"))
sys.path.insert(0, HERE)
os.environ.setdefault("B12X_AUTOTUNE", "0")
os.environ.setdefault("B12X_W4A8_TINY_DECODE", "0")

STEPS = {0: [1, 5, None, 3, 8, 16], 1: [7, 2, 4, None, 8, 1]}  # None = placeholder step


class _SchedOut:
    def __init__(self, n):
        self.num_batched_tokens = n


def _worker(rank: int, world: int, init_file: str, q):
    import torch.distributed as dist
    import sm120_moe
    import test_ep_moe as T
    from deepinfer.worker.cuda_graph_runner import CudaGraphRunner, build_graph_batch_sizes
    from sm120_ep import dp_sync
    from sm120_ep.moe import EPRoutedExperts
    from sm120_ep.transport import GlooTransport
    try:
        torch.cuda.set_device(0)
        dev = torch.device("cuda", 0)
        dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
        group = dist.group.WORLD
        runner = CudaGraphRunner.__new__(CudaGraphRunner)
        runner.batch_sizes = build_graph_batch_sizes(16, 4)
        runner.graphs = {b: f"graph{b}" for b in runner.batch_sizes}

        class Runner:
            use_cuda_graph = True
            cuda_graph_runner = runner

        wts = T.make_weights(4)
        nl = T.E // world
        local = sm120_moe.B12xRoutedExperts(T.make_moe(wts, range(rank * nl, (rank + 1) * nl), dev))
        full = sm120_moe.B12xRoutedExperts(T.make_moe(wts, range(0, T.E), dev))
        ep = EPRoutedExperts(local, group=None, num_local_experts=nl, ep_rank=rank, hidden=T.K, topk=T.TOPK,
                             decode_rows=10 ** 6, transport=GlooTransport(group))  # fixed capacity would be huge

        class Worker:
            model_runner = Runner()

            def execute_model_step(self, is_prompt, is_placeholder, seq_metadata_list=None, scheduler_outputs=None):
                n = 1 if is_placeholder else scheduler_outputs.num_batched_tokens  # placeholder = 1 dummy token
                g_zos = None if is_placeholder else self.model_runner.cuda_graph_runner.select(n)[0]
                g, _ = self.model_runner.cuda_graph_runner.select(n)
                x, ids, w = T.make_tokens(g, 77 * rank + n, dev)
                ids[n:] = torch.randint(0, T.E, ids[n:].shape, device=dev)  # padding rows route like real ones
                out = ep(x, ids, w, is_prompt=False)
                ref = full(x[:n], ids[:n], w[:n])
                return g, g_zos, T.cos(out[:n], ref), out.shape[0]

        dp_sync.patch_worker(Worker, group)
        dp_sync.patch_graph_runner(CudaGraphRunner)
        dp_sync._STATE["active"] = True
        wk = Worker()
        res = []
        for n in STEPS[rank]:
            if n is None:
                res.append((None,) + wk.execute_model_step(False, True, None, None))
            else:
                res.append((n,) + wk.execute_model_step(False, False, None, _SchedOut(n)))
        q.put((rank, "ok", res, runner.batch_sizes, dp_sync._STATE["n_sync"]))
        dist.barrier()
    except Exception:  # noqa: BLE001
        import traceback
        q.put((rank, "fail", traceback.format_exc()[-3000:], None, None))


def main():
    import torch.multiprocessing as mp
    from bisect import bisect_left
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    world = 2
    with tempfile.TemporaryDirectory() as td:
        procs = [ctx.Process(target=_worker, args=(r, world, os.path.join(td, "pg"), q)) for r in range(world)]
        for p in procs:
            p.start()
        results = {r: (st, res, bs, ns) for r, st, res, bs, ns in (q.get(timeout=1800) for _ in procs)}
        for p in procs:
            p.join(60)
    ok = True
    for r, (st, res, *_rest) in results.items():
        if st != "ok":
            print(f"rank {r} FAILED:\n{res}")
            ok = False
    if not ok:
        sys.exit(1)
    bs = results[0][2]
    for i in range(len(STEPS[0])):
        need = max(1 if STEPS[r][i] is None else STEPS[r][i] for r in range(world))
        want = bs[bisect_left(bs, need)]
        row = [results[r][1][i] for r in range(world)]
        line = "  ".join(f"rank{r}: local={n} graph={g} zos_graph={gz} ep_rows={rows} cos={c:.6f}"
                         for r, (n, g, gz, c, rows) in enumerate(row))
        good = all(g == want and rows == want and (gz is None or gz == g) and c > 0.9999 for _n, g, gz, c, rows in row)
        print(f"step {i}: expected graph {want}: {line}  {'OK' if good else 'MISMATCH'}")
        ok &= good
    print(f"gloo all-reduces per rank: {results[0][3]} (one per decode step)")
    print("ALL OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


def test_dp_sync():
    import subprocess
    assert subprocess.call([sys.executable, os.path.abspath(__file__)]) == 0


if __name__ == "__main__":
    main()
