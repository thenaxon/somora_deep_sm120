"""DP-wide max-sync of the decode graph size and of the prefill row count (sm120_ep.dp_sync), 2 gloo processes on ONE GPU.

    PYTHONPATH=/opt/sm120/sm120:/opt/deepinfer python3 /opt/sm120/tests/test_ep_dp_sync.py

Each process patches DeepInfer's real CudaGraphRunner.select plus a minimal worker whose execute_model_step
follows DeepInfer's order: select() for the ZOS correction, select() again in input preparation (padding the
batch with dummy tokens up to the graph size), then one EP MoE layer (sm120_ep.EPRoutedExperts over b12x local
experts, gloo transport) on graph-size rows with is_prompt=False. Ranks get different batch sizes, including
placeholder steps. Prefill steps (real or placeholder, one dummy token) run the EP layer PREFILL_LAYERS times
with is_prompt=True. Checks: both ranks pick the same graph size every decode step (= select(max of local
sizes)), both selects inside a step agree, every prefill layer pads to the step's DP-max token count without a
per-layer count exchange, one all-reduce per step, and the EP output of the real rows matches a full-expert run.
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

# (kind, n): kind "d" decode with n tokens, "p" prefill with seqlen_q list n; n None = placeholder step
STEPS = {0: [("d", 1), ("d", 5), ("d", None), ("p", [300, 20]), ("d", 3), ("p", None), ("d", 8), ("d", 16)],
         1: [("d", 7), ("d", 2), ("d", 4), ("p", None), ("d", None), ("p", [50]), ("d", 8), ("d", 1)]}
PREFILL_LAYERS = 3


class _SchedOut:
    def __init__(self, n):
        self.num_batched_tokens = n


class _Seq:
    def __init__(self, n):
        self.seqlen_q = n


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
        tr = GlooTransport(group)
        seen = {"rows": [], "counts": 0}
        ag, agc = tr.all_gather, tr.all_gather_counts

        def all_gather(out, inp):
            seen["rows"].append(inp.shape[0])
            return ag(out, inp)

        def all_gather_counts(n, device):
            seen["counts"] += 1
            return agc(n, device)

        tr.all_gather, tr.all_gather_counts = all_gather, all_gather_counts
        ep = EPRoutedExperts(local, group=None, num_local_experts=nl, ep_rank=rank, hidden=T.K, topk=T.TOPK,
                             decode_rows=10 ** 6, transport=tr)  # fixed capacity would be huge

        class Worker:
            model_runner = Runner()

            def execute_model_step(self, is_prompt, is_placeholder, seq_metadata_list=None, scheduler_outputs=None):
                if is_prompt:
                    n = 1 if is_placeholder else sum(sq.seqlen_q for sq in seq_metadata_list)
                    x, ids, w = T.make_tokens(n, 91 * rank + n, dev)
                    ref = full(x, ids, w)
                    seen["rows"].clear()
                    cs = [T.cos(ep(x, ids, w, is_prompt=True), ref) for _ in range(PREFILL_LAYERS)]
                    return None, None, min(cs), tuple(seen["rows"])
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
        for kind, n in STEPS[rank]:
            prompt = kind == "p"
            if n is None:
                res.append((None,) + wk.execute_model_step(prompt, True, None, None))
            elif prompt:
                res.append((n,) + wk.execute_model_step(True, False, [_Seq(m) for m in n], _SchedOut(sum(n))))
            else:
                res.append((n,) + wk.execute_model_step(False, False, None, _SchedOut(n)))
        q.put((rank, "ok", res, runner.batch_sizes, (dp_sync._STATE["n_sync"], seen["counts"])))
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
    def tokens(r, i):
        n = STEPS[r][i][1]
        return 1 if n is None else (sum(n) if isinstance(n, list) else n)

    for i in range(len(STEPS[0])):
        need = max(tokens(r, i) for r in range(world))
        row = [results[r][1][i] for r in range(world)]
        if STEPS[0][i][0] == "p":
            line = "  ".join(f"rank{r}: local={tokens(r, i)} ep_rows/layer={rows} cos={c:.6f}"
                             for r, (_n, _g, _gz, c, rows) in enumerate(row))
            good = all(rows == (need,) * PREFILL_LAYERS and c > 0.9999 for _n, _g, _gz, c, rows in row)
            print(f"step {i} (prefill, {PREFILL_LAYERS} layers): expected rows {need}: {line}  {'OK' if good else 'MISMATCH'}")
        else:
            want = bs[bisect_left(bs, need)]
            line = "  ".join(f"rank{r}: local={n} graph={g} zos_graph={gz} ep_rows={rows} cos={c:.6f}"
                             for r, (n, g, gz, c, rows) in enumerate(row))
            good = all(g == want and rows == want and (gz is None or gz == g) and c > 0.9999 for _n, g, gz, c, rows in row)
            print(f"step {i}: expected graph {want}: {line}  {'OK' if good else 'MISMATCH'}")
        ok &= good
    n_sync, n_counts = results[0][3]
    print(f"all-reduces per rank: {n_sync} (one per step, {len(STEPS[0])} steps); per-layer count exchanges: {n_counts}")
    ok &= n_sync == len(STEPS[0]) and n_counts == 0
    print("ALL OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


def test_dp_sync():
    import subprocess
    assert subprocess.call([sys.executable, os.path.abspath(__file__)]) == 0


if __name__ == "__main__":
    main()
