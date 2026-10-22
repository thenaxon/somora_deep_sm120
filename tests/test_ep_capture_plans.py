"""EP decode graph capture at start-up plans every b12x bucket in the eager warm-up (sm120_ep.dp_sync.patch_capture),
4 gloo processes on ONE GPU (world 4 emulated).

    PYTHONPATH=/opt/sm120/sm120:/opt/deepinfer python3 /opt/sm120/tests/test_ep_capture_plans.py

DeepInfer's CudaGraphRunner.capture (and DFlashCudaGraphRunner.capture) runs, for every batch size of the list,
largest first, one eager warm-up forward and then the capture. The rank-local experts (the real
sm120_moe.B12xRoutedExperts) can create a b12x plan only outside a capture, so the warm-up must request the same
bucket as the capture. Each rank runs the same loop through a runner whose capture() is wrapped by
dp_sync.patch_capture; "capture" means sm120_moe._capturing() and sm120_ep.moe._capturing() return True, so a plan
request raises the production error. Checks: no plan is requested during capture, warm-up and capture use the
graph size as EP rows, the output matches a full-expert run. Control: the same loop without the wrapper (the
v0.2.3 behaviour: warm-up at the fixed capacity) must hit the error, so this test would have caught it.
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

WORLD = 4
MAX_GRAPH = 32  # graph batch sizes 1, 2, 4, 8, 16, 24, 32; fixed capacity = 32 rows per rank (world*32 = bucket 512)


def _worker(rank: int, init_file: str, q):
    import torch.distributed as dist
    import sm120_moe
    import sm120_ep.moe as M
    import test_ep_moe as T
    from deepinfer.worker.cuda_graph_runner import build_graph_batch_sizes
    from sm120_ep import dp_sync
    from sm120_ep.transport import GlooTransport
    try:
        torch.cuda.set_device(0)
        dev = torch.device("cuda", 0)
        dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD)
        tr = GlooTransport(dist.group.WORLD)
        rows_seen = []
        ag = tr.all_gather

        def all_gather(out, inp):
            rows_seen.append(inp.shape[0])
            return ag(out, inp)

        tr.all_gather = all_gather
        wts = T.make_weights(4)
        nl = T.E // WORLD
        full = sm120_moe.B12xRoutedExperts(T.make_moe(wts, range(0, T.E), dev))
        sizes = build_graph_batch_sizes(MAX_GRAPH, 8)
        dp_sync._STATE["active"] = True  # DP sync installed; no step synced yet (start-up)

        def make_ep():  # fresh local experts: no plans yet
            local = sm120_moe.B12xRoutedExperts(T.make_moe(wts, range(rank * nl, (rank + 1) * nl), dev))
            return M.EPRoutedExperts(local, group=None, num_local_experts=nl, ep_rank=rank, hidden=T.K, topk=T.TOPK,
                                     decode_rows=MAX_GRAPH, transport=tr)

        class Runner:
            """DeepInfer's capture loop: per size, largest first, an eager warm-up then the capture."""
            batch_sizes = sizes

            def capture(self, ep, log):
                for bs in reversed(self.batch_sizes):
                    x, ids, w = T.make_tokens(bs, 17 * rank + bs, dev)
                    rows_seen.clear()
                    ep(x, ids, w, is_prompt=False)  # warm-up (eager)
                    warm = rows_seen[-1]
                    rows_seen.clear()
                    cap_m, cap_e = sm120_moe._capturing, M._capturing
                    sm120_moe._capturing = M._capturing = lambda: True
                    try:
                        out = ep(x, ids, w, is_prompt=False)
                        err = None
                    except RuntimeError as exc:  # the production "plan requested during CUDA graph capture"
                        out, err = None, str(exc)
                    finally:
                        sm120_moe._capturing, M._capturing = cap_m, cap_e
                    c = T.cos(out[:bs], full(x, ids, w)) if out is not None else None
                    log.append((bs, warm, rows_seen[-1] if rows_seen else None, err, c))
                    # all ranks take the same branch (same sizes, same buckets), so no rank is left in a collective

        class Wrapped(Runner):
            pass

        dp_sync.patch_capture(Wrapped)
        fixed_log, fixed_flag = [], []
        Wrapped().capture(make_ep(), fixed_log)
        ctl_log = []
        Runner().capture(make_ep(), ctl_log)
        fixed_flag.append(dp_sync._STATE["in_capture"])  # restored after capture()
        q.put((rank, "ok", fixed_log, ctl_log, fixed_flag))
        dist.barrier()
    except Exception:  # noqa: BLE001
        import traceback
        q.put((rank, "fail", traceback.format_exc()[-3000:], None, None))


def main():
    import torch.multiprocessing as mp
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    with tempfile.TemporaryDirectory() as td:
        procs = [ctx.Process(target=_worker, args=(r, os.path.join(td, "pg"), q)) for r in range(WORLD)]
        for p in procs:
            p.start()
        results = {r: (st, a, b, c) for r, st, a, b, c in (q.get(timeout=1800) for _ in procs)}
        for p in procs:
            p.join(60)
    ok = True
    for r, (st, a, *_x) in sorted(results.items()):
        if st != "ok":
            print(f"rank {r} FAILED:\n{a}")
            ok = False
    if not ok:
        sys.exit(1)
    for r, (_st, log, ctl, flag) in sorted(results.items()):
        good = all(err is None and warm == bs and cap == bs and c > 0.9999 for bs, warm, cap, err, c in log)
        good &= flag == [False]
        print(f"rank {r} with patch_capture: " + ", ".join(
            f"{bs}: warm-up {warm} / capture {cap} rows{' ERROR ' + err[:60] if err else ''}"
            for bs, warm, cap, err, _c in log) + f", min cos {min(c for *_y, c in log if c is not None):.6f}  "
            f"{'OK' if good else 'MISMATCH'}")
        ok &= good
        fails = [bs for bs, _w, _c, err, _cos in ctl if err and "during CUDA graph capture" in err]
        cgood = bool(fails)
        print(f"rank {r} control (no patch_capture, warm-up at the fixed capacity {MAX_GRAPH}): plan requested during "
              f"capture for graph sizes {fails}  {'OK (caught)' if cgood else 'NOT CAUGHT'}")
        ok &= cgood
    print("ALL OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


def test_capture_plans():
    import subprocess
    assert subprocess.call([sys.executable, os.path.abspath(__file__)]) == 0


if __name__ == "__main__":
    main()
