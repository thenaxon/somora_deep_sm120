"""Shutdown of the --dp ranks' engine loops (server/engine.py EngineLoop._run / _poll), 2 gloo processes, CPU only.

    PYTHONPATH=/opt/deepinfer python3 /opt/sm120/tests/test_shutdown_dp.py

Every rank runs the real EngineLoop._run over a fake LLM whose calls make the collectives DeepInfer's do with
DP > 1: has_unfinished_requests() is an all-reduce (MAX) over the DP group, step() is one more collective, close() is
a barrier (Worker.stop). The ranks have work at different times, and the stop request (SIGTERM from the proxy)
reaches them at different times, as with docker stop. Checks: both loops stop at the same iteration and close
together (no hang, no collective timeout), the requests in flight are dropped. Control: the same loop with the stop
checked locally (0.3.0 behaviour) leaves the other rank waiting in a collective until the gloo timeout.
"""

from __future__ import annotations

import datetime
import os
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "server"))

TIMEOUT = 4.0  # gloo collective timeout: a rank left waiting raises instead of hanging the test


def _worker(rank: int, world: int, init_file: str, local_stop: bool, q):
    import torch
    import torch.distributed as dist
    import engine as E

    try:
        dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world,
                                timeout=datetime.timedelta(seconds=TIMEOUT))
        group = dist.group.WORLD
        calls = {"polls": 0, "steps": 0, "closed": None}

        class Scheduler:
            def has_unfinished_seqs(self):  # rank 0 has work in polls 20-60, rank 1 from poll 40 on
                n = calls["polls"]
                return 20 <= n < 60 if rank == 0 else n >= 40

        class Engine:
            scheduler = Scheduler()
            eos_id = 0

            def so_builder_has_pending(self):
                return False

            def has_unfinished_requests(self):  # DeepInfer's: an all-reduce over the DP group
                t = torch.tensor([int(self.scheduler.has_unfinished_seqs())], dtype=torch.int64)
                dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
                return t.item() > 0

            def step(self):  # a DP step: collectives with the other ranks
                t = torch.ones(1)
                dist.all_reduce(t, group=group)
                calls["steps"] += 1
                time.sleep(0.002)
                return []

            def flush_inflight_reqs(self):
                return []

        class LLM:
            llm_engine = Engine()

            def close(self):  # Worker.stop: barriers with the other ranks
                dist.barrier(group=group)
                calls["closed"] = time.monotonic()

        loop = E.EngineLoop("/nonexistent", dp_size=world, dp_rank=rank)
        loop._build = LLM
        loop._poll_group = (group, "cpu")
        poll = loop._poll

        def counted_poll():
            calls["polls"] += 1
            if local_stop:  # 0.3.0: each rank checks its own stop flag and leaves on its own
                return loop.engine.has_unfinished_requests(), loop._stop.is_set()
            return poll()

        loop._poll = counted_poll
        loop.start()
        # the proxy's SIGTERM reaches the ranks at different times, rank 1 while both have work
        threading.Timer(0.3 + 0.4 * rank, loop._stop.set).start()
        loop._thread.join(TIMEOUT + 3)
        clean = not loop._thread.is_alive() and loop._error is None and calls["closed"] is not None
        q.put((rank, clean, calls["polls"], calls["steps"], repr(loop._error) if loop._error else None))
    except BaseException as exc:  # noqa: BLE001
        q.put((rank, False, -1, -1, repr(exc)))
    q.close()
    q.join_thread()  # the queue's feeder thread sends the result before os._exit
    os._exit(0)


def run(local_stop: bool) -> list:
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    with tempfile.TemporaryDirectory() as d:
        init_file = os.path.join(d, "init")
        procs = [ctx.Process(target=_worker, args=(r, 2, init_file, local_stop, q)) for r in range(2)]
        for p in procs:
            p.start()
        out = sorted(q.get(timeout=60) for _ in procs)
        for p in procs:
            p.join(10)
            if p.is_alive():
                p.kill()
    return out


def main() -> None:
    ok = True
    res = run(local_stop=False)
    for rank, clean, polls, steps, err in res:
        print(f"agreed stop, rank {rank}: clean {clean}, polls {polls}, steps {steps}, error {err}")
    good = all(r[1] for r in res) and res[0][2] == res[1][2] and res[0][3] == res[1][3]
    print(f"agreed stop: both ranks stopped at the same iteration and closed together: {'OK' if good else 'FAILED'}")
    ok &= good
    res = run(local_stop=True)
    for rank, clean, polls, steps, err in res:
        print(f"control (local stop), rank {rank}: clean {clean}, polls {polls}, steps {steps}, error {err}")
    good = not all(r[1] for r in res)
    print(f"control: a rank is left in a collective: {'OK (as expected)' if good else 'UNEXPECTED (no hang)'}")
    ok &= good
    print("ALL OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


def test_shutdown_dp():
    import subprocess
    assert subprocess.call([sys.executable, os.path.abspath(__file__)]) == 0


if __name__ == "__main__":
    main()
