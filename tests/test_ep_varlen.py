"""Exact-size EP prefill collectives (SM120_EP_VARLEN) against the padded path, 4 gloo processes on ONE GPU.

    PYTHONPATH=/opt/sm120/sm120:/opt/deepinfer python3 /opt/sm120/tests/test_ep_varlen.py

Every rank runs one EP prefill layer (sm120_ep.EPRoutedExperts, DP sync active, the step's per-rank token counts
given) for several count patterns: one busy rank, two busy ranks, ranks with 0 tokens, uniform counts. Per pattern it
runs SM120_EP_VARLEN = 0 (padded all-gather / reduce-scatter, the reference), ag, 1 and force.

1. Bit-identity: the rank-local experts are replaced by a deterministic per-row function of (hidden, local ids,
   weights) that is zero for rows without a local route, like the experts (b12x's fused MoE combines the top-k with
   atomics, so its own output is not run-to-run reproducible). Every mode must equal the reference bit for bit, the
   auto modes must fall back to the padded path where it moves fewer bytes (uniform counts), and the all-gather
   must deliver sum(counts) instead of world * max(counts) rows.
2. With the real b12x experts (B12xRoutedExperts), modes 0 and 1 against a full-expert run: cos > 0.9999.
With gloo both reduce paths sum in fp32 and round once; the NCCL reduce of mode 1 sums in bf16 like NCCL's
reduce-scatter, so mode 1's bit-identity on NCCL is checked on the multi-GPU host (bench/test there).
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

WORLD = 4
PATTERNS = [(300, 1, 1, 1), (300, 200, 1, 1), (0, 50, 0, 7), (64, 64, 64, 64)]
MODES = ("0", "ag", "1", "force")
B12X_MODES = ("0", "1")


def _worker(rank: int, init_file: str, q):
    import torch.distributed as dist
    import sm120_moe
    import test_ep_moe as T
    from sm120_ep import dp_sync
    from sm120_ep.moe import EPRoutedExperts
    from sm120_ep.transport import GlooTransport
    try:
        torch.cuda.set_device(0)
        dev = torch.device("cuda", 0)
        dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD)
        tr = GlooTransport(dist.group.WORLD)
        moved = {"ag": 0, "agv": 0, "rs": 0, "rv": 0}
        ag, agv, rs, rv = tr.all_gather, tr.all_gather_v, tr.reduce_scatter_sum, tr.reduce_v

        def all_gather(out, inp):
            moved["ag"] += out.shape[0]
            return ag(out, inp)

        def all_gather_v(out, inp, counts):
            moved["agv"] += out.shape[0]
            return agv(out, inp, counts)

        def reduce_scatter_sum(out, inp):
            moved["rs"] += inp.shape[0]
            return rs(out, inp)

        def reduce_v(inp, counts):
            moved["rv"] += inp.shape[0]
            return rv(inp, counts)

        tr.all_gather, tr.all_gather_v, tr.reduce_scatter_sum, tr.reduce_v = (all_gather, all_gather_v,
                                                                                 reduce_scatter_sum, reduce_v)
        wts = T.make_weights(5)
        nl = T.E // WORLD

        def row_fn(xg, idl, wg):  # deterministic stand-in for the local experts: per row, zero without local routes
            s = (wg.float() * (idl >= 0) * (idl.float() + 2.0)).sum(-1, keepdim=True)
            return (xg.float() * s).to(torch.bfloat16)

        def row_ref(x, ids, w):  # the same, summed over all ranks' expert ranges in fp32
            acc = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
            for r in range(WORLD):
                loc = ids.to(torch.int64) - r * nl
                loc = torch.where((ids >= 0) & (loc >= 0) & (loc < nl), loc, torch.full_like(loc, -1))
                acc += row_fn(x, loc, w).float()
            return acc.to(torch.bfloat16)

        local = sm120_moe.B12xRoutedExperts(T.make_moe(wts, range(rank * nl, (rank + 1) * nl), dev))
        full = sm120_moe.B12xRoutedExperts(T.make_moe(wts, range(0, T.E), dev))
        eps = {k: EPRoutedExperts(inner, group=None, num_local_experts=nl, ep_rank=rank, hidden=T.K, topk=T.TOPK,
                                  decode_rows=64, transport=tr) for k, inner in (("fn", row_fn), ("b12x", local))}
        dp_sync._STATE["active"] = True
        res = []
        for pat in PATTERNS:
            n = pat[rank]
            x, ids, w = T.make_tokens(n, 1000 * rank + n, dev)
            dp_sync._STATE["prefill_counts"] = list(pat)
            outs, rows, cos = {}, {}, {}
            for m in MODES:
                os.environ["SM120_EP_VARLEN"] = m
                for k in moved:
                    moved[k] = 0
                outs[m] = eps["fn"](x, ids, w, is_prompt=True)
                rows[m] = dict(moved)
                if m in B12X_MODES:  # b12x JIT per bucket and rank is slow: the real experts on two modes only
                    got = eps["b12x"](x, ids, w, is_prompt=True)  # every rank joins the collectives, also with 0 rows
                    cos[m] = T.cos(got, full(x, ids, w)) if n else 1.0
            os.environ.pop("SM120_EP_VARLEN", None)
            ref = outs["0"]
            same = {m: ref.shape == outs[m].shape and torch.equal(ref.view(torch.int16), outs[m].view(torch.int16))
                    for m in MODES}
            exact_ref = torch.equal(ref.view(torch.int16), row_ref(x, ids, w).view(torch.int16)) if n else True
            res.append((pat, same, rows, min(cos.values()), exact_ref))
        q.put((rank, "ok", res))
        dist.barrier()
    except Exception:  # noqa: BLE001
        import traceback
        q.put((rank, "fail", traceback.format_exc()[-3000:]))


def main():
    import torch.multiprocessing as mp
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    with tempfile.TemporaryDirectory() as td:
        procs = [ctx.Process(target=_worker, args=(r, os.path.join(td, "pg"), q)) for r in range(WORLD)]
        for p in procs:
            p.start()
        results = {r: (st, res) for r, st, res in (q.get(timeout=1800) for _ in procs)}
        for p in procs:
            p.join(60)
    ok = True
    for r, (st, res) in sorted(results.items()):
        if st != "ok":
            print(f"rank {r} FAILED:\n{res}")
            ok = False
    if not ok:
        sys.exit(1)
    for i, pat in enumerate(PATTERNS):
        uneven = sum(pat) < (WORLD - 1) * max(pat)
        good = True
        for r in range(WORLD):
            _p, same, rows, c, exact_ref = results[r][1][i]
            good &= all(same.values()) and c > 0.9999
            # padded all-gather delivers world*max rows; the exact-size one sum(counts)
            good &= rows["0"]["ag"] == WORLD * max(pat) and rows["force"]["agv"] == sum(pat)
            good &= (rows["ag"]["agv"] == sum(pat)) == uneven and (rows["1"]["rv"] == sum(pat)) == uneven
        _p, same, rows, c, _e = results[0][1][i]
        print(f"counts {pat}: bit-identical to padded on all ranks "
              f"{ {m: all(results[r][1][i][1][m] for r in range(WORLD)) for m in MODES} }, "
              f"padded == fp32 row reference {all(results[r][1][i][4] for r in range(WORLD))}, all-gather rows "
              f"padded {rows['0']['ag']} -> exact {sum(pat)} ({'used' if uneven else 'not used: padded is cheaper'} "
              f"by ag/1); b12x experts min cos vs full {min(results[r][1][i][3] for r in range(WORLD)):.6f}  "
              f"{'OK' if good else 'MISMATCH'}")
        ok &= good
    print("ALL OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


def test_ep_varlen():
    import subprocess
    assert subprocess.call([sys.executable, os.path.abspath(__file__)]) == 0


if __name__ == "__main__":
    main()
