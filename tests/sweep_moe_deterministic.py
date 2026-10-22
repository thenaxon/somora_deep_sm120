#!/usr/bin/env python3
"""Shape sweep: b12x deterministic MoE (SM120_MOE_DETERMINISTIC=1) against the default b12x path.

    PYTHONPATH=/opt/sm120/sm120:/opt/deepinfer python3 /opt/sm120/tests/sweep_moe_deterministic.py \\
        [--experts 12,16,96] [--tokens 1,2,...] [--json out.json]

V4.1 expert dimensions (K=5120, n=2048, top-6, swiglu limit 10), random FP4 weights. For every E, T and routing
pattern the same tokens go through two B12xRoutedExperts built from the same weights: one in deterministic mode, one
default, and through DeepInfer's exact torch reference (deep_gemm._moe_ref, W4A8 like b12x). Checks: the
deterministic path is no less accurate than the default one (1 - cos(det, exact) <= 2 * (1 - cos(default, exact))
+ 1e-6), deterministic repeats bit-identical, exact zeros for rows without any route, finite output. Also reported:
cos(det, default) and the default path's own run-to-run noise (two default calls). All shapes run back to back on the same two
objects, so stale scratch from an earlier shape would show up too.

Routing patterns:
  unmasked  every route local
  rand60    60 % of the routes masked (-1), the first T/8 rows fully masked
  ep4       rank 0 of 4 expert-parallel ranks: only experts [0, E/4) are local, the rest -1, as with --dp 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "sm120"))
sys.path.insert(0, HERE)
os.environ.setdefault("B12X_AUTOTUNE", "0")

TOKENS = ("1,2,3,4,5,6,7,8,9,12,15,16,17,24,31,32,33,48,63,64,65,96,100,127,128,129,192,255,256,257,384,511,512,513,"
          "640,768,1000,1023,1024,1025,1536,2047,2048,2049,2100")


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def routes(T, E, pattern, seed, dev):
    import test_ep_moe as TE

    TE.E = E
    x, ids, w = TE.make_tokens(T, seed, dev)
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    if pattern == "rand60":
        ids[(torch.rand(ids.shape, generator=g) < 0.6).to(dev)] = -1
        ids[: T // 8] = -1
    elif pattern == "ep4":
        ids[ids >= E // 4] = -1  # local experts 0 .. E/4-1 keep their (local = global) index
    return x, ids, w


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experts", default="12,16,96")
    ap.add_argument("--tokens", default=TOKENS)
    ap.add_argument("--patterns", default="unmasked,rand60,ep4")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--json")
    args = ap.parse_args()
    import sm120_moe
    import test_ep_moe as TE

    dev = torch.device("cuda", 0)
    os.environ["B12X_W4A8_TINY_DECODE"] = os.environ.get("B12X_W4A8_TINY_DECODE", "1")
    rows, fails = [], 0
    t0 = time.time()
    for E in [int(x) for x in args.experts.split(",")]:
        TE.E, TE.K, TE.N, TE.TOPK = E, 5120, 2048, 6
        wts = TE.make_weights(E)

        def build():
            m = TE.make_moe(wts, range(0, E), dev)
            m.swiglu_clamp_value = 10.0
            m.use_fp8_act = True
            return sm120_moe.B12xRoutedExperts(m)

        exact_moe = TE.make_moe(wts, range(0, E), dev)
        exact_moe.swiglu_clamp_value, exact_moe.use_fp8_act = 10.0, True

        os.environ["SM120_MOE_DETERMINISTIC"] = "1"
        det = build()
        os.environ["SM120_MOE_DETERMINISTIC"] = "0"
        ref = build()
        del wts
        from deep_gemm._moe_ref import moe_reference

        def exact(x, ids, w):
            m = exact_moe
            return moe_reference(x, ids, w, m.gate_up_proj_weight, m.gate_up_proj_weight.sf, m.down_proj_weight,
                                 m.down_proj_weight.sf, m.swiglu_clamp_value, True)

        for T in [int(x) for x in args.tokens.split(",")]:
            for pattern in args.patterns.split(","):
                x, ids, w = routes(T, E, pattern, T * 1000 + E, dev)
                os.environ["SM120_MOE_DETERMINISTIC"] = "0"
                r = ref(x, ids, w)
                r2 = ref(x, ids, w).clone()
                ex = exact(x, ids, w)
                os.environ["SM120_MOE_DETERMINISTIC"] = "1"
                outs = [det(x, ids, w).clone() for _ in range(args.repeats)]
                os.environ["SM120_MOE_DETERMINISTIC"] = "0"
                torch.cuda.synchronize()
                none = (ids < 0).all(-1)
                def ccos(a, b):
                    return cos(a, b) if b.float().norm() > 0 else (1.0 if a.float().norm() == 0 else 0.0)
                c = ccos(outs[0], r)
                c_det, c_def, noise = ccos(outs[0], ex), ccos(r, ex), ccos(r2, r)
                rep = all(torch.equal(outs[0].view(torch.int16), o.view(torch.int16)) for o in outs[1:])
                zero = bool((outs[0][none] == 0).all())
                finite = bool(torch.isfinite(outs[0]).all())
                accurate = (1 - c_det) <= 2 * (1 - c_def) + 1e-6
                ok = accurate and rep and zero and finite
                fails += not ok
                rows.append(dict(E=E, T=T, pattern=pattern, cos=c, cos_det_exact=c_det, cos_default_exact=c_def,
                                 default_noise=noise, repeat_identical=rep, empty_rows_zero=zero,
                                 empty_rows=int(none.sum()), finite=finite, ok=ok))
                if not ok:
                    print(f"FAIL E={E} T={T} {pattern}: cos det/exact {c_det:.6f} default/exact {c_def:.6f} "
                          f"det/default {c:.6f} repeat {rep} zero-rows {zero} finite {finite}", flush=True)
        n = sum(1 for r in rows if r["E"] == E)
        sel = [r for r in rows if r["E"] == E]
        print(f"E={E}: {sum(r['ok'] for r in sel)}/{n} ok | vs exact: det worst cos {min(r['cos_det_exact'] for r in sel):.7f}, "
              f"default worst {min(r['cos_default_exact'] for r in sel):.7f} | det vs default worst "
              f"{min(r['cos'] for r in sel):.7f} | default run-to-run worst {min(r['default_noise'] for r in sel):.7f}",
              flush=True)
        del det, ref
        torch.cuda.empty_cache()
    os.environ.pop("SM120_MOE_DETERMINISTIC", None)
    print(f"{len(rows) - fails}/{len(rows)} ok in {time.time() - t0:.0f}s")
    if args.json:
        json.dump(rows, open(args.json, "w"), indent=1)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
