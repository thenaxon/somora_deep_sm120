#!/usr/bin/env python3
"""Test (and optionally time) b12x fused_moe w4a8_mx tiny-decode with swiglu_limit.

Checks patches/b12x-tiny-decode-swiglu.patch on random MXFP4 experts (no checkpoint needed):
for T = 1..4 the default path (B12X_W4A8_TINY_DECODE=1 -> "micro"/tiny-decode) must match the fp32
reference WITH the SwiGLU clamp, and agree with the dynamic path (B12X_W4A8_TINY_DECODE=0). A run
without swiglu_limit checks that the unclamped kernel is unchanged. Exit code 0 = all PASS.

    python3 test_b12x_tiny_decode_swiglu.py                    # small shape, correctness only
    python3 test_b12x_tiny_decode_swiglu.py --shape v41 --bench
            # V4.1 expert geometry (E=384, K=5120, n=2304, top-6; verify n against the checkpoint config)
            # --bench: CUDA-graph replay latency of one MoE call, tiny-decode vs dynamic, T = 1..4

On an unpatched b12x the tiny-decode cases FAIL ("clamp not applied"), which is the bug repro.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F

DEV = torch.device("cuda", 0)  # b12x scratch checks want an explicit device index
FP4 = None
SHAPES = {
    "small": dict(E=8, K=2048, N=512, TOPK=2),
    "v41": dict(E=384, K=5120, N=2304, TOPK=6),
}


def dequant(packed, scale_u8):
    """uint8 [R, C/2] (low nibble = even column) + UE8M0 [R, C/32] -> fp32 [R, C]."""
    r, half = packed.shape
    vals = torch.stack((FP4[(packed & 0xF).long()], FP4[(packed >> 4).long()]), -1).reshape(r, 2 * half)
    sc = torch.exp2(scale_u8.float() - 127.0)
    return (vals.view(r, -1, 32) * sc.unsqueeze(-1)).reshape(r, -1)


def reference(x, ids, w, w13, s13, w2, s2, N, limit):
    out = torch.zeros(x.shape[0], x.shape[1], device=DEV)
    hits = tot = 0
    for t in range(x.shape[0]):
        for j in range(ids.shape[1]):
            e = int(ids[t, j])
            h = x[t].float() @ dequant(w13[e], s13[e]).T
            g, u = h[:N], h[N:]
            if limit is not None:
                hits += int((g > limit).sum() + (u.abs() > limit).sum())
                tot += 2 * N
                g, u = g.clamp(max=limit), u.clamp(-limit, limit)
            out[t] += w[t, j] * ((F.silu(g) * u) @ dequant(w2[e], s2[e]).T)
    return out, (hits / tot if tot else 0.0)


class Runner:
    def __init__(self, fm, tiny: str, shape: dict, weights, limit, T: int):
        os.environ["B12X_W4A8_TINY_DECODE"] = tiny  # read by b12x at plan time
        E, K, N = shape["E"], shape["K"], shape["N"]
        w13, s13, w2, s2 = weights
        wplan = fm.plan_weights(quant_modes="w4a8_mx", source_format="fp4_e8m0_k32", activation="silu",
                                params_dtype=torch.bfloat16, num_experts=E, hidden_size=K,
                                intermediate_size=N, w13_layout="w31")
        ones = torch.ones(E, device=DEV)
        self.experts = fm.prepare_weights(plan=wplan, params_dtype=torch.bfloat16,
                                          w1_fp4=w13.clone(), w2_fp4=w2.clone(),  # repacked in place
                                          w1_blockscale=s13.clone(), w2_blockscale=s2.clone(),
                                          w1_global_scale=ones, w2_global_scale=ones,
                                          a1_gscale=ones, a2_gscale=ones)
        self.caps = fm.Caps(max_tokens=T, num_topk=shape["TOPK"], device=DEV, weight_plan=wplan,
                            quant_mode="w4a8_mx", swiglu_limit=limit)
        self.plan = fm.plan(self.caps)
        self.scratch = tuple(torch.zeros(s.shape, dtype=s.dtype, device=s.device) for s in self.plan.scratch_specs())
        self.fm = fm
        self.backend = self.caps.decode_config.backend
        self.impl = self.plan.launch_plan.implementation

    def bind(self, x, ids, w, out):
        return self.fm.bind(self.plan, scratch=self.scratch, a=x, experts=self.experts, topk_weights=w,
                            topk_ids=ids, output=out, input_scales_static=True)

    def run(self, x, ids, w):
        out = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bfloat16, device=DEV)
        self.fm.run(binding=self.bind(x, ids, w, out))
        torch.cuda.synchronize()
        return out.float()

    def graph_latency_us(self, x, ids, w, iters: int) -> float:
        out = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bfloat16, device=DEV)
        b = self.bind(x, ids, w, out)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            self.fm.run(binding=b)  # warm-up / JIT outside capture
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            self.fm.run(binding=b)
        g.replay()
        torch.cuda.synchronize()
        t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(iters):
            g.replay()
        t1.record()
        torch.cuda.synchronize()
        return t0.elapsed_time(t1) * 1000.0 / iters


def cos(a, b):
    return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def main() -> int:
    global FP4
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape", choices=sorted(SHAPES), default="small")
    ap.add_argument("--limit", type=float, default=10.0)
    ap.add_argument("--x-std", type=float, default=8.0, help="activation std (8 clamps ~1/3 of gate/up values)")
    ap.add_argument("--tokens", default="1,2,3,4")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    import b12x
    from b12x.moe._shared.kernels import tiny_decode
    from b12x.moe import fused_moe as fm

    FP4 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], device=DEV)
    patched = bool(getattr(tiny_decode, "SWIGLU_LIMIT_SUPPORTED", False))
    shape = SHAPES[args.shape]
    E, K, N, TOPK = shape["E"], shape["K"], shape["N"], shape["TOPK"]
    print(f"b12x {getattr(b12x, '__version__', '?')} from {b12x.__file__}; tiny-decode swiglu patch "
          f"{'PRESENT' if patched else 'ABSENT'}; {torch.cuda.get_device_name()}; shape {args.shape} {shape}")

    torch.manual_seed(0)
    # MXFP4 weights: random nibbles, UE8M0 scale 2^-7 (byte 120) like real checkpoints (119..122)
    w13 = torch.randint(0, 256, (E, 2 * N, K // 2), dtype=torch.uint8, device=DEV)
    w2 = torch.randint(0, 256, (E, K, N // 2), dtype=torch.uint8, device=DEV)
    s13 = torch.full((E, 2 * N, K // 32), 120, dtype=torch.uint8, device=DEV)
    s2 = torch.full((E, K, N // 32), 120, dtype=torch.uint8, device=DEV)
    weights = (w13, s13, w2, s2)

    ok = True
    bench_rows = []
    for T in [int(t) for t in args.tokens.split(",")]:
        x = (torch.randn(T, K, device=DEV) * args.x_std).to(torch.bfloat16)
        ids = torch.stack([torch.randperm(E, device=DEV)[:TOPK] for _ in range(T)]).to(torch.int32)
        if T > 1:
            ids[-1, -1] = -1  # one masked route (graph padding / non-local EP expert)
        w = torch.softmax(torch.randn(T, TOPK, device=DEV), -1)
        for limit in (args.limit, None):
            ref, frac = reference(x, ids.clamp(min=0), torch.where(ids >= 0, w, 0.0), w13, s13, w2, s2, N, limit)
            outs = {}
            for tiny in ("1", "0"):
                r = Runner(fm, tiny, shape, weights, limit, T)
                out = r.run(x, ids, w)
                outs[tiny] = out
                c = cos(out, ref)
                good = c > 0.99 and bool(torch.isfinite(out).all())
                tag = f"T={T} limit={limit} B12X_W4A8_TINY_DECODE={tiny} backend={r.backend}/{r.impl}"
                if tiny == "1" and r.backend != "micro":
                    print(f"INFO {tag}: tiny-decode not selected for this shape; check inconclusive")
                note = "" if good else ("  <- clamp not applied" if limit is not None and tiny == "1" else "")
                print(f"{'PASS' if good else 'FAIL'} {tag}: cos vs fp32 ref = {c:.5f}"
                      + (f" (clamped fraction {frac:.1%})" if limit is not None else "") + note)
                ok &= good
                if args.bench and limit is not None:
                    bench_rows.append((T, tiny, r.backend, r.graph_latency_us(x, ids, w, args.iters)))
                del r
            c = cos(outs["1"], outs["0"])
            good = c > 0.995
            ok &= good
            print(f"{'PASS' if good else 'FAIL'} T={T} limit={limit}: tiny vs dynamic cos = {c:.5f}")

    if bench_rows:
        print("\nCUDA-graph replay latency per MoE call (us):")
        print(f"{'T':>3} {'tiny-decode':>14} {'dynamic':>10} {'speedup':>8}")
        by = {(t, k): (b, us) for t, k, b, us in bench_rows}
        for T in sorted({t for t, *_ in bench_rows}):
            tu, du = by.get((T, "1"), (None, float("nan")))[1], by.get((T, "0"), (None, float("nan")))[1]
            print(f"{T:>3} {tu:>14.1f} {du:>10.1f} {du / tu:>7.2f}x")
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
