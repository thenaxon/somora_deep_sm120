#!/usr/bin/env python3
"""Compare the torch fallbacks (ref_kernels.py) with the official tilelang kernels on random inputs.

    python3 kernel_selftest.py [--ref-src .../ds41-inference/inference]

Run on the target GPU inside the container. Expected: act_quant / fp4_act_quant bit-exact,
hc_split_sinkhorn within ~1e-6, GEMMs/sparse_attn within bf16 rounding (rare 1-ulp differences from
fp32 summation order). Any tilelang exception is printed (that function would fall back to torch).
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-src", default=os.environ.get("REF_SRC", os.path.join(HERE, "..", "..", "third_party", "ds41-inference", "inference")))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    os.environ["REF_KERNELS"] = "tilelang"
    import ref_kernels as rk

    disp = rk.install(os.path.abspath(args.ref_src))
    tl = disp.tl
    if tl is None:
        print(f"tilelang kernel.py could not be imported: {disp.tl_import_error}")
        return 2
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.bfloat16)
    dev = "cuda"
    results = []

    def cmp(name, a, b, tol):
        a = a if isinstance(a, tuple) else (a,)
        b = b if isinstance(b, tuple) else (b,)
        worst = 0.0
        exact = True
        for x, y in zip(a, b):
            xf, yf = rk._to_cmp(x), rk._to_cmp(y)
            d = (xf - yf).abs()
            d = torch.where(torch.isnan(xf) & torch.isnan(yf), torch.zeros_like(d), d)
            m = float(d.max()) if d.numel() else 0.0
            rel = m / max(float(xf.abs().max()), 1e-30)
            worst = max(worst, rel)
            exact &= m == 0.0
        ok = worst <= tol
        results.append((name, ok, exact, worst))
        print(f"{'OK  ' if ok else 'FAIL'} {name:40s} bitexact={exact} max_rel={worst:.3e} (tol {tol:g})")

    def run(name, fn_t, fn_tl, tol):
        try:
            out_tl = fn_tl()
            torch.cuda.synchronize()
        except Exception:
            print(f"EXC  {name}: tilelang raised\n{traceback.format_exc(limit=3)}")
            results.append((name, False, False, float("nan")))
            return
        cmp(name, fn_t(), out_tl, tol)

    e8 = torch.float8_e8m0fnu
    # act_quant (linear path, non-inplace, ue8m0 scales) and inplace (kv / o paths)
    for shape in ((37, 5120), (1, 4096), (300, 2304)):
        x = (torch.randn(shape, device=dev) * 3).bfloat16()
        x[0, :32] = 0  # all-zero block -> amax floor
        run(f"act_quant{shape}", lambda: rk.act_quant_torch(x, 32, "ue8m0", e8), lambda: tl.act_quant(x, 32, "ue8m0", e8), 0)
        run(
            f"act_quant_inplace{shape}",
            lambda: rk.act_quant_torch(x.clone(), 32, "ue8m0", e8, True),
            lambda: tl.act_quant(x.clone(), 32, "ue8m0", e8, True),
            0,
        )
    # fp4_act_quant: indexer (E8M0, 32) and compressed KV (E4M3, 16), inplace like model.py
    q = (torch.randn(5, 7, 32, 128, device=dev) * 2).bfloat16()
    run("fp4_act_quant_e8m0_inplace", lambda: rk.fp4_act_quant_torch(q.clone(), 32, True), lambda: tl.fp4_act_quant(q.clone(), 32, True), 0)
    lat = (torch.randn(1, 150, 512, device=dev)).bfloat16()
    lat[0, 0, :16] = 0
    run(
        "fp4_act_quant_e4m3_inplace",
        lambda: rk.fp4_act_quant_torch(lat.clone(), 16, True, scale_dtype=torch.float8_e4m3fn),
        lambda: tl.fp4_act_quant(lat.clone(), 16, True, scale_dtype=torch.float8_e4m3fn),
        0,
    )
    # fp8_gemm with block-32 E8M0 weight scales
    for M, N, K in ((37, 1280, 5120), (1, 32768, 1280), (300, 5120, 8192)):
        xa = torch.randn(M, K, device=dev).bfloat16()
        a, sa = rk.act_quant_torch(xa, 32, "ue8m0", e8)
        b = (torch.randn(N, K, device=dev) * 100).clamp(-448, 448).to(torch.float8_e4m3fn)
        sb = torch.randint(110, 125, ((N + 31) // 32, K // 32), device=dev, dtype=torch.uint8).view(e8)
        run(f"fp8_gemm M{M} N{N} K{K}", lambda: rk.fp8_gemm_torch(a, sa, b, sb, e8, 32), lambda: tl.fp8_gemm(a, sa, b, sb, e8, 32), 1e-2)
    # fp4_gemm (routed experts)
    for M, N, K in ((5, 2304, 5120), (64, 5120, 2304)):
        xa = torch.randn(M, K, device=dev).bfloat16()
        a, sa = rk.act_quant_torch(xa, 32, "ue8m0", e8)
        b = torch.randint(0, 256, (N, K // 2), device=dev, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
        sb = torch.randint(115, 125, (N, K // 32), device=dev, dtype=torch.uint8).view(e8)
        run(f"fp4_gemm M{M} N{N} K{K}", lambda: rk.fp4_gemm_torch(a, sa, b, sb, e8, 32), lambda: tl.fp4_gemm(a, sa, b, sb, e8, act_block_size=32), 1e-2)
    # sparse_attn: prefill-like (window + compressed, with -1 holes) and decode-like
    for m, n, topk in ((37, 200, 160), (1, 300, 309)):
        qq = torch.randn(1, m, 64, 512, device=dev).bfloat16()
        kv = torch.randn(1, n, 512, device=dev).bfloat16()
        idx = torch.randint(0, n, (1, m, topk), device=dev, dtype=torch.int32)
        idx[:, :, ::7] = -1
        sink = torch.randn(64, device=dev, dtype=torch.float32)
        run(f"sparse_attn m{m} n{n} topk{topk}", lambda: rk.sparse_attn_torch(qq, kv, sink, idx, 512**-0.5), lambda: tl.sparse_attn(qq, kv, sink, idx, 512**-0.5), 2e-2)
    # all-invalid rows must give zeros, not NaN
    idx = torch.full((1, 3, 64), -1, device=dev, dtype=torch.int32)
    qq = torch.randn(1, 3, 64, 512, device=dev).bfloat16()
    kv = torch.randn(1, 10, 512, device=dev).bfloat16()
    sink = torch.zeros(64, device=dev, dtype=torch.float32)
    run("sparse_attn all-invalid", lambda: rk.sparse_attn_torch(qq, kv, sink, idx, 0.05), lambda: tl.sparse_attn(qq, kv, sink, idx, 0.05), 0)
    # hc_split_sinkhorn
    mixes = torch.randn(1, 37, 24, device=dev, dtype=torch.float32)
    sc = torch.rand(3, device=dev, dtype=torch.float32)
    base = torch.randn(24, device=dev, dtype=torch.float32)
    run("hc_split_sinkhorn", lambda: rk.hc_split_sinkhorn_torch(mixes, sc, base, 4, 20, 1e-6), lambda: tl.hc_split_sinkhorn(mixes, sc, base, 4, 20, 1e-6), 1e-5)

    bad = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(bad)}/{len(results)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
