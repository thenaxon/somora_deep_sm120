"""Micro-benchmark of the SM120 attention / indexer / top-k shims: torch reference vs b12x.

    PYTHONPATH=/opt/sm120/sm120:/opt/deepinfer python3 /opt/sm120/tests/bench_attn.py
    ... bench_attn.py --only prefill,decode --prefill 327,2048,8192 --decode 1,8,32 --torch-max-prefill 2048

Shapes follow DeepSeek V4.1 (64 heads x 512 = 448 nope + 64 rope, SWA window 128, top-k 512 over
the compressed cache, indexer 32 heads x 128 MXFP4). Prefill KV is DeepInfer-shaped
[SWA cached | current SWA | compressed] (FP8-exact / FP4-exact tokens). Decode reads paged 528 B
SWA + 288 B compressed caches. "graph" rows replay a CUDA graph of the same call (b12x only, and
torch where capture-safe). Times are medians of CUDA-event timings in milliseconds.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("B12X_AUTOTUNE", "0")

import test_attn_shims as T  # noqa: E402  (shared case builders)

DEV = T.DEV


def timeit(fn, iters=20, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def time_graph(fn, iters=50):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        fn()
    return timeit(g.replay, iters=iters)


def env(**kv):
    return T.EnvSet(**{k: str(v) for k, v in kv.items()})


ROWS = []


def report(group, shape, impl, ms, ref_ms=None):
    sp = f"{ref_ms / ms:7.1f}x" if (ref_ms and ms) else ""
    ROWS.append((group, shape, impl, ms, sp))
    print(f"{group:<22} {shape:<34} {impl:<14} {ms:10.3f} ms {sp}", flush=True)


def safe(fn):
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        print(f"    !! {type(exc).__name__}: {str(exc)[:160]}", flush=True)
        return None


# ---------------------------------------------------------------------------------------------
def bench_prefill(sizes, torch_max):
    import flash_mla
    for s_q in sizes:
        c = T.build_prefill_case(seed=7, s_q=s_q, ratio=4, holes=0.0)
        width = c["idx"].shape[-1]
        shape = f"s_q={s_q} H=64 w={width} kv={c['kv'].shape[0]}"

        def run(impl, c=c):
            with env(SM120_ATTN_PREFILL_IMPL=impl, SM120_ATTN_STRICT=int(impl == "b12x")):
                return flash_mla.flash_mla_sparse_fwd(c["q"], c["kv"], c["idx"], c["sm_scale"], 512, c["sink"], None)

        t_ref = None
        if s_q <= torch_max:
            t_ref = safe(lambda: timeit(lambda: run("torch"), iters=3, warmup=1))
            if t_ref:
                report("flash_mla_sparse_fwd", shape, "torch", t_ref)
        t = safe(lambda: timeit(lambda: run("b12x"), iters=10))
        if t:
            report("flash_mla_sparse_fwd", shape, "b12x(bf16)", t, t_ref)
        with env(SM120_ATTN_COMPUTE="fp8"):
            t8 = safe(lambda: timeit(lambda: run("b12x"), iters=10))
        if t8:
            report("flash_mla_sparse_fwd", shape, "b12x(fp8)", t8, t_ref)
        pos = torch.randint(0, 4096, (s_q,), device=DEV).to(torch.int32)
        cs = T.cos_sin_cache()
        qf = T.F_.q_to_fused_layout(c["q"])

        def fused(impl, c=c, pos=pos, qf=qf):
            with env(SM120_ATTN_PREFILL_IMPL=impl, SM120_ATTN_STRICT=int(impl == "b12x")):
                return T.flash_mla.fused_norm_rope_attn_rope_cast.prefill(
                    True, 1e-6, pos, False, 64, cs, 8, 32, True, True, True, qf, c["kv"], c["idx"], c["sm_scale"],
                    512, c["sink"], None)

        f_ref = None
        if s_q <= torch_max:
            f_ref = safe(lambda: timeit(lambda: fused("torch"), iters=3, warmup=1))
            if f_ref:
                report("fused.prefill", shape, "torch", f_ref)
        f = safe(lambda: timeit(lambda: fused("b12x"), iters=10))
        if f:
            report("fused.prefill", shape, "b12x(bf16)", f, f_ref)


def bench_decode(batches):
    import flash_mla
    for b in batches:
        c = T.build_decode_case(seed=5, rows=b, invalid_frac=0.0)
        shape = f"batch={b} H=64 swa128+topk512"

        def run(c=c):
            sm, _ = flash_mla.get_mla_metadata()
            return flash_mla.flash_mla_with_kvcache(
                q=c["q"].unsqueeze(1), k_cache=c["swa"], block_table=None, cache_seqlens=None, head_dim_v=512,
                tile_scheduler_metadata=sm, softmax_scale=c["sm_scale"], causal=False, is_fp8_kvcache=True,
                indices=c["swa_idx"].unsqueeze(1), attn_sink=c["sink"], extra_k_cache=c["main"],
                extra_indices_in_kvcache=c["main_idx"].unsqueeze(1), topk_length=None,
                extra_topk_length=c["main_len"])

        with env(SM120_ATTN_DECODE_IMPL="torch"):
            t_ref = safe(lambda: timeit(run))
            if t_ref:
                report("flash_mla_with_kvcache", shape, "torch", t_ref)
            g_ref = safe(lambda: time_graph(run))
            if g_ref:
                report("flash_mla_with_kvcache", shape, "torch graph", g_ref)
        for mode in ("bf16", "fp8"):
            with env(SM120_ATTN_DECODE_IMPL="b12x", SM120_ATTN_COMPUTE=mode, SM120_ATTN_STRICT=1):
                t = safe(lambda: timeit(run))
                if t:
                    report("flash_mla_with_kvcache", shape, f"b12x({mode})", t, t_ref)
                g = safe(lambda: time_graph(run))
                if g:
                    report("flash_mla_with_kvcache", shape, f"b12x({mode}) graph", g, t_ref)


def bench_indexer(batches, prefill_sizes, ctx, torch_max):
    dg = T.DG
    for b in batches:
        c = T.build_index_case(seed=9, Hi=32, nq=b, nkv=ctx)
        ke = torch.full((b,), ctx, dtype=torch.int32, device=DEV)
        lens = ke.view(b, 1).contiguous()
        ar = torch.arange(b, dtype=torch.int32, device=DEV)
        meta = dg["get_paged_mqa_logits_metadata"](lens, c["page"], 8, indices=ar)
        qp = (c["q"][0].unsqueeze(1), c["q"][1].unsqueeze(1))
        shape = f"batch={b} H=32 ctx={ctx}"

        def run(c=c, lens=lens, ar=ar, meta=meta, qp=qp):
            return dg["fp8_fp4_paged_mqa_logits"](qp, c["cache"], c["w"], lens, c["bt"], meta, ctx, indices=ar)

        def run_topk(run=run, lens=lens):
            return T.ds_topk(run(), 512, end=lens.view(-1), indices_type=torch.int32, sorted_index=True,
                             idx_oob_fill_value=-1, return_value=False)

        with env(SM120_INDEXER_IMPL="torch"):
            t_ref = safe(lambda: timeit(run))
            if t_ref:
                report("paged_mqa_logits", shape, "torch", t_ref)
        with env(SM120_INDEXER_IMPL="b12x", SM120_ATTN_STRICT=1):
            t = safe(lambda: timeit(run))
            if t:
                report("paged_mqa_logits", shape, "b12x", t, t_ref)
            g = safe(lambda: time_graph(run_topk))
            if g:
                report("paged logits+topk", shape, "b12x graph", g)
        with env(SM120_INDEXER_IMPL="torch"):
            tk = safe(lambda: timeit(run_topk))
            if tk:
                report("paged logits+topk", shape, "torch", tk)
    for s_q in prefill_sizes:
        nkv = max(64, s_q // 4)
        c = T.build_index_case(seed=11, Hi=32, nq=s_q, nkv=nkv)
        ks = torch.zeros((s_q,), dtype=torch.int32, device=DEV)
        ke = ((torch.arange(s_q, device=DEV) + 1) // 4).clamp(min=1).to(torch.int32)
        W = nkv
        shape = f"s_q={s_q} H=32 kv={nkv}"

        def run(c=c, ks=ks, ke=ke, W=W):
            return dg["fp8_fp4_mqa_logits"](c["q"], c["k"], c["w"], ks, ke, max_seqlen_k=W)

        t_ref = None
        if s_q <= torch_max:
            with env(SM120_INDEXER_IMPL="torch"):
                t_ref = safe(lambda: timeit(run, iters=3, warmup=1))
                if t_ref:
                    report("mqa_logits (prefill)", shape, "torch", t_ref)
        with env(SM120_INDEXER_IMPL="b12x", SM120_ATTN_STRICT=1):
            t = safe(lambda: timeit(run, iters=10))
            if t:
                report("mqa_logits (prefill)", shape, "b12x", t, t_ref)


def bench_topk(shapes):
    from b12x.attention.dsa_indexer.tiled_topk import run_row_topk
    for rows, W, k in shapes:
        x = torch.randn((rows, W), device=DEV).to(torch.bfloat16)
        end = torch.randint(W // 2, W + 1, (rows,), device=DEV).to(torch.int32)
        shape = f"rows={rows} W={W} k={k}"

        def shim(x=x, end=end, k=k):
            return T.ds_topk(x, k, end=end, indices_type=torch.int32, sorted_index=True, idx_oob_fill_value=-1,
                             return_value=False)

        def raw(x=x, k=k):
            return torch.topk(x, k, dim=1, sorted=False)

        def b12x(x=x, end=end, k=k):
            return run_row_topk(row_logits=x.float(), lengths=end, topk=k, write_values=False)

        with env(SM120_TOPK_IMPL="torch"):
            t_ref = safe(lambda: timeit(shim))
        if t_ref:
            report("deep_select.topk", shape, "shim torch", t_ref)
        t_raw = safe(lambda: timeit(raw))
        if t_raw:
            report("deep_select.topk", shape, "torch.topk only", t_raw, t_ref)
        t_b = safe(lambda: timeit(b12x))
        if t_b:
            report("deep_select.topk", shape, "b12x row_topk", t_b, t_ref)
        with env(SM120_TOPK_IMPL="triton"):
            t_s = safe(lambda: timeit(shim))
        if t_s:
            report("deep_select.topk", shape, "shim triton", t_s, t_ref)


def bench_mhc_einsum(batches, prefill):
    import deep_gemm
    import test_mhc_einsum as M
    gen = torch.Generator(device=DEV).manual_seed(0)
    for T in list(batches) + list(prefill):
        x = torch.randn((T, 4 * 5120), device=DEV).to(torch.bfloat16)
        fn = torch.randn((24, 4 * 5120), device=DEV) * 0.02
        dd = torch.empty((16, T, 24), device=DEV)
        sq = torch.empty((16, T), device=DEV)
        a, sfa = M._fp8_operand((T, 8), 4096, gen, colmajor_rows=False)
        sfa = M._a_colmajor(sfa)
        w, sfw = M._fp8_operand((8, 1024), 4096, gen, colmajor_rows=True)
        d8 = torch.empty((T, 8, 1024), dtype=torch.float8_e4m3fn, device=DEV)
        dsf = torch.empty_strided((T, 64), (1, (T + 3) // 4 * 4), dtype=torch.int32, device=DEV)

        def hc():
            deep_gemm.tf32_hc_prenorm_gemm(x, fn, dd, sq, 16)

        def ein():
            deep_gemm.fp8_einsum("bhr,hdr->bhd", (a, sfa), (w, sfw), (d8, dsf), recipe=(1, 1, 32))

        for name, f, var in (("tf32_hc_prenorm_gemm", hc, "SM120_MHC_IMPL"), ("fp8_einsum wo_a", ein, "SM120_EINSUM_IMPL")):
            shape = f"T={T}"
            with env(**{var: "torch"}):
                t_ref = safe(lambda: timeit(f))
            if t_ref:
                report(name, shape, "torch", t_ref)
            with env(**{var: "triton"}):
                t = safe(lambda: timeit(f))
                g = safe(lambda: time_graph(f))
            if t:
                report(name, shape, "triton", t, t_ref)
            if g:
                report(name, shape, "triton graph", g, t_ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="prefill,decode,indexer,topk,mhc")
    ap.add_argument("--topk", default="1x65536x512,32x65536x512,32x262144x512,2048x16384x512,4096x65536x512,"
                                      "2048x4096x512,8x8192x1024")
    ap.add_argument("--prefill", default="327,2048,8192")
    ap.add_argument("--decode", default="1,8,32")
    ap.add_argument("--ctx", type=int, default=8192, help="indexer decode context (compressed tokens)")
    ap.add_argument("--torch-max-prefill", type=int, default=8192,
                    help="skip the (slow) torch reference above this many prefill rows")
    a = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA required")
    print(f"device: {torch.cuda.get_device_name()}  torch {torch.__version__}  b12x "
          f"{'yes' if T.b12x_available() else 'NO'} / dsa_indexer {'yes' if T.b12x_indexer_available() else 'NO'}")
    print(f"{'group':<22} {'shape':<34} {'impl':<14} {'median':>13} speedup-vs-torch")
    only = set(a.only.split(","))
    pre = [int(x) for x in a.prefill.split(",") if x]
    dec = [int(x) for x in a.decode.split(",") if x]
    with torch.inference_mode():
        if "prefill" in only:
            bench_prefill(pre, a.torch_max_prefill)
        if "decode" in only:
            bench_decode(dec)
        if "indexer" in only:
            bench_indexer(dec, pre, a.ctx, a.torch_max_prefill)
        if "mhc" in only:
            bench_mhc_einsum(dec, pre)
        if "topk" in only:
            bench_topk([tuple(int(v) for v in t.split("x")) for t in a.topk.split(",") if t])


if __name__ == "__main__":
    main()
