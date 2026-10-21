#!/usr/bin/env python3
"""FP8 (weight-only) lm_head vs bf16 on real activations: logit cos / top-1 / top-5 / KL, plus kernel timing.

    python3 lmhead_eval.py --ckpt /work/ckpt_full --dumps /work/full_short_di.pt,/work/full_long_di.pt

Hidden states: the dump's hidden["prefill"]["final_norm"] (the head input). Head weight: <ckpt>/head.weight.
The bf16 reference is the deep_gemm shim's bf16_gemm_nt with fp32 output (what get_logits runs); it is checked
against the dump's own prefill logits when they are stored.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "sm120"))


def load_head(ckpt: str) -> torch.Tensor:
    from safetensors import safe_open
    idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    with safe_open(os.path.join(ckpt, idx["head.weight"]), framework="pt", device="cuda") as f:
        return f.get_tensor("head.weight").to(torch.bfloat16).contiguous()


def timeit(fn, iters=50):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
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
    return a.elapsed_time(b) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/work/ckpt_full")
    ap.add_argument("--dumps", default="/work/full_short_di.pt,/work/full_long_di.pt")
    a = ap.parse_args()
    import deep_gemm
    import sm120_lmhead as LH

    W = load_head(a.ckpt)
    V, K = W.shape
    print(f"head {tuple(W.shape)} bf16 {W.numel() * 2 / 2**30:.2f} GiB")
    w8, sf = LH.quantize_weight(W)
    deq = (w8.float().view(V, -1, 128) * sf.unsqueeze(-1)).view(V, K)
    print(f"weight quantisation: rel L2 err {((deq - W.float()).norm() / W.float().norm()).item():.4e}")
    del deq
    for path in [p for p in a.dumps.split(",") if p]:
        if not os.path.exists(path):
            print(f"{path}: missing")
            continue
        d = torch.load(path, map_location="cpu", weights_only=False)
        hid = ((d.get("hidden") or {}).get("prefill") or {}).get("final_norm")
        if hid is None:
            print(f"{path}: no hidden final_norm (rerun without --no-hidden)")
            continue
        x = hid.reshape(-1, K).to("cuda", torch.bfloat16).contiguous()
        M = x.shape[0]
        ref = torch.empty((M, V), dtype=torch.float32, device="cuda")
        deep_gemm.bf16_gemm_nt(x, W, ref)
        got = torch.empty_like(ref)
        for a0 in range(0, M, 64):
            LH.lmhead_fp8(x[a0:a0 + 64], w8, sf, got[a0:a0 + 64])
        own = d.get("prefill_logits_last64")
        if isinstance(own, torch.Tensor):
            c_own = F.cosine_similarity(own.float().cuda(), ref[-own.shape[0]:], dim=-1)
            print(f"  sanity: recomputed bf16 logits vs dump's own last-64 logits cos min {c_own.min():.6f}")
        cos = F.cosine_similarity(got, ref, dim=-1)
        t1 = (got.argmax(-1) == ref.argmax(-1)).float().mean().item()
        top5 = ref.topk(5, -1).indices
        t5 = torch.tensor([len(set(top5[i].tolist()) & set(got[i].topk(5).indices.tolist())) / 5
                           for i in range(M)]).mean().item()
        kl = F.kl_div(F.log_softmax(got, -1), F.log_softmax(ref, -1), log_target=True, reduction="none").sum(-1)
        dmax = (got - ref).abs().max().item()
        print(f"{os.path.basename(path)}: {M} positions  logits cos mean {cos.mean():.6f} min {cos.min():.6f}  "
              f"top-1 agree {t1:.4f}  top-5 overlap {t5:.4f}  KL(ref||fp8) mean {kl.mean():.3e} max {kl.max():.3e}  "
              f"max|d| {dmax:.3f}")
    print("timing (CUDA graph replay, fp32 logits out):")
    for M in (1, 8, 32, 64):
        x = torch.randn((M, K), device="cuda").to(torch.bfloat16)
        o = torch.empty((M, V), dtype=torch.float32, device="cuda")
        tb = timeit(lambda: deep_gemm.bf16_gemm_nt(x, W, o))
        tf = timeit(lambda: LH.lmhead_fp8(x, w8, sf, o))
        print(f"  M={M:4d}: bf16 (shim/cuBLAS) {tb:.3f} ms   fp8 triton {tf:.3f} ms   {tb / tf:.2f}x")


if __name__ == "__main__":
    main()
