#!/usr/bin/env python3
"""Long-context correctness with CHUNKED prefill (run_deepinfer.py refuses chunking): one request of a tiled
prompt, greedy decode, decode logits taken from sm120_capture's sampler hook. Runs every --variants entry
in its own process (bench_e2e.worker_env), then compares tokens and per-step decode-logit cosine.

    python3 longctx_check.py --ckpt /work/ckpt3 --prompt-len 32768 --variants "b12x:graph,torch:eager"
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.abspath(os.path.join(HERE, "..", "tools"))
sys.path.insert(0, TOOLS)


def worker(a):
    import bench_e2e as BE
    import sm120_patches

    sm120_patches.install()
    for p in (os.path.join(a.deepinfer_root, "examples"), a.deepinfer_root):
        if p not in sys.path:
            sys.path.insert(0, p)
    from _config import create_llm
    import torch

    bargs = BE.parse_args(["--prompt-len", str(a.prompt_len), "--prompt-ids", a.prompt_ids])
    ids = BE.make_prompts(bargs, 1, 0, 0)[0]
    t0 = time.time()
    llm = create_llm(a.ckpt, 0, ep_size=1, cp_size=1, data_parallel_size=1,
                     max_num_batched_tokens=a.chunk, max_model_len=a.max_model_len, max_num_seqs=1,
                     gpu_memory_utilization=a.gpu_memory_utilization, lookahead_tokens=0, disable_log_stats=True)
    init_s = time.time() - t0
    import sm120_capture as cap
    from deepinfer._C.engine import RequestOutput
    from deepinfer.sampling_params import SamplingParams

    eng = llm.llm_engine
    sp = SamplingParams(temperature=0.0, max_tokens=a.gen, ignore_eos=True, top_logprobs=0, seq_seed=0)
    cap.arm(len(ids), keep_full_prefill=False)
    rid = str(llm._add_request(None, sp, ids).request_id)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    t_first = None
    n = 0
    while True:
        unfinished = eng.has_unfinished_requests()
        outs = eng.step() if unfinished else eng.flush_inflight_reqs()
        for raw in outs:
            o = RequestOutput.clone(raw)
            if str(o.request_id) == rid:
                n += 1
                if t_first is None:
                    t_first = time.perf_counter()
        if not unfinished:
            break
    torch.cuda.synchronize()
    t_end = time.perf_counter()
    cap.disarm()
    rows = cap.decode_rows(rid) or cap.decode_rows(None)
    dec = torch.stack([r[0].float() for r in rows]) if rows else torch.empty(0)
    toks = [int(x) for x in dec.argmax(-1).tolist()] if rows else []
    sampled = cap.STATE.get("prefill_sampled") or []
    first = int(sampled[0].float().argmax(-1).reshape(-1)[-1]) if sampled else None
    res = dict(variant=a.variant, prompt_len=len(ids), init_s=init_s, prefill_s=(t_first or t_end) - t0,
               decode_s=t_end - (t_first or t_end), gen=n, first_token=first, tokens=toks,
               peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30, errors=cap.STATE.get("errors", [])[:3])
    torch.save(dict(res, decode_logits=dec), a.out)
    print("[longctx_check] " + json.dumps(res), flush=True)
    llm.close()
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/work/ckpt3")
    ap.add_argument("--prompt-len", type=int, default=16384)
    ap.add_argument("--prompt-ids", default="/work/prompt_ids.json")
    ap.add_argument("--gen", type=int, default=17)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--max-model-len", type=int, default=0, help="0: prompt + 1024 rounded up to 1024")
    ap.add_argument("--max-blocks", type=int, default=0, help="0: 2 * max_model_len / 128 + 64")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    ap.add_argument("--variants", default="b12x:graph,b12x:eager,torch:eager:0.95:2048",
                    help="comma list of <attn impl>:<eager|graph>[:<gpu_memory_utilization>[:<prefill chunk>]]")
    ap.add_argument("--deepinfer-root", default=os.environ.get("DEEPINFER_ROOT", "/opt/deepinfer"))
    ap.add_argument("--variant", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if not a.max_model_len:
        a.max_model_len = max(16384, (a.prompt_len + 1024 + 1023) // 1024 * 1024)
    if not a.max_blocks:
        a.max_blocks = 2 * a.max_model_len // 128 + 64
    if a.variant:
        return worker(a)
    import bench_e2e as BE
    import torch
    import torch.nn.functional as F

    outs = {}
    for v in a.variants.split(","):
        impl, mode, *rest = v.split(":")  # optional 3rd / 4th field: gpu_memory_utilization, prefill chunk
        env = BE.worker_env(BE.parse_args(["--max-blocks", str(a.max_blocks)]), mode, 0, 0)
        impl, *flags = impl.split("+")  # e.g. b12x+idxtorch: single components back to torch
        env["SM120_ATTN_IMPL"] = impl
        for f in flags:
            env[{"idxtorch": "SM120_INDEXER_IMPL", "dectorch": "SM120_ATTN_DECODE_IMPL",
                 "pretorch": "SM120_ATTN_PREFILL_IMPL", "fusedtorch": "SM120_ATTN_FUSED_IMPL",
                 "topktorch": "SM120_TOPK_IMPL"}[f]] = "torch"
        if impl == "torch":
            env.setdefault("SM120_ATTN_CHUNK_MB", "128")  # torch reference working set (KV takes most memory)
        out = f"/work/p3lcc_{a.prompt_len}_{len(outs)}.pt"
        cmd = [sys.executable, os.path.abspath(__file__), *sys.argv[1:], "--variant", v, "--out", out,
               "--max-model-len", str(a.max_model_len), "--max-blocks", str(a.max_blocks)]
        if rest and rest[0]:
            cmd += ["--gpu-memory-utilization", rest[0]]
        if len(rest) > 1:
            cmd += ["--chunk", rest[1]]
        rc = subprocess.call(cmd, env=env)
        if rc == 0 and os.path.exists(out):
            outs[v] = torch.load(out, map_location="cpu", weights_only=False)
            os.remove(out)
        else:
            print(f"[longctx_check] variant {v} failed rc={rc}", flush=True)
    keys = list(outs)
    for k in keys:
        r = outs[k]
        print(f"  {k:<14} L={r['prompt_len']} prefill {r['prefill_s']:.2f}s ({r['prompt_len'] / max(r['prefill_s'], 1e-9):.0f} "
              f"tok/s incl. first sample) decode {1000 * r['decode_s'] / max(r['gen'] - 1, 1):.2f} ms/tok peak "
              f"{r['peak_gib']:.1f} GiB first={r['first_token']} tokens={r['tokens']}")
    if len(keys) >= 2:
        a0 = outs[keys[0]]
        ta = [a0["first_token"]] + a0["tokens"]
        for k in keys[1:]:
            b0 = outs[k]
            tb = [b0["first_token"]] + b0["tokens"]
            pre = 0
            for x, y in zip(ta, tb):
                if x != y:
                    break
                pre += 1
            da, db = a0["decode_logits"], b0["decode_logits"]
            m = min(da.shape[0], db.shape[0])
            c = F.cosine_similarity(da[:m], db[:m], dim=-1) if m else torch.zeros(1)
            t1 = (da[:m].argmax(-1) == db[:m].argmax(-1)).float().mean().item() if m else 0.0
            print(f"  {keys[0]} vs {k}: identical token prefix {pre}/{min(len(ta), len(tb))}; decode step-0 logits cos "
                  f"{c[0]:.5f}; all-step cos mean {c.mean():.5f} (contexts diverge after the first differing token)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
