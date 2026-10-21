#!/usr/bin/env python3
"""Per-op GPU time breakdown of DeepInfer prefill and decode steps (torch.profiler / CUPTI), one GPU.

    PYTHONPATH=/opt/sm120/sm120:/opt/deepinfer python3 /opt/sm120/tests/prof_e2e.py --ckpt /work/ckpt_yoco \
        --batches 1,8,32 --prompt-len 1024 --decode-steps 16 --modes graph

Per batch size B: B requests of P random tokens are added; all prefill engine steps run inside one
profiler session, then --decode-steps decode steps inside a second one. For each phase it prints
GPU kernel time per category (attention, indexer, top-k, GEMM, MoE, mHC/norm, sampler, ...), the
wall time of the engine steps, the GPU-busy time (union of kernel intervals) and the difference
(= host overhead / idle GPU between and inside steps), plus the top kernels by time.
Reuses tools/bench_e2e.py for the engine environment, prompts and request handling.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.abspath(os.path.join(HERE, "..", "tools"))
sys.path.insert(0, TOOLS)

CATS = [  # first match wins; patterns on the lower-cased kernel name
    ("comm.nccl", r"nccl"),
    ("ep.pack", r"_pack_kernel|_unpack_kernel"),
    ("attn.fused_prep_cast", r"_prepare_q_kernel|_rope_cast_kernel"),
    ("attn.sparse_mla", r"mla|prefill_mg|unifieddecode|unified_decode|split_kv|merge_kernel|map_indexed"),
    ("indexer.score", r"mxfp4|pagedscore|tensorcorepagedscore|_score|dsa_|indexer"),
    ("topk", r"topk|radix|sort|_key_kernel|bitonic|select"),
    ("moe", r"moe|expert|grouped|routing|permute|dispatch|combine"),
    ("gemm", r"gemm|gemv|nvjet|xmma|matmul|block_fp8|wgmma|hgmma|cutlass|cublas|sm80_|sm90_|sm100_|sm120_|ampere|einsum"),
    ("mhc/norm", r"rms|norm|hc_|mhc|sinkhorn|hadamard"),
    ("sampler", r"sampl|softmax|argmax|logit|gumbel|penalt"),
    ("kv-cache/rope/quant", r"store_kv|cache|rotary|rope|quant|cast|compress"),
    ("torch elementwise/copy", r"elementwise|copy|fill|cat|index|gather|scatter|reduce|memcpy|memset|vectorized"),
]


def category(name: str) -> str:
    n = name.lower()
    for cat, pat in CATS:
        if re.search(pat, n):
            return cat
    return "other"


def union_us(intervals):
    tot, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        tot += cur_e - cur_s
    return tot


def kernels(prof):
    from torch.autograd import DeviceType
    out = []
    for ev in prof.events():
        if (ev.device_type == DeviceType.CUDA and ev.time_range.end > ev.time_range.start
                and not ev.name.startswith(("deepinfer:", "ProfilerStep")) and not getattr(ev, "is_user_annotation", False)):
            out.append((ev.name, ev.time_range.start, ev.time_range.end))
    return out


def summarize(label, ks, wall_ms, steps, top=18):
    by_cat = collections.Counter()
    by_name = collections.Counter()
    cnt = collections.Counter()
    for n, s, e in ks:
        d = (e - s) / 1000.0
        by_cat[category(n)] += d
        by_name[n] += d
        cnt[n] += 1
    gpu_sum = sum(by_cat.values())
    busy = union_us([(s, e) for _, s, e in ks]) / 1000.0
    res = dict(label=label, steps=steps, wall_ms=wall_ms, gpu_busy_ms=busy, gpu_kernel_sum_ms=gpu_sum,
               host_gap_ms=max(0.0, wall_ms - busy), cats={k: v for k, v in by_cat.most_common()},
               top=[(n[:90], by_name[n], cnt[n]) for n, _ in by_name.most_common(top)])
    print(f"\n=== {label}: {steps} engine steps, wall {wall_ms:.2f} ms ({wall_ms / max(steps, 1):.3f} ms/step), "
          f"GPU busy {busy:.2f} ms ({100 * busy / max(wall_ms, 1e-9):.1f} %), host/idle gap {res['host_gap_ms']:.2f} ms")
    for k, v in by_cat.most_common():
        print(f"    {k:<24} {v:9.3f} ms  {100 * v / max(gpu_sum, 1e-9):5.1f} %   {v / max(steps, 1):8.4f} ms/step")
    print("  top kernels:")
    for n, t, c in res["top"]:
        print(f"    {t:9.3f} ms  x{c:<5d} [{category(n)}] {n}")
    if os.environ.get("PROF_CAT_TOP"):
        per = collections.defaultdict(list)
        for n, t in by_name.items():
            per[category(n)].append((t, n))
        for cat_name in sorted(per, key=lambda c: -by_cat[c]):
            items = sorted(per[cat_name], reverse=True)[: int(os.environ["PROF_CAT_TOP"])]
            print(f"  [{cat_name}] " + "; ".join(f"{t / max(steps, 1):.3f} ms/step x{cnt[n] // max(steps, 1)} {n[:80]}"
                                                for t, n in items))
    return res


def install_module_ranges(max_depth: int = 6):
    """record_function range per submodule (layer index folded to '*'); returns the hook handles."""
    import gc
    import torch

    roots = [o for o in gc.get_objects() if isinstance(o, torch.nn.Module) and type(o).__name__.endswith("ForCausalLM")]
    if not roots:
        print("[prof_e2e] no *ForCausalLM module found; module attribution disabled", flush=True)
        return []
    hooks = []
    for name, mod in roots[0].named_modules():
        if name.count(".") >= max_depth:
            continue
        key = "mod:" + re.sub(r"\.\d+(\.|$)", r".*\1", name or type(mod).__name__)

        def pre(m, args, key=key):
            r = torch.profiler.record_function(key)
            r.__enter__()
            m.__dict__.setdefault("_prof_ranges", []).append(r)

        def post(m, args, out):
            st = m.__dict__.get("_prof_ranges")
            if st:
                st.pop().__exit__(None, None, None)

        hooks.append(mod.register_forward_pre_hook(pre))
        hooks.append(mod.register_forward_hook(post))
    print(f"[prof_e2e] module ranges on {len(hooks) // 2} modules of {type(roots[0]).__name__}", flush=True)
    return hooks


def install_shim_ranges():
    """Wrap the shim entry points DeepInfer calls (attention, indexer, top-k) in 'sm120:<name>' ranges."""
    import importlib
    import torch

    def wrap(obj, attr, label):
        fn = getattr(obj, attr, None)
        if fn is None or getattr(fn, "_prof_wrapped", False):
            return
        def w(*a, **k):
            with torch.profiler.record_function("sm120:" + label):
                return fn(*a, **k)
        w._prof_wrapped = True
        setattr(obj, attr, w)

    import deep_gemm
    import flash_mla
    ab = importlib.import_module("deepinfer.model_executor.layers.attention_backend")
    ia = importlib.import_module("deepinfer.model_executor.models.v4_modules.index_attn")
    for name in ("flash_mla_sparse_fwd", "flash_mla_with_kvcache"):
        wrap(ab, name, name)
    for name in ("prefill", "decode"):
        wrap(flash_mla.fused_norm_rope_attn_rope_cast, name, f"fused.{name}")
    for name in ("fp8_fp4_mqa_logits", "fp8_fp4_paged_mqa_logits", "fp8_fp4_sparse_mqa_logits",
                 "fp8_fp4_paged_sparse_mqa_logits"):
        wrap(deep_gemm, name, name)
    wrap(ia, "topk", "deep_select.topk")


def module_breakdown(prof, label, steps, top=30):
    """GPU kernel time and CPU time per innermost module range (eager only)."""
    gpu = collections.Counter()
    gk = collections.defaultdict(collections.Counter)   # module -> kernel name -> ms
    cs = collections.defaultdict(collections.Counter)   # module -> cpu op (with python fn parent) -> self cpu ms

    def owner(ev):
        p = ev.cpu_parent
        while p is not None and not p.name.startswith(("mod:", "sm120:")):
            p = p.cpu_parent
        if p is None:
            return "(outside modules)"
        return p.name if p.name.startswith("sm120:") else p.name[4:]

    for ev in prof.events():
        if ev.name.startswith(("mod:", "sm120:")) or ev.device_type.name != "CPU":
            continue
        key = owner(ev)
        cs[key][ev.name] += ev.self_cpu_time_total / 1000.0
        ks = getattr(ev, "kernels", None)
        if not ks:
            continue
        for k in ks:
            gk[key][k.name[:70]] += k.duration / 1000.0
        gpu[key] += sum(k.duration for k in ks) / 1000.0
    cpu = collections.Counter()
    for ev in prof.events():
        if ev.name.startswith("mod:"):
            cpu[ev.name[4:]] += ev.cpu_time_total / 1000.0
        elif ev.name.startswith("sm120:"):
            cpu[ev.name] += ev.cpu_time_total / 1000.0
    tot = sum(gpu.values())
    print(f"\n--- {label}: GPU kernel time by innermost module ({tot:.1f} ms total), with inclusive CPU time")
    for k, v in gpu.most_common(top):
        print(f"    {v:9.3f} ms {100 * v / max(tot, 1e-9):5.1f} %  {v / max(steps, 1):8.3f} ms/step  cpu {cpu.get(k, 0):9.1f} ms  {k}")
    hot = [k for k, _ in gpu.most_common(4)] + [k for k, _ in sorted(cpu.items(), key=lambda kv: -kv[1])[:3]]
    for k in dict.fromkeys(hot):
        print(f"  [{k}] top kernels: " + "; ".join(f"{n} {t:.1f}ms" for n, t in gk[k].most_common(8)))
        print(f"  [{k}] top self-CPU ops: " + "; ".join(f"{n} {t:.1f}ms" for n, t in cs[k].most_common(10)))


def worker(a):
    import bench_e2e as BE
    import sm120_patches

    sm120_patches.install()
    for p in (os.path.join(a.deepinfer_root, "examples"), a.deepinfer_root):
        if p not in sys.path:
            sys.path.insert(0, p)
    from _config import create_llm
    import torch
    from torch.profiler import ProfilerActivity, profile

    batches = [int(x) for x in a.batches.split(",")]
    bargs = BE.parse_args(["--ckpt", a.ckpt, "--prompt-len", str(a.prompt_len), "--gen", str(a.decode_steps + 4)]
                          + (["--prompt-ids", a.prompt_ids] if a.prompt_ids else []))
    llm = create_llm(a.ckpt, 0, ep_size=1, cp_size=1, data_parallel_size=1, profile_init=True,
                     max_num_batched_tokens=a.max_num_batched_tokens, max_model_len=a.max_model_len,
                     max_num_seqs=max(batches), gpu_memory_utilization=a.gpu_memory_utilization,
                     lookahead_tokens=min(1, a.nextn), disable_log_stats=True)
    from deepinfer._C.engine import RequestOutput
    from deepinfer.sampling_params import SamplingParams
    if a.modules:
        install_module_ranges()
        install_shim_ranges()
    eng = llm.llm_engine
    vocab = eng.model_config.get_vocab_size()
    results = []
    for i, B in enumerate(batches):
        BE.run_batch(llm, bargs, B, 4, seed=1000 + i)  # warm-up: plans / JIT for this batch size
        sp = SamplingParams(temperature=0.0, max_tokens=a.decode_steps + 4, ignore_eos=True, top_logprobs=0, seq_seed=i)
        rids = {str(llm._add_request(None, sp, p).request_id) for p in BE.make_prompts(bargs, B, vocab, i)}
        first = set()

        def step():
            outs = eng.step() if eng.has_unfinished_requests() else eng.flush_inflight_reqs()
            n = 0
            for raw in outs:
                o = RequestOutput.clone(raw)
                if str(o.request_id) in rids:
                    first.add(str(o.request_id))
                    n += 1
            return n

        torch.cuda.synchronize()
        pkw = dict(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=bool(a.stack),
                   record_shapes=False)
        with profile(**pkw) as pp:
            t0 = time.perf_counter()
            nsteps = 0
            while len(first) < B and nsteps < 10000:
                step()
                nsteps += 1
            torch.cuda.synchronize()
            pre_ms = 1000 * (time.perf_counter() - t0)
        r1 = summarize(f"B={B} prefill (P={a.prompt_len}, {B * a.prompt_len} tokens)", kernels(pp), pre_ms, nsteps)
        with profile(**pkw) as pd:
            t0 = time.perf_counter()
            for _ in range(a.decode_steps):
                step()
            torch.cuda.synchronize()
            dec_ms = 1000 * (time.perf_counter() - t0)
        r2 = summarize(f"B={B} decode", kernels(pd), dec_ms, a.decode_steps)
        if a.modules:
            module_breakdown(pp, f"B={B} prefill", nsteps)
            module_breakdown(pd, f"B={B} decode", a.decode_steps)
        if a.stack:
            for lab, pr in (("prefill", pp), ("decode", pd)):
                print(f"\n--- B={B} {lab}: top CPU ops by self CUDA time (stack, eager)")
                tab = pr.key_averages(group_by_stack_n=a.stack).table(
                    sort_by="self_cuda_time_total", row_limit=a.stack_rows, max_name_column_width=50,
                    max_src_column_width=110)
                print(tab)
        if a.trace_dir:
            os.makedirs(a.trace_dir, exist_ok=True)
            pd.export_chrome_trace(os.path.join(a.trace_dir, f"decode_b{B}.json"))
        while eng.has_unfinished_requests():
            step()
        eng.flush_inflight_reqs()
        results += [dict(batch=B, **r1), dict(batch=B, **r2)]
    if a.out:
        with open(a.out, "a") as f:
            for r in results:
                f.write(json.dumps({**r, "mode": a.mode, "ckpt": a.ckpt}) + "\n")
    llm.close()
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/work/ckpt_yoco")
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--prompt-len", type=int, default=1024)
    ap.add_argument("--prompt-ids", default=None)
    ap.add_argument("--decode-steps", type=int, default=16)
    ap.add_argument("--modes", default="graph")
    ap.add_argument("--nextn", type=int, default=0, help="DEEPINFER_USE_NEXTN_DECODE (MTP draft tokens)")
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--max-num-batched-tokens", type=int, default=4096)
    ap.add_argument("--max-blocks", type=int, default=512)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    ap.add_argument("--out", default=None)
    ap.add_argument("--trace-dir", default=None)
    ap.add_argument("--stack", type=int, default=0, help="N>0: record Python stacks (use --modes eager) and print "
                                                          "the top CPU ops grouped by N stack frames")
    ap.add_argument("--stack-rows", type=int, default=30)
    ap.add_argument("--modules", action="store_true", help="per-module attribution (use --modes eager)")
    ap.add_argument("--deepinfer-root", default=os.environ.get("DEEPINFER_ROOT", "/opt/deepinfer"))
    ap.add_argument("--mode", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.mode:
        return worker(a)
    import bench_e2e as BE
    rc = 0
    for mode in a.modes.split(","):
        bargs = BE.parse_args(["--max-blocks", str(a.max_blocks)])
        env = BE.worker_env(bargs, mode, a.nextn, 0)
        cmd = [sys.executable, os.path.abspath(__file__), *sys.argv[1:], "--mode", mode]
        print(f"### mode={mode}", flush=True)
        rc |= subprocess.call(cmd, env=env)
    return rc


if __name__ == "__main__":
    sys.exit(main())
