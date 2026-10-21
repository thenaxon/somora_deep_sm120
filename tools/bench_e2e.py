#!/usr/bin/env python3
"""End-to-end DeepInfer benchmark on ONE SM120 GPU with the SM120 shims (offline engine, no server).

For every configuration (mode x MTP) a fresh worker process builds the engine once and runs, per batch
size B, N=B requests of P random prompt tokens and G generated tokens (greedy, ignore_eos):

    prefill tok/s   B*P / time until every request has its first token (all prefill steps)
    decode tok/s    B*(G-1) / time from "all first tokens" to the last token
    TPOT            that decode time / (G-1)            (per-request time per output token)
    tokens/step     decode tokens / (decode steps * B)  (1 without MTP; 1 + accepted drafts with MTP)
    acceptance      (tokens/step - 1) / nextn           (MTP only)

    PYTHONPATH=/opt/sm120/sm120:/opt/deepinfer python3 /opt/sm120/tools/bench_e2e.py --ckpt /work/ckpt3 \
        --batches 1,4,16,64 --prompt-len 256 --gen 64 --modes eager,graph --mtp 0 --out /work/bench.jsonl

    # MTP / DSpark (needs a checkpoint that still has the mtp.* layers, see docs/RUNTIME.md):
    ... --ckpt /work/ckpt3_mtp --mtp 0,5 --force-topk 0,1024

Notes:
  * Each (mode, nextn, force_topk) runs in its own process: DeepInfer reads DEEPINFER_ENFORCE_EAGER /
    DEEPINFER_USE_NEXTN_DECODE at engine construction. Init time (incl. memory profiling, b12x JIT and
    CUDA-graph capture) is reported separately.
  * --warmup (default 1) runs every batch once with G=4 before measuring, so lazy b12x plans/JIT of new
    token buckets (eager mode) are not timed.
  * The truncated test model is numerically chaotic and its DSpark heads read features of other layers:
    acceptance rates on it only exercise the plumbing. Use --prompt-ids with real text for anything else.
  * Wall-clock per engine step includes scheduler/sampler host time and ZOS overlap, like a real server.
  * DEEPINFER_PREFILL_(ANY_)ATTEMPT_INTERVAL=1: the engine default (70/100) delays prefill by ~70 idle steps.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="/work/ckpt3")
    ap.add_argument("--batches", default="1,4,16,64")
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--gen", type=int, default=64, help="generated tokens per request (>= 2)")
    ap.add_argument("--modes", default="eager,graph", help="comma list of eager|graph")
    ap.add_argument("--mtp", default="0", help="comma list of DEEPINFER_USE_NEXTN_DECODE values (0 = off)")
    ap.add_argument("--force-topk", default="0", help="comma list of DEEPINFER_SAMPLER_FORCE_TOPK values")
    ap.add_argument("--dflash-eager", action="store_true", help="graph mode: keep the DSpark draft model eager")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--prompt-ids", default=None, help="json list of token ids, tiled/cut to --prompt-len")
    ap.add_argument("--max-model-len", type=int, default=16384)  # >= 2048*8 (index_attn.py:213)
    ap.add_argument("--max-num-batched-tokens", type=int, default=4096)
    ap.add_argument("--max-blocks", type=int, default=512, help="SM120_MAX_GPU_BLOCKS (128-token KV blocks)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.95)  # 0.90 leaves no KV room on 32 GB
    ap.add_argument("--zos", choices=("default", "0", "1"), default="default")
    ap.add_argument("--out", default="/work/bench_e2e.jsonl")
    ap.add_argument("--deepinfer-root", default=os.environ.get("DEEPINFER_ROOT", "/opt/deepinfer"))
    ap.add_argument("--timeout", type=int, default=7200, help="per worker process, seconds")
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def _ints(s: str) -> list[int]:
    return [int(x) for x in str(s).split(",") if x.strip()]


# ------------------------------------------------------------------------------------------------ worker
def worker_env(args, mode: str, nextn: int, force_topk: int) -> dict:
    env = dict(os.environ)
    eager = mode == "eager"
    env.update({
        "DEEPINFER_USE_NEXTN_DECODE": str(nextn),
        "DEEPINFER_ENFORCE_EAGER": "1" if eager else "0",
        "DEEPINFER_DFLASH_ENFORCE_EAGER": "1" if (eager or args.dflash_eager) else "0",
        "DEEPINFER_SAMPLER_FORCE_TOPK": str(force_topk),
        "DEEPINFER_DISABLE_SYNC_GUARD": "1",  # eager prefill: torch reference paths in the shims synchronise
        "DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS": "0",
        "DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL": "0",
        "DEEPINFER_EXPERT_COUNT_WINDOW_SIZE": "1",
        "DEEPINFER_LOGGING_INTERVAL_STEP": "0",
        # NORMAL engines attempt prefill only every 70 (ALL) / 100 (ANY) decode steps (environ.py "Scheduling"):
        # offline, that is pure waiting in placeholder steps and would be timed as prefill
        "DEEPINFER_PREFILL_ATTEMPT_INTERVAL": "1",
        "DEEPINFER_PREFILL_ANY_ATTEMPT_INTERVAL": "1",
        "SM120_MAX_GPU_BLOCKS": str(args.max_blocks),
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
        "RANK": "0",
        "WORLD_SIZE": "1",
    })
    if args.zos != "default":
        env["DEEPINFER_ZERO_OVERHEAD_SCHEDULING"] = args.zos
    shims = os.path.join(os.path.dirname(HERE), "sm120")
    pp = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    for p in (args.deepinfer_root, shims):  # shims first: sitecustomize + drop-in kernel packages
        if p not in pp:
            pp.insert(0, p)
    env["PYTHONPATH"] = os.pathsep.join(pp)
    return env


def make_prompts(args, n: int, vocab: int, seed: int) -> list[list[int]]:
    import random

    if args.prompt_ids:
        with open(args.prompt_ids) as f:
            obj = json.load(f)
        base = [int(x) for x in (obj["input_ids"] if isinstance(obj, dict) else obj)]
        ids = (base * (args.prompt_len // max(1, len(base)) + 1))[: args.prompt_len]
        return [list(ids) for _ in range(n)]  # identical prompts: the scheduler may share nothing (no prefix cache)
    rng = random.Random(seed)
    lo, hi = 1000, max(1001, vocab - 1000)  # stay clear of special / reserved ids
    return [[rng.randrange(lo, hi) for _ in range(args.prompt_len)] for _ in range(n)]


def run_batch(llm, args, B: int, gen: int, seed: int) -> dict:
    import torch
    from deepinfer._C.engine import RequestOutput  # step outputs are raw pointers; clone() like LLM._run_engine
    from deepinfer.sampling_params import SamplingParams

    eng = llm.llm_engine
    vocab = eng.model_config.get_vocab_size()
    sp = SamplingParams(temperature=args.temperature, max_tokens=gen, ignore_eos=True, top_logprobs=0, seq_seed=seed)
    prompts = make_prompts(args, B, vocab, seed)
    rids = [str(llm._add_request(None, sp, p).request_id) for p in prompts]
    want = set(rids)
    count = {r: 0 for r in rids}
    first_at: dict[str, float] = {}
    steps = []  # (t_end, tokens emitted in this step)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while True:
        unfinished = eng.has_unfinished_requests()
        outs = eng.step() if unfinished else eng.flush_inflight_reqs()
        now = time.perf_counter()
        n_tok = 0
        for raw in outs:
            o = RequestOutput.clone(raw)
            rid = str(o.request_id)
            if rid not in want:
                continue
            count[rid] += 1
            n_tok += 1
            first_at.setdefault(rid, now)
        steps.append((now, n_tok))
        if not unfinished:
            break
    torch.cuda.synchronize()
    t_end = time.perf_counter()
    if len(first_at) != B:
        raise RuntimeError(f"only {len(first_at)}/{B} requests produced a token")
    t_first_all = max(first_at.values())
    prefill_s = t_first_all - t0
    decode_s = t_end - t_first_all
    dec_tokens = sum(count.values()) - B
    # decode steps = engine steps after the last first token that emitted something
    dec_steps = sum(1 for t, n in steps if t > t_first_all and n > 0)
    nextn = int(os.environ.get("DEEPINFER_USE_NEXTN_DECODE", "0"))
    per_step = dec_tokens / max(1, dec_steps * B)
    return dict(
        batch=B, prompt_len=args.prompt_len, gen=gen,
        tokens_out=sum(count.values()), short_requests=sum(1 for c in count.values() if c < gen),
        prefill_s=prefill_s, prefill_tok_s=B * args.prompt_len / max(prefill_s, 1e-9),
        decode_s=decode_s, decode_tok_s=dec_tokens / max(decode_s, 1e-9),
        tpot_ms=1000.0 * decode_s / max(1, gen - 1),
        engine_steps=len(steps), decode_steps=dec_steps,
        prefill_engine_steps=sum(1 for t, _ in steps if t <= t_first_all), tokens_per_step_per_req=per_step,
        acceptance=(per_step - 1.0) / nextn if nextn > 0 else None,
    )


def worker_main(cfg: dict) -> int:
    args = parse_args(cfg["argv"])
    mode, nextn, force_topk = cfg["mode"], cfg["nextn"], cfg["force_topk"]
    for p in (os.path.join(args.deepinfer_root, "examples"), args.deepinfer_root):
        if p not in sys.path:
            sys.path.insert(0, p)
    batches = _ints(args.batches)
    tag = dict(mode=mode, nextn=nextn, force_topk=force_topk, ckpt=args.ckpt)

    def emit(rec: dict) -> None:
        rec = {**tag, **rec}
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[bench_e2e] {json.dumps(rec)}", flush=True)

    import sm120_patches

    sm120_patches.install()
    from _config import create_llm

    import torch

    t0 = time.time()
    try:
        llm = create_llm(
            args.ckpt, 0, ep_size=1, cp_size=1, data_parallel_size=1,
            max_num_batched_tokens=args.max_num_batched_tokens, max_model_len=args.max_model_len,
            max_num_seqs=max(batches), gpu_memory_utilization=args.gpu_memory_utilization,
            lookahead_tokens=min(1, nextn), disable_log_stats=True,
        )
    except Exception as exc:  # noqa: BLE001
        hint = ""
        if nextn and isinstance(exc, (IndexError, AssertionError, KeyError, NotImplementedError)):
            hint = " (MTP needs a checkpoint with the mtp.* DSpark layers and dflash config, see docs/RUNTIME.md)"
        emit(dict(status="init_failed", error=f"{type(exc).__name__}: {exc}{hint}"))
        return 2
    init_s = time.time() - t0
    import deep_gemm._b12x as gb
    import sm120_moe

    emit(dict(status="init", init_s=init_s, gpu=torch.cuda.get_device_name(), moe=sm120_moe.stats(), gemm=gb.stats(),
              tiny_decode=sm120_patches.TINY_DECODE_POLICY,
              mem_alloc_gib=torch.cuda.memory_allocated() / 2**30))
    rc = 0
    for i, B in enumerate(batches):
        try:
            for w in range(args.warmup):
                run_batch(llm, args, B, min(4, args.gen), seed=10_000 + 97 * i + w)
            torch.cuda.reset_peak_memory_stats()
            rec = run_batch(llm, args, B, args.gen, seed=97 * i)
            rec.update(status="ok", peak_mem_gib=torch.cuda.max_memory_allocated() / 2**30,
                       moe=sm120_moe.stats(), gemm=gb.stats())
            emit(rec)
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            emit(dict(status="failed", batch=B, error=f"{type(exc).__name__}: {exc}"))
            rc = 1
            break  # engine state is unknown after a failure
    llm.close()
    return rc


# ------------------------------------------------------------------------------------------------ driver
def _table(recs: list[dict]) -> str:
    hdr = ("mode", "mtp", "ftopk", "B", "prefill tok/s", "decode tok/s", "TPOT ms", "tok/step", "accept", "init s")
    rows = [hdr]
    init = {(r["mode"], r["nextn"], r["force_topk"]): r.get("init_s") for r in recs if r.get("status") == "init"}
    for r in recs:
        key = (r["mode"], r["nextn"], r["force_topk"])
        if r.get("status") == "ok":
            rows.append((r["mode"], str(r["nextn"]), str(r["force_topk"]), str(r["batch"]),
                         f"{r['prefill_tok_s']:.0f}", f"{r['decode_tok_s']:.1f}", f"{r['tpot_ms']:.1f}",
                         f"{r['tokens_per_step_per_req']:.2f}",
                         "-" if r["acceptance"] is None else f"{r['acceptance']:.1%}",
                         f"{init.get(key) or 0:.0f}"))
        elif r.get("status") in ("failed", "init_failed"):
            rows.append((r["mode"], str(r["nextn"]), str(r["force_topk"]), str(r.get("batch", "-")),
                         r["status"].upper(), r.get("error", "")[:60], "", "", "", ""))
    w = [max(len(row[i]) for row in rows) for i in range(len(hdr))]
    line = lambda row: "| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(row)) + " |"  # noqa: E731
    return "\n".join([line(rows[0]), "|" + "|".join("-" * (x + 2) for x in w) + "|"] + [line(r) for r in rows[1:]])


def main() -> int:
    args = parse_args()
    if args.worker:
        return worker_main(json.loads(args.worker))
    if args.gen < 2:
        raise SystemExit("--gen must be >= 2 (decode needs at least one token after the prefill token)")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    start = 0
    if os.path.exists(args.out):
        with open(args.out) as f:
            start = sum(1 for _ in f)
    argv = list(sys.argv[1:])
    rc = 0
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        if mode not in ("eager", "graph"):
            raise SystemExit(f"unknown mode {mode!r}")
        for nextn in _ints(args.mtp):
            for ftk in _ints(args.force_topk) if nextn else [0]:
                cfg = dict(mode=mode, nextn=nextn, force_topk=ftk, argv=argv)
                print(f"[bench_e2e] === mode={mode} nextn={nextn} force_topk={ftk} ===", flush=True)
                try:
                    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", json.dumps(cfg)],
                                       env=worker_env(args, mode, nextn, ftk), timeout=args.timeout)
                    rc |= p.returncode != 0
                except subprocess.TimeoutExpired:
                    with open(args.out, "a") as f:
                        f.write(json.dumps(dict(mode=mode, nextn=nextn, force_topk=ftk, status="failed",
                                                error=f"timeout {args.timeout}s")) + "\n")
                    rc = 1
    with open(args.out) as f:
        recs = [json.loads(line) for i, line in enumerate(f) if i >= start and line.strip()]
    print("\n" + _table(recs))
    print(f"\n[bench_e2e] raw records appended to {args.out}")
    return int(rc)


if __name__ == "__main__":
    sys.exit(main())
