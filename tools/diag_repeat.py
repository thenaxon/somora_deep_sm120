#!/usr/bin/env python3
"""Determinism / latency probe without the HTTP layer: the same prompt through LLM.generate several times in one
process (greedy with seq_seed 0, then seeded sampling), plus per-request prefill latency (max_tokens=1).

    python3 /opt/sm120/tools/diag_repeat.py --ckpt /work/ckpt3 [--max-num-seqs 8] [--repeats 3]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/work/ckpt3")
    ap.add_argument("--max-num-seqs", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=16)
    args = ap.parse_args()
    import argparse as _ap

    import run_deepinfer

    run_deepinfer.setup_env(_ap.Namespace(max_blocks=256, zos="default", batch_invariant=False, nextn=0, graphs=False,
                                          force_topk=None, deepinfer_root=os.environ.get("DEEPINFER_ROOT", "/opt/deepinfer")))
    import sm120_patches

    sm120_patches.install()
    from _config import create_llm

    from deepinfer.sampling_params import SamplingParams

    llm = create_llm(args.ckpt, 0, ep_size=1, cp_size=1, data_parallel_size=1, max_num_batched_tokens=4096,
                     max_model_len=16384, max_num_seqs=args.max_num_seqs, gpu_memory_utilization=0.9, lookahead_tokens=0,
                     disable_log_stats=True)
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(args.ckpt, "tokenizer.json"))
    ids = tok.encode("The lighthouse keeper wrote in his logbook that", add_special_tokens=False).ids

    def run(sp):
        t0 = time.perf_counter()
        out = llm.generate(prompt_token_ids=[ids], sampling_params=sp, use_tqdm=False)
        return [int(x.token_id) for x in out[0]], time.perf_counter() - t0

    for name, mk in (("greedy seed0", lambda: SamplingParams(temperature=0.0, max_tokens=args.tokens, ignore_eos=True, seq_seed=0)),
                     ("greedy seedNone", lambda: SamplingParams(temperature=0.0, max_tokens=args.tokens, ignore_eos=True)),
                     ("t=1 seed1234", lambda: SamplingParams(temperature=1.0, top_p=0.95, max_tokens=args.tokens, ignore_eos=True, seq_seed=1234))):
        outs = [run(mk()) for _ in range(args.repeats)]
        same = all(o[0] == outs[0][0] for o in outs)
        print(f"[diag] {name}: identical over {args.repeats} runs: {same}; times {[round(o[1], 2) for o in outs]}")
        for o in outs:
            print(f"[diag]    {o[0]}")
    # per-step timing of one request (engine API directly)
    from deepinfer._C.engine import RequestOutput

    eng = llm.llm_engine
    for trial in range(2):
        llm._add_request(None, SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True, seq_seed=0), ids)
        times, n_out = [], []
        while eng.has_unfinished_requests():
            t0 = time.perf_counter()
            outs = eng.step()
            times.append(time.perf_counter() - t0)
            n_out.append(len(outs))
        t0 = time.perf_counter()
        outs = eng.flush_inflight_reqs()
        times.append(time.perf_counter() - t0)
        n_out.append(len(outs))
        print(f"[diag] steps (s) trial {trial}: {[round(t, 3) for t in times]} outputs per step {n_out}")
    for n in (1, 2):
        _, t = run(SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True, seq_seed=0))
        print(f"[diag] max_tokens={n}: {t:.3f}s")
    llm.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
