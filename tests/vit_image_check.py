#!/usr/bin/env python3
"""Full-size images through DeepInfer's own API (vision encoder on the SM120 shims).

    PYTHONPATH=/opt/sm120/sm120:/opt/deepinfer python3 /opt/sm120/tests/vit_image_check.py --ckpt /work/ckpt3

Sends text+image requests whose images are resized to the full 1024-token budget (4000x3000 and 1400x1400 inputs,
plus a small 448x448 one) and a text-only request, eager, greedy. Prints the expanded prompt lengths, the GPU peak
during generate and the tokens; exit 0 if every request produced its tokens. It exercises the shapes and memory of
the vision path (the SDPA shim allocates by shape). The test checkpoints on the RTX 5090 test machine synthesise the
vision tower as zeros (truncation.json "vision_synthesized": true, shard 00001 missing), so there the ViT outputs 0
and the tokens say nothing about image content; numerics need a checkpoint with real vision weights.
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="/work/ckpt3")
    ap.add_argument("--prompt-len", type=int, default=1536, help="expanded prompt length per image request")
    ap.add_argument("--gen", type=int, default=8)
    ap.add_argument("--deepinfer-root", default=os.environ.get("DEEPINFER_ROOT", "/opt/deepinfer"))
    args = ap.parse_args()
    for k, v in {"DEEPINFER_USE_NEXTN_DECODE": "0", "DEEPINFER_ENFORCE_EAGER": "1", "DEEPINFER_DISABLE_SYNC_GUARD": "1",
                 "DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS": "0", "DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL": "0",
                 "DEEPINFER_PREFILL_ATTEMPT_INTERVAL": "1", "SM120_MAX_GPU_BLOCKS": "256", "RANK": "0",
                 "WORLD_SIZE": "1"}.items():
        os.environ.setdefault(k, v)
    for p in (os.path.join(args.deepinfer_root, "examples"), args.deepinfer_root):
        if p not in sys.path:
            sys.path.insert(0, p)
    import sm120_patches

    sm120_patches.install()
    import torch
    from _config import create_llm
    from _prompts import VLPromptParams, generate_random_image_prompt

    from deepinfer.sampling_params import SamplingParams

    llm = create_llm(args.ckpt, 0, ep_size=1, cp_size=1, data_parallel_size=1, max_num_batched_tokens=4096,
                     max_model_len=16384, max_num_seqs=4, gpu_memory_utilization=0.95, lookahead_tokens=0,
                     disable_log_stats=True, decode_sequence=False)
    params = VLPromptParams.from_llm(llm)
    sizes = [(4000, 3000), (1400, 1400), (448, 448)]
    reqs = [generate_random_image_prompt(params, args.prompt_len, size, seed=i) for i, size in enumerate(sizes)]
    text_ids = params.text_token_pool[:256].tolist()
    sp = SamplingParams(temperature=0.0, max_tokens=args.gen, ignore_eos=True, seq_seed=0)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t0 = time.time()
    out = llm.generate(prompt_token_ids=[r[0] for r in reqs] + [text_ids], sampling_params=sp, use_tqdm=False,
                       multi_modal_data=[r[1] for r in reqs] + [None])
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - base) / 2**30
    toks = [[int(x.token_id) for x in o] for o in out]
    print(f"[vit_check] generate {time.time() - t0:.1f}s, GPU peak above the loaded engine {peak:.2f} GiB", flush=True)
    for (size, (ids, _)), t in zip(zip(sizes + ["text"], reqs + [(text_ids, None)]), toks):
        print(f"[vit_check] {size}: prompt {len(ids)} tokens -> {t}", flush=True)
    ok = all(len(t) == args.gen for t in toks)
    llm.close()
    print(f"[vit_check] {'OK' if ok else 'FAILED'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
