#!/usr/bin/env python3
"""Image inputs with and without DeepInfer's multimodal shared pool (DEEPINFER_USE_MULTIMODAL_POOL).

The pool is a pinned host buffer (multiprocessing.shared_memory, /dev/shm/psm_*) through which the multimodal worker
processes hand preprocessed image tensors to the engine; without it the tensors travel through the worker's
multiprocessing queue. Run this once per setting and compare the printed token ids:

    DEEPINFER_USE_MULTIMODAL_POOL=1 SM120_MOE_DETERMINISTIC=1 python3 tests/mm_pool_check.py --ckpt /work/ckpt3
    DEEPINFER_USE_MULTIMODAL_POOL=0 SM120_MOE_DETERMINISTIC=1 python3 tests/mm_pool_check.py --ckpt /work/ckpt3

Prints the pool setting, generation time, the /dev/shm psm_* bytes while the engine runs, and the greedy tokens of
two text+image requests and one text-only request. The truncated test checkpoints carry zero vision weights, so the
tokens only show that both settings agree, not that the image is understood.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time


def shm_psm_bytes() -> int:
    return sum(os.path.getsize(p) for p in glob.glob("/dev/shm/psm_*"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="/work/ckpt3")
    ap.add_argument("--prompt-len", type=int, default=1024)
    ap.add_argument("--image-size", type=int, default=448)
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
    from _config import create_llm
    from _prompts import VLPromptParams, generate_random_image_prompt

    from deepinfer.sampling_params import SamplingParams

    llm = create_llm(args.ckpt, 0, ep_size=1, cp_size=1, data_parallel_size=1, max_num_batched_tokens=4096,
                     max_model_len=16384, max_num_seqs=4, gpu_memory_utilization=0.95, lookahead_tokens=0,
                     disable_log_stats=True, decode_sequence=False)
    from deepinfer import envs

    print(f"[mm_pool] DEEPINFER_USE_MULTIMODAL_POOL={int(envs.DEEPINFER_USE_MULTIMODAL_POOL)} "
          f"pool_size_gb={envs.DEEPINFER_MULTIMODAL_POOL_SIZE_GB} psm_bytes_after_init={shm_psm_bytes()}", flush=True)
    params = VLPromptParams.from_llm(llm)
    reqs = [generate_random_image_prompt(params, args.prompt_len, (args.image_size, args.image_size), seed=s)
            for s in (1, 2)]
    text_ids = params.text_token_pool[: args.prompt_len // 4].tolist()
    sp = SamplingParams(temperature=0.0, max_tokens=args.gen, ignore_eos=True, seq_seed=0)
    t0 = time.time()
    out = llm.generate(prompt_token_ids=[r[0] for r in reqs] + [text_ids], sampling_params=sp, use_tqdm=False,
                       multi_modal_data=[r[1] for r in reqs] + [None])
    dt = time.time() - t0
    toks = [[int(x.token_id) for x in o] for o in out]
    print(f"[mm_pool] generate {dt:.1f}s psm_bytes_while_running={shm_psm_bytes()}", flush=True)
    for name, t in zip(("image1", "image2", "text"), toks):
        print(f"[mm_pool] {name}: {t}", flush=True)
    ok = all(len(t) == args.gen for t in toks)
    llm.close()
    print(f"[mm_pool] {'OK' if ok else 'FAILED'} psm_bytes_after_close={shm_psm_bytes()}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
