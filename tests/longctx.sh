#!/bin/bash
# Long-context check on /work/ckpt3 (real candidate_topk_blocks 2048): for each prompt length L
#   1. correctness: run_deepinfer (no chunked prefill: --max-num-batched-tokens = max_model_len) with b12x
#      attention (graphs) vs SM120_ATTN_IMPL=torch (eager), same tiled prompt,
#      compare decoded tokens and decode / last-64 prefill logits (cosine)
#   2. performance: bench_e2e (graph mode, batch 1) prefill tok/s and TPOT
# usage: longctx.sh "16384 32768 65536"
#   env: TEST_CKPT (default /work/ckpt3), TEST_CHUNK (max-num-batched-tokens, default 8192), TEST_SKIP_TORCH=1
#   logs: /work/lc_logs/
# Container paths (override via env): SM120_ROOT=/opt/sm120 (this repo), WORK=/work (work dir),
# HF_CKPT=/ckpt (original HF checkpoint shards, read-only).
SM120_ROOT=${SM120_ROOT:-/opt/sm120}; WORK=${WORK:-/work}; HF_CKPT=${HF_CKPT:-/ckpt}; export SM120_ROOT WORK HF_CKPT
cd $SM120_ROOT
export B12X_AUTOTUNE=0
CKPT=${TEST_CKPT:-$WORK/ckpt3}
LOGD=$WORK/lc_logs; mkdir -p $LOGD
LENS=${1:-"16384 32768 65536"}
CHUNK=${TEST_CHUNK:-8192}
for L in $LENS; do
  ML=$(( (L + 1024 + 1023) / 1024 * 1024 ))       # max_model_len (>= 16384 = 2048*8 candidates for L >= 16k)
  BLK=$(( 2 * ML / 128 + 64 ))                     # KV block cap generous: GPU memory is the real limit
  python3 - "$L" <<'PY'
import os
import json, sys
L = int(sys.argv[1])
base = json.load(open(f"{os.environ.get('WORK', '/work')}/prompt_ids.json"))
base = base["input_ids"] if isinstance(base, dict) else base
ids = (base[1:] * (L // len(base) + 2))[: L - 1]
json.dump({"input_ids": [base[0]] + ids}, open(f"{os.environ.get('WORK', '/work')}/lc_{L}.json", "w"))
PY
  echo "=== L=$L max_model_len=$ML blocks=$BLK chunk=$CHUNK"
  for v in b12x torch; do
    if [ "$v" = torch ] && [ -n "${TEST_SKIP_TORCH:-}" ]; then continue; fi
    if [ "$v" = b12x ]; then extra="--graphs"; envs=""; else extra=""; envs="SM120_ATTN_IMPL=torch"; fi
    env $envs SM120_MAX_GPU_BLOCKS=$BLK python3 $SM120_ROOT/tools/run_deepinfer.py \
        --ckpt $CKPT --prompt-ids $WORK/lc_$L.json --max-model-len $ML --max-num-batched-tokens $ML \
        --max-blocks $BLK --gpu-memory-utilization 0.95 --compact --no-hidden $extra --out $WORK/lc_${L}_$v.pt \
        > $LOGD/lc_${L}_$v.log 2>&1 || { echo "  [$v] FAILED"; grep -E 'Error|rror:|out of memory' $LOGD/lc_${L}_$v.log | tail -4; }
    tail -1 $LOGD/lc_${L}_$v.log 2>/dev/null | grep -oE 'init [0-9.]+s generate [0-9.]+s' | sed "s/^/  [$v] /"
    grep -E 'b12x .* failed|unsupported|falling back|triton .* failed' $LOGD/lc_${L}_$v.log | sort | uniq -c | head -4
  done
  python3 - "$L" <<'PY'
import os
import os, sys, torch
import torch.nn.functional as F
L = sys.argv[1]
p = {v: f"{os.environ.get('WORK', '/work')}/lc_{L}_{v}.pt" for v in ("b12x", "torch")}
if not all(os.path.exists(x) for x in p.values()):
    sys.exit(0)
a, b = (torch.load(p[v], map_location="cpu", weights_only=False) for v in ("b12x", "torch"))
ta, tb = a["decoded_ids"].tolist(), b["decoded_ids"].tolist()
pre = sum(1 for _ in __import__("itertools").takewhile(lambda t: t[0] == t[1], zip(ta, tb)))
print(f"  decoded b12x : {ta}\n  decoded torch: {tb}\n  identical prefix {pre}/{len(ta)}")
for k in ("prefill_logits_last64", "decode_logits_first4"):
    x, y = a.get(k), b.get(k)
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor) and x.shape == y.shape and x.numel():
        c = F.cosine_similarity(x.float(), y.float(), dim=-1)
        t1 = (x.argmax(-1) == y.argmax(-1)).float().mean().item()
        print(f"  {k:<24} cos mean {c.mean():.5f} min {c.min():.5f}  top1 agree {t1:.3f}")
PY
  # performance (graph mode, batch 1); bench_e2e tiles --prompt-ids to --prompt-len
  python3 $SM120_ROOT/tools/bench_e2e.py --ckpt $CKPT --batches 1 --prompt-len $L --gen 32 --modes graph \
      --prompt-ids $WORK/lc_$L.json --max-model-len $ML --max-num-batched-tokens $CHUNK --max-blocks $BLK \
      --warmup 0 --out $WORK/lc_bench.jsonl > $LOGD/lc_bench_$L.log 2>&1 || echo "  bench FAILED"
  grep '"status": "ok"' $LOGD/lc_bench_$L.log | python3 -c "
import json, sys
for line in sys.stdin:
    r = json.loads(line.split('[bench_e2e] ', 1)[1])
    print(f\"  bench L={r['prompt_len']}: prefill {r['prefill_tok_s']:.0f} tok/s ({r['prefill_s']:.2f} s), TPOT {r['tpot_ms']:.2f} ms, peak {r.get('peak_mem_gib', 0):.1f} GiB\")"
  grep -E '"status": "(failed|init_failed)"' $LOGD/lc_bench_$L.log | cut -c1-300
  rm -f $WORK/lc_${L}_*.pt
done
rm -f $WORK/lc_bench.jsonl
