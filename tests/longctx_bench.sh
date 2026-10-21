#!/bin/bash
# bench_e2e (graph, batch 1, 32 generated tokens) vs prompt length and prefill chunk on /work/ckpt3
# Container paths (override via env): SM120_ROOT=/opt/sm120 (this repo), WORK=/work (work dir),
# HF_CKPT=/ckpt (original HF checkpoint shards, read-only).
SM120_ROOT=${SM120_ROOT:-/opt/sm120}; WORK=${WORK:-/work}; HF_CKPT=${HF_CKPT:-/ckpt}; export SM120_ROOT WORK HF_CKPT
cd $SM120_ROOT
export B12X_AUTOTUNE=0
CKPT=${TEST_CKPT:-$WORK/ckpt3}
for CH in ${TEST_CHUNKS:-2048 8192}; do
  for L in ${1:-16384 32768 65536}; do
    ML=$(( (L + 1024 + 1023) / 1024 * 1024 )); [ $ML -lt 16384 ] && ML=16384
    BLK=$(( 2 * ML / 128 + 64 ))
    python3 $SM120_ROOT/tools/bench_e2e.py --ckpt $CKPT --batches 1 --prompt-len $L --gen 32 --modes graph \
        --prompt-ids $WORK/prompt_ids.json --max-model-len $ML --max-num-batched-tokens $CH --max-blocks $BLK \
        --warmup 0 --out $WORK/lcb.jsonl > $WORK/lc_logs/bench_${L}_$CH.log 2>&1
    grep -E '"status": "(ok|failed|init_failed)"' $WORK/lc_logs/bench_${L}_$CH.log | python3 -c "
import json, sys
for line in sys.stdin:
    r = json.loads(line.split('[bench_e2e] ', 1)[1])
    if r['status'] == 'ok':
        print(f\"  L={$L:6d} chunk={$CH:5d}: prefill {r['prefill_tok_s']:7.0f} tok/s ({r['prefill_s']:.2f} s)  TPOT {r['tpot_ms']:.2f} ms  peak {r.get('peak_mem_gib', 0):.1f} GiB\")
    else:
        print(f\"  L={$L:6d} chunk={$CH:5d}: {r['status']} {r.get('error', '')[:200]}\")"
  done
done
rm -f $WORK/lcb.jsonl
