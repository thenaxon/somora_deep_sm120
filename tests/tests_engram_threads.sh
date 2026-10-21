#!/bin/bash
# Full-depth model, graphs, async engram fetch: host gather threads vs decode TPOT (pass 1 cold, pass 2 warm).
# Container paths (override via env): SM120_ROOT=/opt/sm120 (this repo), WORK=/work (work dir),
# HF_CKPT=/ckpt (original HF checkpoint shards, read-only).
SM120_ROOT=${SM120_ROOT:-/opt/sm120}; WORK=${WORK:-/work}; HF_CKPT=${HF_CKPT:-/ckpt}; export SM120_ROOT WORK HF_CKPT
set -u
B=${BATCHES:-8}
for th in ${THREADS_LIST:-1 8 64}; do for pass in cold warm; do
  name=th${th}_$pass
  SM120_ENGRAM_THREADS=$th SM120_ENGRAM_STATS=1 python3 $SM120_ROOT/tools/bench_e2e.py --ckpt $WORK/ckpt_full --batches $B \
      --prompt-len 256 --gen 64 --modes graph --max-num-batched-tokens 2048 --out $WORK/engram_th_$name.jsonl > $WORK/engram_th_$name.log 2>&1
  echo "== $name rc=$? $(grep -hE '^\| graph' $WORK/engram_th_$name.log | awk -F'|' '{print "B="$5" TPOT="$8}' | tr '\n' ' ')"
  grep -h "sm120_engram\] stats" $WORK/engram_th_$name.log | grep -oE "'(gpu_spin_iters_per_replay|graph_gather_ms_per_replay|graph_ready_ms_per_replay)': [^]]*\]?" | tr '\n' ' '; echo
done; done
