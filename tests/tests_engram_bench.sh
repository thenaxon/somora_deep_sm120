#!/bin/bash
# End-to-end decode cost of Engram on the YOCO test model: with engram (layer 1, host mode; CUDA graphs with the
# in-graph "async" fetch or "prestage", and eager) vs the same model without engram. Every engram config is run
# twice with the same prompts: pass 1 faults the rows in (cold), pass 2 hits the page cache (warm).
# Container paths (override via env): SM120_ROOT=/opt/sm120 (this repo), WORK=/work (work dir),
# HF_CKPT=/ckpt (original HF checkpoint shards, read-only).
SM120_ROOT=${SM120_ROOT:-/opt/sm120}; WORK=${WORK:-/work}; HF_CKPT=${HF_CKPT:-/ckpt}; export SM120_ROOT WORK HF_CKPT
set -u
B=${BATCHES:-1,16,64}
bench() { name=$1; ck=$2; modes=$3; shift 3
  env "$@" SM120_ENGRAM_STATS=1 python3 $SM120_ROOT/tools/bench_e2e.py --ckpt $ck --batches $B --prompt-len 256 --gen 64 \
      --modes $modes --out $WORK/engram_e2e_$name.jsonl > $WORK/engram_e2e_$name.log 2>&1
  echo "== $name rc=$?"; grep -hE "^\| (graph|eager)" $WORK/engram_e2e_$name.log
  grep -h "sm120_engram\] stats" $WORK/engram_e2e_$name.log | sed 's/.*| per fetch/   per fetch/' | cut -c1-220; }
bench noengram $WORK/ckpt_yoco graph,eager
bench async_cold $WORK/ckpt_yoco_engram graph SM120_ENGRAM_GRAPH=async
bench async_warm $WORK/ckpt_yoco_engram graph SM120_ENGRAM_GRAPH=async
bench prestage_warm $WORK/ckpt_yoco_engram graph SM120_ENGRAM_GRAPH=prestage
bench eager_warm $WORK/ckpt_yoco_engram eager
