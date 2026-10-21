#!/bin/bash
# Full-depth model: decode cost of engram (layers 1 and 14) under CUDA graphs vs the same model without engram.
#   /work/ckpt_full           the full-depth test model (16 experts, engram 1+14)
#   /work/ckpt_full_noengram  the same truncation without engram (built here once)
# Container paths (override via env): SM120_ROOT=/opt/sm120 (this repo), WORK=/work (work dir),
# HF_CKPT=/ckpt (original HF checkpoint shards, read-only).
SM120_ROOT=${SM120_ROOT:-/opt/sm120}; WORK=${WORK:-/work}; HF_CKPT=${HF_CKPT:-/ckpt}; export SM120_ROOT WORK HF_CKPT
set -u
B=${BATCHES:-1,8,32}
if [ ! -f $WORK/ckpt_full_noengram/config.json ]; then
  python3 $SM120_ROOT/tools/ref/make_truncated_ckpt.py --src $HF_CKPT --dst $WORK/ckpt_full_noengram --layers 0-39 --num-experts 16 \
      --mtp 3 --mtp-num-experts 32 --allow-engram-drop > $WORK/ckpt_full_noengram.build.log 2>&1 || { echo build failed; tail -5 $WORK/ckpt_full_noengram.build.log; }
fi
bench() { name=$1; ck=$2; modes=$3; shift 3
  env "$@" SM120_ENGRAM_STATS=1 python3 $SM120_ROOT/tools/bench_e2e.py --ckpt $ck --batches $B --prompt-len 256 \
      --gen 64 --modes $modes --max-num-batched-tokens 2048 --out $WORK/engram_full_$name.jsonl > $WORK/engram_full_$name.log 2>&1
  echo "== $name rc=$?"; grep -hE "^\| (graph|eager)" $WORK/engram_full_$name.log
  grep -h "sm120_engram\] stats" $WORK/engram_full_$name.log | grep -oE "'graph_(fetches|gather_s)': [0-9.]+|prestage [0-9.]+ ms \(sync [0-9.]+, gather\+H2D [0-9.]+\)" | tr '\n' ' '; echo; }
# CFGS="name:ckpt:ENV=1,ENV2=2+name2:..." (default below)
DEFAULT="noengram:$WORK/ckpt_full_noengram:+async_cold:$WORK/ckpt_full:SM120_ENGRAM_GRAPH=async+async_warm:$WORK/ckpt_full:SM120_ENGRAM_GRAPH=async+prestage_warm:$WORK/ckpt_full:SM120_ENGRAM_GRAPH=prestage"
LIST="${CFGS:-$DEFAULT}"
for cfg in ${LIST//+/ }; do
  IFS=: read -r name ck envs <<< "$cfg"
  bench $name $ck graph ${envs//,/ }
done
