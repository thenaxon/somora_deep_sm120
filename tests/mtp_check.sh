#!/bin/bash
# MTP/DSpark draft path check: nextn 0 / 5 eager / 5 graphs on an MTP checkpoint, greedy tokens compared.
# Container paths (override via env): SM120_ROOT=/opt/sm120 (this repo), WORK=/work (work dir),
# HF_CKPT=/ckpt (original HF checkpoint shards, read-only).
SM120_ROOT=${SM120_ROOT:-/opt/sm120}; WORK=${WORK:-/work}; HF_CKPT=${HF_CKPT:-/ckpt}; export SM120_ROOT WORK HF_CKPT
cd $SM120_ROOT
export B12X_AUTOTUNE=0
CKPT=${TEST_CKPT:-$WORK/ckpt3_mtp}
for spec in "n0 --nextn 0" "n5 --nextn 5" "n5g --nextn 5 --graphs" "n0g --nextn 0 --graphs" "n5_attntorch --nextn 5 SM120_ATTN_IMPL=torch"; do
  set -- $spec; tag=$1; shift
  args=""; envs=""
  for a in "$@"; do case $a in *=*) envs="$envs $a";; *) args="$args $a";; esac; done
  env $envs SM120_ATTN_DEBUG=0 python3 $SM120_ROOT/tools/run_deepinfer.py --ckpt $CKPT --prompt-ids $WORK/prompt_ids.json \
      --max-model-len 16384 --gpu-memory-utilization 0.95 --no-hidden $args --out $WORK/mtp_$tag.pt > /tmp/mtp_$tag.log 2>&1
  rc=$?
  echo "== $tag rc=$rc ($spec)  $(grep -oE 'init [0-9.]+s generate [0-9.]+s' /tmp/mtp_$tag.log)"
  grep -E 'accept|nextn|NEXTN|draft' /tmp/mtp_$tag.log | grep -v -i warn | tail -3
  grep -E 'b12x .* failed|unsupported|falling back|triton fused .* failed|Error|rror:' /tmp/mtp_$tag.log | sort | uniq -c | head -8
done
python3 - <<'PY'
import os
import torch, os
outs = {}
for tag in ("n0", "n5", "n5g", "n0g", "n5_attntorch"):
    p = f"{os.environ.get('WORK', '/work')}/mtp_{tag}.pt"
    if os.path.exists(p):
        d = torch.load(p, map_location="cpu", weights_only=False)
        toks = d["decoded_ids"].tolist()
        acc = (d.get("meta") or {}).get("accepted_log")
        outs[tag] = toks
        print(f"{tag:<14} {toks}  accepted={str(acc)[:120]}")
base = outs.get("n0")
for tag, t in outs.items():
    if base is not None:
        same = sum(a == b for a, b in zip(base, t))
        print(f"  {tag:<14} vs n0: {same}/{min(len(base), len(t))} positions equal")
PY
rm -f $WORK/mtp_*.pt
