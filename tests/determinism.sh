#!/bin/bash
# Run-to-run determinism check: runs tools/run_deepinfer.py twice per config on $WORK/ckpt3 and diffs the two
# outputs (prefill/decode logits, hidden taps, tokens). Each argument is one env config, e.g.
#   determinism.sh "SM120_MOE_DETERMINISTIC=1" "SM120_MOE_IMPL=torch" ""   ('' = defaults)
# The default b12x fused_moe is not run-to-run deterministic (atomic top-k combine); the two configs above are.
# Container paths (override via env): SM120_ROOT=/opt/sm120 (this repo), WORK=/work (work dir),
# HF_CKPT=/ckpt (original HF checkpoint shards, read-only).
SM120_ROOT=${SM120_ROOT:-/opt/sm120}; WORK=${WORK:-/work}; HF_CKPT=${HF_CKPT:-/ckpt}; export SM120_ROOT WORK HF_CKPT
cd $SM120_ROOT
export B12X_AUTOTUNE=0
for cfg in "$@"; do
  for r in 1 2; do
    env $cfg python3 $SM120_ROOT/tools/run_deepinfer.py --ckpt $WORK/ckpt3 --prompt-ids $WORK/prompt_ids.json --max-model-len 16384 \
        --gpu-memory-utilization 0.97 --out $WORK/det_$r.pt > /tmp/det_$r.log 2>&1 || { echo "$cfg run $r FAILED"; tail -3 /tmp/det_$r.log; }
  done
  echo "== $cfg"
  python3 - <<'PY'
import os
import torch
a = torch.load(f"{os.environ.get('WORK', '/work')}/det_1.pt", map_location="cpu", weights_only=False)
b = torch.load(f"{os.environ.get('WORK', '/work')}/det_2.pt", map_location="cpu", weights_only=False)
def walk(x, y, k=""):
    if isinstance(x, dict):
        for kk in x:
            if kk in y: walk(x[kk], y[kk], f"{k}.{kk}" if k else str(kk))
    elif isinstance(x, (list, tuple)) and x and not isinstance(x[0], (int, float)):
        for i, (u, v) in enumerate(zip(x, y)): walk(u, v, f"{k}[{i}]")
    elif torch.is_tensor(x) and x.is_floating_point() and x.shape == y.shape:
        d = (x.float() - y.float()).abs().max().item()
        print(f"  {k:<40} {tuple(x.shape)} max|d|={d:.3e}{'' if d == 0 else '  <-- differs'}")
    elif isinstance(x, list) and x != y:
        print(f"  {k:<40} tokens differ: {x[:12]} vs {y[:12]}")
walk(a, b)
PY
done
rm -f $WORK/det_*.pt
