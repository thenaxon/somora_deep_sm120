#!/bin/bash
# Run-to-run determinism check: runs tools/run_deepinfer.py twice per config on $WORK/ckpt3 and compares the two
# outputs bit for bit (prefill/decode logits, hidden taps, token ids and argmax). Each argument is one env config:
#   determinism.sh "SM120_MOE_DETERMINISTIC=1" "SM120_MOE_IMPL=torch" "~"     ('' = defaults)
# A config prefixed with "~" is a control that is expected to differ (the default b12x fused_moe is not run-to-run
# deterministic: atomic top-k combine): it is reported, but does not affect the exit status.
# Exit status: non-zero if a run fails or a (non-control) config is not bit-identical.
# Container paths (override via env): SM120_ROOT=/opt/sm120 (this repo), WORK=/work (work dir),
# HF_CKPT=/ckpt (original HF checkpoint shards, read-only).
SM120_ROOT=${SM120_ROOT:-/opt/sm120}; WORK=${WORK:-/work}; HF_CKPT=${HF_CKPT:-/ckpt}; export SM120_ROOT WORK HF_CKPT
set -uo pipefail
cd $SM120_ROOT
export B12X_AUTOTUNE=0
RC=0
for spec in "$@"; do
  control=0; cfg=$spec
  case $spec in "~"*) control=1; cfg=${spec#\~};; esac
  rm -f $WORK/det_1.pt $WORK/det_2.pt
  failed=0
  for r in 1 2; do
    env $cfg python3 $SM120_ROOT/tools/run_deepinfer.py --ckpt $WORK/ckpt3 --prompt-ids $WORK/prompt_ids.json --max-model-len 16384 \
        --gpu-memory-utilization 0.97 --out $WORK/det_$r.pt > /tmp/det_$r.log 2>&1 || { echo "== '$cfg' run $r FAILED"; tail -3 /tmp/det_$r.log; failed=1; }
  done
  if [ $failed -eq 1 ] || [ ! -f $WORK/det_1.pt ] || [ ! -f $WORK/det_2.pt ]; then RC=1; continue; fi
  echo "== '$cfg'$([ $control -eq 1 ] && echo ' (control: expected to differ)')"
  python3 - <<'PY'
import os
import sys

import torch

w = os.environ.get("WORK", "/work")
a = torch.load(f"{w}/det_1.pt", map_location="cpu", weights_only=False)
b = torch.load(f"{w}/det_2.pt", map_location="cpu", weights_only=False)
n_cmp, n_diff = 0, 0
def walk(x, y, k=""):
    global n_cmp, n_diff
    if isinstance(x, dict):
        for kk in x:
            if kk in ("meta", "timings"):
                continue
            if kk not in y:
                print(f"  {k}.{kk}: missing in run 2  <-- differs"); n_diff += 1; continue
            walk(x[kk], y[kk], f"{k}.{kk}" if k else str(kk))
    elif isinstance(x, (list, tuple)) and x and not isinstance(x[0], (int, float)):
        for i, (u, v) in enumerate(zip(x, y)):
            walk(u, v, f"{k}[{i}]")
    elif torch.is_tensor(x):
        n_cmp += 1
        if not torch.is_tensor(y) or x.shape != y.shape or x.dtype != y.dtype:
            print(f"  {k:<40} shape/dtype differs  <-- differs"); n_diff += 1; return
        if x.is_floating_point():
            d = (x.float() - y.float()).abs().max().item() if x.numel() else 0.0
            same = torch.equal(x, y)
            print(f"  {k:<40} {tuple(x.shape)} max|d|={d:.3e}{'' if same else '  <-- differs'}")
        else:
            same = torch.equal(x, y)  # token ids, argmax: integer tensors compared exactly
            print(f"  {k:<40} {tuple(x.shape)} {x.dtype} {'equal' if same else 'DIFFER  <-- differs'}")
        n_diff += not same
    elif isinstance(x, (list, int, float, str)) and x != y:
        n_cmp += 1; n_diff += 1
        print(f"  {k:<40} differs: {str(x)[:60]} vs {str(y)[:60]}")
walk(a, b)
print(f"  compared {n_cmp} tensors/values, {n_diff} differ")
sys.exit(2 if n_cmp == 0 else (1 if n_diff else 0))
PY
  rc=$?
  if [ $control -eq 1 ]; then
    echo "  control: $([ $rc -eq 1 ] && echo 'differs, as expected' || echo "rc=$rc (identical this time or nothing compared)")"
    [ $rc -eq 2 ] && RC=1
  else
    echo "  $([ $rc -eq 0 ] && echo BIT-IDENTICAL || echo "NOT DETERMINISTIC (rc=$rc)")"
    [ $rc -eq 0 ] || RC=1
  fi
done
rm -f $WORK/det_1.pt $WORK/det_2.pt
exit $RC
