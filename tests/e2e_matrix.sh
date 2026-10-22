#!/bin/bash
# End-to-end matrix. Each argument is "<tag> ENV=VAL ...". Exit status non-zero if a run fails or compare.py flags
# suspicious layers / an invalid comparison.
# Checkpoint / prompt / references via env (defaults: the 3-layer test model):
#   TEST_CKPT TEST_PROMPT TEST_REF TEST_REF2 TEST_ARGS (extra run_deepinfer args) TEST_CAPTURE (CAPTURE_LAYER)
#   TEST_L0=1 also runs compare_l0.py --layer $TEST_CAPTURE (needs fine taps) and prints its key lines.
# Container paths (override via env): SM120_ROOT=/opt/sm120 (this repo), WORK=/work (work dir),
# HF_CKPT=/ckpt (original HF checkpoint shards, read-only).
SM120_ROOT=${SM120_ROOT:-/opt/sm120}; WORK=${WORK:-/work}; HF_CKPT=${HF_CKPT:-/ckpt}; export SM120_ROOT WORK HF_CKPT
cd $SM120_ROOT
export B12X_AUTOTUNE=0
set -o pipefail
RC=0
CKPT=${TEST_CKPT:-$WORK/ckpt3}
PROMPT=${TEST_PROMPT:-$WORK/prompt_ids.json}
REF=${TEST_REF:-$WORK/ref_out.pt}
REF2=${TEST_REF2:-$WORK/ref_out_torch.pt}
ARGS=${TEST_ARGS:---max-model-len 16384 --gpu-memory-utilization 0.97}
run() {
  tag=$1; shift
  env CAPTURE_LAYER=${TEST_CAPTURE:-0} "$@" python3 $SM120_ROOT/tools/run_deepinfer.py --ckpt $CKPT --prompt-ids $PROMPT $ARGS \
      --out $WORK/matrix_$tag.pt > /tmp/matrix_$tag.log 2>&1 || { echo "== $tag FAILED"; grep -E 'Error|rror:' /tmp/matrix_$tag.log | tail -5; RC=1; return; }
  echo "== $tag ($*)  $(grep -oE 'init [0-9.]+s generate [0-9.]+s' /tmp/matrix_$tag.log)"
  python3 $SM120_ROOT/tools/compare.py --di $WORK/matrix_$tag.pt --ref $REF --ref2 $REF2 \
      --known-deviation "${TEST_KNOWN_DEVIATION-layer0}" > /tmp/matrix_$tag.cmp 2>&1
  local crc=$?
  [ $crc -eq 0 ] || { RC=1; echo "  compare.py exit $crc: $(grep -E 'INVALID|suspicious hidden layers' /tmp/matrix_$tag.cmp | tail -2 | tr '\n' ' ')"; }
  awk '/^== logits: ref2 vs ref/{exit} {print}' /tmp/matrix_$tag.cmp | grep -E 'prefill \(all|decoded (ref|x)|identical prefix' | head -4
  grep -E '^  (layer[0-9]+|final_norm) ' /tmp/matrix_$tag.cmp | awk -F'|' '{split($1,k," "); split($2,d," "); split($4,z," "); split($5,r," ")
      printf "    %-10s DI-ref %s noise %s ratio %s %s\n", k[1], d[1], z[1], r[1], r[2]}'
  grep -E 'b12x .* failed|unsupported|falling back|triton fused .* failed' /tmp/matrix_$tag.log | sort | uniq -c | head -5
  if [ -n "${TEST_L0:-}" ]; then
    python3 $SM120_ROOT/tools/compare_l0.py --layer ${TEST_CAPTURE} --di $WORK/matrix_$tag.pt --ref $REF --ref2 $REF2 > /tmp/matrix_$tag.l0 2>&1
    grep -E '^== layer|compressed positions|candidate positions|^  (DI|ref2) +(identical|-)|flagged' /tmp/matrix_$tag.l0 | head -40
  fi
}
for spec in "$@"; do eval "run $spec"; done
rm -f $WORK/matrix_*.pt
echo "e2e_matrix: $([ $RC -eq 0 ] && echo ALL OK || echo FAILED)"
exit $RC
