#!/bin/bash
# Engram + CUDA graphs on the YOCO engram test model, teacher-forced decode (same prefixes in every run):
#   eager vs eager (run-to-run noise), graph/async vs eager, graph/prestage vs eager,
#   graph/prestage with SM120_ENGRAM_PRESTAGE=0 (negative control: stale rows -> must diverge).
# Container paths (override via env): SM120_ROOT=/opt/sm120 (this repo), WORK=/work (work dir),
# HF_CKPT=/ckpt (original HF checkpoint shards, read-only).
SM120_ROOT=${SM120_ROOT:-/opt/sm120}; WORK=${WORK:-/work}; HF_CKPT=${HF_CKPT:-/ckpt}; export SM120_ROOT WORK HF_CKPT
set -u
CK=${CK:-$WORK/ckpt_yoco_engram}
PROMPT=${PROMPT:-$WORK/prompt_ids_yoco.json}
REF=${REF:-$WORK/yoco_engram_ref.pt}
EXTRA=${EXTRA:---max-num-batched-tokens 4096}
TAG=${TAG:-}
R="python3 $SM120_ROOT/tools/run_deepinfer.py --ckpt $CK --prompt-ids $PROMPT --ref $REF --max-model-len 16384 $EXTRA --no-hidden --teacher-force"
RC=0
run() { name=$1; shift; env "$@" $R ${GRAPHS:-} --out $WORK/eg${TAG}_$name.pt > $WORK/eg${TAG}_$name.log 2>&1; local rc=$?; [ $rc -eq 0 ] || RC=1; echo "$name rc=$rc $(grep -h 'generate' $WORK/eg${TAG}_$name.log | tail -1)"; }
run eager_tf
run eager_tf2
GRAPHS=--graphs run graph_async_tf SM120_ENGRAM_GRAPH=async
GRAPHS=--graphs run graph_prestage_tf SM120_ENGRAM_GRAPH=prestage
GRAPHS=--graphs run graph_nopre_tf SM120_ENGRAM_GRAPH=prestage SM120_ENGRAM_PRESTAGE=0
for x in eager_tf2 graph_async_tf graph_prestage_tf graph_nopre_tf; do
  [ $x = graph_nopre_tf ] && { note=" (NEGATIVE CONTROL: SM120_ENGRAM_PRESTAGE=0 serves stale rows, must diverge)"; chk=diverge; } || { note=""; chk=none; }
  echo "==== $x vs eager_tf$note"
  python3 $SM120_ROOT/tools/compare.py --di $WORK/eg${TAG}_$x.pt --ref $WORK/eg${TAG}_eager_tf.pt --ref2 '' \
      --allow-missing-hidden --decode-check $chk > $WORK/eg${TAG}_$x.cmp 2>&1
  crc=$?
  grep -E "decode (first|all)|decoded x|identical prefix|decode check|INVALID" $WORK/eg${TAG}_$x.cmp
  [ $crc -eq 0 ] || { RC=1; echo "  compare.py exit $crc"; }
done
echo "tests_engram_graph: $([ $RC -eq 0 ] && echo ALL OK || echo FAILED)"
exit $RC
