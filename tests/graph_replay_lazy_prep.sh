#!/usr/bin/env bash
# Regression: decode CUDA graphs must survive b12x plans prepared lazily after capture (a first-seen longer
# prefill). Before the fix, b12x evicted the captured programs at the end of that preparation and the next
# graph replay died with "CUDA error: an illegal instruction was encountered".
#   graph_replay_lazy_prep.sh [ckpt ...]     (default: ckpt3, ckpt_yoco_engram, ckpt_full)
# Negative control (NEG=1): the same on the first ckpt with SM120_RETAIN_GRAPH_PROGRAMS=0, expected to FAIL.
# Container paths (override via env): SM120_ROOT=/opt/sm120 (this repo), WORK=/work (work dir).
SM120_ROOT=${SM120_ROOT:-/opt/sm120}; WORK=${WORK:-/work}
set -u
CKPTS=("$@"); [ ${#CKPTS[@]} -eq 0 ] && CKPTS=($WORK/ckpt3 $WORK/ckpt_yoco_engram $WORK/ckpt_full)
PORT=${PORT:-8125}
SARGS=${SARGS:---gpu-memory-utilization 0.9}
rc=0
run() {  # run <tag> <ckpt> [env...]
    local tag=$1 ck=$2; shift 2
    local log=$WORK/graph_replay_${tag}_$(basename $ck).log
    out=$(env "$@" bash $SM120_ROOT/server/with_server.sh $ck $PORT $log --graphs $SARGS -- \
          python3 $SM120_ROOT/tests/graph_replay_lazy_prep.py --url http://127.0.0.1:$PORT 2>&1 | grep -E "^(OK|FAIL)|did not")
    echo "$tag $(basename $ck): ${out:-no result (see $log)}"
    [[ $out == OK* ]]
}
for ck in "${CKPTS[@]}"; do
    [ -f $ck/config.json ] || { echo "skip $ck (missing)"; continue; }
    run graphs $ck || rc=1
done
if [ "${NEG:-0}" = 1 ]; then
    if run neg ${CKPTS[0]} SM120_RETAIN_GRAPH_PROGRAMS=0; then echo "negative control did not fail (trigger not reproduced)"; else echo "negative control failed as expected"; fi
fi
echo "graph_replay_lazy_prep: $([ $rc = 0 ] && echo ALL OK || echo FAILED)"
exit $rc
