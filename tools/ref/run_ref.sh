#!/usr/bin/env bash
# Full reference pipeline inside the container (see README.md, docs/MTP_YOCO.md).
#   SRC=/ckpt WORK=/work REF_SRC=<ds41-inference/inference> ./run_ref.sh [tilelang|torch|check|all]
# YOCO test model (other checkpoint, longer prompt, separate outputs):
#   CKPT=/work/ckpt_yoco REFMP=/work/ref-mp1-yoco OUT=/work/yoco_ref PROMPT_JSON=/work/prompt_ids_yoco.json \
#   MAKE_ARGS="--preset yoco" RUN_ARGS="--prompt-tokens 2048 --max-seq-len 4096" CAPTURE_LAYER=all ./run_ref.sh all
# Container paths (override via env): SM120_ROOT=/opt/sm120 (this repo), WORK=/work (work dir),
# HF_CKPT=/ckpt (original HF checkpoint shards, read-only).
SM120_ROOT=${SM120_ROOT:-/opt/sm120}; WORK=${WORK:-/work}; HF_CKPT=${HF_CKPT:-/ckpt}; export SM120_ROOT WORK HF_CKPT
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="${SRC:-$HF_CKPT}"
WORK="${WORK:-$WORK}"
REF_SRC="${REF_SRC:-$HERE/../../third_party/ds41-inference/inference}"
CKPT="${CKPT:-$WORK/ckpt3}"                         # truncated HF checkpoint
REFMP="${REFMP:-$WORK/ref-mp1}"                     # reference-format checkpoint
OUT="${OUT:-$WORK/ref_out}"                         # -> $OUT.pt (tilelang), ${OUT}_torch.pt, ${OUT}_check.pt
PROMPT_JSON="${PROMPT_JSON:-$WORK/prompt_ids.json}" # written by the tilelang run, reused by the others if present
MAKE_ARGS="${MAKE_ARGS:-}"
RUN_ARGS="${RUN_ARGS:-}"
MODE="${1:-tilelang}"

if [ ! -f "$CKPT/model.safetensors.index.json" ]; then
    # shellcheck disable=SC2086
    python3 "$HERE/make_truncated_ckpt.py" --src "$SRC" --dst "$CKPT" $MAKE_ARGS
fi
if [ ! -f "$REFMP/model0-mp1.safetensors" ]; then
    python3 "$HERE/ref_convert.py" --hf-ckpt-path "$CKPT" --save-path "$REFMP"
fi

run() {  # $1 = REF_KERNELS mode, $2 = output file
    local prompt=(--save-prompt-json "$PROMPT_JSON")
    if [ "$1" != tilelang ] && [ -f "$PROMPT_JSON" ]; then
        prompt=(--prompt-ids-json "$PROMPT_JSON" --save-prompt-json "")
    fi
    # shellcheck disable=SC2086
    REF_KERNELS="$1" python3 "$HERE/ref_runner.py" \
        --ckpt "$REFMP" --hf-config "$CKPT/config.json" --ref-src "$REF_SRC" \
        --out "$2" "${prompt[@]}" $RUN_ARGS
}

case "$MODE" in
    tilelang) run tilelang "$OUT.pt" ;;
    torch) run torch "${OUT}_torch.pt" ;;
    check) run check "${OUT}_check.pt" ;;
    all)
        run tilelang "$OUT.pt"
        run torch "${OUT}_torch.pt"
        ;;
    *) echo "usage: $0 [tilelang|torch|check|all]" >&2; exit 1 ;;
esac
