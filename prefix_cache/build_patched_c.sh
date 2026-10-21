#!/usr/bin/env bash
# Build DeepInfer's _C with patches/deepinfer-sequence-block-table.patch into a persistent copy (inside the
# container; /opt/deepinfer itself is part of the image and stays untouched). Run once; later runs reuse it with
#   PYTHONPATH=$SM120_ROOT/sm120:/work/deepinfer_pc
# Image build alternative: apply the patch to src/ before `RUN bash scripts/build.sh build` in docker/Dockerfile.sm120.
# Container paths (override via env): SM120_ROOT=/opt/sm120 (this repo), WORK=/work (work dir),
# HF_CKPT=/ckpt (original HF checkpoint shards, read-only).
SM120_ROOT=${SM120_ROOT:-/opt/sm120}; WORK=${WORK:-/work}; HF_CKPT=${HF_CKPT:-/ckpt}; export SM120_ROOT WORK HF_CKPT
set -euo pipefail
DST="${DST:-$WORK/deepinfer_pc}"
PATCH=$SM120_ROOT/patches/deepinfer-sequence-block-table.patch
if [ ! -f "$DST/.prefix_cache_patched" ]; then
    rm -rf "$DST"
    cp -a /opt/deepinfer "$DST"          # includes the image's build/ dir -> incremental rebuild of one file
    cd "$DST"
    if command -v patch >/dev/null; then patch -p1 < "$PATCH"; else git apply -p1 "$PATCH"; fi
    MAX_JOBS="${MAX_JOBS:-16}" DEEPINFER_ACCELERATOR=nvidia bash scripts/build.sh build
    touch .prefix_cache_patched
fi
cd /
PYTHONPATH="$SM120_ROOT/sm120:$DST" python3 -c "
import deepinfer, deepinfer._C.engine as E
assert hasattr(E.Sequence, 'block_table'), 'block_table binding missing'
print('patched _C OK:', deepinfer.__file__)"
