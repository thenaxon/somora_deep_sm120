#!/bin/bash
# Runtime unit tests (no checkpoint): tiny-decode patch, P8 sampler op, GEMM/MoE graph tests.
# Container paths (override via env): SM120_ROOT=/opt/sm120 (this repo), WORK=/work (work dir),
# HF_CKPT=/ckpt (original HF checkpoint shards, read-only).
SM120_ROOT=${SM120_ROOT:-/opt/sm120}; WORK=${WORK:-/work}; HF_CKPT=${HF_CKPT:-/ckpt}; export SM120_ROOT WORK HF_CKPT
set -x
python3 -c "import sm120_patches as p; print(p.TINY_DECODE_POLICY, p.b12x_tiny_decode_patched())"
if python3 -c "import sys, sm120_patches as p; sys.exit(0 if p.b12x_tiny_decode_patched() else 1)"; then
  echo "b12x tiny-decode clamp patch already applied (serving image): skipping the unpatched run"
else
  python3 $SM120_ROOT/tests/test_b12x_tiny_decode_swiglu.py; echo "unpatched exit=$?"
  SITE=$(python3 -c 'import b12x, os; print(os.path.dirname(os.path.dirname(b12x.__file__)))')
  patch -p1 -d "$SITE" < $SM120_ROOT/patches/b12x-tiny-decode-swiglu.patch || exit 3
fi
python3 -c "import sm120_patches as p; print(p.TINY_DECODE_POLICY, p.b12x_tiny_decode_patched())"
python3 $SM120_ROOT/tests/test_b12x_tiny_decode_swiglu.py; echo "patched exit=$?"
python3 $SM120_ROOT/tests/test_b12x_tiny_decode_swiglu.py --shape v41 --bench; echo "bench exit=$?"
python3 $SM120_ROOT/patches/repro_b12x_tiny_decode_swiglu.py; echo "repro exit=$?"
pytest -q -x $SM120_ROOT/tests/test_gemm_moe_graphs.py 2>&1 | tail -30
pytest -q -x $SM120_ROOT/tests/test_fail_closed.py 2>&1 | tail -15
cd /opt/deepinfer/deepinfer/model_executor/layers/tilelang_ops
python3 -m pytest -q sampler.py -k "not benchmark" 2>&1 | tail -15
python3 -m pytest -q dflash.py dflash_copy.py store_mtp_info.py schedule_mtp_varlen.py fused_sample.py -k "not benchmark" 2>&1 | tail -15
