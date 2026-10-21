"""Runtime switches of the SM120 deep_gemm shim (read at call time, never at import time).

    SM120_GEMM_IMPL        b12x (default) | torch
                           b12x : FP8 block-scaled linears (fp8_fp4_gemm_nt with FP8 weights) go through
                                  b12x.gemm.block_fp8_linear (native SM120 MXFP8 GEMM).
                                  Everything without a validated b12x mapping uses the torch path.
                           torch: exact dequant-to-bf16 + cuBLAS reference for every function.
    SM120_GEMM_STRICT      0 (default) | 1   1: raise instead of falling back b12x -> torch
    SM120_GEMM_DEBUG       0 (default) | 1   print every dispatch decision once
    SM120_DEQUANT_CACHE    einsum (default) | 1 | 0
                           cache the bf16 dequantisation of static FP8/FP4 weights used by the torch
                           path: einsum = only fp8_einsum weights, 1 = all weights, 0 = never.
    SM120_B12X_LINEAR_BUCKETS  16,256,2048,8192 (default)
                           max_tokens capacities of the b12x block-FP8 plans (one JIT compile per
                           (N, K, bucket)); rows above the largest bucket are processed in chunks.
    SM120_B12X_AUTOTUNE    0 (default) | 1   1: prepare each block_fp8_linear plan through a b12x
                           PreparationSession (candidate race on the real weight) instead of the lazy
                           default config. Decisions are cached in SM120_B12X_TUNE_CACHE.
    SM120_B12X_TUNE_CACHE  /cache/b12x/preparation (default when /cache is writable, else b12x's default
                           <B12X_COMPILE_CACHE_DIR>/preparation)
    SM120_MK_ALIGNMENT     128 (default)   value of get_theoretical_mk_alignment_for_contiguous_layout
                           (DeepGEMM returns 128 on every non-SM100 arch)
    SM120_CHUNK_MB         512 (default)   working-set budget for the fp32 paths of the torch reference
"""

from __future__ import annotations

import os

_LOGGED: set = set()


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip().lower()


def _flag(name: str, default: str = "0") -> bool:
    return _env(name, default) in ("1", "true", "yes", "on")


def gemm_impl() -> str:
    v = _env("SM120_GEMM_IMPL", "b12x")
    if v not in ("b12x", "torch"):
        raise ValueError(f"SM120_GEMM_IMPL must be 'b12x' or 'torch', got {v!r}")
    return v


def strict() -> bool:
    return _flag("SM120_GEMM_STRICT")


def debug() -> bool:
    return _flag("SM120_GEMM_DEBUG")


def dequant_cache_mode() -> str:
    v = _env("SM120_DEQUANT_CACHE", "einsum")
    if v in ("0", "false", "off", "no", "none"):
        return "none"
    if v in ("1", "true", "on", "yes", "all"):
        return "all"
    return "einsum"


def linear_buckets() -> tuple[int, ...]:
    raw = os.environ.get("SM120_B12X_LINEAR_BUCKETS", "16,256,2048,8192")
    vals = sorted({max(1, int(x)) for x in raw.split(",") if x.strip()})
    return tuple(vals) or (16, 256, 2048, 8192)


def autotune() -> bool:
    return _flag("SM120_B12X_AUTOTUNE")


def tune_cache_dir() -> str | None:
    v = os.environ.get("SM120_B12X_TUNE_CACHE")
    if v:
        return v
    root = os.environ.get("SM120_CACHE_ROOT", "/cache")
    if os.path.isdir(root) and os.access(root, os.W_OK):
        return os.path.join(root, "b12x", "preparation")
    return None


def mk_alignment() -> int:
    return max(1, int(_env("SM120_MK_ALIGNMENT", "128")))


def chunk_bytes() -> int:
    return max(16, int(_env("SM120_CHUNK_MB", "512"))) << 20


def log_once(key, msg: str) -> None:
    if key in _LOGGED:
        return
    _LOGGED.add(key)
    if debug():
        print(f"[sm120 deep_gemm] {msg}", flush=True)


def warn_once(key, msg: str) -> None:
    if key in _LOGGED:
        return
    _LOGGED.add(key)
    print(f"[sm120 deep_gemm] WARNING: {msg}", flush=True)
