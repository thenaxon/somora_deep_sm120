"""Runtime switches for the SM120 attention / indexer / top-k shims.

All switches are read at call time (not import time) so tests and the runner can flip them
per call via ``os.environ``.

    SM120_ATTN_IMPL      b12x (default) | torch
                         b12x: use b12x kernels where a mapping exists (sparse decode, sparse
                         prefill, fused decode/prefill cores, MXFP4 indexer logits), torch elsewhere.
                         torch: pure-torch fp32 reference for every function.
    SM120_ATTN_COMPUTE   bf16 (default) | fp8      b12x compressed_sparse_mla v41_compute_mode
    SM120_ATTN_PERMUTE   1 (default) | 0
                         1: permute_q_b_proj / permute_wv_proj reproduce FlashMLA's exact
                            weight permutations and the fused functions consume/produce the
                            permuted layouts (bit-for-bit layout contract of the SM100 kernel).
                         0: both permute functions are identity and the fused functions use
                            the natural [h, d] layouts (escape hatch if the weight scale-factor
                            layout used by the GEMM shim is not row-indexed).
                         MUST be identical at weight-load time and at run time.
    SM120_ATTN_STRICT    0 (default) | 1   1: raise instead of falling back b12x -> torch
    SM120_ATTN_CHUNK_MB  512 (default)     working-set budget of the torch paths (MiB)
    SM120_ATTN_B12X_LSE  0 (default) | 1   ask b12x for the (natural-log) LSE; otherwise the
                                            b12x path returns a NaN-filled LSE (DeepInfer
                                            discards LSE at every call site)
    SM120_ATTN_TF32      0 (default) | 1   allow TF32 in the torch prefill path (bf16 inputs are
                                            exact in TF32; P@V loses ~2^-11 relative)
    SM120_ATTN_DEBUG     0 (default) | 1   log every dispatch decision once

  Finer switches (; each defaults to SM120_ATTN_IMPL):
    SM120_ATTN_DECODE_IMPL   b12x | torch   sparse decode (flash_mla_with_kvcache, fused decode)
    SM120_ATTN_PREFILL_IMPL  b12x | torch   sparse prefill (flash_mla_sparse_fwd, fused prefill)
    SM120_INDEXER_IMPL       b12x | torch   DeepGEMM MQA-logits functions (b12x dsa_indexer MXFP4 score)
    SM120_ATTN_FUSED_IMPL    triton | torch (default: torch iff SM120_ATTN_IMPL=torch)  q unpermute/RMS/RoPE and inverse-RoPE/FP8-cast of
                             fused_norm_rope_attn_rope_cast
    SM120_ATTN_PREFILL_KV    auto (default) | fp8
                         how the bf16 prefill KV is re-encoded for b12x: auto puts every token that is
                         exactly representable as a V4.1 FP4 record (compressed-cache tokens) into a
                         288 B cache and everything else into a 528 B FP8 cache (exact for SWA tokens,
                         which DeepInfer casts through FP8 already); fp8 puts every token into the
                         528 B cache (lossy for FP4-origin tokens, one cache, fewer index lists)
    SM120_ATTN_PREFILL_ROWS  2048 (default)  query rows per b12x prefill launch (scratch ~ rows*H*1 KiB)
    SM120_INDEXER_SCRATCH_MB 1024 (default)  per-call scratch budget of the b12x indexer (rows chunking)
    SM120_INDEXER_FILL_TAIL  0 (default) | 1  b12x fp8_fp4_mqa_logits: also write -inf to the columns
                                              past the longest valid row (DeepGEMM leaves them
                                              uninitialized; DeepInfer always passes `end` to top-k)
"""

from __future__ import annotations

import contextlib
import logging
import os

import torch

logger = logging.getLogger("sm120_attn")
_LOGGED: set = set()


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip().lower()


def impl() -> str:
    v = _env("SM120_ATTN_IMPL", "b12x")
    if v not in ("b12x", "torch"):
        raise ValueError(f"SM120_ATTN_IMPL must be 'b12x' or 'torch', got {v!r}")
    return v


def _impl_override(name: str) -> str:
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return impl()
    if v not in ("b12x", "torch"):
        raise ValueError(f"{name} must be 'b12x' or 'torch', got {v!r}")
    return v


def fused_impl() -> str:
    v = _env("SM120_ATTN_FUSED_IMPL", "torch" if impl() == "torch" else "triton")
    if v not in ("triton", "torch"):
        raise ValueError(f"SM120_ATTN_FUSED_IMPL must be 'triton' or 'torch', got {v!r}")
    return v


def decode_impl() -> str:
    return _impl_override("SM120_ATTN_DECODE_IMPL")


def prefill_impl() -> str:
    return _impl_override("SM120_ATTN_PREFILL_IMPL")


def indexer_impl() -> str:
    return _impl_override("SM120_INDEXER_IMPL")


def prefill_kv_mode() -> str:
    v = _env("SM120_ATTN_PREFILL_KV", "auto")
    if v not in ("auto", "fp8"):
        raise ValueError(f"SM120_ATTN_PREFILL_KV must be 'auto' or 'fp8', got {v!r}")
    return v


def prefill_rows() -> int:
    return max(16, int(_env("SM120_ATTN_PREFILL_ROWS", "2048")))


def indexer_scratch_bytes() -> int:
    return max(16, int(_env("SM120_INDEXER_SCRATCH_MB", "1024"))) * (1 << 20)


def indexer_fill_tail() -> bool:
    return _env("SM120_INDEXER_FILL_TAIL", "0") in ("1", "true", "yes", "on")


def capturing() -> bool:
    """True while the current stream is being captured into a CUDA graph (no host syncs allowed)."""
    try:
        return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    except Exception:  # noqa: BLE001
        return False


def compute_mode() -> str:
    v = _env("SM120_ATTN_COMPUTE", "bf16")
    if v not in ("bf16", "fp8"):
        raise ValueError(f"SM120_ATTN_COMPUTE must be 'bf16' or 'fp8', got {v!r}")
    return v


def permute_enabled() -> bool:
    return _env("SM120_ATTN_PERMUTE", "1") not in ("0", "false", "no", "off")


def strict() -> bool:
    return _env("SM120_ATTN_STRICT", "0") in ("1", "true", "yes", "on")


def b12x_lse() -> bool:
    return _env("SM120_ATTN_B12X_LSE", "0") in ("1", "true", "yes", "on")


def allow_tf32() -> bool:
    return _env("SM120_ATTN_TF32", "0") in ("1", "true", "yes", "on")


def debug() -> bool:
    return _env("SM120_ATTN_DEBUG", "0") in ("1", "true", "yes", "on")


def chunk_bytes() -> int:
    return max(16, int(_env("SM120_ATTN_CHUNK_MB", "512"))) * (1 << 20)


def log_once(key, msg: str, level: int = logging.INFO) -> None:
    if key in _LOGGED:
        return
    _LOGGED.add(key)
    logger.log(level, msg)
    if debug() or level >= logging.WARNING:
        print(f"[sm120_attn] {msg}", flush=True)


def rows_per_chunk(bytes_per_row: int, total_rows: int) -> int:
    """Rows per chunk such that rows * bytes_per_row stays within the working-set budget."""
    if total_rows <= 0:
        return 1
    return max(1, min(total_rows, chunk_bytes() // max(1, int(bytes_per_row))))


@contextlib.contextmanager
def matmul_precision(tf32: bool):
    """Temporarily set TF32 for cuBLAS matmuls (eager only)."""
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


def align(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def next_pow2(x: int, lo: int = 1) -> int:
    b = lo
    while b < x:
        b *= 2
    return b
