"""Opt-in FP8 (weight-only) lm_head for DeepInfer on SM120: SM120_LMHEAD_IMPL = bf16 (default) | fp8.

DeepInfer computes the logits in sampler_utils.get_logits as bf16_gemm_nt(hidden bf16 [M, K], head bf16 [V, K],
logits fp32 [M, V]). Decode is a pure weight stream (V=129280, K=5120: 1.32 GB per step, ~0.8 ms on a 5090).
With SM120_LMHEAD_IMPL=fp8 the head is quantised once to e4m3 with one fp32 scale per (row, 128 columns)
(amax / 448) and the logits come from a Triton kernel that streams the FP8 weight (0.66 GB): the raw e4m3
values (exact in bf16) are dotted with the bf16 hidden state on tensor cores and each 128-column partial is
multiplied by its fp32 scale. The only deviation from bf16 is the weight quantisation itself.

install() wraps deepinfer.model_executor.layers.sampler_utils.bf16_gemm_nt (the name get_logits calls): calls
with an fp32 output go to the FP8 path when enabled and M <= 64 rows (decode; the kernel is a weight
stream and only wins there: 0.41 vs 0.81-0.85 ms for M = 1-32); larger M keeps cuBLAS bf16. The quantised copy is cached per weight tensor (static) and
costs V*K bytes + V*K/128*4 (0.66 GB + 21 MB); it is built on the first eager call (graph warm-up).
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001
    triton = None

_CACHE: dict = {}
_LOGGED: set = set()
BLOCK_K = 128
MAX_M = 64  # weight-streaming regime (decode); larger M (e.g. MTP verify batches >= 11 requests) keeps bf16


def impl() -> str:
    v = os.environ.get("SM120_LMHEAD_IMPL", "bf16").strip().lower()
    if v not in ("bf16", "fp8"):
        raise ValueError(f"SM120_LMHEAD_IMPL must be 'bf16' or 'fp8', got {v!r}")
    return v


def _log_once(key, msg):
    if key not in _LOGGED:
        _LOGGED.add(key)
        print(f"[sm120_lmhead] {msg}", flush=True)


def quantize_weight(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 [V, K] -> (e4m3 [V, K], fp32 scales [V, K/128]); scale = amax/448 per (row, 128 columns)."""
    V, K = w.shape
    assert K % BLOCK_K == 0, K
    out8 = torch.empty((V, K), dtype=torch.float8_e4m3fn, device=w.device)
    sf = torch.empty((V, K // BLOCK_K), dtype=torch.float32, device=w.device)
    step = 8192
    for a in range(0, V, step):  # bounded fp32 working set
        x = w[a:a + step].float().view(-1, K // BLOCK_K, BLOCK_K)
        s = x.abs().amax(-1).clamp_min(1e-12) / 448.0
        out8[a:a + step] = (x / s.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(-1, K)
        sf[a:a + step] = s
    return out8, sf


def _quantized(w: torch.Tensor):
    key = (w.data_ptr(), tuple(w.shape), w.dtype)
    hit = _CACHE.get(key)
    if hit is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("FP8 lm_head weight not quantised before CUDA graph capture")
        hit = quantize_weight(w)
        _CACHE[key] = hit
        _log_once(("quant", key), f"quantised lm_head {tuple(w.shape)} to FP8 (per-row-128 fp32 scales)")
    return hit


if triton is not None:

    @triton.jit
    def _lmhead_fp8_kernel(x_ptr, x_s, w_ptr, w_s, sf_ptr, sf_s, o_ptr, o_s, M, V, K,
                           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pn = tl.program_id(0)
        pm = tl.program_id(1)
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        mm = rm < M
        nm = rn < V
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            rk = k0 + tl.arange(0, BK)
            x = tl.load(x_ptr + rm[:, None].to(tl.int64) * x_s + rk[None, :], mask=mm[:, None], other=0.0)
            w = tl.load(w_ptr + rn[:, None].to(tl.int64) * w_s + rk[None, :], mask=nm[:, None], other=0.0)
            s = tl.load(sf_ptr + rn.to(tl.int64) * sf_s + k0 // BK, mask=nm, other=0.0)
            part = tl.dot(x.to(tl.bfloat16), tl.trans(w.to(tl.bfloat16)))
            acc += part * s[None, :]
        tl.store(o_ptr + rm[:, None].to(tl.int64) * o_s + rn[None, :], acc, mask=mm[:, None] & nm[None, :])


def lmhead_fp8(x: torch.Tensor, w8: torch.Tensor, sf: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """x bf16 [M, K], w8 e4m3 [V, K], sf fp32 [V, K/128] -> out fp32 [M, V] (unit inner strides)."""
    M, K = x.shape
    V = w8.shape[0]
    if M == 0:
        return out
    if M > MAX_M:
        raise ValueError(f"lmhead_fp8 supports M <= {MAX_M} (got {M})")
    BM, BN, warps, stages = (16 if M <= 16 else (32 if M <= 32 else 64)), 64, 4, 4
    grid = (triton.cdiv(V, BN), triton.cdiv(M, BM))
    _lmhead_fp8_kernel[grid](x, x.stride(0), w8, w8.stride(0), sf, sf.stride(0), out, out.stride(0), M, V, K,
                             BM=BM, BN=BN, BK=BLOCK_K, num_warps=warps, num_stages=stages)
    return out


def logits(hidden: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> bool:
    """FP8 logits into out (fp32 [M, V]); False if the FP8 path does not apply."""
    if triton is None or impl() != "fp8" or hidden.device.type != "cuda":
        return False
    if (out.dtype != torch.float32 or weight.dim() != 2 or hidden.dim() != 2 or hidden.shape[0] > MAX_M
            or weight.shape[1] % BLOCK_K or hidden.stride(1) != 1 or out.stride(1) != 1 or weight.stride(1) != 1):
        return False
    w8, sf = _quantized(weight)
    x = hidden if hidden.dtype == torch.bfloat16 else hidden.to(torch.bfloat16)
    lmhead_fp8(x, w8, sf, out)
    _log_once("fp8_on", f"lm_head -> FP8 weight-only Triton kernel ({tuple(weight.shape)})")
    return True


def install() -> bool:
    """Wrap sampler_utils.bf16_gemm_nt (used by get_logits). Idempotent; inert unless SM120_LMHEAD_IMPL=fp8."""
    try:
        from deepinfer.model_executor.layers import sampler_utils as su
    except Exception as exc:  # noqa: BLE001
        _log_once("noimport", f"sampler_utils not importable ({exc}); FP8 lm_head not installed")
        return False
    orig = su.bf16_gemm_nt
    if getattr(orig, "_sm120_lmhead", False):
        return True

    def bf16_gemm_nt(a, b, d, *args, **kwargs):
        if not args and not kwargs and logits(a, b, d):
            return None
        return orig(a, b, d, *args, **kwargs)

    bf16_gemm_nt._sm120_lmhead = True
    su.bf16_gemm_nt = bf16_gemm_nt
    return True
