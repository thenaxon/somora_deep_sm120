"""b12x fast path for DeepGEMM's FP8 x FP8 (MXFP8, UE8M0 1x32 / 32x32) dense linear.

Mapping:  deep_gemm.fp8_fp4_gemm_nt((x_fp8, x_sf), (w_fp8, w_sf), d, c, recipe=(1, 1, 32))
     ->   b12x.gemm.block_fp8_linear  (native SM120 MXFP8 dense GEMM, same numerics class:
          e4m3 x e4m3 products, UE8M0 per-32 scales, fp32 accumulation, bf16 output).

* Weights are packed once per (data_ptr, shape) on first use. ``pack_weight`` keeps the E4M3 values
  in place (``weight.contiguous()`` of a contiguous parameter is the same storage), only the
  expanded scale grids are new (N*K/32 bytes + MMA swizzle copy).
  - 32x32-block-uniform SFs (the normal case: checkpoint 32x32 scales broadcast by our
    ``transform_sf_into_required_layout``) use the public ``pack_weight(..., block_size=(32, 32))``.
  - Row-varying SFs (e.g. after a row permutation such as FlashMLA's q_b permute) are packed from
    the per-row grid via b12x's internal MXFP8Rows builder (same object the public packer returns).
* Activations: DeepInfer hands us MXFP8 (e4m3 + packed UE8M0 per 32). b12x's linear quantises a bf16
  source itself, so we dequantise to bf16 first. This round trip is lossless: e4m3*2^e is exact in
  bf16, and re-quantising with ceil-UE8M0(amax/448) reproduces values that are exactly representable
  (the new scale is <= the old one by a power of two and amax/scale stays <= 448).
* Plans: one per (N, K, bucket) from SM120_B12X_LINEAR_BUCKETS, prepared lazily with b12x's default
  (heuristic, non-autotuned) configuration; scratch buffers are shared between plans with identical
  scratch specs. SM120_B12X_AUTOTUNE=1 prepares each plan through a b12x PreparationSession instead
  (candidate race, decisions cached under SM120_B12X_TUNE_CACHE, default /cache/b12x/preparation).
* CUDA graphs: packing, plan preparation (JIT) and scratch allocation happen on the first call of a
  (weight, bucket), which DeepInfer's eager warm-up before every graph capture provides; bind/run then
  only launch retained kernels (no allocation besides the graph-pool output, no host sync). A first
  call during capture raises instead of failing inside the CUDA driver.
"""

from __future__ import annotations

import torch

from . import _dequant_triton, _env, _sf

_WEIGHTS: dict = {}
_PLANS: dict = {}
_SCRATCH: dict = {}
_UNSUPPORTED: set = set()
_SESSION: dict = {}
_STATS = {"weights": 0, "plans": 0, "tuned": 0, "scratch_bytes": 0}


def stats() -> dict:
    return dict(weights=_STATS["weights"], plans=_STATS["plans"], tuned=_STATS["tuned"],
                scratch_mib=_STATS["scratch_bytes"] >> 20)


def _capturing() -> bool:
    return torch.cuda.is_current_stream_capturing()


def _not_in_capture(what: str) -> None:
    if _capturing():
        raise RuntimeError(f"deep_gemm shim: {what} first requested during CUDA graph capture; run the same shape "
                           "eagerly first (DeepInfer's per-size warm-up does) so packing/JIT happen outside capture")


def _api():
    import b12x.preparation  # noqa: F401  (registers the lazy default preparer)
    from b12x.gemm import block_fp8_linear as bfl
    from b12x.preparation import require_prepared

    return bfl, require_prepared


def eligible(a: torch.Tensor, sfa, b: torch.Tensor, sfb, d: torch.Tensor, alpha, epilogue) -> str | None:
    """None if the call can go to b12x, else the reason."""
    if epilogue is not None:
        return "custom epilogue"
    if alpha is not None:
        return "alpha"
    if a.dtype != torch.float8_e4m3fn or b.dtype != torch.float8_e4m3fn:
        return f"dtypes a={a.dtype} b={b.dtype} (only e4m3 x e4m3)"
    if a.dim() != 2 or b.dim() != 2:
        return "non-2D operands"
    if sfa is None or sfb is None:
        return "missing scale factors"
    if d.dtype != torch.bfloat16:
        return f"output dtype {d.dtype}"
    n, k = b.shape
    if k % 128 != 0:
        return f"K={k} not a multiple of 128"
    if not b.is_contiguous():
        return "non-contiguous weight"
    if not a.is_cuda:
        return "cpu tensor"
    return None


def _scale_bytes_32(sf: torch.Tensor, mn: int, k: int, gran_k: int | None) -> torch.Tensor:
    scales, g = _sf.sf_to_group_scales(sf, mn, k, gran_k)
    if g == 128:
        scales = scales.repeat_interleave(4, dim=-1)[..., : k // 32]
    elif g != 32:
        raise ValueError(f"unsupported weight SF granularity {g}")
    return _sf.float_to_ue8m0_bytes(scales).contiguous()  # uint8 [mn, k/32]


def _pack(b: torch.Tensor, sfb: torch.Tensor, gran_k: int | None):
    key = (b.data_ptr(), tuple(b.shape), sfb.data_ptr(), tuple(sfb.shape), b.device)
    w = _WEIGHTS.get(key)
    if w is not None:
        return w
    _not_in_capture(f"b12x weight packing N,K={tuple(b.shape)}")
    bfl, _ = _api()
    n, k = map(int, b.shape)
    rows = _scale_bytes_32(sfb, n, k, gran_k)  # [n, k/32]
    uniform = False
    if n % 32 == 0:
        blk = rows.view(n // 32, 32, k // 32)
        uniform = bool((blk == blk[:, :1]).all().item())
    if uniform:
        block = rows.view(n // 32, 32, k // 32)[:, 0].contiguous().view(torch.float8_e8m0fnu)
        w = bfl.pack_weight(b, block, block_size=(32, 32))
        how = "pack_weight(32x32)"
    else:
        from b12x.gemm._shared.block_fp8 import BlockFP8LinearWeight
        from b12x.gemm._shared.wo_mxfp8 import MXFP8Rows, pack_mxfp8_scales_for_dense_gemm

        scale_rows = rows.view(1, n, k // 32).view(torch.float8_e8m0fnu)
        scale_mma = pack_mxfp8_scales_for_dense_gemm(scale_rows, m=n, k=k, num_groups=1)
        w = BlockFP8LinearWeight(
            weight=MXFP8Rows(values=b.contiguous(), scale_rows=scale_rows, scale_mma=scale_mma),
            in_features=k,
            out_features=n,
            block_size=(32, 32),
        )
        how = "per-row MXFP8Rows"
    _env.log_once(("b12x_pack", n, k, how), f"b12x block_fp8_linear weight N={n} K={k} packed via {how}")
    _WEIGHTS[key] = w
    _STATS["weights"] += 1
    return w


def _session(device: torch.device):
    s = _SESSION.get(device)
    if s is None:
        from b12x.preparation import PreparationSession

        s = PreparationSession(device=device, autotune=True, cache_dir=_env.tune_cache_dir(),
                               namespace={"user": "deepinfer-sm120", "op": "block_fp8_linear"})
        _SESSION[device] = s
    return s


def _prepare_tuned(plan, n: int, k: int, bucket: int, weight, device: torch.device) -> None:
    """Race b12x's candidate configs for this (N, K, bucket) on the real packed weight (cached on disk)."""
    from b12x.preparation import PreparedCall

    def make_call(state):
        q = state.query
        source = torch.empty((q.max_tokens, q.in_features), device=state.device, dtype=getattr(torch, q.source_dtype))
        output = torch.empty((q.max_tokens, q.out_features, 1), device=state.device,
                             dtype=getattr(torch, q.output_dtype))
        scratch = [torch.empty(spec.shape, dtype=spec.dtype, device=state.device)
                   for spec in state.scratch.scratch_specs()]
        binding = state.bind(scratch=scratch, source=source, packed_weight=weight, output=output)
        return PreparedCall(run=lambda: state.run_binding(binding), produce=lambda: source.normal_(std=0.25),
                            owners=(weight,))

    req = plan.request(name=f"sm120.block32.n{n}.k{k}.m{bucket}", prepare_call=make_call, benchmark_call=make_call)
    _session(device).prepare((req,))
    _STATS["tuned"] += 1


def _plan(n: int, k: int, bucket: int, device: torch.device, weight=None):
    key = (n, k, bucket, device)
    hit = _PLANS.get(key)
    if hit is not None:
        return hit
    _not_in_capture(f"b12x block_fp8_linear plan N={n} K={k} max_tokens={bucket}")
    bfl, require_prepared = _api()
    plan = bfl.plan(
        bfl.Caps(device=device, max_tokens=bucket, in_features=k, out_features=n, block_size=(32, 32))
    )
    if _env.autotune() and weight is not None:
        try:
            _prepare_tuned(plan, n, k, bucket, weight, device)
        except Exception as exc:  # noqa: BLE001  tuning is optional: fall back to the default config
            if _env.strict():
                raise
            _env.warn_once(("b12x_tune", n, k, bucket), f"b12x autotune N={n} K={k} m={bucket} failed "
                           f"({type(exc).__name__}: {exc}); using the default config")
    require_prepared(plan, "gemm.block_fp8_linear")  # default config (or the tuned one), compiles on first use
    specs = tuple(plan.scratch_specs())
    skey = (device, tuple((tuple(s.shape), s.dtype) for s in specs))
    scratch = _SCRATCH.get(skey)
    if scratch is None:
        scratch = [torch.empty(tuple(s.shape), dtype=s.dtype, device=device) for s in specs]
        _SCRATCH[skey] = scratch
        _STATS["scratch_bytes"] += sum(t.numel() * t.element_size() for t in scratch)
    _env.log_once(("b12x_plan", key), f"b12x block_fp8_linear plan N={n} K={k} max_tokens={bucket}")
    _PLANS[key] = (plan, scratch)
    _STATS["plans"] += 1
    return plan, scratch


def linear(a: torch.Tensor, sfa: torch.Tensor, gran_k_a: int | None,
           b: torch.Tensor, sfb: torch.Tensor, gran_k_b: int | None) -> torch.Tensor:
    """(e4m3 [M, K], SF) x (e4m3 [N, K], SF)^T -> bf16 [M, N] via b12x."""
    bfl, _ = _api()
    m, k = map(int, a.shape)
    n = int(b.shape[0])
    w = _pack(b, sfb, gran_k_b)
    x = _dequant_triton.dequant_fp8_rows(a, sfa, gran_k_a)  # one kernel, bit-identical to _sf.dequant_fp8
    if x is None:
        x = _sf.dequant_fp8(a, sfa, gran_k_a, torch.bfloat16).contiguous()
    out = torch.empty((m, n, 1), dtype=torch.bfloat16, device=a.device)
    if m == 0:
        return out[:, :, 0]
    buckets = _env.linear_buckets()
    top = buckets[-1]
    for s in range(0, m, top):
        e = min(m, s + top)
        rows = e - s
        bucket = next(bk for bk in buckets if bk >= rows)
        plan, scratch = _plan(n, k, bucket, a.device, w)
        binding = bfl.bind(plan, scratch=scratch, source=x[s:e], packed_weight=w, output=out[s:e])
        bfl.run(binding=binding)
    return out[:, :, 0]


def try_linear(a, sfa, gran_k_a, b, sfb, gran_k_b, d, alpha, epilogue) -> torch.Tensor | None:
    """Return the b12x result or None (caller falls back to torch)."""
    reason = eligible(a, sfa, b, sfb, d, alpha, epilogue)
    if reason is not None:
        _env.log_once(("b12x_skip", reason), f"fp8_fp4_gemm_nt -> torch ({reason})")
        return None
    key = (tuple(b.shape), b.data_ptr())
    if key in _UNSUPPORTED:
        return None
    try:
        return linear(a, sfa, gran_k_a, b, sfb, gran_k_b)
    except Exception as exc:  # noqa: BLE001
        # inside a CUDA graph capture (e.g. a first-time pack / plan, _not_in_capture): raise instead of recording the
        # torch path into the graph and marking the weight unsupported for good
        if _env.strict() or _capturing():
            raise
        _UNSUPPORTED.add(key)
        _env.warn_once(("b12x_fail", tuple(b.shape)), f"b12x block_fp8_linear failed for N,K={tuple(b.shape)}: "
                       f"{type(exc).__name__}: {exc}; using the torch reference for this weight")
        return None
