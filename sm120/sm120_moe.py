"""SM120 routed-expert backends for DeepInfer's ``DeepseekMoE`` (v4_modules/mlp.py).

Installed by sm120_patches (wraps DeepseekMoE.__init__ / DeepseekMoE.forward). Selected with

    SM120_MOE_IMPL = b12x   (default when SM120_GEMM_IMPL=b12x)
                     b12x.moe.fused_moe, recipe w4a8_mx (MXFP4 e2m1 weights + UE8M0/32 scales, MXFP8
                     activations), source_format fp4_e8m0_k32, w13_layout "w31". No dispatch/combine at
                     all for EP=1 (the whole routed MoE is one fused kernel); the shared expert and the
                     gate stay DeepInfer's own code.
                   = torch  (default when SM120_GEMM_IMPL=torch)
                     exact torch reference (deep_gemm._moe_ref.moe_reference) on DeepInfer's own weights.
                   = deepep
                     DeepInfer's original non-mega path unchanged: deep_ep EPBuffer (single-rank shim) +
                     deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous (torch shim) + tile_kernels swiglu.
    SM120_MOE_BUCKETS       4,64,512,2048 (default)  token capacities of the b12x plans
    SM120_MOE_SHARE_SCRATCH 1 (default)               share b12x scratch between layers (sequential use)
    SM120_MOE_STRICT        0 (default)               1: never fall back from b12x to torch
    SM120_MOE_TINY_DECODE   0 (default)               1: b12x tiny-decode path for T<=4 (patched b12x only)
    SM120_MOE_GRAPH_BUCKETS 0 (default)               1: one exact plan per captured CUDA-graph token count
    SM120_MOE_PLAN          caps (default) | session  caps: plan(Caps) per token bucket, heuristic config.
                                                      session: one plan_execution per layer with exact variants
                                                      for every CUDA-graph size, prepared by a b12x
                                                      PreparationSession (autotuned when B12X_AUTOTUNE=1,
                                                      decisions cached in SM120_B12X_TUNE_CACHE)
    SM120_MOE_DETERMINISTIC auto (default)            1: b12x deterministic_output (no atomic top-k combine);
                                                      auto = on with DEEPINFER_BATCH_INVARIANT=1

CUDA graphs (DEEPINFER_ENFORCE_EAGER=0): only the b12x backend is capturable. Plans, scratch and JIT happen on
the first call per token bucket, i.e. in DeepInfer's eager warm-up run that precedes every graph capture
(cuda_graph_runner.py); bind/run are allocation- and sync-free. A plan first requested *during* capture raises
(see _plan). The torch reference and the deepep path synchronise and raise under capture.

Weight handling (b12x): DeepInfer's HF loader already stores the routed experts in exactly the raw
checkpoint form b12x wants (hugging_face/weights.py:_load_expert_params):
    gate_up_proj_weight  int8 [E, 2n, K/2]  = cat(w1 (gate), w3 (up)) raw e2m1 bytes, low nibble first
    gate_up_proj_weight.sf  int32 [E, 2n, K/128] (column-major view) = the checkpoint's F8_E8M0 [2n, K/32]
                         bytes reinterpreted 4-per-int32 -> ``.contiguous().view(uint8)`` gives them back
    down_proj_weight     int8 [E, K, n/2] (w2), .sf int32 [E, K, n/128]
b12x's w4a8_mx preparation repacks the e2m1 bytes IN PLACE (it views the uint8 storage), so we hand it
DeepInfer's own parameter storage: no second copy of the ~7 GB/layer-group of expert weights. The UE8M0
grids are copied to contiguous uint8 (b12x swizzles those into new tensors) and DeepInfer's int32 SF
tensors are freed afterwards. The DeepInfer parameters are marked ``sm120_b12x_repacked`` -- their
bytes are no longer in checkpoint layout and must not be used by any other path.
"""

from __future__ import annotations

import os

import torch

_SCRATCH: dict = {}
_STATS = {"plans": 0, "plans_by_impl": {}, "scratch_bytes": 0}


def stats() -> dict:
    """Number of b12x fused_moe plans (all layers) per implementation, and shared scratch bytes."""
    return dict(plans=_STATS["plans"], by_impl=dict(_STATS["plans_by_impl"]), scratch_mib=_STATS["scratch_bytes"] >> 20)


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip().lower()


def moe_impl() -> str:
    gemm = _env("SM120_GEMM_IMPL", "b12x")
    v = _env("SM120_MOE_IMPL", "b12x" if gemm == "b12x" else "torch")
    if v not in ("b12x", "torch", "deepep"):
        raise ValueError(f"SM120_MOE_IMPL must be b12x|torch|deepep, got {v!r}")
    return v


_EXTRA_BUCKETS: set = set()


def add_graph_buckets(sizes) -> None:
    """SM120_MOE_GRAPH_BUCKETS=1: give every CUDA-graph token count its own exact plan (b12x then picks the
    launch config, incl. tiny-decode for 1..4, for that exact T instead of the enclosing bucket's)."""
    if _env("SM120_MOE_GRAPH_BUCKETS", "0") in ("1", "true", "yes", "on"):
        _EXTRA_BUCKETS.update(int(x) for x in sizes if int(x) > 0)


def _buckets() -> tuple[int, ...]:
    raw = os.environ.get("SM120_MOE_BUCKETS", "4,64,512,2048")
    vals = {max(1, int(x)) for x in raw.split(",") if x.strip()} or {4, 64, 512, 2048}
    if _EXTRA_BUCKETS:
        top = max(vals)
        vals |= {b for b in _EXTRA_BUCKETS if b <= top}
    return tuple(sorted(vals))


def deterministic() -> bool:
    """SM120_MOE_DETERMINISTIC=auto|0|1 (auto: on with DEEPINFER_BATCH_INVARIANT=1 or B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1).
    b12x's dynamic path scatter-adds the top-k expert outputs with atomics, so the fp summation order (and the
    bf16 result) varies run to run; deterministic_output writes per-route rows and sums them in a separate
    top-k-sum kernel. The tiny-decode path (B12X_W4A8_TINY_DECODE=1) always uses atomic adds."""
    v = _env("SM120_MOE_DETERMINISTIC", "auto")
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return (_env("DEEPINFER_BATCH_INVARIANT", "0") in ("1", "true", "yes", "on")
            or _env("B12X_DYNAMIC_DETERMINISTIC_OUTPUT", "0") in ("1", "true", "yes", "on"))


def moe_plan_mode() -> str:
    v = _env("SM120_MOE_PLAN", "caps")
    if v not in ("caps", "session"):
        raise ValueError(f"SM120_MOE_PLAN must be caps|session, got {v!r}")
    return v


_SESSIONS: dict = {}


def _moe_session(device):
    s = _SESSIONS.get(device)
    if s is None:
        from b12x.preparation import PreparationSession

        from deep_gemm import _env as gemm_env

        s = PreparationSession(device=device, autotune=True, cache_dir=gemm_env.tune_cache_dir(),
                               namespace={"user": "deepinfer-sm120", "op": "fused_moe"})
        _SESSIONS[device] = s
    return s


def _strict() -> bool:
    return _env("SM120_MOE_STRICT", "0") in ("1", "true", "yes", "on")


def _log(msg: str) -> None:
    print(f"[sm120_moe] {msg}", flush=True)


def _sf_bytes(sf: torch.Tensor, k: int) -> torch.Tensor:
    """DeepInfer packed SF int32 [E, rows, ceil(k/128)] (any strides) -> contiguous uint8 [E, rows, k/32]."""
    assert sf.dtype in (torch.int32, torch.int16), sf.dtype
    b = sf.contiguous().view(torch.uint8)
    # always a fresh tensor: the caller frees DeepInfer's SF storage after prepare_weights, and b12x may keep
    # (or convert in place) the grid it was given. .contiguous() alone aliases when the SF is already contiguous.
    return b[..., : k // 32].clone(memory_format=torch.contiguous_format)


class B12xRoutedExperts:
    """b12x fused_moe over all local experts of one DeepInfer MoE layer."""

    def __init__(self, moe) -> None:
        from b12x.moe import fused_moe as fm

        self.fm = fm
        w13 = moe.gate_up_proj_weight
        w2 = moe.down_proj_weight
        if w13.dtype != torch.int8 or w13.dim() != 3 or not getattr(moe, "use_fp4_weight", False):
            raise TypeError(f"b12x MoE needs MXFP4 experts (int8 [E, 2n, K/2]); got {w13.dtype} {tuple(w13.shape)}")
        E, two_n, k_half = map(int, w13.shape)
        K, n = 2 * k_half, two_n // 2
        if tuple(w2.shape) != (E, K, n // 2):
            raise ValueError(f"down_proj shape {tuple(w2.shape)} != {(E, K, n // 2)}")
        if not (w13.is_contiguous() and w2.is_contiguous()):
            raise ValueError("expert weights must be contiguous for in-place b12x preparation")
        dev = w13.device
        if not fm.is_supported(dev):
            raise RuntimeError("b12x fused_moe reports unsupported device")
        self.E, self.K, self.n = E, K, n
        self.topk = int(moe.num_physical_topk)
        self.swiglu_limit = None if moe.swiglu_clamp_value is None else float(moe.swiglu_clamp_value)
        self.device = dev
        self.layer_idx = moe.layer_idx
        _check_tiny_decode(self.swiglu_limit, n, self.layer_idx)

        # session plans did not give bit-identical repeats with deterministic_output (5090 test): caps path then
        self.session_mode = moe_plan_mode() == "session" and not deterministic()
        self.graph_tokens = int(getattr(moe, "max_num_decoding_tokens", 0) or 0)
        s13 = _sf_bytes(w13.sf, K)  # [E, 2n, K/32]
        s2 = _sf_bytes(w2.sf, n)  # [E, K, n/32]
        ones = lambda: torch.ones(E, dtype=torch.float32, device=dev)  # noqa: E731
        w13_u8 = w13.data.view(torch.uint8)
        w2_u8 = w2.data.view(torch.uint8)
        self._prepare_started = True  # from here on DeepInfer's expert bytes may be mutated
        if self.session_mode:
            # canonical API (same in-place prepare_b12x_fp4_moe_weights underneath); needed for plan_execution
            self.wplan = fm.plan_weights(
                source=fm.PackedSource(format=fm.PackedSourceFormat.MXFP4_E8M0_K32, w13_layout=fm.W13Layout.W31),
                activation=fm.ActivationSpec(mode=fm.ActivationMode.A8, nonlinearity="silu", io_dtype=torch.bfloat16,
                                             swiglu_limit=self.swiglu_limit),
                geometry=fm.MoEGeometry(num_experts=E, hidden_size=K, intermediate_size=n),
            )
            self.experts = fm.prepare_weights(plan=self.wplan, weights=fm.PackedWeights(
                w13=w13_u8, w2=w2_u8, w13_block_scales=s13, w2_block_scales=s2,
                w13_global_scales=ones(), w2_global_scales=ones(), immutable_input_scales=True))
        else:
            self.wplan = fm.plan_weights(
                quant_modes="w4a8_mx",
                source_format="fp4_e8m0_k32",
                activation="silu",
                params_dtype=torch.bfloat16,
                num_experts=E,
                hidden_size=K,
                intermediate_size=n,
                w13_layout="w31",
            )
            self.experts = fm.prepare_weights(
                plan=self.wplan,
                params_dtype=torch.bfloat16,
                w1_fp4=w13_u8,
                w2_fp4=w2_u8,
                w1_global_scale=ones(),
                w2_global_scale=ones(),
                w1_blockscale=s13,
                w2_blockscale=s2,
                a1_gscale=ones(),
                a2_gscale=ones(),
            )
        del s13, s2
        # Free what b12x does not alias: the int32 SF grids always, the e2m1 storage only if b12x
        # allocated a fresh repacked tensor (ceil-tiled tails) instead of repacking in place.
        for p in (w13, w2):
            p.sm120_b12x_repacked = True
            old_sf = p.sf
            p.sf = torch.zeros((1,), dtype=old_sf.dtype, device=dev).expand(old_sf.shape)
            old_sf.untyped_storage().resize_(0)
        impl = getattr(self.experts, "_impl", self.experts)  # canonical PreparedExperts wraps the tensor impl
        aliased = {}
        for name, p, attr in (("w13", w13, "w1_fp4"), ("w2", w2, "w2_fp4")):
            t = getattr(impl, attr, None)
            same = isinstance(t, torch.Tensor) and t.untyped_storage().data_ptr() == p.untyped_storage().data_ptr()
            aliased[name] = same
            if isinstance(t, torch.Tensor) and not same:
                p.data.untyped_storage().resize_(0)
        self._plans: dict = {}
        self._session_plan = None
        if moe.layer_idx in (0, 1):
            _log(f"layer {moe.layer_idx}: b12x fused_moe w4a8_mx prepared E={E} K={K} n={n} topk={self.topk} "
                 f"swiglu_limit={self.swiglu_limit} in-place={aliased} "
                 f"B12X_W4A8_TINY_DECODE={os.environ.get('B12X_W4A8_TINY_DECODE')} plan_mode={moe_plan_mode()}")

    # -------------------------------------------------------------------------------- session mode
    def _session_counts(self) -> tuple[int, int]:
        """(capacity, exact variants): every DeepInfer CUDA-graph size (build_graph_batch_sizes of
        max_num_decoding_tokens) plus 1..8; other token counts run on the capacity variant (b12x variant_for)."""
        cap = _buckets()[-1]
        counts = {1, 2, 3, 4, 8}
        m = self.graph_tokens
        if m > 0:
            interval = max(1, int(os.environ.get("DEEPINFER_CUDAGRAPH_BATCH_SIZE_INTERVAL", "8")))
            size = 1
            while size < min(interval, m):
                counts.add(size)
                size *= 2
            counts.update(range(min(interval, m), m, interval))
            counts.add(m)
        counts.update(_EXTRA_BUCKETS)
        return cap, tuple(sorted(c for c in counts if 0 < c <= cap))

    def _prepare_session(self):
        if self._session_plan is not None:
            return self._session_plan
        if _capturing():
            raise RuntimeError(f"sm120_moe layer {self.layer_idx}: session plan requested during CUDA graph capture")
        fm = self.fm
        from b12x.preparation import PreparedCall
        from b12x.moe.fused_moe.workloads import make_tuning_routes

        cap, counts = self._session_counts()
        plan = fm.plan_execution(
            experts=self.experts,
            capacity=fm.ExecutionCapacity(max_tokens=cap, top_k=self.topk, warmup_token_counts=counts,
                                          route_num_experts=0),
            routing=fm.RoutingSpec(deterministic_output=True if deterministic() else None),
        )
        dev, K, topk, E = self.device, self.K, self.topk, self.E

        def factory(tokens):
            def make(state):
                g = torch.Generator(device=dev).manual_seed(42)
                hidden = torch.empty((tokens, K), dtype=torch.bfloat16, device=dev)
                src = torch.empty_like(hidden).normal_(mean=0.0, std=0.125, generator=g)
                out = torch.empty_like(hidden)
                routes = make_tuning_routes(tokens, topk, E, device=dev)
                logits = (torch.arange(tokens, device=dev, dtype=torch.float32)[:, None] * 0.03125
                          + torch.arange(topk, device=dev, dtype=torch.float32)[None] * 0.125)
                rw = torch.softmax(logits, -1).contiguous()
                ids, w = torch.empty_like(routes[0]), torch.empty_like(rw)
                scratch = tuple(torch.empty(sp.shape, dtype=sp.dtype, device=sp.device)
                                for sp in state.scratch.scratch_specs())
                binding = state.bind(scratch=scratch, a=hidden, experts=self.experts, topk_weights=w, topk_ids=ids,
                                     output=out, input_scales_static=True)

                def produce(pattern: int = 0):
                    hidden.copy_(src)
                    ids.copy_(routes[pattern])
                    w.copy_(rw)

                def reset():
                    out.zero_()
                    for b in scratch:
                        b.zero_()

                return PreparedCall(run=binding.run, output=out, produce=produce, reset=reset,
                                    restore=lambda: (reset(), produce()), capture_safe=False,
                                    benchmark_producers=tuple((lambda p=p: produce(p)) for p in range(routes.shape[0])))
            return make

        calls = {c: factory(c) for c in plan.token_counts}
        req = plan.request(name=f"sm120.moe.L{self.layer_idx}.{id(self):x}", prepare_calls=calls, benchmark_calls=calls)
        import time

        t0 = time.time()
        _moe_session(dev).prepare((req,))
        nbytes = sum(sp.nbytes for sp in plan.scratch_specs())
        key = ("session", dev, nbytes)
        buf = _SCRATCH.get(key)
        if buf is None:
            buf = torch.zeros(max(1, nbytes), dtype=torch.uint8, device=dev)
            _SCRATCH[key] = buf
            _STATS["scratch_bytes"] += nbytes
        _STATS["plans"] += len(plan.token_counts)
        _STATS["plans_by_impl"]["session"] = _STATS["plans_by_impl"].get("session", 0) + len(plan.token_counts)
        if self.layer_idx in (0, 1):
            _log(f"layer {self.layer_idx}: session plan capacity={cap} variants={tuple(plan.token_counts)} "
                 f"autotune={os.environ.get('B12X_AUTOTUNE')} prepared in {time.time() - t0:.1f}s scratch={nbytes >> 20} MiB")
        self._session_plan = (plan, buf, cap)
        return self._session_plan

    def _call_session(self, x, ids, w, out):
        fm = self.fm
        plan, buf, cap = self._prepare_session()
        T = x.shape[0]
        for i in range(0, T, cap):
            j = min(T, i + cap)
            binding = fm.bind(plan, scratch=buf, a=x[i:j], experts=self.experts, topk_weights=w[i:j],
                              topk_ids=ids[i:j], output=out[i:j], input_scales_static=True)
            fm.run(binding=binding)
        return out

    def _plan(self, bucket: int):
        hit = self._plans.get(bucket)
        if hit is not None:
            return hit
        if _capturing():
            raise RuntimeError(f"sm120_moe layer {self.layer_idx}: b12x plan for max_tokens={bucket} requested during "
                               "CUDA graph capture; plans (JIT, scratch) must be created by an eager warm-up run "
                               "with the same token count first")
        fm = self.fm
        caps_kw = dict(max_tokens=bucket, num_topk=self.topk, device=self.device, weight_plan=self.wplan,
                       quant_mode="w4a8_mx", swiglu_limit=self.swiglu_limit)
        caps = fm.Caps(**caps_kw, decode_config=None)
        if deterministic():
            # b12x 1.5.1: the heuristic picks dynamic_route_mode "direct" assuming non-deterministic output
            # (_heuristic_dynamic_route_mode passes deterministic_output=False), and validation then rejects
            # direct + deterministic_output ("planned dynamic direct routing is unsupported for this launch
            # shape"). Keep the heuristic config but route "grouped", which supports the top-k-sum combine.
            import dataclasses

            # b12x 1.5.1 also computes wrong results with deterministic_output at dynamic_tile_m < 64 (5090: cos ~0
            # against the default path, E=16 at 5..64 tokens, E=96 at 65..512; b12x's own choice for
            # deterministic_output, decode_config=None, has the same tile): use at least 64 rows per tile.
            cfg = caps.decode_config
            if cfg.backend == "dynamic":
                cfg = dataclasses.replace(cfg, dynamic_route_mode="grouped" if cfg.dynamic_route_mode == "direct"
                                          else cfg.dynamic_route_mode, dynamic_tile_m=max(64, cfg.dynamic_tile_m))
            caps = fm.Caps(**caps_kw, decode_config=cfg, deterministic_output=True)
        plan = fm.plan(caps)
        specs = tuple(plan.scratch_specs())
        share = _env("SM120_MOE_SHARE_SCRATCH", "1") not in ("0", "false", "no", "off")
        skey = (self.device, tuple((tuple(s.shape), s.dtype) for s in specs))
        scratch = _SCRATCH.get(skey) if share else None
        if scratch is None:
            scratch = tuple(torch.zeros(tuple(s.shape), dtype=s.dtype, device=self.device) for s in specs)
            _STATS["scratch_bytes"] += sum(t.numel() * t.element_size() for t in scratch)
            if share:
                _SCRATCH[skey] = scratch
        lp = plan.launch_plan
        per_launch = int(lp.max_tokens_per_launch) if lp.implementation == "dynamic" else bucket
        if self.layer_idx in (0, 1):
            _log(f"layer {self.layer_idx}: plan max_tokens={bucket} impl={lp.implementation} "
                 f"deterministic={deterministic()} "
                 f"decode_config={caps.decode_config} per_launch={per_launch}")
        self._plans[bucket] = (plan, scratch, max(1, per_launch))
        _STATS["plans"] += 1
        _STATS["plans_by_impl"][lp.implementation] = _STATS["plans_by_impl"].get(lp.implementation, 0) + 1
        return self._plans[bucket]

    def __call__(self, x: torch.Tensor, topk_idx: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor:
        fm = self.fm
        T = int(x.shape[0])
        out = torch.empty((T, self.K), dtype=torch.bfloat16, device=x.device)
        if T == 0:
            return out
        x = x.contiguous()
        ids = topk_idx.to(torch.int32).contiguous()
        w = topk_weights.to(torch.float32).contiguous()
        if self.session_mode:
            return self._call_session(x, ids, w, out)
        if deterministic():
            ids, w = _unmask_routes(ids, w, self.E)
        buckets = _buckets()
        top = buckets[-1]
        for s0 in range(0, T, top):
            s1 = min(T, s0 + top)
            bucket = next(b for b in buckets if b >= s1 - s0)
            plan, scratch, per_launch = self._plan(bucket)
            for i in range(s0, s1, per_launch):
                j = min(s1, i + per_launch)
                binding = fm.bind(
                    plan,
                    scratch=scratch,
                    a=x[i:j],
                    experts=self.experts,
                    topk_weights=w[i:j],
                    topk_ids=ids[i:j],
                    output=out[i:j],
                    input_scales_static=True,
                )
                fm.run(binding=binding)
        return out


def _unmask_routes(ids: torch.Tensor, w: torch.Tensor, n_experts: int):
    """b12x 1.5.1 deterministic_output: a masked route (id -1) leaves its per-route output row unwritten, and the
    top-k-sum kernel then adds whatever that scratch row holds from an earlier call (5090: cos 0.66-0.87 against the
    same routes without the masked ones; rows without any route not zero). With --dp every rank masks the routes to
    the other ranks' experts, so real tokens were wrong. Each masked route gets a local expert with weight 0 instead
    (spread round-robin, not all on one expert): it adds exactly 0. No host sync (CUDA-graph capturable)."""
    m = ids < 0
    spread = torch.arange(ids.numel(), device=ids.device, dtype=torch.int32).view(ids.shape) % n_experts
    return torch.where(m, spread, ids).contiguous(), torch.where(m, torch.zeros_like(w), w).contiguous()


def _check_tiny_decode(swiglu_limit, n: int, layer_idx: int) -> None:
    """Warn once if the b12x tiny-decode path (T<=4, n % 128 == 0) is enabled on a b12x without the clamp fix."""
    if swiglu_limit is None or n % 128 != 0 or os.environ.get("B12X_W4A8_TINY_DECODE", "1") == "0":
        return
    if _STATS.get("tiny_warned"):
        return
    try:
        from b12x.moe._shared.kernels import tiny_decode
        patched = bool(getattr(tiny_decode, "SWIGLU_LIMIT_SUPPORTED", False))
    except Exception:  # noqa: BLE001
        patched = False
    if not patched:
        _STATS["tiny_warned"] = True
        _log(f"layer {layer_idx}: WARNING B12X_W4A8_TINY_DECODE=1 but this b12x tiny-decode kernel ignores "
             f"swiglu_limit={swiglu_limit}: decode with T<=4 tokens is numerically wrong. Apply "
             "patches/b12x-tiny-decode-swiglu.patch or set B12X_W4A8_TINY_DECODE=0.")


class TorchRoutedExperts:
    """Exact torch reference on DeepInfer's own (unmodified) expert weights."""

    def __init__(self, moe) -> None:
        self.moe = moe

    def __call__(self, hidden_states, hidden_states_bf16, topk_idx, topk_weights) -> torch.Tensor:
        if _capturing():
            raise RuntimeError("SM120_MOE_IMPL=torch is not CUDA-graph capturable (host syncs); use b12x or "
                               "DEEPINFER_ENFORCE_EAGER=1")
        from deep_gemm import _sf
        from deep_gemm._moe_ref import moe_reference

        moe = self.moe
        if isinstance(hidden_states, (tuple, list)):
            x = _sf.dequant_fp8(hidden_states[0], hidden_states[1], 32, torch.bfloat16)
        else:
            x = hidden_states_bf16 if hidden_states_bf16 is not None else hidden_states
        w13, w2 = moe.gate_up_proj_weight, moe.down_proj_weight
        out = moe_reference(
            x, topk_idx, topk_weights,
            w13, getattr(w13, "sf", None), w2, getattr(w2, "sf", None),
            moe.swiglu_clamp_value, bool(moe.use_fp8_act),
        )
        return out


# ------------------------------------------------------------------------------------------------
# DeepseekMoE hooks (called by sm120_patches)
# ------------------------------------------------------------------------------------------------
def _wrap_ep(moe) -> None:
    """Hand-off to multi-GPU EP (sm120/sm120_ep, Option B): wraps the rank-local b12x experts when the EP
    group has > 1 rank; returns them unchanged for EP=1 or SM120_EP_IMPL=off."""
    import importlib.util

    if importlib.util.find_spec("sm120_ep") is None:
        return
    import sm120_ep

    moe.sm120_routed = sm120_ep.wrap(moe)


def setup(moe) -> None:
    """Called at the end of DeepseekMoE.__init__ (weights are loaded at that point)."""
    impl = moe_impl()
    moe.sm120_moe_impl = impl
    moe.sm120_routed = None
    if moe.use_mega_moe:
        _log(f"layer {moe.layer_idx}: use_mega_moe=True (unexpected on SM120); keeping DeepInfer's mega path "
             "with the torch mega reference")
        moe.sm120_moe_impl = "mega"
        return
    if impl == "deepep":
        return
    if impl == "b12x":
        try:
            moe.sm120_routed = B12xRoutedExperts(moe)
            _wrap_ep(moe)
            return
        except Exception as exc:  # noqa: BLE001
            mutated = bool(getattr(moe.gate_up_proj_weight, "sm120_b12x_repacked", False))
            if _strict() or mutated:
                raise
            _log(f"layer {moe.layer_idx}: b12x MoE unavailable ({type(exc).__name__}: {exc}); using torch reference")
            moe.sm120_moe_impl = "torch"
    moe.sm120_routed = TorchRoutedExperts(moe)


def forward(moe, orig_forward, hidden_states, hidden_states_bf16, input_metadata):
    impl = getattr(moe, "sm120_moe_impl", None)
    routed = getattr(moe, "sm120_routed", None)
    if impl in (None, "deepep", "mega") or routed is None:
        if impl in ("deepep", "mega") and _capturing():
            raise RuntimeError(f"SM120_MOE_IMPL={impl} path (single-rank DeepEP / mega shims) synchronises and is not "
                               "CUDA-graph capturable; use SM120_MOE_IMPL=b12x or DEEPINFER_ENFORCE_EAGER=1")
        return orig_forward(moe, hidden_states, hidden_states_bf16, input_metadata)
    topk_idx, topk_weights = moe.gate_forward(hidden_states_bf16, input_metadata)
    if impl == "b12x":
        if getattr(routed, "sm120_ep", False):  # multi-GPU EP wrapper: collective sizes depend on the step type
            out = routed(hidden_states_bf16, topk_idx, topk_weights, is_prompt=input_metadata.is_prompt)
        else:
            out = routed(hidden_states_bf16, topk_idx, topk_weights)
        shared = moe.shared_forward(hidden_states)
        if shared is not None:
            out = (out.float() + shared.float()).to(torch.bfloat16)
        return out
    out = routed(hidden_states, hidden_states_bf16, topk_idx, topk_weights)  # fp32
    shared = moe.shared_forward(hidden_states)
    if shared is not None:
        out += shared.float()
    return out.to(torch.bfloat16)
