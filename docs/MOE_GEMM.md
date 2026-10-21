# GEMM and MoE on SM120

DeepInfer's dense and expert GEMMs come from DeepGEMM and its expert dispatch from DeepEP, both SM100-only. The
`sm120/deep_gemm/` and `sm120/deep_ep/` packages implement every symbol DeepInfer touches, on b12x where a mapping
exists and on exact torch references otherwise. The routed-expert part of the MoE layer is replaced at module level by
one b12x `fused_moe` call (`sm120/sm120_moe.py`, runtime patch P3, see [patches/PATCHES.md](../patches/PATCHES.md)).
Line numbers refer to DeepInfer at commit 8b16149.

Files: `sm120/deep_gemm/**`, `sm120/deep_ep/**`, `sm120/sm120_moe.py`, `sm120/sm120_patches.py`,
`sm120/sm120_capture.py`. Multi-GPU expert parallelism is in [MULTI_GPU.md](MULTI_GPU.md); the CUDA-graph, MTP and
benchmark side in [RUNTIME.md](RUNTIME.md).

## 1. Switches

| env | values | effect |
|---|---|---|
| `SM120_GEMM_IMPL` | `b12x` (default), `torch` | `b12x`: FP8 linears run through `b12x.gemm.block_fp8_linear`; other functions use the torch reference (or the Triton kernels listed in [ATTENTION.md](ATTENTION.md)). `torch`: torch reference everywhere. |
| `SM120_MOE_IMPL` | `b12x` (default when GEMM=b12x), `torch` (default when GEMM=torch), `deepep` | routed-expert backend, section 4 |
| `SM120_GEMM_STRICT` / `SM120_MOE_STRICT` | 0/1 | raise instead of falling back from b12x to torch |
| `SM120_GEMM_DEBUG` | 0/1 | log each dispatch decision once |
| `SM120_DEQUANT_CACHE` | `einsum` (default), `1`, `0` | cache the bf16 dequantisation of static weights on the torch paths |
| `SM120_B12X_LINEAR_BUCKETS` | `16,256,2048,8192` | max_tokens capacities of the b12x linear plans; each (N, K, bucket) is one JIT compile |
| `SM120_MOE_BUCKETS` | `4,64,512,2048` | max_tokens capacities of the b12x fused_moe plans |
| `SM120_MOE_SHARE_SCRATCH` | 1 | layers share b12x MoE scratch |
| `SM120_MK_ALIGNMENT` | 128 | `get_theoretical_mk_alignment_for_contiguous_layout` (the value DeepGEMM returns on any arch other than SM100) |
| `SM120_MOE_TINY_DECODE` | `0` (default), `1`, `auto` | sets `B12X_W4A8_TINY_DECODE` unless set explicitly. `0`: dynamic path (faster, section 7). `1`/`auto`: tiny-decode only if b12x carries `patches/b12x-tiny-decode-swiglu.patch` |
| `B12X_W4A8_TINY_DECODE` | from the above | `0` = b12x's dynamic path for T <= 4 (correct on unpatched b12x 1.5.1, whose tiny-decode ignores `swiglu_limit`) |
| `SM120_MOE_DETERMINISTIC` | `auto` | `1`: b12x `deterministic_output` (no atomic top-k combine); `auto` = on with `DEEPINFER_BATCH_INVARIANT=1` (section 5) |
| `SM120_MOE_PLAN` | `caps` | `session`: one b12x `plan_execution` per layer with exact variants for every CUDA-graph size, prepared by a `PreparationSession` (autotuned with `SM120_B12X_AUTOTUNE=1`). Optional, within about 5 % of `caps`; forced off in deterministic mode |
| `SM120_MOE_GRAPH_BUCKETS` | 0 | `1`: every captured CUDA-graph token count gets its own exact MoE plan (section 6) |
| `SM120_B12X_AUTOTUNE` | 0 | `1`: block_fp8_linear plans are prepared by a b12x `PreparationSession` (candidate race, section 7) |
| `SM120_B12X_TUNE_CACHE` | `/cache/b12x/preparation` | tuning decisions (only with `SM120_B12X_AUTOTUNE=1`) |
| `B12X_AUTOTUNE` | 0 (1 with `SM120_B12X_AUTOTUNE=1`) | read only by b12x preparation sessions; the lazy default path ignores it |

## 2. Scale-factor layout contract

`transform_sf_into_required_layout` (`deep_gemm/_layout.py`) returns **exactly the SM100 DeepGEMM layout**:

- the FP32 block SF `[G?, ceil(MN/gran_mn), ceil(K/gran_k)]` is broadcast along MN, so row r uses block r//gran_mn;
- each value becomes a UE8M0 byte (`bits>>23`), padded with 127 to a multiple of 4 and packed 4 per int32;
- the result is int32 `[G?, MN, ceil(K/gran_k/4)]`, MN-major with TMA-aligned stride `align(MN,4)`;
- int32 inputs with gran_mn = 1 are returned unchanged.

Because the layout is per row, FlashMLA-style row permutations of `(weight, sf)` (`attn.py:114` q_b, `attn.py:143` wv,
done by the attention shim), DeepInfer's `.view()`s (`attn.py:202`) and EPLB's `sf.mT` all stay valid. Every reader
(`_sf.sf_to_group_scales`) works with any stride, accepts per-row or per-block and int32 or fp32 input, and decodes byte
j of int32 i as K-group 4i+j. Activations come from `tile_kernels.quant.per_token_cast(..., num_per_channels=32,
round_sf=True, use_packed_ue8m0=True, use_tma_aligned_col_major_sf=True)`, the same packed MN-major format
(`linear.py:17-24`).

## 3. Symbol map

| Symbol | Call site(s) | Implementation |
|---|---|---|
| `fp8_fp4_gemm_nt(x, w, out, out_acc, recipe=(1,1,32))` | `layers/linear.py:165` (all FP8 `ReplicatedLinear`s: attention projections, shared experts, wo_a/b ...) | **b12x** `block_fp8_linear`. `pack_weight(w, 32x32 UE8M0)` runs on first use and is cached by `(data_ptr, shape)`; E4M3 values are not copied. Weights whose SF is not 32x32-block-uniform (after a row permutation) go through the per-row `MXFP8Rows` builder. The activation is dequantised to bf16 and b12x requantises it; the round trip is lossless (argument in `_b12x.py`). Plans per (N, K, bucket) use lazy default preparation. `d = r + c` in fp32. **torch**: exact bf16 dequant + cuBLAS. |
| `bf16_gemm_nt(a,b,d,c)` | `linear.py:33` (bf16 weights, gate with fp32 out), `sampler_utils.py:26` (lm-head logits, fp32 out), `vit.py:47` | torch: bf16 out via `torch.mm`; fp32 out via `torch.mm(..., out_dtype=float32)`, falling back to a chunked exact fp32 GEMM without TF32. No bf16 rounding of logits. Optional FP8 lm_head: [ATTENTION.md](ATTENTION.md). |
| `cublaslt_gemm_nt` | `vit.py:50` | same as `bf16_gemm_nt` |
| `fp8_einsum("bhr,hdr->bhd", (ctx,sf), (wv,sf), out, recipe=(1,1,32))` | `attn.py:215` (wo_a grouped projection) | Triton kernel (`sm120_einsum.py`); torch fallback: exact dequant (cached, `SM120_DEQUANT_CACHE=einsum`) + bf16 bmm. FP8 output `(fp8, sf)` (`use_fp8_einsum_output`) follows DeepGEMM's epilogue semantics, `per_token_cast_to_fp8(d_bf16.view(b,h*d), 32, packed UE8M0)` written into the caller's MN-major SF. Other expressions raise. |
| `transform_sf_into_required_layout` | `linear.py:70`, `attn.py:134`, `mlp.py:73` | section 2 |
| `get_tma_aligned_size` | `attn.py:208` | `align(x, 16/elem)` |
| `m_grouped_fp8_fp4_gemm_nt_contiguous` | `mlp.py:119` (only `SM120_MOE_IMPL=deepep`) | torch: psum layout (group g covers rows `[align(psum[g-1], A), psum[g])`, DeepEP expand semantics) or a per-row layout; padding rows zeroed; host sync via `.tolist()` |
| `m_grouped_bf16_gemm_nt_contiguous` | `mlp.py:130` (bf16 experts) | torch, same layout handling |
| `get/set_mk_alignment_for_contiguous_layout`, `get_theoretical_...` | `mlp.py:250-251,477` | state knob, default 128; DeepInfer passes the same value to `tile_kernels.set_token_alignment`, so dispatch, GEMM and SwiGLU agree |
| `tf32_hc_prenorm_gemm(x, fn, out, sqr, n_splits)` | `mhc_ops.py:111,113` (mHC pre on every sub-layer) | Triton kernel (`sm120_mhc.py`), full fp32; torch fallback exact fp32. Split-K partials cover the real K chunks `ceil(K/S)`; the consumer sums them. More accurate than DeepGEMM's TF32. |
| `mega_mhc` | `hc.py:270` (disabled by P4) | torch port of `tests/test_mega_mhc.py::mhc_reference` |
| `bf16_mega_gate` | `mlp.py:398` (disabled by P3) | fp32 logits + `tile_kernels.moe.moe_topk_gate(**same kwargs)`, i.e. DeepInfer's non-mega branch |
| `SymmBuffer`, `transform_weights_for_mega_moe`, `fp8_fp4_mega_moe`, `bf16_mega_moe`, `get_block_m_for_mega_moe` | `mlp.py:96-98,191,255,293`, `userbuffer.py:183-225` (mega path, disabled by P2) | single-rank torch reference; multi-rank raises |
| `get_num_sms`, `set_num_sms` | `userbuffer.py:44-51`, `index_attn.py:394` | device SM count, settable |
| `set_pdl`, `use_deterministic_algorithms` | `runtime_config.py:65,68` | state only |
| `locality_domain.*` | `userbuffer.py:218` | stubs |
| indexer `fp8_fp4_*mqa_logits` + metadata | `index_attn.py:273-447` | re-exported from `sm120_attn.indexer.DEEP_GEMM_EXPORTS` ([ATTENTION.md](ATTENTION.md)); imported at the end of `deep_gemm/__init__`, with a stub that raises a clear error if the module is missing |
| `deep_ep.EPBuffer`, `EPHandle`, `EventHandle`, `EventOverlap` | `mlp.py:17,469-511`, `userbuffer.py:63-95` | single-rank implementation with DeepEP's exact contracts (`deep_ep/_ep.py` docstring): expand mode gives `recv_x` grouped per expert at aligned offsets plus a packed MN-major SF, 1-D weights, psum = aligned start + count, `recv_src_metadata` slots; combine = fp32 slot sum + bias; `get_theoretical_num_sms` -> 0. On a group with more than one rank it is an inert placeholder under the b12x EP path ([MULTI_GPU.md](MULTI_GPU.md)). |
| `deep_ep.get_physical_domain_size` | `parallel_state.py:134` | `(group.size(), 1)`: an NVLink domain of 1 GPU (PCIe) |
| `deep_ep.get_num_rdma_alignment`, `get_num_allocation_alignment` | `userbuffer.py:108-109` (CP > 1 only) | constants |
| `deep_ep.BucketBuffer/BucketSession` | `parallel_utils/cp.py:6` | import-only stubs; constructing one raises (CP = 1 only) |
| `deep_ep.EngramBuffer` | `engram.py:14`, `userbuffer.py:127-181` | implemented on `sm120_engram` ([ENGRAM.md](ENGRAM.md)) |
| aliases (`fp8_gemm_nt`, `fp4_gemm_nt`, `m_grouped_*_masked`, `*_nn/_tn/_tt`, `einsum`, `k_grouped_*`) | unused by DeepInfer | NT/masked/bf16 variants implemented; FP8 NN/TN/TT and K-grouped raise `NotImplementedError` |

## 4. MoE: design

**Decision:** patch at module level (P3 + `sm120_moe.py`), with b12x `fused_moe` (`w4a8_mx`) as the default.

DeepInfer's non-mega path chains EPBuffer dispatch (expand), grouped GEMM1,
`tile_kernels.swiglu_forward_and_per_token_cast`, grouped GEMM2 and combine. For EP = 1 dispatch and combine only
permute data, and b12x has no grouped GEMM that takes DeepGEMM's psum layout. Replacing the routed part of
`DeepseekMoE.forward` with one `fused_moe` call is less invasive and more robust:

- no layout emulation in the hot path: no expand buffers with 128-row padding per expert (about 50k rows at prefill),
  no psum host sync, no reliance on tile_kernels' psum SwiGLU semantics;
- the calling convention was validated on an RTX 5090 with real layer-2 experts (`tests/test_moe_b12x_local_subset.py`:
  cos 0.999 against the fp32 reference, masked `-1` ids work, partial sums correct): `plan_weights(quant_modes="w4a8_mx",
  source_format="fp4_e8m0_k32", activation="silu", w13_layout="w31")`, `prepare_weights(...)`, `Caps(...,
  swiglu_limit=cfg.swiglu_limit)`, `plan`, `bind(input_scales_static=True)`, `run`, with dynamic-launch chunking;
- DeepInfer's gate (`gate_forward`: noaux_tc, sqrtsoftplus, bias, EPLB logical-to-physical map,
  `routed_scaling_factor`) and shared expert (`shared_forward`: FP8 linears and tile_kernels SwiGLU) are unchanged;
  routed and shared outputs are added in fp32, as combine's `bias=` does;
- the same structure extends to EP > 1 on PCIe: all-gather the tokens, run fused_moe on the local experts with
  non-local ids set to -1, then reduce-scatter ([MULTI_GPU.md](MULTI_GPU.md)).

The alternatives remain selectable: `SM120_MOE_IMPL=torch` (exact reference on the same weights) and
`SM120_MOE_IMPL=deepep` (DeepInfer's original path on the `deep_ep` and grouped-GEMM shims).

**Weights.** `hugging_face/weights.py:_load_expert_params` (lines 278-327) stores the experts in raw checkpoint form
(no transform for FP4):

- `gate_up_proj_weight` is int8 `[E, 2n, K/2]` = `cat(w1, w3)` raw e2m1 bytes, gate first, which is b12x
  `w13_layout="w31"`;
- `.sf` is int32 `[E, 2n, K/128]`, a column-major view whose bytes are the checkpoint's F8_E8M0 `[2n, K/32]`.

`sm120_moe.B12xRoutedExperts` passes DeepInfer's own e2m1 storage (viewed as uint8) to `prepare_weights`. The w4a8
repack works in place (V4.1 shapes have no tail), so **there is no second copy of the expert weights**; only the scale
grids are copied (about 0.4 GB per layer, transient). DeepInfer's int32 SF storages are then freed and replaced by
zero-stride placeholders, so EPLB's `sf.mT` still works. If b12x allocates a fresh repacked tensor, the DeepInfer
storage is freed instead (checked through `data_ptr`). The parameters are tagged `sm120_b12x_repacked`. SF bytes are
always copied (a view of a freed SF caused an illegal memory access during development).

**Numerics.**

- Activations: b12x quantises the bf16 hidden states itself to MXFP8, the same scheme as DeepInfer's `per_token_cast`.
- SwiGLU clamp: gate `min(g, L)`, up `clamp(u, +-L)`, on every path used by default (section 7).
- Routing weights: b12x applies the routing weight after FC2; DeepInfer applies it before the FP8 cast of the SwiGLU
  output. This is a rounding-order difference only.

## 5. Determinism

The torch paths use cuBLAS default algorithms; `use_deterministic_algorithms` is only recorded.

**b12x fused_moe is not run-to-run deterministic by default:** the dynamic path scatter-adds the top-k expert outputs
into the bf16 output with atomics, so the summation order varies. Layer-0 `routed_out` differed by up to 0.09 between
two identical eager runs (7/7 repeats differ, max 0.25 in a unit test), while attention was bit-identical. On the
truncated test models this flips near-tied greedy tokens after a few steps.

b12x has a deterministic mode, `Caps(deterministic_output=True)` (env `B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1`): per-route
rows plus a separate top-k-sum kernel, at some extra memory traffic. `SM120_MOE_DETERMINISTIC=auto|0|1` sets it; `auto`
turns it on with `DEEPINFER_BATCH_INVARIANT=1` (`tools/run_deepinfer.py --batch-invariant`). With it, repeats are
bit-identical, and two `--batch-invariant` runs give bitwise equal prefill and decode logits and tokens. b12x 1.5.1's
heuristic can pick `dynamic_route_mode="direct"` for a deterministic plan, which then fails with "planned dynamic
direct routing is unsupported for this launch shape" ([flashinfer-ai/flashinfer#6941](https://github.com/flashinfer-ai/flashinfer/issues/6941)); the shim works
around this by forcing `"grouped"`
(`patches/repro_b12x_deterministic_direct.py` reproduces it standalone). The tiny-decode path always uses bf16 atomic
adds and has no deterministic variant.

Cost (RTX 5090, CUDA graphs, prompt 256, 128 generated tokens, TPOT ms default -> deterministic):

| model | B=1 | B=8 | B=32 |
|---|---|---|---|
| 3-layer test model | 2.3 -> 4.6 | 2.6 -> 5.2 | 3.6 -> 6.7 |
| YOCO test model (8 layers) | 4.3 -> 11.0 | 6.0 -> 11.9 | 8.3 -> 16.1 |
| full-depth test model (40 layers, 16/384 experts) | 16.1 -> 46.1 | 20.4 -> 42.2 | 25.6 -> 47.8 |

The decode step is 2-2.9x slower, so the deterministic mode stays opt-in, for tests and reproducibility only
(`server.py --deterministic` is a different switch: it uses the torch MoE reference, `SM120_MOE_IMPL=torch`).
`SM120_MOE_PLAN=session` is forced off in deterministic mode: session plans did not give bit-identical repeats. Whether b12x `block_fp8_linear` is bit-reproducible
(split-K configs) is not checked. Test: `tests/test_gemm_moe_graphs.py::test_moe_determinism`.

## 6. CUDA graphs (`DEEPINFER_ENFORCE_EAGER=0`)

DeepInfer captures only the main-model forward of decode steps (`worker/cuda_graph_runner.py`) and, with MTP, the
whole DSpark draft step including sampling (`worker/dflash_cuda_graph_runner.py`, unless
`DEEPINFER_DFLASH_ENFORCE_EAGER=1`). Prefill and the main sampler stay eager.

- **Sizes.** `build_graph_batch_sizes(max_num_seqs * (1 + nextn), DEEPINFER_CUDAGRAPH_BATCH_SIZE_INTERVAL=8)`: powers
  of two below 8, then steps of 8, then the maximum. The size counts tokens (`total_q`), so with MTP a batch of B
  requests uses the graph >= B·(1+nextn). DFlash graphs are sized in requests (its MoE sees B·nextn tokens). Padding
  rows get `valid_token_mask=False`, so the gate gives them id -1, which b12x skips.
- **Warm-up.** Before each capture DeepInfer runs the same size once eagerly on the capture stream (largest first).
  That run creates everything the shims cache per shape: b12x weight packing (one `.item()` sync), `block_fp8_linear`
  plans per (N, K, bucket) incl. JIT, fused_moe plans per (layer, bucket) incl. JIT, shared scratch, the torch einsum
  dequant cache. Under capture only `bind` + `run` remain; both are allocation-free (outputs come from the graph pool)
  and do not sync. Cached buffers are allocated outside capture and never freed, so the graph's baked-in addresses
  stay valid; scratch shared between layers is safe because DeepInfer runs one stream.
- **Guards.** A first-time packing / plan / dequant-cache fill during capture raises a clear `RuntimeError`
  (`deep_gemm._b12x._not_in_capture`, `sm120_moe._plan`). `SM120_MOE_IMPL=torch|deepep` raise under capture (they
  sync); graphs need the b12x MoE. GEMMs on the torch fallback are capturable.
- **Buckets.** A graph of T tokens runs the plan of the smallest bucket >= T, so T = 8 runs a 64-token MoE plan (b12x
  picks its launch config from the plan's `max_tokens`). `SM120_MOE_GRAPH_BUCKETS=1` adds every main-graph size as an
  exact MoE bucket: more plans and scratch sets (about 11 sizes for 64 sequences, about 52 with nextn = 5). Whether that
  is faster has not been measured.
- **P9** logs the plans created per capture (`sm120_moe.stats()`, `deep_gemm._b12x.stats()`).

Host syncs remain on the `deepep` path, the torch grouped GEMM, the torch MoE reference, the b12x lazy preparation
(first call only) and the capture hooks; eager steps therefore need `DEEPINFER_DISABLE_SYNC_GUARD=1`.

## 7. Tiny-decode MoE and autotune

**Tiny-decode clamp bug** ([flashinfer-ai/flashinfer#6866](https://github.com/flashinfer-ai/flashinfer/issues/6866)).
b12x 1.5.1 runs `w4a8_mx` MoE with 1 <= T <= 4 and `n % 128 == 0` through `moe/_shared/kernels/tiny_decode.py` (bf16
activations, two plain launches). Its FC2 computes `silu(g)·u` without the SwiGLU clamp, because `_launch_micro` never
passes `swiglu_limit`. V4.1 with full experts per GPU has n = 2304 (`n % 128 == 0`) and is affected; a TP4 expert shard
(n = 576, `n % 128 == 64`) goes to the compact-micro kernel, which clamps. With real V4.1 layer-2 experts the
tiny-decode output had cos 0.79 against the clamped reference (dynamic path 0.998).

- Workaround (default): `B12X_W4A8_TINY_DECODE=0`, set through `SM120_MOE_TINY_DECODE=0`.
- Fix ([posted upstream](https://github.com/flashinfer-ai/flashinfer/issues/6866#issuecomment-6312667479)): `patches/b12x-tiny-decode-swiglu.patch` makes `swiglu_limit` a compile-time constant of the tiny-decode kernel
  (part of the cache key) and clamps like the reference (`g = min(g, L)`, `u = clamp(u, +-L)`, then `silu(g)·u·w`);
  without a limit the generated code is unchanged. The custom op gets a `float? swiglu_limit` argument and the
  preparation prewarm resolves the same kernel. The serving image applies the patch.
- `tests/test_b12x_tiny_decode_swiglu.py` checks T = 1..4 against the fp32 reference and the dynamic path, with and
  without a limit; `--bench` times both paths under CUDA-graph replay. With the patch everything passes, but tiny-decode
  is slower than the dynamic path on the RTX 5090 (0.90-0.95x; T=1 71.4 vs 67.6 us, T=4 305.8 vs 287.8 us), so the
  default stays `SM120_MOE_TINY_DECODE=0` even on a patched b12x. `1` turns tiny-decode on (refused on unpatched b12x;
  the shim warns if `B12X_W4A8_TINY_DECODE=1` is forced there).
- With MTP the path is irrelevant: verify batches have B·(1+nextn) >= 6 tokens, draft batches B·nextn >= 5.

**Autotune.** The plans use `plan(Caps)` + b12x's lazy default preparation, which always takes the heuristic config
and ignores `B12X_AUTOTUNE`. Tuning needs a `PreparationSession`:

- `block_fp8_linear`: opt-in `SM120_B12X_AUTOTUNE=1` (`_b12x._prepare_tuned`). On the first call of each (N, K, bucket)
  the session races the candidate configs on the real packed weight and caches the decision in
  `/cache/b12x/preparation` (keyed by device). A failure falls back to the default config. Costs start-up time,
  synchronises, and uses up to half of the free memory as race budget, so it happens in the profile run / graph
  warm-ups. Measured effect: within about 5 % of the default ([RUNTIME.md](RUNTIME.md), "MoE tuning levers").
- `fused_moe`: the default `Caps` path is heuristic-only by design. Opt-in `SM120_MOE_PLAN=session` uses
  `plan_execution(experts, ExecutionCapacity(max_tokens, top_k, warmup_token_counts=graph sizes))` plus a session
  request with one prepared call per token count (raced with `SM120_B12X_AUTOTUNE=1`). Measured: within about 5 % of
  the default, so the default is unchanged.
- b12x compile artifacts are cached on disk: `sm120_patches` points `B12X_COMPILE_CACHE_DIR` to `/cache/b12x/compile`,
  so repeated container runs skip most JIT.

## 8. Known gaps

1. **b12x linear internals.** `block_fp8_linear` is used through `plan(Caps)` + lazy default preparation. The per-row
   fallback for non-block-uniform SFs uses b12x internals (`gemm._shared.wo_mxfp8.MXFP8Rows`,
   `pack_mxfp8_scales_for_dense_gemm`). On any exception the shim falls back to torch for that weight and warns once,
   unless `SM120_GEMM_STRICT=1`.
2. **JIT time.** Each distinct (N, K, bucket) of the linear and each (layer, bucket) of fused_moe compiles on first
   use; the profile and warm-up runs trigger the large buckets, decode triggers buckets 16 / 4. Mount `/cache` to keep
   the compile cache.
3. **Plan capacity.** Plans are bound with T <= max_tokens. If a b12x launch path turned out to require
   T == max_tokens, set `SM120_MOE_BUCKETS` / `SM120_B12X_LINEAR_BUCKETS` to the exact sizes used.
4. **tile_kernels on SM120.** DeepInfer still calls `tile_kernels` (activation `per_token_cast`, `moe_topk_gate`, mHC
   kernels, shared-expert SwiGLU) and `batched_transpose_weight_sf` at load; these work on SM120 in all tests run so
   far but are not separately validated.
5. **Determinism** of the default MoE path, section 5.

## 9. Tests

```bash
cd /opt/sm120
pytest -q tests/test_gemm_moe_graphs.py        # warm-up/capture/replay == eager, plan-in-capture guards, determinism
python3 tests/test_moe_b12x_local_subset.py --ckpt /ckpt   # b12x fused_moe on real experts of one layer (needs the HF shards)
```

`tests/test_gemm_moe_graphs.py`: 14 passed, 2 xfailed on unpatched b12x (the tiny-decode cases, by design).
