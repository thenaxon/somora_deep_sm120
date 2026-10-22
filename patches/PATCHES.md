# Patches

DeepInfer's source is not modified on disk except for one optional C++ binding (below). Everything SM120-specific is
either a drop-in kernel package on `PYTHONPATH` (`sm120/deep_gemm`, `sm120/deep_ep`, `sm120/flash_mla`,
`sm120/deep_select`) or a runtime patch applied in-process by `sm120/sm120_patches.py`. Kernel replacements are not
patches; see [docs/DESIGN.md](../docs/DESIGN.md), [docs/MOE_GEMM.md](../docs/MOE_GEMM.md) and
[docs/ATTENTION.md](../docs/ATTENTION.md).

## 1. Runtime patches (`sm120/sm120_patches.py`)

`sm120/sitecustomize.py` imports `sm120_patches` when the interpreter starts (it chains to any other `sitecustomize`
on `sys.path`). `sm120_patches` registers post-import hooks: each patch runs right after DeepInfer imports the target
module. Nothing is imported eagerly; in particular `torch` is not imported at start-up, which DeepInfer's
`deepinfer/__init__.py` forbids.

- Every patch only takes effect on a GPU with compute capability 12.x. `SM120_FORCE_SM120=0|1` overrides the
  detection; `SM120_PATCHES=0` turns all patches off.
- `sm120_patches.APPLIED` lists the patches that ran; `tools/run_deepinfer.py` saves it in its output under
  `meta.patches`.

Line numbers refer to DeepInfer at commit 8b16149.

| # | Target | Change | Why |
|---|---|---|---|
| P1 | `deep_cluster/hardware/device.py:151` `get_bf16_tflops_limit`, called at `engine/llm_engine.py:304` | On SM12x, return the dense BF16 peak instead of raising `RuntimeError("unknown GPU type")`: RTX 5090 210, RTX PRO 6000 Max-Q 190, RTX PRO 6000 250, otherwise 200 TFLOPS. `SM120_BF16_TFLOPS` overrides it. Both `deep_cluster.hardware` and `.device` are patched, because llm_engine imports the name directly. | Without it the engine aborts at construction. The value only feeds the MFU statistic. `is_sm_major_10()` correctly returns False, which sends ViT's bf16 GEMM to `cublaslt_gemm_nt`. |
| P2 | `parallel_utils/parallel_state.py:131-134` `ParallelProcessGroup.is_within_nvl_domain` | Returns False on SM12x (NPU behaviour unchanged). | It decides `DeepseekMoE.use_mega_moe` and the EPLB symmetric-memory shuffle. For EP=1 the original returns True through its `group.size()==1` short-circuit, which selects the SM100-only mega-MoE path. SM120 machines have no NVLink, so False is also the truthful answer for EP>1; the `deep_ep` shim additionally reports an NVLink domain of 1. |
| P3 | `models/v4_modules/mlp.py` `DeepseekMoE.__init__` / `.forward` | After `__init__`: `use_mega_gate = False` and `sm120_moe.setup(self)` (routed-expert backend). `forward` delegates to `sm120_moe.forward`: DeepInfer's gate (`gate_forward`) and shared expert (`shared_forward`) are kept; dispatch, grouped GEMMs, SwiGLU and combine are replaced by one b12x `fused_moe` call. `SM120_MOE_IMPL=deepep` runs the original forward. | `bf16_mega_gate` is an SM100 fusion; the non-fused branch is DeepInfer's own code. Design: [docs/MOE_GEMM.md](../docs/MOE_GEMM.md). |
| P4 | `models/v4_modules/hc.py:99-101` `HCLayerMixin.use_mega_mhc` | Set to False after `__init__`. `SM120_MEGA_MHC=1` keeps it (the torch `deep_gemm.mega_mhc` reference then runs). | `mega_mhc` is an SM100 fusion of mHC post+pre+RMSNorm+cast. The unfused path (tile_kernels mHC kernels plus `deep_gemm.tf32_hc_prenorm_gemm`) is DeepInfer's own code. |
| P5 | `models/v4_modules/attn.py:87-92` `V4AttentionBase.use_fused_norm_rope_attn_rope_cast` | `SM120_FUSED_NORM_ROPE_ATTN=auto` (default) keeps DeepInfer's decision only if the `flash_mla` shim exposes `fused_norm_rope_attn_rope_cast.{prefill,decode,permute_q_b_proj,permute_wv_proj}`, and forces False otherwise. `0` always forces False, `1` never touches it. Applied in the base `__init__`, so the q_b/wv weight permutations follow the same decision. | The flag selects FlashMLA's SM100 fused kernel; the gate follows what the shim provides. |
| P6 | `layers/sampler.py` `Sampler.forward` and module-level `get_logits` | Wrapped to call `sm120_capture` hooks, inert until `tools/run_deepinfer.py` calls `sm120_capture.arm()`. `forward` also passes `seq_metadata_list`: sampler calls of placeholder steps (all sequences `is_placeholder`) are not recorded, and every decode entry carries request ids / rows (`STATE["decode_meta"]`, `sm120_capture.decode_rows()`). | Instrumentation only: fp32 logits for every prefill position (same GEMM as `sampler_utils.get_logits`) and per-step decode logits. The engine runs a dummy sequence (token 0, position 0) whenever the schedule is empty; without the filter those constant logits were stored as decode steps. |
| P7 | `worker/base.py:212` `WorkerBase.profile_num_available_blocks` | Active only when `SM120_MAX_GPU_BLOCKS` is set (`run_deepinfer.py` sets 256): caps the profiled KV block count. | Otherwise the KV cache takes all memory left after profiling and later first-use allocations (b12x plan scratch for new token buckets, torch-path temporaries) have no headroom. |
| P8 | `model_executor/layers/tilelang_ops/cuda/sampler.py:155` `remap_draft_topk_tl` (MTP rejection with `DEEPINFER_SAMPLER_FORCE_TOPK>0`) | The factory is re-defined from its own source with `pool_size = remap_pool_size(vocab_size)` instead of `25856 if force_topk <= 1024 else 44032`: the fewest equal vocab windows that fit `SM120_REMAP_SMEM_BYTES` (default 96 KiB), i.e. 21568 floats (84 KiB) for vocab 129280. The patched text is registered in `linecache` under `<sm120_patches:...>` so TileLang's source parser and source-keyed caches see it. Runs while `tilelang_ops/ops.py` imports the backend; a second hook refreshes the `ops.remap_draft_topk_tl` re-export if needed. If the source line is not found exactly once, nothing is patched and a note is logged. | The kernel's shared vocab window needs 103 KB / 176 KB; SM120 allows 99 KB per block ("Failed to set dynamic shared memory size"). The kernel loops over `ceil(vocab / pool_size)` windows, so the result does not depend on the window size. |
| P9 | `worker/cuda_graph_runner.py` `CudaGraphRunner.capture`, `worker/dflash_cuda_graph_runner.py` `DFlashCudaGraphRunner.capture` | Wrapped: log the captured sizes and the b12x plans / scratch created by the warm-up runs (`sm120_moe.stats()`, `deep_gemm._b12x.stats()`), print a hint on failure and exit the process with 4 (`SM120_CAPTURE_FAIL_FATAL=0`: re-raise; a raised capture error left `--dp` ranks hanging). With `SM120_MOE_GRAPH_BUCKETS=1` the main graph sizes become exact MoE plan buckets. The warm-up + capture runs inside b12x `retain_compiled_programs()`, whose record is kept for the life of the process (`SM120_RETAIN_GRAPH_PROGRAMS=0` disables it). | Logging and opt-in bucket control; graph safety itself is in the shims ([docs/MOE_GEMM.md](../docs/MOE_GEMM.md)). Retention: every b12x preparation made while serving (a prefill / indexer shape seen for the first time) ends with `evict_unretained()`, which unloads every program that is neither in that session's plans nor retained, including the MoE `tp_moe.dynamic` programs a captured decode graph launches. The next replay then failed with `CUDA error: an illegal instruction` (regression test `tests/graph_replay_lazy_prep.sh`). |
| P10 | `deep_cluster/topology/bind.py` `bind_numa`, called by `worker/base.py:347` `_bind_numa` and `ve/multimodal_processor.py:174` | Skipped (returns 0, one info line) when the host has one NUMA node (`/sys/devices/system/node/node<N>`) or the GPU reports node `-1`; multi-node hosts with a valid node keep DeepInfer's binding. | On single-socket hosts the GPU's sysfs `numa_node` is `-1`, and `bind_numa(-1)` segfaults inside libnuma (`numa_bitmask_setbit(mask, (unsigned)-1)`). Without libnuma1 the same calls only log long tracebacks. With libnuma1 (now in the image) and one node there is nothing to bind. On multi-socket hosts the memory policy additionally needs `--cap-add SYS_NICE` (Docker's seccomp profile blocks `set_mempolicy` otherwise: "failed to set memory policy: Operation not permitted"; CPU affinity is still applied). |
| P12 | `engine/llm_engine.py` `LLMEngine.post_process` | Wrapped (class level; bound into the worker callbacks at construction): after each decode step, add the sampler output's `num_generated_tokens` / `num_verified_tokens` (`[num_seqs, n at draft position 1, ...]`) to `sm120_patches.MTP_STATS`. Placeholder and prompt steps are skipped. | Statistics only. DeepInfer aggregates these only in its stats logger, which the server turns off. With `--mtp-varlen` the verified draft count varies per step, so the acceptance cannot be derived from tokens per step. |
| P11 | `ve/multimodal_processor.py:92` `MultimodalProcessor._start_workers` | After the original call, `get_multi_modal_pool()._buffer_shm.unlink()` (shm_unlink plus resource_tracker unregister), and the pool's own later `unlink()` becomes a no-op. | DeepInfer unlinks the pinned multimodal pool (`multiprocessing.shared_memory`, `/dev/shm/psm_*`, `DEEPINFER_MULTIMODAL_POOL_SIZE_GB` per engine) only in `LLMEngine.close()`, which a SIGTERM'd/SIGKILL'd server never reaches. The workers are forked and inherit the mapping, and nothing reopens the segment by name, so the name can go at once. The memory is freed with the last mapping. |
| - | `model_executor/layers/sampler_utils.py` import | `sm120_lmhead.install()` wraps `sampler_utils.bf16_gemm_nt`; inert unless `SM120_LMHEAD_IMPL=fp8`. | Optional FP8 lm_head ([docs/ATTENTION.md](../docs/ATTENTION.md)). Composes with P6, because `get_logits` looks `bf16_gemm_nt` up at call time. |
| - | `sm120_moe.setup` / `forward`, `deep_ep.EPBuffer` (shims) | Multi-GPU expert parallelism (`sm120/sm120_ep`): `setup` passes the b12x experts to `sm120_ep.wrap(moe)` (unchanged for EP=1 or `SM120_EP_IMPL=off`); `forward` passes `is_prompt=` to wrappers marked `sm120_ep`; `EPBuffer` on a group with more than one rank is an inert placeholder under `SM120_EP_IMPL=allgather` (default), whose dispatch/combine raise. | DeepInfer still builds an EPBuffer per MoE layer; the b12x EP path never uses it. See [docs/MULTI_GPU.md](../docs/MULTI_GPU.md). |
| - | `deepinfer` package import | `sm120_patches_engram.install()` (`SM120_ENGRAM_PATCHES=0` disables it). | Engram support, [docs/ENGRAM.md](../docs/ENGRAM.md). |
| - | `deepinfer` package import | If `sm120_attn.patches` exists, its `install()` is called. | Extension point for attention-specific patches; such an `install()` must stay lazy and register `sm120_patches.when_imported(...)` hooks instead of importing torch or DeepInfer eagerly. |

Environment defaults set by `sm120_patches` at start-up (an explicit value always wins):

- `B12X_W4A8_TINY_DECODE`, decided by `SM120_MOE_TINY_DECODE=0|1|auto`. `0` (default) keeps b12x's dynamic path for
  T <= 4. `1`/`auto` enable tiny-decode only if the installed b12x carries `b12x-tiny-decode-swiglu.patch` (detected by
  the `SWIGLU_LIMIT_SUPPORTED = True` marker in `tiny_decode.py`, read without importing b12x).
  `sm120_patches.TINY_DECODE_POLICY` records the decision.
- `B12X_AUTOTUNE=0`, or `1` with `SM120_B12X_AUTOTUNE=1` ([docs/MOE_GEMM.md](../docs/MOE_GEMM.md) section 7).
- `B12X_COMPILE_CACHE_DIR=/cache/b12x/compile` and `TRITON_CACHE_DIR=/cache/triton` when `/cache` (`SM120_CACHE_ROOT`)
  is writable, so b12x CuTe-DSL and Triton JIT survive container restarts.

Things that need **no** patch:

- `allocate_userbuffer`: `userbuffer.allocate_ep` builds the single-rank `deep_ep.EPBuffer` shim, which allocates
  nothing; `get_theoretical_num_sms` returns 0, so no SMs are reserved; `allocate_cp` returns None for CP=1.
- `runtime_config.configure_worker_kernels`: `deep_gemm.set_pdl` and `use_deterministic_algorithms` are state-only.
- `init_device_groups`: single-rank NCCL all_reduce works on SM120.
- CUDA graphs and MTP need no further DeepInfer patch beyond P8: graph safety is a property of the shims.
- DeepInfer's `forbid_cuda_sync` guard (`model_runner.py:324,359`) is turned off with `DEEPINFER_DISABLE_SYNC_GUARD=1`
  (set by `run_deepinfer.py` and the image): the torch reference paths, the single-rank DeepEP dispatch, the b12x lazy
  preparation and the capture hooks synchronise.

## 2. Source patches (files in this directory)

| file | applies to | what |
|---|---|---|
| `deepinfer-sequence-block-table.patch` | DeepInfer `csrc/engine/pybind_sequence.cpp` | Exposes `Sequence.block_table` (read-only) to Python. Needed only by the prefix cache ([docs/PREFIX_CACHE.md](../docs/PREFIX_CACHE.md)). The serving image applies it before building `_C`; `prefix_cache/build_patched_c.sh` builds a patched copy inside a dev container. |
| `b12x-tiny-decode-swiglu.patch` | installed b12x 1.5.1 (`patch -p1 -d <site-packages>`) | Makes the `w4a8_mx` tiny-decode MoE kernel (1-4 tokens, `n % 128 == 0`) apply `swiglu_limit`. Upstream bug: [flashinfer-ai/flashinfer#6866](https://github.com/flashinfer-ai/flashinfer/issues/6866). The serving image applies it; the runtime still defaults to the dynamic path, which clamps correctly and is faster on SM120. |
| `repro_b12x_tiny_decode_swiglu.py` | standalone, GPU + b12x only | Reproduces the tiny-decode clamp bug on random MXFP4 experts. |
| `repro_b12x_deterministic_direct.py` | standalone, GPU + b12x only | Reproduces the `deterministic_output=True` failure (section 3, second item; [#6941](https://github.com/flashinfer-ai/flashinfer/issues/6941)). |

Related tests: `tests/test_b12x_tiny_decode_swiglu.py` (T = 1..4 against the fp32 reference and the dynamic path,
with and without a limit; `--shape v41 --bench` for latency) and `tests/run_runtime_unit.sh` (tiny-decode patch test,
P8 sampler ops, GEMM/MoE graph tests).

## 3. Upstream b12x issues found

Both reproduce with b12x 1.5.1 (PyPI), torch 2.13.0+cu132, CUDA 13.2 on an RTX 5090, without a checkpoint.

**`fused_moe` w4a8_mx tiny-decode ignores `swiglu_limit`**
([flashinfer-ai/flashinfer#6866](https://github.com/flashinfer-ai/flashinfer/issues/6866)).
With `quant_mode="w4a8_mx"`, `source_format="fp4_e8m0_k32"` and an intermediate size with `n % 128 == 0`, plans with
`max_tokens <= 4` select the micro backend, which dispatches to `moe/_shared/kernels/tiny_decode.py`
(`_impl._launch_micro` -> `b12x::tp_moe_tiny_decode_launch`). `swiglu_limit` is never passed, and FC2 computes
`silu(gate) * up` unclamped. The dynamic path and the compact-micro kernel (`n % 128 == 64`) clamp correctly.

- Effect: models with a SwiGLU clamp (DeepSeek-V4.1, `swiglu_limit=10`) get wrong expert outputs in single-user
  decode whenever the expert shard has `n % 128 == 0`, e.g. full experts per GPU (n = 2304). A TP4 shard (n = 576) is
  not affected. With real V4.1 layer-2 experts: cos 0.79 against the clamped reference for tiny-decode, 0.998 for the
  dynamic path.
- Repro: `repro_b12x_tiny_decode_swiglu.py` (E=8, K=2048, N=512, top-2, T=1, x std 8 so that about a third of
  gate/up values exceed the limit). Unpatched, tiny-decode matches the unclamped reference and the dynamic path the
  clamped one.
- Fix (`b12x-tiny-decode-swiglu.patch`, about 100 lines, also posted upstream in
  [this comment](https://github.com/flashinfer-ai/flashinfer/issues/6866#issuecomment-6312667479)): `swiglu_limit` becomes a compile-time parameter of
  `MoETinyDecodeKernelBackend` and part of its cache key; FC2 clamps like the reference (`gate = min(gate, L)`,
  `up = clamp(up, -L, L)`, then `silu(gate) * up * w`); without a limit the generated code is unchanged. It is threaded
  through `_get_tiny_decode_kernel`, `_launch_tiny_decode_flat`, the custom op (new `float? swiglu_limit` argument),
  `_launch_micro` and the preparation prewarm.
- Workaround without the patch: `B12X_W4A8_TINY_DECODE=0` (the default here).
- Side note: under CUDA-graph replay on the RTX 5090, tiny-decode is slower than the dynamic path for the V4.1 shape
  (T=1 71.4 vs 67.6 us, T=4 305.8 vs 287.8 us).

**`fused_moe` `deterministic_output=True` fails for shapes routed "direct"**
([flashinfer-ai/flashinfer#6941](https://github.com/flashinfer-ai/flashinfer/issues/6941)).
`Caps(..., decode_config=None, deterministic_output=True)` for w4a8_mx raises `RuntimeError: planned dynamic direct
routing is unsupported for this launch shape` when the heuristic selects `dynamic_route_mode="direct"`.
`_heuristic_dynamic_route_mode()` evaluates the direct-routing candidate with `deterministic_output=False`, so it can
return "direct" for a deterministic plan, which `_dynamic_direct_routing_selected()` then rejects.

- Repro: `repro_b12x_deterministic_direct.py` (random experts with the V4.1 shape E=384, K=5120, N=2304, top-6;
  `--tokens 1,4,8,64,512`). On an RTX 5090, T = 1 and 4 (route "direct") fail; the same config with
  `dynamic_route_mode="grouped"` works and gives bit-identical repeats; T = 8, 64, 512 route "grouped" and work.
- Workaround (used by `sm120_moe`): resolve the heuristic config without `deterministic_output`, replace "direct" with
  "grouped", then pass that config with `deterministic_output=True`.
- Motivation: without `deterministic_output` the dynamic path combines the top-k expert outputs with atomic adds, so
  repeated runs differ (max abs diff about 0.25 in the test).
