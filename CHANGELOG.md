# Changelog

All notable changes to this project are documented here. Versions follow [Semantic Versioning](https://semver.org/);
before 1.0 any minor release may change interfaces.

## [0.1.1] - unreleased

### Fixed
- `libnuma1` in both images; DeepInfer's NUMA binding no longer logs tracebacks. On single-node hosts the binding is skipped
  (new runtime patch P10): with libnuma present, `bind_numa(-1)` would segfault.
- The serving image sets `DEEPINFER_MULTIMODAL_POOL_SIZE_GB=2` (DeepInfer default 8): 2 GB instead of 8 GB pinned host
  memory per rank for image inputs. Sizing and the `/dev/shm/psm_*` leftovers after a crash are in
  `docs/MULTI_GPU_RUNBOOK.md` §3.

- No more `/dev/shm/psm_*` leaks (2 GB per engine and stop): runtime patch P11 unlinks the multimodal pool's name after
  the worker fork; `server.py` turns SIGTERM into a clean shutdown (as PID 1 the default action was to ignore it until
  `docker stop` killed it), and the `--dp` proxy forwards it to the ranks.

### Added
- `tests/mm_pool_check.py`: text+image requests through DeepInfer's own API, with and without the multimodal pool.

## [0.1.0] - unreleased

First release. Experimental.

### Added
- Drop-in SM120 replacements for DeepInfer's SM100-only kernel libraries (`sm120/`): `flash_mla` (sparse MLA decode
  and prefill, fused norm/RoPE/cast), `deep_gemm` (FP8 block GEMM, MQA indexer logits, grouped/fused MoE), `deep_ep`
  (single-GPU buffers, Engram buffer), `deep_select` (top-k), all on b12x 1.5.1, Triton, or exact torch references.
- Runtime patches installed through `sitecustomize` (`sm120/sm120_patches.py`, `sm120_patches_engram.py`), documented
  in `patches/PATCHES.md`.
- CUDA-graph decode and MTP/DSpark speculative decoding on SM120.
- Engram on SM120 without RDMA: host-mmapped tables, gathered asynchronously under CUDA graphs.
- Multi-GPU expert parallelism for PCIe-only machines (DP attention + EP MoE over NCCL, DP proxy). Compile-checked and
  tested in single-GPU emulation only.
- Optional MoE tuning switches (`SM120_MOE_PLAN=session`, `SM120_MOE_GRAPH_BUCKETS=1`, `SM120_B12X_AUTOTUNE=1`); defaults unchanged.
- OpenAI-compatible server (`server/`) with streaming, cancellation, seeded sampling, logprobs and exact prefix caching
  (`prefix_cache/`).
- Optional weight-only FP8 lm_head for decode (`SM120_LMHEAD_IMPL=fp8`, off by default).
- Validation tooling: DeepSeek reference runner, checkpoint truncation (3-layer, YOCO, full-depth presets), per-layer
  and intra-layer comparison against the reference noise floor (`tools/`).
- Serving image (`docker/Dockerfile`), which pins DeepInfer `8b16149741a4935e1dec8cea27853f273b374c5b` and b12x 1.5.1 and
  applies the b12x tiny-decode SwiGLU clamp fix ([flashinfer-ai/flashinfer#6866](https://github.com/flashinfer-ai/flashinfer/issues/6866), patch posted
  [upstream](https://github.com/flashinfer-ai/flashinfer/issues/6866#issuecomment-6312667479)). There is also a dev image (`docker/Dockerfile.dev`).

### Known limitations
- Validated on one RTX 5090 with truncated checkpoints only. The full model has not been run (it needs multiple GPUs).
- Layer 0 deviates from the reference by more than the noise floor. This is a property of DeepInfer's `fused_transform` rounding.
- b12x 1.5.1 bugs, worked around here and filed upstream: tiny-decode ignores `swiglu_limit`
  ([flashinfer-ai/flashinfer#6866](https://github.com/flashinfer-ai/flashinfer/issues/6866); patched in the image, default uses the dynamic path anyway) and
  `deterministic_output` fails with direct routing ([flashinfer-ai/flashinfer#6941](https://github.com/flashinfer-ai/flashinfer/issues/6941); grouped routing is forced).
- The default b12x MoE is not run-to-run deterministic. `SM120_MOE_DETERMINISTIC=1` is bit-exact but 2-2.9x slower in decode.
- MTP: nextn 1 with `DEEPINFER_SAMPLER_FORCE_TOPK=1024` at batch > 1 accepts no drafts (correct output, no speed-up).
