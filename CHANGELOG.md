# Changelog

All notable changes to this project are documented here. Versions follow [Semantic Versioning](https://semver.org/);
before 1.0 any minor release may change interfaces.

## [0.2.0] - unreleased

### Added
- Tool calling in `/v1/chat/completions`: `tools`, `tool_choice` (`auto`, `none`, `required`, named function),
  `parallel_tool_calls`, history with assistant `tool_calls` and `role: "tool"` messages, `message.tool_calls` with
  `finish_reason: "tool_calls"`, and streaming `tool_calls` deltas. Rendering and parsing use the checkpoint's
  `encoding/encoding.py` (DSML format) (`server/tools.py`).
- `response_format`: `text`, `json_object`, `json_schema` (`strict` honoured).
- Grammar-constrained decoding through DeepInfer's structured output (xgrammar EBNF in `SamplingParams.so_args`):
  `tool_choice` `required` or a named function only admits valid calls whose arguments follow the JSON schema;
  `response_format` output follows the schema. In thinking mode the grammar starts after `</think>`.
- Tests: `tests/test_tools.py` (CPU), `tests/test_tools_server.py`, `tests/test_openai_sdk.py` (official `openai`
  SDK incl. `chat.completions.parse` with pydantic), `tests/run_tools_tests.sh`.
- Image input in `/v1/chat/completions`: OpenAI `image_url` content parts (`data:` base64 and `http(s)` URLs), passed to
  DeepInfer's multimodal path (`server/images.py`). Limits: `--max-images`, `--max-image-bytes`, `--max-image-pixels`,
  `--no-image-urls`; URLs on private addresses are refused unless `--image-fetch-private`. Image tokens count in
  `usage.prompt_tokens`. Tests: `tests/test_images.py` (CPU), `tests/test_images_server.py`, an SDK image check.
- `eval/quality_set.jsonl`: 8 tool-calling, 3 structured-output and 3 image items; `eval/quality_check.py` scores them
  and reports scores per category.

### Changed
- Thinking-mode streaming sends the reasoning as `delta.reasoning_content` (it was sent as `content`).
- `eval/quality_check.py` reports scores per category (core / tool / json / image).

### Fixed
- xgrammar 0.2.2 admits raw control characters in length-bounded JSON strings; the server patches that character class.
- xgrammar 0.2.2 builds wrong digit patterns for some integer ranges (`minimum: 0, maximum: 120` admits 121-129); ranges
  with up to 1024 values are passed to it as an explicit `enum`.
- Text-only prompts carry an empty `multi_modal_data` when the model has a vision processor (as DeepInfer's
  `LLM.generate` does); a batch that mixed text and image prompts failed an assertion in `deepseek_vl`.
- ViT attention shim: 4-D SDPA, so a full-size image no longer falls back to the math kernel (about 10 GiB, CUDA OOM).

## [0.1.1] - 2026-10-21

### Fixed
- CUDA-graph server crash (`CUDA error: an illegal instruction` in decode graph replay) after a prompt that needed a b12x plan not seen before: the lazy preparation evicted and unloaded b12x programs that captured graphs still launch. The programs launched during graph warm-up and capture are now retained (`sm120_patches` P9, test `tests/graph_replay_lazy_prep.sh`).
- `libnuma1` in both images; DeepInfer's NUMA binding no longer logs tracebacks. On single-node hosts the binding is
  skipped (new runtime patch P10): with libnuma present, `bind_numa(-1)` would segfault.
- The serving image sets `DEEPINFER_MULTIMODAL_POOL_SIZE_GB=2` (DeepInfer default 8): 2 GB instead of 8 GB pinned host
  memory per rank for image inputs. Sizing and the `/dev/shm/psm_*` leftovers after a crash are in
  `docs/MULTI_GPU_RUNBOOK.md` §3.
- `tests/run_server_tests.sh` stops the whole server process group and waits for GPU memory; the mid-decode server
  check no longer depends on timing under CUDA graphs.
- No more `/dev/shm/psm_*` leaks (2 GB per engine and stop): runtime patch P11 unlinks the multimodal pool's name after
  the worker fork; `server.py` turns SIGTERM into a clean shutdown (as PID 1 the default action was to ignore it until
  `docker stop` killed it), and the `--dp` proxy forwards it to the ranks. `docker stop` now takes about 1.5 s, exits
  with 0 and logs no tracebacks: forked worker processes exit quietly and the engine is closed before interpreter exit.
- `/v1/completions` accepts a prompt of a single token id; after an engine crash, new requests fail at once instead of
  waiting for the per-token timeout.

### Added
- Start-up banner: somora lettering with "deep sm120", version, checkpoint, DP size and port
  (`server/banner.py`, `SM120_BANNER=0` disables it).
- `tests/graph_replay_lazy_prep.sh`: regression test for the CUDA-graph crash (new prompt lengths after a request history).
- `tests/mm_pool_check.py`: text+image requests through DeepInfer's own API, with and without the multimodal pool
  (pipeline check; the truncated test checkpoints carry no real vision weights).

### Changed
- Multi-GPU runbook: Hugging Face hub caches must be mounted at their root (snapshot files are symlinks into
  `blobs/`); realistic image size (about 7 GB); building when the host cannot reach the repository; host-RAM
  budgeting for the engram modes (per-rank memory, page cache vs. pinned memory, baseline stopped), the multimodal
  pool and NUMA (`--cap-add SYS_NICE` on multi-socket hosts).
- `eval/quality_check.py`: items can be marked `known_fail` (reported as KFAIL/KPASS, not counted). `math-reason`
  is marked: the model answers 2 without thinking mode, independent of the engine.

### Known limitations
- b12x MoE programs used only by eager prefill are evicted by later lazy b12x preparations and reloaded on next use
  (load time on some first-seen prefill shapes; graphs are not affected). See docs/RUNTIME.md, open point 6.

## [0.1.0] - 2026-10-21

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
