# Changelog

All notable changes to this project are documented here. Versions follow [Semantic Versioning](https://semver.org/);
before 1.0 any minor release may change interfaces.

## [Unreleased]

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
- Multi-GPU runbook: Hugging Face hub caches must be mounted at their root (snapshot files are symlinks into
  `blobs/`); realistic image size (about 7 GB); building when the host cannot reach the repository; host-RAM
  budgeting for the engram modes (per-rank memory, page cache vs. pinned memory, baseline stopped).
- `eval/quality_check.py`: items can be marked `known_fail` (reported as KFAIL/KPASS, not counted). `math-reason`
  is marked: the model answers 2 without thinking mode, independent of the engine.
- Thinking-mode streaming sends the reasoning as `delta.reasoning_content` (it was sent as `content`).
- Requests submitted after an engine crash now fail at once instead of waiting for `--request-timeout`.

### Fixed
- `/v1/completions` rejected a one-token token-id prompt (`[id]`) as a malformed string list.
- xgrammar 0.2.2 admits raw control characters in length-bounded JSON strings; the server patches that character class.
- xgrammar 0.2.2 builds wrong digit patterns for some integer ranges (`minimum: 0, maximum: 120` admits 121-129); ranges
  with up to 1024 values are passed to it as an explicit `enum`.

### Known issues
- With CUDA graphs (`--graphs`, the image default), a server with some request history can lose its engine on the first
  prefill of a new size (illegal instruction / illegal memory access in the decode graph replay). This also happens without
  tools. Eager mode is not affected. Details and a partial workaround: [docs/SERVER.md](docs/SERVER.md#tool-calling-and-structured-output).

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
