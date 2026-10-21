# Changelog

All notable changes to this project are documented here. Versions follow [Semantic Versioning](https://semver.org/);
before 1.0 any minor release may change interfaces.

## [Unreleased]

### Fixed
- `eval/scenarios.py realistic`: `tokens_per_step` divided by every engine step, and behind the `--dp` proxy by the
  sum over all ranks (which step in lockstep), so speculative decoding read 0.3-0.6 instead of above 1. The server now
  counts, per request, the steps in which it received tokens (`/health` `stats.emit_steps`, plus `stats.tokens` and
  `config.nextn`), and the scenario reports tokens per decode step of a sequence at any concurrency, with
  `draft_acceptance_est` = (tokens_per_step - 1) / nextn. The `--dp` proxy reports `steps` / `flushes` per rank
  instead of summing them.

### Changed
- `eval/scenarios.py capacity`: a `missed` column names the limit a level broke (ttft, tpot, errors).
- `eval/compare_endpoints.py --teacher-forced`: the error for a recording that lacks the requested continuation says
  how to record it.

## [0.2.1] - 2026-10-21

### Fixed
- Prefill with `--prefix-cache` was split into `--prefix-cache-grain`-sized engine steps (512 tokens by default): a
  4k prompt took 8 steps instead of 1. Checkpoints now sit at multiples of the new `--prefix-cache-checkpoint-stride`
  (default: max(grain, `--max-num-batched-tokens`)), so the cache never adds steps; prompt-end and generation-end
  snapshots are unchanged. On one RTX 5090, 32k TTFT with the prefix cache went from 20.5 s to 8 s. Hits inside a
  prompt shorter than the stride need a smaller stride (see docs/PREFIX_CACHE.md).
- Multi-GPU EP prefill: the token counts are synchronised once per step instead of in every MoE layer (40 host syncs
  per step before), and the local experts run on the real rows only instead of on every rank's padded rows.
- Engram: `SM120_ENGRAM_STATS=1` now also prints periodically (`SM120_ENGRAM_STATS_INTERVAL`, default 60 s), not only
  at a clean exit.
- `chat_template_kwargs` in chat requests was ignored. `{"thinking": true}` (and `enable_thinking`), `reasoning_effort`
  and `drop_thinking` are now honoured; top-level `thinking` / `reasoning_effort` win. Non-boolean `thinking` values
  are rejected with 400 instead of being coerced.

### Added
- Engram page cache: start-up and after-load log lines report how much of the tables is resident; after all ranks
  have loaded, the server drops clean page-cache pages of the other weight shards and prefaults the tables if they fit
  into free RAM (`SM120_ENGRAM_PREFAULT_AT=ready`, `SM120_ENGRAM_DROP_WEIGHT_PAGES=0` to keep the pages).
- `--dp`: a peer-copy self-test at start-up sets `NCCL_P2P_LEVEL=SYS` for the ranks when GPU peer access works
  (an explicit `NCCL_P2P_LEVEL`/`NCCL_P2P_DISABLE` or `SM120_NCCL_P2P_LEVEL` wins).
- Faster `fp8_einsum` for long prompts (17–23 %, bit-identical).
- `SM120_EP_TRANSPORT=b12x`: optional b12x PCIe all-gather / reduce-scatter for EP prefill (4 ranks), with
  `tests/test_ep_b12x_transport.py`. NCCL stays the default; on the 4-GPU target it was faster above one row.
- `eval/scenarios.py` + `eval/prompts.json`: workload scenarios against any OpenAI-compatible endpoint (realistic
  prompts for speculative decoding, many long prompts, shared system prompt over several turns, mixed long/short
  load, capacity under a latency SLO).
- `eval/compare_endpoints.py --teacher-forced`: both endpoints (and an optional third reference: a file or an
  endpoint) score the same continuation position by position: top-1 agreement, KL per position, forced-token
  logprob delta, NLL per endpoint, and which endpoint is closer to the reference. Continuations from either endpoint,
  the reference, or human-written text. `tools/ref/ref_to_fidelity.py` turns reference-implementation dumps into
  reference files.
- docs/EVAL.md: KV-cache precision of the reference, DeepInfer and an FP8-KV vLLM deployment, and how to tell which
  engine is numerically right on the full model.
- Engram prefetch at request admission (`sm120_engram.EngramPrefetcher`, patch E7):
  - A prompt's rows are computed on the CPU exactly like DeepInfer's and read into the page cache by a background
    thread, ahead of the prefill chunks that need them, in admission order.
  - Bounded by `SM120_ENGRAM_PREFETCH_AHEAD`; aborts cancel queued prompts; results stay bit-identical.
  - Switches `SM120_ENGRAM_PREFETCH*`; `SM120_ENGRAM_TRACE=1` prints per-gather and per-prefetch timing.
  - Cold-cache TTFT on the RTX 5090 test machine: 32k-token prompt 9.2 s -> 6.7 s (`host`), 11.4 s -> 6.6 s
    (`uring`). Without engram it is 6.3 s.
- `tests_engram.py` T7: the prefetcher's row ids and page ranges against DeepInfer's.

### Changed
- Engram scale planes are kept in the page cache (`SM120_ENGRAM_SCALES=resident`, 5.7 GiB for both tables): one NVMe
  read per cold lookup instead of two (cold 4k-token gather 0.78 s -> 0.30 s). `uring` reads only the weight rows with
  O_DIRECT.
- `uring` mode gathers prefill-sized batches (`SM120_ENGRAM_URING_MMAP_TOKENS`, 256 tokens) through the mmap, so
  prefetched rows are used; decode batches stay O_DIRECT.

## [0.2.0] - 2026-10-21

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
