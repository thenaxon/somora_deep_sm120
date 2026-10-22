# Changelog

All notable changes to this project are documented here. Versions follow [Semantic Versioning](https://semver.org/);
before 1.0 any minor release may change interfaces.

## [Unreleased]

### Fixed
- `eval/loadgen.py`, `eval/scenarios.py`: a streamed request that ended with an error event, a malformed event or
  without `[DONE]` / a finish reason counted as a success; it is now an error, and its tokens are reported separately
  (`partial_out_tokens`). All eval and bench clients (`loadgen`, `scenarios`, `server/bench_client.py`,
  `tests/test_server.py`, `tests/ttft_conc.py`) read SSE from the decoded HTTP body: with chunked transfer encoding
  (vLLM) the raw socket could split an event. Shared parser `eval/sse.py`, test `tests/test_sse.py`.
- A missing TPOT was reported as 0 ms and passed the `capacity` SLO; it is now None and a miss.
- `tests/run_runtime_unit.sh`, `tests/determinism.sh`, `tests/e2e_matrix.sh`, `tests/mtp_check.sh`,
  `tests/tests_engram_graph.sh` and `tools/compare.py` exit non-zero on failure (pipefail, collected exit codes,
  integer tensors compared exactly, stale dumps removed per config, the engram negative control checked).
  `compare.py`: exit 2 for an invalid comparison (different input ids, capture errors, missing hidden states unless
  `--allow-missing-hidden`), `--decode-check identical|diverge` (exit 3; teacher-forced dumps are judged by the
  decode logits, top-1 and `--diverge-kl`, because their decoded ids are the forced ones), `--known-deviation layer0`
  for the documented layer-0 deviation (`e2e_matrix.sh` passes it, `TEST_KNOWN_DEVIATION` overrides).
  `determinism.sh` never ran its comparison. `mtp_check.sh` printed the wrong `accepted_log` key. Test
  `tests/test_compare_exit.py`.

### Changed
- Eval method (`eval/measure.py`, docs/EVAL.md): a per-run nonce at the start of every prompt (no prefix-cache
  help, also across runs), an unmeasured warm-up per level with the same lengths, 3 repeats per level (median and
  range), steady-state throughput, tails as p99 only from >= 100 samples (else the maximum, labelled),
  `output_tokens_mean` and per-user tok/s always reported. `scenarios.py`: `ignore_eos` by default (`--allow-eos`),
  `mixed` counts short requests only within the phase. `loadgen.py`: requests per level max(16, 3 x concurrency).

### Added
- `SM120_PROFILE_DECODE=N` (diagnostics, off by default): every engine process that holds requests profiles N decode
  steps once (torch profiler; same per-category GPU and host tables as `SM120_PROFILE_PREFILL`), also per rank with
  `--dp` (docs/SERVER.md).

### Fixed
- b12x's deterministic MoE: `SM120_MOE_DETERMINISTIC=1` (also on with `DEEPINFER_BATCH_INVARIANT=1` or `B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1`) gave
  wrong MoE outputs in two cases, both b12x 1.5.1 `deterministic_output` bugs:
  - plans with `dynamic_tile_m` < 64 (b12x's own choice, e.g. E=16 at 5-64 tokens, E=96 at 65-512): cos ~0 against
    the default path. The shim now uses at least 64 rows per tile in deterministic mode.
  - masked routes (id -1) summed stale scratch rows. With `--dp` every rank masks the routes to the other ranks'
    experts, so real tokens were wrong (single GPU: only discarded padding rows). They now go to a local expert with
    weight 0.
  The default b12x MoE and `--deterministic` (torch MoE) were not affected. New tests in `tests/test_gemm_moe_graphs.py`
  and `tests/sweep_moe_deterministic.py`. Upstream: [flashinfer-ai/flashinfer#7021](https://github.com/flashinfer-ai/flashinfer/issues/7021).

### Changed
- The server no longer refuses b12x's deterministic MoE (0.3.0 did). It logs one start-up line (bit-exact, slower
  decode, with `--dp N` the masked routes computed at weight 0) and a warning if the installed b12x is not 1.5.1,
  the version the workarounds were validated on. `SM120_ALLOW_B12X_DETERMINISTIC`, the 0.3.0 override, is still
  accepted and has no effect.
- Deterministic MoE cost (full-depth test model, RTX 5090, CUDA graphs, TPOT B=1/8/32): 16.1/20.8/25.4 ->
  19.1/23.4/27.0 ms, 1.06-1.19x (0.2.x: 2-2.9x with the small tiles that gave wrong rows).

## [0.3.0] - 2026-10-22

First public release, experimental. Accepted on 4× RTX PRO 6000 Blackwell as v0.3.0-rc1 (same code). 0.2.5 was not
tagged; its changes are listed here.

### Fixed
- b12x's deterministic MoE (`SM120_MOE_DETERMINISTIC=1`, or `DEEPINFER_BATCH_INVARIANT=1` /
  `B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1`, which turn it on) gives wrong outputs with b12x 1.5.1: `deterministic_output`
  writes wrong rows when the dynamic tile is below 64 (b12x's own choice for some batch sizes, also on one GPU), and it
  does not zero the rows of masked routes (`topk_ids` -1), which every expert-parallel rank uses for the other ranks'
  experts. The server now refuses it at start-up (`SM120_ALLOW_B12X_DETERMINISTIC=1` overrides, for tests).
  `--deterministic` (torch MoE) and the default b12x MoE are not affected. The docs no longer recommend the mode.
- The start log shows the effective prefill intervals (`prefill intervals: ...`).
- `--dp > 1`: TTFT of about 2 s even for short prompts. With several DP ranks DeepInfer votes the step type, and a
  prefill happens at the `DEEPINFER_PREFILL_ATTEMPT_INTERVAL` points only if every rank has a waiting prompt. A
  request that only one rank holds waited for the next `DEEPINFER_PREFILL_ANY_ATTEMPT_INTERVAL` point (default 100
  decode steps). The server now sets it (`--prefill-any-interval`, docs/SERVER.md).
- `--dp > 1` with both prefill intervals at 1: while any rank had prompt chunks waiting, the decoders of all ranks got
  about one token per prefill chunk (mixed load: TPOT p99 401 ms). `--prefill-interval` and `--prefill-any-interval`
  now default to `auto`: 1 / 1 for `--dp 1`, 16 / 16 for `--dp > 1`, so that at least 16 decode steps run between two
  chunks. Measured on 4× RTX PRO 6000 (`--nextn 7`), against 0.2.4: TTFT 58 tokens 2.0 → 0.85 s, 4k 2.9 → 1.4 s, 32k
  24 → 11 s; mixed load short-chat TPOT p99 57 → 65 ms (401 ms with 1 / 1); loadgen 4096×16 376 → 388 tok/s. The image no longer sets the two `DEEPINFER_PREFILL_*` variables (an explicit `-e` still overrides).

### Changed
- Documentation for a public release: the README states the hardware the full model needs (4× RTX PRO 6000, about
  256 GB RAM; a single RTX 5090 only runs truncated test checkpoints), the recommended configuration
  (`SM120_ENGRAM_MODE=pinned`, `--gpu-memory-utilization 0.95`, `--nextn 7`) and the current results against vLLM;
  docs/RESULTS.md has the v0.3.0 measurements and the progress since v0.2.0; the runbook's launch command is the
  recommended configuration. Start-up banner: "RTX PRO 6000 Blackwell" instead of "RTX 5090 / RTX PRO 6000".
- `SM120_ENGRAM_MODE=pinned`: the start-up check counts the driver's host memory for the registration (about 14 MiB per
  GiB of tables per rank), so `--dp 4` needs about 225 GiB available before start (was 215). A KV sizing failure in
  pinned mode now says how much GPU memory the registration took and suggests raising `--gpu-memory-utilization`.
  Logs: the registration's GPU and host cost per rank, and a host-memory breakdown after the fill.

### Documentation
- README and `docs/RESULTS.md`: the numbers of the v0.3.0-rc1 acceptance run on 4× RTX PRO 6000 (single user 136 / 84–87
  tok/s for code / chat, 4k × 64 users 460 tok/s, 32 users within the latency targets like vLLM, 15-minute soak
  without errors).
- `docker run` examples use `--stop-timeout 120`: with `--dp 4` and pinned engram tables the orderly shutdown takes
  longer than Docker's default 10 s, after which Docker kills the server (harmless, nothing is left behind).
- README: about 12 GiB of host RAM stay free while serving on a 246 GiB host.

## [0.2.4] - 2026-10-22

### Fixed
- `--dp` with CUDA graphs failed at start-up in 0.2.3 (`sm120_moe layer 0: b12x plan for max_tokens=64 requested
  during CUDA graph capture`; the other ranks then hung). DeepInfer's eager warm-up before each graph capture still ran
  the EP layer at the fixed capacity, so the rank-local experts were planned only for that bucket, while the capture
  itself used the graph size (0.2.3). `CudaGraphRunner.capture` and `DFlashCudaGraphRunner.capture` are now marked as a
  whole (`dp_sync.patch_capture`), and warm-up and capture both use the graph size as EP rows, so the warm-up creates
  every bucket the capture needs. New `tests/test_ep_capture_plans.py` (world 4 emulated, real b12x plan path) fails
  on the 0.2.3 behaviour. The extra plans (buckets 4 and 64 next to 512) add about 17 MiB of shared b12x scratch
  per GPU, once, not per layer.
- A failed engine start (e.g. a CUDA-graph capture that raised) left the server process hanging in the interpreter
  teardown, and with `--dp` the proxy waited for it forever. A failed CUDA-graph capture now exits the process at
  once with code 4 (runtime patch P9; `SM120_CAPTURE_FAIL_FATAL=0` re-raises instead), any other start-up failure
  with code 3; the `--dp` proxy stops the other ranks (SIGTERM, `kill` after 10 s) and exits non-zero.

### Changed
- `SM120_ENGRAM_MODE=pinned`: each rank logs the GPU memory its registration of the shared segment costs (page tables,
  about 2 MiB per GiB of tables, ~0.35-0.4 GiB per GPU for the full tables). It comes out of the KV cache: on a tight
  budget raise `--gpu-memory-utilization` (on 4× RTX PRO 6000 with `--max-model-len 40960`: 0.92 left 0.17 GiB of KV,
  which pinned mode used up). Documented in docs/MULTI_GPU_RUNBOOK.md §3 and docs/ENGRAM.md, which also corrects the
  gather time to ~2.3 ms per 98k rows after a one-time warm-up.

## [0.2.3] - 2026-10-22

0.2.2 was tested but not tagged; its changes ship with 0.2.3.

### Fixed
- `eval/scenarios.py realistic`: `tokens_per_step` divided by every engine step, and behind the `--dp` proxy by the
  sum over all ranks (which step in lockstep), so speculative decoding read 0.3-0.6 instead of above 1. The server now
  counts, per request, the steps in which it received tokens (`/health` `stats.emit_steps`, plus `stats.tokens` and
  `config.nextn`), and the scenario reports tokens per decode step of a sequence at any concurrency, with
  `draft_acceptance_est` = (tokens_per_step - 1) / nextn. The `--dp` proxy reports `steps` / `flushes` per rank
  instead of summing them.
- `--dp` decode graphs: the EP MoE layers were captured at start-up, before any step had been DP-synced, with the
  fixed capacity (`max_num_decoding_tokens` rows per rank, 128 at 16 seqs x nextn 7), and every replayed graph
  padded its all-gather / reduce-scatter and expert call to it. With the DP sync on, each graph is now captured with
  its own size as rows (all ranks capture the same sizes in the same order). A `--mtp-varlen` decode cost curve
  profiled with an older version (`/cache/decode_perf`) measured the padded graphs; delete it once.

### Added
- `SM120_ENGRAM_MODE=pinned`: the engram tables are kept once in shared, pinned host memory for all ranks, the way the
  baseline keeps them in RAM.
  - A memfd created by the server or the `--dp` proxy is registered with `cudaHostRegister` in every rank and filled
    from the shards after load.
  - Lookups are GPU gathers from it, also inside CUDA graphs, and never touch the NVMe.
  - The start-up fails with the numbers when tables + ranks' memory + headroom do not fit.
  - `tests/test_engram_pinned.py` checks one copy for two ranks, bit-exact rows and graph replay.
- Adaptive speculative decoding: `server.py --mtp-varlen` (with `--graphs --nextn K`) enables DeepInfer's
  cost-model draft trimming (`DEEPINFER_ENABLE_MTP_VARLEN`). The decode graphs are profiled at start-up (cached in
  `/cache/decode_perf`); each step verifies only as many drafts per sequence (≤ K) as the profiled step cost and the
  DSpark confidence head justify. Measured on the RTX 5090 test model: K=5/7 is +35-45 % output tok/s over K=3 at one
  user; the trimming itself is neutral at ≤ 8 users (docs/MTP_YOCO.md §9).
- `/health` reports verified / accepted draft tokens (`stats.mtp_verified`, `stats.mtp_accepted`, runtime patch P12);
  `eval/scenarios.py realistic` reports `draft_acceptance` (accepted / verified) and `drafts_verified_per_step`.
- YOCO compact prefill truncation, opt-in: `server.py --yoco-compact [KEEP]` (KEEP default 140, ≥ 128 + nextn + 1)
  runs the YOCO consumer half (layers 20-39) on only the last KEEP rows of each prefill (DeepInfer's
  `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact`). TTFT on the RTX 5090 test model: 4k 1.04-1.11 s → 0.65-0.77 s,
  32k 6.68-6.97 s → 4.34-4.70 s; greedy tokens identical. With `--dp` the consumer half's EP layers exchange the kept-row
  counts once per step (runtime patch P13). Not exact with `--prefix-cache`, so the server refuses the combination
  (docs/MTP_YOCO.md §8a).
- `SM120_ENGRAM_PIN=mlock`: after load, rank 0 prefaults the engram tables and locks them. All ranks share that one
  locked page-cache copy (memory counted once), so the tables stay fully in RAM like the baseline's pinned copy.
  - The server refuses to start, with a message naming the numbers, if the tables plus
    `SM120_ENGRAM_PIN_HEADROOM_GB` (default 8) do not fit into MemAvailable, or if `RLIMIT_MEMLOCK` is too low.
  - With `--dp` the proxy then stops all ranks.
- `SM120_PROFILE_PREFILL=N`: every engine process profiles the prefill of its N-th request (torch profiler) and
  writes a trace plus a table to `SM120_PROFILE_DIR`: GPU time per kernel category (incl. NCCL and EP pack), host time
  per named range (engine step, engram fetch/wait, DP step sync, prefix cache, DeepInfer's forward/sample ranges),
  blocking CUDA calls and the engine thread's timeline.
- Diagnostic switches, defaults unchanged: `SM120_ATTN_PREFILL_SYNC=0` (prefill attention without the per-layer
  width read-back; about 2× more attention work on one GPU) and `SM120_EP_COMPACT=0` (EP prefill without the row
  compaction).
- `SM120_ENGRAM_RESIDENCY_INTERVAL`: periodic page-cache residency log of the engram tables while serving.
- The server logs each rank's host footprint after load (anon, shmem, page tables, torch's pinned cache).

### Changed
- `eval/scenarios.py capacity`: a `missed` column names the limit a level broke (ttft, tpot, errors).
- `eval/compare_endpoints.py --teacher-forced`: the error for a recording that lacks the requested continuation says
  how to record it.
- Engram prefault after load (`SM120_ENGRAM_PREFAULT=1`) runs when the tables fit into MemAvailable minus
  `SM120_ENGRAM_PREFAULT_HEADROOM_GB` (default 4). The old rule wanted 5 % above the tables and skipped hosts with
  210 GB available for 203 GB of tables. `SM120_ENGRAM_PREFAULT=force` prefaults regardless.

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
