# OpenAI-compatible server

A minimal, dependency-free HTTP server (stdlib `ThreadingHTTPServer`) in front of one DeepInfer engine with continuous
batching. DeepInfer's source is not modified: the engine is built exactly like `tools/run_deepinfer.py` (SM120 shims +
runtime patches) and driven through its public `LLMEngine` API. The serving image's entrypoint (`docker/entrypoint.sh`)
starts this server.

| file | content |
|---|---|
| `server/server.py` | HTTP layer, OpenAI request/response mapping, chat template, incremental detokenisation, stop strings, SSE |
| `server/engine.py` | `EngineLoop`: owns the DeepInfer engine on one thread, continuous batching, add/abort between steps |
| `server/logprob_tap.py` | raw-logit logprobs (OpenAI `logprobs` / `top_logprobs`); DeepInfer's own logprobs are post-temperature |
| `server/dp_proxy.py` | `--dp N`: one engine process per GPU (DP attention + EP MoE) behind a least-loaded proxy |
| `server/bench_client.py` | TTFT / TPOT client: multi-turn streaming chat (exercises the prefix cache) + concurrent burst |
| `server/with_server.sh` | run a command against a temporary server; stops the whole process group and waits until the GPU memory is free |
| `tests/test_server.py` | plumbing tests against a running server (stdlib client) |
| `tests/run_server_tests.sh` | start the server, wait for `/health`, run the tests, stop |
| `tools/diag_repeat.py` | determinism and per-step latency probe of the bare engine (no HTTP) |

## Run

Inside the image (`PYTHONPATH=/opt/sm120/sm120:/opt/deepinfer` is preset):

```bash
python3 /opt/sm120/server/server.py --ckpt /work/ckpt3 --port 8000 --max-num-seqs 8     # b12x backends (fast)
python3 /opt/sm120/server/server.py --ckpt /work/ckpt3 --port 8000 --deterministic      # reproducible outputs
# options: --max-model-len 16384 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.9 --max-blocks 256
#          --nextn N (MTP, needs mtp.* layers) --graphs (CUDA graphs) --zos 0|1 --prefill-interval 1 --model-name ID
#          --prefix-cache [--prefix-cache-grain 512 --prefix-cache-entries 64 --prefix-cache-max-block-frac 0.5]

curl -s localhost:8000/v1/models
curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"ckpt3","prompt":"The lighthouse keeper","max_tokens":32,"temperature":0}'
curl -N localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"ckpt3","messages":[{"role":"user","content":"Hi"}],"max_tokens":32,"stream":true}'

# tests (server must be running), or all in one go:
python3 /opt/sm120/tests/test_server.py --url http://127.0.0.1:8000
/opt/sm120/tests/run_server_tests.sh /work/ckpt3 8000 /work/srv.log --deterministic -- \
  --di-ref /work/srv_ref_di_torchmoe.pt --prompt-ids /work/prompt_ids.json
```

Through the image entrypoint, `--model PATH` maps to `--ckpt` and CUDA graphs are on by default (`--no-graphs` turns them off);
see [../docker/README.md](../docker/README.md).

The OpenAI Python client works with `base_url="http://host:8000/v1"` and any `api_key` (it is not checked).

## API

| endpoint | supported |
|---|---|
| `GET /v1/models` | the one model (`--model-name`, default the checkpoint directory name) and its `max_model_len` |
| `GET /health` | `{"status": "ok" \| "dead", "stats": {...}}`: engine steps, flushes, added, aborted, finished, max_running, busy_s (+ `prefix_cache` stats when enabled) |
| `POST /v1/completions` | `prompt`: a string, a one-element list of strings, or a list of token ids. `max_tokens`, `temperature`, `top_p`, `top_k`, `stop` (string or up to 4 strings), `seed`, `ignore_eos`, `stream`, `stream_options.include_usage`. Extension: `return_token_ids` (non-streaming) |
| `POST /v1/chat/completions` | `messages` rendered with the checkpoint's `encoding/encoding.py` (`encode_messages`, `thinking_mode="chat"`). `"thinking": true` or `reasoning_effort` select `thinking` mode; the text before `</think>` is returned as `message.reasoning_content`. Same sampling fields, plus `max_completion_tokens` |

**Logprobs** (non-streaming, nextn 0): completions `logprobs: K` → `{tokens, token_logprobs, top_logprobs, text_offset}`;
chat `logprobs: true, top_logprobs: K` → `{content: [{token, logprob, bytes, top_logprobs}]}`. K <= 20. These are log_softmax of the
**raw** logits, like vLLM's default, not DeepInfer's post-temperature values. `return_tokens_as_token_ids: true` gives
`"token_id:<n>"` strings.

**Multi-GPU:** `--dp N [--ep M]` spawns N rank processes (engine_rank r, data_parallel_size N, expert_parallel_size M=N) on ports
PORT+1..PORT+N. `dp_proxy.py` serves PORT and routes each POST to the least-loaded rank; `/health` aggregates the ranks. Idle
ranks poll every 5 ms, because `has_unfinished_requests` is a DP-group all-reduce. Tested only with one rank (`--force-proxy`);
EP with N=4 has not run on real hardware yet ([MULTI_GPU_RUNBOOK.md](MULTI_GPU_RUNBOOK.md)).

**Not supported:** `n > 1`, `best_of`, streaming logprobs, `echo`, tools/function calling (encoding.py can render tools; output
parsing is not wired), images, and `presence`/`frequency_penalty`/`logit_bias` (DeepInfer's `SamplingParams` has none of them).
`max_tokens` defaults to the remaining context and is capped at `max_model_len - prompt - 1`.

Details:

- **Tokenisation:** raw `tokenizers` with `add_special_tokens=False`. DeepInfer's tokenizer wrapper prepends BOS to plain text
  (`hugging_face/tokenizer.py:26`). The chat template already contains BOS; a plain completion prompt gets exactly what the caller sends.
- **Detokenisation:** the full output is decoded on every token. A trailing U+FFFD (an incomplete UTF-8 sequence) and the last
  `len(longest stop) - 1` characters are held back until they are safe to send.
- **Stop strings:** matched on the decoded text. The text is cut before the match, `finish_reason="stop"` is returned, and the
  request is **aborted in the engine** so it stops consuming batch slots. EOS ends with `"stop"`, `max_tokens` with `"length"`.
- **Cancellation:** a streaming client that disconnects (the next write raises `BrokenPipe`/`ConnectionReset`) aborts its request
  (`engine.abort_request`). A non-streaming request whose client goes away runs to completion. A per-token timeout
  (`--request-timeout`, default 600 s) aborts stuck requests.
- The listen backlog is 1024 (the stdlib default of 5 hangs closed-loop load tests at concurrency 64).

## How the engine is driven (`server/engine.py`)

DeepInfer has no online/async entry point: `LLM.generate` adds all prompts, then calls `LLMEngine.step()` until
`has_unfinished_requests()` is False, then calls `flush_inflight_reqs()` once. With the zero-overhead scheduler (ZOS, the
default), `step()` returns the previous step's outputs, so the last ones only come out of the flush.

The scheduler is fully dynamic. `add_request` only appends to the C++ scheduler's waiting queue, and `abort_request` removes a
sequence; both are legal between any two `step()` calls (`llm_engine.py:472-656`, `csrc/engine/scheduler.hpp`). The server
therefore owns the engine on one thread and replaces `LLM._run_engine` with a loop that drains an inbox between steps:

```
loop:  apply queued add/abort  ->  step() while unfinished  |  one flush_inflight_reqs() when it goes idle  |  block on inbox
       clone every RequestOutput (pool slots are recycled by the next step), route by request_id to the handler's queue
```

Every DeepInfer and CUDA call happens on that thread. HTTP handler threads only exchange messages with it through queues.

**`DEEPINFER_PREFILL_ATTEMPT_INTERVAL`:** the NORMAL scheduler only tries a prefill every N decode steps (default 70), and idle
placeholder steps count (`scheduler.hpp:296-330`, built for DP-wide prefill voting). At about 32 ms per eager step that alone is
about 2.3 s of TTFT. The server sets it to `--prefill-interval 1`. The cost is that one decode step of the running requests waits
for each prefill chunk (bounded by chunked prefill, `max_num_batched_tokens`).

## Determinism

The bare engine (`tools/diag_repeat.py`, no HTTP) gives different tokens for the same prompt, run after run, with the default b12x
backends. This holds for greedy decoding and for seeded sampling (DeepInfer's sampler itself is deterministic in
`(seq_seed, position)`, `tilelang_ops/random_sampling.py`). Bisected with `--max-num-seqs 1`:

| config | identical over 3 runs |
|---|---|
| defaults (b12x GEMM / MoE / attention) | no |
| `SM120_GEMM_IMPL=torch` (b12x MoE) | no |
| `SM120_MOE_IMPL=torch` | **yes** (greedy and seeded sampling) |
| `SM120_ATTN_IMPL=torch` | out of memory at profiling (torch prefill attention) |

So **b12x `fused_moe` is not run-to-run deterministic**: the logits differ slightly, and the chaotic truncated test model amplifies
that into different tokens. `--deterministic` sets `SM120_MOE_IMPL=torch`; then the server's greedy tokens equal
`run_deepinfer.py`'s exactly (checked by `test_server.py`). See [RUNTIME.md](RUNTIME.md) for the deterministic b12x mode
(`SM120_MOE_DETERMINISTIC=1`).

## Test results (RTX 5090, 3-layer test checkpoint `ckpt3`, eager)

`--deterministic`: **18/18 passed**.

- `/v1/models` and `/health` respond.
- Greedy is deterministic and its token ids equal `run_deepinfer.py` (torch MoE).
- Streaming text equals the non-streaming text.
- Stop strings work, streaming and non-streaming.
- Chat works: streaming, non-streaming and thinking mode.
- 8 concurrent streams with 16-72 tokens each all finish and overlap (engine `max_running` 8, every stream's first token at 0.20 s).
- A request added while a 200-token request decodes finishes after 0.65 s, long before the long one.
- A disconnecting stream is aborted in the engine (`aborted +1`), and the server keeps serving.
- Seeded sampling is reproducible, and a different seed gives a different output.
- Bad requests get 400.

`--deterministic --prefix-cache`: **21/21 passed** (adds prefix-cache hit checks; a repeated 1503-token prompt hits 1024 tokens,
TTFT 1.82 s → 0.56 s, identical tokens).

Default b12x mode: **12/12 hard checks passed**. The 6 repeatability checks (greedy repeat, stream == non-stream, stop strings derived
from a previous greedy run, seeded sampling) differ, as expected from the b12x MoE nondeterminism above.

| | `--deterministic` (torch MoE) | default (b12x MoE) |
|---|---|---|
| TTFT, single 9-token prompt | 0.16-0.20 s | 0.04-0.07 s |
| 8 concurrent streams (16-72 tokens each), wall | 7.2 s | 2.6 s |
| 12 tokens, single request, total | 0.48 s | 0.47 s |
| TTFT with DeepInfer's default `PREFILL_ATTEMPT_INTERVAL=70` | 2.1-2.4 s | |

All of these runs are eager on the 3-layer test checkpoint. Its outputs are not meaningful text (truncated model); only the plumbing
is tested. Numbers with CUDA graphs on the full-depth test checkpoint are in [../docker/README.md](../docker/README.md).

## Prefix caching

`--prefix-cache` enables exact prefix caching (hit == miss, bit for bit, with the deterministic MoE). It needs DeepInfer's `_C`
built with the one-line `Sequence.block_table` binding, which the serving image includes. Design, exactness tests and TTFT numbers:
[PREFIX_CACHE.md](PREFIX_CACHE.md).
