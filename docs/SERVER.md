# OpenAI-compatible server

A minimal, dependency-free HTTP server (stdlib `ThreadingHTTPServer`) in front of one DeepInfer engine with continuous
batching. DeepInfer's source is not modified: the engine is built exactly like `tools/run_deepinfer.py` (SM120 shims +
runtime patches) and driven through its public `LLMEngine` API. The serving image's entrypoint (`docker/entrypoint.sh`)
starts this server.

| file | content |
|---|---|
| `server/server.py` | HTTP layer, OpenAI request/response mapping, chat template, incremental detokenisation, stop strings, SSE |
| `server/engine.py` | `EngineLoop`: owns the DeepInfer engine on one thread, continuous batching, add/abort between steps |
| `server/tools.py` | tool calling and `response_format`: request → encoding.py messages, DSML tool-call parser (complete and streaming), output grammars |
| `server/images.py` | image input: `image_url` parts (data: / http(s)) → validated image bytes and sizes for DeepInfer's multimodal path |
| `server/logprob_tap.py` | raw-logit logprobs (OpenAI `logprobs` / `top_logprobs`); DeepInfer's own logprobs are post-temperature |
| `server/dp_proxy.py` | `--dp N`: one engine process per GPU (DP attention + EP MoE) behind a least-loaded proxy |
| `server/bench_client.py` | TTFT / TPOT client: multi-turn streaming chat (exercises the prefix cache) + concurrent burst |
| `server/with_server.sh` | run a command against a temporary server; stops the whole process group and waits until the GPU memory is free |
| `tests/test_server.py` | plumbing tests against a running server (stdlib client) |
| `tests/run_server_tests.sh` | start the server, wait for `/health`, run the tests, stop |
| `tests/test_tools.py` | CPU unit tests of `server/tools.py` (rendering, parsing, streaming parser, grammars) |
| `tests/test_tools_server.py`, `tests/test_openai_sdk.py` | tool calling / `response_format` against a running server, with a stdlib client and with the official `openai` SDK |
| `tests/test_images.py`, `tests/test_images_server.py` | image input: CPU unit tests of `server/images.py`; server tests (formats, token counts, image changes the output, limits, 400s) |
| `tests/run_tools_tests.sh` | unit tests, then start the server and run the tool, image and SDK tests (optionally `test_server.py` too) |
| `tools/diag_repeat.py` | determinism and per-step latency probe of the bare engine (no HTTP) |

## Run

Inside the image (`PYTHONPATH=/opt/sm120/sm120:/opt/deepinfer` is preset):

```bash
python3 /opt/sm120/server/server.py --ckpt /work/ckpt3 --port 8000 --max-num-seqs 8     # b12x backends (fast)
python3 /opt/sm120/server/server.py --ckpt /work/ckpt3 --port 8000 --deterministic      # reproducible outputs
# options: --max-model-len 16384 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.9 --max-blocks 256
#          --nextn N (MTP, needs mtp.* layers) --graphs (CUDA graphs) --zos 0|1 --prefill-interval 1 --model-name ID
#          --prefix-cache [--prefix-cache-grain 512 --prefix-cache-checkpoint-stride 0 (= --max-num-batched-tokens)
#                          --prefix-cache-entries 64 --prefix-cache-max-block-frac 0.5]
#          (tool-calling workloads with short shared system prompts: --prefix-cache-checkpoint-stride 1024 or 2048, see PREFIX_CACHE.md)
#          --yoco-compact [KEEP] (opt-in, about 1.5x prefill on long prompts, not with --prefix-cache; MTP_YOCO.md §8a)

curl -s localhost:8000/v1/models
curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"ckpt3","prompt":"The lighthouse keeper","max_tokens":32,"temperature":0}'
curl -N localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"ckpt3","messages":[{"role":"user","content":"Hi"}],"max_tokens":32,"stream":true}'

# tests (server must be running), or all in one go:
python3 /opt/sm120/tests/test_server.py --url http://127.0.0.1:8000
# greedy reference for --di-ref: make it with the same image and code as the server (torch MoE, like --deterministic);
# a dump from an older version can differ and fails the "greedy == run_deepinfer.py" check
SM120_MOE_IMPL=torch python3 /opt/sm120/tools/run_deepinfer.py --ckpt /work/ckpt3 --prompt-ids /work/prompt_ids.json \
  --out /work/srv_ref_torchmoe.pt
/opt/sm120/tests/run_server_tests.sh /work/ckpt3 8000 /work/srv.log --deterministic -- \
  --di-ref /work/srv_ref_torchmoe.pt --prompt-ids /work/prompt_ids.json
```

Through the image entrypoint, `--model PATH` maps to `--ckpt` and CUDA graphs are on by default (`--no-graphs` turns them off);
see [../docker/README.md](../docker/README.md).

At start-up the server prints the somora banner with the version, checkpoint, `--dp` and port (once, not per
`--dp` rank). It is coloured only on a terminal, plain in `docker logs`, ASCII-only without a UTF-8 locale;
`SM120_BANNER=0` (or `NO_COLOR` for colour only) turns it off.

The OpenAI Python client works with `base_url="http://host:8000/v1"` and any `api_key` (it is not checked).

## API

| endpoint | supported |
|---|---|
| `GET /v1/models` | the one model (`--model-name`, default the checkpoint directory name) and its `max_model_len` |
| `GET /health` | `{"status": "ok" \| "dead", "stats": {...}, "config": {"nextn": N}}`: engine steps, flushes, added, aborted, finished, max_running, busy_s, tokens (generated), emit_steps (per request, the steps in which it got tokens: (tokens - added) / (emit_steps - added) = tokens per decode step) (+ `prefix_cache` stats when enabled). Behind the `--dp` proxy: summed over ranks, except the lockstep counters steps / flushes (per rank), plus each rank's own report under `ranks` |
| `POST /v1/completions` | `prompt`: a string, a one-element list of strings, or a list of token ids. `max_tokens`, `temperature`, `top_p`, `top_k`, `stop` (string or up to 4 strings), `seed`, `ignore_eos`, `stream`, `stream_options.include_usage`. Extension: `return_token_ids` (non-streaming) |
| `POST /v1/chat/completions` | `messages` rendered with the checkpoint's `encoding/encoding.py` (`encode_messages`, `thinking_mode="chat"`). `"thinking": true` or `reasoning_effort` select `thinking` mode, also inside `chat_template_kwargs` (`thinking`, `enable_thinking`, `reasoning_effort`, `drop_thinking`; top-level fields win); the text before `</think>` is returned as `message.reasoning_content` (streaming: `delta.reasoning_content`). Same sampling fields, plus `max_completion_tokens`. `tools`, `tool_choice`, `parallel_tool_calls`, `response_format`: see [Tool calling and structured output](#tool-calling-and-structured-output); `image_url` content parts: see [Image input](#image-input) |

**Logprobs** (non-streaming, nextn 0): completions `logprobs: K` → `{tokens, token_logprobs, top_logprobs, text_offset}`;
chat `logprobs: true, top_logprobs: K` → `{content: [{token, logprob, bytes, top_logprobs}]}`. K <= 20. These are log_softmax of the
**raw** logits, like vLLM's default, not DeepInfer's post-temperature values. `return_tokens_as_token_ids: true` gives
`"token_id:<n>"` strings.

**Multi-GPU:** `--dp N [--ep M]` spawns N rank processes (engine_rank r, data_parallel_size N, expert_parallel_size M=N) on ports
PORT+1..PORT+N. `dp_proxy.py` serves PORT and routes each POST to the least-loaded rank; `/health` aggregates the ranks. Idle
ranks poll every 5 ms, because `has_unfinished_requests` is a DP-group all-reduce. Tested only with one rank (`--force-proxy`);
EP with N=4 has not run on real hardware yet ([MULTI_GPU_RUNBOOK.md](MULTI_GPU_RUNBOOK.md)).

**Not supported:** `n > 1`, `best_of`, streaming logprobs, `echo`, the legacy `functions` / `function_call` fields, audio, and `presence`/`frequency_penalty`/`logit_bias` (DeepInfer's `SamplingParams` has none of them).
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

## Tool calling and structured output

`/v1/chat/completions` accepts OpenAI's `tools`, `tool_choice`, `parallel_tool_calls` and `response_format`. Everything model
specific comes from the checkpoint's `encoding/encoding.py`: it renders the tool schemas into the system prompt, renders
earlier tool calls and tool results, and its `parse_tool_calls` parses the model's tool calls. `server/tools.py` maps between
that and the OpenAI format.

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "m", "messages": [{"role": "user", "content": "Weather in Vienna?"}],
  "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object",
             "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
  "tool_choice": "required"}'
# -> "message": {"role": "assistant", "content": null, "tool_calls": [{"id": "call_...", "type": "function",
#      "function": {"name": "get_weather", "arguments": "{\"city\": \"Vienna\"}"}}]}, "finish_reason": "tool_calls"
```

**Request.** `tools` go into the first system message (one with empty text is inserted if the conversation has none), where
encoding.py renders them as its `## Tools` section. `response_format` with a schema is rendered the same way (`## Response
Format`). History messages use the OpenAI shapes: assistant messages with `tool_calls` (`arguments` as a JSON string), `role:
"tool"` messages with `tool_call_id` (encoding.py turns them into `<tool_result>` blocks, sorted by call order), `developer` as
`system`, and text content-part lists.

**Response.** `message.tool_calls` holds `[{id: "call_<hex>", type: "function", function: {name, arguments}}]`; `arguments` is a
JSON string formatted like encoding.py's `decode_dsml_to_arguments` (`{"k": v, ...}`). Text that the model wrote before its
tool calls is `message.content` (`null` if there is none). `finish_reason` is `"tool_calls"` when calls are returned. Streaming
sends OpenAI deltas: for each call one `tool_calls[i]` delta with `index`, `id`, `type`, `function.name` and empty `arguments`,
then one `function.arguments` piece per completed parameter, and the closing `}`; the concatenated pieces are the non-streaming
string. In thinking mode the reasoning streams as `delta.reasoning_content` (also without tools; v0.1 streamed it as
`content`).

**Decoding.** The model writes tool calls as one DSML block after its text (`\n\n<｜DSML｜ calls>` … `</｜DSML｜ calls>`, then
EOS). How a request is decoded:

| request | decoding | result |
|---|---|---|
| `tools`, `tool_choice` `auto` (the default with tools) | free | a valid block (known functions, JSON-valid arguments) becomes `tool_calls`; anything else is returned as plain `content` |
| `tool_choice` `none` | free, generation stops where a block would start | `content` only |
| `tool_choice` `required` | grammar: a block with 1 to 8 calls of the declared functions (exactly 1 with `parallel_tool_calls: false`) | `tool_calls` |
| `tool_choice` `{"type": "function", "function": {"name": N}}` | grammar: calls of N only | `tool_calls` |
| `response_format` `json_object` | grammar: a JSON object | `content` is JSON |
| `response_format` `json_schema` | grammar: the schema (`strict: false` relaxes xgrammar's strict mode, e.g. extra properties) | `content` is valid against the schema |
| `auto` + `response_format` | grammar: either a tool-call block or the response JSON | either |

The grammars are xgrammar EBNF, passed to DeepInfer's own structured-output support (`SamplingParams.so_args`,
`schema_type: "ebnf"`; DeepInfer compiles them asynchronously and masks the logits in its sampler, including the MTP draft
verification). In a tool-call grammar the parameters come in schema order, required ones are mandatory, undeclared ones are not
allowed. String parameters are raw text (`string="true"`, as the model was trained) with `enum`, `minLength` and `maxLength`
honoured. All other parameter types are JSON validated against their sub-schema (`$defs` are carried along). JSON whitespace
in `response_format` output is limited to runs of 16 characters, so a degenerate model cannot loop on whitespace. In thinking
mode the grammar only starts after `</think>` (DeepInfer `trigger_token_ids`), so the reasoning stays free.

**Limitations.**

- If `max_tokens` runs out inside a grammar, the output is incomplete: `finish_reason` `"length"`, no `tool_calls`, the partial
  text as `content`. The same happens in thinking mode when the model does not close its reasoning within `max_tokens`.
- `auto` is unconstrained (best effort). In streaming, tool calls that were already sent cannot be taken back: if the block
  turns out malformed after the first call, the open call is closed with the parameters streamed so far and no further calls
  follow. Non-streaming returns the whole text as `content` in that case.
- Constrained requests (grammar) bypass the prefix cache (`prefix_cache.py` skips requests with `so_args`); unconstrained tool
  requests (`auto`, `none`) are cached as usual.
- String parameter values cannot contain `</` (the DSML parser ends a value at the first `</｜DSML｜`); in a length-bounded
  JSON string (`minLength` / `maxLength`) xgrammar 0.2.2 allows no escape sequences. xgrammar 0.2.2 also lets raw control
  characters into length-bounded JSON strings; `tools.py` patches that character class. It also builds wrong digit patterns
  for some integer ranges; ranges with up to 1024 values are therefore sent as an `enum`, larger ones as they are.
- Tool names may not contain `"` or newlines. The legacy `functions` / `function_call` fields are not supported.
- `logprobs` are those of the raw logits, before the grammar mask.

**Test results** (RTX 5090 test machine, 3-layer test checkpoints, whose text is garbage; the checks are structural). `tests/test_tools.py`
(CPU, 15 tests) passes everywhere. `tests/test_tools_server.py` checks that every constrained output that finished is valid:
known function, arguments valid against the JSON schema, `json_schema` content valid against the schema. It also covers
streaming deltas, auto / none, multi-turn tool history, auto + `response_format`, 9 concurrent mixed requests, thinking mode
and 400s. `tests/test_openai_sdk.py` uses openai 3.35.1: typed `tool_calls`, a tool-result round trip, streaming chunks, the
stream helper, `chat.completions.parse` with a pydantic `response_format` and with `pydantic_function_tool`, `json_object`,
and `BadRequestError`.

| server | test_tools_server | openai SDK | notes |
|---|---|---|---|
| `ckpt3`, eager, b12x MoE | 35/35 | 10/10 | `test_server.py` (v0.1 plumbing) 12/12 hard checks |
| `ckpt3`, eager, `--deterministic` | 36/36 | n/a | incl. greedy tool-call stream == non-stream; `test_server.py` 17/17 |
| `ckpt3`, eager, `--prefix-cache` (grain and checkpoint stride 128) | 36/36 | 10/10 | repeated tool prompts hit the cache, constrained requests bypass it |
| `ckpt3_mtp`, eager, `--nextn 3` | 34/34 | 10/10 | constrained decoding works with MTP draft verification |
| `ckpt3`, `--graphs` (v0.1.1) | 35/35 | 11/11 | images 18/18; `test_server.py` 14/14 |
| `ckpt3`, `--graphs --prefix-cache` (grain and checkpoint stride 128) | 36/36 | 11/11 | images 18/18 |
| `ckpt3_mtp`, `--graphs --nextn 3` | 34/34 | 11/11 | images 19/19 |
| `ckpt3_vision`, `--graphs` (real ViT weights) | 35/35 | 11/11 | images 18/18 (different pictures change the logits) |
| `ckpt3`, `--graphs`, 19 requests of 469 tokens, then 9 concurrent constrained requests | 9/9 valid | | server healthy; this crashed before v0.1.1 |

A random model can run out of `max_tokens` inside an unbounded JSON string (`json_object`) or never close its reasoning
(thinking mode); the tests report those cases as INFO, not as failures.

## Image input

Chat messages can contain OpenAI image parts: `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}` or an
`http(s)://` URL (`detail` is accepted and ignored). The checkpoint's encoding.py puts one `<｜deepseek_image｜>` placeholder
per image into the prompt. The server fetches and checks every image (PNG, JPEG, WebP, GIF, BMP; full decode with Pillow)
and passes the encoded bytes and the decoded size to DeepInfer (`add_request(multi_modal_data={"image_bytes", "image_sizes"})`).
DeepInfer expands each placeholder to the image's token span and runs its own preprocessing and vision encoder. An image costs
a size-dependent number of prompt tokens, at most `vision_config.max_image_tokens` (1024 for V4.1-Flash); `usage.prompt_tokens`
and the `max_tokens` budget include them.

| option | default | |
|---|---|---|
| `--max-images` | 8 | images per request (all messages together) |
| `--max-image-bytes` | 20 MiB | encoded size per image |
| `--max-image-pixels` | 50,000,000 | width × height per image, checked before decoding |
| `--no-image-urls` | off | refuse `http(s)` URLs, accept only `data:` URLs |
| `--image-fetch-private` | off | allow URLs whose host resolves to a loopback / private / link-local address (SSRF protection when off); redirects are never followed |
| `--image-fetch-timeout` | 10 s | per URL |

Images are accepted in user (and system) messages; image parts in assistant or tool messages, local file paths, `file:` URLs,
unsupported formats, corrupt files and anything over a limit get a 400 with the reason. Requests with images bypass the prefix
cache (`prefix_cache.py` does not cache multimodal prompts). The image preprocessing runs in DeepInfer's multimodal worker
processes, which use a pinned host memory pool (`DEEPINFER_USE_MULTIMODAL_POOL`, `DEEPINFER_MULTIMODAL_POOL_SIZE_GB`). One
in-flight request holds its resized images there until its prefill: about 5.1 MB per full-size image (1024 tokens), about
0.9 MB at the minimum size, plus 4 bytes per prompt token. A request that does not fit the whole pool would wait until
`DEEPINFER_MULTI_MODAL_WORKER_TIMEOUT`, so keep `--max-images` × 5.1 MB well below the pool size (the default 8 images need
about 41 MB; a 2 GB pool per GPU is plenty). Pillow's decompression-bomb guard is off in DeepInfer's worker, which is why the
server checks `--max-image-pixels` before anything is decoded.

**Test results** (RTX 5090 test machine). The truncated test checkpoints normally have a vision tower of zeros (shard
`model-00001` holds the vision weights; `make_truncated_ckpt.py` synthesises zeros without it, `truncation.json:
vision_synthesized`). `/work/ckpt3_vision` (layers 0/2/20, 16 experts, real ViT + aligner) was built for these checks.

- `tests/test_images.py` (CPU): 4/4. `tests/test_images_server.py`: 19/19 on `ckpt3_vision` (`--deterministic`,
  `--image-fetch-private`). This includes 4000x3000 input, PNG/JPEG/WebP/GIF/BMP, two images per message, streaming,
  images in two turns, image + `tool_choice: "required"`, 4 image and 2 text requests at once, an http URL, and the 400s.
  The first-token logits differ between two different pictures and repeat for the same picture. On `ckpt3`
  (zero ViT): 18/18 plus that logit check as INFO; with `--nextn 3` (`ckpt3_mtp`): 19/19, logit checks skipped (no logprobs with MTP).
- openai SDK image message (data: URL): passes.
- **ViT + aligner vs DeepSeek's reference `vision.py`** (`tools/ref/vit_compare.py`). Same patches, reference in fp32 =
  truth, reference in bf16 = noise floor, 5 synthetic pictures (448², 1344x896, 1600x600 text, 300x500 noise upscaled,
  3000x2000 capped at 977 tokens). DeepInfer's rel-L2 / noise floor for patch_embed, block 0, block 15, block 31, ViT output
  and aligner output: 0.65 to 1.15 everywhere. Aligner output cosine vs fp32: 0.9990 / 0.9807 / 0.9982 / 0.9996 / 0.9986;
  the bf16 reference reaches 0.9992 / 0.9794 / 0.9985 / 0.9997 / 0.9985 on the same inputs. DeepInfer's preprocessing
  matches an independent Pillow resize/normalise at cosine 1.000 (exact layout sizes) to 0.91 (a noise image, resize filter
  differences). The ViT runs in DeepInfer's eager prefill path (`--graphs` does not capture it).
- Before `fix/vit-sdpa-batch` the ViT attention shim fell back to torch's math SDPA kernel: about 10 GiB for a full-size
  image, CUDA OOM, dead engine. Now 17 MiB extra.
- Not tested yet: whether the full model answers questions about images (the quality set's `image` items), on the 4-GPU host.

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

## Profiling a prefill (`SM120_PROFILE_PREFILL`)

`-e SM120_PROFILE_PREFILL=N` makes every engine process profile the N-th request it admits (torch profiler, CPU +
CUDA) from admission to its first token, i.e. its whole prefill, once. It writes `prefill_r<rank>_p<pid>.trace.json`
(chrome://tracing / Perfetto) and `.txt` (per-category GPU kernel table: attention, GEMM, MoE, mHC, NCCL, EP pack,
...; wall vs GPU busy) to `SM120_PROFILE_DIR` (default `/work`), and logs the table. With `--dp`, a single request on
an idle server goes to rank 0. The profiled request is slower (about +2.5 s for the trace export); time the others.

```bash
docker run ... -e SM120_PROFILE_PREFILL=2 -e SM120_PROFILE_DIR=/work ...   # then send 2 fresh requests
docker logs <container> 2>&1 | grep -A20 "SM120_PROFILE_PREFILL: wrote"
```

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

`--deterministic --prefix-cache --prefix-cache-checkpoint-stride 512`: **22/22 passed** (adds prefix-cache hit checks; a repeated 1503-token prompt hits 1024 tokens,
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
