# Evaluation tools: endpoint comparison, load generation, quality spot-check

All tools in `eval/` are stdlib-only clients of any OpenAI-compatible endpoint: this repo's server (`server/server.py`, image
`somora_deep_sm120:0.1.0`) or a vLLM baseline deployment. The multi-GPU bring-up procedure that uses them is in
[MULTI_GPU_RUNBOOK.md](MULTI_GPU_RUNBOOK.md).

| tool | what |
|---|---|
| `eval/compare_endpoints.py` | greedy A/B on identical token ids: exact match, first divergence, position-wise agreement, chosen-token logprob deltas, top-K overlap and deltas, near-tie classification of the divergence. Record/replay (`--save-a`, `--a file:PATH`) for endpoints that cannot run at the same time |
| `eval/loadgen.py` | closed-loop load at concurrency 1/4/16/64 × prompt 512/4k/32k, 256 output tokens: TTFT, TPOT, E2E p50/p99, output and total tok/s; random token-id prompts (no prefix-cache help) |
| `eval/quality_check.py` + `eval/quality_set.jsonl` | 28 prompts with known answers: math (5), Python code with asserts (3, `--exec-code` runs them in a separate `python3 -I` process with a timeout), German (4), a fact, an 8k-token passkey needle, tool calling (8), structured output (3) and images (3), see below. Items with `known_fail` (currently `math-reason`, which the model gets wrong without thinking mode) are reported as KFAIL/KPASS and not counted. Scores are reported per category |

## Requirements on the endpoint

- `/v1/completions` with a token-id prompt, `logprobs: K` (K <= 20), `return_tokens_as_token_ids: true`, `ignore_eos: true`,
  and `stream` with `stream_options.include_usage`. vLLM supports all of these. `server.py` supports them (logprobs only
  non-streaming, and not with `--nextn > 0`).
- **Logprob semantics.** vLLM reports raw-logit logprobs. DeepInfer's own sampler only has *processed* logprobs (after temperature
  and top-p/k, which is degenerate for greedy), so `server.py` computes log_softmax of the raw logits in `server/logprob_tap.py`.

## Tool calling and structured-output items

The items with a `"request"` field send extra request fields (`tools`, `tool_choice`, `response_format`) and are scored on
the structured result, greedy like the rest:

| id | request | passes when |
|---|---|---|
| `tool-pick-weather`, `tool-pick-stock` | 3 tools (weather, stock price, e-mail), `tool_choice` auto | exactly one call of the right tool with the right arguments (`city` Vienna/Wien + `unit` celsius; `ticker` NVDA) |
| `tool-args-email` | same tools | `send_email` with the address, subject and body extracted from the prompt |
| `tool-parallel` | same tools | two `get_weather` calls, one for Vienna and one for Berlin (any order) |
| `tool-not-needed` | same tools, an arithmetic question | no tool call, answer 42 |
| `tool-result-turn` | history: the weather call + its `role: "tool"` result | no new call; the answer uses the result (21 °C, sunny) |
| `tool-enum-required` | thermostat tool with an enum and an integer range, `tool_choice: "required"` | `set_thermostat(mode="heat", temperature=23)` (integer, not a string) |
| `tool-de-named` | German prompt, named `tool_choice` | `get_weather` with `city` Graz |
| `json-extract-nested` | `json_schema` (strict, nested address object) | valid JSON, valid against the schema, extracted fields correct (age 34, zip 1010, city Wien/Vienna, ...) |
| `json-enum-sentiment` | `json_schema` with enums | valid against the schema, `sentiment` = negative |
| `json-object` | `json_object` | valid JSON, `primes` = [2, 3, 5, 7, 11] |

Image items (`image` category; the pictures are small synthetic PNGs embedded as `data:` URLs):

| id | picture | passes when |
|---|---|---|
| `image-color` | a green square on white | answer "green" |
| `image-count` | three blue circles and a small outlined rectangle | answer 3 |
| `image-read-number` | the number 4862 in large black digits | answer 4862 |

The baseline needs a vLLM deployment with the model's vision support for these; otherwise they fail there for that reason only.

Expectation keys (`quality_check.py --help`): `tool_calls` (each expected call must match a distinct returned call: name and
argument specs, strings case-insensitive, `{"regex": ...}`, `{"len": n}`), `exact_count`, `no_tool_calls`, `json` with `schema`
(checked against the request's `json_schema`) and `fields` (dotted paths).

For the vLLM baseline, tool calls need the server started with `--enable-auto-tool-choice` and a tool-call parser for the model;
without them the tool items fail on the baseline for that reason only. `response_format` works on vLLM without extra flags.

## Verified results (RTX 5090, 3-layer test checkpoint `ckpt3`, two server instances one after the other)

These runs check the harness, not model quality: the 3-layer truncated model produces meaningless text.

**compare_endpoints.py** (8 built-in prompts, 48 tokens, top-5):

| A | B | result |
|---|---|---|
| server `--deterministic` (recorded with `--save-a`) | the same config, a new server instance (replayed with `--a file:`) | **8/8 exact, logprob deltas 0, top-K Jaccard 1.0** (identical stacks agree bit for bit) |
| server `--deterministic` (torch MoE) | server default (b12x MoE), through the DP proxy | divergence at token 0-30, agreement 0.10, 6/8 divergences at near-ties. Expected on the chaotic 3-layer model (ref-vs-ref top-1 is about 0.7, see [VALIDATION.md](VALIDATION.md)); shows that the harness reports and classifies divergences |

**loadgen.py** (b12x server through the DP proxy `--force-proxy`, `--max-num-seqs 64`, eager, output 256):

| prompt | conc | out tok/s | total tok/s | TTFT p50 / p99 | TPOT p50 / p99 |
|---|---|---|---|---|---|
| 512 | 1 | 17.9 | 53.7 | 0.11 / 0.12 s | 55.5 / 63.0 ms |
| 512 | 4 | 73.6 | 220.9 | 0.14 / 0.14 s | 55.1 / 55.1 ms |
| 512 | 16 | 254.6 | 763.8 | 0.24 / 0.25 s | 65.1 / 65.4 ms |
| 512 | 64 | 679.4 | 2038.3 | 0.54 / 16.5 s | 61.1 / 155.5 ms |
| 4096 | 1 | 17.4 | 296.3 | 0.14 / 0.15 s | 57.6 / 62.3 ms |
| 4096 | 4 | 67.8 | 1152.2 | 0.31 / 0.42 s | 61.1 / 62.2 ms |
| 4096 | 16 | 105.3 | 1790.0 | 15.5 / 33.4 s | 58.2 / 69.8 ms |
| 4096 | 64 | 110.4 | 1877.2 | 125 / 142 s | 58.4 / 69.6 ms |
| 32768 | - | skipped (> max_model_len 16384) | | | |

- The 4k × 16/64 TTFTs are KV-capacity queueing: `--max-blocks 256` (the default, 32768 tokens) holds only about 7 sequences of
  4.3k tokens. Raise `--max-blocks` on a real deployment.
- These are eager-mode numbers; CUDA graphs (`--graphs`, default in the image entrypoint) cut decode TPOT by about an order of
  magnitude ([RUNTIME.md](RUNTIME.md)).

**quality_check.py** scored 0/14 against `ckpt3`, as expected for a 3-layer model. Retrieval, code-block detection and the 2k-token
needle prompt all ran through. Meaningful quality numbers need the full checkpoint on multi-GPU hardware. The tool and JSON items
(added in 0.2.0) were run against `ckpt3` to check the scoring paths: 1/25 overall. Every tool and JSON item was sent and scored. The constrained ones (`tool-enum-required`, `tool-de-named`, the JSON items) came back structurally valid but with wrong values, which is what the 3-layer model should produce.
