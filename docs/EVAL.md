# Evaluation tools: endpoint comparison, load generation, quality spot-check

All tools in `eval/` are stdlib-only clients of any OpenAI-compatible endpoint: this repo's server (`server/server.py`, image
`somora_deep_sm120:0.2.0`) or a vLLM baseline deployment. The multi-GPU bring-up procedure that uses them is in
[MULTI_GPU_RUNBOOK.md](MULTI_GPU_RUNBOOK.md).

| tool | what |
|---|---|
| `eval/compare_endpoints.py` | greedy A/B on identical token ids: exact match, first divergence, position-wise agreement, chosen-token logprob deltas, top-K overlap and deltas, near-tie classification of the divergence. Record/replay (`--save-a`, `--a file:PATH`) for endpoints that cannot run at the same time |
| `compare_endpoints.py --teacher-forced` | numerical fidelity: both endpoints (and optionally DeepSeek's reference implementation) score the same continuation; top-1 agreement and KL per position; see [Numerical fidelity](#numerical-fidelity) |
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

## Numerical fidelity

`compare_endpoints.py` without options compares two endpoints' greedy outputs; they diverge at the first near-tie, and
everything after that is incomparable. `--teacher-forced` scores a fixed continuation instead: for every position j of
the continuation each endpoint gets prompt + continuation[:j] and returns its top-K logprobs for the next token
(`max_tokens 1`). Requests run `--parallel` at a time and benefit from prefix caching.

- The continuation is the greedy output of `--cont-from a|b|ref` (default: the reference if given, else A), or a prompt
  item's own `"continuation"` text (e.g. human-written; then the per-endpoint NLL is a small perplexity test).
- `tools/ref/ref_to_fidelity.py` converts dumps of DeepSeek's reference implementation (`tools/ref/ref_runner.py`) into
  such a file. Positions 0-4 have exact reference distributions; later ones are normalised over the dump's top-50.
- Per pair (A-B, ref-A, ref-B): **top-1 agreement** (same argmax), **KL(P‖Q)** on the union of both top-K sets (a
  missing token gets that side's K-th logprob, both renormalised over the union; an estimate, comparable between runs
  with the same K), and the logprob delta of the forced token.
- `--reference` is a file (reference dump, or any recording) or a live endpoint URL; with it the summary states which of
  A and B is closer to it (lower KL mean, higher top-1 agreement).
- `--save-a/--save-b` record the scored positions; `--a/--b file:PATH` replay them, for endpoints that cannot run at the
  same time.

```bash
python3 eval/compare_endpoints.py --tokenizer-dir /models/V4.1 --teacher-forced --max-tokens 256 --top-logprobs 20 \
    --prompts prompts.jsonl --a http://host:8000 --save-a di_tf.json            # DeepInfer, then stop it
python3 eval/compare_endpoints.py --tokenizer-dir /models/V4.1 --teacher-forced --max-tokens 256 --top-logprobs 20 \
    --prompts prompts.jsonl --a file:di_tf.json --b http://host:8001 --json tf_di_vllm.json   # vLLM baseline
```

Both endpoints must return raw-logit logprobs (vLLM's default; this server's `logprobs`, not available with
`--nextn > 0`). Run the fidelity check with speculative decoding off on both sides.

### KV-cache precision of the implementations

| implementation | sliding-window KV | compressed KV (C4 / C128) | indexer keys |
|---|---|---|---|
| DeepSeek reference (`inference/model.py`) | FP8 e4m3, per-32 scales | FP4 (e2m1, per-16 scales) | FP4 |
| DeepInfer / this repo | FP8 e4m3 (528 B records) | FP4 (288 B records) | MXFP4 (68 B) |
| vLLM with `--kv-cache-dtype fp8` | FP8 | FP8 by that flag (check the vLLM version's V4 KV layout) | per vLLM's V4 kernels |

DeepInfer stores the KV in the same formats as the reference implementation (that is DeepSeek's design, see
[ATTENTION.md](ATTENTION.md)). A vLLM deployment with an FP8 KV cache presumably keeps the compressed tokens in FP8,
not in the model's FP4 (verify for the deployed version). A greedy A/B difference between the two is therefore expected even if both are implemented
correctly, and "higher precision" is not the same as "closer to the reference".

### Who is right? (full model)

A divergence at token 1 without a near-tie and a chosen-token logprob delta of ~0.1 nats in the greedy mode can come
from one early difference that every later token inherits. To find out:

1. **Teacher-forced A-B in both directions** (`--cont-from a`, then `--cont-from b`, 8+ prompts × 256 tokens). The KL
   per position and the top-1 agreement on a shared continuation show the real per-step distance, without compounding.
2. **A third, independent reference.** The DeepSeek reference implementation does not fit the full model on four 96 GB
   GPUs (476 GB), so: (a) the same truncated checkpoint served by both engines and run by `tools/ref/ref_runner.py`
   (`ref_to_fidelity.py` → `--reference`), which measures each engine's numerical error per layer stack; and/or (b) a
   third endpoint with a different numeric path (e.g. vLLM with the model's native KV format, or this server with
   `--deterministic`) as `--reference URL`.
3. **Human-written continuations** (prompt items with `"continuation"`): the per-endpoint NLL on real text is the
   practical "who models the text better" number. On the full model a clearly lower NLL settles the question;
   differences below ~0.01 nats/token are numerics noise.
4. The quality set (`quality_check.py`) stays the end-to-end check: equal scores mean the differences do not matter
   for answers.

### Verified on the RTX 5090 (plumbing, 3-layer `ckpt3`)

A = this server with `--deterministic` (torch MoE, eager), B = default (b12x MoE, CUDA graphs), reference = DeepSeek's
reference implementation on the same checkpoint (`ref_runner.py` dump, 17 continuation positions), top-20, 64 tokens:

| continuation | pair | positions | top-1 agreement | KL mean | KL p99 (worst prompt) |
|---|---|---|---|---|---|
| reference greedy | ref-A / ref-B / A-B | 17 | 0.941 / 1.000 / 0.941 | 0.031 / 0.025 / 0.026 | 0.077 / 0.069 / 0.048 |
| A greedy, 8 prompts | A-B | 512 | 0.910 | 0.048 | 0.397 |
| B greedy, 8 prompts | A-B | 512 | 0.897 | 0.049 | 0.295 |
| 4 human-written texts | A-B | 98 | 0.587 | 0.143 | 0.422 |

On this chaotic 3-layer model the two MoE paths are about as far from each other as each is from the reference, and
the verdict tool reports B marginally closer. The per-endpoint NLL of human text is 5.25 (A) vs 5.24 (B); the 3-layer
model does not model text at all, so only the full model makes those numbers meaningful.

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
