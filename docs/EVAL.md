# Evaluation tools: endpoint comparison, load generation, quality spot-check

All tools in `eval/` are stdlib-only clients of any OpenAI-compatible endpoint: this repo's server (`server/server.py`, image
`somora_deep_sm120:0.1.1`) or a vLLM baseline deployment. The multi-GPU bring-up procedure that uses them is in
[MULTI_GPU_RUNBOOK.md](MULTI_GPU_RUNBOOK.md).

| tool | what |
|---|---|
| `eval/compare_endpoints.py` | greedy A/B on identical token ids: exact match, first divergence, position-wise agreement, chosen-token logprob deltas, top-K overlap and deltas, near-tie classification of the divergence. Record/replay (`--save-a`, `--a file:PATH`) for endpoints that cannot run at the same time |
| `eval/loadgen.py` | closed-loop load at concurrency 1/4/16/64 × prompt 512/4k/32k, 256 output tokens: TTFT, TPOT, E2E p50/p99, output and total tok/s; random token-id prompts (no prefix-cache help) |
| `eval/quality_check.py` + `eval/quality_set.jsonl` | 14 prompts with known answers: math (5), Python code with asserts (3, `--exec-code` runs them in a separate `python3 -I` process with a timeout), German (4), a fact, and an 8k-token passkey needle. Items with `known_fail` (currently `math-reason`, which the model gets wrong without thinking mode) are reported as KFAIL/KPASS and not counted |

## Requirements on the endpoint

- `/v1/completions` with a token-id prompt, `logprobs: K` (K <= 20), `return_tokens_as_token_ids: true`, `ignore_eos: true`,
  and `stream` with `stream_options.include_usage`. vLLM supports all of these. `server.py` supports them (logprobs only
  non-streaming, and not with `--nextn > 0`).
- **Logprob semantics.** vLLM reports raw-logit logprobs. DeepInfer's own sampler only has *processed* logprobs (after temperature
  and top-p/k, which is degenerate for greedy), so `server.py` computes log_softmax of the raw logits in `server/logprob_tap.py`.

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
needle prompt all ran through. Meaningful quality numbers need the full checkpoint on multi-GPU hardware.
