# Results on the 4-GPU target

Measured results of the full DeepSeek-V4.1-Flash model on the multi-GPU target, next to a tuned vLLM deployment of the
same checkpoint on the same machine. Single-GPU results with truncated test checkpoints are in
[VALIDATION.md](VALIDATION.md) and the README. Discussion: [issue #2](https://github.com/thenaxon/somora_deep_sm120/issues/2).

## v0.2.0

**In short:** DeepInfer runs the full DeepSeek-V4.1-Flash on 4 SM120 GPUs over PCIe. It is stable and correct, and its quality matches or exceeds the baseline. It is not yet competitive on speed: decode is about 2–3× and prefill about 15× behind. The main prefill causes are identified, and fixes are on the way for v0.2.1.

### Setup

- **Hardware:** 4× RTX PRO 6000 Blackwell Max-Q (96 GB, SM120), PCIe only (all GPU pairs `NODE`, no NVLink), driver 595.91, 246 GB host RAM.
- **Model:** DeepSeek-V4.1-Flash (all 48 shards, unmodified), mounted read-only.
- **This engine:** `somora_deep_sm120:0.2.0` with `--dp 4` (DP attention + EP MoE over NCCL, `SM120_EP_IMPL=allgather`), `SM120_ENGRAM_MODE=host`, `SM120_ENGRAM_PREFAULT=1`, `--gpu-memory-utilization 0.92 --max-model-len 40960 --max-num-seqs 16 --max-num-batched-tokens 4096 --max-blocks 4096 --prefix-cache`.
- **Baseline:** a vLLM fork with b12x kernels, TP=4, DSpark speculative decoding (adaptive, up to 7 drafts), FP8 KV cache, engram tables pinned in host RAM.
- **Tools:** the repo's `eval/` tools (`compare_endpoints.py`, `loadgen.py`, `quality_check.py`) plus a few real code/chat prompts. The two engines could not share the GPUs, so they ran one after the other.

### Summary

| | this engine, v0.2.0 (best config) | vLLM baseline (TP=4) |
|---|---|---|
| Quality, 28-item set | **27/27** (+ the known-fail item passes) | 26/27 |
| Stability | 5 starts incl. CUDA graphs and DSpark: no crash, clean shutdown, no `/dev/shm` leftovers | – |
| Decode, 1 user, real prompts | code 111 / chat 73 tok/s | code 339 / chat 165 tok/s |
| TTFT, new 4k prompt | 3.9–4.6 s | 0.24 s |
| TTFT, new 32k prompt | 32.6 s | 2.0 s |
| Prefill rate | ~1,000 tok/s | ~16,000 tok/s |
| 4k prompts, 64 users, output tok/s | 248 | 425 |

"Best config" means CUDA graphs, `--nextn 3` and `--prefix-cache-grain 4096` (see the prefill section).

### Quality (`quality_check.py --exec-code`, graphs, nextn 0)

| category | this engine | vLLM baseline |
|---|---|---|
| core (math, code with asserts, German, fact, 8k needle) | 13/13 (+ `math-reason` KPASS) | 13/13 (+1 known-fail) |
| tool calling | 8/8 | 8/8 |
| structured output (JSON schema) | 3/3 | 2/3 (unterminated string in an enum item) |
| image input | 3/3 | 3/3 |

Hand checks passed:
- an image was described correctly;
- `json_schema` output validated;
- tool calls came back as `tool_calls`;
- top-level `"thinking": true` worked.

### Load test (`loadgen.py`, random token-id prompts, 256 output tokens)

Output tok/s, TTFT p50 in parentheses.

| prompt | users | vLLM baseline | graphs | graphs + nextn 3 | graphs + nextn 3 + grain 4096 |
|---|---|---|---|---|---|
| 512 | 1 | 193.0 (0.18 s) | 38.4 (1.10 s) | 54.4 (2.35 s) | 56.2 (1.41 s) |
| 512 | 16 | 1009.3 (0.27 s) | 376.2 (1.90 s) | 444.4 (1.97 s) | 493.7 (1.88 s) |
| 512 | 64 | 1324.3 (6.0 s) | 922.3 (4.66 s) | 928.9 (3.59 s) | 944.0 (3.39 s) |
| 4096 | 1 | 179.8 (0.24 s) | 12.1 (15.5 s) | 11.0 (20.6 s) | 48.3 (2.31 s) |
| 4096 | 16 | 416.6 (5.77 s) | 187.6 (11.2 s) | 108.0 (9.0 s) | 223.5 (7.57 s) |
| 4096 | 64 | 424.9 (33.8 s) | 233.4 (32.7 s) | 141.0 (24.5 s) | 247.5 (20.4 s) |

Random token ids make speculative decoding accept almost nothing on either engine. The real-prompt numbers in the summary are the fairer single-user comparison.

### Prefill: what we found

1. **The prefix cache split every prefill into 512-token engine steps.** With `--prefix-cache`, checkpoints at every grain (default 512) ended every chunk there. A 4k prompt took 8 steps instead of 1, a 32k prompt 64 instead of 8. Each step had a fixed cost of about 2 s on this machine, which matches the measured 15.5 s and 134 s almost exactly. `--prefix-cache-grain 4096` cut TTFT by 3.5–4× (4k: 3.9–4.6 s, 32k: 32.6 s). The prefix cache itself works: a repeated ~21k-token prompt took 114 s cold and 3.0 / 2.7 s when it hit.
2. **GPU communication is not the bottleneck.**
   - NCCL 2.29.7 used SHM for all channels by default, even though peer access works.
   - With `NCCL_P2P_LEVEL=SYS` all 128 channels used P2P/CUMEM, but TTFT stayed exactly the same: 4k 4.2–4.5 s, 32k 32.6 s.
3. **Remaining cost:** about 4 s per 4096-token step, against about 0.9 s on one RTX 5090 running the same 40 layers with 16 experts. Candidates:
   - In EP prefill, a host sync on the token counts ran in every MoE layer.
   - Idle ranks were padded to the busy rank's token count: with one 4k request, every rank ran its 96 local experts over 16k rows, three quarters of them empty. At V4.1-Flash size that is 22.5 ms instead of 6.8 ms per layer.
   - Only about 80 of the 203 GB of engram tables stayed in the page cache. Each missing row costs a random NVMe read. On one GPU a cold cache adds 0.4–0.7 s per 4k prompt.

#### Collectives and EP MoE layer (`tests/bench_ep_comm.py --moe`, CUDA graphs, µs)

| rows | NCCL all-gather | NCCL reduce-scatter | b12x PCIe all-gather | b12x PCIe reduce-scatter |
|---|---|---|---|---|
| 1 | 25.0 | 21.1 | 13.7 | 7.3 |
| 16 | 49.3 | 47.4 | 160.3 | 112.2 |
| 512 | 502.2 | 502.0 | 2814.4 | 2908.0 |
| 2048 | 1964.6 | 1964.7 | 10463.7 | 10148.0 |

| rows per rank | EP layer | EP fixed-capacity | full layer on 1 GPU |
|---|---|---|---|
| 1 | 117 | 4441 | 60 |
| 64 | 1256 | 8070 | 2495 |
| 2048 | 9346 | 9356 | 4862 |

All results match the single-GPU reference (cos ≥ 0.9999). The fixed-capacity variant costs 4–10 ms per layer regardless of load, so we are checking whether any serving path uses it.

### Numerics vs. the baseline

`compare_endpoints.py` (eager, 8 prompts × 256 greedy tokens) gave:
- 0/8 identical outputs, with the median first divergence at token 19;
- 2 prompts that diverge at token 1 without a near-tie;
- a mean chosen-token logprob delta of about 0.11.

The two engines differ in KV precision: this engine uses DeepInfer's FP4 main KV and FP4 indexer, the baseline FP8 KV. On truncated checkpoints, every layer of this engine matches DeepSeek's reference implementation within the reference's own noise floor (see `docs/VALIDATION.md`), and quality is on par, so we do not expect a numerics bug. For the full model we are adding a teacher-forced comparison, where both engines score the same continuation, so divergences do not compound.

### Other findings

- `chat_template_kwargs: {"thinking": true}` (as sent by common OpenAI clients and proxies) was ignored; only top-level `"thinking"` worked.
- `SM120_ENGRAM_STATS=1` printed no stats line in server mode.
- DSpark `--nextn 3` helped with short prompts at 1 user (+40 %). With 4k prompts under load and the 512-token chunking, it was slower than no DSpark; with grain 4096 it was neutral to slightly positive.

### What v0.2.1 changes

Verified for correctness on one GPU; the multi-GPU speed-up still has to be measured on the target:
- Prefix-cache checkpoints only where chunked prefill splits anyway (new `--prefix-cache-checkpoint-stride`, default = `--max-num-batched-tokens`). On one GPU `--prefix-cache` is now as fast as running without it.
- EP prefill: one count sync per step instead of one per layer, and the local experts run on real rows only.
- fp8 einsum retuned for long prompts (17–23 % faster, bit-identical).
- Engram: a start-up log line reporting how much of the tables is resident, and a prefault after all ranks have loaded.
- NCCL P2P self-test in the DP proxy; opt-in b12x PCIe transport for prefill (`SM120_EP_TRANSPORT=b12x`).
- `chat_template_kwargs.thinking` support and server-side engram stats.

Expectation: a clear prefill improvement, possibly 2–3× from here. A single long prompt will stay well behind TP=4, because in this design one GPU reads each prompt while TP spreads it over four. Tensor parallelism for the dense layers is tracked in #1.

