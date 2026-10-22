# Results on the 4-GPU target

Measured results of the full DeepSeek-V4.1-Flash model on the multi-GPU target, next to a tuned vLLM deployment of the
same checkpoint on the same machine. Single-GPU results with truncated test checkpoints are in
[VALIDATION.md](VALIDATION.md) and the README. Discussion: [issue #2](../../../issues/2).

## v0.3.0

**In short:** stable on 4× RTX PRO 6000 over PCIe, quality on par with the tuned vLLM deployment (27/27 vs 26/27).
With many concurrent users and long prompts it now matches vLLM's throughput and reaches the first token sooner, and
it serves as many users within the latency targets (32). A single user gets about 55–65 % of vLLM's speed, and a
single long prompt is read about 5–6× slower,
because one GPU reads each prompt here while vLLM's tensor parallelism spreads it over four ([#1](../../../issues/1)).

### Setup

- **Hardware, model, baseline:** as for v0.2.0 below (4× RTX PRO 6000 Blackwell Max-Q, 96 GB, PCIe only; full
  DeepSeek-V4.1-Flash-1017; vLLM fork with b12x kernels, TP=4, adaptive DSpark up to 7 drafts, engram tables in RAM).
- **This engine:** `--dp 4` with CUDA graphs, `--nextn 7`, `--prefix-cache`, `--gpu-memory-utilization 0.95
  --max-model-len 40960 --max-num-seqs 16 --max-num-batched-tokens 4096 --max-blocks 4096`,
  `SM120_ENGRAM_MODE=pinned` (one copy of the 189 GiB engram tables in pinned host RAM for all ranks).
- The column 0.3.0 is the acceptance run of the release candidate (`v0.3.0-rc1`, 2026-10-22), started exactly as in
  [MULTI_GPU_RUNBOOK.md](MULTI_GPU_RUNBOOK.md) §5: 5 starts in 181–186 s, a 15-minute soak (1,400 requests, 0 errors,
  same memory before and after), the README quickstart from an empty kernel cache (up after 351 s).
- Single measurements vary by about 10 %: 4k × 16 gave 321 tok/s in one run of 32 requests and 361 tok/s over the
  15-minute soak at the same load (v0.2.4 with the 0.3.0 intervals: 388).

### Progress since v0.2.0

| | v0.2.0 | v0.2.1 | v0.2.4 | **0.3.0** | vLLM (TP=4) |
|---|---|---|---|---|---|
| TTFT, new 58-token prompt | – | – | 2.0 s | **0.45 s** | ~0.2 s |
| TTFT, new 4k prompt | 3.9–4.6 s | 3.7–4.0 s | 2.9 s | **1.35 s** | 0.24 s |
| TTFT, new 32k prompt | 32.6 s | 33.8–35.7 s | 24.0 s | **11.2 s** | ~2.0 s |
| 4k prompts × 16 users, output tok/s | 224 | 117 | 376 | **321 / 361 (soak)** | 417 |
| 4k prompts × 64 users, output tok/s (TTFT p50) | 248 (20.4 s) | 116 | 420 (7.9 s) | **460 (9.2 s)** | 425 (33.8 s) |
| 512-token prompts × 64 users, output tok/s | 944 | 623 | 1,150 | **1,256** | 1,324 |
| users within TTFT p99 ≤ 2 s, TPOT p99 ≤ 50 ms | – | – | 0 | **32** | 32 |
| quality, 28-item set | 27/27 | – | 27/27 | **27/27** | 26/27 |
| teacher-forced vs vLLM: top-1 agreement / mean KL | – | 0.949 / 0.059 | 0.949 / 0.063 | **0.952 / 0.055** | – |

`loadgen.py` uses random token-id prompts and 256 output tokens (speculative decoding accepts almost nothing on them).

### Real prompts (`eval/scenarios.py realistic`, output tok/s, `--nextn 7`)

| set | users | v0.2.4 | **0.3.0** | vLLM |
|---|---|---|---|---|
| code | 1 | 101 | **136** | 209 |
| code | 16 | 951 | **917** | 1,200 |
| chat (German / English) | 1 | 67 / 74 | **84 / 87** | 148 / 157 |
| chat (German / English) | 16 | 496 / 560 | **548 / 533** | 797 / 787 |

- Draft acceptance with 7 drafts: code 0.52, chat 0.22–0.25. `--nextn 7` was the best fixed setting; `--mtp-varlen`
  (DeepInfer's adaptive draft length) won only for German chat at 16 users and lost at 1 user.
- One decode step at batch 1 takes 20.3 ms (v0.2.3: 32.1 ms, before the decode-graph fix).

### Mixed load and the prefill intervals

With several DP ranks DeepInfer decides per step, for all ranks together, whether to read a prompt or to decode. The
two intervals set how many decode steps run between two prefill chunks. 16 / 16 was the best trade-off measured:

| intervals | TTFT 58 / 4k / 32k | short chats during long prompts: TPOT p50 / p99 | short requests done in 60 s | 4k × 16 tok/s |
|---|---|---|---|---|
| 1 / 100 (v0.2.4) | 2.0 / 2.9 / 24.0 s | 27 / 57 ms | 105 | 376 |
| 1 / 1 | 0.24 / 1.1 / 8.8 s | 226 / 401 ms | 13 | – |
| 8 / 8 | 0.70 / 1.24 / 9.8 s | 62 / 118 ms | 50 | 325 |
| **16 / 16** | 0.85 / 1.39 / 11.0 s | 36 / 65 ms | 68 | 388 |
| 16 / 16, 0.3.0 acceptance | 0.45 / 1.35 / 11.2 s | 36 / 63 ms | 66 | 321 / 361 |
| vLLM | ~0.2 / 0.24 / ~2.0 s | – / 25 ms | 191 | 417 |

DeepInfer cannot mix decode tokens into a prefill step, so while a long prompt is read the other users' streams pause
for each chunk. vLLM's mixed batches avoid that.

### What changed since v0.2.1

- **v0.2.1** fixed the prefix-cache chunking and the EP prefill padding (single-GPU gains), but on the 4-GPU host the
  load numbers dropped (4k × 64: 248 → 116 tok/s). The cause was the engram tables: they did not stay in the page
  cache, and every miss became an NVMe read through the host's IOMMU (perf: 18 % of the CPU time in
  `amd_iommu_iotlb_sync`).
- **Engram tables in pinned RAM (0.2.3):** `SM120_ENGRAM_MODE=pinned` keeps one copy for all ranks in shared, pinned host
  memory, and the GPUs gather rows from it directly, also inside CUDA graphs. perf no longer shows the IOMMU; the load
  numbers recovered (4k × 64: 116 → 420 tok/s). It needs about 225 GiB of free RAM before start and about 0.4 GiB of
  GPU memory per rank, so `--gpu-memory-utilization 0.95` instead of 0.92.
- **Decode graphs without padding (0.2.3/0.2.4):** the captured decode graphs exchanged the full batch capacity between
  the GPUs in every step, also for one user. Now each graph exchanges only its own rows: 32.1 → 20.3 ms per step at
  batch 1, +15–60 % output tok/s.
- **Prefill vote (0.3.0):** a request that only one rank held waited up to 100 decode steps before its prefill
  (DeepInfer's `DEEPINFER_PREFILL_ANY_ATTEMPT_INTERVAL`), before every 4096-token chunk. Fixed by the prefill
  intervals above.
- Numerics are unchanged: teacher-forced against vLLM's continuations, top-1 agreement 0.949, mean KL 0.063.

### Where a prefill spends its time (v0.2.4, 4k prompt, all 4 ranks busy)

GPU busy 94 %: NCCL collectives 27 % (part of it waiting for the rank with the heaviest experts), MoE 20 %, sparse MLA
attention 19 %, GEMMs 12 %. Larger MoE buckets (`SM120_MOE_BUCKETS=...,8192`) cut the MoE time by 23 % but did not change
the TTFT; NCCL already picks the Simple protocol at these message sizes.

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

