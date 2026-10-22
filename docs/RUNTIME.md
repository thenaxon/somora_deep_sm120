# Runtime: CUDA graphs, MTP/DSpark, logit capture, end-to-end benchmark

This document covers the engine-level parts of the port: running DeepInfer's CUDA-graph decode and its DSpark (MTP)
speculative decoding on SM120, the instrumentation used for validation, and the offline end-to-end benchmark.
Relevant files: `sm120/sm120_patches.py` (runtime patches P6, P8, P9, P10, P11, see [patches/PATCHES.md](../patches/PATCHES.md)),
`sm120/sm120_moe.py`, `sm120/sm120_capture.py`, `sm120/deep_gemm/_b12x.py`, `tools/run_deepinfer.py`,
`tools/bench_e2e.py`. The GEMM/MoE graph contract is in [MOE_GEMM.md](MOE_GEMM.md) section 6, the attention side in
[ATTENTION.md](ATTENTION.md) section 5.

All numbers below are from one RTX 5090 with truncated test checkpoints ([VALIDATION.md](VALIDATION.md)).

## 1. Status

| check | result |
|---|---|
| `tests/test_gemm_moe_graphs.py` | 14 passed, 2 xfailed (tiny-decode on unpatched b12x, by design) |
| P8 `test_remap_draft_topk` (DeepInfer's TileLang sampler tests) | 6/6 pass (fail on SM120 without P8) |
| DeepInfer sampler / DFlash TileLang op tests | 74 + 124 passed |
| b12x tiny-decode SwiGLU patch ([flashinfer#6866](https://github.com/flashinfer-ai/flashinfer/issues/6866)) | all checks pass incl. the V4.1 shape; tiny-decode is 0.90-0.95x the dynamic path, so the default stays dynamic (`SM120_MOE_TINY_DECODE=0`) |
| logit capture (`tools/run_deepinfer.py`, 3-layer model) | decode argmax == sampled token for all 16 steps |
| determinism | default b12x MoE: repeats differ (atomic top-k combine). `SM120_MOE_DETERMINISTIC=1` / `--batch-invariant`: bit-identical repeats, prefill + decode logits and tokens; on the full-depth test model eager, graphs and a second graphs run are bit-identical (17/17 tokens, 0.0 error on every layer) |
| CUDA graphs end to end | 3-layer, YOCO and full-depth test models; no b12x fallback warnings under capture |
| MTP / DSpark | nextn 1 and 5, eager and graphs; speculative == greedy holds exactly with the deterministic MoE ([VALIDATION.md](VALIDATION.md)) |

### Benchmark, 3-layer test model (`tools/bench_e2e.py`, prompt 256, 64 generated tokens)

| mode | B | prefill tok/s | decode tok/s | TPOT ms |
|---|---|---|---|---|
| eager | 1 | 1957 | 23.9 | 41.8 |
| eager | 4 | 10814 | 94.9 | 42.1 |
| eager | 16 | 32945 | 390.9 | 40.9 |
| eager | 64 | 45020 | 1520.1 | 42.1 |
| graph | 1 | 5691 | 309.1 | 3.2 |
| graph | 4 | 14712 | 943.7 | 4.2 |
| graph | 16 | 33118 | 3305.3 | 4.8 |
| graph | 64 | 52511 | 11466.3 | 5.6 |

Eager decode is launch-bound (TPOT about 42 ms at every B); CUDA graphs give 13x at B=1 and 7.5x at B=64. Prefill at
small B includes host/scheduler overhead. These numbers predate the mHC/einsum Triton kernels; deeper models are in
[ATTENTION.md](ATTENTION.md) section 9 and [VALIDATION.md](VALIDATION.md).

### MTP / DSpark on the full-depth test model

`tools/bench_e2e.py` on the full-depth test checkpoint (40 layers, real DSpark heads and targets, but only 16/384
routed experts per layer and 32/128 per MTP layer), `SM120_ENGRAM_MODE=host`, CUDA graphs, random prompts of 256
tokens, 256 generated tokens, `--max-num-batched-tokens 512`:

| nextn | force_topk | B=1 decode tok/s / TPOT ms | tok/step | accept | B=8 decode tok/s / TPOT ms | tok/step | accept |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 58.2 / 17.2 | 1.00 | - | 388.4 / 20.6 | 1.00 | - |
| 1 | 0 | 98.4 / 10.2 | 1.98 | 97.7 % | 642.2 / 12.5 | 1.96 | 96.2 % |
| 1 | 1024 | 97.7 / 10.2 | 1.96 | 96.2 % | **358.2 / 22.3** | **1.00** | **0.0 %** (known issue) |
| 3 | 0 | 139.3 / 7.2 | 3.31 | 77.1 % | 927.5 / 8.6 | 3.23 | 74.3 % |
| 3 | 1024 | 141.7 / 7.1 | 3.45 | 81.5 % | 887.1 / 9.0 | 3.07 | 69.1 % |
| 5 | 0 | 152.3 / 6.6 | 3.59 | 51.8 % | 908.9 / 8.8 | 3.36 | 47.1 % |
| 5 | 1024 | 141.3 / 7.1 | 3.31 | 46.2 % | 920.1 / 8.7 | 3.36 | 47.1 % |

DSpark speculative decoding works on SM120: **2.6x** decode speed-up at B=1 (nextn 5) and **2.4x** at B=8 (nextn 3,
the best setting there). These numbers come from a truncated checkpoint whose experts are cut on both the target and the
draft side; they show that the MTP path works and what it costs, and are **not a prediction for the full model**.

**Known issue:** nextn = 1 together with `DEEPINFER_SAMPLER_FORCE_TOPK=1024` (`--force-topk 1024`) at batch > 1 accepts
0 % of the drafts. The output is still correct (rejected drafts fall back to the target token), there is just no
speed-up. Every other combination is fine, including nextn 1 with force-topk at B=1 and nextn 3/5 with force-topk at
B=8. Not yet investigated; likely causes are the force-top-k rejection path for 2-row requests (`remap_draft_topk` read
rows / `mtp_info_read_rows`) or the MTP info pool with several requests. The server and the image do not set
`DEEPINFER_SAMPLER_FORCE_TOPK` and default to nextn 0, so the default configuration is not affected.

### Deterministic MoE: cost

b12x 1.5.1's `deterministic_output` is wrong for some batch sizes (dynamic tiles below 64) and with expert parallelism (masked routes). `sm120_moe` works around both since 0.3.1 (the 0.3.0 server refused the mode); the server logs a warning if the installed b12x is not 1.5.1, the version the workarounds were validated on (`tests/sweep_moe_deterministic.py`). `SM120_MOE_DETERMINISTIC=1` (or `tools/run_deepinfer.py --batch-invariant`) makes b12x `fused_moe` bit-exact run to run
([MOE_GEMM.md](MOE_GEMM.md) section 5). CUDA graphs, prompt 256, 128 generated tokens, TPOT ms default -> deterministic:

| model | B=1 | B=8 | B=32 |
|---|---|---|---|
| full-depth test model, 0.3.1 | 16.1 -> 19.1 | 20.8 -> 23.4 | 25.4 -> 27.0 |
| same, 0.2.x (smaller tiles, wrong for some shapes) | 16.1 -> 46.1 | 20.4 -> 42.2 | 25.6 -> 47.8 |

That is 1.06-1.19x slower decode on one GPU (with `--dp N` plus the masked routes each rank computes at weight 0,
not measured yet); the mode stays off by default. The server's
`--deterministic` flag is a different mechanism: it switches the MoE to the exact torch reference
(`SM120_MOE_IMPL=torch`), which is also reproducible and is what the server test suite uses. The torch reference is
single-GPU only (not expert-parallel; refused with `--dp` / `--ep` > 1); with `--dp` use `SM120_MOE_DETERMINISTIC=1`.

### MoE tuning levers (optional)

YOCO test model, CUDA graphs, TPOT ms at B=1/8/32 (default 4.3 / 6.0 / 8.3):

| variant | TPOT ms | init s |
|---|---|---|
| `SM120_MOE_GRAPH_BUCKETS=1` (exact MoE plan per CUDA-graph size) | 4.1 / 5.8 / 8.2 | 33 |
| `SM120_MOE_PLAN=session` (b12x `plan_execution`, exact variants, heuristic configs) | 4.1 / 5.8 / 8.2 | 35 |
| `SM120_MOE_PLAN=session` + `SM120_B12X_AUTOTUNE=1` (MoE + linear race) | 4.1 / 5.8 / 8.3 | 122 |
| `SM120_B12X_AUTOTUNE=1` (linear only) | 4.2 / 5.7 / 8.3 | 24 |

All are within about 5 % of the default: b12x's heuristic configs are already what the race picks for these shapes, so
the defaults stay (caps plans, no tuning). The remaining MoE time is kernel time. `SM120_MOE_PLAN=session` is forced
off in deterministic mode (session plans did not give bit-identical repeats with `deterministic_output`). Deterministic
mode also forces grouped routing, working around [flashinfer-ai/flashinfer#6941](https://github.com/flashinfer-ai/flashinfer/issues/6941).

`DEEPINFER_PREFILL_ATTEMPT_INTERVAL`: DeepInfer's NORMAL scheduler only attempts a prefill every 70 steps by default,
idle steps included, which shows up as about 2.2 s of "prefill" (TTFT) for every batch size. `bench_e2e.py`, the server
and the image set it to 1.

## 2. What the runtime layer does

| item | change | where |
|---|---|---|
| CUDA graphs | The GEMM/MoE shims follow DeepInfer's capture contract: everything per shape (weight packing, b12x plans + JIT, scratch, dequant cache) is created by the eager warm-up run DeepInfer does before each capture; under capture only `bind`/`run` remain. First-time creation during capture raises a clear error. The torch/deepep MoE paths refuse capture. Optional exact MoE plans per graph size (`SM120_MOE_GRAPH_BUCKETS=1`). P9 logs plans per capture. | `deep_gemm/_b12x.py`, `deep_gemm/_gemm.py`, `sm120_moe.py`, `sm120_patches.py` P9 |
| MTP / DSpark | P8: the `sampler.remap_draft_topk` TileLang kernel is rebuilt with a 21568-float shared window (84 KB) instead of 25856/44032 (103/176 KB, above SM120's 99 KB per block). Only used with `DEEPINFER_SAMPLER_FORCE_TOPK>0`. | `sm120_patches.py` P8 |
| MoE decode | b12x tiny-decode SwiGLU clamp fix as a patch (applied in the image), default stays on the faster dynamic path; opt-in `block_fp8_linear` autotune (`SM120_B12X_AUTOTUNE=1`); persistent JIT caches under `/cache`. | `patches/b12x-tiny-decode-swiglu.patch`, `sm120_patches.py`, `deep_gemm/_b12x.py` |
| logit capture | Placeholder steps (the dummy sequence, token 0 / position 0, that the worker runs when the schedule is empty) are not recorded as decode steps; recording them produced a constant decode argmax. Each decode entry carries request ids and rows. | `sm120_capture.py`, `sm120_patches.py` P6 |
| benchmark | prefill tok/s, decode tok/s, TPOT, tokens/step, acceptance; eager vs graph, MTP on/off, any batch list | `tools/bench_e2e.py` |

## 3. MTP / DSpark

DeepInfer's only speculative head is DFlash/DSpark (`models/mtp/dflash.py`): `num_nextn_predict_layers=3` layers in
`mtp.{0,1,2}.*`, fed by the hidden states of `dspark_target_layer_ids` (37, 38, 39 in V4.1-Flash), proposing
`DEEPINFER_USE_NEXTN_DECODE` (default 5) drafts per step.

- Verify batches have B·(1+nextn) rows. Graphs are sized in tokens, and the MoE sees >= 6 tokens per request, so the
  b12x tiny-decode path (T <= 4) is never used under MTP.
- The DSpark layers' MoE (`dspark_n_routed_experts`) also goes through `sm120_moe` (DeepseekMoE class patch); their
  attention runs on the b12x decode plan without an extra cache ([ATTENTION.md](ATTENTION.md)).
- With `DEEPINFER_SAMPLER_FORCE_TOPK=0` (the default of `run_deepinfer.py`) `remap_draft_topk` is not called; `1024`
  is what DeepSeek's benchmark uses and needs P8.
- `tools/run_deepinfer.py --nextn N` sets `DEEPINFER_USE_NEXTN_DECODE` and `create_llm(..., lookahead_tokens=min(1,
  nextn))` as in DeepInfer's `examples/benchmark.py`. One engine step then emits up to 1+nextn tokens with 1+nextn
  logit rows per request; `sm120_capture.decode_rows()` keeps rows 0..a (a = accepted drafts) so `decode_logits_all`
  has one row per emitted token, and `meta.mtp` records rows per step, emitted tokens per step, accepted drafts and the
  acceptance rate.
- MTP test checkpoints are built with `tools/ref/make_truncated_ckpt.py --mtp K --dspark-targets ...`
  ([MTP_YOCO.md](MTP_YOCO.md)). On checkpoints whose DSpark targets are not the original layers 37-39 the drafts are
  meaningless (acceptance 0); only the plumbing is tested. The full-depth test model keeps the real targets.

## 4. How to test (inside the image)

```bash
docker run --rm -it --gpus '"device=0"' --ipc=host --shm-size=16g \
  -v /srv/work:/work -v /srv/hf-shards:/ckpt:ro -v /srv/cache:/cache \
  --entrypoint bash somora_deep_sm120:0.3.1
cd /opt/sm120

# policy decisions printed at start-up
python3 -c "import sm120_patches as p; print(p.TINY_DECODE_POLICY, p.b12x_tiny_decode_patched())"

# GEMM/MoE shims alone: warm-up/capture/replay == eager, plan-in-capture guards
pytest -q tests/test_gemm_moe_graphs.py
# runtime unit bundle (no checkpoint): tiny-decode patch test, P8 sampler ops, GEMM/MoE graph tests
bash tests/run_runtime_unit.sh

# whole engine with graphs on the 3-layer model; compare decoded_ids with an eager run
python3 tools/run_deepinfer.py --ckpt /work/ckpt3 --out /work/di_out.pt
python3 tools/run_deepinfer.py --ckpt /work/ckpt3 --graphs --out /work/di_out_graph.pt 2>&1 | tee /work/graph.log
grep -E "sm120_patches\] CudaGraphRunner|cuda_graph_batch_sizes|capture" /work/graph.log
```

Expected: `capture done ... moe={plans: ...}` lines, then identical or near-identical decode tokens (the truncated
model is chaotic and the default MoE is not deterministic). Prefill logits are computed eagerly in both runs. If
capture fails, the P9 hint names the rule that was broken.

```bash
# P8 kernel against DeepInfer's own reference, plus the other sampler/DFlash ops
cd /opt/deepinfer/deepinfer/model_executor/layers/tilelang_ops
python3 -m pytest -q sampler.py -k "not benchmark"
python3 -m pytest -q dflash.py dflash_copy.py store_mtp_info.py schedule_mtp_varlen.py fused_sample.py -k "not benchmark"
cd /opt/sm120
# DSpark on an MTP checkpoint: nextn 5, with forced top-k, and with graphs
python3 tools/run_deepinfer.py --ckpt /work/ckpt3_mtp --nextn 5 --out /work/di_out_mtp.pt
python3 tools/run_deepinfer.py --ckpt /work/ckpt3_mtp --nextn 5 --force-topk 1024 --out /work/di_out_mtp_tk.pt
python3 tools/run_deepinfer.py --ckpt /work/ckpt3_mtp --nextn 5 --graphs --out /work/di_out_mtp_graph.pt

# tiny-decode patch (the image already has it applied; on a plain b12x install the first run reproduces the bug)
python3 tests/test_b12x_tiny_decode_swiglu.py
python3 tests/test_b12x_tiny_decode_swiglu.py --shape v41 --bench     # T=1..4 latency, tiny vs dynamic
SM120_MOE_TINY_DECODE=1 python3 tools/bench_e2e.py --ckpt /work/ckpt3 --batches 1,4 --modes graph --out /work/bench_tiny_on.jsonl
python3 tools/bench_e2e.py --ckpt /work/ckpt3 --batches 1,4 --modes graph --out /work/bench_tiny_off.jsonl
# opt-in block_fp8_linear autotune (decisions cached in /cache/b12x/preparation, first run slow)
SM120_B12X_AUTOTUNE=1 python3 tools/bench_e2e.py --ckpt /work/ckpt3 --batches 1,16 --modes graph --out /work/bench_tune.jsonl

# logit capture sanity
python3 tools/run_deepinfer.py --ckpt /work/ckpt3 --out /work/di_out.pt 2>&1 | grep -E "WARNING|decoded ids"
python3 -c "import torch; m=torch.load('/work/di_out.pt',weights_only=False)['meta']; print(m['errors'], m['n_decode_captures'])"
```

There should be no `decode step s: argmax ... != sampled token` warnings. If they appear, rerun with `--zos 0` and
inspect `sm120_capture.STATE["decode_meta"]` (request ids and argmax per row).

## 5. End-to-end benchmark (`tools/bench_e2e.py`)

```bash
python3 tools/bench_e2e.py --ckpt /work/ckpt3 --batches 1,4,16,64 --prompt-len 256 --gen 64 \
    --modes eager,graph --mtp 0 --out /work/bench_e2e.jsonl
python3 tools/bench_e2e.py --ckpt /work/ckpt3_mtp --batches 1,4,16,64 --modes eager,graph --mtp 0,5 \
    --force-topk 0,1024 --out /work/bench_e2e_mtp.jsonl
```

- For every configuration (mode x nextn x force_topk) a fresh worker process builds the engine once and runs, per
  batch size B, B requests of P random prompt tokens and G generated tokens (greedy, `ignore_eos`). DeepInfer reads
  `DEEPINFER_ENFORCE_EAGER` / `DEEPINFER_USE_NEXTN_DECODE` at engine construction, hence one process per config.
- Metrics: prefill tok/s = B·P / time until every request has its first token; decode tok/s = B·(G-1) / time from
  "all first tokens" to the last token; TPOT = decode time / (G-1); tokens/step = decode tokens / (decode steps x B);
  acceptance = (tokens/step - 1) / nextn. Init time (memory profiling, JIT, graph capture) is reported separately.
- `--warmup` (default 1) runs every batch once with G = 4 before measuring, so lazy b12x plans/JIT of new token buckets
  are not timed. Wall-clock per step includes scheduler/sampler host time and zero-overhead-scheduling overlap.
- Defaults: `max_model_len` 16384, `max_num_batched_tokens` 4096, `gpu_memory_utilization` 0.95,
  `SM120_MAX_GPU_BLOCKS=512`. 64 x (256+64) tokens need 192 blocks; raise `--max-blocks` for longer runs.
- Acceptance on truncated models only proves the plumbing; use `--prompt-ids` with real text for anything else.

## 6. Graph-safety contract for shim code

Everything inside a captured decode step must follow these rules (the attention, GEMM, MoE and engram shims do):

1. No host sync inside the forward of a decode step (`.item()`, `.tolist()`, `.cpu()`, `nonzero`, `torch.unique`,
   data-dependent shapes).
2. No host-side values derived from per-step metadata: a Python int computed from `input_metadata` tensors (sequence
   lengths, block counts, top-k lengths) during capture is frozen into the graph and wrong on every replay. Everything
   data-dependent stays in device tensors; only shapes equal to the graph size may be used on the host.
3. Plans, scratch and JIT per shape are created on the first eager call (DeepInfer warms every size before capturing
   it, largest first) and cached; creation under `torch.cuda.is_current_stream_capturing()` is refused with a clear
   error.
4. Buffers referenced by a graph are never freed or re-allocated (module-level caches).
   The same holds for compiled code: every lazy b12x preparation ends with `evict_unretained()`, which unloads the
   programs that are not retained, so `sm120_patches` (P9) runs warm-up + capture inside b12x
   `retain_compiled_programs()` and keeps the record for the life of the process.
5. The DFlash graph (`DEEPINFER_DFLASH_ENFORCE_EAGER=0`) also captures the draft layers' attention and the draft
   sampling (TileLang `fused_sample`, `topk`), so `deep_select.topk` must be capturable too.

## 7. Open points

1. **P8 TileLang source rewrite** relies on TVMScript reading the prim_func through `inspect`/`linecache`. If a
   source-keyed TileLang kernel cache served a stale 103 KB kernel, clear `/cache/tilelang` once.
2. **Tiny-decode** stays off by default because it is slower on the RTX 5090, even with the clamp fix. In the
   unclamped kernel FC2 activations are packed to f16 and could also overflow on large activations.
3. **MTP:** nextn 1 + force-topk 1024 at batch > 1 accepts no drafts (see above).
4. **MoE speed:** the tuning levers above give at most about 5 %. Further gains would need changes outside the shim
   (FP8 activations straight from DeepInfer's quantized input instead of bf16 -> MXFP8 inside b12x, or fusing the
   shared expert into the routed call).
5. **Long decodes:** acceptance and step-time numbers so far come from short runs (16-64 decode steps).
6. **Eager prefill MoE reloads:** b12x MoE programs used only by eager prefill shapes (not captured in a graph)
   are still evicted by later lazy preparations and loaded again on their next use. This is safe since the P9
   retention fix, but costs load time on some first-seen prefill shapes. A fix would keep `sm120_moe`'s plans in
   one b12x session or retain them as well.
