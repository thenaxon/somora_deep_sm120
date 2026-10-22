# Validation against the DeepSeek reference

DeepInfer runs unchanged on SM120: its SM100-only kernel libraries are replaced by the drop-in packages in `sm120/`, and
a few runtime patches are applied at start-up ([DESIGN.md](DESIGN.md), [../patches/PATCHES.md](../patches/PATCHES.md)).
To check numerics, truncated real-weight V4.1-Flash checkpoints (`tools/ref/make_truncated_ckpt.py`) are fed to both
DeepInfer and DeepSeek's official reference code (the `inference/` directory shipped with the model, `inference/model.py`),
and the outputs are compared layer by layer.

All numbers below were measured on a single RTX 5090 (32 GB) with **truncated test checkpoints**, not with the full
model. They show that the port reproduces the reference within the reference's own kernel-to-kernel noise; they say
nothing about the quality of the full model.

## 1. Test checkpoints

| name | built with `make_truncated_ckpt.py` | layers (original ids) | routed experts | params DeepInfer / ref | use |
|---|---|---|---|---|---|
| `ckpt3` (+ `ref-mp1`) | default | 0, 2, 20 | 384 | 24.0 / 24.5 GiB | 3-layer test model |
| `ckpt_yoco` (+ `ref-mp1-yoco`) | `--preset yoco` | 0, 1, 2, 8*, 14, 20, 21, 24 | 128 | 22.6 / 23.2 GiB | YOCO test model ([MTP_YOCO.md](MTP_YOCO.md)); `8*` = demoted to a C2 consumer |
| `ckpt3_mtp` | `--layers 0,2,20 --mtp 3 --dspark-targets 0,2,20 --num-experts 128` | 0, 2, 20 + mtp.0-2 | 128 | 18.0 GiB | MTP/DSpark plumbing |
| `ckpt_yoco_mtp` | `--preset yoco --mtp 3 --dspark-targets 20,21,24 --num-experts 64` | YOCO + mtp.0-2 | 64 | 21.0 GiB | MTP on the YOCO model |
| `ckpt_yoco_engram` (+ `ref-mp1-yoco-engram`) | `--preset yoco --keep-engram 1` | YOCO, engram on layer 1 (shard symlinked) | 128 | 22.6 GiB + table | Engram ([ENGRAM.md](ENGRAM.md)) |
| `ckpt_full` (+ `ref-mp1-full`) | `--layers 0-39 --num-experts 16 --mtp 3 --mtp-num-experts 32 --keep-engram 1,14` | all 40 + mtp.0-2 (real targets 37-39), engram 1 and 14 | 16 (MTP: 32) | 23.2 / 22.5 GiB | full-depth test model (§6) |

Always validate first with `--dry-run`, which prints the plan, the DeepInfer topology check and the GPU estimate.
When shard 00001 (vision) is not available, the vision/aligner tensors are synthesized; text-only prompts never run
the ViT. See [../tools/ref/README.md](../tools/ref/README.md) for the truncation details.

## 2. Running (inside the container)

Mounts: `/ckpt` = raw HF shards (ro), `/work` = work dir, `/ds41ref` = the reference `inference/` code (`REF_SRC`),
`/cache` = JIT caches. The image already has `PYTHONPATH=/opt/sm120/sm120:/opt/deepinfer`.

```bash
docker run --rm -it --gpus '"device=0"' --ipc=host --shm-size=16g --entrypoint bash \
  -v $CKPT_HOST:/ckpt:ro -v $WORK_HOST:/work -v $REF_HOST/inference:/ds41ref/inference:ro -v $CACHE_HOST:/cache \
  somora_deep_sm120:0.3.1

# 0. sanity: the shims shadow the (absent) vendor libs, the patches are active
python3 -c "import deep_gemm, deep_ep, sm120_patches; print(deep_gemm.__version__, deep_ep.__version__, sm120_patches.enabled())"
```

### 3-layer test model

```bash
# reference (tilelang + torch kernels = noise floor); builds ckpt3 / ref-mp1 if missing, writes /work/prompt_ids.json
REF_SRC=/ds41ref/inference /opt/sm120/tools/ref/run_ref.sh all
# DeepInfer with the default backends (b12x), then with the pure-torch shim backends (localizes a b12x mapping problem)
python3 /opt/sm120/tools/run_deepinfer.py --ckpt /work/ckpt3 --prompt-ids /work/prompt_ids.json --out /work/di_out.pt
SM120_GEMM_IMPL=torch SM120_MOE_IMPL=torch SM120_ATTN_IMPL=torch python3 /opt/sm120/tools/run_deepinfer.py --out /work/di_out_torch.pt
# optional: teacher-forced decode (every step sees the reference prefix, so per-step logits are comparable)
python3 /opt/sm120/tools/run_deepinfer.py --teacher-force --ref /work/ref_out.pt --out /work/di_out_tf.pt
# compare: per-layer prefill hidden error vs the ref-vs-ref noise floor (the pass signal), plus logits
python3 /opt/sm120/tools/compare.py --di /work/di_out.pt --ref /work/ref_out.pt --ref2 /work/ref_out_torch.pt
```

### YOCO test model, MTP, Engram

See [MTP_YOCO.md](MTP_YOCO.md) §7-8 and [ENGRAM.md](ENGRAM.md) §6.

### Full-depth test model

```bash
python3 /opt/sm120/tools/ref/make_truncated_ckpt.py --src /ckpt --dst /work/ckpt_full --layers 0-39 --num-experts 16 \
  --mtp 3 --mtp-num-experts 32 --keep-engram 1,14          # 23.2 GiB written (+189 GiB engram symlinked)
python3 /opt/sm120/tools/ref/ref_convert.py --hf-ckpt-path /work/ckpt_full --save-path /work/ref-mp1-full
cd /opt/sm120/tools/ref; for k in tilelang torch; do  # noise floor = tilelang vs torch kernels
  CAPTURE_LAYER=-1 REF_KERNELS=$k python3 ref_runner.py --ckpt /work/ref-mp1-full --hf-config /work/ckpt_full/config.json \
    --ref-src /ds41ref/inference --prompt-ids-json /work/prompt_ids.json --save-prompt-json '' --max-seq-len 4096 --compact \
    --out /work/full_short_ref$([ $k = torch ] && echo _torch).pt; done      # likewise prompt_ids_yoco.json -> full_long_*
CAPTURE_LAYER=-1 SM120_ENGRAM_MODE=host python3 /opt/sm120/tools/run_deepinfer.py --ckpt /work/ckpt_full \
  --prompt-ids /work/prompt_ids.json --compact --max-num-batched-tokens 512 --gpu-memory-utilization 0.95 \
  --out /work/full_short_di.pt                                     # long prompt: --max-num-batched-tokens 2304
#   variants: --graphs, --nextn 1, --nextn 5, --graphs --nextn 5 (with --no-hidden)
python3 /opt/sm120/tools/compare.py --di /work/full_short_di.pt --ref /work/full_short_ref.pt --ref2 /work/full_short_ref_torch.pt
```

With the defaults (`--gpu-memory-utilization 0.90`, `--max-num-batched-tokens 4096`) DeepInfer stops at start-up with
"No available memory for the cache blocks": the profiling pass over 4096 tokens x 40 layers no longer fits beside the
parameters. `0.95` plus a batch of at most the prompt length rounded up works.

## 3. Switches

| env | default | meaning |
|---|---|---|
| `SM120_GEMM_IMPL` | `b12x` | `b12x` or `torch` for the deep_gemm shim |
| `SM120_MOE_IMPL` | follows GEMM | `b12x` (fused_moe), `torch` (exact reference) or `deepep` (DeepInfer's original EPBuffer + grouped-GEMM path on the shims) |
| `SM120_ATTN_IMPL` | `b12x` | attention shim backend (finer switches in `sm120/sm120_attn/_config.py`, [ATTENTION.md](ATTENTION.md)) |
| `SM120_FUSED_NORM_ROPE_ATTN` | `auto` | DeepInfer's FlashMLA fused norm/rope/attention gate (patch P5) |
| `SM120_MAX_GPU_BLOCKS` | 256 (set by run_deepinfer) | KV-cache cap in 128-token blocks |
| `SM120_GEMM_STRICT`, `SM120_MOE_STRICT` | 0 | 1 = raise instead of silently falling back to torch |
| `SM120_PATCHES` | 1 | 0 disables all runtime patches |
| `CAPTURE_LAYER` | `0` | fine taps for both runners: `0`, `3,5,6`, `all`, or `-1` = off |
| `REF_KERNELS` | `tilelang` | reference kernels: `tilelang`, `torch`, `check` ([../tools/ref/README.md](../tools/ref/README.md)) |

`tools/run_deepinfer.py` sets these unless they are already set (env wins, except where a flag sets them explicitly):

- `DEEPINFER_USE_NEXTN_DECODE=0` (`--nextn N`), `DEEPINFER_ENFORCE_EAGER=1` and `DEEPINFER_DFLASH_ENFORCE_EAGER=1`
  (`--graphs` sets both to 0), `DEEPINFER_DISABLE_SYNC_GUARD=1`
- `DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS=0`, `DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL=0`,
  `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=off` (YOCO consumers see every prefill query, as in the reference)
- `create_llm(..., lookahead_tokens=min(1, nextn))`, as in DeepInfer's `examples/benchmark.py`

Flags:

| flag | default | |
|---|---|---|
| `--ckpt`, `--prompt-ids`, `--out` | `/work/ckpt3`, `/work/prompt_ids.json`, `/work/di_out.pt` | |
| `--decode-steps` | 16 | |
| `--max-model-len` | 16384 | DeepInfer needs >= candidate_topk_blocks x block size |
| `--max-num-batched-tokens` | 4096 | must be at least the prompt length (no chunked prefill) |
| `--max-blocks` | 256 | `SM120_MAX_GPU_BLOCKS` |
| `--gpu-memory-utilization` | 0.90 | |
| `--nextn N` | env or 0 | MTP/DSpark draft tokens per step |
| `--graphs` | off | CUDA graphs |
| `--force-topk K` | - | `DEEPINFER_SAMPLER_FORCE_TOPK` |
| `--teacher-force` | off | force the reference tokens (`--ref`) |
| `--zos` | engine default | `0` or `1` sets the zero-overhead scheduler |
| `--batch-invariant`, `--compact`, `--no-hidden` | off | |

## 4. Dump format (`sm120-ref-v1`, the same for `tools/ref/ref_runner.py` and `tools/run_deepinfer.py`)

- `input_ids`
- `decoded_ids`: `[prefill argmax, 16 greedy decode tokens]`
- `prefill_logits` `[L, V]` (None with `--compact`), `prefill_logits_last64`, `prefill_top20_values` / `ids`, `prefill_argmax`
- `decode_top50_values` / `ids` `[16, 50]`, `decode_logits_first4` `[4, V]`; DeepInfer also `decode_logits_all`
  `[16, V]` (one row per emitted token, also with MTP)
- `hidden["prefill"]`: `embed_hc` `[1, L, 4, 5120]` (block-0 input), `layer{i}` `[1, L, 4, 5120]` (block outputs;
  DeepInfer: after the deferred mHC post), `final_norm` `[1, L, 5120]`, fine taps `L<i>/<name>`
- `meta`: both runners record `truncation` (the checkpoint's `truncation.json`) and `capture_layers`. DeepInfer also
  records env, patches, errors, peak memory, `decode_meta`, `n_placeholder_skipped`, `step_timing`, and `mtp`
  (`nextn`, `rows_per_step`, `emitted_per_step`, `accepted_log`, `tokens_per_step`, `draft_acceptance`). The reference
  also records `model_args`, kernel backends and fallbacks, and `engram`.

How the DeepInfer logits are captured (`sm120/sm120_capture.py`):

- Prefill logits for all positions are recomputed from the sampler's input hidden states with the same
  bf16 x bf16 -> fp32 GEMM as `sampler_utils.get_logits`.
- Decode logits are what DeepInfer's sampler sees, with placeholder steps dropped.
- With MTP each step has 1 + nextn rows; rows 0..a are kept, where a is the step's accepted draft count.

**Fine taps.** Both runners record about 30 intermediates per tapped layer (`CAPTURE_LAYER`) into
`hidden["prefill"]["L<i>/<name>"]`: hc trunk / mixes / branch inputs, q, kv, RoPE tables, SWA indices, attention core,
gate, routed / shared / ffn outputs, plus `compress_idx` (compressed positions attended) and `cand_mask` (on the
candidate source). At about 2k tokens a tapped layer costs about 1 GB per dump, so prefer a few layers over `all`.

```bash
python3 /opt/sm120/tools/compare_l0.py --layer 0 --di /work/di_out_torch.pt --ref /work/ref_out.pt --ref2 /work/ref_out_torch.pt
python3 /opt/sm120/tools/compare_l0.py --layer all ...    # every layer tapped in both dumps, plus a summary
```

## 5. Reading the comparison

Truncated models amplify ULP-level differences. Two reference runs that differ only in kernel rounding (tilelang vs
torch kernels) agree on top-1 for only about 70-74 % of prefill positions on the 3-layer and YOCO models, and from the
third layer on their hidden states differ by 0.1-0.2 rel-L2. End-to-end logits are therefore informational.

`compare.py` flags a hidden-state layer as **SUSPICIOUS** when DeepInfer-vs-ref rel-L2 exceeds 3x the ref-vs-ref2 noise
of that layer (`--factor`, with a `--floor` of 1e-4 for exact layers such as `embed_hc`). Each row is annotated with the
original layer id and its role (e.g. `o21 C1 kv<5 idx<5 yoco`). The first suspicious layer localizes a bug; bisect with
`SM120_GEMM_IMPL` / `SM120_MOE_IMPL` / `SM120_ATTN_IMPL=torch`, then run `compare_l0.py` on that layer.

### Known deviation: layer 0

Layer 0 is flagged in every run (ratio 5-20x), also with the pure-torch shims, so it is not caused by b12x. Fine taps
localize it: everything up to the KV projection matches (<= 1e-4; RoPE tables, SWA mask and sink are identical). The
first difference is `kv` after norm, RoPE and FP8 cast (2.2e-3), inside DeepInfer's own TileLang kernel
`fused_transform`: DeepInfer rotates the fp32 norm result, the reference rotates the bf16-rounded one, and individual
FP8 codes flip. The output projection amplifies this, and 10-12 % of tokens then pick different experts. The kernel
matches its own torch reference on SM120. This is a rounding property of DeepInfer relative to the reference, not a
port error; it is expected to appear on SM100 too (not verified). Later layers return to the noise floor.
`compare.py --known-deviation layer0` prints the row as `known` and leaves it out of the exit status
(`tests/e2e_matrix.sh` passes it by default; `TEST_KNOWN_DEVIATION=` turns it off).

Exit status of `compare.py`: 0 ok, 1 suspicious layers, 2 invalid comparison (different input ids, capture errors,
missing hidden states without `--allow-missing-hidden`), 3 a failed `--decode-check identical|diverge`. Teacher-forced
dumps decode the forced tokens, so there the check uses the decode logits: identical = top-1 agreement 1.0 and mean
top-50 KL at most `--diverge-kl` (0.5), diverge = above it.

## 6. Results (RTX 5090 test machine)

### 3-layer test model (327-token prompt, 16 decode steps)

- DeepInfer runs end to end on SM120: loading, memory profiling, paged KV cache, C++ scheduler, prefill, decode, sampler.
- b12x contribution in isolation (DeepInfer + b12x vs DeepInfer + exact torch shims, layer 0): attention path `q_a`
  8e-7, `attn_core_out` 3e-6, `attn_out` 1.6e-4 rel-L2; MoE routing identical (100 % same expert sets), `routed_out`
  3.1e-2 rel-L2 (the expected FP8 intermediate-quantization difference of b12x w4a8_mx).
- Against the reference: layers 1, 2 and `final_norm` are within the noise floor (factor 0.8-2.1); layer 0 deviates as
  described in §5. Prefill top-1 68.5 % (noise 74 %).

### YOCO test model (2081-token prompt)

Index top-k is active on every compressed layer; candidate filtering is active beyond 1024 positions. Prefill hidden
rel-L2 of DeepInfer (b12x) vs reference, against the noise floor (ref tilelang vs ref torch):

| layer | role | DI-ref | noise | ratio |
|---|---|---|---|---|
| 0 | o0 SWA | 1.84e-2 | 9.0e-4 | 20.4 **flagged** (§5) |
| 1 | o1 SWA | 3.52e-2 | 1.44e-2 | 2.43 ok |
| 2 | o2 C2 src | 5.53e-2 | 3.97e-2 | 1.39 ok |
| 3 | **o8\* C2 consumer** | 1.33e-1 | 1.21e-1 | 1.09 ok |
| 4 | o14 C2 src | 1.69e-1 | 1.59e-1 | 1.06 ok |
| 5 | **o20 C1 src + candidate src (YOCO boundary)** | 1.59e-1 | 1.52e-1 | 1.05 ok |
| 6 | **o21 C1 consumer** | 1.58e-1 | 1.50e-1 | 1.05 ok |
| 7 | **o24 C1 index src using candidates** | 1.80e-1 | 1.71e-1 | 1.05 ok |
| final_norm | | 2.02e-1 | 1.87e-1 | 1.08 ok |

- Decoded tokens identical to the reference over all 17 steps (b12x eager, torch attention eager, and b12x with CUDA
  graphs). Prefill top-1 0.677 (ref vs ref2: 0.702).
- `compare_l0 --layer 3,5,6,7` flags 0 tensors for both DeepInfer variants (b12x and pure-torch shims). kv, attn_out,
  ffn_in, routed_out and ffn_out all sit at about 1.05-1.2x their noise column; SWA key sets are identical in 100 % of rows.
- DeepInfer's selections vs the reference, against the ref2-vs-ref noise:

  | layer | role | `compress_idx` mean set overlap, DI / ref2 | `cand_mask` overlap, DI / ref2 |
  |---|---|---|---|
  | 3 | C2 consumer | 0.986 / 0.991 | - |
  | 5 | C1 source + candidate source | 0.913 / 0.916 | 0.966 / 0.968 |
  | 6 | C1 consumer | 0.913 / 0.916 | - |
  | 7 | index source using candidates | 0.902 / 0.905 | - |

- Short prompt (327 tokens, all 8 layers tapped): layers 1-7 at 1.1-2.2x noise, layer 0 at 4.7x.
- Measured peaks: reference 26.0 GiB, DeepInfer 26.5 GiB. The tilelang reference falls back to torch for `sparse_attn`
  (it needs 141 KB of shared memory, more than SM120 has).

### Full-depth test model

All 40 layers with every real attention, norm and mHC weight and the shared experts; routed experts cut to 16/384 per
layer and 32/128 per MTP layer; all 3 DSpark/MTP heads with their real targets 37, 38, 39; engram on layers 1 and 14
(tables mmapped from the symlinked shards).

Memory:

| run | params | peak |
|---|---|---|
| reference, 327 / 2081 tokens (tilelang) | 22.5 GiB (fp32 head, no vision/MTP) | 23.2 / 25.6 GiB |
| reference, torch kernels | | 24.0 / 25.6 GiB |
| DeepInfer eager, 327 tokens (`max_num_batched_tokens` 512) | 23.2 GiB | 24.9 GiB |
| DeepInfer eager, 2081 tokens (2304) | | 27.5 GiB |
| DeepInfer `--graphs` / `--nextn 1,5` (327 tokens) | | 24.9 / 27.5 GiB |

The KV cache is tiny (114 KB per 128-token block for 40 layers). Engram tables stay in host page cache through the
mmap. 24 routed experts per layer would need 28.7 GiB of parameters and does not fit in 32 GB.

Prefill hidden rel-L2 per layer, DeepInfer (b12x, eager) vs reference, divided by the ref-vs-ref noise:

| layers | 327 tokens | 2081 tokens |
|---|---|---|
| 0 (SWA) | 11.6 flagged (§5) | 17.2 flagged (§5) |
| 1 (SWA + engram) | 2.19 | 2.30 |
| 2-19 (C2: sources 2/8/14 + consumers) | 1.10-1.54 | 1.01-1.33 |
| 20-27 (C1: candidate source 20, consumers, index source 24) | 1.08-1.16 | 1.01-1.03 |
| 28-39 (C1, index sources 28/32/36 + consumers) | 2.37-2.87 (noise only about 2.5e-2 here; DI-ref 6-7e-2) | 0.51-0.74 |
| final_norm | 1.06 | 1.01 |

No layer other than layer 0 is flagged (threshold 3x noise).

Logits:

| | prefill top-1 DI-ref | ref vs ref2 | decode |
|---|---|---|---|
| 327 tokens, eager | 0.927 | 0.933 | 5/17 identical prefix, 11/17 position-wise; first 4 decode steps top-1 1.000 |
| 2081 tokens, eager | 0.949 | 0.953 | 9/17 prefix, 15/17 position-wise; first 4 top-1 1.000 |
| 327 tokens, `--graphs` | 0.927 | 0.933 | one run: 17/17 identical to the torch-kernel reference, 5/17 vs tilelang (see below: one draw of the non-deterministic MoE) |

- With all 40 layers the model is far less chaotic than the shorter truncations (top-1 0.93-0.95 vs 0.68-0.74), and
  the decoded tokens look like real text.
- **Run-to-run variation and eager vs graphs.** The default b12x `fused_moe` combines the top-k expert outputs with
  atomics and is not run-to-run deterministic. On this truncated checkpoint the greedy tokens therefore flip at
  near-ties (positions 2 and 5) from run to run, in eager and graph mode alike, while the per-layer errors against the
  reference stay within noise (ratios 1.06-1.18). The "17/17" graph run above is one such draw; an eager run picked a
  different token at position 5.
- **Tested: deterministic eager == graphs.** With `SM120_MOE_DETERMINISTIC=1` (b12x `deterministic_output`), eager,
  graphs and a second graphs run are bit-identical: 17/17 tokens and 0.0 difference on every layer. So CUDA-graph
  capture/replay introduces no error. The deterministic tokens
  (`[304, 270, 1840, 5090, 294, 270, 14108, 3554, 33, 455, ...]`) leave the torch-kernel reference at position 2
  (1840 vs 3072), a near-tie decided only by the MoE combine order.
- For reproducible outputs on one GPU use the exact torch MoE (`SM120_MOE_IMPL=torch`, which the server's
  `--deterministic` sets; single-GPU only, refused with `--dp` / `--ep` > 1), with `--dp` `SM120_MOE_DETERMINISTIC=1`. b12x 1.5.1's `deterministic_output` is wrong for some batch sizes (dynamic tiles below 64) and with expert parallelism (masked routes); 0.3.0 refused `SM120_MOE_DETERMINISTIC=1`, and since 0.3.1 `sm120_moe` works around both bugs (validated on b12x 1.5.1).

Speed (327-token prompt, 16 decode steps, single request; step times from `meta.step_timing`):

| mode | time to the prefill sample | decode step (median) | tokens/step | per token |
|---|---|---|---|---|
| eager, nextn 0 | 41 s (includes first-call JIT) | 393 ms | 1 | 393 ms |
| **`--graphs`, nextn 0** | **2.2 s** | **28.4 ms** | 1 | **28.4 ms (35 tok/s)** |
| eager, nextn 1 | 38.6 s | 423 ms | 1.45 | 291 ms |
| eager, nextn 5 | 34.8 s | 547 ms | 2.29 | 239 ms |
| `--graphs`, nextn 5 | 2.8 s | 44.8 ms | 1.60 | 28.0 ms |

The eager prefill time is dominated by first-call JIT and the eager paths; with graphs, warm-up happens during init.
For throughput numbers use `tools/bench_e2e.py` ([RUNTIME.md](RUNTIME.md)) and the server benchmarks
([SERVER.md](SERVER.md)).

MTP acceptance (real DSpark targets, but 16/384 backbone experts and 32/128 draft experts, 7-11 steps per run):

| run | accepted drafts per step | draft acceptance | tokens/step |
|---|---|---|---|
| eager, nextn 1 | `[0,0,1,0,0,1,1,0,1,1,1]` | 45 % | 1.45 |
| eager, nextn 5 | `[0,0,1,0,5,2,5]` | 26 % | 2.29 |
| graphs, nextn 5 | `[0,0,1,0,0,1,2,0,1,5]` | 12 % | 1.60 |

- Unlike the shorter truncations (0 % acceptance), drafts are often accepted, including full 5-token runs. This is still
  not representative of the real model.
- **The speculative == greedy invariant holds exactly:** with the deterministic MoE (`SM120_MOE_IMPL=torch`, eager),
  nextn 0, 1 and 5 decode the identical 17 tokens (nextn 1 accepts 60 % of drafts, nextn 5 accepts 33 %). The token
  differences in the b12x runs above come from b12x `fused_moe` nondeterminism, not from the MTP path.

## 7. Open points

- The default b12x `fused_moe` is not run-to-run deterministic; use the torch MoE (`--deterministic`) when
  bit-reproducibility matters. `SM120_MOE_DETERMINISTIC=1` (b12x, bit-exact, 1.06-1.19x slower decode on one GPU) is usable since 0.3.1, which works around b12x 1.5.1's `deterministic_output` bugs (dynamic tiles below 64, masked routes).
- Longer decodes (hundreds of steps) are needed for stable acceptance and step-time numbers.
- No validation of the full (untruncated) model yet: it does not fit one 32 GB GPU and needs the multi-GPU path
  ([MULTI_GPU.md](MULTI_GPU.md)).
