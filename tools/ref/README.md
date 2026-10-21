# tools/ref: truncated checkpoints and the DeepSeek reference runner

These scripts build **truncated DeepSeek-V4.1-Flash checkpoints from real weights** and run DeepSeek's official
minimal inference code (the `inference/` directory shipped with the model, here called the reference) on them. DeepInfer
(on SM120) and the reference load the same checkpoint, so their outputs can be compared directly
([../../docs/VALIDATION.md](../../docs/VALIDATION.md)). Everything runs inside the container on one 32 GB GPU.

The reference code is not included. Point `REF_SRC` / `--ref-src` at its `inference/` directory (in the container
conventionally mounted at `/ds41ref/inference`; by default `<repo>/third_party/ds41-inference/inference` is tried).

| file | purpose |
|---|---|
| `make_truncated_ckpt.py` | `/ckpt` -> `/work/ckpt3` (default: original layers 0/2/20 renamed to 0/1/2). Writes shards (each under 10 GB), a new index, the edited `config.json` and `truncation.json`, and copies the tokenizer files and `encoding/`. Checks the result against both loaders and against an offline replica of DeepInfer's topology checks. Options for larger test models (`--layers`, `--preset yoco`, `--num-experts`, `--as-consumer`, `--candidate-topk-blocks`, `--keep-engram`, `--mtp`, `--mtp-num-experts`, `--dspark-targets`): see [../../docs/MTP_YOCO.md](../../docs/MTP_YOCO.md). |
| `ref_convert.py` | `/work/ckpt3` -> `/work/ref-mp1/model0-mp1.safetensors`. A streaming reimplementation of the reference `convert.py` for MP=1 with fp4 experts. |
| `ref_runner.py` | The reference run: prefill, then 16 greedy decode steps; writes `/work/ref_out.pt` and `/work/ref_out.summary.json`. |
| `ref_kernels.py` | Pure-torch versions of every function in the reference `kernel.py`, plus a per-function dispatcher between tilelang and torch. |
| `ref_capture.py` | Fine prefill taps inside reference blocks (counterpart of `sm120/sm120_capture.py`), for `tools/compare_l0.py`. |
| `ref_engram.py` | Engram for the reference: an mmap-backed table instead of a 98 GB CUDA parameter ([../../docs/ENGRAM.md](../../docs/ENGRAM.md)). |
| `kernel_selftest.py` | Compares tilelang and torch kernels on random inputs, on the GPU. |
| `compare_out.py` | Compares two output files, e.g. tilelang vs torch, or the reference vs a DeepInfer dump in the same format. |
| `st_io.py` | Dependency-free streaming safetensors reader and writer that copies raw bytes. |
| `run_ref.sh` | Runs the whole pipeline (`SRC`, `WORK`, `REF_SRC`, `CKPT`, `REFMP`, `OUT`, `PROMPT_JSON`, `MAKE_ARGS`, `RUN_ARGS`). |

## How to run (inside the container)

```bash
cd /opt/sm120/tools/ref
# 1. truncated HF checkpoint (about 26 GB written, less than 200 MB RAM)
python3 make_truncated_ckpt.py --src /ckpt --dst /work/ckpt3            # --dry-run validates without writing
#    optional: --verify-bytes re-reads all tensors and compares hashes; --vision copy|synth|auto (default auto)

# 2. reference-format checkpoint (MP=1, fp4 experts)
python3 ref_convert.py --hf-ckpt-path /work/ckpt3 --save-path /work/ref-mp1

# 3. kernel sanity check on SM120 (recommended once)
python3 kernel_selftest.py --ref-src /ds41ref/inference

# 4. reference run (default: tilelang kernels, falling back to torch per function)
REF_KERNELS=tilelang python3 ref_runner.py --ckpt /work/ref-mp1 --hf-config /work/ckpt3/config.json \
  --ref-src /ds41ref/inference --out /work/ref_out.pt
REF_KERNELS=torch    python3 ref_runner.py --ckpt /work/ref-mp1 --hf-config /work/ckpt3/config.json \
  --ref-src /ds41ref/inference --out /work/ref_out_torch.pt
REF_KERNELS=check    python3 ref_runner.py ... --out /work/ref_out_check.pt   # runs both on every call and logs the diffs
python3 compare_out.py /work/ref_out.pt /work/ref_out_torch.pt

# or simply
SRC=/ckpt WORK=/work REF_SRC=/ds41ref/inference ./run_ref.sh all
```

### Environment variables (`ref_kernels.py`)

| var | values | meaning |
|---|---|---|
| `REF_KERNELS` | `tilelang` (default), `torch`, `check` | `tilelang` imports the official `kernel.py`. Any function that raises (import, JIT compile or launch error) switches to torch for the rest of the run; the switch is logged and saved in `meta.kernels.fallbacks`. `torch` never imports tilelang. `check` runs torch on cloned inputs, then tilelang, and records `max_abs`, `max_rel` and the bit-exact count per function. |
| `REF_CHECK_USE` | `tilelang` (default), `torch` | In `check` mode, which result the model continues with. |
| `REF_KERNELS_TORCH` | e.g. `sparse_attn,fp4_gemm` | Forces these functions to torch in tilelang or check mode. Use it to bisect, or after a CUDA fault. |
| `REF_KERNEL_SYNC` | `1` | Synchronizes after every tilelang call, so asynchronous CUDA faults are attributed to the right kernel. |
| `REF_SRC` | path | Location of the reference `inference/` directory. Same as `--ref-src`. |

On SM120 the reference's tilelang `sparse_attn` needs 141 KB of shared memory and falls back to torch; everything else
runs in tilelang.

### ref_runner options

- `--max-seq-len` (default 1024): sizes the KV caches.
- `--decode-steps` (default 16).
- `--compact`: drops the full prefill logits (about 170 MB).
- `--no-hidden`: skips the per-layer hidden states.
- `--prompt-ids-json`: reuses exact ids, e.g. `/work/prompt_ids.json`.
- `--prompt-tokens N`: lengthens the built prompt to at least N tokens with deterministic synthetic logbook lines (makes
  the indexer top-k and the candidate blocks select).
- `--capture-layer` / env `CAPTURE_LAYER`: fine taps of one layer (`0`), several (`3,5,6`) or `all`; `-1` = off.
- MTP layers in the HF config are ignored (DSpark off); engram layers are served by `ref_engram.py` (`--engram-ckpt`,
  default the HF config dir; `--engram-mode host|uva`).
- `--save-prompt-json` (default `/work/prompt_ids.json`): DeepInfer should be fed exactly these ids.

### Output: `/work/ref_out.pt` (torch.save dict, all CPU tensors, format `sm120-ref-v1`)

| key | content |
|---|---|
| `input_ids` | `[L]` int64. About 330 tokens: BOS + `<｜User｜>` passage + question + `<｜Assistant｜>`, built by `encoding.encode_messages(..., thinking_mode="chat")` and tokenized with `tokenizers` using `add_special_tokens=False`. |
| `prefill_logits` | `[L, 129280]` fp32, every position. `None` with `--compact`. |
| `prefill_logits_last64`, `prefill_top20_values` / `_ids` (`[L, 20]`), `prefill_argmax` | |
| `decoded_ids` | `[17]`. Entry 0 is the argmax of the last prefill position; entry s+1 is the argmax of decode step s, whose input is `decoded_ids[s]` at position `L+s`. This matches the reference `generate.py`. |
| `decode_top50_values` / `_ids` | `[16, 50]` |
| `decode_logits_first4` | `[4, 129280]` fp32 |
| `hidden` | Prefill: `embed_hc` (block-0 input), `layer{i}` (block output, hc stream `[1, L, 4, 5120]` bf16), `final_norm` (head input), fine taps `L<i>/<name>`. |
| `meta` | ModelArgs, kernel backends, fallbacks and check statistics, timings, peak memory, versions, `truncation`, `capture_layers`, `engram`. |

### Runtime and memory (3-layer model, RTX 5090)

- `make_truncated_ckpt.py` writes about 26 GB (3 layer shards of about 7.4 GB plus about 3.6 GB for embed, head and the
  vision placeholders), bounded by disk speed. RSS stays under 200 MB because copies stream in 64 MiB chunks.
- `ref_convert.py` writes about 23.5 GB; RSS is at most a few hundred MB plus the largest tensor.
- `ref_runner.py`: about 24.5 GiB of GPU parameters (routed experts fp4 20.2 GiB, head kept as fp32 2.47 GiB, embed bf16
  1.23 GiB, the rest about 0.6 GiB), measured peak 23.7 GiB at 327 tokens. Host RAM stays under 6 GB because weights are
  loaded one tensor at a time. The first tilelang run compiles about 25 kernels; later runs use the tilelang cache
  (`TILELANG_CACHE_DIR`). Torch mode takes about 1 minute for prefill plus decode.

## The 3-layer test model

| new layer | original | compress ratio | role |
|---|---|---|---|
| 0 | 0 | 0 | SWA only |
| 1 | 2 | 2 | kv source + index source (has `compressor.wgate`, indexer `wk` / `k_norm`) |
| 2 | 20 | 1 | kv source + index source + candidate source; first ratio-1 layer, so DeepInfer `n_prefill_layer = 2` |

Kept tensors: `embed.weight`, `norm.weight`, `head.weight`, `image_start`, `image_end`, `image_newline`; all
`layers.{0,2,20}.*`, renamed; vision and aligner tensors (263, see below). There are no `hc_head_*` tensors: both stacks
use shifted HC, where the last block's FFN pre-mix collapses the stream before the head (DeepInfer `hc_shifted`
defaults to True). Dropped: all `mtp.*` tensors (DSpark) and all other layers.

### config.json edits (`text_config`; everything else is byte-identical JSON content)

| key | original | truncated | why |
|---|---|---|---|
| `num_hidden_layers` | 40 | 3 | |
| `compress_ratios` | 43 entries (40 + 3 MTP) | `[0, 2, 1]` | DeepInfer asserts `len == n_layer + nextn`; the reference indexes it by `layer_id`. |
| `kv_source_layer_ids` | `[2, 8, 14, 20]` | `[1, 2]` | |
| `index_source_layer_ids` | `[2, 8, 14, 20, 24, 28, 32, 36]` | `[1, 2]` | |
| `candidate_source_layer_id` | 20 | 2 | |
| `engram_layer_ids` | `[1, 14]` | `[]` | DeepInfer `_build_engram` and the reference `EngramLayout.from_args` both return None. |
| `engram_num_embeddings` | `[384006168, 384016682]` | `[]` | Kept consistent with the ids. |
| `num_nextn_predict_layers` | 3 | 0 | |
| `dspark_target_layer_ids` | `[37, 38, 39]` | `[]` | Unused with nextn 0. The other `dspark_*` keys stay because DeepInfer reads them unconditionally. |

`index_k_source_layer_ids` is not present, so DeepInfer defaults it to the kv sources; it would be remapped if it
existed. `vision_config`, `quantization_config`, `image_token_id` and the bos/eos/pad ids are unchanged; DeepInfer needs
`vision_config`. The reference runner turns vision off itself with `vision_n_layers = 0`.

DeepInfer derives from the edited config: `n_prefill_layer = 2` (the first ratio-1 layer), `layer_kv_map =
layer_index_module_map = {1: 1, 2: 2}`, `layer_candidate_source_map = {2: 2}`, YOCO enabled (2 < 3).

### Vision / aligner

DeepInfer always builds the ViT and aligner parameter tree, refuses to load if any model parameter is not covered by the
HF index, and requires matching dtypes. Those tensors live in `model-00001`. With `--vision auto`,
`make_truncated_ckpt.py` copies them if the shard exists; otherwise it synthesizes the 263 tensors as bf16 zeros (norm
weights 1), with shapes from `vision_config` and DeepInfer's `vit.py` / `mlp_projector.py`. A text-only prompt never runs
the ViT. Do not send images through a checkpoint with synthesized vision weights. The reference drops these tensors, along
with `image_*` and `ffn.gate.bias_vl`; DeepInfer needs `bias_vl` and `image_*`, so both stay in the truncated checkpoint.

## Reference runner details

1. **No torchrun, no `init_process_group`.** `Transformer.__init__` uses `world_size = 1` when distributed is not
   initialized, and every collective is guarded by `world_size > 1`. Setup mirrors `generate.py`: `set_device`,
   expandable segments, default dtype bf16, seed, build under `torch.device("cuda")`, then `set_default_device("cuda")`
   after loading (required because `get_window_topk_idxs` allocates without a device).
2. **ModelArgs** are built from `inference/config.json` as a template, overlaid with values derived from the truncated HF
   config (`hf_to_model_args`). Field mapping: `hidden_size->dim`, `moe_intermediate_size->moe_inter_dim`,
   `num_experts_per_tok->n_activated_experts`, `scoring_func->score_func`, `routed_scaling_factor->route_scale`,
   `qk_rope_head_dim->rope_head_dim`, `rms_norm_eps->norm_eps`, `sliding_window->window_size`,
   `*_source_layer_ids->*_source_layers`, `candidate_source_layer_id->candidate_source_layer` (None becomes -1),
   `rope_scaling.{original_max_position_embeddings,factor,beta_fast,beta_slow}`, `engram_pad_token_id->engram_pad_id`,
   `dspark_num_experts_per_tok->dspark_n_activated_experts`. Overrides: `n_mtp_layers = 0`, `dspark_block_size = 0`,
   `vision_n_layers = 0`, `max_batch_size = 1`, `temperature = 0` (argmax). Non-layer fields that differ between the
   template and the HF config are saved in `meta.config_template_mismatches`. The runner asserts that the HF
   `quantization_config` matches the reference's hard-coded block sizes and scale format (32 / ue8m0).
3. **Missing module.** `model.py` imports `image_processor`, which is not in the published code; `ref_kernels.install`
   stubs it (`TEXT=-1`, `IMAGE*=0..3`), used only by `merge_image_embeddings`, which is never reached. If sympy is
   missing, a stub `isprime` is installed. `model.py`, `engram.py` and `vision.py` are imported unmodified.
4. **Loading.** Strict: every `model.state_dict()` key must be present, and any extra key except
   vision/aligner/image_*/mtp/bias_vl raises. Same-dtype tensors are copied as raw bytes; bf16 -> fp32 conversions happen
   for the head and the ratio-2 compressor, as `load_state_dict` would do. FP4 experts are stored by `ref_convert.py` as
   safetensors `I8` (the HF bytes unchanged) and viewed as `float4_e2m1fn_x2`, as `convert.py` does. `wo_a` is dequantized
   to bf16 as in `convert.py`. The resulting file is therefore not drop-in for the official `generate.py`, but it is
   byte-equivalent.
5. **Prompt.** `encoding.encode_messages([{role: user, content: passage + question}], thinking_mode="chat")` already
   includes the BOS text, so the runner tokenizes with `add_special_tokens=False` to get exactly one BOS (the official
   `generate.py` uses `AutoTokenizer.encode`, which may add a second one). The ids are saved to `/work/prompt_ids.json`;
   feed these to DeepInfer directly instead of re-tokenizing, because DeepInfer's tokenizer inserts BOS for plain text.
   If `encoding.py` cannot be imported, the runner falls back to raw `BOS + text` and logs a warning.
6. **Prefill and decode** use the official `Transformer.forward`; for prefill, `head.forward` is temporarily wrapped
   with `full_logits=True`. Decode feeds one token at `start_pos = L + s`. Generation does not stop at EOS. Greedy =
   argmax of the fp32 logits (first index on ties).
7. **Torch kernels** (`ref_kernels.py`) mirror `kernel.py` step by step: `act_quant` (fp32 amax with a 1e-4 floor,
   power-of-two scale via the same exponent/mantissa trick, RNE cast to e4m3, E8M0 storage); `fp4_act_quant` (E8M0 and
   E4M3 scale paths, e2m1 with RNE ties-to-even); `fp8_gemm` / `fp4_gemm` (dequantize, then fp32 GEMM with TF32 off;
   products are exact and scales are powers of two, so only summation order differs); `sparse_attn` (gather, invalid
   slots -inf, online softmax over blocks of 64 in index order from a -1e30 max, P cast to bf16 before P*V, sink only in
   the fp32 denominator); `hc_split_sinkhorn` (same operation order). Remaining differences are ULP-level;
   `kernel_selftest.py` and `REF_KERNELS=check` quantify them.

## Layer-index-dependent logic (why layers are renumbered)

DeepInfer (`deepinfer/`):

- `hugging_face/config.py`: `_source_map` builds `layer_index_module_map`, `layer_kv_map`, `layer_index_k_map` and
  `layer_candidate_source_map`; `compress_ratios` length must equal `n_layer + nextn`; `n_prefill_layer` is the first
  ratio-1 layer (StopIteration if none); `dspark_*` keys are read unconditionally; `engram_layer_ids`.
- `v4_modules/builder.py`, `v4_modules/cross_layer.py`: per-layer kv/index source flags and shared index-K modules;
  share-kv must not cross the YOCO boundary; the candidate source must be an index source with `index_topk > 0` and
  ratio > 0.
- `model_metadata_utils.py`, `v4_modules/hc_blocks.py`, `worker/cache_engine.py`: YOCO enablement; SWA caches of layers
  `>= n_prefill_layer` are treated as dropped.
- `hugging_face/names.py`: HF `layers.L` maps to internal `module.module.{L+1}`, head and norm to
  `module.module.{n_layer+1}`, MTP to `{n_layer+2+i}`. Renumbering to 0..N-1 is therefore required.
- `hugging_face/weights.py`: a layer is only covered if all `n_routed_experts` experts are present (hence the
  `n_routed_experts` rewrite with `--num-experts`).

Reference (`inference/model.py`, `engram.py`): per-layer `compress_ratios`, kv/index source lists, candidate source,
`get_moe_config` (`layer_id < n_layers` means backbone), Indexer `owns_k` / `is_candidate_source` /
`uses_candidates = 0 <= cand < layer_id`, the RoPE choice per ratio, engram per layer, MTP construction and target
layers. Engram hash multipliers are seeded per layer id (`10007 * layer_id`), so engram layers must keep their original
index.

## Caveats

- **Vision placeholders.** Synthesized ViT/aligner weights are harmless for text-only runs only.
- **Engram removal.** In the full model, engram at layer 1 feeds layer 2's input. A truncated model without it is a
  different function from any prefix of the full model; it is only a consistency target between the two stacks.
- **YOCO in DeepInfer.** Keep `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=off` (pinned by `tools/run_deepinfer.py`);
  otherwise DeepInfer drops prefill tokens for consumer layers and the comparison breaks.
- **Candidate semantics.** DeepInfer's candidate map includes the source layer itself; the reference only masks layers
  strictly after the source. On the 3-layer model at about 330 tokens every block is selected, so this has no effect.
- **Tilelang 0.1.15 vs the reference's pinned 0.1.12.** If `kernel.py` fails to import, every function falls back to
  torch (logged); per-function compile errors fall back individually. A CUDA fault inside a tilelang kernel poisons the
  context and cannot be caught; rerun with `REF_KERNELS_TORCH=<fn>`. A silently miscomputing kernel is only caught by
  `REF_KERNELS=check` or `kernel_selftest.py`.
- **Tie-breaking.** Greedy argmax ties and `topk(sorted=False)` inside the indexer can legitimately differ between
  implementations. Compare logits and top-k, not just token ids.
- **Shape checks.** `make_truncated_ckpt.py` uses a static model of both loaders' expected shapes
  (`layer_tensor_specs`); if it disagrees with the real checkpoint, the run stops. `--no-shape-check` turns mismatches
  into warnings; a mismatch means one of the script's assumptions about the checkpoint is wrong.
