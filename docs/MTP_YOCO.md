# YOCO cross-layer sharing, truncated test checkpoints and MTP/DSpark

DeepSeek-V4.1-Flash shares compressed KV, index keys, top-k selections and candidate blocks across layers ("YOCO").
A 3-layer truncation (original layers 0, 2, 20) never has a layer that *consumes* another layer's KV, top-k or
candidates. The **YOCO test model** is a larger truncation that contains every layer kind, so both DeepInfer (on
SM120) and the DeepSeek reference exercise all sharing paths on one 32 GB GPU. This document describes how the
sharing works in both stacks, how the truncated checkpoints are built (`tools/ref/make_truncated_ckpt.py`), and the
MTP/DSpark test checkpoints. Validation results are in [VALIDATION.md](VALIDATION.md).

## 1. How the sharing works

V4.1 config: `compress_ratios` = `[0,0, 2x18, 1x20, 0,0,0]` (40 backbone layers + 3 MTP), `kv_source_layer_ids [2,8,14,20]`,
`index_source_layer_ids [2,8,14,20,24,28,32,36]`, `candidate_source_layer_id 20`, `candidate_topk_blocks 2048`,
`candidate_block_size 8`, `index_topk 512`. There is no `index_k_source_layer_ids`; DeepInfer then uses the kv sources.

| concept | reference (`inference/model.py`) | DeepInfer |
|---|---|---|
| ratio 0 / 2 / 1 | 0 = sliding window only (layers 0, 1, MTP). r > 0 adds `index_topk` compressed positions, one latent per r tokens. RoPE is `compress_rope_theta` + YaRN if r > 0, plain `rope_theta` otherwise | `_win_spec` / `_compress_spec` (`hugging_face/config.py`), the same RoPE split |
| own SWA | every layer has its own `wkv`, `kv_norm` and window cache, fed by its own hidden | the same. Layers >= `n_prefill_layer` keep their SWA cache in the "dropped" state region (`cache_engine.py`), which only means it is not written to disk |
| kv source | owns `Compressor` and `compress_kv_cache`, publishes `shared_attn.compress_kv` | `layer_kv_map[l] == l`: owns `PackedWkv` and the FP4 main-KV `ca_cache` (288 B/token) |
| kv consumer (C2: 3-7, 9-13, 15-19; C1: 21-39) | no compressor, reads the nearest earlier source's cache | `ca_cache` aliased from `layer_kv_map[l]`; in prefill the source's loaded KV is reused (`CrossLayerRuntime.loaded_prefill_kv`) |
| index-K owner | `Indexer.owns_k = layer in kv_source_layers`: `wk`, `k_norm`, `k_cache`, published as `shared_attn.index_k` | `layer_index_k_map` (default = kv sources) owns `k_proj`, `k_norm` and the FP4 index cache `ica_cache` (68 B/token); other index sources share them |
| index source (2, 8, 14, 20, 24, 28, 32, 36) | own `Indexer` (`wq_b`, `weights_proj`): scores its own queries against the shared index K and publishes `shared_attn.topk_idxs` | `layer_index_module_map[l] == l`: an `index_attn` MQA with its own `q_B` and `weights_proj` |
| non-index compressed layer | reuses `topk_idxs` of the nearest earlier index source | reuses `CrossLayerRuntime.topk_indices` |
| candidate source 20 | after its own top-k scores, `select_candidate_blocks`: the best 2048 blocks of 8 compressed positions by max score, with the block holding the query's newest position pinned in. The source itself is not restricted | `is_candidate_topk_source`: `candidate_block_max` (bf16 block max, same rule) plus top-k gives `block_candidate_idxs` (`index_attn.py`, `tilelang_ops/candidate_blocks.py`) |
| candidate users (24, 28, 32, 36) | `uses_candidates = 0 <= cand < layer_id`: these index sources mask their scores to the candidates, then take their own top-k | `layer_candidate_source_map = {l: 20 for l >= 20}`; index sources after 20 compute sparse MQA logits on the candidate tokens only, then top-k and restore |
| ordering | layers run in order and each value has one slot, so the nearest earlier source wins implicitly | explicit maps validated at build time (`cross_layer.py`): contiguous segments, same ratio and `index_topk` as the source, chain index-topk ⊆ index-K ⊆ KV, share-KV must not cross `n_prefill_layer`, candidate consumers need capacity >= `index_topk` |
| YOCO | no special case | `n_prefill_layer` = first ratio-1 layer = 20; layers >= 20 are "YOCO consumers". With `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=off` (pinned by `tools/run_deepinfer.py`) they run all prefill queries, as the reference does. `mask`/`compact` only run the last `KEEP_TOKENS` queries in consumer layers, so per-position prefill output is not comparable |

What gets exercised depends on the prompt length L. A C1 layer selects all positions while L <= 512, a C2 layer while
L <= 1024 (`index_topk` 512). With the original `candidate_topk_blocks` 2048 x 8, candidate filtering only starts beyond
16384 compressed positions. The reference cannot prefill that on 32 GB (the indexer score tensor alone is [L, 32, L]
fp32 and there is no chunked prefill), so the YOCO test model lowers `candidate_topk_blocks` to 128. Filtering then
starts at L > 1024, and a ~2k-token prompt exercises top-k on every compressed layer and the candidate restriction on
layer 24. The default 327-token prompt exercises neither.

## 2. The YOCO test model (`--preset yoco`)

The preset uses original layers 0, 1, 2, 8, 14, 20, 21, 24 (plus head, embed). None of them is a natural C2 consumer,
so `--as-consumer 8` demotes layer 8 to a C2 consumer of layer 2: its compressor and indexer weights are dropped and it
reads layer 2's compressed KV and top-k. Its remaining weights are its own, so it acts as a real (if untrained)
consumer. Both stacks see the same model.

| new | orig | ratio | role in the truncated model | what it exercises |
|---|---|---|---|---|
| 0 | 0 | 0 | SWA | first SWA layer |
| 1 | 1 | 0 | SWA (engram layer in V4.1; engram off by default) | second SWA layer, engram hook point (`--keep-engram 1`) |
| 2 | 2 | 2 | C2 kv + index source | C2 compressor (ratio-2 pooling state), index-K owner |
| 3 | 8* | 2 | **C2 consumer** of 2 | aliased C2 `ca_cache`, reused C2 top-k |
| 4 | 14 | 2 | C2 kv + index source | a second C2 segment (segment switch) |
| 5 | 20 | 1 | C1 kv + index source + **candidate source**; `n_prefill_layer` | YOCO boundary, candidate production |
| 6 | 21 | 1 | **C1 consumer** (kv, idx, cand of 5) | YOCO consumer: aliased C1 KV, reused top-k |
| 7 | 24 | 1 | C1 **index source** without its own index K; **uses candidates** | shared `ica_cache` with its own q and weights; candidate-restricted top-k |

DeepInfer derives from the written config (checked with its own `load_hf_model_config` and `SharedLayerTopology`):
`kv_map [0,1,2,2,4,5,5,5]`, `idx_map [0,1,2,2,4,5,5,7]`, `idxk_map [0,1,2,2,4,5,5,5]`, `cand_map {5:5,6:5,7:5}`,
`n_prefill_layer 5`, YOCO consumers `[5,6,7]`. In the reference, consumers produce identical `compress_idx` to their
source, and the index source's top-k stays inside the candidate mask.

Not covered: a C1 consumer between two index sources (25-27) and a late layer (28-39) in this model; neither is a new
code path (25 behaves like 21, 28 like 24). The full-depth test model in [VALIDATION.md](VALIDATION.md) has all 40 layers.
If shard 00006 (layer 3, a natural C2 consumer) is available, `--layers 0,1,2,3,8,14,20,21,24 --as-consumer "" --num-experts 96`
gives a 9-layer variant (20.0 / 20.6 GiB parameters, DeepInfer / reference).

## 3. Routed-expert subset (`--num-experts N`, `--expert-select first|stride`)

- The same original experts are kept in every backbone layer and renumbered 0..N-1
  (`truncation.json:expert_ids_new_to_orig`). `layers.L.ffn.experts.{e}.*` are renamed, and the router rows
  `ffn.gate.weight [E,dim]`, `ffn.gate.bias [E]` (= DeepInfer `e_score_correction_bias`) and `ffn.gate.bias_vl [E]` are
  sliced row by row. `n_routed_experts` is set to N. Both stacks then take top-6 over the same N scores, so the
  comparison stays exact.
- V4.1 has no hash routing (no `tid2eid`); the script refuses `num_hash_layers`, any `tid2eid` tensor, and any other
  per-layer tensor with a 384-row leading dimension it does not know how to slice.
- MTP layers keep their own `dspark_n_routed_experts=128` unless `--mtp-num-experts` slices them too.
- Constraints: tile_kernels `moe_topk_gate` (used because patch P3 disables the mega gate) needs
  `num_routed_experts % 4 == 0` and topk <= 32, so the script enforces `N % 4 == 0` and `6 <= N <= 384`. DeepInfer only
  needs `(N + extra) % ep_size == 0`. There is no hard-coded 384 in DeepInfer's Python, the shims or `sm120_moe`; b12x
  `fused_moe` takes `num_experts` from the weights.

## 4. Config remap (`truncate_config`)

- Layer-valued fields are mapped old -> new: `kv_source_layer_ids`, `index_source_layer_ids`,
  `index_k_source_layer_ids` (if present) and `candidate_source_layer_id`. Demoted layers are removed from the source
  lists.
- `compress_ratios` = kept backbone ratios + (with `--mtp K`) the first K MTP entries; `num_nextn_predict_layers = K`.
- Checked relationships: every kept compressed layer's effective kv / index (/ index-K) source must be kept, with the
  same ratio and no other source in between; no ratio-0 layer may follow the first kept source; at least one ratio-1
  layer. If the candidate source is dropped while a kept index source after it uses candidates, the script errors
  (`--allow-candidate-drop` overrides). Finally the whole DeepInfer topology is re-checked offline
  (`deepinfer_topology_check`, a replica of `config.py` + `cross_layer.py` + `index_attn.py`).
- `--candidate-topk-blocks` must be a multiple of 8, <= 4096, and x block size >= `index_topk`. DeepInfer needs
  `--max-model-len >= candidate_topk_blocks x block size` (16384 for the original config).
- `--layers` accepts lists and ranges (`0-39`).
- **Engram:** hash multipliers are seeded with `10007*layer_id` and bucket primes are drawn in `engram_layer_ids` order,
  so `--keep-engram 1` works only because original layer 1 stays at index 1 and `[1]` is a prefix of `[1,14]`. In the
  YOCO preset, layer 14 (new index 4) cannot keep its engram; its tensors in shard 00048 are named `layers.14.*` and a
  symlinked file cannot be renamed. Tables are about 94 GiB per layer; by default (`--engram-mode link`) the original
  shard is symlinked (`engram-model-00047-of-00048.safetensors -> /ckpt/...`) instead of copied. Without the flag,
  engram is dropped (the preset sets `--allow-engram-drop`). The full-depth model keeps both engram layers
  (`--layers 0-39 --keep-engram 1,14`). Reference side: `ref_convert.py` drops only the tables (`.engram.embed.*`), and
  `ref_runner.py` serves them through `ref_engram.py` (see [ENGRAM.md](ENGRAM.md)).
- **MTP** (`--mtp K --dspark-targets a,b,c`, **original** layer ids, kept and distinct, the same count as the original
  `[37,38,39]`): `compress_ratios` keeps the K trailing entries (`[0,0,0]`), all `mtp.{i}.*` tensors are copied
  (`--mtp-num-experts M` slices their experts and gate rows and rewrites `dspark_n_routed_experts`), and the other
  `dspark_*` keys are untouched. Each MTP layer costs about 2.4 GiB on the DeepInfer side. The reference never runs
  DSpark (`ref_runner` ignores MTP, `ref_convert` drops `mtp.*`).

## 5. Runners and comparison for N layers

- `tools/ref/ref_runner.py`: `--capture-layer` / `CAPTURE_LAYER` takes `0`, `3,5,6` or `all`; `--prompt-tokens N` builds a
  deterministic, non-repeating long prompt (passage + synthetic logbook lines + question). `meta.truncation` and
  `meta.capture_layers` are recorded.
- `tools/ref/ref_capture.py`: several layers can be tapped at once. Extra taps: `compress_idx` [L, k] (compressed
  positions attended, -1 masked; for consumers this is the source's result), `cand_mask` [L, C] (candidate source
  only), and source flags in `meta`.
- `tools/run_deepinfer.py`: installs the shim's `install_layer_captures` on every requested layer and reopens the capture
  window per layer. `sm120/sm120_attn/capture.py` records DeepInfer's `compress_idx` (from the index list passed to the
  prefill attention, so YOCO consumers report the top-k they reuse) and `cand_mask` (from `deep_select.topk` when called
  by the candidate-block selection).
- `tools/compare.py` annotates each `layer{i}` row with its original id and role, e.g. `o21 C1 kv<5 idx<5 yoco`,
  `o8* C2 kv<2 idx<2` (* = demoted).
- `tools/compare_l0.py`: `--layer 0 | 3,5,6 | all`, one section per layer plus a summary, and set comparisons for
  `compress_idx` and `cand_mask` (identical rows and mean overlap).

## 6. Memory (RTX 5090, 31.4 GiB usable)

Parameter bytes are computed by `make_truncated_ckpt.py` from the planned tensors, printed with `--dry-run` and stored
in `truncation.json`.

| config | DeepInfer params | reference params |
|---|---|---|
| 3 layers (0, 2, 20; 384 experts) | 24.0 GiB | 24.5 GiB |
| **8 layers, 128 experts (`--preset yoco`)** | **22.6 GiB** (routed 17.9) | **23.2 GiB** (fp32 head +1.2, no vision -0.9) |
| 8 layers, 96 experts | 18.1 GiB | 18.7 GiB |
| 8 layers, 192 experts | 31.6 GiB | 32.2 GiB, does not fit |
| preset + `--mtp 3` | 29.9 GiB, does not fit for DeepInfer | 23.2 GiB |
| preset + `--mtp 3 --num-experts 64` | 20.9 GiB | 14.2 GiB |
| 0,2,20 + `--mtp 3 --num-experts 128` | 17.9 GiB | 11.0 GiB |
| 40 layers, 16 experts, MTP 3 with 32 experts, engram 1+14 (full-depth model) | 23.2 GiB | 22.5 GiB |

Per routed expert per layer: 18.9 MB (w1/w2/w3 FP4 + UE8M0 scales). Per layer without routed experts: about 0.17 GiB.

KV is negligible at these lengths (about 91 KB per 128-token block for the YOCO model, 114 KB for 40 layers).
Activations at L of about 2.1k: full-prefill logits [L, 129280] fp32 = 1.1 GB on both sides; indexer score [L, 32, L]
fp32 = 0.56 GB plus temporaries. Measured peaks at 2081 tokens: reference 26.0 GiB, DeepInfer (b12x) 26.5 GiB. Fine taps
cost about 0.7 GB (reference) and 1 GB (DeepInfer) per tapped layer at that length, so prefer a few layers over `all`.

## 7. Commands (inside the container)

Mounts: `/ckpt` = raw HF shards (ro), `/work` = work dir, `/ds41ref` = the reference `inference/` code (`REF_SRC`).

```bash
# 0. build: validate first (no writes), then ~22.6 GiB written + ~21.9 GiB reference format
python3 /opt/sm120/tools/ref/make_truncated_ckpt.py --src /ckpt --dst /work/ckpt_yoco --preset yoco --dry-run
python3 /opt/sm120/tools/ref/make_truncated_ckpt.py --src /ckpt --dst /work/ckpt_yoco --preset yoco
python3 /opt/sm120/tools/ref/ref_convert.py --hf-ckpt-path /work/ckpt_yoco --save-path /work/ref-mp1-yoco

# 1. the YOCO test: ~2.1k-token prompt (top-k active everywhere, candidates active on new layer 7);
#    reference with tilelang and torch kernels (noise floor); the torch run reuses the saved ids
cd /opt/sm120/tools/ref
CAPTURE_LAYER=3,5,6,7 REF_KERNELS=tilelang python3 ref_runner.py --ckpt /work/ref-mp1-yoco \
  --hf-config /work/ckpt_yoco/config.json --ref-src /ds41ref/inference \
  --prompt-tokens 2048 --max-seq-len 4096 --save-prompt-json /work/prompt_ids_yoco.json --out /work/yoco_ref.pt
CAPTURE_LAYER=3,5,6,7 REF_KERNELS=torch python3 ref_runner.py --ckpt /work/ref-mp1-yoco \
  --hf-config /work/ckpt_yoco/config.json --ref-src /ds41ref/inference \
  --prompt-ids-json /work/prompt_ids_yoco.json --save-prompt-json '' --max-seq-len 4096 --out /work/yoco_ref_torch.pt
#    (same via the script: CKPT=/work/ckpt_yoco REFMP=/work/ref-mp1-yoco OUT=/work/yoco_ref \
#     PROMPT_JSON=/work/prompt_ids_yoco.json RUN_ARGS="--prompt-tokens 2048 --max-seq-len 4096" \
#     REF_SRC=/ds41ref/inference CAPTURE_LAYER=3,5,6,7 ./run_ref.sh all)

# 2. DeepInfer: default backends (b12x), then the exact torch shims (isolates a b12x mapping problem)
CAPTURE_LAYER=3,5,6,7 python3 /opt/sm120/tools/run_deepinfer.py --ckpt /work/ckpt_yoco \
  --prompt-ids /work/prompt_ids_yoco.json --ref /work/yoco_ref.pt --out /work/yoco_di.pt \
  --max-model-len 16384 --max-num-batched-tokens 4096
CAPTURE_LAYER=3,5,6,7 SM120_GEMM_IMPL=torch SM120_MOE_IMPL=torch SM120_ATTN_IMPL=torch \
  python3 /opt/sm120/tools/run_deepinfer.py --ckpt /work/ckpt_yoco --prompt-ids /work/prompt_ids_yoco.json \
  --ref /work/yoco_ref.pt --out /work/yoco_di_torch.pt --max-model-len 16384 --max-num-batched-tokens 4096

# 3. compare: per-layer hidden error vs the ref-vs-ref2 noise floor, then the per-layer breakdown
python3 /opt/sm120/tools/compare.py --di /work/yoco_di.pt --ref /work/yoco_ref.pt --ref2 /work/yoco_ref_torch.pt
python3 /opt/sm120/tools/compare_l0.py --layer all --di /work/yoco_di.pt --ref /work/yoco_ref.pt --ref2 /work/yoco_ref_torch.pt

# engram variant (layer 1 keeps engram; layer 14's engram is dropped because it would move to index 4)
python3 /opt/sm120/tools/ref/make_truncated_ckpt.py --src /ckpt --dst /work/ckpt_yoco_engram --preset yoco --keep-engram 1
python3 /opt/sm120/tools/ref/ref_convert.py --hf-ckpt-path /work/ckpt_yoco_engram --save-path /work/ref-mp1-yoco-engram
REF_KERNELS=tilelang python3 /opt/sm120/tools/ref/ref_runner.py --ckpt /work/ref-mp1-yoco-engram \
  --hf-config /work/ckpt_yoco_engram/config.json --ref-src /ds41ref/inference \
  --prompt-ids-json /work/prompt_ids_yoco.json --save-prompt-json '' --max-seq-len 4096 --out /work/yoco_engram_ref.pt
```

The same flow through `tests/e2e_matrix.sh` (several env variants, each fed through `compare.py` and, with
`TEST_L0=1`, `compare_l0.py`):

```bash
TEST_CKPT=/work/ckpt_yoco TEST_PROMPT=/work/prompt_ids_yoco.json TEST_REF=/work/yoco_ref.pt TEST_REF2=/work/yoco_ref_torch.pt \
TEST_CAPTURE=3,5,6,7 TEST_L0=1 TEST_ARGS="--max-model-len 16384 --max-num-batched-tokens 4096 --ref /work/yoco_ref.pt" \
  bash /opt/sm120/tests/e2e_matrix.sh "eager DEEPINFER_ENFORCE_EAGER=1" "attntorch DEEPINFER_ENFORCE_EAGER=1 SM120_ATTN_IMPL=torch" \
    "graph DEEPINFER_ENFORCE_EAGER=0 DEEPINFER_DFLASH_ENFORCE_EAGER=0"
```

## 8. MTP / DSpark test checkpoints

```bash
# 3-layer backbone (original 0,2,20) + mtp.0-2, DSpark targets = all three kept layers (new [0,1,2])
python3 /opt/sm120/tools/ref/make_truncated_ckpt.py --src /ckpt --dst /work/ckpt3_mtp --layers 0,2,20 \
  --mtp 3 --dspark-targets 0,2,20 --num-experts 128            # 17.9 GiB; with 384 experts 31.3 GiB does not fit
# YOCO backbone + mtp.0-2, DSpark targets = original 20,21,24 (new [5,6,7])
python3 /opt/sm120/tools/ref/make_truncated_ckpt.py --src /ckpt --dst /work/ckpt_yoco_mtp --preset yoco \
  --mtp 3 --dspark-targets 20,21,24 --num-experts 64             # 20.9 GiB
# run with nextn 0/1/5 (add --graphs for CUDA graphs, --force-topk K for DEEPINFER_SAMPLER_FORCE_TOPK)
for n in 0 1 5; do
  python3 /opt/sm120/tools/run_deepinfer.py --ckpt /work/ckpt3_mtp --prompt-ids /work/prompt_ids.json --no-hidden --nextn $n \
    --out /work/mtp_n$n.pt
done
bash /opt/sm120/tests/mtp.sh       # nextn 0 / 5, eager / graphs, and nextn 5 with torch attention; greedy tokens compared
```

- `--nextn N` sets `DEEPINFER_USE_NEXTN_DECODE` and `lookahead_tokens=min(1, N)`. A decode step then carries 1+N logit
  rows; `decode_logits_all` keeps one row per emitted token, and `meta.mtp` holds `rows_per_step` (= 1 + nextn), the
  accepted drafts per step (`num_accepted_tokens_gpu`), tokens per step and draft acceptance.
- Expected on these checkpoints: identical greedy tokens for every nextn; on `ckpt3_mtp` the `decode_logits_all` rows
  match the nextn=0 run (min cos 0.991).
- The DSpark heads read features of layers they were not trained on, so acceptance on these checkpoints is meaningless
  (it is 0); only the plumbing is tested. The full-depth test model keeps the real targets 37-39 and gives non-zero
  acceptance ([VALIDATION.md](VALIDATION.md)).
- Both MTP backbones differ from `ckpt3` / `ckpt_yoco` in expert count, so a backbone logit comparison needs its own
  reference run (which ignores the MTP layers).

## 9. Caveats

- DeepInfer scores candidate blocks in bf16 (`candidate_block_max` on bf16 logits) while the reference uses fp32 scores.
  Near-tied blocks can flip, and top-k ties at the 512 boundary too. Judge `compress_idx` and `cand_mask` by mean
  overlap against the ref-vs-ref2 line, not by exact rows.
- The demoted layer 8 is untrained as a consumer; activations can be atypical, but the comparison remains valid.
- Fewer experts change routing (top-6 of 128); clamp and SwiGLU statistics differ from the full model. Fine for a
  consistency test, not for quality.
- The lowered `candidate_topk_blocks` (128 instead of 2048) is a config deviation that exercises the same code paths at
  a different capacity. The original 2048 is only reachable with prompts > 16k, which the reference cannot prefill on
  32 GB (DeepInfer itself is tested at long context with chunked prefill, see [ATTENTION.md](ATTENTION.md)).
- Symlinked engram shards must be visible at the same absolute path (`/ckpt/...`) wherever the checkpoint is loaded.
