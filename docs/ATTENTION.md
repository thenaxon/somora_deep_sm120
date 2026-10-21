# Attention, indexer and top-k on SM120

DeepInfer calls three SM100-only kernel libraries for attention: FlashMLA (sparse MLA prefill/decode and the fused
norm/RoPE/attention/cast kernel), DeepSelect (top-k) and the DeepGEMM indexer functions (MQA logits). The packages
under `sm120/` are drop-in replacements with the same Python API, backed by b12x 1.5.1, Triton kernels and exact
pure-torch fp32 references. `sm120/` must be first on `PYTHONPATH`. Paths below are relative to the DeepInfer
checkout unless they start with `sm120/` or `tests/`.

All numbers in this document were measured on one RTX 5090 with truncated test checkpoints (see
[VALIDATION.md](VALIDATION.md)); they are not results for the full model.

| Package / file | What it provides |
|---|---|
| `sm120/flash_mla/` | `flash_mla_interface.{FlashMLASchedMeta, get_mla_metadata, flash_mla_with_kvcache, flash_mla_sparse_fwd, flash_attn_varlen_*}`, `fused_norm_rope_attn_rope_cast.{prefill, decode, permute_q_b_proj, permute_wv_proj}` |
| `sm120/deep_select/` | `interface.{topk, get_stride_requirement}` |
| `sm120/sm120_attn/indexer.py` | `DEEP_GEMM_EXPORTS`: the 8 DeepGEMM indexer functions, re-exported by the `deep_gemm` shim |
| `sm120/sm120_attn/` | implementation: `_config` (env switches), `quant` (byte formats), `rope`, `sparse_attn` (torch reference + dispatcher), `b12x_mla` (b12x decode), `b12x_prefill` (b12x extend), `b12x_indexer`, `fused`, `fused_triton`, `indexer`, `topk`, `capture` |
| `sm120/sm120_mhc.py`, `sm120/sm120_einsum.py` | Triton kernels behind `deep_gemm.tf32_hc_prenorm_gemm` and `deep_gemm.fp8_einsum` |
| `sm120/sm120_lmhead.py` | optional FP8 lm_head (opt-in) |
| `tests/test_attn_shims.py`, `tests/test_mhc_einsum.py` | pytest suites, run on the GPU inside the container |

## 1. Switches

All switches are read on every call (`sm120/sm120_attn/_config.py`).

| env | default | meaning |
|---|---|---|
| `SM120_ATTN_IMPL` | `b12x` | `b12x` uses b12x/Triton where a mapping exists. `torch` uses the pure-torch fp32 reference everywhere (useful for bisecting). |
| `SM120_ATTN_DECODE_IMPL` | `$SM120_ATTN_IMPL` | sparse decode backend |
| `SM120_ATTN_PREFILL_IMPL` | `$SM120_ATTN_IMPL` | sparse prefill backend |
| `SM120_INDEXER_IMPL` | `$SM120_ATTN_IMPL` | MQA-logits backend |
| `SM120_ATTN_FUSED_IMPL` | `triton` (`torch` if `SM120_ATTN_IMPL=torch`) | fused q-prep and RoPE/FP8-cast kernels |
| `SM120_TOPK_IMPL` | `triton` (`torch` if `SM120_ATTN_IMPL=torch`) | top-k key kernel |
| `SM120_ATTN_COMPUTE` | `bf16` | b12x `v41_compute_mode`. `fp8` is faster but less accurate (cos about 0.99-0.997). |
| `SM120_ATTN_PERMUTE` | `1` | `1` reproduces FlashMLA's exact q_b_proj/wv_proj permutations and fused I/O layouts. `0` makes the permute functions identity and uses natural layouts. **Must be the same at weight load and at run time.** |
| `SM120_ATTN_STRICT` | `0` | `1` raises instead of falling back from b12x to torch |
| `SM120_ATTN_CHUNK_MB` | `512` | working-set budget of the torch paths |
| `SM120_ATTN_B12X_LSE` | `0` | ask b12x for the natural-log LSE (layout unverified). Otherwise the b12x paths return a NaN LSE. |
| `SM120_ATTN_TF32` | `0` | allow TF32 in the torch prefill matmuls |
| `SM120_ATTN_PREFILL_KV` | `auto` | how the bf16 prefill KV is re-encoded for b12x (section 4). `fp8` puts every token into the 528 B cache. |
| `SM120_ATTN_PREFILL_ROWS` | `2048` | query rows per b12x prefill launch (fixed plan capacity). Scratch is about rows x H x 1 KiB, 128 MiB at H=64. |
| `SM120_ATTN_PREFILL_LONELY_FIX` | `0` | `1` restores the full-output fix-up for queries without keys (b12x already writes 0 there) |
| `SM120_ATTN_DECODE_COMPACT` | `0` | `1` compacts decode index lists before b12x (not needed: b12x skips -1 entries) |
| `SM120_INDEXER_SCRATCH_MB` | `1024` | per-call scratch budget of the b12x indexer; rows are chunked to fit it |
| `SM120_INDEXER_FILL_TAIL` | `0` | `1`: b12x `fp8_fp4_mqa_logits` also writes -inf past the longest valid row (DeepGEMM leaves it uninitialized; DeepInfer always passes `end` to top-k) |
| `SM120_MHC_IMPL` | `triton` | `torch` keeps the torch loop for `tf32_hc_prenorm_gemm` |
| `SM120_EINSUM_IMPL` | `triton` | `torch` keeps the dequant + bmm path for `fp8_einsum` |
| `SM120_LMHEAD_IMPL` | `bf16` | `fp8` enables the FP8 lm_head (section 9) |
| `SM120_ATTN_DEBUG` | `0` | print each dispatch decision once, plus plan lists and prefill KV class counts. Fallback warnings are always printed. |

`SM120_ATTN_STRICT`, `SM120_ATTN_COMPUTE` and `SM120_ATTN_B12X_LSE` apply to every b12x path. A b12x error or an
unsupported operand prints one warning and runs the torch path; with `STRICT=1` it raises instead.

## 2. What runs where

| DeepInfer call | default backend | fallback / `=torch` |
|---|---|---|
| `flash_mla_with_kvcache`, `fused...decode` | b12x `compressed_sparse_mla`, mode `decode` | `sparse_decode_torch` |
| `flash_mla_sparse_fwd`, `fused...prefill` | b12x `compressed_sparse_mla`, mode `extend` (`sm120_attn/b12x_prefill.py`) | `sparse_prefill_torch` |
| `fp8_fp4_paged_mqa_logits` (decode, dense) | b12x `dsa_indexer` MXFP4 `score`, page table = `block_table` | torch (capture-safe) |
| `fp8_fp4_paged_sparse_mqa_logits` (decode, candidate blocks) | b12x MXFP4 `score`, candidates mode | torch |
| `fp8_fp4_mqa_logits` (prefill, contiguous K) | b12x MXFP4 `score`, candidates = `ks + column`, over a page pool built from the contiguous K | torch |
| `fp8_fp4_sparse_mqa_logits` (prefill, candidate blocks) | b12x MXFP4 `score`, candidates mode, over the same pool | torch |
| `deep_select.topk` | Triton key kernel + `torch.topk` | `torch.topk` |
| fused q unpermute / RMS / RoPE, and inverse RoPE / FP8 per-32 cast | two Triton kernels (`sm120_attn/fused_triton.py`) | torch |
| `deep_gemm.tf32_hc_prenorm_gemm` (mHC) | Triton (`sm120_mhc.py`) | torch loop |
| `deep_gemm.fp8_einsum` (wo_a) | Triton (`sm120_einsum.py`) | dequant + bf16 bmm |

## 3. Call sites in DeepInfer

| Call site | Calls |
|---|---|
| `deepinfer/model_executor/layers/attention_backend.py:2-3` | imports `flash_mla_sparse_fwd, flash_mla_with_kvcache, fused_norm_rope_attn_rope_cast, FlashMLASchedMeta` |
| `attention_backend.py:51` | `fused_norm_rope_attn_rope_cast.prefill(q=, kv=, indices=index_topk.unsqueeze(1), sm_scale=, attn_sink=, topk_length=, **fused_kwargs)` |
| `attention_backend.py:64` | `flash_mla_sparse_fwd(query, kv, index_topk.unsqueeze(1), sm_scale=, attn_sink=, topk_length=)`; heads are zero-padded to 64/128 first |
| `attention_backend.py:108` | `fused_norm_rope_attn_rope_cast.decode(q=, k_cache=, indices_in_kvcache=, sm_scale=, d_v=, attn_sink=, topk_length=, extra_k_cache=, extra_indices_in_kvcache=, extra_topk_length=, **fused_kwargs)` |
| `attention_backend.py:125` | `flash_mla_with_kvcache(q=[T,1,H,512], k_cache=, block_table=None, cache_seqlens=None, head_dim_v=, tile_scheduler_metadata=sched_meta, softmax_scale=, causal=False, is_fp8_kvcache=True, indices=[T,1,topk], attn_sink=, extra_k_cache=, extra_indices_in_kvcache=[T,1,etopk], topk_length=, extra_topk_length=, enable_batch_invariant=)`, then `out.squeeze(1)`, `lse.squeeze(-1)` |
| `models/v4_modules/attn.py:87` | sets `use_fused_norm_rope_attn_rope_cast` (true for H in {64,128} on GPU, except dense packed-KV decode) |
| `attn.py:114` | `permute_q_b_proj((weight[:H*D], sf[:H*D]), H, D)`, copied back in place **at weight load** |
| `attn.py:143` | `permute_wv_proj((wv_weight [G,N,gs*D], wv_sf), wv_group_size=H//G, d_o=D)`, **at weight load**; `wv_sf` comes from `deep_gemm.transform_sf_into_required_layout(..., recipe=(1,1,32))` |
| `attn.py:174` | fused kwargs: `enable_q_norm, rms_norm_eps, token_positions, is_rope_neox_style, rope_dim, cos_sin_cache, n_wv_group, num_per_channels=32, use_tma_aligned_col_major_sf=True, round_sf=True, use_packed_ue8m0=True` |
| `attn.py:617`, `:660`, `:994` | prefill, decode and DFlash decode; all discard the LSE (`context, _ =`) |
| `attn.py:215` | `(out_fp8, out_sf)` from the fused functions goes straight into `deep_gemm.fp8_einsum("bhr,hdr->bhd", ..., recipe=(1,1,32))` |
| `models/v4_modules/index_attn.py:273` | `deep_gemm.get_sparse_mqa_logits_metadata(ks, ke, num_kv_tokens, abs_candidate_blocks, q_dtype, cbs, use_unaligned_ks=True)` |
| `index_attn.py:299` | `deep_gemm.fp8_fp4_mqa_logits((q_fp4,q_sf), (k_fp4,k_sf), weights, ks, ke, max_seqlen_k=max_kv_len)` |
| `index_attn.py:317` | `deep_gemm.fp8_fp4_sparse_mqa_logits(q, kv, weights, meta, nblk, cbs, use_unaligned_ks=True)` |
| `index_attn.py:392` | `deep_gemm.get_paged_mqa_logits_metadata(klen.unsqueeze(1), page, deep_gemm.get_num_sms(), indices=query_indices)` |
| `index_attn.py:410` | `deep_gemm.get_paged_sparse_mqa_logits_metadata(klen, block_tables, query_indices, page, cand_blocks, q_dtype, cbs)` |
| `index_attn.py:434` | `deep_gemm.fp8_fp4_paged_mqa_logits((q.unsqueeze(1), sf.unsqueeze(1)), ica_cache, weights, klen.unsqueeze(1), block_tables, meta, max_kv_len, indices=query_indices)` |
| `index_attn.py:447` | `deep_gemm.fp8_fp4_paged_sparse_mqa_logits(q, ica_cache, weights, meta, nblk, cbs)` |
| `index_attn.py:216`, `:361`, `:494` | `deep_select.topk(..., sorted=False, end=, indices_type=int32, sorted_index=True, output_idx=, output_idx_offset=, idx_oob_fill_value=-1, return_value=False)` |
| `layers/sampler.py:643` | `topk(logprobs, k, True, indices_type=int32)` (sorted values) |
| `layers/sampler_utils.py:68`, `:167` | `topk(logits, k, False, indices_type=int32)` |
| `input_metadata/input_metadata.py:9,389`, `input_metadata/dflash.py:18,83` | `defaultdict(FlashMLASchedMeta)`; a zero-argument constructor is required |

Vendored originals used as the specification:

- `vendor/FlashMLA/flash_mla/flash_mla_interface.py`, `vendor/FlashMLA/flash_mla/fused_norm_rope_attn_rope_cast.py`
- `vendor/FlashMLA/csrc/cuda_kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/{permute_q_b_proj,permute_wv_proj}/kernel.cu`
- `vendor/FlashMLA/csrc/api/fused_norm_rope_attn_rope_cast_fwd.cpp` (output allocation and stride checks)
- `vendor/FlashMLA/tests/{ref.py,quant.py,test-fused-norm-rope-attn-rope-cast.py}`
- `vendor/DeepGEMM/csrc/apis/attention.hpp`, `vendor/DeepGEMM/tests/test_attention.py`
- `vendor/DeepSelect/deep_select/interface.py`, `vendor/DeepSelect/tests/test.py`

## 4. Function semantics and mapping

### `FlashMLASchedMeta`, `get_mla_metadata`

Same dataclass and fields. `flash_mla_with_kvcache` reproduces the first-call init and the later consistency asserts
on `config`, so reusing one meta with a different top-k, batch or page size still asserts like the original.
`tile_scheduler_metadata` and `num_splits` stay `None`; nothing reads them. `get_mla_metadata()` returns
`(FlashMLASchedMeta(), None)`.

### `flash_mla_with_kvcache` (sparse decode)

- Inputs: `q` `[b, s_q, h, 512]` bf16; `k_cache` `[blocks, page, 1, B]` with B = 528 (V4.1 FP8) or 288 (V4.1 FP4),
  any block stride (the caches are regions of multi-layer block buffers); `indices` `[b, s_q, topk]`, each entry
  `block*page + off`, -1 = invalid; `topk_length` `[b]`; optional `extra_*` with the same formats.
- Semantics: gather the selected tokens, dequantize exactly, fp32 softmax over main + extra, apply the sink factor
  `1/(1+exp(sink-lse))`. A query with no valid token gets output 0 and LSE +inf.
- Output: `out` `[b, s_q, h, d_v]` bf16, `lse` `[b, h, s_q]` fp32 without the sink.
- b12x mapping: `b12x.attention.compressed_sparse_mla` with `cache_format="deepseek_v41"`, mode `decode`, when the main
  cache is 528 B, the extra cache absent or 288 B, H a multiple of 16 and d = d_v = 512. Physical slots and 2-D page
  views are used directly (the KV cache bytes are compatible, see `tests/test_attn_b12x_compat.py`). Index lists are
  passed uncompacted (b12x skips -1 entries). Plans are cached per (H, row bucket = next power of two >= 8, widths,
  page sizes, cache shape/stride/alignment, sink, compute); the first call per shape compiles. Every other case runs
  the torch path (`sparse_attn.sparse_decode_torch`, chunked over rows) with a warning.

### `flash_mla_sparse_fwd` (sparse prefill)

- Inputs: `q` `[s_q, h, 512]` bf16, `kv` `[s_kv, 1, 512]` bf16, `indices` `[s_q, 1, topk]` (invalid if < 0 or
  >= s_kv), optional `topk_length`.
- Output `(out bf16, max_logits, lse)`: `max_logits` is -inf and `lse` +inf for a query with no valid token; `lse`
  excludes the sink.

b12x has no bf16-KV sparse kernel. Its prefill is the single-pass MG kernel reached through mode `extend`, which
reads paged V4.1 caches. DeepInfer instead passes an assembled bf16 buffer `[loaded SWA cache | current SWA kv |
loaded compressed cache]` (`attn.py:813-857`). Two facts make an exact mapping possible:

- **SWA tokens are FP8-exact.** The current-chunk kv goes through `CastArgs("e4m3", ue8m0, cast_back=True)`
  (`attn.py:412-421`), and cached SWA tokens are dequantized 528 B records.
- **Compressed tokens are FP4-exact.** They are dequantized 288 B records.

On every call the shim therefore:

1. re-encodes the buffer into two transient caches with page 64 and slot = kv row: a 528 B FP8 "swa" cache and a
   288 B FP4 "indexed" cache (one Triton pass, `fused_triton.encode_v41`, bit-identical to `quant.quantize_v41_*`;
   the page count is grow-only and a power of two because the cache shape is part of the plan identity);
2. classifies every token: "FP4" if its FP4 round trip is bit-exact, otherwise "FP8";
3. splits every query's index list by class and compacts each part to a prefix with per-row lengths;
4. runs b12x with main = FP8 list and extra = FP4 list (a single-cache plan if no FP4 token is selected).

So **for every token DeepInfer produces, b12x sees KV bit-identical to FlashMLA's bf16 contract.** A token that is
neither FP8- nor FP4-exact would be FP8-quantized; that does not happen in DeepInfer (`SM120_ATTN_DEBUG=1` prints the
counts).

Remaining deviations from FlashMLA:

- **Kernel math:** with `SM120_ATTN_COMPUTE=bf16` b12x uses BF16 QK and P·V, as FlashMLA's contract does. `fp8`
  uses FP8-internal math (cos about 0.99).
- **Fused prefill RMS factor** is folded into q in bf16 (FlashMLA applies it to the fp32 logits): about 2^-9
  relative error on q.
- **`max_logits` is NaN**, and `lse` too unless `SM120_ATTN_B12X_LSE=1`. Queries with no valid key still return
  out = 0, lse = +inf, max = -inf. DeepInfer discards both at every prefill call site.
- **Overhead:** one host sync per call (`.tolist()` of the two maximum list lengths, used to size the widths). Prefill
  is never graph-captured in DeepInfer. Under shared KV (YOCO) the same buffer is re-encoded once per consumer layer;
  caching it is possible but not done.
- **Plan count:** widths are bucketed to powers of two >= 64, the row capacity is fixed and the pages are grow-only,
  so a handful of compiles happen at the first prefills. Widths up to (1024, 1024) are tested; widths beyond 2048
  (dense layers at long context) have not been exercised.

### `fused_norm_rope_attn_rope_cast.prefill` / `.decode`

Unfused but equivalent to the reference in FlashMLA's `tests/test-fused-norm-rope-attn-rope-cast.py`:

1. undo the q_b_proj permutation;
2. RMS factor `r = rsqrt(mean(q^2)+eps)` from the pre-RoPE q (no weight, applied to the logits; on the b12x path
   folded into q in bf16);
3. q RoPE on the last `rope_dim` channels (interleaved pairs, or NeoX), fp32, rounded to bf16;
4. sparse attention core (same dispatchers as above);
5. inverse (conjugate) RoPE on the fp32 output;
6. per-32 FP8 e4m3 cast with ue8m0 scale `ceil_pow2(max(amax,1e-4)/448)`.

Steps 1-3 and 5-6 are two Triton kernels by default. Output layout: `out_fp8` `[s_q, G, gs*512]`, within a wv group
ordered `[(d//32), head_in_group, 32]`; `out_sf` int32 `[s_q, G, gs*4]` (4 packed ue8m0 bytes, stride(0) == 1, storage
`[G, gs*4, align4(s_q)]` transposed, as `allocate_scale_factor` builds it). Returns `prefill`: `(fp8, sf, max_logits
[s,h], lse [s,h])`, `decode`: `(fp8, sf, lse [s,h])`. Only `num_per_channels=32` and `tma/round/packed=True` are
implemented, which is all FlashMLA supports.

### `permute_q_b_proj((w, sf), h, d)` and `permute_wv_proj((w [G,N,gs*d], sf), gs, d)`

- q_b: row `head*d + dim` moves to `dim%16 + head*16 + (dim//16)*h*16`, the exact kernel formula. `w` and `sf` are
  permuted on dim 0; `sf` must be row-indexed (`sf.shape[0] == h*d`, e.g. DeepGEMM's `[N, K/128]` int32 packed). A
  block-scaled `[N/128, ...]` sf is rejected, because rows inside a block would be mixed.
- wv: the 32-element input chunk `head*(d/32)+c` moves to `c*gs+head`. `sf` may be int32 packed ue8m0 `[G,N,gs*d/128]`
  (byte j of int k = scale 4k+j) or unpacked per-32 `[G,N,gs*d/32]`; the returned `sf` keeps the input strides.
- With `SM120_ATTN_PERMUTE=0` both are identity.

The fused output layout and the permuted wv columns use the same ordering, so a standard `einsum("bhr,hdr->bhd")`
gives the same result as natural-layout output times natural wv (`test_weight_permutations_are_consistent`).

Weight-load contract for the GEMM shim: `q_b_proj.weight.sf` must be row-indexed on dim 0, which is what
`transform_sf_into_required_layout(sf, N, K, recipe=(1,32,32), is_sfa=False)` returns (`[N, K/128]` int32 MN-major).
The wv scales are `transform_sf_into_required_layout(wv_sf.repeat_interleave(32, dim=1), N, K, num_groups=G,
recipe=(1,1,32), is_sfa=False)`. `fp8_einsum` receives `ctx_fp8` `[T, G, gs*512]` e4m3 and `ctx_sf` int32 packed ue8m0
with strides `(1, K4*align4(T), align4(T))`, the same format `tile_kernels.quant.per_token_cast(...,
use_tma_aligned_col_major_sf=True, use_packed_ue8m0=True)` gives on the non-fused path: a plain per-32 block-scaled
GEMM along r.

### DeepGEMM indexer (`sm120_attn/indexer.py`, `sm120_attn/b12x_indexer.py`)

`logits = sum_h relu(q_h·k)·w_h`; output bf16 for bf16 weights, fp32 for fp32 weights.

- q: `(int8 packed e2m1 [..,D/2] | e4m3 [..,D], int32 packed ue8m0 gran-32 | None)`, even elements in the low nibble.
- Contiguous kv: `(fp [N,..], sf [N] int32)`. Paged index cache: `[pages, page, 1, D/2+4]` uint8 with a block-wide
  `[values | scales]` layout (68 B/token for MXFP4, D = 128).

| Function | Output (rows have a 1024 B-aligned stride) |
|---|---|
| `fp8_fp4_mqa_logits` | `[s_q, max_seqlen_k]`; column j is token `ks+j`, valid for `j < ke-ks` |
| `fp8_fp4_paged_mqa_logits` | `[B, max_context_len]`; column j is logical token j, valid for `j < ctx`; physical slot via `block_table[row]` |
| `fp8_fp4_sparse_mqa_logits` | `[s_q, nblk*sbk]`; column t is token `blocks[t//sbk]*sbk + (ks%sbk if unaligned) + t%sbk` |
| `fp8_fp4_paged_sparse_mqa_logits` | logical token `blocks[t//sbk]*sbk + t%sbk < ctx` |

For the sparse functions the first `min(ceil(len/sbk), nblk)` block slots are valid. The originals leave invalid
entries uninitialized; the shim writes -inf. The `*_metadata` functions return small tensors (dense) or dataclasses
carrying the arguments (sparse). The `deep_gemm` shim provides `get_num_sms()`, which `index_attn.py:394` passes back to
the metadata function.

b12x mapping: the b12x MXFP4 index page (`dsa_indexer/mxfp4.py`, `index_mxfp4_page_bytes`) is `[page, 64]` packed
e2m1 followed by `[page, 4]` ue8m0, byte-identical to DeepInfer's `FP4_INDEX_KV_FORMAT`, so `ica_cache` is used in
place through a 2-D page view. Only the staged `score()` is called (dense bf16 logits, -inf at invalid columns); the
fused `score + select` route fixes top-k at 512 and is not used because DeepInfer calls logits and top-k separately.

- `fp8_fp4_paged_mqa_logits`: no candidates; `cache_lengths = context_lens`, `page_table = block_table`,
  `score_width = align(max_context_len, 512)` (gives DeepGEMM's 1024 B row stride, returned without a copy).
- Sparse variants: DeepGEMM's token-per-column map (-1 = invalid) is computed in torch and passed as
  `candidate_indices`; b12x caps candidates at 16384, so wider rows are scored in column chunks.
- `fp8_fp4_mqa_logits`: the contiguous K is copied into a transient pool of 64-token pages; candidates =
  `ks + c0 + column` in chunks of 16384 columns. Chunks past the longest valid row are skipped (one host sync, only
  when `max_seqlen_k > 16384`).

Numerics: b12x rounds `q·k`, `relu(q·k)·w` and the head sum to bf16, mirroring the published V4.1 `model.py` scorer;
DeepGEMM accumulates in fp32 and rounds once. Expect about 1e-2 row-relative differences and occasional swaps of
near-tied top-k entries; the tests require > 97 % top-k overlap. The torch path is used when q is MXFP8 (e4m3), the
weights are fp32, there are more than 32 index heads or 32 % H != 0, or the pages are not 16 B aligned.

Memory: DeepInfer's dense paged indexer call uses `max_context_len = runtime_max_model_len // ratio`, so the scratch
per call is `rows_bucket x align(max_kv_len, 512) x 6 B`, about 50 MiB at batch 32 and 1 M context / 4. Graph capture
keeps those blocks in the graph pool; lower `SM120_INDEXER_SCRATCH_MB` if memory is tight.

### `deep_select.topk`

`torch.topk` plus the original contract: candidates `[0, end)`; the first `min(end,k)` slots are valid, the rest get
`idx_oob_fill_value` (without offset) and `value_oob_fill_value`; `output_idx_offset` is added to valid indices;
`sorted_index` gives ascending indices, `sorted` descending values; `output_idx` is written in place; rows have a 32 B
aligned stride.

A Triton key kernel (default) builds the masked key in one pass instead of the fp32 copy, isnan/isinf checks, `where`
and `end` mask. The key is **fp32**: with a bf16 key, `torch.topk`'s bf16 kernel breaks the indexer's many exact ties
(relu zeros, repeated tokens) differently from the fp32 reference path; at 64k context that changed decode logits to
cos 0.58 against the torch reference, with the fp32 key it is 0.975 (noise level). With `sorted_index`,
`torch.topk(sorted=False)` plus an index sort suffices because OOB columns always sort last, and no values gather is
done when `return_value=False`. NaN ranks lowest (the original aborts).

| shape | torch-only shim | Triton key + `torch.topk` |
|---|---|---|
| 4096 x 65536, k 512 | 13.4 ms | 2.8 ms |
| 2048 x 16384, k 512 | 1.9 ms | 0.52 ms |

Small shapes are bound by about 0.35 ms of host launch overhead, which graph replay removes. b12x 1.5.1's exact row
top-k (`dsa_indexer.tiled_topk.run_row_topk`, k in {512, 1024, 2048}) was measured and is not faster (0.33-1.65 ms vs
0.06-1.7 ms including its host overhead).

## 5. CUDA graphs (decode)

DeepInfer captures decode only (`deepinfer/worker/cuda_graph_runner.py`, `model_runner.py`): capture is on unless
`DEEPINFER_ENFORCE_EAGER=1`; DFlash/DSpark has its own runner and switch (`DEEPINFER_DFLASH_ENFORCE_EAGER`). For every
batch size the runner does one eager warm-up forward on the graph stream, then captures (largest size first) with a
shared pool. Input metadata is copied into static buffers before each replay.

What the shims do so that capture works:

- **Plans are created in the eager warm-up** (decode attention per row bucket, indexer per row capacity). Under
  capture `prepare_plan` raises before b12x would compile; the dispatcher then falls back to torch with a warning that
  the graph records the torch path (or raises with `STRICT=1`). A capture-time miss is not marked as permanently
  failed.
- **No host syncs on the decode path.** b12x decode reads lengths in the kernel; the torch paged indexer uses the
  static `max_context_len` while capturing; top-k is sync-free.
- **Persistent staging buffers per plan.** Padded rows get length 0 on every call, so a graph does not depend on the
  capture order.
- **Per-call allocations come from the graph pool** with fixed sizes per shape, so the logits of one layer never alias
  the next layer's call.
- **One shared decode scratch arena** (`b12x_mla._shared_scratch`, grow-only). Without it, MTP with nextn 5 ran out of
  memory during capture (verify batches up to 32 x 6 = 192 rows, each row bucket with its own split-KV scratch).
- Shapes (batch size), `max_context_len` and the cache tensors must stay constant between capture and replay;
  DeepInfer keeps them static. Prefill (attention and indexer) is never captured.

`test_decode_path_cuda_graph` captures fused decode + paged logits + top-k, replays after changing the inputs and
compares with eager (bitwise).

## 6. mHC prenorm GEMM and `fp8_einsum` (wo_a)

DeepInfer's mHC path (`v4_modules/mhc_ops.py`, `hc.py`) uses tile_kernels for expand, post, pre_apply_mix, sinkhorn,
split_mixes and reduce+RMSNorm (about 0.24 ms per decode step on the YOCO test model). The exception is
`deep_gemm.tf32_hc_prenorm_gemm(x [m, 4*5120] bf16, fn [24, 20480] fp32, d [16, m, 24], sqr_sum [16, m], 16)`. As a
torch loop over the 16 K-splits it was 256 tiny launches per decode step, about 2.7 ms/step at B=8 (about half of the
decode step on that model). Both functions now have Triton implementations, hooked at the top of the two functions in
`deep_gemm/_gemm.py`; failures fall back to torch with one log line.

- **`sm120_mhc.tf32_hc_prenorm_gemm`:** one launch computes all split-K partials plus the square sums. The dot is full
  fp32 FMA (`input_precision="ieee"`), more accurate than DeepGEMM's TF32. For few rows every split is cut into up to
  16 sub-splits (about 2x SMs CTAs) and summed in a fixed order (deterministic, no atomics). Graph replay: 16-18 us per
  call at m = 1-32 vs 1.34-1.44 ms for the torch loop; 0.28 vs 1.44 ms at m = 2048; 0.91 vs 1.73 ms at m = 8192.
  Accuracy against fp64: rtol 1e-5.
- **`sm120_einsum.fp8_einsum_bhr_hdr`** (`"bhr,hdr->bhd"`, `recipe=(1,1,32)`): batched Triton GEMM over (G, T tiles, N
  tiles). Both FP8 operands are dequantized in registers with their per-32 ue8m0 scales (e4m3·2^e is exact in bf16);
  bf16 dot with fp32 accumulation, rounded to bf16. The fused per-32 FP8 output cast writes the scale bytes into
  DeepInfer's TMA-aligned int32 tensor. The weight is read as FP8 from the parameter, so the bf16 dequant cache is not
  filled. Decode tiles use BN=32 for T <= 32. Graph replay: 25-35 us at T = 1-32 vs 0.52 ms torch; 0.88 vs 1.87 ms at
  T = 2048; 3.4 vs 7.8 ms at T = 8192. bf16 output cos > 0.99999 against the torch path and an independent fp32
  reference; FP8 output matches > 99 % of scale bytes.
- **Tests:** `tests/test_mhc_einsum.py` (17): prenorm vs fp64 for m in {1, 7, 33, 300} x splits {None, 16}; einsum vs
  torch and an independent reference for T in {1, 8, 32, 300} x {bf16, FP8 out}; CUDA-graph capture/replay. The tests
  set `SM120_DEQUANT_CACHE=none`, because the torch path's dequant cache is keyed by `data_ptr` and version and a fresh
  test tensor at a reused address would hit a stale entry (static model weights are unaffected).

b12x `norm.mhc` (a fully fused V4.1 residual op) could replace DeepInfer's tile_kernels mHC chain, but would need a
runtime patch of `hc.py` with exact semantic matching; the remaining mHC cost (about 0.39 ms/step on the YOCO test
model) limits the upside to 0.2-0.3 ms/step. Not done.

## 7. Selection capture (`sm120_attn/capture.py`)

For layer-by-layer comparison with the reference (`tools/compare_l0.py`, [MTP_YOCO.md](MTP_YOCO.md)), DeepInfer's
sparse selections can be recorded. Capture is active only while `sm120_capture` is armed, during the tapped layer's
attention in the first prompt forward; otherwise it costs one `sys.modules` lookup.

- **`L<i>/compress_idx`** comes from the index list passed to the prefill attention, so YOCO consumers report the
  top-k they reuse. The compressed region of the assembled KV is the trailing run of FP4-exact tokens; entries are
  relative to the first compressed latent, -1 = SWA, masked or invalid.
- **`L<i>/cand_mask`** is recorded in `deep_select.topk` when called from `index_attn._topk_candidate_block`; the
  caller's `k_len`, `offset` and block size are read from the calling frame. Each block covers `b*bs .. b*bs+bs-1`,
  clipped to the compressed width. The reference's `select_candidate_blocks` (`model.py:584-611`) does **not** clip
  blocks to the row's own valid length; clipping would cut the identical-row fraction to 6 %, without it it matches
  the noise floor.

## 8. FP8 lm_head (`sm120/sm120_lmhead.py`, opt-in)

Decode lm_head is a bandwidth-bound GEMV: 129280 x 5120 bf16 = 1.32 GB per step. At about 1.7-1.8 TB/s the floor is
0.74-0.78 ms, and cuBLAS measures about 0.8 ms/step (about 90 % of peak), so only fewer bytes help.

- `SM120_LMHEAD_IMPL=fp8` (default `bf16`): weight-only FP8, the head quantized once to e4m3 with one fp32 scale per
  (row, 128 columns), amax/448. A Triton kernel streams 0.66 GB instead of 1.32 GB, dots raw e4m3 values (exact in
  bf16) with the bf16 hidden state and applies the scale per 128-column partial. Used for M <= 64 rows (decode);
  larger M (big MTP verify batches, prompt logprobs) keeps cuBLAS bf16. Extra memory 0.66 GB.
- Hook: `sm120_lmhead.install()` wraps `sampler_utils.bf16_gemm_nt` (called by `get_logits`); `sm120_patches`
  installs it on import of `sampler_utils`. It composes with the logit capture (P6), which looks `bf16_gemm_nt` up at
  call time.

Measured on full-depth test model activations (`tests/lmhead_eval.py`, `hidden.prefill.final_norm` from a
`tools/run_deepinfer.py` dump, head from the same checkpoint). The bf16 recomputation equals DeepInfer's own logits
(cos 1.000000); the weight quantization error is rel-L2 2.6e-2.

| prompt | positions | logits cos mean / min | top-1 agreement | top-5 overlap | KL(bf16 ‖ fp8) mean / max | max abs logit diff |
|---|---|---|---|---|---|---|
| short | 327 | 0.999886 / 0.999729 | 93.9 % | 97.6 % | 1.36e-3 / 2.2e-3 | 0.35 |
| long | 2081 | 0.999870 / 0.999716 | 97.7 % | 98.0 % | 6.2e-4 / 3.9e-3 | 0.35 |

| M | cuBLAS bf16 | FP8 Triton |
|---|---|---|
| 1 | 0.85 ms | 0.41 ms |
| 8 | 0.81 ms | 0.42 ms |
| 32 | 0.85 ms | 0.43 ms |

That is about -0.4 ms per decode step (about -2 % at full depth) for a 2-6 % change of top-1 greedy picks (measured on
the reduced-expert test model). The deviation comes only from the weight rounding. Because it changes outputs, it is
off by default.

## 9. Measured results

### Kernel microbenchmarks (`tests/bench_attn.py`, median ms)

| path | shape | torch | b12x | speedup |
|---|---|---|---|---|
| `flash_mla_sparse_fwd` | 327 q x 640 idx x 64 heads | 2.40 | 2.21 | 1.1x (host overhead dominated) |
| `flash_mla_sparse_fwd` | 2048 q | 14.3 | 4.2 | 3.4x |
| `flash_mla_sparse_fwd` | 8192 q | 56.8 | 12.9 | 4.4x |
| `fused.prefill` (b12x core + Triton norm/rope/cast) | 2048 q / 8192 q | 18.2 / 73.8 | 4.5 / 14.9 | 4.0x / 5.0x |
| `flash_mla_with_kvcache` decode, eager | batch 1 / 8 / 32 | 1.03 / 1.03 / 1.04 | 1.02 / 1.02 / 1.02 | host bound |
| same, CUDA graph replay | batch 1 / 8 / 32 | 0.10 / 0.18 / 0.55 (torch graph) | 0.064 / 0.078 / 0.120 | |
| `paged_mqa_logits` | batch 1 / 8 / 32, ctx 8192, H 32 | 0.87 / 0.91 / 2.69 | 0.28 / 0.28 / 0.28 | 3x / 3x / 10x |
| paged logits + top-k, graph | batch 1 / 8 / 32 | 1.23 / 1.25 / 2.79 (eager torch) | 0.09 / 0.10 / 0.11 | 13-25x |
| `mqa_logits` (prefill) | 327 / 2048 / 8192 q | 0.79 / 1.16 / 9.76 | 0.54 / 0.51 / 1.06 | 1.5x / 2.3x / 9.2x |

These were taken before the Triton KV encoder and the sync removals; afterwards prefill attention for 327 / 2048 /
8192 q went from 2.35 / 4.21 / 12.9 ms to 1.43 / 3.25 / 11.85 ms. In the 8192-q profile the b12x extend kernel takes
2.6 ms per 2048 rows (about 66 TFLOPS bf16); the rest is torch-side work and 1-2 ms of host launch overhead per call.

### Decode step breakdown, YOCO test model (8 layers, 2048-token prompts, CUDA graphs, `tests/prof_e2e.py`)

After the mHC/einsum Triton kernels (before: 6.72 / 9.44 / 9.08 ms/step at B = 1 / 8 / 32):

| B | ms/step | GEMM bucket | mHC/norm (incl. prenorm) | elementwise/copy |
|---|---|---|---|---|
| 1 | 4.54 | 2.03 | 0.38 | 0.33 |
| 8 | 5.02 | 2.06 | 0.39 | 0.35 |
| 32 | 4.84 | 2.25 | 0.39 | 0.35 |

Remaining at B=8 (5.0 ms/step): bf16 GEMMs incl. lm_head 0.86, b12x dense GEMMs 0.7, `_einsum_kernel` 0.31, MoE 1.06,
mHC 0.39, top-k 0.24, kv/rope/quant 0.32, copies 0.35, attention 0.17, host gap 0.35. Attention, indexer, top-k and
the fused kernels together are about 0.35 ms/step.

Prefill, B=32 x 2048 = 65536 tokens, chunked at 4096: 3.10 s (21.1k tok/s) before, 2.61 s (25.1k tok/s) after the
mHC/einsum kernels.

### Full-depth test model (40 layers, 16/384 routed experts, engram on layers 1 and 14, CUDA graphs, 512-token prompts)

`tests/prof_e2e.py --ckpt /work/ckpt_full --batches 1,8,32 --prompt-len 512 --decode-steps 16 --modes graph --nextn
{0,5} --max-num-batched-tokens 2304 --gpu-memory-utilization 0.95 --max-blocks 512`

| B | decode ms/step, nextn 0 | decode ms/step, nextn 5 (6 rows per request verified per step) |
|---|---|---|
| 1 | 18.1 | 15.8 |
| 8 | 23.2 | 22.9 |
| 32 | 28.0 | 51.4 |

| category (nextn 0), ms/step | B=1 | B=8 | B=32 |
|---|---|---|---|
| GEMM | 6.8 | 7.9 | 7.8 |
| MoE | 3.5 | 5.6 | 6.2 |
| mHC/norm | 1.9 | 1.9 | 2.0 |
| other (mostly `wait_kernel`, likely the engram host-mode prefetch wait) | 1.4 | 2.3 | 3.6 |
| elementwise/copy | 1.4 | 1.5 | 1.7 |
| kv/rope/quant | 1.4 | 1.4 | 1.5 |
| top-k | 0.6 | 0.7 | 0.6 |
| attention | 0.56 | 1.13 | 3.55 |
| fused prep/cast | 0.08 | 0.11 | 0.19 |
| indexer | 0.03 | 0.05 | 0.15 |
| host gap | ~0.6 | ~0.6 | ~0.8 |

Within GEMM at B=8: b12x dense projections 3.5, `_einsum_kernel` (wo_a, 38 us x 40) 1.5, lm_head 0.85, per-layer
fp32-out bf16 GEMMs 1.2 ms/step.

### Long context (3-layer test model with the real `candidate_topk_blocks=2048`, batch 1, CUDA graphs, chunked prefill)

`tools/run_deepinfer.py` does not support chunked prefill, and unchunked prefill at >= 16k runs out of memory on
32 GB. Correctness therefore uses `tests/longctx_check.py` (drives the engine with chunking and reads decode logits
through `sm120_capture`); performance uses `tests/longctx_bench.sh` (`tools/bench_e2e.py`).

| prompt | chunk 2048: prefill tok/s / TPOT | chunk 8192: prefill tok/s / TPOT | peak |
|---|---|---|---|
| 4k | 19.0k / 3.62 ms | 21.9k / 3.50 ms | 25.6-26.5 GiB |
| 16k | 33.4k / 3.51 ms | 39.2k / 3.52 ms | 25.6-28.2 GiB |
| 32k | 38.6k / 3.53 ms | 43.7k / 3.52 ms | 25.7-28.2 GiB |
| 64k | 38.2k / 3.54 ms | 42.7k / 3.54 ms | 25.9-28.3 GiB |

TPOT is flat in context length (sparse attention); prefill throughput saturates at about 40k tok/s from 32k on.

| prompt (tiled) | first token, b12x vs torch (chunk 2048) | step-0 decode logits cos, b12x vs torch | noise: b12x vs b12x (graph/eager or chunk 2048/8192) |
|---|---|---|---|
| 16k | equal | 0.986 | 0.991 |
| 32k | equal | 0.990 | 0.973-0.989 |
| 64k | equal | 0.975 (graph) / 0.969 (eager) | b12x graph == eager, 18/18 tokens |

The truncated 3-layer model plus the b12x MoE run-to-run nondeterminism ([MOE_GEMM.md](MOE_GEMM.md)) make identical
configurations diverge after a few tokens, so the comparison is step-0 cosine against that noise level.

### End-to-end correctness

- **3-layer test model:** b12x eager is equivalent to the torch attention path; per-layer rel-L2 against the
  reference is unchanged (layer 0/1/2: 2.37e-2 / 3.54e-2 / 6.92e-2). Layer 0 is flagged at about 20x the ref-vs-ref
  noise in every configuration: that is DeepInfer's own `fused_transform` rounding, not the shims (see
  [VALIDATION.md](VALIDATION.md)). Graph mode printed no b12x fallback warnings.
- **Run-to-run nondeterminism comes from MoE, not attention:** in two identical eager runs (`tests/determinism.sh`)
  the first difference is at `L0/routed_out` (max abs diff 0.094); every attention tap (`q_post_rope`,
  `attn_core_out`, `attn_out`) is bit-identical with both `SM120_ATTN_IMPL=torch` and `b12x`.
- **YOCO test model** (2081-token prompt, `tests/e2e_matrix.sh` with `TEST_CKPT=/work/ckpt_yoco
  TEST_CAPTURE=3,5,6,7 TEST_L0=1`): b12x eager, torch attention and b12x with graphs all decode 17/17 tokens identical
  to the reference; every layer except layer 0 is at about 1.05-1.2x noise; `compare_l0` on layers 3/5/6/7 flags 0
  tensors. DeepInfer's selections overlap the reference at the ref-vs-ref noise level:

| layer | role | `compress_idx` set overlap, DI / ref2 | `cand_mask` overlap, DI / ref2 |
|---|---|---|---|
| 3 | C2 consumer | 0.986 / 0.991 | - |
| 5 | C1 source + candidate source | 0.913 / 0.916 | 0.966 / 0.968 |
| 6 | C1 consumer | 0.913 / 0.916 | - |
| 7 | index source using candidates | 0.902 / 0.905 | - |

- **MTP / DSpark draft path** (`tests/mtp.sh` on an MTP test checkpoint, nextn 0 / 5, eager / graphs, and nextn 5
  with `SM120_ATTN_IMPL=torch`): greedy tokens identical in all runs. The draft layers' attention runs on b12x under
  graphs (no-extra decode plan, `swa_width=192`), and the 20k-token profile prefill also runs on b12x. Draft acceptance
  on the truncated checkpoint is 0 because its DSpark targets are not the trained ones; it says nothing about attention.
- **Unit tests:** `tests/test_attn_shims.py` + `tests/test_mhc_einsum.py`, 77 passed.

## 10. Known gaps

1. **b12x LSE** is not returned by default (NaN); DeepInfer never uses it. `SM120_ATTN_B12X_LSE=1` requests it, but
   the layout and sink handling are unverified.
2. **Coverage:** validated on the truncated test checkpoints and the unit shapes. A b12x error on a new shape falls
   back to torch with one warning, so a failure costs speed, not correctness; use `SM120_ATTN_STRICT=1` to surface it.
   Main cache 288 B (non-SWA layers) or extra cache 528 B goes to torch.
3. **Plan creation under capture** (a shape not warmed up) records the torch path: correct but slow, with a warning.
4. **Top-k is still `torch.topk`** over the full `max_kv_len` width, which costs bandwidth on decode.
5. **Prefill overhead:** the KV re-encode is repeated per YOCO consumer layer, and there is one host sync per call.
6. **NaN handling in top-k:** NaN ranks lowest; the original aborts.
7. **Dense varlen `flash_attn_varlen_*`** is a forward-only torch loop with bottom-right causal alignment (unused by
   DeepInfer).
8. **Rounding differences, all within FlashMLA's own test tolerances:** the FP8 out-cast clamps amax at 1e-4
   (DeepGEMM style; FlashMLA's tests clamp `amax/448`), and the decode output is rotated in fp32 without the
   reference's intermediate bf16 rounding.
9. **`enable_batch_invariant`** is recorded but has no effect on attention. The torch paths are deterministic and the
   attention taps were bit-identical across runs on b12x too.

## 11. Tests and tools

```bash
# inside the image (PYTHONPATH=/opt/sm120/sm120:/opt/deepinfer is preset)
cd /opt/sm120
B12X_AUTOTUNE=0 pytest -q tests/test_attn_shims.py                   # everything
B12X_AUTOTUNE=0 pytest -q tests/test_attn_shims.py -k "b12x or graph" # b12x and CUDA-graph tests only
pytest -q tests/test_mhc_einsum.py
python3 tests/bench_attn.py                                            # prefill 327/2048/8192, decode 1/8/32, indexer, topk
python3 tests/bench_attn.py --only decode --decode 1,8,32
```

Caches are written with DeepInfer's `store_kv_cache_ref` when importable, otherwise with FlashMLA-equivalent encoders
(the test prints a line when it falls back).

| test | covers |
|---|---|
| `test_prefill_b12x_vs_torch[ratio 1/2/4 x bf16/fp8]` | b12x prefill against the torch reference |
| `test_prefill_b12x_chunked_rows_and_fp8_kv[sink]` | row-chunk loop and `SM120_ATTN_PREFILL_KV=fp8` |
| `test_prefill_b12x_realistic_2048` | 2048 q x (128 SWA + 512 top-k) x 64 heads |
| `test_fused_prefill_b12x_vs_torch` | fused prefill core on b12x |
| `test_decode_b12x_batch_sizes[1/8/32]` | decode at realistic batch sizes |
| `test_indexer_b12x_vs_torch[H 32/16]` | all 4 logits functions plus top-k overlap |
| `test_indexer_b12x_wide_and_chunked` | > 16384 columns and a small scratch budget |
| `test_indexer_fp32_weights_keep_torch_path` | fp32 weights stay on torch |
| `test_decode_path_cuda_graph[b12x/torch]` | capture/replay of fused decode + paged logits + top-k vs eager |
| `test_fused_triton_vs_torch[permute x neox x norm]` | Triton q-prep and rope/cast kernels |
| `test_encode_v41_triton_matches_torch` | Triton V4.1 KV encoder |
| `test_topk_contract[impl]`, `test_topk_triton_inf_nan_rank_above_oob` | top-k contract |
| `test_weight_permutations_are_consistent` | q_b/wv permutations vs natural layout |

| script | what it does |
|---|---|
| `tests/bench_attn.py` | kernel microbenchmarks, torch vs b12x |
| `tests/prof_prefill.py` | torch-profiler breakdown of the b12x prefill |
| `tests/prof_e2e.py` | engine profile: `--modes graph` gives per-category GPU kernel time, wall time, GPU-busy time and host gap per step; `--modes eager --modules` attributes GPU/CPU time per innermost `nn.Module` and per shim entry point (`sm120:*` ranges) |
| `tests/e2e_matrix.sh` | end-to-end runs with env variants, each fed through `tools/compare.py` (`TEST_CKPT`, `TEST_PROMPT`, `TEST_REF`, `TEST_REF2`, `TEST_ARGS`, `TEST_CAPTURE`, `TEST_L0`) |
| `tests/determinism.sh` | two identical runs per config, then a diff of every tap |
| `tests/mtp.sh` | MTP/DSpark draft path, greedy tokens compared across configs |
| `tests/longctx.sh`, `tests/longctx_check.py`, `tests/longctx_bench.sh` | long-context correctness and performance |
| `tests/lmhead_eval.py` | FP8 lm_head accuracy on dumped hidden states |

Operational notes: keep `B12X_AUTOTUNE=0` (avoids multi-candidate races during the first compile); set
`SM120_ATTN_DEBUG=1` on a first run to see the plan list and prefill KV class counts. If the host `/dev/shm` is small
or nearly full, engine start can fail; `DEEPINFER_MULTIMODAL_POOL_SIZE_GB=1` reduces DeepInfer's shared-memory pool.
