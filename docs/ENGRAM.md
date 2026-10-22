# Engram on SM120

DeepSeek-V4.1-Flash has two Engram layers (1 and 14): n-gram hashed lookups into two ~100 GB FP8 tables whose rows are
projected and gated into the hc residual stream. Stock DeepInfer serves the tables through DeepEP's `EngramBuffer`
(GPU-initiated RDMA between ranks). This port replaces that with a process-local fetch engine that reads the rows
directly from the HF safetensors shards.

Status: runs end to end on the RTX 5090 test machine, eager and with CUDA graphs, for layers 1 and 14, in `host` and
`uring` mode.

- Tables are mmapped straight from the original shards (`model-00047`/`00048`); nothing is copied or converted.
- Outputs match the official reference within the reference's own noise floor.
- With CUDA graphs, decode costs +0.0 to 1.3 ms per step over the same model without engram (2-5 %).
- `tests/tests_engram.py` passes T0-T6 for both layers.
- `uva` mode does not work (`cudaHostRegister` of the read-only shard mapping: `cudaErrorNotSupported`). Guaranteed
  residency comes from `SM120_ENGRAM_PIN=mlock`. That was tested on the RTX 5090 test machine on 16 GiB shared by two
  processes; the full 189 GiB needs the 4-GPU host.

| file | purpose |
|---|---|
| `sm120/sm120_engram.py` | table access (mmap of the HF shard, tensors, UVA, io_uring), fetch engine, torch ports of the tile_kernels Engram ops, hash metadata, safetensors locator |
| `sm120/deep_ep/_engram.py` (+ `deep_ep/__init__.py`, `deep_ep/buffers/engram.py`) | `deep_ep.EngramBuffer` with the DeepEP v2 API, process-local |
| `sm120/sm120_patches_engram.py` | runtime patches E1-E6 for DeepInfer (loader, runtime, graph replay), installed from `sm120_patches.py` |
| `tools/ref/ref_engram.py` | Engram for the reference runner: an mmap-backed table instead of a 98 GB CUDA parameter |
| `tests/tests_engram.py` | correctness tests T0-T6 (DeepInfer side vs the official `engram.py`/`model.py`) and a micro-benchmark |
| `tests/tests_engram_graph.sh`, `tests_engram_bench.sh`, `tests_engram_full.sh`, `tests_engram_threads.sh` | end-to-end checks: graph vs eager (teacher forced, with a negative control), decode cost with vs without engram, gather-thread sweep |

## 1. Reference semantics (`inference/engram.py`, `model.py`)

Config (V4.1-Flash): `engram_layer_ids=[1,14]`, `max_ngram_size=4`, `n_heads=8`, `head_dim=256`,
`engram_vocab_size=16_000_000`, `pad_token_id=2`, `compressed_vocab_size=99092`, `hc_mult=4`, `dim=5120`.

1. **Compressed tokens** (`build_compressed_token_map`). Every token id is decoded alone with the raw `tokenizers`
   backend and normalized (NFKC, NFD, strip accents, lowercase, whitespace runs to one space, strip; a single-space
   token survives through a sentinel). Equal keys share one id; partial UTF-8 tokens are keyed by their raw vocab
   string. Result: 99092 ids. Every hash multiplier depends on that count.
2. **N-gram window** (`NgramHashState.forward`). For position p the window is `[c(p), c(p-1), c(p-2), c(p-3)]` over
   compressed ids. Before the sequence start, and from the first image token backwards, the slot holds `c(pad)`. A
   per-request cache carries this over the prefill/decode split.
3. **Hash.** Multipliers come from `np.random.default_rng(10007*layer_id)`, odd and bounded by
   `int64_max // 99092 // 2`, one per lookback. The value is a running XOR `h_n = x0*m0 ^ x1*m1 ^ ... ^ x_{n-1}*m_{n-1}`
   for n = 2, 3, 4. Each of the 3 orders x 8 heads has its own prime bucket count (the next primes above 16M-1, drawn
   in order over layers, orders, heads and never reused). A row id is `h_n % prime[n][head] + offset[n][head]` with a
   running-sum offset. That gives 24 row ids per token per layer; the table has sum(primes) rows: **384006168**
   (layer 1) and **384016682** (layer 14).
4. **Lookup** (`ParallelEngramEmbedding`). Rows are FP8 E4M3 `[rows, 256]` plus E8M0 `[rows, 8]` (one scale per 32),
   dequantized to bf16: `[B, L, 24*256 = 6144]`.
5. **Projection and gate** (`Engram.forward`). `kv = wkv(rows)`, an FP8 linear 6144 -> 5*5120 with 32x32 block
   scales (the reference re-quantizes its bf16 input with `act_quant`, ue8m0). `key = kv[:4*5120]` gives one key per
   hc copy, `value = kv[4*5120:]` is shared.
   `gate = sigmoid(signed_sqrt(<h*(q_w*k_w), key> * rstd(h) * rstd(key) / sqrt(dim)))` per (token, hc copy), |dot|
   clamped at 1e-6; `h <- h + gate * value`. Image tokens get gate 0.
6. **Where it applies:** to the hc residual stream `[B, L, 4, dim]` before blocks 1 and 14, i.e. on the input of that
   block's attention-side `hc_pre`.

## 2. How stock DeepInfer implements it

| step | DeepInfer code | runs on |
|---|---|---|
| enable | `hugging_face/config.py _build_engram`: on iff the HF config has non-empty `engram_layer_ids`; no runtime switch | - |
| compressed map | `engram_metadata.py CompressedVocabulary` (same normalizer); `worker/base.py init_engram` -> C++ `set_engram_lookup_table` | CPU |
| window | C++ `fill_engram_row` (`csrc/engine/input_metadata/prepare_tensor_input_base.cpp`), ZOS variant in `zos.cpp`. Output `ngram_token_ids` int32 `[T, 4]`, image tokens = -1 | CPU, pinned |
| hash metadata | `engram_metadata.py EngramLayout` (sympy `nextprime`, same prime order), `NgramHashMapping` (same RNG and bound) | - |
| hash | `tile_kernels.engram.engram_hash(...)` -> int32 `[n_layers, T, 24]` global row ids, -1 for image tokens | TileLang |
| fetch | `EngramRuntime.fetch` -> `deep_ep.EngramBuffer.fetch(hashes, use_tma_aligned_col_major_sf=True)` -> one hook per layer | DeepEP NCCL-GIN RDMA gets |
| apply | `decoder_layer.py`: an `hc_post_intervention` in `HCLayerMixin.attn_norm_forward`, after the previous layer's hc_post and before this layer's hc_pre (same place as the reference); disables mega_mhc for that layer | - |
| projection | `Engram.forward`: `fused_kv_projection(embed_hook())`, an FP8 `ReplicatedLinear` -> `deep_gemm.fp8_fp4_gemm_nt` on the fetched fp8 rows and their packed sf (no re-quantization) | DeepGEMM (here: shim) |
| gate | `tile_kernels.engram.fused_weight(q_w, k_w)` (once) and `engram_gate_fwd(h, kv, w, eps, 1e-6, image_token_mask=..., inplace=True)` | TileLang |

**Where the tables live (stock).** `NgramEmbeddingParameters` pads each table to a multiple of `lcm(256, enp)` rows and
gives every rank of the engram group (`engram_parallel_size`, default all ranks) a contiguous slice. The loader
materializes that slice in CPU RAM (a full 98 GB per table at enp=1) and the fp32 scales on the GPU (12 GB per table
at enp=1). `EngramRuntime.initialize` packs the scales to int32 ue8m0 and all-gathers them, so every rank holds all
scales in GPU memory (3 GB per table), then copies the slice into the RDMA-registered storage window. At run time
every rank fetches the rows of its own tokens from the owning rank via GPU-initiated RDMA.

**Hook output contract** (DeepEP `csrc/buffers/engram.hpp`): for FP8 tables a hook returns
`(data fp8 [T, 24*256], sf int32 [T, 24*2])`. sf holds 4 E8M0 bytes per int32 (byte k = group k, little endian, the
same bytes as the HF `.scale` tensor). With `use_tma_aligned_col_major_sf` sf is MN-major with strides
`(1, align(T, 4))`. Ids outside the table give zero rows.

## 3. What the SM120 port does

**Fetch engine** (`sm120_engram.EngramFetcher` behind `deep_ep.EngramBuffer`). Each process serves **global** row ids
from a table that covers the whole global table. There are no peers, so any engram group size works. The hook contract
above is met exactly, so the FP8 GEMM sees the same operand as on SM100.

| `SM120_ENGRAM_MODE` | table | gather | CUDA graphs | RAM needed |
|---|---|---|---|---|
| `host` (default) | mmap of the HF shard (`MADV_RANDOM`); page cache = RAM tier | D2H of the ids + one stream sync, CPU `index_select` in worker threads (each page-faults its rows from NVMe), pinned staging, async H2D. The layer-14 gather runs while the GPU computes layers 1-13 | yes (§3.1) | none beyond page cache |
| `pinned` | anonymous shared memory (memfd) holding the whole tables once for all ranks, created by the server / `--dp` proxy before the ranks start, filled by rank 0 after load (reads from the shards), registered with `cudaHostRegister(Portable\|Mapped)` in every rank (costs about 2 MiB of GPU memory per GiB for the page tables, 0.34 GiB per rank for the full tables, taken from the KV cache) | GPU `index_select` over PCIe from the pinned host memory (8-byte elements): about 2.3 ms per 98k rows (both planes) after a one-time warm-up of ~0.1 s per process on the RTX 5090 test machine; no host sync | yes (inside the graph) | tables (188.8 GiB) + the ranks' own memory; start-up refuses if it does not fit (MULTI_GPU_RUNBOOK.md §3) |
| `uva` | the same mmap, `cudaHostRegister(Mapped\|ReadOnly)`, CUDA view | GPU `index_select` over PCIe, no host sync | yes | >= table size (pinned). **Does not work:** registering the read-only file pages fails with `cudaErrorNotSupported` (RTX 5090 test machine); pin with `SM120_ENGRAM_PIN=mlock` instead |
| `uring` | b12x `DiskRowCache` (O_DIRECT io_uring, `b12x.sequence._shared.disk_table`) for the weight rows; scale rows and prefill-sized batches (>= `SM120_ENGRAM_URING_MMAP_TOKENS` tokens, default 256) through the mmap like `host` | decode: per-batch reads into mapped-host staging, then a GPU copy; prefill: as `host` | yes (§3.1) | staging only; container needs io_uring allowed by seccomp and a memlock limit (§7) |
| (tensors) | `EngramBuffer.write` (stock path, enp=1 only), CPU or CUDA | as above | GPU: yes | as stock |

`SM120_ENGRAM_PREFAULT=1` reads the whole table into the page cache at start-up. It only makes sense when free RAM
exceeds the table size.

**Resident scale planes** (`SM120_ENGRAM_SCALES=resident`, default, `host` and `uring`). A lookup reads one 256-byte
weight row and one 8-byte E8M0 scale row, which sit on different 4 KiB pages: a cold lookup used to be two random
NVMe reads. The scale planes are 2.9 GiB per table (5.7 GiB for both, one page-cache copy for all ranks of a host).
`open_table` reads them sequentially at attach (about 1 s each on the RTX 5090 test machine) and
`post_load_cache_policy` once more after all ranks have loaded (their weight streaming may have evicted them). Every
lookup touches the scale pages, so the LRU keeps them. A cold lookup is then one read: cold 4k-token gather 0.78 s ->
0.30 s (one process), 1.65 s -> 1.16 s (4 processes at once, NVMe-bound at about 0.8 M random reads/s). `uring` then
reads only the weight rows with O_DIRECT. `SM120_ENGRAM_SCALES=disk` restores the old behaviour.

### 3.1 CUDA graphs

The rows of step N+1 depend on the token sampled at the end of step N, which only the GPU knows. With `host`/`uring`
tables the gather is CPU work, so it cannot itself be part of a graph. Two modes (`SM120_ENGRAM_GRAPH`):

- **`async` (default): in-graph host fetch.** Under capture, `fetch()` records into the graph (1) a D2H copy of the
  hash ids into pinned memory and (2) a Triton `signal` kernel that bumps a device generation counter and writes
  `(generation, T)` into a pinned flag with one 64-bit store. A host service thread polls that flag, gathers each
  layer's rows into pinned staging (for `uring`, b12x's native reader writes its mapped-host staging) and writes a
  per-layer done flag. The hook of engram layer l, also captured, runs a Triton `wait` kernel that spins on that flag
  (with a timeout and an error flag), then copies the rows H2D.
  - No host code runs per replay; the fetch starts right after the previous step's sampling (the earliest point the
    tokens exist) and overlaps the embedding and layer 0 (and layers 1-13 for layer 14). ZOS keeps its overlap.
  - The pinned buffers, thread and Triton JIT are set up at the first eager fetch, never inside a capture (compiling
    Triton inside a capture recorded a launch that never ran again).
  - `SM120_ENGRAM_SWITCH_INTERVAL` (default 0.2 ms) shortens Python's GIL switch interval, so the thread is not stuck
    behind DeepInfer's scheduler for 5 ms.
- **`prestage`.** Patch E6 wraps `CudaGraphRunner.replay`: it hashes the windows `_prepare_input_tensors` already wrote
  into the graph's `input_metadata_source` (ZOS included) and calls `EngramBuffer.prestage()`, which fills the static
  row buffers before `graph.replay()`. Simple, but it waits for the previous step on the host and serializes host and
  GPU. `SM120_ENGRAM_PRESTAGE=0` skips it; that is the negative control (stale rows).

Device and `uva` tables gather on the GPU inside the graph directly.

### 3.2 Runtime patches (`sm120_patches_engram.py`)

- **E1** `deepseek.materialize_hf_parameter`: an Engram table becomes a 1-byte `expand()` placeholder with the right
  shape and attributes (saves 98 GB CPU and 12 GB GPU memory per table).
- **E2** `weights._load_engram_tables`: records `(shard path, byte offset)` of `...engram.embed.weight` and `.scale`
  instead of copying, using DeepInfer's own `_shard_of` / `resolve_hf_name` / `scale_name` plus a header-only
  safetensors parse.
- **E3** `NgramEmbedding.__init__` keeps that record; `EngramRuntime.initialize` opens the tables
  (`sm120_engram.open_table`, mode from the env) and calls `EngramBuffer.attach_tables`. It skips `write()` and
  `gather_sf()`, so no GPU-resident scales are needed (scales are fetched with the rows).
- **E4** (only with `SM120_ENGRAM_TK=torch`) replaces the three tile_kernels Engram ops with the torch ports, in case
  the TileLang kernels misbehave. Not needed on SM120 so far.
- **E5** `hugging_face.loading.safe_open` (shard validation and shape checks): when safetensors cannot map a shard,
  falls back to a header-only handle. safetensors maps files privately, which the kernel refuses (ENOMEM) for a
  101 GB shard on a host with less RAM+swap, so the stock loader fails with "HF shard is unreadable". Only `keys()` and
  `get_slice().get_shape()/get_dtype()` are needed there; the ordinary tensors in that shard are read by DeepInfer's
  own pread-based `_TensorReader`.
- **E6** `DeepseekModel.__init__` registers models that use engram (and hands their hash tensors to the prefetcher);
  `CudaGraphRunner.replay` prestages rows, but only in `SM120_ENGRAM_GRAPH=prestage`.
- **E7** `LLMEngine.add_request` / `abort_request` report admitted prompts (in admission order) to the prefetcher and
  cancel aborted ones (§3.3). E3 starts the prefetcher.

`SM120_ENGRAM_PATCHES=0` disables all of them (E1-E7). Hashing, the gate (tile_kernels), the wkv GEMM (deep_gemm shim) and the
C++ windows are unchanged DeepInfer code.

**Why not `b12x.sequence.engram` as the op.** Its hash has the same semantics, but it is bound to vLLM request state
(query_start_loc, committed history), and its lookup returns dequantized bf16 `[T, 6144]`, while DeepInfer's projection
wants the fp8 rows and sf. DeepInfer already computes the hash itself. b12x's O_DIRECT row reader is reused as the
`uring` backend.

**Reference side** (`tools/ref/ref_engram.py`). The official model would allocate each table as a 98 GB CUDA parameter.
`install(M)` swaps `M.ParallelEngramEmbedding` for `MmapEngramEmbedding`: the same forward math, no parameters, rows
read through `sm120_engram.EngramHostTable` from the mmap. `NgramHashState`, `EngramLayout` and `Engram` (wkv + gate)
stay the official code. `tokenizer_for_engram()` adapts a raw `tokenizers.Tokenizer` to what
`build_compressed_token_map` expects.

### 3.3 Prefetch at request admission (`EngramPrefetcher`)

The rows a prompt needs are fully determined by its token ids. When a request is submitted (`server/engine.py`
`EngineLoop.submit`, and E7 for every `LLMEngine.add_request`), its prompt goes to a per-process prefetcher. A
background thread works through the queued prompts in blocks of `SM120_ENGRAM_PREFETCH_BLOCK` tokens (4096, one
prefill chunk):
- It hashes the block on the CPU exactly as DeepInfer does: compressed ids (`CompressedTokenizer.encode`), n-gram
  windows (a vectorised `fill_engram_row`, with look-back across block boundaries), `engram_hash`. That is about 1 ms
  per 1k tokens.
- It reads the weight rows through the mmap into a scratch buffer from `SM120_ENGRAM_PREFETCH_THREADS` threads (16).
  This is the `touch` method, the default; `=fadvise` issues `posix_fadvise(WILLNEED)` for the page ranges, sorted and
  coalesced, from a small C helper built with the system compiler.

The NVMe reads overlap the queue wait and the GPU compute of earlier chunks. The prefill's own gather then finds the
rows in the page cache: 17-35 ms per 4096-token chunk and layer instead of 250 ms.
- **Exact:** the data path is unchanged (same rows, same file), so the results are bit-identical. T7 checks that the
  prefetcher computes exactly DeepInfer's row ids and that its page ranges cover every row. A wrong or late prefetch
  only costs I/O.
- **Order:** E7 reports prompts in the scheduler's admission order. The server's earlier submit-time call can be out
  of order across HTTP threads, so admitted requests are served in admission order. Without that, the second of 4
  concurrent 32k prompts often read the wrong prompt's rows.
- **Bounded:** at most `SM120_ENGRAM_PREFETCH_AHEAD` prompt tokens (32768, about 6 GiB of page cache for both tables)
  are read ahead of what the prefills have consumed.
- **Cancellation:** abort removes a queued request.
- **Graph-safe:** host-only work outside any graph.
- **`touch` vs `fadvise`:** `touch` maps the pages like the gathers do. `fadvise` pages are unmapped, and the kernel
  reclaims those first under page-cache pressure, which is before they are used: `fadvise` measured 21 s against
  `touch` 16 s total gather time over the same prompt series.
- **Off switch:** `SM120_ENGRAM_PREFETCH=0`.
- **Prefix cache:** a prompt prefix served from the prefix cache is prefetched anyway (wasted I/O, no harm).

## 4. Correctness notes

- **Row values.** DeepInfer GEMMs the stored fp8 rows; the reference GEMMs `act_quant(dequant(rows))`. The bytes differ
  for roughly half the rows (the scale exponent shifts when the dequantized amax falls just below a power of two), but
  the values are identical for 100 % of rows, so both GEMMs see the same operand (T5).
- **Ids -1 and padding rows** (image tokens, rows that pad the table to a multiple of 256) give zero weights, matching
  DeepEP and the stock loader.
- **Layer numbering.** Multipliers are seeded with the model layer id and primes depend on the order of
  `engram_layer_ids`. A truncated test model must keep original layer 1 **at index 1** with `engram_layer_ids=[1]`;
  its primes are then identical to the full model (sum = 384006168).

## 5. Results (RTX 5090 test machine)

**`tests_engram.py`, real shard `model-00047` (layer 1), CUDA, DeepInfer ops (tile_kernels TileLang on SM120):**

| check | result |
|---|---|
| T0 | compressed map identical: 129280 -> 99092 ids |
| T1 | rows 384006168 / 384016682 |
| T2 | hash ids identical to the official `NgramHashState`, including the prefill/decode split and an image span; image tokens give -1 |
| T3 | 22k rows bit-exact against a pread of the shard |
| T5 | 100 % of rows reproduce their values after `act_quant` (60.3 % identical bytes) |
| T4 | real wkv/q/k: max\|diff\| 7.8e-3 against an engram contribution of 6.1, rel 1.3e-3, 99.995 % bitwise equal |
| T6 | captured graph vs eager fetch, bit-exact over 5 replays, `host` and `uring`, layers 1 and 14 |

`engram_hash`, `fused_weight` and `engram_gate_fwd` compile and run on SM120 (TileLang); E4 is not needed. The TileLang
gate needs dim >= 512. The synthetic mode (`--synthetic DIR`, a small generated table with the real geometry) passes as
well and runs on CPU.

**End to end, YOCO test model with engram on layer 1** (`make_truncated_ckpt.py --preset yoco --keep-engram 1`,
2081-token prompt). Per-layer prefill hidden rel-L2:

| layer | DI vs ref | noise (ref torch vs tilelang) | ratio | ref without engram vs ref with engram |
|---|---|---|---|---|
| 0 (no engram) | 1.84e-2 | 9.0e-4 | 20 (flagged) | 0 |
| **1 (engram)** | 3.20e-2 | 1.30e-2 | **2.45 ok** | **0.53** (engram changes the stream a lot) |
| 2-7, final | 5.1e-2 ... 2.0e-1 | 3.7e-2 ... 2.0e-1 | 1.0-1.4 ok | 0.50-0.69 |

- Prefill top-1 vs ref 0.63 (ref vs ref2: 0.63). Decode identical for 11 of 17 tokens; first 4 decode steps match top-1.
- The layer-0 flag is unrelated to engram: it is the known DeepInfer `fused_transform` rounding difference (see
  [VALIDATION.md](VALIDATION.md)) and shows up without engram too.
- Inside layer 1 (`compare_l0.py`), the trunk after engram has DI-ref 1.57e-2, i.e. the error is inherited from layer 0
  and not grown by engram. A wrong engram would give about 0.5.

The full-depth test model (all 40 layers, engram on layers 1 and 14) is covered in [VALIDATION.md](VALIDATION.md).

**Table micro-benchmark** (`--bench`, layer 1, uniform random row ids = worst case, H2D included, 32 threads; cold =
page cache dropped first):

| tokens | `host` cold ms | `host` warm ms | `host` warm tok/s | `uring` ms (cold = warm) |
|---|---|---|---|---|
| 1 | 33 | 0.21 | 4.7k | 0.37 |
| 8 | 14 | 0.23 | 35k | 1.0 |
| 64 | 75 | 0.40 | 160k | 5.6 |
| 512 | 250 | 1.9 | 276k | 42 |
| 2048 | 249 | 3.4 | 601k | 161 |
| 8192 | 605 (480 with 128 threads) | 12.7 | 646k | 590 |

- `host` warm: about 15 M rows/s (3.9 GB/s gathered). `host` cold: about 0.3-0.4 M rows/s (each row costs a weight and
  a scale page fault); more threads barely help, so the NVMe or fault path is the limit.
- `uring`: a flat 0.3 M rows/s; O_DIRECT never uses the page cache.
- Both lines were measured with the scale planes on disk. Cold per-row costs, one process, 98k random rows:

  | how | weight and scale rows from disk | weight rows only (scales resident) |
  |---|---|---|
  | page faults, 32 threads | 0.24 M rows/s | 0.76 M rows/s |
  | `posix_fadvise(WILLNEED)` from a C helper, 32 threads, then the gather | — | 0.67 M rows/s |
  | 4x the rows in one process | — | 0.68-0.80 M rows/s |

  The device tops out at about 0.8 M random 4 KiB reads per second (Samsung 990 PRO behind LVM). Coalescing adjacent
  pages saves little: 98k rows touch 97.7k ranges. Per-row cost can therefore only go down by issuing fewer reads (the
  resident scales) and by hiding them (prefetch).

**Prefill TTFT with a cold engram cache** (server on `ckpt_full`, engram on layers 1 and 14). Setup:
- Shards 47/48 were dropped from the page cache before the server started.
- Every request uses fresh random token ids, so every lookup is effectively cold.
- `--max-num-batched-tokens 4096`. Repeated runs vary by about ±5 %.
- before = `fix/prefill-throughput`; after = resident scales + prefetch + `uring` prefill through the mmap; ablation =
  after with `SM120_ENGRAM_PREFETCH=0`.

TTFT in seconds; "wall" is until the last of the concurrent requests has its first token.

| case | no engram | `host` before | `host` after | `host` ablation | `uring` before | `uring` after | `uring` ablation |
|---|---|---|---|---|---|---|---|
| 4k x1 (second request) | 0.81 | 1.20 | 0.98 | 0.97 | 1.44 | 0.89 | 0.97 |
| 32k x1 | 6.29 | 9.23 | 6.65 | 7.74 | 11.36 | 6.64 | 7.90 |
| 4k x4, wall | 3.13 | 4.22 | 3.33 | 3.86 | 5.67 | 3.31 | 3.85 |
| 32k x4, wall | 25.13 | 32.70 | 26.11 | 31.23 | 45.41 | 25.99 | 30.39 |
| engram gather time, whole series | — | 27.5 s | 3.3 s | 19.1 s | — | 2.9 s | 18.0 s |

What engram adds on top of the no-engram prefill:
- 32k prompt: 2.94 s -> 0.36 s (`host`), 5.07 s -> 0.35 s (`uring`).
- 4 x 32k: 7.6 s -> 1.0 s (`host`), 20.3 s -> 0.9 s (`uring`).
- 4k prompt: 0.39 s -> 0.17 s (`host`), 0.63 s -> 0.08 s (`uring`). This case gets the resident scales but no
  overlap.
- `uring` prefill now runs through the mmap (prefetched rows); decode stays O_DIRECT.

The 4-rank case (4 processes gathering cold 4k prefills at once, `conc4.py`) went from 1.65 s to 1.16 s per prefill.
That is NVMe-bound; prefetch hides it as far as compute time allows. Exactness: `tests_engram.py` T0-T7 pass for
layers 1 and 14 in `host` and `uring`.

**CUDA graphs vs eager** (`tests_engram_graph.sh`, teacher-forced, 16 decode steps, decode KL over the top 50 against
an eager run):

| model | eager run 2 (noise) | graph `async` | graph `prestage` | graph without prestage (control) |
|---|---|---|---|---|
| YOCO, engram on layer 1 | 0.145 | 0.128 | 0.156 | 1.24 (top-1 0.25) |
| full depth, engram on layers 1 and 14 | 5.8e-3 | 6.6e-3 | 4.5e-3 | 8.1 (top-1 0) |

Both graph modes stay inside run-to-run noise; the control shows the test catches stale rows.

**Decode cost** (`tests_engram_full.sh`: full-depth test model, `bench_e2e.py`, graphs, prompt 256, 64 generated tokens;
TPOT in ms; baseline = the same truncation built without engram):

| B | no engram | `host` + `async` | `uring` + `async` | `host` + `prestage` |
|---|---|---|---|---|
| 1 | 16.4 | 16.8 (+0.4) | 16.4 (+0.0) | 17.6 |
| 8 | 20.7 | 21.5 (+0.8) | 21.1 (+0.4) | 25.8 |
| 32 | 24.8 | 26.1 (+1.3) | 26.0 (+1.2) | 28.8 |

- Gather tasks carry at least `SM120_ENGRAM_MIN_CHUNK` (default 64) rows, and weight and scale rows are separate tasks
  (different pages). With tiny tasks, Python per-task overhead (20-50 us) dominated cached rows; thread sweep at B=8
  (`tests_engram_threads.sh`): 1 thread 21.4 ms, 8 threads 21.2 ms, 64 threads 22.8-23.7 ms.
- At B=8 the rows of engram layer 1 are ready about 0.9 ms after the flag; embedding plus layer 0 take about 0.5 ms, so
  about 0.4 ms of wait remains.
- `uring` has the smallest decode overhead because its io_uring queue (QD 128) serves a decode batch's random rows in
  about one device latency, but it never hits a cache: B=8 prefill 2140 tok/s vs 5419 in `host`.
- `prestage` is correct but slower (+1-5 ms): every replay waits on the host for the previous step, which removes the
  ZOS overlap.
- Not done: a GPU-resident hot-row cache (n-gram rows are Zipf-like).

## 6. Commands (inside the container)

```bash
# correctness on real rows of shard 00047 (mmap; reads ~15k rows + wkv)
python3 /opt/sm120/tests/tests_engram.py --ckpt /ckpt --layer 1 --json /work/engram_test.json
python3 /opt/sm120/tests/tests_engram.py --ckpt /ckpt --layer 14 --impl torch          # without tile_kernels
python3 /opt/sm120/tests/tests_engram.py --synthetic /work/engram_synth                 # tiny generated table, no checkpoint
# micro-benchmark: cold vs warm page cache, decode and prefill sizes, both layers
python3 /opt/sm120/tests/tests_engram.py --ckpt /ckpt --skip-tests --bench --bench-all-layers --json /work/engram_bench.json
SM120_ENGRAM_MODE=uring python3 /opt/sm120/tests/tests_engram.py --ckpt /ckpt --skip-tests --bench --bench-all-layers
# DeepInfer with engram: a truncated checkpoint that keeps layer 1 + the symlinked shard 00047
SM120_ENGRAM_MODE=host python3 /opt/sm120/tools/run_deepinfer.py --ckpt /work/ckpt_yoco_engram ...
# prefill TTFT with a cold engram cache, one server per mode (docker host; see the script header)
IMG=somora_deep_sm120:0.3.0 CKPT_DIR=... WORK=... CACHE_DIR=... MODEL=/work/ckpt_full \
  MODES="host host_nopf:SM120_ENGRAM_PREFETCH=0 uring" bash tests/engram_cold_ttft.sh
# CUDA graphs: graph vs eager (teacher forced, negative control); decode cost with vs without engram
bash /opt/sm120/tests/tests_engram_graph.sh
CK=/work/ckpt_full PROMPT=/work/prompt_ids.json REF=/work/full_short_ref.pt \
  EXTRA='--max-num-batched-tokens 512 --gpu-memory-utilization 0.95' TAG=full bash /opt/sm120/tests/tests_engram_graph.sh
bash /opt/sm120/tests/tests_engram_full.sh        # CFGS="name:ckpt:ENV=1,ENV2=2+..." selects configs
SM120_ENGRAM_STATS=1 ...                          # per-fetch timing summary at exit
```

The shell scripts take `SM120_ROOT` (default `/opt/sm120`), `WORK` (default `/work`) and `HF_CKPT` (default `/ckpt`).

Truncated checkpoints symlink the engram shards (e.g. `engram-model-00047-of-00048.safetensors -> /ckpt/...`), so the
original shard directory must be visible at the same absolute path wherever the checkpoint is loaded.

## 7. Multi-GPU and deployment notes

- DeepInfer runs attention as DP and MoE as EP, so every rank needs lookups for its own tokens. With the mmap design
  each rank maps the same files and the page cache holds **one shared copy** (203 GB for both tables). Nothing is
  replicated per rank and nothing is exchanged between GPUs. Row sharding plus all-to-all only pays off when shards sit
  in GPU memory, and ~51 GB per GPU next to the weights does not fit on 96 GB cards.
- **MemAvailable before start-up >= tables + 5 GiB per rank + the registration (2.6 GiB per rank) + 6 GiB (225 GiB for 4 ranks):** `pinned`. There is
  one shared copy in pinned host memory and every rank gathers on the GPU. On the RTX 5090 test machine (tables
  truncated with `SM120_ENGRAM_SHM_MAX_ROWS`), the cost of the lookups disappeared: TTFT 4k 0.81 s, 32k 6.32 s,
  4 x 4k 3.15 s, the same as without engram (0.81 / 6.29 / 3.13 s).
- **MemAvailable after start-up >= tables + 8 GiB (about 212 GB):** `host` with `SM120_ENGRAM_PIN=mlock`. Rank 0
  prefaults the tables and locks them once. All ranks share that locked page-cache copy, so lookups never do NVMe I/O,
  which matters most where the NVMe sits behind a translating IOMMU (MULTI_GPU_RUNBOOK.md §3). The start-up refuses
  to lock when the headroom is missing (locked memory without swap would turn process growth into an OOM).
- **Tables + 4 GiB:** `SM120_ENGRAM_PREFAULT=1` caches them best effort, without a lock (reclaimable, no OOM risk).
  `uva` does not work with the shard mmaps (§9).
- **RAM < 203 GB:** `host` mode (page cache acts as an LRU over the Zipf-distributed rows, NVMe serves misses), or
  `uring` (O_DIRECT, predictable latency, no cache). Raise `SM120_ENGRAM_THREADS` and budget CPU cores: ranks x gather
  threads, plus one polling service thread per rank for `async` graphs.
- **Choosing a mode:** `uring` has the lowest decode latency on a cold or too-small page cache (queue depth 128, about
  1 ms for 192 random rows, constant); `host` wins when the hot rows are cached, which includes every prefill with a
  warm cache.
- **`uring` in containers:** Docker's default seccomp profile blocks `io_uring_setup/enter/register` ("Operation not
  permitted"), and the registered buffers need a memlock limit above the default ("io_uring buffer registration
  failed: Cannot allocate memory"). Tests ran with `--security-opt seccomp=unconfined --ulimit memlock=-1`; a custom
  seccomp profile that allows only the three io_uring syscalls is the cleaner choice. b12x builds its native reader
  once (`-e XDG_CACHE_HOME=/cache` keeps it); the serving image includes `liburing-dev` and `pkg-config` for that.
- Start-up: E1/E2 make loading the engram part O(header); the stock loader would read 203 GB.

## 8. Switches

| env | default | meaning |
|---|---|---|
| `SM120_ENGRAM_MODE` | `auto` (= `host`) | `host`, `pinned`, `uva`, `uring` (see §3); the serving image sets `host` |
| `SM120_ENGRAM_GRAPH` | `async` | CUDA-graph fetch mode: `async` or `prestage` |
| `SM120_ENGRAM_PRESTAGE` | 1 | 0 skips prestaging (negative control only) |
| `SM120_ENGRAM_PREFAULT` | 0 | 1 reads the whole tables into the page cache. The server does it once after every rank has loaded its weights, before it accepts requests (`--dp`: the proxy asks rank 0), and skips it when the tables do not fit into MemAvailable minus `SM120_ENGRAM_PREFAULT_HEADROOM_GB` (use `uring` there). `force` prefaults regardless |
| `SM120_ENGRAM_PREFAULT_HEADROOM_GB` | 4 | MemAvailable that the prefault leaves free. A partial prefault is not offered: on the RTX 5090 test machine it made cold prefills 20 % slower (page-cache pressure), and the second table's pages evicted the first's |
| `SM120_ENGRAM_RANK_HOST_GB` | 5 | `pinned`: host memory budgeted per rank in the start-up check (tables + ranks x (this + the registration, 14 MiB per GiB of tables) + `SM120_ENGRAM_PIN_HEADROOM_GB`, default 6 in this mode, must fit into MemAvailable before the ranks start) |
| `SM120_ENGRAM_PINNED_ALLOC` | register | `pinned`: `register` = every rank registers the memfd with `cudaHostRegister`; `vmm` (opt-in) = rank 0 allocates the tables once with `cuMemCreate` (pinned host memory, POSIX-fd shareable) and the other ranks import it for their GPU only: 2.6 GiB less host memory per extra rank for the full tables, same lookups; falls back to `register` if the allocation fails (MULTI_GPU_RUNBOOK.md §3) |
| `SM120_ENGRAM_VMM_WAIT_S` | 1800 | `vmm`: how long the other ranks wait for rank 0 to share the allocation |
| `SM120_ENGRAM_SHM_MAX_ROWS` | 0 | tests only: truncate the shared tables to this many rows so that `pinned` fits a small machine (lookups beyond read zeros) |
| `SM120_ENGRAM_PIN` | 0 | `mlock`: after load, prefault the tables and `mlock` them once in rank 0's process. The pages are the shared page cache of the shard files, so all ranks get guaranteed residency from one locked copy. Start-up fails if the tables plus `SM120_ENGRAM_PIN_HEADROOM_GB` do not fit into MemAvailable, or if `RLIMIT_MEMLOCK` is too low (`--ulimit memlock=-1`). The prefetcher is turned off (nothing to read ahead) |
| `SM120_ENGRAM_PIN_HEADROOM_GB` | 8 | MemAvailable that must remain after pinning (locked memory is not reclaimable and there is no swap) |
| `SM120_ENGRAM_RESIDENCY_INTERVAL` | 0 | seconds between residency log lines while serving (process that ran the post-load policy; mincore, about 1 s per 200 GiB) |
| `SM120_ENGRAM_PREFAULT_AT` | `attach` (library), `ready` (server) | when the prefault runs |
| `SM120_ENGRAM_DROP_WEIGHT_PAGES` | 1 | server: after load, drop the other shards' clean page-cache pages (posix_fadvise DONTNEED) so they do not compete with the tables |
| `SM120_ENGRAM_RESIDENCY` | 1 | log the tables' page-cache residency and MemAvailable at attach and after load (mincore) |
| `SM120_ENGRAM_THREADS` | 32 | host gather threads |
| `SM120_ENGRAM_MIN_CHUNK` | 64 | minimum rows per gather task |
| `SM120_ENGRAM_URING_QD` | 128 | io_uring queue depth (`uring` mode) |
| `SM120_ENGRAM_URING_MMAP_TOKENS` | 256 | `uring`: batches of at least this many tokens (prefill chunks) gather through the mmap, so prefetched rows hit the page cache; smaller ones (decode) are O_DIRECT reads |
| `SM120_ENGRAM_SCALES` | `resident` | read the scale planes into the page cache at attach and after load (one NVMe read per cold lookup instead of two); `disk` = old behaviour |
| `SM120_ENGRAM_PREFETCH` | 1 | read a prompt's rows into the page cache from request admission on (§3.3) |
| `SM120_ENGRAM_PREFETCH_METHOD` | `touch` | `touch` (threads read the rows through the mmap) or `fadvise` (posix_fadvise WILLNEED from a C helper) |
| `SM120_ENGRAM_PREFETCH_AHEAD` | 32768 | prompt tokens read ahead of the prefills' consumption (page cache: about 192 KiB per token for both tables) |
| `SM120_ENGRAM_PREFETCH_BLOCK` | 4096 | tokens per prefetch step |
| `SM120_ENGRAM_PREFETCH_THREADS` | 16 | prefetch I/O threads |
| `SM120_ENGRAM_TRACE` | 0 | 1 prints one line per prefill gather and prefetch block (timing, prefetch position) |
| `SM120_ENGRAM_SWITCH_INTERVAL` | 0.2 ms | Python GIL switch interval while the `async` service thread runs |
| `SM120_ENGRAM_TK` | (tile_kernels) | `torch` replaces the TileLang Engram ops with torch ports (E4) |
| `SM120_ENGRAM_PATCHES` | 1 | 0 disables E1-E7 |
| `SM120_ENGRAM_LAZY` | 1 | 0 keeps the stock table loader (E1, E2, E5 off) |
| `SM120_ENGRAM_CKPT` | HF dir | where E2 looks for the table shards |
| `SM120_ENGRAM_STATS` | 0 | 1 prints the fetch / gather / wait statistics (`[sm120_engram] stats (pid, device) {...}`, and `prefetch stats`) at exit and every `SM120_ENGRAM_STATS_INTERVAL` seconds (default 60) while they change |
| `DEEPINFER_ENGRAM_FORCE_GPU_STORAGE` | 0 | stock DeepInfer: put the table slice in GPU memory |

## 9. Open points

1. `uva` does not work: `cudaHostRegister` of the read-only file mapping returns `cudaErrorNotSupported`. Pinning goes
   through `SM120_ENGRAM_PIN=mlock` instead: one locked copy in the page cache, shared by all ranks.
2. The `async` mode depends on one host thread per process. If it dies, the GPU wait kernel times out (seconds) with
   stale rows and the error is printed.
3. On hosts where the tables cannot be fully cached, the first prefill and DeepInfer's profiling run are I/O bound, and
   page-cache pressure competes with weight loading.
4. DeepInfer's `skip_load` (dummy weights) path still allocates the full tables (`torch.zeros`); not patched.
5. E2 assumes table and scale live in safetensors shards under the HF dir (or `SM120_ENGRAM_CKPT`) and resolve through
   DeepInfer's `resolve_hf_name`.
6. **Prefetch under page-cache pressure** (test machine, 62 GB RAM).
   - **Cold, clean cache** (only the table pages dropped, the rest of the page cache mostly free): 4 x 32k wall
     26.1 s (no engram 25.1 s).
   - **Full cache** (other weight shards read first, `free` about 0; this matches the 0.2.1 release retest, which
     measured 30.5 s `host` / 30.0 s `uring`): 26-29 s, with the second 4 x 32k batch of a series 3-5 s slower.
     Every read then allocates through reclaim, and the prefill gathers' misses wait behind the prefetch reads on the
     same NVMe. 30-35 % of the 4096-token gathers take 100-600 ms instead of 17-35 ms.
   - **The throttle is not the cause.** `SM120_ENGRAM_PREFETCH_AHEAD` 8192 and 32768 give the same wall time (28.5 vs
     28.9 s). `throttled_s` only counts the time the prefetcher waits because it is far enough ahead.
   - **Things that did not help, so they are not in the code:**
     - unmapping the consumed weight pages (`MADV_DONTNEED`): the gathers stalled behind the 150-500 ms zap;
     - keeping the prefetched rows in private memory (rows only, 256 B each, so eviction cannot touch them): 95 %
       staged hits, but 15.9 s gather time against 10.8 s for `touch`, because the per-gather lookup costs 30-50 ms of
       CPU and the remaining misses still wait behind the prefetch reads;
     - `fadvise` readahead: 21 s against 16 s.
   - The prefetcher does not see which chunk the scheduler runs next. It assumes FIFO prefill in admission order,
     which is DeepInfer's behaviour in these tests.
   - **4-GPU host** (4 ranks, about 230 GB RAM, one NVMe):
     - Random prompts are the worst case. A 32k prompt reads about 1.6 s worth of rows at the device's random-read
       limit (about 0.8 M reads/s), against about 6.3 s of prefill compute. Four ranks prefilling 32k prompts at the
       same time therefore need about 6.4 s of NVMe per 6.3 s of compute: the shared NVMe, not the throttle, becomes
       the limit at full prefill load.
     - What helps there is a page cache that holds most of the tables. With about 190 GB available after start-up,
       about 90 % of the 189 GiB fits. It fills by use: the start-up prefault is skipped below tables + 10 %, see
       MULTI_GPU_RUNBOOK.md §3. The Zipf-like reuse of real text helps on top.
     - With the tables fully in RAM (pinned, one copy for all ranks, `SM120_ENGRAM_PIN`, MULTI_GPU_RUNBOOK.md §3)
       the shared NVMe stops mattering.
7. Prefetch cannot help a lone request's first chunk, whose gather starts right after admission. For that chunk only the
   resident scales help.
