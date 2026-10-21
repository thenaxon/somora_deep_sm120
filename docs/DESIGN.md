# Design: DeepInfer on SM120

This repo runs DeepSeek's [DeepInfer](https://github.com/deepseek-ai/DeepInfer) engine on SM120 GPUs (RTX 5090,
RTX PRO 6000 Blackwell). DeepInfer's own source is not modified. Its SM100-only kernel libraries are replaced by drop-in
packages on `PYTHONPATH`, and a small set of runtime patches is applied when the interpreter starts.

## 1. Why DeepInfer does not run on SM120 as shipped

DeepInfer targets SM100 (B200/GB200) and Ascend. Its hot path depends on four vendored DeepSeek libraries:

| library | used for | why it fails on SM120 |
|---|---|---|
| FlashMLA | sparse MLA prefill/decode, fused norm/RoPE/attention/cast | tcgen05 / TMEM, 2-CTA MMA |
| DeepGEMM | FP8×FP4 dense and grouped GEMMs, indexer MQA logits, mega-MoE, mHC fusions, `fp8_einsum` | tcgen05, up to 227 KB shared memory |
| DeepEP | MoE dispatch/combine, CP all-gather, Engram row exchange | NVLink / NVSHMEM / RDMA |
| DeepSelect | top-k | shared-memory size and 16-CTA clusters |

SM120 has about 99 KB of shared memory per block, no tcgen05/TMEM, and workstation systems have no NVLink. There is no
SM90 variant of these kernels to fall back to.

What *is* portable without changes:

- `csrc/` (host-only C++): the scheduler (async batch scheduling, preemption, `fork_seq` with copy-on-write), the block
  manager (128-token blocks, ref counts, state slots), the zero-overhead scheduler (ZOS) and input preparation. It contains
  no CUDA code and builds unchanged with `scripts/build.sh`.
- Engine and worker: lifecycle, memory profiling, CUDA graphs, the single-slab cache engine.
- The V4 model code: MLA with compressed KV (C1/C2) and the sparse indexer, SWA with sinks, YOCO, mHC, Engram, noaux_tc MoE.
- DeepInfer's own TileLang kernels: almost all run on SM120 against their torch references. One DSpark sampler kernel
  needs a smaller shared-memory window (patch P8, see [../patches/PATCHES.md](../patches/PATCHES.md)).
- Sampler, DSpark/MTP verification, structured outputs (xgrammar), batch invariance.

## 2. Kernel backend: b12x

[b12x](https://pypi.org/project/b12x/) (Apache-2.0; upstream issues are tracked in the FlashInfer repository) provides SM120 kernels for exactly the
missing pieces: `attention/compressed_sparse_mla`, `attention/dsa_indexer`, `gemm/block_fp8_linear`, `moe/fused_moe`,
`norm/mhc`, `comm/pcie`, `sequence/engram`. DeepSeek-V4.1 needs **b12x >= 1.5.1** (`cache_format="deepseek_v41"`,
MXFP4 indexer, hidden size 5120 in mHC, 32×32-block FP8 linears).

Key compatibility facts (verified on SM120 with `tests/test_attn_b12x_compat.py` and `tests/test_moe_b12x_local_subset.py`):

- **KV cache formats are byte-identical.** DeepInfer's SWA cache (528 B/token: 512 × E4M3 + 16 × UE8M0), main compressed
  cache (288 B/token FP4: 256 B packed E2M1 + 32 × E4M3 per-16 scales) and indexer cache (68 B/token MXFP4) are what b12x's
  `deepseek_v41` format reads. DeepInfer's TileLang writer and b12x's decoder see bit-identical values from the same bytes,
  zero-copy; b12x's `page_nbytes` equals DeepInfer's block bytes.
- DeepInfer's physical slot indices and logical indices plus a page table give bitwise identical results, so DeepInfer's
  indexing stays as it is.
- b12x attention in bf16 compute mode matches an fp32 reference at cos 0.999999 (decode and extend). The FP8-internal mode is
  slightly less exact (cos 0.996–0.998), still below the KV quantization error itself.
- b12x `fused_moe` (w4a8_mx) on real V4.1 experts: cos 0.999–0.99995 vs fp32 for T in 1..2048. Masking routes with `-1` works on
  every launch path, and tokens without a local expert give exact zeros. This is what makes expert parallelism over PCIe
  possible without a custom MoE kernel (section 5).
- Found on the way: b12x's tiny-decode MoE path (T <= 4) ignores `swiglu_limit`
  ([flashinfer-ai/flashinfer#6866](https://github.com/flashinfer-ai/flashinfer/issues/6866)). The repo ships a fix
  (`patches/b12x-tiny-decode-swiglu.patch`) and defaults to b12x's dynamic path anyway.

## 3. How the replacement is wired

### Drop-in packages (`sm120/`)

`sm120/` is a `PYTHONPATH` root (not a package). It is put **before** DeepInfer, so `import deep_gemm`, `import flash_mla`
etc. resolve to the shims with the same import paths and signatures:

| path | replaces | backend | docs |
|---|---|---|---|
| `sm120/flash_mla/`, `sm120/sm120_attn/` | FlashMLA | b12x `compressed_sparse_mla` (decode and extend), Triton kernels for the fused q prep / inverse RoPE / FP8 cast, torch reference | [ATTENTION.md](ATTENTION.md) |
| `sm120/deep_gemm/` | DeepGEMM | b12x `block_fp8_linear` (dense FP8), cuBLAS (bf16), b12x `dsa_indexer` (indexer logits), Triton `fp8_einsum`, torch references for mHC / mega symbols | [MOE_GEMM.md](MOE_GEMM.md) |
| `sm120/sm120_moe.py` | DeepGEMM grouped GEMMs + DeepEP dispatch/combine in the MoE layer | one b12x `fused_moe` call per layer | [MOE_GEMM.md](MOE_GEMM.md) |
| `sm120/deep_ep/` | DeepEP | single-rank buffers with identical layouts; Engram via `sm120_engram`; inert placeholders for EP > 1 | [MULTI_GPU.md](MULTI_GPU.md) |
| `sm120/sm120_ep/` | DeepEP all-to-all for EP > 1 | all-gather → local `fused_moe` → reduce-scatter (NCCL) | [MULTI_GPU.md](MULTI_GPU.md) |
| `sm120/deep_select/` | DeepSelect | `torch.topk` with the original contract | [ATTENTION.md](ATTENTION.md) |
| `sm120/flash_attn/` | FlashAttention (ViT) | SDPA | |
| `sm120/sm120_engram.py`, `sm120/sm120_patches_engram.py` | DeepEP `EngramBuffer` (RDMA) | mmap of the HF shard + host gather, UVA, or io_uring | [ENGRAM.md](ENGRAM.md) |
| `sm120/sm120_lmhead.py` | bf16 lm_head GEMM (opt-in) | weight-only FP8 Triton kernel | [ATTENTION.md](ATTENTION.md) |
| `sm120/sm120_capture.py` | (instrumentation) | logit / hidden-state / fine-tap capture for validation | [VALIDATION.md](VALIDATION.md) |

Every component can be switched to an exact torch reference (`SM120_GEMM_IMPL`, `SM120_MOE_IMPL`, `SM120_ATTN_IMPL=torch`).
That is the main debugging tool: bisect a numerical problem by switching one backend at a time.

### Runtime patches (`sitecustomize` → `sm120_patches`)

`sm120/sitecustomize.py` is imported automatically by Python at start-up because it is on `PYTHONPATH`. It chains to any
other `sitecustomize` further down the path and imports `sm120_patches`, which registers **post-import hooks**: each patch runs
right after DeepInfer imports its target module. Nothing is imported eagerly (DeepInfer's `__init__` forbids importing torch
early), and patches only act on compute capability 12.x (`SM120_FORCE_SM120` overrides, `SM120_PATCHES=0` disables all).

The patches are small and listed with file:line justifications in [../patches/PATCHES.md](../patches/PATCHES.md). Examples:
a BF16 TFLOPS value for SM12x so engine construction does not abort (P1), "no NVLink domain" so the SM100 mega-MoE path is
never selected (P2), the MoE layer delegating routed experts to `sm120_moe` (P3), unfused mHC (P4), the fused-attention gate
following what the shim provides (P5), and the DSpark sampler's shared-memory window (P8).

Two optional source patches exist for things that cannot be done from Python: the b12x tiny-decode SwiGLU clamp fix and a
one-line `Sequence.block_table` pybind binding that the prefix cache needs. The serving image applies both at build time.

## 4. What this repo adds on top of DeepInfer

DeepInfer's open-source release is an offline engine. Added here:

| piece | where | docs |
|---|---|---|
| OpenAI-compatible HTTP server (completions, chat, SSE streaming, logprobs) with continuous batching | `server/` | [SERVER.md](SERVER.md) |
| Exact prefix caching built on DeepInfer's own `fork_seq` prefix-hit path | `prefix_cache/` | [PREFIX_CACHE.md](PREFIX_CACHE.md) |
| Engram tables served from host page cache / NVMe instead of RDMA peers | `sm120/sm120_engram.py` | [ENGRAM.md](ENGRAM.md) |
| CUDA-graph safety of all shims, MTP/DSpark on SM120 | `sm120/` | [RUNTIME.md](RUNTIME.md) |
| Expert parallelism over PCIe (DP attention + EP MoE), DP proxy | `sm120/sm120_ep/`, `server/dp_proxy.py` | [MULTI_GPU.md](MULTI_GPU.md) |
| Validation harness against DeepSeek's reference `inference/model.py` with truncated real-weight checkpoints | `tools/` | [VALIDATION.md](VALIDATION.md), [MTP_YOCO.md](MTP_YOCO.md) |
| Endpoint A/B comparison, load generator, quality spot-check | `eval/` | [EVAL.md](EVAL.md) |

## 5. Parallelism on PCIe-only systems

DeepInfer has **no tensor parallelism**. It serves V4.1 as DP attention (each rank is its own engine with replicated attention
weights and its own KV cache) plus expert-parallel MoE, with tokens moved by DeepEP all-to-all over NVLink/RDMA. Other SM120
stacks (b12x's vLLM integration) use TP with all-reduce instead.

Options considered:

| option | content | trade-off |
|---|---|---|
| A. add TP to DeepInfer | column/row split of attention, dense and MoE | proven layout and lower VRAM, but the deepest change to DeepInfer's model and scheduler assumptions |
| **B. DP attention + "EP by all-gather"** (chosen) | all-gather tokens, each rank runs `fused_moe` on its local experts with non-local routes masked to -1, reduce-scatter | close to DeepInfer's design, no custom kernel; replicated attention costs VRAM |
| C. DP attention + b12x `ep_moe` | as B with b12x's EP op | W4A16 only, loses the FP8 activation path |

Option B is implemented in `sm120/sm120_ep/`, using DeepInfer's own EP expert sharding at load and its DP-wide step voting,
so every rank enters every MoE layer in lockstep. **Status: runs on 4 physical GPUs** ([RESULTS.md](RESULTS.md)); also tested in emulation
(4 in-process ranks, 2 processes on one GPU with gloo, a 1-rank NCCL group captured in a CUDA graph). Expected costs, the transport choice and the open follow-ups (DP max-sync of the decode graph size, b12x PCIe collectives)
are in [MULTI_GPU.md](MULTI_GPU.md); the bring-up procedure is in [MULTI_GPU_RUNBOOK.md](MULTI_GPU_RUNBOOK.md).

Memory budget for the full DeepSeek-V4.1-Flash checkpoint (about 510 GB: 296 GB FP4 routed experts, 203 GB Engram tables,
5.2 GB attention, 2.6 GB embedding + head, 3.2 GB shared experts / MTP / vision / rest) on 4 × 96 GB with DP attention + EP=4:
experts 74 + attention 5.2 + embed/head 2.6 + rest ~2 ≈ 84 GB per GPU, leaving roughly 10 GB for KV cache and activations.
Engram tables do not go on the GPU (section 3). There is no room for EPLB redundant experts
(`DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS=0`). Long contexts are therefore tighter than with a TP layout.

Other consequences of having no NVLink: the mega-MoE (symmetric memory) path and the EPLB symmetric-memory shuffle are off
(patch P2), context parallelism stays at CP=1, and NCCL latency dominates single-user decode collectives.

## 6. Known numerical properties

- **Layer 0 deviates from the reference** by more than the reference's own noise. This is a property of DeepInfer, not of the
  port: DeepInfer's TileLang `fused_transform` rotates the fp32 norm result while the reference rotates the bf16-rounded one, so
  individual FP8 codes of `kv` flip. It reproduces with pure-torch shims, and the kernel matches its own torch reference on SM120.
  Every other layer is within the reference noise floor ([VALIDATION.md](VALIDATION.md)).
- **b12x `fused_moe` is not run-to-run deterministic** (atomic top-k combine). `SM120_MOE_IMPL=torch` (or the server's
  `--deterministic`) gives bit-reproducible outputs; `SM120_MOE_DETERMINISTIC=1` (or `--batch-invariant` in the tools) keeps b12x and gives bit-identical repeats.
- DeepInfer needs `max_model_len >= 16384` (candidate_topk_blocks 2048 × block 8) and loads vision/aligner weights even for
  text-only prompts; the truncated test checkpoints synthesize them.
