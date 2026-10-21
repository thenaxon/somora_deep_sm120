# somora_deep_sm120

Part of the [somora](https://somora.ai) family.

Run DeepSeek's [DeepInfer](https://github.com/deepseek-ai/DeepInfer) inference engine on NVIDIA **SM120** GPUs
(GeForce RTX 5090, RTX PRO 6000 Blackwell). DeepInfer itself targets SM100 (B200/GB200): its hot path depends on
FlashMLA, DeepGEMM, DeepEP and DeepSelect, which do not run on SM120. This repo replaces those four libraries with
drop-in Python packages backed by [b12x](https://pypi.org/project/b12x/) SM120 kernels (or exact torch references),
applies a small set of runtime patches at start-up, and adds an OpenAI-compatible server and exact prefix caching.
DeepInfer's own source is not modified: the drop-ins and patches sit in front of it on `PYTHONPATH`. The server's
prefix cache needs one extra read-only C++ binding, added by a one-line patch at image build time.

Target model: DeepSeek-V4.1-Flash (MoE, FP4 experts, compressed sparse attention, mHC, Engram, MTP/DSpark).

> **Status: experimental (v0.2.2).** DeepInfer runs the full DeepSeek-V4.1-Flash on 4× RTX PRO 6000 Blackwell Max-Q
> over PCIe: stable, with quality on par with a tuned vLLM deployment, but not yet competitive on speed (decode about
> 2-3× and prefill about 15× behind as of v0.2.0, see [docs/RESULTS.md](docs/RESULTS.md)). On a single RTX 5090 it is
> validated layer by layer against DeepSeek's reference implementation with truncated test checkpoints. Not affiliated
> with DeepSeek or NVIDIA.

## Hardware

| GPU | status |
|---|---|
| GeForce RTX 5090 (SM120, 32 GB) | **tested**: truncated test checkpoints, all single-GPU results below |
| RTX PRO 6000 Blackwell (Max-Q), 4× over PCIe, no NVLink | **tested** with the full model (`--dp 4`): [docs/RESULTS.md](docs/RESULTS.md) |
| other SM120 / SM121 | untested |

NVIDIA driver >= 595 is required (CUDA 13.2). Pinned versions are listed in [COMPATIBILITY.md](COMPATIBILITY.md).

## Quickstart

```bash
git clone <this repo> somora_deep_sm120 && cd somora_deep_sm120
docker build -t somora_deep_sm120:0.2.2 -f docker/Dockerfile .      # clones DeepInfer at the pinned commit

docker run --rm --gpus all --ipc=host --shm-size=16g -p 8000:8000 \
  -v /models:/models:ro -v sm120-cache:/cache \
  somora_deep_sm120:0.2.2 --model /models/<checkpoint> --prefix-cache

curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model": "<checkpoint>", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 32}'
```

- Mount a volume on `/cache`: the first start compiles many kernels (b12x, TileLang, Triton), and later starts reuse them.
- `--help` lists the options: CUDA graphs are on by default (`--no-graphs` turns them off). The other options are
  `--nextn N` (MTP/DSpark), `--deterministic`, `--gpu-memory-utilization`, `--max-num-batched-tokens` and more.
- The model weights are not included. Use your own copy of the checkpoint, which is under its own license.
  `tools/ref/make_truncated_ckpt.py` builds the smaller test checkpoints that fit one GPU
  ([tools/ref/README.md](tools/ref/README.md)).

More in [docker/README.md](docker/README.md) and [docs/SERVER.md](docs/SERVER.md).

## Features

- **Drop-in kernels** (`sm120/`): `flash_mla`, `deep_gemm`, `deep_ep` and `deep_select` packages with DeepInfer's
  APIs. They map to b12x (sparse MLA decode/prefill on the byte-identical `deepseek_v41` KV format, MXFP4 indexer,
  FP8 block GEMM, W4A8 fused MoE), to Triton kernels (mHC einsum, KV encode), or to exact torch references.
  `SM120_*_IMPL=torch` switches each family to the reference to bisect problems.
- **Runtime patches** (`sm120/sm120_patches.py`, installed by `sitecustomize`), each listed and justified in
  [patches/PATCHES.md](patches/PATCHES.md).
- **CUDA graphs** for decode, including MTP/DSpark speculative decoding ([docs/RUNTIME.md](docs/RUNTIME.md)).
- **Engram** without RDMA: n-gram embedding tables are mmapped from the checkpoint and gathered on the host
  (asynchronously inside the CUDA graph) ([docs/ENGRAM.md](docs/ENGRAM.md)).
- **OpenAI-compatible server**: `/v1/completions` and `/v1/chat/completions` with SSE streaming, cancellation,
  seeded sampling and logprobs ([docs/SERVER.md](docs/SERVER.md)).
- **Tool calling and structured output**: OpenAI `tools` / `tool_choice` / `parallel_tool_calls` (streaming included)
  in the model's own DSML format, and `response_format` `json_object` / `json_schema`. `tool_choice` `required`, a named
  function and `response_format` are grammar-constrained through DeepInfer's xgrammar structured output; tested with the
  official `openai` SDK ([docs/SERVER.md](docs/SERVER.md#tool-calling-and-structured-output)).
- **Image input**: OpenAI `image_url` content parts (base64 `data:` URLs or `http(s)`), run through DeepInfer's
  vision encoder ([docs/SERVER.md](docs/SERVER.md#image-input)).
- **Exact prefix caching**: a cache hit gives the same result as a miss, bit for bit, with the deterministic MoE
  ([docs/PREFIX_CACHE.md](docs/PREFIX_CACHE.md)).
- **Multi-GPU expert parallelism for PCIe**: DP attention + EP MoE over NCCL instead of DeepEP, with a DP proxy in front
  of the per-GPU engines. Runs the full model on 4× RTX PRO 6000 ([docs/RESULTS.md](docs/RESULTS.md), [docs/MULTI_GPU.md](docs/MULTI_GPU.md),
  [docs/MULTI_GPU_RUNBOOK.md](docs/MULTI_GPU_RUNBOOK.md)).
- **Validation tooling** (`tools/`): a runner for DeepSeek's reference implementation, a checkpoint truncator,
  per-layer and intra-layer comparison against the ref-vs-ref noise floor ([docs/VALIDATION.md](docs/VALIDATION.md)).
- Optional **FP8 lm_head** for decode (`SM120_LMHEAD_IMPL=fp8`, off by default) ([docs/ATTENTION.md](docs/ATTENTION.md)).

## Results

### Full model, 4× RTX PRO 6000 Blackwell Max-Q (v0.2.0)

| | this engine | tuned vLLM, TP=4 |
|---|---|---|
| quality, 28-item set (core / tool / json / image) | 27/27 | 26/27 |
| decode, 1 user, real code / chat prompts | 111 / 73 tok/s | 339 / 165 tok/s |
| prefill rate | ~1,000 tok/s | ~16,000 tok/s |
| 4k prompts, 64 users, output | 248 tok/s | 425 tok/s |

Stable in all runs (CUDA graphs, DSpark, clean shutdown). Causes of the slow prefill and the v0.2.1 fixes:
[docs/RESULTS.md](docs/RESULTS.md).

### Single RTX 5090, truncated test checkpoints

All numbers come from **one RTX 5090** with **truncated test checkpoints** built from the real V4.1-Flash weights. The
main one is the *full-depth test model*: all 40 layers, real attention/norm/mHC/shared-expert weights, engram on
layers 1 and 14, the 3 MTP/DSpark heads, but only **16 of 384 routed experts** per layer. The full model streams many
more expert weights per token, so the speed numbers are **not** a prediction for it.

| | result |
|---|---|
| correctness, full-depth test model (327 / 2081-token prompt) | prefill hidden-state error within 3× the ref-vs-ref noise floor on every layer except layer 0 (see limitations); prefill top-1 agreement with the reference 0.927 / 0.949 (reference vs itself with other kernels: 0.933 / 0.953) |
| decode step, CUDA graphs, batch 1 / 8 / 32 (`tests/prof_e2e.py`) | 18.0 / 21.8 / 26.6 ms |
| server, single stream, CUDA graphs (prefix cache on) | TTFT 0.34 s, TPOT 16.6 ms (60 tok/s) |
| server, 4 concurrent streams | 147 tok/s aggregate, median TPOT 22.2 ms |
| MTP/DSpark decode, `bench_e2e.py`, CUDA graphs, B=1 | 58 tok/s (nextn 0) -> 98 / 139 / 152 tok/s (nextn 1 / 3 / 5, acceptance 98 / 77 / 52 %): 2.6x |
| MTP/DSpark decode, B=8 | 388 tok/s (nextn 0) -> 642 / 927 / 909 tok/s (nextn 1 / 3 / 5): 2.4x with nextn 3 |
| deterministic MoE cost (`SM120_MOE_DETERMINISTIC=1`), TPOT B=1 / 8 / 32 | 16.1 / 20.4 / 25.6 -> 46.1 / 42.2 / 47.8 ms (2-2.9x; opt-in for tests and reproducibility) |
| determinism | with `SM120_MOE_DETERMINISTIC=1`, eager and CUDA graphs are bit-identical (17/17 tokens, 0.0 error on every layer); nextn 0 / 1 / 5 decode identical tokens with a deterministic MoE |
| server test suite, 3-layer test model | 22/22 with `--deterministic --prefix-cache` |
| prefix cache hit, 1503-token prompt | TTFT 1.82 s → 0.56 s, identical tokens |

Details and more tables: [docs/VALIDATION.md](docs/VALIDATION.md), [docs/ATTENTION.md](docs/ATTENTION.md),
[docs/RUNTIME.md](docs/RUNTIME.md), [docker/README.md](docker/README.md).

## Known limitations

- **Speed on the 4-GPU target is behind a tuned vLLM TP=4 deployment** (decode about 2-3×, prefill about 15× as of
  v0.2.0). DP attention runs each request's prefill on one GPU; tensor parallelism is tracked in
  [#1](../../issues/1). The full model needs more than one GPU.
- **Layer 0 deviates from the reference** beyond the noise floor (about 12-20× noise, not at other layers). This is a
  known rounding property of DeepInfer's `fused_transform` path, not a port bug ([docs/VALIDATION.md](docs/VALIDATION.md)).
- **b12x `fused_moe` (the default MoE) is not run-to-run deterministic.** The same prompt can decode different tokens
  after a few steps. `--deterministic` / `SM120_MOE_IMPL=torch` is exact but slower, and `SM120_MOE_DETERMINISTIC=1`
  gives a deterministic b12x mode, recommended when reproducibility matters but 2-2.9x slower in decode
  ([docs/VALIDATION.md](docs/VALIDATION.md), [docs/RUNTIME.md](docs/RUNTIME.md)).
- **MTP known issue:** nextn 1 together with `DEEPINFER_SAMPLER_FORCE_TOPK=1024` at batch > 1 accepts no drafts (output
  correct, no speed-up). Other combinations are fine; the server defaults (nextn 0, no forced top-k) are not affected.
- The b12x tiny-decode MoE kernel misses the SwiGLU clamp
  ([flashinfer-ai/flashinfer#6866](https://github.com/flashinfer-ai/flashinfer/issues/6866)). The image patches it, and
  the default uses the dynamic path anyway ([patches/PATCHES.md](patches/PATCHES.md)).
- FP8 lm_head is opt-in: it saves about 0.4 ms per decode step but changes 2-6 % of greedy top-1 picks.
- Eager mode is slow (launch-bound). Use CUDA graphs (the server default). The first start pays a lot of JIT compilation.
- The truncated test models produce meaningless text. They test plumbing and numerics, not output quality.
- Single-node only. There is no PD disaggregation, and DeepInfer's mega-MoE and NVLink paths are disabled on SM120.

## Layout

```
sm120/          PYTHONPATH root: drop-in flash_mla / deep_gemm / deep_ep / deep_select / flash_attn packages,
                sm120_* modules (MoE, attention, engram, EP, lm_head, capture), sitecustomize.py -> sm120_patches
server/         OpenAI-compatible server, engine thread, DP proxy, logprob tap, benchmark client
prefix_cache/   exact prefix caching on top of DeepInfer's block manager
eval/           endpoint A/B comparison, load generator, quality check set
patches/        the two patches applied at image build (b12x clamp fix, DeepInfer Sequence.block_table) + repros
tools/          run_deepinfer.py (offline runner + dumps), bench_e2e.py, compare*.py, diag_repeat.py,
                ref/ (DeepSeek reference runner, checkpoint truncation and conversion)
tests/          unit, graph, EP, engram, server, prefix-cache and end-to-end test scripts
docker/         Dockerfile (serving image), Dockerfile.dev, entrypoint.sh
docs/           design and component documentation
```

Inside the image the repo is at `/opt/sm120`, with `PYTHONPATH=/opt/sm120/sm120:/opt/deepinfer`. Test scripts
use the container paths `/work` (work dir), `/ckpt` (original HF shards, read-only) and `/cache`; override them with
`WORK`, `HF_CKPT` and `SM120_ROOT`. Start with [docs/DESIGN.md](docs/DESIGN.md) for how the pieces fit together.

## Running the tests

```bash
docker run --rm --gpus all --ipc=host --shm-size=16g --entrypoint bash somora_deep_sm120:0.2.2 -c '
  cd /opt/sm120/tests && pytest -q test_attn_shims.py test_mhc_einsum.py test_gemm_moe_graphs.py test_ep_moe.py \
  && python3 test_ep_dp_sync.py && bash run_runtime_unit.sh'
```

Tests that need checkpoints (server, prefix cache, end-to-end, validation) are described in
[docs/VALIDATION.md](docs/VALIDATION.md) and [docs/SERVER.md](docs/SERVER.md).

## Credits

- [DeepInfer](https://github.com/deepseek-ai/DeepInfer) by DeepSeek (Apache-2.0): the engine this repo runs.
- [b12x](https://pypi.org/project/b12x/) (Apache-2.0): the SM120 kernels behind the drop-in packages. See also
  [FlashInfer](https://github.com/flashinfer-ai/flashinfer).
- DeepSeek-V4.1-Flash and its reference implementation by DeepSeek: used for validation. They are not included here.

A community project of [somora](https://somora.ai). See [NOTICE](NOTICE). Licensed under the [Apache License 2.0](LICENSE).
