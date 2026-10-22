# somora_deep_sm120

Part of the [somora](https://somora.ai) family.

DeepSeek's [DeepInfer](https://github.com/deepseek-ai/DeepInfer) inference engine on NVIDIA **SM120** GPUs (the
Blackwell workstation generation). It runs the full **DeepSeek-V4.1-Flash** on **4× RTX PRO 6000 Blackwell** over
PCIe, behind an OpenAI-compatible server with tool calling, structured output and image input.

DeepInfer targets SM100 (B200/GB200): its hot path depends on FlashMLA, DeepGEMM, DeepEP and DeepSelect, which do not
run on SM120. This repo replaces those four libraries with drop-in Python packages backed by
[b12x](https://pypi.org/project/b12x/) SM120 kernels, adapts the multi-GPU path to PCIe (expert parallelism over NCCL
instead of NVLink), and applies a small set of runtime patches at start-up. The model code, the scheduler, the KV
cache, sampling and speculative decoding (DSpark) are DeepInfer's own and unmodified: the drop-ins and patches sit in
front of it on `PYTHONPATH`, plus a one-line patch for a read-only binding the prefix cache needs. The GPU kernels
are b12x's, not DeepSeek's; the numerics were checked layer by layer against DeepSeek's reference implementation.

As far as we know this is the first port of DeepInfer to SM120 workstation GPUs.

> **What you need for the full model**
> - **4× RTX PRO 6000 Blackwell (96 GB each)**. PCIe is enough, NVLink is not needed. The weights take about 80 GiB per
>   GPU, so fewer or smaller GPUs do not work; **a single RTX 5090 cannot run the full model**.
> - **About 256 GB of host RAM**: the engram tables (189 GiB) are kept in RAM, and the recommended mode needs about
>   225 GiB free before start ([docs/MULTI_GPU_RUNBOOK.md](docs/MULTI_GPU_RUNBOOK.md) §3 lists modes for less RAM).
> - Linux, Docker with the NVIDIA container toolkit, NVIDIA driver >= 595 (CUDA 13.2), the DeepSeek-V4.1-Flash
>   checkpoint (~300 GB, not included).

> **Status: experimental (v0.3.0).** Stable on 4× RTX PRO 6000, quality on par with a tuned vLLM deployment of the
> same model. With many concurrent users and long prompts it matches vLLM's throughput and reaches the first token
> sooner; a single user gets about half of vLLM's decode speed, and a single long prompt is read about 5–6× slower
> ([docs/RESULTS.md](docs/RESULTS.md)). If you want the fastest single-user serving on this hardware, use vLLM. This
> project is for running DeepSeek's own engine on it. Not affiliated with DeepSeek or NVIDIA.

## Hardware

| GPU | status |
|---|---|
| RTX PRO 6000 Blackwell (Max-Q), 4× over PCIe, no NVLink | **tested** with the full model (`--dp 4`): [docs/RESULTS.md](docs/RESULTS.md) |
| GeForce RTX 5090 (SM120, 32 GB) | development and validation with **truncated test checkpoints** only (a few layers or experts of the real weights); it cannot hold the full model |
| other SM120 / SM121 | untested |

Pinned versions are listed in [COMPATIBILITY.md](COMPATIBILITY.md).

## Quickstart (4× RTX PRO 6000)

```bash
git clone https://github.com/thenaxon/somora_deep_sm120 && cd somora_deep_sm120
docker build -t somora_deep_sm120:0.3.0 -f docker/Dockerfile .      # ~6 min; clones DeepInfer at the pinned commit

docker run -d --name sm120-serve --gpus all --ipc=host --shm-size=32g --ulimit memlock=-1 -p 8000:8000 \
  -v /models/DeepSeek-V4.1-Flash:/models/ds41:ro -v sm120-cache:/cache \
  -e SM120_EP_IMPL=allgather -e SM120_ENGRAM_MODE=pinned -e DEEPINFER_MULTIMODAL_POOL_SIZE_GB=1 \
  somora_deep_sm120:0.3.0 --model /models/ds41 --served-model-name deepseek-v4.1-flash \
    --dp 4 --nextn 7 --prefix-cache --gpu-memory-utilization 0.95 --max-model-len 40960 \
    --max-num-seqs 16 --max-num-batched-tokens 4096 --max-blocks 4096
docker logs -f sm120-serve      # first start: weights + engram tables load, kernels compile (several minutes)

curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 32}'
```

- Mount a volume on `/cache`: the first start compiles many kernels (b12x, TileLang, Triton), and later starts reuse them.
- Read [docs/MULTI_GPU_RUNBOOK.md](docs/MULTI_GPU_RUNBOOK.md) before the first start: preflight checks, engram modes by
  free RAM, NCCL P2P, what the start log should show, and how to evaluate against another endpoint.
- `--help` lists all options. CUDA graphs are on by default (`--no-graphs` turns them off).
- The weights are not included. Use your own copy of the checkpoint, which is under its own license.
- Development on one GPU: `tools/ref/make_truncated_ckpt.py` builds smaller test checkpoints from the real weights
  ([tools/ref/README.md](tools/ref/README.md)). They test plumbing and numerics; their output text is meaningless.

More in [docker/README.md](docker/README.md) and [docs/SERVER.md](docs/SERVER.md).

## Features

- **Drop-in kernels** (`sm120/`): `flash_mla`, `deep_gemm`, `deep_ep` and `deep_select` packages with DeepInfer's
  APIs. They map to b12x (sparse MLA decode/prefill on the byte-identical `deepseek_v41` KV format, MXFP4 indexer,
  FP8 block GEMM, W4A8 fused MoE), to Triton kernels (mHC einsum, KV encode), or to exact torch references.
  `SM120_*_IMPL=torch` switches each family to the reference to bisect problems.
- **Runtime patches** (`sm120/sm120_patches.py`, installed by `sitecustomize`), each listed and justified in
  [patches/PATCHES.md](patches/PATCHES.md).
- **CUDA graphs** for decode, including MTP/DSpark speculative decoding ([docs/RUNTIME.md](docs/RUNTIME.md)).
- **Engram** without RDMA: the n-gram embedding tables (189 GiB) are kept once in pinned host RAM for all GPUs and
  gathered by the GPUs directly, also inside CUDA graphs (`SM120_ENGRAM_MODE=pinned`); modes for hosts with less RAM read
  them from the page cache or the NVMe ([docs/ENGRAM.md](docs/ENGRAM.md)).
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

### Full model, 4× RTX PRO 6000 Blackwell Max-Q, against a tuned vLLM (TP=4) on the same machine

| | this engine | vLLM, TP=4 |
|---|---|---|
| quality, 28-item set (core / tool / json / image) | 27/27 | 26/27 |
| 4k prompts × 64 users: output tok/s, median time to first token | 420 tok/s, 7.9 s | 425 tok/s, 33.8 s |
| 4k prompts × 16 users, output tok/s | 388 | 417 |
| real prompts, 1 user: code / chat, output tok/s | 101 / 67–74 | 209 / 148–157 |
| real prompts, 16 users: code, output tok/s | 951 | 1,200 |
| time to first token, new 4k / 32k prompt | 1.4 s / 11 s | 0.24 s / ~2 s |

Measured with `--nextn 7` and `SM120_ENGRAM_MODE=pinned`; the time to first token with the 0.3.0 prefill defaults.
Stable in all runs (CUDA graphs, DSpark, clean shutdown). Setup, progress since v0.2.0, mixed load and where the time
goes: [docs/RESULTS.md](docs/RESULTS.md).

### Development GPU: one RTX 5090, truncated test checkpoints (not the full model)

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

- **Single-user speed is behind a tuned vLLM TP=4 deployment**: about half the decode speed, and a single long
  prompt is read about 5–6× slower. DP attention reads each request's prompt on one GPU; tensor parallelism for the
  attention and dense layers is tracked in [#1](../../issues/1).
- **Mixed load:** DeepInfer cannot mix decode tokens into a prefill step, so while a long prompt is read the other
  users' streams pause for each 4096-token chunk (TPOT p99 65 ms vs vLLM's 25 ms; [docs/RESULTS.md](docs/RESULTS.md)).
- **Memory is tight on 4× 96 GB:** the recommended `SM120_ENGRAM_MODE=pinned` needs `--gpu-memory-utilization 0.95`
  (at 0.92 no KV cache is left) and about 225 GiB of free host RAM before start.
- `--yoco-compact` (opt-in, about 1.15–1.2× faster prefill at 32k) cannot be combined with `--prefix-cache`.
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
docker run --rm --gpus all --ipc=host --shm-size=16g --entrypoint bash somora_deep_sm120:0.3.0 -c '
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
- [vLLM](https://github.com/vllm-project/vllm) and the community's DeepSeek-V4.1 work on RTX PRO 6000: several ideas
  (pinned engram tables gathered by the GPU, the evaluation scenarios) come from studying how it runs this model.

A community project of [somora](https://somora.ai). See [NOTICE](NOTICE). Licensed under the [Apache License 2.0](LICENSE).
