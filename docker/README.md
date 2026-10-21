# Docker images

| file | image | what |
|---|---|---|
| `docker/Dockerfile` | `somora_deep_sm120:0.2.0` | **self-contained serving image**: DeepInfer (pinned commit) + SM120 shims + OpenAI server + prefix cache; the entrypoint starts the server |
| `docker/Dockerfile.dev` | `somora_deep_sm120:dev` | dev/test image: the same base and DeepInfer build, but the repo is **mounted** at `/opt/sm120` instead of copied |
| `docker/entrypoint.sh` | | entrypoint of the serving image (`/opt/sm120/docker/entrypoint.sh`) |

## Build

The build context is the repo root:

```bash
docker build -t somora_deep_sm120:0.2.0 -f docker/Dockerfile .
# a different DeepInfer commit (untested):
docker build -t somora_deep_sm120:custom --build-arg DEEPINFER_COMMIT=<sha> -f docker/Dockerfile .
```

The DeepInfer source is not part of this repo. The Dockerfile fetches `https://github.com/deepseek-ai/DeepInfer` at
`ARG DEEPINFER_COMMIT=8b16149741a4935e1dec8cea27853f273b374c5b` (without the vendored SM100 kernel submodules, which SM120
does not use), so the build needs network access to GitHub and PyPI.

What the image contains:

- Base: `nvidia/cuda:13.2.1-devel-ubuntu24.04`, Python 3.12 venv, torch 2.13.0+cu132, DeepInfer's requirements (transformers
  bumped to 4.51.3 for b12x), tilelang 0.1.15, tile-kernels 2.1.0, b12x 1.5.1. These layers are identical in `Dockerfile` and
  `Dockerfile.dev`, so building one reuses the other's cache. Versions: [../COMPATIBILITY.md](../COMPATIBILITY.md).
- `liburing-dev`, `pkg-config` (b12x O_DIRECT io_uring row reader, engram `SM120_ENGRAM_MODE=uring`), `patch`, and `libnuma1`
  (DeepInfer's NUMA binding; on single-node hosts runtime patch P10 skips the binding, which would otherwise segfault).
- **b12x tiny-decode SwiGLU fix** applied to the installed b12x (`patches/b12x-tiny-decode-swiglu.patch`,
  upstream issue [flashinfer-ai/flashinfer#6866](https://github.com/flashinfer-ai/flashinfer/issues/6866)). The default still uses
  b12x's faster dynamic path (`SM120_MOE_TINY_DECODE=0`).
- **DeepInfer `_C` built with the `Sequence.block_table` binding** (`patches/deepinfer-sequence-block-table.patch`), which the
  prefix cache needs.
- The repo at `/opt/sm120`. `/opt/sm120/sm120` holds the drop-in FlashMLA / DeepGEMM / DeepEP / DeepSelect replacements on b12x
  plus the runtime patches (`sitecustomize` → `sm120_patches`); `server/`, `prefix_cache/`, `tools/`, `eval/`, `tests/` are there too.

Environment defaults (override with `-e`):

| var | value | why |
|---|---|---|
| `PYTHONPATH` | `/opt/sm120/sm120:/opt/deepinfer` | the shims shadow the SM100-only vendor libs |
| `DEEPINFER_ROOT` | `/opt/deepinfer` | `examples/_config.py` (engine construction) |
| `SM120_CACHE_ROOT`, `DEEPINFER_JIT_CACHE_DIR`, `TILELANG_CACHE_DIR`, `TRITON_CACHE_DIR` | `/cache`, `/cache/jit`, `/cache/tilelang`, `/cache/triton` | mount a volume on `/cache` to keep JIT and autotune results across runs (the first start compiles a lot) |
| `B12X_AUTOTUNE` | 0 | deterministic start-up; `SM120_B12X_AUTOTUNE=1` opts in |
| `SM120_MOE_TINY_DECODE` | 0 | dynamic MoE path (faster on SM120, [../docs/RUNTIME.md](../docs/RUNTIME.md)) |
| `SM120_ENGRAM_MODE` | `host` | engram tables mmapped from the checkpoint ([../docs/ENGRAM.md](../docs/ENGRAM.md)) |
| `DEEPINFER_PREFILL_ATTEMPT_INTERVAL` | 1 | DeepInfer's default 70 costs about 2 s TTFT ([../docs/SERVER.md](../docs/SERVER.md)) |
| `DEEPINFER_MULTIMODAL_POOL_SIZE_GB` | 2 | pinned host buffer per engine (rank) for preprocessed images, DeepInfer's default is 8; sizing in [../docs/MULTI_GPU_RUNBOOK.md](../docs/MULTI_GPU_RUNBOOK.md) §3 |
| `DEEPINFER_DISABLE_SYNC_GUARD` | 1 | the shims do CPU syncs |
| `HF_HUB_OFFLINE`, `TOKENIZERS_PARALLELISM` | 1, false | |

## Run

```bash
docker run --rm --gpus all --ipc=host --shm-size=16g -p 8000:8000 \
  -v /models:/models:ro -v /var/cache/deepinfer:/cache \
  somora_deep_sm120:0.2.0 --model /models/DeepSeek-V4.1-Flash --prefix-cache
# entrypoint options: --model PATH (required) --served-model-name NAME --no-graphs (CUDA graphs are on by default)
#   --host 0.0.0.0 --port 8000; everything else goes to server.py:
#   --gpu-memory-utilization 0.9 --max-model-len 16384 --max-num-seqs 8 --max-num-batched-tokens 4096 --prefix-cache
#   [--prefix-cache-grain 512 --prefix-cache-entries 64 --prefix-cache-max-block-frac 0.5] --nextn N --deterministic
docker run --rm somora_deep_sm120:0.2.0 --help
```

The API (`/v1/models`, `/v1/completions`, `/v1/chat/completions` incl. SSE streaming, `/health`) is described in
[../docs/SERVER.md](../docs/SERVER.md).

**Checkpoint symlinks:** truncated test checkpoints link the engram shards instead of copying them (e.g.
`engram-model-00047-of-00048.safetensors -> /ckpt/...`), so mount the original shard directory at the same path
(`-v <hf shards>:/ckpt:ro`) next to the work directory (`-v <work dir>:/work`).

**The full V4.1-Flash does not fit one 32 GB GPU.** It needs multi-GPU expert parallelism, which is implemented but has only
been tested in emulation so far ([../docs/MULTI_GPU.md](../docs/MULTI_GPU.md)). On a single RTX 5090 the smoke test uses the
full-depth test checkpoint `ckpt_full`: all 40 layers, engram, MTP, but only 16/384 routed experts per layer. It needs
`--gpu-memory-utilization 0.95 --max-num-batched-tokens 512`, because DeepInfer's memory profiling over 4096 tokens × 40 layers no
longer fits next to the 23 GiB of parameters.

## Dev image

```bash
docker build -t somora_deep_sm120:dev -f docker/Dockerfile.dev .
docker run --rm -it --gpus all --ipc=host --shm-size=16g \
  -v "$PWD":/opt/sm120 -v <work dir>:/work -v <hf shards>:/ckpt:ro -v <cache dir>:/cache \
  somora_deep_sm120:dev bash
```

The dev image's `/opt/deepinfer` may lack the `Sequence.block_table` binding; `prefix_cache/build_patched_c.sh` builds a patched
copy under `/work` ([../docs/PREFIX_CACHE.md](../docs/PREFIX_CACHE.md)).

## Smoke test (single RTX 5090, full-depth test checkpoint with 16/384 routed experts)

The entrypoint runs on `ckpt_full` with CUDA graphs and `--prefix-cache`
(`--gpu-memory-utilization 0.95 --max-num-batched-tokens 512 --max-num-seqs 4`). The client is `server/bench_client.py`: a 4-turn
streaming chat (each turn = previous messages + answer + a new question, 64 tokens each), then 4 concurrent streaming chats.
The server comes up in about 66 s (load + graph capture).

Warm numbers (second pass in the same container). The first pass after a cold start pays first-call JIT: turn-1 TTFT 8.4 s,
burst TTFT 3.3 s.

| | nextn 0 (graphs) | nextn 5 (graphs, DSpark) |
|---|---|---|
| turn 1 (143 prompt tokens, 0 cached): TTFT / TPOT | 0.315 s / 16.6 ms | 0.310 s / 5.7 ms |
| turn 2 (225 tokens, 206 cached) | 0.341 s / 16.6 ms | 0.342 s / 4.9 ms |
| turn 3 (309 tokens, 288 cached) | 0.339 s / 16.6 ms | 0.341 s / 4.9 ms |
| turn 4 (395 tokens, 372 cached) | 0.341 s / 16.8 ms | 0.339 s / 5.7 ms |
| 4 concurrent chats × 64 tokens: throughput / median TTFT / median TPOT | 147 tok/s / 0.339 s / 22.2 ms | 342 tok/s / 0.340 s / 6.0 ms |
| server start (load, profile, graph capture) | 29-66 s | 60 s |

- Single-stream decode is 16.6 ms/token (60 tok/s) on this reduced-expert model. The real model streams far more expert weights
  per token; these numbers are not a prediction for it.
- DSpark's 5 ms/token is optimistic: the reduced-expert model's output is repetitive, so drafts are accepted unusually often.
- TTFT stays at about 0.34 s whether 0 or 372 prompt tokens are cached. At these short prompts it is dominated by the fixed cost
  of one prefill step plus scheduling, not by the prefill length. With 1-2k-token prefixes, hits cut TTFT 2-3.5×
  ([../docs/PREFIX_CACHE.md](../docs/PREFIX_CACHE.md)).
- Prefix-cache stats after both passes: 16 lookups, 6 hits, 1732/2290 prompt tokens served from the cache (76 %), 9 generation-end
  snapshots.
- TTFT counts from sending the request to the first streamed content chunk. TPOT = (last chunk − first chunk) / (tokens − 1).
  "cached" = prompt tokens served from the prefix cache; from turn 2 on, almost the whole prompt hits the generation-end snapshot
  of the previous turn.
- The outputs are not meaningful text, because the experts are reduced (16/384).
