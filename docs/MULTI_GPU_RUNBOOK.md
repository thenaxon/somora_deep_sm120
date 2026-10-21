# Runbook: full DeepSeek-V4.1-Flash on a 4-GPU PCIe SM120 host

This is the procedure for running the serving image on a host with 4 SM120 GPUs and no NVLink (for example
4x RTX PRO 6000 Blackwell Max-Q, 96 GB each) and comparing it against an existing baseline such as a vLLM
deployment of the same model.

Measured results on such a host are in [RESULTS.md](RESULTS.md). Do the validation steps in §4
before trusting any numbers, and start eager (`--no-graphs`).

If a baseline (e.g. a vLLM deployment with TP=4) occupies all four GPUs, both cannot run at the same time. Plan a
maintenance window, stop the baseline container (do not remove it or its image), and start it again afterwards.

## 0. Preflight (read-only, safe while the baseline runs)

```bash
nvidia-smi                       # driver >= 595 (CUDA 13.2) for the cu132 image
nvidia-smi topo -m               # PCIe P2P paths
free -g; nproc; df -h            # host RAM decides the engram mode (§3); ~10 GB disk for the image + build cache
```

If a baseline container is running, `docker inspect <baseline> --format '{{json .Mounts}} {{json .HostConfig.PortBindings}}'`
shows its checkpoint path and port.

## 1. Image

Build on the host (about 6 minutes with a warm network; the finished image is about 7 GB):

```bash
git clone <this repo> somora_deep_sm120 && cd somora_deep_sm120
docker build -t somora_deep_sm120:0.2.1 -f docker/Dockerfile .
```

If the host has no access to the repository (e.g. no GitHub credentials for a private checkout), clone the tag on
another machine and copy the tree over, or build elsewhere and transfer the image:
`docker save somora_deep_sm120:0.2.1 | ssh <host> docker load`. The build itself only needs network access to
PyPI, the PyTorch wheel index and GitHub (for the pinned DeepInfer commit).

## 2. Checkpoint and mounts

- Mount the HF checkpoint dir of DeepSeek-V4.1-Flash read-only (about 510 GB: shards 00001-00048, including the
  engram tables 00047/00048 at 203 GB). If a baseline already uses it, mount the same host path.
- `-v /models/DeepSeek-V4.1-Flash:/models/ds41:ro`, `-v /srv/sm120/cache:/cache` (JIT, autotune and TileLang caches;
  the first start compiles a lot), `-v /srv/work:/work` (eval outputs).
- **Hugging Face hub cache:** if the checkpoint lives in an HF hub cache (`.../hub/models--.../snapshots/<rev>/`), the
  files in the snapshot dir are symlinks into `.../hub/models--.../blobs/`. Mounting only the snapshot dir leaves
  dangling links (`No such file or directory` for `tokenizer.json` or the shards). Mount the whole cache root
  read-only **at the same path** and pass the snapshot path to `--model`, e.g.
  `-v /data/hf-cache:/data/hf-cache:ro ... --model /data/hf-cache/hub/models--<org>--<name>/snapshots/<rev>`.
  The same applies to `--tokenizer-dir` of the eval tools.
- No conversion step: DeepInfer loads the HF shards directly, and the engram tables are mmapped, not loaded.

## 3. Engram mode (by free host RAM; see [ENGRAM.md](ENGRAM.md) §7)

The two engram tables are about 203 GB (189 GiB). What counts is the memory left **after** the 4 ranks have started:
each rank holds its own CUDA context, pinned staging, the 2 GB multimodal pool and loader buffers (budget about
10 GB per rank, about 40 GB for 4). The server logs `MemAvailable` together with the tables' page-cache residency
at start-up and again after the post-load step (`[sm120_engram] engram tables resident in the page cache ...`).

| MemAvailable after start-up | mode | flags |
|---|---|---|
| >= tables + 10 % (about 225 GB; i.e. about 265 GB free before start-up) and room to page-lock them | `uva`: one pinned copy, pure GPU gathers (untested) | `-e SM120_ENGRAM_MODE=uva --ulimit memlock=-1` |
| >= tables + 10 % (about 225 GB) | `host` + prefault: once all ranks have loaded their weights, rank 0 drops the weight shards' page-cache pages and reads both tables into the page cache | `-e SM120_ENGRAM_MODE=host -e SM120_ENGRAM_PREFAULT=1` |
| below that (e.g. 246 GB total RAM: about 190 GB after start-up) | **`uring`** (O_DIRECT reads per lookup batch, no page-cache thrash, flat latency) | `-e SM120_ENGRAM_MODE=uring -e SM120_ENGRAM_THREADS=16 --security-opt seccomp=unconfined --ulimit memlock=-1` |

`host` without prefault also works below the threshold (page cache as LRU, NVMe serves misses). Two defaults make
the cold case cheap in both modes:
- The scale planes are kept in the page cache (5.7 GiB for both tables, one copy per host). A cold lookup is then one
  NVMe read instead of two.
- Each rank reads a prompt's rows ahead from request admission on, overlapping the queue wait and the compute of
  earlier chunks (ENGRAM.md §3.3). Budget about 6 GiB of page cache per rank for that (`SM120_ENGRAM_PREFETCH_AHEAD`,
  32768 tokens).

On the RTX 5090 test machine a cold 32k-token prompt now costs 0.36 s more than without engram (it was 2.9 s), and
`uring` prefills the same way (its prefill batches read through the mmap). `uring` keeps the lower decode latency on a
cold page cache. `uring` needs io_uring, which
Docker's default seccomp profile blocks ("io_uring initialization failed: Operation not permitted"), and locked
memory for its registered buffers ("Cannot allocate memory" without `--ulimit memlock=-1`).

All 4 rank processes map the same files, so the page cache holds one copy.

- Count free RAM **with the baseline stopped**: a running baseline may pin its own copy of the engram tables.
- The 4 rank processes also need host memory of their own (CUDA contexts, pinned staging buffers, the loader).
  Budget roughly 5-10 GB per rank and check `free -g` after start-up.
- Prefaulted pages are ordinary page cache, not locked memory: under memory pressure the kernel evicts them and
  misses are read from NVMe again (slower, but no OOM). This also holds without swap. `uva` instead page-locks the
  whole 203 GB and needs that much RAM on top of everything else.
- **Multimodal pool (pinned host RAM per rank, on top of the engram budget above).** Each engine process pins a
  `DEEPINFER_MULTIMODAL_POOL_SIZE_GB` buffer (`multiprocessing.shared_memory`, `/dev/shm/psm_*`). The multimodal
  worker processes use it to hand preprocessed images to the engine. DeepInfer's default is 8 GB, i.e. 32 GB pinned
  for 4 ranks. The image sets **2 GB** (8 GB for 4 ranks).
  - What it holds: one block per in-flight request, with all of the request's resized RGB images plus 4 bytes per
    prompt token. The block is freed once the prefill has copied it to the GPU. A full-resolution image (1024 vision
    tokens, about 1302×1302 px) takes about 5.1 MB, the smallest (`min_pixels`, 544×544) about 0.9 MB. 1 GB is about 200
    full-resolution images in flight.
  - When it is full, the worker waits for space. A single request whose images exceed the whole pool waits until
    `DEEPINFER_MULTI_MODAL_WORKER_TIMEOUT` (1800 s) aborts the process. So size it to at least your largest request's
    images: 2 GB covers about 400 full-resolution images.
  - `DEEPINFER_USE_MULTIMODAL_POOL=0` removes the pinned buffer; images then travel through the worker queue (same
    tokens on the test model, slower copies). Text-only serving does not use the pool.
  - Image inputs come in through the OpenAI server's `image_url` content parts (see [SERVER.md](SERVER.md)). The vision
    encoder on SM120 matches DeepSeek's reference within the bf16 noise floor on a truncated checkpoint with the real
    vision weights; whether the full model answers questions about images is checked by the `image` items of the
    quality set.
  - No `/dev/shm/psm_*` leftovers. Patch P11 unlinks the segment's name right after DeepInfer forks its multimodal
    workers, so the memory exists only while a process maps it, also after SIGKILL, OOM or `docker kill`. The server
    handles SIGTERM (`docker stop`; the server is PID 1 through `exec` in the entrypoint) with a clean shutdown, and
    the `--dp` proxy forwards it to its ranks. Before v0.1.1 every stopped engine left a 2 GB `psm_*` file in the
    host's `/dev/shm` (with `--ipc=host`); delete leftovers from older runs by hand.
- **NUMA:** the image has libnuma1. On a multi-socket host add `--cap-add SYS_NICE`, otherwise the memory policy part of
  DeepInfer's binding is refused (CPU affinity still applies). Single-socket hosts skip the binding (patch P10).

## 4. Validate the multi-GPU path first

```bash
docker run --rm --gpus all --ipc=host --shm-size=32g --entrypoint bash somora_deep_sm120:0.2.1 -c '
  nvidia-smi topo -m
  torchrun --nproc_per_node 4 /opt/sm120/tests/bench_ep_comm.py --out /tmp/bench_ep_comm.json --moe
  python3 /opt/sm120/tests/test_ep_moe.py'
```

Then follow the remaining steps of [MULTI_GPU.md](MULTI_GPU.md) §6 (W=4 NCCL test, DeepInfer EP=4 vs single-GPU).

## 5. Launch

```bash
docker run -d --name sm120-serve --gpus all --ipc=host --shm-size=32g --ulimit memlock=-1 -p 8100:8000 \
  -v /models/DeepSeek-V4.1-Flash:/models/ds41:ro -v /srv/sm120/cache:/cache -v /srv/work:/work \
  -e SM120_EP_IMPL=allgather -e SM120_EP_TRANSPORT=nccl -e SM120_ENGRAM_MODE=<per §3> \
  somora_deep_sm120:0.2.1 --model /models/ds41 --served-model-name deepseek-v4.1-flash \
    --dp 4 --gpu-memory-utilization 0.92 --max-model-len 40960 --max-num-seqs 16 --max-num-batched-tokens 4096 \
    --max-blocks 4096 --prefix-cache --no-graphs        # drop --no-graphs once eager works
docker logs -f sm120-serve     # 4 ranks start (one engine process per GPU, NCCL EP group), then the dp proxy serves
curl -s localhost:8100/health | python3 -m json.tool
```

- NCCL P2P: the dp proxy self-tests peer copies and then starts the ranks with `NCCL_P2P_LEVEL=SYS` (without it,
  NCCL 2.29 routes all EP traffic through host memory on hosts whose GPUs hang off different root ports). Add
  `-e NCCL_DEBUG=INFO` once and check that the channel lines say "via P2P" ([MULTI_GPU.md](MULTI_GPU.md) §4a). If P2P
  is not usable, `-e SM120_EP_TRANSPORT=b12x -e SM120_EP_B12X_MAX_ROWS=4096` moves the prefill collectives to b12x's
  CUDA-IPC two-shot kernels (run `tests/test_ep_b12x_transport.py` with torchrun first).
- `--prefix-cache` splits prefills only at multiples of `--max-num-batched-tokens` (v0.2.1); with v0.2.0 add
  `--prefix-cache-grain 4096`.
- `--dp 4` means DP attention + EP MoE (`--ep` defaults to `--dp`). It runs one engine process per GPU on ports
  8001-8004 inside the container, plus a least-loaded proxy on 8000 (`server/dp_proxy.py`). Prefix caches are per rank.
- Memory per GPU (estimate): about 73 GB of routed experts (384/4 per layer x 40 layers) plus about 10 GB replicated
  (attention, mHC, embed/head, vision, MTP). That leaves about 10 GB for KV and activations at `0.92`. If it does not
  start, lower `--max-num-batched-tokens` to 2048 first.
- With CUDA graphs, the DP-wide decode graph-size sync (`SM120_EP_DP_SYNC`, default on) keeps the EP row count at the
  synced graph size. Measure eager first.
- `--nextn 5` (DSpark/MTP) can be tried after the plain run.
- `--max-blocks` caps the KV cache in 128-token blocks. The server default of 256 (32k tokens) is sized for the
  single-GPU test machine; at 4k prompts it queues requests.

## 6. Evaluation (`eval/`, from any machine that reaches the ports)

The tools need the checkpoint's `tokenizer.json` and `encoding/` (`--tokenizer-dir`). See [EVAL.md](EVAL.md).

**Step A: baseline** (before the window, with the baseline running; this adds load to it).

```bash
E=/opt/sm120/eval     # or eval/ in a checkout
python3 $E/compare_endpoints.py --a http://<host>:<baseline-port> --tokenizer-dir <ckpt> --max-tokens 256 --save-a /work/base_cmp.json
python3 $E/loadgen.py   --url http://<host>:<baseline-port> --json /work/base_load.json     # conc 1/4/16/64 x prompt 512/4k/32k, out 256
python3 $E/quality_check.py --url http://<host>:<baseline-port> --exec-code --json /work/base_quality.json
```

**Step B: this image** (in the window, serving on 8100).

```bash
python3 $E/compare_endpoints.py --a http://<host>:8100 --b file:/work/base_cmp.json --tokenizer-dir <ckpt> --max-tokens 256 --json /work/cmp.json
python3 $E/loadgen.py   --url http://<host>:8100 --json /work/sm120_load.json
python3 $E/quality_check.py --url http://<host>:8100 --exec-code --json /work/sm120_quality.json
```

**Step C: close the window:** `docker stop sm120-serve && docker rm sm120-serve`, start the baseline again and check
that it answers.

Reading the results:

- **compare:** both sides get identical token ids. Expect exact or long common prefixes; every first divergence should
  be a near-tie (`margin < 0.1`), and chosen-token logprob deltas should be about 1e-2. A divergence that is not at a
  near-tie, within the first tokens, points at a numerics bug. vLLM reports raw logprobs; this server's are raw-logit
  logprobs too (`server/logprob_tap.py`). The comparison sends `return_tokens_as_token_ids`, `ignore_eos` and
  `logprobs` (up to 20), which the baseline must accept (check in step A).
- **quality:** both should pass (nearly) all 14 items, including the 8k-token needle.
- **loadgen:** compare TTFT/TPOT p50/p99 and tok/s per level. The 32k prompts need `--max-model-len` > 32768+256 on
  both sides; otherwise that level is skipped.
