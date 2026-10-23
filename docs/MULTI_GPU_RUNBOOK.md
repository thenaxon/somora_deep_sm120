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
docker build -t somora_deep_sm120:0.3.1 -f docker/Dockerfile .
```

If the host has no access to the repository (e.g. no GitHub credentials for a private checkout), clone the tag on
another machine and copy the tree over, or build elsewhere and transfer the image:
`docker save somora_deep_sm120:0.3.1 | ssh <host> docker load`. The build itself only needs network access to
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

| MemAvailable | mode | flags |
|---|---|---|
| **before** start-up >= tables + (5 GiB + 2.6 GiB registration) per rank + 6 GiB (225 GiB for `--dp 4`) | **`pinned`** (recommended, the way the baseline keeps its tables):<br>- The server (the `--dp` proxy) creates one anonymous shared segment (memfd) of 188.8 GiB before the ranks start. Every rank registers it with `cudaHostRegister` (portable, mapped).<br>- Once all ranks have loaded, rank 0 reads the shards into it (buffered reads, the shard pages dropped behind them, about 1 min at 3.5 GB/s). Every rank's lookups then gather on the GPU straight from that pinned host memory.<br>- One copy for all ranks: two processes registering a 15.7 GiB segment raised Shmem by 15.8 GiB, not 31.<br>- Shared memory without swap is never reclaimed, and the registration pins it: residency is guaranteed, and there is no I/O while serving.<br>- Start-up fails with a message naming the numbers if it does not fit. A failed fill (a read error, or no confirmed fill from rank 0 behind the `--dp` proxy) also stops the start-up with a non-zero exit: partly filled tables are never served | `-e SM120_ENGRAM_MODE=pinned --ulimit memlock=-1 -e DEEPINFER_MULTIMODAL_POOL_SIZE_GB=1` |
| after start-up >= tables + 8 GiB (about 212 GB, tables 188.8 GiB) | `host` + locked page cache: once all ranks have loaded their weights, rank 0 drops the weight shards' page-cache pages, reads both tables into the page cache (sequential reads, about 1 min at 3.5 GB/s) and `mlock`s them. The locked pages are the shared page cache of the shard files, so the 4 ranks share one locked copy (measured: two processes locking the same 16 GiB raise `Mlocked` by 16 GiB, not 32). The kernel cannot evict them; lookups never do I/O. Start-up fails with a clear message if the tables plus `SM120_ENGRAM_PIN_HEADROOM_GB` (8) do not fit | `-e SM120_ENGRAM_MODE=host -e SM120_ENGRAM_PIN=mlock --ulimit memlock=-1 -e DEEPINFER_MULTIMODAL_POOL_SIZE_GB=1` |
| >= tables + 4 GiB | `host` + prefault (best effort: page cache, no lock; process growth evicts table pages) | `-e SM120_ENGRAM_MODE=host -e SM120_ENGRAM_PREFAULT=1` (0.2.1: `=force`) |
| below that | `uring` (O_DIRECT reads per lookup batch, flat latency) | `-e SM120_ENGRAM_MODE=uring -e SM120_ENGRAM_THREADS=16 --security-opt seccomp=unconfined --ulimit memlock=-1` |

`uva` (pinning through `cudaHostRegister`) does not work with the shard mmaps: registering read-only file pages fails
with `cudaErrorNotSupported` (measured on the RTX 5090 test machine), so `mlock` is the way to pin.

**Memory budget for `--dp 4` on a 246 GiB host without swap.**
- In `pinned` mode the tables are 188.8 GiB of shared memory (`Shmem` in `/proc/meminfo`), counted once (with
  `SM120_ENGRAM_PINNED_ALLOC=vmm` driver memory instead, not in `Shmem`).
- The ranks no longer map the shard files, so their page tables stay small: VmPTE 0.03 GiB instead of 0.4.
- Each rank's registration costs host memory besides the segment: the driver's per-page bookkeeping (12 MiB of
  `VmallocUsed` per GiB) and the rank's page tables for the mapping (2 MiB per GiB), both for 4 KiB pages and not
  reclaimable while registered. For the full tables that is 2.6 GiB per rank, 10.4 GiB for `--dp 4` (measured on the
  RTX 5090; the start-up log prints it per rank: "registered the shared segment ... host: page tables +X MiB, driver
  vmalloc +Y MiB"). Only 2 MiB host pages would shrink it, and those need hugetlbfs or shmem THP, i.e. host settings.
- The start-up check uses `SM120_ENGRAM_RANK_HOST_GB=5` per rank plus the registration cost above per rank and
  `SM120_ENGRAM_PIN_HEADROOM_GB=6`: tables + 4 x (5 + 2.6) + 6 = 225 GiB for `--dp 4`.
- **Recommended option (opt-in in 0.3.1, planned as the default for 0.3.2): `-e SM120_ENGRAM_PINNED_ALLOC=vmm`.**
  Measured on the 4-GPU host (v0.3.1-rc1, `--dp 4`, 256 users without errors): `docker stop` 4.9 s instead of 13-14 s,
  attach about 16 s per rank instead of 44 s (the other ranks' 16 s include 12 s waiting for rank 0's allocation),
  VmallocUsed 4.3 instead of 9.1 GiB and PageTables 0.5 instead of 1.6 GiB after the fill (about 5.9 GiB less kernel
  memory that cannot be reclaimed), the fill at 6.3 GB/s. Two things change for monitoring: the tables are no longer
  `Shmem` (a check for "Shmem about 189 GiB" reads about 4 GiB, the multimodal pools; check MemAvailable and the
  "allocated" / "imported" log lines instead), and the start-up check needs 217 GiB instead of 225 GiB.
  Rank 0 allocates the segment once as pinned host
  memory (`cuMemCreate`) and passes it to the other ranks over a Unix socket; they map it for their GPU only. The
  driver's bookkeeping then exists once and the other ranks need no CPU page tables for it: about 2.6 GiB more
  MemAvailable per extra rank, 7.8 GiB for `--dp 4` (RTX 5090: a second rank's attach costs 0 MiB of vmalloc and page
  tables instead of 193 + 32 MiB for 16 GiB). The start-up check then counts the registration once: tables + 4 x 5 +
  2.6 + 6 = 217 GiB for `--dp 4`. Lookups and GPU memory are unchanged (same gather speed inside CUDA graphs, rows
  bit-exact, still about 0.4 GiB of GPU memory per rank, so keep `--gpu-memory-utilization 0.95`). The tables no
  longer show up as `Shmem` but as a MemAvailable drop. Process exit frees them; nothing is left in `/dev/shm`, also
  after SIGKILL. If rank 0 cannot allocate it, every rank falls back to the memfd registration (log line "vmm failed
  ... registering the memfd instead"; the start-up stops instead if MemAvailable cannot take the fallback). The log
  says "allocated (cuMemCreate ...)" on rank 0 and "imported (cuMemCreate) ... (after X s waiting for rank 0)" on the
  other ranks; their page tables and vmalloc are counted from the receipt of the allocation (VmallocUsed is
  system-wide: ranks attaching at the same time still see each other's). An import that fails after rank 0's
  allocation stops the start-up ("unset SM120_ENGRAM_PINNED_ALLOC"); only a failed allocation falls back.
- After the fill the server prints where the host memory is ("read ... GiB of engram tables ... MemAvailable X GiB
  (Shmem, AnonPages, VmallocUsed, PageTables, page cache, SUnreclaim)"). On the 4-GPU host about 14 GiB stay
  available after the fill: about 232 GiB before start - 188.8 GiB tables - 4 x (about 4.5 GiB per rank + 2.6 GiB
  registration). The fill itself holds nothing afterwards (it reads straight into the segment and drops the shard
  pages behind it), and the shard page cache that remains is reclaimable and counted in MemAvailable.
- **GPU memory:** the registration also costs GPU memory for the driver's page tables of the mapping, about 2 MiB per
  GiB of tables: 0.34 GiB per rank for the full tables (measured on the 4-GPU host: non-torch memory 3.73 GiB vs 3.39
  with `host` + `mlock`; 30 MiB for 15.7 GiB on the RTX 5090). DeepInfer sizes the KV cache from the free GPU memory, so
  that comes out of the KV cache. If the KV budget is already tight (`kv_cache_size` in the "Memory profiling results"
  line close to 0), raise `--gpu-memory-utilization` (e.g. 0.92 -> 0.95) or lower `--max-model-len`; otherwise start-up stops
  with "No available memory for the cache blocks" plus a pinned-mode hint naming the cost. The start-up log prints the cost per rank ("registered the shared
  segment ... MiB of GPU memory"; device-wide, so ranks sharing one GPU see each other's registrations).

Per rank, measured on the RTX 5090 test machine after load and a few 4k/32k requests:

| | per rank | 4 ranks |
|---|---|---|
| anonymous memory (CUDA runtime, torch, DeepInfer, Python; `MALLOC_ARENA_MAX=2` saves nothing) | 2.2-2.8 GiB | 9-11 GiB |
| multimodal pool (`/dev/shm`, not reclaimable), image default 2 GB | 2.0 GiB, **1.0 GiB with `DEEPINFER_MULTIMODAL_POOL_SIZE_GB=1`** | 8 GiB -> 4 GiB |
| page tables (VmPTE; grows while a rank maps more table pages, up to about 0.4 GiB per rank for 189 GiB) | 0.35-0.4 GiB | 1.5 GiB |
| torch's pinned host cache | 0.07 GiB | 0.3 GiB |
| engram staging (pinned) | 0.05 GiB | 0.2 GiB |
| sum | 5.5 GiB, 4.5 GiB with the 1 GB pool | 22 GiB, 18 GiB |

The server prints this per rank after load (`[sm120_engram] host footprint of this rank (GiB): ...`).
- On the 4-GPU host MemAvailable after load was about 195.6 GiB. Pinning the 188.8 GiB of tables leaves about 6.8 GiB
  free; with the 1 GB pools about 10.8 GiB.
- The rest of the about 50 GiB in use there is the host's own (kernel, GPU driver, other services). A read-only check
  of who holds it (`smem -tk`, `/proc/meminfo`) is worth adding to the next host report.
- `DEEPINFER_MULTIMODAL_POOL_SIZE_GB=0.5` frees another 2 GiB: about 100 full-resolution images in flight per rank.
- Without swap, the locked tables and `/dev/shm` are fixed. A process that grows past the remaining headroom is
  OOM-killed instead of evicting table pages, so keep the headroom at 8 GiB or more and watch
  `SM120_ENGRAM_RESIDENCY_INTERVAL=300` (residency and MemAvailable every 5 minutes) on the first runs.

**NVMe behind a translating IOMMU** (read the facts for a report: `cat /sys/kernel/iommu_groups/*/type`,
`/sys/bus/pci/devices/<nvme>/untrusted`).
- If the model NVMe sits in a group of type `DMA` despite `iommu=pt`, for example behind a USB4/Thunderbolt switch,
  every NVMe completion pays an IOTLB flush under a global spinlock.
- On one 4-GPU host that was about 18 % of a rank's CPU time.
- A 4k chunk's gather took 1.6 s with 16 gather threads plus prefetch, and 0.16 s with 4 threads and no prefetch.

Pinned tables avoid it entirely, because there is no I/O while serving. Without them, use few I/O threads:
`SM120_ENGRAM_THREADS=4`, `SM120_ENGRAM_PREFETCH=0`, `OMP_NUM_THREADS=8`. In `uring` mode,
`SM120_ENGRAM_URING_MMAP_TOKENS=1000000000` keeps prefill on O_DIRECT (one submitter, batched completions).

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
- Prefaulted pages (`SM120_ENGRAM_PREFAULT=1`) are ordinary page cache: under memory pressure the kernel evicts them
  and misses are read from NVMe again (slower, but no OOM). Pinned pages (`SM120_ENGRAM_PIN=mlock`) cannot be
  evicted: one locked copy of 188.8 GiB, which is why the start-up checks the headroom first.
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
    handles SIGTERM (`docker stop`; the server is PID 1 through `exec` in the entrypoint) with an orderly shutdown, and
    the `--dp` proxy forwards it to its ranks. The ranks stop together, a rank still there after 7 s is killed, and
    every stage logs a `shutdown: ...` line (docs/SERVER.md). Most of the time goes into freeing the pinned engram
    tables: after a rank's `os._exit` the kernel and the driver unpin and unmap every 4 KiB page of its
    `cudaHostRegister` of the 188.8 GiB segment (and with a translated IOMMU unmap them page by page). With the default
    memfd registration a `docker stop` of `--dp 4` takes **13-14 s on the 4-GPU host (v0.3.1-rc1), and that is
    expected**: the ranks reach their fast exit within 0.7 s, the teardown then takes 5-11 s per rank (the 7 s kill
    does not shorten it, the kernel finishes it either way), and the proxy frees the memfds in about 2 s. Neither an
    explicit `cudaHostUnregister` before the exit nor unregistering chunks from several threads is faster (measured:
    the driver serializes it). **`--stop-timeout 30` is required** (§5); with docker's default of 10 s the stop ends
    with SIGKILL (exit 137). That loses nothing (in the v0.3.0 acceptance it left no processes, GPU memory or shared
    memory behind), but it is not an orderly stop. With `SM120_ENGRAM_PINNED_ALLOC=vmm` (below) the stop took 4.9 s. Before v0.1.1 every stopped engine left a 2 GB `psm_*` file in the
    host's `/dev/shm` (with `--ipc=host`); delete leftovers from older runs by hand.
- **NUMA:** the image has libnuma1. On a multi-socket host add `--cap-add SYS_NICE`, otherwise the memory policy part of
  DeepInfer's binding is refused (CPU affinity still applies). Single-socket hosts skip the binding (patch P10).

## 4. Validate the multi-GPU path first

```bash
docker run --rm --gpus all --ipc=host --shm-size=32g --entrypoint bash somora_deep_sm120:0.3.1 -c '
  nvidia-smi topo -m
  torchrun --nproc_per_node 4 /opt/sm120/tests/bench_ep_comm.py --out /tmp/bench_ep_comm.json --moe
  python3 /opt/sm120/tests/test_ep_moe.py'
```

Then follow the remaining steps of [MULTI_GPU.md](MULTI_GPU.md) §6 (W=4 NCCL test, DeepInfer EP=4 vs single-GPU).

## 5. Launch

```bash
docker run -d --name sm120-serve --gpus all --ipc=host --shm-size=32g --ulimit memlock=-1 --stop-timeout 30 -p 8100:8000 \
  -v /models/DeepSeek-V4.1-Flash:/models/ds41:ro -v /srv/sm120/cache:/cache -v /srv/work:/work \
  -e SM120_EP_IMPL=allgather -e SM120_EP_TRANSPORT=nccl -e SM120_ENGRAM_MODE=pinned \
  -e DEEPINFER_MULTIMODAL_POOL_SIZE_GB=1 \
  somora_deep_sm120:0.3.1 --model /models/ds41 --served-model-name deepseek-v4.1-flash \
    --dp 4 --nextn 7 --gpu-memory-utilization 0.95 --max-model-len 40960 --max-num-seqs 16 \
    --max-num-batched-tokens 4096 --max-blocks 4096 --prefix-cache
docker logs -f sm120-serve     # 4 ranks start (one engine process per GPU, NCCL EP group), then the dp proxy serves
# recommended configuration on 4x RTX PRO 6000 (v0.3.1). Other engram modes (§3): replace SM120_ENGRAM_MODE.
# A first bring-up can add --no-graphs --nextn 0 to check the eager path; then start as above.
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
  (attention, mHC, embed/head, vision, MTP); measured on the host: 80.1 GiB of weights per GPU. At `0.92` only about
  0.17 GiB was left for the KV cache (1,600 blocks), and `pinned` mode (0.38 GiB of page tables per GPU) left none, so
  the recommended setting is `0.95` (4,096 blocks, the `--max-blocks` cap). If it does not start, lower
  `--max-num-batched-tokens` to 2048 or `--max-model-len` first.
- With CUDA graphs, the DP-wide decode graph-size sync (`SM120_EP_DP_SYNC`, default on) makes every rank replay the
  same graph size in each step. The start-up capture (its eager warm-ups included) uses each graph's own size as EP
  rows (rows = T; all ranks capture the same sizes in the same order), so the collectives carry only the graph's
  rows. `SM120_EP_DP_SYNC=0` is the A/B switch back: no sync, and every graph pads to the fixed capacity (`max_num_seqs` × (1 + nextn) rows per
  rank). Before 0.2.3 the capture always padded; delete a `/cache/decode_perf` profiled by an older version once.
- DSpark/MTP after the plain run: smoke `--nextn 7`, then `--nextn 7 --mtp-varlen`, before any benchmark (§5a).
- `--max-blocks` caps the KV cache in 128-token blocks. The server default of 256 (32k tokens) is sized for the
  single-GPU test machine; at 4k prompts it queues requests.

## 5a. MTP / DSpark on `--dp 4`

Start the server as in §5 with CUDA graphs (no `--no-graphs`) plus the MTP flags below. Speculative decoding has run
with CUDA graphs on one GPU (`--dp 1`). On `--dp 4`, each new combination of flags gets a smoke run first.

**Smoke order.** Run each step once, with a 10-minute start-up timeout, before any benchmark:
1. `--nextn 7` (fixed draft length).
2. `--nextn 7 --mtp-varlen` (adaptive draft length, [MTP_YOCO.md](MTP_YOCO.md) §9).

For each step:
- Wait for `/health` with a timeout instead of an unbounded loop, e.g.
  `timeout 600 sh -c 'until curl -sf localhost:8100/health >/dev/null; do sleep 5; done'`.
- Then send one chat request with `max_tokens` 64 and a concurrent burst of 16. `eval/scenarios.py realistic
  --concurrency 1,16 --min-requests 16` does both.

**Expected log lines** (`docker logs`), per rank, in this order:
- `[sm120_patches] CudaGraphRunner capture done in ...` (main decode graphs)
- with `--mtp-varlen` only:
  - `Decode performance curve profiled in ...s: (bs=1 ms=...) ...`: all ranks replay every graph size in lockstep
    and average the times.
  - Or `Loaded cached decode performance curve` within one hour of a previous start: rank 0 reads the cache in
    `/cache/decode_perf` and broadcasts it, so all ranks agree.
- `[sm120_patches] DFlashCudaGraphRunner capture done in ...` (draft graphs)
- `engine ready` with `'nextn': 7` (and `'mtp_varlen': True`)

**Hang symptom.** The log stops after `CudaGraphRunner capture done` on some or all ranks, `/health` never answers, and
the GPUs sit at 100 % utilization with no progress (ranks waiting in a collective). Stop the container
(`docker stop`; a hung rank is killed after the stop timeout). Keep the `docker logs` of all ranks (each line is prefixed with the
rank's port) and report which rank stopped where. A hang during serving shows as `/health` timing out or TPOT
growing without bound; the same applies.

**Fallback.**
- If `--mtp-varlen` hangs or errors, drop it: fixed `--nextn 7` is the recommended setting anyway.
- If `--nextn 7` itself fails, try `--nextn 3`, the configuration measured before.
- Both are plain server flags; no image change is needed.

**A/B on real text** (after both smokes pass). Same image, same flags otherwise, one server per row:

| run | flags |
|---|---|
| n5 | `--nextn 5` |
| n7 | `--nextn 7` |
| n7v | `--nextn 7 --mtp-varlen` |

- Command per run: `python3 eval/scenarios.py realistic --url http://host:8100 --concurrency 1,16 --max-tokens 256 --label <run> --out results/<run>`.
- Compare `output_tok_s`, `tpot_p50_ms` and `tpot_p99_ms`. Also compare `draft_acceptance` (accepted / verified drafts)
  and `drafts_verified_per_step` (below nextn with `--mtp-varlen`, which trims drafts per step).
- Prefer `--mtp-varlen` as the default only if n7v beats n7 at concurrency 16 without losing at 1. On the single-GPU test
  machine they were within noise at ≤ 8 sequences (MTP_YOCO.md §9).

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
