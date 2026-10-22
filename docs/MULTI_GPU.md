# Multi-GPU on PCIe-only SM120: expert parallelism without DeepEP

Target: a host with several SM120 GPUs and no NVLink, e.g. the 4x RTX PRO 6000 Blackwell Max-Q (PCIe) target.

Status: **runs the full model on 4× RTX PRO 6000 Blackwell Max-Q over PCIe** (results: [RESULTS.md](RESULTS.md)). Originally developed and tested in emulation: The EP MoE layer, the DP-wide decode graph-size sync, the DeepInfer
hooks and the multi-rank server front end (`server.py --dp N`) are in place. They have been tested on one GPU
(in-process emulation, several processes on one GPU over gloo, and a 1-rank NCCL group under CUDA graphs). Run the
steps in §6 on every new host before serving.

## 1. What DeepInfer does on SM100, and what changes

DeepInfer serves V4.1 as **DP attention + EP MoE**:

- Each DP rank is its own engine process with its own scheduler, requests and KV cache; it runs attention only for
  its own tokens.
- MoE layers are expert-parallel. Physical expert slots are split evenly: rank `r` owns `[r*n_local, (r+1)*n_local)`.
- Tokens move between ranks with DeepEP all-to-all `dispatch` / `combine` over NVLink/NVSHMEM (`v4_modules/mlp.py`);
  inside an NVLink domain it uses the fused mega MoE.

On PCIe-only SM120 hosts there is no DeepEP transport, and `PG.is_within_nvl_domain` is patched to False (patch P2),
so the mega path is off.

**"Option B"** replaces dispatch/combine per MoE layer with:

1. **all-gather** of every rank's tokens: bf16 hidden, global physical top-k ids and routing weights, packed into one
   byte row per token (one collective).
2. Each rank runs **its own routed experts** (the unchanged `sm120_moe.B12xRoutedExperts`, b12x `fused_moe` w4a8_mx
   over its `n_local` experts) on all `world x rows` tokens. Every route to a non-local expert is masked to **-1**;
   b12x skips those routes, and tokens without a local route produce exact zeros on every launch path.
3. **reduce-scatter (sum)** of the partial outputs, so every rank gets the full routed output of its own tokens.
4. The shared expert stays rank-local; DeepInfer adds it after the routed output.

Implementation: `sm120/sm120_ep/` (`moe.py`: `EPRoutedExperts`, `wrap(moe)`; `transport.py`; `dp_sync.py`).

## 2. How it plugs into DeepInfer

| concern | how it is handled |
|---|---|
| **Expert sharding at load** | DeepInfer's own: `hugging_face/weights.py:_local_expert_ids` loads only this rank's physical experts, and `DeepseekMoE.num_local_physical_experts = physical // ep_size`. `sm120_moe` builds the b12x layer over exactly those weights. The gate emits global physical ids (EPLB `logical_to_physical_map`), so the local id is `id - ep_rank*n_local` when in range, else -1. EPLB redundant slots work unchanged. |
| **Where the hook sits** | At the MoE layer, not in DeepEP: `sm120_moe.setup` sets `moe.sm120_routed = sm120_ep.wrap(moe)` (a no-op for EP=1 or `SM120_EP_IMPL=off`), and `sm120_moe.forward` passes `is_prompt=input_metadata.is_prompt`. DeepEP `dispatch`/`combine` are never called. The `deep_ep.EPBuffer` shim is an inert object for `world_size > 1` under `SM120_EP_IMPL=allgather`. |
| **Step type across DP ranks** | Already uniform: DeepInfer's C++ scheduler (`csrc/engine/scheduler.hpp`) votes over the DP group on whether all ranks prefill or decode. Idle ranks run a placeholder step, so every rank enters every MoE layer in lockstep. |
| **Different token counts per rank** | Collectives need equal row counts. **Prefill** (eager): the step's per-rank token counts are all-gathered once per step by the DP sync (below; without it once per layer, a host sync each), every rank pads to the maximum, and the local experts run only on the ranks' real rows (padding rows are dropped before the expert call and scattered back as zeros). **Decode** (CUDA graphs): see the DP sync below; without it, a fixed capacity of `SM120_EP_DECODE_ROWS` (default: the layer's `max_num_decoding_tokens`) rows per rank, padding rows masked, no host sync. |
| **DP-wide decode graph-size sync** (`dp_sync.py`, default on) | `Worker.execute_model_step` is wrapped: at the top of every decode step (real or placeholder) the local decode token count is all-reduced with MAX on DeepInfer's CPU DP group (one gloo all-reduce of an int, tens of us). `CudaGraphRunner.select(total_q)` then selects for the synced size, so every rank replays the same graph and the EP rows are simply that graph size. The start-up capture (before any step is synced) uses each graph's own size as rows too, in the eager warm-up before each capture as well (so the warm-up creates the b12x plan buckets the capture needs), since every rank runs the same sizes in the same order. `DFlashCudaGraphRunner.select` (MTP draft graph) is synced the same way. `SM120_EP_DP_SYNC=0` falls back to the fixed capacity. |
| **CUDA graphs** | NCCL collectives are capturable, and DeepInfer captures the same batch-size list in the same order on every rank. The decode path allocates only from the graph pool. |
| **Transport** | `SM120_EP_TRANSPORT=nccl` (default): `all_gather_into_tensor` / `reduce_scatter_tensor` on DeepInfer's EP group (already NCCL). `b12x`: eager prefill collectives through b12x `PCIeTwoShotBF16` (CUDA-IPC peer writes, independent of NCCL's P2P decision), NCCL for decode graphs and everything else; see §4a. `gloo`: CPU staging, eager only, for single-GPU multi-process tests. |
| **Serving** | `server/server.py --dp N [--ep M]` spawns N engine processes (engine_rank r, `data_parallel_size` N, `expert_parallel_size` M, default N), one GPU each, behind `server/dp_proxy.py`, which routes each request to the least-loaded rank. See [SERVER.md](SERVER.md). |

## 3. Tests on one GPU

Setup: random MXFP4 experts in DeepInfer's parameter layout, E=8, K=2048, n=512, top-4, with the real
`B12xRoutedExperts`.

| test | what | result |
|---|---|---|
| `tests/test_ep_moe.py` level 1 | in-process emulation of W=4: sum of the masked local partials of 4 ranks vs the full 8-expert run | cos 0.99998 (T=1) ... 0.999997 (T=300) |
| `tests/test_ep_moe.py` level 2 | **W=2 processes on the same GPU**, gloo, `EPRoutedExperts` end to end. Per-rank counts differ: prefill 37/63, decode 3/12 with capacity 16, and prefill with one rank at 0 tokens. Each rank vs a full-expert run of its own tokens | cos >= 0.99998, max\|d\| <= 0.125 (bf16 partial-sum rounding) |
| `tests/test_ep_moe.py` level 3 | W=1 NCCL group, fixed-capacity path captured in a CUDA graph and replayed with new inputs | cos 0.999996 vs eager |
| `tests/test_ep_capture_plans.py` | 4 gloo processes on one GPU run DeepInfer's capture loop (eager warm-up, then capture, per graph size) with the real `B12xRoutedExperts` plan path; control without `dp_sync.patch_capture` | no plan requested during capture, warm-up and capture use the graph size as EP rows; the control (0.2.3 behaviour) hits the capture-time plan error |
| `tests/test_ep_dp_sync.py` | 2 gloo processes on one GPU patch DeepInfer's real `CudaGraphRunner.select` plus a minimal worker in DeepInfer's step order, with different batch sizes and placeholder steps | both ranks select the same graph size every step, both selects in a step agree, EP output of the real rows matches a full-expert run |

What one GPU **cannot** test: NCCL with more than one rank (it refuses duplicate devices), and DeepInfer itself with
`expert_parallel_size > 1` (it maps one GPU per rank).

## 4. Transport choice and expected cost

Traffic per MoE layer and rank, with R rows per rank and H=5120 (rows are 10.3 KB packed), on 4 GPUs: all-gather
receives 3*R*10.3 KB, reduce-scatter sends/receives 3*R*10 KB.

| rows per rank | payload per layer | ~time at 20-25 GB/s effective PCIe | x 43 MoE layers |
|---|---|---|---|
| 1 (single user) | ~60 KB | latency-bound: 2 collectives x 15-30 us (NCCL) | ~1.5-2.5 ms/step |
| 8 | ~0.5 MB | ~20 us + latency | ~2.5-3.5 ms/step |
| 32 | ~2 MB | ~80 us + latency | ~4.5-5.5 ms/step |
| 192 (fixed capacity at nextn 5 x 32 seqs) | ~12 MB | ~0.5 ms | ~20 ms/step: too slow, which is why the DP sync exists |

These are estimates, not measurements.

- **NCCL latency dominates single-user decode.** `b12x.comm.pcie` has CUDA-IPC collectives for this regime:
  `PCIeTwoShotBF16` (`reduce_scatter` / `all_gather` / `all_reduce`, world size 4 only, fp32 accumulation with one
  bf16 rounding) and `OneshotAllReduce` for small messages. Both need PCIe peer access between the GPUs, a one-time IPC
  setup per process group, and graph capture inside `runtime.capture(plan=...)`. `tests/bench_ep_comm.py` compares
  NCCL and `PCIeTwoShotBF16` (and with `--moe` a full EP MoE layer) on a 4-GPU host. The `b12x` transport
  (§4a) uses it for prefill; decode under CUDA graphs would additionally need `runtime.capture(plan=...)` around
  DeepInfer's capture and stays on NCCL.
- **Halving the all-gather:** an FP8 all-gather of the hidden state would halve that traffic, but needs a b12x entry
  that accepts pre-quantized activations (not found in 1.5.1).
- **Compute side:** each rank streams only 1/W of the expert weights. Attention, mHC, dense GEMMs and lm_head stay
  replicated per rank (DP attention).

## 4a. NCCL P2P and the b12x prefill transport

**Measured on the 4x RTX PRO 6000 host (v0.2.0, NCCL 2.29.7):** every channel between the 4 GPUs ran "via SHM" (host
memory), not P2P, although all pairs report peer access (topology NODE). NCCL's default P2P level only accepts paths
up to PXB (a common PCIe switch), so GPUs behind different root ports of one CPU fall back to SHM. With the
`--prefix-cache-grain 4096` workaround a 4096-token prefill step still took about 4 s there vs about 0.9 s on one
RTX 5090. With `NCCL_P2P_LEVEL=SYS` all 128 channels ran via P2P/CUMEM, but TTFT was identical (4k 4.2-4.5 s, 32k
32.6 s): prefill is not bound by the collectives there (all-gather + reduce-scatter of 2048 rows: about 2 ms each).
`tests/bench_ep_comm.py` on that host: b12x `PCIeTwoShotBF16` beats NCCL only at 1 row per rank (all-gather 13.7 vs
25.0 us) and is 2-12x slower from 4 rows on, so the b12x transport is not a prefill speed-up there either; it stays
opt-in for hosts where NCCL cannot use P2P at all.

* **`NCCL_P2P_LEVEL`** (`server/dp_proxy.py`, `--dp > 1`): before the ranks start, the proxy runs a peer-copy
  self-test on all GPU pairs in a subprocess (`can_device_access_peer` plus a verified 16 MB copy, 60 s timeout). If
  it passes, the ranks get `NCCL_P2P_LEVEL=SYS`, otherwise NCCL keeps its default. An explicit `NCCL_P2P_LEVEL` or
  `NCCL_P2P_DISABLE` always wins; `SM120_NCCL_P2P_LEVEL=<LOC|NVL|PIX|PXB|PHB|SYS>` forces a level and
  `SM120_NCCL_P2P_LEVEL=default` skips the test. Check with `NCCL_DEBUG=INFO`: the channel lines should read
  "via P2P/CUMEM" (or P2P/IPC) instead of "via SHM". The self-test proves peer copies are correct, not that P2P is
  faster through the root complex on every platform; compare `tests/bench_ep_comm.py` with and without it.
* **`SM120_EP_TRANSPORT=b12x`** (`sm120_ep/transport.py` `B12xTransport`): the prefill all-gather and reduce-scatter
  go through `PCIeTwoShotBF16` (CUDA-IPC, fp32-accumulating reduce-scatter with one bf16 rounding; world size 4 only).
  Prefill rows per rank are padded to multiples of 256 (one prepared plan per size, built on first use); the
  experts skip the padding rows. Everything else stays on NCCL: decode under CUDA graphs, row counts, sizes above
  `SM120_EP_B12X_MAX_ROWS` (default 4096, set it to `--max-num-batched-tokens`), other world sizes. Each switch
  between b12x and NCCL depends only on values that are equal on all ranks, and runtime / plan creation all-reduce
  a success flag, so the ranks never mix transports in one collective. Extra GPU memory: about 1 GB per rank
  (staging plus owned max-size buffers). **Not yet run on 4 GPUs**: validate with
  `torchrun --nproc_per_node 4 /opt/sm120/tests/test_ep_b12x_transport.py` (b12x vs NCCL vs full experts, per-layer
  times) before serving with it.

## 5. Switches

| env | values | meaning |
|---|---|---|
| `SM120_EP_IMPL` | `allgather` (default if EP > 1) \| `off` | Option-B wrapper on/off |
| `SM120_EP_TRANSPORT` | `nccl` (default) \| `b12x` \| `gloo` | collectives backend: b12x = PCIe two-shot for prefill (§4a); gloo for single-GPU tests only |
| `SM120_EP_B12X_MAX_ROWS` | int (default 4096) | b12x transport: max prefill rows per rank (= `--max-num-batched-tokens`) |
| `SM120_NCCL_P2P_LEVEL` | unset (self-test, then SYS) \| `default` \| NCCL level | `NCCL_P2P_LEVEL` for the `--dp` ranks (§4a) |
| `SM120_EP_DP_SYNC` | 1 (default) \| 0 | DP-wide MAX sync of the decode graph size |
| `SM120_EP_DECODE_ROWS` | int (default `max_num_decoding_tokens`) | fixed per-rank decode row capacity when the DP sync is off |
| `SM120_EP_FUSED_PACK` | 1 (default) \| 0 | Triton kernel that packs hidden/ids/weights into the all-gather rows |

## 6. To validate on a real multi-GPU host (in this order)

1. **Interconnect:** `nvidia-smi topo -m`, `torch.cuda.can_device_access_peer`, and
   `torchrun --nproc_per_node 4 /opt/sm120/tests/bench_ep_comm.py --out /work/bench_ep_comm.json [--moe]`
   (NCCL vs b12x all-gather / reduce-scatter at 10 KB - 4 MB).
2. **`tests/test_ep_moe.py` at W=4 with NCCL**, one GPU per rank.
3. **DeepInfer `data_parallel_size = expert_parallel_size = 4`:** compare 4 requests per rank against the single-GPU run
   of the same 16 requests (where the model fits). `SM120_MOE_IMPL=torch` gives a deterministic comparison; its
   reference MoE must then accept -1 ids or have non-local weights zeroed.
4. **CUDA graphs on 4 ranks:** DP sync on/off, MTP verify batches (rows = B*(1+nextn)).
5. **Transport:** `tests/test_ep_b12x_transport.py` (4 ranks), then end-to-end prefill TTFT and TPOT at B = 1/8/32 with
   NCCL (with and without `NCCL_P2P_LEVEL=SYS`) vs `SM120_EP_TRANSPORT=b12x`.
6. Engram (`engram_parallel_size`) and prefix-cache interplay.

The operational side (image, mounts, memory budget, engram mode by host RAM, evaluation against a baseline) is in
[MULTI_GPU_RUNBOOK.md](MULTI_GPU_RUNBOOK.md).
