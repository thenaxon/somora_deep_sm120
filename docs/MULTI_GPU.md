# Multi-GPU on PCIe-only SM120: expert parallelism without DeepEP

Target: a host with several SM120 GPUs and no NVLink, e.g. the 4x RTX PRO 6000 Blackwell Max-Q (PCIe) target.

Status: **implemented, but only emulated-tested.** The EP MoE layer, the DP-wide decode graph-size sync, the DeepInfer
hooks and the multi-rank server front end (`server.py --dp N`) are in place. They have been tested on one GPU
(in-process emulation, several processes on one GPU over gloo, and a 1-rank NCCL group under CUDA graphs). Nothing
has run on 4 real GPUs yet. Treat multi-GPU serving as untested until the steps in §6 pass.

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
| **Different token counts per rank** | Collectives need equal row counts. **Prefill** (eager): per-rank counts are all-gathered first (one host sync per layer) and every rank pads to the maximum with masked rows. **Decode** (CUDA graphs): see the DP sync below; without it, a fixed capacity of `SM120_EP_DECODE_ROWS` (default: the layer's `max_num_decoding_tokens`) rows per rank, padding rows masked, no host sync. |
| **DP-wide decode graph-size sync** (`dp_sync.py`, default on) | `Worker.execute_model_step` is wrapped: at the top of every decode step (real or placeholder) the local decode token count is all-reduced with MAX on DeepInfer's CPU DP group (one gloo all-reduce of an int, tens of us). `CudaGraphRunner.select(total_q)` then selects for the synced size, so every rank replays the same graph and the EP rows are simply that graph size. `DFlashCudaGraphRunner.select` (MTP draft graph) is synced the same way. `SM120_EP_DP_SYNC=0` falls back to the fixed capacity. |
| **CUDA graphs** | NCCL collectives are capturable, and DeepInfer captures the same batch-size list in the same order on every rank. The decode path allocates only from the graph pool. |
| **Transport** | `SM120_EP_TRANSPORT=nccl` (default): `all_gather_into_tensor` / `reduce_scatter_tensor` on DeepInfer's EP group (already NCCL). `gloo`: CPU staging, eager only, for single-GPU multi-process tests. |
| **Serving** | `server/server.py --dp N [--ep M]` spawns N engine processes (engine_rank r, `data_parallel_size` N, `expert_parallel_size` M, default N), one GPU each, behind `server/dp_proxy.py`, which routes each request to the least-loaded rank. See [SERVER.md](SERVER.md). |

## 3. Tests on one GPU

Setup: random MXFP4 experts in DeepInfer's parameter layout, E=8, K=2048, n=512, top-4, with the real
`B12xRoutedExperts`.

| test | what | result |
|---|---|---|
| `tests/test_ep_moe.py` level 1 | in-process emulation of W=4: sum of the masked local partials of 4 ranks vs the full 8-expert run | cos 0.99998 (T=1) ... 0.999997 (T=300) |
| `tests/test_ep_moe.py` level 2 | **W=2 processes on the same GPU**, gloo, `EPRoutedExperts` end to end. Per-rank counts differ: prefill 37/63, decode 3/12 with capacity 16, and prefill with one rank at 0 tokens. Each rank vs a full-expert run of its own tokens | cos >= 0.99998, max\|d\| <= 0.125 (bf16 partial-sum rounding) |
| `tests/test_ep_moe.py` level 3 | W=1 NCCL group, fixed-capacity path captured in a CUDA graph and replayed with new inputs | cos 0.999996 vs eager |
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
  NCCL and `PCIeTwoShotBF16` (and with `--moe` a full EP MoE layer) on a 4-GPU host. A `b12x` transport in
  `sm120_ep/transport.py` is not wired yet; it needs 4 distinct GPUs to test.
- **Halving the all-gather:** an FP8 all-gather of the hidden state would halve that traffic, but needs a b12x entry
  that accepts pre-quantized activations (not found in 1.5.1).
- **Compute side:** each rank streams only 1/W of the expert weights. Attention, mHC, dense GEMMs and lm_head stay
  replicated per rank (DP attention).

## 5. Switches

| env | values | meaning |
|---|---|---|
| `SM120_EP_IMPL` | `allgather` (default if EP > 1) \| `off` | Option-B wrapper on/off |
| `SM120_EP_TRANSPORT` | `nccl` (default) \| `gloo` | collectives backend; gloo for single-GPU tests only |
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
5. **Transport:** end-to-end TPOT at B = 1/8/32, NCCL vs (once wired) b12x `PCIeTwoShotBF16`.
6. Engram (`engram_parallel_size`) and prefix-cache interplay.

The operational side (image, mounts, memory budget, engram mode by host RAM, evaluation against a baseline) is in
[MULTI_GPU_RUNBOOK.md](MULTI_GPU_RUNBOOK.md).
