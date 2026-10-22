"""DP-wide max-sync of the decode CUDA-graph size, so that every EP rank replays the same graph size.

Why: with Option-B EP the per-layer all-gather / reduce-scatter need equal row counts on all ranks. Without a
sync every rank picks the graph for its own batch (DeepInfer CudaGraphRunner.select(total_q)), so the EP layer
has to pad every decode step to a fixed capacity (max_num_decoding_tokens rows per rank, up to 192 with
nextn 5 x 32 seqs: ~12 MB per layer over PCIe). With the sync every rank pads its batch to the DP-wide
maximum graph size instead, and the EP rows are simply that graph size.

Mechanism (one gloo all-reduce MAX per decode step on DeepInfer's cpu_data_parallel_group):
  * Worker.execute_model_step is wrapped: at the top of every decode step - real or placeholder, which every
    DP rank runs in lockstep (the C++ scheduler votes the step type across DP) - the local decode token count
    (scheduler_outputs.num_batched_tokens; 0 for placeholder steps) is all-reduced with MAX;
  * CudaGraphRunner.select(total_q) is wrapped to select for max(total_q, synced). Both calls in a step
    (worker.py _correct_seqlens_qk for ZOS and model_runner._prepare_input_driver) therefore see the same
    graph, and DeepInfer pads the batch with its own padding_seq_metadata_list as usual;
  * DFlashCudaGraphRunner.select(req_cnt) (MTP draft graph; its MoE layers are EP too) all-reduces its request
    count the same way.
Capture is unaffected (all ranks capture the same batch-size list in the same order); replay picks the synced
size. The extra cost is one localhost gloo all-reduce of an int (~tens of us) per decode step (+1 with MTP).

Prefill steps (eager) are synced the same way: the step's token count (sum of seqlen_q, 1 for a placeholder
step, whose forward runs DeepInfer's one-token placeholder sequence) is all-gathered once per step, and every
EP layer of the step pads to the maximum and skips the other ranks' padding rows (EPRoutedExperts, moe.py)
instead of exchanging its row count per layer (40 host syncs per step, each draining the GPU and waiting for
the slowest rank).

install() is called by sm120_ep.wrap() when the EP group has > 1 rank (SM120_EP_DP_SYNC=0 disables it; the
EP layer then falls back to the fixed decode capacity).
"""

from __future__ import annotations

import os

import sm120_prof
import torch
import torch.distributed as dist

_STATE = {"active": False, "synced": None, "prefill_rows": None, "prefill_counts": None, "consumer_counts": None,
          "group": None, "n_sync": 0, "yoco_boundary": None}


def enabled_by_env() -> bool:
    return os.environ.get("SM120_EP_DP_SYNC", "1").strip().lower() not in ("0", "false", "off", "no")


def active() -> bool:
    return bool(_STATE["active"])


def dp_max(value: int, group) -> int:
    t = torch.tensor([int(value)], dtype=torch.int64)
    with sm120_prof.rng("sm120:dp_sync.all_reduce"):
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    _STATE["n_sync"] += 1
    return int(t.item())


def dp_counts(value: int, group) -> list[int]:
    """All ranks' values (one gloo all-gather of an int)."""
    world = dist.get_world_size(group)
    parts = [torch.zeros((1,), dtype=torch.int64) for _ in range(world)]
    with sm120_prof.rng("sm120:dp_sync.all_gather"):
        dist.all_gather(parts, torch.tensor([int(value)], dtype=torch.int64), group=group)
    _STATE["n_sync"] += 1
    return [int(p.item()) for p in parts]


def patch_worker(cls, group) -> None:
    orig = cls.execute_model_step
    if getattr(orig, "_sm120_dp_sync", False):
        return

    def execute_model_step(self, is_prompt, is_placeholder, seq_metadata_list=None, scheduler_outputs=None):
        mr = getattr(self, "model_runner", None)
        _STATE["synced"] = _STATE["prefill_rows"] = _STATE["prefill_counts"] = _STATE["consumer_counts"] = None
        if is_prompt:
            local = 1 if is_placeholder or not seq_metadata_list else sum(int(s.seqlen_q) for s in seq_metadata_list)
            counts = dp_counts(local, group)
            _STATE["prefill_counts"], _STATE["prefill_rows"] = counts, max(counts)
        elif getattr(mr, "use_cuda_graph", False):
            local = 0
            if not is_placeholder and scheduler_outputs is not None:
                local = int(scheduler_outputs.num_batched_tokens)
            _STATE["synced"] = dp_max(local, group)
        return orig(self, is_prompt, is_placeholder, seq_metadata_list, scheduler_outputs)

    execute_model_step._sm120_dp_sync = True
    cls.execute_model_step = execute_model_step


def patch_graph_runner(cls) -> None:
    orig = cls.select
    if getattr(orig, "_sm120_dp_sync", False):
        return

    def select(self, total_q):
        s = _STATE["synced"]
        return orig(self, max(int(total_q), s) if s is not None else total_q)

    select._sm120_dp_sync = True
    cls.select = select


def patch_dflash_runner(cls, group) -> None:
    orig = cls.select
    if getattr(orig, "_sm120_dp_sync", False):
        return

    def select(self, req_cnt):
        return orig(self, dp_max(req_cnt, group) if _STATE["active"] else req_cnt)

    select._sm120_dp_sync = True
    cls.select = select


def install(group=None) -> bool:
    """Patch DeepInfer's worker / graph runners (idempotent). Returns True when active."""
    if _STATE["active"]:
        return True
    if not enabled_by_env():
        return False
    try:
        from deepinfer.worker import cuda_graph_runner as cgr
        from deepinfer.worker import worker as wk
        from deepinfer.model_executor.parallel_utils.parallel_state import PG
        group = group if group is not None else PG.cpu_data_parallel_group()
        patch_worker(wk.Worker, group)
        patch_graph_runner(cgr.CudaGraphRunner)
        try:
            from deepinfer.worker import dflash_cuda_graph_runner as dcgr
            patch_dflash_runner(dcgr.DFlashCudaGraphRunner, group)
        except ImportError:
            pass
    except Exception as exc:  # noqa: BLE001
        print(f"[sm120_ep] DP graph-size sync not installed ({type(exc).__name__}: {exc}); "
              "EP decode uses the fixed capacity", flush=True)
        return False
    _STATE["group"] = group
    _STATE["active"] = True
    print("[sm120_ep] DP-wide decode graph-size max-sync installed", flush=True)
    return True
