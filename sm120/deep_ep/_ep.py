"""Single-rank (EP=1) DeepEP v2 ``EPBuffer`` with DeepEP's exact tensor contracts (vendor
DeepEP/deep_ep/buffers/ep.py docstrings + utils/refs.py), implemented with plain torch ops on the
current stream.

dispatch(do_expand=True)   (the mode DeepInfer uses, mlp.py:469)
    recv_x              [num_expanded, hidden] (bf16) or (e4m3 [num_expanded, hidden], SF) where the
                        rows are grouped by local expert, each expert block starting at an
                        ``expert_alignment``-aligned row; inside a block tokens are in source order.
                        Padding rows are zero (values and SF).
                        SF: int32 [num_expanded, hidden/128] (packed UE8M0 per 32), MN-major TMA-aligned
                        when use_tma_aligned_col_major_sf else row-major.
    recv_topk_idx       [num_expanded] int64 local expert id per row (-1 on padding rows)
    recv_topk_weights   [num_expanded] fp32 routing weight per row (0 on padding rows)
    handle.psum_num_recv_tokens_per_expert  int32 [E]: psum[i] = aligned start of expert i + count_i
    handle.recv_src_metadata  int64 [T, 2 + topk]: (src token idx, src rank, row slot per top-k or -1)
dispatch(do_expand=False)
    recv_x = x (token major, every token once), recv_topk_idx / weights = inputs,
    psum = inclusive prefix sum of expert_alignment-padded counts.
combine
    expand:     out[t] = bias[t] + sum_k x[slot(t, k)]   (fp32 accumulation, slot order)
    non-expand: out[t] = bias[t] + x[t]
The CPU sync DeepEP does for ``do_cpu_sync=True`` is always performed (the output size needs it);
set DEEPINFER_DISABLE_SYNC_GUARD=1 when DeepInfer's forbid_cuda_sync guard is active.
"""

from __future__ import annotations

from typing import Optional

import os

import torch

from ._event import EventHandle, EventOverlap

topk_idx_t = torch.int64


def _align(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def _group_size(group) -> int:
    if group is None:
        return 1
    try:
        return int(group.size())
    except Exception:  # noqa: BLE001
        return 1


def _require_single_rank(group, what: str) -> None:
    n = _group_size(group)
    if n != 1:
        raise NotImplementedError(
            f"SM120 deep_ep.{what}: only a single-rank group is implemented (got {n} ranks). Multi-GPU EP over "
            "PCIe needs an NCCL all-to-all / all-gather+reduce-scatter backend (see docs/MOE_GEMM.md)."
        )


def _ep_allgather() -> bool:
    """SM120_EP_IMPL=allgather (default; sm120/sm120_ep, Option B): routed experts never use EPBuffer, so a
    multi-rank EPBuffer is only an inert placeholder for DeepInfer's userbuffer bookkeeping."""
    return os.environ.get("SM120_EP_IMPL", "allgather").strip().lower() == "allgather"


def _mn_major_copy(sf: torch.Tensor) -> torch.Tensor:
    mn, k = sf.shape
    tma = _align(mn, 4)
    out = torch.empty_strided((mn, k), (1, tma), dtype=sf.dtype, device=sf.device)
    out.copy_(sf)
    return out


class EPHandle:
    def __init__(self, do_expand, num_experts, expert_alignment, num_max_tokens_per_rank, num_sms, topk_idx,
                 num_recv_tokens, num_expanded_tokens, num_recv_tokens_per_expert_list,
                 psum_num_recv_tokens_per_scaleup_rank, psum_num_recv_tokens_per_expert,
                 num_unaligned_recv_tokens_per_expert, recv_src_metadata, dst_buffer_slot_idx,
                 token_metadata_at_forward=None, channel_linked_list=None):
        assert topk_idx is not None
        self.do_expand = do_expand
        self.num_experts = num_experts
        self.expert_alignment = expert_alignment
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_sms = num_sms
        self.psum_num_recv_tokens_per_scaleup_rank = psum_num_recv_tokens_per_scaleup_rank
        self.psum_num_recv_tokens_per_expert = psum_num_recv_tokens_per_expert
        self.num_unaligned_recv_tokens_per_expert = num_unaligned_recv_tokens_per_expert
        self.num_recv_tokens_per_expert_list = num_recv_tokens_per_expert_list
        self.recv_src_metadata = recv_src_metadata
        self.dst_buffer_slot_idx = dst_buffer_slot_idx
        self.token_metadata_at_forward = token_metadata_at_forward
        self.channel_linked_list = channel_linked_list
        self.num_recv_tokens = num_recv_tokens
        self.num_expanded_tokens = num_expanded_tokens
        self.cached_recv_src_metadata_before_sort = None
        self._topk_idx = topk_idx

    @property
    def topk_idx(self) -> torch.Tensor:
        return self._topk_idx


class EPBuffer:
    def __init__(self, group, num_bytes: Optional[int] = None, num_max_tokens_per_rank: int = 0, hidden: int = 0,
                 num_topk: int = 0, use_fp8_dispatch: bool = False, lb_allocation_plan_or_num_bytes=0,
                 deterministic: bool = False, allow_hybrid_mode: bool = True, allow_multiple_reduction: bool = True,
                 prefer_overlap_with_compute: bool = True, sl_idx=None, num_allocated_qps: int = 0,
                 num_cpu_timeout_secs: int = 300, num_gpu_timeout_secs: int = 100, explicitly_destroy: bool = False):
        n = _group_size(group)
        self.inert = n > 1 and _ep_allgather()
        if not self.inert:
            _require_single_rank(group, "EPBuffer")
        self.group = group
        self.rank_idx = int(group.rank()) if self.inert else 0
        self.num_ranks = n if self.inert else 1
        self.allow_hybrid_mode = allow_hybrid_mode
        self.allow_multiple_reduction = allow_multiple_reduction
        self.prefer_overlap_with_compute = prefer_overlap_with_compute
        self.deterministic = deterministic
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_bytes = int(num_bytes) if num_bytes is not None else self.get_buffer_size_hint(
            group, num_max_tokens_per_rank, hidden, num_topk, use_fp8_dispatch)
        self.num_lb_bytes = 0
        self.num_allocated_qps = num_allocated_qps or 17
        self.explicitly_destroy = explicitly_destroy
        self.num_scaleout_ranks, self.num_scaleup_ranks = 1, 1
        self.scaleout_rank_idx = self.scaleup_rank_idx = 0
        self.num_rdma_ranks, self.num_nvlink_ranks = 1, 1
        self.context = None
        self.runtime = None

    # --- comm helpers (DeepEP exposes them on the buffer class)
    @staticmethod
    def barrier(*args, **kwargs) -> None:
        return None

    @staticmethod
    def get_comm_stream(*args, **kwargs):
        return torch.cuda.current_stream()

    def get_physical_domain_size(self):
        return 1, 1

    def get_logical_domain_size(self):
        return 1, 1

    def destroy(self) -> None:
        self.group = None

    @staticmethod
    def get_buffer_size_hint(group, num_max_tokens_per_rank: int, hidden: int, num_topk: int = 0,
                             use_fp8_dispatch: bool = False, allow_hybrid_mode: bool = True,
                             allow_multiple_reduction: bool = True) -> int:
        # nothing is pre-allocated by this shim; report a plausible, 2 MiB-aligned size
        per_token = hidden * (1 if use_fp8_dispatch else 2) + hidden // 32 + max(num_topk, 1) * 12
        return _align(max(int(num_max_tokens_per_rank), 1) * per_token, 2 << 20)

    @staticmethod
    def capture() -> EventHandle:
        return EventHandle()

    def get_theoretical_num_sms(self, num_experts: int = 0, num_topk: int = 0, *args, **kwargs) -> int:
        # No communication at EP=1: do not reserve SMs for it (DeepInfer subtracts this from the compute SMs)
        return 0

    def get_theoretical_num_qps(self, num_sms: int) -> int:
        return 1

    # ----------------------------------------------------------------------------------- dispatch
    def _not_inert(self, what: str) -> None:
        if getattr(self, "inert", False):
            raise NotImplementedError(f"SM120 deep_ep.EPBuffer.{what}: multi-rank buffer is inert (SM120_EP_IMPL="
                                      "allgather routes experts through sm120/sm120_ep, not DeepEP dispatch/combine)")

    def dispatch(self, x, topk_idx=None, topk_weights=None, cumulative_local_expert_recv_stats=None,
                 num_experts=None, num_max_tokens_per_rank=None, expert_alignment=None, num_sms: int = 0,
                 num_qps: int = 0, previous_event=None, async_with_compute_stream: bool = False,
                 allocate_on_comm_stream: bool = False, handle: Optional[EPHandle] = None,
                 do_handle_copy: bool = False, do_cpu_sync=None, do_expand: bool = False,
                 do_zero_padding: bool = False, use_tma_aligned_col_major_sf: bool = False,
                 defer_epilogue: bool = False):
        self._not_inert("dispatch")
        assert not do_handle_copy, "`do_handle_copy` must be False"
        if previous_event is not None:
            previous_event.current_stream_wait()
        if handle is not None:
            assert topk_idx is None
            topk_idx = handle.topk_idx
            num_experts = num_experts if num_experts is not None else handle.num_experts
            expert_alignment = expert_alignment if expert_alignment is not None else handle.expert_alignment
            num_max_tokens_per_rank = num_max_tokens_per_rank or handle.num_max_tokens_per_rank
            do_expand = handle.do_expand
        num_max_tokens_per_rank = num_max_tokens_per_rank or self.num_max_tokens_per_rank
        expert_alignment = int(expert_alignment or 1)
        assert num_experts is not None, "num_experts is required"
        x_data, x_sf = (x[0], x[1]) if isinstance(x, (tuple, list)) else (x, None)
        result = (_dispatch_expand if do_expand else _dispatch_plain)(
            x_data, x_sf, topk_idx, topk_weights, int(num_experts), expert_alignment,
            use_tma_aligned_col_major_sf, cumulative_local_expert_recv_stats)
        recv_x, recv_sf, recv_idx, recv_w, meta = result
        if handle is None:
            handle = EPHandle(do_expand, int(num_experts), expert_alignment, num_max_tokens_per_rank, num_sms,
                              topk_idx, **meta)
        recv = (recv_x, recv_sf) if recv_sf is not None else recv_x
        event = EventOverlap(EventHandle())
        if defer_epilogue:
            assert async_with_compute_stream, "defer_epilogue requires async_with_compute_stream"
            event.register_hook_after_wait(lambda: (recv, recv_idx, recv_w, handle))
            return event
        return recv, recv_idx, recv_w, handle, event

    # ----------------------------------------------------------------------------------- combine
    def combine(self, x: torch.Tensor, handle: EPHandle, topk_weights=None, bias=None, num_sms: int = 0,
                num_qps: int = 0, previous_event=None, async_with_compute_stream: bool = False,
                allocate_on_comm_stream: bool = False, defer_epilogue: bool = False):
        self._not_inert("combine")
        if previous_event is not None:
            previous_event.current_stream_wait()
        meta = handle.recv_src_metadata
        T = int(meta.shape[0])
        hidden = int(x.shape[-1])
        out = torch.zeros((T, hidden), dtype=torch.float32, device=x.device)
        if handle.do_expand:
            slots = meta[:, 2:]
            for k in range(slots.shape[1]):
                s = slots[:, k]
                m = s >= 0
                if bool(m.any()):
                    out[m] += x[s[m]].float()
        elif T > 0:
            out += x[:T].float()
        if bias is not None:
            for b in (bias if isinstance(bias, (tuple, list)) else (bias,)):
                if b is not None:
                    out += b.float()
        combined_x = out.to(torch.bfloat16)
        combined_w = topk_weights if topk_weights is not None and not handle.do_expand else None
        event = EventOverlap(EventHandle())
        if defer_epilogue:
            assert async_with_compute_stream, "defer_epilogue requires async_with_compute_stream"
            event.register_hook_after_wait(lambda: (combined_x, combined_w))
            return event
        return combined_x, combined_w, event

    # --- EPLB weight prefetch (needs peers)
    def lb_get_theoretical_num_sms(self) -> int:
        return 0

    def lb_prefetch_weights(self, *args, **kwargs):
        raise NotImplementedError("SM120 deep_ep: EPLB load-balance weight prefetch requires multiple ranks")

    def lb_reduce_grads(self, *args, **kwargs):
        raise NotImplementedError("SM120 deep_ep: lb_reduce_grads is not implemented")


def _counts(topk_idx: torch.Tensor, num_experts: int):
    idx = topk_idx.to(torch.int64)
    valid = (idx >= 0) & (idx < num_experts)
    counts = torch.bincount(idx[valid], minlength=num_experts)[:num_experts]
    return idx, valid, counts


def _dispatch_expand(x, sf, topk_idx, topk_weights, E, alignment, tma_sf, stats):
    T, K = topk_idx.shape
    hidden = x.shape[-1]
    dev = x.device
    idx, valid, counts = _counts(topk_idx, E)
    counts_list = [int(c) for c in counts.tolist()]  # CPU sync (output size)
    aligned = [_align(c, alignment) for c in counts_list]
    starts, acc = [], 0
    for a in aligned:
        starts.append(acc)
        acc += a
    num_expanded = acc
    starts_t = torch.tensor(starts, dtype=torch.int64, device=dev)
    # stable order: by expert, then token, then slot
    flat_idx = idx.flatten()
    flat_valid = valid.flatten()
    pos = torch.arange(T * K, device=dev, dtype=torch.int64)
    key = torch.where(flat_valid, flat_idx * (T * K) + pos, torch.full_like(pos, E * T * K))
    order = torch.argsort(key)
    n_valid = int(sum(counts_list))
    order = order[:n_valid]
    exp_sorted = flat_idx[order]
    # rank inside the expert block
    first = torch.zeros(E + 1, dtype=torch.int64, device=dev)
    first[1:] = torch.cumsum(counts, 0)
    rank = torch.arange(n_valid, device=dev, dtype=torch.int64) - first[exp_sorted]
    dst = starts_t[exp_sorted] + rank
    tok = order // K
    slot = order % K

    recv_x = torch.zeros((num_expanded, hidden), dtype=x.dtype, device=dev)
    if n_valid:
        # index_put on float8 is not implemented everywhere: move bytes for 1-byte dtypes
        if x.element_size() == 1 and x.dtype != torch.uint8:
            recv_x.view(torch.uint8)[dst] = x.contiguous().view(torch.uint8)[tok]
        else:
            recv_x[dst] = x[tok]
    recv_sf = None
    if sf is not None:
        sf_rows = torch.zeros((num_expanded, sf.shape[-1]), dtype=sf.dtype, device=dev)
        if n_valid:
            sf_rows[dst] = sf[tok]
        recv_sf = _mn_major_copy(sf_rows) if tma_sf else sf_rows
    recv_w = torch.zeros((num_expanded,), dtype=torch.float32, device=dev)
    recv_idx = torch.full((num_expanded,), -1, dtype=torch.int64, device=dev)
    slots = torch.full((T, K), -1, dtype=torch.int64, device=dev)
    if n_valid:
        if topk_weights is not None:
            recv_w[dst] = topk_weights[tok, slot].float()
        recv_idx[dst] = exp_sorted
        slots[tok, slot] = dst
    meta = torch.cat([torch.arange(T, device=dev, dtype=torch.int64).unsqueeze(1),
                      torch.zeros((T, 1), dtype=torch.int64, device=dev), slots], dim=1)
    psum = torch.tensor([s + c for s, c in zip(starts, counts_list)], dtype=torch.int32, device=dev)
    if stats is not None:
        stats += counts[: stats.shape[0]].to(stats.dtype)
    num_recv_tokens = int(valid.any(dim=1).sum().item()) if T else 0
    info = dict(
        num_recv_tokens=num_recv_tokens,
        num_expanded_tokens=num_expanded,
        num_recv_tokens_per_expert_list=counts_list,
        psum_num_recv_tokens_per_scaleup_rank=torch.tensor([num_recv_tokens], dtype=torch.int32, device=dev),
        psum_num_recv_tokens_per_expert=psum,
        num_unaligned_recv_tokens_per_expert=counts.to(torch.int32),
        recv_src_metadata=meta,
        dst_buffer_slot_idx=slots,
    )
    return recv_x, recv_sf, recv_idx, recv_w if topk_weights is not None else None, info


def _dispatch_plain(x, sf, topk_idx, topk_weights, E, alignment, tma_sf, stats):
    T, K = topk_idx.shape
    dev = x.device
    idx, valid, counts = _counts(topk_idx, E)
    counts_list = [int(c) for c in counts.tolist()]
    psum, acc = [], 0
    for c in counts_list:
        acc += _align(c, alignment)
        psum.append(acc)
    recv_idx = torch.where(valid, idx, torch.full_like(idx, -1))
    recv_sf = None
    if sf is not None:
        recv_sf = _mn_major_copy(sf.contiguous()) if tma_sf else sf.contiguous()
    meta = torch.cat([torch.arange(T, device=dev, dtype=torch.int64).unsqueeze(1),
                      torch.zeros((T, 1), dtype=torch.int64, device=dev)], dim=1)
    if stats is not None:
        stats += counts[: stats.shape[0]].to(stats.dtype)
    info = dict(
        num_recv_tokens=T,
        num_expanded_tokens=T,
        num_recv_tokens_per_expert_list=counts_list,
        psum_num_recv_tokens_per_scaleup_rank=torch.tensor([T], dtype=torch.int32, device=dev),
        psum_num_recv_tokens_per_expert=torch.tensor(psum, dtype=torch.int32, device=dev),
        num_unaligned_recv_tokens_per_expert=counts.to(torch.int32),
        recv_src_metadata=meta,
        dst_buffer_slot_idx=torch.arange(T, device=dev, dtype=torch.int64),
    )
    return x, recv_sf, recv_idx, topk_weights, info
