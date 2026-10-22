"""Option-B expert parallelism for DeepInfer's routed MoE on PCIe-only SM120 machines.

DeepInfer's model is DP attention + EP MoE. On SM100 the MoE goes through DeepEP all-to-all dispatch/combine
(NVLink/RDMA). Here every EP rank instead:

  1. all-gathers its tokens (bf16 hidden, global physical top-k ids, routing weights) packed in ONE row per
     token, so the whole layer costs one all-gather + one reduce-scatter;
  2. runs the rank-local routed experts (b12x fused_moe over the rank's n_local physical experts, exactly
     the object sm120_moe builds today) on all world*n rows, with every non-local route masked to -1
     (b12x skips them; tokens without local routes give exact zeros - feasibility test 10.4);
  3. reduce-scatters (sum) the partial outputs back, so each rank gets the full routed output of its tokens.
The shared expert stays rank-local (DeepInfer adds it after the routed output). Expert sharding at load is
DeepInfer's own: the HF loader keeps physical slots [ep_rank*n_local, (ep_rank+1)*n_local)
(hugging_face/weights.py _local_expert_ids) and the gate emits global physical ids.

Row counts per rank must match inside a collective:
  * decode (CUDA graphs): with the DP-wide graph-size sync (dp_sync.py, default) every rank replays the same
    graph size, so the rows are simply the step's token count, also inside the start-up capture; without it (SM120_EP_DP_SYNC=0 or eager
    decode) a fixed capacity C = SM120_EP_DECODE_ROWS (default: the layer's max_num_decoding_tokens) rows per
    rank, padding rows masked (ids -1, weight 0); no host sync, graph-capturable with NCCL;
  * prefill (eager; DeepInfer's scheduler makes the step type uniform across DP ranks by vote, see
    csrc/engine/scheduler.hpp NORMAL engine): every rank pads to the DP-wide maximum token count of the step,
    synced once per step by dp_sync.py; without it the per-rank counts are all-gathered in every layer (host
    sync).
The step type comes from input_metadata.is_prompt when the caller passes it (runtime hook); without it,
calls inside CUDA graph capture use the fixed capacity and eager calls the count exchange (correct as long
as all ranks are eager or all are captured).

Switches: SM120_EP_IMPL = allgather (default when the EP group has > 1 rank) | off;
          SM120_EP_TRANSPORT = nccl | gloo | b12x (see transport.py); SM120_EP_DECODE_ROWS = <int>;
          SM120_EP_COMPACT = 1 (default) | 0 (prefill experts on the padded rows too, for A/B).
"""

from __future__ import annotations

import os

import torch

from . import dp_sync
from .transport import make_transport

try:
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001
    triton = None

if triton is not None:

    @triton.jit
    def _pack_kernel(x_ptr, x_s, ids_ptr, ids_s, w_ptr, w_s, out_ptr, T, H2: tl.constexpr, K: tl.constexpr,
                     KP: tl.constexpr, ROW: tl.constexpr, BLOCK: tl.constexpr):
        """One packed row per program: [hidden bf16 bytes | int32 ids | fp32 weights]; rows >= T: no routes."""
        r = tl.program_id(0)
        r64 = r.to(tl.int64)
        live = r < T
        dst = out_ptr + r64 * ROW
        for c0 in range(0, H2 // 2, BLOCK):  # hidden as int16 words
            c = c0 + tl.arange(0, BLOCK)
            m = c < H2 // 2
            v = tl.load(x_ptr + r64 * x_s + c, mask=m & live, other=0)
            tl.store(dst.to(tl.pointer_type(tl.int16)) + c, v, mask=m)
        j = tl.arange(0, KP)
        jm = j < K
        ids = tl.load(ids_ptr + r64 * ids_s + j, mask=live & jm, other=-1).to(tl.int32)
        w = tl.load(w_ptr + r64 * w_s + j, mask=live & jm, other=0.0).to(tl.float32)
        tl.store((dst + H2).to(tl.pointer_type(tl.int32)) + j, ids, mask=jm)
        tl.store((dst + H2 + 4 * K).to(tl.pointer_type(tl.float32)) + j, w, mask=jm)

    @triton.jit
    def _unpack_kernel(in_ptr, x_ptr, ids_ptr, w_ptr, first, n_local, H2: tl.constexpr, K: tl.constexpr,
                       KP: tl.constexpr, ROW: tl.constexpr, BLOCK: tl.constexpr):
        """Packed row -> contiguous hidden, rank-local expert ids (-1 if not local), weights."""
        r = tl.program_id(0)
        r64 = r.to(tl.int64)
        src = in_ptr + r64 * ROW
        for c0 in range(0, H2 // 2, BLOCK):
            c = c0 + tl.arange(0, BLOCK)
            m = c < H2 // 2
            v = tl.load(src.to(tl.pointer_type(tl.int16)) + c, mask=m)
            tl.store(x_ptr + r64 * (H2 // 2) + c, v, mask=m)
        j = tl.arange(0, KP)
        jm = j < K
        ids = tl.load((src + H2).to(tl.pointer_type(tl.int32)) + j, mask=jm, other=-1)
        loc = ids - first
        ok = (ids >= 0) & (loc >= 0) & (loc < n_local)
        tl.store(ids_ptr + r64 * K + j, tl.where(ok, loc, -1), mask=jm)
        tl.store(w_ptr + r64 * K + j, tl.load((src + H2 + 4 * K).to(tl.pointer_type(tl.float32)) + j, mask=jm),
                 mask=jm)


def ep_impl() -> str:
    v = os.environ.get("SM120_EP_IMPL", "allgather").strip().lower()
    if v not in ("allgather", "off"):
        raise ValueError(f"SM120_EP_IMPL must be allgather|off, got {v!r}")
    return v


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


class EPRoutedExperts:
    """Callable with the signature of sm120_moe.B12xRoutedExperts: (x bf16 [T,H], topk_idx [T,k] global
    physical ids, topk_weights [T,k]) -> bf16 [T,H] routed output of this rank's tokens over ALL experts."""

    sm120_ep = True  # marker for sm120_moe.forward: pass is_prompt=input_metadata.is_prompt

    def __init__(self, inner, *, group, num_local_experts: int, ep_rank: int, hidden: int, topk: int,
                 decode_rows: int, transport=None):
        self.inner = inner
        self.transport = transport if transport is not None else make_transport(group)
        self.world = self.transport.world
        self.rank = self.transport.rank
        assert self.rank == ep_rank, (self.rank, ep_rank)
        self.n_local = int(num_local_experts)
        self.first = self.rank * self.n_local
        self.hidden = int(hidden)
        self.topk = int(topk)
        self.decode_rows = int(os.environ.get("SM120_EP_DECODE_ROWS", decode_rows))
        # packed row: bf16 hidden | int32 ids [k] | fp32 weights [k]  (bytes)
        self.row_bytes = 2 * self.hidden + 8 * self.topk
        assert self.row_bytes % 4 == 0

    def _yoco_consumer(self) -> bool:
        """True in the YOCO consumer half (layer >= n_prefill_layer) when compact prefill truncation is on."""
        if os.environ.get("DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE", "off") != "compact":
            return False
        b = dp_sync._STATE.get("yoco_boundary")
        idx = getattr(self.inner, "layer_idx", None)
        return b is not None and idx is not None and int(idx) >= int(b)

    # ----------------------------------------------------------------------------------------- pack/unpack
    def _fused_ok(self, x) -> bool:
        return triton is not None and x.is_cuda and os.environ.get("SM120_EP_FUSED_PACK", "1") != "0"

    def _pack(self, x: torch.Tensor, ids: torch.Tensor, w: torch.Tensor, rows: int) -> torch.Tensor:
        T = x.shape[0]
        buf = torch.empty((rows, self.row_bytes), dtype=torch.uint8, device=x.device)
        if self._fused_ok(x) and rows:
            xi = x.contiguous().view(torch.int16) if x.stride(1) != 1 else x.view(torch.int16)
            _pack_kernel[(rows,)](xi, xi.stride(0), ids, ids.stride(0), w, w.stride(0), buf, T, H2=2 * self.hidden,
                                  K=self.topk, KP=triton.next_power_of_2(self.topk), ROW=self.row_bytes, BLOCK=1024,
                                  num_warps=4)
            return buf
        H2, k = 2 * self.hidden, self.topk
        buf[:T, :H2] = x.contiguous().view(torch.uint8).view(T, H2)
        buf[:T, H2:H2 + 4 * k] = ids.to(torch.int32).contiguous().view(torch.uint8).view(T, 4 * k)
        buf[:T, H2 + 4 * k:] = w.to(torch.float32).contiguous().view(torch.uint8).view(T, 4 * k)
        if rows > T:  # padding rows: no routes
            buf[T:, :H2] = 0
            buf[T:, H2:H2 + 4 * k] = 0xFF  # int32 -1
            buf[T:, H2 + 4 * k:] = 0
        return buf

    def _unpack_local(self, buf: torch.Tensor):
        """-> (hidden bf16 [n,H], rank-local ids int32 [n,k] (-1 = not local), weights fp32 [n,k])."""
        n = buf.shape[0]
        if self._fused_ok(buf) and n:
            x = torch.empty((n, self.hidden), dtype=torch.bfloat16, device=buf.device)
            ids = torch.empty((n, self.topk), dtype=torch.int32, device=buf.device)
            w = torch.empty((n, self.topk), dtype=torch.float32, device=buf.device)
            _unpack_kernel[(n,)](buf, x.view(torch.int16), ids, w, self.first, self.n_local, H2=2 * self.hidden,
                                 K=self.topk, KP=triton.next_power_of_2(self.topk), ROW=self.row_bytes,
                                 BLOCK=1024, num_warps=4)
            return x, ids, w
        x, ids, w = self._unpack(buf)
        return x, self._local_ids(ids), w

    def _unpack(self, buf: torch.Tensor):
        n = buf.shape[0]
        H2, k = 2 * self.hidden, self.topk
        x = buf[:, :H2].contiguous().view(torch.bfloat16).view(n, self.hidden)
        ids = buf[:, H2:H2 + 4 * k].contiguous().view(torch.int32).view(n, k)
        w = buf[:, H2 + 4 * k:].contiguous().view(torch.float32).view(n, k)
        return x, ids, w

    def _local_ids(self, ids: torch.Tensor) -> torch.Tensor:
        loc = ids - self.first
        ok = (ids >= 0) & (loc >= 0) & (loc < self.n_local)
        return torch.where(ok, loc, torch.full_like(loc, -1))

    # ----------------------------------------------------------------------------------------------- run
    def __call__(self, x: torch.Tensor, topk_idx: torch.Tensor, topk_weights: torch.Tensor, *,
                 is_prompt: bool | None = None) -> torch.Tensor:
        T = int(x.shape[0])
        fixed = (not is_prompt) if is_prompt is not None else _capturing()
        if fixed:
            if dp_sync.active() and (dp_sync._STATE["synced"] is not None or dp_sync._STATE["in_capture"]
                                     or _capturing()):
                # every DP rank replays the same (DP-max) graph size, see dp_sync.py. Start-up capture (no step synced
                # yet): every rank runs the same batch-size list in the same order, each size an eager warm-up and
                # then the capture, so both use T rows; the warm-up thereby creates the local experts' b12x plan for
                # the world*T bucket that the capture needs (plans cannot be created while capturing)
                rows = T
            else:
                rows = self.decode_rows
                if T > rows:
                    raise RuntimeError(f"EP decode step has {T} rows > SM120_EP_DECODE_ROWS={rows}")
            counts = None
        elif dp_sync.active() and dp_sync._STATE["prefill_counts"] is not None:
            counts = dp_sync._STATE["prefill_counts"]
            if self._yoco_consumer():
                # YOCO compact truncation: layers >= n_prefill_layer see only the kept rows. Every rank reaches the
                # first consumer layer in the same step, so all exchange their row counts there once (one host sync
                # per step) and the consumer layers pad to the kept rows instead of the step's tokens.
                if dp_sync._STATE["consumer_counts"] is None:
                    dp_sync._STATE["consumer_counts"] = self.transport.all_gather_counts(T, x.device)
                counts = dp_sync._STATE["consumer_counts"]
            rows = max(counts)
            if T > counts[self.rank]:  # the step's token count bounds every layer's T (else the collectives desync)
                raise RuntimeError(f"EP prefill layer has {T} rows > the step's token count {counts[self.rank]}")
        else:
            counts = self.transport.all_gather_counts(T, x.device)
            rows = max(counts)
        if counts is not None and hasattr(self.transport, "round_rows"):
            rows = self.transport.round_rows(rows)  # b12x: plan bucket (padding rows are skipped by the experts)
        send = self._pack(x, topk_idx, topk_weights, rows)
        recv = torch.empty((self.world * rows, self.row_bytes), dtype=torch.uint8, device=x.device)
        self.transport.all_gather(recv, send)
        compact = os.environ.get("SM120_EP_COMPACT", "1") != "0"  # 0: experts see the padding rows (A/B only)
        real = None if (counts is None or not compact or sum(counts) == self.world * rows) \
            else _real_rows(counts, rows, x.device)
        if real is None:
            xg, idl, wg = self._unpack_local(recv)
            part = self.inner(xg, idl, wg)  # bf16 [world*rows, H], zeros where no local route
        else:
            # prefill: run the experts on the ranks' real rows only (padding rows have no routes, but b12x
            # still quantizes and schedules them: 22.5 vs 6.8 ms per layer for one busy rank of four)
            xg, idl, wg = self._unpack_local(recv.index_select(0, real))
            sub = self.inner(xg, idl, wg)
            part = torch.zeros((self.world * rows, sub.shape[1]), dtype=sub.dtype, device=x.device)
            part.index_copy_(0, real, sub)
        out = torch.empty((rows, self.hidden), dtype=part.dtype, device=x.device)
        self.transport.reduce_scatter_sum(out, part.contiguous())
        return out[:T]


_REAL_ROWS: dict = {}


def _real_rows(counts, rows: int, device) -> torch.Tensor:
    """Row index of every rank's real (non-padding) rows in the all-gathered [world*rows] buffer; built once per
    (counts, rows), i.e. once per prefill step, and shared by all layers (no host sync)."""
    key = (tuple(counts), rows, str(device))
    t = _REAL_ROWS.get(key)
    if t is None:
        _REAL_ROWS.clear()
        t = torch.cat([torch.arange(r * rows, r * rows + c, device=device) for r, c in enumerate(counts) if c])
        _REAL_ROWS[key] = t
    return t


def wrap(moe, inner=None):
    """Hook for sm120_moe.setup (runtime): wrap the rank-local routed experts when the EP group has > 1 rank.

    Returns the object to store in moe.sm120_routed (the inner one unchanged for EP=1 / SM120_EP_IMPL=off)."""
    inner = inner if inner is not None else moe.sm120_routed
    group = moe.ep_group
    size = group.size() if hasattr(group, "size") else 1
    if size <= 1 or ep_impl() == "off":
        return inner
    rows = int(getattr(moe, "max_num_decoding_tokens", 0) or 0) or 256
    dp_sync.install()  # DP-wide decode graph-size sync (SM120_EP_DP_SYNC=0: fixed decode capacity instead)
    return EPRoutedExperts(inner, group=group, num_local_experts=moe.num_local_physical_experts, ep_rank=moe.ep_rank,
                           hidden=moe.hidden_size, topk=moe.num_physical_topk, decode_rows=rows)
