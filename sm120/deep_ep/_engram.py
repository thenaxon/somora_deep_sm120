"""SM120 `deep_ep.EngramBuffer`: DeepEP v2's Engram API on top of sm120_engram.EngramFetcher.

DeepEP stores each rank's table shard in RDMA-registered CPU/GPU memory and every rank fetches rows of
any shard with GPU-initiated NCCL GIN gets. Here nothing is remote: every process can read every row
itself, because the table comes from

  * `attach_tables(...)` (installed by sm120_patches_engram: a per-layer `EngramHostTable` that mmaps the
    *whole* global table out of the HF safetensors shard; works for any engram group size, since each
    rank serves global row ids locally and the page cache is shared between the ranks of a machine), or
  * DeepInfer's stock `write(storages, sfs)` (this rank's shard as CPU/CUDA tensors). That only holds
    the full table when the engram group has one rank, so it is limited to that.

`fetch()` returns one hook per layer with DeepEP's output contract (csrc/buffers/engram.hpp:158-243).
"""

from __future__ import annotations

import functools
from typing import Callable, List, Optional, Sequence, Tuple

import torch

from ._stubs import BufferBase

_ALLOC_ALIGN = 2 << 20


def _align(x: int, y: int) -> int:
    return (x + y - 1) // y * y


def _group_size(group) -> int:
    return 1 if group is None else int(group.size())


def _group_rank(group) -> int:
    return 0 if group is None else int(group.rank())


def _accept_legacy_storage_size(init):
    @functools.wraps(init)
    def wrapper(self, *args, **kwargs):
        if "num_rdma_storage_size" in kwargs:
            if "num_rdma_storage_bytes" in kwargs:
                raise TypeError("Specify only one of num_rdma_storage_bytes and num_rdma_storage_size")
            kwargs["num_rdma_storage_bytes"] = kwargs.pop("num_rdma_storage_size")
        return init(self, *args, **kwargs)

    return wrapper


class EngramBuffer(BufferBase):
    """Process-local Engram row fetch (host RAM / mmap / NVMe / GPU), DeepEP v2 2.5.0 call surface."""

    @_accept_legacy_storage_size
    def __init__(self, group, num_gpu_bytes: int = 0, num_rdma_storage_bytes: int = 0,
                 use_cpu_rdma_storage: bool = True, num_allocated_qps: int = 1, qp_depth: int = 1,
                 restrict_rd_atomic: bool = False, allow_hybrid_mode: bool = True, sl_idx: Optional[int] = None,
                 num_gpu_timeout_secs: int = 100, explicitly_destroy: bool = False) -> None:
        super().__init__(explicitly_destroy)
        self.group = group
        self.rank_idx, self.num_ranks = _group_rank(group), _group_size(group)
        self.num_gpu_bytes, self.num_rdma_storage_bytes = num_gpu_bytes, num_rdma_storage_bytes
        self.use_cpu_rdma_storage = use_cpu_rdma_storage
        self.num_allocated_qps = max(1, int(num_allocated_qps))
        self.allow_hybrid_mode = allow_hybrid_mode
        # every rank reads rows itself: one "scale-out" rank per process, no NVLink peers
        self.num_scaleout_ranks, self.num_scaleup_ranks = self.num_ranks, 1
        self.scaleout_rank_idx, self.scaleup_rank_idx = self.rank_idx, 0
        self.num_rdma_ranks, self.num_nvlink_ranks = self.num_ranks, 1
        self.context = None
        self._fetcher = None
        self._config = None

    # ---- DeepEP static helpers
    @staticmethod
    def get_storage_size_hint(group, num_entries_per_layer: List[int], hidden: int, num_max_tokens: int,
                              num_entries_per_token: int, dtype: torch.dtype = torch.bfloat16,
                              num_sf_packs: int = 0) -> Tuple[int, int]:
        """(GPU receive bytes, per-rank storage bytes) as DeepEP computes them (calculate_storage_size);
        informational here, nothing is pre-allocated from it."""
        row = hidden * dtype.itemsize
        recv = _align(len(num_entries_per_layer) * num_max_tokens * num_entries_per_token * row, _ALLOC_ALIGN)
        storage = _align(sum(num_entries_per_layer) * row, _ALLOC_ALIGN)
        return recv, storage

    @staticmethod
    def get_theoretical_config(group, num_layers: int, num_max_tokens: int, num_entries_per_token: int,
                               num_max_read_requests_per_qp: int = 16) -> Tuple[int, int, bool]:
        return 1, 1, False  # no QPs

    # ---- comm helpers DeepInfer may call on any DeepEP buffer
    def barrier(self, *args, **kwargs) -> None:
        if self.group is not None and self.num_ranks > 1:
            torch.distributed.barrier(group=self.group)

    def get_comm_stream(self):
        return torch.cuda.current_stream()

    def get_physical_domain_size(self) -> Tuple[int, int]:
        return self.num_rdma_ranks, self.num_nvlink_ranks

    def get_logical_domain_size(self) -> Tuple[int, int]:
        return self.num_scaleout_ranks, self.num_scaleup_ranks

    def destroy(self) -> None:
        if self._fetcher is not None:
            self._fetcher.close()
            self._fetcher = None
        super().destroy()

    # ---- config / storage
    def set_config(self, num_entries_per_layer: List[int], hidden: int, num_max_tokens: int,
                   num_entries_per_token: int, dtype: torch.dtype = torch.bfloat16, num_sf_packs: int = 0) -> None:
        if self._fetcher is not None:
            self._fetcher.close()
        self._make_fetcher(num_entries_per_layer, hidden, num_max_tokens, num_entries_per_token, dtype, num_sf_packs)

    def _make_fetcher(self, num_entries_per_layer, hidden, num_max_tokens, num_entries_per_token, dtype, num_sf_packs):
        import sm120_engram

        sf_dtype = torch.int32  # PLATFORM.fp8_sf_type on CUDA (4 UE8M0 bytes per pack)
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        self._config = dict(num_entries_per_layer=list(num_entries_per_layer), hidden=int(hidden),
                            num_max_tokens=int(num_max_tokens), num_entries_per_token=int(num_entries_per_token),
                            dtype=dtype, num_sf_packs=int(num_sf_packs))
        self._fetcher = sm120_engram.EngramFetcher(
            num_entries_per_layer, hidden, dtype.itemsize, num_max_tokens, num_entries_per_token, num_sf_packs,
            sf_dtype=sf_dtype, device=device)

    def write(self, storages: List[torch.Tensor], sfs: Optional[List[torch.Tensor]] = None) -> None:
        """DeepInfer's stock path: this rank's shard [num_entries, hidden] per layer (+ the all-gathered
        packed sf [num_ranks * num_entries, packs]). Kept by reference, not copied."""
        import sm120_engram

        if self._fetcher is None:
            raise RuntimeError("EngramBuffer.write before set_config")
        if self.num_ranks > 1:
            raise NotImplementedError(
                "SM120 EngramBuffer.write: with engram_parallel_size > 1 a rank only holds its shard, but the "
                "shim has no remote reads. Use the checkpoint-backed tables (sm120_patches_engram) instead.")
        tables = []
        for i, w in enumerate(storages):
            if w.is_cuda and not _is_device_memory(w):
                raise RuntimeError(
                    "SM120 EngramBuffer.write got a host tensor relabelled as CUDA (EngramRuntime.initialize's "
                    "cuda_view_for_write); install sm120_patches_engram so the CPU table is passed as is.")
            sf = None if sfs is None else sfs[i]
            if sf is not None and w.is_cuda != sf.is_cuda:
                sf = sf.to(w.device)
            tables.append(sm120_engram.EngramHostTable.from_tensors(w, sf, name=f"engram[{i}]"))
        self._fetcher.attach(tables)

    def attach_tables(self, tables: Sequence, row_offsets: Optional[Sequence[int]] = None) -> None:
        """SM120 extension: per-layer `sm120_engram.EngramHostTable` covering global row ids."""
        if self._fetcher is None:
            raise RuntimeError("EngramBuffer.attach_tables before set_config")
        self._fetcher.attach(tables, row_offsets)

    def prestage(self, indices: torch.Tensor) -> None:
        """SM120 extension: gather the rows for the next CUDA-graph replay into the static buffers that the
        hooks captured in the graph read (see sm120_patches_engram E6)."""
        self._fetcher.prestage(indices)

    @property
    def fetcher(self):
        return self._fetcher

    def fetch(self, indices: torch.Tensor, num_qps: int = 0, use_tma_aligned_col_major_sf: bool = True) -> List[Callable]:
        """indices int32 [num_layers, num_tokens, num_entries_per_token] -> one hook per layer returning
        (fp8 [T, E*hidden], int32 sf [T, E*packs]) for FP8 tables, else a bf16 [T, E*hidden] tensor."""
        if self._fetcher is None:
            raise RuntimeError("EngramBuffer.fetch before set_config")
        return self._fetcher.fetch(indices, use_tma_aligned_col_major_sf)


def _is_device_memory(t: torch.Tensor) -> bool:
    """False for host memory that was relabelled as a CUDA tensor (DeepInfer's cuda_view_for_write)."""
    try:
        from cuda.bindings import runtime as rt

        err, attrs = rt.cudaPointerGetAttributes(t.data_ptr())
        if err != rt.cudaError_t.cudaSuccess:
            return True
        return attrs.type in (rt.cudaMemoryType.cudaMemoryTypeDevice, rt.cudaMemoryType.cudaMemoryTypeManaged,
                              rt.cudaMemoryType.cudaMemoryTypeHost)
    except Exception:  # noqa: BLE001  (no cuda-python: trust the tensor)
        return True
