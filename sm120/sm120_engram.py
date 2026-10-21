"""SM120 Engram: serve DeepSeek-V4.1 n-gram table lookups from host RAM or NVMe instead of DeepEP RDMA.

DeepInfer keeps each Engram table (layer 1 and 14 of V4.1-Flash: 384M rows x 256 FP8 bytes + 8 E8M0
scale bytes, ~101.5 GB each) sharded over the engram ranks and fetches rows with DeepEP's
GPU-initiated RDMA gets (`deep_ep.EngramBuffer`). SM120 boxes have neither NVLink nor the NCCL GIN/RDMA
path, and the tables do not fit into GPU memory. This module provides:

* `EngramHostTable`: one table, addressed by row id. Backed by
    - an mmap of the HF safetensors shard (default: no copy, the kernel page cache is the RAM tier,
      shared by every process on the machine),
    - plain tensors (CPU or CUDA; tests, small models, DeepInfer's stock `EngramBuffer.write` path),
    - optionally CUDA-registered ("uva") mmap pages, so a GPU kernel gathers rows over PCIe,
    - optionally b12x's O_DIRECT io_uring row reader ("uring", `b12x.sequence._shared.disk_table`).
* `EngramFetcher`: the engine behind the `deep_ep.EngramBuffer` shim. `fetch(indices)` returns one hook
  per layer; a hook returns exactly what DeepEP returns:
  ``(fp8 [T, E*hidden], packed-ue8m0 int32 sf [T, E*hidden/128])`` with the sf MN-major and TMA aligned
  (strides ``(1, align(T, 4))``), or a bf16 tensor for bf16 tables.
* Torch versions of the tile_kernels Engram ops DeepInfer calls (`engram_hash`, `fused_weight`,
  `engram_gate_fwd`) and of DeepInfer's hash metadata / C++ n-gram window builder, used by
  `tests/tests_engram.py` and, with ``SM120_ENGRAM_TK=torch``, by `sm120_patches_engram.py`.

Environment (all optional):
    SM120_ENGRAM_MODE=auto|host|uva|uring   how a checkpoint-backed table is read (default auto = host)
    SM120_ENGRAM_THREADS=<n>                CPU gather threads (default 32; page faults are I/O bound)
    SM120_ENGRAM_CHUNK=<rows>               rows per gather task (default 4096)
    SM120_ENGRAM_ADVICE=random|normal       madvise for the mmap (default random: no 128 KiB readahead per row)
    SM120_ENGRAM_PREFAULT=0|1               read the whole table into the page cache ("ram mode"): at attach, or
                                            with SM120_ENGRAM_PREFAULT_AT=ready (server default) once all ranks
                                            have loaded their weights (post_load_cache_policy)
    SM120_ENGRAM_DROP_WEIGHT_PAGES=1|0      server: drop the other shards' page-cache pages after load
    SM120_ENGRAM_STATS=0|1                  print fetch / gather / wait statistics at exit and every
                                            SM120_ENGRAM_STATS_INTERVAL seconds (default 60) while they change
    SM120_ENGRAM_CKPT=<dir>                 directory used to resolve shards (default: the loader's hf_dir)
    SM120_ENGRAM_URING_QD=<n>               io_uring queue depth for mode=uring (default 128)

See docs/ENGRAM.md for the semantics and DeepInfer's data path.
"""

from __future__ import annotations

import json
import math
import mmap
import os
import re
import struct
import sys
import threading
import time
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch

E8M0_ONE = 127  # E8M0 byte of 2^0
FP8_GROUP = 32  # one E8M0 scale per 32 FP8 values along a row


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _align(x: int, y: int) -> int:
    return (x + y - 1) // y * y


# ------------------------------------------------------------------------------------------------ safetensors
@dataclass(frozen=True)
class TensorLoc:
    """Byte location of one tensor inside a safetensors file (offset is absolute in the file)."""

    path: str
    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int


_HEADER_CACHE: dict[str, tuple[dict, int]] = {}


def read_safetensors_header(path: str) -> tuple[dict, int]:
    """(header json, absolute offset of the data buffer). Reads only the header bytes."""
    path = os.path.realpath(path)
    hit = _HEADER_CACHE.get(path)
    if hit is not None:
        return hit
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        if n <= 0 or n > 512 << 20:
            raise ValueError(f"{path}: implausible safetensors header length {n}")
        header = json.loads(f.read(n))
    _HEADER_CACHE[path] = (header, 8 + n)
    return header, 8 + n


def locate_tensor(path: str, name: str) -> TensorLoc:
    header, start = read_safetensors_header(path)
    if name not in header:
        raise KeyError(f"{name} not in {path}")
    info = header[name]
    begin, end = info["data_offsets"]
    return TensorLoc(os.path.realpath(path), name, info["dtype"], tuple(info["shape"]), start + begin, end - begin)


def _engram_names(names, layer_id: int) -> tuple[str | None, list[str]]:
    pat = re.compile(rf"(^|\.)layers\.{layer_id}\.engram\.embed\.weight$")
    weights = [n for n in names if pat.search(n)]
    if len(weights) > 1:
        raise ValueError(f"several Engram tables for layer {layer_id}: {weights}")
    if not weights:
        return None, []
    w = weights[0]
    stem = w[: -len(".weight")]
    return w, [stem + ".scale", stem + ".weight_scale_inv"]


def find_engram_tensors(ckpt_dir: str, layer_id: int) -> tuple[TensorLoc, TensorLoc]:
    """Locate ``layers.{layer_id}.engram.embed.{weight,scale}`` in an HF checkpoint directory (via
    model.safetensors.index.json, else by scanning the shard headers). `layer_id` is the layer index
    *as named in that checkpoint*."""
    index_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        weight_map = json.load(open(index_path))["weight_map"]
        w, scales = _engram_names(weight_map, layer_id)
        if w is not None:
            wpath = os.path.join(ckpt_dir, weight_map[w])
            for s in scales:
                if s in weight_map:
                    return locate_tensor(wpath, w), locate_tensor(os.path.join(ckpt_dir, weight_map[s]), s)
            header, _ = read_safetensors_header(wpath)  # scale missing from the index: same shard?
            for s in scales:
                if s in header:
                    return locate_tensor(wpath, w), locate_tensor(wpath, s)
            raise KeyError(f"no scale tensor ({scales}) for {w} in {ckpt_dir}")
    for fn in sorted(os.listdir(ckpt_dir)):
        if not fn.endswith(".safetensors"):
            continue
        path = os.path.join(ckpt_dir, fn)
        header, _ = read_safetensors_header(path)
        w, scales = _engram_names(header, layer_id)
        if w is None:
            continue
        for s in scales:
            if s in header:
                return locate_tensor(path, w), locate_tensor(path, s)
        raise KeyError(f"no scale tensor ({scales}) next to {w} in {path}")
    raise FileNotFoundError(f"no Engram table for layer {layer_id} under {ckpt_dir}")


# ------------------------------------------------------------------------------------------------ mmaps
_MMAPS: dict[str, mmap.mmap] = {}
_MMAP_LOCK = threading.Lock()


def _mmap_file(path: str) -> mmap.mmap:
    path = os.path.realpath(path)
    with _MMAP_LOCK:
        mm = _MMAPS.get(path)
        if mm is None:
            with open(path, "rb") as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            _MMAPS[path] = mm
        return mm


def _madvise(mm: mmap.mmap, advice: str, offset: int = 0, length: int | None = None) -> None:
    flag = {"random": "MADV_RANDOM", "normal": "MADV_NORMAL", "willneed": "MADV_WILLNEED",
            "sequential": "MADV_SEQUENTIAL", "dontneed": "MADV_DONTNEED"}[advice]
    if not hasattr(mmap, flag):
        return
    start = offset // mmap.PAGESIZE * mmap.PAGESIZE
    end = len(mm) if length is None else min(len(mm), offset + length)
    try:
        mm.madvise(getattr(mmap, flag), start, end - start)
    except OSError as exc:  # advisory only
        warnings.warn(f"madvise({flag}) failed: {exc}", RuntimeWarning, stacklevel=2)


def drop_page_cache(path: str, offset: int = 0, length: int = 0) -> None:
    """posix_fadvise(DONTNEED): evict clean cached pages of `path` (no root needed). Used by the
    benchmark for cold-cache numbers; pages that are still mapped and referenced may survive."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def resident_bytes(path: str, offset: int = 0, length: int | None = None, chunk: int = 1 << 30) -> int | None:
    """Bytes of path[offset:offset+length] in the page cache (mincore(2) on a temporary mapping), None if unknown."""
    try:
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        libc.mmap.restype = ctypes.c_void_p
        libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
        libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        pg = mmap.PAGESIZE
        fd = os.open(path, os.O_RDONLY)
        try:
            size = os.fstat(fd).st_size
            end = size if length is None else min(size, offset + length)
            start = offset // pg * pg
            total = 0
            for a in range(start, end, chunk):
                n = min(chunk, end - a)
                p = libc.mmap(None, n, mmap.PROT_READ, mmap.MAP_SHARED, fd, a)
                if p in (None, ctypes.c_void_p(-1).value):
                    return None
                try:
                    vec = np.zeros((n + pg - 1) // pg, dtype=np.uint8)
                    if libc.mincore(p, n, vec.ctypes.data) != 0:
                        return None
                    total += int(np.count_nonzero(vec & 1)) * pg
                finally:
                    libc.munmap(p, n)
            return total
        finally:
            os.close(fd)
    except Exception:  # noqa: BLE001 - diagnostics only
        return None


_TABLES: list = []  # every checkpoint-backed table opened in this process (post_load_cache_policy)


def post_load_cache_policy(ckpt_dir: str | None) -> dict:
    """Page-cache policy after the weights of every rank are loaded (server: once, on one process per host).

    1. posix_fadvise(DONTNEED) on the checkpoint's other *.safetensors shards: their pages served the weight load
       and now only compete with the Engram tables for the page cache (clean, read-only pages; pages a process
       still maps stay). SM120_ENGRAM_DROP_WEIGHT_PAGES=0 skips it.
    2. If SM120_ENGRAM_PREFAULT=1 with SM120_ENGRAM_PREFAULT_AT=ready: prefault the tables now.
    3. Log the tables' page-cache residency."""
    t0 = time.time()
    res = {}
    tables = [t for t in _TABLES if t.source is not None and t._uring is None]
    engram_files = {os.path.realpath(loc.path) for t in tables for loc in t.source if loc is not None}
    if ckpt_dir and _env("SM120_ENGRAM_DROP_WEIGHT_PAGES", "1") not in ("0", "false", "no"):
        dropped, nbytes = 0, 0
        for name in sorted(os.listdir(ckpt_dir)):
            path = os.path.realpath(os.path.join(ckpt_dir, name))
            if not name.endswith(".safetensors") or path in engram_files or not os.path.isfile(path):
                continue
            try:
                drop_page_cache(path)
                dropped += 1
                nbytes += os.path.getsize(path)
            except OSError:
                pass
        res["dropped_shards"], res["dropped_gib"] = dropped, round(nbytes / 2**30, 1)
        print(f"[sm120_engram] page cache: dropped clean pages of {dropped} weight shards ({nbytes / 2**30:.0f} GiB of "
              f"files) after load (SM120_ENGRAM_DROP_WEIGHT_PAGES=0 keeps them)", flush=True)
    if (_env("SM120_ENGRAM_PREFAULT", "0") in ("1", "true", "yes") and _env("SM120_ENGRAM_PREFAULT_AT", "attach") == "ready"
            and tables):
        size, avail = sum(t.nbytes for t in tables), _mem_available()
        if avail is not None and size > 0.95 * avail:
            # reading tables that do not fit only churns the cache (and the disk) for minutes while serving
            res["prefault_s"] = None
            print(f"[sm120_engram] prefault skipped: tables {size / 2**30:.0f} GiB > MemAvailable {avail / 2**30:.0f} "
                  f"GiB; use SM120_ENGRAM_MODE=uring on this host (docs/ENGRAM.md)", flush=True)
        else:
            dt = sum(t.prefault() for t in tables)
            res["prefault_s"] = round(dt, 1)
            print(f"[sm120_engram] prefaulted {len(tables)} tables after load in {dt:.0f} s", flush=True)
    resident = [t.resident() for t in tables]
    if tables and all(r is not None for r in resident):
        size = sum(t.nbytes for t in tables)
        avail = _mem_available()
        res["resident_gib"], res["tables_gib"] = round(sum(resident) / 2**30, 1), round(size / 2**30, 1)
        print(f"[sm120_engram] engram tables resident in the page cache after load: {sum(resident) / 2**30:.1f} / "
              f"{size / 2**30:.1f} GiB ({100 * sum(resident) / max(size, 1):.0f} %); MemAvailable "
              f"{'?' if avail is None else f'{avail / 2**30:.1f} GiB'}", flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    return res


# ------------------------------------------------------------------------------------------------ table
def _scale_bytes(scale: torch.Tensor, rows: int) -> torch.Tensor:
    """Any Engram scale representation -> [rows, groups] uint8 E8M0 bytes (same device)."""
    if scale.dtype == torch.float32:  # DeepInfer's loader keeps fp32 powers of two
        bits = scale.contiguous().view(torch.int32)
        return torch.bitwise_right_shift(bits, 23).to(torch.uint8)
    if scale.dtype in (torch.int32, torch.int16):  # packed ue8m0 (DeepInfer gather_sf, DeepEP write)
        return scale.contiguous().view(torch.uint8).view(rows, -1)
    if scale.dtype in (torch.uint8, getattr(torch, "float8_e8m0fnu", torch.uint8)):
        return scale.contiguous().view(torch.uint8)
    raise TypeError(f"unsupported Engram scale dtype {scale.dtype}")


class EngramHostTable:
    """One Engram table addressed by row id: weight rows of `row_bytes` (FP8: hidden bytes, BF16: 2*hidden)
    and optional E8M0 scale rows of `groups` bytes. Rows outside ``[0, rows)`` (DeepInfer's image tokens
    hash to -1) read as zero weights with unit scales."""

    def __init__(self, weight_u8: torch.Tensor | None, scale_u8: torch.Tensor | None, *, rows: int,
                 row_bytes: int, groups: int, name: str = "",
                 source: tuple[TensorLoc, TensorLoc] | None = None) -> None:
        self.weight = weight_u8  # [rows, row_bytes] uint8 torch view (CPU, CUDA or UVA) or None
        self.scale = scale_u8  # [rows, groups] uint8 or None
        self.rows, self.row_bytes, self.groups = int(rows), int(row_bytes), int(groups)
        self.name = name
        self.source = source
        self.kind = "cuda" if weight_u8 is not None and weight_u8.is_cuda else "host"
        self._registered: list[int] = []
        self._uring = None

    # ---- constructors
    @classmethod
    def from_tensors(cls, weight: torch.Tensor, scale: torch.Tensor | None = None, name: str = "") -> "EngramHostTable":
        """weight: [rows, hidden] fp8/uint8 (or bf16); scale: fp32 / e8m0 / uint8 / packed int32 per row."""
        assert weight.dim() == 2, weight.shape
        rows = weight.shape[0]
        w = weight.contiguous().view(torch.uint8).view(rows, -1)
        s = None if scale is None else _scale_bytes(scale, rows).view(rows, -1)
        if s is not None:
            assert s.shape[0] == rows, (s.shape, rows)
        groups = 0 if s is None else s.shape[1]
        if s is not None:
            s = s.to(w.device)
        return cls(w, s, rows=rows, row_bytes=w.shape[1], groups=groups, name=name)

    @classmethod
    def from_locations(cls, wloc: TensorLoc, sloc: TensorLoc | None, name: str = "", *,
                       advice: str | None = None, prefault: bool | None = None) -> "EngramHostTable":
        """mmap a table straight out of its safetensors shard(s): no copy, no RAM beyond the page cache."""
        if wloc.dtype not in ("F8_E4M3", "U8", "I8", "BF16"):
            raise TypeError(f"{wloc.name}: unexpected Engram weight dtype {wloc.dtype}")
        rows = wloc.shape[0]
        row_bytes = wloc.nbytes // rows
        wmm = _mmap_file(wloc.path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # read-only buffer: torch warns, we never write
            w = torch.frombuffer(wmm, dtype=torch.uint8, count=rows * row_bytes, offset=wloc.offset).view(rows, row_bytes)
        s = None
        groups = 0
        if sloc is not None:
            if sloc.dtype not in ("F8_E8M0", "U8"):
                raise TypeError(f"{sloc.name}: unexpected Engram scale dtype {sloc.dtype}")
            assert sloc.shape[0] == rows, (sloc.shape, wloc.shape)
            groups = sloc.nbytes // rows
            smm = _mmap_file(sloc.path)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s = torch.frombuffer(smm, dtype=torch.uint8, count=rows * groups, offset=sloc.offset).view(rows, groups)
        table = cls(w, s, rows=rows, row_bytes=row_bytes, groups=groups, name=name or wloc.name,
                    source=(wloc, sloc))
        table.advise(advice or _env("SM120_ENGRAM_ADVICE", "random"))
        want = prefault if prefault is not None else _env("SM120_ENGRAM_PREFAULT", "0") in ("1", "true", "yes")
        # SM120_ENGRAM_PREFAULT_AT=ready (set by server/server.py): prefault later, in post_load_cache_policy(),
        # once every rank has loaded its weights (weight streaming through the page cache would evict the tables)
        if want and _env("SM120_ENGRAM_PREFAULT_AT", "attach") != "ready":
            table.prefault()
        _TABLES.append(table)
        return table

    @classmethod
    def from_checkpoint(cls, ckpt_dir: str, layer_id: int, **kw) -> "EngramHostTable":
        wloc, sloc = find_engram_tensors(ckpt_dir, layer_id)
        return cls.from_locations(wloc, sloc, name=f"layers.{layer_id}.engram", **kw)

    # ---- page cache control
    @property
    def nbytes(self) -> int:
        return self.rows * (self.row_bytes + self.groups)

    def advise(self, advice: str) -> None:
        if self.source is None:
            return
        for loc in self.source:
            if loc is not None:
                _madvise(_mmap_file(loc.path), advice, loc.offset, loc.nbytes)

    def resident(self) -> int | None:
        """Bytes of the table currently in the page cache (host tables backed by a shard), None if unknown."""
        if self.source is None:
            return None
        total = 0
        for loc in self.source:
            if loc is not None:
                r = resident_bytes(os.path.realpath(loc.path), loc.offset, loc.nbytes)
                if r is None:
                    return None
                total += r
        return total

    def prefault(self, threads: int | None = None, chunk_bytes: int = 256 << 20) -> float:
        """Read the whole table once (sequential, parallel) so later lookups hit the page cache. Returns
        seconds. On a machine with less free RAM than the table this only churns the cache: it warns."""
        if self.source is None:
            return 0.0
        avail = _mem_available()
        if avail is not None and self.nbytes > avail:
            warnings.warn(f"{self.name}: table {self.nbytes / 2**30:.1f} GiB > MemAvailable "
                          f"{avail / 2**30:.1f} GiB, prefault cannot keep it resident", RuntimeWarning, stacklevel=2)
        t0 = time.time()
        jobs = []
        for loc in self.source:
            if loc is None:
                continue
            mm = _mmap_file(loc.path)
            _madvise(mm, "sequential", loc.offset, loc.nbytes)
            flat = np.frombuffer(mm, dtype=np.uint8, count=loc.nbytes, offset=loc.offset)
            for a in range(0, loc.nbytes, chunk_bytes):
                jobs.append(flat[a : a + chunk_bytes])
        with ThreadPoolExecutor(max_workers=threads or int(_env("SM120_ENGRAM_THREADS", "32"))) as pool:
            list(pool.map(lambda x: int(x[:: mmap.PAGESIZE].sum()), jobs))
        self.advise(_env("SM120_ENGRAM_ADVICE", "random"))
        return time.time() - t0

    # ---- GPU access
    def to_uva(self, device: torch.device | str | None = None) -> "EngramHostTable":
        """Page-lock the mmap'd table with cudaHostRegister(Mapped|ReadOnly) and expose CUDA views, so
        the gather runs on the GPU (no host sync, CUDA-graph capturable). Needs RAM >= table size: the
        whole table becomes pinned page cache (shared between processes mapping the same file)."""
        if self.source is None:
            raise ValueError("uva mode needs a file-backed table")
        device = torch.device(device or "cuda")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        cudart = torch.cuda.cudart()
        views = []
        for loc, width in zip(self.source, (self.row_bytes, self.groups)):
            if loc is None:
                views.append(None)
                continue
            mm = _mmap_file(loc.path)
            base = np.frombuffer(mm, dtype=np.uint8, count=1).ctypes.data
            start = loc.offset // mmap.PAGESIZE * mmap.PAGESIZE
            end = min(len(mm), _align(loc.offset + loc.nbytes, mmap.PAGESIZE))
            host_ptr = base + start
            flags = 0x01 | 0x02 | 0x08  # Portable | Mapped | ReadOnly
            err = cudart.cudaHostRegister(host_ptr, end - start, flags)
            if int(getattr(err, "value", err)) != 0:
                raise RuntimeError(f"cudaHostRegister({(end - start) / 2**30:.1f} GiB of {loc.path}) failed: {err}")
            self._registered.append(host_ptr)
            dev_ptr = _device_pointer(host_ptr)
            nbytes = loc.nbytes
            storage = torch._C._construct_storage_from_data_pointer(dev_ptr + (loc.offset - start), device, nbytes)
            t = torch.empty(0, dtype=torch.uint8, device=device).set_(storage, 0, (self.rows, width), (width, 1))
            views.append(t)
        out = EngramHostTable(views[0], views[1], rows=self.rows, row_bytes=self.row_bytes, groups=self.groups,
                              name=self.name, source=self.source)
        out._registered, self._registered = self._registered, []
        out.kind = "uva"
        return out

    def to_uring(self, device: torch.device | str, max_lookups: int) -> "EngramHostTable":
        """Rows are read per batch with b12x's O_DIRECT io_uring reader into mapped-host staging."""
        if self.source is None:
            raise ValueError("uring mode needs a file-backed table")
        from b12x.sequence._shared.disk_table import DiskRowCache

        device = torch.device(device)
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        cache = DiskRowCache(device=device, max_lookups=max_lookups, table_rows=self.rows, shard_start=0,
                             shard_end=self.rows, shard_rows=self.rows, weight_row_bytes=self.row_bytes,
                             scale_row_bytes=self.groups, queue_depth=int(_env("SM120_ENGRAM_URING_QD", "128")))
        wloc, sloc = self.source
        cache.add_shard(0, wloc.path, wloc.offset)
        if sloc is not None:
            cache.add_shard(0, sloc.path, sloc.offset, scale=True)
        cache.freeze()
        out = EngramHostTable(None, None, rows=self.rows, row_bytes=self.row_bytes, groups=self.groups,
                              name=self.name, source=self.source)
        out.kind = "uring"
        out._uring = cache
        return out

    def close(self) -> None:
        if self._registered:
            cudart = torch.cuda.cudart()
            torch.cuda.synchronize()
            for p in self._registered:
                cudart.cudaHostUnregister(p)
            self._registered = []
        if self._uring is not None:
            self._uring.close()
            self._uring = None

    # ---- gathers
    def gather_host(self, idx: torch.Tensor, out_w: torch.Tensor, out_s: torch.Tensor | None,
                    pool: ThreadPoolExecutor | None = None, chunk: int = 4096) -> None:
        """CPU gather: idx int64 CPU [n] -> out_w [n, row_bytes], out_s [n, groups] (uint8, CPU/pinned).
        index_select releases the GIL, so `pool` threads fault pages in parallel (that is the NVMe queue
        depth of the mmap path); on cached rows it runs at memory bandwidth."""
        n = idx.shape[0]
        if n == 0:
            return

        bad = (idx < 0) | (idx >= self.rows)
        any_bad = bool(bad.any())
        sel_all = idx.masked_fill(bad, 0) if any_bad else idx
        planes = [(self.weight, out_w, 0)]
        if out_s is not None and self.scale is not None:
            planes.append((self.scale, out_s, E8M0_ONE))

        @torch.inference_mode()
        def run(src: torch.Tensor, out: torch.Tensor, a: int, b: int) -> None:
            torch.index_select(src, 0, sel_all[a:b], out=out[a:b])

        if pool is None:
            for src, out, _ in planes:
                run(src, out, 0, n)
        else:
            # weight and scale rows live on different pages: fault them in separate tasks; small batches
            # (decode) are spread over all threads so that cache misses fault in parallel
            workers = getattr(pool, "_max_workers", 1)
            # >= SM120_ENGRAM_MIN_CHUNK rows per task: a task costs ~20-50 us of Python, which dominates cached
            # rows (decode on the full-depth model: 64 threads with 4-row tasks cost +1.5 ms per step)
            chunk = max(int(_env("SM120_ENGRAM_MIN_CHUNK", "64")), min(chunk, -(-n * len(planes) // workers)))
            futs = [pool.submit(run, src, out, a, min(a + chunk, n)) for src, out, _ in planes for a in range(0, n, chunk)]
            for f in futs:
                f.result()
        if any_bad:
            for _, out, fill in planes:
                out[bad] = fill

    def gather_cuda(self, idx: torch.Tensor, out_w: torch.Tensor, out_s: torch.Tensor | None) -> None:
        """GPU gather for device / UVA tables: idx int64 CUDA [n]; out_w [n, row_bytes], out_s [n, groups]."""
        valid = (idx >= 0) & (idx < self.rows)
        safe = torch.where(valid, idx, torch.zeros_like(idx))
        torch.index_select(self.weight, 0, safe, out=out_w)
        out_w.masked_fill_(~valid.unsqueeze(-1), 0)
        if out_s is not None and self.scale is not None:
            torch.index_select(self.scale, 0, safe, out=out_s)
            out_s.masked_fill_(~valid.unsqueeze(-1), E8M0_ONE)


def _device_pointer(host_ptr: int) -> int:
    try:
        from cuda.bindings import runtime as rt  # cuda-python (a b12x dependency)

        err, dev = rt.cudaHostGetDevicePointer(host_ptr, 0)
        if err == rt.cudaError_t.cudaSuccess:
            return int(dev)
    except Exception:  # noqa: BLE001
        pass
    return host_ptr  # UVA on 64-bit Linux: registered host memory has the same device address


def _mem_available() -> int | None:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def open_table(ckpt_dir: str | None, layer_id: int, *, wloc: TensorLoc | None = None, sloc: TensorLoc | None = None,
               mode: str | None = None, device: torch.device | str | None = None,
               max_lookups: int = 0) -> EngramHostTable:
    """Open a checkpoint-backed table in the requested mode (SM120_ENGRAM_MODE by default)."""
    mode = mode or _env("SM120_ENGRAM_MODE", "auto")
    if wloc is None:
        wloc, sloc = find_engram_tensors(ckpt_dir, layer_id)
    table = EngramHostTable.from_locations(wloc, sloc, name=f"layers.{layer_id}.engram",
                                           prefault=None if mode in ("auto", "host") else False)
    if mode in ("auto", "host"):
        return table
    if mode == "uva":
        return table.to_uva(device)
    if mode == "uring":
        return table.to_uring(device, max_lookups)
    raise ValueError(f"SM120_ENGRAM_MODE={mode!r}: expected auto, host, uva or uring")


# ------------------------------------------------------------------------------------------------ fetcher
class EngramFetcher:
    """Row fetch engine with DeepEP EngramBuffer.fetch semantics, for one process (no peers).

    indices: int32 [num_layers, T, E] global row ids (DeepInfer's `engram_hash` output, -1 for image
    tokens). Each layer's hook returns its gathered rows as `[T, E*hidden]` FP8 plus the per-row packed
    UE8M0 scales `[T, E*num_sf_packs]` (MN-major/TMA aligned when `use_tma_aligned_col_major_sf`), or
    `[T, E*hidden]` BF16 for BF16 tables.

    Host tables: one D2H of the indices and a stream sync, then a CPU gather per layer runs in the
    background (layer 14's gather overlaps the GPU work of layers 1..13); the hook waits for it and
    issues an async H2D copy from pinned staging. Device/UVA tables gather on the GPU inside the hook.
    """

    def __init__(self, num_entries_per_layer: Sequence[int], hidden: int, elem_size: int, num_max_tokens: int,
                 num_entries_per_token: int, num_sf_packs: int = 0, *, sf_dtype: torch.dtype = torch.int32,
                 device: torch.device | str = "cuda", threads: int | None = None) -> None:
        self.num_entries_per_layer = list(num_entries_per_layer)
        self.num_layers = len(self.num_entries_per_layer)
        self.hidden, self.elem_size = int(hidden), int(elem_size)
        self.row_bytes = self.hidden * self.elem_size
        self.num_max_tokens, self.num_entries_per_token = int(num_max_tokens), int(num_entries_per_token)
        self.num_sf_packs, self.sf_dtype = int(num_sf_packs), sf_dtype
        self.groups = self.num_sf_packs * sf_dtype.itemsize  # E8M0 bytes per row
        self.use_fp8 = self.num_sf_packs > 0
        if self.use_fp8:
            assert self.elem_size == 1 and self.groups == self.hidden // FP8_GROUP, (hidden, num_sf_packs, sf_dtype)
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        self.is_cuda = device.type == "cuda"
        self.tables: list[EngramHostTable | None] = [None] * self.num_layers
        self.row_offsets = [0] * self.num_layers
        self.threads = int(threads or _env("SM120_ENGRAM_THREADS", "32"))
        self.chunk = int(_env("SM120_ENGRAM_CHUNK", "4096"))
        self._pool: ThreadPoolExecutor | None = None
        self._layer_pool: ThreadPoolExecutor | None = None
        self._bufs: dict | None = None
        self._futures: list[Future | None] = [None] * self.num_layers
        self._gen = 0
        self.owns_tables = True  # close() closes the attached tables (False when they are shared)
        self.graph_mode = None  # "prestage" | "async" once hooks were captured into a CUDA graph
        self._async = None
        self._lock = threading.Lock()
        self.stats = {"fetches": 0, "rows": 0, "gather_s": 0.0, "wait_s": 0.0, "sync_s": 0.0}
        _FETCHERS.append(self)

    # ---- setup
    def attach(self, tables: Sequence[EngramHostTable], row_offsets: Sequence[int] | None = None) -> None:
        assert len(tables) == self.num_layers, (len(tables), self.num_layers)
        for t in tables:
            assert t.row_bytes == self.row_bytes, (t.name, t.row_bytes, self.row_bytes)
            if self.use_fp8:
                assert t.groups == self.groups, (t.name, t.groups, self.groups)
        self.tables = list(tables)
        self.row_offsets = list(row_offsets or [0] * self.num_layers)
        if any(t.kind == "host" for t in tables) and self.threads > 1 and self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self.threads, thread_name_prefix="sm120-engram")
            self._layer_pool = ThreadPoolExecutor(max_workers=max(1, self.num_layers), thread_name_prefix="sm120-engram-l")

    def close(self) -> None:
        if self._async is not None:
            self._async.stop()
            self._async = None
        self._drain()
        for p in (self._pool, self._layer_pool):
            if p is not None:
                p.shutdown(wait=True)
        self._pool = self._layer_pool = None
        if self.owns_tables:
            for t in self.tables:
                if t is not None:
                    t.close()

    def _buffers(self) -> dict:
        if self._bufs is None:
            with torch.inference_mode(False):  # normal tensors: the gather threads are not in inference mode
                self._bufs = self._alloc_buffers()
        return self._bufs

    def _alloc_buffers(self) -> dict:
        L, n = self.num_layers, self.num_max_tokens * self.num_entries_per_token
        pin = self.is_cuda
        host = any(t is not None and t.kind == "host" for t in self.tables)
        b: dict = {"idx_host": None, "stage_w": [None] * L, "stage_s": [None] * L, "dev_w": [], "dev_s": [], "ids64": None}
        if host:
            b["idx_host"] = torch.empty((L, self.num_max_tokens, self.num_entries_per_token), dtype=torch.int32,
                                        device="cpu", pin_memory=pin)
        for l, t in enumerate(self.tables):
            if t is not None and t.kind == "host":
                b["stage_w"][l] = torch.empty((n, self.row_bytes), dtype=torch.uint8, device="cpu", pin_memory=pin)
                b["stage_s"][l] = torch.empty((n, max(self.groups, 1)), dtype=torch.uint8, device="cpu", pin_memory=pin)
            b["dev_w"].append(torch.empty((n, self.row_bytes), dtype=torch.uint8, device=self.device)
                              if self.is_cuda or t is None or t.kind != "host" else b["stage_w"][l])
            b["dev_s"].append(torch.empty((n, max(self.groups, 1)), dtype=torch.uint8, device=self.device)
                              if self.is_cuda or t is None or t.kind != "host" else b["stage_s"][l])
        if any(t is not None and t.kind == "uring" for t in self.tables):
            b["ids64"] = torch.empty((L, n), dtype=torch.int64, device=self.device)
        return b

    def _drain(self) -> None:
        for i, f in enumerate(self._futures):
            if f is not None:
                try:
                    f.result()
                finally:
                    self._futures[i] = None

    # ---- fetch
    def fetch(self, indices: torch.Tensor, use_tma_aligned_col_major_sf: bool = True) -> list[Callable]:
        L, T, E = indices.shape
        assert L == self.num_layers and E == self.num_entries_per_token, (indices.shape, self.num_layers)
        assert T <= self.num_max_tokens, f"fetch of {T} tokens exceeds num_max_tokens={self.num_max_tokens}"
        assert indices.dtype == torch.int32 and indices.is_contiguous()
        assert all(t is not None for t in self.tables), "EngramFetcher: tables not attached (write/attach_tables first)"
        kinds = {t.kind for t in self.tables}
        if self.is_cuda and torch.cuda.is_current_stream_capturing() and kinds - {"cuda", "uva"}:
            if kinds <= {"host", "uring"} and _env("SM120_ENGRAM_GRAPH", "async") == "async":
                return self._async_graph_hooks(indices, use_tma_aligned_col_major_sf)
            return self._graph_hooks(T, use_tma_aligned_col_major_sf)
        with self._lock:
            self._drain()
            self._gen += 1
            gen = self._gen
            b = self._buffers()
            n = T * E
            if (self.is_cuda and self._async is None and kinds <= {"host", "uring"}
                    and _env("SM120_ENGRAM_GRAPH", "async") == "async"):
                self._async = _AsyncGraphFetch(self)  # eager: pinned buffers, thread, Triton JIT before capture
            if "host" in kinds:
                t0 = time.perf_counter()
                idx_host = b["idx_host"][:, :T]
                idx_host.copy_(indices, non_blocking=self.is_cuda)
                if self.is_cuda:
                    # also orders the previous fetch's H2D reads of the staging buffers before we rewrite them.
                    # An event sync: DeepInfer's forbid_cuda_sync guard does not cover events (utils.py:119).
                    ev = torch.cuda.Event()
                    ev.record(torch.cuda.current_stream(self.device))
                    ev.synchronize()
                self.stats["sync_s"] += time.perf_counter() - t0
                for l, t in enumerate(self.tables):
                    if t.kind == "host":
                        job = (self._gather_layer, l, n)
                        self._futures[l] = (self._layer_pool.submit(*job) if self._layer_pool is not None
                                            else _done(job[0](*job[1:])))
            if "uring" in kinds:
                for l, t in enumerate(self.tables):
                    if t.kind == "uring":
                        b["ids64"][l, :n].copy_(indices[l].reshape(-1))
                        if self.row_offsets[l]:
                            b["ids64"][l, :n].sub_(self.row_offsets[l])
            self.stats["fetches"] += 1
            self.stats["rows"] += L * n
        cache: dict[int, object] = {}
        return [self._make_hook(l, T, gen, indices, use_tma_aligned_col_major_sf, cache) for l in range(L)]

    # ---- CUDA graphs (host / uring tables)
    def _graph_hooks(self, T: int, tma: bool) -> list[Callable]:
        """Under CUDA-graph capture the hooks only read the static per-layer device buffers; `prestage()`
        fills them before every replay, outside the graph. The buffers exist from the eager warm-up run."""
        if self._bufs is None:
            raise RuntimeError("SM120 Engram: graph capture before any eager fetch (no static buffers yet)")
        self.graph_mode = "prestage"
        n = T * self.num_entries_per_token
        b = self._bufs
        return [(lambda l=l: self._format(b["dev_w"][l][:n], b["dev_s"][l][:n], T, tma)) for l in range(self.num_layers)]

    def _async_graph_hooks(self, indices: torch.Tensor, tma: bool) -> list[Callable]:
        """In-graph host fetch: the graph copies the hash ids to pinned memory and raises a host-visible
        flag; a host thread gathers the rows into the pinned staging buffers and raises per-layer done
        flags; the hook of layer l spins on its flag (GPU kernel), then copies the rows H2D. The gather
        overlaps the GPU work before the first engram layer, and no host code runs per replay."""
        if self._bufs is None:
            raise RuntimeError("SM120 Engram: graph capture before any eager fetch (no static buffers yet)")
        if self._async is None:
            raise RuntimeError("SM120 Engram: async graph fetch was not prepared by an eager fetch")
        self.graph_mode = "async"
        return self._async.capture(indices, tma)

    def prestage(self, indices: torch.Tensor) -> None:
        """Fill the static device buffers for the next graph replay: same gather as fetch() + hooks, on the
        current stream, eager. indices: int32 CUDA [num_layers, T, E] (T = the graph's padded batch)."""
        t_start = time.perf_counter()
        sync0 = self.stats["sync_s"]
        hooks = self.fetch(indices, use_tma_aligned_col_major_sf=False)
        self.stats["pre_sync_s"] = self.stats.get("pre_sync_s", 0.0) + self.stats["sync_s"] - sync0
        t_gather = time.perf_counter()
        for l, t in enumerate(self.tables):
            if t.kind == "host":  # wait for the CPU gather and issue the H2D copies (no formatting needed)
                fut = self._futures[l]
                if fut is not None:
                    fut.result()
                    self._futures[l] = None
                n = indices.shape[1] * self.num_entries_per_token
                b = self._bufs
                b["dev_w"][l][:n].copy_(b["stage_w"][l][:n], non_blocking=True)
                if self.use_fp8:
                    b["dev_s"][l][:n].copy_(b["stage_s"][l][:n], non_blocking=True)
            else:
                hooks[l]()
        self.stats["prestages"] = self.stats.get("prestages", 0) + 1
        now = time.perf_counter()
        self.stats["prestage_s"] = self.stats.get("prestage_s", 0.0) + now - t_start
        self.stats["pre_gather_s"] = self.stats.get("pre_gather_s", 0.0) + now - t_gather

    @torch.inference_mode()
    def _gather_layer(self, l: int, n: int) -> None:
        t0 = time.perf_counter()
        b = self._bufs
        idx = b["idx_host"][l, : n // self.num_entries_per_token].reshape(-1).to(torch.int64)
        if self.row_offsets[l]:
            idx -= self.row_offsets[l]
        t = self.tables[l]
        t.gather_host(idx, b["stage_w"][l][:n], b["stage_s"][l][:n] if t.groups else None, pool=self._pool,
                      chunk=self.chunk)
        self.stats["gather_s"] += time.perf_counter() - t0

    def _make_hook(self, l: int, T: int, gen: int, indices: torch.Tensor, tma: bool, cache: dict) -> Callable:
        def hook():
            if l in cache:
                return cache[l]
            if gen != self._gen:
                raise RuntimeError("SM120 Engram: stale fetch hook (a newer fetch() was issued)")
            n = T * self.num_entries_per_token
            b = self._bufs
            t = self.tables[l]
            dev_w, dev_s = b["dev_w"][l][:n], b["dev_s"][l][:n]
            if t.kind == "host":
                t0 = time.perf_counter()
                fut = self._futures[l]
                if fut is not None:
                    fut.result()
                    self._futures[l] = None
                self.stats["wait_s"] += time.perf_counter() - t0
                if self.is_cuda:
                    dev_w.copy_(b["stage_w"][l][:n], non_blocking=True)
                    if self.use_fp8:
                        dev_s.copy_(b["stage_s"][l][:n], non_blocking=True)
            elif t.kind in ("cuda", "uva"):
                idx = indices[l].reshape(-1).to(torch.int64)
                if self.row_offsets[l]:
                    idx = idx - self.row_offsets[l]
                t.gather_cuda(idx, dev_w, dev_s if self.use_fp8 else None)
            elif t.kind == "uring":
                cache_ = t._uring
                with cache_.transaction():
                    cache_.read_rows(b["ids64"][l], n)
                    dev_w.copy_(cache_.weight[:n])
                    if self.use_fp8:
                        dev_s.copy_(cache_.scale[:n])
            else:
                raise RuntimeError(f"unknown table kind {t.kind}")
            out = self._format(dev_w, dev_s, T, tma)
            cache[l] = out
            return out

        return hook

    def _format(self, dev_w: torch.Tensor, dev_s: torch.Tensor, T: int, tma: bool):
        E = self.num_entries_per_token
        if not self.use_fp8:
            return dev_w.view(torch.bfloat16 if self.elem_size == 2 else torch.float8_e4m3fn).view(T, E * self.hidden)
        data = dev_w.view(torch.float8_e4m3fn).view(T, E * self.hidden)
        packs = E * self.num_sf_packs
        sf = dev_s.view(self.sf_dtype).view(T, packs)  # per row: E entries x num_sf_packs, little-endian groups
        if tma:
            backing = torch.empty((packs, _align(T, 16 // self.sf_dtype.itemsize)), dtype=self.sf_dtype, device=sf.device)
            backing[:, :T].copy_(sf.t())
            sf = backing[:, :T].t()  # [T, packs], strides (1, align(T, 4))
        else:
            sf = sf.clone()
        return data, sf


_FETCHERS: list = []


def _host_view_on_device(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """CUDA alias of a pinned host tensor (UVA: cudaHostAlloc'd memory has the same device address)."""
    storage = torch._C._construct_storage_from_data_pointer(t.data_ptr(), device, t.numel() * t.element_size())
    return torch.empty(0, dtype=t.dtype, device=device).set_(storage, 0, tuple(t.shape), t.stride())


_TRITON = None


def _triton_kernels():
    global _TRITON
    if _TRITON is None:
        import triton
        import triton.language as tl

        @triton.jit
        def signal_kernel(cnt_ptr, ready_ptr, T):
            c = tl.load(cnt_ptr) + 1
            tl.store(cnt_ptr, c)
            tl.store(ready_ptr, c * 1048576 + T)  # one 64-bit store: generation and batch size

        @triton.jit
        def wait_kernel(cnt_ptr, done_ptr, err_ptr, max_iters, stat_ptr):
            c = tl.load(cnt_ptr)
            v = tl.load(done_ptr, volatile=True)
            i = 0
            while (v < c) & (i < max_iters):
                v = tl.load(done_ptr, volatile=True)
                i += 1
            if v < c:
                tl.store(err_ptr, c)
            tl.atomic_add(stat_ptr, i)  # spin iterations (each one host-memory read over PCIe, ~1 us)

        _TRITON = (signal_kernel, wait_kernel)
    return _TRITON


class _AsyncGraphFetch:
    """Host side of EngramFetcher's in-graph fetch: one service thread per fetcher."""

    MAX_SPIN = 1 << 31  # GPU wait-loop iterations before giving up (seconds); the error flag is then raised

    def __init__(self, fetcher: "EngramFetcher") -> None:
        f = self.f = fetcher
        dev = f.device
        L, n = f.num_layers, f.num_max_tokens * f.num_entries_per_token
        with torch.inference_mode(False):
            self.idx = torch.zeros((L * n,), dtype=torch.int32, device="cpu", pin_memory=True)
            self.flags = torch.zeros((2 + L,), dtype=torch.int64, device="cpu", pin_memory=True)  # ready, err, done[L]
            self.cnt = torch.zeros((1,), dtype=torch.int64, device=dev)
            self.spins = torch.zeros((fetcher.num_layers,), dtype=torch.int64, device=dev)  # per-layer spin sum
        self.flags_dev = _host_view_on_device(self.flags, dev)
        self.idx_dev = _host_view_on_device(self.idx, dev)
        self.np = self.flags.numpy()
        self.served = 0
        signal_kernel, wait_kernel = _triton_kernels()  # compile now, never inside a capture
        scratch = torch.zeros((3,), dtype=torch.int64, device=dev)
        signal_kernel[(1,)](scratch[0:1], scratch[1:2], 1)
        wait_kernel[(1,)](scratch[2:3], scratch[0:1], scratch[1:2], 1, scratch[1:2])  # done (1) >= cnt (0)
        torch.cuda.synchronize(dev)
        self._stop = False
        # The service thread must get the GIL quickly once the GPU raises the flag; with the default 5 ms
        # switch interval it can wait behind DeepInfer's scheduler thread for a whole interval.
        si = float(_env("SM120_ENGRAM_SWITCH_INTERVAL", "0.0002"))
        if si > 0 and sys.getswitchinterval() > si:
            sys.setswitchinterval(si)
        self.thread = threading.Thread(target=self._serve, name="sm120-engram-graph", daemon=True)
        self.thread.start()

    def capture(self, indices: torch.Tensor, tma: bool) -> list[Callable]:
        signal_kernel, wait_kernel = _triton_kernels()
        f, b = self.f, self.f._bufs
        L, T, E = indices.shape
        n = T * E
        # all recorded into the graph: D2H of the hash ids through the device alias of pinned memory, then the flag
        self.idx_dev[: L * n].view(L, T, E).copy_(indices)
        signal_kernel[(1,)](self.cnt, self.flags_dev[0:1], T)
        hooks = []
        for l in range(L):
            t = f.tables[l]
            if t.kind == "uring":  # rows land in b12x's mapped-host staging (CUDA views)
                src_w, src_s = t._uring.weight, t._uring.scale
            else:
                src_w, src_s = b["stage_w"][l], b["stage_s"][l]

            def hook(l=l, src_w=src_w, src_s=src_s):
                wait_kernel[(1,)](self.cnt, self.flags_dev[2 + l : 3 + l], self.flags_dev[1:2], self.MAX_SPIN,
                                  self.spins[l : l + 1])
                b["dev_w"][l][:n].copy_(src_w[:n], non_blocking=True)
                if f.use_fp8:
                    b["dev_s"][l][:n].copy_(src_s[:n], non_blocking=True)
                return f._format(b["dev_w"][l][:n], b["dev_s"][l][:n], T, tma)

            hooks.append(hook)
        return hooks

    @torch.inference_mode()
    def _serve(self) -> None:
        f = self.f
        idle = 0
        while not self._stop:
            v = int(self.np[0])
            gen, T = v >> 20, v & 0xFFFFF
            if gen <= self.served:
                idle += 1
                time.sleep(0 if idle < 2000 else 5e-5)
                continue
            idle = 0
            t0 = time.perf_counter()
            E, L = f.num_entries_per_token, f.num_layers
            n = T * E
            b = f._bufs
            for l in range(L):  # layer order: the first engram layer is released first
                idx = self.idx[l * n : (l + 1) * n].to(torch.int64)
                if f.row_offsets[l]:
                    idx -= f.row_offsets[l]
                t = f.tables[l]
                if t.kind == "uring":
                    c = t._uring  # b12x DiskRowCache: drive its native reader directly from host ids
                    c.ids_host[:n].copy_(idx)
                    c._native.ple_reader_run(c._reader, c._ids_buffer, c._weight_buffer, c._scale_buffer, n)
                else:
                    t.gather_host(idx, b["stage_w"][l][:n], b["stage_s"][l][:n] if t.groups else None,
                                  pool=f._pool, chunk=f.chunk)
                self.np[2 + l] = gen
                key = f"graph_ready_l{l}_s"  # flag seen -> this layer's rows staged
                f.stats[key] = f.stats.get(key, 0.0) + time.perf_counter() - t0
            self.served = gen
            f.stats["graph_fetches"] = f.stats.get("graph_fetches", 0) + 1
            f.stats["graph_gather_s"] = f.stats.get("graph_gather_s", 0.0) + time.perf_counter() - t0
            if int(self.np[1]):
                print(f"[sm120_engram] ERROR: GPU engram wait timed out at generation {int(self.np[1])}", flush=True)

    def stop(self) -> None:
        self._stop = True
        self.thread.join(timeout=5)




def _dump_stats(only_changed: bool = False) -> None:
    for f in _FETCHERS:
        st = dict(f.stats)
        if only_changed:
            seen = (st.get("fetches", 0), st.get("graph_fetches", 0), st.get("prestages", 0))
            if getattr(f, "_stats_seen", None) == seen:
                continue
            f._stats_seen = seen
        if f._async is not None:
            try:
                g = max(1, st.get("graph_fetches", 0))
                st["gpu_spin_iters_per_replay"] = [round(x / g, 1) for x in f._async.spins.tolist()]
                st["graph_gather_ms_per_replay"] = round(st.get("graph_gather_s", 0.0) / g * 1e3, 3)
                st["graph_ready_ms_per_replay"] = [round(st.get(f"graph_ready_l{l}_s", 0.0) / g * 1e3, 3)
                                                   for l in range(f.num_layers)]
            except Exception:  # noqa: BLE001
                pass
        n = max(1, st["fetches"])
        print(f"[sm120_engram] stats (pid {os.getpid()}, {f.device}) {st} | per fetch: sync {st['sync_s'] / n * 1e3:.3f} ms, "
              f"gather {st['gather_s'] / n * 1e3:.3f} ms, hook wait {st['wait_s'] / n * 1e3:.3f} ms, "
              f"prestage {st.get('prestage_s', 0.0) / max(1, st.get('prestages', 0)) * 1e3:.3f} ms "
              f"(sync {st.get('pre_sync_s', 0.0) / max(1, st.get('prestages', 0)) * 1e3:.3f}, gather+H2D "
              f"{st.get('pre_gather_s', 0.0) / max(1, st.get('prestages', 0)) * 1e3:.3f})", flush=True)


def _stats_loop(interval: float) -> None:
    while True:
        time.sleep(interval)
        try:
            _dump_stats(only_changed=True)
        except Exception:  # noqa: BLE001 - diagnostics only
            pass


if os.environ.get("SM120_ENGRAM_STATS", "0") not in ("0", ""):
    import atexit

    atexit.register(_dump_stats)
    # also every SM120_ENGRAM_STATS_INTERVAL seconds (default 60; 0 = only at exit), when something changed: a
    # server stopped with SIGKILL (docker stop timeout) never reaches atexit
    _iv = float(os.environ.get("SM120_ENGRAM_STATS_INTERVAL", "60"))
    if _iv > 0:
        threading.Thread(target=_stats_loop, args=(_iv,), name="sm120-engram-stats", daemon=True).start()


def _done(value) -> Future:
    f: Future = Future()
    f.set_result(value)
    return f


# ------------------------------------------------------------------------------------------------ hash metadata
def isprime(n: int) -> bool:
    """Deterministic Miller-Rabin (exact for n < 3.3e24)."""
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def layout_primes(layer_ids: Sequence[int], max_ngram_size: int, vocab_size: int, n_heads: int) -> list[list[list[int]]]:
    """[layer][ngram-1][head] bucket primes, drawn in order, never reused (engram.py:EngramLayout and
    DeepInfer engram_metadata.py:EngramLayout agree)."""
    seen: set[int] = set()
    out = []
    for _ in layer_ids:
        per = []
        for _ in range(max_ngram_size - 1):
            cur, sizes = vocab_size - 1, []
            for _ in range(n_heads):
                c = cur + 1
                while not isprime(c) or c in seen:
                    c += 1
                seen.add(c)
                sizes.append(c)
                cur = c
            per.append(sizes)
        out.append(per)
    return out


def hash_tensors(layer_ids: Sequence[int], max_ngram_size: int, vocab_size: int, n_heads: int,
                 compressed_vocab_size: int, seed: int = 0, device="cpu"):
    """DeepInfer NgramHashMapping (engram_metadata.py:85-116) without tokenizer/tile_kernels:
    multipliers int64 [L, max_ngram], vocab_sizes int32 [L, max_ngram-1, heads], offsets int32 [L, cols]."""
    primes = layout_primes(layer_ids, max_ngram_size, vocab_size, n_heads)
    bound = max(1, (np.iinfo(np.int64).max // compressed_vocab_size) // 2)
    mult = []
    for lid in layer_ids:
        rng = np.random.default_rng(int(seed + 10007 * int(lid)))
        mult.append(rng.integers(0, bound, size=(max_ngram_size,), dtype=np.int64) * 2 + 1)
    multipliers = torch.tensor(np.stack(mult), dtype=torch.int64, device=device)
    vocab_sizes = torch.tensor(primes, dtype=torch.int32, device=device)
    offsets = make_offsets(vocab_sizes)
    return multipliers, vocab_sizes, offsets


def make_offsets(vocab_sizes: torch.Tensor) -> torch.Tensor:
    """tile_kernels.torch.engram.make_offsets."""
    rows = []
    for l in range(vocab_sizes.shape[0]):
        flat = vocab_sizes[l].reshape(-1)
        rows.append(torch.cat([torch.zeros(1, dtype=torch.int32, device=flat.device),
                               flat[:-1].cumsum(0, dtype=torch.int32)]))
    return torch.stack(rows)


def build_ngram_token_ids(compressed: Sequence[int], positions: Sequence[int], max_ngram_size: int,
                          pad: int, image_sentinel: int = -1) -> torch.Tensor:
    """DeepInfer's C++ `fill_engram_row` (csrc/engine/input_metadata/prepare_tensor_input_base.cpp:10-35):
    row r of position p is the compressed id at p-r, PAD before the sequence start and from the first
    image token (sentinel) on. Returns int32 [len(positions), max_ngram_size]."""
    out = torch.empty((len(positions), max_ngram_size), dtype=torch.int32, device="cpu")
    for i, pos in enumerate(positions):
        hit = False
        for r in range(max_ngram_size):
            if hit:
                out[i, r] = pad
                continue
            s = pos - r
            v = pad if s < 0 else int(compressed[s])
            if v == image_sentinel:
                hit, v = True, pad
            out[i, r] = v
    return out


# ------------------------------------------------------------------------------------------------ torch ops
def engram_hash(ngram_token_ids: torch.Tensor, multipliers: torch.Tensor, vocab_sizes: torch.Tensor,
                offsets: torch.Tensor, image_token_mask: torch.Tensor | None = None) -> torch.Tensor:
    """tile_kernels.engram.engram_hash (torch; same as tile_kernels.torch.engram.engram_hash_ref).
    ngram_token_ids int32 [T, N] -> int32 [L, T, (N-1)*heads] global row ids, -1 on image tokens."""
    L, N = multipliers.shape
    prod = ngram_token_ids.to(torch.int64).unsqueeze(0) * multipliers.unsqueeze(1)  # [L, T, N]
    h = prod[:, :, 0].clone()
    cols = []
    for i in range(1, N):
        h = torch.bitwise_xor(h, prod[:, :, i])
        cols.append(h.unsqueeze(-1) % vocab_sizes[:, i - 1].to(torch.int64).unsqueeze(1))  # [L, T, heads]
    out = (torch.cat(cols, dim=-1) + offsets.to(torch.int64).unsqueeze(1)).to(torch.int32)
    if image_token_mask is not None:
        out = out.masked_fill(image_token_mask.view(1, -1, 1), -1)
    return out.contiguous()


def fused_weight(weight_hidden: torch.Tensor, weight_embed: torch.Tensor) -> torch.Tensor:
    """tile_kernels.engram.fused_weight: bf16 x bf16 -> fp32 elementwise product."""
    return weight_hidden.float() * weight_embed.float()


def engram_gate_fwd(hidden_states: torch.Tensor, kv: torch.Tensor, weight_fused: torch.Tensor, eps: float,
                    clamp_value: float, save_for_backward: bool = True, image_token_mask: torch.Tensor | None = None,
                    inplace: bool = False):
    """tile_kernels.engram.engram_gate_fwd (torch). hidden_states [T, hc, D] bf16, kv [T, hc+1, D] bf16,
    weight_fused [hc, D] fp32. out = x + sigmoid(signed_sqrt(<x*w, k> * rstd_x * rstd_k / sqrt(D))) * v."""
    T, hc, D = hidden_states.shape
    kv = kv.view(T, hc + 1, D)
    x = hidden_states.float()
    k = kv[:, :hc].float()
    v = kv[:, hc].float()
    rstd_x = torch.rsqrt(x.square().mean(-1) + eps)
    rstd_k = torch.rsqrt(k.square().mean(-1) + eps)
    raw = (x * weight_fused.float().unsqueeze(0) * k).sum(-1)
    dot = raw * rstd_x * rstd_k * D**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(clamp_value).sqrt(), dot))
    out = (x + gate.unsqueeze(-1) * v.unsqueeze(1)).to(hidden_states.dtype)
    if image_token_mask is not None:
        out = torch.where(image_token_mask.view(-1, 1, 1), hidden_states, out)
    if inplace:
        hidden_states.copy_(out)
        out = hidden_states
    if save_for_backward:
        return out, raw, gate, rstd_x, rstd_k
    return out, None, None, None, None


def dequant_rows(data: torch.Tensor, sf: torch.Tensor | None, hidden: int) -> torch.Tensor:
    """Fetched rows -> bf16 [T, E*hidden]: fp8 * 2^(e8m0-127) per 32 values (sf: packed int32 [T, E*packs]
    any strides, or uint8 [T, E*groups])."""
    if sf is None:
        return data.to(torch.bfloat16)
    T = data.shape[0]
    e = sf.contiguous().view(torch.uint8).view(T, -1).to(torch.int32)  # [T, E*groups]
    scale = (e - 127).float().exp2()  # byte 0 is 2^-127 (not zero), as in b12x / DeepGEMM
    vals = data.float().view(T, -1, FP8_GROUP) * scale.unsqueeze(-1)
    return vals.view(T, -1).to(torch.bfloat16)
