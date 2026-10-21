"""Minimal, dependency-free safetensors I/O used by the reference tooling.

Why not `safetensors` directly:
  * the truncation step must copy ~23 GB of FP4/FP8/E8M0 tensors byte-for-byte while staying far
    below 64 GB host RAM; `safetensors.torch.save_file` materialises a bytes copy of every tensor
    (`_flatten`/`_tobytes`), so a dict-based save would need ~2x the checkpoint size in RAM;
  * raw byte copies are dtype-agnostic (F8_E8M0, I8-packed FP4, ...), so no torch/safetensors
    version has to understand those dtypes for the copy to be lossless.

Format (https://github.com/huggingface/safetensors): u64-LE header length N, N bytes of JSON
header {name: {dtype, shape, data_offsets:[begin,end]}, "__metadata__": {str: str}}, then the data
buffer; offsets are relative to the start of the data buffer. We pad the header with spaces to a
multiple of 8 bytes, which is what the reference writer does.
"""

from __future__ import annotations

import json
import math
import os
import struct
from typing import Iterable, Iterator

# bytes per element for every dtype we can meet in the V4.1 checkpoint
DTYPE_SIZES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
}

MAX_HEADER = 100_000_000
COPY_CHUNK = 64 << 20


def tensor_nbytes(dtype: str, shape: Iterable[int]) -> int:
    if dtype not in DTYPE_SIZES:
        raise ValueError(f"unsupported safetensors dtype {dtype!r} (extend st_io.DTYPE_SIZES)")
    return math.prod(int(s) for s in shape) * DTYPE_SIZES[dtype]


def read_header(path: str) -> tuple[dict, dict, int, int]:
    """Return (entries, metadata, data_start, file_size). entries excludes __metadata__."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path}: truncated safetensors file")
        (n,) = struct.unpack("<Q", prefix)
        if not 0 < n <= min(MAX_HEADER, size - 8):
            raise ValueError(f"{path}: invalid header length {n}")
        header = json.loads(f.read(n))
    metadata = header.pop("__metadata__", {}) or {}
    data_start = 8 + n
    for name, e in header.items():
        b, en = e["data_offsets"]
        if not (0 <= b <= en <= size - data_start):
            raise ValueError(f"{path}:{name}: data_offsets {e['data_offsets']} outside file")
        want = tensor_nbytes(e["dtype"], e["shape"])
        if en - b != want:
            raise ValueError(f"{path}:{name}: byte range {en - b} != {want} for {e['dtype']}{e['shape']}")
    return header, metadata, data_start, size


class ShardReader:
    """Random access to the raw bytes of tensors in one safetensors file."""

    def __init__(self, path: str):
        self.path = path
        self.entries, self.metadata, self.data_start, self.size = read_header(path)
        self._f = open(path, "rb")

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __contains__(self, name: str) -> bool:
        return name in self.entries

    def keys(self):
        return self.entries.keys()

    def info(self, name: str) -> tuple[str, list[int], int]:
        e = self.entries[name]
        b, en = e["data_offsets"]
        return e["dtype"], list(e["shape"]), en - b

    def iter_bytes(self, name: str, chunk: int = COPY_CHUNK) -> Iterator[bytes]:
        e = self.entries[name]
        b, en = e["data_offsets"]
        self._f.seek(self.data_start + b)
        left = en - b
        while left > 0:
            buf = self._f.read(min(chunk, left))
            if not buf:
                raise EOFError(f"{self.path}:{name}: short read")
            left -= len(buf)
            yield buf

    def iter_rows(self, name: str, rows: Iterable[int], chunk: int = COPY_CHUNK) -> Iterator[bytes]:
        """Raw bytes of the dim-0 rows `rows` (in that order) of a row-major tensor, e.g. the kept experts'
        rows of a gate weight / bias. Consecutive rows are coalesced into one read."""
        e = self.entries[name]
        b, en = e["data_offsets"]
        n0 = int(e["shape"][0]) if e["shape"] else 0
        if n0 == 0 or (en - b) % n0:
            raise ValueError(f"{self.path}:{name}: cannot slice rows of shape {e['shape']}")
        row = (en - b) // n0
        rows = [int(r) for r in rows]
        if any(not 0 <= r < n0 for r in rows):
            raise IndexError(f"{self.path}:{name}: row ids outside 0..{n0 - 1}")
        i = 0
        while i < len(rows):
            j = i + 1
            while j < len(rows) and rows[j] == rows[j - 1] + 1:
                j += 1
            self._f.seek(self.data_start + b + rows[i] * row)
            left = (j - i) * row
            while left > 0:
                buf = self._f.read(min(chunk, left))
                if not buf:
                    raise EOFError(f"{self.path}:{name}: short read")
                left -= len(buf)
                yield buf
            i = j

    def read_bytes(self, name: str) -> bytearray:
        _, _, n = self.info(name)
        out = bytearray(n)
        view = memoryview(out)
        pos = 0
        for buf in self.iter_bytes(name):
            view[pos : pos + len(buf)] = buf
            pos += len(buf)
        return out

    def read_torch(self, name: str):
        """CPU torch tensor with the stored dtype (via a uint8 buffer + view, so exotic dtypes work)."""
        import torch

        dtype, shape, n = self.info(name)
        raw = self.read_bytes(name)
        if n == 0:
            return torch.empty(shape, dtype=torch_dtype(dtype))
        t = torch.frombuffer(raw, dtype=torch.uint8)
        return t.view(torch_dtype(dtype)).reshape(shape)


def torch_dtype(st_dtype: str):
    import torch

    table = {
        "F64": torch.float64,
        "F32": torch.float32,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "I64": torch.int64,
        "I32": torch.int32,
        "I16": torch.int16,
        "I8": torch.int8,
        "U8": torch.uint8,
        "BOOL": torch.bool,
        "F8_E4M3": torch.float8_e4m3fn,
        "F8_E5M2": torch.float8_e5m2,
        "F8_E8M0": torch.float8_e8m0fnu,
    }
    return table[st_dtype]


def st_dtype_of(t) -> str:
    import torch

    table = {
        torch.float64: "F64",
        torch.float32: "F32",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.int64: "I64",
        torch.int32: "I32",
        torch.int16: "I16",
        torch.int8: "I8",
        torch.uint8: "U8",
        torch.bool: "BOOL",
        torch.float8_e4m3fn: "F8_E4M3",
        torch.float8_e5m2: "F8_E5M2",
        torch.float8_e8m0fnu: "F8_E8M0",
    }
    return table[t.dtype]


def torch_to_bytes(t) -> bytes:
    import torch

    t = t.detach().contiguous().cpu()
    return t.view(torch.uint8).numpy().tobytes() if t.numel() else b""


class StreamWriter:
    """Write a safetensors file tensor by tensor without holding the payload in memory.

    The full layout (names, dtypes, shapes) must be declared up front; payloads are then streamed
    in exactly that order. The file is written to `<path>.tmp` and atomically renamed on close().
    """

    def __init__(self, path: str, entries: list[tuple[str, str, list[int]]], metadata: dict | None = None):
        self.path = path
        self.tmp = path + ".tmp"
        header: dict = {}
        if metadata:
            header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
        offset = 0
        self.order: list[tuple[str, int]] = []
        seen = set()
        for name, dtype, shape in entries:
            if name in seen:
                raise ValueError(f"duplicate tensor {name}")
            seen.add(name)
            n = tensor_nbytes(dtype, shape)
            header[name] = {"dtype": dtype, "shape": [int(s) for s in shape], "data_offsets": [offset, offset + n]}
            self.order.append((name, n))
            offset += n
        self.total = offset
        blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
        blob += b" " * (-len(blob) % 8)
        self._f = open(self.tmp, "wb")
        self._f.write(struct.pack("<Q", len(blob)))
        self._f.write(blob)
        self._idx = 0
        self._closed = False

    def write_tensor(self, name: str, chunks: Iterable[bytes | bytearray | memoryview]) -> None:
        if self._idx >= len(self.order) or self.order[self._idx][0] != name:
            want = self.order[self._idx][0] if self._idx < len(self.order) else None
            raise RuntimeError(f"out-of-order write: got {name}, expected {want}")
        expected = self.order[self._idx][1]
        written = 0
        for c in chunks:
            self._f.write(c)
            written += len(c)
        if written != expected:
            raise RuntimeError(f"{name}: wrote {written} bytes, expected {expected}")
        self._idx += 1

    def close(self) -> None:
        if self._closed:
            return
        if self._idx != len(self.order):
            self._f.close()
            raise RuntimeError(f"{self.path}: only {self._idx}/{len(self.order)} tensors written")
        self._f.flush()
        os.fsync(self._f.fileno())
        self._f.close()
        os.replace(self.tmp, self.path)
        self._closed = True

    def abort(self) -> None:
        try:
            self._f.close()
        finally:
            if os.path.exists(self.tmp):
                os.remove(self.tmp)


def const_bytes(nbytes: int, pattern: bytes, chunk: int = COPY_CHUNK) -> Iterator[bytes]:
    """Stream `nbytes` of a repeated element pattern (e.g. b'\\x80\\x3f' = bf16 1.0)."""
    assert nbytes % len(pattern) == 0
    per = max(len(pattern), (chunk // len(pattern)) * len(pattern))
    block = pattern * (per // len(pattern))
    left = nbytes
    while left > 0:
        n = min(left, len(block))
        yield block[:n]
        left -= n
