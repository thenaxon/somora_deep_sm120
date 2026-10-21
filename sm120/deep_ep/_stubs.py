"""DeepEP buffers that need NVLink/RDMA peers. The single-GPU path runs CP=1 and engram disabled, so they only have
to be importable; constructing one raises a clear error instead of crashing in a missing C++ extension."""

from __future__ import annotations


def _unsupported(what: str):
    raise NotImplementedError(
        f"SM120 deep_ep.{what} is not implemented (needs NVLink/RDMA symmetric memory). The single-GPU path requires "
        "context_parallel_size=1 (no BucketBuffer) and engram disabled (no EngramBuffer); see patches/PATCHES.md."
    )


class BufferBase:
    def __init__(self, explicitly_destroy: bool = False) -> None:
        self.explicitly_destroy = explicitly_destroy

    def destroy(self) -> None:
        pass


class BufferAllocator:
    def __init__(self, *args, **kwargs) -> None:
        self.num_bytes = 0
        self.materialized = False

    def materialize(self, *args, **kwargs) -> None:
        _unsupported("BufferAllocator.materialize")


class BucketSession:
    def __init__(self, *args, **kwargs) -> None:
        _unsupported("BucketSession")


class BucketBuffer(BufferBase):
    def __init__(self, *args, **kwargs) -> None:
        _unsupported("BucketBuffer (context-parallel all-gather)")

    @staticmethod
    def get_buffer_size_hint(*args, **kwargs) -> int:
        _unsupported("BucketBuffer.get_buffer_size_hint")


class EngramBuffer(BufferBase):
    def __init__(self, *args, **kwargs) -> None:
        _unsupported("EngramBuffer (RDMA engram table storage)")

    @staticmethod
    def get_storage_size_hint(*args, **kwargs):
        _unsupported("EngramBuffer.get_storage_size_hint")

    @staticmethod
    def get_theoretical_config(*args, **kwargs):
        _unsupported("EngramBuffer.get_theoretical_config")


class PPBuffer(BufferBase):
    def __init__(self, *args, **kwargs) -> None:
        _unsupported("PPBuffer (pipeline parallel)")
