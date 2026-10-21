"""deep_ep EventHandle / EventOverlap (vendor deep_ep/utils/event.py semantics) on top of torch.cuda.Event."""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import torch


class EventHandle:
    """A CUDA event recorded on the current stream at construction (deep_ep._C.EventHandle)."""

    def __init__(self) -> None:
        self.event = torch.cuda.Event()
        self.event.record(torch.cuda.current_stream())

    def current_stream_wait(self) -> None:
        torch.cuda.current_stream().wait_event(self.event)


class EventOverlap:
    def __init__(self, event: Optional[EventHandle] = None, extra_tensors: Optional[Tuple[torch.Tensor]] = None) -> None:
        self.event = event
        self.extra_tensors = extra_tensors
        self._release_handle_by_call = False
        self.hook_after_wait: Optional[Callable] = None

    def current_stream_wait(self, release_handle: bool = False) -> Any:
        assert self.event is not None
        self.event.current_stream_wait()
        result = None
        if self.hook_after_wait is not None:
            result = self.hook_after_wait()
            self.hook_after_wait = None
        if release_handle:
            self.event = None
        return result

    def wait(self) -> Any:
        return self.current_stream_wait()

    def register_hook_after_wait(self, hook_after_wait: Callable) -> None:
        assert self.hook_after_wait is None, "A hook is already registered on this `EventOverlap`"
        self.hook_after_wait = hook_after_wait

    def __call__(self, release_handle: bool = False) -> "EventOverlap":
        self._release_handle_by_call = release_handle
        return self

    def __enter__(self) -> Any:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.event is not None:
            self.current_stream_wait(release_handle=self._release_handle_by_call)
        self._release_handle_by_call = False
