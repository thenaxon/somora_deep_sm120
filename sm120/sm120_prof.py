"""Named CPU ranges for SM120_PROFILE_PREFILL (server/engine.py): `with sm120_prof.rng("sm120:..."):` is a no-op
unless a prefill profile is being recorded, then a torch.profiler.record_function range (CPU timeline + table)."""

from __future__ import annotations

import contextlib

ACTIVE = False


def rng(name: str):
    if not ACTIVE:
        return contextlib.nullcontext()
    import torch
    return torch.profiler.record_function(name)
