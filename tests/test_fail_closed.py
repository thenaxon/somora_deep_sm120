"""Fail-closed paths of the MoE / GEMM / attention shims and the DP sync, on one GPU.

    PYTHONPATH=/opt/sm120/sm120:/opt/deepinfer pytest -q /opt/sm120/tests/test_fail_closed.py

- sm120_moe.setup: a b12x failure before any weight mutation may fall back to the torch reference; a failure after
  the in-place repack started, or any MoE failure with expert parallelism (> 1 rank), raises. SM120_MOE_IMPL=torch
  with expert parallelism raises (the torch reference is not expert-parallel). A missing sm120_ep with EP > 1 raises.
- deep_gemm b12x linear: an error inside CUDA graph capture is raised and does not mark the weight unsupported;
  eagerly it still falls back and marks it.
- attention indexer: a b12x error inside capture is raised instead of recording the torch path.
- dp_sync: without captured graphs (DeepInfer then decodes eagerly) no graph size is synced.
- b12x indexer: the active-width scalar is one buffer per device, not one cached tensor per value.
"""

from __future__ import annotations

import os
import sys
import types

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "sm120"))
sys.path.insert(0, HERE)
os.environ.setdefault("B12X_AUTOTUNE", "0")

CUDA = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12
pytestmark = pytest.mark.skipif(not CUDA, reason="needs an SM12x GPU")
DEV = torch.device("cuda", 0)


class _Group:
    def __init__(self, n):
        self.n = n

    def size(self):
        return self.n


def _moe(ep: int = 1):
    import test_ep_moe as T
    m = T.make_moe(T.make_weights(3), range(0, T.E), DEV)
    m.use_mega_moe = False
    m.use_fp8_act = False
    m.ep_group = _Group(ep)
    return m


def _setup(monkeypatch, impl="b12x"):
    import sm120_moe
    monkeypatch.setenv("SM120_MOE_IMPL", impl)
    monkeypatch.delenv("SM120_MOE_STRICT", raising=False)
    return sm120_moe


def test_moe_failure_before_mutation_falls_back(monkeypatch):
    sm120_moe = _setup(monkeypatch)
    from b12x.moe import fused_moe as fm
    monkeypatch.setattr(fm, "plan_weights", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no plan")))
    m = _moe()
    sm120_moe.setup(m)
    assert m.sm120_moe_impl == "torch" and isinstance(m.sm120_routed, sm120_moe.TorchRoutedExperts)


def test_moe_failure_after_partial_repack_raises(monkeypatch):
    sm120_moe = _setup(monkeypatch)
    from b12x.moe import fused_moe as fm

    def half_repack(*a, **k):
        w = k.get("w1_fp4", k.get("weights", None))
        if isinstance(w, torch.Tensor):
            w.view(-1)[:16].zero_()  # mutated in place, then fails
        raise RuntimeError("out of memory half-way")

    monkeypatch.setattr(fm, "prepare_weights", half_repack)
    m = _moe()
    with pytest.raises(RuntimeError, match="half-way"):
        sm120_moe.setup(m)
    assert m.gate_up_proj_weight.sm120_b12x_prepare_started


def test_moe_failure_with_ep_raises(monkeypatch):
    sm120_moe = _setup(monkeypatch)
    from b12x.moe import fused_moe as fm
    monkeypatch.setattr(fm, "plan_weights", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no plan")))
    with pytest.raises(RuntimeError, match="no plan"):
        sm120_moe.setup(_moe(ep=2))


def test_moe_torch_impl_with_ep_raises(monkeypatch):
    sm120_moe = _setup(monkeypatch, impl="torch")
    with pytest.raises(RuntimeError, match="expert parallelism"):
        sm120_moe.setup(_moe(ep=4))
    m = _moe(ep=1)
    sm120_moe.setup(m)  # single GPU: allowed
    assert isinstance(m.sm120_routed, sm120_moe.TorchRoutedExperts)


def test_wrap_ep_without_sm120_ep_raises(monkeypatch):
    import importlib.util
    import sm120_moe
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "sm120_ep" else real(name, *a))
    m = types.SimpleNamespace(layer_idx=3, ep_group=_Group(2))
    with pytest.raises(RuntimeError, match="sm120_ep"):
        sm120_moe._wrap_ep(m)
    sm120_moe._wrap_ep(types.SimpleNamespace(layer_idx=3, ep_group=_Group(1)))  # EP=1: no-op


def _linear_args():
    a = torch.zeros(4, 256, device=DEV)
    b = torch.zeros(128, 256, device=DEV)
    return a, None, None, b, None, None, None, None, None


def test_try_linear_capture_error_is_raised_and_not_cached(monkeypatch):
    from deep_gemm import _b12x
    monkeypatch.setattr(_b12x, "eligible", lambda *a: None)
    monkeypatch.setattr(_b12x, "linear", lambda *a: _b12x._not_in_capture("test plan"))
    monkeypatch.setattr(_b12x, "_capturing", lambda: True)
    args = _linear_args()
    key = (tuple(args[3].shape), args[3].data_ptr())
    _b12x._UNSUPPORTED.discard(key)
    with pytest.raises(RuntimeError, match="first requested during CUDA graph capture"):
        _b12x.try_linear(*args)
    assert key not in _b12x._UNSUPPORTED
    # eager: the old fallback (torch for this weight) still applies
    monkeypatch.setattr(_b12x, "_capturing", lambda: False)
    monkeypatch.setattr(_b12x, "linear", lambda *a: (_ for _ in ()).throw(ValueError("bad layout")))
    assert _b12x.try_linear(*args) is None and key in _b12x._UNSUPPORTED
    _b12x._UNSUPPORTED.discard(key)


def test_indexer_b12x_error_in_capture_is_raised(monkeypatch):
    from sm120_attn import _config as cfg
    from sm120_attn import b12x_indexer, indexer
    monkeypatch.setattr(cfg, "indexer_impl", lambda: "b12x")
    monkeypatch.setattr(b12x_indexer, "fake_entry", lambda *a: (_ for _ in ()).throw(RuntimeError("no plan")),
                        raising=False)
    monkeypatch.setattr(cfg, "capturing", lambda: True)
    with pytest.raises(RuntimeError, match="no plan"):
        indexer._b12x("fake", "fake_entry")
    monkeypatch.setattr(cfg, "capturing", lambda: False)
    monkeypatch.setattr(cfg, "strict", lambda: False)
    assert indexer._b12x("fake", "fake_entry") is None  # eager: torch fallback as before


def test_dp_sync_without_graphs_does_not_sync(monkeypatch):
    from sm120_ep import dp_sync
    calls = []
    monkeypatch.setattr(dp_sync, "dp_max", lambda v, g: calls.append(v) or v)

    class Worker:
        def __init__(self, graphs):
            self.model_runner = types.SimpleNamespace(use_cuda_graph=True,
                                                      cuda_graph_runner=types.SimpleNamespace(graphs=graphs))

        def execute_model_step(self, is_prompt, is_placeholder, seq_metadata_list=None, scheduler_outputs=None):
            return dp_sync._STATE["synced"]

    dp_sync.patch_worker(Worker, None)
    out = types.SimpleNamespace(num_batched_tokens=5)
    assert Worker({}).execute_model_step(False, False, None, out) is None and calls == []
    assert Worker(None).execute_model_step(False, False, None, out) is None and calls == []
    assert Worker({8: object()}).execute_model_step(False, False, None, out) == 5 and calls == [5]


def test_indexer_active_is_one_buffer():
    from sm120_attn import b12x_indexer
    b12x_indexer._ACTIVE.clear()
    for v in range(1000, 1100):
        t = b12x_indexer._active(v, DEV)
    assert len(b12x_indexer._ACTIVE) == 1 and int(t.item()) == 1099
    assert int(b12x_indexer._active(2 ** 40, DEV).item()) == 2 ** 31 - 1
