"""CUDA-graph tests for the GEMM / MoE shims (deep_gemm b12x block_fp8_linear, sm120_moe b12x fused_moe).

Run inside the SM120 container with the shims first on PYTHONPATH:
    PYTHONPATH=/opt/sm120/sm120:/opt/deepinfer pytest -q /opt/sm120/tests/test_gemm_moe_graphs.py

Each case follows DeepInfer's capture contract (worker/cuda_graph_runner.py): one eager warm-up call of the same
shape on the capture stream, then capture, then replay with new data written into the static inputs. Checks:
replay == eager on the same data, eager vs an fp32 torch reference, and that the shims refuse to create plans
during capture. Random weights, no checkpoint. MoE: E=16, K=2048, n=512, top-4, swiglu_limit 10 (both
B12X_W4A8_TINY_DECODE settings; the tiny-decode case only matches the clamped reference on a patched b12x).
"""

from __future__ import annotations

import os
import sys
import types

import pytest
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
SHIMS = os.path.join(os.path.dirname(HERE), "sm120")
if SHIMS not in sys.path:
    sys.path.insert(0, SHIMS)
os.environ.setdefault("B12X_AUTOTUNE", "0")

CUDA = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12
pytestmark = pytest.mark.skipif(not CUDA, reason="needs an SM12x GPU")
DEV = torch.device("cuda", 0)


def _cos(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def _capture(fn, side: torch.cuda.Stream):
    """Warm up once on the capture stream (as DeepInfer does), then capture fn into a graph."""
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=side):
        fn()
    torch.cuda.synchronize()
    return g


# ------------------------------------------------------------------------------------------------ linear
def _block_quant_weight(n: int, k: int):
    from deep_gemm import _sf, transform_sf_into_required_layout

    w = torch.randn(n, k, device=DEV) * 0.05
    blk = w.view(n // 32, 32, k // 32, 32).abs().amax(dim=(1, 3)).clamp(min=1e-4)
    scale = _sf.ceil_to_ue8m0(blk / 448.0)  # [n/32, k/32] power-of-two
    full = scale.repeat_interleave(32, 0).repeat_interleave(32, 1)
    wq = (w / full).to(torch.float8_e4m3fn)
    sfb = transform_sf_into_required_layout(scale, n, k, recipe=(1, 32, 32), is_sfa=False)
    return wq, sfb, (wq.float() * full)


def _act(x: torch.Tensor):
    from deep_gemm import _sf, transform_sf_into_required_layout

    q, s = _sf.quant_fp8_rows(x, 32)
    return q, transform_sf_into_required_layout(s, x.shape[0], x.shape[1], recipe=(1, 32))


@pytest.mark.parametrize("T", [1, 7, 16, 100])
def test_linear_graph(T):
    import deep_gemm
    from deep_gemm import _b12x, _sf

    os.environ["SM120_GEMM_IMPL"] = "b12x"
    os.environ["SM120_GEMM_STRICT"] = "1"
    N, K = 1536, 2048
    wq, sfb, w_deq = _block_quant_weight(N, K)
    xq, sfa = _act(torch.randn(T, K, device=DEV).to(torch.bfloat16))
    out = torch.empty(T, N, dtype=torch.bfloat16, device=DEV)

    def fn():
        deep_gemm.fp8_fp4_gemm_nt((xq, sfa), (wq, sfb), out, recipe=(1, 1, 32))

    g = _capture(fn, torch.cuda.Stream())
    assert _b12x.stats()["plans"] > 0, "b12x block_fp8_linear was not used"
    xq2, sfa2 = _act(torch.randn(T, K, device=DEV).to(torch.bfloat16))
    xq.copy_(xq2)
    sfa.copy_(sfa2)
    g.replay()
    torch.cuda.synchronize()
    replay = out.clone()
    fn()
    torch.cuda.synchronize()
    assert _cos(replay, out) > 0.99999, "graph replay differs from eager"
    ref = _sf.dequant_fp8(xq, sfa, 32, torch.float32) @ w_deq.t()
    assert _cos(out, ref) > 0.999


def test_linear_plan_in_capture_raises():
    import deep_gemm

    os.environ["SM120_GEMM_IMPL"] = "b12x"
    os.environ["SM120_GEMM_STRICT"] = "1"
    N, K = 768, 1024  # a (weight, shape) never seen eagerly
    wq, sfb, _ = _block_quant_weight(N, K)
    xq, sfa = _act(torch.randn(3, K, device=DEV).to(torch.bfloat16))
    out = torch.empty(3, N, dtype=torch.bfloat16, device=DEV)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with pytest.raises(RuntimeError, match="during CUDA graph capture"):
        with torch.cuda.graph(g, stream=s):
            deep_gemm.fp8_fp4_gemm_nt((xq, sfa), (wq, sfb), out, recipe=(1, 1, 32))
    torch.cuda.synchronize()


# ------------------------------------------------------------------------------------------------ MoE
E, K_MOE, N_MOE, TOPK, LIMIT = 16, 2048, 512, 4, 10.0


def _fake_moe():
    """Just the DeepseekMoE attributes sm120_moe.B12xRoutedExperts reads, in DeepInfer's HF-loader layout."""
    w13 = torch.randint(-128, 128, (E, 2 * N_MOE, K_MOE // 2), dtype=torch.int8, device=DEV)
    w2 = torch.randint(-128, 128, (E, K_MOE, N_MOE // 2), dtype=torch.int8, device=DEV)
    s13 = torch.randint(118, 122, (E, 2 * N_MOE, K_MOE // 32), dtype=torch.uint8, device=DEV)
    s2 = torch.randint(118, 122, (E, K_MOE, N_MOE // 32), dtype=torch.uint8, device=DEV)
    w13.sf = s13.view(torch.int32)  # [E, 2n, K/128] int32 = 4 UE8M0 bytes each, like the loader
    w2.sf = s2.view(torch.int32)
    ref = (w13.clone(), s13.clone(), w2.clone(), s2.clone())
    moe = types.SimpleNamespace(gate_up_proj_weight=w13, down_proj_weight=w2, use_fp4_weight=True,
                                num_physical_topk=TOPK, swiglu_clamp_value=LIMIT, layer_idx=99)
    return moe, ref


def _moe_ref(x, ids, w, ref):
    from deep_gemm import _sf

    w13, s13, w2, s2 = ref
    out = torch.zeros(x.shape[0], K_MOE, device=DEV)
    for t in range(x.shape[0]):
        for j in range(TOPK):
            e = int(ids[t, j])
            if e < 0:
                continue
            h = x[t].float() @ _sf.dequant_fp4(w13[e], s13[e], 32, torch.float32).t()
            g, u = h[:N_MOE].clamp(max=LIMIT), h[N_MOE:].clamp(-LIMIT, LIMIT)
            out[t] += w[t, j] * ((F.silu(g) * u) @ _sf.dequant_fp4(w2[e], s2[e], 32, torch.float32).t())
    return out


def _routes(T):
    ids = torch.stack([torch.randperm(E, device=DEV)[:TOPK] for _ in range(T)]).long()
    if T > 1:
        ids[-1] = -1  # a CUDA-graph padding row (valid_token_mask=False -> gate gives -1)
    w = torch.softmax(torch.randn(T, TOPK, device=DEV), -1)
    return ids, w


@pytest.mark.parametrize("tiny", ["0", "1"])
@pytest.mark.parametrize("T", [1, 4, 8, 64])
def test_moe_graph(T, tiny):
    import sm120_moe
    from b12x.moe._shared.kernels import tiny_decode

    os.environ["B12X_W4A8_TINY_DECODE"] = tiny
    moe, ref = _fake_moe()
    experts = sm120_moe.B12xRoutedExperts(moe)
    x = (torch.randn(T, K_MOE, device=DEV) * 4).to(torch.bfloat16)
    ids, w = _routes(T)
    holder = {}

    def fn():
        holder["out"] = experts(x, ids, w)

    g = _capture(fn, torch.cuda.Stream())
    x.copy_((torch.randn(T, K_MOE, device=DEV) * 4).to(torch.bfloat16))
    ids2, w2 = _routes(T)
    ids.copy_(ids2)
    w.copy_(w2)
    g.replay()
    torch.cuda.synchronize()
    replay = holder["out"].clone()
    eager = experts(x, ids, w)
    torch.cuda.synchronize()
    assert _cos(replay, eager) > 0.9999, "graph replay differs from eager"
    if T > 1:
        assert torch.count_nonzero(eager[-1]) == 0, "masked (-1) row must stay zero"
    ref_out = _moe_ref(x, ids, w, ref)
    c = _cos(eager, ref_out)
    if tiny == "1" and T <= 4 and not getattr(tiny_decode, "SWIGLU_LIMIT_SUPPORTED", False):
        pytest.xfail(f"unpatched b12x tiny-decode ignores swiglu_limit (cos {c:.4f} vs clamped ref)")
    assert c > 0.99, f"cos vs clamped fp32 reference {c:.4f}"


def test_moe_plan_in_capture_raises():
    import sm120_moe

    os.environ["B12X_W4A8_TINY_DECODE"] = "0"
    moe, _ = _fake_moe()
    experts = sm120_moe.B12xRoutedExperts(moe)
    x = torch.randn(32, K_MOE, device=DEV).to(torch.bfloat16)
    ids, w = _routes(32)
    g = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="during CUDA graph capture"):
        with torch.cuda.graph(g, stream=torch.cuda.Stream()):
            experts(x, ids, w)
    torch.cuda.synchronize()


@pytest.mark.parametrize("det", ["0", "1"])
def test_moe_determinism(det):
    """SM120_MOE_DETERMINISTIC=1 must give bit-identical results over repeated calls; 0 only reports."""
    import sm120_moe

    os.environ["B12X_W4A8_TINY_DECODE"] = "0"
    os.environ["SM120_MOE_DETERMINISTIC"] = det
    try:
        moe, _ = _fake_moe()
        experts = sm120_moe.B12xRoutedExperts(moe)
        x = (torch.randn(512, K_MOE, device=DEV) * 4).to(torch.bfloat16)
        ids, w = _routes(512)
        outs = [experts(x, ids, w).clone() for _ in range(8)]
        torch.cuda.synchronize()
        n_diff = sum(not torch.equal(outs[0], o) for o in outs[1:])
        print(f"SM120_MOE_DETERMINISTIC={det}: {n_diff}/7 repeats differ, "
              f"max|d|={max((o.float() - outs[0].float()).abs().max().item() for o in outs[1:]):.3g}")
        if det == "1":
            assert n_diff == 0
    finally:
        os.environ.pop("SM120_MOE_DETERMINISTIC", None)
