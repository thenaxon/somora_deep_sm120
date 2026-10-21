"""Triton mHC prenorm GEMM and fp8_einsum (sm120_mhc / sm120_einsum) vs the deep_gemm shim's torch path.

    PYTHONPATH=/opt/sm120/sm120:/opt/deepinfer pytest -q /opt/sm120/tests/test_mhc_einsum.py
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "sm120"))

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="CUDA required")
DEV = torch.device("cuda", 0) if CUDA else torch.device("cpu")


class EnvSet:
    def __init__(self, **kv):
        self.kv, self.old = kv, {}

    def __enter__(self):
        for k, v in self.kv.items():
            self.old[k] = os.environ.get(k)
            os.environ[k] = str(v)

    def __exit__(self, *exc):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------------------------------------
# tf32_hc_prenorm_gemm
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("m", [1, 7, 33, 300])
@pytest.mark.parametrize("splits", [None, 16])
def test_hc_prenorm_triton_vs_fp64(m, splits):
    import deep_gemm

    gen = torch.Generator(device=DEV).manual_seed(m)
    k, n = 4 * 5120, 24
    a = torch.randn((m, k), device=DEV, generator=gen).to(torch.bfloat16)
    b = torch.randn((n, k), device=DEV, generator=gen) * 0.02
    S = 1 if splits is None else splits
    out = {}
    for impl in ("torch", "triton"):
        d = torch.full((S, m, n), float("nan"), device=DEV) if splits else torch.full((m, n), float("nan"), device=DEV)
        sq = torch.full((S, m), float("nan"), device=DEV) if splits else torch.full((m,), float("nan"), device=DEV)
        with EnvSet(SM120_MHC_IMPL=impl):
            deep_gemm.tf32_hc_prenorm_gemm(a, b, d, sq, splits)
        out[impl] = (d.view(S, m, n), sq.view(S, m))
    ref_d = (a.double() @ b.double().t())
    ref_sq = a.double().square().sum(-1)
    for impl, (d, sq) in out.items():
        assert torch.isfinite(d).all() and torch.isfinite(sq).all(), impl
        assert torch.allclose(d.sum(0).double(), ref_d, rtol=1e-5, atol=1e-4), impl
        assert torch.allclose(sq.sum(0).double(), ref_sq, rtol=1e-5), impl
    # per-split partials identical up to fp32 summation order
    assert torch.allclose(out["triton"][0], out["torch"][0], rtol=1e-4, atol=1e-4)
    assert torch.allclose(out["triton"][1], out["torch"][1], rtol=1e-5)


# ---------------------------------------------------------------------------------------------
# fp8_einsum("bhr,hdr->bhd", recipe=(1,1,32))
# ---------------------------------------------------------------------------------------------
def _fp8_operand(rows_shape, k, gen, colmajor_rows: bool):
    """Random e4m3 values with per-32 ue8m0 scales, packed int32 [..., k/128] (MN-major if requested)."""
    x = (torch.randn((*rows_shape, k), device=DEV, generator=gen) * 2).to(torch.float8_e4m3fn)
    e = torch.randint(118, 130, (*rows_shape, k // 32), device=DEV, generator=gen).to(torch.uint8)
    packed = e.contiguous().view(torch.int32)  # [..., k/128]
    if colmajor_rows:
        shp = packed.shape
        if len(shp) == 3:  # [A, B, K4] -> strides (B*K4, 1, B): MN (= B) major, like transform_sf / TMA layout
            t = torch.empty_strided(shp, (shp[1] * shp[2], 1, shp[1]), dtype=torch.int32, device=DEV)
        else:
            t = torch.empty_strided(shp, (1, shp[0]), dtype=torch.int32, device=DEV)
        t.copy_(packed)
        packed = t
    return x, packed


def _a_colmajor(x_sf):
    """A scale layout as the fused attention cast produces it: [T, G, K4] with stride(0) == 1."""
    T, G, K4 = x_sf.shape
    aT = (T + 3) // 4 * 4
    base = torch.empty((G, K4, aT), dtype=torch.int32, device=DEV)
    v = base.transpose(1, 2)[:, :T].transpose(0, 1)
    v.copy_(x_sf)
    return v


@pytest.mark.parametrize("T", [1, 8, 32, 300])
@pytest.mark.parametrize("out_fp8", [False, True])
def test_fp8_einsum_triton_vs_torch(T, out_fp8):
    import deep_gemm

    gen = torch.Generator(device=DEV).manual_seed(100 + T)
    G, N, K = 8, 1024, 4096
    a, sfa = _fp8_operand((T, G), K, gen, colmajor_rows=False)
    sfa = _a_colmajor(sfa)
    w, sfw = _fp8_operand((G, N), K, gen, colmajor_rows=True)
    res = {}
    for impl in ("torch", "triton"):
        # SM120_DEQUANT_CACHE=none: the torch path caches dequantised weights by data_ptr, and a fresh test
        # tensor can reuse a previous case's address (static model weights never do)
        with EnvSet(SM120_EINSUM_IMPL=impl, SM120_ATTN_STRICT=1 if impl == "triton" else 0, SM120_DEQUANT_CACHE="none"):
            if out_fp8:
                d8 = torch.empty((T, G, N), dtype=torch.float8_e4m3fn, device=DEV)
                aT = (T + 3) // 4 * 4
                dsf = torch.empty_strided((T, G * N // 128), (1, aT), dtype=torch.int32, device=DEV)
                deep_gemm.fp8_einsum("bhr,hdr->bhd", (a, sfa), (w, sfw), (d8, dsf), recipe=(1, 1, 32))
                sc = dsf.contiguous().view(torch.uint8).view(T, -1)
                val = d8.float().view(T, -1, 32) * (sc.int() << 23).view(torch.float32).view(T, -1, 1)
                res[impl] = (val.view(T, G, N), sc)
            else:
                d = torch.empty((T, G, N), dtype=torch.bfloat16, device=DEV)
                deep_gemm.fp8_einsum("bhr,hdr->bhd", (a, sfa), (w, sfw), d, recipe=(1, 1, 32))
                res[impl] = (d.float(), None)
    # independent reference: logical int32 scale tensors -> bytes -> per-32 scales, fp32 einsum, bf16 round
    def deq(v, sf):
        sc = (sf.contiguous().view(torch.uint8).int() << 23).view(torch.float32)  # [..., K/32]
        return v.float().view(*v.shape[:-1], -1, 32).mul(sc.unsqueeze(-1)).view(v.shape)

    ref = torch.einsum("bhr,hdr->bhd", deq(a, sfa), deq(w, sfw)).to(torch.bfloat16).float()
    for impl in ("triton", "torch"):
        c = torch.nn.functional.cosine_similarity(res[impl][0].flatten(), ref.flatten(), dim=0).item()
        print(f"T={T} out_fp8={out_fp8} {impl} vs independent reference: cos {c:.6f}")
        # bf16 output: rounding only; fp8 output: the reference is not FP8-quantised (one e4m3 step ~ 6 %)
        assert c > (0.999 if out_fp8 else 0.9999), f"{impl} vs independent reference cos {c}"
    x, y = res["triton"][0], res["torch"][0]
    scale = y.abs().amax() + 1e-6
    err = (x - y).abs().max().item() / scale.item()
    cos = torch.nn.functional.cosine_similarity(x.flatten(), y.flatten(), dim=0).item()
    # bf16 rounding of differently ordered fp32 sums (+ one FP8 step on ties for the fp8 output)
    assert cos > 0.99999 and err < (0.07 if out_fp8 else 0.01), (cos, err)
    if out_fp8:
        assert (res["triton"][1] == res["torch"][1]).float().mean().item() > 0.99


def test_fp8_einsum_cuda_graph():
    import deep_gemm

    gen = torch.Generator(device=DEV).manual_seed(7)
    T, G, N, K = 8, 8, 1024, 4096
    a, sfa = _fp8_operand((T, G), K, gen, colmajor_rows=False)
    sfa = _a_colmajor(sfa)
    w, sfw = _fp8_operand((G, N), K, gen, colmajor_rows=True)
    d = torch.empty((T, G, N), dtype=torch.bfloat16, device=DEV)
    x = torch.randn((T, 4 * 5120), device=DEV).to(torch.bfloat16)
    fn = torch.randn((24, 4 * 5120), device=DEV) * 0.02
    dd = torch.empty((16, T, 24), device=DEV)
    sq = torch.empty((16, T), device=DEV)

    def step():
        deep_gemm.fp8_einsum("bhr,hdr->bhd", (a, sfa), (w, sfw), d, recipe=(1, 1, 32))
        deep_gemm.tf32_hc_prenorm_gemm(x, fn, dd, sq, 16)

    with EnvSet(SM120_EINSUM_IMPL="triton", SM120_MHC_IMPL="triton", SM120_ATTN_STRICT=1):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            step()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            step()
        a.copy_(a.flip(0))
        x.copy_(x.flip(0))
        g.replay()
        torch.cuda.synchronize()
        d_g, dd_g, sq_g = d.clone(), dd.clone(), sq.clone()
        step()
        assert torch.equal(d_g, d) and torch.equal(dd_g, dd) and torch.equal(sq_g, sq)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
