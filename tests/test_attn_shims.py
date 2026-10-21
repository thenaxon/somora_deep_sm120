"""Unit tests for the SM120 attention / indexer / top-k shims (flash_mla, deep_select, deep_gemm indexer).

Run inside the SM120 container with the shim tree first on PYTHONPATH, e.g.
    PYTHONPATH=/opt/sm120/sm120:/opt/deepinfer pytest -q tests/test_attn_shims.py

* b12x vs torch: the decode tests run the public FlashMLA entry points twice, with
  SM120_ATTN_IMPL=b12x and SM120_ATTN_IMPL=torch, on the same DeepInfer-format caches.
* b12x prefill (extend mode over re-encoded KV), b12x MXFP4 indexer logits (all four
  DeepGEMM functions, plus top-k overlap) and CUDA-graph capture/replay of the decode path
  (fused decode + paged logits + deep_select.topk), selected with `-k "b12x or graph"`.
* torch vs independent naive references (per-row loops over fully dequantized caches).
* Caches are written with DeepInfer's own store_kv_cache_ref (tilelang_ops/store_kv_cache.py)
  when DeepInfer is importable (DEEPINFER_ROOT, /opt/deepinfer or the repo checkout); otherwise
  with the FlashMLA-equivalent reference encoders in sm120_attn.quant (reported via a print).
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
SHIMS = os.path.join(os.path.dirname(HERE), "sm120")
if SHIMS not in sys.path:
    sys.path.insert(0, SHIMS)
for _p in (os.environ.get("DEEPINFER_ROOT"), "/opt/deepinfer",
           os.path.abspath(os.path.join(SHIMS, "..", "..", "DeepInfer"))):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)
os.environ.setdefault("B12X_AUTOTUNE", "0")

import flash_mla  # noqa: E402
from deep_select.interface import topk as ds_topk  # noqa: E402
from sm120_attn import fused as F_  # noqa: E402
from sm120_attn import quant as Q  # noqa: E402
from sm120_attn.indexer import DEEP_GEMM_EXPORTS as DG  # noqa: E402

CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="CUDA required")
DEV = torch.device("cuda", 0) if CUDA else torch.device("cpu")

H, D, ROPE = 64, 512, 64


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
class EnvSet:
    def __init__(self, **kv):
        self.kv = kv
        self.old = {}

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


def cos_sim(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def assert_close(a, b, cos=0.9999, rel=0.02, name=""):
    a, b = a.float(), b.float()
    assert torch.isfinite(a).all(), f"{name}: non-finite"
    c = cos_sim(a, b)
    err = (a - b).abs().max().item()
    scale = b.abs().max().item() + 1e-6
    assert c > cos and err <= rel * scale + 1e-3, f"{name}: cos={c:.6f} max_abs={err:.3e} ref_absmax={scale:.3e}"


def latent(n, gen):
    x = torch.randn((n, D), generator=gen, device=DEV)
    tok = torch.exp2(torch.empty((n, 1), device=DEV).uniform_(-2, 2, generator=gen))
    grp = torch.exp2(torch.empty((n, D // 16, 1), device=DEV).uniform_(-1.5, 1.5, generator=gen))
    return (x.view(n, D // 16, 16) * grp).view(n, D).mul(tok).to(torch.bfloat16)


_DI = {}


def deepinfer_writer():
    """(store_kv_cache_ref, V41_FORMAT, V41_FP4_FORMAT, load_paged_kv_cache_ref) or None."""
    if "ok" not in _DI:
        try:
            from deepinfer.model_executor.layers.tilelang_ops import cache_common as cc
            from deepinfer.model_executor.layers.tilelang_ops.load_paged_kv_cache import load_paged_kv_cache_ref
            from deepinfer.model_executor.layers.tilelang_ops.store_kv_cache import store_kv_cache_ref
            _DI.update(ok=True, store=store_kv_cache_ref, load=load_paged_kv_cache_ref, cc=cc)
        except Exception as exc:  # noqa: BLE001
            print(f"[test] DeepInfer writer unavailable ({type(exc).__name__}: {exc}); using sm120_attn encoders")
            _DI.update(ok=False)
    return _DI if _DI["ok"] else None


def make_cache(num_blocks, page, rec_bytes, gen, block_stride_pad=96):
    """Paged cache view [nb, page, 1, rec] inside a larger buffer (non-trivial block stride/offset)."""
    stride = Q_align(page * rec_bytes + block_stride_pad, 32)
    buf = (torch.randint(0, 256, (num_blocks, stride + 64), dtype=torch.uint8, device=DEV, generator=gen) & 0x77)
    view = buf[:, 32:32 + page * rec_bytes].as_strided((num_blocks, page, 1, rec_bytes),
                                                       (buf.stride(0), rec_bytes, rec_bytes, 1))
    return buf, view


def Q_align(x, a):
    return (x + a - 1) // a * a


def store(cache, kv, slots, fp4: bool):
    di = deepinfer_writer()
    if di is not None:
        layout = di["cc"].V41_FP4_FORMAT if fp4 else di["cc"].V41_FORMAT
        di["store"](kv, cache, slots.to(torch.int32).contiguous(), layout=layout)
        return
    rec = Q.quantize_v41_fp4_records(kv) if fp4 else Q.quantize_v41_fp8_records(kv)
    s = slots.long()
    cache[s // cache.shape[1], s % cache.shape[1], 0, :] = rec


def dequant_all(cache):
    nb, page = cache.shape[:2]
    idx = torch.arange(nb * page, device=DEV)
    return Q.dequant_v41_records(Q.gather_paged_records(cache, idx))  # [nb*page, 512]


def naive_decode(q, deq_main, idx, ln, deq_extra, eidx, eln, sm_scale, sink, q_scale=None):
    """Per-row fp32 loop (independent of sm120_attn.sparse_attn)."""
    R = q.shape[0]
    out = torch.zeros((R, q.shape[1], D), device=DEV)
    lse = torch.full((R, q.shape[1]), float("inf"), device=DEV)
    for r in range(R):
        ks = []
        ii = idx[r][: (ln[r] if ln is not None else idx.shape[1])]
        ks.append(deq_main[ii[ii >= 0].long()])
        if eidx is not None:
            ee = eidx[r][: (eln[r] if eln is not None else eidx.shape[1])]
            ks.append(deq_extra[ee[ee >= 0].long()])
        K = torch.cat(ks, 0)
        if K.shape[0] == 0:
            continue
        s = q[r].float() @ K.t() * sm_scale
        if q_scale is not None:
            s = s * q_scale[r].view(-1, 1)
        l_ = torch.logsumexp(s, -1)
        p = torch.exp(s - l_.view(-1, 1))
        o = p @ K
        if sink is not None:
            o = o / (1 + torch.exp(sink - l_)).view(-1, 1)
        out[r] = o
        lse[r] = l_
    return out, lse


def build_decode_case(seed=1, rows=13, swa_page=134, window=128, main_page=64, topk=512, with_extra=True,
                      with_len=True, invalid_frac=0.15):
    """Paged V4.1 decode case: 528 B SWA cache (window 128) + 288 B compressed cache (top-k 512)."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    nb_swa, nb_main = 1 + rows, 1 + rows * ((topk * 2) // main_page + 1)
    _, swa = make_cache(nb_swa, swa_page, 528, gen)
    _, main = make_cache(nb_main, main_page, 288, gen)
    swa_slots = torch.randperm(nb_swa * swa_page, device=DEV, generator=gen)[: rows * window]
    main_slots = torch.randperm(nb_main * main_page, device=DEV, generator=gen)[: rows * topk * 2]
    store(swa, latent(swa_slots.numel(), gen), swa_slots, fp4=False)
    store(main, latent(main_slots.numel(), gen), main_slots, fp4=True)
    swa_idx = swa_slots.view(rows, window).to(torch.int32).clone()
    main_idx = main_slots.view(rows, -1)[:, :topk].to(torch.int32).clone()
    # -1 holes (FlashMLA: only -1 is invalid) and SWA windows that are a valid prefix
    swa_len = torch.randint(1, window + 1, (rows,), device=DEV, generator=gen)
    swa_idx[torch.arange(window, device=DEV).view(1, -1) >= swa_len.view(-1, 1)] = -1
    holes = torch.rand((rows, topk), device=DEV, generator=gen) < invalid_frac
    main_idx[holes] = -1
    main_len = torch.randint(topk // 4, topk + 1, (rows,), device=DEV, generator=gen).to(torch.int32)
    main_len[0] = 0  # a row with no extra tokens
    q = (torch.randn((rows, H, D), device=DEV, generator=gen) * 0.5).to(torch.bfloat16)
    sink = (torch.randn((H,), device=DEV, generator=gen) * 2).float()
    return dict(q=q, swa=swa, main=main if with_extra else None, swa_idx=swa_idx,
                main_idx=main_idx if with_extra else None, main_len=main_len if (with_extra and with_len) else None,
                sink=sink, sm_scale=D ** -0.5)


def run_flash_mla_decode(c, impl, compute="bf16"):
    with EnvSet(SM120_ATTN_IMPL=impl, SM120_ATTN_COMPUTE=compute, SM120_ATTN_STRICT=1 if impl == "b12x" else 0):
        rows = c["q"].shape[0]
        sm, _ = flash_mla.get_mla_metadata()
        out, lse = flash_mla.flash_mla_with_kvcache(
            q=c["q"].unsqueeze(1), k_cache=c["swa"], block_table=None, cache_seqlens=None, head_dim_v=D,
            tile_scheduler_metadata=sm, softmax_scale=c["sm_scale"], causal=False, is_fp8_kvcache=True,
            indices=c["swa_idx"].unsqueeze(1), attn_sink=c["sink"], extra_k_cache=c["main"],
            extra_indices_in_kvcache=None if c["main_idx"] is None else c["main_idx"].unsqueeze(1),
            topk_length=None, extra_topk_length=c["main_len"])
        assert out.shape == (rows, 1, H, D) and out.dtype == torch.bfloat16
        assert lse.shape == (rows, H, 1)
        return out.squeeze(1), lse.squeeze(-1)


def b12x_available():
    try:
        from b12x.attention import compressed_sparse_mla as mla
        return bool(mla.is_supported(DEV))
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------------------------
# deep_select.topk
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("impl", ["torch", "triton"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mode", ["plain", "sorted", "sorted_index"])
def test_topk_contract(dtype, mode, impl):
    if impl == "triton" and not CUDA:
        pytest.skip("CUDA required")
    with EnvSet(SM120_TOPK_IMPL=impl):
        _topk_contract(dtype, mode)


def _topk_contract(dtype, mode):
    gen = torch.Generator(device=DEV).manual_seed(0)
    N, V, k = 37, 3000, 64
    x = torch.randn((N, V), device=DEV, generator=gen).to(dtype)
    x[3, :10] = float("-inf")
    end = torch.randint(0, V + 1, (N,), device=DEV, generator=gen).to(torch.int32)
    end[0], end[1], end[2] = 0, 5, k
    off = torch.randint(0, 1000, (N,), device=DEV, generator=gen).to(torch.int32)
    out_idx = torch.empty((N, k), dtype=torch.int32, device=DEV)
    vals, idx = ds_topk(x, k, sorted=(mode == "sorted"), end=end, indices_type=torch.int32,
                        sorted_index=(mode == "sorted_index"), output_idx=out_idx, output_idx_offset=off,
                        idx_oob_fill_value=-1, value_oob_fill_value=float("-inf"), return_value=True)
    assert idx.data_ptr() == out_idx.data_ptr()
    for i in range(N):
        c = min(int(end[i]), k)
        sel = idx[i, :c].long() - off[i]
        assert (idx[i, c:] == -1).all()
        assert torch.isneginf(vals[i, c:].float()).all()
        assert ((sel >= 0) & (sel < end[i])).all() and sel.unique().numel() == c
        assert torch.equal(vals[i, :c], x[i, sel])
        if c:
            ref = torch.topk(x[i, : int(end[i])].float(), c).values
            assert torch.equal(torch.sort(x[i, sel].float()).values, torch.sort(ref).values)
        if mode == "sorted_index":
            assert (sel[1:] > sel[:-1]).all()
        if mode == "sorted" and c > 1:
            assert (vals[i, 1:c].float() <= vals[i, : c - 1].float()).all()
    # no end / no offset / return_value=False, vocab < topk
    v2, i2 = ds_topk(x[:, :40], k, indices_type=torch.int64, return_value=False)
    assert v2 is None and i2.dtype == torch.int64 and (i2[:, 40:] == 2147483647).all()


@needs_cuda
@pytest.mark.parametrize("V", [700, 4096, 16384])
@pytest.mark.parametrize("ties", [False, True])
def test_topk_fused_index_kernel_matches_torch(V, ties):
    """Fused radix-select index top-k == the torch path (same set incl. tie order: lowest index first)."""
    gen = torch.Generator(device=DEV).manual_seed(V + ties)
    N, k = 9, 512
    if ties:
        x = torch.randint(0, 6, (N, V + 40), device=DEV, generator=gen).float().to(torch.bfloat16)[:, :V]
    else:
        x = torch.randn((N, V + 40), device=DEV, generator=gen).to(torch.bfloat16)[:, :V]  # strided rows
    x[0, :5] = float("-inf")
    x[1, 3] = float("nan")
    end = torch.randint(0, V + 1, (N,), device=DEV, generator=gen).to(torch.int32)
    end[2], end[3] = 3, V
    off = torch.randint(0, 100, (N,), device=DEV, generator=gen).to(torch.int32)
    res = {}
    for fused in ("0", "1"):
        with EnvSet(SM120_TOPK_IMPL="triton", SM120_TOPK_FUSED=fused):
            o = torch.full((N, k), 12345, dtype=torch.int32, device=DEV)
            ds_topk(x, k, end=end, indices_type=torch.int32, sorted_index=True, output_idx=o, output_idx_offset=off,
                    idx_oob_fill_value=-1, return_value=False)
            res[fused] = o
    if ties:
        assert torch.equal(res["1"], res["0"]), "tie order differs from torch gatherTopK"
    else:
        for i in range(N):
            assert set(res["1"][i].tolist()) == set(res["0"][i].tolist()), i
        assert torch.equal(res["1"], res["0"])


@needs_cuda
def test_topk_triton_inf_nan_rank_above_oob():
    x = torch.full((2, 600), float("-inf"), device=DEV, dtype=torch.bfloat16)
    x[0, :3] = torch.tensor([1.0, float("nan"), 2.0])
    end = torch.tensor([520, 4], dtype=torch.int32, device=DEV)
    with EnvSet(SM120_TOPK_IMPL="triton"):
        _, idx = ds_topk(x, 512, end=end, indices_type=torch.int32, sorted_index=True, idx_oob_fill_value=-1,
                         return_value=False)
    assert ((idx[0] >= 0) & (idx[0] < 520)).all() and idx[0].unique().numel() == 512
    assert sorted(idx[1, :4].tolist()) == [0, 1, 2, 3] and (idx[1, 4:] == -1).all()


# ---------------------------------------------------------------------------------------------
# cache formats
# ---------------------------------------------------------------------------------------------
@needs_cuda
def test_v41_dequant_matches_deepinfer_loader():
    di = deepinfer_writer()
    if di is None:
        pytest.skip("DeepInfer not importable")
    gen = torch.Generator(device=DEV).manual_seed(3)
    for fp4, rec, page in ((False, 528, 134), (True, 288, 64)):
        _, cache = make_cache(9, page, rec, gen)
        slots = torch.randperm(9 * page, device=DEV, generator=gen)[: 5 * page].to(torch.int32)
        kv = latent(slots.numel(), gen)
        store(cache, kv, slots, fp4)
        b = torch.arange(9, dtype=torch.int32, device=DEV)
        segs = torch.stack([b, torch.zeros_like(b), torch.full_like(b, page), b * page], 1).contiguous()
        layout = di["cc"].V41_FP4_FORMAT if fp4 else di["cc"].V41_FORMAT
        ref = di["load"](cache, segs, 9 * page, layout=layout).bf16_data.float()
        ours = dequant_all(cache)
        s = slots.long()
        assert torch.equal(ours[s].to(torch.bfloat16).float(), ref[s]), f"fp4={fp4}"


@needs_cuda
def test_v41_encoders_roundtrip():
    gen = torch.Generator(device=DEV).manual_seed(4)
    x = latent(256, gen)
    assert cos_sim(Q.dequant_v41_fp8_records(Q.quantize_v41_fp8_records(x)), x) > 0.999
    assert cos_sim(Q.dequant_v41_fp4_records(Q.quantize_v41_fp4_records(x)), x) > 0.98


# ---------------------------------------------------------------------------------------------
# decode: torch vs naive, b12x vs torch
# ---------------------------------------------------------------------------------------------
@needs_cuda
@pytest.mark.parametrize("with_extra", [True, False])
def test_decode_torch_vs_naive(with_extra):
    c = build_decode_case(seed=11, with_extra=with_extra)
    out, lse = run_flash_mla_decode(c, "torch")
    deq_s = dequant_all(c["swa"])
    deq_m = dequant_all(c["main"]) if with_extra else None
    ref, ref_lse = naive_decode(c["q"], deq_s, c["swa_idx"], None, deq_m, c["main_idx"], c["main_len"],
                                c["sm_scale"], c["sink"])
    assert_close(out, ref, cos=0.99999, rel=0.01, name="decode torch vs naive")
    assert torch.allclose(lse, ref_lse, rtol=1e-4, atol=1e-4)


@needs_cuda
@pytest.mark.parametrize("compute", ["bf16", "fp8"])
@pytest.mark.parametrize("with_extra", [True, False])
def test_decode_b12x_vs_torch(compute, with_extra):
    if not b12x_available():
        pytest.skip("b12x compressed_sparse_mla not available")
    c = build_decode_case(seed=5, with_extra=with_extra)
    ref, _ = run_flash_mla_decode(c, "torch")
    out, _ = run_flash_mla_decode(c, "b12x", compute)
    assert_close(out, ref, cos=0.9995 if compute == "bf16" else 0.99, rel=0.05 if compute == "bf16" else 0.15,
                 name=f"decode b12x({compute}) vs torch")
    # second call with fewer rows reuses the plan (row bucket) and must not see stale rows
    c2 = dict(c)
    for k in ("q", "swa_idx", "main_idx", "main_len"):
        if c2[k] is not None:
            c2[k] = c2[k][:5].contiguous()
    ref2, _ = run_flash_mla_decode(c2, "torch")
    out2, _ = run_flash_mla_decode(c2, "b12x", compute)
    assert_close(out2, ref2, cos=0.9995 if compute == "bf16" else 0.99, rel=0.05 if compute == "bf16" else 0.15,
                 name="decode b12x reuse")


# ---------------------------------------------------------------------------------------------
# prefill (flash_mla_sparse_fwd)
# ---------------------------------------------------------------------------------------------
@needs_cuda
def test_sparse_prefill_vs_naive():
    gen = torch.Generator(device=DEV).manual_seed(7)
    s_q, s_kv, topk = 50, 700, 192
    q = (torch.randn((s_q, H, D), device=DEV, generator=gen) * 0.5).to(torch.bfloat16)
    kv = latent(s_kv, gen).view(s_kv, 1, D)
    idx = torch.randint(-50, s_kv + 50, (s_q, 1, topk), device=DEV, generator=gen).to(torch.int32)
    idx[2] = -1  # lonely query
    tl = torch.randint(0, topk + 1, (s_q,), device=DEV, generator=gen).to(torch.int32)
    sink = torch.randn((H,), device=DEV, generator=gen).float()
    with EnvSet(SM120_ATTN_PREFILL_IMPL="torch"):  # the torch reference (b12x returns NaN max_logits)
        out, mx, lse = flash_mla.flash_mla_sparse_fwd(q, kv, idx, D ** -0.5, 512, sink, tl)
    assert out.dtype == torch.bfloat16 and out.shape == (s_q, H, D) and lse.shape == (s_q, H)
    for r in range(s_q):
        ii = idx[r, 0, : tl[r]].long()
        ii = ii[(ii >= 0) & (ii < s_kv)]
        if ii.numel() == 0:
            assert (out[r] == 0).all() and torch.isinf(lse[r]).all() and (lse[r] > 0).all()
            assert torch.isneginf(mx[r]).all()
            continue
        K = kv[ii, 0].float()
        s = q[r].float() @ K.t() * D ** -0.5
        l_ = torch.logsumexp(s, -1)
        o = (torch.softmax(s, -1) @ K) / (1 + torch.exp(sink - l_)).view(-1, 1)
        assert_close(out[r], o, cos=0.9999, rel=0.02, name=f"prefill row {r}")
        assert torch.allclose(lse[r], l_, rtol=1e-4, atol=1e-4)
        assert torch.allclose(mx[r], s.amax(-1), rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------------------------
# fused norm/rope/attn/rope/cast
# ---------------------------------------------------------------------------------------------
def cos_sin_cache(n_pos=4096):
    inv = 1.0 / (10000 ** (torch.arange(0, ROPE, 2, dtype=torch.float, device=DEV) / ROPE))
    f = torch.outer(torch.arange(n_pos, dtype=torch.float, device=DEV), inv)
    cis = torch.polar(torch.ones_like(f), f)
    return torch.cat((cis.real, cis.imag), -1)


def dequant_fused_out(fp8, sf, G):
    """(fp8 [s,G,gs*512], sf int32 [s,G,gs*4]) -> natural fp32 [s, H, 512] (undo the wv layout)."""
    s = fp8.shape[0]
    scales = Q.unpack_ue8m0_int32(sf.contiguous()).view(s, G, -1)  # [s,G,gs*16]
    x = fp8.float().view(s, G, -1, 32) * scales.unsqueeze(-1)
    gs = fp8.shape[-1] // 512
    x = x.view(s, G, gs * 512)
    if F_.cfg.permute_enabled():
        x = x.view(s, G, 512 // 32, gs, 32).permute(0, 1, 3, 2, 4)
    return x.reshape(s, G * gs, 512)


def ref_fused_from_natural(q_nat, positions, cs, norm, eps, attend):
    """Manual composition: norm scale, rope q -> bf16, attention, inverse rope (all fp32)."""
    q_scale = torch.rsqrt(q_nat.float().pow(2).mean(-1) + eps) if norm else None
    qr = q_nat.float().clone()
    from sm120_attn.rope import apply_rope_
    apply_rope_(qr, positions, cs, ROPE, False, conjugate=False)
    out = attend(qr.to(torch.bfloat16), q_scale).float().clone()
    apply_rope_(out, positions, cs, ROPE, False, conjugate=True)
    return out


@needs_cuda
@pytest.mark.parametrize("permute", ["1", "0"])
@pytest.mark.parametrize("norm", [True, False])
def test_fused_decode_torch_vs_manual(permute, norm):
    with EnvSet(SM120_ATTN_PERMUTE=permute, SM120_ATTN_IMPL="torch"):
        c = build_decode_case(seed=21)
        rows = c["q"].shape[0]
        pos = torch.randint(0, 4096, (rows,), device=DEV).to(torch.int32)
        cs = cos_sin_cache()
        G = H // 8
        q_nat = c["q"]
        q_in = F_.q_to_fused_layout(q_nat)
        assert torch.equal(F_.q_from_fused_layout(q_in), q_nat)
        fp8, sf, lse = flash_mla.fused_norm_rope_attn_rope_cast.decode(
            enable_q_norm=norm, rms_norm_eps=1e-4, token_positions=pos, is_rope_neox_style=False, rope_dim=ROPE,
            cos_sin_cache=cs, n_wv_group=G, num_per_channels=32, use_tma_aligned_col_major_sf=True, round_sf=True,
            use_packed_ue8m0=True, q=q_in, k_cache=c["swa"], indices_in_kvcache=c["swa_idx"], sm_scale=c["sm_scale"],
            d_v=D, attn_sink=c["sink"], topk_length=None, extra_k_cache=c["main"],
            extra_indices_in_kvcache=c["main_idx"], extra_topk_length=c["main_len"])
        assert fp8.dtype == torch.float8_e4m3fn and fp8.shape == (rows, G, 8 * D)
        assert sf.dtype == torch.int32 and sf.shape == (rows, G, 8 * D // 128) and sf.stride(0) == 1
        assert lse.shape == (rows, H)
        deq_s, deq_m = dequant_all(c["swa"]), dequant_all(c["main"])

        def attend(qr, q_scale):
            return naive_decode(qr, deq_s, c["swa_idx"], None, deq_m, c["main_idx"], c["main_len"],
                                c["sm_scale"], c["sink"], q_scale)[0]

        ref = ref_fused_from_natural(q_nat, pos, cs, norm, 1e-4, attend)
        got = dequant_fused_out(fp8, sf, G)
        assert_close(got, ref, cos=0.9995, rel=0.08, name="fused decode (fp8 out) vs manual")


@needs_cuda
def test_fused_decode_b12x_vs_torch():
    if not b12x_available():
        pytest.skip("b12x compressed_sparse_mla not available")
    c = build_decode_case(seed=23)
    rows = c["q"].shape[0]
    pos = torch.randint(0, 4096, (rows,), device=DEV).to(torch.int32)
    cs = cos_sin_cache()
    G = H // 8
    res = {}
    for impl in ("torch", "b12x"):
        with EnvSet(SM120_ATTN_IMPL=impl, SM120_ATTN_STRICT=1 if impl == "b12x" else 0):
            fp8, sf, _ = flash_mla.fused_norm_rope_attn_rope_cast.decode(
                True, 1e-4, pos, False, ROPE, cs, G, 32, True, True, True, F_.q_to_fused_layout(c["q"]), c["swa"],
                c["swa_idx"], c["sm_scale"], D, c["sink"], None, c["main"], c["main_idx"], c["main_len"])
            res[impl] = dequant_fused_out(fp8, sf, G)
    assert_close(res["b12x"], res["torch"], cos=0.999, rel=0.1, name="fused decode b12x vs torch")


@needs_cuda
def test_fused_prefill_vs_manual():
    gen = torch.Generator(device=DEV).manual_seed(31)
    s_q, s_kv, topk = 24, 300, 128
    q_nat = (torch.randn((s_q, H, D), device=DEV, generator=gen) * 0.5).to(torch.bfloat16)
    kv = latent(s_kv, gen).view(s_kv, 1, D)
    idx = torch.randint(-5, s_kv, (s_q, 1, topk), device=DEV, generator=gen).to(torch.int32)
    sink = torch.randn((H,), device=DEV, generator=gen).float()
    pos = torch.randint(0, 4096, (s_q,), device=DEV).to(torch.int32)
    cs = cos_sin_cache()
    G = H // 8
    with EnvSet(SM120_ATTN_PREFILL_IMPL="torch"):
        fp8, sf, mx, lse = flash_mla.fused_norm_rope_attn_rope_cast.prefill(
            True, 1e-4, pos, False, ROPE, cs, G, 32, True, True, True, F_.q_to_fused_layout(q_nat), kv, idx,
            D ** -0.5, D, sink, None)
    assert mx.shape == lse.shape == (s_q, H)

    def attend(qr, q_scale):
        from sm120_attn.sparse_attn import sparse_prefill_torch
        return sparse_prefill_torch(qr, kv[:, 0], idx[:, 0], D ** -0.5, D, sink, None, q_scale, None)[0]

    ref = ref_fused_from_natural(q_nat, pos, cs, True, 1e-4, attend)
    assert_close(dequant_fused_out(fp8, sf, G), ref, cos=0.9995, rel=0.08, name="fused prefill vs manual")


@needs_cuda
@pytest.mark.parametrize("permute", ["1", "0"])
def test_weight_permutations_are_consistent(permute):
    """permute_q_b_proj: (x @ W_perm^T) is the fused q layout of (x @ W^T);
    permute_wv_proj: einsum(fused_out, Wv_perm) == einsum(natural_out, Wv)."""
    with EnvSet(SM120_ATTN_PERMUTE=permute):
        gen = torch.Generator(device=DEV).manual_seed(41)
        K = 256
        w = (torch.randn((H * D, K), device=DEV, generator=gen) / 10).to(torch.float8_e4m3fn)
        sf = torch.randint(100, 130, (H * D, K // 128), device=DEV, generator=gen).to(torch.int32)
        wp, sfp = flash_mla.fused_norm_rope_attn_rope_cast.permute_q_b_proj((w, sf), H, D)
        x = torch.randn((3, K), device=DEV, generator=gen)
        q_nat = (x @ w.float().t()).view(3, H, D)
        q_perm = (x @ wp.float().t()).view(3, H, D)
        assert torch.allclose(F_.q_from_fused_layout(q_perm), q_nat)
        if permute == "1":
            out_row = F_.q_b_proj_row_permutation(H, D, DEV)
            assert torch.equal(sfp[out_row], sf)
        # wv: [G, N, gs*D] with MN-major packed sf
        G, gs, N = H // 8, 8, 64
        wv = (torch.randn((G, N, gs * D), device=DEV, generator=gen) / 10).to(torch.float8_e4m3fn)
        exps = torch.randint(120, 130, (G, N, gs * D // 32), device=DEV, generator=gen).to(torch.uint8)
        packed = exps.contiguous().view(torch.int32)  # [G, N, gs*D/128]
        sf_mn = torch.empty_strided(packed.shape, (packed.shape[1] * packed.shape[2], 1, packed.shape[1]),
                                    dtype=torch.int32, device=DEV)
        sf_mn.copy_(packed)
        wvp, sfwp = flash_mla.fused_norm_rope_attn_rope_cast.permute_wv_proj((wv, sf_mn), gs, D)
        assert sfwp.stride() == sf_mn.stride()

        def deq(wt, s):
            sc = Q.unpack_ue8m0_int32(s.contiguous()).view(G, N, -1)
            return (wt.float().view(G, N, -1, 32) * sc.unsqueeze(-1)).view(G, N, -1)

        out_nat = torch.randn((5, H, D), device=DEV, generator=gen)
        fp8, osf = F_.cast_output(out_nat.clone(), G)
        o_fused = fp8.float().view(5, G, -1, 32) * Q.unpack_ue8m0_int32(osf.contiguous()).view(5, G, -1).unsqueeze(-1)
        o_fused = o_fused.view(5, G, gs * D)
        o_nat_q = dequant_fused_out(fp8, osf, G).view(5, G, gs * D)
        y_fused = torch.einsum("bhr,hdr->bhd", o_fused, deq(wvp, sfwp))
        y_nat = torch.einsum("bhr,hdr->bhd", o_nat_q, deq(wv, sf_mn))
        assert torch.allclose(y_fused, y_nat, rtol=1e-4, atol=1e-3)


# ---------------------------------------------------------------------------------------------
# indexer (deep_gemm MQA logits)
# ---------------------------------------------------------------------------------------------
def fp4_cast(x):
    """deep_gemm per_token_cast_to_fp4(use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)."""
    m, n = x.shape
    xv = x.float().view(m, n // 32, 32)
    sf = Q.ceil_to_ue8m0(xv.abs().amax(-1).clamp_min(1e-4) / 6.0)
    packed = Q.pack_e2m1(xv / sf.unsqueeze(-1)).view(m, n // 2).view(torch.int8)
    nsf = sf.shape[-1]
    if nsf % 4:
        sf = F.pad(sf, (0, 4 - nsf % 4), value=1.0)
    return packed, Q.pack_ue8m0(sf)


def naive_logits(qf, kf, w):
    return (torch.einsum("mhd,nd->mhn", qf, kf).relu() * w.float().unsqueeze(-1)).sum(1)


@needs_cuda
def test_indexer_contiguous_paged_sparse():
    gen = torch.Generator(device=DEV).manual_seed(51)
    Hi, Di, nkv, nq = 64, 128, 1000, 40
    q = torch.randn((nq, Hi, Di), device=DEV, generator=gen)
    k = torch.randn((nkv, Di), device=DEV, generator=gen)
    q_fp, q_sf = fp4_cast(q.view(-1, Di))
    q_fp, q_sf = q_fp.view(nq, Hi, Di // 2), q_sf.view(nq, Hi)
    k_fp, k_sf = fp4_cast(k)
    k_sf = k_sf.view(nkv)
    qf = Q.dequant_index_values(q_fp, q_sf, Di)
    kf = Q.dequant_index_values(k_fp, k_sf, Di)
    w_store = torch.zeros((nq, 72), dtype=torch.bfloat16, device=DEV)
    w = w_store[:, :Hi]
    w.copy_(torch.randn((nq, Hi), device=DEV, generator=gen))
    ks = torch.randint(0, 300, (nq,), device=DEV, generator=gen).to(torch.int32)
    ke = (ks + torch.randint(0, 600, (nq,), device=DEV, generator=gen)).clamp(max=nkv).to(torch.int32)
    max_k = int((ke - ks).max())
    logits = DG["fp8_fp4_mqa_logits"]((q_fp, q_sf), (k_fp, k_sf), w, ks, ke, max_seqlen_k=max_k)
    assert logits.shape == (nq, max_k) and logits.dtype == torch.bfloat16
    assert (logits.stride(0) * 2) % 1024 == 0
    full = naive_logits(qf, kf, w)
    for i in range(nq):
        L = int(ke[i] - ks[i])
        assert torch.allclose(logits[i, :L].float(), full[i, ks[i]:ke[i]].to(torch.bfloat16).float(), rtol=1e-2, atol=1e-2)
        assert torch.isneginf(logits[i, L:].float()).all()

    # paged: put the same K into a block-layout cache, one request per query
    page = 64
    nb = (nkv + page - 1) // page + 3
    stride = Q_align(page * 68, 128) + 128
    buf = torch.zeros((nb, stride), dtype=torch.uint8, device=DEV)
    cache = buf[:, :page * 68].view(nb, page, 1, 68)
    vals, sfs = Q.split_index_cache(cache, 64, 4)
    perm = torch.randperm(nb, device=DEV, generator=gen)
    kp = torch.zeros((nb * page, Di), device=DEV)
    kp[:nkv] = k
    kpf, kps = fp4_cast(kp)
    nlog = (nkv + page - 1) // page
    for lb in range(nb):
        pb = perm[lb]
        vals[pb] = kpf[lb * page:(lb + 1) * page].view(torch.uint8)
        sfs[pb] = kps[lb * page:(lb + 1) * page].contiguous().view(torch.uint8).view(page, 4)
    bt = perm[:nlog].to(torch.int32).view(1, -1).expand(nq, -1).contiguous()
    ctx = ke.view(nq, 1).contiguous()
    meta = DG["get_paged_mqa_logits_metadata"](ctx, page, 8, indices=torch.arange(nq, dtype=torch.int32, device=DEV))
    pl = DG["fp8_fp4_paged_mqa_logits"]((q_fp.unsqueeze(1), q_sf.unsqueeze(1)), cache, w, ctx, bt, meta, nkv,
                                        indices=torch.arange(nq, dtype=torch.int32, device=DEV))
    assert pl.shape == (nq, nkv)
    for i in range(nq):
        L = int(ke[i])
        assert torch.allclose(pl[i, :L].float(), full[i, :L].to(torch.bfloat16).float(), rtol=1e-2, atol=1e-2)
        assert torch.isneginf(pl[i, L:].float()).all()

    # sparse (candidate blocks) == gathered dense
    sbk, nblk = 16, 8
    klen = ke - ks
    cand = torch.full((nq, nblk), -1, dtype=torch.int32, device=DEV)
    for i in range(nq):
        avail = (int(klen[i]) + sbk - 1) // sbk
        sel = torch.randperm(max(avail, 1), device=DEV, generator=gen)[: min(avail, nblk)].sort().values
        cand[i, : sel.numel()] = sel.to(torch.int32)
    abs_blocks = cand + torch.div(ks, sbk, rounding_mode="floor").unsqueeze(1)
    smeta = DG["get_sparse_mqa_logits_metadata"](ks, ke, nkv, abs_blocks, q_fp.dtype, sbk, use_unaligned_ks=True)
    sl = DG["fp8_fp4_sparse_mqa_logits"]((q_fp, q_sf), (k_fp, k_sf), w, smeta, nblk, sbk, use_unaligned_ks=True)
    assert sl.shape == (nq, nblk * sbk)
    for i in range(nq):
        for t in range(nblk * sbk):
            b = int(cand[i, t // sbk])
            tok = int(ks[i]) + b * sbk + t % sbk
            if b >= 0 and tok < int(ke[i]):
                assert torch.allclose(sl[i, t].float(), full[i, tok].to(torch.bfloat16).float(), rtol=1e-2, atol=1e-2)
            elif b < 0:
                assert torch.isneginf(sl[i, t].float())
    pmeta = DG["get_paged_sparse_mqa_logits_metadata"](ke.contiguous(), bt, torch.arange(nq, dtype=torch.int32, device=DEV),
                                                       page, cand + torch.div(ks, sbk, rounding_mode="floor").unsqueeze(1),
                                                       q_fp.dtype, sbk)
    psl = DG["fp8_fp4_paged_sparse_mqa_logits"]((q_fp.unsqueeze(1), q_sf.unsqueeze(1)), cache, w, pmeta, nblk, sbk)
    assert psl.shape == (nq, nblk * sbk)
    for i in range(nq):
        nv = min((int(ke[i]) + sbk - 1) // sbk, nblk)
        for t in range(nv * sbk):
            tok = (int(cand[i, t // sbk]) + int(ks[i]) // sbk) * sbk + t % sbk
            if 0 <= tok < int(ke[i]) and int(cand[i, t // sbk]) >= 0:
                assert torch.allclose(psl[i, t].float(), full[i, tok].to(torch.bfloat16).float(), rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------------------------
# DeepInfer call-site wrappers (only if DeepInfer's attention_backend is importable)
# ---------------------------------------------------------------------------------------------
@needs_cuda
def test_deepinfer_attention_backend_decode():
    try:
        from deepinfer.model_executor.layers.attention_backend import decode_mla_mqa_sparse
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"DeepInfer attention_backend not importable: {exc}")
    c = build_decode_case(seed=61)
    with EnvSet(SM120_ATTN_IMPL="torch"):
        sm = flash_mla.flash_mla_interface.FlashMLASchedMeta()
        out, lse = decode_mla_mqa_sparse(c["q"], c["swa"], D, c["swa_idx"], sm, c["sm_scale"], attn_sink=c["sink"],
                                         key_cache_extra=c["main"], indices_in_kvcache_extra=c["main_idx"],
                                         index_length_extra=c["main_len"])
        ref, _ = run_flash_mla_decode(c, "torch")
    assert out.shape == (c["q"].shape[0], H, D) and lse.shape == (c["q"].shape[0], H)
    assert torch.equal(out, ref)
    # FlashMLASchedMeta reuse with a different topk must trip the original consistency assert
    with pytest.raises(AssertionError):
        decode_mla_mqa_sparse(c["q"], c["swa"], D, c["swa_idx"][:, :64].contiguous(), sm, c["sm_scale"],
                              attn_sink=c["sink"], key_cache_extra=c["main"],
                              indices_in_kvcache_extra=c["main_idx"], index_length_extra=c["main_len"])


# ---------------------------------------------------------------------------------------------
# b12x prefill / indexer / CUDA graphs
# ---------------------------------------------------------------------------------------------
def b12x_indexer_available():
    try:
        from b12x.attention import dsa_indexer
        return bool(dsa_indexer.is_supported(DEV))
    except Exception:  # noqa: BLE001
        return False


def fp8_exact(x):
    return Q.dequant_v41_fp8_records(Q.quantize_v41_fp8_records(x)).to(torch.bfloat16)


def fp4_exact(x):
    return Q.dequant_v41_fp4_records(Q.quantize_v41_fp4_records(x)).to(torch.bfloat16)


def build_prefill_case(seed=71, s_q=96, ratio=4, window=128, topk=512, heads=H, holes=0.05, with_len=False):
    """DeepInfer-shaped prefill KV [SWA cached (window) | current SWA (s_q) | compressed ((window+s_q)//ratio)].

    SWA tokens are FP8-exact (DeepInfer cast_back), compressed tokens FP4-exact (loaded FP4 cache);
    ratio 1 = SWA-only layer. Indices per query: [SWA window | top-k compressed] (kv_index_cat)."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    n_swa = window + s_q
    n_comp = 0 if ratio <= 1 else n_swa // ratio
    kv = torch.cat((fp8_exact(latent(n_swa, gen)), fp4_exact(latent(max(n_comp, 1), gen))[:n_comp]), 0)
    pos = window + torch.arange(s_q, device=DEV)  # absolute position of each query in the SWA region
    w = torch.arange(window, device=DEV).view(1, -1)
    swa_idx = pos.view(-1, 1) - window + 1 + w
    swa_idx = torch.where(swa_idx >= 0, swa_idx, torch.full_like(swa_idx, -1))
    parts = [swa_idx]
    if n_comp:
        avail = ((pos + 1) // ratio).clamp(max=n_comp)
        k = min(topk, n_comp)
        score = torch.rand((s_q, n_comp), device=DEV, generator=gen)
        score.masked_fill_(torch.arange(n_comp, device=DEV).view(1, -1) >= avail.view(-1, 1), -1.0)
        val, col = torch.topk(score, k, dim=1)
        comp = torch.where(val >= 0, n_swa + col, torch.full_like(col, -1))
        if k < topk:
            comp = F.pad(comp, (0, topk - k), value=-1)
        parts.append(comp)
    idx = torch.cat(parts, 1)
    idx[torch.rand(idx.shape, device=DEV, generator=gen) < holes] = -1
    idx[1] = -1  # lonely query
    tl = torch.randint(window // 2, idx.shape[1] + 1, (s_q,), device=DEV, generator=gen).to(torch.int32) \
        if with_len else None
    q = (torch.randn((s_q, heads, D), device=DEV, generator=gen) * 0.5).to(torch.bfloat16)
    sink = (torch.randn((heads,), device=DEV, generator=gen) * 2).float()
    return dict(q=q, kv=kv.view(-1, 1, D), idx=idx.to(torch.int32).view(s_q, 1, -1), tl=tl, sink=sink,
                sm_scale=D ** -0.5)


def run_prefill(c, impl, compute="bf16", kv_mode="auto", sink=True, rows=None):
    env = dict(SM120_ATTN_PREFILL_IMPL=impl, SM120_ATTN_COMPUTE=compute, SM120_ATTN_PREFILL_KV=kv_mode,
               SM120_ATTN_STRICT=1 if impl == "b12x" else 0)
    if rows is not None:
        env["SM120_ATTN_PREFILL_ROWS"] = rows
    with EnvSet(**env):
        return flash_mla.flash_mla_sparse_fwd(c["q"], c["kv"], c["idx"], c["sm_scale"], D,
                                              c["sink"] if sink else None, c["tl"])


@needs_cuda
@pytest.mark.parametrize("ratio", [1, 2, 4])
@pytest.mark.parametrize("compute", ["bf16", "fp8"])
def test_prefill_b12x_vs_torch(ratio, compute):
    if not b12x_available():
        pytest.skip("b12x compressed_sparse_mla not available")
    c = build_prefill_case(seed=71 + ratio, s_q=96, ratio=ratio, with_len=(ratio == 2))
    ref, _, _ = run_prefill(c, "torch")
    out, mx, lse = run_prefill(c, "b12x", compute)
    assert out.shape == ref.shape and out.dtype == torch.bfloat16
    assert (out[1] == 0).all() and torch.isinf(lse[1]).all() and torch.isneginf(mx[1]).all()
    tight = compute == "bf16"
    assert_close(out, ref, cos=0.9999 if tight else 0.99, rel=0.03 if tight else 0.15,
                 name=f"prefill b12x({compute}) ratio={ratio}")


@needs_cuda
@pytest.mark.parametrize("sink", [True, False])
def test_prefill_b12x_chunked_rows_and_fp8_kv(sink):
    """s_q > SM120_ATTN_PREFILL_ROWS (row chunk loop), and SM120_ATTN_PREFILL_KV=fp8 (single cache)."""
    if not b12x_available():
        pytest.skip("b12x compressed_sparse_mla not available")
    c = build_prefill_case(seed=81, s_q=200, ratio=4)
    ref, _, _ = run_prefill(c, "torch", sink=sink)
    out, _, _ = run_prefill(c, "b12x", rows=64, sink=sink)
    assert_close(out, ref, cos=0.9999, rel=0.03, name="prefill b12x row-chunked")
    out8, _, _ = run_prefill(c, "b12x", kv_mode="fp8", rows=64, sink=sink)
    assert_close(out8, ref, cos=0.999, rel=0.08, name="prefill b12x fp8-kv")


@needs_cuda
def test_prefill_b12x_realistic_2048():
    if not b12x_available():
        pytest.skip("b12x compressed_sparse_mla not available")
    c = build_prefill_case(seed=91, s_q=2048, ratio=4, holes=0.0)
    ref, _, _ = run_prefill(c, "torch")
    out, _, _ = run_prefill(c, "b12x")
    assert_close(out, ref, cos=0.9999, rel=0.03, name="prefill b12x 2048x(128+512)")


@needs_cuda
def test_fused_prefill_b12x_vs_torch():
    if not b12x_available():
        pytest.skip("b12x compressed_sparse_mla not available")
    c = build_prefill_case(seed=101, s_q=64, ratio=4)
    s_q = c["q"].shape[0]
    pos = torch.randint(0, 4096, (s_q,), device=DEV).to(torch.int32)
    cs = cos_sin_cache()
    G = H // 8
    res = {}
    for impl in ("torch", "b12x"):
        with EnvSet(SM120_ATTN_PREFILL_IMPL=impl, SM120_ATTN_STRICT=1 if impl == "b12x" else 0):
            fp8, sf, _, _ = flash_mla.fused_norm_rope_attn_rope_cast.prefill(
                True, 1e-4, pos, False, ROPE, cs, G, 32, True, True, True, F_.q_to_fused_layout(c["q"]), c["kv"],
                c["idx"], c["sm_scale"], D, c["sink"], None)
            res[impl] = dequant_fused_out(fp8, sf, G)
    assert_close(res["b12x"], res["torch"], cos=0.999, rel=0.1, name="fused prefill b12x vs torch")


@needs_cuda
@pytest.mark.parametrize("rows", [1, 8, 32])
def test_decode_b12x_batch_sizes(rows):
    if not b12x_available():
        pytest.skip("b12x compressed_sparse_mla not available")
    c = build_decode_case(seed=111 + rows, rows=rows)
    ref, _ = run_flash_mla_decode(c, "torch")
    out, _ = run_flash_mla_decode(c, "b12x")
    assert_close(out, ref, cos=0.9995, rel=0.05, name=f"decode b12x rows={rows}")


@needs_cuda
@pytest.mark.parametrize("permute", ["1", "0"])
@pytest.mark.parametrize("neox", [False, True])
@pytest.mark.parametrize("norm", [True, False])
def test_fused_triton_vs_torch(permute, neox, norm):
    """Triton q-prep / rope+cast kernels against the torch reference (same attention core)."""
    from sm120_attn import fused_triton
    if not fused_triton.available():
        pytest.skip("triton not available")
    c = build_decode_case(seed=161, rows=37)
    rows = c["q"].shape[0]
    pos = torch.randint(0, 4096, (rows,), device=DEV).to(torch.int32)
    cs = cos_sin_cache()
    G = H // 8
    res = {}
    with EnvSet(SM120_ATTN_PERMUTE=permute, SM120_ATTN_DECODE_IMPL="torch"):
        q_in = F_.q_to_fused_layout(c["q"])
        for impl in ("torch", "triton"):
            with EnvSet(SM120_ATTN_FUSED_IMPL=impl, SM120_ATTN_STRICT=1):
                fp8, sf, _ = flash_mla.fused_norm_rope_attn_rope_cast.decode(
                    norm, 1e-4, pos, neox, ROPE, cs, G, 32, True, True, True, q_in, c["swa"], c["swa_idx"],
                    c["sm_scale"], D, c["sink"], None, c["main"], c["main_idx"], c["main_len"])
                assert sf.stride(0) == 1 and sf.shape == (rows, G, 8 * D // 128)
                res[impl] = (fp8, sf, dequant_fused_out(fp8, sf, G))
        # prep kernel alone: q_r and q_scale
        q_r, sc, _ = fused_triton.prepare_q(q_in, norm, 1e-4, pos, neox, ROPE, cs, permute == "1", fold=False)
        q_r0, sc0 = F_._prepare_q(q_in, norm, 1e-4, pos, neox, ROPE, cs)
        assert (q_r.float() - q_r0.float()).abs().max().item() <= 2 ** -6 * q_r0.float().abs().max().item()
        if norm:
            assert torch.allclose(sc, sc0, rtol=1e-5)
    a, b = res["triton"][2], res["torch"][2]
    # one FP8 step (12.5 % relative) on an element is possible: last-ulp RoPE differences (FMA) flip rounding
    assert_close(a, b, cos=0.99999, rel=0.05, name="fused triton vs torch")
    same = (res["triton"][1] == res["torch"][1]).float().mean().item()
    assert same > 0.99, f"scale bytes agree on {same:.4f} only"


@needs_cuda
def test_encode_v41_triton_matches_torch():
    from sm120_attn import fused_triton
    if not fused_triton.available():
        pytest.skip("triton not available")
    gen = torch.Generator(device=DEV).manual_seed(171)
    x = torch.cat((latent(300, gen), fp8_exact(latent(200, gen)), fp4_exact(latent(200, gen)),
                   torch.zeros((3, D), device=DEV, dtype=torch.bfloat16)), 0)
    x[5, :7] = torch.tensor([1e-30, -0.0, 6.0, -5.0, 0.25, 448.0, -1e4], device=DEV).to(torch.bfloat16)
    n = x.shape[0]
    r8 = torch.empty((n + 5, 528), dtype=torch.uint8, device=DEV)
    r4 = torch.empty((n + 5, 288), dtype=torch.uint8, device=DEV)
    flag = fused_triton.encode_v41(x, r8, r4)
    assert torch.equal(r8[:n], Q.quantize_v41_fp8_records(x))
    assert torch.equal(r4[:n], Q.quantize_v41_fp4_records(x))
    ref = (Q.dequant_v41_fp4_records(Q.quantize_v41_fp4_records(x)) == x.float()).all(-1)
    assert torch.equal(flag, ref)
    assert flag[500:700].all() and not flag[:300].any()


# ---- indexer -------------------------------------------------------------------------------
def build_index_case(seed=121, Hi=32, nq=48, nkv=1500, page=64, weights_dtype=torch.bfloat16):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    Di = 128
    q = torch.randn((nq, Hi, Di), device=DEV, generator=gen)
    k = torch.randn((nkv, Di), device=DEV, generator=gen)
    q_fp, q_sf = fp4_cast(q.view(-1, Di))
    q_fp, q_sf = q_fp.view(nq, Hi, Di // 2), q_sf.view(nq, Hi)
    k_fp, k_sf = fp4_cast(k)
    k_sf = k_sf.view(nkv)
    w = torch.randn((nq, Hi), device=DEV, generator=gen).to(weights_dtype)
    ks = torch.randint(0, 300, (nq,), device=DEV, generator=gen).to(torch.int32)
    ke = (ks + torch.randint(1, nkv, (nq,), device=DEV, generator=gen)).clamp(max=nkv).to(torch.int32)
    nb = (nkv + page - 1) // page + 3
    stride = Q_align(page * 68, 128) + 128
    buf = torch.zeros((nb, stride), dtype=torch.uint8, device=DEV)
    cache = buf[:, :page * 68].view(nb, page, 1, 68)
    vals, sfs = Q.split_index_cache(cache, 64, 4)
    perm = torch.randperm(nb, device=DEV, generator=gen)
    kp = torch.zeros((nb * page, Di), device=DEV)
    kp[:nkv] = k
    kpf, kps = fp4_cast(kp)
    for lb in range(nb):
        pb = perm[lb]
        vals[pb] = kpf[lb * page:(lb + 1) * page].view(torch.uint8)
        sfs[pb] = kps[lb * page:(lb + 1) * page].contiguous().view(torch.uint8).view(page, 4)
    nlog = (nkv + page - 1) // page
    bt = perm[:nlog].to(torch.int32).view(1, -1).expand(nq, -1).contiguous()
    return dict(q=(q_fp, q_sf), k=(k_fp, k_sf), w=w, ks=ks, ke=ke, cache=cache, bt=bt, nkv=nkv, page=page, gen=gen)


def logits_close(got, ref, valid, name):
    got, ref = got.float(), ref.float()
    assert torch.equal(torch.isneginf(got) & valid, torch.isneginf(ref) & valid), f"{name}: -inf pattern"
    fin = valid & torch.isfinite(ref)
    scale = ref.masked_fill(~fin, 0).abs().amax(1, keepdim=True) + 1e-3
    err = ((got - ref).abs() / scale).masked_fill(~fin, 0).max().item()
    assert err < 0.03, f"{name}: max row-relative err {err:.4f}"


def topk_overlap(a, b, k, end):
    ia = ds_topk(a, k, end=end, indices_type=torch.int32, sorted_index=True, idx_oob_fill_value=-1,
                 return_value=False)[1]
    ib = ds_topk(b, k, end=end, indices_type=torch.int32, sorted_index=True, idx_oob_fill_value=-1,
                 return_value=False)[1]
    hit = tot = 0
    for i in range(a.shape[0]):
        sa, sb = set(ia[i][ia[i] >= 0].tolist()), set(ib[i][ib[i] >= 0].tolist())
        hit += len(sa & sb)
        tot += max(len(sa), 1)
    return hit / tot


def run_indexer(fn, impl, *args, **kw):
    with EnvSet(SM120_INDEXER_IMPL=impl, SM120_ATTN_STRICT=1 if impl == "b12x" else 0):
        return DG[fn](*args, **kw)


@needs_cuda
@pytest.mark.parametrize("heads", [32, 16])
def test_indexer_b12x_vs_torch(heads):
    if not b12x_indexer_available():
        pytest.skip("b12x dsa_indexer not available")
    c = build_index_case(Hi=heads)
    nq, nkv = c["q"][0].shape[0], c["nkv"]
    ks, ke = c["ks"], c["ke"]
    cols = torch.arange(nkv, device=DEV).view(1, -1)
    # contiguous prefill logits
    W = int((ke - ks).max())
    res = {impl: run_indexer("fp8_fp4_mqa_logits", impl, c["q"], c["k"], c["w"], ks, ke, max_seqlen_k=W)
           for impl in ("torch", "b12x")}
    assert res["b12x"].shape == (nq, W) and res["b12x"].dtype == torch.bfloat16
    assert (res["b12x"].stride(0) * 2) % 1024 == 0
    valid = torch.arange(W, device=DEV).view(1, -1) < (ke - ks).view(-1, 1)
    logits_close(res["b12x"], res["torch"], valid, "fp8_fp4_mqa_logits")
    assert topk_overlap(res["b12x"], res["torch"], 128, (ke - ks)) > 0.97
    # paged decode logits
    ctx = ke.view(nq, 1).contiguous()
    ar = torch.arange(nq, dtype=torch.int32, device=DEV)
    meta = DG["get_paged_mqa_logits_metadata"](ctx, c["page"], 8, indices=ar)
    pres = {impl: run_indexer("fp8_fp4_paged_mqa_logits", impl, (c["q"][0].unsqueeze(1), c["q"][1].unsqueeze(1)),
                              c["cache"], c["w"], ctx, c["bt"], meta, nkv, indices=ar) for impl in ("torch", "b12x")}
    assert pres["b12x"].shape == (nq, nkv) and (pres["b12x"].stride(0) * 2) % 1024 == 0
    logits_close(pres["b12x"], pres["torch"], cols < ke.view(-1, 1), "fp8_fp4_paged_mqa_logits")
    assert topk_overlap(pres["b12x"], pres["torch"], 512, ke) > 0.97
    # sparse candidate blocks (prefill, contiguous K) and paged sparse (decode)
    sbk, nblk = 16, 24
    klen = ke - ks
    cand = torch.full((nq, nblk), -1, dtype=torch.int32, device=DEV)
    for i in range(nq):
        avail = (int(klen[i]) + sbk - 1) // sbk
        sel = torch.randperm(max(avail, 1), device=DEV, generator=c["gen"])[: min(avail, nblk)].sort().values
        cand[i, : sel.numel()] = sel.to(torch.int32)
    base = torch.div(ks, sbk, rounding_mode="floor").unsqueeze(1)
    smeta = DG["get_sparse_mqa_logits_metadata"](ks, ke, nkv, cand + base, c["q"][0].dtype, sbk, use_unaligned_ks=True)
    sres = {impl: run_indexer("fp8_fp4_sparse_mqa_logits", impl, c["q"], c["k"], c["w"], smeta, nblk, sbk,
                              use_unaligned_ks=True) for impl in ("torch", "b12x")}
    sv = torch.isfinite(sres["torch"].float()) | torch.isneginf(sres["torch"].float())
    logits_close(sres["b12x"], sres["torch"], sv, "fp8_fp4_sparse_mqa_logits")
    pmeta = DG["get_paged_sparse_mqa_logits_metadata"](ke.contiguous(), c["bt"], ar, c["page"], cand + base,
                                                       c["q"][0].dtype, sbk)
    qp = (c["q"][0].unsqueeze(1), c["q"][1].unsqueeze(1))
    psres = {impl: run_indexer("fp8_fp4_paged_sparse_mqa_logits", impl, qp, c["cache"], c["w"], pmeta, nblk, sbk)
             for impl in ("torch", "b12x")}
    nvalid = ((ke + sbk - 1) // sbk).clamp(max=nblk).view(-1, 1)
    pv = (torch.arange(nblk * sbk, device=DEV).view(1, -1) // sbk) < nvalid
    logits_close(psres["b12x"], psres["torch"], pv, "fp8_fp4_paged_sparse_mqa_logits")


@needs_cuda
def test_indexer_b12x_wide_and_chunked():
    """max_seqlen_k > 16384 (column chunks) and a small scratch budget (row chunks)."""
    if not b12x_indexer_available():
        pytest.skip("b12x dsa_indexer not available")
    c = build_index_case(seed=131, nq=200, nkv=20000)
    ks = torch.randint(0, 64, c["ks"].shape, device=DEV, generator=c["gen"]).to(torch.int32)
    ke = torch.randint(17000, 20001, ks.shape, device=DEV, generator=c["gen"]).to(torch.int32)
    W = 20480
    with EnvSet(SM120_INDEXER_SCRATCH_MB=16, SM120_INDEXER_FILL_TAIL=1):  # 16 MiB -> 64-row chunks
        res = {impl: run_indexer("fp8_fp4_mqa_logits", impl, c["q"], c["k"], c["w"], ks, ke, max_seqlen_k=W)
               for impl in ("torch", "b12x")}
        valid = torch.arange(W, device=DEV).view(1, -1) < (ke - ks).view(-1, 1)
        logits_close(res["b12x"], res["torch"], valid, "fp8_fp4_mqa_logits wide")
        assert torch.isneginf(res["b12x"][~valid].float()).all()


@needs_cuda
def test_indexer_fp32_weights_keep_torch_path():
    c = build_index_case(seed=141, weights_dtype=torch.float32)
    W = int((c["ke"] - c["ks"]).max())
    with EnvSet(SM120_INDEXER_IMPL="b12x", SM120_ATTN_STRICT=0):
        out = DG["fp8_fp4_mqa_logits"](c["q"], c["k"], c["w"], c["ks"], c["ke"], max_seqlen_k=W)
    assert out.dtype == torch.float32


# ---- CUDA graphs ---------------------------------------------------------------------------
def _graph_capture(fn):
    """DeepInfer CudaGraphRunner pattern: eager warm-up on the graph stream, then capture."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        out = fn()
    return g, out


@needs_cuda
@pytest.mark.parametrize("impl", ["b12x", "torch"])
def test_decode_path_cuda_graph(impl):
    """fused decode + paged indexer logits + deep_select.topk captured in one graph; replay after
    changing the static inputs must match an eager run."""
    if impl == "b12x" and not (b12x_available() and b12x_indexer_available()):
        pytest.skip("b12x not available")
    c = build_decode_case(seed=151, rows=8)
    ic = build_index_case(seed=152, nq=8, nkv=3000)
    rows = 8
    pos = torch.randint(0, 4096, (rows,), device=DEV).to(torch.int32)
    cs = cos_sin_cache()
    G = H // 8
    q_in = F_.q_to_fused_layout(c["q"]).contiguous()
    ctx = ic["ke"].view(rows, 1).contiguous()
    ar = torch.arange(rows, dtype=torch.int32, device=DEV)
    qp = (ic["q"][0].unsqueeze(1).contiguous(), ic["q"][1].unsqueeze(1).contiguous())
    meta = DG["get_paged_mqa_logits_metadata"](ctx, ic["page"], 8, indices=ar)

    def step():
        fp8, sf, _ = flash_mla.fused_norm_rope_attn_rope_cast.decode(
            True, 1e-4, pos, False, ROPE, cs, G, 32, True, True, True, q_in, c["swa"], c["swa_idx"], c["sm_scale"],
            D, c["sink"], None, c["main"], c["main_idx"], c["main_len"])
        logits = DG["fp8_fp4_paged_mqa_logits"](qp, ic["cache"], ic["w"], ctx, ic["bt"], meta, ic["nkv"], indices=ar)
        idx = ds_topk(logits, 512, end=ctx.view(-1), indices_type=torch.int32, sorted_index=True,
                      idx_oob_fill_value=-1, return_value=False)[1]
        return fp8, sf, logits, idx

    with EnvSet(SM120_ATTN_IMPL=impl, SM120_ATTN_STRICT=1 if impl == "b12x" else 0):
        g, (g_fp8, g_sf, g_log, g_idx) = _graph_capture(step)
        gen = torch.Generator(device=DEV).manual_seed(153)
        q_in.copy_(torch.randn(q_in.shape, device=DEV, generator=gen).mul(0.5).to(torch.bfloat16))
        c["swa_idx"].copy_(c["swa_idx"].flip(0))
        ctx.copy_((ctx - torch.randint(0, 500, ctx.shape, device=DEV, generator=gen)).clamp(min=1).to(torch.int32))
        qp[0].copy_(qp[0].flip(0))
        g.replay()
        torch.cuda.synchronize()
        e_fp8, e_sf, e_log, e_idx = step()
        assert torch.equal(g_fp8.view(torch.uint8), e_fp8.view(torch.uint8)) and torch.equal(g_sf, e_sf)
        valid = torch.arange(ic["nkv"], device=DEV).view(1, -1) < ctx.view(-1, 1)
        if impl == "b12x":  # same kernels, same launch shapes: bit-identical
            assert torch.equal(g_log.float()[valid], e_log.float()[valid])
            assert torch.equal(g_idx, e_idx)
        else:  # the captured torch path chunks by the static bound instead of max(ctx)
            logits_close(g_log, e_log, valid, "captured torch paged logits")
            assert topk_overlap(g_log, e_log, 512, ctx.view(-1)) > 0.99


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
