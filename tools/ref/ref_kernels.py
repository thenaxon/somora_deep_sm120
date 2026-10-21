"""Pure-torch re-implementations of ds41-inference/inference/kernel.py plus a per-function dispatcher.

`install(ref_src)` registers this module's dispatcher as `sys.modules["kernel"]` *before*
ds41 `model.py` is imported, so `from kernel import act_quant, ...` (model.py:12-19) binds to the
dispatcher. The official tilelang kernel.py is loaded under a private module name.

Environment
-----------
REF_KERNELS        = tilelang (default) | torch | check
                     tilelang: use the official tilelang kernels; if one raises (import, JIT compile,
                               launch), that function permanently falls back to torch (logged).
                     torch:    use only the torch implementations below (tilelang never imported).
                     check:    run BOTH on every call (torch on cloned inputs first), log/aggregate the
                               differences and continue with the tilelang result
                               (REF_CHECK_USE=torch continues with the torch result instead).
REF_KERNELS_TORCH  = comma list of functions forced to torch in tilelang/check mode (bisecting)
REF_KERNEL_SYNC    = 1 -> torch.cuda.synchronize() after every tilelang call so async CUDA faults
                     are attributed to the right kernel (slower).

Numerics: every torch function mirrors the tilelang kernel step by step (see the per-function
comments with kernel.py line numbers): fp32 amax, the same clamp floors, power-of-two scale via the
same IEEE bit trick (fast_log2_ceil/fast_pow2, kernel.py:22-37), RNE casts to e4m3 / e2m1, E8M0/E4M3
scale storage, fp32 accumulation, and for sparse_attn the blockwise (64) online softmax with the
probability tile cast to bf16 before the PV product (kernel.py:358-387). Remaining differences are
fp32 summation order inside GEMMs and possible fast-math exp/div in tilelang (ulp level).
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import math
import os
import sys
import types
from collections import defaultdict

import torch

log = logging.getLogger("ref_kernels")

FUNCS = ("act_quant", "fp4_act_quant", "fp8_gemm", "fp4_gemm", "sparse_attn", "hc_split_sinkhorn")

FP8_MAX = 448.0
FP4_MAX = 6.0


# ----------------------------------------------------------------------------------------------
# bit-exact helpers
# ----------------------------------------------------------------------------------------------
def e8m0_to_f32(s: torch.Tensor) -> torch.Tensor:
    """float8_e8m0fnu (or raw uint8 exponent) -> fp32, exactly: 2**(u-127); 0 -> 2**-127; 255 -> NaN."""
    if s.dtype == torch.float32:
        return s
    u = s.view(torch.uint8).to(torch.int32)
    bits = torch.where(u == 0, torch.full_like(u, 0x00400000), u << 23)
    bits = torch.where(u == 255, torch.full_like(u, 0x7FC00000), bits)
    return bits.view(torch.float32)


def f32_to_e8m0(s: torch.Tensor) -> torch.Tensor:
    """Power-of-two fp32 -> float8_e8m0fnu by taking the exponent field (exact for powers of two)."""
    bits = s.contiguous().view(torch.int32)
    e = (bits >> 23) & 0xFF
    return e.to(torch.uint8).view(torch.float8_e8m0fnu)


def scale_to_f32(s: torch.Tensor) -> torch.Tensor:
    if s.dtype == torch.float8_e8m0fnu:
        return e8m0_to_f32(s)
    return s.float()


def fast_round_scale(amax: torch.Tensor, max_inv: float) -> torch.Tensor:
    """kernel.py:22-37: 2**ceil(log2(amax * max_inv)) using the exponent/mantissa bits of the fp32 product."""
    t = amax.float() * torch.tensor(max_inv, dtype=torch.float32, device=amax.device)
    bits = t.contiguous().view(torch.int32)
    exp = (bits >> 23) & 0xFF
    man = bits & ((1 << 23) - 1)
    e = exp - 127 + (man != 0).to(torch.int32)
    return ((e + 127) << 23).to(torch.int32).view(torch.float32)


_FP4_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def round_to_e2m1(v: torch.Tensor) -> torch.Tensor:
    """fp32 -> nearest e2m1 value, ties to even mantissa (cvt.rn.satfinite.e2m1), returned as fp32."""
    a = v.abs()
    q = torch.full_like(a, 6.0)
    q = torch.where(a <= 5.0, 4.0, q)
    q = torch.where(a < 3.5, 3.0, q)
    q = torch.where(a <= 2.5, 2.0, q)
    q = torch.where(a < 1.75, 1.5, q)
    q = torch.where(a <= 1.25, 1.0, q)
    q = torch.where(a < 0.75, 0.5, q)
    q = torch.where(a <= 0.25, 0.0, q)
    q = torch.where(torch.isnan(v), v, q)
    return torch.copysign(q, v)


def e2m1_codes(q: torch.Tensor) -> torch.Tensor:
    """Grid values (from round_to_e2m1) -> 4-bit codes (sign in bit 3)."""
    a = q.abs()
    code = torch.zeros_like(a, dtype=torch.uint8)
    for i, g in enumerate(_FP4_GRID):
        code = torch.where(a == g, torch.tensor(i, dtype=torch.uint8, device=q.device), code)
    sign = torch.signbit(q).to(torch.uint8) << 3
    return code | sign


_E2M1_LUT_CACHE: dict = {}


def e2m1_lut(device) -> torch.Tensor:
    key = str(device)
    if key not in _E2M1_LUT_CACHE:
        vals = list(_FP4_GRID) + [-x for x in _FP4_GRID]
        _E2M1_LUT_CACHE[key] = torch.tensor(vals, dtype=torch.float32, device=device)
    return _E2M1_LUT_CACHE[key]


def as_u8(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8)


def unpack_fp4(b: torch.Tensor) -> torch.Tensor:
    """[N, K//2] float4_e2m1fn_x2 (or int8/uint8 bytes) -> [N, K] fp32. Low nibble = even element
    (convert.py:33-36 packing, which is also tilelang's FP4 buffer layout)."""
    u = as_u8(b)
    lut = e2m1_lut(b.device)
    lo = lut[(u & 0x0F).long()]
    hi = lut[(u >> 4).long()]
    return torch.stack([lo, hi], dim=-1).flatten(-2)


# ----------------------------------------------------------------------------------------------
# torch implementations (same signatures as kernel.py)
# ----------------------------------------------------------------------------------------------
def act_quant_torch(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    """kernel.py:40-124. amax per (row, block) in fp32, floor 1e-4; scale = amax/448 or 2**ceil(log2(.))
    when scale_fmt is set; y = e4m3(clamp(x/s)); inplace writes bf16(e4m3(...) * s) back into x."""
    N = x.size(-1)
    assert N % block_size == 0
    xf = x.float().unflatten(-1, (N // block_size, block_size))
    amax = xf.abs().amax(-1).clamp_min(1e-4)
    if scale_fmt is not None:
        s = fast_round_scale(amax, 1.0 / FP8_MAX)
    else:
        s = amax * torch.tensor(1.0 / FP8_MAX, dtype=torch.float32, device=x.device)
    q = (xf / s.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    if inplace:
        y = (q.float() * s.unsqueeze(-1)).flatten(-2).to(x.dtype)
        x.copy_(y)
        return x
    if scale_dtype == torch.float8_e8m0fnu:
        s_out = f32_to_e8m0(s)
    else:
        s_out = s.to(scale_dtype)
    return q.flatten(-2).contiguous(), s_out.contiguous()


def fp4_act_quant_torch(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    """kernel.py:127-204. E8M0: amax floor 6*2**-126, s = 2**ceil(log2(amax * (1/6))).
    E4M3: amax floor 6*2**-9, s = fp32(e4m3(amax / 6)). y = e2m1(clamp(x/s, -6, 6)) (RNE)."""
    assert scale_dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn)
    N = x.size(-1)
    assert N % block_size == 0
    xf = x.float().unflatten(-1, (N // block_size, block_size))
    amax = xf.abs().amax(-1)
    if scale_dtype == torch.float8_e4m3fn:
        amax = amax.clamp_min(6 * 2.0**-9)
        s = (amax / FP4_MAX).to(torch.float8_e4m3fn).float()
    else:
        amax = amax.clamp_min(6 * 2.0**-126)
        s = fast_round_scale(amax, 1.0 / FP4_MAX)
    q = round_to_e2m1((xf / s.unsqueeze(-1)).clamp(-FP4_MAX, FP4_MAX))
    if inplace:
        y = (q * s.unsqueeze(-1)).flatten(-2).to(x.dtype)
        x.copy_(y)
        return x
    codes = e2m1_codes(q.flatten(-2))
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous().view(torch.float4_e2m1fn_x2)
    s_out = f32_to_e8m0(s) if scale_dtype == torch.float8_e8m0fnu else s.to(torch.float8_e4m3fn)
    return packed, s_out.contiguous()


def _dequant_rows(a: torch.Tensor, a_s: torch.Tensor, group: int) -> torch.Tensor:
    K = a.size(-1)
    af = a.float().reshape(-1, K // group, group)
    sa = scale_to_f32(a_s).reshape(-1, K // group, 1)
    return (af * sa).reshape(-1, K)


def fp8_gemm_torch(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    """kernel.py:207-307. C = sum_k (A_k @ B_k^T) * sa[m,k] * sb[n//bs,k]. fp8 x fp8 products are exact
    in fp32 and the scales are powers of two, so dequantise-then-fp32-GEMM equals the kernel up to
    fp32 summation order. TF32 must be off (ref_runner sets it)."""
    K = a.size(-1)
    N = b.size(0)
    A = _dequant_rows(a, a_s, block_size)
    sb = scale_to_f32(b_s)  # [ceil(N/bs), K/bs]
    if N % block_size == 0:
        B = (b.float().view(N // block_size, block_size, K // block_size, block_size) * sb[:, None, :, None]).view(N, K)
    else:
        B = b.float() * sb.repeat_interleave(block_size, 0)[:N].repeat_interleave(block_size, 1)
    c = torch.matmul(A, B.t())
    return c.to(torch.get_default_dtype()).view(*a.shape[:-1], N)


def fp4_gemm_torch(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    """kernel.py:477-591. A fp8 with per-act_block scales, B packed e2m1 [N, K//2] with per-32 E8M0
    scales along K. fp8 x e2m1 products are exact in fp32."""
    K = a.size(-1)
    N = b.size(0)
    A = _dequant_rows(a, a_s, act_block_size)
    B = unpack_fp4(b)  # [N, K]
    assert B.size(-1) == K, (B.shape, K)
    sb = scale_to_f32(b_s)  # [N, K/32]
    B = (B.view(N, K // 32, 32) * sb[:, :, None]).view(N, K)
    c = torch.matmul(A, B.t())
    return c.to(torch.get_default_dtype()).view(*a.shape[:-1], N)


def sparse_attn_torch(q, kv, attn_sink, topk_idxs, softmax_scale, q_chunk: int = 32, block: int = 64):
    """kernel.py:310-403. Per query row: gather kv[idx] (zero where idx == -1), scores = q.k * scale in
    fp32 (-inf where idx == -1), online softmax over blocks of 64 slots in index order starting from
    a finite max of -1e30, P tile cast to bf16 before P@V (fp32 accumulate), denominator in fp32,
    attn_sink added as exp(sink - max) to the denominator only. Output bf16."""
    b, m, h, d = q.shape
    topk = topk_idxs.size(-1)
    out = torch.empty_like(q)
    sink = attn_sink.float()[:h]
    if topk == 0:
        out.zero_()
        return out
    for bi in range(b):
        kvb = kv[bi]
        for m0 in range(0, m, q_chunk):
            m1 = min(m, m0 + q_chunk)
            qc = q[bi, m0:m1].float()  # [mc, h, d]
            idx = topk_idxs[bi, m0:m1].long()  # [mc, topk]
            valid = idx != -1
            g = kvb[idx.clamp_min(0)]  # [mc, topk, d]
            g = g.masked_fill(~valid.unsqueeze(-1), 0).float()
            mc = m1 - m0
            mx = torch.full((mc, h), -1e30, dtype=torch.float32, device=q.device)
            se = torch.zeros((mc, h), dtype=torch.float32, device=q.device)
            acc = torch.zeros((mc, h, d), dtype=torch.float32, device=q.device)
            for t0 in range(0, topk, block):
                kb = g[:, t0 : t0 + block]
                vb = valid[:, t0 : t0 + block]
                s = torch.matmul(qc, kb.transpose(1, 2))  # [mc, h, blk]
                s = s.masked_fill(~vb.unsqueeze(1), -math.inf) * softmax_scale
                m_new = torch.maximum(mx, s.amax(-1))
                sc = torch.exp(mx - m_new)
                p = torch.exp(s - m_new.unsqueeze(-1))
                se = se * sc + p.sum(-1)
                acc = acc * sc.unsqueeze(-1) + torch.matmul(p.to(torch.bfloat16).float(), kb)
                mx = m_new
            se = se + torch.exp(sink.unsqueeze(0) - mx)
            out[bi, m0:m1] = (acc / se.unsqueeze(-1)).to(q.dtype)
    return out


def hc_split_sinkhorn_torch(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """kernel.py:406-474, same operation order: pre = sigmoid(.)+eps, post = 2*sigmoid(.),
    comb = row-softmax + eps, column normalise, then (iters-1) x (row, column) normalisation."""
    hc = hc_mult
    m = mixes.float()
    sc = hc_scale.float()
    base = hc_base.float()
    pre = torch.sigmoid(m[..., :hc] * sc[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc : 2 * hc] * sc[1] + base[hc : 2 * hc])
    comb = (m[..., 2 * hc :] * sc[2] + base[2 * hc :]).unflatten(-1, (hc, hc))
    comb = torch.exp(comb - comb.amax(-1, keepdim=True))
    comb = comb / comb.sum(-1, keepdim=True) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre.contiguous(), post.contiguous(), comb.contiguous()


TORCH_IMPLS = {
    "act_quant": act_quant_torch,
    "fp4_act_quant": fp4_act_quant_torch,
    "fp8_gemm": fp8_gemm_torch,
    "fp4_gemm": fp4_gemm_torch,
    "sparse_attn": sparse_attn_torch,
    "hc_split_sinkhorn": hc_split_sinkhorn_torch,
}


# ----------------------------------------------------------------------------------------------
# dispatcher
# ----------------------------------------------------------------------------------------------
def _to_cmp(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.float8_e8m0fnu:
        return e8m0_to_f32(t)
    if t.dtype == torch.float4_e2m1fn_x2:
        return unpack_fp4(t)
    return t.float()


def _clone_tree(x):
    if isinstance(x, torch.Tensor):
        return x.clone()
    if isinstance(x, (list, tuple)):
        return type(x)(_clone_tree(v) for v in x)
    if isinstance(x, dict):
        return {k: _clone_tree(v) for k, v in x.items()}
    return x


class Dispatcher:
    def __init__(self):
        self.mode = os.environ.get("REF_KERNELS", "tilelang").strip().lower()
        if self.mode not in ("tilelang", "torch", "check"):
            raise ValueError(f"REF_KERNELS must be tilelang|torch|check, got {self.mode!r}")
        forced = {f.strip() for f in os.environ.get("REF_KERNELS_TORCH", "").split(",") if f.strip()}
        unknown = forced - set(FUNCS)
        if unknown:
            raise ValueError(f"REF_KERNELS_TORCH: unknown functions {sorted(unknown)}")
        self.check_use_torch = os.environ.get("REF_CHECK_USE", "tilelang").strip().lower() == "torch"
        self.sync = os.environ.get("REF_KERNEL_SYNC", "0") == "1"
        self.backend = {f: ("torch" if self.mode == "torch" or f in forced else "tilelang") for f in FUNCS}
        self.forced = sorted(forced)
        self.tl = None
        self.tl_path = None
        self.tl_import_error = None
        self.fallbacks: dict[str, str] = {}
        self.calls = defaultdict(lambda: defaultdict(int))
        self.check = defaultdict(lambda: {"calls": 0, "bitexact": 0, "max_abs": 0.0, "max_rel": 0.0, "worst_frac_mismatch": 0.0})
        self._sigs = {f: inspect.signature(TORCH_IMPLS[f]) for f in FUNCS}

    # -- tilelang loading
    def load_tilelang(self, kernel_path: str):
        self.tl_path = kernel_path
        if all(b == "torch" for b in self.backend.values()):
            return
        try:
            spec = importlib.util.spec_from_file_location("ds41_tilelang_kernel", kernel_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.tl = mod
            try:
                import tilelang

                log.info(f"tilelang {getattr(tilelang, '__version__', '?')} loaded from {kernel_path}")
            except Exception:
                pass
        except Exception as e:  # tilelang missing / API drift between 0.1.12 and 0.1.15
            self.tl_import_error = f"{type(e).__name__}: {e}"
            log.warning(f"tilelang kernel.py import failed -> ALL functions use torch: {self.tl_import_error}")
            for f in FUNCS:
                if self.backend[f] == "tilelang":
                    self.backend[f] = "torch"
                    self.fallbacks[f] = f"import: {self.tl_import_error}"

    def _fallback(self, name, exc):
        msg = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
        log.warning(f"tilelang {name} failed -> torch from now on: {msg}")
        self.backend[name] = "torch"
        self.fallbacks[name] = msg
        if "CUDA error" in msg or "illegal memory" in msg.lower():
            log.error("this looks like a CUDA fault; the CUDA context is probably poisoned -- rerun with "
                      f"REF_KERNELS_TORCH={name} (or REF_KERNELS=torch)")

    def _compare(self, name, ref_out, tl_out):
        refs = ref_out if isinstance(ref_out, tuple) else (ref_out,)
        tls = tl_out if isinstance(tl_out, tuple) else (tl_out,)
        st = self.check[name]
        st["calls"] += 1
        exact = True
        for r, t in zip(refs, tls):
            if r.dtype != t.dtype or r.shape != t.shape:
                log.error(f"check {name}: dtype/shape differ torch={r.dtype}{tuple(r.shape)} tl={t.dtype}{tuple(t.shape)}")
                exact = False
                continue
            rf, tf = _to_cmp(r), _to_cmp(t)
            diff = (rf - tf).abs()
            both_nan = torch.isnan(rf) & torch.isnan(tf)
            diff = torch.where(both_nan, torch.zeros_like(diff), diff)
            mx = float(diff.max()) if diff.numel() else 0.0
            if mx != 0.0 or torch.isnan(diff).any():
                exact = False
            denom = rf.abs().max().clamp_min(1e-30)
            st["max_abs"] = max(st["max_abs"], mx)
            st["max_rel"] = max(st["max_rel"], float(mx / denom) if diff.numel() else 0.0)
            frac = float((diff != 0).float().mean()) if diff.numel() else 0.0
            st["worst_frac_mismatch"] = max(st["worst_frac_mismatch"], frac)
        st["bitexact"] += int(exact)

    def call(self, name, args, kwargs):
        torch_fn = TORCH_IMPLS[name]
        if self.backend[name] == "torch":
            self.calls[name]["torch"] += 1
            return torch_fn(*args, **kwargs)
        tl_fn = getattr(self.tl, name)
        bound = self._sigs[name].bind(*args, **kwargs)
        inplace = bool(bound.arguments.get("inplace", False))
        if self.mode == "check":
            t_args = _clone_tree(args)
            t_kwargs = _clone_tree(kwargs)
            ref_out = torch_fn(*t_args, **t_kwargs)
            ref_cmp = t_args[0] if inplace else ref_out
            try:
                out = tl_fn(*args, **kwargs)
                torch.cuda.synchronize()
            except Exception as e:
                self._fallback(name, e)
                # torch result was computed on clones: replay it on the real inputs
                self.calls[name]["torch"] += 1
                return torch_fn(*args, **kwargs)
            self.calls[name]["check"] += 1
            self._compare(name, ref_cmp, args[0] if inplace else out)
            if self.check_use_torch:
                if inplace:
                    args[0].copy_(t_args[0])
                    return args[0]
                return ref_out
            return out
        try:
            out = tl_fn(*args, **kwargs)
            if self.sync:
                torch.cuda.synchronize()
        except Exception as e:
            self._fallback(name, e)
            self.calls[name]["torch"] += 1
            return torch_fn(*args, **kwargs)
        self.calls[name]["tilelang"] += 1
        return out

    def report(self) -> dict:
        return {
            "mode": self.mode,
            "forced_torch": self.forced,
            "backend_final": dict(self.backend),
            "tilelang_kernel_path": self.tl_path,
            "tilelang_import_error": self.tl_import_error,
            "fallbacks": dict(self.fallbacks),
            "calls": {k: dict(v) for k, v in self.calls.items()},
            "check": {k: dict(v) for k, v in self.check.items()},
        }


DISPATCH: Dispatcher | None = None


def _make_entry(name):
    def entry(*args, **kwargs):
        return DISPATCH.call(name, args, kwargs)

    entry.__name__ = name
    entry.__doc__ = TORCH_IMPLS[name].__doc__
    return entry


def _isprime(n: int) -> bool:  # only used if sympy is absent (engram.py imports it; unused with engram off)
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


def install(ref_src: str) -> Dispatcher:
    """Register the dispatcher as module `kernel`, stub the missing `image_processor` module, then
    put ref_src on sys.path so model.py / engram.py / vision.py import from there."""
    global DISPATCH
    DISPATCH = Dispatcher()
    kmod = types.ModuleType("kernel")
    kmod.__file__ = __file__
    for f in FUNCS:
        setattr(kmod, f, _make_entry(f))
    sys.modules["kernel"] = kmod
    DISPATCH.load_tilelang(os.path.join(ref_src, "kernel.py"))

    # model.py:11 and generate.py:25 import image_processor, which is not part of the published repo.
    # Only the token-type constants are referenced by model.py (merge_image_embeddings, never reached
    # for text-only input). TEXT = -1 per model.py:1252 ("TEXT is -1").
    if "image_processor" not in sys.modules and not os.path.exists(os.path.join(ref_src, "image_processor.py")):
        ip = types.ModuleType("image_processor")
        ip.TEXT, ip.IMAGE, ip.IMAGE_START, ip.IMAGE_END, ip.IMAGE_NEW_LINE = -1, 0, 1, 2, 3

        def _no_vl(*a, **k):
            raise NotImplementedError("image_processor stub: vision input is not supported by this runner")

        ip.prepare_vl_inputs = _no_vl
        sys.modules["image_processor"] = ip
    try:
        import sympy  # noqa: F401
    except ImportError:
        sm = types.ModuleType("sympy")
        sm.isprime = _isprime
        sys.modules["sympy"] = sm
        log.warning("sympy not installed: using a stub isprime (only engram uses it; engram is disabled)")
    if ref_src not in sys.path:
        sys.path.insert(0, ref_src)
    log.info(f"REF_KERNELS={DISPATCH.mode} backends={DISPATCH.backend}")
    return DISPATCH
