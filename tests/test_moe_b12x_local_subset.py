#!/usr/bin/env python3
"""Feasibility test: b12x 1.5.1 fused_moe (w4a8_mx) on a LOCAL EXPERT SUBSET
with non-local routes masked as expert id -1 (DeepInfer EP=4 "Option B").

Question answered:
  Does b12x fused_moe, prepared for only 96 of 384 experts, with topk_ids where
  non-local routes are -1, produce exactly the partial sum over local experts?
  And do 4 such partials (experts 0-95, 96-191, 192-287, 288-383) sum to the
  full 384-expert result?

Inputs: real DeepSeek V4 layer-2 routed experts from the HF checkpoint
(MXFP4 e2m1 nibbles + UE8M0 per-32 scales), random hidden states and random
top-k routing.  Only torch + safetensors + b12x are imported (no DeepInfer).

Stages
  0 env/config        read /ckpt/config.json (text_config), print geometry
  1 load              load all 384 experts of layer L (raw, never mutated)
                      + self-check of our FP4/UE8M0 dequant vs b12x helper
  2 reference         fp32 torch reference per rank partial; variants
                      fp32 (clamped), mx8 (clamped + MXFP8 activation
                      emulation), fp32_noclamp; cross-check vs b12x
                      moe_reference_w4a8_mx oracle
  3 full control      b12x on all 384 experts, unmasked ids, vs full ref
  4 rank subsets      b12x on each 96-expert subset with -1 masked ids vs
                      rank-partial ref; all-non-local tokens == 0; all -1
                      batches == 0; NaN-sentinel output contract; clamp probe
  5 sum check         sum of 4 partials vs full b12x run and vs full ref
  Path variants (stage 3 and rank 0 of stage 4): default heuristic path,
  B12X_W4A8_TINY_DECODE=0 (tiny-decode "micro" -> dynamic) for T<=4, and the
  flipped dynamic route mode (direct <-> grouped) via an explicit
  MoeDecodeConfig.

Output: [PASS]/[FAIL]/[INFO] lines, final "SUMMARY:" line; exit 0 iff no FAIL.

Usage (inside the container):
  PYTHONPATH=/opt/deepinfer python3 test_moe_b12x_local_subset.py --ckpt /ckpt
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import os
import re
import sys
import time
import traceback

os.environ.setdefault("B12X_AUTOTUNE", "0")  # b12x/preparation/session.py:271

import torch  # noqa: E402

# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
RESULTS: list[tuple[str, str, str]] = []
COS_THR_REF = 0.995      # b12x vs torch reference (FP8 activation quantization)
COS_THR_SUM = 0.9999     # masked partial sum vs full b12x run
ZERO_TOL = 1e-6


def report(status: str, name: str, msg: str = "") -> None:
    RESULTS.append((status, name, msg))
    print(f"[{status}] {name}: {msg}", flush=True)


def info(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def metrics(out: torch.Tensor, ref: torch.Tensor) -> dict:
    o = out.detach().double().reshape(-1)
    r = ref.detach().double().reshape(-1).to(o.device)
    finite = bool(torch.isfinite(o).all().item())
    o = torch.nan_to_num(o, nan=0.0, posinf=0.0, neginf=0.0)
    d = o - r
    rn = r.norm().item()
    on = o.norm().item()
    if rn > 0 and on > 0:
        cos = (o @ r).item() / (rn * on)
    elif rn == 0 and on == 0:
        cos = 1.0
    else:
        cos = 0.0
    dn = d.norm().item()
    return {
        "cos": cos,
        "max_abs": d.abs().max().item() if d.numel() else 0.0,
        "ref_max": r.abs().max().item() if r.numel() else 0.0,
        "rel_l2": (dn / rn) if rn > 0 else (0.0 if dn == 0 else float("inf")),
        "ref_norm": rn,
        "finite": finite,
    }


def fmt(m: dict) -> str:
    return (
        f"cos={m['cos']:.6f} max_abs={m['max_abs']:.4e} ref_max={m['ref_max']:.4e} "
        f"rel_l2={m['rel_l2']:.4e} finite={m['finite']}"
    )


def check(name: str, out: torch.Tensor, ref: torch.Tensor, thr: float, extra: str = "") -> dict:
    m = metrics(out, ref)
    if m["ref_norm"] > 0:
        ok = m["finite"] and m["cos"] > thr
    else:
        ok = m["finite"] and m["max_abs"] <= ZERO_TOL
    report("PASS" if ok else "FAIL", name, fmt(m) + (f" thr={thr}" if m["ref_norm"] > 0 else " (ref==0)") + extra)
    return m


def run_stage(name: str, fn, *args, **kwargs):
    print(f"\n===================== STAGE {name} =====================", flush=True)
    t0 = time.time()
    try:
        result = fn(*args, **kwargs)
        info(f"stage {name} finished in {time.time() - t0:.1f}s")
        return result
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        report("FAIL", f"stage:{name}", f"exception {type(exc).__name__}: {exc}")
        return None


@contextlib.contextmanager
def temp_env(env: dict | None):
    env = env or {}
    saved = {k: os.environ.get(k) for k in env}
    try:
        for k, v in env.items():
            os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def mem(tag: str) -> None:
    if torch.cuda.is_available():
        info(
            f"mem[{tag}] allocated={torch.cuda.memory_allocated() / 2**30:.2f}GiB "
            f"reserved={torch.cuda.memory_reserved() / 2**30:.2f}GiB"
        )


# --------------------------------------------------------------------------
# config / checkpoint
# --------------------------------------------------------------------------
@dataclasses.dataclass
class Geo:
    hidden: int
    inter: int
    n_experts: int
    topk: int
    swiglu_limit: float | None
    scoring: str | None
    hidden_act: str | None


def load_config(ckpt: str) -> Geo:
    with open(os.path.join(ckpt, "config.json")) as f:
        cfg = json.load(f)
    text = cfg.get("text_config", cfg)

    def get(*keys, default=None):
        for k in keys:
            for d in (text, cfg):
                if isinstance(d, dict) and d.get(k) is not None:
                    return d[k]
        return default

    limit = get("swiglu_limit", "swiglu_clamp_value", "swiglu_clamp", "clamp_limit")
    geo = Geo(
        hidden=int(get("hidden_size")),
        inter=int(get("moe_intermediate_size")),
        n_experts=int(get("n_routed_experts", "num_experts")),
        topk=int(get("num_experts_per_tok")),
        swiglu_limit=None if limit is None else float(limit),
        scoring=get("scoring_func"),
        hidden_act=get("hidden_act"),
    )
    info(f"config: {geo}")
    q = cfg.get("quantization_config") or text.get("quantization_config")
    if q:
        info(f"quantization_config: {json.dumps(q)[:400]}")
    return geo


_SAFE_DTYPE_BYTES = {"I8": 1, "U8": 1, "F8_E8M0": 1, "F8_E4M3": 1, "F4": 1}


def _raw_read_u8(path: str, name: str) -> torch.Tensor:
    """Fallback reader for 1-byte dtypes if safetensors cannot decode the dtype."""
    with open(path, "rb") as f:
        hlen = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(hlen))
        meta = header[name]
        if meta["dtype"] not in _SAFE_DTYPE_BYTES:
            raise TypeError(f"raw reader supports only 1-byte dtypes, got {meta['dtype']}")
        start, end = meta["data_offsets"]
        f.seek(8 + hlen + start)
        buf = bytearray(f.read(end - start))
    t = torch.frombuffer(buf, dtype=torch.uint8)
    return t.reshape(meta["shape"])


def load_layer_experts(ckpt: str, layer: int, geo: Geo, device: torch.device):
    """Return raw uint8 tensors for all routed experts of one layer.

    w13: [E, 2n, K/2]  rows [0,n)=w1 (gate), [n,2n)=w3 (up)   -> b12x w13_layout="w31"
    s13: [E, 2n, K/32] UE8M0 bytes
    w2 : [E, K, n/2]
    s2 : [E, K, n/32]
    """
    from safetensors import safe_open

    with open(os.path.join(ckpt, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    pat = re.compile(
        rf"^(?:model\.)?layers\.{layer}\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$"
    )
    names: dict[tuple[int, str, str], tuple[str, str]] = {}
    for name, shard in weight_map.items():
        m = pat.match(name)
        if m:
            names[(int(m.group(1)), m.group(2), m.group(3))] = (name, shard)
    E, K, n = geo.n_experts, geo.hidden, geo.inter
    missing = [
        (e, w, k) for e in range(E) for w in ("w1", "w2", "w3") for k in ("weight", "scale")
        if (e, w, k) not in names
    ]
    if missing:
        raise RuntimeError(f"layer {layer}: {len(missing)} expert tensors missing, e.g. {missing[:4]}")
    shards = sorted({s for (_, s) in names.values()})
    info(f"layer {layer}: {len(names)} expert tensors in shards {shards}")

    w13 = torch.empty((E, 2 * n, K // 2), dtype=torch.uint8, device=device)
    s13 = torch.empty((E, 2 * n, K // 32), dtype=torch.uint8, device=device)
    w2 = torch.empty((E, K, n // 2), dtype=torch.uint8, device=device)
    s2 = torch.empty((E, K, n // 32), dtype=torch.uint8, device=device)

    by_shard: dict[str, list[tuple[int, str, str, str]]] = {}
    for (e, w, k), (name, shard) in names.items():
        if e < E:
            by_shard.setdefault(shard, []).append((e, w, k, name))
    dtypes_seen: set[str] = set()
    for shard, items in by_shard.items():
        path = os.path.join(ckpt, shard)
        with safe_open(path, framework="pt", device=str(device)) as f:
            for e, w, k, name in items:
                try:
                    t = f.get_tensor(name)
                    dtypes_seen.add(str(t.dtype))
                    if t.element_size() != 1:
                        raise TypeError(f"{name}: unexpected element size {t.element_size()} ({t.dtype})")
                    t = t.view(torch.uint8)
                except (TypeError, RuntimeError, ValueError) as exc:
                    if "element size" in str(exc):
                        raise
                    t = _raw_read_u8(path, name).to(device)
                    dtypes_seen.add("raw-u8-fallback")
                if w in ("w1", "w3"):
                    rows = slice(0, n) if w == "w1" else slice(n, 2 * n)
                    dst = (w13 if k == "weight" else s13)[e, rows]
                else:
                    dst = (w2 if k == "weight" else s2)[e]
                if tuple(t.shape) != tuple(dst.shape):
                    raise RuntimeError(f"{name}: shape {tuple(t.shape)} != expected {tuple(dst.shape)}")
                dst.copy_(t)
    torch.cuda.synchronize()
    info(f"checkpoint tensor dtypes seen: {sorted(dtypes_seen)}")
    return w13, s13, w2, s2


# --------------------------------------------------------------------------
# torch reference
# --------------------------------------------------------------------------
_FP4_LUT_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                   -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def dequant_mxfp4(packed: torch.Tensor, scale_u8: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """packed uint8 [R, C/2] (low nibble = even column), UE8M0 [R, C/32] -> fp32 [R, C]."""
    R, half = packed.shape
    C = half * 2
    p = packed.view(torch.uint8)
    lo = (p & 0x0F).long()
    hi = (p >> 4).long()
    vals = torch.stack((lut[lo], lut[hi]), dim=-1).reshape(R, C)
    sc = torch.exp2(scale_u8.view(torch.uint8).float() - 127.0)
    return (vals.view(R, C // 32, 32) * sc[:, : C // 32].unsqueeze(-1)).reshape(R, C)


@dataclasses.dataclass
class Case:
    name: str
    T: int
    scale: float
    x: torch.Tensor          # bf16 [T, K]
    ids: torch.Tensor        # int32 [T, topk] global expert ids
    w: torch.Tensor          # fp32 [T, topk]
    probe: bool = False
    refs: dict | None = None  # variant -> fp32 [ranks, T, K]
    clamp_frac: float = 0.0


def make_routing(T: int, E: int, topk: int, epr: int, ranks: int, gen: torch.Generator,
                 want_all_nonlocal_r0: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Random top-k over E experts.  Resample (bounded) so that rank 0 has >=1
    local route, and (for T>=7) >=1 token whose routes are all non-local for
    rank 0; for T<=8 also require every rank to own >=1 route."""
    best = None
    for _ in range(2000):
        scores = torch.randn((T, E), generator=gen)
        top, ids = scores.topk(topk, dim=-1)
        rank_of = ids // epr
        r0_local = (rank_of == 0)
        ok = bool(r0_local.any())
        if want_all_nonlocal_r0:
            ok = ok and bool((~r0_local.any(dim=1)).any())
        if T <= 8:
            ok = ok and all(bool((rank_of == r).any()) for r in range(ranks))
        best = (ids, top)
        if ok:
            break
    ids, top = best
    w = torch.softmax(top.float(), dim=-1)
    return ids.to(torch.int32), w


def build_cases(geo: Geo, tokens: list[int], scale: float, probe_scale: float,
                probe_tokens: list[int], ranks: int, epr: int, device, seed: int) -> list[Case]:
    cases = []
    gen = torch.Generator().manual_seed(seed)
    for T in tokens:
        x = (torch.randn((T, geo.hidden), generator=gen) * scale).to(torch.bfloat16)
        ids, w = make_routing(T, geo.n_experts, geo.topk, epr, ranks, gen, want_all_nonlocal_r0=T >= 7)
        cases.append(Case(f"T{T}", T, scale, x.to(device), ids.to(device).contiguous(), w.to(device).contiguous()))
    for T in probe_tokens:
        x = (torch.randn((T, geo.hidden), generator=gen) * probe_scale).to(torch.bfloat16)
        ids, w = make_routing(T, geo.n_experts, geo.topk, epr, ranks, gen, want_all_nonlocal_r0=False)
        cases.append(Case(f"probeT{T}_s{probe_scale:g}", T, probe_scale, x.to(device),
                          ids.to(device).contiguous(), w.to(device).contiguous(), probe=True))
    for c in cases:
        r0 = (c.ids // epr == 0)
        info(f"case {c.name}: T={c.T} scale={c.scale} routes={c.ids.numel()} rank0_local_routes={int(r0.sum())} "
             f"rank0_all_nonlocal_tokens={int((~r0.any(dim=1)).sum())}")
    return cases


def compute_references(cases: list[Case], raw, geo: Geo, ranks: int, epr: int, mx8_fn) -> None:
    w13, s13, w2, s2 = raw
    n, K, L = geo.inter, geo.hidden, geo.swiglu_limit
    device = w13.device
    lut = torch.tensor(_FP4_LUT_VALUES, dtype=torch.float32, device=device)
    variants = ["fp32", "fp32_noclamp"] + (["mx8"] if mx8_fn is not None else [])
    x32 = {}
    xq = {}
    clamp_hits = {c.name: 0 for c in cases}
    clamp_tot = {c.name: 0 for c in cases}
    for c in cases:
        c.refs = {v: torch.zeros((ranks, c.T, K), dtype=torch.float32, device=device) for v in variants}
        x32[c.name] = c.x.float()
        if mx8_fn is not None:
            xq[c.name] = mx8_fn(x32[c.name])
    E_total = ranks * epr
    for e in range(E_total):
        r = e // epr
        W13 = None
        for c in cases:
            tok, slot = (c.ids == e).nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            if W13 is None:
                W13 = dequant_mxfp4(w13[e], s13[e], lut)   # [2n, K]
                W2 = dequant_mxfp4(w2[e], s2[e], lut)      # [K, n]
            wt = c.w[tok, slot].unsqueeze(1)
            for v in variants:
                xin = (xq[c.name] if v == "mx8" else x32[c.name])[tok]
                h = xin @ W13.T
                g, u = h[:, :n], h[:, n:]
                if v == "fp32" and L is not None:
                    clamp_hits[c.name] += int((g > L).sum()) + int((u.abs() > L).sum())
                    clamp_tot[c.name] += 2 * g.numel()
                if v != "fp32_noclamp" and L is not None:
                    g = g.clamp(max=L)
                    u = u.clamp(min=-L, max=L)
                act = torch.nn.functional.silu(g) * u
                if v == "mx8":
                    act = mx8_fn(act)
                y = act @ W2.T
                c.refs[v][r].index_add_(0, tok, y * wt)
    for c in cases:
        c.clamp_frac = clamp_hits[c.name] / max(1, clamp_tot[c.name])
        full = c.refs["fp32"].sum(0)
        info(f"ref {c.name}: |ref| max={full.abs().max().item():.4e} rms={full.pow(2).mean().sqrt().item():.4e} "
             f"swiglu clamp fraction={c.clamp_frac:.3e} (limit={L})")
    torch.cuda.synchronize()


# --------------------------------------------------------------------------
# b12x runner (tensor-based API: plan_weights(quant_modes=..), prepare_weights(tensors),
# plan(Caps(..)), bind, run)
# --------------------------------------------------------------------------
class B12xRunner:
    def __init__(self, fm, label: str, w13, s13, w2, s2, geo: Geo, device):
        self.fm = fm
        self.label = label
        self.geo = geo
        self.device = device
        E = int(w13.shape[0])
        self.E = E
        self.wplan = fm.plan_weights(
            quant_modes="w4a8_mx",
            source_format="fp4_e8m0_k32",
            activation="silu",
            params_dtype=torch.bfloat16,
            num_experts=E,
            hidden_size=geo.hidden,
            intermediate_size=geo.inter,
            w13_layout="w31",
        )
        ones = lambda: torch.ones(E, dtype=torch.float32, device=device)  # noqa: E731
        t0 = time.time()
        # NOTE: w4a8_mx preparation repacks w13/w2 and the scale grids IN PLACE
        # (_impl.py _logical_weight_to_w4a8_rp_inplace); callers pass clones.
        self.experts = fm.prepare_weights(
            plan=self.wplan,
            params_dtype=torch.bfloat16,
            w1_fp4=w13,
            w2_fp4=w2,
            w1_global_scale=ones(),
            w2_global_scale=ones(),
            w1_blockscale=s13,
            w2_blockscale=s2,
            a1_gscale=ones(),
            a2_gscale=ones(),
        )
        torch.cuda.synchronize()
        info(f"[{label}] prepare_weights(E={E}) done in {time.time() - t0:.1f}s; "
             f"transforms={sorted(str(t) for t in getattr(self.wplan, 'transforms', ()))}")
        self._plans: dict = {}

    def _caps(self, T: int, decode_config, env):
        with temp_env(env):
            return self.fm.Caps(
                max_tokens=T,
                num_topk=self.geo.topk,
                device=self.device,
                weight_plan=self.wplan,
                quant_mode="w4a8_mx",
                swiglu_limit=self.geo.swiglu_limit,
                decode_config=decode_config,
            )

    def decode_config(self, T: int, env=None):
        return self._caps(T, None, env).decode_config

    def plan(self, T: int, vname: str, decode_config=None, env=None):
        key = (T, vname)
        if key not in self._plans:
            t0 = time.time()
            caps = self._caps(T, decode_config, env)
            with temp_env(env):
                plan = self.fm.plan(caps)
            scratch = tuple(
                torch.zeros(spec.shape, dtype=spec.dtype, device=spec.device)
                for spec in plan.scratch_specs()
            )
            lp = plan.launch_plan
            info(f"[{self.label}] plan T={T} variant={vname}: impl={lp.implementation} "
                 f"decode_config={caps.decode_config} max_tokens_per_launch={lp.max_tokens_per_launch} "
                 f"scratch={sum(s.nbytes for s in plan.scratch_specs()) / 2**20:.1f}MiB "
                 f"swiglu_limit={caps.swiglu_limit} ({time.time() - t0:.1f}s)")
            self._plans[key] = (plan, scratch, dict(env or {}))
        return self._plans[key]

    def run(self, T: int, vname: str, x, ids, w, *, prefill: float = 0.0, decode_config=None, env=None):
        plan, scratch, penv = self.plan(T, vname, decode_config, env)
        out = torch.full((x.shape[0], self.geo.hidden), prefill, dtype=torch.bfloat16, device=self.device)
        lp = plan.launch_plan
        chunk = x.shape[0]
        if lp.implementation == "dynamic":
            chunk = max(1, min(chunk, int(lp.max_tokens_per_launch)))
        t0 = time.time()
        with temp_env(penv):
            for i in range(0, x.shape[0], chunk):
                j = min(x.shape[0], i + chunk)
                binding = self.fm.bind(
                    plan,
                    scratch=scratch,
                    a=x[i:j],
                    experts=self.experts,
                    topk_weights=w[i:j],
                    topk_ids=ids[i:j],
                    output=out[i:j],
                    input_scales_static=True,
                )
                self.fm.run(binding=binding)
        torch.cuda.synchronize()
        dt = time.time() - t0
        if chunk < x.shape[0]:
            info(f"[{self.label}] T={T} {vname}: ran in {math.ceil(x.shape[0] / chunk)} chunks of {chunk}")
        return out, dt

    def variants(self, T: int):
        """[(vname, decode_config|None, env)] : default, tiny-decode off, flipped route mode."""
        out = []
        dc0 = self.decode_config(T)
        out.append(("default", None, {}))
        dyn_dc, dyn_env = dc0, {}
        if dc0.backend == "micro":
            env = {"B12X_W4A8_TINY_DECODE": "0"}  # _impl.py:12313
            dyn_dc = self.decode_config(T, env)
            dyn_env = env
            out.append(("tiny_decode_off", None, env))
        if dyn_dc.backend == "dynamic" and dyn_dc.dynamic_route_mode in ("direct", "grouped"):
            flip = "grouped" if dyn_dc.dynamic_route_mode == "direct" else "direct"
            prefix = "tiny_off+" if dyn_env else ""
            out.append((f"{prefix}route_{flip}", dataclasses.replace(dyn_dc, dynamic_route_mode=flip), dyn_env))
        return out


def local_ids(ids: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    return torch.where((ids >= lo) & (ids < hi), ids - lo, torch.full_like(ids, -1)).to(torch.int32).contiguous()


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------
def stage_dequant_selfcheck(raw, geo: Geo):
    w13, s13, w2, s2 = raw
    lut = torch.tensor(_FP4_LUT_VALUES, dtype=torch.float32, device=w13.device)
    try:
        from b12x.moe._shared.kernels.reference import _dequant_w4a8_weight_e8m0_k32, _make_fp4_lut
    except Exception as exc:  # noqa: BLE001
        report("INFO", "dequant_vs_b12x_helper", f"b12x reference helper unavailable: {exc}")
        return
    blut = _make_fp4_lut(w13.device)
    for e in sorted({0, geo.n_experts // 2, geo.n_experts - 1}):
        mine = dequant_mxfp4(w13[e], s13[e], lut)
        theirs = _dequant_w4a8_weight_e8m0_k32(w13[e], s13[e], 2 * geo.inter, geo.hidden, blut).reshape(2 * geo.inter, geo.hidden)
        mine2 = dequant_mxfp4(w2[e], s2[e], lut)
        theirs2 = _dequant_w4a8_weight_e8m0_k32(w2[e], s2[e], geo.hidden, geo.inter, blut).reshape(geo.hidden, geo.inter)
        ok = torch.equal(mine, theirs) and torch.equal(mine2, theirs2)
        report("PASS" if ok else "FAIL", f"dequant_vs_b12x_helper[e{e}]",
               f"w13 std={mine.std().item():.4e} absmax={mine.abs().max().item():.4e}; "
               f"w2 std={mine2.std().item():.4e}; scale byte range w13=[{int(s13[e].min())},{int(s13[e].max())}]")


def stage_oracle_crosscheck(cases, raw, geo, epr):
    try:
        from b12x.moe._shared.kernels.reference import moe_reference_w4a8_mx
    except Exception as exc:  # noqa: BLE001
        report("INFO", "oracle_crosscheck", f"moe_reference_w4a8_mx unavailable: {exc}")
        return
    if "mx8" not in cases[0].refs:
        report("INFO", "oracle_crosscheck", "mx8 reference variant unavailable; skipped")
        return
    w13, s13, w2, s2 = raw
    c = next((c for c in cases if not c.probe and 4 <= c.T <= 64), None) or cases[0]
    E = epr
    ones = torch.ones(E, dtype=torch.float32, device=w13.device)
    lid = local_ids(c.ids, 0, E)
    oracle = moe_reference_w4a8_mx(
        c.x, w13[:E], s13[:E], None, ones, w2[:E], s2[:E], None, ones,
        lid, c.w, E, geo.hidden, geo.inter,
        activation="silu", swiglu_limit=geo.swiglu_limit, w13_layout="w31",
    )
    m = metrics(oracle, c.refs["mx8"][0])
    ok = m["cos"] > 0.99999 or (m["ref_norm"] == 0 and m["max_abs"] == 0)
    report("PASS" if ok else "FAIL", f"oracle_crosscheck[{c.name} rank0 mx8 ref vs b12x moe_reference_w4a8_mx]", fmt(m))


def compare_all_refs(prefix: str, out: torch.Tensor, refs: dict, rank_sel, thr: float) -> dict:
    """Primary PASS/FAIL vs 'fp32' ref, INFO for 'mx8' and 'fp32_noclamp'."""
    def pick(v):
        t = refs[v]
        return t.sum(0) if rank_sel is None else t[rank_sel]
    m = check(f"{prefix} vs fp32_ref", out, pick("fp32"), thr)
    for v in ("mx8", "fp32_noclamp"):
        if v in refs:
            mv = metrics(out, pick(v))
            info(f"{prefix} vs {v}_ref: {fmt(mv)}")
    return m


def stage_full(fm, cases, raw, geo, device, do_variants: bool, store: dict):
    w13, s13, w2, s2 = raw
    runner = B12xRunner(fm, "full384", w13.clone(), s13.clone(), w2.clone(), s2.clone(), geo, device)
    mem("full prepared")
    for c in cases:
        if c.probe:
            continue
        variants = runner.variants(c.T) if do_variants else [("default", None, {})]
        for vname, dc, env in variants:
            tag = f"full384 {c.name} [{vname}]"
            try:
                out, dt = runner.run(c.T, vname, c.x, c.ids, c.w, decode_config=dc, env=env)
                info(f"{tag}: run {dt * 1e3:.1f}ms (incl. first-call compile)")
                compare_all_refs(tag, out, c.refs, None, COS_THR_REF)
                if vname == "default":
                    store[c.name] = out.float()
                elif c.name in store:
                    m = metrics(out, store[c.name])
                    info(f"{tag} vs full384 default: {fmt(m)}")
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                status = "FAIL" if vname == "default" else "INFO"
                report(status, tag, f"exception {type(exc).__name__}: {exc}")
    del runner
    torch.cuda.empty_cache()
    mem("full freed")


def stage_rank(fm, r: int, cases, raw, geo, device, epr: int, do_variants: bool, partials: dict):
    w13, s13, w2, s2 = raw
    lo, hi = r * epr, (r + 1) * epr
    label = f"rank{r}[{lo}:{hi}]"
    runner = B12xRunner(fm, label, w13[lo:hi].clone(), s13[lo:hi].clone(), w2[lo:hi].clone(),
                        s2[lo:hi].clone(), geo, device)
    mem(f"{label} prepared")
    for c in cases:
        lid = local_ids(c.ids, lo, hi)
        nonlocal_tok = ~(lid >= 0).any(dim=1)
        if c.probe:
            if r != 0:
                continue
            variants = runner.variants(c.T)
            for vname, dc, env in variants:
                tag = f"{label} {c.name} clamp-probe [{vname}]"
                try:
                    out, _ = runner.run(c.T, vname, c.x, lid, c.w, decode_config=dc, env=env)
                    mc = metrics(out, c.refs["fp32"][r])
                    mn = metrics(out, c.refs["fp32_noclamp"][r])
                    detail = (f"clamp_frac={c.clamp_frac:.3e} | vs clamped ref: {fmt(mc)} | "
                              f"vs UNclamped ref: {fmt(mn)}")
                    if c.clamp_frac == 0 or geo.swiglu_limit is None:
                        report("INFO", tag, "clamp never triggered; probe inconclusive. " + detail)
                    elif mc["cos"] > COS_THR_REF and mc["rel_l2"] < mn["rel_l2"]:
                        report("PASS", tag, "swiglu_limit applied. " + detail)
                    else:
                        report("FAIL", tag, "output matches UNclamped reference better -> swiglu_limit "
                               "not applied on this path (expected for tiny-decode, see report). " + detail)
                except Exception as exc:  # noqa: BLE001
                    traceback.print_exc()
                    report("FAIL" if vname == "default" else "INFO", tag, f"exception {type(exc).__name__}: {exc}")
            continue

        variants = runner.variants(c.T) if (do_variants and r == 0) else [("default", None, {})]
        for vname, dc, env in variants:
            tag = f"{label} {c.name} masked [{vname}]"
            try:
                out, dt = runner.run(c.T, vname, c.x, lid, c.w, decode_config=dc, env=env)
                info(f"{tag}: local_routes={int((lid >= 0).sum())}/{lid.numel()} run {dt * 1e3:.1f}ms")
                compare_all_refs(tag, out, c.refs, r, COS_THR_REF)
                n_nl = int(nonlocal_tok.sum())
                if n_nl:
                    sub = out[nonlocal_tok].float()
                    amax = torch.nan_to_num(sub, nan=float("inf")).abs().max().item()
                    ok = bool(torch.isfinite(sub).all()) and amax <= ZERO_TOL
                    report("PASS" if ok else "FAIL", f"{tag} all-nonlocal tokens == 0",
                           f"tokens={n_nl} max_abs={amax:.3e} exact_zero={amax == 0.0}")
                if vname == "default":
                    partials.setdefault(c.name, {})[r] = out.float()
                elif c.name in partials and r in partials[c.name]:
                    info(f"{tag} vs {label} default: {fmt(metrics(out, partials[c.name][r]))}")

                if r == 0:
                    # Output-buffer contract: prefill with NaN, see if every row is written.
                    out_nan, _ = runner.run(c.T, vname, c.x, lid, c.w, prefill=float("nan"), decode_config=dc, env=env)
                    nan_rows = int((~torch.isfinite(out_nan.float())).any(dim=1).sum())
                    same = bool(torch.equal(torch.nan_to_num(out_nan.float()), out.float()))
                    report("INFO", f"{tag} NaN-prefilled output",
                           f"rows still non-finite={nan_rows}/{c.T} (0 => kernel fully writes/zeroes output); "
                           f"bitwise equal to zero-prefilled run={same}")
                    # Whole batch masked: rank receives tokens but owns none of their routes.
                    all_m = torch.full_like(lid, -1)
                    out_m, _ = runner.run(c.T, vname, c.x, all_m, c.w, prefill=float("nan"), decode_config=dc, env=env)
                    nan_rows = int((~torch.isfinite(out_m.float())).any(dim=1).sum())
                    out_z, _ = runner.run(c.T, vname, c.x, all_m, c.w, prefill=0.0, decode_config=dc, env=env)
                    amax = torch.nan_to_num(out_z.float(), nan=float("inf")).abs().max().item()
                    ok = amax <= ZERO_TOL and bool(torch.isfinite(out_z.float()).all())
                    report("PASS" if ok else "FAIL", f"{tag} ALL routes -1 (zero-prefilled) == 0",
                           f"max_abs={amax:.3e}")
                    report("INFO", f"{tag} ALL routes -1 (NaN-prefilled)",
                           f"rows still non-finite={nan_rows}/{c.T} (>0 => caller must zero output when a rank has no work)")
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                report("FAIL" if vname == "default" else "INFO", tag, f"exception {type(exc).__name__}: {exc}")
    del runner
    torch.cuda.empty_cache()
    mem(f"{label} freed")


def stage_sum(cases, partials: dict, full_out: dict, ranks: int):
    for c in cases:
        if c.probe:
            continue
        parts = partials.get(c.name, {})
        if len(parts) != ranks:
            report("INFO", f"sum {c.name}", f"only {sorted(parts)} rank partials available; skipped")
            continue
        s = sum(parts[r] for r in range(ranks))
        check(f"sum4partials {c.name} vs fp32_ref(full)", s, c.refs["fp32"].sum(0), COS_THR_REF)
        if "mx8" in c.refs:
            info(f"sum4partials {c.name} vs mx8_ref(full): {fmt(metrics(s, c.refs['mx8'].sum(0)))}")
        if c.name in full_out:
            check(f"sum4partials {c.name} vs b12x full384", s, full_out[c.name], COS_THR_SUM)
        else:
            report("INFO", f"sum4partials {c.name} vs b12x full384", "full384 output unavailable")


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="/ckpt")
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--tokens", default="1,4,7,64,512,2048",
                    help="token counts (1..4 -> tiny-decode micro path by default; 4 -> 24 routed rows "
                         "is also the dynamic direct-routing band when tiny-decode is off)")
    ap.add_argument("--scale", type=float, default=1.0, help="hidden-state std")
    ap.add_argument("--probe-scale", type=float, default=8.0, help="hidden std for swiglu clamp probe")
    ap.add_argument("--probe-tokens", default="1,7", help="token counts for the clamp probe ('' disables)")
    ap.add_argument("--ranks", type=int, default=4)
    ap.add_argument("--skip-full", action="store_true", help="skip the 384-expert control run")
    ap.add_argument("--no-variants", action="store_true", help="only the default path")
    ap.add_argument("--only-rank", type=int, default=None, help="run only this rank subset (no sum check)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--mem-frac", type=float, default=0.85)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    tokens = [int(t) for t in args.tokens.split(",") if t.strip()]
    probe_tokens = [int(t) for t in args.probe_tokens.split(",") if t.strip()]

    # ---- stage 0
    def s0():
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(args.mem_frac, device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        info(f"torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(device)} "
             f"cc={torch.cuda.get_device_capability(device)} mem_total="
             f"{torch.cuda.get_device_properties(device).total_memory / 2**30:.1f}GiB")
        try:
            from importlib.metadata import version
            info(f"b12x version={version('b12x')}")
        except Exception as exc:  # noqa: BLE001
            info(f"b12x version unknown: {exc}")
        info("env: " + " ".join(f"{k}={v}" for k, v in sorted(os.environ.items()) if k.startswith("B12X_")))
        from b12x.moe import fused_moe as fm
        info(f"fused_moe supported on device: {fm.is_supported(device)}")
        geo = load_config(args.ckpt)
        if geo.hidden_act not in (None, "silu"):
            report("INFO", "config", f"hidden_act={geo.hidden_act} (test assumes SiLU-gated SwiGLU)")
        if geo.swiglu_limit is None:
            report("INFO", "config", "no swiglu_limit found in config; running unclamped")
        if geo.n_experts % args.ranks:
            raise RuntimeError(f"{geo.n_experts} experts not divisible by {args.ranks} ranks")
        mx8 = None
        try:
            from b12x._lib.intrinsics import quant_dequant_mxfp8_torch as mx8
        except Exception as exc:  # noqa: BLE001
            report("INFO", "mx8_emulation", f"quant_dequant_mxfp8_torch unavailable ({exc}); mx8 ref skipped")
        return device, fm, geo, mx8

    r0 = run_stage("0-env-config", s0)
    if r0 is None:
        return finish()
    device, fm, geo, mx8 = r0
    epr = geo.n_experts // args.ranks
    info(f"EP emulation: ranks={args.ranks} experts/rank={epr} topk={geo.topk} hidden={geo.hidden} inter={geo.inter}")

    raw = run_stage("1-load", load_layer_experts, args.ckpt, args.layer, geo, device)
    if raw is None:
        return finish()
    mem("raw loaded")
    run_stage("1b-dequant-selfcheck", stage_dequant_selfcheck, raw, geo)

    cases = run_stage("2a-cases", build_cases, geo, tokens, args.scale, args.probe_scale, probe_tokens,
                      args.ranks, epr, device, args.seed)
    if cases is None:
        return finish()
    run_stage("2b-reference", compute_references, cases, raw, geo, args.ranks, epr, mx8)
    if any(c.refs is None for c in cases):
        return finish()
    mem("refs computed")
    run_stage("2c-oracle-crosscheck", stage_oracle_crosscheck, cases, raw, geo, epr)

    full_out: dict = {}
    if not args.skip_full and args.only_rank is None:
        run_stage("3-full384-control", stage_full, fm, cases, raw, geo, device, not args.no_variants, full_out)

    partials: dict = {}
    rank_list = [args.only_rank] if args.only_rank is not None else list(range(args.ranks))
    for r in rank_list:
        run_stage(f"4-rank{r}-masked", stage_rank, fm, r, cases, raw, geo, device, epr,
                  not args.no_variants, partials)

    if args.only_rank is None:
        run_stage("5-sum-check", stage_sum, cases, partials, full_out, args.ranks)
    return finish()


def finish() -> int:
    n_pass = sum(1 for s, _, _ in RESULTS if s == "PASS")
    n_fail = sum(1 for s, _, _ in RESULTS if s == "FAIL")
    n_info = sum(1 for s, _, _ in RESULTS if s == "INFO")
    print("\n===================== RESULTS =====================")
    for s, name, _ in RESULTS:
        if s == "FAIL":
            print(f"  FAIL: {name}")
    verdict = "OK" if n_fail == 0 and n_pass > 0 else "NOT-OK"
    print(f"SUMMARY: {verdict} pass={n_pass} fail={n_fail} info={n_info}", flush=True)
    return 0 if (n_fail == 0 and n_pass > 0) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        print("SUMMARY: CRASH", flush=True)
        sys.exit(2)
