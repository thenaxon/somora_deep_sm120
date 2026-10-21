#!/usr/bin/env python3
"""Fine-grained prefill comparison of decoder layers (default 0): DeepInfer-SM120 vs the reference.

    python3 /opt/sm120/tools/compare_l0.py --di /work/di_out_torch.pt --ref /work/ref_out.pt --ref2 /work/ref_out_torch.pt
    python3 /opt/sm120/tools/compare_l0.py --layer all ...      # or --layer 3,5,6; one section per layer + a summary

Inputs are the normal dumps; the taps live in hidden["prefill"]["L<i>/<name>"] and are produced by
  DeepInfer: sm120/sm120_capture.py install_layer_captures   (env CAPTURE_LAYER, default 0)
  reference: tools/ref/ref_capture.py via ref_runner.py --capture-layer (env CAPTURE_LAYER, default 0)
Both sides use identical names and semantics (documented in those two files). For every tensor this prints
rel-L2 = |di-ref|/|ref|, max|d|, cosine (flattened), the worst token's rel-L2, and the ref-vs-ref2 noise
(ref_out.pt = tilelang kernels vs ref_out_torch.pt = torch kernels). q / kv / attention-core tensors are
also split into the nope part and the RoPE tail (last rope_dim channels): a RoPE theta / pairing / position
bug shows up as a large rope-part error with a clean nope part. Gate: fraction of tokens whose expert SETS
agree, and max|d| of the weights on those tokens (both sorted by expert id). swa_idx: fraction of query rows
whose visible key-position SET is identical; compress_idx (compressed positions each query attends to,
the indexer result of the layer's index source) and cand_mask (candidate positions of the candidate source)
the same way, plus the mean set overlap. Lines are in data-flow order, so the first row that jumps
far above its noise column is where the stacks diverge.

====================================================================================================
Semantic notes from reading both implementations of a ratio-0 (SWA-only) layer such as layer 0
(ref = third_party/ds41-inference/inference/model.py, DI = DeepInfer/deepinfer/model_executor/...)
====================================================================================================
 RoPE for SWA layers -- consistent by construction (verified at run time by rope_cos/rope_sin):
   ref: compress_ratio == 0 -> original_seq_len = 0 (no YaRN), theta = args.rope_theta (model.py:684-697);
        complex pairs (2i, 2i+1) on the LAST rope_head_dim channels (model.py:392-407, 772-773, 706-707);
        inverse rotation of the output (model.py:782).
   DI : _win_spec sets cfg.rope_theta = text.rope_theta, no rope_variant -> _GPT_DEFAULTS "base"
        (hugging_face/config.py:20-47, 101-111); LoraRopeCompressFlashMHAAutoRecomp picks rope_theta when
        n_packed_heads == 0 (v4_modules/mla_attn.py build_rotary_emb call, ~:270); rope_interleaved=True ->
        is_neox_style=False (utils.py convert_rope:39-46) -> adjacent pairs; offset = head_dim - rope_dim
        (attn.py _q_b_norm_rope / _prepare_swa_kv). cos_sin_cache = polar(outer(arange, inv_freq)) in fp32
        (layers/rotary_embedding.py:35-82), same formula as precompute_freqs_cis (model.py:369-389).
        Only difference: ref computes RoPE on bf16-rounded kv_norm output; DI's fused norm+rope kernel may
        rotate the fp32 norm result (sub-ULP).
 SWA window / mask -- consistent (verified at run time by swa_idx):
   ref: get_window_topk_idxs prefill row t = [max(0, t-W+1) .. t] (W = 128, INCLUSIVE of t: 128 keys incl.
        self), -1 for not-yet-existing slots (model.py:410-427).
   DI : pivots abs_lo = max(pos-W+1, valid_start, 0), section-2 = [abs_lo, pos] inclusive, history section
        empty for a fresh prompt (csrc/engine/input_metadata/prepare_tensor_input_v4.cpp:487-527);
        window_size = cfg.window_size[0] = sliding_window (config.py window_size=(sliding_window, 0)).
        DI's kv buffer is [swa history (n_seq*W rows, zero) | current kv]; indices are offset by
        total_k_swa (attn.py:818-857) -- compare_l0 normalises that.
 Attention sink: both add exp(sink) to the softmax denominator only (kernel.py sparse_attn / ref_kernels.py
   sparse_attn_torch:262 vs sm120/sm120_attn/sparse_attn.py _attend_rows:39-44 logaddexp). Same.
 Attention P precision (REAL difference, small): ref casts the probability tile P to bf16 before P@V
   (kernel.py:339,377-380 acc_s_cast, ref_kernels.py:260); DI SM120 torch path keeps P and the output in fp32
   (sm120_attn/sparse_attn.py:44-45, fused.py:120 out_dtype=None). Ref also rounds o to bf16 BEFORE the
   inverse RoPE (model.py:781-782); DI rounds only at the fp8 cast. Expect attn_core_* rel-L2 ~1e-3.
 q per-head RMSNorm: ref has none (model.py:771-773). DI applies a weightless per-head q RMSNorm iff
   cfg.use_qnorm (attn.py:106-109; _GPT_DEFAULTS use_qnorm=False, config.py:44). meta.use_q_b_norm and the
   presence of q_rms_scale show what ran. If it is True the logits are rescaled -> large error.
 KV cast: both quantise the full 512-dim post-RoPE kv (rope tail included) to e4m3 with 1x32 ue8m0 scales and
   use the cast-back values (model.py:708 act_quant(..., inplace) vs attn.py _prepare_swa_kv CastArgs
   num_per_channels=32 round_sf ue8m0 cast_back). Same.
 Output projection: ref act_quant(o) per 32 in natural [g, 8*512] order, bf16 einsum with the dequantised wo_a,
   bf16 out, then wo_b with fp8 act_quant (model.py:787-790). DI casts fp32 o per 32 (32-chunk permuted, same
   blocks), fp8_einsum, wv quantised to fp8 again for out_proj (use_fp8_einsum_output, attn.py:197-230). The
   fp8 grids match; the inputs to the first cast differ by the P/o rounding above.
 RMSNorm eps: both take rms_norm_eps from the HF config (ModelArgs.norm_eps default 1e-20 is overridden by
   ref_runner.hf_to_model_args); mHC prenorm eps = rms_norm_eps on both sides (model.py:955, hc_blocks.py:46).
   Gate renormalisation uses +1e-20 in ref (model.py:827, "not norm_eps") -- tile_kernels moe_topk_gate eps
   not visible here; irrelevant unless weights are tiny.
 mHC: identical structure (shifted pre-mix: layer-0 attention uses one-hot pre_mix -> hc_pre_attn == embedding,
   FFN uses the ATTENTION branch's pre mix, the FFN's pre feeds the next layer; model.py:970-996 vs
   hc.py:317-364). post = 2*sigmoid (kernel.py:429) vs sigmoid * hc_post_mult_value=2.0 (config.py:28);
   pre = sigmoid + hc_eps; comb = softmax + eps, column-normalise, (iters-1) x (row, column) with hc_eps on
   both sides (kernel.py:426-458 vs sm120/deep_gemm/_fused.py:22-28 reference of tile_kernels sinkhorn).
   The prenorm GEMM is fp32 in both (ref F.linear with TF32 disabled by ref_runner; DI shim
   tf32_hc_prenorm_gemm computes full fp32 despite the name, _gemm.py:425-452) but split-K (16 partials) on DI.
 MoE: routing weight multiplies the fp32 SwiGLU activation BEFORE the bf16/fp8 cast and w2 on both sides
   (model.py:852-853 vs swiglu_forward_and_per_token_cast(topk_weights=...) / _moe_ref.py:63). SwiGLU clamps:
   up clamped to [-limit, limit], gate clamped from above only, routed AND shared (model.py:847-849,
   Expert(swiglu_limit) for shared_experts model.py:889 vs mlp.py shared_forward clamp_value). Gate logits:
   ref fp32(x) @ fp32(W) (model.py:813), DI bf16 GEMM with fp32 accumulation (mlp.py gate ReplicatedLinear
   use_fp32_output) -- same values up to summation order; near-tied experts can flip.
   Routed sum: ref fp32 accumulation + shared, one bf16 rounding (model.py:895-906); DI torch backend the same
   (sm120_moe.py:276-280); DI b12x backend rounds the routed sum to bf16 first (sm120_moe.py:270-275).
   Expert input: ref act_quant of the bf16 ffn_norm output; DI uses the fp8 copy emitted by the RMSNorm kernel
   (may be quantised from the fp32 norm result, i.e. before bf16 rounding -> rare 1-code differences).
 YOCO: config.py:_n_prefill_layer makes the first ratio-1 layer the YOCO boundary (layer 2 for [0,2,1]); DI may
   compact the prefill queries of later consumer layers (deepseek.py:191-207). Irrelevant for layer 0.
"""

from __future__ import annotations

import argparse
import os
import sys

ORDER = [
    "positions", "hc_trunk_in", "attn_pre_mix_in", "hc_pre_attn", "attn_mix_pre", "attn_mix_post", "attn_mix_comb",
    "attn_in", "q_a", "q_pre_rope", "kv_proj", "kv_normed_pre_rope", "rope_cos", "rope_sin", "q_post_rope",
    "q_rms_scale", "kv", "swa_idx", "attn_sink", "attn_core_pre_unrope", "attn_core_out", "attn_out",
    "hc_trunk_mid", "ffn_pre_mix_in", "hc_pre_ffn", "ffn_mix_pre", "ffn_mix_post", "ffn_mix_comb", "ffn_in",
    "gate_idx", "gate_w", "routed_out", "shared_out", "ffn_out",
]
ROPE_SPLIT = {"q_pre_rope", "q_post_rope", "kv_proj", "kv_normed_pre_rope", "kv", "attn_core_pre_unrope",
              "attn_core_out"}
SPECIAL = {"gate_idx", "gate_w", "swa_idx", "compress_idx", "cand_mask", "positions"}


def load(path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def taps(d, layer: int) -> dict:
    hp = ((d or {}).get("hidden") or {}).get("prefill") or {}
    p = f"L{layer}/"
    return {k[len(p):]: v for k, v in hp.items() if k.startswith(p)}


def stats(a, b):
    """a = reference, b = candidate (same shape after reshape). -> dict or None."""
    import torch

    if a is None or b is None:
        return None
    a, b = a.float(), b.float()
    if a.shape != b.shape:
        if a.numel() != b.numel():
            return dict(err=f"shape {tuple(a.shape)} vs {tuple(b.shape)}")
        b = b.reshape(a.shape)
    d = a - b
    rel = (d.norm() / a.norm().clamp_min(1e-30)).item()
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    res = dict(rel=rel, maxd=d.abs().max().item(), cos=cos, ref_absmax=a.abs().max().item())
    if a.dim() >= 2 and a.shape[0] > 1:
        per = d.flatten(1).norm(dim=1) / a.flatten(1).norm(dim=1).clamp_min(1e-30)
        w = int(per.argmax())
        res.update(worst_tok=w, worst_rel=per[w].item(), med_rel=per.median().item())
    return res


def fmt(s):
    if s is None:
        return f"{'-':>10s} {'-':>9s} {'-':>10s}"
    if "err" in s:
        return f"{s['err']:>31s}"
    return f"{s['rel']:10.3e} {s['maxd']:9.2e} {s['cos']:10.7f}"


def gate_stats(ia, wa, ib, wb):
    """expert-set agreement and weight error on agreeing tokens (both sorted by expert id)."""
    import torch

    if any(t is None for t in (ia, wa, ib, wb)):
        return None
    ia, ib = ia.long(), ib.long()
    if ia.shape != ib.shape:
        return dict(err=f"shape {tuple(ia.shape)} vs {tuple(ib.shape)}")
    sa, oa = ia.sort(-1)
    sb, ob = ib.sort(-1)
    same = (sa == sb).all(-1)
    nsame = int(same.sum())
    wa_s, wb_s = wa.float().gather(-1, oa), wb.float().gather(-1, ob)
    res = dict(frac_same=nsame / ia.shape[0], n_diff=int((~same).sum()),
               first_diff=[int(i) for i in (~same).nonzero().flatten()[:8]],
               overlap=float((torch.stack([torch.isin(sa[t], sb[t]).float().mean() for t in range(sa.shape[0])])).mean()))
    if nsame:
        d = (wa_s[same] - wb_s[same]).abs()
        res.update(w_maxd=d.max().item(), w_rel=((wa_s[same] - wb_s[same]).norm() / wa_s[same].norm()).item())
    res["wsum_ref"] = wa.float().sum(-1).mean().item()
    res["wsum_x"] = wb.float().sum(-1).mean().item()
    return res


def fmt_gate(tag, g):
    if g is None:
        return f"  {tag:10s} -"
    if "err" in g:
        return f"  {tag:10s} {g['err']}"
    s = (f"  {tag:10s} identical expert sets {g['frac_same']:.4f} ({g['n_diff']} tokens differ, mean overlap "
         f"{g['overlap']:.4f}; first {g['first_diff']})")
    if "w_maxd" in g:
        s += f"  weights on identical sets: max|d| {g['w_maxd']:.3e} rel {g['w_rel']:.3e}"
    s += f"  sum(w) ref {g['wsum_ref']:.4f} x {g['wsum_x']:.4f}"
    return s


def idx_sets_equal(a, b):
    """rows of int index tensors -> fraction of rows whose valid (>=0) SETS are equal, and first bad row."""
    import torch

    if a is None or b is None:
        return None
    if a.shape[0] != b.shape[0]:
        return dict(err=f"rows {a.shape[0]} vs {b.shape[0]}")
    a, b = a.long(), b.long()
    bad = []
    for t in range(a.shape[0]):
        x = a[t][a[t] >= 0].sort().values
        y = b[t][b[t] >= 0].sort().values
        if x.numel() != y.numel() or not torch.equal(x, y):
            bad.append(t)
    ov = []
    for t in range(a.shape[0]):
        x, y = a[t][a[t] >= 0], b[t][b[t] >= 0]
        ov.append(1.0 if x.numel() == 0 and y.numel() == 0 else
                  torch.isin(x, y).sum().item() / max(x.numel(), y.numel()))
    res = dict(frac=1 - len(bad) / a.shape[0], bad=bad[:8], n_bad=len(bad), overlap=sum(ov) / len(ov),
               hist_refs=int((b == -2).sum()) if (b == -2).any() else 0)
    if bad:
        t = bad[0]
        res["example"] = (t, a[t][a[t] >= 0].tolist()[:6], a[t][a[t] >= 0].tolist()[-3:],
                          b[t][b[t] >= 0].tolist()[:6], b[t][b[t] >= 0].tolist()[-3:])
    return res


def tapped_layers(d) -> list[int]:
    hp = ((d or {}).get("hidden") or {}).get("prefill") or {}
    return sorted({int(k[1:].split("/", 1)[0]) for k in hp if k.startswith("L") and "/" in k and k[1:].split("/", 1)[0].isdigit()})


def print_idx_sets(title, R, D, R2, name):
    print(f"\n-- {title}")
    for tag, x in (("DI", D.get(name)), ("ref2", R2.get(name))):
        r = idx_sets_equal(R.get(name), x)
        if r is None:
            print(f"  {tag:5s} -" + ("  (not captured on this side)" if R.get(name) is not None and x is None else ""))
        elif "err" in r:
            print(f"  {tag:5s} {r['err']}")
        else:
            print(f"  {tag:5s} identical rows {r['frac']:.4f} ({r['n_bad']} differ, first {r['bad']}), mean overlap "
                  f"{r['overlap']:.4f}"
                  + (f"; {r['hist_refs']} DI entries point into the swa history buffer" if r["hist_refs"] else "")
                  + (f"\n        example row {r['example'][0]}: ref {r['example'][1]}..{r['example'][2]}"
                     f"  x {r['example'][3]}..{r['example'][4]}" if "example" in r else ""))


def compare_layer(args, di_d, ref_d, ref2_d, layer: int, role: str = "") -> int | None:
    """Print the breakdown of one layer; return the number of flagged tensors (None = no taps)."""
    import torch

    D, R, R2 = taps(di_d, layer), taps(ref_d, layer), taps(ref2_d, layer) if ref2_d else {}
    if not D or not R:
        print(f"\n== layer {layer}: no L{layer}/ taps in {'DI' if not D else 'ref'} dump (rerun with CAPTURE_LAYER={layer})")
        return None
    L = int(ref_d["input_ids"].numel())
    md, mr = D.get("meta") or {}, R.get("meta") or {}
    rope_dim = args.rope_dim or md.get("rope_dim") or mr.get("rope_dim") or 64

    print(f"\n{'=' * 100}\n== layer {layer} [{role}]: L={L}  DI env="
          f"{ {k: v for k, v in ((di_d.get('meta') or {}).get('env') or {}).items() if k.startswith('SM120_')} }")
    for e in (di_d.get("meta") or {}).get("errors") or []:
        print(f"  DI capture WARNING: {e}")
    for e in R.get("errors") or []:
        print(f"  ref capture WARNING: {e}")
    print("\n-- meta (DI | ref)")
    for k in sorted(set(md) | set(mr)):
        print(f"  {k:28s} {str(md.get(k, '-')):>24s} | {str(mr.get(k, '-'))}")

    # token order
    pos = D.get("positions")
    if pos is not None:
        ok = torch.equal(pos.long(), torch.arange(L))
        print(f"\n-- DI positions == arange(L): {ok}" + ("" if ok else f"  first: {pos[:8].tolist()}"))

    thr_bad = 0
    print(f"\n-- tensors (rel-L2 = |DI-ref|/|ref|, cos flattened, worst = token with the largest per-token rel-L2;"
          f" noise = ref2 vs ref; flag if DI-ref > {args.factor:g} x max(noise, {args.floor:g}))")
    print(f"  {'name':24s} {'shape':>16s} | {'DI-ref relL2':>10s} {'max|d|':>9s} {'cos':>10s} | {'worst tok':>9s} "
          f"{'rel':>9s} {'median':>9s} | {'noise relL2':>10s} {'max|d|':>9s} {'cos':>10s} | flag")
    names = [n for n in ORDER if n in R or n in D] + sorted((set(R) | set(D)) - set(ORDER) - {"meta", "errors"})
    for n in names:
        if n in SPECIAL:
            continue
        a, b, c = R.get(n), D.get(n), R2.get(n)
        if not isinstance(a, torch.Tensor) and not isinstance(b, torch.Tensor):
            continue
        shp = str(tuple((b if isinstance(b, torch.Tensor) else a).shape))
        if a is None or b is None:
            print(f"  {n:24s} {shp:>16s} | {'(only in ' + ('DI' if a is None else 'ref') + ')':>31s}")
            continue
        s, sn = stats(a, b), stats(a, c) if isinstance(c, torch.Tensor) else None
        worst = (f"{s['worst_tok']:9d} {s['worst_rel']:9.2e} {s['med_rel']:9.2e}" if s and "worst_tok" in s
                 else f"{'':9s} {'':9s} {'':9s}")
        flag = ""
        if s and "rel" in s:
            noise = sn["rel"] if sn and "rel" in sn else 0.0
            if s["rel"] > args.factor * max(noise, args.floor):
                flag = "<-- DIVERGES" if sn else "<-- (no noise ref)"
                thr_bad += 1
        print(f"  {n:24s} {shp:>16s} | {fmt(s)} | {worst} | {fmt(sn)} | {flag}")
        if n in ROPE_SPLIT and s and "rel" in s and a.shape[-1] > rope_dim:
            for part, sl in (("nope", slice(0, a.shape[-1] - rope_dim)), ("rope", slice(a.shape[-1] - rope_dim, None))):
                bb = b.reshape(a.shape) if b.shape != a.shape else b
                sp = stats(a[..., sl], bb[..., sl])
                spn = stats(a[..., sl], c.reshape(a.shape)[..., sl]) if isinstance(c, torch.Tensor) and c.numel() == a.numel() else None
                print(f"    {'.' + part:22s} {'':>16s} | {fmt(sp)} | {'':29s} | {fmt(spn)} |")

    # window mask
    print_idx_sets("SWA visible key positions per query (sets, -1 masked)", R, D, R2, "swa_idx")
    if R.get("swa_idx") is not None:
        ri = R["swa_idx"].long()
        nvis = (ri >= 0).sum(-1)
        print(f"  ref visible keys per row: min {int(nvis.min())} max {int(nvis.max())} "
              f"(row {L - 1}: {ri[L - 1][ri[L - 1] >= 0].min().item()}..{ri[L - 1][ri[L - 1] >= 0].max().item()})")

    # compressed top-k (YOCO: consumers carry their index source's result) and candidate blocks
    if R.get("compress_idx") is not None or D.get("compress_idx") is not None:
        print_idx_sets("compressed positions attended per query (indexer top-k, sets, -1 masked)", R, D, R2, "compress_idx")
        ci = R.get("compress_idx")
        if ci is not None:
            nsel = (ci.long() >= 0).sum(-1)
            print(f"  ref selected per row: min {int(nsel.min())} max {int(nsel.max())} (index_topk caps it; rows "
                  f"with fewer compressed positions select all of them)")
    if R.get("cand_mask") is not None or D.get("cand_mask") is not None:
        def as_idx(m):
            if m is None:
                return None
            m = m.bool()
            ar = torch.arange(m.shape[-1]).expand_as(m)
            return torch.where(m, ar, torch.full_like(ar, -1))
        R_c, D_c, R2_c = ({"c": as_idx(x.get("cand_mask"))} for x in (R, D, R2))
        print_idx_sets("candidate positions published by the candidate source (sets)", R_c, D_c, R2_c, "c")

    # gate
    print("\n-- MoE routing")
    print(fmt_gate("DI-ref", gate_stats(R.get("gate_idx"), R.get("gate_w"), D.get("gate_idx"), D.get("gate_w"))))
    if R2:
        print(fmt_gate("ref2-ref", gate_stats(R.get("gate_idx"), R.get("gate_w"), R2.get("gate_idx"), R2.get("gate_w"))))

    # derived checks
    print("\n-- derived")
    if isinstance(D.get("ffn_out"), torch.Tensor) and isinstance(D.get("shared_out"), torch.Tensor) and "routed_out" not in D:
        der = D["ffn_out"].float() - D["shared_out"].float()
        print(f"  DI routed (ffn_out - shared_out, derived): {fmt(stats(R.get('routed_out'), der))}")
    if layer == 0 and isinstance(R.get("hc_pre_attn"), torch.Tensor) and isinstance(R.get("hc_trunk_in"), torch.Tensor):
        x0 = R["hc_trunk_in"][:, 0]
        print(f"  ref hc_pre_attn vs trunk copy 0 (one-hot mix check): {fmt(stats(x0, R['hc_pre_attn']))}")
    emb = ((di_d.get("hidden") or {}).get("prefill") or {}).get("embed_hc")
    if isinstance(emb, torch.Tensor) and isinstance(D.get("hc_trunk_in"), torch.Tensor) and layer == 0:
        print(f"  DI hc_trunk_in vs DI embed_hc: {fmt(stats(emb[0].float(), D['hc_trunk_in']))}")
    print(f"\nlayer {layer}: flagged tensors: {thr_bad}")
    return thr_bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--di", default="/work/di_out.pt")
    ap.add_argument("--ref", default="/work/ref_out.pt")
    ap.add_argument("--ref2", default="/work/ref_out_torch.pt", help="second reference for the noise column")
    ap.add_argument("--layer", default=os.environ.get("CAPTURE_LAYER", "0"),
                    help="'0', '3,5,6' or 'all' (= every layer tapped in both dumps)")
    ap.add_argument("--rope-dim", type=int, default=None, help="default: from meta (64 for V4.1)")
    ap.add_argument("--factor", type=float, default=3.0, help="flag DI-ref > factor x max(noise, floor)")
    ap.add_argument("--floor", type=float, default=1e-4)
    args = ap.parse_args()

    import torch

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from compare import layer_roles

    di_d, ref_d = load(args.di), load(args.ref)
    ref2_d = load(args.ref2) if args.ref2 and os.path.isfile(args.ref2) else None
    if not torch.equal(ref_d["input_ids"], di_d["input_ids"]):
        print("input_ids DIFFER between DI and ref -- comparison is meaningless")
        return 2
    if args.layer.strip().lower() == "all":
        layers = sorted(set(tapped_layers(di_d)) & set(tapped_layers(ref_d)))
    else:
        layers = sorted({int(x) for x in args.layer.split(",") if x.strip()})
    if not layers:
        print(f"no common tapped layers (DI {tapped_layers(di_d)}, ref {tapped_layers(ref_d)})")
        return 2
    roles = layer_roles(ref_d.get("meta"))
    results = {li: compare_layer(args, di_d, ref_d, ref2_d, li, roles.get(li, "")) for li in layers}
    if len(layers) > 1:
        print(f"\n{'=' * 100}\n== summary (flagged tensors per layer; first flagged layer localises the divergence)")
        for li, n in results.items():
            print(f"  layer {li:3d} {roles.get(li, ''):40s} {'no taps' if n is None else n}")
    if all(n is None for n in results.values()):
        return 2
    return 1 if any(n for n in results.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
