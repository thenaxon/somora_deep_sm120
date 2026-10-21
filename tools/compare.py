#!/usr/bin/env python3
"""Compare a DeepInfer-SM120 dump (run_deepinfer.py) against the reference dumps (ref/ref_runner.py).

    python3 /opt/sm120/tools/compare.py --di /work/di_out.pt --ref /work/ref_out.pt --ref2 /work/ref_out_torch.pt

The truncated 3-layer model is numerically chaotic (two reference runs that only differ in ULP-level
kernel rounding reach ~74% prefill top-1 agreement), so end-to-end logit agreement is reported for
information only. The pass signal is the per-layer prefill hidden-state error: DI vs ref is compared
with the ref-vs-ref2 noise floor of the same layer and flagged SUSPICIOUS above --factor x noise.
Works for any number of layers; each row is annotated with the original layer id and its role (from the
reference meta: model_args + the checkpoint's truncation.json), e.g. "o21 C1 kv<5 idx<5 yoco".

Prints: prefill top-1 agreement (all positions), KL(ref||di) and cosine on every position with full
logits (prefill all/last-64, decode first 4 / all), top-50-renormalised KL for decode steps that
only have top-k in the reference, decoded-token match, per-layer hidden rel-L2 / max|d|.
"""

from __future__ import annotations

import argparse
import os
import sys


def load(path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def full_logit_stats(x, y):
    """x = reference logits, y = candidate; rows = positions. -> dict of floats."""
    import torch

    x, y = x.float(), y.float()
    lp, lq = x.log_softmax(-1), y.log_softmax(-1)
    kl = (lp.exp() * (lp - lq)).sum(-1)
    cos = torch.nn.functional.cosine_similarity(x, y, dim=-1)
    top1 = (x.argmax(-1) == y.argmax(-1)).float()
    return dict(n=x.shape[0], kl_mean=kl.mean().item(), kl_max=kl.max().item(), cos_mean=cos.mean().item(),
                cos_min=cos.min().item(), top1=top1.mean().item(), maxabs=(x - y).abs().max().item())


def fmt_stats(name, s):
    return (f"  {name:28s} n={s['n']:4d}  KL(ref||x) mean={s['kl_mean']:.4e} max={s['kl_max']:.4e}  "
            f"cos mean={s['cos_mean']:.6f} min={s['cos_min']:.6f}  top1={s['top1']:.3f}  max|d|={s['maxabs']:.3e}")


def topk_renorm_kl(ref_vals, ref_ids, cand_full):
    """KL over the reference's top-k support, both distributions renormalised on it."""
    import torch

    q = cand_full.float().gather(-1, ref_ids.long())
    lp, lq = ref_vals.float().log_softmax(-1), q.log_softmax(-1)
    kl = (lp.exp() * (lp - lq)).sum(-1)
    cos = torch.nn.functional.cosine_similarity(ref_vals.float(), q, dim=-1)
    return kl, cos


def compare_logits(tag, ref, x):
    print(f"\n== logits: {tag}")
    import torch

    if not torch.equal(ref["input_ids"], x["input_ids"]):
        print("  input_ids DIFFER -- comparison is meaningless")
        return
    L = ref["input_ids"].numel()
    pa, pb = ref.get("prefill_argmax"), x.get("prefill_argmax")
    if pa is not None and pb is not None and pa.numel() == pb.numel():
        print(f"  prefill top-1 agreement (all {L} positions): {(pa == pb).float().mean().item():.4f}")
    if ref.get("prefill_logits") is not None and x.get("prefill_logits") is not None:
        print(fmt_stats("prefill (all positions)", full_logit_stats(ref["prefill_logits"], x["prefill_logits"])))
    if ref.get("prefill_logits_last64") is not None and x.get("prefill_logits_last64") is not None:
        print(fmt_stats("prefill (last 64)", full_logit_stats(ref["prefill_logits_last64"], x["prefill_logits_last64"])))
    ra, xa = ref["decoded_ids"].tolist(), x["decoded_ids"].tolist()
    n = min(len(ra), len(xa))
    prefix = next((i for i in range(n) if ra[i] != xa[i]), n)
    same = sum(int(a == b) for a, b in zip(ra, xa))
    print(f"  decoded ref: {ra}\n  decoded x  : {xa}\n  identical prefix {prefix}/{n}, position-wise match {same}/{n}"
          + ("  (x was teacher-forced)" if x.get("meta", {}).get("teacher_forced") else ""))
    if prefix < n:
        print("  NOTE: decode steps after the first divergence see different prefixes unless teacher-forced")
    d4r, d4x = ref.get("decode_logits_first4"), x.get("decode_logits_first4")
    if d4r is not None and d4x is not None and d4r.numel() and d4x.numel():
        k = min(d4r.shape[0], d4x.shape[0])
        print(fmt_stats(f"decode first {k} (full)", full_logit_stats(d4r[:k], d4x[:k])))
    xall = x.get("decode_logits_all")
    if xall is None and x.get("decode_logits_first4") is not None:
        xall = x["decode_logits_first4"]
    if ref.get("decode_top50_values") is not None and xall is not None and xall.numel():
        k = min(ref["decode_top50_values"].shape[0], xall.shape[0])
        kl, cos = topk_renorm_kl(ref["decode_top50_values"][:k], ref["decode_top50_ids"][:k], xall[:k])
        t1 = (ref["decode_top50_ids"][:k, 0] == xall[:k].argmax(-1)).float().mean().item()
        print(f"  {'decode all steps (top-50)':28s} n={k:4d}  KL_top50 mean={kl.mean().item():.4e} "
              f"max={kl.max().item():.4e}  cos_top50 mean={cos.mean().item():.6f}  top1={t1:.3f}")
        print("  per-step KL_top50: " + " ".join(f"{v:.3f}" for v in kl.tolist()))


def rel_l2(a, b):
    a, b = a.float(), b.float()
    if a.shape != b.shape:
        if a.numel() == b.numel():
            b = b.reshape(a.shape)
        else:
            return float("nan"), float("nan")
    return ((a - b).norm() / a.norm().clamp_min(1e-30)).item(), (a - b).abs().max().item()


def layer_roles(meta: dict | None) -> dict[int, str]:
    """new layer -> 'o<orig>[*] SWA | C<r> kv-src|kv<src idx-src|idx<src [cand-src|cand<src] [yoco]'.

    * = demoted to a consumer by make_truncated_ckpt --as-consumer; 'yoco' = layer >= n_prefill_layer (the first
    ratio-1 layer, DeepInfer config.py:158-161). Sources follow model.py:500-503,654-656."""
    a = (meta or {}).get("model_args") or {}
    if not a:
        return {}
    tr = (meta or {}).get("truncation") or {}
    orig = {int(k): v for k, v in (tr.get("layer_map_new_to_orig") or {}).items()}
    demoted = set(tr.get("demoted_to_consumer") or [])
    n, ratios = a["n_layers"], a["compress_ratios"]
    kv, idx, cand = list(a["kv_source_layers"]), list(a["index_source_layers"]), a["candidate_source_layer"]
    n_prefill = next((i for i, r in enumerate(ratios[:n]) if r == 1), n)
    out = {}
    for i in range(n):
        r = ratios[i]
        if r == 0:
            role = "SWA"
        else:
            role = f"C{r} " + ("kv-src" if i in kv else f"kv<{max((s for s in kv if s <= i), default='?')}")
            role += " idx-src" if i in idx else f" idx<{max((s for s in idx if s <= i), default='?')}"
            if i == cand:
                role += " cand-src"
            elif 0 <= cand < i and i in idx:
                role += f" cand<{cand}"
        if i >= n_prefill:
            role += " yoco"
        if i in orig:
            role = f"o{orig[i]}{'*' if orig[i] in demoted else ''} {role}"
        out[i] = role
    return out


def compare_hidden(ref, ref2, di, factor: float, floor: float) -> int:
    hr = (ref.get("hidden") or {}).get("prefill") or {}
    hd = (di.get("hidden") or {}).get("prefill") or {}
    h2 = ((ref2 or {}).get("hidden") or {}).get("prefill") or {}
    if not hr or not hd:
        print("\n== hidden: missing in ref or di")
        return 0
    order = ["embed_hc"] + sorted([k for k in hr if k.startswith("layer")], key=lambda k: int(k[5:])) + ["final_norm"]
    # "L<i>/<name>" keys are the fine per-layer taps (CAPTURE_LAYER): compared by compare_l0.py, not here
    order = [k for k in order if k in hr] + [k for k in hr if k not in order and "/" not in k]
    print(f"\n== prefill hidden states (rel-L2 = |a-b|/|ref|; noise = ref vs ref2; SUSPICIOUS if DI-vs-ref > "
          f"{factor:g} x max(noise, {floor:g}))")
    print(f"  {'key':12s} {'shape':>20s} | {'DI-ref relL2':>12s} {'max|d|':>10s} | {'DI-ref2 relL2':>13s} {'max|d|':>10s} | "
          f"{'noise relL2':>11s} {'max|d|':>10s} | ratio  flag        role")
    roles = layer_roles(ref.get("meta"))
    bad = 0
    for k in order:
        a = hr[k]
        if k not in hd:
            print(f"  {k:12s} missing in DI dump")
            bad += 1
            continue
        r_di, m_di = rel_l2(a, hd[k])
        if k in h2:
            r_d2, m_d2 = rel_l2(h2[k], hd[k])
            r_nz, m_nz = rel_l2(a, h2[k])
        else:
            r_d2 = m_d2 = r_nz = m_nz = float("nan")
        noise = r_nz if r_nz == r_nz else 0.0
        thr = factor * max(noise, floor)
        ratio = r_di / max(noise, floor)
        flag = "SUSPICIOUS" if (r_di != r_di or r_di > thr) else "ok"
        bad += flag != "ok"
        role = roles.get(int(k[5:]), "") if k.startswith("layer") and k[5:].isdigit() else ""
        print(f"  {k:12s} {str(tuple(hd[k].shape)):>20s} | {r_di:12.4e} {m_di:10.3e} | {r_d2:13.4e} {m_d2:10.3e} | "
              f"{r_nz:11.4e} {m_nz:10.3e} | {ratio:6.2f} {flag:11s} {role}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--di", default="/work/di_out.pt")
    ap.add_argument("--ref", default="/work/ref_out.pt", help="reference (tilelang kernels)")
    ap.add_argument("--ref2", default="/work/ref_out_torch.pt", help="second reference (pure torch) for the noise floor")
    ap.add_argument("--factor", type=float, default=3.0)
    ap.add_argument("--floor", type=float, default=1e-4, help="minimum noise floor (rel-L2) for exact layers")
    args = ap.parse_args()
    ref, di = load(args.ref), load(args.di)
    ref2 = load(args.ref2) if args.ref2 and os.path.isfile(args.ref2) else None
    meta = di.get("meta", {})
    print(f"DI: gpu={meta.get('gpu')} teacher_forced={meta.get('teacher_forced')} "
          f"env={ {k: v for k, v in (meta.get('env') or {}).items() if k.startswith(('SM120_', 'B12X_'))} }")
    for e in meta.get("errors") or []:
        print(f"DI capture WARNING: {e}")
    compare_logits("DI vs ref", ref, di)
    if ref2 is not None:
        compare_logits("DI vs ref2", ref2, di)
        compare_logits("ref2 vs ref (noise floor)", ref, ref2)
    bad = compare_hidden(ref, ref2, di, args.factor, args.floor)
    print(f"\nsuspicious hidden layers: {bad}")
    if any("/" in k for k in ((ref.get("hidden") or {}).get("prefill") or {})):
        print("fine per-layer taps present: run compare_l0.py for the intra-layer breakdown")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
