#!/usr/bin/env python3
"""Compare two ref_runner outputs (e.g. REF_KERNELS=tilelang vs torch, or reference vs a DeepInfer dump
written in the same format: input_ids, decoded_ids, prefill_logits_last64, decode_logits_first4, ...).

    python3 compare_out.py /work/ref_out.pt /work/ref_out_torch.pt
"""

from __future__ import annotations

import sys


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    import torch

    a = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    b = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
    if not torch.equal(a["input_ids"], b["input_ids"]):
        print("input_ids DIFFER -- comparison is meaningless")
        return 1
    print(f"prompt tokens: {a['input_ids'].numel()}")
    da, db = a["decoded_ids"].tolist(), b["decoded_ids"].tolist()
    n_same = next((i for i, (x, y) in enumerate(zip(da, db)) if x != y), min(len(da), len(db)))
    print(f"decoded A: {da}\ndecoded B: {db}\nidentical prefix: {n_same}/{min(len(da), len(db))}")

    def logit_stats(name, x, y):
        x, y = x.float(), y.float()
        d = (x - y).abs()
        lp, lq = x.log_softmax(-1), y.log_softmax(-1)
        kl = (lp.exp() * (lp - lq)).sum(-1)
        top1 = (x.argmax(-1) == y.argmax(-1)).float().mean()
        print(f"{name:24s} max|d|={d.max():.4e} mean|d|={d.mean():.4e} KL(A||B) max={kl.max():.4e} mean={kl.mean():.4e} top1 agree={top1:.3f}")

    if a.get("prefill_logits") is not None and b.get("prefill_logits") is not None:
        logit_stats("prefill (all pos)", a["prefill_logits"], b["prefill_logits"])
    logit_stats("prefill (last 64)", a["prefill_logits_last64"], b["prefill_logits_last64"])
    agree = (a["prefill_argmax"] == b["prefill_argmax"]).float().mean()
    print(f"{'prefill argmax agree':24s} {agree:.4f}")
    n4 = min(a["decode_logits_first4"].shape[0], b["decode_logits_first4"].shape[0])
    if n4:
        logit_stats("decode first steps", a["decode_logits_first4"][:n4], b["decode_logits_first4"][:n4])
    ha, hb = a.get("hidden"), b.get("hidden")
    if ha and hb:
        for k in ha["prefill"]:
            if "/" not in k and k in hb["prefill"]:  # "L<i>/..." fine taps: see tools/compare_l0.py
                x, y = ha["prefill"][k].float(), hb["prefill"][k].float()
                rel = (x - y).norm() / x.norm().clamp_min(1e-30)
                print(f"hidden prefill {k:12s} rel L2 {rel:.4e} max|d| {(x - y).abs().max():.4e}")
    for name, m in (("A", a.get("meta", {})), ("B", b.get("meta", {}))):
        k = m.get("kernels", {})
        print(f"{name}: kernels mode={k.get('mode')} backends={k.get('backend_final')} fallbacks={k.get('fallbacks')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
