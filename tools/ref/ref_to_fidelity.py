#!/usr/bin/env python3
"""Convert reference-runner dumps (ref_runner.py --out, format sm120-ref-v1) into the fidelity reference format of
eval/compare_endpoints.py --teacher-forced --reference.

    python3 ref_to_fidelity.py --out /work/ref_fidelity.json /work/ref_out_p0.pt /work/ref_out_p1.pt ... [--top-k 20]

Per dump: prompt = input_ids, continuation = decoded_ids (the reference's greedy tokens), and for continuation position
j the reference's top-K logprobs of the distribution that predicted it:
- j = 0: the last prefill position (full logits in the dump -> exact log_softmax);
- j = 1..4: decode steps 0..3 (full logits stored -> exact);
- j >= 5: decode step j-1, of which the dump keeps the top-50 logits only. They are normalised over those 50
  (logsumexp of the top-50 instead of the whole vocabulary), which overstates each logprob by the tail mass
  (log(1 / (1 - tail))); for a peaked distribution that is far below the bf16 differences being measured.
The dump must come from the same checkpoint (and prompt tokens) the endpoints serve.
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()
    import torch

    runs = []
    for path in args.dumps:
        d = torch.load(path, map_location="cpu", weights_only=False)
        if not str(d.get("format", "")).endswith("-ref-v1"):  # sm120-ref-v1 and its older name
            raise SystemExit(f"{path}: not a ref_runner.py dump")
        ids = d["input_ids"].tolist()
        cont = d["decoded_ids"].tolist()
        n_dec = d["decode_top50_values"].shape[0]
        cont = cont[: n_dec + 1]  # decoded_ids[j] for j <= n_dec has a recorded distribution
        dists = [d["prefill_logits_last64"][-1].float().log_softmax(-1)]
        full = d.get("decode_logits_first4")
        for s in range(n_dec):
            if full is not None and s < full.shape[0]:
                dists.append(full[s].float().log_softmax(-1))
            else:
                vals, idx = d["decode_top50_values"][s].float(), d["decode_top50_ids"][s]
                dists.append((idx, vals - vals.logsumexp(0)))
        top = []
        for dist in dists:
            if isinstance(dist, tuple):
                idx, lp = dist
                order = lp.argsort(descending=True)[: args.top_k]
                top.append({str(int(idx[i])): float(lp[i]) for i in order})
            else:
                v, i = dist.topk(args.top_k)
                top.append({str(int(t)): float(x) for t, x in zip(i, v)})
        runs.append(dict(ids=ids, cont=cont, top=top[: len(cont)], source=path))
        print(f"{path}: {len(ids)} prompt tokens, {len(cont)} continuation positions")
    json.dump(dict(model="reference", top_k=args.top_k, runs=runs), open(args.out, "w"))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
