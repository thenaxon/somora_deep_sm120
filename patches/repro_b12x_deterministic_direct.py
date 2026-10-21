"""Standalone repro: b12x fused_moe w4a8_mx with deterministic_output=True fails for shapes whose heuristic
picks dynamic_route_mode="direct" ("planned dynamic direct routing is unsupported for this launch shape").

_heuristic_dynamic_route_mode() evaluates the direct-routing candidate with deterministic_output=False, so it can
select "direct" for a deterministic plan; _dynamic_direct_routing_selected() then rejects that combination.
Forcing dynamic_route_mode="grouped" in the decode config works and is bit-reproducible.

Random MXFP4 experts with DeepSeek-V4.1 geometry (E=384, hidden 5120, intermediate 2304, top-6), no checkpoint.
Usage: python repro_b12x_deterministic_direct.py [--tokens 1,4,8,64,512]
"""

import argparse
import dataclasses

import torch

import b12x
from b12x.moe import fused_moe as fm

E, K, N, TOPK = 384, 5120, 2304, 6
DEV = torch.device("cuda", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="1,4,8,64,512")
    args = ap.parse_args()
    torch.manual_seed(0)
    print(f"b12x {getattr(b12x, '__version__', '?')} {torch.cuda.get_device_name()}")
    wplan = fm.plan_weights(quant_modes="w4a8_mx", source_format="fp4_e8m0_k32", activation="silu",
                            params_dtype=torch.bfloat16, num_experts=E, hidden_size=K, intermediate_size=N,
                            w13_layout="w31")
    ones = torch.ones(E, device=DEV)
    experts = fm.prepare_weights(
        plan=wplan, params_dtype=torch.bfloat16,
        w1_fp4=torch.randint(0, 256, (E, 2 * N, K // 2), dtype=torch.uint8, device=DEV),
        w2_fp4=torch.randint(0, 256, (E, K, N // 2), dtype=torch.uint8, device=DEV),
        w1_blockscale=torch.full((E, 2 * N, K // 32), 120, dtype=torch.uint8, device=DEV),
        w2_blockscale=torch.full((E, K, N // 32), 120, dtype=torch.uint8, device=DEV),
        w1_global_scale=ones, w2_global_scale=ones, a1_gscale=ones, a2_gscale=ones)
    bad = 0
    for T in [int(t) for t in args.tokens.split(",")]:
        kw = dict(max_tokens=T, num_topk=TOPK, device=DEV, weight_plan=wplan, quant_mode="w4a8_mx", swiglu_limit=10.0)
        heur = fm.Caps(**kw).decode_config
        x = torch.randn(T, K, device=DEV).to(torch.bfloat16)
        ids = torch.stack([torch.randperm(E, device=DEV)[:TOPK] for _ in range(T)]).to(torch.int32)
        w = torch.softmax(torch.randn(T, TOPK, device=DEV), -1)

        def run(cfg):
            caps = fm.Caps(**kw, decode_config=cfg, deterministic_output=True)
            plan = fm.plan(caps)
            scratch = tuple(torch.zeros(s.shape, dtype=s.dtype, device=s.device) for s in plan.scratch_specs())
            outs = []
            for _ in range(3):
                out = torch.empty(T, K, dtype=torch.bfloat16, device=DEV)
                fm.run(binding=fm.bind(plan, scratch=scratch, a=x, experts=experts, topk_weights=w, topk_ids=ids,
                                       output=out, input_scales_static=True))
                outs.append(out)
            torch.cuda.synchronize()
            return all(torch.equal(outs[0], o) for o in outs[1:])

        try:
            same = run(None)
            print(f"T={T}: heuristic backend={heur.backend} route={heur.dynamic_route_mode}: OK, repeats identical={same}")
        except RuntimeError as exc:
            bad += 1
            fixed = dataclasses.replace(heur, dynamic_route_mode="grouped")
            same = run(fixed)
            print(f"T={T}: heuristic backend={heur.backend} route={heur.dynamic_route_mode}: FAILS with "
                  f"deterministic_output=True: {exc}\n      same config with route=grouped: OK, repeats identical={same}")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
