"""Standalone repro: b12x fused_moe (w4a8_mx) tiny-decode path ignores swiglu_limit.

Random MXFP4 expert weights (no checkpoint needed). Runs the same T=1 problem twice:
default path selection (tiny-decode for T<=4) and B12X_W4A8_TINY_DECODE=0 (dynamic),
and compares both against an fp32 torch reference with and without the SwiGLU clamp.

Works with standalone b12x (PyPI) and with FlashInfer's bundled b12x (`import b12x`).
Usage: python repro_tiny_decode_swiglu.py
"""

import os

import torch
import torch.nn.functional as F

import b12x

E, K, N, TOPK, T = 8, 2048, 512, 2, 1  # K % 256 == 0, N % 32 == 0 -> tiny-decode eligible
LIMIT = 10.0
X_STD = 8.0  # gives roughly a third of gate/up values beyond LIMIT with these weights
DEV = torch.device("cuda", 0)  # b12x scratch checks require an explicit device index

FP4 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], device=DEV)


def dequant(packed, scale_u8):
    """uint8 [R, C/2] (low nibble = even column) + UE8M0 [R, C/32] -> fp32 [R, C]."""
    r, half = packed.shape
    vals = torch.stack((FP4[(packed & 0xF).long()], FP4[(packed >> 4).long()]), -1).reshape(r, 2 * half)
    sc = torch.exp2(scale_u8.float() - 127.0)
    return (vals.view(r, -1, 32) * sc.unsqueeze(-1)).reshape(r, -1)


def reference(x, ids, w, w13, s13, w2, s2, clamp):
    out = torch.zeros(x.shape[0], K, device=DEV)
    hits = tot = 0
    for t in range(x.shape[0]):
        for j in range(TOPK):
            e = int(ids[t, j])
            h = x[t].float() @ dequant(w13[e], s13[e]).T
            g, u = h[:N], h[N:]
            hits += int((g > LIMIT).sum() + (u.abs() > LIMIT).sum())
            tot += 2 * N
            if clamp:
                g, u = g.clamp(max=LIMIT), u.clamp(-LIMIT, LIMIT)
            out[t] += w[t, j] * ((F.silu(g) * u) @ dequant(w2[e], s2[e]).T)
    return out, hits / tot


def run_b12x(env_tiny, x, ids, w, w13, s13, w2, s2):
    os.environ["B12X_W4A8_TINY_DECODE"] = env_tiny
    fm = b12x.moe.fused_moe
    wplan = fm.plan_weights(quant_modes="w4a8_mx", source_format="fp4_e8m0_k32", activation="silu",
                            params_dtype=torch.bfloat16, num_experts=E, hidden_size=K,
                            intermediate_size=N, w13_layout="w31")
    ones = torch.ones(E, device=DEV)
    experts = fm.prepare_weights(plan=wplan, params_dtype=torch.bfloat16,
                                 w1_fp4=w13.clone(), w2_fp4=w2.clone(),  # repacked in place
                                 w1_blockscale=s13.clone(), w2_blockscale=s2.clone(),
                                 w1_global_scale=ones, w2_global_scale=ones,
                                 a1_gscale=ones, a2_gscale=ones)
    caps = fm.Caps(max_tokens=T, num_topk=TOPK, device=DEV, weight_plan=wplan,
                   quant_mode="w4a8_mx", swiglu_limit=LIMIT)
    plan = fm.plan(caps)
    scratch = tuple(torch.zeros(s.shape, dtype=s.dtype, device=s.device) for s in plan.scratch_specs())
    out = torch.zeros(T, K, dtype=torch.bfloat16, device=DEV)
    b = fm.bind(plan, scratch=scratch, a=x, experts=experts, topk_weights=w, topk_ids=ids,
                output=out, input_scales_static=True)
    fm.run(binding=b)
    torch.cuda.synchronize()
    return out.float(), caps.decode_config.backend, plan.launch_plan.implementation


def cos(a, b):
    return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def main():
    torch.manual_seed(0)
    print(f"b12x from {b12x.__file__}  torch {torch.__version__}  {torch.cuda.get_device_name()}")
    # MXFP4 weights: random nibbles, UE8M0 scale 2^-7 (byte 120) like real checkpoints (119..122)
    w13 = torch.randint(0, 256, (E, 2 * N, K // 2), dtype=torch.uint8, device=DEV)
    w2 = torch.randint(0, 256, (E, K, N // 2), dtype=torch.uint8, device=DEV)
    s13 = torch.full((E, 2 * N, K // 32), 120, dtype=torch.uint8, device=DEV)
    s2 = torch.full((E, K, N // 32), 120, dtype=torch.uint8, device=DEV)
    x = (torch.randn(T, K, device=DEV) * X_STD).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(E, device=DEV)[:TOPK] for _ in range(T)]).to(torch.int32)
    w = torch.softmax(torch.randn(T, TOPK, device=DEV), -1)

    ref_c, frac = reference(x, ids, w, w13, s13, w2, s2, clamp=True)
    ref_u, _ = reference(x, ids, w, w13, s13, w2, s2, clamp=False)
    print(f"swiglu_limit={LIMIT}, fraction of gate/up values beyond limit: {frac:.1%}")

    ok = True
    for env in ("1", "0"):
        out, backend, impl = run_b12x(env, x, ids, w, w13, s13, w2, s2)
        cc, cu = cos(out, ref_c), cos(out, ref_u)
        good = cc > 0.99
        ok &= good
        print(f"B12X_W4A8_TINY_DECODE={env}: backend={backend} impl={impl}  "
              f"cos vs clamped ref={cc:.4f}  cos vs UNclamped ref={cu:.4f}  -> {'OK' if good else 'BUG: clamp not applied'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
