"""Option-B EP MoE (sm120_ep) on ONE GPU.

    PYTHONPATH=/opt/sm120/sm120:/opt/deepinfer python3 /opt/sm120/tests/test_ep_moe.py      # all levels
    PYTHONPATH=/opt/sm120/sm120:/opt/deepinfer pytest -q /opt/sm120/tests/test_ep_moe.py

Experts: sm120_moe.B12xRoutedExperts (the production b12x fused_moe wrapper) built from random MXFP4 weights in
DeepInfer's parameter layout (int8 e2m1 [E, 2n, K/2] + int32 packed ue8m0 .sf), E=12 global experts, top-6.
  level 1  in-process: sum over W emulated ranks of the masked local partials == full-expert run
  level 2  W=2 processes on the same GPU, gloo transport, EPRoutedExperts end to end, different token counts per
           rank, prefill mode (count exchange) and decode mode (fixed capacity)
  level 3  W=1 NCCL group: the fixed-capacity path captured in a CUDA graph == eager
"""

from __future__ import annotations

import os
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "sm120"))
os.environ.setdefault("B12X_AUTOTUNE", "0")
os.environ.setdefault("B12X_W4A8_TINY_DECODE", "0")  # tiny-decode ignores swiglu_limit (feasibility 10.4); unused here

E, K, N, TOPK = 12, 2048, 512, 6  # V4.1 top-6; E divisible by 2 and 4 ranks


class FakeMoE:
    pass


def make_weights(seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)
    w13 = torch.randint(-128, 128, (E, 2 * N, K // 2), generator=g, dtype=torch.int8)
    s13 = torch.randint(118, 124, (E, 2 * N, K // 32), generator=g, dtype=torch.int32).to(torch.uint8)
    w2 = torch.randint(-128, 128, (E, K, N // 2), generator=g, dtype=torch.int8)
    s2 = torch.randint(118, 124, (E, K, N // 32), generator=g, dtype=torch.int32).to(torch.uint8)
    return w13, s13, w2, s2


def make_moe(weights, experts: range, device, layer_idx=7):
    """Fake DeepseekMoE with the given (physical) experts, in DeepInfer's runtime parameter layout."""
    w13, s13, w2, s2 = weights
    sl = slice(experts.start, experts.stop)
    m = FakeMoE()
    m.layer_idx = layer_idx
    m.use_fp4_weight = True
    m.num_physical_topk = TOPK
    m.swiglu_clamp_value = None
    p13 = w13[sl].clone().to(device)
    p13.sf = s13[sl].contiguous().view(torch.int32).to(device)  # [e, 2n, K/128] packed
    p2 = w2[sl].clone().to(device)
    p2.sf = s2[sl].contiguous().view(torch.int32).to(device)
    m.gate_up_proj_weight, m.down_proj_weight = p13, p2
    return m


def make_tokens(T: int, seed: int, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn((T, K), generator=g) * 0.5).to(torch.bfloat16).to(device)
    scores = torch.rand((T, E), generator=g)
    ids = scores.topk(TOPK, -1).indices.to(torch.int64).to(device)
    w = torch.softmax(torch.randn((T, TOPK), generator=g), -1).float().to(device)
    return x, ids, w


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


# ---------------------------------------------------------------------------------------------- level 1
def level1(world: int = 4):
    import sm120_moe
    from sm120_ep.moe import EPRoutedExperts
    dev = torch.device("cuda", 0)
    wts = make_weights(1)
    full = sm120_moe.B12xRoutedExperts(make_moe(wts, range(0, E), dev))
    nl = E // world
    locs = [sm120_moe.B12xRoutedExperts(make_moe(wts, range(r * nl, (r + 1) * nl), dev)) for r in range(world)]
    res = []
    for T in (1, 7, 64, 300):
        x, ids, w = make_tokens(T, 100 + T, dev)
        ref = full(x, ids, w)
        acc = torch.zeros((T, K), dtype=torch.float32, device=dev)
        for r in range(world):
            helper = EPRoutedExperts.__new__(EPRoutedExperts)
            helper.first, helper.n_local = r * nl, nl
            acc += locs[r](x, helper._local_ids(ids.to(torch.int32)), w).float()
        c = cos(acc, ref)
        res.append((T, c))
        assert c > 0.9999, (T, c)
    print(f"[level1] W={world} sum of masked local partials vs full experts: " +
          ", ".join(f"T={T} cos {c:.6f}" for T, c in res), flush=True)


# ---------------------------------------------------------------------------------------------- level 2
def _level2_worker(rank: int, world: int, init_file: str, q):
    import torch.distributed as dist
    import sm120_moe
    from sm120_ep.moe import EPRoutedExperts
    from sm120_ep.transport import GlooTransport
    try:
        torch.cuda.set_device(0)
        dev = torch.device("cuda", 0)
        dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
        wts = make_weights(2)
        nl = E // world
        local = sm120_moe.B12xRoutedExperts(make_moe(wts, range(rank * nl, (rank + 1) * nl), dev))
        full = sm120_moe.B12xRoutedExperts(make_moe(wts, range(0, E), dev))
        ep = EPRoutedExperts(local, group=None, num_local_experts=nl, ep_rank=rank, hidden=K, topk=TOPK,
                             decode_rows=16, transport=GlooTransport(dist.group.WORLD))
        out = []
        for name, T, is_prompt in (("prefill", 37 + 26 * rank, True), ("decode", 3 + 9 * rank, False),
                                   ("prefill-empty", 0 if rank == 1 else 20, True)):
            x, ids, w = make_tokens(T, 1000 * rank + T, dev)
            got = ep(x, ids, w, is_prompt=is_prompt)
            assert got.shape == (T, K), got.shape
            if T:
                ref = full(x, ids, w)
                out.append((name, T, cos(got, ref), (got.float() - ref.float()).abs().max().item()))
            else:
                out.append((name, T, 1.0, 0.0))
        dist.barrier()
        q.put((rank, "ok", out))
    except Exception as exc:  # noqa: BLE001
        import traceback
        q.put((rank, "fail", traceback.format_exc()[-2000:]))


def level2(world: int = 2):
    import torch.multiprocessing as mp
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    with tempfile.TemporaryDirectory() as td:
        init_file = os.path.join(td, "pg")
        procs = [ctx.Process(target=_level2_worker, args=(r, world, init_file, q)) for r in range(world)]
        for p in procs:
            p.start()
        results = [q.get(timeout=1800) for _ in procs]
        for p in procs:
            p.join(60)
    ok = True
    for rank, status, payload in sorted(results):
        if status != "ok":
            print(f"[level2] rank {rank} FAILED:\n{payload}", flush=True)
            ok = False
            continue
        for name, T, c, d in payload:
            print(f"[level2] W={world} rank {rank} {name:<14} T={T:3d}: cos vs full-expert {c:.6f} max|d| {d:.4f}",
                  flush=True)
            ok &= c > 0.9999
    assert ok, "level 2 failed"


# ---------------------------------------------------------------------------------------------- level 3
def level3():
    import torch.distributed as dist
    import sm120_moe
    from sm120_ep.moe import EPRoutedExperts
    from sm120_ep.transport import NcclTransport
    dev = torch.device("cuda", 0)
    with tempfile.TemporaryDirectory() as td:
        dist.init_process_group("nccl", init_method=f"file://{os.path.join(td, 'pg')}", rank=0, world_size=1,
                                device_id=dev)
        try:
            wts = make_weights(3)
            inner = sm120_moe.B12xRoutedExperts(make_moe(wts, range(0, E), dev))
            ep = EPRoutedExperts(inner, group=None, num_local_experts=E, ep_rank=0, hidden=K, topk=TOPK,
                                 decode_rows=16, transport=NcclTransport(dist.group.WORLD))
            x, ids, w = make_tokens(8, 5, dev)
            ref = inner(x, ids, w)
            eager = ep(x, ids, w, is_prompt=False)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                ep(x, ids, w, is_prompt=False)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                out_g = ep(x, ids, w, is_prompt=False)
            x2, ids2, w2 = make_tokens(8, 6, dev)
            x.copy_(x2)
            ids.copy_(ids2)
            w.copy_(w2)
            g.replay()
            torch.cuda.synchronize()
            ref2 = inner(x, ids, w)
            c1, c2 = cos(eager, ref), cos(out_g, ref2)
            print(f"[level3] W=1 NCCL fixed-capacity path: eager vs inner cos {c1:.6f}; CUDA-graph replay with new "
                  f"inputs vs inner cos {c2:.6f}", flush=True)
            assert c1 > 0.99999 and c2 > 0.99999
        finally:
            dist.destroy_process_group()


def test_level1():
    level1()


def test_level2():
    level2()


def test_level3():
    level3()


if __name__ == "__main__":
    level1()
    level3()
    level2()
    print("ALL OK", flush=True)
