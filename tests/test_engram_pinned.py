"""SM120_ENGRAM_MODE=pinned check: one memfd segment (both tables truncated to SM120_ENGRAM_SHM_MAX_ROWS rows),
two rank processes register it and gather on the GPU; memory counted once; rows bit-exact against the shard files."""
import multiprocessing as mp, os, sys, time
# usage (container, GPU): SM120_ENGRAM_MODE=pinned [SM120_ENGRAM_SHM_MAX_ROWS=32000000] python3 tests/test_engram_pinned.py [CKPT]
#   --ulimit memlock=-1; SM120_ENGRAM_SHM_MAX_ROWS truncates the tables so that they fit a small machine.
#   Prints: memory before/after (the segment must count once for both ranks), registration and fill times, GPU gather
#   speed, and "exact (True, True, ...)" per rank: rows bit-identical to preads of the shard, graph replay == eager.
CKPT = sys.argv[1] if len(sys.argv) > 1 else "/ckpt"
os.environ["SM120_ENGRAM_MODE"] = "pinned"  # the serving image defaults to host
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sm120"))
def mi():
    d = {l.split(":")[0]: int(l.split()[1]) / 2**20 for l in open("/proc/meminfo")}
    return " ".join(f"{k} {d[k]:.1f}" for k in ("MemAvailable", "Shmem", "Mlocked", "AnonPages", "Cached"))
def rank(r, q, filled, done):
    import torch, numpy as np, sm120_engram as E
    from deep_ep._engram import EngramBuffer
    torch.cuda.set_device(0)
    tabs = []
    for lid in E._engram_layer_ids(CKPT):
        w, s = E.find_engram_tensors(CKPT, lid)
        tabs.append(E.open_table(None, lid, wloc=w, sloc=s, mode="pinned", device="cuda"))
    q.put((r, "registered", mi()))
    if r == 0:
        q.put((r, "fill", E.fill_shared_tables()))
        filled.set()
    filled.wait()
    ok, g = True, torch.Generator().manual_seed(r)
    for t in tabs:
        ids = torch.randint(-1, t.rows, (98304,), generator=g)
        ow = torch.empty((ids.numel(), t.row_bytes), dtype=torch.uint8, device="cuda")
        os_ = torch.empty((ids.numel(), t.groups), dtype=torch.uint8, device="cuda")
        t.gather_cuda(ids.cuda(), ow, os_); torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10): t.gather_cuda(ids.cuda(), ow, os_)
        torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / 10
        # reference: plain preads of the shard
        w, s = t.source
        ref_w = np.zeros((ids.numel(), t.row_bytes), np.uint8); ref_s = np.full((ids.numel(), t.groups), 127, np.uint8)
        fd = os.open(w.path, os.O_RDONLY)
        for k, i in enumerate(ids.tolist()):
            if i >= 0:
                ref_w[k] = np.frombuffer(os.pread(fd, t.row_bytes, w.offset + i * t.row_bytes), np.uint8)
                ref_s[k] = np.frombuffer(os.pread(fd, t.groups, s.offset + i * t.groups), np.uint8)
        os.close(fd)
        ok &= bool((ow.cpu().numpy() == ref_w).all() and (os_.cpu().numpy() == ref_s).all())
        q.put((r, f"{t.name}: GPU gather of 98304 rows {dt * 1e3:.2f} ms ({98304 * 264 / dt / 1e9:.1f} GB/s)", None))
    # graph capture: an EngramBuffer over the pinned tables, fetch inside a CUDA graph vs eager
    buf = EngramBuffer(None); buf.set_config([t.rows for t in tabs], 256, 64, 24, torch.float8_e4m3fn, 2); buf.attach_tables(tabs)
    idx = torch.randint(0, min(t.rows for t in tabs), (len(tabs), 16, 24), dtype=torch.int64).to(torch.int32).cuda()
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        [h() for h in buf.fetch(idx)]
    torch.cuda.current_stream().wait_stream(st)
    graph, outs = torch.cuda.CUDAGraph(), []
    with torch.cuda.graph(graph, stream=st):
        outs = [E.dequant_rows(*h(), 256) for h in buf.fetch(idx)]
    new = torch.randint(0, min(t.rows for t in tabs), idx.shape, dtype=torch.int64).to(torch.int32).cuda()
    idx.copy_(new); graph.replay()
    eager = [E.dequant_rows(*h(), 256) for h in buf.fetch(new)]; torch.cuda.synchronize()
    gok = all(torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in zip(outs, eager))
    buf.fetcher.owns_tables = False
    q.put((r, "exact", (ok, gok, mi())))
    done.wait()
if __name__ == "__main__":
    import sm120_engram as E
    print("before:", mi(), flush=True)
    fd = E.create_shared_tables(CKPT, ranks=2)
    print("segment created:", mi(), flush=True)
    ctx = mp.get_context("spawn")
    q, filled, done = ctx.Queue(), ctx.Event(), ctx.Event()
    ps = [ctx.Process(target=rank, args=(r, q, filled, done)) for r in range(2)]
    for p in ps: p.start()
    left, all_ok = 2, True
    while left:
        try:
            r, what, info = q.get(timeout=900)
        except Exception:  # a rank died (its traceback is above) or hangs
            print("FAIL: a rank did not report", flush=True)
            for p in ps:
                p.kill()
            sys.exit(1)
        print(f"rank {r}: {what} {info if info is not None else ''}", flush=True)
        if what == "exact":
            left -= 1
            all_ok &= bool(info[0] and info[1])
    print("both ranks done:", mi(), flush=True)
    done.set()
    for p in ps: p.join()
    os.close(fd)
    print("after exit:", mi(), flush=True)
    sys.exit(0 if all_ok else 1)
