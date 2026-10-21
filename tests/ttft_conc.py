"""TTFT with fresh random prompts (no reuse): for each (length, concurrency), C streamed completions at once."""
import argparse, http.client, json, random, statistics, threading, time
ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--cases", default="4096x1,4096x1,32768x1,4096x4,32768x4")
ap.add_argument("--max-tokens", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tag", default="")
a = ap.parse_args()
rng = random.Random(a.seed)
def one(ids, out, k):
    c = http.client.HTTPConnection("127.0.0.1", a.port, timeout=3600)
    t0 = time.perf_counter()
    c.request("POST", "/v1/completions", json.dumps(dict(model="m", prompt=ids, max_tokens=a.max_tokens, ignore_eos=True,
                                                          temperature=0, stream=True)), {"Content-Type": "application/json"})
    r = c.getresponse(); tf = None
    while True:
        line = r.fp.readline()
        if not line: break
        if tf is None and line.startswith(b"data: ") and line.strip() != b"data: [DONE]": tf = time.perf_counter()
    out[k] = (tf - t0) if tf else float("nan")
for case in a.cases.split(","):
    n, cc = (int(x) for x in case.split("x"))
    prompts = [[rng.randrange(1000, 120000) for _ in range(n)] for _ in range(cc)]
    out = [None] * cc
    t0 = time.perf_counter()
    th = [threading.Thread(target=one, args=(p, out, k)) for k, p in enumerate(prompts)]
    for t in th: t.start()
    for t in th: t.join()
    wall = time.perf_counter() - t0
    print(f"[ttft]{a.tag} prompt {n:6d} x{cc}: TTFT mean {statistics.mean(out):7.2f}s max {max(out):7.2f}s "
          f"min {min(out):7.2f}s  wall {wall:6.2f}s  ({n * cc / wall:6.0f} prompt tok/s)", flush=True)
