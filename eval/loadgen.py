#!/usr/bin/env python3
"""Throughput / latency load generator for any OpenAI-compatible endpoint (stdlib only; threads).

    python3 loadgen.py --url http://host:8000 [--concurrency 1,4,16,64] [--prompt-lens 512,4096,32768] \
        [--output-len 256] [--requests-per-level 0 (= max(8, 2 x concurrency))] [--json out.json]

For each (prompt length, concurrency):
- Sends `requests` streaming /v1/completions with random token-id prompts of exactly that length. They are unique per
  request, so prefix caches cannot help.
- Uses `max_tokens = output_len`, `ignore_eos: true`, temperature 0, `stream_options.include_usage`.
- Keeps `concurrency` requests in flight (closed loop), after one warm-up request per prompt length.

Measured per request:
- TTFT (send -> first text chunk);
- TPOT = (last chunk - first chunk) / (completion_tokens - 1);
- E2E.

Per level: p50 / p99 of each, output tok/s and total (prompt + output) tok/s over the level's wall time, errors.
Levels with prompt + output > the endpoint's max_model_len (from /v1/models, if reported) are skipped.

Token ids: random in [--id-lo, --id-hi). The default 1000..100000 avoids V4.1's special tokens (> 128000).
"""

from __future__ import annotations

import argparse
import http.client
import json
import random
import statistics
import sys
import threading
import time
import urllib.parse


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, int(round(q / 100 * (len(xs) - 1)))))
    return xs[i]


class Client:
    def __init__(self, url, model, timeout):
        u = urllib.parse.urlparse(url)
        self.host, self.port = u.hostname, u.port or 80
        self.prefix = u.path.rstrip("/").removesuffix("/v1")
        self.timeout = timeout
        info = self.get("/v1/models")["data"][0]
        self.model = model or info["id"]
        self.max_model_len = info.get("max_model_len")

    def get(self, path):
        c = http.client.HTTPConnection(self.host, self.port, timeout=60)
        c.request("GET", self.prefix + path)
        return json.loads(c.getresponse().read())

    def stream(self, ids, max_tokens):
        c = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        body = dict(model=self.model, prompt=ids, max_tokens=max_tokens, temperature=0.0, ignore_eos=True, stream=True,
                    stream_options={"include_usage": True})
        t0 = time.perf_counter()
        c.request("POST", self.prefix + "/v1/completions", json.dumps(body), {"Content-Type": "application/json"})
        r = c.getresponse()
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}: {r.read()[:200]!r}")
        t_first = t_last = None
        usage = None
        n_chunks = 0
        while True:
            line = r.fp.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data: "):
                continue
            if line[6:] == b"[DONE]":
                break
            ev = json.loads(line[6:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                if ch.get("text"):
                    now = time.perf_counter()
                    t_first = t_first or now
                    t_last = now
                    n_chunks += 1
        c.close()
        t_end = time.perf_counter()
        n_out = (usage or {}).get("completion_tokens") or n_chunks
        return dict(ttft=(t_first - t0) if t_first else None, e2e=t_end - t0, out=n_out, inp=len(ids),
                    tpot=((t_last - t_first) / (n_out - 1)) if (t_first and n_out > 1) else None)


def run_level(cl, plen, conc, nreq, olen, rng, lo, hi):
    prompts = [[rng.randrange(lo, hi) for _ in range(plen)] for _ in range(nreq)]
    results, errors = [], []
    lock = threading.Lock()
    it = iter(prompts)

    def worker():
        while True:
            with lock:
                ids = next(it, None)
            if ids is None:
                return
            try:
                res = cl.stream(ids, olen)
                with lock:
                    results.append(res)
            except Exception as exc:  # noqa: BLE001
                with lock:
                    errors.append(repr(exc)[:200])

    t0 = time.perf_counter()
    ths = [threading.Thread(target=worker) for _ in range(conc)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.perf_counter() - t0
    ttft = [r["ttft"] for r in results if r["ttft"] is not None]
    tpot = [r["tpot"] for r in results if r["tpot"] is not None]
    e2e = [r["e2e"] for r in results]
    out = sum(r["out"] for r in results)
    inp = sum(r["inp"] for r in results)
    return dict(prompt_len=plen, concurrency=conc, requests=nreq, ok=len(results), errors=len(errors),
                error_samples=errors[:3], wall_s=wall, output_tok_s=out / wall, total_tok_s=(out + inp) / wall,
                ttft_p50=pct(ttft, 50), ttft_p99=pct(ttft, 99), tpot_p50_ms=(pct(tpot, 50) or 0) * 1e3,
                tpot_p99_ms=(pct(tpot, 99) or 0) * 1e3, e2e_p50=pct(e2e, 50), e2e_p99=pct(e2e, 99),
                tpot_mean_ms=statistics.fmean(tpot) * 1e3 if tpot else None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--model")
    ap.add_argument("--concurrency", default="1,4,16,64")
    ap.add_argument("--prompt-lens", default="512,4096,32768")
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--requests-per-level", type=int, default=0, help="0 = max(8, 2 x concurrency)")
    ap.add_argument("--id-lo", type=int, default=1000)
    ap.add_argument("--id-hi", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--json")
    args = ap.parse_args()
    cl = Client(args.url, args.model, args.timeout)
    rng = random.Random(args.seed)
    levels = []
    print(f"endpoint {args.url} model {cl.model} max_model_len {cl.max_model_len}")
    print(f"{'prompt':>7s} {'conc':>5s} {'req':>4s} {'err':>3s} | {'out tok/s':>9s} {'total tok/s':>11s} | "
          f"{'TTFT p50':>8s} {'p99':>7s} | {'TPOT p50':>8s} {'p99':>7s} ms | {'E2E p50':>7s} {'p99':>7s}")
    for plen in [int(x) for x in args.prompt_lens.split(",")]:
        if cl.max_model_len and plen + args.output_len > cl.max_model_len:
            print(f"{plen:7d}  skipped: prompt + output > max_model_len {cl.max_model_len}")
            continue
        try:
            cl.stream([rng.randrange(args.id_lo, args.id_hi) for _ in range(plen)], 8)  # warm-up (JIT, graphs)
        except Exception as exc:  # noqa: BLE001
            print(f"{plen:7d}  warm-up failed: {exc!r}"[:200])
            continue
        for conc in [int(x) for x in args.concurrency.split(",")]:
            nreq = args.requests_per_level or max(8, 2 * conc)
            r = run_level(cl, plen, conc, nreq, args.output_len, rng, args.id_lo, args.id_hi)
            levels.append(r)
            f = lambda x, s="{:7.3f}": s.format(x) if x is not None else "      -"  # noqa: E731
            print(f"{plen:7d} {conc:5d} {nreq:4d} {r['errors']:3d} | {r['output_tok_s']:9.1f} {r['total_tok_s']:11.1f} | "
                  f"{f(r['ttft_p50'], '{:8.3f}')} {f(r['ttft_p99'])} | {r['tpot_p50_ms']:8.1f} {r['tpot_p99_ms']:7.1f}    | "
                  f"{f(r['e2e_p50'])} {f(r['e2e_p99'])}", flush=True)
    if args.json:
        json.dump(dict(url=args.url, model=cl.model, args=vars(args), levels=levels), open(args.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
