#!/usr/bin/env python3
"""Throughput / latency load generator for any OpenAI-compatible endpoint (stdlib only; threads).

    python3 loadgen.py --url http://host:8000 [--concurrency 1,4,16,64] [--prompt-lens 512,4096,32768] \\
        [--output-len 256] [--requests-per-level 0 (= max(16, 3 x concurrency))] [--repeats 3] [--json out.json]

For each (prompt length, concurrency):
- An unmeasured warm-up with the same prompt and output lengths (min(concurrency, --warmup) requests at once), so
  lazily prepared kernels / plans of first-seen shapes are not timed.
- Then --repeats measured runs. Each sends `requests` streaming /v1/completions with random token-id prompts of exactly
  that length: the first --nonce-tokens ids come from a per-run nonce, the rest from --seed, so no prefix cache (of
  either engine, also across runs against the same server) can help, while the body stays the same for every engine.
- `max_tokens = output_len`, `ignore_eos: true` (both engines generate the same number of tokens), temperature 0,
  `stream_options.include_usage`. `concurrency` requests in flight (closed loop).

A request counts only if its stream completes (eval/sse.py: [DONE] or a finish_reason, no error event). Failed
requests are errors; the tokens they produced are reported separately as partial_out_tokens.

Per level (eval/measure.py): output tok/s over the wall time and in the steady-state window (all slots busy), each as
the median and range over the repeats; TTFT, TPOT and E2E p50 and tail (p99 from >= 100 samples, else the maximum,
over the samples of all repeats); per-user decode speed 1000 / TPOT p50; mean output tokens; errors.
Levels with prompt + output > the endpoint's max_model_len (from /v1/models, if reported) are skipped.

Token ids: random in [--id-lo, --id-hi). The default 1000..100000 avoids V4.1's special tokens (> 128000).
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import random
import statistics
import sys
import threading
import time
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from measure import combine, ms, pct, steady_tok_s, tail  # noqa: E402
from sse import read_stream  # noqa: E402


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
        try:
            c.request("GET", self.prefix + path)
            return json.loads(c.getresponse().read())
        finally:
            c.close()

    def stream(self, ids, max_tokens) -> dict:
        c = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        body = dict(model=self.model, prompt=ids, max_tokens=max_tokens, temperature=0.0, ignore_eos=True, stream=True,
                    stream_options={"include_usage": True})
        t0 = time.perf_counter()
        try:
            c.request("POST", self.prefix + "/v1/completions", json.dumps(body), {"Content-Type": "application/json"})
            r = c.getresponse()
            if r.status != 200:
                return dict(ok=False, error=f"HTTP {r.status}: {r.read()[:200]!r}", inp=len(ids), out=0, partial_out=0)
            s = read_stream(r)
        except Exception as exc:  # noqa: BLE001
            return dict(ok=False, error=f"{type(exc).__name__}: {exc}"[:200], inp=len(ids), out=0, partial_out=0)
        finally:
            c.close()
        t_end = time.perf_counter()
        n_out = (s.usage or {}).get("completion_tokens") or len(s.stamps)
        first, last = (s.stamps[0], s.stamps[-1]) if s.stamps else (None, None)
        if not s.ok:
            return dict(ok=False, error=s.error, inp=len(ids), out=0, partial_out=len(s.stamps))
        return dict(ok=True, t_send=t0, t_first=first, t_last=last, t_end=t_end, inp=len(ids), out=n_out,
                    ttft=(first - t0) if first else None, e2e=t_end - t0,
                    tpot=((last - first) / (n_out - 1)) if first and last and n_out > 1 and len(s.stamps) > 1 else None)


def prompts(n, plen, seed, nonce, nonce_tokens, lo, hi, tag):
    """n distinct prompts: --nonce-tokens ids from the run nonce, the rest from the seed (same for every engine)."""
    out = []
    for i in range(n):
        body = random.Random(f"{seed}/{tag}/{i}")
        head = random.Random(f"{nonce}/{tag}/{i}")
        k = min(nonce_tokens, plen)
        out.append([head.randrange(lo, hi) for _ in range(k)] + [body.randrange(lo, hi) for _ in range(plen - k)])
    return out


def run_pool(cl, prompts_, conc, olen):
    results, lock = [], threading.Lock()
    it = iter(prompts_)

    def worker():
        while True:
            with lock:
                ids = next(it, None)
            if ids is None:
                return
            res = cl.stream(ids, olen)
            with lock:
                results.append(res)

    t0 = time.perf_counter()
    ths = [threading.Thread(target=worker, daemon=True) for _ in range(conc)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    return results, time.perf_counter() - t0


def run_level(cl, plen, conc, nreq, olen, args):
    tag = f"{plen}x{conc}"
    warm = prompts(min(conc, args.warmup), plen, args.seed + 1, args.nonce + "-warm", args.nonce_tokens, args.id_lo,
                   args.id_hi, tag)
    if warm:
        run_pool(cl, warm, len(warm), olen)  # unmeasured: same lengths, different content
    reps, ok_all = [], []
    for rep in range(args.repeats):
        ps = prompts(nreq, plen, args.seed, f"{args.nonce}-r{rep}", args.nonce_tokens, args.id_lo, args.id_hi, tag)
        res, wall = run_pool(cl, ps, conc, olen)
        ok = [r for r in res if r["ok"]]
        bad = [r for r in res if not r["ok"]]
        ok_all += ok
        out, inp = sum(r["out"] for r in ok), sum(r["inp"] for r in ok)
        st, win = steady_tok_s(ok, conc)
        reps.append(dict(wall_s=wall, output_tok_s=out / wall, total_tok_s=(out + inp) / wall, steady_output_tok_s=st,
                         steady_window_s=win, ok=len(ok), errors=len(bad),
                         partial_out_tokens=sum(r["partial_out"] for r in bad),
                         error_samples=[r["error"] for r in bad][:3]))
    ttft, tpot, e2e = [r["ttft"] for r in ok_all], [r["tpot"] for r in ok_all], [r["e2e"] for r in ok_all]
    t_ttft, t_tpot, t_e2e = tail(ttft), tail(tpot), tail(e2e)
    tpot50 = pct(tpot, 50)
    lev = dict(prompt_len=plen, concurrency=conc, requests=nreq, repeats=args.repeats,
               ok=sum(r["ok"] for r in reps), errors=sum(r["errors"] for r in reps),
               partial_out_tokens=sum(r["partial_out_tokens"] for r in reps),
               error_samples=[e for r in reps for e in r["error_samples"]][:3],
               **combine(reps, ["output_tok_s", "steady_output_tok_s", "total_tok_s", "wall_s", "steady_window_s"]),
               ttft_p50=pct(ttft, 50), ttft_tail=t_ttft[0], ttft_tail_kind=t_ttft[1],
               tpot_p50_ms=ms(tpot50), tpot_tail_ms=ms(t_tpot[0]), tpot_tail_kind=t_tpot[1],
               e2e_p50=pct(e2e, 50), e2e_tail=t_e2e[0], e2e_tail_kind=t_e2e[1], samples=len(ok_all),
               tok_s_per_user=(1.0 / tpot50) if tpot50 else None,
               output_tokens_mean=statistics.fmean([r["out"] for r in ok_all]) if ok_all else None,
               per_repeat=reps)
    # legacy names (earlier JSON readers): p99 fields only when they really are p99s
    lev["ttft_p99"] = lev["ttft_tail"] if t_ttft[1] == "p99" else None
    lev["tpot_p99_ms"] = lev["tpot_tail_ms"] if t_tpot[1] == "p99" else None
    return lev


def f(x, spec="{:7.3f}"):
    return spec.format(x) if x is not None else "      -"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--model")
    ap.add_argument("--concurrency", default="1,4,16,64")
    ap.add_argument("--prompt-lens", default="512,4096,32768")
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--requests-per-level", type=int, default=0, help="0 = max(16, 3 x concurrency)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=8, help="warm-up requests per level (at most the concurrency)")
    ap.add_argument("--nonce", default=None, help="prompt nonce (default: random per run; recorded in the JSON)")
    ap.add_argument("--nonce-tokens", type=int, default=8)
    ap.add_argument("--id-lo", type=int, default=1000)
    ap.add_argument("--id-hi", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--json")
    args = ap.parse_args()
    args.nonce = args.nonce or os.urandom(6).hex()
    cl = Client(args.url, args.model, args.timeout)
    levels = []
    print(f"endpoint {args.url} model {cl.model} max_model_len {cl.max_model_len} nonce {args.nonce} repeats {args.repeats}")
    print(f"{'prompt':>7s} {'conc':>5s} {'req':>4s} {'err':>3s} | {'out tok/s':>9s} {'(min-max)':>15s} {'steady':>8s} | "
          f"{'TTFT p50':>8s} {'tail':>11s} | {'TPOT p50':>8s} {'tail':>11s} ms | {'tok/s/user':>10s} {'out mean':>8s}")
    for plen in [int(x) for x in args.prompt_lens.split(",")]:
        if cl.max_model_len and plen + args.output_len > cl.max_model_len:
            print(f"{plen:7d}  skipped: prompt + output > max_model_len {cl.max_model_len}")
            continue
        for conc in [int(x) for x in args.concurrency.split(",")]:
            nreq = args.requests_per_level or max(16, 3 * conc)
            r = run_level(cl, plen, conc, nreq, args.output_len, args)
            levels.append(r)
            rng = f"({f(r.get('output_tok_s_min'), '{:.0f}').strip()}-{f(r.get('output_tok_s_max'), '{:.0f}').strip()})"
            print(f"{plen:7d} {conc:5d} {nreq:4d} {r['errors']:3d} | {f(r['output_tok_s'], '{:9.1f}')} {rng:>15s} "
                  f"{f(r['steady_output_tok_s'], '{:8.1f}')} | {f(r['ttft_p50'], '{:8.3f}')} "
                  f"{f(r['ttft_tail'], '{:7.3f}')} {r['ttft_tail_kind'] or '':>3s} | {f(r['tpot_p50_ms'], '{:8.1f}')} "
                  f"{f(r['tpot_tail_ms'], '{:7.1f}')} {r['tpot_tail_kind'] or '':>3s}    | "
                  f"{f(r['tok_s_per_user'], '{:10.1f}')} {f(r['output_tokens_mean'], '{:8.1f}')}", flush=True)
            if r["errors"]:
                print(f"         errors {r['errors']} (partial output tokens {r['partial_out_tokens']}): "
                      f"{r['error_samples'][:2]}", flush=True)
    if args.json:
        json.dump(dict(url=args.url, model=cl.model, args=vars(args), levels=levels), open(args.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
