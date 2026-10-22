#!/usr/bin/env python3
"""TTFT / TPOT client for server.py (stdlib): a multi-turn streaming chat, then a concurrent burst.

    python3 /opt/sm120/server/bench_client.py --url http://127.0.0.1:8000 [--turns 4 --max-tokens 64 --concurrency 4]

TTFT = request sent -> first content chunk; TPOT = (last chunk - first chunk) / (completion_tokens - 1), with
completion_tokens from the final usage chunk (stream_options.include_usage). Each chat turn appends the previous
answer and a new user message, so with --prefix-cache turn n hits the snapshot taken at the end of turn n-1.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import statistics
import sys
import threading
import time
import urllib.parse

TOPICS = ["the history of lighthouses", "how tides work", "why the sky is blue", "the life of a ship's carpenter",
          "how oil lamps were maintained", "storms in the nineteenth century"]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval"))
from sse import read_stream  # noqa: E402


def stream_chat(host, port, model, messages, max_tokens, temperature=0.0):
    c = http.client.HTTPConnection(host, port, timeout=1200)
    body = dict(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature, stream=True,
                stream_options={"include_usage": True})
    t0 = time.perf_counter()
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    if r.status != 200:
        raise RuntimeError(f"HTTP {r.status}: {r.read()[:300]!r}")
    try:
        s = read_stream(r)  # eval/sse.py: decoded body (chunked transfer), error events, truncation
    finally:
        c.close()
    if not s.ok:
        raise RuntimeError(f"stream failed after {len(s.stamps)} chunks: {s.error}")
    usage, text = s.usage, [s.text]
    t_first, t_last = (s.stamps[0], s.stamps[-1]) if s.stamps else (None, None)
    n = (usage or {}).get("completion_tokens", 0)
    ttft = (t_first - t0) if t_first else None
    tpot = ((t_last - t_first) / (n - 1)) if (t_first and n > 1) else None
    return dict(ttft=ttft, tpot=tpot, tokens=n, prompt_tokens=(usage or {}).get("prompt_tokens"),
                total=time.perf_counter() - t0, text="".join(text))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    u = urllib.parse.urlparse(args.url)
    host, port = u.hostname, u.port or 80
    c = http.client.HTTPConnection(host, port, timeout=60)
    c.request("GET", "/v1/models")
    model = json.loads(c.getresponse().read())["data"][0]["id"]

    def health():
        c = http.client.HTTPConnection(host, port, timeout=60)
        c.request("GET", "/health")
        return json.loads(c.getresponse().read())

    out = dict(model=model, turns=[], burst=None)
    # 1. multi-turn chat
    msgs = [{"role": "system", "content": "You are a concise assistant. " * 20}]
    for t in range(args.turns):
        msgs.append({"role": "user", "content": f"Turn {t + 1}: tell me about {TOPICS[t % len(TOPICS)]}, in detail."})
        h0 = (health().get("prefix_cache") or {})
        r = stream_chat(host, port, model, msgs, args.max_tokens)
        h1 = (health().get("prefix_cache") or {})
        r["cache_hit_tokens"] = (h1.get("hit_tokens", 0) - h0.get("hit_tokens", 0)) if h1 else None
        msgs.append({"role": "assistant", "content": r.pop("text")})
        out["turns"].append(r)
        print(f"[turn {t + 1}] prompt {r['prompt_tokens']} tok, cached {r['cache_hit_tokens']}, TTFT "
              f"{r['ttft']:.3f}s, TPOT {r['tpot'] * 1e3 if r['tpot'] else float('nan'):.1f} ms, {r['tokens']} tok",
              flush=True)
    # 2. concurrent burst of independent chats
    res = [None] * args.concurrency
    t0 = time.perf_counter()

    def worker(i):
        m = [{"role": "user", "content": f"Request {i}: summarize {TOPICS[i % len(TOPICS)]} in a paragraph."}]
        res[i] = stream_chat(host, port, model, m, args.max_tokens)

    ths = [threading.Thread(target=worker, args=(i,)) for i in range(args.concurrency)]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    wall = time.perf_counter() - t0
    toks = sum(r["tokens"] for r in res)
    tp = [r["tpot"] for r in res if r["tpot"]]
    out["burst"] = dict(concurrency=args.concurrency, wall_s=wall, tokens=toks, tok_per_s=toks / wall,
                        ttft_median=statistics.median(r["ttft"] for r in res),
                        tpot_median_ms=statistics.median(tp) * 1e3 if tp else None)
    print(f"[burst x{args.concurrency}] {toks} tokens in {wall:.2f}s = {toks / wall:.1f} tok/s; median TTFT "
          f"{out['burst']['ttft_median']:.3f}s, median TPOT {out['burst']['tpot_median_ms'] or float('nan'):.1f} ms")
    out["health"] = health()
    print("[health]", json.dumps(out["health"])[:600])
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
