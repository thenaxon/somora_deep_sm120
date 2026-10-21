#!/usr/bin/env python3
"""Plumbing tests for server.py against a running server (stdlib client; outputs of truncated models are garbage).

    python3 /opt/sm120/tests/test_server.py --url http://127.0.0.1:8000 [--di-ref /work/di_out.pt --prompt-ids /work/prompt_ids.json]

Checks: models/health; greedy non-streaming == run_deepinfer.py greedy tokens (optional, --di-ref); streaming deltas
concatenate to the non-streaming text; stop strings; chat (stream + non-stream); N concurrent streams that overlap
in time; a request submitted while another decodes finishes first; cancelling a stream (client disconnect) aborts it
in the engine and the server keeps serving; sampling with temperature/top_p/seed; bad requests -> 400.
"""

from __future__ import annotations

import argparse
import http.client
import json
import sys
import threading
import time
import urllib.parse

RESULTS: list[tuple[str, bool, str]] = []


SOFT: set[str] = set()


def check(name: str, ok: bool, info: str = "") -> bool:
    if name in SOFT and not ok:
        print(f"[INFO] {name}  {info} (expected without --deterministic)", flush=True)
        return True
    RESULTS.append((name, bool(ok), info))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {info}", flush=True)
    return ok


class Client:
    def __init__(self, url: str):
        u = urllib.parse.urlparse(url)
        self.host, self.port = u.hostname, u.port or 80

    def conn(self, timeout=600):
        return http.client.HTTPConnection(self.host, self.port, timeout=timeout)

    def get(self, path):
        c = self.conn()
        c.request("GET", path)
        r = c.getresponse()
        return r.status, json.loads(r.read() or b"{}")

    def post(self, path, body):
        c = self.conn()
        c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
        r = c.getresponse()
        return r.status, json.loads(r.read() or b"{}")

    def stream(self, path, body, on_event=None, stop_after=None):
        """-> (events, t_first, t_end, raw_done). stop_after=N: close the socket after N data events."""
        c = self.conn()
        c.request("POST", path, json.dumps(dict(body, stream=True)), {"Content-Type": "application/json"})
        r = c.getresponse()
        assert r.status == 200, (r.status, r.read())
        events, t_first, done = [], None, False
        buf = b""
        while True:
            line = r.fp.readline()
            if not line:
                break
            buf = line.strip()
            if not buf.startswith(b"data: "):
                continue
            data = buf[6:]
            if data == b"[DONE]":
                done = True
                break
            ev = json.loads(data)
            if t_first is None:
                t_first = time.perf_counter()
            events.append(ev)
            if on_event:
                on_event(ev)
            if stop_after is not None and len(events) >= stop_after:
                r.close()
                c.close()  # client-side cancellation
                break
        return events, t_first, time.perf_counter(), done


def text_of(events, chat):
    if chat:
        return "".join((e["choices"][0]["delta"].get("content") or "") for e in events if e.get("choices"))
    return "".join(e["choices"][0]["text"] for e in events if e.get("choices"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--di-ref", default=None, help="run_deepinfer.py dump on the same ckpt (greedy token check)")
    ap.add_argument("--prompt-ids", default=None, help="prompt ids json used for that dump")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--prefix-cache", action="store_true", help="server runs --prefix-cache: test hits")
    ap.add_argument("--nondeterministic-ok", action="store_true",
                    help="server runs b12x MoE (not --deterministic): report repeatability checks as INFO")
    args = ap.parse_args()
    cl = Client(args.url)
    if args.nondeterministic_ok:
        SOFT.update({"greedy is deterministic", "greedy == run_deepinfer.py (token ids)",
                     "completion stream == non-stream text", "chat stream", "stop string (stream)", "stop string (non-stream)",
                     "seeded sampling reproducible", "prefix cache: same tokens on a hit"})

    st, models = cl.get("/v1/models")
    check("GET /v1/models", st == 200 and models["data"], json.dumps(models)[:120])
    model = models["data"][0]["id"]
    st, h = cl.get("/health")
    check("GET /health", st == 200 and h["status"] == "ok")

    # 1. greedy non-streaming, optionally against run_deepinfer.py
    prompt = "The lighthouse keeper wrote in his logbook that"
    base = dict(model=model, prompt=prompt, max_tokens=24, temperature=0, ignore_eos=True, return_token_ids=True)
    st, r = cl.post("/v1/completions", base)
    check("completion non-stream", st == 200 and r["choices"][0]["finish_reason"] == "length"
          and r["usage"]["completion_tokens"] == 24, f"{r['usage']} {r['choices'][0]['text'][:60]!r}")
    greedy_text = r["choices"][0]["text"]
    st, r2 = cl.post("/v1/completions", base)
    check("greedy is deterministic", r2["choices"][0]["token_ids"] == r["choices"][0]["token_ids"])
    if args.di_ref and args.prompt_ids:
        import torch

        ref = torch.load(args.di_ref, map_location="cpu", weights_only=False)
        ids = json.load(open(args.prompt_ids))["input_ids"]
        n = len(ref["decoded_ids"])
        st, r3 = cl.post("/v1/completions", dict(model=model, prompt=ids, max_tokens=n, temperature=0, ignore_eos=True,
                                                 return_token_ids=True))
        got, want = r3["choices"][0]["token_ids"], ref["decoded_ids"].tolist()
        check("greedy == run_deepinfer.py (token ids)", got == want, f"server {got[:8]} ref {want[:8]}")

    # 2. streaming == non-streaming
    ev, _, _, done = cl.stream("/v1/completions", base)
    check("completion stream == non-stream text", done and text_of(ev, False) == greedy_text
          and ev[-1]["choices"][0]["finish_reason"] == "length", f"{len(ev)} events")

    # 3. stop string taken from the greedy text
    if len(greedy_text) > 12:
        mid = len(greedy_text) // 2  # a stop string from the middle, so some text precedes it
        stop = greedy_text[mid:mid + 3]
        st, r = cl.post("/v1/completions", dict(base, stop=[stop]))
        t = r["choices"][0]["text"]
        check("stop string (non-stream)", r["choices"][0]["finish_reason"] == "stop" and stop not in t
              and greedy_text.startswith(t), f"stop={stop!r} -> {t!r}")
        ev, _, _, done = cl.stream("/v1/completions", dict(base, stop=stop))
        check("stop string (stream)", done and text_of(ev, False) == t and ev[-1]["choices"][0]["finish_reason"] == "stop")

    # 4. chat
    msgs = [{"role": "user", "content": "Say hello in one word."}]
    st, r = cl.post("/v1/chat/completions", dict(model=model, messages=msgs, max_tokens=16, temperature=0))
    check("chat non-stream", st == 200 and r["choices"][0]["message"]["role"] == "assistant",
          f"{r.get('usage')} {r['choices'][0]['message']['content'][:40]!r}")
    ev, _, _, done = cl.stream("/v1/chat/completions", dict(model=model, messages=msgs, max_tokens=16, temperature=0,
                                                            stream_options={"include_usage": True}))
    check("chat stream", done and ev[0]["choices"][0]["delta"].get("role") == "assistant"
          and text_of(ev, True) == r["choices"][0]["message"]["content"] and "usage" in ev[-1],
          f"{len(ev)} events")
    st, r = cl.post("/v1/chat/completions", dict(model=model, messages=msgs, max_tokens=8, thinking=True))
    check("chat thinking mode", st == 200 and "reasoning_content" in r["choices"][0]["message"])
    # chat_template_kwargs (as LiteLLM / vLLM-style clients send it). The rendered prompt ends with <think> in thinking
    # mode and with </think> in chat mode, so the prompt's last token shows which mode was rendered.
    think_ids = {True: 128821, False: 128822}  # V4.1 <think> / </think>

    def mode(extra, stream=False):
        body = dict(model=model, messages=msgs, max_tokens=8, temperature=0, return_token_ids=True, **extra)
        st, r = cl.post("/v1/chat/completions", body)
        if st != 200:
            return st, None, None
        return st, r["choices"][0]["prompt_token_ids"][-1] == think_ids[True], r["choices"][0]["message"]

    for label, extra, want in [("thinking", {"chat_template_kwargs": {"thinking": True}}, True),
                               ("enable_thinking", {"chat_template_kwargs": {"enable_thinking": True}}, True),
                               ("reasoning_effort", {"chat_template_kwargs": {"reasoning_effort": "high"}}, True),
                               ("thinking false", {"chat_template_kwargs": {"thinking": False}}, False),
                               ("top-level wins", {"thinking": False, "chat_template_kwargs": {"thinking": True}}, False)]:
        st, on, m = mode(extra)
        check(f"chat_template_kwargs {label}", st == 200 and on == want and (("reasoning_content" in m) == want),
              f"st {st} thinking prompt {on}")
    ev, _, _, done = cl.stream("/v1/chat/completions", dict(model=model, messages=msgs, max_tokens=16, temperature=0,
                                                            chat_template_kwargs={"thinking": True}))
    rc = "".join((e["choices"][0]["delta"].get("reasoning_content") or "") for e in ev if e.get("choices"))
    check("chat_template_kwargs thinking (stream)", done and len(rc) > 0, f"{len(rc)} reasoning chars")
    tool = {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}}}
    st, on, m = mode({"chat_template_kwargs": {"thinking": True}, "tools": [tool]})
    check("chat_template_kwargs thinking + tools", st == 200 and on and "reasoning_content" in m, f"st {st}")
    st, _ = cl.post("/v1/chat/completions", dict(model=model, messages=msgs, chat_template_kwargs={"thinking": "yes"}))
    st2, _ = cl.post("/v1/chat/completions", dict(model=model, messages=msgs, chat_template_kwargs=["thinking"]))
    check("chat_template_kwargs bad types -> 400", st == st2 == 400, f"{st} {st2}")

    # 5. concurrency: N streams at once, different lengths; they must overlap and all finish
    N = args.concurrency
    res = [None] * N
    t0 = time.perf_counter()

    def worker(i):
        body = dict(model=model, prompt=f"Entry {i}: the wind was", max_tokens=16 + 8 * i, temperature=0.7, top_p=0.9,
                    seed=i, ignore_eos=True)
        res[i] = cl.stream("/v1/completions", body)

    ths = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.perf_counter() - t0
    oks = [r is not None and r[3] and r[0][-1]["choices"][0]["finish_reason"] == "length" for r in res]
    firsts = [r[1] - t0 for r in res]
    ends = [r[2] - t0 for r in res]
    overlap = max(firsts) < min(ends)  # every stream had produced a token before the first one ended
    st, h = cl.get("/health")
    check(f"{N} concurrent streams finish", all(oks), f"wall {wall:.1f}s; ttft {min(firsts):.2f}-{max(firsts):.2f}s; "
          f"end {min(ends):.2f}-{max(ends):.2f}s")
    check("concurrent streams overlap (continuous batching)", overlap and h["stats"]["max_running"] > 1,
          f"engine max_running {h['stats']['max_running']}")

    # 6. a short request submitted while a long one decodes finishes first. The short one is sent once the long
    # one has streamed a few tokens (not after a fixed sleep: with CUDA graphs a small model decodes 200 tokens
    # in well under a second), and the long one is long enough to still be decoding when the short one ends.
    long_res, short_res = {}, {}
    long_decoding = threading.Event()

    def long_job():
        n = [0]

        def on_event(ev):
            n[0] += 1
            if n[0] == 5:
                long_decoding.set()

        try:
            long_res["r"] = cl.stream("/v1/completions", dict(model=model, prompt="Long:", max_tokens=600,
                                                              temperature=0, ignore_eos=True), on_event=on_event)
        finally:
            long_decoding.set()

    tl = threading.Thread(target=long_job)
    tl.start()
    long_decoding.wait(timeout=120)
    ts = time.perf_counter()
    short_res["r"] = cl.stream("/v1/completions", dict(model=model, prompt="Short:", max_tokens=8, temperature=0,
                                                       ignore_eos=True))
    t_short_end = short_res["r"][2]
    tl.join()
    check("request added mid-decode is served immediately", short_res["r"][3] and t_short_end < long_res["r"][2],
          f"short done after {t_short_end - ts:.2f}s, long done after {long_res['r'][2] - ts:.2f}s")

    # 7. cancellation: disconnect after 5 events; the engine must abort and keep serving
    st, h0 = cl.get("/health")
    ev, _, _, done = cl.stream("/v1/completions", dict(model=model, prompt="Cancel me:", max_tokens=4000,
                                                       temperature=0, ignore_eos=True), stop_after=5)
    time.sleep(2.0)
    st, h1 = cl.get("/health")
    aborted = h1["stats"]["aborted"] - h0["stats"]["aborted"]
    check("client disconnect aborts the request", not done and aborted >= 1, f"aborted +{aborted}")
    st, r = cl.post("/v1/completions", dict(model=model, prompt="after cancel", max_tokens=4, temperature=0))
    check("server healthy after cancel", st == 200 and h1["status"] == "ok")

    # 8. sampling with a seed is reproducible
    sb = dict(model=model, prompt=prompt, max_tokens=12, temperature=1.0, top_p=0.95, seed=1234, ignore_eos=True,
              return_token_ids=True)
    a = cl.post("/v1/completions", sb)[1]["choices"][0]["token_ids"]
    b = cl.post("/v1/completions", sb)[1]["choices"][0]["token_ids"]
    c = cl.post("/v1/completions", dict(sb, seed=99))[1]["choices"][0]["token_ids"]
    check("seeded sampling reproducible", a == b, f"seed99 differs: {a != c}")

    # 10. prefix cache: the same long prefix twice (and a multi-turn extension) -> hits, faster TTFT, same tokens
    if args.prefix_cache:
        import random

        rnd = random.Random(5)
        prefix = [rnd.randrange(1000, 100000) for _ in range(1500)]
        h0 = cl.get("/health")[1]["prefix_cache"]

        def ttft_of(ids, n=8):
            ev, t1, _, done = cl.stream("/v1/completions", dict(model=model, prompt=ids, max_tokens=n, temperature=0,
                                                                 ignore_eos=True))
            return done, t1, text_of(ev, False)

        t0 = time.perf_counter()
        d1, t1, txt1 = ttft_of(prefix + [11111, 22222, 33333])
        miss = t1 - t0
        t0 = time.perf_counter()
        d2, t2, txt2 = ttft_of(prefix + [11111, 22222, 33333])
        hit = t2 - t0
        h1 = cl.get("/health")[1]["prefix_cache"]
        check("prefix cache: repeated prompt hits", h1["hits"] - h0["hits"] >= 1,
              f"hits +{h1['hits'] - h0['hits']}, cached tokens +{h1['hit_tokens'] - h0['hit_tokens']}" + ("" if h1["hits"] > h0["hits"] else f" (server checkpoint stride {h1.get('checkpoint_stride')}: in-prompt hits need "
                                                   "--prefix-cache-checkpoint-stride <= 1024)"))
        check("prefix cache: TTFT faster on a hit", hit < miss, f"miss {miss:.3f}s hit {hit:.3f}s")
        check("prefix cache: same tokens on a hit", d1 and d2 and txt1 == txt2)  # soft without --deterministic
        st, r = cl.post("/v1/completions", dict(model=model, prompt=prefix + [11111, 22222, 33333] + [444] * 40,
                                                max_tokens=4, temperature=0))
        h2 = cl.get("/health")[1]["prefix_cache"]
        check("prefix cache: multi-turn extension hits the prompt-end snapshot", st == 200 and h2["hits"] > h1["hits"],
              f"{h2}")

    # 9. a one-token token-id prompt is a token-id prompt (not a one-element list of strings)
    st, r = cl.post("/v1/completions", dict(model=model, prompt=[1000], max_tokens=4, temperature=0, ignore_eos=True,
                                            return_token_ids=True))
    check("one-token token-id prompt", st == 200 and r["usage"]["prompt_tokens"] == 1
          and r["choices"][0]["prompt_token_ids"] == [1000], json.dumps(r)[:120])
    st, r = cl.post("/v1/completions", dict(model=model, prompt=["Hello"], max_tokens=4, temperature=0))
    check("one-element string-list prompt", st == 200 and r["usage"]["prompt_tokens"] >= 1, json.dumps(r)[:120])

    # 10. bad requests
    st, _ = cl.post("/v1/completions", dict(model=model, prompt="x", top_p=0))
    st2, _ = cl.post("/v1/chat/completions", dict(model=model, messages=[]))
    st3, _ = cl.post("/v1/completions", dict(model=model, prompt="x", n=2))
    check("bad requests -> 400", st == st2 == st3 == 400, f"{st} {st2} {st3}")

    st, h = cl.get("/health")
    print("engine stats:", h["stats"])
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"{len(RESULTS) - n_fail}/{len(RESULTS)} passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
