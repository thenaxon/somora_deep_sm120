#!/usr/bin/env python3
"""Quality spot-check of a real (full) V4.1 model behind an OpenAI-compatible endpoint: a few math / code / German
prompts with known answers (quality_set.jsonl), greedy, via /v1/chat/completions.

    python3 quality_check.py --url http://host:8000 [--set quality_set.jsonl] [--exec-code] [--max-tokens 512]
        [--needle-tokens 8000] [--json out.json]

Expectations per item: {"regex": ...} (case-insensitive search), {"all": [substrings]} (case-insensitive), or
{"code": "asserts"}. With --exec-code, the code check takes the first ```python block of the answer and runs it plus the
asserts in a separate `python3 -I` process (10 s timeout, temp dir). Without --exec-code it only checks that a
block was returned. Only use --exec-code against a model you trust.
"needle": true builds a long-context retrieval prompt: a passkey 7319 hidden in --needle-tokens of filler.
"known_fail": "<reason>" marks an item that the model itself is known to fail (not an engine problem). It is reported
as KFAIL / KPASS and left out of the pass count and the exit status, so it cannot mask real regressions.

Run it against both endpoints (DeepInfer and vLLM): on the real model both should pass (nearly) everything. A
truncated test checkpoint is expected to fail; that only checks the plumbing.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))


def chat(url, model, messages, max_tokens, timeout):
    u = urllib.parse.urlparse(url)
    prefix = u.path.rstrip("/").removesuffix("/v1")
    c = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    c.request("POST", prefix + "/v1/chat/completions", json.dumps(dict(
        model=model, messages=messages, max_tokens=max_tokens, temperature=0.0)), {"Content-Type": "application/json"})
    r = c.getresponse()
    data = json.loads(r.read())
    if r.status != 200:
        raise RuntimeError(f"HTTP {r.status}: {str(data)[:300]}")
    m = data["choices"][0]["message"]
    return (m.get("content") or ""), data.get("usage", {})


def needle_messages(n_tokens: int):
    filler = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
    reps = max(1, n_tokens // 22)  # ~22 tokens per filler sentence group
    half = reps // 2
    text = filler * half + "The passkey is 7319. Remember it. 7319 is the passkey. " + filler * (reps - half)
    return [{"role": "user", "content": text + "\n\nWhat is the passkey? Answer with the number only."}]


def check(answer: str, expect: dict, exec_code: bool):
    if "regex" in expect:
        return bool(re.search(expect["regex"], answer, re.I | re.M)), None
    if "all" in expect:
        miss = [s for s in expect["all"] if s.lower() not in answer.lower()]
        return not miss, (f"missing {miss}" if miss else None)
    if "code" in expect:
        m = re.search(r"```(?:python|py)?\s*\n(.*?)```", answer, re.S)
        if not m:
            return False, "no code block"
        if not exec_code:
            return True, "code block found (not executed; pass --exec-code)"
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.py")
            open(p, "w").write(m.group(1) + "\n\n" + expect["code"] + "\nprint('OK')\n")
            try:
                r = subprocess.run([sys.executable, "-I", p], capture_output=True, text=True, timeout=10, cwd=d)
            except subprocess.TimeoutExpired:
                return False, "timeout"
            return r.returncode == 0 and "OK" in r.stdout, (r.stderr.strip().splitlines() or [""])[-1][:200]
    raise ValueError(f"unknown expectation {expect}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--model")
    ap.add_argument("--set", default=os.path.join(HERE, "quality_set.jsonl"))
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--needle-tokens", type=int, default=8000)
    ap.add_argument("--exec-code", action="store_true")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--json")
    args = ap.parse_args()
    u = urllib.parse.urlparse(args.url)
    c = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=60)
    c.request("GET", u.path.rstrip("/").removesuffix("/v1") + "/v1/models")
    model = args.model or json.loads(c.getresponse().read())["data"][0]["id"]
    items = [json.loads(l) for l in open(args.set) if l.strip()]
    rows, n_ok, n_known = [], 0, 0
    for it in items:
        msgs = needle_messages(args.needle_tokens) if it.get("needle") else it["messages"]
        t0 = time.perf_counter()
        try:
            ans, usage = chat(args.url, model, msgs, args.max_tokens, args.timeout)
            ok, note = check(ans, it["expect"], args.exec_code)
        except Exception as exc:  # noqa: BLE001
            ans, usage, ok, note = "", {}, False, f"error: {exc!r}"[:200]
        dt = time.perf_counter() - t0
        known = it.get("known_fail")
        if known:
            n_known += 1
            note = f"known fail: {known}" + (f"; {note}" if note else "")
        else:
            n_ok += ok
        rows.append(dict(id=it["id"], ok=ok, known_fail=bool(known), note=note, seconds=dt, usage=usage, answer=ans[:2000]))
        tag = ("KPASS" if ok else "KFAIL") if known else ("PASS" if ok else "FAIL")
        print(f"[{tag}] {it['id']:16s} {dt:6.1f}s {usage.get('prompt_tokens', '?'):>6} prompt tok"
              f"{'  ' + note if note else ''}\n        {ans.strip()[:160]!r}", flush=True)
    total = len(items) - n_known
    print(f"{n_ok}/{total} passed" + (f" (+{n_known} known-fail item(s) not counted)" if n_known else "")
          + f" ({model} @ {args.url})")
    if args.json:
        json.dump(dict(url=args.url, model=model, passed=n_ok, total=total, known_fail=n_known, rows=rows), open(args.json, "w"), indent=1)
    return 0 if n_ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
