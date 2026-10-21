#!/usr/bin/env python3
"""Quality spot-check of a real (full) V4.1 model behind an OpenAI-compatible endpoint: a few math / code / German /
tool-calling / structured-output prompts with known answers (quality_set.jsonl), greedy, via /v1/chat/completions.

    python3 quality_check.py --url http://host:8000 [--set quality_set.jsonl] [--exec-code] [--max-tokens 512]
        [--needle-tokens 8000] [--json out.json]

Expectations per item: {"regex": ...} (case-insensitive search), {"all": [substrings]} (case-insensitive), or
{"code": "asserts"}; tool calling / structured output (the item's "request" adds tools, tool_choice, response_format,
... to the request): {"tool_calls": [{"name": ..., "args": {key: value-spec}}], "exact_count": bool} (every expected
call matched by a distinct returned call, any order), {"no_tool_calls": true} (optionally with "regex" on the content),
{"json": true, "schema": true, "fields": {"a.b": value-spec}} (content is JSON, valid against the request's
json_schema, field checks). A value-spec is a literal (strings compare case-insensitively) or {"regex": ...} /
{"len": n}. With --exec-code, the code check takes the first ```python block of the answer and runs it plus the
asserts in a separate `python3 -I` process (10 s timeout, temp dir). Without --exec-code it only checks that a
block was returned. Only use --exec-code against a model you trust.
Scores are also reported per category (an item's "category", else "tool" / "json" / "image" by id prefix, else
"core" for the general items).
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


def chat(url, model, messages, max_tokens, timeout, extra=None):
    u = urllib.parse.urlparse(url)
    prefix = u.path.rstrip("/").removesuffix("/v1")
    c = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    c.request("POST", prefix + "/v1/chat/completions", json.dumps(dict(
        model=model, messages=messages, max_tokens=max_tokens, temperature=0.0, **(extra or {}))),
        {"Content-Type": "application/json"})
    r = c.getresponse()
    data = json.loads(r.read())
    if r.status != 200:
        raise RuntimeError(f"HTTP {r.status}: {str(data)[:300]}")
    return data["choices"][0]["message"], data.get("usage", {})


def match_value(v, spec) -> bool:
    if isinstance(spec, dict) and "regex" in spec:
        return isinstance(v, (str, int, float)) and bool(re.search(spec["regex"], str(v), re.I))
    if isinstance(spec, dict) and "len" in spec:
        return isinstance(v, (list, dict, str)) and len(v) == spec["len"]
    if isinstance(spec, str):
        return isinstance(v, str) and v.strip().lower() == spec.lower()
    return v == spec and type(v) is type(spec)


def schema_errors(v, s, path="$") -> list[str]:
    """Small JSON-schema check (type, enum, required, properties, additionalProperties, items, min/max)."""
    t = s.get("type")
    py = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}
    if t == "integer" and (not isinstance(v, int) or isinstance(v, bool)) or \
            t == "number" and (not isinstance(v, (int, float)) or isinstance(v, bool)) or \
            t in py and not isinstance(v, py[t]):
        return [f"{path}: not {t}"]
    errs = [f"{path}: {v!r} not in enum"] if "enum" in s and v not in s["enum"] else []
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if "minimum" in s and v < s["minimum"] or "maximum" in s and v > s["maximum"]:
            errs.append(f"{path}: {v} out of range")
    if isinstance(v, dict):
        props = s.get("properties", {})
        errs += [f"{path}: missing {k}" for k in s.get("required", []) if k not in v]
        if s.get("additionalProperties") is False:
            errs += [f"{path}: unexpected {k}" for k in v if k not in props]
        for k, x in v.items():
            if k in props:
                errs += schema_errors(x, props[k], f"{path}.{k}")
    if isinstance(v, list):
        for i, x in enumerate(v):
            errs += schema_errors(x, s.get("items", {}), f"{path}[{i}]")
    return errs


def check_tools(message: dict, expect: dict, request: dict):
    calls = message.get("tool_calls") or []
    if expect.get("no_tool_calls"):
        if calls:
            return False, f"unexpected tool calls {[c['function']['name'] for c in calls]}"
        if "regex" in expect:
            return bool(re.search(expect["regex"], message.get("content") or "", re.I | re.M)), None
        return True, None
    got = []
    for c in calls:
        try:
            got.append((c["function"]["name"], json.loads(c["function"]["arguments"])))
        except (KeyError, TypeError, ValueError):
            return False, f"malformed tool call {str(c)[:120]}"
    if not got:
        return False, "no tool calls"
    if expect.get("exact_count") and len(got) != len(expect["tool_calls"]):
        return False, f"{len(got)} calls, expected {len(expect['tool_calls'])}"
    free = list(got)
    for want in expect["tool_calls"]:
        hit = next((g for g in free if g[0] == want["name"] and isinstance(g[1], dict) and all(
            k in g[1] and match_value(g[1][k], spec) for k, spec in want.get("args", {}).items())), None)
        if hit is None:
            return False, f"no call matching {want['name']}{json.dumps(want.get('args', {}))}; got {json.dumps(got)[:200]}"
        free.remove(hit)
    return True, None


def check_json(content: str, expect: dict, request: dict):
    try:
        v = json.loads(content)
    except ValueError as exc:
        return False, f"not JSON: {exc}"
    rf = request.get("response_format") or {}
    if expect.get("schema") and rf.get("type") == "json_schema":
        errs = schema_errors(v, rf["json_schema"]["schema"])
        if errs:
            return False, f"schema: {errs[:3]}"
    for path, spec in (expect.get("fields") or {}).items():
        x = v
        for part in path.split("."):
            x = x.get(part) if isinstance(x, dict) else None
        if not match_value(x, spec):
            return False, f"{path} = {x!r}"
    return True, None


def needle_messages(n_tokens: int):
    filler = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
    reps = max(1, n_tokens // 22)  # ~22 tokens per filler sentence group
    half = reps // 2
    text = filler * half + "The passkey is 7319. Remember it. 7319 is the passkey. " + filler * (reps - half)
    return [{"role": "user", "content": text + "\n\nWhat is the passkey? Answer with the number only."}]


def check(answer: str, expect: dict, exec_code: bool, message=None, request=None):
    if "tool_calls" in expect or expect.get("no_tool_calls"):
        return check_tools(message or {}, expect, request or {})
    if expect.get("json"):
        return check_json(answer, expect, request or {})
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


def category(it: dict) -> str:
    """Scoring group: an item's "category", else tool / json by id prefix, else core (the original general set)."""
    if it.get("category"):
        return it["category"]
    for c in ("tool", "json", "image"):
        if it["id"].startswith(c + "-"):
            return c
    return "core"


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
            req = it.get("request") or {}
            msg, usage = chat(args.url, model, msgs, args.max_tokens, args.timeout, req)
            ans = msg.get("content") or ""
            ok, note = check(ans, it["expect"], args.exec_code, msg, req)
            if msg.get("tool_calls"):
                ans += "".join(f"\n[tool_call] {c['function']['name']}({c['function']['arguments']})"
                               for c in msg["tool_calls"])
        except Exception as exc:  # noqa: BLE001
            ans, usage, ok, note = "", {}, False, f"error: {exc!r}"[:200]
        dt = time.perf_counter() - t0
        known = it.get("known_fail")
        if known:
            n_known += 1
            note = f"known fail: {known}" + (f"; {note}" if note else "")
        else:
            n_ok += ok
        rows.append(dict(id=it["id"], category=category(it), ok=ok, known_fail=bool(known), note=note, seconds=dt,
                         usage=usage, answer=ans[:2000]))
        tag = ("KPASS" if ok else "KFAIL") if known else ("PASS" if ok else "FAIL")
        print(f"[{tag}] {it['id']:16s} {dt:6.1f}s {usage.get('prompt_tokens', '?'):>6} prompt tok"
              f"{'  ' + note if note else ''}\n        {ans.strip()[:160]!r}", flush=True)
    total = len(items) - n_known
    cats: dict = {}
    for r in rows:
        c = cats.setdefault(r["category"], dict(passed=0, total=0, known_fail=0))
        if r["known_fail"]:
            c["known_fail"] += 1
        else:
            c["total"] += 1
            c["passed"] += r["ok"]
    for name, c in cats.items():
        print(f"  {name:6s} {c['passed']}/{c['total']}" + (f" (+{c['known_fail']} known-fail)" if c["known_fail"] else ""))
    print(f"{n_ok}/{total} passed" + (f" (+{n_known} known-fail item(s) not counted)" if n_known else "")
          + f" ({model} @ {args.url})")
    if args.json:
        json.dump(dict(url=args.url, model=model, passed=n_ok, total=total, known_fail=n_known, categories=cats, rows=rows),
                  open(args.json, "w"), indent=1)
    return 0 if n_ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
