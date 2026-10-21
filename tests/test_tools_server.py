#!/usr/bin/env python3
"""Tool calling and response_format against a running server.py (stdlib client).

    python3 /opt/sm120/tests/test_tools_server.py --url http://127.0.0.1:8000 [--thinking]

Written for truncated test models, whose text is garbage: the checks are about structure, not quality. With a grammar
(tool_choice required / named, response_format) every finished output must be valid (known function, arguments that
satisfy the JSON schema, content that satisfies the response schema); auto / none must come back as sane OpenAI
responses. The schemas are bounded (enums, maxLength, integer ranges) so that a random model also terminates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_server  # noqa: E402
from test_server import RESULTS, Client  # noqa: E402

SOFT: set[str] = set()  # names whose failure only means "ran out of max_tokens inside the grammar"


def check(name: str, ok: bool, info: str = "") -> bool:
    if name in SOFT and not ok:
        print(f"[INFO] {name}  {info} (max_tokens reached before the grammar completed)", flush=True)
        return True
    return test_server.check(name, ok, info)

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "maxLength": 24},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        "days": {"type": "integer", "minimum": 1, "maximum": 7}},
        "required": ["city", "unit"]}}}
EMAIL = {"type": "function", "function": {
    "name": "send_email", "description": "Send an e-mail",
    "parameters": {"type": "object", "properties": {
        "to": {"type": "string", "maxLength": 32},
        "urgent": {"type": "boolean"},
        "meta": {"type": "object", "properties": {"lang": {"type": "string", "enum": ["de", "en"]},
                                                  "retries": {"type": "integer", "minimum": 0, "maximum": 3}},
                 "required": ["lang"], "additionalProperties": False}},
        "required": ["to", "urgent", "meta"]}}}
TOOLS = [WEATHER, EMAIL]
SCHEMA = {"type": "object", "properties": {
    "name": {"type": "string", "maxLength": 16},
    "age": {"type": "integer", "minimum": 0, "maximum": 120},
    "role": {"type": "string", "enum": ["admin", "user", "guest"]},
    "address": {"type": "object", "properties": {"zip": {"type": "string", "maxLength": 6},
                                                 "country": {"type": "string", "enum": ["AT", "DE", "CH"]}},
                "required": ["zip", "country"], "additionalProperties": False},
    "tags": {"type": "array", "items": {"type": "string", "enum": ["a", "b", "c"]}, "maxItems": 3}},
    "required": ["name", "age", "role", "address"], "additionalProperties": False}


def validate(v, s, path="$") -> list[str]:
    """Minimal JSON-schema check for the keywords used above."""
    errs = []
    t = s.get("type")
    types = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}
    if t == "integer":
        if not isinstance(v, int) or isinstance(v, bool):
            return [f"{path}: not an integer"]
    elif t == "number":
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return [f"{path}: not a number"]
    elif t in types and not isinstance(v, types[t]):
        return [f"{path}: not {t}"]
    if "enum" in s and v not in s["enum"]:
        errs.append(f"{path}: {v!r} not in enum")
    if isinstance(v, str) and "maxLength" in s and len(v) > s["maxLength"]:
        errs.append(f"{path}: longer than {s['maxLength']}")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if "minimum" in s and v < s["minimum"] or "maximum" in s and v > s["maximum"]:
            errs.append(f"{path}: {v} out of range")
    if isinstance(v, dict):
        props = s.get("properties", {})
        errs += [f"{path}: missing {k}" for k in s.get("required", []) if k not in v]
        if s.get("additionalProperties") is False:
            errs += [f"{path}: extra {k}" for k in v if k not in props]
        for k, x in v.items():
            if k in props:
                errs += validate(x, props[k], f"{path}.{k}")
    if isinstance(v, list):
        if "maxItems" in s and len(v) > s["maxItems"]:
            errs.append(f"{path}: more than {s['maxItems']} items")
        for i, x in enumerate(v):
            errs += validate(x, s.get("items", {}), f"{path}[{i}]")
    return errs


def strict(params: dict) -> dict:
    """The tool grammar admits only declared parameters (as strict function calling does)."""
    return dict(params, additionalProperties=False)


def check_calls(calls, allowed, info="") -> tuple[bool, str]:
    by_name = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}
    if not calls:
        return False, "no tool_calls"
    ids = set()
    for c in calls:
        if c.get("type") != "function" or not str(c.get("id", "")).startswith("call_") or c["id"] in ids:
            return False, f"bad id/type {c}"
        ids.add(c["id"])
        name = c["function"]["name"]
        if name not in allowed:
            return False, f"function {name} not allowed"
        try:
            args = json.loads(c["function"]["arguments"])
        except ValueError:
            return False, f"arguments not JSON: {c['function']['arguments'][:80]!r}"
        errs = validate(args, strict(by_name[name]))
        if errs:
            return False, f"{name}: {errs[:3]}"
    return True, info + " " + "; ".join(f"{c['function']['name']}({c['function']['arguments'][:70]})" for c in calls)


def assemble(events):
    content, reasoning, calls, finish = "", "", {}, None
    for e in events:
        if not e.get("choices"):
            continue
        ch = e["choices"][0]
        d = ch["delta"]
        content += d.get("content") or ""
        reasoning += d.get("reasoning_content") or ""
        for tc in d.get("tool_calls") or []:
            c = calls.setdefault(tc["index"], {"id": None, "type": "function", "function": {"name": None, "arguments": ""}})
            if tc.get("id"):
                c["id"], c["function"]["name"] = tc["id"], tc["function"]["name"]
            c["function"]["arguments"] += (tc.get("function") or {}).get("arguments") or ""
        finish = ch.get("finish_reason") or finish
    return content, reasoning, [calls[i] for i in sorted(calls)], finish


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--repeats", type=int, default=3, help="seeds per constrained case")
    ap.add_argument("--thinking", action="store_true", help="also run the thinking-mode cases")
    ap.add_argument("--nondeterministic-ok", action="store_true",
                    help="server runs b12x MoE (not --deterministic): greedy repeat comparisons are INFO")
    ap.add_argument("--prefix-cache", action="store_true", help="server runs --prefix-cache (small grain): test hits")
    args = ap.parse_args()
    if args.nondeterministic_ok:
        SOFT.add("tools greedy stream == non-stream")
    cl = Client(args.url)
    model = cl.get("/v1/models")[1]["data"][0]["id"]
    user = [{"role": "user", "content": "What is the weather in Vienna, in celsius? Then mail it to the ops team."}]
    base = dict(model=model, messages=user, tools=TOOLS, max_tokens=args.max_tokens, temperature=0.8)

    # 1. tool_choice required / named / parallel_tool_calls false, non-streaming
    for label, extra, allowed in [
            ("required", dict(tool_choice="required"), {"get_weather", "send_email"}),
            ("named send_email", dict(tool_choice={"type": "function", "function": {"name": "send_email"}}), {"send_email"}),
            ("required, parallel_tool_calls=false", dict(tool_choice="required", parallel_tool_calls=False),
             {"get_weather", "send_email"})]:
        for seed in range(args.repeats):
            st, r = cl.post("/v1/chat/completions", dict(base, seed=seed, **extra))
            if st != 200:
                check(f"tools {label} seed {seed}", False, json.dumps(r)[:200])
                continue
            ch = r["choices"][0]
            calls = ch["message"].get("tool_calls")
            ok, info = check_calls(calls, allowed, f"finish {ch['finish_reason']}")
            if ch["finish_reason"] == "length":  # a random model can exhaust max_tokens inside the grammar
                SOFT.add(f"tools {label} seed {seed}")
            ok = ok and ch["finish_reason"] == "tool_calls"
            if "parallel" in label:
                ok = ok and len(calls) == 1
            check(f"tools {label} seed {seed}", ok, info[:300])

    # 2. streaming, required: deltas assemble into valid calls; same structure as non-streaming
    for seed in range(args.repeats):
        ev, _, _, done = cl.stream("/v1/chat/completions", dict(base, seed=seed, tool_choice="required"))
        content, _, calls, finish = assemble(ev)
        ok, info = check_calls(calls, {"get_weather", "send_email"}, f"finish {finish}, {len(ev)} events")
        first = [tc for e in ev if e.get("choices") for tc in (e["choices"][0]["delta"].get("tool_calls") or [])]
        ok = ok and done and finish == "tool_calls" and all("index" in tc for tc in first)
        if finish == "length":
            SOFT.add(f"tools stream required seed {seed}")
        check(f"tools stream required seed {seed}", ok, info[:300])
    # greedy stream == greedy non-stream (same calls; ids differ)
    g = dict(base, temperature=0, tool_choice="required")
    st, r = cl.post("/v1/chat/completions", g)
    ev, _, _, _ = cl.stream("/v1/chat/completions", g)
    _, _, scalls, _ = assemble(ev)
    ncalls = r["choices"][0]["message"].get("tool_calls") or []
    same = [(c["function"]["name"], c["function"]["arguments"]) for c in scalls] == \
           [(c["function"]["name"], c["function"]["arguments"]) for c in ncalls]
    check("tools greedy stream == non-stream", same, f"{len(ncalls)} calls")

    # 3. auto / none: sane responses
    for seed in range(args.repeats):
        st, r = cl.post("/v1/chat/completions", dict(base, seed=seed, max_tokens=64))
        ch = r["choices"][0] if st == 200 else {}
        m = ch.get("message", {})
        calls = m.get("tool_calls")
        ok = st == 200 and ch["finish_reason"] in ("stop", "length", "tool_calls") and \
            (isinstance(m.get("content"), str) or calls) and (not calls or check_calls(calls, {"get_weather", "send_email"})[0])
        check(f"tools auto seed {seed}", ok, f"finish {ch.get('finish_reason')} calls {len(calls or [])} "
                                            f"content {str(m.get('content'))[:60]!r}")
    st, r = cl.post("/v1/chat/completions", dict(base, tool_choice="none", max_tokens=64))
    check("tools none", st == 200 and not r["choices"][0]["message"].get("tool_calls")
          and r["choices"][0]["finish_reason"] in ("stop", "length"), r["choices"][0]["finish_reason"] if st == 200 else r)
    ev, _, _, done = cl.stream("/v1/chat/completions", dict(base, max_tokens=64))
    content, _, calls, finish = assemble(ev)
    check("tools auto stream", done and finish in ("stop", "length", "tool_calls"), f"finish {finish} calls {len(calls)}")

    # 4. multi-turn history: assistant tool_calls + tool results, then a forced follow-up call
    hist = user + [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "get_weather",
                                                          "arguments": '{"city": "Vienna", "unit": "celsius"}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": '{"temp": 21, "sky": "clear"}'}]
    st, r = cl.post("/v1/chat/completions", dict(base, messages=hist, tool_choice={"type": "function", "function": {
        "name": "send_email"}}))
    ok, info = check_calls(r["choices"][0]["message"].get("tool_calls") if st == 200 else None, {"send_email"})
    check("tools history + tool result -> named call", st == 200 and ok, info[:200])
    st, r = cl.post("/v1/chat/completions", dict(base, messages=hist, max_tokens=32))
    check("tools history, auto", st == 200, r["choices"][0]["finish_reason"] if st == 200 else r)

    # 5. response_format
    for seed in range(args.repeats):
        st, r = cl.post("/v1/chat/completions", dict(model=model, messages=[{"role": "user", "content": "Reply in JSON."}],
                                                     response_format={"type": "json_object"}, seed=seed,
                                                     max_tokens=args.max_tokens, temperature=0.8))
        ch = r["choices"][0]
        try:
            v = json.loads(ch["message"]["content"])
            ok = isinstance(v, dict)
        except (ValueError, TypeError):
            ok, v = False, None
        if ch["finish_reason"] == "length":  # an unbounded object from a random model may run out of tokens
            SOFT.add(f"json_object seed {seed}")
        check(f"json_object seed {seed}", ok and ch["finish_reason"] == "stop",
              f"finish {ch['finish_reason']} {ch['message']['content'][:100]!r}")
    rf = {"type": "json_schema", "json_schema": {"name": "person", "schema": SCHEMA, "strict": True}}
    for seed in range(args.repeats):
        st, r = cl.post("/v1/chat/completions", dict(model=model, messages=[{"role": "user", "content": "A person."}],
                                                     response_format=rf, seed=seed, max_tokens=args.max_tokens,
                                                     temperature=0.8))
        ch = r["choices"][0]
        try:
            errs = validate(json.loads(ch["message"]["content"]), SCHEMA)
        except (ValueError, TypeError) as exc:
            errs = [str(exc)]
        check(f"json_schema seed {seed}", not errs and ch["finish_reason"] == "stop",
              f"finish {ch['finish_reason']} {errs[:2]} {ch['message']['content'][:120]!r}")
    ev, _, _, done = cl.stream("/v1/chat/completions", dict(model=model, messages=[{"role": "user", "content": "A person."}],
                                                           response_format=rf, max_tokens=args.max_tokens, temperature=0))
    content, _, _, finish = assemble(ev)
    try:
        errs = validate(json.loads(content), SCHEMA)
    except ValueError as exc:
        errs = [str(exc)]
    check("json_schema stream", done and not errs and finish == "stop", f"{errs[:2]} {content[:100]!r}")
    # auto tools + response_format: either a valid call or schema-valid JSON
    st, r = cl.post("/v1/chat/completions", dict(base, response_format=rf, max_tokens=args.max_tokens))
    m = r["choices"][0]["message"]
    if m.get("tool_calls"):
        ok, info = check_calls(m["tool_calls"], {"get_weather", "send_email"})
    else:
        try:
            errs = validate(json.loads(m["content"]), SCHEMA)
        except (ValueError, TypeError) as exc:
            errs = [str(exc)]
        ok, info = not errs, f"json {errs[:2]}"
    check("auto tools + json_schema", ok, info[:200])

    # 6. concurrency: constrained and unconstrained requests in one batch
    res = {}

    def worker(i):
        body = [dict(base, seed=i, tool_choice="required"),
                dict(model=model, messages=user, response_format=rf, seed=i, max_tokens=args.max_tokens, temperature=0.8),
                dict(model=model, messages=user, seed=i, max_tokens=48, temperature=0.8)][i % 3]
        res[i] = cl.post("/v1/chat/completions", body)
    th = [threading.Thread(target=worker, args=(i,)) for i in range(9)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    good, truncated = 0, 0
    for i, (st, r) in sorted(res.items()):
        if st != 200:
            continue
        m = r["choices"][0]["message"]
        if i % 3 != 2 and r["choices"][0]["finish_reason"] == "length":
            truncated += 1
            continue
        if i % 3 == 0:
            good += check_calls(m.get("tool_calls"), {"get_weather", "send_email"})[0]
        elif i % 3 == 1:
            try:
                good += not validate(json.loads(m["content"]), SCHEMA)
            except (ValueError, TypeError):
                pass
        else:
            good += isinstance(m.get("content"), str)
    check("concurrent mixed constrained / free (9)", good + truncated == 9 and good >= 6,
          f"{good}/9 valid, {truncated} hit max_tokens")

    # 7. thinking mode: the grammar is armed after </think>
    if args.thinking:
        st, r = cl.post("/v1/chat/completions", dict(base, thinking=True, tool_choice="required", temperature=0.8, seed=1))
        ch = r["choices"][0]
        m = ch["message"]
        if m.get("tool_calls"):
            ok, info = check_calls(m["tool_calls"], {"get_weather", "send_email"})
        else:  # a random model may never close its reasoning
            SOFT.add("thinking + required")
            ok, info = False, f"no </think> within max_tokens (finish {ch['finish_reason']})"
        check("thinking + required", ok and "reasoning_content" in m, info[:200])
        ev, _, _, done = cl.stream("/v1/chat/completions", dict(base, thinking=True, max_tokens=48))
        _, reasoning, _, finish = assemble(ev)
        check("thinking stream has reasoning_content", done and bool(reasoning), f"{len(reasoning)} chars, finish {finish}")

    # 8. prefix cache: unconstrained tool requests are cached; constrained ones bypass the cache (prefix_cache.py)
    if args.prefix_cache:
        pc0 = cl.get("/health")[1].get("prefix_cache") or {}
        hist_body = dict(base, messages=hist, max_tokens=8, temperature=0)
        for _ in range(2):
            st, r = cl.post("/v1/chat/completions", hist_body)
        pc1 = cl.get("/health")[1].get("prefix_cache") or {}
        check("prefix cache: repeated tools prompt hits", st == 200 and pc1.get("hits", 0) > pc0.get("hits", 0),
              f"hits {pc0.get('hits')} -> {pc1.get('hits')}, hit_tokens {pc1.get('hit_tokens')}")
        ok = 0
        for _ in range(2):
            st, r = cl.post("/v1/chat/completions", dict(hist_body, tool_choice="required", max_tokens=args.max_tokens))
            ok += st == 200 and (check_calls(r["choices"][0]["message"].get("tool_calls"), {"get_weather", "send_email"})[0]
                                 or r["choices"][0]["finish_reason"] == "length")
        pc2 = cl.get("/health")[1].get("prefix_cache") or {}
        check("prefix cache: constrained requests bypass it", ok == 2 and pc2.get("hits") == pc1.get("hits"),
              f"{ok}/2 valid, hits {pc1.get('hits')} -> {pc2.get('hits')}")

    # 9. bad requests
    for label, body in [
            ("unknown named function", dict(base, tool_choice={"type": "function", "function": {"name": "nope"}})),
            ("required without tools", dict(model=model, messages=user, tool_choice="required")),
            ("bad response_format", dict(model=model, messages=user, response_format={"type": "yaml"})),
            ("bad tools", dict(model=model, messages=user, tools=[{"type": "function"}])),
            ("tools on /v1/completions", "completions")]:
        if body == "completions":
            st, r = cl.post("/v1/completions", dict(model=model, prompt="x", tools=TOOLS))
        else:
            st, r = cl.post("/v1/chat/completions", body)
        check(f"400: {label}", st == 400, json.dumps(r)[:120])
    st, h = cl.get("/health")
    check("server healthy after the tool tests", st == 200 and h["status"] == "ok")

    fails = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(fails)}/{len(RESULTS)} passed" + (f"; FAILED: {fails}" if fails else ""))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
