#!/usr/bin/env python3
"""Compatibility of server.py with the official `openai` Python SDK: chat, streaming, tools (non-streaming and
streaming), response_format json_schema through the SDK's pydantic helper (`chat.completions.parse`).

    pip install openai pydantic        # e.g. in a venv: python3 -m venv /work/sdkvenv && /work/sdkvenv/bin/pip install ...
    python3 /opt/sm120/tests/test_openai_sdk.py --url http://127.0.0.1:8000

The checks are structural (truncated test models produce garbage text): the SDK must parse every response, tool
calls must come back typed, and the parsed pydantic object must exist whenever the grammar finished.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Literal

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, info: str = "") -> None:
    RESULTS.append((name, bool(ok), info))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {info}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--max-tokens", type=int, default=768)
    args = ap.parse_args()

    import openai
    from openai import OpenAI
    from pydantic import BaseModel, Field

    print(f"openai {openai.__version__}")
    client = OpenAI(base_url=args.url.rstrip("/") + "/v1", api_key="unused", max_retries=0, timeout=600)
    model = client.models.list().data[0].id
    msgs = [{"role": "user", "content": "What's the weather in Vienna? Answer with a tool call."}]
    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "Weather for a city",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "maxLength": 24},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
            "required": ["city", "unit"], "additionalProperties": False}}}]

    r = client.chat.completions.create(model=model, messages=msgs, max_tokens=16, temperature=0)
    check("chat.completions.create", r.choices[0].message.role == "assistant" and r.usage.completion_tokens > 0,
          repr(r.choices[0].message.content)[:60])
    text = "".join(c.choices[0].delta.content or "" for c in client.chat.completions.create(
        model=model, messages=msgs, max_tokens=16, temperature=0, stream=True) if c.choices)
    check("chat stream", isinstance(text, str) and len(text) > 0, repr(text)[:60])

    # tools, non-streaming
    r = client.chat.completions.create(model=model, messages=msgs, tools=tools, tool_choice="required",
                                       max_tokens=args.max_tokens, temperature=0.7, seed=3)
    m = r.choices[0].message
    tc = (m.tool_calls or [None])[0]
    ok = r.choices[0].finish_reason == "tool_calls" and tc is not None and tc.type == "function" and \
        tc.function.name == "get_weather" and json.loads(tc.function.arguments)["unit"] in ("celsius", "fahrenheit")
    check("tools (SDK typed tool_calls)", ok, f"{tc.function.name}({tc.function.arguments})" if tc else str(m)[:120])

    # the call goes back into the history with a tool result, as a client application does
    if tc is not None:
        msgs2 = msgs + [m.model_dump(exclude_none=True), {"role": "tool", "tool_call_id": tc.id, "content": "21 C, clear"}]
        r2 = client.chat.completions.create(model=model, messages=msgs2, tools=tools, max_tokens=32, temperature=0)
        check("tools: tool result round trip", r2.choices[0].finish_reason in ("stop", "length", "tool_calls"),
              r2.choices[0].finish_reason)

    # tools, streaming: the SDK's chunk types; arguments arrive in pieces
    name, argbuf, ids, finish, n = None, "", set(), None, 0
    for c in client.chat.completions.create(model=model, messages=msgs, tools=tools, tool_choice="required",
                                            max_tokens=args.max_tokens, temperature=0.7, seed=4, stream=True):
        if not c.choices:
            continue
        d = c.choices[0].delta
        for t in d.tool_calls or []:
            n += 1
            if t.id:
                ids.add(t.id)
            if t.function and t.function.name:
                name = t.function.name
            if t.function and t.function.arguments:
                argbuf += t.function.arguments
        finish = c.choices[0].finish_reason or finish
    try:
        args_ok = json.loads(argbuf)["unit"] in ("celsius", "fahrenheit")
    except (ValueError, KeyError):
        args_ok = False
    check("tools stream (SDK chunks)", name == "get_weather" and args_ok and finish == "tool_calls" and len(ids) >= 1,
          f"{n} tool_call deltas, args {argbuf[:80]}")

    # streaming helper that accumulates tool calls (client.chat.completions.stream / beta), if this SDK has it
    stream_api = getattr(client.chat.completions, "stream", None) or getattr(
        getattr(getattr(client, "beta", None), "chat", None) and client.beta.chat.completions, "stream", None)
    if stream_api is not None:
        with stream_api(model=model, messages=msgs, tools=tools, tool_choice="required", max_tokens=args.max_tokens,
                        temperature=0.7, seed=5) as s:
            for _ in s:
                pass
            fin = s.get_final_completion()
        fc = fin.choices[0].message.tool_calls or []
        check("tools stream helper (get_final_completion)", bool(fc) and fc[0].function.name == "get_weather",
              f"{fc[0].function.arguments[:80] if fc else None}")

    # response_format with a pydantic model through the SDK's parse helper
    class Address(BaseModel):
        zip: str = Field(max_length=6)
        country: Literal["AT", "DE", "CH"]

    class Person(BaseModel):
        name: str = Field(max_length=16)
        role: Literal["admin", "user", "guest"]
        active: bool
        address: Address

    parse = getattr(client.chat.completions, "parse", None) or client.beta.chat.completions.parse
    try:
        r = parse(model=model, messages=[{"role": "user", "content": "Invent a person."}], response_format=Person,
                  max_tokens=args.max_tokens, temperature=0.7, seed=6)
        p = r.choices[0].message.parsed
        check("chat.completions.parse(response_format=Person)", isinstance(p, Person), repr(p)[:160])
    except Exception as exc:  # noqa: BLE001  (LengthFinishReasonError, ValidationError, ...)
        check("chat.completions.parse(response_format=Person)", False, f"{type(exc).__name__}: {exc}"[:200])
    # tools declared through the SDK's pydantic_function_tool (strict schema)
    if hasattr(openai, "pydantic_function_tool"):
        class GetWeather(BaseModel):
            city: str = Field(max_length=24)
            unit: Literal["celsius", "fahrenheit"]
        try:
            r = parse(model=model, messages=msgs, tools=[openai.pydantic_function_tool(GetWeather)],
                      tool_choice="required", max_tokens=args.max_tokens, temperature=0.7, seed=7)
            tc = r.choices[0].message.tool_calls[0]
            check("parse(tools=[pydantic_function_tool])", isinstance(tc.function.parsed_arguments, GetWeather),
                  repr(tc.function.parsed_arguments)[:120])
        except Exception as exc:  # noqa: BLE001
            check("parse(tools=[pydantic_function_tool])", False, f"{type(exc).__name__}: {exc}"[:200])
    r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": "JSON please"}],
                                       response_format={"type": "json_object"}, max_tokens=args.max_tokens,
                                       temperature=0.7, seed=8)
    try:
        ok = isinstance(json.loads(r.choices[0].message.content), dict)
    except ValueError:
        ok = False
    check("response_format json_object", ok or r.choices[0].finish_reason == "length",
          f"finish {r.choices[0].finish_reason} {r.choices[0].message.content[:80]!r}")

    try:
        client.chat.completions.create(model=model, messages=msgs, tools=tools,
                                       tool_choice={"type": "function", "function": {"name": "nope"}})
        check("SDK sees 400 as BadRequestError", False)
    except openai.BadRequestError as exc:
        check("SDK sees 400 as BadRequestError", True, str(exc)[:100])

    fails = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(fails)}/{len(RESULTS)} passed" + (f"; FAILED: {fails}" if fails else ""))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
