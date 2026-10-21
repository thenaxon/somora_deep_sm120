#!/usr/bin/env python3
"""CPU unit tests for server/tools.py: prompt rendering, DSML tool-call parsing (complete and streaming), grammars.

    python3 tests/test_tools.py [--encoding /ckpt/encoding/encoding.py]      (or: pytest tests/test_tools.py)

Needs the checkpoint's encoding/encoding.py (--encoding, $DS41_ENCODING, or /ckpt/encoding/encoding.py) and, for the
grammar tests, xgrammar (skipped if missing). The grammar tests compile against a one-character-per-token vocabulary,
so they check the grammar language itself, independent of the real tokenizer.
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "server"))

import tools  # noqa: E402

ENC_PATH = os.environ.get("DS41_ENCODING", "/ckpt/encoding/encoding.py")
_enc = None


def enc():
    global _enc
    if _enc is None:
        spec = importlib.util.spec_from_file_location("ds41_encoding", ENC_PATH)
        _enc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_enc)
    return _enc


WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "maxLength": 40},
        "days": {"type": "integer", "minimum": 1, "maximum": 7},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        "opts": {"type": "object", "properties": {"hourly": {"type": "boolean"}}, "required": ["hourly"]}},
        "required": ["city"]}}}
NOW = {"type": "function", "function": {"name": "now", "description": "Current time", "parameters": {"type": "object",
                                                                                                    "properties": {}}}}
TOOLS = [WEATHER, NOW]


def plan(**kw):
    return tools.plan_from_body({"tools": TOOLS, **kw})


def render(msgs, p, thinking=False):
    return enc().encode_messages(tools.prepare_messages(msgs, p), thinking_mode="thinking" if thinking else "chat")


def completion_of(calls, content="", thinking=False, reasoning=""):
    """What the model would write for an assistant turn with these calls: the training-format rendering of the turn
    (encoding.py), minus the prompt and EOS."""
    e = enc()
    msgs = [{"role": "user", "content": "q"}]
    asst = {"role": "assistant", "content": content, "reasoning_content": reasoning,
            "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
                           for i, (n, a) in enumerate(calls)]}
    p = plan()
    head = render(msgs, p, thinking)
    full = render(msgs + [asst], p, thinking)
    assert full.startswith(head), "rendering of the prompt must be a prefix of the rendering with the answer"
    return full[len(head):].removesuffix(e.eos_token)


def stream_all(text, p, thinking=False, sizes=None, seed=0):
    sp = tools.StreamParser(enc(), thinking, p)
    rnd = random.Random(seed)
    ev, i = [], 0
    while i < len(text):
        n = sizes if isinstance(sizes, int) else rnd.randint(1, 7)
        ev += sp.feed(text[i:i + n])
        i += n
    ev += sp.finish()
    content = "".join(d.get("content", "") for d in ev)
    reasoning = "".join(d.get("reasoning_content", "") for d in ev)
    calls: dict[int, dict] = {}
    for d in ev:
        for tc in d.get("tool_calls", []):
            c = calls.setdefault(tc["index"], {"id": None, "name": None, "args": ""})
            if "id" in tc:
                assert c["id"] is None, "id sent twice"
                c["id"], c["name"] = tc["id"], tc["function"]["name"]
            c["args"] += tc["function"].get("arguments", "")
    return content, reasoning, [calls[i] for i in sorted(calls)], sp


# ------------------------------------------------------------------------------------------------ request side
def test_plan_validation():
    assert plan().choice == "auto"
    assert tools.plan_from_body({}).choice == "none"
    assert plan(tool_choice={"type": "function", "function": {"name": "now"}}).named == "now"
    for bad in [dict(tool_choice="always"), dict(tool_choice={"type": "function", "function": {"name": "nope"}}),
                dict(response_format={"type": "xml"}), dict(tools=[{"type": "retrieval"}]),
                dict(tools=[NOW, NOW])]:
        try:
            tools.plan_from_body({"tools": TOOLS, **bad})
        except tools.ToolError:
            continue
        raise AssertionError(f"accepted {bad}")
    try:
        tools.plan_from_body({"tool_choice": "required"})
        raise AssertionError("required without tools accepted")
    except tools.ToolError:
        pass
    p = tools.plan_from_body({"response_format": {"type": "json_schema", "json_schema": {
        "name": "x", "schema": {"type": "object"}, "strict": False}}})
    assert p.rf_kind == "json_schema" and p.rf_strict is False


def test_render_tools_and_history():
    e = enc()
    # no system message: one is inserted, carrying the tool schemas
    s = render([{"role": "user", "content": "Weather in Vienna?"}], plan())
    assert s.startswith(e.bos_token + "<｜System｜>") and "## Tools" in s and '"name": "get_weather"' in s
    assert s.endswith("<｜User｜>Weather in Vienna?<｜Assistant｜></think>")
    # an existing system message keeps its text, the tools are appended to it
    s = render([{"role": "developer", "content": [{"type": "text", "text": "Be brief."}]},
                {"role": "user", "content": "hi"}], plan())
    assert "<｜System｜>Be brief.\n\n## Tools" in s
    # history: assistant tool calls + tool results (results sorted by call order)
    msgs = [{"role": "user", "content": "Weather in Vienna and the time?"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Vienna"}'}},
                {"id": "b", "type": "function", "function": {"name": "now", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "b", "content": "12:00"},
            {"role": "tool", "tool_call_id": "a", "content": [{"type": "text", "text": "sunny"}]}]
    s = render(msgs, plan())
    assert '<｜DSML｜ invoke name="get_weather">\n<｜DSML｜ parameter name="city" string="true">Vienna</｜DSML｜ parameter>' in s
    assert "<tool_result>sunny</tool_result>\n\n<tool_result>12:00</tool_result><｜Assistant｜></think>" in s
    # without tools / response_format nothing changes
    m = [{"role": "user", "content": "x"}]
    assert tools.prepare_messages(m, tools.ToolPlan()) == m


def test_render_response_format():
    p = tools.plan_from_body({"response_format": {"type": "json_schema", "json_schema": {
        "name": "p", "schema": {"type": "object", "properties": {"a": {"type": "integer"}}}}}})
    s = render([{"role": "user", "content": "x"}], p)
    assert "## Response Format:" in s and '{"type": "object", "properties": {"a": {"type": "integer"}}}' in s


# ------------------------------------------------------------------------------------------------ parsing
CASES = [
    ("one call", [("get_weather", {"city": "Vienna", "days": 3})], ""),
    ("content + calls", [("get_weather", {"city": "Wien <Innere Stadt>", "opts": {"hourly": True}}),
                         ("now", {})], "Let me check."),
    ("no-arg call", [("now", {})], ""),
    ("unicode + quotes", [("get_weather", {"city": 'São "Paulo"\nBR', "unit": "celsius"})], "Olá"),
]


def test_parse_final_roundtrip():
    for name, calls, content in CASES:
        text = completion_of(calls, content)
        c, r, out = tools.parse_final(enc(), text, False, plan())
        assert r is None, name
        assert c == (content or None), (name, c)
        assert [(o["function"]["name"], json.loads(o["function"]["arguments"])) for o in out] == calls, name
        assert all(o["id"].startswith("call_") and o["type"] == "function" for o in out), name
        assert len({o["id"] for o in out}) == len(out), name


def test_parse_final_thinking():
    text = completion_of([("now", {})], "ok", thinking=True, reasoning="I need the time.")
    c, r, out = tools.parse_final(enc(), text, True, plan())
    assert r == "I need the time." and c == "ok" and out[0]["function"]["name"] == "now"
    # never closed thinking: everything is reasoning
    c, r, out = tools.parse_final(enc(), "still thinking", True, plan())
    assert (c, r, out) == ("", "still thinking", [])


def test_parse_final_malformed_and_none():
    good = completion_of([("get_weather", {"city": "Vienna", "days": 2})])
    bad = {
        "truncated": good[: len(good) - 20],
        "bad json": good.replace('string="false">2<', 'string="false">two<'),
        "unknown tool": good.replace('name="get_weather"', 'name="rm_rf"'),
        "duplicate param": good.replace('string="false">2</｜DSML｜ parameter>',
                                        'string="false">2</｜DSML｜ parameter>\n<｜DSML｜ parameter name="days" '
                                        'string="false">3</｜DSML｜ parameter>'),
        "garbage": "\n\n<｜DSML｜ calls>\nhello",
    }
    for name, text in bad.items():
        c, r, out = tools.parse_final(enc(), text, False, plan())
        assert out == [] and c == text, name
    # tool_choice none: DSML is never parsed
    c, _, out = tools.parse_final(enc(), good, False, plan(tool_choice="none"))
    assert out == [] and c == good
    # plain text
    assert tools.parse_final(enc(), "hello", False, plan()) == ("hello", None, [])


def test_streaming_matches_final():
    for thinking in (False, True):
        for name, calls, content in CASES:
            text = completion_of(calls, content, thinking=thinking, reasoning="hmm <think-ish> " if thinking else "")
            ref_c, ref_r, ref = tools.parse_final(enc(), text, thinking, plan())
            for sizes in (1, 3, None):
                c, r, sc, sp = stream_all(text, plan(), thinking, sizes)
                assert c == (ref_c or ""), (name, sizes, c, ref_c)
                assert r == (ref_r or ""), (name, sizes)
                assert [(x["name"], x["args"]) for x in sc] == \
                       [(o["function"]["name"], o["function"]["arguments"]) for o in ref], (name, sizes, sc)
                assert [x["id"] for x in sc] == [t["id"] for t in sp.tool_calls]
                for x in sc:
                    json.loads(x["args"])


def test_parallel_false_keeps_first_call():
    text = completion_of([("now", {}), ("get_weather", {"city": "x"})])
    p = plan(parallel_tool_calls=False)
    _, _, out = tools.parse_final(enc(), text, False, p)
    _, _, sc, _ = stream_all(text, p, False, 3)
    assert [o["function"]["name"] for o in out] == ["now"] and [c["name"] for c in sc] == ["now"]


def test_streaming_holdback():
    # text that looks like the start of a block but is not: nothing lost, nothing reordered
    for text in ["a\n\n<｜DSM", "x <｜DSML｜ call", "line\n\n", "\n\nok", "</thi"]:
        c, r, calls, _ = stream_all(text, plan(), False, 1)
        assert c == text and not calls, (text, c)
    # newlines before a real block are dropped (as parse_final does), not streamed early
    text = completion_of([("now", {})], "hi")
    c, _, calls, _ = stream_all(text, plan(), False, 1)
    assert c == "hi" and calls[0]["name"] == "now"


def test_streaming_malformed():
    good = completion_of([("get_weather", {"city": "Vienna", "days": 2}), ("now", {})])
    # broken before any call: the whole block comes back as content, like parse_final
    for text in ["\n\n<｜DSML｜ calls>\nhello", good.replace('name="get_weather"', 'name="nope"')]:
        c, _, calls, _ = stream_all(text, plan(), False, 2)
        assert not calls and c == text, text
    # broken after the first call started: the streamed call is closed as valid JSON and streaming stops
    text = good.replace('string="false">2<', 'string="false">two<')
    c, _, calls, _ = stream_all(text, plan(), False, 2)
    assert len(calls) == 1 and json.loads(calls[0]["args"]) == {"city": "Vienna"}
    # truncated (max_tokens) mid-call: same
    c, _, calls, _ = stream_all(good[:good.index("days")], plan(), False, 2)
    assert len(calls) == 1 and json.loads(calls[0]["args"]) == {"city": "Vienna"}


def test_streaming_reasoning_without_tools():
    text = "thought one\nthought two</think>\n\nanswer"
    c, r, calls, _ = stream_all(text, tools.ToolPlan(), True, 1)
    assert (c, r, calls) == ("answer", "thought one\nthought two", [])
    assert tools.split_reasoning(text, True, "</think>") == ("answer", "thought one\nthought two")


# ------------------------------------------------------------------------------------------------ grammars
def _matcher_factory(texts):
    import xgrammar as xg

    vocab = sorted(set("".join(texts)) | set("abcdefghijklmnopqrstuvwxyz0123456789{}[]\":,. <>/\n=-_")) + ["</s>"]
    info = xg.TokenizerInfo(vocab, stop_token_ids=[len(vocab) - 1])
    comp = xg.GrammarCompiler(info)

    def accepts(grammar_str, text):
        m = xg.GrammarMatcher(comp.compile_grammar(grammar_str))
        for ch in text:
            if not m.accept_token(vocab.index(ch)):
                return False
        return m.accept_token(len(vocab) - 1)  # complete: EOS allowed

    return accepts


def _have_xgrammar():
    try:
        import xgrammar  # noqa: F401
        return True
    except ImportError:
        print("  (xgrammar not installed: grammar tests skipped)")
        return False


def test_tool_grammar():
    if not _have_xgrammar():
        return
    e = enc()
    block = lambda calls: completion_of(calls)  # noqa: E731  (no content: the grammar starts at the block)
    ok = [block([("get_weather", {"city": "Vienna"})]),
          block([("get_weather", {"city": "Wien", "days": 3, "unit": "celsius", "opts": {"hourly": False}})]),
          block([("now", {}), ("get_weather", {"city": "x"})]),
          block([("get_weather", {"city": "a<b <<" + "x" * 34})])]  # "<" without "/" is text; 40 = maxLength
    bad = [block([("get_weather", {"days": 3})]),  # missing required
           block([("get_weather", {"city": "x", "unit": "kelvin"})]),  # enum
           block([("get_weather", {"city": "x", "days": 9})]),  # maximum
           block([("get_weather", {"city": "x", "opts": {}})]),  # nested required
           block([("get_weather", {"city": "x" * 41})]),  # maxLength
           block([("get_weather", {"city": "x", "extra": 1})]),  # unknown parameter
           "Sure." + block([("now", {})])]  # content before the block
    accepts = _matcher_factory(ok + bad)
    g = str(tools.build_grammar(e, plan(tool_choice="required")))
    for t in ok:
        assert accepts(g, t), t
    for t in bad:
        assert not accepts(g, t), t
    # named function: only that one; parallel_tool_calls false: exactly one call
    g = str(tools.build_grammar(e, plan(tool_choice={"type": "function", "function": {"name": "now"}})))
    assert accepts(g, block([("now", {})])) and not accepts(g, ok[0])
    g = str(tools.build_grammar(e, plan(tool_choice="required", parallel_tool_calls=False)))
    assert accepts(g, ok[0]) and not accepts(g, ok[2])
    # whatever the grammar admits is parseable: every accepted text above parses back to the same calls
    for t in ok:
        assert tools.parse_final(e, t, False, plan())[2]


def test_json_grammar_and_so_args():
    if not _have_xgrammar():
        return
    e = enc()
    schema = {"type": "object", "properties": {"name": {"type": "string"}, "tags": {"type": "array", "items": {
        "type": "string", "enum": ["a", "b"]}}, "n": {"type": "integer"}}, "required": ["name", "n"]}
    p = tools.plan_from_body({"response_format": {"type": "json_schema", "json_schema": {"name": "s", "schema": schema}}})
    good = ['{"name": "x", "tags": ["a"], "n": 1}', '{"name":"x","n":2}', '{\n  "name": "y",\n  "n": 0\n}']
    bad = ['{"name": "x"}', '{"name": "x", "n": 1, "z": 2}', '{"name": "x", "tags": ["c"], "n": 1}', "hello"]
    tool_text = completion_of([("now", {})])
    accepts = _matcher_factory(good + bad + [tool_text])
    g = str(tools.build_grammar(e, p))
    assert all(accepts(g, t) for t in good) and not any(accepts(g, t) for t in bad)
    # json_object: any object
    g = str(tools.build_grammar(e, tools.plan_from_body({"response_format": {"type": "json_object"}})))
    assert accepts(g, '{"a": [1, {"b": null}]}') and not accepts(g, "[1]") and not accepts(g, "x")
    # auto tools + response_format: JSON or a tool call
    g = str(tools.build_grammar(e, plan(response_format={"type": "json_schema", "json_schema": {"schema": schema}})))
    assert accepts(g, good[0]) and accepts(g, tool_text) and not accepts(g, "hello")
    # auto without response_format: unconstrained; none: unconstrained
    assert tools.so_args(e, plan(), False, [1]) is None and tools.so_args(e, plan(tool_choice="none"), False, [1]) is None
    so = tools.so_args(e, plan(tool_choice="required"), True, [128822])
    assert so["schema_type"] == "ebnf" and so["trigger_token_ids"] == [128822] and so["schema"].startswith("root ::=")
    try:
        tools.so_args(e, tools.plan_from_body({"response_format": {"type": "json_schema", "json_schema": {
            "schema": {"type": "object", "properties": {"a": {"$ref": "#/nowhere"}}}}}}), False, [1])
        raise AssertionError("bad schema accepted")
    except tools.ToolError:
        pass


def test_bounded_json_string_has_no_control_chars():
    """xgrammar 0.2.2 admits raw TAB etc. in length-bounded JSON strings; tools.build_grammar patches the class."""
    if not _have_xgrammar():
        return
    p = tools.plan_from_body({"response_format": {"type": "json_schema", "json_schema": {"schema": {
        "type": "object", "properties": {"s": {"type": "string", "maxLength": 8}}, "required": ["s"]}}}})
    good, bad = '{"s": "ab c"}', '{"s": "ab\tc"}'
    accepts = _matcher_factory([good, bad])
    g = tools.build_grammar(enc(), p)
    assert accepts(g, good) and not accepts(g, bad)


def test_integer_ranges_exact():
    """xgrammar 0.2.2 admits 121-129 for 0..120; tools.py rewrites small ranges as enums."""
    if not _have_xgrammar():
        return
    p = tools.plan_from_body({"response_format": {"type": "json_schema", "json_schema": {"schema": {
        "type": "object", "properties": {"a": {"type": "integer", "minimum": 0, "maximum": 120},
                                         "b": {"type": "integer", "exclusiveMinimum": -3, "exclusiveMaximum": 3}},
        "required": ["a", "b"]}}}})
    ok = ['{"a": 0, "b": -2}', '{"a": 120, "b": 2}', '{"a": 99, "b": 0}']
    bad = ['{"a": 121, "b": 0}', '{"a": 129, "b": 0}', '{"a": -1, "b": 0}', '{"a": 5, "b": 3}', '{"a": 5, "b": -3}']
    accepts = _matcher_factory(ok + bad)
    g = tools.build_grammar(enc(), p)
    assert all(accepts(g, t) for t in ok) and not any(accepts(g, t) for t in bad)


def main() -> int:
    global ENC_PATH
    if "--encoding" in sys.argv:
        ENC_PATH = sys.argv[sys.argv.index("--encoding") + 1]
    if not os.path.isfile(ENC_PATH):
        print(f"encoding.py not found at {ENC_PATH} (--encoding / $DS41_ENCODING)")
        return 2
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"[PASS] {name}")
            except Exception as exc:  # noqa: BLE001
                fails += 1
                print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
    print(f"{'OK' if not fails else f'{fails} FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
