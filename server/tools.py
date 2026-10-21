"""OpenAI tool calling and response_format for the server, on top of the checkpoint's encoding/encoding.py.

Pure Python (xgrammar only for the grammars), so it is unit-testable on a CPU (tests/test_tools.py).

Prompt side (prepare): the request's `tools` and `response_format` go onto the first system message (one is inserted if
the conversation has none), where encoding.py renders them (render_tools, response_format_template). Assistant
`tool_calls` and `role: "tool"` messages are passed through unchanged: encoding.py renders them as DSML blocks and
<tool_result> blocks (merge_tool_messages, sort_tool_results_by_call_order).

Output side: the model writes tool calls as one DSML block after its content (V4.1 format, encoding.py):

    <content>\\n\\n<｜DSML｜ calls>
    <｜DSML｜ invoke name="get_weather">
    <｜DSML｜ parameter name="city" string="true">Vienna</｜DSML｜ parameter>
    <｜DSML｜ parameter name="days" string="false">3</｜DSML｜ parameter>
    </｜DSML｜ invoke>
    </｜DSML｜ calls><EOS>

parse_final() parses a complete completion with encoding.parse_tool_calls. StreamParser turns the text deltas into
OpenAI chat deltas (reasoning_content, content, tool_calls[i] with per-parameter `function.arguments` pieces); the
argument string it streams is byte-identical to encoding.decode_dsml_to_arguments ({"k": v, ...}).

Constrained decoding (so_args): tool_choice "required" / a named function compiles a grammar that only admits a DSML
block with 1..N calls of the allowed functions whose parameters follow the JSON schemas (string-typed parameters raw,
all others JSON; xgrammar), response_format json_object / json_schema compiles the JSON schema. Both run through
DeepInfer's structured output (SamplingParams.so_args, schema_type "ebnf"). In thinking mode the grammar is only armed
after </think> (trigger_token_ids), so the reasoning stays free.
"""

from __future__ import annotations

import copy
import json
import uuid
from dataclasses import dataclass, field

MAX_PARALLEL_CALLS = 8  # grammar bound for parallel_tool_calls (keeps a degenerate model from looping on calls)
JSON_MAX_WS = 16  # response_format: max whitespace run between JSON tokens (unbounded whitespace can loop forever)


class ToolError(ValueError):
    """Invalid tools / tool_choice / response_format in a request (HTTP 400)."""


# ------------------------------------------------------------------------------------------------ request side
@dataclass
class ToolPlan:
    tools: list[dict] = field(default_factory=list)  # OpenAI tool objects ({"type": "function", "function": {...}})
    choice: str = "none"  # none | auto | required | named
    named: str | None = None
    parallel: bool = True
    rf_kind: str | None = None  # None | json_object | json_schema
    rf_schema: dict | None = None
    rf_strict: bool = True

    @property
    def parse_tools(self) -> bool:
        return bool(self.tools) and self.choice != "none"

    @property
    def names(self) -> set[str]:
        return {t["function"]["name"] for t in self.tools}


def _text_of(content) -> str | None:
    """OpenAI content parts -> text (system / assistant messages; encoding.py expects a string there)."""
    if content is None or isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for p in content:
            if isinstance(p, dict) and p.get("type") in ("text", "input_text", "output_text"):
                out.append(p.get("text", ""))
            elif isinstance(p, str):
                out.append(p)
            else:
                raise ToolError(f"unsupported content part {p.get('type') if isinstance(p, dict) else type(p)}")
        return "\n".join(out)
    raise ToolError("message content must be a string, a list of content parts or null")


def _check_tools(tools) -> list[dict]:
    if tools is None:
        return []
    if not isinstance(tools, list):
        raise ToolError("tools must be a list")
    out, seen = [], set()
    for t in tools:
        if not isinstance(t, dict) or t.get("type", "function") != "function" or not isinstance(t.get("function"), dict):
            raise ToolError('each tool must be {"type": "function", "function": {...}}')
        f = dict(t["function"])
        name = f.get("name")
        if not isinstance(name, str) or not name or '"' in name or "\n" in name:
            raise ToolError(f"invalid tool name {name!r}")
        if name in seen:
            raise ToolError(f"duplicate tool name {name!r}")
        seen.add(name)
        params = f.get("parameters")
        if params is None:
            f["parameters"] = {"type": "object", "properties": {}}
        elif not isinstance(params, dict):
            raise ToolError(f"tool {name}: parameters must be a JSON schema object")
        out.append({"type": "function", "function": f})
    return out


def plan_from_body(body: dict) -> ToolPlan:
    p = ToolPlan(tools=_check_tools(body.get("tools")))
    tc = body.get("tool_choice")
    if tc is None:
        p.choice = "auto" if p.tools else "none"
    elif tc in ("none", "auto", "required"):
        p.choice = tc
    elif isinstance(tc, dict) and tc.get("type") == "function" and isinstance(tc.get("function"), dict):
        p.choice, p.named = "named", tc["function"].get("name")
    else:
        raise ToolError('tool_choice must be "none", "auto", "required" or {"type": "function", "function": {"name": ...}}')
    if p.choice in ("required", "named") and not p.tools:
        raise ToolError(f'tool_choice "{p.choice}" needs tools')
    if p.choice == "named" and p.named not in p.names:
        raise ToolError(f"tool_choice names unknown function {p.named!r}")
    p.parallel = bool(body.get("parallel_tool_calls", True))
    rf = body.get("response_format")
    if rf is not None:
        if not isinstance(rf, dict) or rf.get("type") not in ("text", "json_object", "json_schema"):
            raise ToolError('response_format.type must be "text", "json_object" or "json_schema"')
        if rf["type"] == "json_object":
            p.rf_kind, p.rf_schema = "json_object", {"type": "object"}
        elif rf["type"] == "json_schema":
            js = rf.get("json_schema")
            if not isinstance(js, dict) or not isinstance(js.get("schema", {}), dict):
                raise ToolError("response_format.json_schema must be {name, schema, strict?}")
            p.rf_kind, p.rf_schema = "json_schema", js.get("schema") or {"type": "object"}
            p.rf_strict = bool(js.get("strict", True)) if js.get("strict") is not None else True
    return p


def prepare_messages(messages: list[dict], plan: ToolPlan) -> list[dict]:
    """OpenAI messages -> encoding.py messages (tools / response_format on the first system message)."""
    out = []
    for m in messages:
        if not isinstance(m, dict):
            raise ToolError("each message must be an object")
        m = copy.copy(m)
        role = m.get("role")
        if role in ("system", "developer"):
            m["role"] = "system"
            m["content"] = _text_of(m.get("content"))
        elif role == "assistant":
            m["content"] = _text_of(m.get("content"))
            tcs = m.get("tool_calls")
            if tcs:
                fixed = []
                for tc in tcs:
                    fn = (tc or {}).get("function") or {}
                    if not isinstance(fn.get("name"), str):
                        raise ToolError("assistant tool_calls[].function.name is required")
                    args = fn.get("arguments", "{}")
                    fixed.append({"id": tc.get("id", ""), "type": "function",
                                  "function": {"name": fn["name"], "arguments": args if args is not None else "{}"}})
                m["tool_calls"] = fixed
            else:
                m.pop("tool_calls", None)
        elif role == "tool":
            if isinstance(m.get("content"), list):
                m["content"] = _text_of(m["content"])
        out.append(m)
    want_rf = plan.rf_kind is not None
    if not plan.tools and not want_rf:
        return out
    if not out or out[0].get("role") != "system":
        out.insert(0, {"role": "system", "content": ""})
    sys0 = dict(out[0])
    if plan.tools:
        sys0["tools"] = plan.tools
    if want_rf:
        sys0["response_format"] = plan.rf_schema
    out[0] = sys0
    return out


# ------------------------------------------------------------------------------------------------ output side
class Dsml:
    """DSML literals, taken from the checkpoint's encoding.py."""

    def __init__(self, enc):
        d = enc.dsml_token
        self.calls_open = f"<{d}{enc.tool_calls_block_name}"  # "<｜DSML｜ calls"
        self.calls_close = f"</{d}{enc.tool_calls_block_name}>"
        self.invoke_open = f"<{d}{enc.tool_call_tag_name} name=\""
        self.invoke_close = f"</{d}{enc.tool_call_tag_name}>"
        self.param_open = f"<{d}{enc.tool_parameter_tag_name} name=\""
        self.param_close = f"</{d}{enc.tool_parameter_tag_name}>"
        self.think_end = enc.thinking_end_token
        self.to_json = enc.to_json


def new_call_id() -> str:
    return "call_" + uuid.uuid4().hex[:24]


def _openai_name(call: dict) -> str:
    return f"{call['namespace']}::{call['name']}" if call.get("namespace") else call["name"]


def split_reasoning(text: str, thinking: bool, think_end: str) -> tuple[str, str | None]:
    if not thinking:
        return text, None
    if think_end in text:
        r, c = text.split(think_end, 1)
        return c.lstrip("\n"), r
    return "", text


def parse_final(enc, text: str, thinking: bool, plan: ToolPlan) -> tuple[str | None, str | None, list[dict]]:
    """Complete completion text (EOS dropped) -> (content, reasoning_content, OpenAI tool_calls). Text before the DSML
    block is the content; a block that encoding.parse_tool_calls rejects, names an unknown function or carries
    non-JSON arguments is returned as plain content (no tool calls)."""
    D = Dsml(enc)
    content, reasoning = split_reasoning(text, thinking, D.think_end)
    if not plan.parse_tools:
        return content, reasoning, []
    i = content.find(D.calls_open)
    if i < 0:
        return content, reasoning, []
    try:
        _, stop, calls = enc.parse_tool_calls(i + len(D.calls_open), content)
    except (ValueError, AssertionError):
        return content, reasoning, []
    if stop != D.calls_close or not calls:
        return content, reasoning, []
    out = []
    for c in calls:
        name = _openai_name(c)
        if name not in plan.names:
            return content, reasoning, []
        try:
            json.loads(c["arguments"])
        except ValueError:
            return content, reasoning, []
        out.append({"id": new_call_id(), "type": "function", "function": {"name": name, "arguments": c["arguments"]}})
    if not plan.parallel:
        out = out[:1]  # unconstrained (auto) output with several calls: OpenAI returns at most one
    before = content[:i].rstrip("\n")
    return (before or None), reasoning, out


def _prefix_hold(buf: str, token: str) -> int:
    """Length of the longest suffix of buf that is a proper prefix of token (text that may still become token)."""
    for k in range(min(len(buf), len(token) - 1), 0, -1):
        if token.startswith(buf[-k:]):
            return k
    return 0


class StreamParser:
    """Incremental chat output -> OpenAI deltas. feed(text) / finish() return lists of delta dicts
    ({"reasoning_content": ...}, {"content": ...}, {"tool_calls": [...]})."""

    def __init__(self, enc, thinking: bool, plan: ToolPlan):
        self.D = Dsml(enc)
        self.plan = plan
        self.state = "reasoning" if thinking else "content"
        self.lead = thinking  # split_reasoning strips the newlines right after </think>
        self.buf = ""
        self.nl = ""  # newlines between the content and the block (dropped unless the block is invalid)
        self.block = ""  # tool block text after "<｜DSML｜ calls"
        self.pos = 0  # scan position in block
        self.sub = "open"  # open | invoke | body | done | bad
        self.calls: list[dict] = []  # {"id", "name", "args"}
        self.n_params = 0

    # -- public
    def feed(self, text: str) -> list[dict]:
        self.buf += text
        return self._run(final=False)

    def finish(self) -> list[dict]:
        out = self._run(final=True)
        if self.state == "calls":
            if self.sub == "body":
                out += self._close_call()  # truncated mid-call (length): close what was streamed
            if not self.calls:  # nothing valid: the whole block is content after all (as parse_final)
                out.append({"content": self.nl + self.D.calls_open + self.block})
        return out

    @property
    def tool_calls(self) -> list[dict]:
        return [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["args"]}}
                for c in self.calls]

    # -- internals
    def _run(self, final: bool) -> list[dict]:
        out: list[dict] = []
        while True:
            if self.state == "reasoning":
                i = self.buf.find(self.D.think_end)
                if i >= 0:
                    if i:
                        out.append({"reasoning_content": self.buf[:i]})
                    self.buf = self.buf[i + len(self.D.think_end):]
                    self.state = "content"
                    continue
                keep = 0 if final else _prefix_hold(self.buf, self.D.think_end)
                if len(self.buf) > keep:
                    out.append({"reasoning_content": self.buf[:len(self.buf) - keep]})
                    self.buf = self.buf[len(self.buf) - keep:]
                return out
            if self.state == "content":
                if self.lead:
                    self.buf = self.buf.lstrip("\n")
                    if not self.buf:
                        return out
                    self.lead = False
                if self.plan.parse_tools:
                    i = self.buf.find(self.D.calls_open)
                    if i >= 0:
                        before = self.buf[:i].rstrip("\n")
                        self.nl = self.buf[len(before):i]
                        if before:
                            out.append({"content": before})
                        self.block = self.buf[i + len(self.D.calls_open):]
                        self.buf = ""
                        self.state = "calls"
                        continue
                    keep = 0 if final else _prefix_hold(self.buf, self.D.calls_open)
                    if not final:
                        j = len(self.buf) - keep
                        while j > 0 and self.buf[j - 1] == "\n":  # "\n\n" before the block is not content
                            j -= 1
                        keep = len(self.buf) - j
                else:
                    keep = 0
                if len(self.buf) > keep:
                    out.append({"content": self.buf[:len(self.buf) - keep]})
                    self.buf = self.buf[len(self.buf) - keep:]
                return out
            # calls
            self.block += self.buf
            self.buf = ""
            out += self._scan()
            return out

    def _expect(self, lit: str) -> bool | None:
        """True: lit consumed; None: need more text; False: mismatch."""
        rem = self.block[self.pos:]
        if rem.startswith(lit):
            self.pos += len(lit)
            return True
        return None if lit.startswith(rem) else False

    def _bad(self) -> list[dict]:
        self.sub = "bad"
        return self._close_call() if self.calls and self.n_params >= 0 else []

    def _close_call(self) -> list[dict]:
        c = self.calls[-1]
        piece = "}" if self.n_params else "{}"
        c["args"] += piece
        self.n_params = -1  # closed
        return [{"tool_calls": [{"index": len(self.calls) - 1, "function": {"arguments": piece}}]}]

    def _scan(self) -> list[dict]:
        D, out = self.D, []
        while True:
            rem = self.block[self.pos:]
            if self.sub == "open":
                r = self._expect(">\n")
                if r is None:
                    return out
                if not r:
                    return out + self._bad()
                self.sub = "invoke"
            elif self.sub == "invoke":  # expecting an invoke or the end of the block
                if rem.startswith(D.calls_close):
                    if not self.calls:
                        return out + self._bad()
                    self.pos += len(D.calls_close)
                    self.sub = "done"
                    return out
                if rem.startswith(D.invoke_open):
                    j = rem.find('">\n', len(D.invoke_open))
                    if j < 0:
                        return out
                    name = rem[len(D.invoke_open):j]
                    if name not in self.plan.names or "\n" in name:
                        return out + self._bad()
                    self.pos += j + 3
                    call = {"id": new_call_id(), "name": name, "args": ""}
                    self.calls.append(call)
                    self.n_params = 0
                    out.append({"tool_calls": [{"index": len(self.calls) - 1, "id": call["id"], "type": "function",
                                                "function": {"name": name, "arguments": ""}}]})
                    self.sub = "body"
                    continue
                if D.invoke_open.startswith(rem) or D.calls_close.startswith(rem):
                    return out
                return out + self._bad()
            elif self.sub == "body":  # parameters, then </invoke>
                if rem.startswith(D.param_open):
                    j = rem.find(D.param_close, len(D.param_open))
                    if j < 0 or len(rem) < j + len(D.param_close) + 1:
                        return out
                    head, _, value = rem[len(D.param_open):j].partition('">')
                    key, sep, flag = head.partition('" string="')
                    if not sep or flag not in ("true", "false") or rem[j + len(D.param_close)] != "\n":
                        return out + self._bad()
                    if flag == "true":
                        v = D.to_json(value)
                    else:
                        v = value
                        try:
                            json.loads(v)
                        except ValueError:
                            return out + self._bad()
                    piece = ("{" if self.n_params == 0 else ", ") + f"{D.to_json(key)}: {v}"
                    self.n_params += 1
                    self.calls[-1]["args"] += piece
                    self.pos += j + len(D.param_close) + 1
                    out.append({"tool_calls": [{"index": len(self.calls) - 1, "function": {"arguments": piece}}]})
                    continue
                if rem.startswith(D.invoke_close):
                    if len(rem) < len(D.invoke_close) + 1:
                        return out
                    if rem[len(D.invoke_close)] != "\n":
                        return out + self._bad()
                    self.pos += len(D.invoke_close) + 1
                    out += self._close_call()
                    self.sub = "invoke" if self.plan.parallel else "done"
                    continue
                if D.param_open.startswith(rem) or D.invoke_close.startswith(rem):
                    return out
                return out + self._bad()
            else:  # done / bad: ignore the rest
                return out


# ------------------------------------------------------------------------------------------------ grammars
def _xg():
    import xgrammar as xg

    return xg


def _lit(xg, s: str):
    return xg.Grammar.from_ebnf("root ::= " + json.dumps(s))


def _is_raw_string(sub: dict) -> bool:
    """Parameters the model writes as string="true" (raw text): plain string schemas."""
    if not isinstance(sub, dict):
        return False
    if sub.get("type") == "string":
        return not any(k in sub for k in ("anyOf", "oneOf", "allOf", "$ref", "const"))
    return "type" not in sub and isinstance(sub.get("enum"), list) and sub["enum"] and \
        all(isinstance(e, str) for e in sub["enum"])


def _raw_string_grammar(xg, sub: dict):
    if isinstance(sub.get("enum"), list) and sub["enum"]:
        alts = [_lit(xg, e) for e in sub["enum"] if isinstance(e, str)]
        return alts[0] if len(alts) == 1 else xg.Grammar.union(*alts)
    # raw text up to "</｜DSML｜ parameter>", one character per repetition (min/maxLength): any character, "<" only
    # when no "/" follows (parse_tool_calls ends the value at the first "</｜DSML｜ ...")
    lo = int(sub.get("minLength", 0) or 0)
    hi = sub.get("maxLength")
    rep = "{%d,%s}" % (lo, "" if hi is None else int(hi)) if (lo or hi is not None) else "*"
    return xg.Grammar.from_ebnf(f'root ::= rawchar{rep}\nrawchar ::= [^<] | rawlt\nrawlt ::= "<" (=([^/]))')


def _json_grammar(xg, schema: dict, strict: bool, compact: bool):
    kw = dict(any_whitespace=False) if compact else dict(any_whitespace=True, max_whitespace_cnt=JSON_MAX_WS)
    return xg.Grammar.from_json_schema(json.dumps(schema), strict_mode=strict, **kw)


def _invoke_grammar(xg, D: Dsml, fn: dict, strict: bool = True):
    params = fn.get("parameters") or {}
    props = params.get("properties") or {}
    required = set(params.get("required") or [])
    defs = {k: params[k] for k in ("$defs", "definitions") if k in params}
    parts = [_lit(xg, D.invoke_open + fn["name"] + '">\n')]
    for key, sub in props.items():
        sub = sub if isinstance(sub, dict) else {}
        if _is_raw_string(sub):
            flag, value = "true", _raw_string_grammar(xg, sub)
        else:
            flag, value = "false", _json_grammar(xg, {**sub, **defs} if defs else sub, strict, compact=True)
        p = xg.Grammar.concat(_lit(xg, f'{D.param_open}{key}" string="{flag}">'), value,
                              _lit(xg, D.param_close + "\n"))
        if key not in required:
            p = xg.Grammar.union(p, xg.Grammar.from_ebnf('root ::= ""'))
        parts.append(p)
    if not props:
        parts.append(_lit(xg, "\n"))  # encoding.py renders an argument-less call as ">\n\n</invoke>"
    parts.append(_lit(xg, D.invoke_close))
    return xg.Grammar.concat(*parts)


def _calls_grammar(xg, D: Dsml, fns: list[dict], max_calls: int):
    invokes = [_invoke_grammar(xg, D, f) for f in fns]
    one = invokes[0] if len(invokes) == 1 else xg.Grammar.union(*invokes)
    head, tail = json.dumps("\n\n" + D.calls_open + ">\n"), json.dumps("\n" + D.calls_close)
    rep = f' ("\\n" root){{0,{max_calls - 1}}}' if max_calls > 1 else ""
    g = xg.Grammar.from_ebnf(str(one) + f"\ndsml_calls ::= {head} root{rep} {tail}\n", root_rule_name="dsml_calls")
    return xg.Grammar.concat(g)  # re-rooted at "root" (DeepInfer compiles EBNF with root rule "root")


# xgrammar 0.2.2 writes length-bounded JSON strings (minLength / maxLength) with the character class below, which lets
# raw control characters (TAB, ...) into a JSON string: invalid JSON. Its unbounded strings use the correct class.
_XG_BOUNDED_STRING_CHAR = r'[^\"\\\r\n]'
_JSON_STRING_CHAR = r'[^\0-\x1f\"\\\r\n]'


def build_grammar(enc, plan: ToolPlan) -> str | None:
    """EBNF (xgrammar dialect, root rule "root") for the request, or None (unconstrained)."""
    g = _build_grammar(enc, plan)
    if g is None:
        return None
    ebnf = str(g).replace(_XG_BOUNDED_STRING_CHAR, _JSON_STRING_CHAR)
    _xg().Grammar.from_ebnf(ebnf)  # still valid
    return ebnf


def _build_grammar(enc, plan: ToolPlan):
    xg = _xg()
    D = Dsml(enc)
    calls = None
    if plan.choice in ("required", "named") or (plan.choice == "auto" and plan.rf_kind):
        fns = [t["function"] for t in plan.tools if plan.choice != "named" or t["function"]["name"] == plan.named]
        calls = _calls_grammar(xg, D, fns, MAX_PARALLEL_CALLS if plan.parallel else 1)
    if plan.choice in ("required", "named"):
        return calls
    if plan.rf_kind:
        js = _json_grammar(xg, plan.rf_schema, plan.rf_strict, compact=False)
        return xg.Grammar.union(js, calls) if calls is not None else js
    return None


def so_args(enc, plan: ToolPlan, thinking: bool, think_end_ids: list[int]) -> dict | None:
    """DeepInfer SamplingParams.so_args for the request (None: unconstrained). Raises ToolError on a bad schema."""
    try:
        g = build_grammar(enc, plan)
    except Exception as exc:  # noqa: BLE001  (xgrammar: unsupported / invalid JSON schema)
        raise ToolError(f"cannot build the output grammar: {exc}") from None
    if g is None:
        return None
    so = {"schema_type": "ebnf", "schema": g}
    if thinking:
        so["trigger_token_ids"] = list(think_end_ids)
    return so

