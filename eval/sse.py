"""Server-sent events (SSE) for the eval / bench clients: one parser for every OpenAI-compatible stream (stdlib only).

Reads the decoded HTTP body with HTTPResponse.readline(), never the raw socket (`r.fp`): with Transfer-Encoding:
chunked (vLLM, most proxies) a chunk boundary can fall inside an SSE event, and the raw socket would hand chunk-size
lines and event fragments to the JSON parser.

A stream counts as complete only if it ends with `data: [DONE]` or with a choice whose finish_reason is set, and no
`{"error": ...}` event arrived. Anything else (error event, EOF before the end, malformed event) is a failed request;
the tokens received until then are kept as partial output, separate from the successful ones.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field


@dataclass
class StreamResult:
    ok: bool = False
    error: str | None = None
    done: bool = False  # saw data: [DONE]
    finish_reason: str | None = None
    usage: dict | None = None
    stamps: list = field(default_factory=list)  # perf_counter() of every chunk that carried output text
    text: str = ""
    n_events: int = 0


def read_stream(r, on_event=None) -> StreamResult:
    """Consume an SSE response (http.client.HTTPResponse, status already checked). on_event(ev) sees every parsed
    event (e.g. for client-side cancellation: return True to stop reading)."""
    res = StreamResult()
    parts = []
    try:
        while True:
            line = r.readline()  # decoded body (handles chunked transfer encoding)
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                res.done = True
                break
            try:
                ev = json.loads(data)
            except ValueError as exc:
                res.error = f"malformed SSE event: {exc}: {data[:120]!r}"
                break
            res.n_events += 1
            if isinstance(ev, dict) and ev.get("error") is not None:
                err = ev["error"]
                res.error = "error event: " + (err.get("message") if isinstance(err, dict) else str(err))[:300]
                break
            if ev.get("usage"):
                res.usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta")
                txt = ((d.get("content") or "") + (d.get("reasoning_content") or "")) if d is not None \
                    else (ch.get("text") or "")
                if txt:
                    res.stamps.append(time.perf_counter())
                    parts.append(txt)
                if ch.get("finish_reason"):
                    res.finish_reason = ch["finish_reason"]
            if on_event is not None and on_event(ev):
                res.error = res.error or "stopped by the client"
                break
    except Exception as exc:  # noqa: BLE001  (connection reset, timeout, IncompleteRead)
        res.error = f"transport: {type(exc).__name__}: {exc}"[:300]
    res.text = "".join(parts)
    if res.error is None and not (res.done or res.finish_reason):
        res.error = "stream ended without [DONE] or a finish_reason (truncated)"
    res.ok = res.error is None
    return res
