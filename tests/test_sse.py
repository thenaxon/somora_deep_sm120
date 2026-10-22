#!/usr/bin/env python3
"""CPU tests for the eval clients' SSE handling (eval/sse.py) and their error accounting (loadgen.py, scenarios.py).

    python3 tests/test_sse.py        (or: pytest tests/test_sse.py)

Real http.client.HTTPResponse objects over in-memory sockets (chunked transfer encoding with an SSE event split across
chunks, several events per chunk, CRLF, Connection: close), and loadgen / scenarios clients against a local server
that sends chunked SSE streams which end normally, with an error event, or truncated.
"""

from __future__ import annotations

import http.client
import http.server
import io
import json
import os
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "eval"))

import sse  # noqa: E402


class FakeSock:
    def __init__(self, data: bytes):
        self.data = data

    def makefile(self, *a, **k):
        return io.BufferedReader(io.BytesIO(self.data))


def chunked(*chunks: bytes) -> bytes:
    return b"".join(b"%x\r\n%s\r\n" % (len(c), c) for c in chunks) + b"0\r\n\r\n"


def response(body: bytes, chunked_te=True) -> http.client.HTTPResponse:
    head = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
    head += b"Transfer-Encoding: chunked\r\n\r\n" if chunked_te else b"Connection: close\r\n\r\n"
    r = http.client.HTTPResponse(FakeSock(head + body))
    r.begin()
    return r


def ev(obj) -> bytes:
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


TOK = lambda t: ev({"choices": [{"index": 0, "text": t, "finish_reason": None}]})  # noqa: E731
FIN = ev({"choices": [{"index": 0, "text": "", "finish_reason": "length"}]})
USAGE = ev({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 3}})
DONE = b"data: [DONE]\n\n"


def test_event_split_across_chunks():
    full = TOK("Hello") + TOK(" world") + FIN + USAGE + DONE
    cut = full.index(b"world") + 2  # inside the JSON of an event
    s = sse.read_stream(response(chunked(full[:cut], full[cut:])))
    assert s.ok and s.text == "Hello world" and s.usage["completion_tokens"] == 3 and s.done, s
    # the raw socket (what the clients used to read) sees chunk headers and a broken event
    r = response(chunked(full[:cut], full[cut:]))
    raw = r.fp.read()
    assert b"\r\n" in raw and raw.count(b"data:") == full.count(b"data:")  # chunk-size lines are in the raw stream


def test_many_events_per_chunk_and_crlf():
    body = (TOK("a") + TOK("b") + TOK("c") + FIN + DONE).replace(b"\n", b"\r\n")
    s = sse.read_stream(response(chunked(body)))
    assert s.ok and s.text == "abc" and len(s.stamps) == 3, s


def test_chat_deltas_and_reasoning():
    body = (ev({"choices": [{"delta": {"role": "assistant"}}]}) +
            ev({"choices": [{"delta": {"reasoning_content": "think"}}]}) +
            ev({"choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}]}) + DONE)
    s = sse.read_stream(response(chunked(body)))
    assert s.ok and s.text == "thinkanswer" and len(s.stamps) == 2 and s.finish_reason == "stop", s


def test_error_event_before_first_token():
    s = sse.read_stream(response(chunked(ev({"error": {"message": "engine crashed"}}) + DONE)))
    assert not s.ok and "engine crashed" in s.error and not s.stamps, s


def test_error_event_after_tokens_keeps_partial():
    s = sse.read_stream(response(chunked(TOK("x") + TOK("y") + ev({"error": {"message": "boom"}}) + DONE)))
    assert not s.ok and len(s.stamps) == 2 and "boom" in s.error, s


def test_eof_before_done():
    s = sse.read_stream(response(chunked(TOK("x") + TOK("y"))))  # no finish_reason, no [DONE]
    assert not s.ok and "truncated" in s.error, s


def test_eof_inside_a_chunk():
    full = chunked(TOK("x") + TOK("y") + FIN + DONE)
    s = sse.read_stream(response(full[: len(full) // 2]))  # transport cut mid-chunk
    assert not s.ok, s


def test_finish_without_done_is_complete():
    s = sse.read_stream(response(chunked(TOK("x") + FIN)))
    assert s.ok and s.finish_reason == "length" and not s.done, s


def test_connection_close_stream():
    s = sse.read_stream(response(TOK("p") + TOK("q") + FIN + DONE, chunked_te=False))
    assert s.ok and s.text == "pq", s


def test_malformed_event():
    s = sse.read_stream(response(chunked(TOK("x") + b"data: {not json\n\n" + DONE)))
    assert not s.ok and "malformed" in s.error, s


# ------------------------------------------------------------------------------------------------ clients end to end
class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    mode = "ok"

    def log_message(self, *a):
        pass

    def do_GET(self):
        body = json.dumps({"data": [{"id": "m", "max_model_len": 100000}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body.get("prompt") or body["messages"][0]["content"]
        chat = "messages" in body
        mode = Handler.mode
        if mode == "alternate":  # every second request fails mid-stream
            mode = "error" if hash(json.dumps(prompt)) % 2 else "ok"
        piece = (lambda t: ev({"choices": [{"delta": {"content": t}}]})) if chat else TOK
        events = [piece("a"), piece("b"), piece("c")]
        if mode == "ok":
            events += [ev({"choices": [{"delta": {} if chat else None, "text": "", "finish_reason": "length"}]}),
                       ev({"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3}}), DONE]
        elif mode == "error":
            events += [ev({"error": {"message": "engine crashed"}}), DONE]
        # "truncated": just stop
        data = b"".join(events)
        cut = len(data) // 3
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for c in (data[:cut], data[cut:]):  # an event split across two chunks
            self.wfile.write(b"%x\r\n%s\r\n" % (len(c), c))
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


def serve():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    srv.handle_error = lambda *a: None  # clients close early after an error event
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def test_loadgen_counts_errors_and_partials():
    import argparse

    import loadgen

    srv, url = serve()
    try:
        cl = loadgen.Client(url, None, 30)
        args = argparse.Namespace(seed=0, nonce="t", nonce_tokens=4, id_lo=1000, id_hi=2000, repeats=2, warmup=2)
        for mode, want_ok in (("ok", True), ("error", False), ("truncated", False)):
            Handler.mode = mode
            lev = loadgen.run_level(cl, 16, 2, 6, 3, args)
            if want_ok:
                assert lev["ok"] == 12 and lev["errors"] == 0 and lev["output_tok_s"] > 0, lev
                assert lev["output_tokens_mean"] == 3 and lev["tpot_p50_ms"] is not None
                assert lev["ttft_tail_kind"] == "max"  # 12 samples: no p99
            else:
                assert lev["ok"] == 0 and lev["errors"] == 12 and lev["partial_out_tokens"] == 36, lev
                assert lev["output_tok_s"] == 0 and lev["tpot_p50_ms"] is None, lev
    finally:
        srv.shutdown()


def test_scenarios_capacity_error_and_missing_tpot():
    import argparse

    import scenarios

    srv, url = serve()
    try:
        ep = scenarios.Endpoint(url, None, 30)
        a = argparse.Namespace(seed=0, nonce="t", repeats=1, warmup=1, temperature=0.0, min_requests=4,
                               requests_per_user=2, prompt_tokens=0, max_tokens=3, slo_ttft=60.0, slo_tpot=1e9,
                               full=True, concurrency="2", chars_per_token=3.6)
        Handler.mode = "ok"
        rows = scenarios.sc_capacity(ep, a)
        assert rows[0]["meets_slo"] and rows[0]["errors"] == 0, rows[0]
        Handler.mode = "error"
        rows = scenarios.sc_capacity(ep, a)
        assert not rows[0]["meets_slo"] and "errors" in rows[0]["missed"] and "tpot" in rows[0]["missed"], rows[0]
        assert rows[0]["partial_out_tokens"] == 12 and rows[0]["tpot_p50_ms"] is None, rows[0]
    finally:
        srv.shutdown()


def test_steady_state_window():
    from measure import steady_tok_s, tail

    # 2 slots, 4 requests of 10 tokens each, 1 s per request, back to back
    res = [dict(t_send=s, t_first=s + 0.1, t_last=s + 1.0, t_end=s + 1.0, out=10) for s in (0.0, 0.0, 1.0, 1.0)]
    v, win = steady_tok_s(res, 2)
    assert win == 2.0 and abs(v - 20.0) < 1e-9, (v, win)  # window [0, 2]: all 40 tokens, 20 tok/s
    # 2 slots, 6 requests: ramp until the 2nd send, tail after the 4th end; tokens outside are not counted
    res6 = [dict(t_send=s, t_first=s + 0.5, t_last=s + 1.0, t_end=s + 1.0, out=10) for s in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)]
    v, win = steady_tok_s(res6, 2)
    assert win == 2.5 and abs(v - 50 / 2.5) < 1e-9, (v, win)  # window [0.5, 3.0]: requests 0-4 inside, 5 outside
    assert steady_tok_s(res[:2], 2) == (None, None)  # requests <= slots: no steady window
    assert tail([1.0] * 99)[1] == "max" and tail([1.0] * 100)[1] == "p99" and tail([None])[0] is None


def main() -> int:
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"[PASS] {name}")
            except Exception as exc:  # noqa: BLE001
                fails += 1
                print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
    print("OK" if not fails else f"{fails} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
