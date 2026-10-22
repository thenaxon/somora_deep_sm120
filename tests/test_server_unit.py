#!/usr/bin/env python3
"""CPU unit tests for server/server.py without a GPU or a checkpoint: incremental detokenisation, request validation
(everything is checked before the engine sees the request) and the request life cycle (every exit that is not a
finished generation aborts the request).

    python3 tests/test_server_unit.py [--tokenizer /ckpt/tokenizer.json]      (or: pytest tests/test_server_unit.py)

The detokeniser tests run against a byte-level fake tokenizer and, if a tokenizer.json is found (--tokenizer,
$DS41_TOKENIZER or /ckpt/tokenizer.json; needs the `tokenizers` package), against the real one as well.
"""

from __future__ import annotations

import http.client
import json
import os
import queue
import random
import socket
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "server"))

import server as S  # noqa: E402

TOKENIZER = os.environ.get("DS41_TOKENIZER", "/ckpt/tokenizer.json")


# ------------------------------------------------------------------------------------------------ detokeniser
class ByteCodec:
    """Byte-level fake tokenizer (like the checkpoint's ByteLevel decoder): a token is a byte string, decode joins the
    bytes and decodes them as UTF-8 with replacement. Counts the ids it decodes."""

    def __init__(self, vocab: list[bytes]):
        self.vocab = vocab
        self.decoded = 0

    def decode(self, ids):
        self.decoded += len(ids)
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")


class RealCodec:
    def __init__(self, path: str):
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(path)
        self.decoded = 0

    def decode(self, ids):
        self.decoded += len(ids)
        return self.tok.decode(ids, skip_special_tokens=True)

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids


class RefDetok:
    """The previous Detok (full decode after every token), the reference for byte identity."""

    def __init__(self, codec, stop):
        self.codec, self.stop = codec, [s for s in stop if s]
        self.ids, self.sent, self.text, self.stopped = [], 0, "", False
        self.hold = max((len(s) for s in self.stop), default=1) - 1

    def push(self, token_id):
        self.ids.append(token_id)
        self.text = self.codec.decode(self.ids)
        for s in self.stop:
            i = self.text.find(s, max(0, self.sent - len(s)))
            if i >= 0:
                self.text = self.text[:i]
                self.stopped = True

    def emit(self, final=False):
        end = len(self.text)
        if not (final or self.stopped):
            while end > self.sent and self.text[end - 1] == "�":
                end -= 1
            end = max(self.sent, end - self.hold)
        out = self.text[self.sent:end]
        self.sent = end
        return out


def run_detok(cls, codec, ids, stop, stream=True):
    """Feeds ids like server.py does: push, stop check, emit per token (stream) and at the end. -> (pieces, text)."""
    d, pieces = cls(codec, stop), []
    for k, t in enumerate(ids):
        d.push(t)
        finished = d.stopped or k == len(ids) - 1
        if stream:
            pieces.append(d.emit(final=finished))
        if finished:
            break
    return pieces, d.text


TEXTS = ["Grüße aus Wien, 東京 und 🌍🚀 — naïve café ✓", "emoji 👨‍👩‍👧‍👦 family, flags 🇦🇹🇯🇵, math ∑∫√",
         "plain ascii text with STOP inside and more", "日本語のテキスト。終わり。次の文。", "x" * 50 + "🙂" * 20]


def byte_vocab_and_ids(text: str, rnd: random.Random):
    """Token ids that split the UTF-8 bytes of text at random points (also inside multi-byte characters)."""
    b = text.encode()
    vocab, ids, i = [], [], 0
    while i < len(b):
        n = rnd.randint(1, 4)
        vocab.append(b[i:i + n])
        ids.append(len(vocab) - 1)
        i += n
    return vocab, ids


def test_detok_matches_full_decode_bytes():
    rnd = random.Random(1)
    stops_list = [[], ["STOP"], ["。次"], ["🚀"], ["café ✓", "Wien"], ["👧"], ["zzz"], ["xx🙂"]]
    for text in TEXTS:
        for _ in range(20):
            vocab, ids = byte_vocab_and_ids(text, rnd)
            # also invalid bytes: a lone continuation byte and a truncated sequence at the end
            vocab += [b"\x80", "🙂".encode()[:2]]
            for extra in ([], [len(vocab) - 2], [len(vocab) - 1]):
                for stop in stops_list:
                    for stream in (True, False):
                        seq = ids[:len(ids) // 2] + extra + ids[len(ids) // 2:] if extra != [len(vocab) - 1] else ids + extra
                        ref = run_detok(RefDetok, ByteCodec(vocab), seq, stop, stream)
                        new = run_detok(S.Detok, ByteCodec(vocab), seq, stop, stream)
                        assert new == ref, (text, stop, stream, extra, ref, new)


def test_detok_is_linear():
    rnd = random.Random(2)
    text = "".join(rnd.choice(["a", "ä", "東", "🙂", " ", "\n"]) for _ in range(6000))
    vocab, ids = byte_vocab_and_ids(text, rnd)
    c = ByteCodec(vocab)
    run_detok(S.Detok, c, ids, ["never"])
    assert c.decoded < 12 * len(ids), (c.decoded, len(ids))  # the full re-decode was len(ids)^2 / 2
    _, t = run_detok(S.Detok, ByteCodec(vocab), ids, [])
    assert t == text


def test_stop_string_never_leaks():
    """A stop string is matched before anything is emitted: no part of it reaches the client, also when it arrives in
    pieces (the hold back of max(len(stop)) - 1 characters is enough, because push() checks before emit())."""
    pieces = [b"xx", b"a", b"b", b"c", b"yy"]
    for split in ([0, 1, 2, 3, 4], [0, 1, 2, 3], [0, 1, 4]):
        vocab = pieces
        out, text = run_detok(S.Detok, ByteCodec(vocab), split, ["abc"])
        if split == [0, 1, 4]:  # no stop: everything comes out
            assert "".join(out) == "xxayy"
            continue
        assert "".join(out) == "xx" and text == "xx", (split, out)
        assert all("a" not in p for p in out), out
    # stop string at the very start and spanning a multi-byte character
    vocab = ["é".encode()[:1], "é".encode()[1:], b"!"]
    out, text = run_detok(S.Detok, ByteCodec(vocab), [0, 1, 2], ["é!"])
    assert "".join(out) == "" and text == ""


def _real_codec():
    path = TOKENIZER
    if not os.path.isfile(path):
        return None
    try:
        return RealCodec(path)
    except Exception:  # noqa: BLE001 - tokenizers missing
        return None


def test_detok_real_tokenizer():
    c = _real_codec()
    if c is None:
        print(f"  (skipped: no tokenizer at {TOKENIZER})")
        return
    rnd = random.Random(3)
    vocab = c.tok.get_vocab_size()
    cases = [c.encode(t) for t in TEXTS] + [c.encode(" ".join(TEXTS) * 40)]
    cases += [[rnd.randrange(vocab) for _ in range(n)] for n in (50, 500, 2000)]  # arbitrary ids: broken UTF-8 too
    for ids in cases:
        for stop in ([], ["STOP"], ["🚀"], ["。次", "Wien"]):
            for stream in (True, False):
                a, b = RealCodec.__new__(RealCodec), RealCodec.__new__(RealCodec)
                a.tok = b.tok = c.tok
                a.decoded = b.decoded = 0
                assert run_detok(S.Detok, a, ids, stop, stream) == run_detok(RefDetok, b, ids, stop, stream), (stop, len(ids))
    long_ids = [rnd.randrange(vocab) for _ in range(16000)]
    a = RealCodec.__new__(RealCodec)
    a.tok, a.decoded = c.tok, 0
    _, text = run_detok(S.Detok, a, long_ids, [])
    assert text == c.decode(long_ids) and a.decoded < 20 * len(long_ids), a.decoded


# ------------------------------------------------------------------------------------------------ HTTP handler
class FakeEngine:
    def __init__(self):
        self.max_model_len, self.nextn, self.eos_id, self.vision, self.alive = 1000, 0, 0, False, True
        self.stats, self.prefix_cache = {}, None
        self.submitted, self.aborted = [], []
        self.script = [("token", 5, False, None, None), ("token", 6, False, None, None), ("token", 0, True, "stop", None)]
        self.delay = 0.0

    def submit(self, req):
        req.request_id = str(len(self.submitted))
        self.submitted.append(req)

        def feed():
            for m in self.script:
                time.sleep(self.delay)
                req.out.put(m)

        threading.Thread(target=feed, daemon=True).start()
        return req

    def abort(self, rid):
        self.aborted.append(rid)


class FakeCodec:
    enc = None

    def __init__(self):
        self.tok = SimpleNamespace(decode=lambda ids, skip_special_tokens=False: "".join(chr(97 + i % 26) for i in ids))

    def encode(self, text):
        return [ord(ch) % 50 + 1 for ch in text]

    def decode(self, ids):
        return "".join(chr(97 + i % 26) for i in ids)

    def split_reasoning(self, text, thinking):
        return text, None


def make_app(**kw):
    app = SimpleNamespace(engine=FakeEngine(), codec=FakeCodec(), model_name="m", created=0, request_timeout=5.0,
                          max_request_bytes=1 << 16, vocab_size=100, args=SimpleNamespace(mtp_varlen=False, dp_rank=None),
                          image_limits=None)
    for k, v in kw.items():
        setattr(app, k, v)
    return app


class Server:
    def __init__(self, app):
        handler = type("H", (S.Handler,), {"app": app})
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def raw(self, data: bytes) -> tuple[int, dict]:
        s = socket.create_connection(("127.0.0.1", self.port), timeout=10)
        s.sendall(data)
        buf = b""
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
            if b"\r\n\r\n" in buf:
                head, _, rest = buf.partition(b"\r\n\r\n")
                n = int(next((ln.split(b":")[1] for ln in head.split(b"\r\n") if ln.lower().startswith(b"content-length")),
                             b"0"))
                if len(rest) >= n:
                    break
        s.close()
        head, _, rest = buf.partition(b"\r\n\r\n")
        return int(head.split(b" ")[1]), json.loads(rest or b"{}")

    def post(self, path: str, body, headers=None) -> tuple[int, dict]:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.request("POST", path, data, {"Content-Type": "application/json", **(headers or {})})
        r = c.getresponse()
        out = r.read()
        c.close()
        try:
            return r.status, json.loads(out)
        except ValueError:
            return r.status, {"raw": out.decode(errors="replace")}

    def close(self):
        self.httpd.shutdown()


def test_validation_before_submit():
    app = make_app()
    srv = Server(app)
    try:
        bad = [
            ([], 400), ("x", 400), (None, 400), ({"prompt": "x", "max_tokens": "abc"}, 400),
            ({"prompt": "x", "max_tokens": 0}, 400), ({"prompt": "x", "max_tokens": 2.5}, 400),
            ({"prompt": "x", "temperature": "NaN"}, 400), ({"prompt": "x", "temperature": -1}, 400),
            ({"prompt": "x", "top_p": 0}, 400), ({"prompt": "x", "stop": 123}, 400), ({"prompt": "x", "stop": [1]}, 400),
            ({"prompt": "x", "stream": "yes"}, 400), ({"prompt": "x", "stream_options": 1}, 400),
            ({"prompt": "x", "stream_options": {"include_usage": "1"}}, 400), ({"prompt": "x", "n": 2}, 400),
            ({"prompt": "x", "seed": "7"}, 400), ({"prompt": "x", "top_k": -5}, 400), ({"prompt": "x", "logprobs": 21}, 400),
            ({"prompt": "x", "logprobs": 2, "stream": True}, 400), ({"prompt": [1, 2, 500]}, 400),
            ({"prompt": [1, -1]}, 400), ({"prompt": [True, 2]}, 400), ({"prompt": "x", "ignore_eos": "false"}, 400),
            ({"messages": "hi"}, 400), ({"messages": [{"role": "user", "content": "x"}], "logprobs": 1}, 400),
        ]
        for body, code in bad:
            path = "/v1/chat/completions" if isinstance(body, dict) and "messages" in body else "/v1/completions"
            st, r = srv.post(path, body)
            assert st == code and "error" in r, (body, st, r)
        for raw in (b'{"prompt": "x", "temperature": NaN}', b'{"prompt": "x", "top_p": Infinity}', b"\xff\xfe", b"{"):
            st, r = srv.post("/v1/completions", raw)
            assert st == 400, (raw, st, r)
        # over --max-request-bytes: answered from the headers alone, the body is never read
        st, r = srv.raw(b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Length: " + str(1 << 17).encode()
                        + b"\r\n\r\n")
        assert st == 413, (st, r)
        for cl in (b"-5", b"abc"):
            st, r = srv.raw(b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Length: " + cl + b"\r\n\r\n{}")
            assert st == 400, (cl, st, r)
        st, r = srv.raw(b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n2\r\n{}\r\n0\r\n\r\n")
        assert st == 400, (st, r)
        assert app.engine.submitted == [], len(app.engine.submitted)
        # a valid request still works, and a finished generation is not aborted
        st, r = srv.post("/v1/completions", {"prompt": "hi", "max_tokens": 4, "temperature": 0})
        assert st == 200 and r["choices"][0]["finish_reason"] == "stop", (st, r)
        assert len(app.engine.submitted) == 1 and app.engine.aborted == []
        assert srv.post("/v1/completions", {"prompt": "hi", "logprobs": False, "seed": None, "stop": None})[0] == 200
        st, r = srv.post("/v1/completions", {"prompt": [1, 2, 3], "stream": True, "stream_options": {"include_usage": True}})
        assert st == 200 and app.engine.aborted == [], (st, r)
    finally:
        srv.close()


def test_abort_on_every_unfinished_exit():
    # engine error -> abort
    app = make_app()
    app.engine.script = [("token", 5, False, None, None), ("error", "boom")]
    srv = Server(app)
    try:
        st, r = srv.post("/v1/completions", {"prompt": "hi"})
        assert st == 500 and app.engine.aborted == ["0"], (st, r, app.engine.aborted)
        # a stop string ends the request early: one abort (the engine does not know stop strings), not two
        app.engine.script = [("token", i, False, None, None) for i in (1, 2, 3, 4)] + [("token", 0, True, "length", None)]
        st, r = srv.post("/v1/completions", {"prompt": "hi", "stop": ["cd"]})
        assert st == 200 and r["choices"][0]["text"] == "b" and app.engine.aborted == ["0", "1"], (r, app.engine.aborted)
        # client disconnects from a stream -> abort
        app.engine.script = [("token", 1, False, None, None)] * 50 + [("token", 0, True, "stop", None)]
        app.engine.delay = 0.02
        s = socket.create_connection(("127.0.0.1", srv.port), timeout=10)
        b = json.dumps({"prompt": "hi", "stream": True}).encode()
        s.sendall(b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: "
                  + str(len(b)).encode() + b"\r\n\r\n" + b)
        s.recv(100)
        s.close()
        for _ in range(200):
            if len(app.engine.aborted) == 3:
                break
            time.sleep(0.02)
        assert app.engine.aborted == ["0", "1", "2"], app.engine.aborted
    finally:
        srv.close()
    # broken pipe while the SSE headers are written (outside the old abort handler)
    app = make_app()
    h = S.Handler.__new__(S.Handler)

    class Broken:
        def write(self, b):
            raise BrokenPipeError

        def flush(self):
            raise BrokenPipeError

    h.app, h.wfile, h.request_version, h.requestline, h.command, h.path = app, Broken(), "HTTP/1.1", "", "POST", "/"
    h.client_address, h.close_connection = ("127.0.0.1", 0), False
    try:
        h._generate({"prompt": "hi", "stream": True}, chat=False)
        raise AssertionError("expected BrokenPipeError")
    except BrokenPipeError:
        pass
    assert len(app.engine.submitted) == 1 and app.engine.aborted == ["0"], app.engine.aborted


def test_finish_reason_and_misc():
    app = make_app()
    app.engine.script = [("token", 5, False, None, None), ("token", 6, True, "abort", None)]
    srv = Server(app)
    try:
        st, r = srv.post("/v1/completions", {"prompt": "hi"})
        ch = r["choices"][0]
        assert st == 200 and ch["finish_reason"] == "stop" and ch["abort_reason"] == "abort", r
        app.engine.script = [("token", 5, True, "length", None)]
        assert srv.post("/v1/completions", {"prompt": "hi"})[1]["choices"][0]["finish_reason"] == "length"
        # GET errors are answered (500), not a dropped connection
        app.engine.prefix_cache = SimpleNamespace(summary=lambda: 1 / 0)
        c = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=10)
        c.request("GET", "/health")
        resp = c.getresponse()
        assert resp.status == 500 and "ZeroDivisionError" in json.loads(resp.read())["error"]["message"]
        assert resp.getheader("Server", "").startswith("somora_deep_sm120/"), resp.getheader("Server")
        c.close()
    finally:
        srv.close()
    assert S._finish_reason("abort", False) == "stop" and S._finish_reason("weird", False) == "stop"
    assert S._abort_reason("stop", False) is None and S._abort_reason("abort", True) is None


def test_prefix_cache_pinned_blocks_thread_safe():
    """/health (an HTTP thread) counts the pinned blocks while the engine thread adds and evicts entries."""
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "prefix_cache"))
    from collections import OrderedDict

    import prefix_cache as pc

    fake = SimpleNamespace(entries=OrderedDict())
    stop = threading.Event()

    def churn():
        k = 0
        while not stop.is_set():
            fake.entries[(k, b"")] = pc.Entry(k=k, digest=b"", blocks=list(range(k % 7, k % 7 + 20)), state_id=0)
            if len(fake.entries) > 50:
                fake.entries.popitem(last=False)
            k += 1

    t = threading.Thread(target=churn, daemon=True)
    t.start()
    try:
        old = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)  # switch threads as often as possible
        for _ in range(3000):
            pc.PrefixCache._pinned_blocks(fake)
    finally:
        sys.setswitchinterval(old)
        stop.set()
        t.join()


def main() -> int:
    global TOKENIZER
    if "--tokenizer" in sys.argv:
        TOKENIZER = sys.argv[sys.argv.index("--tokenizer") + 1]
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"[PASS] {name}")
            except Exception as exc:  # noqa: BLE001
                fails += 1
                print(f"[FAIL] {name}: {type(exc).__name__}: {str(exc)[:600]}")
    print("OK" if not fails else f"{fails} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
