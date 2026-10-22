#!/usr/bin/env python3
"""CPU unit tests for server/dp_proxy.py against fake ranks (plain HTTP servers): route allowlist, request framing,
upstream errors before the response headers (502 / 503 JSON), parallel /health and the rank watchdog.

    python3 tests/test_dp_proxy_unit.py      (or: pytest tests/test_dp_proxy_unit.py)

The real thing (a killed rank of a running --force-proxy server) is in tests/run_proxy_tests.sh (GPU).
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "server"))

import dp_proxy as P  # noqa: E402


class FakeRank:
    """Answers /health after `delay` seconds; POSTs are recorded and echoed."""

    def __init__(self, delay: float = 0.0, status: str = "ok"):
        self.calls: list[tuple[str, str]] = []
        self.delay, self.status = delay, status
        rank = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                rank.calls.append(("GET", self.path))
                time.sleep(rank.delay)
                self._send(200 if rank.status == "ok" else 503, {"status": rank.status, "stats": {"steps": 3}})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n)
                rank.calls.append(("POST", self.path))
                self._send(200, {"echo": json.loads(body or b"{}"), "path": self.path})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p  # nothing listens here: connection refused


class FakeProc:
    def __init__(self, rc=None):
        self.returncode, self.pid = rc, 4242

    def poll(self):
        return self.returncode


def proxy(ports, procs=None, max_bytes=1 << 20):
    handler = type("PH", (P.ProxyHandler,), {"backends": P.Backends(ports), "max_request_bytes": max_bytes,
                                              "procs": procs or []})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


def request(port, method, path, body=None, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
    data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
    c.request(method, path, data, {"Content-Type": "application/json", **(headers or {})})
    r = c.getresponse()
    out = r.read()
    c.close()
    return r.status, json.loads(out or b"{}")


def raw(port, data: bytes) -> int:
    s = socket.create_connection(("127.0.0.1", port), timeout=10)
    s.sendall(data)
    head = s.recv(65536)
    s.close()
    return int(head.split(b" ")[1])


def test_route_allowlist_and_framing():
    r = FakeRank()
    httpd, port = proxy([r.port])
    try:
        for path in ("/internal/engram_cache_policy", "/internal/x", "/v1/embeddings", "/"):
            st, body = request(port, "POST", path, {})
            assert st == 404 and "error" in body, (path, st, body)
        st, _ = request(port, "GET", "/internal/engram_cache_policy")
        assert st == 404
        assert r.calls == [], r.calls  # nothing reached the rank
        st, body = request(port, "POST", "/v1/completions?x=1", {"prompt": "hi"})
        assert st == 200 and body["echo"] == {"prompt": "hi"} and body["path"] == "/v1/completions?x=1", body
        st, body = request(port, "POST", "/v1/chat/completions", {"messages": []})
        assert st == 200
        st, body = request(port, "GET", "/v1/models")
        assert st == 200 and r.calls[-1] == ("GET", "/v1/models")
        n = len(r.calls)
        for cl in (b"-1", b"abc"):
            assert raw(port, b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Length: " + cl + b"\r\n\r\n{}") == 400
        assert raw(port, b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n") == 400
        # over the limit: answered from the headers alone (the body is never read; sending it could hit a closed socket)
        assert raw(port, b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Length: " + str(2 << 20).encode()
                   + b"\r\n\r\n") == 413
        assert len(r.calls) == n
    finally:
        httpd.shutdown()
        r.close()


def test_upstream_errors_are_json():
    dead = free_port()
    httpd, port = proxy([dead], procs=[FakeProc(None)])
    try:
        st, body = request(port, "POST", "/v1/completions", {"prompt": "hi"})
        assert st == 502 and "unreachable" in body["error"]["message"], (st, body)
        httpd.RequestHandlerClass.procs = [FakeProc(-9)]  # the rank process is gone
        st, body = request(port, "POST", "/v1/completions", {"prompt": "hi"})
        assert st == 503 and "not running" in body["error"]["message"], (st, body)
        st, body = request(port, "GET", "/v1/models")
        assert st == 503
        assert httpd.RequestHandlerClass.backends.inflight == [0]
    finally:
        httpd.shutdown()


def test_health_is_parallel():
    ranks = [FakeRank(delay=1.5) for _ in range(4)]
    httpd, port = proxy([r.port for r in ranks] + [free_port()])
    try:
        t = time.time()
        st, body = request(port, "GET", "/health")
        dt = time.time() - t
        assert dt < 3.0, dt  # serial: 4 x 1.5 s
        assert st == 503 and body["status"] == "degraded" and body["ranks"][-1]["status"] == "unreachable", body
        assert body["stats"]["steps"] == 3
    finally:
        httpd.shutdown()
        for r in ranks:
            r.close()


def test_watchdog():
    procs = [FakeProc(None), FakeProc(None)]
    fired = []
    wd = P.Watchdog(procs, [], fired.append, interval=0.05).start()
    time.sleep(0.2)
    assert not fired
    procs[1].returncode = -9
    for _ in range(100):
        if fired:
            break
        time.sleep(0.02)
    wd.stop()
    assert fired == ["rank 1 (pid 4242) exited with -9"] and wd.failure == fired[0], fired
    # a rank whose engine thread died answers /health with status "dead": a failure after two checks in a row
    r = FakeRank(status="dead")
    wd = P.Watchdog([FakeProc(None)], [r.port], lambda m: None)
    assert wd.check(True) is None and "dead engine" in (wd.check(True) or ""), wd._dead
    # unreachable (busy) is not a failure
    wd = P.Watchdog([FakeProc(None)], [free_port()], lambda m: None)
    assert wd.check(True) is None and wd.check(True) is None
    r.close()


def main() -> int:
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"[PASS] {name}")
            except Exception as exc:  # noqa: BLE001
                fails += 1
                print(f"[FAIL] {name}: {type(exc).__name__}: {str(exc)[:400]}")
    print("OK" if not fails else f"{fails} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
