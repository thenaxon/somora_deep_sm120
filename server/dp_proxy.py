"""Data-parallel front end for server.py (`--dp N`).

DeepInfer serves V4.1 on several GPUs as DP attention + EP MoE (docs/MULTI_GPU.md): every GPU is its own engine
process (`engine_rank` r of `data_parallel_size` N) with its own scheduler, requests and KV cache. The MoE layers
exchange tokens over the EP group, and idle ranks run placeholder steps so that all ranks enter every layer in lockstep
(the scheduler votes over the DP group). Serving therefore needs one engine loop per rank plus a router. This
module:

* spawns N copies of server.py (`--dp-rank r --port base+1+r --host 127.0.0.1`, same arguments otherwise); each builds
  its engine with engine_rank r, data_parallel_size = N, expert_parallel_size = --ep (default N);
* runs a reverse proxy on the public port. Only the public routes are forwarded (PUBLIC_POST / PUBLIC_GET; the
  ranks' /internal/* routes are not reachable through it). POST requests go to the rank with the fewest in-flight
  requests; `/v1/models` goes to rank 0; `/health` aggregates all ranks (queried in parallel). Response bodies are
  streamed through unchanged (SSE). A client disconnect closes the upstream connection, which makes that rank abort
  the request. A rank that cannot be reached before the response headers gives a 502 / 503 JSON error.
* supervises the ranks: a rank that has not answered /health within --start-timeout, or that exits during start-up
  or later (or reports a dead engine), stops the proxy and all other ranks, and the server exits non-zero (the other
  ranks may be blocked in a collective with it; a restart policy such as docker --restart can then act).

Prefix caches are per rank: the proxy is not prefix-aware.

NCCL P2P: on PCIe hosts whose GPU pairs sit behind different root ports (nvidia-smi topo: NODE / SYS), NCCL's
default P2P level (PXB) refuses peer-to-peer and routes every EP collective through host shared memory (all 32
channels "via SHM" on the 4x RTX PRO 6000 host). Before the ranks start, the proxy runs a short peer-copy self-test
on all GPU pairs in a subprocess (cudaDeviceCanAccessPeer + a verified peer copy, 60 s timeout). If it passes,
the ranks get NCCL_P2P_LEVEL=SYS; if not, NCCL keeps its default. An explicit NCCL_P2P_LEVEL / NCCL_P2P_DISABLE
in the environment always wins; SM120_NCCL_P2P_LEVEL=<level> forces a level, =default skips the test.
"""

from __future__ import annotations

import http.client
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger("di_server.dp")

PUBLIC_POST = ("/v1/completions", "/v1/chat/completions")  # forwarded to the least-loaded rank
PUBLIC_GET = ("/v1/models",)  # forwarded to rank 0 (/health and /v1/health are answered by the proxy)
UPSTREAM_ERRORS = (OSError, http.client.HTTPException)  # refused, reset, timeout, malformed / no response


class Backends:
    def __init__(self, ports: list[int]):
        self.ports = ports
        self.inflight = [0] * len(ports)
        self.lock = threading.Lock()

    def acquire(self) -> int:
        with self.lock:
            i = min(range(len(self.ports)), key=lambda j: self.inflight[j])
            self.inflight[i] += 1
            return i

    def release(self, i: int) -> None:
        with self.lock:
            self.inflight[i] -= 1


def rank_health(ports: list[int], timeout: float = 5.0) -> list[dict]:
    """GET /health of every rank, in parallel (a slow or dead rank costs one timeout, not one per rank)."""

    def one(p: int) -> dict:
        try:
            return json.loads(urllib.request.urlopen(f"http://127.0.0.1:{p}/health", timeout=timeout).read())
        except urllib.error.HTTPError as exc:  # 503 from a rank whose engine died: its body says so
            try:
                return json.loads(exc.read())
            except Exception:  # noqa: BLE001
                return {"status": "dead", "error": repr(exc)[:200]}
        except Exception as exc:  # noqa: BLE001
            return {"status": "unreachable", "error": repr(exc)[:200]}

    with ThreadPoolExecutor(max_workers=max(1, len(ports))) as ex:
        return list(ex.map(one, ports))


class Watchdog:
    """Polls the rank processes (and every `health_every` seconds their /health) after start-up. on_fail(msg) is
    called once when a rank exits or reports a dead engine (status "dead" twice in a row: the engine thread is gone
    while the HTTP process lives on). An unreachable or slow /health alone is not a failure (busy rank)."""

    def __init__(self, procs, ports: list[int], on_fail, interval: float = 1.0, health_every: float = 5.0):
        self.procs, self.ports, self.on_fail = procs, ports, on_fail
        self.interval, self.health_every = interval, health_every
        self._stop = threading.Event()
        self._dead = [0] * len(ports)
        self.failure: str | None = None
        self._thread = threading.Thread(target=self._run, name="dp-watchdog", daemon=True)

    def start(self) -> "Watchdog":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def check(self, with_health: bool) -> str | None:
        for r, p in enumerate(self.procs):
            if p.poll() is not None:
                return f"rank {r} (pid {p.pid}) exited with {p.returncode}"
        if with_health and self.ports:
            for r, h in enumerate(rank_health(self.ports)):
                self._dead[r] = self._dead[r] + 1 if h.get("status") == "dead" else 0
                if self._dead[r] >= 2:
                    return f"rank {r} reports a dead engine: {str(h.get('error') or h)[:200]}"
        return None

    def _run(self) -> None:
        last_health = time.monotonic()
        while not self._stop.wait(self.interval):
            now = time.monotonic()
            with_health = now - last_health >= self.health_every
            if with_health:
                last_health = now
            msg = self.check(with_health)
            if msg and not self._stop.is_set():
                self.failure = msg
                log.error("%s: stopping the proxy and all ranks", msg)
                self.on_fail(msg)
                return


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    backends: Backends
    max_request_bytes: int = 1 << 62
    procs: list = []  # rank processes (503 instead of 502 for a rank that has exited)

    def log_message(self, fmt, *args):
        log.debug("%s %s", self.address_string(), fmt % args)

    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code: int, msg: str, typ: str = "invalid_request_error") -> None:
        self._json(code, {"error": {"message": msg, "type": typ, "code": code}})

    def _route(self) -> str:
        return urllib.parse.urlsplit(self.path).path

    def _forward(self, i: int, method: str, body: bytes | None) -> None:
        c = http.client.HTTPConnection("127.0.0.1", self.backends.ports[i], timeout=3600)
        try:
            try:
                c.request(method, self.path, body, {"Content-Type": self.headers.get("Content-Type", "application/json")})
                r = c.getresponse()
            except UPSTREAM_ERRORS as exc:  # nothing sent to the client yet: a proper error response
                gone = i < len(self.procs) and self.procs[i].poll() is not None
                log.warning("rank %d %s: %r", i, "has exited" if gone else "unreachable", exc)
                return self._error(503 if gone else 502, f"rank {i} {'is not running' if gone else 'unreachable'}: "
                                   f"{type(exc).__name__}", "server_error")
            self.send_response(r.status)
            length = r.getheader("Content-Length")
            for k in ("Content-Type", "Cache-Control"):
                if r.getheader(k):
                    self.send_header(k, r.getheader(k))
            if length is not None:
                self.send_header("Content-Length", length)
            else:
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            while True:
                try:
                    chunk = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
                except UPSTREAM_ERRORS as exc:  # headers are out: all that is left is to end the response early
                    log.warning("rank %d: response cut off: %r", i, exc)
                    self.close_connection = True
                    break
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        finally:
            c.close()  # on client disconnect: closes upstream -> the rank's handler sees BrokenPipe and aborts

    def do_GET(self):
        try:
            route = self._route()
            if route in ("/health", "/v1/health"):
                ranks = rank_health(self.backends.ports)
                ok = all(r.get("status") == "ok" for r in ranks)

                def agg(key):
                    vals = [r.get(key) for r in ranks if isinstance(r.get(key), dict)]
                    if not vals:
                        return None
                    out = {}
                    for k in vals[0]:
                        xs = [v.get(k) for v in vals if isinstance(v.get(k), (int, float))]
                        if xs:
                            # all ranks step in lockstep: steps / flushes are a per-rank count, not a sum
                            out[k] = max(xs) if k.startswith("max") or k in ("steps", "flushes", "mtp_decode_steps") else sum(xs)
                    return out

                body = json.dumps({"status": "ok" if ok else "degraded", "inflight": self.backends.inflight,
                                   "stats": agg("stats"), "prefix_cache": agg("prefix_cache"),
                                   "config": next((r.get("config") for r in ranks if r.get("config")), None),
                                   "ranks": ranks}).encode()
                self.send_response(200 if ok else 503)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if route not in PUBLIC_GET:
                return self._error(404, f"no route {route}")
            self._forward(0, "GET", None)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        try:
            route = self._route()
            if route not in PUBLIC_POST:  # e.g. the ranks' /internal/* maintenance routes
                self.close_connection = True  # body left unread
                return self._error(404, f"no route {route}")
            if self.headers.get("Transfer-Encoding"):
                self.close_connection = True
                return self._error(400, "chunked request bodies are not supported: send Content-Length")
            raw_n = self.headers.get("Content-Length")
            try:
                n = int(raw_n) if raw_n is not None else 0
            except ValueError:
                n = -1
            if n < 0:
                self.close_connection = True
                return self._error(400, f"invalid Content-Length {raw_n!r}")
            if n > self.max_request_bytes:
                self.close_connection = True
                return self._error(413, f"request body of {n} bytes exceeds --max-request-bytes {self.max_request_bytes}")
            body = self.rfile.read(n)
            i = self.backends.acquire()
            try:
                self._forward(i, "POST", body)
            finally:
                self.backends.release(i)
        except (BrokenPipeError, ConnectionResetError):
            pass


_P2P_TEST = r"""
import sys, torch
n = torch.cuda.device_count()
bad = []
for i in range(n):
    for j in range(n):
        if i == j:
            continue
        if not torch.cuda.can_device_access_peer(i, j):
            bad.append(f"{i}->{j} no peer access")
            continue
        a = torch.arange(1 << 22, dtype=torch.int32, device=f"cuda:{i}")
        b = a.to(f"cuda:{j}")
        torch.cuda.synchronize(i); torch.cuda.synchronize(j)
        if not torch.equal(b.cpu(), a.cpu()):
            bad.append(f"{i}->{j} copy mismatch")
print("P2P_OK" if n > 1 and not bad else "P2P_FAIL " + "; ".join(bad or [f"{n} GPU(s)"]), flush=True)
"""


def nccl_p2p_env(dp: int) -> dict:
    """Extra environment for the ranks: NCCL_P2P_LEVEL=SYS when peer copies work between all GPU pairs."""
    forced = os.environ.get("SM120_NCCL_P2P_LEVEL", "").strip()
    if dp <= 1 or "NCCL_P2P_LEVEL" in os.environ or "NCCL_P2P_DISABLE" in os.environ or forced.lower() == "default":
        return {}
    if forced:
        log.info("NCCL_P2P_LEVEL=%s (SM120_NCCL_P2P_LEVEL)", forced)
        return {"NCCL_P2P_LEVEL": forced}
    try:
        res = subprocess.run([sys.executable, "-c", _P2P_TEST], capture_output=True, text=True, timeout=60)
        verdict = (res.stdout.strip().splitlines() or [res.stderr.strip()[-200:]])[-1]
    except Exception as exc:  # noqa: BLE001 - a hanging or failing test just keeps NCCL's default
        verdict = f"P2P_FAIL {type(exc).__name__}"
    if verdict == "P2P_OK":
        log.info("GPU peer copies verified on all pairs: NCCL_P2P_LEVEL=SYS for the ranks")
        return {"NCCL_P2P_LEVEL": "SYS"}
    log.warning("GPU peer-copy self-test failed (%s): NCCL keeps its default P2P level", verdict)
    return {}


def policy_fatal(reply: str | None, err: BaseException | None, pinned: bool) -> str | None:
    """The proxy's verdict on rank 0's engram post-load reply: a reason to stop all ranks, or None to serve.
    In pinned mode the shared tables are only usable after a confirmed fill: no reply, an unreadable one or one
    without the fill is fatal. Otherwise only an explicit "fatal" (SM120_ENGRAM_PIN) is."""
    if err is not None:
        if pinned:
            return f"no engram post-load reply from rank 0 ({err!r})"
        log.warning("engram page-cache policy request failed: %r", err)
        return None
    try:
        result = json.loads(reply).get("result") or {}
    except Exception as exc:  # noqa: BLE001
        return f"unreadable engram post-load reply from rank 0 ({exc!r})" if pinned else None
    if result.get("fatal"):
        return result["fatal"]
    if pinned and "pinned_fill_s" not in result:
        return f"rank 0 did not confirm the fill of the shared engram tables: {result}"
    return None


def run(argv: list[str], dp: int, host: str, port: int, start_timeout: float = 0.0,
        max_request_bytes: int = 1 << 62) -> int:
    """argv: server.py arguments without --dp/--host/--port. Spawns the ranks and serves the proxy until interrupted
    (-> 0); a rank that fails (watchdog) or a start-up past start_timeout raises SystemExit (exit code 1) after the
    ranks were stopped. start_timeout: seconds for all ranks to answer /health (0 = no limit)."""
    ports = [port + 1 + r for r in range(dp)]
    shm_fds = None
    if os.environ.get("SM120_ENGRAM_MODE") == "pinned":  # one shared, pinned copy of the engram tables for all ranks
        ckpt = argv[argv.index("--ckpt") + 1] if "--ckpt" in argv else next(
            (a.split("=", 1)[1] for a in argv if a.startswith("--ckpt=")), None)
        import sm120_engram

        try:
            shm_fds = sm120_engram.create_shared_tables(ckpt, ranks=dp)
        except sm120_engram.EngramResidencyError as exc:
            raise SystemExit(f"engram tables cannot be pinned: {exc}") from None
    env = dict(os.environ, **nccl_p2p_env(dp))
    procs = []
    for r, p in enumerate(ports):
        cmd = [sys.executable, __file__.replace("dp_proxy.py", "server.py"), *argv, "--dp", str(dp), "--dp-rank", str(r),
               "--host", "127.0.0.1", "--port", str(p)]
        log.info("rank %d: %s", r, " ".join(cmd))
        # own process group: a rank that has to be killed takes its children (multimodal workers) with it
        procs.append(subprocess.Popen(cmd, env=env, pass_fds=shm_fds or (), start_new_session=True))
    start_failed = False
    try:
        t0 = time.time()
        pending = set(range(dp))
        while pending:
            if start_timeout and time.time() - t0 > start_timeout:
                start_failed = True
                raise SystemExit(f"rank(s) {sorted(pending)} not up after {start_timeout:.0f}s (--start-timeout)")
            for r in list(pending):
                if procs[r].poll() is not None:  # the others may be stuck in a collective with it: stop them all
                    start_failed = True
                    raise SystemExit(f"rank {r} exited with {procs[r].returncode} during start-up")
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{ports[r]}/health", timeout=2)
                    pending.discard(r)
                    log.info("rank %d up after %.0fs", r, time.time() - t0)
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(1)
        reply, err = None, None
        try:  # every rank has loaded its weights: Engram page-cache policy (prefault etc.) once, on rank 0, before serving
            req = urllib.request.Request(f"http://127.0.0.1:{ports[0]}/internal/engram_cache_policy", data=b"{}",
                                         headers={"Content-Type": "application/json"})
            reply = urllib.request.urlopen(req, timeout=3600).read().decode()
            log.info("engram page-cache policy on rank 0: %s", reply)
        except Exception as exc:  # noqa: BLE001
            err = exc
        fatal = policy_fatal(reply, err, pinned=shm_fds is not None)
        if fatal:  # SM120_ENGRAM_PIN cannot be honoured / pinned tables not filled: stop all ranks, never serve
            start_failed = True
            raise SystemExit(f"engram tables cannot be pinned: {fatal}")
        ProxyHandler.backends = Backends(ports)
        ProxyHandler.max_request_bytes = max_request_bytes
        ProxyHandler.procs = procs
        ThreadingHTTPServer.request_queue_size = 1024  # listen backlog (socketserver default 5)
        httpd = ThreadingHTTPServer((host, port), ProxyHandler)
        httpd.daemon_threads = True
        log.info("dp proxy: %d ranks on ports %s, serving on http://%s:%d", dp, ports, host, port)
        # a rank that dies while serving: stop serving; the finally block below stops the other ranks
        watchdog = Watchdog(procs, ports, lambda msg: threading.Thread(target=httpd.shutdown, daemon=True).start()).start()
        try:
            httpd.serve_forever()
        finally:
            watchdog.stop()
        if watchdog.failure:  # the finally block stops the other ranks (they may be stuck in a collective with it)
            start_failed = True
            raise SystemExit(f"{watchdog.failure}: stopped all ranks")
    except KeyboardInterrupt:
        pass
    finally:
        _stop_ranks(procs, start_failed)
        if shm_fds:  # the last references to the pinned segment: the kernel frees its pages here, one memfd per thread
            import sm120_engram

            log.info("shutdown: engram segment (%d memfds) released in %.2f s", len(shm_fds),
                     sm120_engram.release_shared_fds(shm_fds))
    return 0


def _stop_ranks(procs: list, start_failed: bool) -> None:
    """SIGTERM every rank at once, wait for all of them against one deadline, then SIGKILL the rest. The ranks agree
    on the stop in their engine loops (EngineLoop._poll) and close together; one that is still there at the
    deadline (blocked in a collective, e.g. after a failed start-up) is killed. SM120_SHUTDOWN_TIMEOUT (default 7 s)
    keeps the whole stop inside docker stop's default 10 s grace period."""
    t0 = time.monotonic()
    timeout = float(os.environ.get("SM120_SHUTDOWN_TIMEOUT", "7"))
    log.info("shutdown: stopping %d ranks (SIGTERM, %.0f s until SIGKILL)", len(procs), timeout)
    for p in procs:
        if p.poll() is None:
            p.terminate()
    deadline = t0 + timeout
    left = dict(enumerate(procs))
    while left:
        for r, p in list(left.items()):
            if p.poll() is not None:
                log.info("shutdown: rank %d exited with %s after %.2f s", r, p.returncode, time.monotonic() - t0)
                del left[r]
        if not left or time.monotonic() >= deadline:
            break
        time.sleep(0.05)
    for r, p in left.items():
        log.warning("shutdown: rank %d (pid %d) still running after %.1f s: killing it%s", r, p.pid,
                    time.monotonic() - t0, " (failed start-up)" if start_failed else "")
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except OSError:
            p.kill()
    for r, p in left.items():
        p.wait()
        log.info("shutdown: rank %d killed, gone after %.2f s", r, time.monotonic() - t0)
    log.info("shutdown: all ranks stopped after %.2f s", time.monotonic() - t0)
