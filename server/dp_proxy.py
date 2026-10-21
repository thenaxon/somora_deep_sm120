"""Data-parallel front end for server.py (`--dp N`).

DeepInfer serves V4.1 on several GPUs as DP attention + EP MoE (docs/MULTI_GPU.md): every GPU is its own engine
process (`engine_rank` r of `data_parallel_size` N) with its own scheduler, requests and KV cache. The MoE layers
exchange tokens over the EP group, and idle ranks run placeholder steps so that all ranks enter every layer in lockstep
(the scheduler votes over the DP group). Serving therefore needs one engine loop per rank plus a router. This
module:

* spawns N copies of server.py (`--dp-rank r --port base+1+r --host 127.0.0.1`, same arguments otherwise); each builds
  its engine with engine_rank r, data_parallel_size = N, expert_parallel_size = --ep (default N);
* runs a reverse proxy on the public port. POST requests go to the rank with the fewest in-flight requests;
  `/v1/models` goes to rank 0; `/health` aggregates all ranks. Response bodies are streamed through unchanged (SSE).
  A client disconnect closes the upstream connection, which makes that rank abort the request.

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
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger("di_server.dp")


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


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    backends: Backends

    def log_message(self, fmt, *args):
        log.debug("%s %s", self.address_string(), fmt % args)

    def _forward(self, i: int, method: str, body: bytes | None) -> None:
        c = http.client.HTTPConnection("127.0.0.1", self.backends.ports[i], timeout=3600)
        try:
            c.request(method, self.path, body, {"Content-Type": self.headers.get("Content-Type", "application/json")})
            r = c.getresponse()
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
                chunk = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        finally:
            c.close()  # on client disconnect: closes upstream -> the rank's handler sees BrokenPipe and aborts

    def do_GET(self):
        try:
            if self.path in ("/health", "/v1/health"):
                ranks = []
                for p in self.backends.ports:
                    try:
                        ranks.append(json.loads(urllib.request.urlopen(f"http://127.0.0.1:{p}/health", timeout=5).read()))
                    except Exception as exc:  # noqa: BLE001
                        ranks.append({"status": "dead", "error": repr(exc)[:200]})
                ok = all(r.get("status") == "ok" for r in ranks)

                def agg(key):
                    vals = [r.get(key) for r in ranks if isinstance(r.get(key), dict)]
                    if not vals:
                        return None
                    out = {}
                    for k in vals[0]:
                        xs = [v.get(k) for v in vals if isinstance(v.get(k), (int, float))]
                        if xs:
                            out[k] = max(xs) if k.startswith("max") else sum(xs)
                    return out

                body = json.dumps({"status": "ok" if ok else "degraded", "inflight": self.backends.inflight,
                                   "stats": agg("stats"), "prefix_cache": agg("prefix_cache"), "ranks": ranks}).encode()
                self.send_response(200 if ok else 503)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self._forward(0, "GET", None)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n)
        i = self.backends.acquire()
        try:
            self._forward(i, "POST", body)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.backends.release(i)


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


def run(argv: list[str], dp: int, host: str, port: int) -> int:
    """argv: server.py arguments without --dp/--host/--port. Spawns the ranks and serves the proxy until interrupted."""
    ports = [port + 1 + r for r in range(dp)]
    env = dict(os.environ, **nccl_p2p_env(dp))
    procs = []
    for r, p in enumerate(ports):
        cmd = [sys.executable, __file__.replace("dp_proxy.py", "server.py"), *argv, "--dp", str(dp), "--dp-rank", str(r),
               "--host", "127.0.0.1", "--port", str(p)]
        log.info("rank %d: %s", r, " ".join(cmd))
        procs.append(subprocess.Popen(cmd, env=env))
    try:
        t0 = time.time()
        pending = set(range(dp))
        while pending:
            for r in list(pending):
                if procs[r].poll() is not None:
                    raise SystemExit(f"rank {r} exited with {procs[r].returncode} during start-up")
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{ports[r]}/health", timeout=2)
                    pending.discard(r)
                    log.info("rank %d up after %.0fs", r, time.time() - t0)
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(1)
        try:  # every rank has loaded its weights: Engram page-cache policy (prefault etc.) once, on rank 0, before serving
            req = urllib.request.Request(f"http://127.0.0.1:{ports[0]}/internal/engram_cache_policy", data=b"{}",
                                         headers={"Content-Type": "application/json"})
            log.info("engram page-cache policy on rank 0: %s", urllib.request.urlopen(req, timeout=3600).read().decode())
        except Exception as exc:  # noqa: BLE001
            log.warning("engram page-cache policy request failed: %r", exc)
        ProxyHandler.backends = Backends(ports)
        ThreadingHTTPServer.request_queue_size = 1024  # listen backlog (socketserver default 5)
        httpd = ThreadingHTTPServer((host, port), ProxyHandler)
        httpd.daemon_threads = True
        log.info("dp proxy: %d ranks on ports %s, serving on http://%s:%d", dp, ports, host, port)
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                p.kill()
    return 0
