#!/usr/bin/env bash
# DP proxy on a real engine (inside the container, one GPU): server.py --force-proxy (proxy + one rank) on <ckpt>,
# then test_server.py through the proxy, the route allowlist, and a killed rank: the proxy must stop the server with a
# non-zero exit code within a few seconds (watchdog), so that a restart policy can act.
#   run_proxy_tests.sh <ckpt> <port> <log> [server args...] -- [test_server args...]
set -u
SM120_ROOT=${SM120_ROOT:-/opt/sm120}
CKPT=$1; PORT=$2; LOG=$3; shift 3
SARGS=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do SARGS+=("$1"); shift; done; [ $# -gt 0 ] && shift
python3 "$SM120_ROOT/server/server.py" --ckpt "$CKPT" --port "$PORT" --force-proxy "${SARGS[@]}" > "$LOG" 2>&1 &
SPID=$!
python3 - "$PORT" <<'PY' || { echo "proxy did not come up"; tail -30 "$LOG"; kill $SPID; exit 1; }
import sys, time, urllib.request
for i in range(1200):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/health", timeout=2); print(f"proxy up after {i}s"); sys.exit(0)
    except Exception:
        time.sleep(1)
sys.exit(1)
PY
fails=0
timeout 1500 python3 "$SM120_ROOT/tests/test_server.py" --url "http://127.0.0.1:$PORT" "$@" || fails=$((fails + 1))
python3 - "$PORT" <<'PY' || fails=$((fails + 1))
import json, sys, urllib.error, urllib.request
port = sys.argv[1]
def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        return urllib.request.urlopen(req, timeout=60).status
    except urllib.error.HTTPError as e:
        return e.code
ok = True
for path, want in (("/internal/engram_cache_policy", 404), ("/v1/completions", 200)):
    st = post(path, {"prompt": "hi", "max_tokens": 2})
    print(f"[{'PASS' if st == want else 'FAIL'}] proxy POST {path} -> {st} (want {want})")
    ok &= st == want
st = post("/v1/completions", [])
print(f"[{'PASS' if st == 400 else 'FAIL'}] proxy POST non-object body -> {st} (want 400, from the rank)")
ok &= st == 400
sys.exit(0 if ok else 1)
PY
RANK=$(pgrep -f -- "--dp-rank 0" | head -1)
echo "killing rank 0 (pid $RANK) with SIGKILL"
kill -9 "$RANK"
T0=$(date +%s)
for _ in $(seq 60); do kill -0 $SPID 2>/dev/null || break; sleep 1; done
if kill -0 $SPID 2>/dev/null; then
    echo "[FAIL] proxy still running $(( $(date +%s) - T0 ))s after the rank was killed"; kill $SPID; fails=$((fails + 1))
else
    wait $SPID; RC=$?
    echo "[$([ $RC -ne 0 ] && echo PASS || echo FAIL)] proxy exited with $RC $(( $(date +%s) - T0 ))s after the rank was killed"
    [ $RC -ne 0 ] || fails=$((fails + 1))
fi
grep -E "exited with|stopping the proxy|exiting with" "$LOG" | tail -4
echo "proxy tests: $([ $fails = 0 ] && echo OK || echo "$fails FAILED")"
exit $fails
