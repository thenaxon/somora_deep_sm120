#!/usr/bin/env bash
# Server plumbing tests inside the container: start server.py on --ckpt, wait for /health, run test_server.py, stop.
#   run_server_tests.sh <ckpt> <port> <log> [server args...] -- [test args...]
# Container paths (override via env): SM120_ROOT=/opt/sm120 (this repo), WORK=/work (work dir),
# HF_CKPT=/ckpt (original HF checkpoint shards, read-only).
SM120_ROOT=${SM120_ROOT:-/opt/sm120}; WORK=${WORK:-/work}; HF_CKPT=${HF_CKPT:-/ckpt}; export SM120_ROOT WORK HF_CKPT
set -u
CKPT=$1; PORT=$2; LOG=$3; shift 3
SARGS=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do SARGS+=("$1"); shift; done; [ $# -gt 0 ] && shift
python3 $SM120_ROOT/server/server.py --ckpt "$CKPT" --port "$PORT" "${SARGS[@]}" > "$LOG" 2>&1 &
SPID=$!
python3 - "$PORT" <<'PY'
import sys, time, urllib.request
for i in range(600):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/health", timeout=2); print(f"server up after {i}s"); break
    except Exception:
        time.sleep(1)
PY
timeout 1500 python3 $SM120_ROOT/tests/test_server.py --url "http://127.0.0.1:$PORT" "$@"
RC=$?
kill $SPID; wait $SPID 2>/dev/null
grep -E "di_server.*(done|disconnected)" "$LOG" | tail -4
exit $RC
