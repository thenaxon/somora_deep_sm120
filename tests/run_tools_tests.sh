#!/usr/bin/env bash
# Tool calling / response_format tests inside the container: CPU unit tests, then start server.py on --ckpt, run
# test_tools_server.py and (if an `openai` SDK is importable by $SDK_PY) test_openai_sdk.py, optionally the plumbing
# tests (SERVER_TESTS=1: test_server.py), stop the server.
#   run_tools_tests.sh <ckpt> <port> <log> [server args...] -- [test_tools_server.py args...]
# SDK venv, once:  python3 -m venv /work/sdkvenv && /work/sdkvenv/bin/pip install openai pydantic
#   then SDK_PY=/work/sdkvenv/bin/python. Container paths: SM120_ROOT=/opt/sm120 (this repo).
SM120_ROOT=${SM120_ROOT:-/opt/sm120}; SDK_PY=${SDK_PY:-python3}
set -u
CKPT=$1; PORT=$2; LOG=$3; shift 3
SARGS=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do SARGS+=("$1"); shift; done; [ $# -gt 0 ] && shift
RC=0
echo "== unit tests (CPU)"
python3 $SM120_ROOT/tests/test_tools.py --encoding "$CKPT/encoding/encoding.py" || RC=1
echo "== server ${SARGS[*]}"
setsid python3 $SM120_ROOT/server/server.py --ckpt "$CKPT" --port "$PORT" "${SARGS[@]}" > "$LOG" 2>&1 &
SPID=$!
python3 - "$PORT" <<'PY' || RC=1
import sys, time, urllib.request
for i in range(900):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/health", timeout=2); print(f"server up after {i}s"); break
    except Exception:
        time.sleep(1)
else:
    sys.exit("server did not come up")
PY
URL="http://127.0.0.1:$PORT"
echo "== test_tools_server.py"
EXTRA=(); [[ " ${SARGS[*]} " == *" --deterministic "* ]] || EXTRA+=(--nondeterministic-ok)
[[ " ${SARGS[*]} " == *" --prefix-cache "* ]] && EXTRA+=(--prefix-cache)
timeout 1500 python3 $SM120_ROOT/tests/test_tools_server.py --url "$URL" "${EXTRA[@]}" "$@" || RC=1
if "$SDK_PY" -c "import openai, pydantic" 2>/dev/null; then
    echo "== test_openai_sdk.py"
    timeout 900 "$SDK_PY" $SM120_ROOT/tests/test_openai_sdk.py --url "$URL" || RC=1
else
    echo "== test_openai_sdk.py skipped: no openai/pydantic for $SDK_PY"
fi
if [ "${SERVER_TESTS:-0}" = 1 ]; then
    echo "== test_server.py"
    ND=(); [[ " ${SARGS[*]} " == *" --deterministic "* ]] || ND=(--nondeterministic-ok)
    timeout 1500 python3 $SM120_ROOT/tests/test_server.py --url "$URL" "${ND[@]}" ${SERVER_TEST_ARGS:-} || RC=1
fi
kill -- -$SPID 2>/dev/null; wait $SPID 2>/dev/null
for _ in $(seq 90); do  # GPU memory released (the next server in this container must not OOM)
    U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1); [ "${U:-0}" -lt 3000 ] && break; sleep 1
done
grep -E "Traceback|Error|SO\]" "$LOG" | grep -v "BadRequest" | head -20
exit $RC
