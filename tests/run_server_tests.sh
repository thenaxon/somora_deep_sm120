#!/usr/bin/env bash
# Server plumbing tests inside the container: start server.py on --ckpt, wait for /health, run test_server.py, stop.
#   run_server_tests.sh <ckpt> <port> <log> [server args...] -- [test args...]
# Container paths (override via env): SM120_ROOT=/opt/sm120 (this repo), WORK=/work (work dir),
# HF_CKPT=/ckpt (original HF checkpoint shards, read-only).
SM120_ROOT=${SM120_ROOT:-/opt/sm120}; WORK=${WORK:-/work}; HF_CKPT=${HF_CKPT:-/ckpt}; export SM120_ROOT WORK HF_CKPT
set -u
CKPT=$1; PORT=$2; LOG=$3; shift 3
SARGS=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do SARGS+=("$1"); shift; done; [ $# -gt 0 ] && shift
# the in-prompt prefix-cache checks repeat a ~550 / 1500-token prompt: they need checkpoints inside it (the server
# default stride is --max-num-batched-tokens, docs/PREFIX_CACHE.md)
if [[ " ${SARGS[*]} " == *" --prefix-cache "* && " ${SARGS[*]} " != *" --prefix-cache-checkpoint-stride"* ]]; then
    SARGS+=(--prefix-cache-checkpoint-stride ${PREFIX_CACHE_TEST_STRIDE:-512})
fi
# own process group (engine + multimodal workers): stopped as a whole at the end, see stop_server
setsid python3 $SM120_ROOT/server/server.py --ckpt "$CKPT" --port "$PORT" "${SARGS[@]}" > "$LOG" 2>&1 &
SPID=$!
stop_server() {
    kill -TERM -- -"$SPID" 2>/dev/null
    for _ in $(seq 60); do kill -0 -- -"$SPID" 2>/dev/null || break; sleep 1; done
    kill -KILL -- -"$SPID" 2>/dev/null
    python3 - <<'PY'
import time
import torch
for i in range(120):  # the driver releases the context's memory shortly after the process is gone
    free, total = torch.cuda.mem_get_info()
    if free > 0.85 * total:
        break
    time.sleep(1)
print(f"GPU free after stop: {free / 2**30:.1f}/{total / 2**30:.1f} GiB")
PY
}
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
stop_server
grep -E "di_server.*(done|disconnected)" "$LOG" | tail -4
exit $RC
