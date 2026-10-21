#!/usr/bin/env bash
# Run a command against a temporary server.py: with_server.sh <ckpt> <port> <log> [server args...] -- <command...>
# The server runs in its own process group (engine + multimodal workers); on exit the whole group is terminated and
# we wait until the GPU memory is actually free, so a second server can start right after in the same container.
set -u
CKPT=$1; PORT=$2; LOG=$3; shift 3
SARGS=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do SARGS+=("$1"); shift; done; [ $# -gt 0 ] && shift
setsid python3 "$(dirname "$0")/server.py" --ckpt "$CKPT" --port "$PORT" "${SARGS[@]}" > "$LOG" 2>&1 &
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
python3 - "$PORT" <<'PY' || { echo "server did not come up"; tail -20 "$LOG"; stop_server; exit 1; }
import sys, time, urllib.request
for i in range(900):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/health", timeout=2); print(f"server up after {i}s"); sys.exit(0)
    except Exception:
        time.sleep(1)
sys.exit(1)
PY
"$@"
RC=$?
stop_server
exit $RC
