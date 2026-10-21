#!/usr/bin/env bash
# somora_deep_sm120 serving entrypoint: OpenAI-compatible server (server/server.py) on one GPU.
#   --model PATH                 HF checkpoint dir (DeepSeek-V4.1-Flash or a truncated one); required
#   --served-model-name NAME     id reported by /v1/models (default: dir name)
#   --no-graphs                  eager mode (CUDA graphs are ON by default)
#   --host/--port                default 0.0.0.0:8000
#   everything else goes to server.py: --gpu-memory-utilization --max-model-len --max-num-seqs --max-num-batched-tokens
#   --prefix-cache [--prefix-cache-grain 512 ...] --nextn N --deterministic --prefill-interval ...
#   --help                       this text + server.py --help
set -euo pipefail
SERVER="${SM120_ROOT:-/opt/sm120}/server/server.py"
args=() model="" graphs=1 host=0.0.0.0 port=8000
while [ $# -gt 0 ]; do
    case "$1" in
        --model) model="$2"; shift 2 ;;
        --model=*) model="${1#*=}"; shift ;;
        --served-model-name) args+=(--model-name "$2"); shift 2 ;;
        --no-graphs) graphs=0; shift ;;
        --graphs) graphs=1; shift ;;
        --host) host="$2"; shift 2 ;;
        --port) port="$2"; shift 2 ;;
        -h|--help) sed -n '2,12p' "$0"; echo; exec python3 "$SERVER" --help ;;
        *) args+=("$1"); shift ;;
    esac
done
[ -n "$model" ] || { echo "error: --model PATH is required (see --help)" >&2; exit 2; }
[ "$graphs" = 1 ] && args+=(--graphs)
mkdir -p "${SM120_CACHE_ROOT:-/cache}" 2>/dev/null || true
exec python3 "$SERVER" --ckpt "$model" --host "$host" --port "$port" "${args[@]}"
