#!/bin/bash
# Prefill TTFT with a cold Engram cache, one server per mode (docs/ENGRAM.md §5 "Prefill TTFT with a cold engram cache").
# Runs on the docker host. Before every server start the engram shards' clean page-cache pages are dropped
# (posix_fadvise DONTNEED, no root needed); every request uses fresh random token ids (tests/ttft_conc.py).
#   IMG=somora_deep_sm120:0.2.0 CKPT_DIR=/path/to/hf/shards WORK=/path/to/work CACHE_DIR=/path/to/cache MODEL=/work/ckpt_full \
#   MODES="host host_nopf:SM120_ENGRAM_PREFETCH=0 uring" bash tests/engram_cold_ttft.sh
# Mode entries are name[:ENV=VAL,ENV=VAL]; a name containing "uring" runs with io_uring allowed (seccomp, memlock).
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
IMG=${IMG:-somora_deep_sm120:0.2.0}
MODEL=${MODEL:-/work/ckpt_full}
CASES=${CASES:-4096x1,4096x1,32768x1,4096x4,32768x4}
N=engram-ttft-srv
XDG=/work/.cache; [ -n "${CACHE_DIR:-}" ] && XDG=/cache
# CACHE_DIR (optional, recommended): JIT caches (b12x, TileLang, the prefetch helper) survive between the servers, so the
# first request of each server does not pay for compilation
MOUNTS="-v $CKPT_DIR:/ckpt:ro -v $WORK:/work -v $REPO:/opt/sm120 ${CACHE_DIR:+-v $CACHE_DIR:/cache}"
SHARDS=$(docker run --rm $MOUNTS --entrypoint python3 $IMG -c "
import json, os, sys; sys.path.insert(0, '/opt/sm120/sm120'); import sm120_engram as E
cfg = json.load(open('$MODEL/config.json')); ids = cfg.get('text_config', cfg).get('engram_layer_ids') or []
print(' '.join(sorted({os.path.realpath(E.find_engram_tensors('$MODEL', l)[0].path) for l in ids})))")
drop() { docker run --rm $MOUNTS --entrypoint python3 $IMG -c "
import os, sys
for f in sys.argv[1:]:
    fd = os.open(f, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)" $SHARDS; }
up() { for i in $(seq 900); do
         [ "$(docker inspect -f '{{.State.Running}}' $N 2>/dev/null)" = true ] || { docker logs $N 2>&1 | tail -20; return 1; }
         docker exec $N python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" \
             2>/dev/null && return 0; sleep 2; done; return 1; }
for m in ${MODES:-host uring}; do
  name=${m%%:*}; envs=""
  [ "$name" != "$m" ] && for e in $(echo "${m#*:}" | tr , ' '); do envs="$envs -e $e"; done
  case $name in *uring*) envs="$envs -e SM120_ENGRAM_MODE=uring --security-opt seccomp=unconfined --ulimit memlock=-1";;
                *) envs="$envs -e SM120_ENGRAM_MODE=host";; esac
  docker rm -f $N >/dev/null 2>&1; drop
  docker run -d --name $N --gpus all --ipc=host --shm-size=16g -e SM120_ENGRAM_STATS=1 -e XDG_CACHE_HOME=$XDG \
      $envs $MOUNTS $IMG --model $MODEL --max-num-batched-tokens 4096 --max-model-len 40960 --max-blocks 2048 \
      --max-num-seqs 8 --gpu-memory-utilization 0.95 >/dev/null
  echo "## $name ($envs)"
  if up; then docker exec $N python3 /opt/sm120/tests/ttft_conc.py --cases $CASES --tag " $name"; fi
  docker stop $N >/dev/null; docker logs $N 2>&1 | grep -E "\[sm120_engram\] (stats|prefetch stats)" | tail -2 | cut -c1-300
  docker rm $N >/dev/null
done
