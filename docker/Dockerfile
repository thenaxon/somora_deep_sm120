# somora_deep_sm120: DeepInfer on SM120 (RTX 5090 / RTX PRO 6000 Blackwell), self-contained serving image
# (OpenAI-compatible server).
#
#   docker build -t somora_deep_sm120:0.2.4 -f docker/Dockerfile .          # build context = repo root
#   docker run --rm --gpus all --ipc=host --shm-size=16g -p 8000:8000 -v /models:/models:ro -v sm120-cache:/cache \
#       somora_deep_sm120:0.2.4 --model /models/DeepSeek-V4.1-Flash [--prefix-cache] [--nextn 5] [--no-graphs]
#
# DeepInfer is cloned at the pinned commit (without its vendor/ submodules: FlashMLA/DeepGEMM/DeepEP/DeepSelect are
# SM100-only and replaced by sm120/). The layers up to `pip install b12x` are the same as docker/Dockerfile.dev.
FROM nvidia/cuda:13.2.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev git g++ make ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
RUN python3.12 -m venv $VIRTUAL_ENV && pip install --no-cache-dir -U pip wheel setuptools

RUN pip install --no-cache-dir torch==2.13.0+cu132 --index-url https://download.pytorch.org/whl/cu132

# DeepInfer requirements.txt without the dev include; transformers bumped for b12x (>=4.51).
RUN pip install --no-cache-dir \
        einops==0.8.2 huggingface_hub==0.36.2 ninja==1.13.0 numpy==2.3.5 orjson==3.11.7 pillow==11.3.0 \
        posix_ipc==1.3.2 psutil==7.2.2 pybind11==3.0.1 pyzmq==27.1.0 requests==2.32.5 safetensors==0.8.0 \
        scipy==1.17.0 sympy==1.14.0 tabulate==0.9.0 tilelang==0.1.15 tile-kernels==2.1.0 tokenizers==0.21.4 \
        tqdm==4.67.3 transformers==4.51.3 xgrammar==0.2.2 pytest

RUN pip install --no-cache-dir b12x==1.5.1

# ---- serving additions ---------------------------------------------------------------------------------------------
# liburing headers: b12x's O_DIRECT io_uring row reader (engram SM120_ENGRAM_MODE=uring, docs/ENGRAM.md); patch(1);
# libnuma1: DeepInfer's NUMA binding of the engine and multimodal worker processes (deep_cluster/topology/bind.py)
RUN apt-get update && apt-get install -y --no-install-recommends liburing-dev pkg-config patch libnuma1 \
    && rm -rf /var/lib/apt/lists/*

# b12x tiny-decode SwiGLU clamp fix (patches/PATCHES.md, https://github.com/flashinfer-ai/flashinfer/issues/6866).
# With it, SM120_MOE_TINY_DECODE=1 is correct; the default below stays the (faster) dynamic path.
COPY patches/b12x-tiny-decode-swiglu.patch /tmp/
RUN patch -p1 -d "$(python -c 'import b12x,os;print(os.path.dirname(os.path.dirname(b12x.__file__)))')" \
        < /tmp/b12x-tiny-decode-swiglu.patch && rm /tmp/b12x-tiny-decode-swiglu.patch

# DeepInfer at the pinned commit, host extension (_C: scheduler, block manager, cuda_utils) built with the one-line
# Sequence.block_table binding the prefix cache needs (patches/deepinfer-sequence-block-table.patch).
ARG DEEPINFER_REPO=https://github.com/deepseek-ai/DeepInfer.git
ARG DEEPINFER_COMMIT=8b16149741a4935e1dec8cea27853f273b374c5b
ENV DEEPINFER_ACCELERATOR=nvidia
RUN git init -q /opt/deepinfer && git -C /opt/deepinfer remote add origin "$DEEPINFER_REPO" \
    && git -C /opt/deepinfer fetch -q --depth 1 origin "$DEEPINFER_COMMIT" \
    && git -C /opt/deepinfer checkout -q FETCH_HEAD && rm -rf /opt/deepinfer/.git
COPY patches/deepinfer-sequence-block-table.patch /tmp/
WORKDIR /opt/deepinfer
RUN patch -p1 < /tmp/deepinfer-sequence-block-table.patch && rm /tmp/deepinfer-sequence-block-table.patch \
    && MAX_JOBS=8 bash scripts/build.sh build \
    && grep -q '"block_table"' csrc/engine/pybind_sequence.cpp   # (importing _C needs libcuda: not at build time)

# This repo: sm120/ (drop-in FlashMLA/DeepGEMM/DeepEP/DeepSelect replacements on b12x + runtime patches via
# sitecustomize), server/, prefix_cache/, tools/, eval/, tests/
COPY . /opt/sm120
RUN chmod +x /opt/sm120/docker/entrypoint.sh && python -m compileall -q /opt/sm120

# caches live under /cache (mount a volume to keep JIT / autotune results across runs); serving defaults can be
# overridden with -e
ENV SM120_ROOT=/opt/sm120 \
    PYTHONPATH=/opt/sm120/sm120:/opt/deepinfer \
    DEEPINFER_ROOT=/opt/deepinfer \
    HF_HUB_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    SM120_CACHE_ROOT=/cache \
    DEEPINFER_JIT_CACHE_DIR=/cache/jit \
    TILELANG_CACHE_DIR=/cache/tilelang \
    TRITON_CACHE_DIR=/cache/triton \
    B12X_AUTOTUNE=0 \
    SM120_MOE_TINY_DECODE=0 \
    SM120_ENGRAM_MODE=host \
    DEEPINFER_PREFILL_ATTEMPT_INTERVAL=1 \
    DEEPINFER_PREFILL_ANY_ATTEMPT_INTERVAL=1 \
    DEEPINFER_MULTIMODAL_POOL_SIZE_GB=2 \
    DEEPINFER_DISABLE_SYNC_GUARD=1

WORKDIR /
EXPOSE 8000
ENTRYPOINT ["/opt/sm120/docker/entrypoint.sh"]
CMD ["--help"]
