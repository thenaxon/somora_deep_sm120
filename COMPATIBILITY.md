# Compatibility

Version 0.3.0 is built and tested with exactly these versions (pinned in `docker/Dockerfile`):

| component | version |
|---|---|
| DeepInfer | commit [`8b16149741a4935e1dec8cea27853f273b374c5b`](https://github.com/deepseek-ai/DeepInfer/commit/8b16149741a4935e1dec8cea27853f273b374c5b) (build arg `DEEPINFER_COMMIT`), without its `vendor/` submodules |
| b12x | 1.5.1 (PyPI), with `patches/b12x-tiny-decode-swiglu.patch` applied |
| nvidia-cutlass-dsl | 4.8.0 (pulled in by b12x) |
| PyTorch | 2.13.0+cu132 |
| CUDA base image | `nvidia/cuda:13.2.1-devel-ubuntu24.04` |
| Python | 3.12 |
| NVIDIA driver (host) | >= 595 (CUDA 13.2) |
| tilelang | 0.1.15 |
| tile-kernels | 2.1.0 |
| transformers | 4.51.3 (b12x needs >= 4.51) |
| xgrammar | 0.2.2 (structured output; two bugs worked around in `server/tools.py`, see [docs/SERVER.md](docs/SERVER.md)) |
| triton | as shipped with torch 2.13.0 |

Other Python dependencies follow DeepInfer's `requirements.txt` at the pinned commit. The exact pins are in the Dockerfile.

b12x before 1.5 does not work: it lacks APIs the shims use (`b12x.preparation`, `cache_format="deepseek_v41"`,
the MXFP4 `dsa_indexer`). Newer
DeepInfer commits may change the internals that the runtime patches hook into. `sm120_patches` logs every patch it
applies and every patch it skips at start-up, so check that log after changing the commit.

## Tested GPUs

| GPU | arch | result |
|---|---|---|
| GeForce RTX 5090, 32 GB | SM120 | all tests and results in this repo |
| RTX PRO 6000 Blackwell Max-Q, 4× over PCIe | SM120 | not yet tested (multi-GPU path emulation-tested on one GPU) |

## Model

DeepSeek-V4.1-Flash in the Hugging Face layout, with FP4 experts and FP8 attention/dense weights. Truncated test
checkpoints made with `tools/ref/make_truncated_ckpt.py` from the same shards are also supported.
