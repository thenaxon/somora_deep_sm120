"""Engram for the reference runner (ds41-inference/inference/{engram.py,model.py}, unmodified).

The official `ParallelEngramEmbedding` allocates the table as FP8/E8M0 parameters on the default device
(model.py:296-326): 98 GB + 3 GB per table on CUDA, which no SM120 GPU has. This helper swaps that one
class for `MmapEngramEmbedding`, which has the same forward math (model.py:310-325) but reads the rows it
needs from an mmap of the HF safetensors shard (sm120/sm120_engram.py, page cache / NVMe). The rest
of Engram -- `Engram` (wkv projection, gate), `NgramHashState` (hashing), `EngramLayout` (primes) -- stays
the official code.

How ref_runner.py uses it (see docs/ENGRAM.md "Requests"):

    import ref_engram
    ref_kernels.install(ref_src); import model as M
    ref_engram.install(M)                                    # before M.Transformer(...)
    margs = ...  (engram_layer_ids / engram_num_embeddings from the HF config; drop the "no engram" assert)
    tok = ref_engram.tokenizer_for_engram(args.tokenizer)    # NgramHashState needs a tokenizer
    with torch.device("cuda"): model = M.Transformer(margs, tok)
    load_weights(model, ckpt_file)                           # ref-mp1 must contain engram.wkv/q_weight/k_weight
    ref_engram.attach(model, hf_ckpt_dir)                    # the shard with layers.<id>.engram.embed.*

The ref-format checkpoint must not contain the tables (ref_convert drops ``.engram.embed.``) but must keep
``layers.<id>.engram.{wkv.weight,wkv.scale,q_weight,k_weight}``; MmapEngramEmbedding has no parameters,
so a strict load_state_dict needs nothing else.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SHIMS = os.path.normpath(os.path.join(HERE, "..", "..", "sm120"))
if SHIMS not in sys.path:
    sys.path.insert(0, SHIMS)

import torch  # noqa: E402
from torch import nn  # noqa: E402

import sm120_engram  # noqa: E402


class MmapEngramEmbedding(nn.Module):
    """Drop-in for model.py's ParallelEngramEmbedding (world_size 1): no parameters, rows from a table."""

    block_size = 32

    def __init__(self, num_embeddings: int, dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.dim = dim
        self.part_num_embeddings = num_embeddings
        self.vocab_start_idx, self.vocab_end_idx = 0, num_embeddings
        self.table: sm120_engram.EngramHostTable | None = None

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if self.table is None:
            raise RuntimeError("MmapEngramEmbedding: call ref_engram.attach(model, ckpt_dir) after building the model")
        shape = indices.shape
        flat = indices.reshape(-1)
        n = flat.numel()
        mask = (flat < self.vocab_start_idx) | (flat >= self.vocab_end_idx)
        t = self.table
        if t.kind in ("cuda", "uva"):
            w = torch.empty((n, t.row_bytes), dtype=torch.uint8, device=flat.device)
            s = torch.empty((n, t.groups), dtype=torch.uint8, device=flat.device)
            t.gather_cuda(flat.to(torch.int64), w, s)
        else:
            idx = flat.to("cpu", torch.int64)
            w = torch.empty((n, t.row_bytes), dtype=torch.uint8, device="cpu", pin_memory=flat.is_cuda)
            s = torch.empty((n, t.groups), dtype=torch.uint8, device="cpu", pin_memory=flat.is_cuda)
            t.gather_host(idx, w, s)
            w, s = w.to(flat.device), s.to(flat.device)
        weight = w.view(torch.float8_e4m3fn).view(*shape, self.dim)
        scales = s.view(torch.float8_e8m0fnu).view(*shape, self.dim // self.block_size)
        # model.py:316-319, verbatim apart from the source of `values` / `scales`
        values = weight.float().unflatten(-1, (-1, self.block_size)) * scales.float().unsqueeze(-1)
        values = values.flatten(-2).to(torch.bfloat16)
        values = values.masked_fill(mask.view(shape).unsqueeze(-1), 0)
        return values


def install(M) -> None:
    """Swap the table class in the official model module; call before M.Transformer(...)."""
    world = getattr(M, "world_size", 1)
    if world != 1:
        raise NotImplementedError("ref_engram: world_size 1 only")
    M.ParallelEngramEmbedding = MmapEngramEmbedding


def attach(model, ckpt_dir: str, layer_map: dict[int, int] | None = None, *, mode: str = "host",
           device=None) -> list[sm120_engram.EngramHostTable]:
    """Give every Engram layer its table. `layer_map` maps model layer id -> layer id named in `ckpt_dir`
    (identity by default). The hash multipliers are seeded with the *model* layer id, so a renumbered
    layer must keep its original index (truncated checkpoints: keep layer 1 at index 1)."""
    tables = []
    for layer in model.layers:
        eng = getattr(layer, "engram", None)
        if eng is None:
            continue
        src = (layer_map or {}).get(eng.layer_id, eng.layer_id)
        table = sm120_engram.EngramHostTable.from_checkpoint(ckpt_dir, src)
        if mode == "uva":
            table = table.to_uva(device)
        if table.rows != eng.embed.num_embeddings:
            raise ValueError(f"layer {eng.layer_id}: table has {table.rows} rows, config says {eng.embed.num_embeddings}")
        eng.embed.table = table
        tables.append(table)
    return tables


class _TokenizerAdapter:
    """What engram.build_compressed_token_map needs from a transformers tokenizer, on a raw `tokenizers`
    Tokenizer: `len()` and `.backend_tokenizer` (decode / id_to_token)."""

    def __init__(self, tok):
        self.backend_tokenizer = tok

    def __len__(self) -> int:
        return self.backend_tokenizer.get_vocab_size(with_added_tokens=True)


def tokenizer_for_engram(tokenizer_json: str):
    from tokenizers import Tokenizer

    return _TokenizerAdapter(Tokenizer.from_file(tokenizer_json))


def engram_state_dict_keys(model) -> list[str]:
    """The engram parameter names the ref-format checkpoint must provide (for a precise strict-load check)."""
    return [k for k in model.state_dict() if ".engram." in k]


__all__ = ["MmapEngramEmbedding", "install", "attach", "tokenizer_for_engram", "engram_state_dict_keys"]
