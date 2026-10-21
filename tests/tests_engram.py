#!/usr/bin/env python3
"""Engram correctness test (SM120 DeepInfer side vs. the official ds41-inference engram.py) + micro-benchmark.

DeepInfer side  = what DeepInfer runs with the SM120 shims: compressed ids -> n-gram windows (mirror of the
                  C++ fill_engram_row) -> engram_hash (tile_kernels on CUDA if importable, else the torch
                  port) -> deep_ep.EngramBuffer shim fetch from the mmap'd shard -> (fp8, packed sf) ->
                  wkv GEMM (fp32 accumulate on the fetched fp8 rows, as deep_gemm does) -> engram_gate_fwd.
Reference       = ds41-inference/inference/engram.py + model.py, unmodified: NgramHashState,
                  ParallelEngramEmbedding (filled with the needed rows read by the `safetensors` library,
                  i.e. independently of sm120_engram's mmap reader), Engram (act_quant + fp8_gemm through
                  tools/ref/ref_kernels.py in torch mode).

Checks: T0 compressed token map (DeepInfer CompressedVocabulary vs official build_compressed_token_map, if
deepinfer is importable), T1 prime layout / table rows, T2 hash ids (exact, incl. chunked prefill/decode and
image spans), T3 row lookup (bit exact, sf layout), T4 full Engram layer (tolerance, the reference
re-quantizes the bf16 rows), T5 re-quantization identity of the stored rows (explains T4 differences).

Examples (inside the container: PYTHONPATH=/opt/sm120/sm120:/opt/deepinfer, ckpt at /ckpt):
    python3 /opt/sm120/tests/tests_engram.py --ckpt /ckpt --layer 1                    # real rows, CPU or GPU
    python3 /opt/sm120/tests/tests_engram.py --ckpt /ckpt --layer 1 --bench            # + benchmark (cold & warm)
    python3 /opt/sm120/tests/tests_engram.py --synthetic /work/engram_synth            # tiny generated table
    SM120_ENGRAM_MODE=uring python3 /opt/sm120/tests/tests_engram.py --ckpt /ckpt --bench --skip-tests
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import struct
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "..", "sm120"), os.path.join(HERE, "..", "tools", "ref")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import sm120_engram as se  # noqa: E402

RESULTS: dict = {}


def log(msg: str) -> None:
    print(msg, flush=True)


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS[name] = {"ok": bool(ok), "detail": detail}
    log(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    return ok


# ------------------------------------------------------------------------------------------------ setup
def default_ref_src() -> str:
    for p in (os.environ.get("REF_SRC"), "/ds41ref/inference",
              os.path.join(HERE, "..", "third_party", "ds41-inference", "inference")):
        if p and os.path.exists(os.path.join(p, "engram.py")):
            return os.path.normpath(p)
    raise SystemExit("ds41-inference/inference not found (pass --ref-src)")


def import_reference(ref_src: str):
    os.environ.setdefault("REF_KERNELS", "torch")
    import ref_kernels

    ref_kernels.install(ref_src)
    import engram as E  # noqa: N812
    import model as M  # noqa: N812

    return M, E


class FakeBackend:
    """Synthetic tokenizer: token i decodes to word (i % n_keys), so the compressed vocab has n_keys ids."""

    def __init__(self, vocab: int, n_keys: int):
        self.vocab, self.n_keys = vocab, n_keys

    def decode(self, ids, skip_special_tokens=False):
        return f"w{ids[0] % self.n_keys}"

    def id_to_token(self, i):
        return f"t{i}"


class FakeTokenizer:
    def __init__(self, vocab: int, n_keys: int):
        self.backend_tokenizer = FakeBackend(vocab, n_keys)
        self.vocab = vocab

    def __len__(self):
        return self.vocab


@dataclasses.dataclass
class Cfg:
    layer_ids: list
    num_embeddings: list
    max_ngram: int
    vocab_size: int  # engram_vocab_size (bucket prime start)
    n_heads: int
    head_dim: int
    pad_id: int
    compressed_vocab: int
    tok_vocab: int
    dim: int
    hc_mult: int
    norm_eps: float


def cfg_from_hf(ckpt: str) -> Cfg:
    hf = json.load(open(os.path.join(ckpt, "config.json")))
    t = hf.get("text_config", hf)
    return Cfg(list(t["engram_layer_ids"]), list(t["engram_num_embeddings"]), t["engram_max_ngram_size"],
               t["engram_vocab_size"], t["engram_n_heads"], t["engram_head_dim"], t["engram_pad_token_id"],
               t["engram_compressed_vocab_size"], t["vocab_size"], t["hidden_size"], t["hc_mult"], t["rms_norm_eps"])


# ------------------------------------------------------------------------------------------------ synthetic ckpt
def _st_write(path: str, tensors: dict) -> None:
    """Minimal safetensors writer (raw bytes; dtype names as safetensors)."""
    names = {torch.float8_e4m3fn: "F8_E4M3", torch.float8_e8m0fnu: "F8_E8M0", torch.bfloat16: "BF16",
             torch.float32: "F32", torch.uint8: "U8"}
    header, off, blobs = {}, 0, []
    for k, t in tensors.items():
        b = t.contiguous().view(torch.uint8).numpy().tobytes()
        header[k] = {"dtype": names[t.dtype], "shape": list(t.shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    h = json.dumps(header).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


def quantize_rows(x: torch.Tensor):
    """bf16/fp32 [n, K] -> fp8 + E8M0 per 32 (ue8m0 rounding as kernel.py act_quant)."""
    n, K = x.shape
    xf = x.float().view(n, K // 32, 32)
    amax = xf.abs().amax(-1).clamp_min(1e-4)
    e = torch.ceil(torch.log2(amax / 448.0))
    s = torch.exp2(e)
    q = (xf / s.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn).view(n, K)
    sb = (e + 127).to(torch.uint8).view(torch.float8_e8m0fnu)
    return q, sb


def quantize_blocks(w: torch.Tensor):
    """[N, K] -> fp8 + E8M0 per 32x32 block (the checkpoint's Linear weight format)."""
    N, K = w.shape
    wb = w.float().view(N // 32, 32, K // 32, 32)
    e = torch.ceil(torch.log2(wb.abs().amax(dim=(1, 3)).clamp_min(1e-4) / 448.0))
    q = (wb / torch.exp2(e)[:, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn).view(N, K)
    return q, (e + 127).to(torch.uint8).view(torch.float8_e8m0fnu)


def make_synthetic(out_dir: str, seed: int = 0) -> str:
    os.makedirs(out_dir, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    layer_ids, max_ngram, vocab_size, n_heads, head_dim = [1, 3], 4, 1000, 8, 256
    tok_vocab, n_keys, dim, hc = 5000, 3001, 5120, 4  # dim as V4.1 (tile_kernels gate needs dim >= 512)
    primes = se.layout_primes(layer_ids, max_ngram, vocab_size, n_heads)
    rows = [sum(sum(p) for p in layer) for layer in primes]
    tensors, index = {}, {}
    for lid, r in zip(layer_ids, rows):
        q, s = quantize_rows(torch.randn((r, head_dim), generator=g) * 0.05)
        cols = (max_ngram - 1) * n_heads * head_dim
        wq, wsf = quantize_blocks(torch.randn(((hc + 1) * dim, cols), generator=g) * 0.02)
        p = f"model.layers.{lid}.engram."
        tensors[p + "embed.weight"] = q
        tensors[p + "embed.scale"] = s
        tensors[p + "wkv.weight"] = wq
        tensors[p + "wkv.scale"] = wsf
        tensors[p + "q_weight"] = (1 + 0.1 * torch.randn((hc, dim), generator=g)).to(torch.bfloat16)
        tensors[p + "k_weight"] = (1 + 0.1 * torch.randn((hc, dim), generator=g)).to(torch.bfloat16)
    shard = "model-00001-of-00001.safetensors"
    _st_write(os.path.join(out_dir, shard), tensors)
    json.dump({"metadata": {}, "weight_map": {k: shard for k in tensors}},
              open(os.path.join(out_dir, "model.safetensors.index.json"), "w"))
    json.dump({"text_config": {"engram_layer_ids": layer_ids, "engram_num_embeddings": rows,
                               "engram_max_ngram_size": max_ngram, "engram_vocab_size": vocab_size,
                               "engram_n_heads": n_heads, "engram_head_dim": head_dim, "engram_pad_token_id": 2,
                               "engram_compressed_vocab_size": n_keys, "vocab_size": tok_vocab, "hidden_size": dim,
                               "hc_mult": hc, "rms_norm_eps": 1e-6},
               "sm120_synthetic": {"n_keys": n_keys}},
              open(os.path.join(out_dir, "config.json"), "w"))
    log(f"synthetic checkpoint: {out_dir} rows={rows}")
    return out_dir


# ------------------------------------------------------------------------------------------------ helpers
def find_tensor(ckpt: str, name_suffixes: list[str]) -> se.TensorLoc | None:
    idx = os.path.join(ckpt, "model.safetensors.index.json")
    if not os.path.exists(idx):
        return None
    wm = json.load(open(idx))["weight_map"]
    for suf in name_suffixes:
        for k, shard in wm.items():
            if k.endswith(suf):
                return se.locate_tensor(os.path.join(ckpt, shard), k)
    return None


def st_rows(loc: se.TensorLoc, ids: np.ndarray) -> torch.Tensor:
    """Rows `ids` of a tensor by plain pread() at offsets from the file's own header (independent of
    sm120_engram's mmap gather). Not via safetensors.safe_open: it maps the whole file privately, which
    the kernel refuses for a 101 GB shard on a 62 GB machine (ENOMEM)."""
    with open(loc.path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        info = json.loads(f.read(n))[loc.name]
        base = 8 + n + info["data_offsets"][0]
        row = (info["data_offsets"][1] - info["data_offsets"][0]) // info["shape"][0]
        fd = f.fileno()
        out = bytearray(len(ids) * row)
        for j, i in enumerate(ids.tolist()):
            out[j * row : (j + 1) * row] = os.pread(fd, row, base + i * row)
    return torch.frombuffer(out, dtype=torch.uint8).view(len(ids), row)


def st_tensor(loc: se.TensorLoc) -> torch.Tensor:
    """Whole (small) tensor by pread; raw bytes, reshaped. Callers .view() the dtype."""
    with open(loc.path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        info = json.loads(f.read(n))[loc.name]
        a, b = info["data_offsets"]
        buf = bytearray(os.pread(f.fileno(), b - a, 8 + n + a))
    t = torch.frombuffer(buf, dtype=torch.uint8)
    dt = {"BF16": torch.bfloat16, "F32": torch.float32, "F8_E4M3": torch.float8_e4m3fn,
          "F8_E8M0": torch.float8_e8m0fnu, "U8": torch.uint8}[info["dtype"]]
    return t.view(dt).view(info["shape"])


def random_sequences(B: int, L: int, tok_vocab: int, seed: int, image_span: bool):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, tok_vocab, (B, L), generator=g)
    # repeat some n-grams so that equal windows must give equal hashes
    if L > 32:
        ids[:, L // 2 : L // 2 + 8] = ids[:, 4:12]
    mask = torch.ones((B, L), dtype=torch.bool)
    if image_span and L > 24:
        mask[0, 10:18] = False  # sequence 0 has an "image" span
    return ids, mask


def deepinfer_available() -> bool:
    try:
        import deepinfer  # noqa: F401
        import tile_kernels  # noqa: F401

        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


# ------------------------------------------------------------------------------------------------ tests
def run_tests(args, cfg: Cfg, ckpt: str, M, E, tokenizer, device: torch.device) -> None:
    use_di = args.impl == "deepinfer" or (args.impl == "auto" and deepinfer_available())
    log(f"device={device} deepinfer-ops={use_di}")
    layer = args.layer
    li = cfg.layer_ids.index(layer)

    # T1: layout
    primes = se.layout_primes(cfg.layer_ids, cfg.max_ngram, cfg.vocab_size, cfg.n_heads)
    margs = M.ModelArgs(dim=cfg.dim, hc_mult=cfg.hc_mult, norm_eps=cfg.norm_eps, max_batch_size=args.seqs,
                        max_seq_len=args.seq_len + args.decode + 8, engram_layer_ids=tuple(cfg.layer_ids),
                        engram_num_embeddings=tuple(cfg.num_embeddings), engram_max_ngram_size=cfg.max_ngram,
                        engram_vocab_size=cfg.vocab_size, engram_n_heads=cfg.n_heads, engram_head_dim=cfg.head_dim,
                        engram_pad_id=cfg.pad_id, engram_compressed_vocab_size=cfg.compressed_vocab)
    layout = E.EngramLayout.from_args(margs)
    rows_ok = [sum(sum(p) for p in lay) for lay in primes] == list(cfg.num_embeddings)
    check("T1 layout", [[list(h) for h in lay] for lay in layout.primes] == primes and rows_ok,
          f"rows={[sum(sum(p) for p in lay) for lay in primes]} config={cfg.num_embeddings}")

    # T0/T2: compressed map and hashes
    hs = E.NgramHashState(margs, layout, tokenizer)
    token_map = hs.token_map.tolist()
    if use_di and not args.synthetic:
        from deepinfer.model_executor.models.v4_modules.engram_metadata import CompressedVocabulary

        di_map = CompressedVocabulary(args.tokenizer).old2new
        check("T0 compressed token map", di_map == token_map, f"{len(di_map)} tokens, {max(di_map) + 1} ids")
    B, L = args.seqs, args.seq_len + args.decode
    ids, mask = random_sequences(B, L, cfg.tok_vocab, args.seed, image_span=True)
    # reference: prefill [0, seq_len) then one token per step, like generate.py
    ref_parts = [hs(ids[:, : args.seq_len], 0, mask[:, : args.seq_len])]
    for s in range(args.seq_len, L):
        ref_parts.append(hs(ids[:, s : s + 1], s, mask[:, s : s + 1]))
    ref_hash = torch.cat(ref_parts, dim=1)  # [B, L, n_layers, cols] int64

    if use_di:
        from deepinfer import AttrDict
        from deepinfer.model_executor.models.v4_modules.engram_metadata import NgramHashMapping
        import tile_kernels

        n_tab = cfg.max_ngram - 1
        di_cfg = AttrDict(vocab_size=[cfg.vocab_size] * n_tab, n_head_per_ngram=[cfg.n_heads] * n_tab,
                          n_embed_per_ngram=[cfg.head_dim * cfg.n_heads] * n_tab, engram_max_ngram_size=cfg.max_ngram,
                          engram_layer_ids=list(cfg.layer_ids), enable_tokenizer_compression=not args.synthetic,
                          tokenizer_name_or_path=args.tokenizer, tokenizer_vocab_size=cfg.compressed_vocab)
        hm = NgramHashMapping(di_cfg, device="cuda")
        mult, vs, offs = hm.ngram_multipliers, hm.ngram_vocab_sizes, hm.ngram_offsets
        hash_fn = tile_kernels.engram.engram_hash
    else:
        mult, vs, offs = se.hash_tensors(cfg.layer_ids, cfg.max_ngram, cfg.vocab_size, cfg.n_heads, cfg.compressed_vocab)
        hash_fn = se.engram_hash
    pad = token_map[cfg.pad_id]
    di_hash = []
    for b in range(B):
        comp = [token_map[t] if m else -1 for t, m in zip(ids[b].tolist(), mask[b].tolist())]
        rows = se.build_ngram_token_ids(comp, list(range(L)), cfg.max_ngram, pad, image_sentinel=-1)
        img = ~mask[b]
        dev = mult.device
        di_hash.append(hash_fn(rows.to(dev), mult, vs, offs, img.to(dev)).cpu())  # [n_layers, L, cols]
    di_hash = torch.stack(di_hash, dim=1).permute(1, 2, 0, 3)  # [B, L, n_layers, cols]
    text = mask
    eq_text = torch.equal(di_hash[text].long(), ref_hash[text])
    img_neg = bool((di_hash[~text] == -1).all())
    check("T2 hash ids", eq_text and img_neg,
          f"{int(text.sum())} text positions x {di_hash.shape[2]} layers x {di_hash.shape[3]} cols; "
          f"image positions -1: {img_neg}; impl={'tile_kernels' if use_di else 'torch'}")

    # T3: lookups through the EngramBuffer shim vs official ParallelEngramEmbedding on safetensors rows
    from deep_ep._engram import EngramBuffer

    table = se.open_table(ckpt, layer, device=device, max_lookups=(B * L + args.random_rows + 64) * di_hash.shape[3])
    cols = di_hash.shape[3]
    flat_hash = di_hash[:, :, li].reshape(B * L, cols).to(torch.int32)
    T = flat_hash.shape[0]
    n_rand = args.random_rows
    g = torch.Generator().manual_seed(args.seed + 1)
    rand = torch.randint(0, table.rows, (n_rand, cols), generator=g, dtype=torch.int64).to(torch.int32)
    rand[0, :3] = torch.tensor([0, table.rows - 1, -1], dtype=torch.int32)  # edges + an image id
    all_ids = torch.cat([flat_hash, rand])  # [T', cols]
    Tn = all_ids.shape[0]
    buf = EngramBuffer(None)
    buf.set_config([table.rows], cfg.head_dim, Tn, cols, torch.float8_e4m3fn, cfg.head_dim // 32 // 4)
    buf.attach_tables([table])
    idx_dev = all_ids.unsqueeze(0).contiguous().to(device)
    (hook,) = buf.fetch(idx_dev)
    data, sf = hook()
    pack_cols = cols * cfg.head_dim // 128
    layout_ok = (tuple(data.shape) == (Tn, cols * cfg.head_dim) and data.dtype == torch.float8_e4m3fn
                 and tuple(sf.shape) == (Tn, pack_cols) and sf.dtype == torch.int32
                 and sf.stride() == (1, se._align(Tn, 4)))
    di_rows = se.dequant_rows(data, sf, cfg.head_dim).cpu()  # [T', cols*hd] bf16
    wloc, sloc = se.find_engram_tensors(ckpt, layer)
    ids_np = all_ids.numpy().astype(np.int64)
    uniq, inv = np.unique(np.where(ids_np < 0, 0, ids_np), return_inverse=True)
    w_rows, s_rows = st_rows(wloc, uniq), st_rows(sloc, uniq)
    M.world_size, M.rank = 1, 0
    ref_embed = M.ParallelEngramEmbedding(len(uniq) + 1, cfg.head_dim)
    with torch.no_grad():
        ref_embed.weight.data[: len(uniq)] = w_rows.view(torch.float8_e4m3fn)
        ref_embed.scale.data[: len(uniq)] = s_rows.view(torch.float8_e8m0fnu)
    remap = torch.from_numpy(inv.reshape(ids_np.shape)).long()
    remap[torch.from_numpy(ids_np < 0)] = len(uniq) + 5  # out of range -> masked to 0 by the reference
    ref_rows = ref_embed(remap).flatten(-2)  # [T', cols*hd]
    exact = torch.equal(di_rows.view(torch.int16), ref_rows.view(torch.int16))
    check("T3 row lookup", exact and layout_ok,
          f"{Tn}x{cols} rows ({len(uniq)} unique), bit-exact={exact}, sf strides {sf.stride()} ok={layout_ok}, "
          f"table kind={table.kind}")

    # T6: CUDA graph: hooks captured in a graph, rows for every replay from (a) the in-graph host fetch
    # ("async": GPU flag + host service thread + GPU spin-wait) or (b) prestage() before replay
    if device.type == "cuda":
        for mode in ("async", "prestage"):
            os.environ["SM120_ENGRAM_GRAPH"] = mode
            ok6, dt6 = graph_check(cfg, table, cols, device, args.seed + 6, prestage=(mode == "prestage"))
            check(f"T6 graph {mode}", ok6, f"5 replays x 16 tokens, graph output == eager fetch; "
                                           f"{dt6 * 1e3:.3f} ms per replay (incl. host gather)")
        os.environ.pop("SM120_ENGRAM_GRAPH", None)

    # T5: the reference re-quantizes the bf16 rows (act_quant, ue8m0) before its fp8 GEMM; DeepInfer feeds
    # the stored fp8 rows. Equal *values* after the round trip mean both GEMMs see the same operand.
    rq_q, rq_s = quantize_rows(di_rows[:T].float().view(-1, cfg.head_dim))
    rq = (rq_q.float().view(-1, cfg.head_dim // 32, 32) * torch.exp2(rq_s.view(torch.uint8).float() - 127).unsqueeze(-1))
    lossless = (rq.view(-1, cfg.head_dim) == di_rows[:T].float().view(-1, cfg.head_dim)).all(-1)
    same_bytes = (rq_q.view(torch.uint8) == data[:T].cpu().view(-1, cfg.head_dim).view(torch.uint8)).all(-1)
    check("T5 requant identity", True, f"{float(lossless.float().mean()) * 100:.3f}% of rows reproduce their values "
                                       f"after act_quant(ue8m0) ({float(same_bytes.float().mean()) * 100:.1f}% same bytes)")

    # T4: full Engram layer
    wkv = find_tensor(ckpt, [f"layers.{layer}.engram.wkv.weight"])
    wkv_s = find_tensor(ckpt, [f"layers.{layer}.engram.wkv.scale", f"layers.{layer}.engram.wkv.weight_scale_inv"])
    qw = find_tensor(ckpt, [f"layers.{layer}.engram.q_weight"])
    kw = find_tensor(ckpt, [f"layers.{layer}.engram.k_weight"])
    if not args.skip_layer and all(x is not None for x in (wkv, wkv_s, qw, kw)):
        torch.set_default_dtype(torch.bfloat16)
        small = dataclasses.replace(layout, num_embeddings=tuple(len(uniq) + 1 for _ in layout.num_embeddings))
        eng = M.Engram(margs, layer, small)
        with torch.no_grad():
            eng.embed.weight.data.copy_(ref_embed.weight.data)
            eng.embed.scale.data.copy_(ref_embed.scale.data)
            eng.wkv.weight.data.copy_(st_tensor(wkv).view(torch.float8_e4m3fn))
            eng.wkv.scale.data.copy_(st_tensor(wkv_s).view(torch.float8_e8m0fnu))
            eng.wkv.weight.scale = eng.wkv.scale
            eng.q_weight.data.copy_(st_tensor(qw).to(torch.bfloat16))
            eng.k_weight.data.copy_(st_tensor(kw).to(torch.bfloat16))
        nt = min(T, args.layer_tokens)
        gx = torch.Generator().manual_seed(args.seed + 2)
        x = (torch.randn((1, nt, cfg.hc_mult, cfg.dim), generator=gx) * 0.5).to(torch.bfloat16)
        tmask = mask.reshape(-1)[:nt].view(1, nt)
        ref_out = eng(x, remap[:nt].view(1, nt, cols), tmask)  # official forward
        # DeepInfer side: fetched fp8 rows -> fp32-accumulated GEMM with the fp8 wkv -> gate
        Wd = (eng.wkv.weight.float().view(eng.wkv.weight.shape[0] // 32, 32, -1, 32)
              * torch.exp2(eng.wkv.scale.view(torch.uint8).float() - 127)[:, None, :, None])
        Wd = Wd.reshape(eng.wkv.weight.shape)
        kv = (di_rows[:nt].float() @ Wd.t()).to(torch.bfloat16).view(nt, cfg.hc_mult + 1, cfg.dim)
        if use_di:
            import tile_kernels

            wf = tile_kernels.engram.fused_weight(eng.q_weight.data.cuda(), eng.k_weight.data.cuda())
            di_out = tile_kernels.engram.engram_gate_fwd(x[0].cuda(), kv.cuda(), wf, cfg.norm_eps, 1e-6,
                                                         save_for_backward=False, image_token_mask=(~tmask[0]).cuda(),
                                                         inplace=False)[0].cpu()
        else:
            wf = se.fused_weight(eng.q_weight.data, eng.k_weight.data)
            di_out = se.engram_gate_fwd(x[0], kv, wf, cfg.norm_eps, 1e-6, save_for_backward=False,
                                        image_token_mask=~tmask[0])[0]
        d = (di_out.float() - ref_out[0].float()).abs()
        delta = (ref_out[0].float() - x[0].float()).abs()  # size of the engram contribution itself
        rel = float(d.max() / delta.max().clamp_min(1e-12))
        same_bits = float((di_out.view(torch.int16) == ref_out[0].view(torch.int16)).float().mean())
        check("T4 engram layer", rel < args.layer_tol,
              f"{nt} tokens: max|diff|={float(d.max()):.3e} vs max|contribution|={float(delta.max()):.3e} "
              f"(rel {rel:.2e}, tol {args.layer_tol}); bitwise-equal {same_bits * 100:.3f}%")
    else:
        log("[SKIP] T4 engram layer (no wkv/q_weight/k_weight in the index, or --skip-layer)")
    buf.destroy()


def graph_check(cfg, table, cols, device, seed, prestage: bool, Tg: int = 16, reps: int = 5):
    from deep_ep._engram import EngramBuffer

    buf = EngramBuffer(None)
    buf.set_config([table.rows], cfg.head_dim, 64, cols, torch.float8_e4m3fn, cfg.head_dim // 128)
    buf.attach_tables([table])
    g = torch.Generator().manual_seed(seed)

    def ids():
        x = torch.randint(-1, table.rows, (1, Tg, cols), generator=g, dtype=torch.int64)
        return x.to(torch.int32).contiguous().to(device)

    static_idx = ids()
    out = torch.empty((Tg, cols * cfg.head_dim), dtype=torch.bfloat16, device=device)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        (h,) = buf.fetch(static_idx)  # eager warm-up allocates the static buffers
        h()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        (h,) = buf.fetch(static_idx)
        d, f = h()
        out.copy_(se.dequant_rows(d, f, cfg.head_dim))
    ok, dt = True, 0.0
    for _ in range(reps):
        new = ids()
        static_idx.copy_(new)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if prestage:
            buf.prestage(static_idx)
        graph.replay()
        torch.cuda.synchronize()
        dt += time.perf_counter() - t0
        (he,) = buf.fetch(new)
        de, fe = he()
        ref = se.dequant_rows(de, fe, cfg.head_dim)
        torch.cuda.synchronize()
        ok &= torch.equal(out.view(torch.int16), ref.view(torch.int16))
    ok &= buf.fetcher.graph_mode == ("prestage" if prestage else "async")
    buf.fetcher.owns_tables = False  # the table is shared with the other checks
    buf.destroy()
    return ok, dt / reps


# ------------------------------------------------------------------------------------------------ benchmark
def evict(table: se.EngramHostTable) -> None:
    if table.source is None:
        return
    for loc in table.source:
        if loc is not None:
            se._madvise(se._mmap_file(loc.path), "dontneed", loc.offset, loc.nbytes)
            se.drop_page_cache(loc.path, loc.offset, loc.nbytes)


def run_bench(args, cfg: Cfg, ckpt: str, device: torch.device, tokenizer=None) -> None:
    from deep_ep._engram import EngramBuffer

    layers = cfg.layer_ids if args.bench_all_layers else [args.layer]
    sizes = [int(x) for x in args.bench_sizes.split(",")]
    cols = (cfg.max_ngram - 1) * cfg.n_heads
    tmax = max(sizes)
    tables = [se.open_table(ckpt, l, device=device, max_lookups=tmax * cols) for l in layers]
    buf = EngramBuffer(None)
    buf.set_config([t.rows for t in tables], cfg.head_dim, tmax, cols, torch.float8_e4m3fn, cfg.head_dim // 128)
    buf.attach_tables(tables)
    g = torch.Generator().manual_seed(1234)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    log(f"\nbenchmark: layers={layers} mode={tables[0].kind} threads={buf.fetcher.threads} device={device} "
        f"(row = {cfg.head_dim}+{cfg.head_dim // 32} B, {cols} rows/token/layer)")
    log(f"{'tokens':>7} {'case':>6} {'ms':>9} {'tok/s':>10} {'Mrows/s':>8} {'MiB/s':>8}")
    report = []
    for T in sizes:
        ids = torch.stack([torch.randint(0, t.rows, (T, cols), generator=g, dtype=torch.int64) for t in tables])
        ids = ids.to(torch.int32).contiguous().to(device)
        for case in ("cold", "warm"):
            if case == "cold":
                for t in tables:
                    evict(t)
            reps = 1 if case == "cold" else args.bench_reps
            if case == "warm":
                for hk in buf.fetch(ids):  # touch once
                    hk()
                sync()
            t0 = time.perf_counter()
            for _ in range(reps):
                for hk in buf.fetch(ids):
                    hk()
            sync()
            dt = (time.perf_counter() - t0) / reps
            nrows = len(tables) * T * cols
            row = dict(tokens=T, case=case, ms=dt * 1e3, tok_s=T / dt, mrows_s=nrows / dt / 1e6,
                       mib_s=nrows * (cfg.head_dim + cfg.head_dim // 32) / dt / 2**20)
            report.append(row)
            log(f"{T:>7} {case:>6} {row['ms']:>9.3f} {row['tok_s']:>10.0f} {row['mrows_s']:>8.2f} {row['mib_s']:>8.0f}")
    RESULTS["bench"] = {"mode": tables[0].kind, "layers": layers, "rows": report, "stats": dict(buf.fetcher.stats)}
    buf.destroy()


# ------------------------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", help="HF checkpoint dir (index + config.json + tokenizer.json + the engram shard)")
    ap.add_argument("--synthetic", metavar="DIR", help="generate a tiny synthetic checkpoint in DIR and test on it")
    ap.add_argument("--layer", type=int, default=1, help="engram layer id (as named in the checkpoint)")
    ap.add_argument("--tokenizer", help="tokenizer.json (default: <ckpt>/tokenizer.json)")
    ap.add_argument("--ref-src", default=None)
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda")
    ap.add_argument("--impl", choices=("auto", "deepinfer", "torch"), default="auto",
                    help="DeepInfer-side hash/gate ops: tile_kernels+deepinfer (CUDA) or the torch ports")
    ap.add_argument("--seqs", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=200)
    ap.add_argument("--decode", type=int, default=8)
    ap.add_argument("--random-rows", type=int, default=512, help="extra tokens with uniform random row ids for T3")
    ap.add_argument("--layer-tokens", type=int, default=64)
    ap.add_argument("--layer-tol", type=float, default=2e-2, help="T4: max|diff| / max|engram contribution|")
    ap.add_argument("--skip-layer", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--bench-sizes", default="1,8,64,512,2048,8192")
    ap.add_argument("--bench-reps", type=int, default=5)
    ap.add_argument("--bench-all-layers", action="store_true", help="fetch every engram layer per step (as DeepInfer)")
    ap.add_argument("--json", help="write results here")
    args = ap.parse_args()

    if args.synthetic:
        ckpt = make_synthetic(args.synthetic, args.seed)
    elif args.ckpt:
        ckpt = args.ckpt
    else:
        ap.error("--ckpt or --synthetic required")
    cfg = cfg_from_hf(ckpt)
    if args.layer not in cfg.layer_ids:
        ap.error(f"--layer {args.layer} not in engram_layer_ids {cfg.layer_ids}")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))
    torch.set_default_dtype(torch.bfloat16)

    rc = 0
    if not args.skip_tests:
        M, E = import_reference(args.ref_src or default_ref_src())
        if args.synthetic:
            meta = json.load(open(os.path.join(ckpt, "config.json")))["sm120_synthetic"]
            tokenizer = FakeTokenizer(cfg.tok_vocab, meta["n_keys"])
        else:
            import ref_engram

            args.tokenizer = args.tokenizer or os.path.join(ckpt, "tokenizer.json")
            tokenizer = ref_engram.tokenizer_for_engram(args.tokenizer)
        with torch.inference_mode():
            run_tests(args, cfg, ckpt, M, E, tokenizer, device)
        rc = 0 if all(v["ok"] for k, v in RESULTS.items() if k.startswith("T")) else 1
    if args.bench:
        with torch.inference_mode():
            run_bench(args, cfg, ckpt, device)
    if args.json:
        json.dump(RESULTS, open(args.json, "w"), indent=1)
    log(f"\n{'ALL PASS' if rc == 0 else 'FAILURES'}: {sorted(k for k, v in RESULTS.items() if k.startswith('T'))}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
