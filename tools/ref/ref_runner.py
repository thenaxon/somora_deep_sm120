#!/usr/bin/env python3
"""Ground-truth run of DeepSeek's official minimal inference (ds41-inference/inference/model.py) on a
truncated DeepSeek-V4.1-Flash checkpoint (3 layers for the base test model, N layers for YOCO), single GPU, world_size = 1,
no torchrun.

    REF_KERNELS=tilelang python3 ref_runner.py \
        --ckpt /work/ref-mp1 --hf-config /work/ckpt3/config.json --out /work/ref_out.pt

What it does
  1. registers the kernel dispatcher (ref_kernels.py) as module `kernel`, imports model.py unchanged;
  2. builds ModelArgs from the inference/config.json template overlaid with values derived from the
     truncated HF config.json (n_layers, compress_ratios, sources, n_routed_experts, ...), vision/DSpark off, engram via ref_engram.py (mmap) when the config has engram layers;
  3. builds Transformer on cuda (as generate.py:128-131) and loads model0-mp1.safetensors strictly;
  4. encodes a fixed ~300-token English passage + question with encoding.py (chat mode) and the raw
     `tokenizers` tokenizer (no extra BOS; --prompt-tokens N appends deterministic synthetic logbook entries
     until the prompt has >= N tokens), prefills it in one forward with full-sequence logits,
     then runs 16 greedy decode steps exactly like generate.py:66-80 (one token per forward at
     start_pos = previous length);
  5. saves logits / top-k / ids / per-layer hidden states to --out (torch.save) and a JSON summary.

world_size: Transformer.__init__ reads dist.is_initialized() (model.py:1192-1193); we never call
init_process_group, so world_size = 1 / rank = 0 and every dist.* call in model.py is skipped (all are
guarded by `if world_size > 1`).
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import logging
import os
import platform
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DEFAULT_REF_SRC = os.path.normpath(os.path.join(HERE, "..", "..", "third_party", "ds41-inference", "inference"))

log = logging.getLogger("ref_runner")

PASSAGE = (
    "The lighthouse at the end of the northern breakwater was built in 1872, at a time when the harbor "
    "handled more timber than any other port on the coast. Its first keeper, a retired ship's carpenter "
    "named Elias Brandt, kept a meticulous logbook in which he recorded the weather every four hours, the "
    "names of passing vessels, and the amount of oil burned each night. Historians have used his notes to "
    "reconstruct the storms of the late nineteenth century with surprising precision. In the winter of "
    "1881, for example, Brandt described a gale that lasted six days and pushed waves over the top of the "
    "breakwater, flooding the keeper's cottage and extinguishing the lamp twice in a single night. Each "
    "time, he climbed the iron staircase in the dark, relit the wick, and polished the salt from the lens "
    "before returning to bail water from his kitchen. The light was converted to electricity in 1931, and "
    "the last resident keeper left in 1967, when an automatic timer and a backup battery made the position "
    "unnecessary. Today the cottage houses a small museum. Visitors can read photocopied pages of the "
    "logbook, examine a model of the original rotating lens, and climb the same staircase to the gallery, "
    "where on clear days the mountains across the bay are visible. Local schools bring students every "
    "spring, and many of them are surprised to learn that a single person once carried the responsibility "
    "of keeping dozens of ships away from the rocks, night after night, with nothing but oil, glass, and "
    "patience."
)
QUESTION = "In one sentence, what is the main idea of the passage above?"

# vocabulary of the synthetic logbook entries that lengthen the prompt (--prompt-tokens): varied, non-repeating
# text so that the indexer top-k and candidate blocks see distinct keys rather than copies of one passage
LOG_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December")
LOG_WINDS = ("north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west")
LOG_SKY = ("clear", "overcast", "drizzle", "heavy rain", "sleet", "fog", "snow showers", "a low haze", "hail")
LOG_SHIPS = ("schooner", "steam tug", "timber barque", "fishing smack", "mail packet", "collier", "brig", "ketch",
             "revenue cutter", "pilot boat")
LOG_NAMES = ("Margaret", "Fortitude", "Northern Star", "Agnes Reid", "Halcyon", "Dolphin", "Prudence", "Osprey",
             "Clara Bell", "Endeavour", "Swift", "Juniper", "Mary Ann", "Kestrel", "Union")
LOG_TASKS = ("trimmed the wick", "cleaned the lens", "painted the railing", "repaired a cracked pane",
             "carried oil up from the store", "logged a distress flare", "rang the fog bell", "mended the boat",
             "checked the clockwork", "scraped salt from the gallery")

# keys that legitimately differ between the template inference/config.json and the truncated model
LAYER_KEYS = {
    "n_layers", "n_mtp_layers", "compress_ratios", "kv_source_layers", "index_source_layers",
    "candidate_source_layer", "engram_layer_ids", "engram_num_embeddings", "dspark_block_size",
    "dspark_target_layer_ids", "vision_n_layers",
    # make_truncated_ckpt.py --num-experts / --candidate-topk-blocks
    "n_routed_experts", "candidate_topk_blocks",
}
DROP_OK = re.compile(r"^(vision\.|aligner\.|image_start$|image_end$|image_newline$|mtp\.)|\.ffn\.gate\.bias_vl$")


# ------------------------------------------------------------------------------------------------
def hf_to_model_args(hf: dict) -> dict:
    """HF config.json (text_config) -> ds41 ModelArgs field names (model.py:44-136)."""
    t, q = hf["text_config"], hf["quantization_config"]
    # invariants model.py hard-codes (module globals model.py:24-30, MoE assert model.py:888)
    assert q["quant_method"] == "fp8" and q["weight_block_size"] == [32, 32] and q["scale_fmt"] == "ue8m0", q
    assert t["n_shared_experts"] == 1 and t["num_key_value_heads"] == 1 and t["hidden_act"] == "silu"
    assert t.get("topk_method", "noaux_tc") == "noaux_tc"
    rs = t.get("rope_scaling") or {}
    if rs:
        assert rs.get("rope_type", rs.get("type")) == "yarn", rs
    nextn = t.get("num_nextn_predict_layers", 0)
    if nextn:  # the checkpoint may carry MTP layers for DeepInfer; the reference never runs DSpark here
        log.info(f"HF config has {nextn} MTP layers: DSpark stays off in the reference (ref_convert drops mtp.*)")
    n_layers = t["num_hidden_layers"]
    cand = t.get("candidate_source_layer_id")
    return dict(
        vocab_size=t["vocab_size"],
        dim=t["hidden_size"],
        moe_inter_dim=t["moe_intermediate_size"],
        n_layers=n_layers,
        n_mtp_layers=0,
        n_heads=t["num_attention_heads"],
        n_routed_experts=t["n_routed_experts"],
        n_shared_experts=t["n_shared_experts"],
        n_activated_experts=t["num_experts_per_tok"],
        score_func=t["scoring_func"],
        norm_topk_prob=t["norm_topk_prob"],
        route_scale=t["routed_scaling_factor"],
        swiglu_limit=t["swiglu_limit"],
        q_lora_rank=t["q_lora_rank"],
        head_dim=t["head_dim"],
        rope_head_dim=t["qk_rope_head_dim"],
        norm_eps=t["rms_norm_eps"],
        o_groups=t["o_groups"],
        o_lora_rank=t["o_lora_rank"],
        window_size=t["sliding_window"],
        compress_ratios=list(t["compress_ratios"][:n_layers]),
        kv_source_layers=list(t["kv_source_layer_ids"]),
        index_source_layers=list(t["index_source_layer_ids"]),
        compress_rope_theta=t["compress_rope_theta"],
        original_seq_len=rs.get("original_max_position_embeddings", 0) if rs else 0,
        rope_theta=t["rope_theta"],
        rope_factor=rs.get("factor", 1) if rs else 1,
        beta_fast=rs.get("beta_fast", 32) if rs else 32,
        beta_slow=rs.get("beta_slow", 1) if rs else 1,
        index_n_heads=t["index_n_heads"],
        index_head_dim=t["index_head_dim"],
        index_topk=t["index_topk"],
        candidate_source_layer=-1 if cand is None else cand,
        candidate_topk_blocks=t["candidate_topk_blocks"],
        candidate_block_size=t["candidate_block_size"],
        hc_mult=t["hc_mult"],
        hc_sinkhorn_iters=t["hc_sinkhorn_iters"],
        hc_eps=t["hc_eps"],
        engram_layer_ids=list(t.get("engram_layer_ids") or []),
        engram_num_embeddings=list(t.get("engram_num_embeddings") or []),
        engram_max_ngram_size=t["engram_max_ngram_size"],
        engram_vocab_size=t["engram_vocab_size"],
        engram_n_heads=t["engram_n_heads"],
        engram_head_dim=t["engram_head_dim"],
        engram_pad_id=t["engram_pad_token_id"],
        engram_compressed_vocab_size=t["engram_compressed_vocab_size"],
        dtype="fp8",
        expert_dtype=q.get("expert_dtype"),
        image_token_id=hf.get("image_token_id", 129264),
        dspark_block_size=0,
        dspark_noise_token_id=t.get("dspark_noise_token_id", 0),
        dspark_target_layer_ids=[],
        dspark_markov_rank=t.get("dspark_markov_rank", 256),
        dspark_n_routed_experts=t.get("dspark_n_routed_experts", 0),
        dspark_n_activated_experts=t.get("dspark_num_experts_per_tok", 0),
        vision_n_layers=0,  # vision off: no ViT/aligner/image_* params and no Gate.bias_vl (model.py:809,1219)
    )


def build_model_args(M, hf_config: str, ref_src: str, max_seq_len: int):
    hf = json.load(open(hf_config))
    derived = hf_to_model_args(hf)
    template_path = os.path.join(ref_src, "config.json")
    base = json.load(open(template_path)) if os.path.exists(template_path) else {}
    mismatches = {}
    for k, v in derived.items():
        if k in base and k not in LAYER_KEYS and base[k] != v:
            mismatches[k] = (base[k], v)
    for k, (b, v) in mismatches.items():
        log.warning(f"config mismatch {k}: inference/config.json={b!r} HF-derived={v!r} (using HF-derived)")
    cfg = dict(base)
    cfg.update(derived)
    cfg.update(max_batch_size=1, max_seq_len=max_seq_len, temperature=0.0)
    fields = {f.name for f in dataclasses.fields(M.ModelArgs)}
    unknown = sorted(set(cfg) - fields)
    if unknown:
        log.warning(f"ignoring keys unknown to ModelArgs: {unknown}")
    args = M.ModelArgs(**{k: v for k, v in cfg.items() if k in fields})
    # sanity: the truncated topology
    assert len(args.compress_ratios) == args.n_layers
    assert not args.vision_enabled and not args.dspark_block_size
    return args, hf, mismatches


# ------------------------------------------------------------------------------------------------
def load_encoding(paths: list[str]):
    for p in paths:
        if p and os.path.isfile(p):
            spec = importlib.util.spec_from_file_location("ds41_encoding", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod, p
    return None, None


def log_entries(seed: int = 1881):
    """Endless deterministic logbook lines (--prompt-tokens), one distinct sentence each."""
    import random

    rng = random.Random(seed)
    year, day = 1872, 0
    while True:
        day += rng.randint(1, 9)
        if day > 28:
            day -= 28
            year += rng.random() < 0.08
        yield (f"{day} {rng.choice(LOG_MONTHS)} {year}: wind {rng.choice(LOG_WINDS)}, force {rng.randint(1, 11)}, "
               f"{rng.choice(LOG_SKY)}; the {rng.choice(LOG_SHIPS)} {rng.choice(LOG_NAMES)} passed at "
               f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}; {rng.randint(2, 19)} pints of oil burned; "
               f"the keeper {rng.choice(LOG_TASKS)}.")


def build_prompt(args, ref_src: str):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(args.tokenizer)
    if args.prompt_ids_json:
        d = json.load(open(args.prompt_ids_json))
        ids = [int(x) for x in d["input_ids"]]
        return tok, ids, d.get("prompt", tok.decode(ids, skip_special_tokens=False)), f"ids from {args.prompt_ids_json}"
    enc, enc_path = load_encoding([
        args.encoding,
        os.path.join(os.path.dirname(os.path.abspath(args.hf_config)), "encoding", "encoding.py"),
        os.path.join(ref_src, "..", "encoding", "encoding.py"),
    ])
    mode = "raw"
    if enc is None:
        log.warning("encoding.py not found; using raw BOS + text")

    def render(text: str) -> str:
        nonlocal mode, enc
        if enc is not None:
            try:
                out = enc.encode_messages([{"role": "user", "content": text}], thinking_mode="chat")
                mode = f"encoding.encode_messages(chat) from {enc_path}"
                return out
            except Exception as e:
                log.warning(f"encode_messages failed ({e}); using raw BOS + text")
                enc = None
        mode = "raw"
        return "<｜begin▁of▁sentence｜>" + text

    prompt = render(PASSAGE + "\n\n" + QUESTION)
    ids = tok.encode(prompt, add_special_tokens=False).ids
    if args.prompt_tokens and len(ids) < args.prompt_tokens:
        # passage, then logbook excerpts, then the question; grown in batches and re-tokenised as a whole
        lines, gen = [], log_entries()
        while len(ids) < args.prompt_tokens:
            lines += [next(gen) for _ in range(max(1, (args.prompt_tokens - len(ids)) // 60))]
            text = PASSAGE + "\n\nExcerpts from the logbook:\n" + "\n".join(lines) + "\n\n" + QUESTION
            prompt = render(text)
            ids = tok.encode(prompt, add_special_tokens=False).ids
        mode += f" + {len(lines)} synthetic logbook lines (--prompt-tokens {args.prompt_tokens})"
    bos = tok.token_to_id("<｜begin▁of▁sentence｜>")
    if bos is not None and ids[0] != bos:
        log.warning(f"prompt does not start with BOS ({bos}); first ids {ids[:4]}")
    return tok, ids, prompt, mode


# ------------------------------------------------------------------------------------------------
def load_weights(model, path: str):
    """Strict load of model0-mp1.safetensors into the cuda-resident model, tensor by tensor."""
    import torch

    from st_io import ShardReader

    sd = model.state_dict()
    loaded, skipped = set(), []
    float_types = {torch.float32, torch.bfloat16, torch.float16}
    t0 = time.time()
    with ShardReader(path) as r:
        names = list(r.keys())
        for i, name in enumerate(names):
            if name not in sd:
                if DROP_OK.search(name):
                    skipped.append(name)
                    continue
                raise RuntimeError(f"unexpected tensor {name} (not a parameter of the reference model)")
            dst = sd[name]
            src = r.read_torch(name)
            if dst.dtype == torch.float4_e2m1fn_x2:
                if src.dtype not in (torch.int8, torch.uint8):
                    raise RuntimeError(f"{name}: fp4 param needs I8/U8 packed bytes, file has {src.dtype}")
                if tuple(src.shape) != tuple(dst.shape):
                    raise RuntimeError(f"{name}: shape {tuple(src.shape)} != {tuple(dst.shape)}")
                dst.view(torch.uint8).copy_(src.view(torch.uint8))
            elif src.dtype == dst.dtype:
                if tuple(src.shape) != tuple(dst.shape):
                    raise RuntimeError(f"{name}: shape {tuple(src.shape)} != {tuple(dst.shape)}")
                if dst.element_size() == 1:
                    dst.view(torch.uint8).copy_(src.view(torch.uint8))
                else:
                    dst.copy_(src)
            elif src.dtype in float_types and dst.dtype in float_types:
                if tuple(src.shape) != tuple(dst.shape):
                    raise RuntimeError(f"{name}: shape {tuple(src.shape)} != {tuple(dst.shape)}")
                dst.copy_(src.to(dst.dtype))  # e.g. bf16 checkpoint -> fp32 head / compressor (model.py:446-448,1008)
            else:
                raise RuntimeError(f"{name}: cannot load {src.dtype} into {dst.dtype}")
            loaded.add(name)
            if (i + 1) % 1000 == 0:
                log.info(f"  loaded {i + 1}/{len(names)} ({time.time() - t0:.0f}s)")
    missing = sorted(set(sd) - loaded)
    if missing:
        raise RuntimeError(f"{len(missing)} model tensors missing from {path}: {missing[:10]}")
    torch.cuda.synchronize()
    log.info(f"loaded {len(loaded)} tensors in {time.time() - t0:.0f}s; skipped {len(skipped)} (vision/mtp/bias_vl)")
    return {"loaded": len(loaded), "skipped": skipped}


def parse_capture_layers(v: str | None, n_layers: int) -> list[int]:
    """CAPTURE_LAYER / --capture-layer: '0' (default), '0,5,7', 'all', or ''/'none'/'off'/'-1' for no taps."""
    v = (v if v is not None else "0").strip().lower()
    if v in ("", "none", "off", "no", "-1"):
        return []
    if v == "all":
        return list(range(n_layers))
    out = sorted({int(x) for x in v.split(",") if x.strip()})
    bad = [i for i in out if not 0 <= i < n_layers]
    if bad:
        raise SystemExit(f"capture layer(s) {bad} out of range 0..{n_layers - 1}")
    return out


# ------------------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="/work/ref-mp1", help="dir with model0-mp1.safetensors (or the file)")
    ap.add_argument("--hf-config", default="/work/ckpt3/config.json", help="truncated HF config.json")
    ap.add_argument("--ref-src", default=os.environ.get("REF_SRC", DEFAULT_REF_SRC), help="ds41-inference/inference dir")
    ap.add_argument("--tokenizer", default=None, help="tokenizer.json (default: next to --hf-config)")
    ap.add_argument("--encoding", default=None, help="encoding.py (default: <hf dir>/encoding/encoding.py)")
    ap.add_argument("--out", default="/work/ref_out.pt")
    ap.add_argument("--prompt-ids-json", default=None, help="use these input ids instead of building the prompt")
    ap.add_argument("--prompt-tokens", type=int, default=0,
                    help="lengthen the built prompt to >= N tokens with synthetic logbook lines (YOCO: exercises "
                         "the indexer top-k and candidate blocks); ignored with --prompt-ids-json")
    ap.add_argument("--save-prompt-json", default="/work/prompt_ids.json", help="write the ids used ('' = skip)")
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--decode-steps", type=int, default=16)
    ap.add_argument("--compact", action="store_true", help="do not store full prefill logits (only last 64 + top-20)")
    ap.add_argument("--no-hidden", action="store_true", help="do not store per-layer hidden states")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--engram-ckpt", default=None,
                    help="HF checkpoint dir holding layers.<id>.engram.embed.* for configs with engram layers (served "
                         "from an mmap by ref_engram.py). Default: the --hf-config dir (make_truncated_ckpt "
                         "--keep-engram symlinks the original shard there)")
    ap.add_argument("--engram-mode", choices=("host", "uva"), default="host", help="ref_engram.attach mode")
    ap.add_argument("--capture-layer", default=os.environ.get("CAPTURE_LAYER", "0"),
                    help="fine prefill taps of these blocks into hidden['prefill']['L<i>/...'] (ref_capture.py): "
                         "'0', '0,5,7', 'all'; -1 = off. Default: env CAPTURE_LAYER or 0")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if args.tokenizer is None:
        args.tokenizer = os.path.join(os.path.dirname(os.path.abspath(args.hf_config)), "tokenizer.json")
    ref_src = os.path.abspath(args.ref_src)
    ckpt_file = args.ckpt if args.ckpt.endswith(".safetensors") else os.path.join(args.ckpt, "model0-mp1.safetensors")
    if not os.path.isfile(ckpt_file):
        raise SystemExit(f"{ckpt_file} not found; run: python3 {HERE}/ref_convert.py --hf-ckpt-path /work/ckpt3 --save-path {args.ckpt}")
    if not os.path.isfile(os.path.join(ref_src, "model.py")):
        raise SystemExit(f"--ref-src {ref_src} does not contain model.py")

    import torch

    torch.backends.cuda.matmul.allow_tf32 = False  # torch fallbacks emulate fp32 accumulation
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    import ref_kernels

    dispatch = ref_kernels.install(ref_src)
    import model as M  # ds41-inference/inference/model.py, unmodified

    timings = {}
    t0 = time.time()
    margs, hf, mismatches = build_model_args(M, args.hf_config, ref_src, args.max_seq_len)
    log.info(f"ModelArgs: {margs}")
    engram_on = bool(margs.engram_layer_ids)
    if engram_on:
        import ref_engram

        ref_engram.install(M)  # table class -> mmap-backed MmapEngramEmbedding, before M.Transformer
        if args.engram_ckpt is None:
            args.engram_ckpt = os.path.dirname(os.path.abspath(args.hf_config))
        log.info(f"engram layers {list(margs.engram_layer_ids)}: tables from {args.engram_ckpt} ({args.engram_mode})")
    tok, ids, prompt, prompt_mode = build_prompt(args, ref_src)
    L = len(ids)
    log.info(f"prompt: {L} tokens ({prompt_mode})")
    if L + args.decode_steps + 1 > margs.max_seq_len:
        raise SystemExit(f"prompt {L} + {args.decode_steps} steps exceeds --max-seq-len {margs.max_seq_len}")
    if args.save_prompt_json:
        with open(args.save_prompt_json, "w") as f:
            json.dump({"prompt": prompt, "prompt_mode": prompt_mode, "input_ids": ids}, f, ensure_ascii=False)

    # generate.py:120-131
    torch.cuda.set_device(args.device)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(33377335)
    tok_engram = ref_engram.tokenizer_for_engram(args.tokenizer) if engram_on else None  # NgramHashState token map
    with torch.device("cuda"):
        model = M.Transformer(margs, tok_engram)
    assert (model.engram_hash is not None) == engram_on and model.vision is None and len(model.mtp) == 0
    torch.cuda.synchronize()
    log.info(f"model built: {torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated")
    timings["build_s"] = time.time() - t0
    t1 = time.time()
    load_stats = load_weights(model, ckpt_file)
    if engram_on:
        tables = ref_engram.attach(model, args.engram_ckpt, mode=args.engram_mode, device=args.device)
        log.info(f"engram tables attached: {[(t.name, t.rows) for t in tables]}")
    timings["load_s"] = time.time() - t1
    torch.set_default_device("cuda")  # generate.py:133; get_window_topk_idxs etc. allocate on the default device

    hidden = {"prefill": {}, "decode": []}
    cur = {"phase": "prefill", "store": hidden["prefill"]}
    hooks = []
    if not args.no_hidden:
        for i, layer in enumerate(model.layers):
            def hook(mod, inp, out, i=i):
                cur["store"][f"layer{i}"] = out[0].detach().to("cpu")  # [b, s, hc_mult, dim] bf16, block output
            hooks.append(layer.register_forward_hook(hook))

            def pre_hook(mod, inp, i=i):
                if i == 0:
                    cur["store"]["embed_hc"] = inp[0].detach().to("cpu")  # input of block 0
            hooks.append(layer.register_forward_pre_hook(pre_hook))

        def norm_hook(mod, inp, out):
            cur["store"]["final_norm"] = out.detach().to("cpu")  # head input [b, s, dim]
        hooks.append(model.norm.register_forward_hook(norm_hook))

    fine_taps = []
    capture_layers = [] if args.no_hidden else parse_capture_layers(args.capture_layer, len(model.layers))
    if capture_layers:
        import ref_capture

        model._capture_args = margs
        for li in capture_layers:
            fine_taps.append(ref_capture.install(model, M, li, hidden["prefill"], L))
            fine_taps[-1].put_static()
        log.info(f"fine taps on layers {capture_layers}")

    res = {}
    with torch.inference_mode():
        x = torch.tensor([ids], dtype=torch.long, device="cuda")
        head = model.head
        orig_forward = head.forward
        head.forward = lambda h, full_logits=False: orig_forward(h, full_logits=True)  # all positions
        t2 = time.time()
        if fine_taps:
            ref_capture.STATE["armed"] = True
        try:
            _, logits_pre, _ = model.forward(x, 0)
        finally:
            del head.forward
            if fine_taps:
                ref_capture.STATE["armed"] = False
                for taps in reversed(fine_taps):  # each layer's wrappers wrap the previous layer's globals
                    taps.remove()
        torch.cuda.synchronize()
        timings["prefill_s"] = time.time() - t2
        logits_pre = logits_pre[0].float()  # [L, V]
        assert logits_pre.shape == (L, margs.vocab_size), logits_pre.shape
        top20 = logits_pre.topk(20, dim=-1)
        res["prefill_top20_values"] = top20.values.cpu()
        res["prefill_top20_ids"] = top20.indices.cpu()
        res["prefill_argmax"] = logits_pre.argmax(-1).cpu()
        res["prefill_logits_last64"] = logits_pre[-64:].cpu().clone()
        res["prefill_logits"] = None if args.compact else logits_pre.cpu()
        res["prefill_finite"] = bool(torch.isfinite(logits_pre).all())
        gen = [int(logits_pre[-1].argmax())]
        del logits_pre

        dec_vals, dec_ids, dec_full, positions = [], [], [], []
        t3 = time.time()
        for s in range(args.decode_steps):
            pos = L + s
            cur["store"] = {}
            _, logits, _ = model.forward(torch.tensor([[gen[-1]]], dtype=torch.long, device="cuda"), pos)
            logits = logits[0].float()  # [V]
            if not args.no_hidden:
                hidden["decode"].append(cur["store"])
            top = logits.topk(50)
            dec_vals.append(top.values.cpu())
            dec_ids.append(top.indices.cpu())
            if s < 4:
                dec_full.append(logits.cpu())
            positions.append(pos)
            gen.append(int(logits.argmax()))
        torch.cuda.synchronize()
        timings["decode_s"] = time.time() - t3
    for h in hooks:
        h.remove()

    res.update(
        format="sm120-ref-v1",
        input_ids=torch.tensor(ids, dtype=torch.long, device="cpu"),
        prompt_text=prompt,
        prompt_mode=prompt_mode,
        # decoded_ids[0] = argmax of the last prefill position; decoded_ids[s+1] = argmax of decode step s,
        # whose input token is decoded_ids[s] at position len(input_ids)+s.
        decoded_ids=torch.tensor(gen, dtype=torch.long, device="cpu"),
        decode_positions=torch.tensor(positions, dtype=torch.long, device="cpu"),
        decode_top50_values=torch.stack(dec_vals) if dec_vals else torch.empty(0, 50, device="cpu"),
        decode_top50_ids=torch.stack(dec_ids) if dec_ids else torch.empty(0, 50, dtype=torch.long, device="cpu"),
        decode_logits_first4=torch.stack(dec_full) if dec_full else torch.empty(0, margs.vocab_size, device="cpu"),
        hidden=None if args.no_hidden else hidden,
    )
    try:
        import tilelang

        tl_ver = getattr(tilelang, "__version__", "?")
    except Exception:
        tl_ver = None
    trunc_path = os.path.join(os.path.dirname(os.path.abspath(args.hf_config)), "truncation.json")
    meta = dict(
        model_args=dataclasses.asdict(margs),
        truncation=json.load(open(trunc_path)) if os.path.isfile(trunc_path) else None,
        capture_layers=capture_layers,
        engram=dict(layer_ids=list(margs.engram_layer_ids), ckpt=args.engram_ckpt, mode=args.engram_mode) if engram_on else None,
        config_template_mismatches={k: list(v) for k, v in mismatches.items()},
        hf_config=os.path.abspath(args.hf_config),
        ckpt=os.path.abspath(ckpt_file),
        ref_src=ref_src,
        kernels=dispatch.report(),
        load=dict(loaded=load_stats["loaded"], skipped=len(load_stats["skipped"])),
        timings=timings,
        peak_mem_gib=torch.cuda.max_memory_allocated() / 2**30,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(args.device),
        capability=torch.cuda.get_device_capability(args.device),
        tilelang=tl_ver,
        python=platform.python_version(),
        greedy="argmax over fp32 logits (first max on ties); temperature 0",
    )
    res["meta"] = meta
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    torch.save(res, args.out)

    summary = dict(
        out=os.path.abspath(args.out),
        n_prompt=L,
        decoded_ids=gen,
        decoded_text=tok.decode(gen, skip_special_tokens=False),
        last_prefill_top5=list(zip(res["prefill_top20_ids"][-1, :5].tolist(), [round(v, 4) for v in res["prefill_top20_values"][-1, :5].tolist()])),
        prefill_finite=res["prefill_finite"],
        kernels=meta["kernels"],
        timings=timings,
        peak_mem_gib=meta["peak_mem_gib"],
    )
    with open(os.path.splitext(args.out)[0] + ".summary.json", "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False, default=str)
    log.info(f"decoded: {gen}")
    log.info(f"text: {summary['decoded_text']!r}")
    log.info(f"kernel backends: {meta['kernels']['backend_final']} fallbacks: {meta['kernels']['fallbacks']}")
    if meta["kernels"]["check"]:
        for k, v in meta["kernels"]["check"].items():
            log.info(f"check {k}: {v}")
    log.info(f"peak GPU mem {meta['peak_mem_gib']:.2f} GiB; timings {timings}; wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
