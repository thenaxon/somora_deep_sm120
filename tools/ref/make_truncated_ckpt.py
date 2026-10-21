#!/usr/bin/env python3
"""Build a truncated DeepSeek-V4.1-Flash HF checkpoint from (a subset of) the original checkpoint.

    python3 make_truncated_ckpt.py --src /ckpt --dst /work/ckpt3                     # 3-layer test model
    python3 make_truncated_ckpt.py --src /ckpt --dst /work/ckpt_yoco --preset yoco   # YOCO test model, see docs/MTP_YOCO.md

Default (--layers 0,2,20):
new layer 0 <- original layer 0   (compress ratio 0, SWA only)
new layer 1 <- original layer 2   (ratio 2, kv/index source)
new layer 2 <- original layer 20  (ratio 1, kv/index source, candidate source, first ratio-1 = n_prefill_layer)

--preset yoco keeps original layers 0,1,2,8,14,20,21,24 (every layer kind incl. the YOCO consumers after 20),
demotes 8 to a C2 consumer of 2 (--as-consumer 8: no C2 consumer shard is on the RTX 5090 test machine), keeps 128 of the
384 routed experts per layer (--num-experts) and lowers candidate_topk_blocks to 128 so that candidate filtering
is active for prompts > 1024 tokens. The derived model is a different function from any prefix of V4.1; it is a
consistency target: DeepInfer and the reference read the same checkpoint and config.

Tensors are copied byte-for-byte (dtype-agnostic, streamed in 64 MiB chunks, RSS stays << 1 GB); the gate
weight / bias rows of the kept experts are copied row by row. MTP/DSpark (unless --mtp), engram (unless
--keep-engram) and every other layer are dropped. Vision/aligner tensors are *kept* because DeepInfer always
builds the ViT + aligner parameter tree and refuses to load if any parameter is not covered
(DeepInfer/deepinfer/model_executor/models/v4_modules/containers.py:44-56 and
model_executor/model_loader.py:147-149). They live in shard model-00001 which is normally not mounted,
so by default (`--vision auto`) they are copied when that shard exists and otherwise synthesised as
bf16 zeros (norm weights = 1). They are never touched by a text-only prompt.

See README.md and docs/MTP_YOCO.md for the full list of config edits and why each one is needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
from collections import Counter, OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from st_io import COPY_CHUNK, ShardReader, StreamWriter, const_bytes, tensor_nbytes  # noqa: E402

INDEX_FILE = "model.safetensors.index.json"
LAYER_RE = re.compile(r"^(model\.)?layers\.(\d+)\.(.+)$")
MTP_RE = re.compile(r"^(model\.)?mtp\.(\d+)\.")
EXPERT_RE = re.compile(r"^ffn\.experts\.(\d+)\.(.+)$")
# per-expert rows of the router (Gate.weight/bias/bias_vl, model.py:807-809; DeepInfer moe.py:62-67)
GATE_ROW_RE = re.compile(r"^ffn\.gate\.(weight|bias|bias_vl|e_score_correction_bias(_vl)?)$")
BF16_ONE = b"\x80\x3f"  # 1.0 in bf16, little endian

PRESETS = {
    # YOCO test model: all layer kinds whose shards are on the RTX 5090 test machine (docs/MTP_YOCO.md "layer selection")
    "yoco": dict(layers="0,1,2,8,14,20,21,24", as_consumer="8", num_experts=128, candidate_topk_blocks=128,
                 allow_engram_drop=True),
}


# --------------------------------------------------------------------------------------------
# config truncation
# --------------------------------------------------------------------------------------------
def nearest_source(source_ids, layer):
    """DeepInfer/deepinfer/hugging_face/config.py:62-65 `_source_map`: nearest source not later than layer."""
    cands = [s for s in source_ids if s <= layer]
    return max(cands) if cands else None


def _ids(s) -> list[int]:
    if s is None or s == "":
        return []
    if isinstance(s, (list, tuple)):
        return [int(x) for x in s]
    out: list[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if "-" in part[1:]:  # range "a-b", inclusive
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def truncate_config(hf: dict, keep: list[int], opts) -> tuple[dict, dict, list[str], dict]:
    """Return (new_config, changed_keys {key: (old, new)}, warnings, info).

    opts: allow_engram_drop, keep_engram [orig ids], as_consumer [orig ids], num_experts, expert_select,
    mtp, dspark_targets [orig ids], candidate_topk_blocks, allow_candidate_drop (argparse namespace).
    info: new_of {orig: new}, expert_ids (kept original expert ids, in new order), demoted, engram_kept, mtp.
    """
    text = hf["text_config"]
    n_orig = text["num_hidden_layers"]
    ratios = text["compress_ratios"]
    nextn = text.get("num_nextn_predict_layers", 0)
    assert len(ratios) == n_orig + nextn, "original compress_ratios must have n_layer + nextn entries"
    if sorted(keep) != keep or len(set(keep)) != len(keep):
        raise ValueError(f"--layers must be strictly increasing, got {keep}")
    if any(not 0 <= l < n_orig for l in keep):
        raise ValueError(f"--layers out of range 0..{n_orig - 1}: {keep}")
    new_of = {old: new for new, old in enumerate(keep)}
    warnings: list[str] = []
    # DeepSeek-V4 had hash-routed MoE layers (num_hash_layers, ffn.gate.tid2eid); V4.1 has none
    # (model.py Gate has no table, LIL vLLM deepseek_v41 passes num_hash_layers=0). Refuse rather than mis-slice.
    if text.get("num_hash_layers"):
        raise ValueError(f"num_hash_layers={text['num_hash_layers']}: hash-routed layers are not supported here")

    kv_ids = text["kv_source_layer_ids"]
    idx_ids = text["index_source_layer_ids"]
    idxk_ids = text.get("index_k_source_layer_ids")  # optional key, defaults to kv sources in DeepInfer
    cand = text.get("candidate_source_layer_id")

    # --as-consumer: a kept source layer gives up its compressor/indexer and reads the nearest earlier source
    demote = sorted(set(opts.as_consumer))
    for l in demote:
        if l not in new_of:
            raise ValueError(f"--as-consumer {l}: layer is not kept")
        if ratios[l] == 0 or (l not in kv_ids and l not in idx_ids):
            raise ValueError(f"--as-consumer {l}: not a compressed kv/index source layer")
        if l == cand:
            raise ValueError(f"--as-consumer {l}: the candidate source must stay an index source")
    if demote:
        warnings.append(f"layers {demote} demoted to consumers: their compressor/indexer weights are dropped")
    kv_eff = [l for l in kv_ids if l not in demote]
    idx_eff = [l for l in idx_ids if l not in demote]
    idxk_eff = [l for l in idxk_ids if l not in demote] if idxk_ids is not None else None
    new_kv = [new_of[l] for l in keep if l in kv_eff]
    new_idx = [new_of[l] for l in keep if l in idx_eff]
    new_idxk = [new_of[l] for l in keep if l in idxk_eff] if idxk_eff is not None else None

    # every compressed layer must keep its (effective) source and nothing may sit in between, otherwise the
    # shared-KV / shared-index topology of the truncated model differs from the original layer's
    for name, eff_ids, new_ids in (("kv", kv_eff, new_kv), ("index", idx_eff, new_idx)) + (
        (("index_k", idxk_eff, new_idxk),) if idxk_eff is not None else ()
    ):
        for l in keep:
            if ratios[l] == 0:
                continue
            src = nearest_source(eff_ids, l)
            if src is None or src not in new_of:
                raise ValueError(f"layer {l} (ratio {ratios[l]}) reads {name} from layer {src}, which is not kept")
            if nearest_source(new_ids, new_of[l]) != new_of[src]:
                raise ValueError(f"layer {l}: {name} source would change after truncation")
            if ratios[src] != ratios[l]:
                raise ValueError(f"layer {l}: {name} source {src} has a different compress ratio")
    # DeepInfer maps every layer >= the first source into the share maps (config.py:62-65) and then
    # requires equal kv_compress_ratio between source and consumer (cross_layer.py _validate_shared_layer_cfg):
    # a ratio-0 layer after the first kept source would break that.
    first_src = min(new_kv + new_idx) if (new_kv + new_idx) else None
    for l in keep:
        if ratios[l] == 0 and first_src is not None and new_of[l] > first_src:
            raise ValueError(f"layer {l} has ratio 0 but comes after the first kept source; DeepInfer rejects that")
    new_ratios = [ratios[l] for l in keep]
    if 1 not in new_ratios:
        raise ValueError("DeepInfer needs at least one ratio-1 layer (config.py:158-161 n_prefill_layer)")

    # candidate source: index sources after it restrict their top-k to its blocks (reference model.py:502-503,
    # 570-576; DeepInfer config.py:76-83 + index_attn.py:97-100), so dropping it changes kept consumers
    if cand is not None and cand in new_of:
        new_cand = new_of[cand]
    else:
        new_cand = None
        users = [l for l in keep if cand is not None and l > cand and l in idx_eff]
        if users and not opts.allow_candidate_drop:
            raise ValueError(f"candidate source {cand} is dropped but index sources {users} use its candidates "
                             "(keep it, or pass --allow-candidate-drop)")
        if cand is not None and any(l > cand for l in keep):
            warnings.append(f"candidate source {cand} dropped but later layers kept: candidate filtering disabled")

    # engram: hash multipliers are seeded with 10007 * layer_id (reference engram.py:75, DeepInfer
    # engram_metadata.py:112) and the bucket primes are drawn in engram_layer_ids order (engram.py:105-117),
    # so a kept engram layer must keep its index and be a prefix of the original engram layer list.
    engram_ids = list(text.get("engram_layer_ids") or [])
    engram_nums = list(text.get("engram_num_embeddings") or [])
    kept_engram = [l for l in keep if l in engram_ids]
    keep_engram = sorted(set(opts.keep_engram))
    for l in keep_engram:
        if l not in kept_engram:
            raise ValueError(f"--keep-engram {l}: not a kept engram layer (engram layers {engram_ids}, kept {keep})")
        if new_of[l] != l:
            raise ValueError(f"--keep-engram {l}: layer would move to index {new_of[l]}; its n-gram hashes are "
                             "seeded with the layer id, so keep every earlier layer")
    if keep_engram and engram_ids[: len(keep_engram)] != keep_engram:
        raise ValueError(f"--keep-engram {keep_engram} must be a prefix of engram_layer_ids {engram_ids} "
                         "(bucket primes are drawn in that order)")
    dropped_engram = [l for l in kept_engram if l not in keep_engram]
    if dropped_engram and not opts.allow_engram_drop:
        raise ValueError(f"kept layers {dropped_engram} carry engram; pass --allow-engram-drop to remove it "
                         "(or --keep-engram to keep it)")
    if dropped_engram:
        warnings.append(f"engram removed from kept layers {dropped_engram}: their outputs differ from the original model")
    new_engram = [new_of[l] for l in keep_engram]
    new_engram_nums = [engram_nums[engram_ids.index(l)] for l in keep_engram] if engram_nums else []

    # MTP / DSpark: the extra layers keep their names (mtp.i, DeepInfer names.py:141-143) and their own expert
    # count (dspark_n_routed_experts); their compress_ratios entries follow the backbone ones.
    k = int(opts.mtp or 0)
    if not 0 <= k <= nextn:
        raise ValueError(f"--mtp {k} out of range 0..{nextn}")
    new_targets: list[int] = []
    if k:
        targets = list(text["dspark_target_layer_ids"])
        tgt = list(opts.dspark_targets) if opts.dspark_targets else targets
        if len(tgt) != len(targets):
            raise ValueError(f"--dspark-targets needs {len(targets)} layers (main_proj input width), got {tgt}")
        if len(set(tgt)) != len(tgt):  # DeepInfer deepseek.py:73 maps layer -> feature slot; a duplicate leaves one unset
            raise ValueError(f"--dspark-targets must be distinct, got {tgt}")
        missing = [t for t in tgt if t not in new_of]
        if missing:
            raise ValueError(f"DSpark target layers {missing} are not kept; pass --dspark-targets <kept orig ids>")
        if tgt != targets:
            warnings.append(f"DSpark reads layers {tgt} instead of {targets}: the draft model is no longer the trained one")
        new_targets = [new_of[t] for t in tgt]
        new_ratios = new_ratios + list(ratios[n_orig : n_orig + k])

    # routed-expert subset: same original experts in every backbone layer, renumbered 0..N-1
    E, topk = text["n_routed_experts"], text["num_experts_per_tok"]
    expert_ids = list(range(E))
    N = opts.num_experts
    if N and N != E:
        if not topk <= N <= E:
            raise ValueError(f"--num-experts {N} must be in {topk}..{E}")
        if N % 4:
            raise ValueError(f"--num-experts {N}: tile_kernels moe_topk_gate needs a multiple of 4")
        expert_ids = list(range(N)) if opts.expert_select == "first" else [i * E // N for i in range(N)]
        warnings.append(f"routed experts reduced to {N}/{E} ({opts.expert_select}): routing differs from the original model")

    ctb = opts.candidate_topk_blocks
    if ctb is not None and ctb != text["candidate_topk_blocks"]:
        bs = text["candidate_block_size"]
        if not (0 < ctb <= 4096 and ctb % 8 == 0):  # cross_layer.py:351, index_attn.py:105
            raise ValueError(f"--candidate-topk-blocks {ctb}: needs a positive multiple of 8, <= 4096")
        if ctb * bs < text["index_topk"]:  # cross_layer.py:367-368 candidate_capacity >= index_topk
            raise ValueError(f"--candidate-topk-blocks {ctb} x block {bs} < index_topk {text['index_topk']}")
        warnings.append(f"candidate_topk_blocks {text['candidate_topk_blocks']} -> {ctb}: candidate filtering is "
                        f"active beyond {ctb * bs} compressed positions")

    new_text = OrderedDict(text)
    edits = OrderedDict(
        num_hidden_layers=len(keep),
        compress_ratios=new_ratios,
        kv_source_layer_ids=new_kv,
        index_source_layer_ids=new_idx,
        candidate_source_layer_id=new_cand,
        engram_layer_ids=new_engram,
        engram_num_embeddings=new_engram_nums,
        num_nextn_predict_layers=k,
        dspark_target_layer_ids=new_targets,
    )
    if idxk_ids is not None:
        edits["index_k_source_layer_ids"] = new_idxk
    if len(expert_ids) != E:
        edits["n_routed_experts"] = len(expert_ids)

    # MTP layers have their own routed-expert count (dspark_n_routed_experts, model.py:142-149, DeepInfer
    # dflash_cfg.moe); --mtp-num-experts slices them like the backbone
    mtp_expert_ids = None
    MN = getattr(opts, "mtp_num_experts", None)
    if k and MN:
        EM, topk_m = text["dspark_n_routed_experts"], text["dspark_num_experts_per_tok"]
        if not topk_m <= MN <= EM or MN % 4:
            raise ValueError(f"--mtp-num-experts {MN} must be a multiple of 4 in {topk_m}..{EM}")
        if MN != EM:
            mtp_expert_ids = list(range(MN)) if opts.expert_select == "first" else [i * EM // MN for i in range(MN)]
            edits["dspark_n_routed_experts"] = MN
            warnings.append(f"MTP routed experts reduced to {MN}/{EM}")
    if ctb is not None:
        edits["candidate_topk_blocks"] = ctb
    changed = OrderedDict()
    for key, v in edits.items():
        old = text.get(key, "<absent>")
        if old != v:
            changed[f"text_config.{key}"] = (old, v)
        new_text[key] = v
    new_hf = OrderedDict(hf)
    new_hf["text_config"] = new_text
    # vision_config is deliberately kept: DeepInfer config.py:217-246 (_build_vision) and
    # hugging_face/processor.py:25 index it unconditionally. The reference runner disables vision itself.
    info = dict(new_of=new_of, expert_ids=expert_ids, demoted=demote, engram_kept=keep_engram, mtp=k,
                mtp_expert_ids=mtp_expert_ids)
    return new_hf, changed, warnings, info


def deepinfer_topology_check(text: dict) -> list[str]:
    """Offline replica of DeepInfer's config-time topology checks on the truncated text_config.

    hugging_face/config.py:62-84 (_source_map, candidate map), :146-161 (ratios length, n_prefill_layer) and
    model_executor/models/v4_modules/cross_layer.py:219-368 (segments, shared cfg, refinement chain, YOCO
    boundary, candidate constraints) + index_attn.py:101-105. Returns a list of violations (empty = OK).
    """
    n = text["num_hidden_layers"]
    allr = text["compress_ratios"]
    if len(allr) != n + text.get("num_nextn_predict_layers", 0):
        return [f"compress_ratios has {len(allr)} entries for {n} + nextn layers (config.py:151)"]
    ratios = allr[:n]
    if 1 not in ratios:
        return ["no ratio-1 layer (config.py:158-161 n_prefill_layer)"]
    n_prefill = ratios.index(1)
    probs: list[str] = []

    def smap(ids):
        m = list(range(n))
        ids = sorted(ids)
        if ids:
            for l in range(ids[0], n):
                m[l] = max(s for s in ids if s <= l)
        return m

    kv = smap(text["kv_source_layer_ids"])
    idx = smap(text["index_source_layer_ids"])
    ik = smap(text.get("index_k_source_layer_ids", text["kv_source_layer_ids"]))
    topk = [text["index_topk"] if r > 0 else 0 for r in ratios]
    for name, m in (("index-topk", idx), ("index-k", ik), ("kv", kv)):
        for l, s in enumerate(m):
            if s == l:
                continue
            if ratios[l] == 0:  # only _compress_spec layers carry layer_info_shared_cfg
                probs.append(f"{name}: consumer {l} has ratio 0 / no layer_info_shared_cfg (cross_layer.py:240)")
            elif ratios[l] != ratios[s] or topk[l] != topk[s]:
                probs.append(f"{name}: layer {l} and source {s} differ in ratio/index_topk (cross_layer.py:383-391)")
    for fname, fine, cname, coarse in (("index-topk", idx, "index-k", ik), ("index-k", ik, "kv", kv)):
        for l, s in enumerate(fine):
            if coarse[s] != coarse[l]:
                probs.append(f"{fname} map must refine {cname}: layer {l} (cross_layer.py:312-319)")
    if n_prefill < n and kv[n_prefill] != n_prefill:
        probs.append(f"share-kv crosses the YOCO boundary n_prefill_layer={n_prefill} (cross_layer.py:246)")
    cand = text.get("candidate_source_layer_id")
    if cand is not None:
        tb, bs = text["candidate_topk_blocks"], text["candidate_block_size"]
        if not 0 <= cand < n:
            probs.append(f"candidate source {cand} out of range")
            return probs
        if idx[cand] != cand or topk[cand] <= 0 or ratios[cand] <= 0:
            probs.append(f"candidate source {cand} must be a compressed index source (cross_layer.py:343-352)")
        if not (0 < tb <= 4096 and tb % 8 == 0) or bs not in (8, 16):
            probs.append(f"candidate_topk_blocks {tb} / block {bs} (cross_layer.py:351, index_attn.py:104-105)")
        for l in range(cand + 1, n):
            if ratios[l] != ratios[cand]:
                probs.append(f"candidate consumer {l}: ratio {ratios[l]} != source ratio (cross_layer.py:363)")
                continue
            cbs = bs * ratios[cand] // ratios[l]
            if cbs > 16 or tb * cbs < topk[l]:
                probs.append(f"candidate consumer {l}: capacity {tb * cbs} < index_topk (cross_layer.py:366-368)")
    return probs


# --------------------------------------------------------------------------------------------
# expected tensors (static model of both loaders)
# --------------------------------------------------------------------------------------------
def _cd(x):
    return (x + 31) // 32


def layer_tensor_specs(text: dict, layer: int) -> dict[str, tuple[list[int], str | None, set[str]]]:
    """suffix -> (shape, required dtype or None, consumers). consumers subset of {"deepinfer","reference"}.

    Reference: ds41-inference/inference/model.py (Attention 623-699, Compressor 437-456, Indexer
    496-525, Gate 798-809, Expert 836-841, MoE 860-889, Block 921-948, Linear 214-240).
    DeepInfer: hugging_face/names.py:42-77 (+ moe.py:63-64 bias_vl, always built).
    """
    both = {"deepinfer", "reference"}
    dim, ql, H, hd = text["hidden_size"], text["q_lora_rank"], text["num_attention_heads"], text["head_dim"]
    G, ol, inter, E = text["o_groups"], text["o_lora_rank"], text["moe_intermediate_size"], text["n_routed_experts"]
    ih, ihd, hc = text["index_n_heads"], text["index_head_dim"], text["hc_mult"]
    mix = (2 + hc) * hc
    ratio = text["compress_ratios"][layer]
    kv_src = layer in text["kv_source_layer_ids"]
    idx_src = layer in text["index_source_layer_ids"]
    F8, E8, F32 = "F8_E4M3", "F8_E8M0", "F32"

    s: dict[str, tuple[list[int], str | None, set[str]]] = {}

    def fp8_linear(name, out_f, in_f):
        s[f"{name}.weight"] = ([out_f, in_f], F8, both)
        s[f"{name}.scale"] = ([_cd(out_f), _cd(in_f)], E8, both)

    s["attn.attn_sink"] = ([H], F32, both)  # DeepInfer loading.py:113 requires F32
    fp8_linear("attn.wq_a", ql, dim)
    s["attn.q_norm.weight"] = ([ql], None, both)
    fp8_linear("attn.wq_b", H * hd, ql)
    fp8_linear("attn.wkv", hd, dim)
    s["attn.kv_norm.weight"] = ([hd], None, both)
    fp8_linear("attn.wo_a", G * ol, H * hd // G)
    fp8_linear("attn.wo_b", dim, G * ol)
    s["attn_norm.weight"] = ([dim], None, both)
    s["ffn_norm.weight"] = ([dim], None, both)
    for sub in ("attn", "ffn"):
        s[f"hc_{sub}_fn"] = ([mix, hc * dim], F32, both)  # DeepInfer loading.py:113 requires F32
        s[f"hc_{sub}_base"] = ([mix], None, both)
        s[f"hc_{sub}_scale"] = ([3], None, both)
    s["ffn.gate.weight"] = ([E, dim], None, both)
    s["ffn.gate.bias"] = ([E], None, both)
    s["ffn.gate.bias_vl"] = ([E], None, {"deepinfer"})  # reference builds it only with vision enabled
    fp8_linear("ffn.shared_experts.w1", inter, dim)
    fp8_linear("ffn.shared_experts.w3", inter, dim)
    fp8_linear("ffn.shared_experts.w2", dim, inter)
    for e in range(E):
        for w, (o, i) in (("w1", (inter, dim)), ("w3", (inter, dim)), ("w2", (dim, inter))):
            s[f"ffn.experts.{e}.{w}.weight"] = ([o, i // 2], "I8", both)  # packed e2m1 x2
            s[f"ffn.experts.{e}.{w}.scale"] = ([o, i // 32], E8, both)
    if ratio and kv_src:
        s["attn.compressor.wkv.weight"] = ([hd, dim], None, both)
        s["attn.compressor.norm.weight"] = ([hd], None, both)
        if ratio > 1:
            s["attn.compressor.wgate.weight"] = ([hd, dim], None, both)
    if ratio and idx_src:
        fp8_linear("attn.indexer.wq_b", ih * ihd, ql)
        s["attn.indexer.weights_proj.weight"] = ([ih, dim], None, both)
        if kv_src:  # reference Indexer.owns_k (model.py:500); DeepInfer index-k source (default = kv sources)
            s["attn.indexer.wk.weight"] = ([ihd, hd], None, both)
            s["attn.indexer.k_norm.weight"] = ([ihd], None, both)
    return s


def vision_tensor_specs(hf: dict) -> dict[str, tuple[list[int], str]]:
    """HF names/shapes of the ViT + aligner (reference vision.py:36-119, DeepInfer vit.py:24-31,126-194,
    mlp_projector.py; name map DeepInfer names.py:79-91). All bf16 (DeepInfer builds them under the
    bf16 default dtype and validate_hf_shapes requires equal dtypes)."""
    v, text = hf["vision_config"], hf["text_config"]
    d, L, inter, p, r = v["hidden_size"], v["num_hidden_layers"], v["intermediate_size"], v["patch_size"], v["downsample_ratio"]
    D = text["hidden_size"]
    out = {
        "vision.patch_embed.proj.weight": [d, 3 * p * p],
        "vision.patch_embed.proj.bias": [d],
        "vision.norm.weight": [d],
        "aligner.w1.weight": [D, d * r * r],
        "aligner.w1.bias": [D],
        "aligner.w2.weight": [D, D],
        "aligner.w2.bias": [D],
    }
    for i in range(L):
        b = f"vision.blocks.{i}."
        out[b + "norm1.weight"] = [d]
        out[b + "norm2.weight"] = [d]
        out[b + "attn.wqkv.weight"] = [3 * d, d]
        out[b + "attn.wqkv.bias"] = [3 * d]
        out[b + "attn.wo.weight"] = [d, d]
        out[b + "attn.wo.bias"] = [d]
        out[b + "mlp.w1.weight"] = [2 * inter, d]
        out[b + "mlp.w2.weight"] = [d, inter]
    return {k: (s, "BF16") for k, s in out.items()}


def global_tensor_specs(text: dict) -> dict[str, tuple[list[int], str | None, set[str]]]:
    V, D = text["vocab_size"], text["hidden_size"]
    both = {"deepinfer", "reference"}
    return {
        "embed.weight": ([V, D], None, both),
        "norm.weight": ([D], None, both),
        "head.weight": ([V, D], None, both),
        # DeepInfer containers.py:26-35 always builds these; reference only with vision enabled
        "image_start": ([D], None, {"deepinfer"}),
        "image_end": ([D], None, {"deepinfer"}),
        "image_newline": ([D], None, {"deepinfer"}),
    }


def expected_tensors(hf: dict) -> dict[str, tuple[list[int], str | None, set[str]]]:
    text = hf["text_config"]
    out = dict(global_tensor_specs(text))
    for l in range(text["num_hidden_layers"]):
        for suf, spec in layer_tensor_specs(text, l).items():
            out[f"layers.{l}.{suf}"] = spec
    for k, (shape, dt) in vision_tensor_specs(hf).items():
        out[k] = (shape, dt, {"deepinfer"})
    return out


# --------------------------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------------------------
def classify(name: str):
    plain = name.removeprefix("model.")
    m = MTP_RE.match(name)
    if m:
        return "mtp", int(m.group(2))
    if plain.startswith("vision.") or plain.startswith("aligner."):
        return "vision", None
    m = LAYER_RE.match(name)
    if m:
        return "layer", (int(m.group(2)), m.group(3))
    return "global", None


def sort_key(name: str):
    m = re.search(r"\.experts\.(\d+)\.", name)
    return (1, int(m.group(1)), name) if m else (0, 0, name)


def gpu_estimate(entries: dict, text: dict) -> dict:
    """GiB of parameters each stack keeps on the GPU, from the planned tensors.

    DeepInfer: everything it loads (vision placeholders included), engram tables excluded (placement depends on
    the engram backend). Reference (ref_convert.py + ref_runner.py): no vision/aligner/image_*/bias_vl/mtp/engram,
    head promoted to fp32 (model.py:1008), ratio>1 compressor wkv/wgate promoted to fp32 (model.py:446-448),
    wo_a dequantised to bf16 without its scale (convert.py:157-177).
    """
    gib = 1 << 30
    c: Counter = Counter()
    ratios = text["compress_ratios"]
    for name, (dt, _sh, nb, _src) in entries.items():
        m = LAYER_RE.match(name)
        suffix = m.group(3) if m else ""
        if suffix.startswith("engram."):
            c["engram"] += nb
            continue
        routed = bool(EXPERT_RE.match(suffix))
        c["di_routed" if routed else "di_other"] += nb
        if name.startswith(("vision.", "aligner.", "image_", "mtp.")) or suffix == "ffn.gate.bias_vl":
            continue
        if name == "head.weight":
            nb = nb * 4 // tensor_nbytes(dt, [1])
        elif m and re.fullmatch(r"attn\.compressor\.(wkv|wgate)\.weight", suffix) and ratios[int(m.group(2))] > 1:
            nb = nb * 4 // tensor_nbytes(dt, [1])
        elif suffix == "attn.wo_a.weight":
            nb *= 2
        elif suffix == "attn.wo_a.scale":
            nb = 0
        c["ref_routed" if routed else "ref_other"] += nb
    return {k: round(v / gib, 3) for k, v in sorted(c.items())} | dict(
        deepinfer_total=round((c["di_routed"] + c["di_other"]) / gib, 3),
        reference_total=round((c["ref_routed"] + c["ref_other"]) / gib, 3),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="/ckpt")
    ap.add_argument("--dst", default=None, help="default /work/ckpt3 (/work/ckpt_yoco with --preset yoco)")
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None,
                    help="named option set (explicit flags still win), see PRESETS / docs/MTP_YOCO.md")
    ap.add_argument("--layers", default=None, help="original layer ids to keep, in order (default 0,2,20)")
    ap.add_argument("--as-consumer", default=None,
                    help="original kv/index source layers to demote to consumers of the nearest earlier source")
    ap.add_argument("--num-experts", type=int, default=None, help="routed experts kept per layer (default: all)")
    ap.add_argument("--expert-select", choices=("first", "stride"), default="first",
                    help="first: experts 0..N-1; stride: every (E/N)-th expert")
    ap.add_argument("--candidate-topk-blocks", type=int, default=None, help="override candidate_topk_blocks")
    ap.add_argument("--keep-engram", default=None,
                    help="original engram layer ids whose engram is kept (must keep their index, e.g. '1')")
    ap.add_argument("--engram-mode", choices=("link", "copy"), default="link",
                    help="link: symlink the original engram shard (needs a shard holding only those tensors); copy: copy")
    ap.add_argument("--mtp", type=int, default=0, help="keep the first K MTP/DSpark layers (default 0 = none)")
    ap.add_argument("--mtp-num-experts", type=int, default=None,
                    help="routed experts kept per MTP layer (default: all dspark_n_routed_experts=128)")
    ap.add_argument("--dspark-targets", default=None, help="original layer ids DSpark reads (when 37-39 are dropped)")
    ap.add_argument("--max-shard-gb", type=float, default=9.5, help="max output shard size (GiB)")
    ap.add_argument("--vision", choices=("auto", "copy", "synth"), default="auto")
    ap.add_argument("--allow-engram-drop", action="store_true", default=None)
    ap.add_argument("--allow-candidate-drop", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="plan + validate only, write nothing")
    ap.add_argument("--overwrite", action="store_true", help="allow a non-empty --dst")
    ap.add_argument("--verify-bytes", action="store_true", help="re-read every written tensor and compare hashes")
    ap.add_argument("--no-shape-check", action="store_true", help="downgrade expected-shape mismatches to warnings")
    args = ap.parse_args()
    for k, v in (PRESETS.get(args.preset) or {}).items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.layers is None:
        args.layers = "0,2,20"
    if args.dst is None:
        args.dst = "/work/ckpt_yoco" if args.preset == "yoco" else "/work/ckpt3"
    args.as_consumer = _ids(args.as_consumer)
    args.keep_engram = _ids(args.keep_engram)
    args.dspark_targets = _ids(args.dspark_targets)
    args.allow_engram_drop = bool(args.allow_engram_drop)

    keep = _ids(args.layers)
    src, dst = os.path.abspath(args.src), os.path.abspath(args.dst)
    hf = json.load(open(os.path.join(src, "config.json")))
    index = json.load(open(os.path.join(src, INDEX_FILE)))["weight_map"]
    try:
        new_hf, changed, warnings, info = truncate_config(hf, keep, args)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from None
    new_of, expert_ids = info["new_of"], info["expert_ids"]
    new_expert = {old: new for new, old in enumerate(expert_ids)}
    E_orig = hf["text_config"]["n_routed_experts"]
    slice_experts = len(expert_ids) != E_orig
    for w in warnings:
        print(f"WARNING: {w}")
    topo = deepinfer_topology_check(new_hf["text_config"])
    if topo:
        for p in topo:
            print("ERROR: DeepInfer topology:", p)
        raise SystemExit(f"truncated config violates {len(topo)} DeepInfer topology constraints")

    # ---- select and rename ---------------------------------------------------------------
    plan_globals: dict[str, str] = {}  # new name -> src name
    plan_layers: dict[int, dict[str, str]] = {i: {} for i in range(len(keep))}
    plan_rows: dict[str, tuple[str, list[int]]] = {}  # new name -> (src name, kept rows) for gate rows
    plan_mtp: dict[str, str] = {}
    plan_vision: dict[str, str] = {}
    dropped = Counter(mtp=0, other_layers=0, engram=0, experts=0, demoted_attn=0)
    for name in index:
        kind, info_ = classify(name)
        if kind == "mtp":
            if info_ >= info["mtp"]:
                dropped["mtp"] += 1
                continue
            plain = name.removeprefix("model.")
            prefix, msuffix = plain.split(".", 2)[:2], plain.split(".", 2)[2]
            mids = info["mtp_expert_ids"]
            if mids is not None and (m := EXPERT_RE.match(msuffix)):
                e = int(m.group(1))
                if e in mids:
                    plan_mtp[f"{'.'.join(prefix)}.ffn.experts.{mids.index(e)}.{m.group(2)}"] = name
                else:
                    dropped["experts"] += 1
            elif mids is not None and GATE_ROW_RE.match(msuffix):
                plan_rows[plain] = (name, mids)
            else:
                plan_mtp[plain] = name
        elif kind == "vision":
            plan_vision[name.removeprefix("model.")] = name
        elif kind == "layer":
            old, suffix = info_
            if old not in new_of:
                dropped["other_layers"] += 1
                continue
            new = new_of[old]
            if "tid2eid" in suffix:
                raise SystemExit(f"{name}: hash-routing table found; remap it before slicing experts")
            if suffix.startswith("engram."):
                if old in info["engram_kept"]:
                    plan_layers[new][f"layers.{new}.{suffix}"] = name
                else:
                    dropped["engram"] += 1
            elif old in info["demoted"] and suffix.startswith(("attn.compressor.", "attn.indexer.")):
                dropped["demoted_attn"] += 1
            elif (m := EXPERT_RE.match(suffix)) and slice_experts:
                e = int(m.group(1))
                if e in new_expert:
                    plan_layers[new][f"layers.{new}.ffn.experts.{new_expert[e]}.{m.group(2)}"] = name
                else:
                    dropped["experts"] += 1
            elif GATE_ROW_RE.match(suffix) and slice_experts:
                plan_rows[f"layers.{new}.{suffix}"] = (name, expert_ids)
            else:
                plan_layers[new][f"layers.{new}.{suffix}"] = name
        else:
            plan_globals[name.removeprefix("model.")] = name

    # ---- source headers --------------------------------------------------------------------
    needed = (set(plan_globals.values()) | {n for d in plan_layers.values() for n in d.values()}
              | {s for s, _ in plan_rows.values()} | set(plan_mtp.values()))
    shard_of = {n: index[n] for n in needed}
    missing_shards = sorted({s for s in shard_of.values() if not os.path.isfile(os.path.join(src, s))})
    if missing_shards:
        lost = sorted(n for n, s in shard_of.items() if s in missing_shards)
        raise SystemExit(f"missing source shards {missing_shards} needed for {len(lost)} tensors, e.g. {lost[:5]}")
    vision_shards = sorted(set(index[n] for n in plan_vision.values()))
    vision_available = bool(plan_vision) and all(os.path.isfile(os.path.join(src, s)) for s in vision_shards)
    if args.vision == "copy" and not vision_available:
        raise SystemExit(f"--vision copy but vision shards {vision_shards} are not present in {src}")
    synth_vision = args.vision == "synth" or not vision_available
    if not synth_vision:
        for n in plan_vision.values():
            shard_of[n] = index[n]

    readers: dict[str, ShardReader] = {}

    def reader(shard: str) -> ShardReader:
        if shard not in readers:
            readers[shard] = ShardReader(os.path.join(src, shard))
        return readers[shard]

    def src_info(src_name: str):
        r = reader(shard_of[src_name])
        if src_name not in r:
            raise SystemExit(f"{src_name} is listed in the index under {shard_of[src_name]} but missing from that shard")
        return r.info(src_name)

    # entries: new name -> (dtype, shape, nbytes, source)
    # source = ("copy", src_name) | ("rows", src_name, rows) | ("const", pattern) | ("link", link_file)
    entries: dict[str, tuple[str, list[int], int, tuple]] = {}
    for new, old in plan_globals.items():
        dt, sh, n = src_info(old)
        entries[new] = (dt, sh, n, ("copy", old))
    for layer_plan in plan_layers.values():
        for new, old in layer_plan.items():
            dt, sh, n = src_info(old)
            if slice_experts and sh and sh[0] == E_orig and not EXPERT_RE.match(LAYER_RE.match(new).group(3)):
                # an expert-indexed tensor the slicing does not know about (e.g. a new routing table)
                raise SystemExit(f"{old}: shape {sh} has a {E_orig}-row dim; extend GATE_ROW_RE before slicing experts")
            entries[new] = (dt, sh, n, ("copy", old))
    for new, (old, rows) in plan_rows.items():
        dt, sh, n = src_info(old)
        n_full = E_orig if new.startswith("layers.") else hf["text_config"]["dspark_n_routed_experts"]
        if not sh or sh[0] != n_full:
            raise SystemExit(f"{old}: shape {sh} is not [{n_full}, ...]; cannot select expert rows")
        new_sh = [len(rows)] + list(sh[1:])
        entries[new] = (dt, new_sh, tensor_nbytes(dt, new_sh), ("rows", old, rows))
    for new, old in plan_mtp.items():
        dt, sh, n = src_info(old)
        entries[new] = (dt, sh, n, ("copy", old))
    vspecs = vision_tensor_specs(new_hf)
    if synth_vision:
        if plan_vision and set(plan_vision) != set(vspecs):
            diff = sorted(set(plan_vision) ^ set(vspecs))
            raise SystemExit(f"vision tensor names in the index do not match the synthesis table: {diff[:10]}")
        print(f"vision/aligner: synthesising {len(vspecs)} bf16 placeholder tensors (source shard not available)")
        for new, (sh, dt) in vspecs.items():
            pattern = BF16_ONE if re.search(r"norm\d?\.weight$", new) else b"\x00\x00"
            entries[new] = (dt, sh, tensor_nbytes(dt, sh), ("const", pattern))
    else:
        print(f"vision/aligner: copying {len(plan_vision)} tensors from {vision_shards}")
        for new, old in plan_vision.items():
            dt, sh, n = src_info(old)
            entries[new] = (dt, sh, n, ("copy", old))

    # engram tables (~100 GB per layer): reuse the original shard through a symlink when it holds exactly the
    # kept engram tensors under unchanged names (a kept engram layer never moves, see truncate_config)
    linked: dict[str, str] = {}  # link file name in dst -> absolute source shard
    if args.engram_mode == "link":
        eng = {n: e for n, e in entries.items() if (m := LAYER_RE.match(n)) and m.group(3).startswith("engram.")}
        by_shard: dict[str, set[str]] = {}
        for n, e in eng.items():
            by_shard.setdefault(shard_of[e[3][1]], set()).add(n)
        for shard, names in sorted(by_shard.items()):
            listed = {k for k, s in index.items() if s == shard}
            if listed == names and all(entries[n][3][1] == n for n in names):
                link = f"engram-{shard}"
                linked[link] = os.path.join(src, shard)
                for n in names:
                    dt, sh, nb, _ = entries[n]
                    entries[n] = (dt, sh, nb, ("link", link))
            else:
                raise SystemExit(f"--engram-mode link: {shard} also holds other tensors or names change; use copy")

    # ---- validation against both loaders --------------------------------------------------
    exp = expected_tensors(new_hf)
    problems, shape_warn = [], []
    for name, (shape, dt, consumers) in exp.items():
        if name not in entries:
            problems.append(f"missing {name} (needed by {sorted(consumers)})")
            continue
        got_dt, got_sh = entries[name][0], entries[name][1]
        if list(got_sh) != list(shape):
            shape_warn.append(f"{name}: shape {got_sh} != expected {shape}")
        if dt is not None and got_dt != dt:
            shape_warn.append(f"{name}: dtype {got_dt} != expected {dt}")
    unexpected = sorted(set(entries) - set(exp))
    if unexpected:
        groups = sorted({".".join(n.split(".")[:3]) for n in unexpected})
        print(f"note: {len(unexpected)} tensors not in the static expectation (kept anyway): "
              f"{groups[:10]}{' ...' if len(groups) > 10 else ''}")
    if shape_warn:
        for w in shape_warn[:40]:
            print(("WARNING: " if args.no_shape_check else "ERROR: ") + w)
        if not args.no_shape_check:
            problems.append(f"{len(shape_warn)} shape/dtype mismatches (use --no-shape-check to override)")
    # DeepInfer requires exactly n_routed_experts experts per layer (weights.py:116-150)
    E = new_hf["text_config"]["n_routed_experts"]
    for l in range(len(keep)):
        ids = {int(m.group(1)) for n in entries if (m := re.match(rf"^layers\.{l}\.ffn\.experts\.(\d+)\.w1\.weight$", n))}
        if ids != set(range(E)):
            problems.append(f"layer {l}: experts {len(ids)} != {E}")
    if problems:
        for p in problems[:50]:
            print("ERROR:", p)
        raise SystemExit(f"validation failed ({len(problems)} problems)")
    n_ref = sum(1 for v in exp.values() if "reference" in v[2])
    n_di = sum(1 for v in exp.values() if "deepinfer" in v[2])
    print(f"validation OK: {len(entries)} tensors; reference expects {n_ref}, DeepInfer expects {n_di}, all present; "
          "DeepInfer topology checks OK")

    # ---- shard plan ------------------------------------------------------------------------
    max_bytes = int(args.max_shard_gb * (1 << 30))
    written = {n: e for n, e in entries.items() if e[3][0] != "link"}
    groups: list[list[str]] = []
    groups.append(sorted([n for n in written if not n.startswith(("layers.", "mtp."))], key=sort_key))
    for l in range(len(keep)):
        groups.append(sorted([n for n in written if n.startswith(f"layers.{l}.")], key=sort_key))
    for i in range(info["mtp"]):
        groups.append(sorted([n for n in written if n.startswith(f"mtp.{i}.")], key=sort_key))
    shards: list[list[str]] = []
    for g in groups:
        cur, cur_b = [], 0
        for n in g:
            b = entries[n][2]
            if cur and cur_b + b > max_bytes:
                shards.append(cur)
                cur, cur_b = [], 0
            cur.append(n)
            cur_b += b
        if cur:
            shards.append(cur)
    nsh = len(shards)
    shard_names = [f"model-{i + 1:05d}-of-{nsh:05d}.safetensors" for i in range(nsh)]
    total = sum(e[2] for e in written.values())
    for sn, names in zip(shard_names, shards):
        print(f"  {sn}: {len(names):5d} tensors, {sum(entries[n][2] for n in names) / 2**30:7.2f} GiB")
    for link, target in linked.items():
        n_l = sum(1 for e in entries.values() if e[3] == ("link", link))
        print(f"  {link}: {n_l:5d} tensors -> symlink {target}")
    print(f"total {total / 2**30:.2f} GiB written in {nsh} shards"
          + (f" (+ {sum(e[2] for e in entries.values()) / 2**30 - total / 2**30:.2f} GiB linked)" if linked else ""))
    print("config edits:")
    for k, (o, n) in changed.items():
        print(f"  {k}: {o} -> {n}")
    print(f"dropped: {dict(dropped)}")
    est = gpu_estimate(entries, new_hf["text_config"])
    print(f"GPU parameter estimate (GiB): DeepInfer {est['deepinfer_total']} (routed {est['di_routed']}), "
          f"reference {est['reference_total']} (routed {est['ref_routed']})"
          + (f", engram tables {est['engram']} not counted" if est.get("engram") else ""))

    if args.dry_run:
        print("dry run: nothing written")
        return 0

    # ---- write -----------------------------------------------------------------------------
    os.makedirs(dst, exist_ok=True)
    if os.listdir(dst) and not args.overwrite:
        raise SystemExit(f"{dst} is not empty (use --overwrite)")
    free = shutil.disk_usage(dst).free
    if free < total * 1.02:
        raise SystemExit(f"not enough space in {dst}: need {total / 2**30:.1f} GiB, have {free / 2**30:.1f} GiB")
    for f in os.listdir(dst):  # stale shards from an earlier run with a different shard count
        if re.fullmatch(r"(model-\d{5}-of-\d{5}\.safetensors(\.tmp)?|engram-.*\.safetensors)", f) and f not in shard_names:
            os.remove(os.path.join(dst, f))

    def source_chunks(entry):
        kind = entry[3]
        if kind[0] == "copy":
            return reader(shard_of[kind[1]]).iter_bytes(kind[1])
        if kind[0] == "rows":
            return reader(shard_of[kind[1]]).iter_rows(kind[1], kind[2])
        return const_bytes(entry[2], kind[1])

    weight_map: dict[str, str] = {}
    for sn, names in zip(shard_names, shards):
        path = os.path.join(dst, sn)
        w = StreamWriter(path, [(n, entries[n][0], entries[n][1]) for n in names], metadata={"format": "pt"})
        try:
            for i, n in enumerate(names):
                w.write_tensor(n, source_chunks(entries[n]))
                weight_map[n] = sn
                if (i + 1) % 500 == 0:
                    print(f"  {sn}: {i + 1}/{len(names)}", flush=True)
            w.close()
        except BaseException:
            w.abort()
            raise
        print(f"wrote {path}", flush=True)
    for link, target in linked.items():
        os.symlink(target, os.path.join(dst, link))
        for n, e in entries.items():
            if e[3] == ("link", link):
                weight_map[n] = link
        print(f"linked {link} -> {target}")

    with open(os.path.join(dst, INDEX_FILE), "w") as f:
        json.dump({"metadata": {"total_size": sum(e[2] for e in entries.values())},
                   "weight_map": dict(sorted(weight_map.items()))}, f, indent=2)
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(new_hf, f, indent=2, ensure_ascii=False)
        f.write("\n")
    record = {
        "source": src,
        "layer_map_new_to_orig": {str(n): o for n, o in enumerate(keep)},
        "config_edits": {k: {"old": o, "new": n} for k, (o, n) in changed.items()},
        "expert_ids_new_to_orig": expert_ids if slice_experts else None,
        "mtp_expert_ids_new_to_orig": info["mtp_expert_ids"],
        "demoted_to_consumer": info["demoted"],
        "engram_kept": info["engram_kept"],
        "mtp_kept": info["mtp"],
        "linked": linked,
        "dropped_tensor_counts": dict(dropped),
        "vision_synthesized": synth_vision,
        "warnings": warnings,
        "gpu_estimate_gib": est,
        "n_tensors": len(entries),
        "total_bytes": total,
    }
    with open(os.path.join(dst, "truncation.json"), "w") as f:
        json.dump(record, f, indent=2)

    # aux files: tokenizer, encoding/, generation config, ... (everything that is not weights/index/config)
    for item in sorted(os.listdir(src)):
        s = os.path.join(src, item)
        if item.endswith(".safetensors") or item in (INDEX_FILE, "config.json"):
            continue
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
            print(f"copied dir {item}/")
        elif os.path.getsize(s) < (256 << 20):
            shutil.copyfile(s, d)
            print(f"copied {item}")
        else:
            print(f"WARNING: skipped large non-weight file {item}")
    for req in ("tokenizer.json", "tokenizer_config.json"):
        if not os.path.isfile(os.path.join(dst, req)):
            print(f"WARNING: {req} missing in {src}; both DeepInfer and the reference runner need it")

    # ---- post-write verification -------------------------------------------------------------
    print("verifying written shards ...")
    for sn, names in list(zip(shard_names, shards)) + [(lk, [n for n, e in entries.items() if e[3] == ("link", lk)])
                                                      for lk in linked]:
        with ShardReader(os.path.join(dst, sn)) as r:
            if set(r.keys()) != set(names):
                raise SystemExit(f"{sn}: tensor set mismatch after write")
            for n in names:
                dt, sh, nb = r.info(n)
                if (dt, list(sh), nb) != (entries[n][0], list(entries[n][1]), entries[n][2]):
                    raise SystemExit(f"{sn}:{n}: header mismatch after write")
            if args.verify_bytes and sn not in linked:
                for n in names:
                    h_dst = hashlib.blake2b(digest_size=16)
                    for c in r.iter_bytes(n, COPY_CHUNK):
                        h_dst.update(c)
                    h_src = hashlib.blake2b(digest_size=16)
                    for c in source_chunks(entries[n]):
                        h_src.update(c)
                    if h_src.digest() != h_dst.digest():
                        raise SystemExit(f"{sn}:{n}: byte mismatch")
    for r in readers.values():
        r.close()
    print(f"done: {dst} ({len(entries)} tensors, {total / 2**30:.2f} GiB){' [bytes verified]' if args.verify_bytes else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
