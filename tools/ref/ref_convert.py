#!/usr/bin/env python3
"""MP=1 / expert_dtype=fp4 re-implementation of ds41-inference/inference/convert.py, streaming.

    python3 ref_convert.py --hf-ckpt-path /work/ckpt3 --save-path /work/ref-mp1

Produces <save-path>/model0-mp1.safetensors + tokenizer files, with the same tensor names and values
that `convert.py --model-parallel 1 --expert-dtype fp4` produces, except:

  * Tensors the reference model does not instantiate when vision is disabled are dropped, so a strict
    load works: vision.*, aligner.*, image_start/end/newline (model.py:1219-1225) and
    layers.*.ffn.gate.bias_vl (model.py:809, only built when vision_enabled). mtp.* is dropped too
    (no DSpark in the truncated model). convert.py would keep them and load_model(strict) would fail.
    Engram n-gram tables (layers.*.engram.embed.*) are dropped; ref_engram.py reads them from the HF shard.
  * Routed-expert FP4 weights are written as safetensors dtype "I8" (the same packed bytes as in the
    HF checkpoint) instead of torch.float4_e2m1fn_x2, because the F4 dtype string/shape convention of
    safetensors differs across versions. ref_runner.py views them as float4_e2m1fn_x2 on load
    (convert.py:191 does exactly `.view(torch.float4_e2m1fn_x2)` on the same bytes).
  * Streaming writer: peak RSS ~ the largest single tensor (head, 1.3 GB) instead of the whole model.

Kept from convert.py (lines 105-176): `model.` prefix strip, self_attn->attn, mlp->ffn (not under
vision.), weight_scale_inv->scale, e_score_correction_bias->bias; wo_a is dequantised to bf16 with its
block scale (convert.py:163-177) and wo_a.scale removed; all other tensors keep their bytes/dtype.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from st_io import ShardReader, StreamWriter, torch_to_bytes  # noqa: E402

INDEX_FILE = "model.safetensors.index.json"
# engram: only the n-gram tables (.engram.embed.*, ~98 GB) are dropped -- ref_engram.py serves them from an mmap of
# the HF shard; engram.{wkv.weight,wkv.scale,q_weight,k_weight} stay so the official Engram module loads strictly
DROP_RE = re.compile(r"^(vision\.|aligner\.|image_start$|image_end$|image_newline$|mtp\.)|\.ffn\.gate\.bias_vl$|\.engram\.embed\.")


def convert_name(name: str) -> str:
    """convert.py:105-127 for mp=1 (the `mapping` table is the identity on names for mp=1)."""
    if name.startswith("model."):
        name = name[len("model.") :]
    name = name.replace("self_attn", "attn")
    if not name.startswith("vision."):
        name = name.replace("mlp", "ffn")
    name = name.replace("weight_scale_inv", "scale")
    name = name.replace("e_score_correction_bias", "bias")
    return name


def is_routed_expert(name: str) -> bool:
    return "experts" in name and "shared_experts" not in name


def dequant_wo_a(weight, scale):
    """convert.py:163-177 (fp8 weight * block scale -> bf16)."""
    import torch

    from ref_kernels import scale_to_f32

    assert weight.size(0) % scale.size(0) == 0 and weight.size(1) % scale.size(1) == 0
    ob, ib = weight.size(0) // scale.size(0), weight.size(1) // scale.size(1)
    assert (ob, ib) in ((32, 32), (128, 128)), (weight.shape, scale.shape)
    w = weight.unflatten(0, (-1, ob)).unflatten(-1, (-1, ib)).float() * scale_to_f32(scale)[:, None, :, None]
    return w.flatten(2, 3).flatten(0, 1).to(torch.bfloat16)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-ckpt-path", default="/work/ckpt3")
    ap.add_argument("--save-path", default="/work/ref-mp1")
    ap.add_argument("--tokenizer-path", default=None)
    args = ap.parse_args()

    src = os.path.abspath(args.hf_ckpt_path)
    dst = os.path.abspath(args.save_path)
    os.makedirs(dst, exist_ok=True)
    index = json.load(open(os.path.join(src, INDEX_FILE)))["weight_map"]
    shards = sorted(set(index.values()))
    readers = {s: ShardReader(os.path.join(src, s)) for s in shards}
    for s, r in readers.items():  # convert.py:179-183 completeness check
        listed = {n for n, sh in index.items() if sh == s}
        if listed != set(r.keys()):
            raise SystemExit(f"{s}: shard contents differ from the index")

    plan = []  # (new_name, src_name, kind)
    dropped = []
    conv = {convert_name(n): n for n in index}
    for name in sorted(index):
        new = convert_name(name)
        if DROP_RE.search(new):
            dropped.append(new)
            continue
        if new.endswith("wo_a.scale"):
            continue  # folded into wo_a.weight
        if new.endswith("wo_a.weight"):
            sname = conv.get(new[: -len("weight")] + "scale")
            if sname is None:
                raise SystemExit(f"{new}: no block scale found")
            plan.append((new, (name, sname), "wo_a"))
        else:
            plan.append((new, name, "copy"))
    if len({p[0] for p in plan}) != len(plan):
        raise SystemExit("name collision after conversion")

    entries = []
    for new, srcn, kind in plan:
        if kind == "wo_a":
            dt, sh, _ = readers[index[srcn[0]]].info(srcn[0])
            assert dt == "F8_E4M3", (new, dt)
            entries.append((new, "BF16", sh))
        else:
            dt, sh, _ = readers[index[srcn]].info(srcn)
            if is_routed_expert(new) and new.endswith(".weight") and dt != "I8":
                raise SystemExit(f"{new}: expected I8-packed FP4 expert weight, got {dt}")
            entries.append((new, dt, sh))

    out = os.path.join(dst, "model0-mp1.safetensors")
    meta = {"format": "pt", "producer": "tools/ref/ref_convert.py", "fp4_experts_stored_as": "I8"}
    w = StreamWriter(out, entries, metadata=meta)
    try:
        for i, (new, srcn, kind) in enumerate(plan):
            if kind == "wo_a":
                wt = readers[index[srcn[0]]].read_torch(srcn[0])
                sc = readers[index[srcn[1]]].read_torch(srcn[1])
                w.write_tensor(new, [torch_to_bytes(dequant_wo_a(wt, sc))])
            else:
                w.write_tensor(new, readers[index[srcn]].iter_bytes(srcn))
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(plan)}", flush=True)
        w.close()
    except BaseException:
        w.abort()
        raise
    for r in readers.values():
        r.close()

    tok = args.tokenizer_path or src
    for f in ("tokenizer.json", "tokenizer_config.json"):
        if os.path.exists(os.path.join(tok, f)):
            shutil.copyfile(os.path.join(tok, f), os.path.join(dst, f))
    with open(os.path.join(dst, "ref_convert.json"), "w") as f:
        json.dump({"source": src, "n_tensors": len(entries), "dropped": dropped}, f, indent=1)
    print(f"wrote {out}: {len(entries)} tensors; dropped {len(dropped)} (vision/aligner/image_*/bias_vl/mtp/engram tables)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
