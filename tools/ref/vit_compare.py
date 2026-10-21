#!/usr/bin/env python3
"""Vision encoder check: DeepInfer's ViT + aligner on SM120 against DeepSeek's reference vision.py, stage by stage.

    python3 /opt/sm120/tools/ref/vit_compare.py --ckpt /work/ckpt3_vision --ref-src /ds41ref/inference \\
        [--images a.png b.jpg ...] [--json /work/vit_compare.json]

DeepInfer side: the server's own path in-process (server/engine.py EngineLoop + server/images.py: image bytes and
sizes into add_request, DeepInfer's worker preprocessing, rgb_pad_to_patches, ViT, aligner). Hooks capture the patches
the ViT receives, the patch embedding, the outputs of blocks 0 / 15 / 31, the ViT output and the aligner output.
Reference side: vision.py's ViT and Aligner (unmodified) with the checkpoint's vision.* / aligner.* tensors, fed the
SAME patches, once in fp32 (ground truth) and once in bf16 (the noise floor of a bf16 implementation).
Per stage: rel-L2 = |a - b| / |b| and cosine, DeepInfer vs fp32 reference and bf16 reference vs fp32 reference.
DeepInfer's preprocessing (resize, pad, normalise) is compared against an independent Pillow implementation (no
reference preprocessing is published); the resize filters differ, so that number is only a sanity check.
The checkpoint needs real vision weights (shard model-00001; make_truncated_ckpt.py writes zeros without it).
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import math
import os
import sys
import time
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "server"))

STAGES = ["patch_embed", "blocks.0", "blocks.15", "blocks.31", "vit_out", "aligner_out"]


def test_images():
    """Varied synthetic pictures (structure, colour, text, noise) in the sizes that matter for the layout:
    upscaled small, square, landscape, wide, portrait, larger than the 1024-token cap."""
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    out = []
    im = Image.new("RGB", (448, 448), "white")
    d = ImageDraw.Draw(im)
    for i in range(8):
        d.ellipse([20 + i * 50, 40 + (i % 3) * 120, 90 + i * 50, 110 + (i % 3) * 120],
                  fill=(30 * i, 255 - 30 * i, (90 * i) % 255))
    out.append(("shapes_448", im))
    g = Image.linear_gradient("L").resize((1344, 896))
    out.append(("gradient_1344x896", Image.merge("RGB", (g, g.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
                                                        g.rotate(90).resize((1344, 896))))))
    im = Image.new("RGB", (1600, 600), (250, 245, 230))
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default(size=90)
    except TypeError:
        font = ImageFont.load_default()
    d.text((40, 200), "DeepInfer on SM120 4862", fill=(10, 10, 10), font=font)
    out.append(("text_1600x600", im))
    noise = Image.effect_noise((300, 500), 64).convert("RGB").filter(ImageFilter.GaussianBlur(2))
    out.append(("noise_300x500", noise))
    big = Image.radial_gradient("L").resize((3000, 2000))
    out.append(("radial_3000x2000", Image.merge("RGB", (big, big.rotate(180), big.transpose(Image.Transpose.FLIP_TOP_BOTTOM)))))
    return out


def png_bytes(im) -> bytes:
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def metrics(a, b):
    import torch

    a, b = a.double().flatten(), b.double().flatten()
    rel = (a - b).norm() / b.norm().clamp_min(1e-30)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0)
    return float(rel), float(cos)


def capture_deepinfer(ckpt: str, pictures, args):
    """Run each picture through the server's engine path; return per-picture stage tensors (CPU fp32)."""
    import torch

    import images as img_mod
    from engine import EngineLoop, GenRequest
    from server import TextCodec

    caps: list[dict] = []
    cur: dict = {}

    def keep(name, t):
        if not torch.is_tensor(t):  # ViT Block returns (y, residual): the residual add is deferred to the next norm
            t = t[0] if len(t) < 2 or t[1] is None else t[0].float() + t[1].float()
        cur[name] = t.detach().float().cpu()

    import deepinfer.model_executor.layers.tilelang_ops.image_embed_ops as ieo
    import deepinfer.model_executor.models.vision_encoder.vl_aligner as vla

    o_pad = ieo.rgb_pad_to_patches

    def pad(*a, **k):
        out = o_pad(*a, **k)
        keep("patches", out)
        return out

    ieo.rgb_pad_to_patches = pad
    o_enc = vla.VisionLanguageInstructTuner._encode_images

    def enc(self, images, *a, **k):
        ve = self.visual_encoder
        hooks = [ve.register_forward_hook(lambda m, i, o: keep("vit_out", o))]
        for n, m in ve.named_modules():
            if n in ("patch_embed", "blocks.0", "blocks.15", "blocks.31"):
                hooks.append(m.register_forward_hook(lambda m, i, o, n=n: keep(n, o)))
        try:
            out = o_enc(self, images, *a, **k)
        finally:
            for h in hooks:
                h.remove()
        keep("aligner_out", out)
        return out

    vla.VisionLanguageInstructTuner._encode_images = enc

    loop = EngineLoop(ckpt, max_num_seqs=1, gpu_memory_utilization=args.gpu_memory_utilization)  # server defaults
    loop.start()
    codec = TextCodec(ckpt)
    lim = img_mod.ImageLimits(max_pixels=100_000_000)
    try:
        for name, im in pictures:
            data = png_bytes(im)
            url = "data:image/png;base64," + __import__("base64").b64encode(data).decode()
            prompt, recs = codec.render_chat([{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": "Describe."}]}], False)
            mm = img_mod.load_images(recs, lim)
            size = mm["image_sizes"][0]
            cur.clear()
            req = loop.submit(GenRequest(prompt_ids=codec.encode(prompt), max_tokens=1, temperature=0.0,
                                         multi_modal_data=mm))
            while True:
                msg = req.out.get(timeout=600)
                if msg[0] == "error":
                    raise RuntimeError(msg[1])
                if msg[2]:
                    break
            caps.append(dict(name=name, size=size, image=im, **cur))
            print(f"captured {name} {size}: " + ", ".join(f"{k}{tuple(v.shape)}" for k, v in cur.items()), flush=True)
        n_tok = {c["name"]: loop.image_tokens(*c["size"]) for c in caps}
    finally:
        loop.shutdown()
    return caps, n_tok


def reference_models(ckpt: str, ref_src: str):
    import torch
    from safetensors import safe_open

    spec = importlib.util.spec_from_file_location("ds41_vision", os.path.join(ref_src, "vision.py"))
    V = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(V)
    cfg = json.load(open(os.path.join(ckpt, "config.json")))
    vc, tc = cfg["vision_config"], cfg.get("text_config", cfg)
    a = SimpleNamespace(vision_dim=vc["hidden_size"], vision_n_heads=vc["num_attention_heads"],
                        vision_inter_dim=vc["intermediate_size"], vision_patch_size=vc["patch_size"],
                        vision_rope_theta=float(vc.get("rope_theta", 10000)), vision_n_layers=vc["num_hidden_layers"],
                        vision_downsample_ratio=vc["downsample_ratio"], dim=tc["hidden_size"])
    idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    sd = {}
    for k, f in idx.items():
        if k.startswith(("vision.", "aligner.")):
            with safe_open(os.path.join(ckpt, f), "pt") as fh:
                sd[k] = fh.get_tensor(k)
    if all(float(sd[k].float().abs().max()) == 0 for k in sd if k.endswith("proj.weight")):
        raise SystemExit("the checkpoint's vision weights are zeros (built without shard model-00001)")
    models = {}
    for dt in (torch.float32, torch.bfloat16):
        vit, al = V.ViT(a), V.Aligner(a)
        vit.load_state_dict({k[len("vision."):]: v for k, v in sd.items() if k.startswith("vision.")}, strict=True)
        al.load_state_dict({k[len("aligner."):]: v for k, v in sd.items() if k.startswith("aligner.")}, strict=True)
        models[dt] = (vit.to("cuda", dt).eval(), al.to("cuda", dt).eval())
    return models, a


def reference_run(models, dt, patches, n_h, n_w):
    import torch

    vit, al = models[dt]
    st = {}
    hooks = [vit.patch_embed.register_forward_hook(lambda m, i, o: st.__setitem__("patch_embed", o.float().cpu()))]
    for i in (0, 15, 31):
        hooks.append(vit.blocks[i].register_forward_hook(lambda m, i_, o, i=i: st.__setitem__(f"blocks.{i}", o.float().cpu())))
    with torch.no_grad(), torch.device("cuda"):  # the reference builds its RoPE tables on the default device
        x = vit(patches.to("cuda", dt), n_h, n_w)
        st["vit_out"] = x.float().cpu()
        st["aligner_out"] = al(x, n_h, n_w).float().cpu()
    for h in hooks:
        h.remove()
    return st


def independent_patches(im, best_w, best_h, patch, pad_color_rgb=None):
    """Pillow resize to the layout DeepInfer chose, normalise (x/255 - 0.5) / 0.5, cut into patches (row-major)."""
    import numpy as np
    import torch

    r = im.convert("RGB").resize((best_w, best_h), resample=3)  # bicubic
    x = torch.from_numpy(np.asarray(r).copy()).float().permute(2, 0, 1) / 255.0
    x = (x - 0.5) / 0.5
    c, h, w = x.shape
    x = x.view(c, h // patch, patch, w // patch, patch).permute(1, 3, 0, 2, 4).reshape(-1, c, patch, patch)
    return x


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ref-src", default=os.environ.get("REF_SRC", "/ds41ref/inference"))
    ap.add_argument("--images", nargs="*", default=None, help="image files (default: built-in synthetic set)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    ap.add_argument("--json", default=None)
    ap.add_argument("--caps", default=None, help="save DeepInfer's captured stage tensors here; reused if it exists")
    args = ap.parse_args()
    import torch
    from PIL import Image

    pictures = [(os.path.basename(p), Image.open(p)) for p in args.images] if args.images else test_images()
    t0 = time.time()
    if args.caps and os.path.exists(args.caps):
        caps, n_tok = torch.load(args.caps, weights_only=False)
        print(f"reusing DeepInfer captures from {args.caps}")
    else:
        caps, n_tok = capture_deepinfer(args.ckpt, pictures, args)
        if args.caps:
            torch.save((caps, n_tok), args.caps)
    torch.cuda.empty_cache()
    models, a = reference_models(args.ckpt, args.ref_src)
    from deepinfer.ve.processor.image_processor import compute_image_layout

    cfg = json.load(open(os.path.join(args.ckpt, "config.json")))["vision_config"]
    rows = []
    for c in caps:
        lay = compute_image_layout(c["size"][0], c["size"][1], patch_size=cfg["patch_size"],
                                   downsample_ratio=cfg["downsample_ratio"], max_n_token=cfg["max_image_tokens"],
                                   min_pixels=cfg["min_pixels"], max_wh_ratio=cfg["max_wh_ratio"])
        bw, bh = lay.best_width, lay.best_height
        n_w, n_h = lay.n_vit_w, lay.n_vit_h
        patches = c["patches"]
        assert patches.shape[0] == n_h * n_w, (patches.shape, n_h, n_w)
        ref32 = reference_run(models, torch.float32, patches, n_h, n_w)
        ref16 = reference_run(models, torch.bfloat16, patches, n_h, n_w)
        row = dict(name=c["name"], size=c["size"], grid=[n_h, n_w], image_tokens=n_tok[c["name"]], stages={})
        for s in STAGES:
            if s not in c:
                continue
            di, r32, r16 = c[s].reshape(ref32[s].shape), ref32[s], ref16[s]
            rel, cos = metrics(di, r32)
            rel16, cos16 = metrics(r16, r32)
            row["stages"][s] = dict(di_rel=rel, di_cos=cos, bf16_rel=rel16, bf16_cos=cos16,
                                    ratio=rel / rel16 if rel16 > 0 else math.inf)
        ind = independent_patches(c["image"], bw, bh, cfg["patch_size"])
        if ind.shape == patches.shape:
            row["preprocess_vs_pillow"] = dict(zip(("rel", "cos"), metrics(patches, ind)))
        rows.append(row)
        print(f"\n{c['name']} {c['size'][0]}x{c['size'][1]} -> grid {n_h}x{n_w}, {n_tok[c['name']]} tokens"
              + (f", preprocessing vs Pillow: cos {row['preprocess_vs_pillow']['cos']:.4f}" if 'preprocess_vs_pillow' in row else ""))
        print(f"  {'stage':12s} {'DI rel-L2':>10s} {'DI cos':>9s} {'bf16 rel':>10s} {'bf16 cos':>9s} {'DI/bf16':>8s}")
        for s, m in row["stages"].items():
            print(f"  {s:12s} {m['di_rel']:10.2e} {m['di_cos']:9.6f} {m['bf16_rel']:10.2e} {m['bf16_cos']:9.6f} {m['ratio']:8.2f}")
    print(f"\n{len(rows)} images in {time.time() - t0:.0f}s")
    if args.json:
        json.dump(dict(ckpt=args.ckpt, rows=rows), open(args.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
