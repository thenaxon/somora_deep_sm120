"""Logit / hidden-state capture for tools/run_deepinfer.py (in-process, driver worker, eager mode).

* Sampler.forward / sampler.get_logits are wrapped by sm120_patches (always installed, inert until
  ``arm()`` is called), so profile/warm-up runs are never recorded.
* Per-layer prefill hidden states use forward hooks on the model instance (``install_model_hooks``),
  stored in the reference format of tools/ref/ref_runner.py:
      hidden["prefill"]["embed_hc"]   [1, L, hc_mult, dim] bf16  (block-0 input; DeepInfer's embedding output
                                       broadcast over hc_mult, which is what expand_to_mhc does)
      hidden["prefill"]["layer{i}"]   [1, L, hc_mult, dim] bf16  (block output after the deferred mHC post)
      hidden["prefill"]["final_norm"] [1, L, dim] bf16           (output_norm output = lm-head input)
  DeepInfer defers each block's mHC "post" into the next block (HCLazyPostData), so the hook applies
  ``HCLayerMixin.hc_post`` (pure function, does not consume the lazy data) to materialise the block output.
* Fine per-layer taps (env CAPTURE_LAYER, default 0): hidden["prefill"]["L<i>/<name>"], see
  ``install_layer_captures`` and the block comment above it; compared by tools/compare_l0.py.
"""

from __future__ import annotations

import torch

STATE: dict = {}


def reset() -> None:
    STATE.clear()
    STATE.update(
        armed=False,
        in_prompt=False,
        prompt_len=None,
        keep_full_prefill=True,
        prefill=None,  # dict of prefill logit products
        prefill_sampled=[],  # logits DeepInfer itself sampled from at prefill (last position)
        decode=[],  # list of [rows, V] fp32 cpu logits, one per decode forward of real requests
        decode_meta=[],  # per decode entry: request ids, seqlen_q (rows per request), argmax per row
        skip_step=False,  # current sampler call belongs to a placeholder step (ZOS / empty schedule)
        step_requests=None,  # (request_id, seqlen_q) of the current sampler call
        n_placeholder_skipped=0,
        hidden={},
        hidden_rows=None,
        errors=[],
        fine_active=None,  # layer index while that layer runs inside the first armed prompt forward
        fine_done=False,
        attn_ctx=False,  # True while the captured layer's V4Attention.forward runs (gates sm120_attn taps)
    )


reset()


def arm(prompt_len: int, keep_full_prefill: bool = True) -> None:
    reset()
    STATE["armed"] = True
    STATE["prompt_len"] = int(prompt_len)
    STATE["keep_full_prefill"] = bool(keep_full_prefill)


def disarm() -> None:
    STATE["armed"] = False


def _val(x):
    v = getattr(x, "_value", None)  # deepinfer.model_executor.utils.Movable (peek, do not move)
    return v if v is not None else x


def _logits_all(hidden: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
    """fp32 logits for every row, exactly like sampler_utils.get_logits (bf16 x bf16 -> fp32 GEMM)."""
    from deep_gemm import bf16_gemm_nt

    rows = hidden.shape[0]
    out = torch.empty((rows, embedding.shape[0]), dtype=torch.float32, device=hidden.device)
    step = 64
    for s in range(0, rows, step):
        e = min(rows, s + step)
        bf16_gemm_nt(hidden[s:e].contiguous(), embedding, out[s:e])
    return out


def _is_placeholder(seq) -> bool:
    flag = getattr(seq, "is_placeholder", None)
    if flag is not None:
        return bool(flag)
    return getattr(seq, "request_id", None) == "placeholder"  # PLACEHOLDER_REQUEST_ID (csrc/engine/sequence.hpp)


def on_sampler_forward(embedding, hidden_states, main_input_metadata, seq_metadata_list=None) -> None:
    if not STATE.get("armed"):
        return
    is_prompt = bool(getattr(main_input_metadata, "is_prompt", False))
    STATE["in_prompt"] = is_prompt
    # Worker.execute_model_step runs a dummy "placeholder" sequence (token 0 at position 0) through forward
    # and sampler whenever the scheduler output is empty (worker.py:126, e.g. while a ZOS decode step is in
    # flight). Its logits are constant; recording them as a decode step was the wrong-row bug (argmax 84562).
    seqs = list(seq_metadata_list) if seq_metadata_list is not None else []
    STATE["skip_step"] = bool(seqs) and all(_is_placeholder(s) for s in seqs)
    STATE["step_requests"] = [(str(getattr(s, "request_id", "?")), int(getattr(s, "seqlen_q", 1) or 1))
                              for s in seqs if not _is_placeholder(s)] or None
    if not is_prompt or STATE["prefill"] is not None or STATE["skip_step"]:
        return
    try:
        hs = _val(hidden_states)
        L = STATE["prompt_len"] or hs.shape[0]
        if hs.shape[0] < L:
            STATE["errors"].append(f"prefill hidden has {hs.shape[0]} rows < prompt_len {L} (chunked prefill?)")
            L = hs.shape[0]
        logits = _logits_all(hs[:L], embedding)
        top = logits.topk(20, dim=-1)
        STATE["prefill"] = dict(
            prefill_top20_values=top.values.cpu(),
            prefill_top20_ids=top.indices.cpu(),
            prefill_argmax=logits.argmax(-1).cpu(),
            prefill_logits_last64=logits[-64:].cpu().clone(),
            prefill_logits=logits.cpu() if STATE["keep_full_prefill"] else None,
            prefill_finite=bool(torch.isfinite(logits).all()),
            prefill_rows=int(hs.shape[0]),
        )
        del logits
    except Exception as exc:  # noqa: BLE001
        STATE["errors"].append(f"prefill logits capture failed: {type(exc).__name__}: {exc}")


def on_get_logits(logits: torch.Tensor) -> None:
    """Sampler's own logits: one row per selected token (decode: seqlen_q rows per request, i.e. 1 + the
    draft tokens under MTP, requests in seq_metadata_list order; CUDA-graph padding is already trimmed)."""
    if not STATE.get("armed"):
        return
    if STATE.get("skip_step"):
        STATE["n_placeholder_skipped"] += 1
        return
    t = logits.detach().float().cpu().clone()
    if STATE["in_prompt"]:
        STATE["prefill_sampled"].append(t)
        return
    reqs = STATE.get("step_requests")
    if reqs is not None and sum(n for _, n in reqs) != t.shape[0]:
        STATE["errors"].append(f"decode capture: {t.shape[0]} logit rows but requests {reqs}")
    STATE["decode"].append(t)
    STATE["decode_meta"].append(dict(requests=reqs, argmax=t.argmax(-1).tolist()))


def decode_rows(request_id: str | None = None) -> list[torch.Tensor]:
    """Per decode step, the logit rows [seqlen_q, V] of one request (default: the first real request)."""
    rows = []
    for t, meta in zip(STATE["decode"], STATE["decode_meta"]):
        reqs = meta["requests"] or [(None, t.shape[0])]
        off = 0
        for rid, n in reqs:
            if request_id is None or rid == request_id:
                rows.append(t[off:off + n])
                break
            off += n
    return rows


# ------------------------------------------------------------------------------------------------ hidden
def install_model_hooks(model) -> list:
    """model: DeepseekForCausalLM (llm.llm_engine.driver_worker.model_runner.model)."""
    from deepinfer.model_executor.models.v4_modules.hc import HCLayerMixin

    inner = model.model
    layers = list(inner.layers)
    hooks = []

    def store(key, t: torch.Tensor, hc: bool) -> None:
        if not STATE.get("armed"):
            return
        hid = STATE["hidden"]
        if key in hid:  # prefill is the first armed forward; later (decode) forwards are ignored
            return
        L = STATE["prompt_len"] or t.shape[0]
        if t.shape[0] < L:
            if t.shape[0] == 1 and L > 1:
                # placeholder forward (dummy seq, 1 row; worker.py:126), e.g. before the prefill: not an error
                STATE["n_placeholder_forwards"] = STATE.get("n_placeholder_forwards", 0) + 1
                return
            STATE["errors"].append(f"hidden {key}: {t.shape[0]} rows < prompt_len {L}")
            return
        t = t[:L].detach()
        hid[key] = t.unsqueeze(0).to(torch.bfloat16).cpu().clone()

    def pre0(mod, args):
        if not STATE.get("armed") or "embed_hc" in STATE["hidden"]:
            return
        try:
            h = _val(args[0])
            hc_mult = getattr(mod, "hc_mult", None) or getattr(model.cfg, "hc_mult", 1)
            if h.dim() == 2:
                h = h.unsqueeze(1).expand(h.shape[0], hc_mult, h.shape[1])
            store("embed_hc", h, True)
        except Exception as exc:  # noqa: BLE001
            STATE["errors"].append(f"embed_hc capture failed: {type(exc).__name__}: {exc}")

    hooks.append(layers[0].register_forward_pre_hook(pre0))

    for i, layer in enumerate(layers):
        def post(mod, args, out, i=i):
            if not STATE.get("armed") or f"layer{i}" in STATE["hidden"]:
                return
            try:
                h, res = (out[0], out[1]) if isinstance(out, (tuple, list)) else (out, None)
                h = _val(h)
                if h.dim() == 2:
                    if res is None or getattr(res, "residual", None) is None:
                        raise RuntimeError("2D block output without lazy mHC post data")
                    h = HCLayerMixin.hc_post(h, res)
                store(f"layer{i}", h, True)
            except Exception as exc:  # noqa: BLE001
                STATE["errors"].append(f"layer{i} capture failed: {type(exc).__name__}: {exc}")

        hooks.append(layer.register_forward_hook(post))

    def norm_hook(mod, args, out):
        if not STATE.get("armed") or "final_norm" in STATE["hidden"]:
            return
        try:
            store("final_norm", _val(out[0] if isinstance(out, (tuple, list)) else out), False)
        except Exception as exc:  # noqa: BLE001
            STATE["errors"].append(f"final_norm capture failed: {type(exc).__name__}: {exc}")

    hooks.append(inner.output_norm.register_forward_hook(norm_hook))
    li = capture_layer()
    if li is not None:
        if 0 <= li < len(layers):
            hooks += install_layer_captures(layers[li], li)
        else:
            STATE["errors"].append(f"CAPTURE_LAYER={li} out of range (model has {len(layers)} layers)")
    return hooks


# ------------------------------------------------------------------------------------------------ fine layer taps
# CAPTURE_LAYER=<i> (default 0; "-1"/"none"/"off" disables) records the intermediates of decoder layer i
# during the FIRST armed prompt forward into hidden["prefill"]["L<i>/<name>"], CPU, prompt order, no batch dim
# (single request, no CP: DeepInfer's prefill rows ARE prompt order; "L<i>/positions" is stored to prove it).
#
# DeepInfer specifics and how each tensor is reconstructed (see tools/compare_l0.py for the reference side):
#   * mHC is "shifted" (hc.py:340-358, V4.1 hc_shifted=True): sub-layer k's input applies the pre-mix computed by
#     sub-layer k-1; layer 0's attention uses a one-hot mix (hc.py:45-49, == model.py make_identity_pre_mix).
#     The previous sub-layer's post is deferred into this sub-layer's attn_norm_forward/ffn_norm_forward
#     (_hc_fused_post_pre, hc.py:174-227), so those two calls are where the trunk and the mixes appear:
#       hc_trunk_in  [L,4,D] bf16  trunk entering the attention branch     (= lazy.residual of attn_norm_forward)
#       hc_trunk_mid [L,4,D] bf16  trunk after the attention hc_post        (= lazy.residual of ffn_norm_forward)
#       attn_pre_mix_in / ffn_pre_mix_in [L,4] fp32  mix APPLIED to form the branch input (prev sub-layer's pre)
#       hc_pre_attn / hc_pre_ffn [L,D] bf16  branch input before the RMSNorm = mhc_pre_apply_mix(trunk, mix_in)
#                    (recomputed with DeepInfer's own kernel: pure function, the fused path never materialises it)
#       attn_in / ffn_in [L,D] bf16  RMSNorm output (hidden_states_bf16; the fp8 copy fed to the GEMMs is not kept)
#       attn_mix_pre/post [L,4], attn_mix_comb [L,4,4] fp32: coefficients computed FROM hc_trunk_in with the
#                    attention hc params (pre is used by the FFN input, post/comb by the attention hc_post);
#       ffn_mix_*    same, computed from hc_trunk_mid with the ffn params (pre feeds the next layer).
#   * attention (v4_modules/attn.py V4Attention.forward:758-873):
#       q_a          [L,q_lora] bf16  wq_a output, BEFORE q_a_norm (split of the fused q_a/kv_swa GEMM)
#       kv_proj      [L,hd]   bf16    wkv output before kv_swa_norm
#       kv           [L,hd]   bf16    after kv norm + RoPE + FP8(e4m3, 1x32 ue8m0) cast-back (_prepare_swa_kv)
#       q_pre_rope   [L,H,hd] bf16    q_b_proj output in natural [h,d] layout (fused path: un-permuted)
#       q_post_rope  [L,H,hd] bf16    after (optional q-norm in the non-fused path) + RoPE; fused path: q_r of
#                                      sm120_attn.fused._prepare_q (q-norm, if enabled, is a logit scale there:
#                                      q_rms_scale [L,H] is then stored too)
#       attn_core_pre_unrope [L,H,hd] attention output before the inverse RoPE (fused: fp32, non-fused: bf16)
#       attn_core_out        [L,H,hd] after the inverse RoPE, before the FP8 cast / wo_a einsum (fused: fp32)
#       swa_idx      [L,W] int32  SWA key positions per query (kv_swa_full_index minus the history offset
#                                 total_k_swa; -1 masked, -2 = points into the (empty) swa_cache history)
#       rope_cos / rope_sin [L,rope_dim/2] fp32 rotary_emb.cos_sin_cache rows at the token positions
#       attn_out     [L,D] bf16   V4Attention output (after wo_a/wo_b)
#   * MoE (mlp.py DeepseekMoE + sm120/sm120_moe.py forward):
#       gate_idx [L,k] int64, gate_w [L,k] fp32  moe.gate_forward (after bias selection, norm, route scale)
#       routed_out [L,D]  routed sum (torch backend fp32, b12x bf16; absent for the deepep/mega paths)
#       shared_out [L,D] bf16 shared expert; ffn_out [L,D] bf16 DeepseekMoE output (routed + shared)
#   * scalars: hidden["prefill"]["L<i>/meta"] (dict) with eps / window / rope / flags actually in effect.
import os as _os

_SKIP = object()


def capture_layer():
    v = _os.environ.get("CAPTURE_LAYER", "0").strip().lower()
    if v in ("", "none", "off", "no", "-1"):
        return None
    return int(v)


def _fine_key(name: str) -> str | None:
    li = STATE.get("fine_active")
    return None if li is None else f"L{li}/{name}"


def _fine_store(name: str, t, rows: bool = True) -> None:
    """Store a CPU copy (dtype kept). rows=True: dim 0 is the token dim, sliced to prompt_len."""
    if not STATE.get("armed"):
        return
    key = _fine_key(name)
    if key is None or key in STATE["hidden"]:
        return
    try:
        t = _val(t)
        if not isinstance(t, torch.Tensor):
            return
        t = t.detach()
        if rows:
            L = STATE["prompt_len"] or t.shape[0]
            if t.shape[0] < L:
                STATE["errors"].append(f"{key}: {t.shape[0]} rows < prompt_len {L}")
                return
            t = t[:L]
        STATE["hidden"][key] = t.to("cpu", copy=True)
    except Exception as exc:  # noqa: BLE001
        STATE["errors"].append(f"{key} capture failed: {type(exc).__name__}: {exc}")


def _fine_meta(**kw) -> None:
    key = _fine_key("meta")
    if key is None:
        return
    STATE["hidden"].setdefault(key, {}).update(kw)


def _fine_err(what: str, exc: Exception) -> None:
    STATE["errors"].append(f"L{STATE.get('fine_active')}/{what}: {type(exc).__name__}: {exc}")


class _Restore:
    """hook-like handle: .remove() undoes an attribute override."""

    def __init__(self, obj, name, had_instance_attr, old):
        self.obj, self.name, self.had, self.old = obj, name, had_instance_attr, old

    def remove(self):
        if self.had:
            setattr(self.obj, self.name, self.old)
        else:
            try:
                delattr(self.obj, self.name)
            except AttributeError:
                pass


def _override(obj, name, make_wrapper) -> _Restore:
    had = name in getattr(obj, "__dict__", {})
    old = obj.__dict__[name] if had else None
    orig = getattr(obj, name)
    setattr(obj, name, make_wrapper(orig))
    return _Restore(obj, name, had, old)


def _mix2d(m, hc: int):
    return None if m is None else m.reshape(-1, hc)


def install_layer_captures(layer, idx: int) -> list:
    """Fine prefill taps on one DeepseekDecoderLayer (see the block comment above for semantics)."""
    from deepinfer.model_executor.models.v4_modules import hc as hc_mod
    from deepinfer.model_executor.models.v4_modules.hc_lazy import HCLazyPostData

    handles: list = []
    hc_mult = int(getattr(layer, "hc_mult", 1))
    attn = layer.self_attn
    mlp = layer.mlp
    try:
        import sm120_attn.fused as sfused
    except Exception:  # noqa: BLE001
        sfused = None

    # ---- layer activation: only inside the first armed prompt forward
    def lpre(mod, args):
        STATE["fine_active"] = None
        STATE["attn_ctx"] = False
        if not STATE.get("armed") or STATE.get("fine_done"):
            return
        md = args[2] if len(args) > 2 else None
        if md is None or not bool(getattr(md, "is_prompt", False)):
            return
        STATE["fine_active"] = idx
        try:
            pos = getattr(md, "positions", None)
            if pos is not None:
                _fine_store("positions", pos)
        except Exception as exc:  # noqa: BLE001
            _fine_err("positions", exc)

    def lpost(mod, args, out):
        if STATE.get("fine_active") == idx:
            STATE["fine_done"] = True
        STATE["fine_active"] = None
        STATE["attn_ctx"] = False

    handles.append(layer.register_forward_pre_hook(lpre))
    handles.append(layer.register_forward_hook(lpost))

    def active() -> bool:
        return STATE.get("armed", False) and STATE.get("fine_active") == idx

    # ---- mHC + norms: attn_norm_forward / ffn_norm_forward
    def norm_wrapper(kind: str):
        def make(orig):
            def wrapper(*a, **kw):
                if not active():
                    return orig(*a, **kw)
                prev_mix, first = None, False
                try:
                    hs = _val(a[0]) if a else _val(kw.get("hidden_states"))
                    prev = a[1] if len(a) > 1 else kw.get("residual")
                    if isinstance(prev, HCLazyPostData) and prev.pre_layer_mix is not None:
                        prev_mix = prev.pre_layer_mix.detach().clone()
                    first = prev is None and isinstance(hs, torch.Tensor) and hs.dim() == 2
                except Exception as exc:  # noqa: BLE001
                    _fine_err(f"{kind}_norm_forward pre", exc)
                out = orig(*a, **kw)
                try:
                    _, normed_bf16, lazy = out
                    trunk = lazy.residual  # [1, n, hc, D] (None only on the DFlash early-post path)
                    _fine_store(f"{kind}_in", normed_bf16)
                    if trunk is not None:
                        _fine_store("hc_trunk_in" if kind == "attn" else "hc_trunk_mid", trunk.squeeze(0))
                    mix_in = prev_mix
                    if mix_in is None and first and getattr(layer, "hc_shifted", False) and trunk is not None:
                        mix_in = hc_mod._make_identity_pre_mix(trunk[:, :, 0], hc_mult)  # [1, n, hc, 1]
                    if mix_in is not None and trunk is not None:
                        _fine_store(f"{kind}_pre_mix_in", _mix2d(mix_in, hc_mult))
                        try:
                            x_in = hc_mod.mhc_pre_apply_mix(trunk.contiguous(), mix_in.float().contiguous())
                        except Exception as exc:  # noqa: BLE001
                            _fine_err(f"hc_pre_{kind} kernel (torch fallback used)", exc)
                            x_in = (trunk.float() * mix_in.float()).sum(-2).to(torch.bfloat16)
                        _fine_store(f"hc_pre_{kind}", x_in.reshape(-1, x_in.shape[-1]))
                    elif not getattr(layer, "hc_shifted", False):
                        STATE["errors"].append(f"L{idx}: hc_shifted=False: hc_pre_{kind} not reconstructable")
                    _fine_store(f"{kind}_mix_pre", _mix2d(lazy.pre_layer_mix, hc_mult))
                    _fine_store(f"{kind}_mix_post", _mix2d(lazy.post_layer_mix, hc_mult))
                    if lazy.comb_res_mix is not None:
                        _fine_store(f"{kind}_mix_comb", lazy.comb_res_mix.reshape(-1, hc_mult, hc_mult))
                except Exception as exc:  # noqa: BLE001
                    _fine_err(f"{kind}_norm_forward post", exc)
                return out

            return wrapper

        return make

    handles.append(_override(layer, "attn_norm_forward", norm_wrapper("attn")))
    handles.append(_override(layer, "ffn_norm_forward", norm_wrapper("ffn")))

    # ---- attention
    fused = bool(getattr(attn, "use_fused_norm_rope_attn_rope_cast", False))

    def apre(mod, args, kwargs):
        STATE["attn_ctx"] = active()

    def apost(mod, args, kwargs, out):
        if STATE.get("attn_ctx"):
            try:
                _fine_store("attn_out", out)
                md = kwargs.get("input_metadata") if kwargs else None
                if md is None and len(args) > 2:
                    md = args[2]
                if md is not None and getattr(md, "kv_swa_full_index", None) is not None:
                    xi = md.kv_swa_full_index.long()
                    off = int(getattr(md, "total_k_swa", 0) or 0)
                    rel = torch.where(xi < 0, torch.full_like(xi, -1),
                                      torch.where(xi < off, torch.full_like(xi, -2), xi - off))
                    _fine_store("swa_idx", rel.to(torch.int32))
                    _fine_meta(swa_total_k_cached=off)
                rot = getattr(attn, "rotary_emb", None)
                pos = getattr(md, "positions", None) if md is not None else None
                if rot is not None and pos is not None and getattr(rot, "cos_sin_cache", None) is not None:
                    cs = rot.cos_sin_cache.float()[pos.long()]
                    half = cs.shape[-1] // 2
                    _fine_store("rope_cos", cs[:, :half])
                    _fine_store("rope_sin", cs[:, half:])
                _fine_store("attn_sink", attn.attn_sink_params, rows=False)
                _fine_meta(
                    fused_norm_rope_attn=fused,
                    use_q_b_norm=bool(getattr(attn, "use_q_b_norm", False)),
                    q_b_norm_eps=getattr(getattr(attn, "q_b_norm", None), "variance_epsilon", None),
                    q_a_norm_eps=getattr(getattr(attn, "q_a_norm", None), "variance_epsilon", None),
                    kv_norm_eps=getattr(getattr(attn, "kv_swa_norm", None), "variance_epsilon", None),
                    window_size=getattr(attn, "window_size", None),
                    softmax_scale=getattr(attn, "softmax_scale", None),
                    kv_compress_ratio=getattr(attn, "kv_compress_ratio", None),
                    rope_base=getattr(rot, "base", None),
                    rope_cls=type(rot).__name__ if rot is not None else None,
                    rope_is_neox=getattr(rot, "is_neox_style", None),
                    rope_dim=getattr(attn, "qk_rope_head_dim", None),
                    use_fp8_einsum_output=getattr(attn, "use_fp8_einsum_output", None),
                    einsum_out_clamp_value=getattr(attn, "einsum_out_clamp_value", None),
                    swa_cache_num_per_channels=getattr(attn, "swa_cache_num_per_channels", None),
                    attn_norm_eps=getattr(layer.attn_layernorm, "variance_epsilon", None),
                    ffn_norm_eps=getattr(layer.ffn_layernorm, "variance_epsilon", None),
                    hc_shifted=getattr(layer, "hc_shifted", None),
                    hc_norm_eps=getattr(layer, "hc_norm_eps", None),
                    hc_pre_eps=getattr(layer, "hc_pre_eps", None),
                    hc_sinkhorn_eps=getattr(layer, "hc_sinkhorn_eps", None),
                    hc_sinkhorn_repeat=getattr(layer, "hc_sinkhorn_repeat", None),
                    hc_post_mult_value=getattr(layer, "hc_post_mult_value", None),
                    use_mega_mhc=getattr(layer, "use_mega_mhc", None),
                )
            except Exception as exc:  # noqa: BLE001
                _fine_err("attn post", exc)
        STATE["attn_ctx"] = False

    handles.append(attn.register_forward_pre_hook(apre, with_kwargs=True))
    handles.append(attn.register_forward_hook(apost, with_kwargs=True))

    def ctx() -> bool:
        return STATE.get("attn_ctx", False) and active()

    def w_proj_q_swa(orig):
        def wrapper(*a, **kw):
            out = orig(*a, **kw)
            if ctx():
                q_a, kv_swa = out
                _fine_store("q_a", q_a)
                _fine_store("kv_proj", kv_swa)
            return out

        return wrapper

    def w_prepare_swa(orig):
        def wrapper(*a, **kw):
            out = orig(*a, **kw)
            if ctx() and out is not None and out[0] is not None:
                kv = out[0]
                _fine_store("kv", kv.reshape(kv.shape[0], -1))
            return out

        return wrapper

    def w_q_norm_rope(orig):
        def wrapper(q, positions, *a, **kw):
            if not ctx():
                return orig(q, positions, *a, **kw)
            q_in = q.detach().clone()
            out = orig(q, positions, *a, **kw)
            try:
                if fused:
                    q_nat = sfused.q_from_fused_layout(q_in) if sfused is not None else None
                    if q_nat is None:
                        STATE["errors"].append(f"L{idx}: sm120_attn.fused not importable: q_pre_rope layout unknown")
                    else:
                        _fine_store("q_pre_rope", q_nat)
                else:
                    _fine_store("q_pre_rope", q_in)
                    _fine_store("q_post_rope", out)  # includes the weightless q_b_norm if use_q_b_norm
            except Exception as exc:  # noqa: BLE001
                _fine_err("q", exc)
            return out

        return wrapper

    def w_wv_o(orig):
        def wrapper(context, positions, *a, **kw):
            if not ctx() or isinstance(context, (tuple, list)):
                return orig(context, positions, *a, **kw)
            _fine_store("attn_core_pre_unrope", context)
            out = orig(context, positions, *a, **kw)  # inverse RoPE is applied in place on `context`
            _fine_store("attn_core_out", context)
            return out

        return wrapper

    handles.append(_override(attn, "_project_q_and_swa", w_proj_q_swa))
    if hasattr(attn, "_prepare_swa_kv"):
        handles.append(_override(attn, "_prepare_swa_kv", w_prepare_swa))
    handles.append(_override(attn, "_q_b_norm_rope", w_q_norm_rope))
    handles.append(_override(attn, "_wv_and_o_proj", w_wv_o))

    if sfused is not None:
        def w_prepare_q(orig):
            def wrapper(*a, **kw):
                out = orig(*a, **kw)
                if ctx():
                    _fine_store("q_post_rope", out[0])
                    if out[1] is not None:
                        _fine_store("q_rms_scale", out[1])
                return out

            return wrapper

        def w_sparse_prefill(orig):
            def wrapper(*a, **kw):
                out = orig(*a, **kw)
                if ctx():
                    _fine_store("attn_core_pre_unrope", out[0])
                return out

            return wrapper

        def w_cast_output(orig):
            def wrapper(out, *a, **kw):
                if ctx():
                    _fine_store("attn_core_out", out)
                return orig(out, *a, **kw)

            return wrapper

        # names follow the attention shim (sparse_prefill_torch was renamed to sparse_prefill); tap what exists
        for name, wrap in (("_prepare_q", w_prepare_q), ("sparse_prefill", w_sparse_prefill),
                           ("sparse_prefill_torch", w_sparse_prefill), ("cast_output", w_cast_output)):
            if hasattr(sfused, name):
                handles.append(_override(sfused, name, wrap))

    # ---- MoE
    def w_gate(orig):
        def wrapper(*a, **kw):
            out = orig(*a, **kw)
            if active():
                idx_t, w_t = out
                _fine_store("gate_idx", idx_t.long())
                _fine_store("gate_w", w_t.float())
            return out

        return wrapper

    def w_shared(orig):
        def wrapper(hidden_states, out_acc=None, *a, **kw):
            out = orig(hidden_states, out_acc, *a, **kw)
            if active() and out_acc is None and out is not None:
                _fine_store("shared_out", out)
            return out

        return wrapper

    handles.append(_override(mlp, "gate_forward", w_gate))
    handles.append(_override(mlp, "shared_forward", w_shared))

    class _RoutedTap:
        def __init__(self, inner):
            self.inner = inner

        def __call__(self, *a, **kw):
            out = self.inner(*a, **kw)
            if active():
                _fine_store("routed_out", out)
            return out

        def __getattr__(self, item):
            return getattr(self.inner, item)

    if getattr(mlp, "sm120_routed", None) is not None:
        handles.append(_override(mlp, "sm120_routed", lambda inner: _RoutedTap(inner)))

    def mlp_post(mod, args, out):
        if active():
            _fine_store("ffn_out", out)
            _fine_meta(
                moe_impl=getattr(mlp, "sm120_moe_impl", None),
                use_mega_moe=getattr(mlp, "use_mega_moe", None),
                use_mega_gate=getattr(mlp, "use_mega_gate", None),
                routed_scaling_factor=getattr(mlp, "routed_scaling_factor", None),
                swiglu_clamp_value=getattr(mlp, "swiglu_clamp_value", None),
                num_topk=getattr(mlp, "num_topk", None),
                use_fp8_act=getattr(mlp, "use_fp8_act", None),
                num_extra_experts=getattr(mlp, "num_extra_experts", None),
            )

    handles.append(mlp.register_forward_hook(mlp_post))
    return handles
