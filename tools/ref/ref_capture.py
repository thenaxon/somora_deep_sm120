"""Fine prefill taps inside reference Blocks (ds41-inference/inference/model.py), for tools/compare_l0.py.

    import ref_capture
    taps = [ref_capture.install(model, M, i, store=hidden["prefill"], prompt_len=L) for i in layers]
    ref_capture.STATE["armed"] = True;  model.forward(x, 0);  ref_capture.STATE["armed"] = False
    for t in reversed(taps): t.remove()     # reverse order: each install wraps the previous one's globals

Several layers may be tapped at once: the attention-internal wrappers only record while STATE["attn_ctx"] holds
their own layer index.

Keys: store["L<i>/<name>"], CPU, prompt order, batch dim dropped (bsz = 1). Same names / semantics as the
DeepInfer side (sm120/sm120_capture.py install_layer_captures):

  hc_trunk_in  [L,4,D] bf16  Block input x                                    (model.py:983)
  attn_pre_mix_in [L,4] fp32 pre_mix argument (previous layer's ffn_pre; one-hot for layer 0, model.py:1263)
  attn_mix_pre/post [L,4], attn_mix_comb [L,4,4] fp32  hc_mixes(x, hc_attn_*)  (model.py:984)
  hc_pre_attn  [L,D] bf16    hc_pre(x, pre_mix)                               (model.py:985)
  attn_in      [L,D] bf16    attn_norm output                                 (model.py:986)
  q_a          [L,q_lora]    wq_a output (before q_norm)                      (model.py:771)
  q_pre_rope   [L,H,hd] bf16 wq_b output                                      (model.py:772)
  q_post_rope  [L,H,hd] bf16 q passed to sparse_attn (RoPE applied in place)   (model.py:773,781)
  kv_proj      [L,hd] bf16   wkv output (before kv_norm)                      (model.py:706)
  kv_normed_pre_rope [L,hd]  kv_norm output (ref only; DeepInfer fuses norm+rope+cast)
  kv           [L,hd] bf16   window kv given to sparse_attn: norm + RoPE + fp8 act_quant round trip
                             (model.py:706-708; first L rows of the kv argument)
  swa_idx      [L,W] int32   window part of topk_idxs (get_window_topk_idxs, model.py:410-427), -1 = masked,
                             W = min(L, window_size)
  compress_idx [L,k] int32   compressed part of topk_idxs minus the window offset L = compressed positions each
                             query attends to, -1 = masked (compressed layers only; for consumers the source's
                             indexer result, model.py:723-739)
  cand_mask    [L,C] bool    candidate positions the candidate source published (model.py:570-573; only on the
                             candidate source layer)
  attn_core_pre_unrope [L,H,hd] bf16 sparse_attn output                       (model.py:781)
  attn_core_out [L,H,hd] bf16 after the inverse RoPE, before act_quant/wo_a   (model.py:782)
  rope_cos / rope_sin [L,rope_dim/2] fp32  freqs_cis[:L].real / .imag (this layer's buffer, model.py:684-697)
  attn_sink    [H] fp32
  attn_out     [L,D] bf16    Attention output                                 (model.py:987)
  hc_trunk_mid [L,4,D] bf16  after the attention hc_post                      (model.py:988)
  ffn_mix_pre/post/comb      hc_mixes(x, hc_ffn_*)                            (model.py:991)
  ffn_pre_mix_in [L,4]       attn_pre (mix applied to the FFN input)          (model.py:992)
  hc_pre_ffn   [L,D] bf16    hc_pre(x, attn_pre)                              (model.py:992)
  ffn_in       [L,D] bf16    ffn_norm output                                  (model.py:993)
  gate_idx [L,k] int64, gate_w [L,k] fp32   Gate output (after bias selection, norm, route_scale; model.py:813-828)
  routed_out   [L,D] fp32    sum of routed experts, before the shared expert  (model.py:895-904)
  shared_out   [L,D] bf16    shared expert                                    (model.py:905)
  ffn_out      [L,D] bf16    MoE output                                       (model.py:994)
  meta (dict)  eps / window / rope parameters in effect

Block.forward and MoE.forward of the captured layer are replaced by line-for-line copies that only add the
recording (same calls in the same order -> bit-identical numerics). Everything is inert unless STATE["armed"].
"""

from __future__ import annotations

import torch

STATE: dict = {"armed": False, "attn_ctx": None, "o_stash": None}  # attn_ctx: index of the layer in attention


class _Taps:
    def __init__(self):
        self.undo: list = []

    def remove(self):
        for fn in reversed(self.undo):
            try:
                fn()
            except Exception:  # noqa: BLE001
                pass
        self.undo.clear()


def install(model, M, layer_idx: int, store: dict, prompt_len: int) -> _Taps:
    blk = model.layers[layer_idx]
    attn, moe = blk.attn, blk.ffn
    taps = _Taps()
    pfx = f"L{layer_idx}/"
    L = int(prompt_len)
    errors = store.setdefault(pfx + "errors", [])

    def put(name, t, rows=True, batch=True):
        key = pfx + name
        if not STATE["armed"] or key in store:
            return
        try:
            t = t.detach()
            if batch:
                t = t[0]
            if rows:
                if t.shape[0] < L:
                    errors.append(f"{name}: {t.shape[0]} rows < {L}")
                    return
                t = t[:L]
            store[key] = t.to("cpu", copy=True)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {type(exc).__name__}: {exc}")

    # ---------------------------------------------------------------- Block.forward copy (model.py:970-996)
    orig_block_forward = blk.forward

    def block_forward(x, start_pos, pre_mix, image_mask, *attn_args):
        if not STATE["armed"] or start_pos != 0:
            return orig_block_forward(x, start_pos, pre_mix, image_mask, *attn_args)
        self = blk
        put("hc_trunk_in", x)
        put("attn_pre_mix_in", pre_mix)
        residual = x
        attn_pre, attn_post, attn_comb = self.hc_mixes(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        put("attn_mix_pre", attn_pre)
        put("attn_mix_post", attn_post)
        put("attn_mix_comb", attn_comb)
        x = self.hc_pre(x, pre_mix)
        put("hc_pre_attn", x)
        x = self.attn_norm(x)
        put("attn_in", x)
        STATE["attn_ctx"], STATE["o_stash"] = layer_idx, None
        try:
            x = self.attn(x, start_pos, *attn_args)
        finally:
            STATE["attn_ctx"], STATE["o_stash"] = None, None
        put("attn_out", x)
        x = self.hc_post(x, residual, attn_post, attn_comb)
        put("hc_trunk_mid", x)

        residual = x
        ffn_pre, ffn_post, ffn_comb = self.hc_mixes(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        put("ffn_mix_pre", ffn_pre)
        put("ffn_mix_post", ffn_post)
        put("ffn_mix_comb", ffn_comb)
        put("ffn_pre_mix_in", attn_pre)
        x = self.hc_pre(x, attn_pre)
        put("hc_pre_ffn", x)
        x = self.ffn_norm(x)
        put("ffn_in", x)
        x = self.ffn(x, image_mask)
        put("ffn_out", x)
        x = self.hc_post(x, residual, ffn_post, ffn_comb)
        return x, ffn_pre

    blk.forward = block_forward
    taps.undo.append(lambda: delattr(blk, "forward"))

    # ---------------------------------------------------------------- MoE.forward copy (model.py:891-906)
    orig_moe_forward = moe.forward

    def moe_forward(x, image_mask=None):
        if not STATE["armed"]:
            return orig_moe_forward(x, image_mask)
        self = moe
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x, None if image_mask is None else image_mask.flatten())
        put("gate_w", weights.float(), batch=False)
        put("gate_idx", indices.long(), batch=False)
        y = torch.zeros_like(x, dtype=torch.float32)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx], weights[idx, top, None])
        if M.world_size > 1:
            M.dist.all_reduce(y)
        put("routed_out", y, batch=False)
        shared = self.shared_experts(x)
        put("shared_out", shared, batch=False)
        y += shared
        return y.type_as(x).view(shape)

    moe.forward = moe_forward
    taps.undo.append(lambda: delattr(moe, "forward"))

    # ---------------------------------------------------------------- attention internals
    def ctx():
        return STATE["armed"] and STATE["attn_ctx"] == layer_idx

    def mod_hook(name, reshape_heads=False):
        def hook(mod, inp, out):
            if not ctx():
                return
            t = out
            if reshape_heads:
                t = t.unflatten(-1, (attn.n_local_heads, attn.head_dim))
            put(name, t)

        return hook

    for mod, name, rh in ((attn.wq_a, "q_a", False), (attn.wq_b, "q_pre_rope", True),
                          (attn.wkv, "kv_proj", False), (attn.kv_norm, "kv_normed_pre_rope", False)):
        h = mod.register_forward_hook(mod_hook(name, rh))
        taps.undo.append(h.remove)

    orig_sparse = M.sparse_attn
    orig_rot = M.apply_rotary_emb

    def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
        o = orig_sparse(q, kv, attn_sink, topk_idxs, softmax_scale)
        if ctx() and STATE["o_stash"] is None:
            put("q_post_rope", q)
            put("kv", kv)  # first L rows = this chunk's window kv (compressed kv, if any, is appended after)
            win = min(L, attn.window_size)  # get_window_topk_idxs width in prefill (model.py:418)
            put("swa_idx", topk_idxs[..., :win].to(torch.int32))
            if attn.compress_ratio:
                c = topk_idxs[..., win:].long()  # model.py:579: idx + offset(= L window rows) or -1
                put("compress_idx", torch.where(c >= 0, c - L, -1).to(torch.int32))
                ind = attn.indexer
                if ind is not None and ind.is_candidate_source and M.shared_attn.candidates is not None:
                    put("cand_mask", M.shared_attn.candidates)
            put("attn_core_pre_unrope", o)
            STATE["o_stash"] = o
        return o

    def apply_rotary_emb(x, freqs_cis, inverse=False):
        y = orig_rot(x, freqs_cis, inverse)
        if inverse and ctx() and isinstance(STATE["o_stash"], torch.Tensor):
            put("attn_core_out", STATE["o_stash"])  # o modified in place through the [..., -rd:] view
            STATE["o_stash"] = False  # recorded; do not record again
        return y

    M.sparse_attn = sparse_attn
    M.apply_rotary_emb = apply_rotary_emb

    def _restore_globals():
        M.sparse_attn = orig_sparse
        M.apply_rotary_emb = orig_rot

    taps.undo.append(_restore_globals)

    # ---------------------------------------------------------------- static per-layer data
    def put_static():
        fc = attn.freqs_cis[:L]
        store[pfx + "rope_cos"] = fc.real.float().cpu().clone()
        store[pfx + "rope_sin"] = fc.imag.float().cpu().clone()
        store[pfx + "attn_sink"] = attn.attn_sink.detach().float().cpu().clone()
        store[pfx + "positions"] = torch.arange(L, dtype=torch.long)
        a = getattr(model, "_capture_args", None)  # ModelArgs, optional (set by ref_runner)
        meta = dict(
            norm_eps=blk.norm_eps,
            attn_norm_eps=blk.attn_norm.eps,
            ffn_norm_eps=blk.ffn_norm.eps,
            q_a_norm_eps=attn.q_norm.eps,
            kv_norm_eps=attn.kv_norm.eps,
            window_size=attn.window_size,
            softmax_scale=attn.softmax_scale,
            compress_ratio=attn.compress_ratio,
            is_kv_source=attn.is_kv_source,
            is_index_source=attn.is_index_source,
            owns_index_k=bool(attn.indexer is not None and attn.indexer.owns_k),
            is_candidate_source=bool(attn.indexer is not None and attn.indexer.is_candidate_source),
            uses_candidates=bool(attn.indexer is not None and attn.indexer.uses_candidates),
            rope_dim=attn.rope_head_dim,
            hc_eps=blk.hc_eps,
            hc_sinkhorn_iters=blk.hc_sinkhorn_iters,
            hc_post_mult_value=2.0,  # hard-coded 2 * sigmoid in hc_split_sinkhorn (kernel.py:429)
            route_scale=moe.gate.route_scale,
            norm_topk_prob=moe.gate.norm_topk_prob,
            score_func=moe.gate.score_func,
            topk=moe.gate.topk,
            swiglu_limit=moe.shared_experts.swiglu_limit,
            q_norm_per_head=False,  # model.py Attention.forward has no per-head q RMSNorm
            sparse_attn_p_bf16=True,  # kernel.py: P tile cast to bf16 before P@V
        )
        if a is not None:
            if attn.compress_ratio:
                meta.update(rope_theta=a.compress_rope_theta, rope_original_seq_len=a.original_seq_len,
                            rope_factor=a.rope_factor)
            else:
                meta.update(rope_theta=a.rope_theta, rope_original_seq_len=0, rope_factor=None)
        store[pfx + "meta"] = meta

    taps.put_static = put_static
    return taps
