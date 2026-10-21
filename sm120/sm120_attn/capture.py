"""Optional capture of DeepInfer's sparse selections for tools/compare_l0.py (docs/MTP_YOCO.md).

Active only while sm120_capture is armed and inside the attention of a CAPTURE_LAYER layer during the
first prompt forward (sm120_capture.STATE["fine_active"] / ["attn_ctx"]); otherwise a dict lookup.

  L<i>/compress_idx [L, W] int32  compressed positions (0 = first compressed latent) the layer's sparse
                                   prefill attention reads, -1 for SWA / masked / invalid entries.
                                   Recorded from the index list passed to flash_mla_sparse_fwd /
                                   fused prefill, so YOCO consumers report the reused top-k.
                                   Compressed region: the trailing run of FP4-exact tokens in the
                                   assembled prefill KV ([SWA cached | SWA current | compressed],
                                   attn.py:813-857; compressed tokens are dequantized FP4 records).
  L<i>/cand_mask    [L, C] bool    compressed positions inside the candidate blocks the candidate
                                   source publishes (index_attn.py _topk_candidate_block: block b
                                   covers b*bs .. b*bs+bs-1, clipped to the compressed width
                                   max(k_len), like the reference's select_candidate_blocks).
"""

from __future__ import annotations

import sys

import torch


def _state():
    cap = sys.modules.get("sm120_capture")
    if cap is None:
        return None, None
    st = getattr(cap, "STATE", None)
    if not st or not st.get("armed") or st.get("fine_active") is None:
        return None, None
    return cap, st


def record_prefill_indices(kv: torch.Tensor, indices: torch.Tensor, topk_length: torch.Tensor | None) -> None:
    cap, st = _state()
    if cap is None or not st.get("attn_ctx"):
        return
    try:
        from .quant import dequant_v41_fp4_records, quantize_v41_fp4_records
        n = kv.shape[0]
        is_fp4 = (dequant_v41_fp4_records(quantize_v41_fp4_records(kv)) == kv.float()).all(-1)
        trailing = int(torch.flip(is_fp4, [0]).int().cumprod(0).sum().item())
        if trailing == 0:
            return  # no compressed region (SWA-only layer)
        off = n - trailing
        idx = indices.long()
        valid = (idx >= off) & (idx < n)
        if topk_length is not None:
            pos = torch.arange(idx.shape[-1], device=idx.device).view(1, -1)
            valid &= pos < topk_length.long().view(-1, 1)
        rel = torch.where(valid, idx - off, torch.full_like(idx, -1)).to(torch.int32)
        cap._fine_store("compress_idx", rel)
        cap._fine_meta(compress_region_start=off, compress_region_len=trailing)
    except Exception as exc:  # noqa: BLE001
        cap._fine_err("compress_idx (sm120_attn)", exc)


def record_candidate_blocks(block_idx: torch.Tensor) -> None:
    """Called from deep_select.topk; picks up the caller's locals in index_attn._topk_candidate_block."""
    cap, st = _state()
    if cap is None:
        return
    try:
        f = sys._getframe(2)
        if f.f_code.co_name != "_topk_candidate_block":
            return
        k_len = f.f_locals.get("k_len")
        offset = int(f.f_locals.get("offset", 0))
        self = f.f_locals.get("self")
        bs = int(self.candidate_topk_cfg.block_size)
        b = block_idx.long()
        rows, nb = b.shape
        C = int(k_len.max().item()) if k_len is not None and k_len.numel() else nb * bs
        pos = (b.clamp_min(0).unsqueeze(-1) * bs + torch.arange(bs, device=b.device)).view(rows, -1)
        # whole blocks, clipped only to the compressed width (= the reference's select_candidate_blocks,
        # ds41 model.py:584-611; it does not clip to the row's own valid length)
        ok = (b >= 0).repeat_interleave(bs, dim=1) & (pos < C)
        mask = torch.zeros((rows, C + 1), dtype=torch.bool, device=b.device)  # column C = sink for invalid
        mask.scatter_(1, torch.where(ok, pos, torch.full_like(pos, C)), True)
        mask = mask[:, :C]
        parts = st.setdefault("_sm120_cand_parts", {}).setdefault(st["fine_active"], {})
        parts[offset] = mask.cpu()
        L = st.get("prompt_len") or 0
        total = sum(m.shape[0] for m in parts.values())
        if total >= L:
            width = max(m.shape[1] for m in parts.values())
            full = torch.cat([torch.nn.functional.pad(parts[k], (0, width - parts[k].shape[1]))
                              for k in sorted(parts)], 0)
            cap._fine_store("cand_mask", full)
            cap._fine_meta(candidate_block_size=bs)
    except Exception as exc:  # noqa: BLE001
        cap._fine_err("cand_mask (sm120_attn)", exc)
