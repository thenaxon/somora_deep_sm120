"""SM120 shim of flash_mla.fused_norm_rope_attn_rope_cast
(vendor/FlashMLA/flash_mla/fused_norm_rope_attn_rope_cast.py): unfused torch norm/rope/cast around
sm120_attn sparse attention (decode and prefill cores may run on b12x). Same signatures and return layouts.
See sm120_attn/fused.py for the exact layout contract."""
from sm120_attn.fused import decode, permute_q_b_proj, permute_wv_proj, prefill  # noqa: F401

__all__ = ["prefill", "decode", "permute_q_b_proj", "permute_wv_proj"]
