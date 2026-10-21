"""Raw-logit logprobs for server.py (OpenAI `logprobs` / `top_logprobs`).

DeepInfer's sampler returns *processed* logprobs: log-softmax after temperature and top-p/top-k
(sampler_utils.process_sampling). With greedy decoding (temperature -> SAMPLING_EPS) these are degenerate (0 for the
argmax, -inf elsewhere). vLLM, the comparison baseline, reports raw logprobs by default. This tap computes
log_softmax(raw logits) instead: it wraps `Sampler.forward` (which sequences the sampler call serves) and the
sampler module's `get_logits` (the raw [rows, vocab] logits), and stores the top-K (K = 20, the OpenAI maximum) for
the rows of requests that asked for logprobs.

Row layout (the same as sm120_capture's): non-placeholder sequences in seq_metadata order.
- prefill: one row per sequence, used only for the last prompt chunk;
- decode: one row per sequence; nextn = 0 only, since MTP verification has 1 + nextn rows per sequence and the server refuses
  logprobs with MTP.
Rows are consumed in order: the n-th recorded row of a request belongs to its n-th output token. With ZOS a finished
request may run one more forward; that row is dropped when the request is closed.
Inert (no sync, no copies) while no live request wants logprobs.
"""

from __future__ import annotations

import collections
import logging

log = logging.getLogger("di_server.logprobs")

K = 20


def _is_placeholder(seq) -> bool:
    flag = getattr(seq, "is_placeholder", None)
    if flag is not None:
        return bool(flag)
    return getattr(seq, "request_id", None) == "placeholder"


class LogprobTap:
    def __init__(self):
        self.want: dict[str, int] = {}
        self.rows: dict[str, collections.deque] = collections.defaultdict(collections.deque)
        self._ctx = None
        self.mismatches = 0
        self.installed = False

    def install(self) -> None:
        import deepinfer.model_executor.layers.sampler as mod

        cls = mod.Sampler
        orig_forward, orig_get_logits = cls.forward, mod.get_logits
        tap = self

        def forward(self_, mtp_module, embedding, hidden_states, sampler_inputs, main_input_metadata, *a, **kw):
            smd = kw.get("seq_metadata_list", a[1] if len(a) > 1 else None)
            tap._ctx = (bool(getattr(main_input_metadata, "is_prompt", False)), smd) if tap.want else None
            try:
                return orig_forward(self_, mtp_module, embedding, hidden_states, sampler_inputs, main_input_metadata,
                                    *a, **kw)
            finally:
                tap._ctx = None

        def get_logits(hidden_states, embedding, vocab_size):
            logits = orig_get_logits(hidden_states, embedding, vocab_size)
            ctx, tap._ctx = tap._ctx, None  # only the target model's call; DSpark draft logits come later
            if ctx is not None:
                try:
                    tap._record(logits, *ctx)
                except Exception:  # noqa: BLE001
                    log.exception("logprob tap failed")
            return logits

        cls.forward = forward
        mod.get_logits = get_logits
        self.installed = True

    def _record(self, logits, is_prompt: bool, smd) -> None:
        import torch

        seqs = [s for s in (smd or []) if not _is_placeholder(s)]
        n = int(logits.shape[0])
        if is_prompt:
            last = [s for s in seqs if bool(getattr(s, "is_last_prompt_chunk", True))]
            if n == len(seqs):
                rows = [(i, s) for i, s in enumerate(seqs) if s in last]
            elif n == len(last):
                rows = list(enumerate(last))
            else:
                rows = None
        else:
            rows = list(enumerate(seqs)) if n == len(seqs) else None
        if rows is None:
            self.mismatches += 1
            if self.mismatches <= 5:
                log.warning("logprob tap: %d logit rows for %d sequences (prompt=%s); skipped", n, len(seqs), is_prompt)
            return
        sel = [(i, s.request_id) for i, s in rows if s.request_id in self.want]
        if not sel:
            return
        idx = torch.tensor([i for i, _ in sel], device=logits.device)
        lp = torch.log_softmax(logits.index_select(0, idx).float(), dim=-1)
        v, ix = lp.topk(K, dim=-1)
        v, ix = v.cpu().tolist(), ix.cpu().tolist()
        for (_, rid), vv, ii in zip(sel, v, ix):
            self.rows[rid].append((ii, vv))

    # ---- engine thread API
    def add(self, request_id: str, k: int) -> None:
        self.want[request_id] = k

    def pop_row(self, request_id: str):
        q = self.rows.get(request_id)
        return q.popleft() if q else None

    def close(self, request_id: str) -> None:
        self.want.pop(request_id, None)
        self.rows.pop(request_id, None)
