#!/usr/bin/env python3
"""Run DeepInfer offline on a truncated V4.1 checkpoint (3 layers for the base test model, N layers for YOCO, see
docs/MTP_YOCO.md) on ONE SM120 GPU and dump logits and per-layer prefill hidden states in the reference format of
tools/ref/ref_runner.py ("sm120-ref-v1").

    python3 /opt/sm120/tools/run_deepinfer.py --ckpt /work/ckpt3 --prompt-ids /work/prompt_ids.json --out /work/di_out.pt

Launch path = DeepInfer's examples (_config.create_llm + _launch.launch_dp, DP=1, EP=1, CP=1), eager
(DEEPINFER_ENFORCE_EAGER=1), no MTP (DEEPINFER_USE_NEXTN_DECODE=0), greedy (temperature 0 ->
SAMPLING_EPS, argmax-equivalent), ignore_eos, max_tokens = decode_steps + 1 so that
decoded_ids[0] is the prefill argmax and decoded_ids[s+1] is the argmax of decode step s (same
convention as the reference).

MTP/DSpark: --nextn N (DEEPINFER_USE_NEXTN_DECODE, lookahead_tokens=min(1, N)); a decode step then carries 1+N logit
rows; decode_logits_all holds one row per emitted token (assign_decode_rows) and meta.mtp the tokens per step and
the draft acceptance. --graphs turns CUDA graphs on, --force-topk sets DEEPINFER_SAMPLER_FORCE_TOPK.

Requires PYTHONPATH=/opt/sm120/sm120:/opt/deepinfer (shims first: deep_gemm / deep_ep / flash_mla /
deep_select replacements + sitecustomize -> sm120_patches). Switches: SM120_GEMM_IMPL, SM120_MOE_IMPL,
SM120_ATTN_IMPL (see docs/VALIDATION.md).

Output keys (all CPU tensors):
  input_ids [L], decoded_ids [S+1], decode_positions [S],
  prefill_top20_values/ids [L, 20], prefill_argmax [L], prefill_logits_last64 [64, V],
  prefill_logits [L, V] (None with --compact), prefill_finite,
  decode_top50_values/ids [S, 50], decode_logits_first4 [4, V], decode_logits_all [S, V] (extra),
  prefill_sampled_logits [1, V] (extra: the logits DeepInfer's sampler itself used),
  hidden = {"prefill": {"embed_hc", "layer{i}", "final_norm", "L<c>/<name>"...}, "decode": []}, meta = {...}
  where L<c>/<name> are the fine taps of the decoder layers in env CAPTURE_LAYER ('0' default, '0,5,7', 'all',
  -1 = off), see sm120/sm120_capture.py install_layer_captures and compare_l0.py. The shim taps one layer per
  run; this script installs them itself on every requested layer (install_fine_taps).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="/work/ckpt3")
    ap.add_argument("--prompt-ids", default="/work/prompt_ids.json", help="json with 'input_ids' (or a list)")
    ap.add_argument("--ref", default="/work/ref_out.pt", help="reference output (only for --teacher-force / ids)")
    ap.add_argument("--out", default="/work/di_out.pt")
    ap.add_argument("--decode-steps", type=int, default=16)
    ap.add_argument("--teacher-force", action="store_true",
                    help="force DeepInfer's sampled tokens to the reference decoded_ids (golden_token_ids) so every "
                         "decode step sees the reference prefix")
    ap.add_argument("--max-model-len", type=int, default=16384)  # >= candidate_topk_blocks(2048) * candidate_block_size(8): index_attn.py:213
    ap.add_argument("--max-num-batched-tokens", type=int, default=4096, help=">= prompt length (no chunked prefill)")
    ap.add_argument("--max-blocks", type=int, default=256, help="SM120_MAX_GPU_BLOCKS (KV-cache cap, 128 tok/block)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--compact", action="store_true", help="do not store full prefill logits")
    ap.add_argument("--no-hidden", action="store_true")
    ap.add_argument("--zos", choices=("default", "0", "1"), default="default",
                    help="DEEPINFER_ZERO_OVERHEAD_SCHEDULING (default: engine default = 1 for NORMAL)")
    ap.add_argument("--batch-invariant", action="store_true", help="DEEPINFER_BATCH_INVARIANT=1")
    ap.add_argument("--nextn", type=int, default=None,
                    help="DEEPINFER_USE_NEXTN_DECODE (MTP/DSpark draft tokens per step; default: env or 0). Needs a "
                         "checkpoint with mtp.* layers (ref/make_truncated_ckpt.py --mtp)")
    ap.add_argument("--graphs", action="store_true",
                    help="CUDA graphs: DEEPINFER_ENFORCE_EAGER=0 and DEEPINFER_DFLASH_ENFORCE_EAGER=0 (default eager)")
    ap.add_argument("--force-topk", type=int, default=None, help="DEEPINFER_SAMPLER_FORCE_TOPK")
    ap.add_argument("--deepinfer-root", default=os.environ.get("DEEPINFER_ROOT", "/opt/deepinfer"))
    return ap.parse_args()


def load_prompt(args) -> list[int]:
    if args.prompt_ids and os.path.isfile(args.prompt_ids):
        with open(args.prompt_ids) as f:
            obj = json.load(f)
        ids = obj["input_ids"] if isinstance(obj, dict) else obj
        return [int(x) for x in ids]
    import torch

    ref = torch.load(args.ref, map_location="cpu", weights_only=False)
    return [int(x) for x in ref["input_ids"].tolist()]


def setup_env(args) -> None:
    env = {
        "DEEPINFER_USE_NEXTN_DECODE": "0",
        "DEEPINFER_ENFORCE_EAGER": "1",
        "DEEPINFER_DFLASH_ENFORCE_EAGER": "1",
        "DEEPINFER_DISABLE_SYNC_GUARD": "1",  # shims + capture do CPU syncs inside forward/sampling
        "DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS": "0",  # no EPLB redundancy at EP=1
        "DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL": "0",
        "DEEPINFER_EXPERT_COUNT_WINDOW_SIZE": "1",
        # YOCO consumers must see every prefill query like the reference (mask/compact drop early queries)
        "DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE": "off",
        "DEEPINFER_LOGGING_INTERVAL_STEP": "1",
        "SM120_MAX_GPU_BLOCKS": str(args.max_blocks),
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
        "RANK": "0",
        "WORLD_SIZE": "1",
    }
    for k, v in env.items():
        os.environ.setdefault(k, v)
    if args.zos != "default":
        os.environ["DEEPINFER_ZERO_OVERHEAD_SCHEDULING"] = args.zos
    if args.batch_invariant:
        os.environ["DEEPINFER_BATCH_INVARIANT"] = "1"
    if args.nextn is not None:
        os.environ["DEEPINFER_USE_NEXTN_DECODE"] = str(args.nextn)
    if args.graphs:
        os.environ["DEEPINFER_ENFORCE_EAGER"] = "0"
        os.environ["DEEPINFER_DFLASH_ENFORCE_EAGER"] = "0"
    if args.force_topk is not None:
        os.environ["DEEPINFER_SAMPLER_FORCE_TOPK"] = str(args.force_topk)
    for p in (os.path.join(args.deepinfer_root, "examples"), args.deepinfer_root):
        if p not in sys.path:
            sys.path.insert(0, p)
    shims = os.path.join(os.path.dirname(HERE), "sm120")
    if shims not in sys.path:
        sys.path.insert(0, shims)


def parse_capture_layers(v: str | None, n_layers: int) -> list[int]:
    """Same syntax as ref/ref_runner.py: '0' (default), '0,5,7', 'all', ''/'none'/'off'/'-1' = no taps."""
    v = (v if v is not None else "0").strip().lower()
    if v in ("", "none", "off", "no", "-1"):
        return []
    if v == "all":
        return list(range(n_layers))
    out = sorted({int(x) for x in v.split(",") if x.strip()})
    bad = [i for i in out if not 0 <= i < n_layers]
    if bad:
        raise SystemExit(f"CAPTURE_LAYER {bad} out of range 0..{n_layers - 1}")
    return out


def install_fine_taps(sm120_capture, layers, idxs: list[int]) -> list:
    """sm120_capture.install_layer_captures on several layers. The shim closes the capture window after the first
    tapped layer (STATE["fine_done"], one layer per run); a pre-hook registered before its own reopens it for every
    layer that has not been recorded yet, a post-hook after its own marks the layer recorded."""
    st = sm120_capture.STATE
    done: set[int] = set()
    hooks: list = []
    for li in idxs:
        layer = layers[li]

        def reopen(mod, args, li=li):
            if st.get("armed") and li not in done:
                st["fine_done"] = False

        def close(mod, args, out, li=li):
            if st.get("fine_done"):
                done.add(li)

        hooks.append(layer.register_forward_pre_hook(reopen))
        try:
            hooks += sm120_capture.install_layer_captures(layer, li)
        except Exception as exc:  # noqa: BLE001
            st.setdefault("errors", []).append(f"L{li}: install_layer_captures failed: {type(exc).__name__}: {exc}")
        hooks.append(layer.register_forward_hook(close))
    return hooks


def install_accept_log(sm120_capture, model) -> tuple[list, object]:
    """Record SamplerOutput.num_accepted_tokens_gpu (accepted drafts per sequence, sampler.py:289) for every armed,
    non-placeholder decode step, in the same order as sm120_capture's decode entries. Returns (log, handle)."""
    st = sm120_capture.STATE
    sampler = model.sampler
    had = "forward" in sampler.__dict__
    orig = sampler.forward
    log: list = []
    times: list = []  # (time.time(), is_prompt) of every armed non-placeholder sampler call

    def forward(*a, **kw):
        out = orig(*a, **kw)
        if st.get("armed") and not st.get("skip_step"):
            times.append((time.time(), bool(st.get("in_prompt"))))
            if not st.get("in_prompt"):
                acc = getattr(out, "num_accepted_tokens_gpu", None)
                log.append(None if acc is None else [int(x) for x in acc.detach().cpu().tolist()])
        return out

    forward.times = times

    sampler.forward = forward

    class _H:
        @staticmethod
        def remove():
            if had:
                sampler.forward = orig
            else:
                del sampler.forward

    return log, _H


def assign_decode_rows(entries: list, gen: list[int], accepted: list | None = None) -> tuple[list, list[int]]:
    """Per-position decode logits from per-step logit rows.

    entries[k] = [seqlen_q, V] rows of the request in decode step k (sm120_capture.decode_rows()). Without MTP each
    step has one row = the logits of the one new token. With MTP (nextn) a step verifies 1+nextn rows and emits the
    accepted prefix plus one token: rows 0..a are the logits of the a+1 emitted tokens, later rows saw a rejected
    draft. Greedy decoding emits argmax(row j) while the drafts are accepted, so the emitted count of a step is
    recovered as the leading run of rows whose argmax equals the next sampled tokens (at least 1). A wrong-context row
    whose argmax happens to equal the next token would be over-assigned; the caller checks the total.
    With `accepted` (per step: accepted drafts of the request, from install_accept_log) the count is exact: a + 1.
    Returns (rows in position order, emitted tokens per step)."""
    out, counts, pos = [], [], 1  # gen[0] comes from the prefill step
    for k, e in enumerate(entries):
        if accepted is not None:
            n = min(accepted[k] + 1, e.shape[0], len(gen) - pos)
        else:
            n = 0
            while n < e.shape[0] and pos + n < len(gen) and (n == 0 or int(e[n].argmax()) == gen[pos + n]):
                n += 1
        # row 0 is always the first new token's logits (greedy: its argmax is gen[pos]); later rows extend the run
        out.extend(e[j] for j in range(n))
        counts.append(n)
        pos += n
        if pos >= len(gen):
            break
    return out, counts


def main(dp_size: int, dp_rank: int, args, ids: list[int]) -> None:
    import sm120_patches

    sm120_patches.install()  # no-op if sitecustomize already did it
    from _config import create_llm  # DeepInfer examples/_config.py

    import torch

    L = len(ids)
    nextn = int(os.environ.get("DEEPINFER_USE_NEXTN_DECODE", "0") or 0)
    if L + args.decode_steps + 2 + nextn > args.max_model_len:
        raise SystemExit(f"prompt {L} + {args.decode_steps} exceeds --max-model-len {args.max_model_len}")
    if L > args.max_num_batched_tokens:
        raise SystemExit(f"prompt {L} > --max-num-batched-tokens {args.max_num_batched_tokens} (would chunk prefill)")

    t0 = time.time()
    llm = create_llm(
        args.ckpt,
        dp_rank,
        ep_size=1,
        cp_size=1,
        data_parallel_size=dp_size,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        lookahead_tokens=min(1, nextn),  # as DeepInfer examples/benchmark.py:184
        disable_log_stats=True,
    )
    t_init = time.time() - t0

    import sm120_capture
    from deepinfer.sampling_params import SamplingParams

    model = llm.llm_engine.driver_worker.model_runner.model
    layers = list(model.model.layers)
    capture_layers = [] if args.no_hidden else parse_capture_layers(os.environ.get("CAPTURE_LAYER"), len(layers))
    hooks = []
    if not args.no_hidden:
        env_cl = os.environ.get("CAPTURE_LAYER")
        os.environ["CAPTURE_LAYER"] = "-1"  # the shim's own single-layer taps stay off; installed below instead
        try:
            hooks = sm120_capture.install_model_hooks(model)
        finally:
            if env_cl is None:
                del os.environ["CAPTURE_LAYER"]
            else:
                os.environ["CAPTURE_LAYER"] = env_cl

    golden = None
    if args.teacher_force:
        ref = torch.load(args.ref, map_location="cpu", weights_only=False)
        golden = [int(x) for x in ref["decoded_ids"].tolist()][: args.decode_steps + 1]
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=args.decode_steps + 1,
        ignore_eos=True,
        top_logprobs=0,
        golden_token_ids=golden,
        seq_seed=0,
    )
    sm120_capture.arm(L, keep_full_prefill=not args.compact)
    accept_log, accept_hook = install_accept_log(sm120_capture, model)
    step_times = model.sampler.forward.times
    hooks.append(accept_hook)
    hooks += install_fine_taps(sm120_capture, layers, capture_layers)  # after arm(): it resets STATE["errors"]
    t1 = time.time()
    outputs = llm.generate(prompt_token_ids=[ids], sampling_params=sp, use_tqdm=False)
    torch.cuda.synchronize()
    t_gen = time.time() - t1
    sm120_capture.disarm()
    for h in reversed(hooks):  # module-level overrides of later layers wrap those of earlier ones
        h.remove()

    st = sm120_capture.STATE
    out0 = outputs[0]
    gen = [int(x.token_id) for x in out0] if not isinstance(out0, str) else []
    S = args.decode_steps
    entries = sm120_capture.decode_rows() if hasattr(sm120_capture, "decode_rows") else [d[:1] for d in st["decode"]]
    errors = list(st["errors"])
    accepted = None
    if nextn > 0 and len(accept_log) == len(entries) and all(a for a in accept_log):
        accepted = [a[0] for a in accept_log]  # max_num_seqs=1: the only real sequence
    elif nextn > 0:
        errors.append(f"accepted-draft log unusable ({len(accept_log)} entries for {len(entries)} decode captures); "
                      "emitted tokens per step inferred from argmax runs")
    if nextn > 0:
        rows, per_step = assign_decode_rows(entries, gen, accepted)
        if sum(per_step) != len(gen) - 1:
            errors.append(f"MTP row assignment: {sum(per_step)} rows for {len(gen) - 1} decoded tokens")
    else:
        rows = [e[0] for e in entries][: max(len(gen) - 1, 0)]
        per_step = [1] * len(rows)
    dec = rows[:S]
    if st["prefill"] is None:
        errors.append("prefill logits were not captured")
    if len(dec) < S:
        errors.append(f"only {len(dec)} decode logit rows for {S} steps")
    if nextn == 0:  # with MTP the sampled tokens of a step are the accepted prefix, not one argmax per entry
        for s, d in enumerate(dec):
            if s + 1 < len(gen) and golden is None and int(d.argmax()) != gen[s + 1]:
                errors.append(f"decode step {s}: argmax {int(d.argmax())} != sampled token {gen[s + 1]}")
    # steps that emitted tokens; the engine may run one more captured step after the last kept token
    n_steps = len(per_step)
    mtp = dict(
        nextn=nextn,
        decode_steps=n_steps,
        decode_captures=len(entries),  # placeholder steps excluded (sm120_capture)
        rows_per_step=[int(e.shape[0]) for e in entries],
        tokens_per_step=(len(gen) - 1) / n_steps if n_steps else None,
        draft_acceptance=((len(gen) - 1) / n_steps - 1) / nextn if n_steps and nextn else None,
        emitted_per_step=per_step,
        emitted_source="num_accepted_tokens_gpu" if accepted is not None else ("argmax-run" if nextn else "one row/step"),
        accepted_log=accept_log,
    )
    # wall time between consecutive real sampler calls (decode steps; the first one is prefill -> first decode)
    ts = [t for t, _ in step_times]
    dts = [b - a for a, b in zip(ts, ts[1:])]
    t_first = next((t for t, p in step_times if p), None)
    timing = dict(
        t_generate_to_prefill_sample=(t_first - t1) if t_first else None,
        decode_step_s=dts,
        decode_step_median_s=sorted(dts)[len(dts) // 2] if dts else None,
    )
    V = dec[0].numel() if dec else (st["prefill"]["prefill_logits_last64"].shape[-1] if st["prefill"] else 0)
    dec_t = torch.stack(dec) if dec else torch.empty(0, V)
    top = dec_t.topk(50, dim=-1) if dec else None

    res = dict(st["prefill"] or {})
    res.update(
        format="sm120-ref-v1",
        producer="deepinfer-sm120",
        input_ids=torch.tensor(ids, dtype=torch.long),
        decoded_ids=torch.tensor(gen, dtype=torch.long),
        decode_positions=torch.tensor([L + s for s in range(len(dec))], dtype=torch.long),
        decode_top50_values=top.values if top is not None else torch.empty(0, 50),
        decode_top50_ids=top.indices if top is not None else torch.empty(0, 50, dtype=torch.long),
        decode_logits_first4=dec_t[:4].clone(),
        decode_logits_all=dec_t,
        prefill_sampled_logits=st["prefill_sampled"][0] if st["prefill_sampled"] else None,
        hidden=None if args.no_hidden else {"prefill": dict(st["hidden"]), "decode": []},
    )
    import deep_gemm

    missing = [li for li in capture_layers if f"L{li}/attn_out" not in st["hidden"]]
    if missing:
        errors.append(f"fine taps requested but not recorded for layers {missing}")
    trunc_path = os.path.join(args.ckpt, "truncation.json")
    res["meta"] = dict(
        teacher_forced=golden is not None,
        n_prompt=L,
        n_layers=len(layers),
        capture_layers=capture_layers,
        truncation=json.load(open(trunc_path)) if os.path.isfile(trunc_path) else None,
        timings=dict(init_s=t_init, generate_s=t_gen),
        env={k: v for k, v in os.environ.items() if k.startswith(("SM120_", "DEEPINFER_", "B12X_"))},
        patches=list(sm120_patches.APPLIED),
        errors=errors,
        n_decode_captures=len(st["decode"]),
        decode_meta=st.get("decode_meta"),
        n_placeholder_skipped=st.get("n_placeholder_skipped"),
        mtp=mtp,
        step_timing=timing,
        graphs=os.environ.get("DEEPINFER_ENFORCE_EAGER") == "0",
        gpu=torch.cuda.get_device_name(),
        capability=torch.cuda.get_device_capability(),
        torch=torch.__version__,
        deep_gemm=deep_gemm.__version__,
        peak_mem_gib=torch.cuda.max_memory_allocated() / 2**30,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    torch.save(res, args.out)
    tok = llm.get_tokenizer()
    print(f"[run_deepinfer] decoded ids: {gen}")
    try:
        print(f"[run_deepinfer] text: {tok.decode(gen)!r}")
    except Exception:  # noqa: BLE001
        pass
    print(f"[run_deepinfer] hidden keys: {sorted(st['hidden'])}")
    print(f"[run_deepinfer] nextn={nextn} decode steps={n_steps} tokens/step={mtp['tokens_per_step']} "
          f"draft acceptance={mtp['draft_acceptance']} emitted/step={per_step} rows/step={mtp['rows_per_step']} placeholder skipped="
          f"{st.get('n_placeholder_skipped')}")
    for e in errors:
        print(f"[run_deepinfer] WARNING: {e}")
    if timing["decode_step_median_s"] is not None:
        print(f"[run_deepinfer] prefill sample after {timing['t_generate_to_prefill_sample']:.3f}s, decode step median "
              f"{timing['decode_step_median_s'] * 1e3:.1f} ms over {len(dts)} steps")
    print(f"[run_deepinfer] init {t_init:.1f}s generate {t_gen:.1f}s -> wrote {args.out}")
    llm.close()


if __name__ == "__main__":
    _args = parse_args()
    setup_env(_args)
    _ids = load_prompt(_args)
    from _launch import launch_dp  # DeepInfer examples/_launch.py

    launch_dp(lambda dp_size, dp_rank: main(dp_size, dp_rank, _args, _ids), 1)
