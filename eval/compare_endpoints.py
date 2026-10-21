#!/usr/bin/env python3
"""Greedy A/B comparison of two OpenAI-compatible endpoints (e.g. DeepInfer-SM120 vs a vLLM baseline deployment).

    python3 compare_endpoints.py --a http://host:8000 --b http://host:8001 --tokenizer-dir /models/DeepSeek-V4.1-Flash \
        [--prompts prompts.jsonl] [--max-tokens 128] [--top-logprobs 5] [--json out.json]

Both endpoints get the *same token ids*: each prompt (chat messages or raw text) is rendered locally with the
checkpoint's encoding/encoding.py (chat mode) and tokenised with its tokenizer.json, then sent to /v1/completions as a
list of token ids. Chat templates and BOS handling therefore cannot differ between the stacks. Request: temperature 0,
`logprobs: K`, `ignore_eos: true` (so both generate exactly max_tokens), `return_tokens_as_token_ids: true` (vLLM and
server.py both answer with "token_id:<n>" strings).

Per prompt:
- exact match;
- first divergence position;
- position-wise agreement;
- before the divergence: |logprob_A - logprob_B| of the chosen token, top-K set overlap, |delta| on shared top-K tokens;
- at the divergence: both sides' top-1/top-2 margin. A divergence at a near-tie (margin < --tie, default 0.1 nats) is
  expected numerics, not a bug.

--teacher-forced (numerical fidelity over long continuations; implied by --reference): free-running greedy outputs
stop being comparable at the first divergence. Here every endpoint scores the SAME continuation instead: the greedy
output of --cont-from (a, b or ref; default ref if given, else a), or a prompt item's own "continuation" text. Each
position j is scored with a 1-token request on prompt + continuation[:j], asking for the top-K logprobs (prefix caches
make that cheap; --parallel requests in flight). --reference is a third source both endpoints are measured against:
a file (DeepSeek's reference implementation via tools/ref/ref_to_fidelity.py, or a --save-a/--save-b recording) or
an endpoint URL; the summary then says which of A and B is closer to it. Per pair (A-B, ref-A, ref-B) and prompt:
- top-1 agreement (same argmax at a position);
- KL(P||Q) on the union of both top-K sets (a token missing on one side gets that side's K-th logprob; both sides
  renormalised over the union);
- |delta logprob| of the forced continuation token;
and per endpoint the mean NLL of the forced continuation.
Files: --save-a / --save-b write the scored positions in the reference format; --a / --b file:PATH replay them, so
endpoints that cannot run at the same time (same GPUs) are compared later. Reference format (JSON):
  {"model": str, "top_k": int, "runs": [{"ids": [prompt ids], "cont": [token ids], "top": [{token_id: logprob}, ...]}]}
"""

from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import math
import importlib.util
import json
import os
import statistics
import sys
import urllib.parse

DEFAULT_PROMPTS = [
    {"messages": [{"role": "user", "content": "Explain in three sentences why the sky is blue."}]},
    {"messages": [{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number."}]},
    {"messages": [{"role": "user", "content": "What is 37 * 41? Show the steps."}]},
    {"messages": [{"role": "user", "content": "Erkläre kurz auf Deutsch, was ein Leuchtturm ist."}]},
    {"messages": [{"role": "user", "content": "List five prime numbers greater than 100."}]},
    {"messages": [{"role": "user", "content": "Summarize the plot of Romeo and Juliet in two sentences."}]},
    {"text": "The lighthouse at the end of the northern breakwater was built in 1872, at a time when"},
    {"text": "def quicksort(arr):\n    "},
]


class Recorded:
    """A previous run of one endpoint (file:PATH written with --save-a/--save-b): replays its parsed responses, so two
    servers that cannot run at the same time (one GPU) can still be compared."""

    def __init__(self, path: str):
        d = json.load(open(path))
        self.url, self.model, self.max_tokens, self.top_k = f"file:{path}", d["model"], d["max_tokens"], d["top_k"]
        self.runs = {tuple(r["ids"]): (r["parsed"][0], r["parsed"][1], [{int(k): v for k, v in t.items()} for t in r["parsed"][2]])
                     for r in d["runs"]}

    def complete_parsed(self, ids, max_tokens, k):
        if max_tokens != self.max_tokens or k != self.top_k:
            raise SystemExit(f"{self.url} was recorded with max_tokens {self.max_tokens}, top_logprobs {self.top_k}")
        try:
            return self.runs[tuple(ids)]
        except KeyError:
            raise SystemExit(f"{self.url}: no recorded run for a prompt of {len(ids)} tokens (different prompt set?)") from None


class Endpoint:
    def __init__(self, url: str, model: str | None, timeout: float):
        u = urllib.parse.urlparse(url)
        self.url, self.host, self.port, self.timeout = url, u.hostname, u.port or 80, timeout
        self.prefix = u.path.rstrip("/").removesuffix("/v1")  # accept http://host:port and http://host:port/v1
        self.model = model or self._get("/v1/models")["data"][0]["id"]

    def _req(self, method, path, body=None):
        c = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        c.request(method, self.prefix + path, json.dumps(body) if body is not None else None,
                  {"Content-Type": "application/json"} if body is not None else {})
        r = c.getresponse()
        data = r.read()
        if r.status != 200:
            raise RuntimeError(f"{self.url}{path}: HTTP {r.status}: {data[:400]!r}")
        return json.loads(data)

    def _get(self, path):
        return self._req("GET", path)

    def complete_parsed(self, ids, max_tokens, k):
        return parse(self.complete(ids, max_tokens, k))

    def complete(self, ids, max_tokens, k):
        return self._req("POST", "/v1/completions", dict(
            model=self.model, prompt=ids, max_tokens=max_tokens, temperature=0.0, logprobs=k, ignore_eos=True,
            return_tokens_as_token_ids=True))


def _tid(tok: str):
    return int(tok.split(":", 1)[1]) if isinstance(tok, str) and tok.startswith("token_id:") else tok


def parse(resp):
    """-> (token ids, chosen logprobs, [ {id: logprob} top-K per position ])"""
    lp = resp["choices"][0].get("logprobs") or {}
    toks = [_tid(t) for t in lp.get("tokens") or []]
    chosen = lp.get("token_logprobs") or [None] * len(toks)
    tops = [{_tid(k): v for k, v in (d or {}).items()} for d in (lp.get("top_logprobs") or [None] * len(toks))]
    return toks, chosen, tops


def margin(top: dict):
    v = sorted(top.values(), reverse=True)
    return (v[0] - v[1]) if len(v) >= 2 else None


def compare_one(a, b, tie: float):
    ta, ca, pa = a
    tb, cb, pb = b
    n = min(len(ta), len(tb))
    div = next((i for i in range(n) if ta[i] != tb[i]), None)
    same_until = div if div is not None else n
    dchosen = [abs(ca[i] - cb[i]) for i in range(same_until) if ca[i] is not None and cb[i] is not None]
    overlap, dtop = [], []
    for i in range(same_until):
        sa, sb = set(pa[i]), set(pb[i])
        if sa and sb:
            overlap.append(len(sa & sb) / len(sa | sb))
            dtop += [abs(pa[i][t] - pb[i][t]) for t in sa & sb]
    res = dict(len_a=len(ta), len_b=len(tb), exact=ta == tb, first_divergence=div,
               agreement=sum(int(x == y) for x, y in zip(ta, tb)) / max(1, n),
               chosen_lp_delta_mean=statistics.fmean(dchosen) if dchosen else None,
               chosen_lp_delta_max=max(dchosen) if dchosen else None,
               topk_jaccard_mean=statistics.fmean(overlap) if overlap else None,
               topk_lp_delta_mean=statistics.fmean(dtop) if dtop else None,
               topk_lp_delta_max=max(dtop) if dtop else None)
    if div is not None:
        ma, mb = margin(pa[div]), margin(pb[div])
        res.update(margin_a=ma, margin_b=mb,
                   near_tie=(ma is not None and ma < tie) or (mb is not None and mb < tie),
                   a_token=ta[div], b_token=tb[div],
                   b_token_in_a_top=tb[div] in pa[div], a_token_in_b_top=ta[div] in pb[div])
    return res


def load_codec(tokenizer_dir: str):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json"))
    enc = None
    p = os.path.join(tokenizer_dir, "encoding", "encoding.py")
    if os.path.isfile(p):
        spec = importlib.util.spec_from_file_location("ds41_encoding", p)
        enc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(enc)

    def render(item) -> list[int]:
        if "ids" in item:
            return list(item["ids"])
        if "messages" in item:
            if enc is None:
                raise SystemExit(f"{p} not found: cannot render chat prompts")
            text = enc.encode_messages(item["messages"], thinking_mode=item.get("thinking_mode", "chat"))
        else:
            text = item["text"]
        return tok.encode(text, add_special_tokens=False).ids

    return render, tok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="endpoint A base URL (e.g. DeepInfer-SM120)")
    ap.add_argument("--b", default=None, help="endpoint B base URL (e.g. vLLM baseline) or file:PATH; omit to only record A")
    ap.add_argument("--save-a", help="record A's parsed responses to this json (replay later as --a file:PATH)")
    ap.add_argument("--save-b")
    ap.add_argument("--model-a")
    ap.add_argument("--model-b")
    ap.add_argument("--tokenizer-dir", required=True, help="checkpoint dir with tokenizer.json (+ encoding/encoding.py)")
    ap.add_argument("--prompts", help="jsonl: {messages:[...]} | {text:...} | {ids:[...]} per line (default: built-in 8)")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--top-logprobs", type=int, default=5)
    ap.add_argument("--tie", type=float, default=0.1, help="top1-top2 logprob margin below which a divergence is a near-tie")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--json")
    ap.add_argument("--teacher-forced", action="store_true", help="score a fixed continuation position by position")
    ap.add_argument("--reference", help="teacher-forced: a third source both endpoints are measured against: a reference "
                    "file (tools/ref/ref_to_fidelity.py, or a --save-a/--save-b recording) or an endpoint URL")
    ap.add_argument("--cont-from", choices=("a", "b", "ref"), default=None,
                    help="teacher-forced: whose greedy output is the continuation (default: ref if given, else a); "
                         "prompt items with a \"continuation\" text use that instead")
    ap.add_argument("--parallel", type=int, default=8, help="teacher-forced: requests in flight per endpoint")
    args = ap.parse_args()

    render, tok = load_codec(args.tokenizer_dir)
    if args.teacher_forced or args.reference:
        return teacher_forced(args, render, tok)
    items = [json.loads(l) for l in open(args.prompts) if l.strip()] if args.prompts else DEFAULT_PROMPTS
    def make(spec, model):
        return Recorded(spec[5:]) if spec.startswith("file:") else Endpoint(spec, model, args.timeout)

    A = make(args.a, args.model_a)
    B = make(args.b, args.model_b) if args.b else None
    print(f"A = {A.url} ({A.model})\nB = {B.url + ' (' + B.model + ')' if B else '- (record only)'}\n{len(items)} prompts, "
          f"{args.max_tokens} tokens, greedy, top-{args.top_logprobs} logprobs")
    rows, rec_a, rec_b = [], [], []
    for i, it in enumerate(items):
        ids = render(it)
        ra = A.complete_parsed(ids, args.max_tokens, args.top_logprobs)
        rec_a.append(dict(ids=ids, parsed=ra))
        if B is None:
            print(f"[{i}] {len(ids):5d} prompt tok | A: {len(ra[0])} tokens recorded", flush=True)
            continue
        rb = B.complete_parsed(ids, args.max_tokens, args.top_logprobs)
        rec_b.append(dict(ids=ids, parsed=rb))
        if not ra[0] or not rb[0] or not all(isinstance(t, int) for t in ra[0] + rb[0]):
            raise SystemExit(f"prompt {i}: an endpoint returned no token ids (needs logprobs + return_tokens_as_token_ids)")
        r = compare_one(ra, rb, args.tie)
        r.update(prompt=i, prompt_tokens=len(ids))
        rows.append(r)
        d = r["first_divergence"]
        print(f"[{i}] {len(ids):5d} prompt tok | exact {str(r['exact']):5s} | first divergence {str(d):>4s} | agreement "
              f"{r['agreement']:.3f} | chosen-lp delta mean {r['chosen_lp_delta_mean'] or 0:.4f} max "
              f"{r['chosen_lp_delta_max'] or 0:.4f} | top-k jaccard {r['topk_jaccard_mean'] or 0:.3f}"
              + (f" | at divergence: margins A {r['margin_a']:.3f} B {r['margin_b']:.3f} "
                 f"{'(near-tie)' if r['near_tie'] else '(NOT a tie)'} A={tok.decode([r['a_token']])!r} "
                 f"B={tok.decode([r['b_token']])!r}" if d is not None and r.get('margin_a') is not None else ""),
              flush=True)
    for path, ep, rec in ((args.save_a, A, rec_a), (args.save_b, B, rec_b)):
        if path and ep is not None:
            # parsed = [ids, logprobs, [{id: lp}]]; json keys become strings -> converted back on replay
            json.dump(dict(model=ep.model, url=ep.url, max_tokens=args.max_tokens, top_k=args.top_logprobs,
                           runs=[dict(ids=r["ids"], parsed=[r["parsed"][0], r["parsed"][1], r["parsed"][2]]) for r in rec]),
                      open(path, "w"))
            print(f"recorded {ep.url} -> {path}")
    if B is None:
        return 0
    n = len(rows)
    divs = [r for r in rows if r["first_divergence"] is not None]
    summary = dict(
        prompts=n, exact=sum(r["exact"] for r in rows), mean_agreement=statistics.fmean(r["agreement"] for r in rows),
        median_first_divergence=statistics.median([r["first_divergence"] for r in divs]) if divs else None,
        divergences_at_near_tie=sum(1 for r in divs if r.get("near_tie")), divergences=len(divs),
        chosen_lp_delta_mean=statistics.fmean([r["chosen_lp_delta_mean"] for r in rows if r["chosen_lp_delta_mean"] is not None] or [0]),
        topk_jaccard_mean=statistics.fmean([r["topk_jaccard_mean"] for r in rows if r["topk_jaccard_mean"] is not None] or [0]),
    )
    print("[summary]", json.dumps(summary))
    if args.json:
        json.dump(dict(a=A.url, b=B.url, model_a=A.model, model_b=B.model, args=vars(args), summary=summary, rows=rows),
                  open(args.json, "w"), indent=1)
    return 0


# ------------------------------------------------------------------------------------------------ teacher-forced fidelity
def load_fidelity(path: str) -> dict:
    d = json.load(open(path))
    for r in d["runs"]:
        r["top"] = [{int(k): v for k, v in t.items()} for t in r["top"]]
    return d


def score_positions(ep, ids, cont, k, parallel):
    """Top-K logprobs of every continuation position j given prompt + cont[:j] (1-token requests)."""
    def one(j):
        toks, _, tops = parse(ep.complete(list(ids) + list(cont[:j]), 1, k))
        return tops[0] if tops else {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as pool:
        return list(pool.map(one, range(len(cont))))


def _f4(x):
    return "-" if x is None else f"{x:.4f}"


def forced_nll(tops, cont):
    """Mean -log p of the forced tokens (a token outside the top-K counts with the K-th logprob, an upper bound)."""
    vals = [-(t.get(c, min(t.values()))) for t, c in zip(tops, cont) if t]
    return statistics.fmean(vals) if vals else None


def kl_topk(p: dict, q: dict):
    """KL(P||Q) on the union of both top-K sets: a token missing on one side gets that side's K-th logprob (an upper
    bound of its true value), then both are renormalised over the union. Always >= 0."""
    if not p or not q:
        return None
    toks = set(p) | set(q)
    fp, fq = min(p.values()), min(q.values())
    lp = {t: p.get(t, fp) for t in toks}
    lq = {t: q.get(t, fq) for t in toks}
    zp = math.log(sum(math.exp(v) for v in lp.values()))
    zq = math.log(sum(math.exp(v) for v in lq.values()))
    return max(0.0, sum(math.exp(lp[t] - zp) * ((lp[t] - zp) - (lq[t] - zq)) for t in toks))


def fidelity_pair(name, P, Q, cont):
    n = min(len(P), len(Q))
    top1 = [max(P[j], key=P[j].get) == max(Q[j], key=Q[j].get) for j in range(n) if P[j] and Q[j]]
    kls = [x for x in (kl_topk(P[j], Q[j]) for j in range(n)) if x is not None]
    dlp = [abs(P[j][cont[j]] - Q[j][cont[j]]) for j in range(n) if cont[j] in P[j] and cont[j] in Q[j]]
    srt = sorted(kls)
    return dict(pair=name, positions=n, top1_agreement=statistics.fmean(top1) if top1 else None,
                kl_mean=statistics.fmean(kls) if kls else None, kl_p99=srt[int(0.99 * (len(srt) - 1))] if srt else None,
                kl_max=srt[-1] if srt else None, forced_lp_delta_mean=statistics.fmean(dlp) if dlp else None)


def teacher_forced(args, render, tok) -> int:
    """Score fixed continuations on A, B and an optional reference (see the module docstring)."""
    def source(spec, model):
        if spec is None:
            return None
        return ("file", load_fidelity(spec[5:] if spec.startswith("file:") else spec)) \
            if spec.startswith("file:") or not spec.startswith(("http://", "https://")) \
            else ("live", Endpoint(spec, model, args.timeout))

    A, B, R = source(args.a, args.model_a), source(args.b, args.model_b), source(args.reference, None)
    if args.prompts:
        items = [json.loads(l) for l in open(args.prompts) if l.strip()]
    elif R and R[0] == "file":
        items = [{"ids": r["ids"]} for r in R[1]["runs"]]
    else:
        items = DEFAULT_PROMPTS
    cont_from = args.cont_from or ("ref" if R else "a")
    srcs = {"a": A, "b": B, "ref": R}
    if srcs.get(cont_from) is None:
        raise SystemExit(f"--cont-from {cont_from}: that source is not given")

    def file_run(src, ids, cont=None):
        run = next((x for x in src[1]["runs"] if x["ids"] == list(ids)
                    and (cont is None or x["cont"][:len(cont)] == list(cont))), None)
        return run

    rows, rec = [], {"a": [], "b": [], "ref": []}
    for i, it in enumerate(items):
        ids = render(it)
        if it.get("continuation"):  # a fixed (e.g. human-written) continuation
            cont = tok.encode(it["continuation"], add_special_tokens=False).ids[:args.max_tokens]
        else:
            src = srcs[cont_from]
            if src[0] == "file":
                run = file_run(src, ids)
                if run is None:
                    raise SystemExit(f"prompt {i}: no run in {cont_from} file")
                cont = run["cont"][:args.max_tokens]
            else:
                cont = parse(src[1].complete(ids, args.max_tokens, 1))[0]
        tops = {}
        for key, src in srcs.items():
            if src is None:
                continue
            if src[0] == "file":
                run = file_run(src, ids, cont)
                if run is None or len(run["top"]) < len(cont):
                    raise SystemExit(
                        f"prompt {i}: the {key} file has no scoring of this continuation. A recording only holds the "
                        f"continuations it was made with (its own greedy output with --cont-from {key}, or the one "
                        f"given then). To compare on another endpoint's continuation, score {key} live, or record it "
                        f"with the same --cont-from source (e.g. --a <url> --b file:<other.json> --cont-from b "
                        f"--save-a ...).")
                tops[key] = run["top"][:len(cont)]
            else:
                tops[key] = score_positions(src[1], ids, cont, args.top_logprobs, args.parallel)
            rec[key].append(dict(ids=list(ids), cont=list(cont), top=tops[key]))
        row = dict(prompt=i, prompt_tokens=len(ids), positions=len(cont), cont_from=cont_from, pairs=[],
                   nll={k: forced_nll(t, cont) for k, t in tops.items()})
        if "a" in tops and "b" in tops:
            row["pairs"].append(fidelity_pair("A-B", tops["a"], tops["b"], cont))
        if "ref" in tops:
            for key in ("a", "b"):
                if key in tops:
                    row["pairs"].append(fidelity_pair(f"ref-{key.upper()}", tops["ref"], tops[key], cont))
        rows.append(row)
        print(f"[{i}] {len(ids):5d} prompt tok, {len(cont)} positions | " + " | ".join(
            f"{p['pair']}: top1 {p['top1_agreement']:.3f} KL mean {p['kl_mean']:.4f} p99 {p['kl_p99']:.4f}"
            for p in row["pairs"] if p["top1_agreement"] is not None) + " | NLL " + " ".join(
            f"{k} {v:.3f}" for k, v in row["nll"].items() if v is not None), flush=True)
    for key, path in (("a", args.save_a), ("b", args.save_b)):
        src = srcs[key]
        if path and src is not None and rec[key]:
            json.dump(dict(model=src[1].model if src[0] == "live" else src[1].get("model"), top_k=args.top_logprobs,
                           runs=[dict(ids=x["ids"], cont=x["cont"], top=[{str(t): v for t, v in d.items()} for d in x["top"]])
                                 for x in rec[key]]), open(path, "w"))
            print(f"recorded {key.upper()} -> {path}")
    summary = []
    for name in sorted({p["pair"] for r in rows for p in r["pairs"]}):
        ps = [p for r in rows for p in r["pairs"] if p["pair"] == name and p["top1_agreement"] is not None]
        if ps:
            summary.append(dict(pair=name, prompts=len(ps), positions=sum(p["positions"] for p in ps),
                                top1_agreement=statistics.fmean(p["top1_agreement"] for p in ps),
                                kl_mean=statistics.fmean(p["kl_mean"] for p in ps), kl_p99_max=max(p["kl_p99"] for p in ps),
                                forced_lp_delta_mean=(statistics.fmean(d) if (d := [p["forced_lp_delta_mean"] for p in ps
                                                      if p["forced_lp_delta_mean"] is not None]) else None)))
    nll = {k: statistics.fmean([r["nll"][k] for r in rows if r["nll"].get(k) is not None] or [float("nan")])
           for k in ("a", "b", "ref") if any(k in r["nll"] for r in rows)}
    md = [f"Continuation from: {cont_from}" + (" (prompts' own continuations where given)" if args.prompts else ""), "",
          "| pair | prompts | positions | top-1 agreement | KL mean | KL p99 (worst prompt) | forced-token lp delta |",
          "|---|---|---|---|---|---|---|"]
    md += [f"| {s['pair']} | {s['prompts']} | {s['positions']} | {s['top1_agreement']:.4f} | {s['kl_mean']:.5f} | "
           f"{s['kl_p99_max']:.4f} | {_f4(s['forced_lp_delta_mean'])} |" for s in summary]
    md += ["", "Mean NLL of the forced continuation (nats/token, lower = the endpoint finds it more likely): " +
           ", ".join(f"{k.upper()} {v:.4f}" for k, v in nll.items())]
    verdict = None
    by = {s["pair"]: s for s in summary}
    if "ref-A" in by and "ref-B" in by:
        a_, b_ = by["ref-A"], by["ref-B"]
        closer = "A" if a_["kl_mean"] < b_["kl_mean"] else "B"
        top1_closer = "A" if a_["top1_agreement"] > b_["top1_agreement"] else \
            "B" if b_["top1_agreement"] > a_["top1_agreement"] else closer
        if top1_closer != closer:
            closer = f"inconclusive (KL favours {closer}, top-1 agreement favours {top1_closer})"
        verdict = dict(closer_to_reference=closer, kl_ref_a=a_["kl_mean"], kl_ref_b=b_["kl_mean"],
                       top1_ref_a=a_["top1_agreement"], top1_ref_b=b_["top1_agreement"])
        md += ["", f"**Closer to the reference: {closer}** (KL mean ref-A {a_['kl_mean']:.5f} vs ref-B {b_['kl_mean']:.5f}; "
               f"top-1 agreement {a_['top1_agreement']:.4f} vs {b_['top1_agreement']:.4f})"]
    print("\n" + "\n".join(md))
    if args.json:
        json.dump(dict(a=args.a, b=args.b, reference=args.reference, args=vars(args), summary=summary, nll=nll,
                       verdict=verdict, rows=rows), open(args.json, "w"), indent=1)
        open(os.path.splitext(args.json)[0] + ".md", "w").write("\n".join(md) + "\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
