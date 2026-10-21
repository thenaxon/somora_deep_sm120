#!/usr/bin/env python3
"""Workload scenarios for any OpenAI-compatible endpoint (stdlib only; threads): where an engine's design shows.

    python3 scenarios.py <scenario> --url http://host:8000 [--out results/name] [scenario options]

Scenarios (all streaming /v1/chat/completions with stream_options.include_usage):

  realistic      Real prompts (eval/prompts.json: code, German chat, English chat) at concurrency 1/4/16. Speculative
                 decoding (MTP / DSpark, EAGLE, ...) depends on the text, so random token ids are not a fair test of it.
                 Reports output tok/s, TPOT, TTFT, and the engine's draft acceptance where it exposes it.
  prefill        Many users with long distinct prompts (8k/16k/32k tokens) all at once (16/32/64 users):
                 TTFT p50/p99/max, time until the last request is done, prompt + output tok/s.
  shared-prefix  Tool-using assistant workload: N users share one long system prompt (tool definitions + instructions, ~6k tokens)
                 with short distinct user turns, and continue their conversations for several turns. Run once with a
                 shared and once with a per-user system prompt: the difference is what prefix caching buys.
                 Reports TTFT per turn, and cache statistics if the endpoint exposes them.
  mixed          Short chats stream continuously while long-prompt requests arrive. Baseline phase without the long
                 requests, then with them: TTFT, TPOT p99 and the longest stall of the short streams (head-of-line
                 blocking by prefills).
  capacity       Ramp concurrency until TTFT p99 > --slo-ttft (s) or TPOT p99 > --slo-tpot (ms): the largest number of
                 concurrent users that meets both, and the throughput there.

Fairness: every scenario builds its prompts from --seed only (no endpoint input), so two endpoints get identical
requests. Greedy (temperature 0) unless --temperature. The same max_tokens everywhere; ignore_eos only where stated
(--ignore-eos). Synthetic long texts are sized by characters (--chars-per-token, default 3.6 for V4.1); the tables
report the real prompt token counts from `usage`.

Engine statistics: before/after each level the script reads GET /health (this repo's server: engine steps, prefix cache
hits) and GET /metrics (Prometheus, e.g. vLLM: prefix cache and speculative decoding counters) and reports the deltas
it understands; endpoints without them just get no such columns.

Output: a table on stdout, and with --out PATH: PATH.json (all numbers, per-request samples) and PATH.md (the tables).
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import random
import re
import statistics
import sys
import threading
import time
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = os.path.join(HERE, "prompts.json")


# ------------------------------------------------------------------------------------------------ helpers
def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return xs[min(len(xs) - 1, max(0, int(round(q / 100 * (len(xs) - 1)))))]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def fmt(x, nd=2):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


class Endpoint:
    def __init__(self, url: str, model: str | None, timeout: float):
        u = urllib.parse.urlparse(url)
        self.url, self.host, self.port, self.timeout = url, u.hostname, u.port or 80, timeout
        self.prefix = u.path.rstrip("/").removesuffix("/v1")
        info = self.get_json("/v1/models")["data"][0]
        self.model = model or info["id"]
        self.max_model_len = info.get("max_model_len")

    def _get(self, path, timeout=30):
        c = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        c.request("GET", self.prefix + path)
        r = c.getresponse()
        return r.status, r.read()

    def get_json(self, path):
        st, data = self._get(path)
        if st != 200:
            raise RuntimeError(f"GET {path}: HTTP {st}")
        return json.loads(data)

    def stats(self) -> dict:
        """Numeric counters the endpoint exposes: /health (this repo's server) and /metrics (Prometheus)."""
        out = {}
        try:
            st, data = self._get("/health", 10)
            if st == 200:
                h = json.loads(data)
                for k, v in (h.get("stats") or {}).items():
                    if isinstance(v, (int, float)):
                        out[f"engine.{k}"] = v
                for k, v in (h.get("prefix_cache") or {}).items():
                    if isinstance(v, (int, float)):
                        out[f"prefix_cache.{k}"] = v
        except Exception:  # noqa: BLE001
            pass
        try:
            st, data = self._get("/metrics", 10)
            if st == 200:
                for line in data.decode(errors="replace").splitlines():
                    if line.startswith("#") or not line.strip():
                        continue
                    m = re.match(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eE]+|NaN|\+Inf)$", line.strip())
                    if not m or not re.search(r"prefix_cache|spec_decode|prompt_tokens|generation_tokens", m.group(1)):
                        continue
                    try:
                        out[f"metrics.{m.group(1)}"] = out.get(f"metrics.{m.group(1)}", 0.0) + float(m.group(3))
                    except ValueError:
                        pass
        except Exception:  # noqa: BLE001
            pass
        return out

    def chat(self, messages, max_tokens, temperature=0.0, ignore_eos=False, seed=None, extra=None) -> dict:
        """One streaming chat request. -> timings, token counts, inter-chunk gaps, text."""
        body = dict(model=self.model, messages=messages, max_tokens=max_tokens, temperature=temperature, stream=True,
                    stream_options={"include_usage": True})
        if ignore_eos:
            body["ignore_eos"] = True
        if seed is not None:
            body["seed"] = seed
        body.update(extra or {})
        c = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        t0 = time.perf_counter()
        c.request("POST", self.prefix + "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        r = c.getresponse()
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}: {r.read()[:200]!r}")
        stamps, usage, parts = [], None, []
        while True:
            line = r.fp.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data: "):
                continue
            if line[6:] == b"[DONE]":
                break
            ev = json.loads(line[6:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                txt = (d.get("content") or "") + (d.get("reasoning_content") or "")
                if txt:
                    stamps.append(time.perf_counter())
                    parts.append(d.get("content") or "")
        c.close()
        t_end = time.perf_counter()
        n_out = (usage or {}).get("completion_tokens") or len(stamps)
        n_in = (usage or {}).get("prompt_tokens")
        gaps = [b - a for a, b in zip(stamps, stamps[1:])]
        return dict(t0=t0, ttft=(stamps[0] - t0) if stamps else None, e2e=t_end - t0, t_end=t_end, out=n_out, inp=n_in,
                    tpot=((stamps[-1] - stamps[0]) / (n_out - 1)) if len(stamps) > 1 and n_out > 1 else None,
                    max_gap=max(gaps) if gaps else None, text="".join(parts))


def stats_delta(before: dict, after: dict) -> dict:
    d = {k: after[k] - before.get(k, 0) for k in after if isinstance(after[k], (int, float))}
    out = {}
    if "engine.steps" in d:
        out["engine_steps"] = d["engine.steps"]
    for k in ("prefix_cache.hits", "prefix_cache.lookups", "prefix_cache.hit_tokens", "prefix_cache.prompt_tokens"):
        if k in d:
            out[k.replace(".", "_")] = d[k]
    for k, v in d.items():  # Prometheus counters (vLLM names vary between versions: keep every matching one)
        if k.startswith("metrics.") and (k.endswith("_total") or "hits" in k or "queries" in k):
            out[k[len("metrics."):]] = v
    acc = [v for k, v in d.items() if "spec_decode_num_accepted_tokens" in k]
    draft = [v for k, v in d.items() if "spec_decode_num_draft_tokens" in k]
    if acc and draft and draft[0]:
        out["spec_acceptance"] = acc[0] / draft[0]
    hits = [v for k, v in d.items() if k.startswith("metrics.") and "prefix_cache_hits" in k]
    queries = [v for k, v in d.items() if k.startswith("metrics.") and "prefix_cache_queries" in k]
    if hits and queries and queries[0]:
        out["prefix_cache_token_hit_rate"] = hits[0] / queries[0]
    if out.get("prefix_cache_prompt_tokens"):
        out["prefix_cache_token_hit_rate"] = out.get("prefix_cache_hit_tokens", 0) / out["prefix_cache_prompt_tokens"]
    return out


def run_pool(jobs, conc):
    """Run callables with at most `conc` in flight (closed loop). -> (results, errors, wall seconds)."""
    results, errors = [], []
    lock = threading.Lock()
    it = iter(jobs)

    def worker():
        while True:
            with lock:
                job = next(it, None)
            if job is None:
                return
            try:
                res = job()
                with lock:
                    results.append(res)
            except Exception as exc:  # noqa: BLE001
                with lock:
                    errors.append(repr(exc)[:200])

    t0 = time.perf_counter()
    ths = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, conc))]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    return results, errors, time.perf_counter() - t0


def summarize(results, wall) -> dict:
    out = sum(r["out"] or 0 for r in results)
    inp = sum(r["inp"] or 0 for r in results)
    return dict(ok=len(results), wall_s=wall, output_tok_s=out / wall if wall else None,
                total_tok_s=(out + inp) / wall if wall else None, prompt_tokens_mean=mean([r["inp"] for r in results]),
                output_tokens_mean=mean([r["out"] for r in results]),
                ttft_p50=pct([r["ttft"] for r in results], 50), ttft_p99=pct([r["ttft"] for r in results], 99),
                ttft_max=max([r["ttft"] for r in results if r["ttft"] is not None], default=None),
                tpot_p50_ms=(pct([r["tpot"] for r in results], 50) or 0) * 1e3,
                tpot_p99_ms=(pct([r["tpot"] for r in results], 99) or 0) * 1e3,
                max_gap_p99_ms=(pct([r["max_gap"] for r in results], 99) or 0) * 1e3,
                e2e_p50=pct([r["e2e"] for r in results], 50), e2e_p99=pct([r["e2e"] for r in results], 99))


# ------------------------------------------------------------------------------------------------ prompt material
def load_set(name: str) -> list[dict]:
    return json.load(open(PROMPTS, encoding="utf-8"))[name]


_NOUNS = ("warehouse", "shipment", "invoice", "supplier", "customer", "turbine", "pipeline", "server", "contract",
          "forecast", "inventory", "budget", "sensor", "vehicle", "assembly line", "audit", "network", "reactor",
          "harbour", "railway", "laboratory", "pharmacy", "school", "library", "bakery", "orchard", "glacier")
_VERBS = ("increased", "decreased", "stabilised", "doubled", "fell sharply", "recovered", "stalled", "improved",
          "fluctuated", "was delayed", "was audited", "was replaced", "was expanded", "was relocated")
_PLACES = ("Linz", "Graz", "Salzburg", "Innsbruck", "Klagenfurt", "Bregenz", "Wels", "Krems", "Villach", "Steyr",
           "Dornbirn", "Leoben", "Hallstatt", "Zell am See", "Bruck", "Melk", "Tulln", "Baden", "Mödling", "Eisenstadt")
_NAMES = ("Anna Berger", "Lukas Huber", "Sophie Wagner", "David Gruber", "Lena Steiner", "Paul Bauer", "Marie Moser",
          "Felix Mayer", "Laura Hofer", "Jonas Pichler", "Emma Leitner", "Tobias Fuchs", "Clara Eder", "Elias Schmid")


def synthetic_document(seed: int, n_tokens: int, cpt: float) -> str:
    """A distinct, natural-language-looking report of about n_tokens tokens (first line unique per seed, so no two
    documents share a prefix)."""
    rng = random.Random(seed * 7919 + n_tokens)
    lines = [f"Report {seed:05d}-{rng.getrandbits(32):08x}: operations review for the {rng.choice(_PLACES)} region, "
             f"prepared by {rng.choice(_NAMES)}."]
    target = int(n_tokens * cpt)
    size = len(lines[0])
    sec = 0
    while size < target:
        sec += 1
        para = [f"\n\nSection {sec}: {rng.choice(_NOUNS).title()} in {rng.choice(_PLACES)}."]
        for _ in range(rng.randint(4, 8)):
            para.append(
                f" In week {rng.randint(1, 52)}, the {rng.choice(_NOUNS)} near {rng.choice(_PLACES)} {rng.choice(_VERBS)} "
                f"by {rng.randint(2, 95)} percent, according to {rng.choice(_NAMES)}, who reported "
                f"{rng.randint(10, 9999)} units and a cost of {rng.randint(1, 900)}.{rng.randint(0, 99):02d} thousand euros.")
        text = "".join(para)
        lines.append(text)
        size += len(text)
    return "".join(lines)[:target]


def tools_system_prompt(n_tokens: int, cpt: float, seed: int, unique_tag: str = "") -> str:
    """An assistant system prompt: instructions + tool definitions (JSON), about n_tokens tokens."""
    rng = random.Random(seed)
    head = (f"{unique_tag}You are an operations assistant for a logistics company. Use the tools below when a request "
            "needs live data or an action. Call at most three tools per turn, never invent tool results, and answer in "
            "the user's language. Confirm destructive actions before executing them. Keep answers short.\n\n## Tools\n")
    parts, size, i = [head], len(head), 0
    target = int(n_tokens * cpt)
    while size < target:
        i += 1
        noun = rng.choice(_NOUNS).replace(" ", "_")
        props = {f"{rng.choice(('id', 'name', 'region', 'date', 'limit', 'status', 'owner', 'priority'))}_{k}":
                 {"type": rng.choice(("string", "integer", "boolean")),
                  "description": f"The {rng.choice(_NOUNS)} {rng.choice(('identifier', 'filter', 'range', 'flag'))}."}
                 for k in range(rng.randint(2, 6))}
        tool = {"name": f"{rng.choice(('get', 'list', 'update', 'cancel', 'schedule', 'audit'))}_{noun}_{i}",
                "description": f"Works on {noun.replace('_', ' ')} records in {rng.choice(_PLACES)}; "
                               f"returns {rng.choice(('a summary', 'a list', 'the new state', 'a report'))}.",
                "parameters": {"type": "object", "properties": props, "required": list(props)[:1]}}
        s = json.dumps(tool, ensure_ascii=False) + "\n"
        parts.append(s)
        size += len(s)
    return "".join(parts)


USER_TURNS = [
    "Which shipments to Graz are delayed this week?",
    "Cancel the second one and tell me what that costs.",
    "Now schedule a replacement delivery for Friday morning.",
    "Summarise everything we changed today in two sentences.",
    "Is the warehouse in Linz over capacity?",
    "Please list the open invoices of the supplier from Wels.",
]


# ------------------------------------------------------------------------------------------------ scenarios
def sc_realistic(ep, a):
    rows = []
    sets = a.sets.split(",")
    for name in sets:
        items = load_set(name)
        for conc in [int(x) for x in a.concurrency.split(",")]:
            n = max(len(items), a.min_requests, 2 * conc)
            msgs = [items[i % len(items)]["messages"] for i in range(n)]
            jobs = [lambda m=m: ep.chat(m, a.max_tokens, a.temperature, a.ignore_eos, seed=a.seed) for m in msgs]
            s0 = ep.stats()
            res, err, wall = run_pool(jobs, conc)
            row = dict(set=name, concurrency=conc, requests=n, errors=len(err), error_samples=err[:2],
                       **summarize(res, wall), **stats_delta(s0, ep.stats()))
            if row.get("engine_steps") and conc == 1:  # one sequence at a time: tokens per engine step (MTP: >1)
                row["tokens_per_step"] = sum(r["out"] or 0 for r in res) / row["engine_steps"]
            rows.append(row)
            show(row, ["set", "concurrency", "requests", "errors", "output_tok_s", "output_tokens_mean", "tpot_p50_ms",
                       "tpot_p99_ms", "ttft_p50", "ttft_p99", "tokens_per_step", "spec_acceptance"])
    return rows


def sc_prefill(ep, a):
    rows = []
    for plen in [int(x) for x in a.prompt_lens.split(",")]:
        if ep.max_model_len and plen + a.max_tokens + 64 > ep.max_model_len:
            print(f"prompt {plen}: skipped (max_model_len {ep.max_model_len})")
            continue
        for conc in [int(x) for x in a.concurrency.split(",")]:
            docs = [synthetic_document(a.seed * 100003 + plen * 101 + conc * 7 + i, plen, a.chars_per_token)
                    for i in range(conc * a.rounds)]
            msgs = [[{"role": "user", "content": d + "\n\nSummarise the report above in three bullet points."}]
                    for d in docs]
            jobs = [lambda m=m: ep.chat(m, a.max_tokens, a.temperature, True, seed=a.seed) for m in msgs]
            s0 = ep.stats()
            res, err, wall = run_pool(jobs, conc)
            prompt_tok = sum(r["inp"] or 0 for r in res)
            row = dict(prompt_len=plen, concurrency=conc, requests=len(msgs), errors=len(err), error_samples=err[:2],
                       time_to_last_s=wall, prefill_tok_s=prompt_tok / wall if wall else None,
                       **summarize(res, wall), **stats_delta(s0, ep.stats()))
            rows.append(row)
            show(row, ["prompt_len", "concurrency", "errors", "prompt_tokens_mean", "ttft_p50", "ttft_p99", "ttft_max",
                       "time_to_last_s", "prefill_tok_s", "total_tok_s"])
    return rows


def sc_shared_prefix(ep, a):
    rows = []
    for variant in a.variants.split(","):
        per_turn: dict[int, list] = {}
        all_res, errs = [], []
        lock = threading.Lock()
        shared = tools_system_prompt(a.system_tokens, a.chars_per_token, a.seed + (0 if variant == "shared" else 1))

        def user(u):
            system = shared if variant == "shared" else tools_system_prompt(
                a.system_tokens, a.chars_per_token, a.seed + 1, unique_tag=f"[session {variant}-{a.seed}-{u:03d}] ")
            msgs = [{"role": "system", "content": system}]
            for t in range(a.turns):
                msgs.append({"role": "user", "content": f"(user {u}) " + USER_TURNS[(u + t) % len(USER_TURNS)]})
                try:
                    r = ep.chat(msgs, a.max_tokens, a.temperature, a.ignore_eos, seed=a.seed)
                except Exception as exc:  # noqa: BLE001
                    with lock:
                        errs.append(repr(exc)[:200])
                    return
                r["turn"] = t
                with lock:
                    per_turn.setdefault(t, []).append(r)
                    all_res.append(r)
                msgs.append({"role": "assistant", "content": r["text"] or "OK."})

        s0 = ep.stats()
        t0 = time.perf_counter()
        ths = [threading.Thread(target=user, args=(u,), daemon=True) for u in range(a.users)]
        for i, t in enumerate(ths):
            t.start()
            time.sleep(a.stagger)
        for t in ths:
            t.join()
        wall = time.perf_counter() - t0
        delta = stats_delta(s0, ep.stats())
        for t in sorted(per_turn):
            rs = per_turn[t]
            row = dict(variant=variant, turn=t + 1, users=a.users, requests=len(rs), errors=len(errs) if t == 0 else 0,
                       prompt_tokens_mean=mean([r["inp"] for r in rs]), ttft_p50=pct([r["ttft"] for r in rs], 50),
                       ttft_p99=pct([r["ttft"] for r in rs], 99), tpot_p50_ms=(pct([r["tpot"] for r in rs], 50) or 0) * 1e3)
            rows.append(row)
            show(row, ["variant", "turn", "requests", "errors", "prompt_tokens_mean", "ttft_p50", "ttft_p99",
                       "tpot_p50_ms"])
        tot = dict(variant=variant, turn="all", users=a.users, requests=len(all_res), errors=len(errs), wall_s=wall,
                   ttft_p50=pct([r["ttft"] for r in all_res], 50), ttft_p99=pct([r["ttft"] for r in all_res], 99),
                   prompt_tokens_total=sum(r["inp"] or 0 for r in all_res), **delta)
        rows.append(tot)
        show(tot, ["variant", "turn", "requests", "errors", "wall_s", "ttft_p50", "ttft_p99",
                   "prefix_cache_token_hit_rate", "prefix_cache_hits"])
    return rows


def sc_mixed(ep, a):
    rows = []
    short_items = load_set("chat_en") + load_set("chat_de")
    for phase in ("short only", "short + long"):
        stop = threading.Event()
        shorts, longs, errs = [], [], []
        lock = threading.Lock()
        rng = random.Random(a.seed)

        def short_user(u):
            k = 0
            while not stop.is_set():
                m = short_items[(u * 7 + k) % len(short_items)]["messages"]
                k += 1
                try:
                    r = ep.chat(m, a.short_tokens, a.temperature, True, seed=a.seed)
                    if not stop.is_set():
                        with lock:
                            shorts.append(r)
                except Exception as exc:  # noqa: BLE001
                    with lock:
                        errs.append(repr(exc)[:200])

        def long_req(i):
            doc = synthetic_document(a.seed * 31 + i + 1000 * (phase == "short + long"), a.long_tokens, a.chars_per_token)
            try:
                r = ep.chat([{"role": "user", "content": doc + "\n\nWhat is this report about? One sentence."}],
                            a.long_out, a.temperature, True, seed=a.seed)
                with lock:
                    longs.append(r)
            except Exception as exc:  # noqa: BLE001
                with lock:
                    errs.append(repr(exc)[:200])

        s0 = ep.stats()
        ths = [threading.Thread(target=short_user, args=(u,), daemon=True) for u in range(a.short_users)]
        for t in ths:
            t.start()
        t0 = time.perf_counter()
        lths = []
        if phase == "short + long":
            i = 0
            while time.perf_counter() - t0 < a.duration:
                th = threading.Thread(target=long_req, args=(i,), daemon=True)
                th.start()
                lths.append(th)
                i += 1
                time.sleep(a.long_interval * (0.5 + rng.random()))
        else:
            time.sleep(a.duration)
        for th in lths:
            th.join()
        stop.set()
        for t in ths:
            t.join()
        row = dict(phase=phase, short_users=a.short_users, short_requests=len(shorts), long_requests=len(longs),
                   errors=len(errs), error_samples=errs[:2],
                   short_ttft_p50=pct([r["ttft"] for r in shorts], 50), short_ttft_p99=pct([r["ttft"] for r in shorts], 99),
                   short_tpot_p50_ms=(pct([r["tpot"] for r in shorts], 50) or 0) * 1e3,
                   short_tpot_p99_ms=(pct([r["tpot"] for r in shorts], 99) or 0) * 1e3,
                   short_max_gap_p99_ms=(pct([r["max_gap"] for r in shorts], 99) or 0) * 1e3,
                   short_max_gap_max_ms=(max([r["max_gap"] for r in shorts if r["max_gap"]], default=0)) * 1e3,
                   long_prompt_tokens=mean([r["inp"] for r in longs]), long_ttft_p50=pct([r["ttft"] for r in longs], 50),
                   long_ttft_p99=pct([r["ttft"] for r in longs], 99), **stats_delta(s0, ep.stats()))
        rows.append(row)
        show(row, ["phase", "short_requests", "long_requests", "errors", "short_ttft_p50", "short_ttft_p99",
                   "short_tpot_p50_ms", "short_tpot_p99_ms", "short_max_gap_p99_ms", "short_max_gap_max_ms",
                   "long_ttft_p50"])
    return rows


def sc_capacity(ep, a):
    rows = []
    items = load_set("chat_en") + load_set("chat_de") + load_set("code")
    best = None
    for conc in [int(x) for x in a.concurrency.split(",")]:
        n = max(a.min_requests, a.requests_per_user * conc)
        if a.prompt_tokens:
            msgs = [[{"role": "user", "content": synthetic_document(a.seed * 977 + conc * 1000 + i, a.prompt_tokens,
                                                                    a.chars_per_token) + "\n\nSummarise this."}]
                    for i in range(n)]
        else:
            msgs = [items[(i * 5 + conc) % len(items)]["messages"] for i in range(n)]
        jobs = [lambda m=m: ep.chat(m, a.max_tokens, a.temperature, True, seed=a.seed) for m in msgs]
        res, err, wall = run_pool(jobs, conc)
        s = summarize(res, wall)
        ok = not err and s["ttft_p99"] is not None and s["ttft_p99"] <= a.slo_ttft and s["tpot_p99_ms"] <= a.slo_tpot
        row = dict(concurrency=conc, requests=n, errors=len(err), meets_slo=ok, **s)
        rows.append(row)
        show(row, ["concurrency", "requests", "errors", "meets_slo", "output_tok_s", "ttft_p50", "ttft_p99",
                   "tpot_p50_ms", "tpot_p99_ms"])
        if ok:
            best = row
        elif not a.full:
            break
    summary = dict(concurrency="max within SLO", slo_ttft_s=a.slo_ttft, slo_tpot_ms=a.slo_tpot,
                   max_users=best["concurrency"] if best else 0, output_tok_s=best["output_tok_s"] if best else None,
                   ttft_p99=best["ttft_p99"] if best else None, tpot_p99_ms=best["tpot_p99_ms"] if best else None)
    print(f"max concurrency within SLO (TTFT p99 <= {a.slo_ttft} s, TPOT p99 <= {a.slo_tpot} ms): "
          f"{summary['max_users']} users, {fmt(summary['output_tok_s'], 1)} output tok/s")
    rows.append(summary)
    return rows


MD_COLUMNS = {
    "realistic": ["set", "concurrency", "requests", "errors", "output_tok_s", "output_tokens_mean", "tpot_p50_ms",
                  "tpot_p99_ms", "ttft_p50", "ttft_p99", "tokens_per_step", "spec_acceptance"],
    "prefill": ["prompt_len", "concurrency", "errors", "prompt_tokens_mean", "ttft_p50", "ttft_p99", "ttft_max",
                "time_to_last_s", "prefill_tok_s", "total_tok_s"],
    "shared-prefix": ["variant", "turn", "requests", "errors", "prompt_tokens_mean", "ttft_p50", "ttft_p99", "tpot_p50_ms",
                      "wall_s", "prefix_cache_token_hit_rate", "prefix_cache_hits"],
    "mixed": ["phase", "short_requests", "long_requests", "errors", "short_ttft_p50", "short_ttft_p99", "short_tpot_p50_ms",
              "short_tpot_p99_ms", "short_max_gap_p99_ms", "short_max_gap_max_ms", "long_prompt_tokens", "long_ttft_p50"],
    "capacity": ["concurrency", "requests", "errors", "meets_slo", "output_tok_s", "ttft_p50", "ttft_p99", "tpot_p50_ms",
                 "tpot_p99_ms", "max_users"],
}

SCENARIOS = {"realistic": sc_realistic, "prefill": sc_prefill, "shared-prefix": sc_shared_prefix, "mixed": sc_mixed,
             "capacity": sc_capacity}


# ------------------------------------------------------------------------------------------------ output
def show(row, cols):
    print("  " + "  ".join(f"{c}={fmt(row.get(c))}" for c in cols if c in row), flush=True)


def markdown(rows, cols=None) -> str:
    present = {k for r in rows for k in r}
    cols = [c for c in (cols or []) if c in present] or \
        [k for k in dict.fromkeys(k for r in rows for k in r) if k != "error_samples"]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario", choices=sorted(SCENARIOS))
    ap.add_argument("--url", required=True)
    ap.add_argument("--model")
    ap.add_argument("--out", help="write OUT.json and OUT.md")
    ap.add_argument("--label", default="", help="free text stored with the results (engine, flags)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--chars-per-token", type=float, default=3.6, help="sizing of the synthetic texts")
    ap.add_argument("--ignore-eos", action="store_true", help="realistic / shared-prefix: generate exactly max_tokens")
    g = ap.add_argument_group("realistic")
    g.add_argument("--sets", default="code,chat_de,chat_en")
    g.add_argument("--min-requests", type=int, default=16)
    g = ap.add_argument_group("prefill / capacity")
    g.add_argument("--concurrency", default=None, help="realistic 1,4,16; prefill 16,32,64; capacity 1,2,4,...,128")
    g.add_argument("--prompt-lens", default="8192,16384,32768")
    g.add_argument("--rounds", type=int, default=1, help="prefill: requests per user")
    g.add_argument("--max-tokens", type=int, default=None, help="realistic/capacity 256, prefill 64, shared-prefix 64")
    g.add_argument("--slo-ttft", type=float, default=2.0)
    g.add_argument("--slo-tpot", type=float, default=50.0)
    g.add_argument("--requests-per-user", type=int, default=2)
    g.add_argument("--prompt-tokens", type=int, default=0, help="capacity: synthetic prompts of this size (0: corpus)")
    g.add_argument("--full", action="store_true", help="capacity: keep ramping after the first SLO miss")
    g = ap.add_argument_group("shared-prefix")
    g.add_argument("--users", type=int, default=16)
    g.add_argument("--turns", type=int, default=4)
    g.add_argument("--system-tokens", type=int, default=6000)
    g.add_argument("--variants", default="unique,shared")
    g.add_argument("--stagger", type=float, default=0.05, help="seconds between user starts")
    g = ap.add_argument_group("mixed")
    g.add_argument("--short-users", type=int, default=8)
    g.add_argument("--short-tokens", type=int, default=128)
    g.add_argument("--long-tokens", type=int, default=16384)
    g.add_argument("--long-out", type=int, default=32)
    g.add_argument("--long-interval", type=float, default=4.0, help="mean seconds between long requests")
    g.add_argument("--duration", type=float, default=30.0, help="seconds per phase")
    a = ap.parse_args()
    a.concurrency = a.concurrency or {"realistic": "1,4,16", "prefill": "16,32,64",
                                      "capacity": "1,2,4,8,16,32,64,128"}.get(a.scenario, "1")
    a.max_tokens = a.max_tokens or {"prefill": 64, "shared-prefix": 64}.get(a.scenario, 256)
    ep = Endpoint(a.url, a.model, a.timeout)
    print(f"{a.scenario}: endpoint {a.url} model {ep.model} max_model_len {ep.max_model_len}")
    ep.chat([{"role": "user", "content": "Hello"}], 4)  # warm-up
    t0 = time.time()
    rows = SCENARIOS[a.scenario](ep, a)
    md = f"### {a.scenario} — {ep.model} @ {a.url} {a.label}\n\n" + markdown(rows, MD_COLUMNS.get(a.scenario)) + "\n"
    print("\n" + md)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        json.dump(dict(scenario=a.scenario, url=a.url, model=ep.model, label=a.label, args=vars(a),
                       started=t0, rows=rows), open(a.out + ".json", "w"), indent=1)
        open(a.out + ".md", "w").write(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
