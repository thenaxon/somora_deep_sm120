"""Client for graph_replay_lazy_prep.sh: short completions, then a longer prompt whose prefill prepares new
b12x plans while the server runs; the decode graphs captured at start-up must keep replaying. The later
prompts are first-seen sizes (short, medium, a chunked prefill beyond 4k), each of which used to be able to
trigger the crash on its own after the history.

    python3 graph_replay_lazy_prep.py --url http://127.0.0.1:8125 [--short 19] [--long 549]

Exit 0 if every request returns 200 and /health stays "ok", 1 otherwise.
"""

import argparse
import http.client
import json
import sys
from urllib.parse import urlparse

# output lengths of the original failing session (14 tool-call turns, then 5 x 64 tokens)
SHORT_MAX_TOKENS = [186, 166, 209, 155, 214, 201, 186, 158, 181, 173, 200, 180, 176, 162] + [64] * 5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8125")
    ap.add_argument("--short", type=int, default=len(SHORT_MAX_TOKENS), help="number of short requests")
    ap.add_argument("--short-len", type=int, default=469)
    ap.add_argument("--long", type=int, nargs="+", default=[30, 549, 1100, 2200, 6000],
                    help="first-seen prompt lengths sent after the history, in order")
    args = ap.parse_args()
    u = urlparse(args.url)

    def call(method, path, body=None):
        c = http.client.HTTPConnection(u.hostname, u.port, timeout=600)
        c.request(method, path, None if body is None else json.dumps(body), {"Content-Type": "application/json"})
        r = c.getresponse()
        return r.status, json.loads(r.read() or b"{}")

    def prompt(n, k):
        return [(i * k) % 120000 + 100 for i in range(n)]

    steps = [(args.short_len, SHORT_MAX_TOKENS[j % len(SHORT_MAX_TOKENS)], 7919) for j in range(args.short)]
    steps += [(n, 64, 104729 + j) for j, n in enumerate(args.long)]
    for j, (plen, max_tokens, k) in enumerate(steps):
        body = dict(model="m", prompt=prompt(plen, k), max_tokens=max_tokens, ignore_eos=True, temperature=0.8)
        try:
            st, r = call("POST", "/v1/completions", body)
            health = call("GET", "/health")[1].get("status")
        except Exception as e:  # noqa: BLE001 - a dead server is the failure being tested
            st, r, health = None, {"error": repr(e)}, None
        if st != 200 or health != "ok":
            print(f"FAIL at request {j + 1}/{len(steps)} ({plen} tok prompt, {max_tokens} out): status {st} "
                  f"health {health} {str(r.get('error'))[:200]}", flush=True)
            return 1
    print(f"OK: {len(steps)} requests ({args.short} x {args.short_len} tok, then {args.long})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
