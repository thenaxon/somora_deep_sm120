"""Shared statistics for the eval clients (loadgen.py, scenarios.py; stdlib only).

- Tail latencies: a p99 is only reported from >= 100 samples; below that the tail is the maximum, labelled as such.
- A missing TPOT (no request produced two output chunks) stays None; it never counts as 0 ms.
- Steady-state throughput: output tokens per second inside the window where all `conc` requests of a closed loop are
  in flight (after the ramp-up, before the tail where workers run out of work). Each request's tokens are spread
  evenly over [first token, last token] and counted by their overlap with the window.
- Repeats: per-repeat numbers are combined as median and range; latency percentiles are taken over the samples of
  all repeats together (more samples, same workload).
"""

from __future__ import annotations

import statistics

P99_MIN_SAMPLES = 100


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return xs[min(len(xs) - 1, max(0, int(round(q / 100 * (len(xs) - 1)))))]


def tail(xs) -> tuple:
    """(value, label, n): the p99 with >= P99_MIN_SAMPLES samples, else the maximum."""
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    if len(xs) >= P99_MIN_SAMPLES:
        return pct(xs, 99), "p99", len(xs)
    return max(xs), "max", len(xs)


def ms(x):
    return None if x is None else x * 1e3


def steady_tok_s(results: list[dict], conc: int):
    """Output tok/s while all `conc` slots are busy. results need t_send, t_end, t_first, t_last, out.
    None when there is no such window (e.g. requests <= conc)."""
    if conc < 1 or len(results) <= conc:
        return None, None
    sends = sorted(r["t_send"] for r in results)
    ends = sorted(r["t_end"] for r in results)
    ta = sends[conc - 1]  # the conc-th request is in flight: all slots busy
    tb = ends[len(results) - conc]  # from here on a worker finds no more work: the tail
    if tb <= ta:
        return None, None
    toks = 0.0
    for r in results:
        a, b, n = r.get("t_first"), r.get("t_last"), r.get("out") or 0
        if a is None or n <= 0:
            continue
        if b is None or b <= a:
            toks += n if ta <= a <= tb else 0
            continue
        ov = max(0.0, min(b, tb) - max(a, ta))
        toks += n * ov / (b - a)
    return toks / (tb - ta), tb - ta


def combine(reps: list[dict], keys: list[str]) -> dict:
    """Median and range over repeats for the given keys (None values skipped)."""
    out = {}
    for k in keys:
        xs = [r.get(k) for r in reps if r.get(k) is not None]
        if xs:
            out[k] = statistics.median(xs)
            if len(xs) > 1:
                out[k + "_min"], out[k + "_max"] = min(xs), max(xs)
        else:
            out[k] = None
    return out
