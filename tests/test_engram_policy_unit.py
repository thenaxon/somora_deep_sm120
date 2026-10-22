"""CPU unit test of the pinned-mode start-up decision (no GPU): the shared tables are only served after a confirmed
fill; any fill failure, a failed post-load step or a lost / unconfirmed reply in the --dp proxy is fatal.

    python3 tests/test_engram_policy_unit.py"""
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "server"))
import dp_proxy  # noqa: E402
import server  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


class Residency(RuntimeError):
    pass


def fake_engram(pinned: bool, policy):
    m = types.ModuleType("sm120_engram")
    m._TABLES, m._SHARED_TABLES = ([] if pinned else [object()]), ([object()] if pinned else [])
    m.EngramResidencyError = Residency
    m.post_load_cache_policy = policy
    return m


def run_policy(pinned, policy):
    sys.modules["sm120_engram"] = fake_engram(pinned, policy)
    server._CACHE_POLICY.clear()
    try:
        return server._engram_cache_policy("/ckpt") or {}
    finally:
        sys.modules.pop("sm120_engram", None)


def raises(exc):
    def f(_):
        raise exc
    return f


# server side (rank 0 / single process): "fatal" decides
cases = [
    ("pinned fill ok", True, lambda _: {"pinned_fill_s": 1.0}, False),
    ("pinned fill raised EngramResidencyError", True, raises(Residency("fill failed")), True),
    ("pinned post-load step raised OSError", True, raises(OSError(5, "EIO")), True),
    ("pinned without fill confirmation", True, lambda _: {"seconds": 0.1}, True),
    ("host hints failed (advisory)", False, raises(OSError(5, "EIO")), False),
    ("host mlock refused", False, raises(Residency("no room")), True),
]
for name, pinned, policy, want in cases:
    r = run_policy(pinned, policy)
    check(f"server: {name}", bool(r.get("fatal")) == want, json.dumps(r)[:160])

# proxy side: the verdict on rank 0's reply
ok_reply = json.dumps({"result": {"pinned_fill_s": 1.0}})
pcases = [
    ("pinned, fill confirmed", ok_reply, None, True, False),
    ("pinned, no reply", None, TimeoutError("timed out"), True, True),
    ("pinned, unreadable reply", "<html>", None, True, True),
    ("pinned, reply without fill", json.dumps({"result": None}), None, True, True),
    ("pinned, fatal reply", json.dumps({"result": {"fatal": "fill failed"}}), None, True, True),
    ("host, no reply (advisory)", None, TimeoutError("timed out"), False, False),
    ("host, fatal reply (mlock)", json.dumps({"result": {"fatal": "no room"}}), None, False, True),
]
for name, reply, err, pinned, want in pcases:
    v = dp_proxy.policy_fatal(reply, err, pinned=pinned)
    check(f"proxy: {name}", bool(v) == want, repr(v)[:160])

print("ALL PASS" if not FAILS else f"FAILURES: {FAILS}")
sys.exit(1 if FAILS else 0)
