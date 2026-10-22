"""CPU tests for tools/compare.py exit codes: 0 ok, 1 suspicious hidden layers, 2 invalid comparison,
3 failed --decode-check (including teacher-forced runs, which are judged by the decode logits).

    python3 -m pytest tests/test_compare_exit.py -q
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

HERE = os.path.dirname(os.path.abspath(__file__))
COMPARE = os.path.join(os.path.dirname(HERE), "tools", "compare.py")
V, STEPS = 64, 8


def dump(seed=0, hidden=True, layer0_noise=0.0, decode_shift=0.0, teacher_forced=False, input_ids=None):
    g = torch.Generator().manual_seed(1234)
    logits = torch.randn(STEPS, V, generator=g) * 4
    top = logits.topk(50, dim=-1)
    d = dict(input_ids=input_ids if input_ids is not None else torch.arange(10),
             decoded_ids=logits.argmax(-1), decode_logits_all=logits.clone(),
             decode_top50_values=top.values, decode_top50_ids=top.indices,
             meta=dict(teacher_forced=teacher_forced))
    if decode_shift:
        # a different distribution; the decoded ids stay the forced ones, as under teacher forcing
        d["decode_logits_all"] = torch.randn(STEPS, V, generator=torch.Generator().manual_seed(99)) * 4 * decode_shift
    if hidden:
        h = {f"layer{i}": torch.randn(10, 16, generator=torch.Generator().manual_seed(i)) for i in range(3)}
        n = torch.Generator().manual_seed(seed)
        h = {k: v + 1e-3 * torch.randn(v.shape, generator=n) for k, v in h.items()}
        if layer0_noise:
            h["layer0"] = h["layer0"] + layer0_noise * torch.randn(10, 16, generator=n)
        d["hidden"] = dict(prefill=h)
    return d


def run(tmp_path, ref, di, ref2=None, *extra):
    paths = {}
    for name, obj in (("ref", ref), ("di", di), ("ref2", ref2)):
        if obj is not None:
            paths[name] = str(tmp_path / f"{name}.pt")
            torch.save(obj, paths[name])
    cmd = [sys.executable, COMPARE, "--ref", paths["ref"], "--di", paths["di"], "--ref2", paths.get("ref2", ""), *extra]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def test_ok(tmp_path):
    rc, out = run(tmp_path, dump(0), dump(1), dump(2))
    assert rc == 0, out


def test_suspicious_layer_and_known_deviation(tmp_path):
    rc, out = run(tmp_path, dump(0), dump(1, layer0_noise=0.5), dump(2))
    assert rc == 1, out
    rc, out = run(tmp_path, dump(0), dump(1, layer0_noise=0.5), dump(2), "--known-deviation", "layer0")
    assert rc == 0 and "known" in out, out


def test_input_ids_differ_is_invalid(tmp_path):
    rc, out = run(tmp_path, dump(0), dump(1, input_ids=torch.arange(1, 11)), dump(2))
    assert rc == 2 and "INVALID" in out, out


def test_missing_hidden(tmp_path):
    rc, out = run(tmp_path, dump(0), dump(1, hidden=False))
    assert rc == 2, out
    rc, out = run(tmp_path, dump(0), dump(1, hidden=False), None, "--allow-missing-hidden")
    assert rc == 0, out


def test_decode_check_untouched_run(tmp_path):
    ref, di = dump(0, hidden=False), dump(1, hidden=False)
    rc, out = run(tmp_path, ref, di, None, "--allow-missing-hidden", "--decode-check", "identical")
    assert rc == 0, out
    rc, out = run(tmp_path, ref, di, None, "--allow-missing-hidden", "--decode-check", "diverge")
    assert rc == 3, out


def test_decode_check_teacher_forced_uses_logits(tmp_path):
    # same forced tokens in both dumps: only the decode logits can show the divergence
    ref = dump(0, hidden=False)
    div = dump(1, hidden=False, teacher_forced=True, decode_shift=1.0)
    same = dump(1, hidden=False, teacher_forced=True)
    a = ("--allow-missing-hidden", "--decode-check")
    for di, chk, want in ((div, "diverge", 0), (same, "diverge", 3), (same, "identical", 0), (div, "identical", 3)):
        rc, out = run(tmp_path, ref, di, None, *a, chk)
        assert rc == want, out
