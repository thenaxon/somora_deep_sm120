"""server.py argument checks that must refuse before any model load (CPU, no GPU needed).

    python3 -m pytest -q /opt/sm120/tests/test_server_args.py

The torch MoE reference (--deterministic / SM120_MOE_IMPL=torch) is single-GPU and not CUDA-graph capturable:
with --dp / --ep > 1 or --graphs the server exits 2 at argument parsing, naming SM120_MOE_DETERMINISTIC=1.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

SERVER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server", "server.py")


def _run(args, env=None):
    e = dict(os.environ)
    e.pop("SM120_MOE_IMPL", None)
    e.update(env or {})
    return subprocess.run([sys.executable, SERVER, "--ckpt", "/nonexistent", *args], env=e, capture_output=True,
                          text=True, timeout=120)


@pytest.mark.parametrize("args,env,what", [
    (["--deterministic", "--graphs"], None, "not CUDA-graph capturable"),
    (["--graphs"], {"SM120_MOE_IMPL": "torch"}, "not CUDA-graph capturable"),
    (["--deterministic", "--dp", "2"], None, "single-GPU only"),
    (["--dp", "2", "--ep", "2"], {"SM120_MOE_IMPL": "torch"}, "single-GPU only"),
])
def test_refused(args, env, what):
    r = _run(args, env)
    assert r.returncode == 2, (r.returncode, r.stderr[-400:])
    assert what in r.stderr and "SM120_MOE_DETERMINISTIC=1" in r.stderr, r.stderr[-400:]
