"""Installed by putting sm120 first on PYTHONPATH: activates the SM120 DeepInfer patches
(sm120_patches; lazy post-import hooks, no torch import here), then chains to the next
``sitecustomize`` on sys.path (e.g. the distro's) so we do not shadow it."""

import os as _os
import sys as _sys

try:
    import sm120_patches as _p

    _p.install()
except Exception as _exc:  # never break interpreter start-up
    print(f"[sitecustomize] sm120_patches failed to install: {_exc!r}", file=_sys.stderr)


def _chain():
    import importlib.machinery
    import importlib.util

    here = _os.path.dirname(_os.path.abspath(__file__))
    paths = [p for p in _sys.path if _os.path.abspath(p or ".") != here]
    spec = importlib.machinery.PathFinder.find_spec("sitecustomize", paths)
    if spec is None or spec.origin is None or _os.path.abspath(spec.origin) == _os.path.abspath(__file__):
        return
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


try:
    _chain()
except Exception as _exc:
    print(f"[sitecustomize] chained sitecustomize failed: {_exc!r}", file=_sys.stderr)
