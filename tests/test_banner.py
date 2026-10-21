"""CPU-only checks of the start-up banner (server/banner.py): lettering shape, colour only on a terminal, ASCII
fallback, SM120_BANNER=0.

    python3 -m pytest -q tests/test_banner.py
"""

import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server"))

import banner  # noqa: E402


def test_lettering_shape():
    assert len({len(l) for l in banner.DEEP_SM120}) == 1
    assert len(banner.DEEP_SM120[0]) <= max(map(len, banner.SOMORA_WIDE)) <= 60  # fits 80 columns with the indent


def test_plain_when_not_a_terminal(monkeypatch):
    monkeypatch.delenv("SM120_BANNER", raising=False)
    out = io.StringIO()
    banner.print_banner("v0 · test", stream=out)
    text = out.getvalue()
    assert "\x1b[" not in text and banner.SOMORA_WIDE[0] in text and "v0 · test" in text


def test_colour_and_ascii_fallback():
    assert banner.MINT in banner.render("x", color=True, utf8=True)
    ascii_text = banner.render("x", color=False, utf8=False)
    assert "somora :: deep sm120" in ascii_text and "█" not in ascii_text


def test_disabled(monkeypatch):
    monkeypatch.setenv("SM120_BANNER", "0")
    out = io.StringIO()
    banner.print_banner("x", stream=out)
    assert out.getvalue() == ""
