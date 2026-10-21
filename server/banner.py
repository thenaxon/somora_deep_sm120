"""Start-up banner: the somora lettering (the same lines as the somora installer and TUI) with "deep sm120" below.

Printed once by server.py before the engine loads. Colour only on a terminal; plain text in `docker logs`;
ASCII fallback without a UTF-8 locale. SM120_BANNER=0 turns it off.
"""

from __future__ import annotations

import os
import sys

SOMORA_WIDE = (
    "███████╗ ██████╗ ███╗   ███╗ ██████╗ ██████╗  █████╗",
    "██╔════╝██╔═══██╗████╗ ████║██╔═══██╗██╔══██╗██╔══██╗",
    "███████╗██║   ██║██╔████╔██║██║   ██║██████╔╝███████║",
    "╚════██║██║   ██║██║╚██╔╝██║██║   ██║██╔══██╗██╔══██║",
    "███████║╚██████╔╝██║ ╚═╝ ██║╚██████╔╝██║  ██║██║  ██║",
    "╚══════╝ ╚═════╝ ╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝",
)

# "deep sm120" in the compact box-drawing letters of the somora compact banner, right-aligned under the lettering
DEEP_SM120 = (
    "┌┬┐┌─┐┌─┐┌─┐  ┌─┐┌┬┐┐┌─┐┌─┐",
    " ││├┤ ├┤ ├─┘  └─┐││││┌─┘│ │",
    "─┴┘└─┘└─┘┴    └─┘┴ ┴┴└─┘└─┘",
)

SOMORA_ASCII = ("somora", "deep sm120")

TAGLINE = "DeepSeek DeepInfer on NVIDIA SM120 (RTX 5090 / RTX PRO 6000 Blackwell)"
MOTTO = "Run. Rest. Dream."

MINT = "\x1b[38;2;126;184;154m"  # somora brand mint #7EB89A
DIM = "\x1b[2m"
RST = "\x1b[0m"


def version() -> str:
    for p in (os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "VERSION"),):
        try:
            return open(p).read().strip()
        except OSError:
            pass
    return "?"


def _utf8(stream) -> bool:
    enc = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
    if enc == "utf8":
        return True
    v = os.environ.get("LC_ALL") or os.environ.get("LC_CTYPE") or os.environ.get("LANG") or ""
    return "utf8" in v.lower().replace("-", "")


def render(subtitle: str, *, color: bool, utf8: bool) -> str:
    c, d, r = (MINT, DIM, RST) if color else ("", "", "")
    if not utf8:
        lines = [f"  {c}{SOMORA_ASCII[0]} :: {SOMORA_ASCII[1]}{r}"]
    else:
        pad = " " * (max(map(len, SOMORA_WIDE)) - len(DEEP_SM120[0]))
        lines = [f"  {c}{l}{r}" for l in SOMORA_WIDE] + [f"  {pad}{c}{l}{r}" for l in DEEP_SM120]
    return "\n".join(["", *lines, f"  {TAGLINE}", f"  {d}{MOTTO}  ·  {subtitle}{r}", ""])


def print_banner(subtitle: str, stream=None) -> None:
    if os.environ.get("SM120_BANNER", "1") == "0":
        return
    stream = stream or sys.stderr
    color = stream.isatty() and os.environ.get("NO_COLOR") is None
    try:
        stream.write(render(subtitle, color=color, utf8=_utf8(stream)) + "\n")
        stream.flush()
    except (OSError, UnicodeEncodeError):
        pass
