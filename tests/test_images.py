#!/usr/bin/env python3
"""CPU unit tests for server/images.py (image fetching / validation) and the encoding.py image rendering it relies on.

    python3 tests/test_images.py [--encoding /ckpt/encoding/encoding.py]      (needs Pillow)
"""

from __future__ import annotations

import base64
import http.server
import importlib.util
import io
import os
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "server"))

import images  # noqa: E402

ENC_PATH = os.environ.get("DS41_ENCODING", "/ckpt/encoding/encoding.py")


def png(w=64, h=48, fmt="PNG", color=(200, 30, 30)) -> bytes:
    from PIL import Image

    b = io.BytesIO()
    Image.new("RGB", (w, h), color).save(b, fmt)
    return b.getvalue()


def data_url(b: bytes, mime="image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode(b).decode()


def expect_error(fn, *a, contains=""):
    try:
        fn(*a)
    except images.ImageError as exc:
        assert contains in str(exc), (contains, str(exc))
        return
    raise AssertionError(f"no ImageError ({contains})")


def test_data_urls():
    lim = images.ImageLimits()
    for fmt, mime in (("PNG", "image/png"), ("JPEG", "image/jpeg"), ("WEBP", "image/webp"), ("GIF", "image/gif")):
        b, size = images.load_image({"url": data_url(png(fmt=fmt), mime)}, lim)
        assert size == (64, 48) and b[:2], fmt
    mm = images.load_images([{"url": data_url(png(10, 20))}, {"url": data_url(png(30, 5))}], lim)
    assert mm["image_sizes"] == [(10, 20), (30, 5)] and len(mm["image_bytes"]) == 2
    assert images.load_images([], lim) is None


def test_rejections():
    lim = images.ImageLimits(max_images=2, max_bytes=50_000, max_pixels=10_000)
    expect_error(images.load_image, {"url": "data:image/png;base64,!!!notbase64"}, lim, contains="")
    expect_error(images.load_image, {"url": data_url(b"hello world", "image/png")}, lim, contains="cannot decode")
    expect_error(images.load_image, {"url": data_url(png(), "text/plain")}, lim, contains="media type")
    expect_error(images.load_image, {"url": "data:image/png,rawnotbase64"}, lim, contains="base64")
    expect_error(images.load_image, {"url": data_url(png(200, 100))}, lim, contains="pixels")
    expect_error(images.load_image, {"url": data_url(png()[:60])}, lim, contains="cannot decode")  # truncated
    expect_error(images.load_image, {"url": "file:///etc/passwd"}, lim, contains="scheme")
    expect_error(images.load_image, {"url": "/etc/passwd"}, lim, contains="scheme")
    expect_error(images.load_image, {"url": ""}, lim, contains="image_url.url")
    expect_error(images.load_images, [{"url": data_url(png(4, 4))}] * 3, lim, contains="limit is 2")
    from PIL import Image

    b = io.BytesIO()
    Image.new("RGB", (8, 8)).save(b, "TIFF")
    expect_error(images.load_image, {"url": data_url(b.getvalue(), "image/tiff")}, lim, contains="format")
    big = os.urandom(60_000)
    expect_error(images.load_image, {"url": data_url(big)}, lim, contains="larger")


def test_http_fetch():
    body = png(16, 16)

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "/img.png")
                self.end_headers()
                return
            self.send_response(200 if self.path == "/img.png" else 404)
            self.send_header("Content-Type", "image/png")
            self.end_headers()
            self.wfile.write(body if self.path == "/img.png" else b"")

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        # loopback is refused by default (SSRF), allowed with fetch_private
        expect_error(images.load_image, {"url": base + "/img.png"}, images.ImageLimits(), contains="non-public")
        lim = images.ImageLimits(fetch_private=True)
        assert images.load_image({"url": base + "/img.png"}, lim)[1] == (16, 16)
        expect_error(images.load_image, {"url": base + "/missing.png"}, lim, contains="fetch")
        expect_error(images.load_image, {"url": base + "/redirect"}, lim, contains="")  # redirects are not followed
        expect_error(images.load_image, {"url": base + "/img.png"}, images.ImageLimits(fetch_urls=False),
                     contains="disabled")
    finally:
        srv.shutdown()


def test_encoding_renders_placeholders():
    if not os.path.isfile(ENC_PATH):
        print("  (encoding.py not found: skipped)")
        return
    spec = importlib.util.spec_from_file_location("ds41_encoding", ENC_PATH)
    enc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(enc)
    u = data_url(png())
    msgs = [{"role": "user", "content": [{"type": "text", "text": "Compare"},
                                         {"type": "image_url", "image_url": {"url": u}},
                                         {"type": "image_url", "image_url": {"url": u, "detail": "high"}},
                                         {"type": "text", "text": "please."}]}]
    prompt, mm = enc.encode_messages(msgs, thinking_mode="chat", return_multi_modal_data=True)
    assert prompt.count(enc.IMAGE_PLACEHOLDER) == 2 and [r["url"] for r in mm["images"]] == [u, u]
    assert "Compare\n\n<｜deepseek_image｜>\n\n<｜deepseek_image｜>\n\nplease." in prompt
    # text-only: no records
    prompt, mm = enc.encode_messages([{"role": "user", "content": "hi"}], thinking_mode="chat",
                                     return_multi_modal_data=True)
    assert mm["images"] == []


def main() -> int:
    global ENC_PATH
    if "--encoding" in sys.argv:
        ENC_PATH = sys.argv[sys.argv.index("--encoding") + 1]
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"[PASS] {name}")
            except Exception as exc:  # noqa: BLE001
                fails += 1
                print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
    print("OK" if not fails else f"{fails} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
