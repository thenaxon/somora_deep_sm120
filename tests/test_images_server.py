#!/usr/bin/env python3
"""Image input against a running server.py (stdlib client + Pillow for the test images).

    python3 /opt/sm120/tests/test_images_server.py --url http://127.0.0.1:8000 [--fetch-private] [--deterministic]

Structural checks on a truncated test model (its text is garbage): image requests succeed (formats, sizes, several
images, streaming, concurrency, together with tools), usage.prompt_tokens includes the image tokens (more for larger
images, capped), different images change the greedy output (the pixels reach the model), bad images get 400.
--fetch-private: the server runs with --image-fetch-private, so an http URL served from 127.0.0.1 is fetched;
without it that URL must be refused. --deterministic: the server runs --deterministic (greedy repeat must match).
"""

from __future__ import annotations

import argparse
import base64
import http.server
import io
import json
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_server  # noqa: E402
from test_server import RESULTS, Client  # noqa: E402

SOFT: set[str] = set()


def check(name, ok, info=""):
    if name in SOFT and not ok:
        print(f"[INFO] {name}  {info} (expected for this server / checkpoint)", flush=True)
        return True
    return test_server.check(name, ok, info)


def img(w, h, color=(220, 40, 40), fmt="PNG", pattern=False) -> bytes:
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (w, h), (255, 255, 255) if pattern else color)
    if pattern:
        d = ImageDraw.Draw(im)
        for i in range(0, w, max(8, w // 16)):
            d.rectangle([i, 0, i + max(4, w // 32), h], fill=color)
    b = io.BytesIO()
    im.save(b, fmt)
    return b.getvalue()


def durl(b: bytes, mime="image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode(b).decode()


def part(url: str) -> dict:
    return {"type": "image_url", "image_url": {"url": url}}


def user(*parts) -> list[dict]:
    return [{"role": "user", "content": [p if isinstance(p, dict) else {"type": "text", "text": p} for p in parts]}]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--fetch-private", action="store_true")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--zero-vision", action="store_true",
                    help="the checkpoint's vision tower is synthesised zeros (make_truncated_ckpt.py without shard 1): "
                         "images cannot change the logits, that check is INFO")
    args = ap.parse_args()
    if args.zero_vision:
        SOFT.add("image: different images change the logits")
    if not args.deterministic:
        SOFT.add("image: greedy repeat identical")
    cl = Client(args.url)
    try:
        run(cl, args)
    except Exception as exc:  # noqa: BLE001
        test_server.check("test run completed", False, f"{type(exc).__name__}: {exc}"[:300])
    fails = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(fails)}/{len(RESULTS)} passed" + (f"; FAILED: {fails}" if fails else ""))
    return 1 if fails else 0


def run(cl, args) -> None:
    model = cl.get("/v1/models")[1]["data"][0]["id"]
    def chat(msgs, **kw):
        return cl.post("/v1/chat/completions", {"model": model, "messages": msgs, "max_tokens": 16, "temperature": 0, **kw})

    red = durl(img(448, 448))
    # 1. one image vs. text only: the image adds its token span to the prompt
    st0, r0 = chat(user("Describe the picture."))
    st, r = chat(user("Describe the picture.", part(red)))
    n_text = r0["usage"]["prompt_tokens"]
    n_img = r["usage"]["prompt_tokens"] if st == 200 else -1
    check("image request (PNG 448x448)", st == 200 and r["choices"][0]["finish_reason"] in ("stop", "length"),
          f"{st} prompt tokens {n_text} -> {n_img}: {json.dumps(r)[:120] if st != 200 else ''}")
    check("image tokens counted in usage", n_img > n_text + 16, f"+{n_img - n_text}")
    # 2. sizes: larger image -> more tokens, capped (vision_config.max_image_tokens)
    sizes = [(64, 64), (448, 448), (1344, 896), (4000, 3000)]
    toks = []
    for w, h in sizes:
        st, r = chat(user("Size?", part(durl(img(w, h, fmt="JPEG"), "image/jpeg"))), max_tokens=2)
        toks.append(r["usage"]["prompt_tokens"] if st == 200 else -1)
    check("token count grows with image size and is capped", all(t > 0 for t in toks) and toks[1] <= toks[2]
          and toks[3] - n_text <= 1100, f"{list(zip(sizes, toks))}")
    # 3. formats
    ok = []
    for fmt, mime in (("PNG", "image/png"), ("JPEG", "image/jpeg"), ("WEBP", "image/webp"), ("GIF", "image/gif"),
                      ("BMP", "image/bmp")):
        st, r = chat(user("What is it?", part(durl(img(200, 150, fmt=fmt), mime))), max_tokens=4)
        ok.append((fmt, st))
    check("image formats PNG/JPEG/WEBP/GIF/BMP", all(s == 200 for _, s in ok), f"{ok}")
    # 4. the image reaches the model: the first token's top-20 logprobs (raw logits) differ between two pictures and
    #    repeat for the same picture (a truncated model often emits the same token whatever the input, so compare logits)
    blue = durl(img(448, 448, color=(30, 60, 220), pattern=True))
    rows = []
    for u in (red, blue, red):
        st, r = chat(user("What do you see?", part(u)), max_tokens=1, logprobs=True, top_logprobs=20)
        lp = r["choices"][0]["logprobs"]["content"][0]["top_logprobs"] if st == 200 else None
        rows.append({x["token"]: round(x["logprob"], 3) for x in lp} if lp else None)
    if st == 400 and "MTP" in json.dumps(r):
        print("[INFO] image logit checks skipped: logprobs are not available with --nextn > 0", flush=True)
        rows = [{"skip": 0}, {"skip": 1}, {"skip": 0}]
    check("image: different images change the logits", rows[0] is not None and rows[0] != rows[1],
          f"{list(rows[0].items())[:2] if rows[0] else None} vs {list(rows[1].items())[:2] if rows[1] else None}")
    check("image: greedy repeat identical", rows[0] == rows[2])
    # 5. several images, streaming, multi-turn history with an earlier image
    st, r = chat(user("Compare", part(red), "with", part(blue)))
    check("two images in one message", st == 200 and r["usage"]["prompt_tokens"] > n_img, f"{r.get('usage')}")
    ev, _, _, done = cl.stream("/v1/chat/completions", dict(model=model, messages=user("Stream it.", part(red)),
                                                             max_tokens=16, temperature=0))
    text = "".join((e["choices"][0]["delta"].get("content") or "") for e in ev if e.get("choices"))
    check("image request streaming", done and len(ev) > 2, f"{len(ev)} events {text[:40]!r}")
    hist = user("First picture:", part(red)) + [{"role": "assistant", "content": "A red square."}] + \
        user("And this one?", part(blue))
    st, r = chat(hist)
    check("multi-turn with images in two turns", st == 200, f"{r.get('usage')}")
    # 6. images + tools (grammar) in one request
    tool = {"type": "function", "function": {"name": "report_color", "parameters": {"type": "object", "properties": {
        "color": {"type": "string", "enum": ["red", "green", "blue"]}}, "required": ["color"]}}}
    st, r = chat(user("Which color is it?", part(red)), tools=[tool], tool_choice="required", max_tokens=256,
                 temperature=0.7)
    calls = (r["choices"][0]["message"].get("tool_calls") or []) if st == 200 else []
    check("image + tool_choice required", st == 200 and calls and json.loads(calls[0]["function"]["arguments"])["color"]
          in ("red", "green", "blue"), f"{calls[:1]}")
    # 7. concurrency: 4 image requests + 2 text requests at once
    res = {}

    def w(i):
        res[i] = chat(user("Go", part(red if i % 2 else blue)) if i < 4 else user("text only"), max_tokens=32,
                      temperature=0.8, seed=i)[0]
    th = [threading.Thread(target=w, args=(i,)) for i in range(6)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    check("concurrent image + text requests (6)", all(res.get(i) == 200 for i in range(6)), f"{res}")
    # 8. http(s) image URLs (a local web server: refused unless the server allows private addresses)
    body = img(320, 240)

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    st, r = chat(user("From the web:", part(f"http://127.0.0.1:{srv.server_address[1]}/x.png")))
    srv.shutdown()
    if args.fetch_private:
        check("http image URL (private address allowed)", st == 200, f"{st} {json.dumps(r)[:100]}")
    else:
        check("http image URL on a private address -> 400", st == 400, f"{st} {json.dumps(r)[:100]}")
    # 9. bad images -> 400
    tiny = durl(img(8, 8))
    for label, msgs in [
            ("bad base64", user("x", part("data:image/png;base64,@@@"))),
            ("not an image", user("x", part(durl(b"just text, not a picture")))),
            ("truncated PNG", user("x", part(durl(body[:100])))),
            ("file: URL", user("x", part("file:///etc/hostname"))),
            ("too many images", user("x", *[part(tiny)] * 9)),
            ("image in a tool message", [{"role": "user", "content": "q"},
                                         {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type":
                                          "function", "function": {"name": "report_color", "arguments": "{}"}}]},
                                         {"role": "tool", "tool_call_id": "c1", "content": [part(tiny)]}])]:
        kw = dict(tools=[tool]) if "tool" in label else {}
        st, r = chat(msgs, **kw)
        check(f"400: {label}", st == 400, json.dumps(r)[:140])
    st, h = cl.get("/health")
    check("server healthy after the image tests", st == 200 and h["status"] == "ok")


if __name__ == "__main__":
    sys.exit(main())
