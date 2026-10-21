"""Image inputs for /v1/chat/completions: OpenAI `image_url` content parts -> DeepInfer multi_modal_data.

The checkpoint's encoding.py replaces every image part with one `<｜deepseek_image｜>` placeholder and returns the image
records in prompt order (encode_messages(..., return_multi_modal_data=True)). DeepInfer's add_request takes
multi_modal_data={"image_bytes": [...], "image_sizes": [(w, h), ...]}; its MultimodalProcessor.overwrite_prompt expands
each placeholder into the image's token span (size-dependent, at most vision_config.max_image_tokens) and its worker
processes decode and resize the bytes. Here each image is fetched (data: URL or http(s)), checked with PIL (format,
size, full decode) and passed on as the original encoded bytes with its decoded size.
"""

from __future__ import annotations

import base64
import binascii
import io
import ipaddress
import socket
import urllib.parse
import urllib.request
from dataclasses import dataclass

FORMATS = ("PNG", "JPEG", "WEBP", "GIF", "BMP")


class ImageError(ValueError):
    """Invalid or unsupported image in a request (HTTP 400)."""


@dataclass
class ImageLimits:
    max_images: int = 8  # per request
    max_bytes: int = 20 << 20  # encoded size per image
    max_pixels: int = 50_000_000  # width * height per image (checked before decoding)
    fetch_urls: bool = True  # http(s) image URLs (off: only data: URLs)
    fetch_private: bool = False  # allow URLs that resolve to loopback / private / link-local addresses
    fetch_timeout: float = 10.0


def _check_host(host: str, allow_private: bool) -> None:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ImageError(f"cannot resolve image host {host!r}: {exc}") from None
    if allow_private:
        return
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise ImageError(f"image URL host {host!r} resolves to a non-public address (server option "
                             "--image-fetch-private allows it)")


def _fetch(url: str, lim: ImageLimits) -> bytes:
    u = urllib.parse.urlparse(url)
    if u.scheme == "data":
        head, sep, payload = url[5:].partition(",")
        if not sep or ";base64" not in head:
            raise ImageError("data: image URLs must be base64 (data:image/<type>;base64,...)")
        mime = head.split(";", 1)[0]
        if mime and not mime.startswith("image/"):
            raise ImageError(f"data: URL has media type {mime!r}, expected image/*")
        if len(payload) > lim.max_bytes * 4 // 3 + 16:
            raise ImageError(f"image larger than {lim.max_bytes} bytes")
        try:
            return base64.b64decode(payload, validate=False)
        except (binascii.Error, ValueError) as exc:
            raise ImageError(f"invalid base64 image data: {exc}") from None
    if u.scheme in ("http", "https"):
        if not lim.fetch_urls:
            raise ImageError("http(s) image URLs are disabled on this server; send the image as a data: URL")
        if not u.hostname:
            raise ImageError(f"invalid image URL {url[:100]!r}")
        _check_host(u.hostname, lim.fetch_private)

        class _NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *a, **k):  # a redirect could point at a private address
                return None

        opener = urllib.request.build_opener(_NoRedirect)
        try:
            with opener.open(urllib.request.Request(url),
                             timeout=lim.fetch_timeout) as r:
                if r.status != 200:
                    raise ImageError(f"image URL returned HTTP {r.status}")
                data = r.read(lim.max_bytes + 1)
        except ImageError:
            raise
        except Exception as exc:  # noqa: BLE001  (URLError, timeout, HTTP errors, redirects)
            raise ImageError(f"cannot fetch image URL: {exc}") from None
        if len(data) > lim.max_bytes:
            raise ImageError(f"image larger than {lim.max_bytes} bytes")
        return data
    raise ImageError(f"unsupported image URL scheme {u.scheme!r} (use data: or http(s))")


def load_image(record: dict, lim: ImageLimits) -> tuple[bytes, tuple[int, int]]:
    """encoding.py image record -> (encoded bytes, (width, height)) as DeepInfer's worker will decode it."""
    from PIL import Image

    url = record.get("url")
    if not isinstance(url, str) or not url:
        raise ImageError("image parts need image_url.url (a data: or http(s) URL)")
    data = _fetch(url, lim)
    if not data:
        raise ImageError("empty image")
    if len(data) > lim.max_bytes:
        raise ImageError(f"image larger than {lim.max_bytes} bytes")
    try:
        with Image.open(io.BytesIO(data)) as img:
            fmt, (w, h) = img.format, img.size
            if fmt not in FORMATS:
                raise ImageError(f"unsupported image format {fmt} (supported: {', '.join(FORMATS)})")
            if w * h > lim.max_pixels:
                raise ImageError(f"image has {w}x{h} pixels, more than the limit of {lim.max_pixels}")
            img.load()  # full decode: a truncated or corrupt file fails here, not in the engine's worker
    except ImageError:
        raise
    except Exception as exc:  # noqa: BLE001  (PIL.UnidentifiedImageError, OSError, DecompressionBombError)
        raise ImageError(f"cannot decode image: {exc}") from None
    if w < 1 or h < 1:
        raise ImageError("image has no pixels")
    return data, (w, h)


def load_images(records: list[dict], lim: ImageLimits) -> dict | None:
    """All images of a request -> DeepInfer multi_modal_data (None without images)."""
    if not records:
        return None
    if len(records) > lim.max_images:
        raise ImageError(f"{len(records)} images in the request; the limit is {lim.max_images}")
    out = [load_image(r, lim) for r in records]
    return {"image_bytes": [b for b, _ in out], "image_sizes": [s for _, s in out]}
