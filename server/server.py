#!/usr/bin/env python3
"""Minimal OpenAI-compatible HTTP server around DeepInfer (single GPU), stdlib only.

    python3 /opt/sm120/server/server.py --ckpt /work/ckpt3 --port 8000 [--max-num-seqs 8] [--nextn N] [--graphs]

Endpoints: GET /health, GET /v1/models, POST /v1/completions, POST /v1/chat/completions (the checkpoint's
encoding/encoding.py chat template; "thinking": true, reasoning_effort or the same keys in chat_template_kwargs ->
thinking mode), streaming (SSE,
"stream": true) and non-streaming. Sampling: temperature, top_p, top_k, max_tokens, stop (str | list), seed,
ignore_eos; extension "return_token_ids": true (non-streaming). Chat: tools / tool_choice / parallel_tool_calls (DSML tool
calls, tools.py), response_format text / json_object / json_schema (grammar-constrained, DeepInfer structured output),
and image_url content parts (data: or http(s) URLs, images.py; DeepInfer's multimodal path).
Concurrent requests share one continuously-batched DeepInfer engine (engine.py). A client that disconnects from a stream
aborts its request in the engine. See docs/SERVER.md.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import queue
import signal
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import images  # noqa: E402
import tools  # noqa: E402
from engine import EngineLoop, GenRequest  # noqa: E402

log = logging.getLogger("di_server")


# ------------------------------------------------------------------------------------------------ text side
class TextCodec:
    """Prompt rendering and detokenisation, independent of DeepInfer's tokenizer wrapper (which prepends BOS to
    plain text, hugging_face/tokenizer.py:26): raw `tokenizers` + the checkpoint's encoding.py, as ref_runner does."""

    def __init__(self, ckpt: str):
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(os.path.join(ckpt, "tokenizer.json"))
        self.bos = "<｜begin▁of▁sentence｜>"
        self.enc = None
        p = os.path.join(ckpt, "encoding", "encoding.py")
        if os.path.isfile(p):
            spec = importlib.util.spec_from_file_location("ds41_encoding", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.enc = mod
        else:
            log.warning("no encoding/encoding.py in %s: chat prompts use a plain fallback template", ckpt)

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

    def render_chat(self, messages: list[dict], thinking: bool, reasoning_effort=None,
                    drop_thinking: bool = True) -> tuple[str, list[dict]]:
        """-> (prompt, image records in prompt order; one image placeholder token per image in the prompt)."""
        if self.enc is not None:
            prompt, mm = self.enc.encode_messages(messages, thinking_mode="thinking" if thinking else "chat",
                                                  reasoning_effort=reasoning_effort, drop_thinking=drop_thinking,
                                                  return_multi_modal_data=True)
            return prompt, mm.get("images") or []
        return self.bos + "".join(f"{m.get('role')}: {m.get('content')}\n" for m in messages) + "assistant: ", []

    def split_reasoning(self, text: str, thinking: bool) -> tuple[str, str | None]:
        if not thinking:
            return text, None
        end = getattr(self.enc, "thinking_end_token", "</think>") if self.enc is not None else "</think>"
        if end in text:
            r, c = text.split(end, 1)
            return c.lstrip("\n"), r
        return "", text  # still thinking (or never closed)


class Detok:
    """Incremental detokenisation with stop strings. emit() returns the text that is safe to send now: it never
    sends an incomplete UTF-8 sequence (U+FFFD at the end) or a prefix of a stop string that may still complete."""

    def __init__(self, codec: TextCodec, stop: list[str]):
        self.codec, self.stop = codec, [s for s in stop if s]
        self.ids: list[int] = []
        self.sent = 0
        self.text = ""
        self.stopped = False
        self.hold = max((len(s) for s in self.stop), default=1) - 1

    def push(self, token_id: int) -> None:
        self.ids.append(token_id)
        self.text = self.codec.decode(self.ids)
        for s in self.stop:
            i = self.text.find(s, max(0, self.sent - len(s)))
            if i >= 0:
                self.text = self.text[:i]
                self.stopped = True

    def emit(self, final: bool = False) -> str:
        end = len(self.text)
        if not (final or self.stopped):
            while end > self.sent and self.text[end - 1] == "�":
                end -= 1
            end = max(self.sent, end - self.hold)
        out = self.text[self.sent:end]
        self.sent = end
        return out


# ------------------------------------------------------------------------------------------------ HTTP
class BadRequest(Exception):
    pass


def chat_options(body: dict) -> tuple[bool, object, bool]:
    """Thinking mode, reasoning effort and drop_thinking of a chat request. Top-level fields ("thinking",
    "reasoning_effort") win over `chat_template_kwargs` (as clients such as LiteLLM or vLLM-style requests send them:
    {"thinking": true}, {"enable_thinking": true}, {"reasoning_effort": ...}, {"drop_thinking": false}). A reasoning
    effort alone switches thinking on. Other chat_template_kwargs keys are ignored."""
    kw = body.get("chat_template_kwargs")
    if kw is None:
        kw = {}
    if not isinstance(kw, dict):
        raise BadRequest("chat_template_kwargs must be an object")
    effort = body.get("reasoning_effort", kw.get("reasoning_effort"))
    thinking = body.get("thinking", kw.get("thinking", kw.get("enable_thinking")))
    if thinking is None:
        thinking = effort is not None
    drop = kw.get("drop_thinking", True)
    for name, v in (("thinking", thinking), ("drop_thinking", drop)):
        if not isinstance(v, bool):
            raise BadRequest(f"{name} must be true or false")
    return thinking, effort, drop


def _finish_reason(di_reason: str, stopped: bool) -> str:
    if stopped:
        return "stop"
    return {"stop": "stop", "length": "length", "abort": "abort"}.get(di_reason, di_reason or "stop")


class Handler(BaseHTTPRequestHandler):
    server_version = "deepinfer-sm120/0.1"
    protocol_version = "HTTP/1.1"
    app: "App"

    def log_message(self, fmt, *args):  # route through logging
        log.debug("%s %s", self.address_string(), fmt % args)

    # ---- helpers
    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code: int, msg: str, typ: str = "invalid_request_error") -> None:
        self._json(code, {"error": {"message": msg, "type": typ, "code": code}})

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError as exc:
            raise BadRequest(f"invalid JSON: {exc}") from None

    # ---- routes
    def do_GET(self):
        if self.path in ("/health", "/v1/health"):
            ok = self.app.engine.alive
            pc = self.app.engine.prefix_cache
            return self._json(200 if ok else 503, {"status": "ok" if ok else "dead",
                                                   "stats": {**self.app.engine.stats, **_mtp_stats()},
                                                   "config": {"nextn": getattr(self.app.engine, "nextn", None),
                                                              "mtp_varlen": bool(self.app.args.mtp_varlen)},
                                                   "prefix_cache": pc.summary() if pc is not None else None})
        if self.path == "/v1/models":
            return self._json(200, {"object": "list", "data": [
                {"id": self.app.model_name, "object": "model", "created": self.app.created, "owned_by": "deepinfer-sm120",
                 "max_model_len": self.app.engine.max_model_len}]})
        self._error(404, f"no route {self.path}")

    def do_POST(self):
        try:
            body = self._body()
            if self.path == "/v1/completions":
                return self._generate(body, chat=False)
            if self.path == "/v1/chat/completions":
                return self._generate(body, chat=True)
            if self.path == "/internal/engram_cache_policy" and self.app.args.dp_rank is not None:
                # --dp rank (listens on 127.0.0.1 only): sent by the dp proxy once all ranks are up
                return self._json(200, {"result": _engram_cache_policy(self.app.args.ckpt)})
            self._error(404, f"no route {self.path}")
        except BadRequest as exc:
            self._error(400, str(exc))
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # noqa: BLE001
            log.exception("request failed")
            try:
                self._error(500, f"{type(exc).__name__}: {exc}", "server_error")
            except Exception:  # noqa: BLE001
                pass

    # ---- generation
    def _params(self, body: dict, prompt_len: int, extra_stop: list[str] = ()) -> dict:
        app = self.app
        if int(body.get("n", 1)) != 1:
            raise BadRequest("only n=1 is supported")
        max_tokens = body.get("max_completion_tokens", body.get("max_tokens"))
        room = app.engine.max_model_len - prompt_len - 1
        if room < 1:
            raise BadRequest(f"prompt has {prompt_len} tokens; max_model_len is {app.engine.max_model_len}")
        max_tokens = room if max_tokens is None else int(max_tokens)
        if max_tokens < 1:
            raise BadRequest("max_tokens must be >= 1")
        stop = body.get("stop") or []
        stop = [stop] if isinstance(stop, str) else list(stop)
        if len(stop) > 4 or not all(isinstance(s, str) for s in stop):
            raise BadRequest("stop must be a string or a list of up to 4 strings")
        temperature = float(body.get("temperature", 1.0))
        top_p = float(body.get("top_p", 1.0))
        if temperature < 0 or not 0 < top_p <= 1:
            raise BadRequest("temperature must be >= 0 and top_p in (0, 1]")
        seed = body.get("seed")
        # logprobs: completions `logprobs: int`, chat `logprobs: bool` + `top_logprobs: int` (raw logits, logprob_tap.py)
        if chat_mode := bool(body.get("messages")):
            lp = (int(body.get("top_logprobs") or 0) or 1) if body.get("logprobs") else 0
            top_n = int(body.get("top_logprobs") or 0) if body.get("logprobs") else 0
        else:
            lp = body.get("logprobs")
            top_n = 0 if lp is None else int(lp)
            lp = 0 if lp is None else max(1, int(lp))
        if not 0 <= top_n <= 20:
            raise BadRequest("logprobs / top_logprobs must be in 0..20")
        if lp and app.engine.nextn > 0:
            raise BadRequest("logprobs are not supported with MTP (--nextn > 0)")
        self._lp = (lp, top_n, bool(body.get("return_tokens_as_token_ids", False)), chat_mode)
        return dict(max_tokens=min(max_tokens, room), temperature=temperature, top_p=top_p,
                    top_k=int(body.get("top_k", -1) or -1), seed=None if seed is None else int(seed) % (2 ** 63),
                    ignore_eos=bool(body.get("ignore_eos", False)), stop=stop + list(extra_stop), logprobs=lp)

    def _generate(self, body: dict, chat: bool) -> None:
        app = self.app
        thinking = False
        plan, so, extra_stop, mm, n_image_tokens, n_images = tools.ToolPlan(), None, [], None, 0, 0
        if chat:
            msgs = body.get("messages")
            if not isinstance(msgs, list) or not msgs:
                raise BadRequest("messages must be a non-empty list")
            thinking, effort, drop_thinking = chat_options(body)
            enc = app.codec.enc
            try:
                plan = tools.plan_from_body(body)
                if enc is None and (plan.tools or plan.rf_kind):
                    raise tools.ToolError("tools / response_format need the checkpoint's encoding/encoding.py")
                if plan.tools or plan.rf_kind:
                    msgs = tools.prepare_messages(msgs, plan)
                    so = tools.so_args(enc, plan, thinking, app.codec.encode(enc.thinking_end_token))
            except tools.ToolError as exc:
                raise BadRequest(str(exc)) from None
            if plan.tools and plan.choice == "none":
                extra_stop = [tools.Dsml(enc).calls_open]  # the tools are shown to the model but must not be called
            try:
                prompt, image_records = app.codec.render_chat(msgs, thinking, effort, drop_thinking)
            except Exception as exc:  # noqa: BLE001  (encoding.py validates roles / content)
                raise BadRequest(f"cannot render messages: {exc}") from None
            ids = app.codec.encode(prompt)
            if image_records:
                if not app.engine.vision:
                    raise BadRequest("this model / engine has no image input (no multimodal processor)")
                try:
                    mm = images.load_images(image_records, app.image_limits)
                except images.ImageError as exc:
                    raise BadRequest(str(exc)) from None
                # each placeholder token becomes the image's token span inside the engine
                n_image_tokens = sum(app.engine.image_tokens(w, h) for w, h in mm["image_sizes"]) - len(image_records)
                n_images = len(image_records)  # (DeepInfer's add_request consumes mm's raw image keys)
        else:
            if body.get("tools") or (body.get("response_format") or {}).get("type", "text") != "text":
                raise BadRequest("tools and response_format are supported on /v1/chat/completions only")
            p = body.get("prompt")
            if isinstance(p, list) and len(p) == 1 and isinstance(p[0], str):
                p = p[0]
            if isinstance(p, str):
                ids = app.codec.encode(p)
            elif isinstance(p, list) and p and all(isinstance(x, int) for x in p):
                ids = list(p)
            else:
                raise BadRequest("prompt must be a string, a one-element list of strings, or a list of token ids")
        if not ids:
            raise BadRequest("empty prompt")
        prm = self._params(body, len(ids) + n_image_tokens, extra_stop)
        stop = prm.pop("stop")
        req = app.engine.submit(GenRequest(prompt_ids=ids, so_args=so, multi_modal_data=mm, **prm))
        rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
        created = int(time.time())
        detok = Detok(app.codec, stop)
        stream = bool(body.get("stream", False))
        include_usage = bool((body.get("stream_options") or {}).get("include_usage", False))
        obj = "chat.completion" if chat else "text_completion"
        log.info("%s engine-id %s prompt %d tok max_tokens %d stream %s tools %s/%d response_format %s grammar %s "
                 "images %d (%d tok)", rid, req.request_id, len(ids) + n_image_tokens, prm["max_tokens"], stream,
                 plan.choice, len(plan.tools), plan.rf_kind, so is not None, n_images, n_image_tokens)
        # chat output -> reasoning_content / content / tool_calls deltas (tools.py); plain text without encoding.py
        parser = tools.StreamParser(app.codec.enc, thinking, plan) if chat and app.codec.enc is not None else None

        def chunk(delta_text: str | None, finish: str | None, role: bool = False, usage=None, delta=None) -> dict:
            if chat:
                delta = dict(delta or {})
                if role:
                    delta["role"] = "assistant"
                if delta_text:
                    delta["content"] = delta_text
                ch = {"index": 0, "delta": delta, "finish_reason": finish}
            else:
                ch = {"index": 0, "text": delta_text or "", "logprobs": None, "finish_reason": finish}
            d = {"id": rid, "object": obj + (".chunk" if chat else ""), "created": created, "model": app.model_name,
                 "choices": [ch]}
            if usage is not None:
                d["usage"] = usage
            return d

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

        def send(d: dict) -> None:
            self.wfile.write(b"data: " + json.dumps(d, ensure_ascii=False).encode() + b"\n\n")
            self.wfile.flush()

        finish, n_tok, err, out_ids, lp_rows = None, 0, None, [], []
        try:
            if stream and chat:
                send(chunk(None, None, role=True))
            while True:
                try:
                    msg = req.out.get(timeout=app.request_timeout)
                except queue.Empty:
                    err = "timeout waiting for the engine"
                    break
                if msg[0] == "error":
                    err = msg[1]
                    break
                _, tok, finished, reason, row = msg
                n_tok += 1
                out_ids.append(tok)
                lp_rows.append(row)
                eos = tok == app.engine.eos_id
                if not eos:
                    detok.push(tok)
                if detok.stopped and not finished:
                    app.engine.abort(req.request_id)
                    finished, reason = True, "stop"
                if stream:
                    piece = detok.emit(final=finished)
                    if piece and parser is not None:
                        for d in parser.feed(piece):
                            send(chunk(None, None, delta=d))
                    elif piece:
                        send(chunk(piece, None))
                if finished:
                    finish = _finish_reason(reason, detok.stopped)
                    break
        except (BrokenPipeError, ConnectionResetError):
            app.engine.abort(req.request_id)  # client went away: cancel the generation
            log.info("%s client disconnected after %d tokens: aborted", rid, n_tok)
            return
        if err is not None:
            app.engine.abort(req.request_id)
            if stream:
                send({"error": {"message": err, "type": "server_error"}})
                self.wfile.write(b"data: [DONE]\n\n")
                return
            return self._error(500, err, "server_error")
        n_prompt = len(ids) + n_image_tokens
        usage = {"prompt_tokens": n_prompt, "completion_tokens": n_tok, "total_tokens": n_prompt + n_tok}
        t = req.first_token_t
        log.info("%s done: %d tokens, finish %s, ttft %.3fs, total %.3fs", rid, n_tok, finish,
                 (t - req.submitted) if t else -1, time.perf_counter() - req.submitted)
        if stream:
            if parser is not None:
                for d in parser.finish():
                    send(chunk(None, None, delta=d))
                if parser.calls and finish == "stop":
                    finish = "tool_calls"
            send(chunk(None, finish))
            if include_usage:
                d = chunk(None, None, usage=usage)
                d["choices"] = []
                send(d)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        text = detok.text
        if chat:
            calls = []
            if app.codec.enc is not None:
                content, reasoning, calls = tools.parse_final(app.codec.enc, text, thinking, plan)
            else:
                content, reasoning = app.codec.split_reasoning(text, thinking)
            message = {"role": "assistant", "content": content}
            if reasoning is not None:
                message["reasoning_content"] = reasoning
            if calls:
                message["tool_calls"] = calls
                if finish == "stop":
                    finish = "tool_calls"
            choice = {"index": 0, "message": message, "finish_reason": finish}
        else:
            choice = {"index": 0, "text": text, "logprobs": None, "finish_reason": finish}
        if body.get("return_token_ids"):  # extension (tests): engine token ids incl. EOS, and the prompt ids
            choice["token_ids"], choice["prompt_token_ids"] = out_ids, ids
        if self._lp[0]:
            choice["logprobs"] = self._logprobs_obj(out_ids, lp_rows, chat)
        self._json(200, {"id": rid, "object": obj, "created": created, "model": app.model_name, "choices": [choice],
                         "usage": usage})


def _tok_str(app, tid: int, as_ids: bool) -> str:
    return f"token_id:{tid}" if as_ids else app.codec.tok.decode([tid], skip_special_tokens=False)


def _logprobs_obj(self, out_ids, rows, chat: bool):
    """OpenAI logprobs for the generated tokens: completions {tokens, token_logprobs, top_logprobs, text_offset},
    chat {content: [{token, logprob, bytes, top_logprobs}]}. The chosen token's logprob comes from the top-20 row;
    None if it is not in it (sampling only; greedy always picks top-1)."""
    app = self.app
    _, top_n, as_ids, _ = self._lp
    toks, lps, tops, offs, content, off = [], [], [], [], [], 0
    for tid, row in zip(out_ids, rows):
        t = _tok_str(app, tid, as_ids)
        lp = None
        top = []
        if row is not None:
            ids_, vals = row
            lp = next((v for i, v in zip(ids_, vals) if i == tid), None)
            top = [(_tok_str(app, i, as_ids), v) for i, v in list(zip(ids_, vals))[:top_n]]
        if chat:
            content.append({"token": t, "logprob": lp if lp is not None else -9999.0, "bytes": list(t.encode()),
                            "top_logprobs": [{"token": a, "logprob": b, "bytes": list(a.encode())} for a, b in top]})
        else:
            toks.append(t)
            lps.append(lp)
            tops.append(dict(top) if top_n else None)
            offs.append(off)
            off += len(t)
    return {"content": content} if chat else {"tokens": toks, "token_logprobs": lps, "top_logprobs": tops,
                                              "text_offset": offs}


Handler._logprobs_obj = _logprobs_obj


class App:
    def __init__(self, args):
        self.args = args
        self.model_name = args.model_name or os.path.basename(os.path.normpath(args.ckpt))
        self.created = int(time.time())
        self.request_timeout = args.request_timeout
        self.codec = TextCodec(args.ckpt)
        self.image_limits = images.ImageLimits(max_images=args.max_images, max_bytes=args.max_image_bytes,
                                               max_pixels=args.max_image_pixels, fetch_urls=not args.no_image_urls,
                                               fetch_private=args.image_fetch_private,
                                               fetch_timeout=args.image_fetch_timeout)
        self.engine = EngineLoop(args.ckpt, max_num_seqs=args.max_num_seqs, max_model_len=args.max_model_len,
                                 max_num_batched_tokens=args.max_num_batched_tokens,
                                 gpu_memory_utilization=args.gpu_memory_utilization, max_blocks=args.max_blocks,
                                 nextn=args.nextn, graphs=args.graphs, zos=args.zos,
                                 prefill_interval=args.prefill_interval,
                                 prefill_any_interval=args.prefill_any_interval, deterministic=args.deterministic,
                                 dp_size=args.dp, dp_rank=args.dp_rank or 0, ep_size=args.ep,
                                 mtp_varlen=args.mtp_varlen,
                                 prefix_cache=dict(grain=args.prefix_cache_grain, checkpoint_stride=args.prefix_cache_checkpoint_stride,
                                                   max_entries=args.prefix_cache_entries,
                                                   max_block_frac=args.prefix_cache_max_block_frac)
                                 if args.prefix_cache else None)


_MAIN_PID = os.getpid()


def _mtp_stats() -> dict:
    """P12 counters (mtp_verified / mtp_accepted draft tokens, mtp_seq_steps); empty if the patches are not loaded."""
    mod = sys.modules.get("sm120_patches")
    return mod.mtp_stats() if mod is not None and hasattr(mod, "mtp_stats") else {}


def _yoco_compact_env(args, ap) -> None:
    """--yoco-compact KEEP -> DeepInfer's compact YOCO prefill truncation; refused together with --prefix-cache.

    With truncation, the consumer layers (>= n_prefill_layer) compute only the last KEEP rows of a prompt, so their
    per-sequence state at a prefix position depends on the prompt's total length: a snapshot taken inside prompt A is
    not the state prompt B would have there (tests/test_prefix_cache.py on ckpt_yoco: 3/17 exact)."""
    mode = os.environ.get("DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE", "off")
    if args.yoco_compact:
        keep = int(args.yoco_compact)
        need = 128 + max(0, args.nextn) + 1
        if keep < need:
            ap.error(f"--yoco-compact {keep}: KEEP must be >= 128 + nextn + 1 = {need} (DSpark window + verify rows)")
        os.environ["DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE"] = mode = "compact"
        os.environ["DEEPINFER_YOCO_PREFILL_KEEP_TOKENS"] = str(keep)
    if mode != "off" and args.prefix_cache:
        ap.error(f"YOCO prefill truncation ({mode}) is not exact with --prefix-cache: drop one of them")


_CACHE_POLICY: dict = {}


def _engram_cache_policy(ckpt: str) -> dict | None:
    """sm120_engram.post_load_cache_policy once per process (host work only: page-cache advice, prefault reads,
    residency), before the server accepts requests: a prefault running while serving starves the prefill's own
    engram reads. None if it already ran or Engram is not loaded."""
    eng = sys.modules.get("sm120_engram")
    if eng is None or not (getattr(eng, "_TABLES", None) or getattr(eng, "_SHARED_TABLES", None)) or _CACHE_POLICY:
        return None
    _CACHE_POLICY["started"] = time.time()
    try:
        _CACHE_POLICY.update(eng.post_load_cache_policy(ckpt))
    except eng.EngramResidencyError as exc:  # SM120_ENGRAM_PIN cannot be honoured: abort the start-up
        log.error("engram tables cannot be pinned: %s", exc)
        _CACHE_POLICY["fatal"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        log.warning("engram page-cache policy failed: %r", exc)
        _CACHE_POLICY["error"] = repr(exc)
    return dict(_CACHE_POLICY)


def _sigterm(signum, frame):
    """docker stop / kill -TERM in the server process: unwind through the finally blocks (KeyboardInterrupt).

    Processes forked from it (DeepInfer's multimodal workers) inherit this handler; DeepInfer stops them with
    terminate() and they have nothing to clean up, so they exit at once instead of printing a KeyboardInterrupt
    traceback from inside their queue read."""
    if os.getpid() != _MAIN_PID:
        os._exit(0)
    raise KeyboardInterrupt


def _exit_start_failed(exc: BaseException, dp_rank) -> None:
    """The engine did not come up: exit non-zero at once (os._exit), never through the interpreter's teardown. After a
    failed start (a CUDA-graph capture that raised, NCCL communicators with work in flight, DeepInfer's threads)
    that teardown can block forever, and with --dp the proxy waits for the rank to exit. Python's resource tracker
    still unlinks the shared-memory segments of this process."""
    log.error("engine start failed%s: %r; exiting with 3", "" if dp_rank is None else f" (rank {dp_rank})", exc)
    for h in logging.getLogger().handlers:
        try:
            h.flush()
        except Exception:  # noqa: BLE001
            pass
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(3)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model-name", default=None, help="id reported by /v1/models (default: ckpt dir name)")
    ap.add_argument("--max-num-seqs", type=int, default=8)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--max-num-batched-tokens", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-blocks", type=int, default=256, help="SM120_MAX_GPU_BLOCKS")
    ap.add_argument("--nextn", type=int, default=0, help="MTP/DSpark draft tokens (needs mtp.* layers)")
    ap.add_argument("--graphs", action="store_true", help="CUDA graphs (default eager)")
    ap.add_argument("--mtp-varlen", action="store_true",
                    help="adaptive MTP: per step, verify only as many drafts (<= --nextn per request) as the profiled "
                         "cost curve and the draft confidences pay for (DEEPINFER_ENABLE_MTP_VARLEN); needs --graphs")
    ap.add_argument("--zos", choices=("default", "0", "1"), default="default")
    ap.add_argument("--prefill-interval", type=int, default=1,
                    help="DEEPINFER_PREFILL_ATTEMPT_INTERVAL (DeepInfer default 70 decode steps; 1 = lowest TTFT)")
    ap.add_argument("--prefill-any-interval", type=int, default=1,
                    help="DEEPINFER_PREFILL_ANY_ATTEMPT_INTERVAL: with --dp > 1, how many decode steps a rank that alone "
                         "has a waiting prompt waits before all ranks do a prefill step (DeepInfer default 100; "
                         "1 = lowest TTFT)")
    ap.add_argument("--deterministic", action="store_true",
                    help="SM120_MOE_IMPL=torch: run-to-run reproducible outputs (b12x fused_moe is not), slower")
    ap.add_argument("--request-timeout", type=float, default=600.0, help="max seconds between two tokens")
    ap.add_argument("--yoco-compact", type=int, nargs="?", const=140, default=0, metavar="KEEP",
                    help="YOCO compact prefill truncation (DeepInfer DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact): layers "
                         ">= n_prefill_layer run only the last KEEP rows of each prefill (default 140, >= 128 + nextn + 1). "
                         "About 1.5x prefill on long prompts. Opt-in; not exact with --prefix-cache (refused)")
    ap.add_argument("--prefix-cache", action="store_true",
                    help="reuse computed prompt prefixes (prefix_cache; needs the patched _C: "
                         "PYTHONPATH=/opt/sm120/sm120:/work/deepinfer_pc DEEPINFER_ROOT=/work/deepinfer_pc)")
    ap.add_argument("--prefix-cache-grain", type=int, default=512,
                    help="snapshot granularity: prefixes are recorded at chunk ends that are multiples of N (multiple "
                         "of 128), plus at every prompt end")
    ap.add_argument("--prefix-cache-checkpoint-stride", type=int, default=0,
                    help="split prefills at multiples of N for snapshots (rounded up to the grain). 0 (default): "
                         "max(grain, --max-num-batched-tokens), i.e. no extra engine steps; a smaller stride gives "
                         "finer prefix hits but costs one prefill step per checkpoint")
    ap.add_argument("--prefix-cache-entries", type=int, default=64, help="max cached prefix snapshots (state slots)")
    ap.add_argument("--prefix-cache-max-block-frac", type=float, default=0.5,
                    help="max fraction of KV blocks the cache may keep pinned")
    ap.add_argument("--dp", type=int, default=1,
                    help="data-parallel ranks = GPUs (DP attention + EP MoE, docs/MULTI_GPU.md): one engine process per "
                         "rank on ports PORT+1.., and a least-loaded proxy on --port (dp_proxy.py)")
    ap.add_argument("--ep", type=int, default=None, help="expert-parallel size (default: --dp)")
    ap.add_argument("--dp-rank", type=int, default=None, help=argparse.SUPPRESS)  # set by the proxy for its children
    ap.add_argument("--force-proxy", action="store_true", help=argparse.SUPPRESS)  # tests: proxy even with --dp 1
    ap.add_argument("--max-images", type=int, default=8, help="images per chat request")
    ap.add_argument("--max-image-bytes", type=int, default=20 << 20, help="encoded size per image")
    ap.add_argument("--max-image-pixels", type=int, default=50_000_000, help="width x height per image")
    ap.add_argument("--no-image-urls", action="store_true", help="refuse http(s) image URLs (data: URLs only)")
    ap.add_argument("--image-fetch-private", action="store_true",
                    help="allow image URLs on loopback / private / link-local addresses (off: SSRF protection)")
    ap.add_argument("--image-fetch-timeout", type=float, default=10.0, help="seconds per image URL fetch")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    _yoco_compact_env(args, ap)
    # docker stop / kill -TERM: run the finally blocks (engine shutdown here, rank shutdown in the dp proxy) instead
    # of dying in place; the default SIGTERM action skips them.
    signal.signal(signal.SIGTERM, _sigterm)
    if args.mtp_varlen and (args.nextn <= 0 or not args.graphs):
        ap.error("--mtp-varlen needs --nextn > 0 and --graphs (DeepInfer profiles the decode graphs for its cost model)")
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if args.dp_rank is None:  # once per server, not in the per-GPU children of --dp
        import banner

        banner.print_banner(f"v{banner.version()}  ·  {os.path.basename(os.path.normpath(args.ckpt))}"
                            f"  ·  dp {args.dp}  ·  :{args.port}")
    # Engram prefault (SM120_ENGRAM_PREFAULT=1) only after every rank has loaded its weights: see _engram_cache_policy
    os.environ.setdefault("SM120_ENGRAM_PREFAULT_AT", "ready")
    if args.dp_rank is None and (args.dp > 1 or args.force_proxy):
        import dp_proxy

        rest, skip = [], 0
        for a in sys.argv[1:]:
            if skip:
                skip -= 1
                continue
            if a in ("--host", "--port", "--dp"):
                skip = 1
                continue
            if a == "--force-proxy" or a.startswith(("--host=", "--port=", "--dp=")):
                continue
            rest.append(a)
        return dp_proxy.run(rest, args.dp, args.host, args.port)
    if args.dp_rank is None:  # SM120_ENGRAM_MODE=pinned: the shared table segment exists before the engine starts
        try:
            import sm120_engram

            sm120_engram.create_shared_tables(args.ckpt, ranks=1)
        except Exception as exc:  # noqa: BLE001 - EngramResidencyError: does not fit; refuse to start
            log.error("%s", exc)
            return 2
    app = App(args)
    try:
        app.engine.start()
    except Exception as exc:  # noqa: BLE001 - e.g. a CUDA-graph capture failure
        _exit_start_failed(exc, args.dp_rank)
    if args.dp_rank is None:  # single process: weights are loaded now; with --dp the proxy calls it on rank 0
        if (_engram_cache_policy(args.ckpt) or {}).get("fatal"):
            app.engine.shutdown()
            return 2
    Handler.app = app
    ThreadingHTTPServer.request_queue_size = 1024  # listen backlog (socketserver default 5)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    log.info("serving %s on http://%s:%d", app.model_name, args.host, args.port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        app.engine.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
