"""OpenAI-compatible proxy that records actual API token usage."""

from __future__ import annotations

import json
import logging
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

LOGGER = logging.getLogger(__name__)


EventCallback = Callable[[str], None] | Callable[..., None]


class OpenAIUsageProxy:
    """Forward OpenAI-compatible requests while recording response ``usage``.

    The proxy is intentionally small and local. It does not estimate tokens; it
    only records token counts that the upstream API returns in response bodies.
    """

    def __init__(
        self,
        target_base_url: str,
        *,
        bind_host: str = "0.0.0.0",
        model: str | None = None,
        event_callback: Callable[..., None] | None = None,
    ) -> None:
        parsed = urllib.parse.urlsplit(target_base_url.rstrip("/"))
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid target_base_url: {target_base_url!r}")
        self._target_origin = f"{parsed.scheme}://{parsed.netloc}"
        self._target_base_path = parsed.path.rstrip("/")
        self._bind_host = bind_host
        self._model = model
        self._event_callback = event_callback
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._calls: list[dict[str, int | str]] = []
        self._missing_usage_responses = 0

    @property
    def port(self) -> int:
        if self._server is None:
            raise RuntimeError("Proxy has not been started")
        return int(self._server.server_address[1])

    @property
    def base_path(self) -> str:
        return self._target_base_path

    def start(self) -> None:
        if self._server is not None:
            return

        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                proxy._handle(self)

            def do_POST(self) -> None:  # noqa: N802
                proxy._handle(self)

            def log_message(self, format: str, *args: Any) -> None:
                LOGGER.debug("OpenAI usage proxy: " + format, *args)

        self._server = ThreadingHTTPServer((self._bind_host, 0), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="ipw-openai-usage-proxy",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        server = self._server
        if server is None:
            return
        server.shutdown()
        server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None

    def reset(self) -> None:
        with self._lock:
            self._calls.clear()
            self._missing_usage_responses = 0

    def base_url_for_client(self, host: str) -> str:
        path = self._target_base_path
        return f"http://{host}:{self.port}{path}"

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            calls = list(self._calls)
            missing = self._missing_usage_responses
        has_complete_usage = bool(calls) and missing == 0
        input_tokens = (
            sum(int(call.get("prompt_tokens") or 0) for call in calls)
            if has_complete_usage
            else None
        )
        output_tokens = (
            sum(int(call.get("completion_tokens") or 0) for call in calls)
            if has_complete_usage
            else None
        )
        total_tokens = (
            sum(int(call.get("total_tokens") or 0) for call in calls)
            if has_complete_usage
            else None
        )
        if has_complete_usage and not total_tokens:
            total_tokens = int(input_tokens or 0) + int(output_tokens or 0)
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "num_turns": len(calls) + missing,
            "cost": 0.0 if has_complete_usage else None,
            "token_source": "openai_api_usage" if has_complete_usage else "missing",
            "missing_usage_responses": missing,
        }

    def _target_url(self, request_path: str) -> str:
        path = request_path or "/"
        if self._target_base_path and not (
            path == self._target_base_path
            or path.startswith(f"{self._target_base_path}/")
        ):
            path = f"{self._target_base_path}{path if path.startswith('/') else '/' + path}"
        return f"{self._target_origin}{path}"

    def _handle(self, handler: BaseHTTPRequestHandler) -> None:
        body = handler.rfile.read(int(handler.headers.get("Content-Length", "0") or 0))
        headers = {
            key: value
            for key, value in handler.headers.items()
            if key.lower()
            not in {"host", "content-length", "connection", "transfer-encoding"}
        }
        headers["Accept-Encoding"] = "identity"
        target_url = self._target_url(handler.path)
        is_completion = self._is_completion_path(handler.path)

        if is_completion:
            self._record_event("lm_inference_start", model=self._model)

        request = urllib.request.Request(
            target_url,
            data=body if handler.command.upper() != "GET" else None,
            headers=headers,
            method=handler.command,
        )
        status = 502
        response_headers: dict[str, str] = {}
        response_body = b""
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                status = int(response.status)
                response_headers = dict(response.headers.items())
                response_body = response.read()
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            response_headers = dict(exc.headers.items())
            response_body = exc.read()
        except Exception as exc:
            LOGGER.warning("OpenAI usage proxy forwarding failed: %s", exc)
            status = 502
            response_body = json.dumps({"error": str(exc)}).encode("utf-8")
            response_headers = {"Content-Type": "application/json"}

        usage = self._record_usage(handler.path, response_body, response_headers)
        if is_completion:
            self._record_event(
                "lm_inference_end",
                model=self._model,
                prompt_tokens=usage.get("prompt_tokens") if usage else None,
                completion_tokens=usage.get("completion_tokens") if usage else None,
                total_tokens=usage.get("total_tokens") if usage else None,
                token_source="openai_api_usage" if usage else "missing",
            )

        handler.send_response(status)
        for key, value in response_headers.items():
            if key.lower() in {
                "connection",
                "content-length",
                "transfer-encoding",
                "content-encoding",
            }:
                continue
            handler.send_header(key, value)
        handler.send_header("Content-Length", str(len(response_body)))
        handler.end_headers()
        handler.wfile.write(response_body)

    def _record_event(self, event_type: str, **metadata: Any) -> None:
        if self._event_callback is None:
            return
        try:
            self._event_callback(event_type, **metadata)
        except Exception:
            LOGGER.debug("OpenAI usage proxy event callback failed", exc_info=True)

    @staticmethod
    def _is_completion_path(path: str) -> bool:
        lowered = path.lower()
        return "/chat/completions" in lowered or lowered.endswith("/completions")

    def _record_usage(
        self,
        path: str,
        response_body: bytes,
        response_headers: dict[str, str],
    ) -> dict[str, int]:
        if not self._is_completion_path(path):
            return {}
        usage = self._extract_usage(response_body, response_headers)
        if usage:
            record = {
                "path": path,
                "prompt_tokens": int(
                    usage.get("prompt_tokens")
                    or usage.get("input_tokens")
                    or 0
                ),
                "completion_tokens": int(
                    usage.get("completion_tokens")
                    or usage.get("output_tokens")
                    or 0
                ),
                "total_tokens": int(usage.get("total_tokens") or 0),
            }
            if not record["total_tokens"]:
                record["total_tokens"] = (
                    int(record["prompt_tokens"]) + int(record["completion_tokens"])
                )
            with self._lock:
                self._calls.append(record)
            return {
                "prompt_tokens": int(record["prompt_tokens"]),
                "completion_tokens": int(record["completion_tokens"]),
                "total_tokens": int(record["total_tokens"]),
            }
        with self._lock:
            self._missing_usage_responses += 1
        return {}

    @staticmethod
    def _extract_usage(
        response_body: bytes,
        response_headers: dict[str, str],
    ) -> dict[str, Any]:
        content_type = ""
        for key, value in response_headers.items():
            if key.lower() == "content-type":
                content_type = value.lower()
                break
        text = response_body.decode("utf-8", errors="ignore")
        if "text/event-stream" in content_type:
            found: dict[str, Any] = {}
            for line in text.splitlines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                payload = line.removeprefix("data:").strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                usage = obj.get("usage")
                if isinstance(usage, dict) and usage:
                    found = usage
            return found
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return {}
        usage = obj.get("usage") if isinstance(obj, dict) else None
        return usage if isinstance(usage, dict) else {}
