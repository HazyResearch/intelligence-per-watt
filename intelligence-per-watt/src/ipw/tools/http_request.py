"""http_request — perform an HTTP request and return the response body.

Ported from /home/ubuntu/lambda-stanford/jonsf/OpenJarvis/src/openjarvis/tools/http_request.py
Uses httpx async client. Method allowlist enforces only standard methods;
URL scheme check requires http:// or https://. Response body truncated to 128KB.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ipw.tools.base import BaseTool, ToolResult, ToolSpec
from ipw.tools.registry import ToolRegistry

_ALLOWED_METHODS = frozenset({"GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"})
_MAX_RESPONSE_BYTES = 128 * 1024


@ToolRegistry.register("http_request")
class HttpRequestTool(BaseTool):
    spec = ToolSpec(
        name="http_request",
        description="Perform an HTTP request and return the response body (truncated to 128KB).",
        parameters={
            "url": {"type": "string", "description": "Target URL (http or https)"},
            "method": {"type": "string", "description": "HTTP method",
                       "enum": sorted(_ALLOWED_METHODS)},
            "headers": {"type": "object", "description": "Optional request headers"},
            "body": {"type": "string", "description": "Optional request body"},
            "timeout": {"type": "number", "description": "Seconds (default: 30)"},
        },
        requires_network=True,
    )

    async def run(
        self,
        url: str = "",
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Any] = None,
        timeout: float = 30.0,
        **kwargs,
    ) -> ToolResult:
        method_upper = method.upper() if method else ""
        if method_upper not in _ALLOWED_METHODS:
            return ToolResult(content="", success=False,
                              error=f"method {method!r} not in allowlist")
        if not url or not (url.startswith("http://") or url.startswith("https://")):
            return ToolResult(content="", success=False,
                              error=f"url {url!r} not http/https")

        try:
            import httpx
        except ImportError:
            return ToolResult(content="", success=False, error="httpx not installed")

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.request(
                    method_upper, url, headers=headers or {}, content=body,
                )
                text = response.text[:_MAX_RESPONSE_BYTES]
                return ToolResult(
                    content=text,
                    success=200 <= response.status_code < 300,
                    error=None if 200 <= response.status_code < 300
                          else f"HTTP {response.status_code}",
                    metadata={
                        "status_code": response.status_code,
                        "headers": dict(response.headers),
                    },
                )
        except Exception as exc:
            return ToolResult(content="", success=False, error=str(exc))


__all__ = ["HttpRequestTool"]
