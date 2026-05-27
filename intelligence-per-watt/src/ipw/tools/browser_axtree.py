"""browser_axtree — Playwright-based tool that fetches a URL and returns the accessibility tree.

Ported from OpenJarvis tools/browser_axtree.py. Uses async_playwright for headless chromium.
Returns a structured indented text representation of the AX tree (role + name + value per node).
Playwright is an optional dependency (install with: ipw[browser]).
"""

from __future__ import annotations

from typing import Any

from ipw.tools.base import BaseTool, ToolResult, ToolSpec
from ipw.tools.registry import ToolRegistry

_MAX_CONTENT_BYTES = 128 * 1024  # 128 KB


@ToolRegistry.register("browser_axtree")
class BrowserAxTreeTool(BaseTool):
    """Fetch a URL with a headless Chromium browser and return the accessibility tree as text."""

    spec = ToolSpec(
        name="browser_axtree",
        description=(
            "Fetch a URL using a headless Chromium browser and return the accessibility tree "
            "as indented text (one line per node: role, name, value). More structured than "
            "raw HTML — useful for agents that reason over page structure."
        ),
        parameters={
            "url": {
                "type": "string",
                "description": "URL to navigate to (http, https, or file://).",
            },
            "timeout": {
                "type": "number",
                "description": "Navigation timeout in seconds (default: 30).",
            },
            "max_depth": {
                "type": "integer",
                "description": "Maximum AX tree depth to traverse (default: 10).",
            },
        },
        requires_network=True,
    )

    async def run(
        self,
        url: str = "",
        timeout: float = 30.0,
        max_depth: int = 10,
        **kwargs: Any,
    ) -> ToolResult:
        """Navigate to url, snapshot the accessibility tree, return as indented text."""

        # Validate URL before touching Playwright
        if not url:
            return ToolResult(
                content="",
                success=False,
                error="url must be a non-empty string",
            )

        # Lazy-import Playwright — optional dep
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return ToolResult(
                content="",
                success=False,
                error="playwright not installed (install with: ipw[browser])",
            )

        timeout_ms = int(timeout * 1000)
        browser = None

        try:
            async with async_playwright() as p:
                launch_kwargs: dict[str, Any] = {"headless": True}

                # If a workspace dir is set, isolate downloads there
                context_kwargs: dict[str, Any] = {}
                if self._default_cwd:
                    context_kwargs["downloads_path"] = self._default_cwd

                browser = await p.chromium.launch(**launch_kwargs)
                context = await browser.new_context(**context_kwargs)
                page = await context.new_page()

                try:
                    await page.goto(url, timeout=timeout_ms)
                except Exception as exc:
                    return ToolResult(
                        content="",
                        success=False,
                        error=str(exc),
                    )

                # aria_snapshot() returns a YAML-like string of the AX tree (Playwright >= 1.46)
                text = await page.aria_snapshot(depth=max_depth)
                if not text or not text.strip():
                    return ToolResult(
                        content="",
                        success=False,
                        error="No accessibility tree available for this page.",
                    )

                # Truncate to 128 KB
                if len(text.encode("utf-8")) > _MAX_CONTENT_BYTES:
                    text = text.encode("utf-8")[:_MAX_CONTENT_BYTES].decode(
                        "utf-8", errors="replace"
                    ) + "\n\n[Content truncated]"

                return ToolResult(
                    content=text,
                    success=True,
                    metadata={"url": url},
                )
        except Exception as exc:
            return ToolResult(
                content="",
                success=False,
                error=str(exc),
            )
        finally:
            # Always close the browser — even on exception paths above
            if browser is not None:
                try:
                    await browser.close()
                except Exception:
                    pass

__all__ = ["BrowserAxTreeTool"]
