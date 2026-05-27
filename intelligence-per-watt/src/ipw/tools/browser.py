"""browser — Playwright-based tool that fetches a URL and returns rendered page text.

Ported from OpenJarvis tools/browser.py. Uses async_playwright for headless chromium.
Playwright is an optional dependency (install with: ipw[browser]).
"""

from __future__ import annotations

from typing import Any, Optional

from ipw.tools.base import BaseTool, ToolResult, ToolSpec
from ipw.tools.registry import ToolRegistry

_MAX_CONTENT_BYTES = 128 * 1024  # 128 KB


@ToolRegistry.register("browser")
class BrowserTool(BaseTool):
    """Fetch a URL with a headless Chromium browser and return the rendered body text."""

    spec = ToolSpec(
        name="browser",
        description=(
            "Fetch a URL using a headless Chromium browser and return the rendered page text. "
            "Supports waiting for a CSS selector to appear before extracting content."
        ),
        parameters={
            "url": {
                "type": "string",
                "description": "URL to navigate to (http, https, or file://).",
            },
            "wait_for": {
                "type": "string",
                "description": (
                    "Optional CSS selector to wait for before extracting text. "
                    "If omitted, extraction starts after page load."
                ),
            },
            "timeout": {
                "type": "number",
                "description": "Navigation timeout in seconds (default: 30).",
            },
        },
        requires_network=True,
    )

    async def run(
        self,
        url: str = "",
        wait_for: Optional[str] = None,
        timeout: float = 30.0,
        **kwargs: Any,
    ) -> ToolResult:
        """Navigate to url, optionally wait for a CSS selector, extract body text."""

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

                if wait_for:
                    try:
                        await page.wait_for_selector(wait_for, timeout=timeout_ms)
                    except Exception as exc:
                        return ToolResult(
                            content="",
                            success=False,
                            error=str(exc),
                        )

                text = await page.inner_text("body")

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


__all__ = ["BrowserTool"]
