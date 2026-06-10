"""Tests for tools/browser.py."""

from __future__ import annotations

import asyncio
import importlib.util

import pytest

# Skip if playwright isn't installed (optional dep)
playwright_available = importlib.util.find_spec("playwright") is not None
pytestmark = pytest.mark.skipif(
    not playwright_available, reason="playwright not installed (install with: ipw[browser])"
)

if playwright_available:
    from ipw.tools.browser import BrowserTool


class TestBrowserToolMetadata:
    """Tests that don't require an actual browser launch."""

    def test_spec(self) -> None:
        spec = BrowserTool.spec
        assert spec.name == "browser"
        assert spec.requires_network is True
        assert "url" in spec.parameters

    def test_invalid_url_returns_error(self) -> None:
        result = asyncio.run(BrowserTool().run(url=""))
        assert result.success is False


@pytest.mark.integration
class TestBrowserToolIntegration:
    """Tests that launch a real Playwright browser. Requires `playwright install chromium`."""

    def test_fetch_file_url(self, tmp_path) -> None:
        html_file = tmp_path / "test.html"
        html_file.write_text("<html><body><h1>Hello Browser</h1></body></html>")
        url = f"file://{html_file}"

        result = asyncio.run(BrowserTool().run(url=url))
        assert result.success is True
        assert "Hello Browser" in result.content

    def test_timeout_on_unreachable_url(self) -> None:
        # http://127.0.0.1:1 — should reject quickly
        result = asyncio.run(BrowserTool().run(url="http://127.0.0.1:1", timeout=2.0))
        assert result.success is False
        # Either timeout or connection refused — both acceptable
        assert any(kw in (result.error or "").lower()
                   for kw in ("timeout", "connection", "refused", "err_"))

    def test_wait_for_selector(self, tmp_path) -> None:
        html_file = tmp_path / "test.html"
        html_file.write_text(
            "<html><body>"
            "<div id='loaded'>READY</div>"
            "</body></html>"
        )
        url = f"file://{html_file}"

        result = asyncio.run(BrowserTool().run(url=url, wait_for="#loaded"))
        assert result.success is True
        assert "READY" in result.content
