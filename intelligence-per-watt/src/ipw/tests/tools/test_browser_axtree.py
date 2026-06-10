"""Tests for tools/browser_axtree.py."""

from __future__ import annotations

import asyncio
import importlib.util

import pytest

playwright_available = importlib.util.find_spec("playwright") is not None
pytestmark = pytest.mark.skipif(
    not playwright_available, reason="playwright not installed"
)


from ipw.tools.browser_axtree import BrowserAxTreeTool  # noqa: E402


class TestBrowserAxTreeMetadata:
    def test_spec(self) -> None:
        spec = BrowserAxTreeTool.spec
        assert spec.name == "browser_axtree"
        assert spec.requires_network is True
        assert "url" in spec.parameters

    def test_empty_url_returns_error(self) -> None:
        result = asyncio.run(BrowserAxTreeTool().run(url=""))
        assert result.success is False


@pytest.mark.integration
class TestBrowserAxTreeIntegration:
    def test_extracts_button_from_simple_page(self, tmp_path) -> None:
        html = tmp_path / "test.html"
        html.write_text(
            "<html><body>"
            "<h1>Title</h1>"
            "<button>Click me</button>"
            "<a href='#'>A link</a>"
            "</body></html>"
        )
        url = f"file://{html}"
        result = asyncio.run(BrowserAxTreeTool().run(url=url))
        assert result.success is True
        # The ax-tree should contain at least the button's role + name
        assert "button" in result.content.lower()
        assert "Click me" in result.content

    def test_extracts_heading(self, tmp_path) -> None:
        html = tmp_path / "h.html"
        html.write_text("<html><body><h1>AxTree Test</h1></body></html>")
        result = asyncio.run(BrowserAxTreeTool().run(url=f"file://{html}"))
        assert result.success is True
        assert "AxTree Test" in result.content
