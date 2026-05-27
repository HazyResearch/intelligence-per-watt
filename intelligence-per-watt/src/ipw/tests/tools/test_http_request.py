"""Tests for tools/http_request.py."""

from __future__ import annotations

import asyncio
import os

import pytest

from ipw.tools.http_request import HttpRequestTool


class TestHttpRequestTool:
    def test_spec(self) -> None:
        spec = HttpRequestTool.spec
        assert spec.name == "http_request"
        assert spec.requires_network is True

    def test_invalid_url_returns_error(self) -> None:
        result = asyncio.run(HttpRequestTool().run(url="not://a.url", method="GET"))
        assert result.success is False

    def test_disallowed_method_rejected(self) -> None:
        result = asyncio.run(HttpRequestTool().run(url="http://example.com", method="EVIL"))
        assert result.success is False
        assert "method" in (result.error or "").lower()

    def test_empty_url_rejected(self) -> None:
        result = asyncio.run(HttpRequestTool().run(url="", method="GET"))
        assert result.success is False

    @pytest.mark.skipif(not os.environ.get("IPW_NETWORK_TESTS"),
                        reason="requires network")
    def test_real_request_succeeds(self) -> None:
        result = asyncio.run(HttpRequestTool().run(
            url="https://httpbin.org/get", method="GET",
        ))
        assert result.success is True
        assert "url" in result.content
