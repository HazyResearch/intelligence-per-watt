"""Tests for tools/image_tool.py (vision understanding)."""

from __future__ import annotations

import asyncio
import importlib.util
import os

import pytest

pillow_available = importlib.util.find_spec("PIL") is not None
pytestmark = pytest.mark.skipif(not pillow_available, reason="Pillow not installed (ipw[multimodal])")


from ipw.tools.image_tool import ImageTool  # noqa: E402


@pytest.fixture
def sample_png(tmp_path):
    from PIL import Image
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))  # solid red
    path = tmp_path / "red.png"
    img.save(path)
    return path


class TestImageToolMetadata:
    def test_spec(self) -> None:
        spec = ImageTool.spec
        assert spec.name == "image_tool"
        assert spec.requires_network is True
        assert "path" in spec.parameters

    def test_empty_path_returns_error(self) -> None:
        result = asyncio.run(ImageTool().run(path=""))
        assert result.success is False

    def test_missing_file_returns_error(self) -> None:
        result = asyncio.run(ImageTool().run(path="/nonexistent/img.png"))
        assert result.success is False

    def test_unsupported_format_returns_error(self, tmp_path) -> None:
        bad = tmp_path / "notimage.xyz"
        bad.write_text("not an image")
        result = asyncio.run(ImageTool().run(path=str(bad)))
        assert result.success is False

    def test_default_cwd_resolves_relative_path(self, tmp_path) -> None:
        from PIL import Image
        img = Image.new("RGB", (8, 8), color=(0, 255, 0))
        img.save(tmp_path / "rel.png")
        tool = ImageTool()
        tool.set_workspace(str(tmp_path))
        # With no key this will fail at the API call, but the file must be FOUND
        # (i.e. the error must NOT be "file not found")
        result = asyncio.run(tool.run(path="rel.png", question="what color?"))
        # Either succeeds (if key present) or fails at API — but not file-not-found
        assert "not found" not in (result.error or "").lower()


@pytest.mark.integration
class TestImageToolVision:
    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="requires OPENAI_API_KEY")
    def test_describe_red_image(self, sample_png) -> None:
        result = asyncio.run(ImageTool().run(
            path=str(sample_png), question="What is the dominant color? One word."
        ))
        assert result.success is True
        assert "red" in result.content.lower()
