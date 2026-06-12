"""Tests for tools/pdf_tool.py."""

from __future__ import annotations

import asyncio
import importlib.util

import pytest

# Skip if pdf deps not installed
pdfplumber_available = importlib.util.find_spec("pdfplumber") is not None
pypdf_available = importlib.util.find_spec("pypdf") is not None
pytestmark = pytest.mark.skipif(
    not (pdfplumber_available and pypdf_available),
    reason="pdfplumber + pypdf not installed (ipw[pdf])",
)


from ipw.tools.pdf_tool import PdfTool  # noqa: E402


@pytest.fixture
def sample_pdf(tmp_path):
    """Create a tiny one-page PDF for testing."""
    try:
        from reportlab.pdfgen import canvas

        pdf_path = tmp_path / "sample.pdf"
        c = canvas.Canvas(str(pdf_path))
        c.drawString(100, 750, "Hello PDF Test")
        c.drawString(100, 730, "Second line of content")
        c.save()
        return pdf_path
    except ImportError:
        pytest.skip("reportlab not installed; cannot generate sample PDF")


class TestPdfToolMetadata:
    def test_spec(self) -> None:
        spec = PdfTool.spec
        assert spec.name == "pdf_tool"
        assert "path" in spec.parameters

    def test_empty_path_returns_error(self) -> None:
        result = asyncio.run(PdfTool().run(path=""))
        assert result.success is False

    def test_missing_file_returns_error(self) -> None:
        result = asyncio.run(PdfTool().run(path="/nonexistent/path/to/file.pdf"))
        assert result.success is False


class TestPdfToolExtraction:
    def test_extracts_text_from_local_pdf(self, sample_pdf) -> None:
        result = asyncio.run(PdfTool().run(path=str(sample_pdf)))
        assert result.success is True
        assert "Hello PDF Test" in result.content

    def test_extract_metadata_returns_dict(self, sample_pdf) -> None:
        result = asyncio.run(PdfTool().run(path=str(sample_pdf), extract_metadata=True))
        assert result.success is True
        # metadata may be empty for our generated PDF but should be a dict in ToolResult.metadata
        assert isinstance(result.metadata, dict)
