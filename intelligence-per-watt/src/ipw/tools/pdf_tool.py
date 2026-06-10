"""pdf_tool — extract text and metadata from a PDF file or URL.

Ported from /home/ubuntu/lambda-stanford/jonsf/OpenJarvis/src/openjarvis/tools/pdf_tool.py.
Uses pdfplumber for text extraction (page-by-page) and pypdf for metadata.
URL support: download via httpx to a temporary file, then process locally.

Deps are lazy-imported so they are only required when the tool is actually invoked.
Install with: uv pip install pdfplumber pypdf  (or ipw[pdf] once the extras group is defined)
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from ipw.tools.base import BaseTool, ToolResult, ToolSpec
from ipw.tools.registry import ToolRegistry

_MAX_BYTES = 128 * 1024  # 128 KB


def _parse_page_range(page_range: str, total_pages: int) -> List[int]:
    """Parse a page-range string (e.g. "1-5") into 0-indexed page numbers.

    Supports:
    - Range:  "1-5"   → [0, 1, 2, 3, 4]
    - List:   "1,3,5" → [0, 2, 4]
    - Mixed:  "1-3,5" → [0, 1, 2, 4]

    Page numbers in the input are 1-based inclusive; output is 0-based.
    """
    indices: list[int] = []
    for part in page_range.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = max(1, int(start_str.strip()))
            end = min(total_pages, int(end_str.strip()))
            indices.extend(range(start - 1, end))
        else:
            page_num = int(part)
            if 1 <= page_num <= total_pages:
                indices.append(page_num - 1)
    return sorted(set(indices))


@ToolRegistry.register("pdf_tool")
class PdfTool(BaseTool):
    """Extract text and optional metadata from a PDF file or URL."""

    spec = ToolSpec(
        name="pdf_tool",
        description=(
            "Extract text and metadata from a PDF file (local path or http/https URL). "
            "Returns the extracted text, optionally including document metadata."
        ),
        parameters={
            "path": {
                "type": "string",
                "description": (
                    "Local file path or http/https URL pointing to a PDF. "
                    "Relative paths are resolved against the workspace directory."
                ),
            },
            "page_range": {
                "type": "string",
                "description": (
                    "Optional page range to extract, e.g. '1-5' or '1,3,5'. "
                    "Omit to extract all pages."
                ),
            },
            "extract_metadata": {
                "type": "boolean",
                "description": (
                    "If true, include document metadata (author, title, etc.) "
                    "in the result. Default: false."
                ),
            },
        },
        requires_network=False,  # set dynamically only when a URL is passed
    )

    async def run(
        self,
        path: str = "",
        page_range: Optional[str] = None,
        extract_metadata: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        if not path:
            return ToolResult(
                content="",
                success=False,
                error="No path provided.",
            )

        # Lazy-import both deps up-front so a missing dep is caught early.
        try:
            import pdfplumber
        except ImportError:
            return ToolResult(
                content="",
                success=False,
                error="pdfplumber not installed. Install with: uv pip install pdfplumber",
            )
        try:
            import pypdf
        except ImportError:
            return ToolResult(
                content="",
                success=False,
                error="pypdf not installed. Install with: uv pip install pypdf",
            )

        # ------------------------------------------------------------------
        # Resolve path — download from URL or resolve local file.
        # ------------------------------------------------------------------
        is_url = path.startswith("http://") or path.startswith("https://")
        tmp_file: Optional[tempfile.NamedTemporaryFile] = None  # type: ignore[type-arg]
        resolved_path: str

        if is_url:
            try:
                import httpx
            except ImportError:
                return ToolResult(
                    content="",
                    success=False,
                    error="httpx not installed (required for URL downloads). Install with: uv pip install httpx",
                )
            try:
                tmp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.get(path, follow_redirects=True)
                    if response.status_code >= 400:
                        tmp_file.close()
                        os.unlink(tmp_file.name)
                        return ToolResult(
                            content="",
                            success=False,
                            error=f"HTTP {response.status_code} fetching {path!r}",
                        )
                    tmp_file.write(response.content)
                    tmp_file.flush()
                resolved_path = tmp_file.name
            except Exception as exc:
                if tmp_file is not None:
                    try:
                        tmp_file.close()
                        os.unlink(tmp_file.name)
                    except OSError:
                        pass
                return ToolResult(
                    content="",
                    success=False,
                    error=f"Failed to download PDF from {path!r}: {exc}",
                )
        else:
            # Local file — respect _default_cwd for relative paths.
            local_path = Path(path)
            if not local_path.is_absolute():
                base = self._default_cwd or os.getcwd()
                local_path = Path(base) / local_path
            if not local_path.exists():
                return ToolResult(
                    content="",
                    success=False,
                    error=f"File not found: {path!r}",
                )
            resolved_path = str(local_path)

        # ------------------------------------------------------------------
        # Extract text via pdfplumber.
        # ------------------------------------------------------------------
        try:
            with pdfplumber.open(resolved_path) as pdf:
                total_pages = len(pdf.pages)

                if page_range:
                    page_indices = _parse_page_range(page_range, total_pages)
                else:
                    page_indices = list(range(total_pages))

                text_parts: list[str] = []
                for idx in page_indices:
                    if 0 <= idx < total_pages:
                        page_text = pdf.pages[idx].extract_text() or ""
                        text_parts.append(page_text)

                text = "\n\n".join(text_parts)
                truncated = False
                if len(text.encode()) > _MAX_BYTES:
                    # Trim to _MAX_BYTES characters (conservative approximation).
                    text = text[:_MAX_BYTES] + "\n\n[Content truncated]"
                    truncated = True

            # ------------------------------------------------------------------
            # Optionally extract metadata via pypdf.
            # ------------------------------------------------------------------
            metadata: Dict[str, Any] = {
                "total_pages": total_pages,
                "pages_extracted": len(page_indices),
                "truncated": truncated,
            }

            if extract_metadata:
                reader = pypdf.PdfReader(resolved_path)
                raw_meta = reader.metadata or {}
                doc_meta: Dict[str, str] = {}
                for k, v in raw_meta.items():
                    key = str(k).lstrip("/")  # pypdf keys look like "/Author"
                    doc_meta[key] = str(v) if v is not None else ""
                metadata["document_metadata"] = doc_meta

            return ToolResult(
                content=text or "No text content found in PDF.",
                success=True,
                metadata=metadata,
            )

        except Exception as exc:
            return ToolResult(
                content="",
                success=False,
                error=f"PDF extraction error: {exc}",
            )
        finally:
            # Clean up temp file if we downloaded from a URL.
            if tmp_file is not None:
                try:
                    tmp_file.close()
                    os.unlink(tmp_file.name)
                except OSError:
                    pass


__all__ = ["PdfTool"]
