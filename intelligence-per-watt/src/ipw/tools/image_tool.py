"""image_tool — analyze a local image file (or URL) using a vision model.

Sends the image to the OpenAI vision API (gpt-4o-mini by default) as a
base64-encoded data-URL and returns the model's answer to an optional question.

Supported formats: PNG, JPEG, GIF, WebP.
Large images are resized to at most 2000 px on the long side before encoding
to keep the base64 payload reasonable.

Deps are lazy-imported:
  - Pillow (PIL) — required for image resizing / re-encoding
  - openai — required for the vision API call
  - httpx — required only when `path` is an http/https URL

Install with: uv pip install pillow openai  (or ipw[multimodal])
"""

from __future__ import annotations

import base64
import io
import os
import tempfile
from typing import Any, Optional

from ipw.tools.base import BaseTool, ToolResult, ToolSpec
from ipw.tools.registry import ToolRegistry

_SUPPORTED = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
_MAX_LONG_SIDE = 2000


@ToolRegistry.register("image_tool")
class ImageTool(BaseTool):
    """Analyze a local image file and answer a question about it using a vision model."""

    spec = ToolSpec(
        name="image_tool",
        description=(
            "Analyze a local image file and answer a question about it using a vision model. "
            "Accepts a local file path or an http/https URL. "
            "Relative paths are resolved against the workspace directory."
        ),
        parameters={
            "path": {
                "type": "string",
                "description": (
                    "Local file path or http/https URL of the image to analyze. "
                    "Supported formats: PNG, JPEG, GIF, WebP."
                ),
            },
            "question": {
                "type": "string",
                "description": (
                    "Question to ask about the image. "
                    "Defaults to 'Describe this image in detail.'"
                ),
            },
            "model": {
                "type": "string",
                "description": (
                    "Vision model to use. Defaults to 'gpt-4o-mini'. "
                    "Accepts bare model IDs or the 'openai/' prefix."
                ),
            },
        },
        requires_network=True,
    )

    async def run(
        self,
        path: str = "",
        question: Optional[str] = None,
        model: str = "gpt-4o-mini",
        **kwargs: Any,
    ) -> ToolResult:
        if not path:
            return ToolResult(content="", success=False, error="No path provided.")

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
                suffix = os.path.splitext(path.split("?")[0])[1] or ".png"
                tmp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
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
                    error=f"Failed to download image from {path!r}: {exc}",
                )
        else:
            # Local file — respect _default_cwd for relative paths.
            if self._default_cwd and not os.path.isabs(path):
                resolved_path = os.path.join(self._default_cwd, path)
            else:
                resolved_path = path

            if not os.path.exists(resolved_path):
                return ToolResult(
                    content="",
                    success=False,
                    error=f"File not found: {path!r}",
                )

        try:
            # ------------------------------------------------------------------
            # Validate extension.
            # ------------------------------------------------------------------
            ext = os.path.splitext(resolved_path)[1].lower()
            if ext not in _SUPPORTED:
                return ToolResult(
                    content="",
                    success=False,
                    error=(
                        f"Unsupported image format {ext!r}. "
                        f"Supported: {', '.join(sorted(_SUPPORTED))}"
                    ),
                )

            # ------------------------------------------------------------------
            # Load + optionally resize with Pillow, then encode to base64 PNG.
            # ------------------------------------------------------------------
            try:
                from PIL import Image
            except ImportError:
                return ToolResult(
                    content="",
                    success=False,
                    error="Pillow not installed. Install with: uv pip install pillow",
                )

            with Image.open(resolved_path) as img:
                # Convert palette/transparency modes to RGB for uniform handling.
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGB")

                # Resize if needed to keep the long side <= _MAX_LONG_SIDE.
                w, h = img.size
                long_side = max(w, h)
                if long_side > _MAX_LONG_SIDE:
                    scale = _MAX_LONG_SIDE / long_side
                    new_w = max(1, int(w * scale))
                    new_h = max(1, int(h * scale))
                    img = img.resize((new_w, new_h), Image.LANCZOS)

                buf = io.BytesIO()
                img.save(buf, format="PNG")
                png_bytes = buf.getvalue()

            b64 = base64.b64encode(png_bytes).decode("ascii")

            # ------------------------------------------------------------------
            # Call the vision API.
            # ------------------------------------------------------------------
            bare_model = model.removeprefix("openai/")
            prompt = question or "Describe this image in detail."

            try:
                from openai import AsyncOpenAI
            except ImportError:
                return ToolResult(
                    content="",
                    success=False,
                    error="openai not installed. Install with: uv pip install openai",
                )

            openai_client = AsyncOpenAI()
            response = await openai_client.chat.completions.create(
                model=bare_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            },
                        ],
                    }
                ],
            )
            answer = response.choices[0].message.content or ""
            return ToolResult(
                content=answer,
                success=True,
                metadata={"path": path, "model": bare_model},
            )

        except Exception as exc:
            return ToolResult(content="", success=False, error=str(exc))

        finally:
            # Clean up temp file if we downloaded from a URL.
            if tmp_file is not None:
                try:
                    tmp_file.close()
                    os.unlink(tmp_file.name)
                except OSError:
                    pass


__all__ = ["ImageTool"]
