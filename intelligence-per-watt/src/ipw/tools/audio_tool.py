"""audio_tool — transcribe a local audio file using OpenAI Whisper.

Sends the audio file to the OpenAI Whisper API (whisper-1 by default) and
returns the transcribed text.

Supported formats: mp3, wav, m4a, ogg, flac, webm.
Maximum file size: 25 MB (Whisper API limit).

Relative paths are resolved against the workspace directory when one has
been set via set_workspace().

Deps are lazy-imported:
  - openai — required for the Whisper API call

Install with: uv pip install openai  (or ipw[multimodal])
"""

from __future__ import annotations

import os
from typing import Any, Optional

from ipw.tools.base import BaseTool, ToolResult, ToolSpec
from ipw.tools.registry import ToolRegistry

_SUPPORTED = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"}
_MAX_BYTES = 25 * 1024 * 1024  # 25 MB


@ToolRegistry.register("audio_tool")
class AudioTool(BaseTool):
    """Transcribe a local audio file to text using OpenAI Whisper."""

    spec = ToolSpec(
        name="audio_tool",
        description=(
            "Transcribe an audio file to text using Whisper. "
            "Supports mp3, wav, m4a, ogg, flac, webm. "
            "Relative paths are resolved against the workspace directory."
        ),
        parameters={
            "path": {
                "type": "string",
                "description": (
                    "Local file path to the audio file to transcribe. "
                    "Supported formats: mp3, wav, m4a, ogg, flac, webm."
                ),
            },
            "language": {
                "type": "string",
                "description": (
                    "Optional ISO-639-1 language code (e.g. 'en', 'es', 'fr'). "
                    "When provided Whisper skips language detection."
                ),
            },
            "model": {
                "type": "string",
                "description": "Whisper model to use. Defaults to 'whisper-1'.",
            },
        },
        requires_network=True,
    )

    async def run(
        self,
        path: str = "",
        language: Optional[str] = None,
        model: str = "whisper-1",
        **kwargs: Any,
    ) -> ToolResult:
        # 1. Empty path check.
        if not path:
            return ToolResult(content="", success=False, error="No path provided.")

        # 2. Resolve relative path against workspace.
        if self._default_cwd and not os.path.isabs(path):
            resolved = os.path.join(self._default_cwd, path)
        else:
            resolved = path

        # 3. Existence check.
        if not os.path.exists(resolved):
            return ToolResult(
                content="",
                success=False,
                error=f"File not found: {path!r}",
            )

        # 4. Format check.
        ext = os.path.splitext(resolved)[1].lower()
        if ext not in _SUPPORTED:
            return ToolResult(
                content="",
                success=False,
                error=(
                    f"Unsupported audio format {ext!r}. "
                    f"Supported formats: {', '.join(sorted(_SUPPORTED))}"
                ),
            )

        # 5. Size check.
        try:
            file_size = os.path.getsize(resolved)
        except OSError as exc:
            return ToolResult(content="", success=False, error=f"Cannot stat file: {exc}")

        if file_size > _MAX_BYTES:
            return ToolResult(
                content="",
                success=False,
                error=(
                    f"File too large ({file_size} bytes). "
                    "Whisper API accepts at most 25 MB per file."
                ),
            )

        # 6. Transcribe via OpenAI Whisper.
        try:
            from openai import OpenAI
        except ImportError:
            return ToolResult(
                content="",
                success=False,
                error="openai not installed. Install with: uv pip install openai",
            )

        try:
            client = OpenAI()  # reads OPENAI_API_KEY from env
            with open(resolved, "rb") as f:
                api_kwargs: dict[str, Any] = {"model": model, "file": f}
                if language:
                    api_kwargs["language"] = language
                transcript = client.audio.transcriptions.create(**api_kwargs)
            text: str = transcript.text
            return ToolResult(
                content=text,
                success=True,
                metadata={"path": path},
            )
        except Exception as exc:
            return ToolResult(content="", success=False, error=str(exc))


__all__ = ["AudioTool"]
