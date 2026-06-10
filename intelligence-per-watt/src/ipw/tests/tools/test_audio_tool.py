"""Tests for tools/audio_tool.py (Whisper transcription)."""

from __future__ import annotations

import asyncio

from ipw.tools.audio_tool import AudioTool


class TestAudioToolMetadata:
    def test_spec(self) -> None:
        spec = AudioTool.spec
        assert spec.name == "audio_tool"
        assert spec.requires_network is True
        assert "path" in spec.parameters

    def test_empty_path_returns_error(self) -> None:
        result = asyncio.run(AudioTool().run(path=""))
        assert result.success is False

    def test_missing_file_returns_error(self) -> None:
        result = asyncio.run(AudioTool().run(path="/nonexistent/audio.mp3"))
        assert result.success is False

    def test_unsupported_format_returns_error(self, tmp_path) -> None:
        bad = tmp_path / "file.xyz"
        bad.write_bytes(b"not audio")
        result = asyncio.run(AudioTool().run(path=str(bad)))
        assert result.success is False
        assert "format" in (result.error or "").lower() or "support" in (result.error or "").lower()

    def test_oversized_file_returns_error(self, tmp_path) -> None:
        big = tmp_path / "big.mp3"
        # Write 26MB (over the 25MB limit)
        big.write_bytes(b"\x00" * (26 * 1024 * 1024))
        result = asyncio.run(AudioTool().run(path=str(big)))
        assert result.success is False
        assert "size" in (result.error or "").lower() or "25" in (result.error or "") or "large" in (result.error or "").lower()

    def test_default_cwd_resolves_relative_path(self, tmp_path) -> None:
        # A small valid-extension file resolved against workspace; should get
        # PAST the file-not-found check (will fail later at the API, that's fine)
        audio = tmp_path / "rel.mp3"
        audio.write_bytes(b"\x00" * 1024)  # tiny fake mp3
        tool = AudioTool()
        tool.set_workspace(str(tmp_path))
        result = asyncio.run(tool.run(path="rel.mp3"))
        # Must not be "file not found" — the workspace resolution worked
        assert "not found" not in (result.error or "").lower()
