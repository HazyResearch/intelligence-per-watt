"""Tests for tools/git_tool.py."""

from __future__ import annotations

import asyncio
import subprocess

import pytest

from ipw.tools.git_tool import GitTool


@pytest.fixture
def temp_repo(tmp_path):
    """Create a minimal git repo with one committed file."""
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "-c", "user.email=t@e.com", "-c", "user.name=T",
                    "commit", "--allow-empty", "-m", "init"],
                   cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "file.txt").write_text("hello\n")
    return tmp_path


class TestGitTool:
    def test_spec(self) -> None:
        spec = GitTool.spec
        assert spec.name == "git_tool"
        assert "subcommand" in spec.parameters

    def test_status(self, temp_repo) -> None:
        result = asyncio.run(GitTool().run(subcommand="status", cwd=str(temp_repo)))
        assert result.success is True
        assert "file.txt" in result.content

    def test_diff_no_changes(self, temp_repo) -> None:
        result = asyncio.run(GitTool().run(subcommand="diff", cwd=str(temp_repo)))
        # Untracked files don't show in diff — content is empty but call succeeds
        assert result.success is True

    def test_log(self, temp_repo) -> None:
        result = asyncio.run(GitTool().run(subcommand="log", cwd=str(temp_repo), args=["-1"]))
        assert result.success is True
        assert "init" in result.content

    def test_unknown_subcommand_rejected(self, temp_repo) -> None:
        result = asyncio.run(GitTool().run(subcommand="evil-rm-rf", cwd=str(temp_repo)))
        assert result.success is False
        assert "allow" in (result.error or "").lower() or "not" in (result.error or "").lower()

    def test_non_repo_dir_returns_error(self, tmp_path) -> None:
        result = asyncio.run(GitTool().run(subcommand="status", cwd=str(tmp_path)))
        assert result.success is False
