"""Tests for the Stirrup agent adapter helpers."""

from __future__ import annotations


def test_submitted_files_resolves_transferred_finish_paths(tmp_path) -> None:
    from ipw.agents.stirrup import _submitted_files

    output_dir = tmp_path / "workspace"
    output_dir.mkdir()
    saved = output_dir / "answer.xlsx"
    saved.write_text("placeholder", encoding="utf-8")

    submitted = _submitted_files(
        output_dir,
        finish_paths=["answer.xlsx", "/tmp/local_exec_env_x/answer.xlsx"],
        transferred_paths=[str(saved)],
        local_files=[str(saved)],
    )

    assert submitted == [str(saved)]
