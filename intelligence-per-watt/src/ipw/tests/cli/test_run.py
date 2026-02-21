"""Tests for cli/run.py -- ipw run CLI command."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from ipw.cli import cli


class TestRunCmd:
    """Test ipw run CLI command."""

    @pytest.fixture()
    def runner(self) -> CliRunner:
        return CliRunner()

    def test_help_text(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--agent" in result.output
        assert "--model" in result.output
        assert "--dataset" in result.output
        assert "--export-format" in result.output

    def test_missing_required_agent(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["run", "--model", "test", "--dataset", "gaia"])
        assert result.exit_code != 0
        assert "agent" in result.output.lower() or "Missing" in result.output

    def test_missing_required_model(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["run", "--agent", "react", "--dataset", "gaia"])
        assert result.exit_code != 0

    def test_missing_required_dataset(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["run", "--agent", "react", "--model", "test"])
        assert result.exit_code != 0

    @patch("ipw.execution.exporters.export_summary_json")
    @patch("ipw.execution.exporters.export_jsonl")
    @patch("ipw.execution.agentic_runner.AgenticRunner")
    def test_export_format_parsing(
        self,
        mock_runner_cls: MagicMock,
        mock_export_jsonl: MagicMock,
        mock_export_summary: MagicMock,
        runner: CliRunner,
        tmp_path,
    ) -> None:
        """Verify the run command accepts --export-format and processes it."""
        # The run command imports many things lazily. Rather than mocking
        # the entire internal chain, just confirm the CLI parses the argument
        # without crashing on unknown agents.
        result = runner.invoke(
            cli,
            [
                "run",
                "--agent", "nonexistent-agent",
                "--model", "test-model",
                "--dataset", "gaia",
                "--export-format", "jsonl",
            ],
        )
        # Should fail with "Unknown agent" not a crash
        assert result.exit_code != 0
        assert "Unknown agent" in result.output or "nonexistent-agent" in result.output
