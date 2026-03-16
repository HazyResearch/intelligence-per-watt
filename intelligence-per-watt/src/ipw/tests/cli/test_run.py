"""Tests for cli/run.py -- ipw run CLI command."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from ipw.cli import cli
from ipw.cli.run import _is_cloud_model


class TestIsCloudModel:
    """Test cloud model detection logic."""

    def test_anthropic_prefix_is_cloud(self) -> None:
        assert _is_cloud_model("anthropic/claude-opus-4-6", "http://localhost:8000") is True

    def test_gemini_prefix_is_cloud(self) -> None:
        assert _is_cloud_model("gemini/gemini-3.1-pro", "http://localhost:8000") is True

    def test_openai_prefix_with_localhost_is_local(self) -> None:
        # openai/ prefix + localhost = local vLLM server, not cloud OpenAI
        assert _is_cloud_model("openai/gpt-oss-120b", "http://localhost:8000") is False

    def test_openai_prefix_with_custom_url_is_cloud(self) -> None:
        # openai/ prefix + non-local URL = cloud OpenAI
        assert _is_cloud_model("openai/gpt-5.4", "http://my-server:8000") is True

    def test_bare_model_with_localhost_is_local(self) -> None:
        assert _is_cloud_model("llama3.2:1b", "http://localhost:8000") is False

    def test_azure_prefix_is_cloud(self) -> None:
        assert _is_cloud_model("azure/gpt-4", "http://localhost:8000") is True

    def test_vertex_ai_prefix_is_cloud(self) -> None:
        assert _is_cloud_model("vertex_ai/gemini-pro", "http://localhost:8000") is True

    def test_openai_prefix_with_127_is_local(self) -> None:
        # 127.0.0.1 = localhost = local server
        assert _is_cloud_model("openai/gpt-5-mini", "http://127.0.0.1:8000") is False


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

    def test_help_includes_concurrency(self, runner: CliRunner) -> None:
        """Verify --concurrency appears in help text."""
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--concurrency" in result.output

    def test_help_includes_dataset_kwargs(self, runner: CliRunner) -> None:
        """Verify --dataset-kwargs appears in help text."""
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--dataset-kwargs" in result.output

    def test_help_includes_agent_kwargs(self, runner: CliRunner) -> None:
        """Verify --agent-kwargs appears in help text."""
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--agent-kwargs" in result.output
