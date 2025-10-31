"""Tests for analyze CLI command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

from click.testing import CliRunner

from trafficbench.analysis.base import AnalysisContext, AnalysisResult
from trafficbench.cli.analyze import analyze


class TestAnalyzeCommand:
    """Test the analyze CLI command."""

    def test_requires_directory_argument(self) -> None:
        runner = CliRunner()
        result = runner.invoke(analyze, [])

        assert result.exit_code != 0
        assert "Missing argument" in result.output or "Error" in result.output

    @patch("trafficbench.cli.analyze.AnalysisRegistry")
    def test_runs_default_regression_analysis(
        self,
        mock_registry: Mock,
        tmp_path: Path,
    ) -> None:
        # Create a dummy directory
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        mock_analysis = Mock()
        mock_analysis.run.return_value = AnalysisResult(
            analysis="regression",
            summary={"total_samples": 10},
            data={},
            warnings=(),
            artifacts={},
        )
        mock_registry.create.return_value = mock_analysis

        runner = CliRunner()
        result = runner.invoke(analyze, [str(data_dir)])

        assert result.exit_code == 0
        mock_registry.create.assert_called_once_with("regression")
        mock_analysis.run.assert_called_once()

    @patch("trafficbench.cli.analyze.AnalysisRegistry")
    def test_accepts_custom_analysis(
        self,
        mock_registry: Mock,
        tmp_path: Path,
    ) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        mock_analysis = Mock()
        mock_analysis.run.return_value = AnalysisResult(
            analysis="custom",
            summary={},
            data={},
            warnings=(),
            artifacts={},
        )
        mock_registry.create.return_value = mock_analysis

        runner = CliRunner()
        result = runner.invoke(analyze, [str(data_dir), "--analysis", "custom"])

        assert result.exit_code == 0
        mock_registry.create.assert_called_once_with("custom")

    @patch("trafficbench.cli.analyze.AnalysisRegistry")
    def test_passes_options_to_analysis(
        self,
        mock_registry: Mock,
        tmp_path: Path,
    ) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        mock_analysis = Mock()
        mock_analysis.run.return_value = AnalysisResult(
            analysis="regression",
            summary={},
            data={},
            warnings=(),
            artifacts={},
        )
        mock_registry.create.return_value = mock_analysis

        runner = CliRunner()
        result = runner.invoke(
            analyze,
            [str(data_dir), "--option", "model=llama3.2:1b"],
        )

        assert result.exit_code == 0
        call_args = mock_analysis.run.call_args
        context = call_args[0][0]
        assert isinstance(context, AnalysisContext)
        assert context.options["model"] == "llama3.2:1b"

    @patch("trafficbench.cli.analyze.AnalysisRegistry")
    def test_handles_multiple_options(
        self,
        mock_registry: Mock,
        tmp_path: Path,
    ) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        mock_analysis = Mock()
        mock_analysis.run.return_value = AnalysisResult(
            analysis="regression",
            summary={},
            data={},
            warnings=(),
            artifacts={},
        )
        mock_registry.create.return_value = mock_analysis

        runner = CliRunner()
        result = runner.invoke(
            analyze,
            [
                str(data_dir),
                "--option",
                "model=llama3.2:1b",
                "--option",
                "skip_zeroes=true",
            ],
        )

        assert result.exit_code == 0
        call_args = mock_analysis.run.call_args
        context = call_args[0][0]
        assert context.options["model"] == "llama3.2:1b"
        assert context.options["skip_zeroes"] == "true"

    @patch("trafficbench.cli.analyze.AnalysisRegistry")
    def test_handles_comma_separated_options(
        self,
        mock_registry: Mock,
        tmp_path: Path,
    ) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        mock_analysis = Mock()
        mock_analysis.run.return_value = AnalysisResult(
            analysis="regression",
            summary={},
            data={},
            warnings=(),
            artifacts={},
        )
        mock_registry.create.return_value = mock_analysis

        runner = CliRunner()
        result = runner.invoke(
            analyze,
            [str(data_dir), "--option", "model=llama,skip_zeroes=true"],
        )

        assert result.exit_code == 0
        call_args = mock_analysis.run.call_args
        context = call_args[0][0]
        assert context.options["model"] == "llama"
        assert context.options["skip_zeroes"] == "true"

    @patch("trafficbench.cli.analyze.AnalysisRegistry")
    def test_displays_summary(
        self,
        mock_registry: Mock,
        tmp_path: Path,
    ) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        mock_analysis = Mock()
        mock_analysis.run.return_value = AnalysisResult(
            analysis="regression",
            summary={"total_samples": 42, "key": "value"},
            data={},
            warnings=(),
            artifacts={},
        )
        mock_registry.create.return_value = mock_analysis

        runner = CliRunner()
        result = runner.invoke(analyze, [str(data_dir)])

        assert result.exit_code == 0
        assert "Summary:" in result.output
        assert "total_samples: 42" in result.output
        assert "key: value" in result.output

    @patch("trafficbench.cli.analyze.AnalysisRegistry")
    def test_displays_warnings(
        self,
        mock_registry: Mock,
        tmp_path: Path,
    ) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        mock_analysis = Mock()
        mock_analysis.run.return_value = AnalysisResult(
            analysis="regression",
            summary={},
            data={},
            warnings=("Warning 1", "Warning 2"),
            artifacts={},
        )
        mock_registry.create.return_value = mock_analysis

        runner = CliRunner()
        result = runner.invoke(analyze, [str(data_dir)])

        assert result.exit_code == 0
        assert "Warnings:" in result.output
        assert "Warning 1" in result.output
        assert "Warning 2" in result.output

    @patch("trafficbench.cli.analyze.AnalysisRegistry")
    def test_displays_artifacts(
        self,
        mock_registry: Mock,
        tmp_path: Path,
    ) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        artifact_path = tmp_path / "report.json"

        mock_analysis = Mock()
        mock_analysis.run.return_value = AnalysisResult(
            analysis="regression",
            summary={},
            data={},
            warnings=(),
            artifacts={"report": artifact_path},
        )
        mock_registry.create.return_value = mock_analysis

        runner = CliRunner()
        result = runner.invoke(analyze, [str(data_dir)])

        assert result.exit_code == 0
        assert "Artifacts:" in result.output
        assert "report:" in result.output

    @patch("trafficbench.cli.analyze.AnalysisRegistry")
    def test_verbose_shows_data(
        self,
        mock_registry: Mock,
        tmp_path: Path,
    ) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        mock_analysis = Mock()
        mock_analysis.run.return_value = AnalysisResult(
            analysis="regression",
            summary={},
            data={"regressions": {"key": "value"}},
            warnings=(),
            artifacts={},
        )
        mock_registry.create.return_value = mock_analysis

        runner = CliRunner()
        result = runner.invoke(analyze, [str(data_dir), "--verbose"])

        assert result.exit_code == 0
        assert "Data:" in result.output
        assert "regressions" in result.output

    @patch("trafficbench.cli.analyze.AnalysisRegistry")
    def test_non_verbose_hides_data(
        self,
        mock_registry: Mock,
        tmp_path: Path,
    ) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        mock_analysis = Mock()
        mock_analysis.run.return_value = AnalysisResult(
            analysis="regression",
            summary={},
            data={"regressions": {"key": "value"}},
            warnings=(),
            artifacts={},
        )
        mock_registry.create.return_value = mock_analysis

        runner = CliRunner()
        result = runner.invoke(analyze, [str(data_dir)])

        assert result.exit_code == 0
        assert "Data:" not in result.output

    @patch("trafficbench.cli.analyze.AnalysisRegistry")
    def test_handles_unknown_analysis(
        self,
        mock_registry: Mock,
        tmp_path: Path,
    ) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        mock_registry.create.side_effect = KeyError("unknown")
        mock_registry.items.return_value = [("regression", None), ("other", None)]

        runner = CliRunner()
        result = runner.invoke(analyze, [str(data_dir), "--analysis", "unknown"])

        assert result.exit_code != 0
        assert "Unknown analysis" in result.output
        assert "regression" in result.output  # Should list available

    @patch("trafficbench.cli.analyze.AnalysisRegistry")
    def test_handles_runtime_errors(
        self,
        mock_registry: Mock,
        tmp_path: Path,
    ) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        mock_analysis = Mock()
        mock_analysis.run.side_effect = RuntimeError("Test error")
        mock_registry.create.return_value = mock_analysis

        runner = CliRunner()
        result = runner.invoke(analyze, [str(data_dir)])

        assert result.exit_code != 0
        assert "Test error" in result.output

