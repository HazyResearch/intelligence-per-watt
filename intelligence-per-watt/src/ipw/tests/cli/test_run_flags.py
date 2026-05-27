"""Tests for --max-retries and --require-dedicated-hardware CLI flags in ipw run."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from ipw.cli import cli

# ---------------------------------------------------------------------------
# Helpers / doubles
# ---------------------------------------------------------------------------


class _CapturingRunner:
    """Drop-in double for AgenticRunner that records constructor kwargs."""

    _last_init_kwargs: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        type(self)._last_init_kwargs = dict(kwargs)

    async def run(self, max_queries=None) -> list:  # noqa: D401
        return []


def _make_patchers():
    """Return a context-managed collection of patches that lets run_cmd reach
    the AgenticRunner constructor without real hardware, datasets, or agents.
    """
    # Fake dataset instance
    fake_dataset = MagicMock()
    fake_dataset.verify_requirements.return_value = []
    fake_dataset.size.return_value = 0

    # Fake agent instance
    fake_agent = MagicMock()

    # Fake telemetry / collector so the non-cloud path doesn't try to start a
    # Rust binary.
    fake_telemetry = MagicMock()
    fake_collector = MagicMock()
    fake_collector.__enter__ = MagicMock(return_value=fake_telemetry)
    fake_collector.__exit__ = MagicMock(return_value=False)

    return [
        # Registry lookups
        patch("ipw.core.registry.AgentRegistry.get", return_value=MagicMock(return_value=fake_agent)),
        patch("ipw.core.registry.DatasetRegistry.get", return_value=MagicMock(return_value=fake_dataset)),
        # Dataset instantiation — already handled by DatasetRegistry.get mock above,
        # but also patch verify_requirements at the dataset class level to be safe.
        # Agent / dataset registration side-effects (lazy imports)
        patch("ipw.clients.ensure_registered"),
        patch("ipw.datasets.ensure_registered"),
        # The runner itself — replaced by _CapturingRunner
        patch("ipw.execution.agentic_runner.AgenticRunner", new=_CapturingRunner),
        # Telemetry / energy collector so local path doesn't launch Rust binary
        patch("ipw.telemetry.EnergyMonitorCollector", return_value=fake_collector),
        patch("ipw.execution.telemetry_session.TelemetrySession", return_value=fake_telemetry),
        # Exporters — return harmless paths
        patch("ipw.execution.exporters.export_jsonl", return_value="/tmp/traces.jsonl"),
        patch("ipw.execution.exporters.export_hf_dataset", return_value="/tmp/hf"),
        patch("ipw.execution.exporters.export_summary_json", return_value="/tmp/summary.json"),
        patch("ipw.execution.exporters.export_artifacts_manifest", return_value=None),
        # EventRecorder — plain stub
        patch("ipw.telemetry.events.EventRecorder", return_value=MagicMock()),
        # _create_model_for_agent — avoid importing agno/openhands
        patch("ipw.cli.run._create_model_for_agent", return_value="test-model"),
        # _is_cloud_model → True so we skip the EnergyMonitorCollector path
        patch("ipw.cli.run._is_cloud_model", return_value=True),
    ]


def _invoke(args: list[str]) -> tuple[Any, dict[str, Any]]:
    """Invoke ``ipw run`` with *args* under all necessary patches.

    Returns ``(result, captured_init_kwargs)`` where *captured_init_kwargs* is
    the keyword-argument dict that ``_CapturingRunner.__init__`` saw.
    """
    _CapturingRunner._last_init_kwargs = {}

    patchers = _make_patchers()
    for p in patchers:
        p.start()
    try:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["run", "--agent", "react", "--model", "gpt-4o", "--dataset", "gaia"] + args,
            catch_exceptions=False,
        )
    finally:
        for p in patchers:
            try:
                p.stop()
            except RuntimeError:
                pass

    return result, _CapturingRunner._last_init_kwargs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMaxRetriesFlag:
    """--max-retries (retries) is converted to attempt count (retries + 1) and
    threaded to the runner as the ``max_attempts`` kwarg."""

    def test_default_retries_gives_four_attempts(self) -> None:
        """No --max-retries flag → default 3 retries → 4 attempts."""
        result, kwargs = _invoke([])
        assert result.exit_code == 0, result.output
        assert kwargs.get("max_attempts") == 4

    def test_zero_retries_gives_one_attempt(self) -> None:
        """--max-retries 0 → 0 retries → 1 attempt (no retry)."""
        result, kwargs = _invoke(["--max-retries", "0"])
        assert result.exit_code == 0, result.output
        assert kwargs.get("max_attempts") == 1

    def test_four_retries_gives_five_attempts(self) -> None:
        """--max-retries 4 → 4 retries → 5 attempts."""
        result, kwargs = _invoke(["--max-retries", "4"])
        assert result.exit_code == 0, result.output
        assert kwargs.get("max_attempts") == 5


class TestRequireDedicatedHardwareFlag:
    """--require-dedicated-hardware is threaded to runner as a bool."""

    def test_flag_absent_gives_false(self) -> None:
        """Without --require-dedicated-hardware, runner gets False."""
        result, kwargs = _invoke([])
        assert result.exit_code == 0, result.output
        assert kwargs.get("require_dedicated_hardware") is False

    def test_flag_present_gives_true(self) -> None:
        """With --require-dedicated-hardware, runner gets True."""
        result, kwargs = _invoke(["--require-dedicated-hardware"])
        assert result.exit_code == 0, result.output
        assert kwargs.get("require_dedicated_hardware") is True


class TestHelpText:
    """Both flags appear in --help output."""

    def test_max_retries_in_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--max-retries" in result.output

    def test_require_dedicated_hardware_in_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--require-dedicated-hardware" in result.output
