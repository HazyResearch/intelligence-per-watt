from __future__ import annotations

import json

from click.testing import CliRunner

from ipj.registry.cli import format_query_table, results


def test_format_query_table_columns():
    rows = [{
        "model": {"id": "openai/Qwen3.6-27B"},
        "benchmark": {"name": "terminal-bench/terminal-bench-2-1"},
        "hardware": {"gpu": "H100"},
        "metrics": {"accuracy": 0.76},
        "energy_sig": "deadbeef00000000",
        "status": "ok",
    }]
    table = format_query_table(rows)
    assert "openai/Qwen3.6-27B" in table
    assert "0.76" in table
    assert "deadbeef" in table


def test_results_group_has_commands():
    assert set(results.commands) >= {"push", "query"}


def test_pull_traces_command_is_registered():
    assert "pull-traces" in results.commands


def test_query_empty_index(monkeypatch):
    monkeypatch.setattr("ipj.registry.cli.pull_index", lambda **kw: [])
    out = CliRunner().invoke(results, ["query"])
    assert out.exit_code == 0
    assert "no runs" in out.output.lower()


def test_trace_push_writes_only_public_index_metadata(
    tmp_path, monkeypatch, make_rec
):
    """The combined CLI flow must never follow sanitized traces with a raw index row."""

    import huggingface_hub

    import ipj.registry.cli as cli
    import ipj.registry.store as store

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    artifact = run_dir / "public_artifact"
    artifact.mkdir()
    record = make_rec()
    record.ipj_git_sha = "deadbeef"
    record.git_dirty = False
    record.accuracy_sig = "a" * 16
    record.energy_sig = "b" * 16
    record.host = "compute-01.internal"
    record.run_dir = "/home/alice/private/run"
    record.raw_config = {"api_key": "sk-proj-abcdefghijklmnop"}
    record.orchestration = {"endpoint": "10.42.0.9"}
    uploaded = {}

    class FakeApi:
        def __init__(self, *, token=None):
            self.token = token

        def upload_file(self, **kwargs):
            uploaded.update(kwargs)

    def raw_push_must_not_run(*args, **kwargs):
        raise AssertionError("raw push_record was called for a public trace upload")

    monkeypatch.setattr(cli, "capture_run", lambda *args, **kwargs: record)
    monkeypatch.setattr(
        cli,
        "push_traces",
        lambda *args, **kwargs: (
            f"hf://ScalingIntelligence/ipj-results/traces/{record.energy_sig}@revision123"
        ),
    )
    monkeypatch.setattr(cli, "push_record", raw_push_must_not_run)
    monkeypatch.setattr(store, "pull_public_index", lambda **kwargs: [])
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)

    result = CliRunner().invoke(
        results,
        ["push", str(run_dir), "--traces"],
    )

    assert result.exit_code == 0, result.output
    assert uploaded["path_in_repo"] == "public-index.jsonl"
    text = uploaded["path_or_fileobj"].decode()
    row = json.loads(text)
    assert row["trace_uri"].endswith("@revision123")
    assert row["accounting"] == {
        "complete_cost_eligible": False,
        "rigorous_energy_eligible": False,
        "rigorous_utility_eligible": False,
        "status": "legacy_run_record_unverified",
    }
    assert row["metrics"]["accuracy"] is None
    assert row["metrics"]["joules"] is None
    assert row["metrics"]["dollars"] is None
    for private_value in (
        "compute-01.internal",
        "/home/alice/private/run",
        "sk-proj-abcdefghijklmnop",
        "10.42.0.9",
        "raw_config",
        "run_dir",
        '"host"',
        "orchestration",
    ):
        assert private_value not in text
