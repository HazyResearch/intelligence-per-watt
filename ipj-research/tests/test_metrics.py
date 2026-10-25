from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import pytest

from ipj.metrics import Cell, append_cell, utility_is_complete, write_cell


def _cell(**kw):
    base = dict(
        model="openai/Qwen/Qwen3.6-27B",
        harness="terminus-2",
        hardware="H100",
        dataset="terminal-bench/terminal-bench-2-1",
        n_tasks=5,
        accuracy=0.6,
        tokens_in=120000,
        tokens_out=18000,
        dollars=0.5,
        dollars_basis="gpu-hours",
        joules=4200.0,
        wall_clock_s=720.0,
        energy_source="energy-monitor",
        run_dir="runs/x",
    )
    base.update(kw)
    return Cell(**base)


def test_write_cell_roundtrip(tmp_path: Path):
    p = tmp_path / "cell.json"
    write_cell(_cell(), p)
    data = json.loads(p.read_text())
    assert data["accuracy"] == 0.6
    assert data["joules"] == 4200.0


def test_write_cell_keeps_cloud_estimate_separate_from_measured_joules(tmp_path: Path):
    from ipj.energy.estimate import JouleEstimate

    p = tmp_path / "hybrid.json"
    estimate = JouleEstimate(
        joules=800.0,
        lo=560.0,
        hi=1040.0,
        j_per_token=2.0,
        band=0.3,
        basis="transfer from measured local reference",
    )
    write_cell(
        _cell(
            joules=4000.0,
            joules_provenance="measured_dedicated",
            joules_scope="local-accelerator",
            joules_complete=False,
            cloud_joules_estimate=estimate,
        ),
        p,
    )

    data = json.loads(p.read_text())
    assert data["joules"] == 4000.0
    assert data["cloud_joules_estimate"]["joules"] == 800.0
    assert data["joules_complete"] is False


def test_cloud_cell_joules_null(tmp_path: Path):
    p = tmp_path / "cloud.json"
    write_cell(_cell(joules=None, energy_source="n/a (cloud)", dollars_basis="api-tokens"), p)
    assert json.loads(p.read_text())["joules"] is None


def test_complete_joules_require_explicit_dedicated_provenance():
    with pytest.raises(ValueError, match="measured_dedicated"):
        _cell(
            joules_complete=True,
            joules_provenance="unknown",
            accelerator_device_coverage_complete=True,
        )


def test_complete_joules_require_device_coverage():
    with pytest.raises(ValueError, match="device coverage"):
        _cell(
            joules_complete=True,
            joules_provenance="measured_dedicated",
        )


def test_complete_joules_allow_a_declared_measured_zero():
    cell = _cell(
        joules=0.0,
        joules_complete=True,
        joules_provenance="measured_dedicated",
        accelerator_device_coverage_complete=True,
    )

    assert cell.joules_complete is True


def test_complete_utility_accepts_task_native_verifier():
    cell = _cell(
        utility_complete=True,
        utility_provenance="task_native_verifier",
        utility_verifier_id="terminalbench-native-verifier",
        utility_verifier_version="2.1.0",
        utility_task_manifest_id="terminalbench-v2.1-fixture",
        utility_membership_complete=True,
        utility_task_manifest_size=5,
    )

    assert cell.utility_complete is True


def test_complete_utility_requires_calibration_id_for_judge():
    with pytest.raises(ValueError, match="calibrated_judge"):
        _cell(
            utility_complete=True,
            utility_provenance="calibrated_judge",
            utility_verifier_id="fixture-judge",
            utility_verifier_version="1.0.0",
            utility_task_manifest_id="fixture-task-manifest-v1",
            utility_membership_complete=True,
            utility_task_manifest_size=5,
        )


def test_utility_completeness_rejects_partial_or_invalid_workloads():
    assert not utility_is_complete(0.5, 4, False, "task_native_verifier")
    assert not utility_is_complete(float("nan"), 4, True, "task_native_verifier")
    assert not utility_is_complete(0.5, 0, True, "task_native_verifier")


@pytest.mark.parametrize(
    "missing_field",
    [
        "utility_verifier_id",
        "utility_verifier_version",
        "utility_task_manifest_id",
        "utility_membership_complete",
        "utility_task_manifest_size",
    ],
)
def test_complete_utility_requires_verifier_and_manifest_bindings(
    missing_field: str,
) -> None:
    evidence = {
        "utility_verifier_id": "terminalbench-native-verifier",
        "utility_verifier_version": "2.1.0",
        "utility_task_manifest_id": "terminalbench-v2.1-fixture",
        "utility_membership_complete": True,
        "utility_task_manifest_size": 5,
    }
    evidence[missing_field] = " "

    with pytest.raises(ValueError, match="pinned verifier"):
        _cell(
            utility_complete=True,
            utility_provenance="task_native_verifier",
            **evidence,
        )


def test_complete_dollars_require_pinned_accounting_evidence() -> None:
    cell = _cell(
        dollars_complete=True,
        dollars_accounting_view="provider_billed",
        dollars_snapshot_id="provider-settlement-2026-08",
        dollars_provenance="provider_invoice",
        dollars_api_charge_kind="provider_reported_or_settled",
    )

    assert cell.dollars_complete is True


@pytest.mark.parametrize(
    "missing_field",
    [
        "dollars_accounting_view",
        "dollars_snapshot_id",
        "dollars_provenance",
        "dollars_api_charge_kind",
    ],
)
def test_complete_dollars_reject_missing_accounting_evidence(
    missing_field: str,
) -> None:
    evidence = {
        "dollars_accounting_view": "provider_billed",
        "dollars_snapshot_id": "provider-settlement-2026-08",
        "dollars_provenance": "provider_invoice",
        "dollars_api_charge_kind": "provider_reported_or_settled",
    }
    evidence[missing_field] = ""

    with pytest.raises(ValueError, match="accounting view"):
        _cell(dollars_complete=True, **evidence)


def test_complete_dollars_reject_unknown_accounting_view() -> None:
    with pytest.raises(ValueError, match="declared accounting view"):
        _cell(
            dollars_complete=True,
            dollars_accounting_view="mixed_unspecified_cost",
            dollars_snapshot_id="cost-fixture-v1",
            dollars_provenance="fixture",
            dollars_api_charge_kind="catalog_quoted",
        )


def test_provider_billed_cell_rejects_catalog_quote_as_complete() -> None:
    with pytest.raises(ValueError, match="typed API charge kind"):
        _cell(
            dollars_complete=True,
            dollars_accounting_view="provider_billed",
            dollars_snapshot_id="catalog-v1",
            dollars_provenance="catalog",
            dollars_api_charge_kind="catalog_quoted",
        )


def test_cell_preserves_existing_positional_field_order() -> None:
    previous_fields = [
        "model",
        "harness",
        "hardware",
        "dataset",
        "n_tasks",
        "accuracy",
        "tokens_in",
        "tokens_out",
        "dollars",
        "dollars_basis",
        "joules",
        "wall_clock_s",
        "energy_source",
        "run_dir",
        "n_local",
        "n_cloud",
        "classifier_dollars",
        "joules_provenance",
        "joules_scope",
        "joules_complete",
        "accelerator_device_coverage_complete",
        "cloud_joules_estimate",
        "utility_complete",
        "utility_provenance",
        "utility_calibration_id",
    ]

    assert [field.name for field in fields(Cell)][: len(previous_fields)] == previous_fields
    cell = _cell()
    assert cell.utility_verifier_id is None
    assert cell.utility_verifier_version is None
    assert cell.utility_task_manifest_id is None
    assert cell.utility_membership_complete is None
    assert cell.utility_task_manifest_size is None
    assert cell.dollars_complete is None
    assert cell.dollars_api_charge_kind is None


def test_append_cell_adds_one_line(tmp_path: Path):
    p = tmp_path / "cells.jsonl"
    append_cell(_cell(), p)
    append_cell(_cell(accuracy=0.8), p)
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["accuracy"] == 0.8
