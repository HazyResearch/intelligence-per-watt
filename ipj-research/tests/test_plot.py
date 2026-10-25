from __future__ import annotations

from pathlib import Path

import pytest

from ipj.plot import (
    DEFAULT_METRICS,
    compute_pareto_frontier,
    cost_multiplier,
    plot_cost_frontier,
    plot_efficiency_scatter,
    plot_tradeoff_grid,
)

# A small realistic sweep: 1 cloud anchor (no joules/flops) + 2 local Qwen cells.
_CELLS = [
    {
        "model": "openai/gpt-5.5-2026-04-23",
        "harness": "terminus-2",
        "accuracy": 0.75,
        "tokens_in": 14_000_000,
        "tokens_out": 770_000,
        "dollars": 41.7,
        "joules": None,
        "wall_clock_s": 14668.0,
        "dollars_basis": "api-billed",
    },
    {
        "model": "openai/Qwen/Qwen3.5-27B-FP8",
        "harness": "terminus-2",
        "accuracy": 0.37,
        "tokens_in": 13_000_000,
        "tokens_out": 584_000,
        "dollars": 39.1,
        "joules": 7_236_271.0,
        "joules_complete": True,
        "joules_provenance": "measured_dedicated",
        "accelerator_device_coverage_complete": True,
        "wall_clock_s": 33796.0,
        "dollars_basis": "gpu-hours",
    },
    {
        "model": "openai/Qwen/Qwen3.5-35B-A3B-FP8",
        "harness": "terminus-2",
        "accuracy": 0.27,
        "tokens_in": 15_900_000,
        "tokens_out": 688_000,
        "dollars": 40.5,
        "joules": 4_953_299.0,
        "joules_complete": True,
        "joules_provenance": "measured_dedicated",
        "accelerator_device_coverage_complete": True,
        "wall_clock_s": 35379.0,
        "dollars_basis": "gpu-hours",
    },
]


def test_plot_writes_pdf_and_png(tmp_path: Path):
    cells = [
        {"model": "Qwen3.5-27B-FP8", "accuracy": 0.6, "dollars": 0.04, "joules": 4200.0, "dollars_basis": "gpu-hours"},
        {"model": "gpt-5.5", "accuracy": 0.82, "dollars": 1.5, "joules": None, "dollars_basis": "api-tokens"},
    ]
    out = tmp_path / "fig1.png"
    p = plot_cost_frontier(cells, out, x="dollars")
    # Both the vector PDF (paper) and raster PNG (preview) are written.
    assert Path(p).with_suffix(".png").stat().st_size > 0
    assert Path(p).with_suffix(".pdf").stat().st_size > 0


def test_plot_joules_skips_none(tmp_path: Path):
    # Cloud cell has no measured joules → skipped on the joules axis; still writes without raising.
    cells = [{"model": "gpt-5.5", "accuracy": 0.5, "dollars": 0.1, "joules": None, "dollars_basis": "api-tokens"}]
    out = tmp_path / "j.png"
    assert Path(plot_cost_frontier(cells, out, x="joules")).with_suffix(".png").exists()


def test_joules_metric_preserves_legacy_default_and_supports_rigorous_mode():
    from ipj.plot import _metric_value

    incomplete = {"joules": 123.0, "joules_complete": False}
    legacy = {"joules": 123.0}
    complete = {
        "joules": 123.0,
        "joules_complete": True,
        "joules_provenance": "measured_dedicated",
        "joules_scope": "local-accelerator",
        "accelerator_device_coverage_complete": True,
        "accuracy": 0.75,
        "n_tasks": 4,
        "utility_complete": True,
        "utility_provenance": "task_native_verifier",
        "utility_verifier_id": "terminalbench-native-verifier",
        "utility_verifier_version": "2.1.0",
        "utility_task_manifest_id": "terminalbench-v2.1-fixture",
        "utility_membership_complete": True,
        "utility_task_manifest_size": 4,
    }
    unsupported_claim = {
        "joules": 123.0,
        "joules_complete": True,
        "joules_provenance": "unknown",
        "accelerator_device_coverage_complete": True,
    }

    assert _metric_value(incomplete, "joules") == 123.0
    assert _metric_value(legacy, "joules") == 123.0
    assert _metric_value(unsupported_claim, "joules") == 123.0
    assert _metric_value(complete, "joules") == 123.0
    assert _metric_value(incomplete, "joules", allow_incomplete_energy=False) is None
    assert _metric_value(legacy, "joules", allow_incomplete_energy=False) is None
    assert _metric_value(unsupported_claim, "joules", allow_incomplete_energy=False) is None
    assert _metric_value(complete, "joules", allow_incomplete_energy=False) == 123.0
    assert (
        _metric_value(
            {**complete, "accelerator_device_coverage_complete": False},
            "joules",
            allow_incomplete_energy=False,
        )
        is None
    )
    assert (
        _metric_value(
            {**complete, "utility_complete": False},
            "joules",
            allow_incomplete_energy=False,
        )
        is None
    )
    assert (
        _metric_value(
            {**complete, "joules_scope": "provider-infrastructure"},
            "joules",
            allow_incomplete_energy=False,
        )
        is None
    )
    assert (
        _metric_value(
            {key: value for key, value in complete.items() if key != "joules_scope"},
            "joules",
            allow_incomplete_energy=False,
        )
        is None
    )
    assert (
        _metric_value(
            {**complete, "utility_provenance": "calibrated_judge"},
            "joules",
            allow_incomplete_energy=False,
        )
        is None
    )
    assert (
        _metric_value(
            {
                **complete,
                "utility_provenance": "calibrated_judge",
                "utility_calibration_id": "judge-calibration-v1",
            },
            "joules",
            allow_incomplete_energy=False,
        )
        == 123.0
    )


def test_dollars_metric_preserves_legacy_default_and_supports_rigorous_mode():
    from ipj.plot import _metric_value

    complete = {
        "dollars": 1.25,
        "dollars_complete": True,
        "dollars_accounting_view": "provider_billed",
        "dollars_snapshot_id": "provider-settlement-2026-08",
        "dollars_provenance": "provider_invoice",
        "dollars_api_charge_kind": "provider_reported_or_settled",
        "accuracy": 0.75,
        "n_tasks": 4,
        "utility_complete": True,
        "utility_provenance": "task_native_verifier",
        "utility_verifier_id": "terminalbench-native-verifier",
        "utility_verifier_version": "2.1.0",
        "utility_task_manifest_id": "terminalbench-v2.1-fixture",
        "utility_membership_complete": True,
        "utility_task_manifest_size": 4,
    }

    assert _metric_value({"dollars": 1.25}, "dollars") == 1.25
    assert _metric_value({"dollars": 1.25}, "dollars", allow_incomplete_cost=False) is None
    assert _metric_value(complete, "dollars", allow_incomplete_cost=False) == 1.25
    assert (
        _metric_value(
            {**complete, "n_tasks": 3},
            "dollars",
            allow_incomplete_cost=False,
        )
        is None
    )
    for missing_field in (
        "dollars_snapshot_id",
        "dollars_provenance",
        "dollars_api_charge_kind",
        "utility_verifier_id",
        "utility_verifier_version",
        "utility_task_manifest_id",
        "utility_membership_complete",
        "utility_task_manifest_size",
    ):
        missing = {**complete, missing_field: ""}
        assert _metric_value(missing, "dollars", allow_incomplete_cost=False) is None
    assert (
        _metric_value(
            {**complete, "dollars_accounting_view": "mixed_unspecified"},
            "dollars",
            allow_incomplete_cost=False,
        )
        is None
    )
    assert (
        _metric_value(
            {**complete, "dollars_api_charge_kind": "catalog_quoted"},
            "dollars",
            allow_incomplete_cost=False,
        )
        is None
    )


@pytest.mark.parametrize("invalid", [-1.0, float("inf"), float("nan"), True, "1.0"])
def test_dollars_metric_always_rejects_invalid_values(invalid):
    from ipj.plot import _metric_value

    assert _metric_value({"dollars": invalid}, "dollars") is None
    assert _metric_value({"dollars": invalid}, "dollars", allow_incomplete_cost=False) is None


@pytest.mark.parametrize("invalid", [-1.0, float("inf"), float("nan"), True])
def test_joules_metric_always_rejects_invalid_values(invalid):
    from ipj.plot import _metric_value

    cell = {
        "joules": invalid,
        "joules_complete": True,
        "joules_provenance": "measured_dedicated",
        "accelerator_device_coverage_complete": True,
    }

    assert _metric_value(cell, "joules") is None
    assert _metric_value(cell, "joules", allow_incomplete_energy=False) is None


def test_pareto_frontier_minimize_cost_maximize_accuracy():
    # Cheaper-and-better dominates; the dominated middle point is dropped.
    px, py = compute_pareto_frontier([0.04, 0.5, 1.5], [0.6, 0.55, 0.82])
    assert list(zip(px, py)) == [(0.04, 0.6), (1.5, 0.82)]


def test_plot_flops_axis(tmp_path: Path):
    # FLOPs computed from tokens×params; cloud (unknown params) is skipped, local plotted.
    cells = [
        {
            "model": "Qwen/Qwen3.5-4B",
            "accuracy": 0.5,
            "tokens_in": 1000,
            "tokens_out": 500,
            "dollars": 1.0,
            "joules": 100.0,
            "dollars_basis": "gpu-hours",
        },
        {
            "model": "openai/gpt-5.5-2026-04-23",
            "accuracy": 0.75,
            "tokens_in": 2000,
            "tokens_out": 1000,
            "dollars": 40.0,
            "joules": None,
            "dollars_basis": "api-billed",
        },
    ]
    out = tmp_path / "fig_flops.png"
    assert Path(plot_cost_frontier(cells, out, x="flops")).with_suffix(".png").stat().st_size > 0


def test_plot_metric_grid_axes(tmp_path: Path):
    # Track E: accuracy vs latency + token axes.
    cells = [
        {
            "model": "Qwen/Qwen3.5-4B",
            "accuracy": 0.5,
            "tokens_in": 1000,
            "tokens_out": 500,
            "dollars": 1.0,
            "joules": 100.0,
            "wall_clock_s": 42.0,
            "dollars_basis": "gpu-hours",
        },
        {
            "model": "gpt-5.5",
            "accuracy": 0.75,
            "tokens_in": 2000,
            "tokens_out": 1000,
            "dollars": 40.0,
            "joules": None,
            "wall_clock_s": 90.0,
            "dollars_basis": "api-billed",
        },
    ]
    for x in ("latency", "tokens", "tokens_in", "tokens_out"):
        out = tmp_path / f"fig_{x}.png"
        assert Path(plot_cost_frontier(cells, out, x=x)).with_suffix(".png").stat().st_size > 0


@pytest.mark.parametrize("metric", DEFAULT_METRICS)
def test_efficiency_scatter_all_axes(tmp_path: Path, metric: str):
    # Rich-encoding single scatter writes a non-empty PDF + PNG for every efficiency axis.
    out = tmp_path / f"eff_{metric}.png"
    p = plot_efficiency_scatter(_CELLS, out, metric=metric)
    assert Path(p).with_suffix(".png").stat().st_size > 0
    assert Path(p).with_suffix(".pdf").stat().st_size > 0


def test_efficiency_scatter_rejects_unknown_metric(tmp_path: Path):
    with pytest.raises(ValueError):
        plot_efficiency_scatter(_CELLS, tmp_path / "x.png", metric="bogus")


def test_tradeoff_grid_writes_pdf_and_png(tmp_path: Path):
    # The combined multi-panel grid renders all default axes into one figure.
    out = tmp_path / "grid.png"
    p = plot_tradeoff_grid(_CELLS, out)
    assert Path(p).with_suffix(".png").stat().st_size > 0
    assert Path(p).with_suffix(".pdf").stat().st_size > 0


def test_tradeoff_grid_rejects_unknown_metric(tmp_path: Path):
    with pytest.raises(ValueError):
        plot_tradeoff_grid(_CELLS, tmp_path / "g.png", metrics=["dollars", "bogus"])


def test_cell_kind_classification():
    from ipj.plot import _cell_kind

    assert _cell_kind({"model": "openai/Qwen/Qwen3.5-27B-FP8", "dollars_basis": "gpu-hours"}) == "local"
    assert _cell_kind({"model": "openai/gpt-5.5-2026-04-23", "dollars_basis": "api-billed"}) == "cloud"
    assert (
        _cell_kind(
            {
                "model": "cascade-v1-realized[27B->gpt-5.5]",
                "harness": "terminus-2-cascade",
                "dollars_basis": "gpu-hours+api",
            }
        )
        == "strategy"
    )


def test_strategy_and_dollars_range_render(tmp_path: Path):
    # A cascade strategy star + a flash dot with a dollars_alt range both render on the $ axis.
    cells = _CELLS + [
        {
            "model": "gemini-2.5-flash",
            "harness": "terminus-2",
            "accuracy": 0.146,
            "dollars": 22.13,
            "dollars_alt": 67.85,
            "joules": None,
            "wall_clock_s": 12643.0,
            "tokens_in": 33_548_934,
            "tokens_out": 5_445_909,
            "dollars_basis": "api-billed",
        },
        {
            "model": "cascade-v1-oracle[27B->gpt-5.5]",
            "harness": "terminus-2-cascade",
            "accuracy": 0.775,
            "dollars": 65.87,
            "dollars_alt": 26.73,
            "joules": 11_795_418.0,
            "wall_clock_s": 56587.0,
            "dollars_basis": "gpu-hours+api",
        },
    ]
    out = tmp_path / "strat.png"
    assert Path(plot_efficiency_scatter(cells, out, metric="dollars")).with_suffix(".png").stat().st_size > 0
    assert Path(plot_tradeoff_grid(cells, tmp_path / "g.png")).with_suffix(".png").stat().st_size > 0


def test_total_params_vs_active_for_moe():
    # MoE size encoding uses TOTAL params (35B) while FLOPs use ACTIVE params (3B).
    from ipj.flops import active_params, total_params

    assert total_params("openai/Qwen/Qwen3.5-35B-A3B-FP8") == 35e9
    assert active_params("openai/Qwen/Qwen3.5-35B-A3B-FP8") == 3e9
    assert total_params("openai/gpt-5.5-2026-04-23") is None  # cloud params unknown


def test_cost_multiplier():
    assert cost_multiplier(0.07, 1.40) == pytest.approx(20.0)  # 1.40/0.07 (float)
    assert cost_multiplier(0.0, 1.40) == float("inf")  # guard against divide-by-zero


def test_plot_with_local_and_cloud_draws_bracket(tmp_path: Path):
    # A local + a cloud point on the $ axis → the "N× cheaper" bracket is drawn without error.
    cells = [
        {"model": "Qwen3.5-27B-FP8", "accuracy": 0.55, "dollars": 0.07, "joules": 5000.0, "dollars_basis": "gpu-hours"},
        {"model": "gpt-5.5", "accuracy": 0.66, "dollars": 1.40, "joules": None, "dollars_basis": "api-billed"},
    ]
    out = tmp_path / "fig1_bracket.png"
    assert Path(plot_cost_frontier(cells, out, x="dollars")).with_suffix(".png").stat().st_size > 0


def test_cost_bracket_distinguishes_legacy_and_rigorous_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels: list[str] = []

    def capture_bracket(*args, **_kwargs) -> None:
        labels.append(args[4])

    monkeypatch.setattr("ipj.plot.draw_improvement_bracket", capture_bracket)
    cells = [
        {
            "model": "local-model",
            "accuracy": 0.55,
            "dollars": 0.07,
            "dollars_basis": "gpu-hours",
        },
        {
            "model": "cloud-model",
            "accuracy": 0.66,
            "dollars": 1.40,
            "dollars_basis": "api-billed",
        },
    ]
    plot_cost_frontier(cells, tmp_path / "legacy.png", x="dollars")

    evidence = {
        "dollars_complete": True,
        "dollars_accounting_view": "provider_billed",
        "dollars_snapshot_id": "provider-settlement-2026-08",
        "dollars_provenance": "provider_invoice",
        "dollars_api_charge_kind": "provider_reported_or_settled",
        "n_tasks": 4,
        "utility_complete": True,
        "utility_provenance": "task_native_verifier",
        "utility_verifier_id": "fixture-native-verifier",
        "utility_verifier_version": "1.0.0",
        "utility_task_manifest_id": "fixture-task-manifest-v1",
        "utility_membership_complete": True,
        "utility_task_manifest_size": 4,
    }
    rigorous_cells = [{**cell, **evidence} for cell in cells]
    plot_cost_frontier(
        rigorous_cells,
        tmp_path / "rigorous.png",
        x="dollars",
        allow_incomplete_cost=False,
    )

    assert labels == ["Legacy 20× cheaper", "20× cheaper"]


def test_strict_efficiency_panel_suppresses_unbound_cost_range() -> None:
    import matplotlib.pyplot as plt

    from ipj.plot import _draw_panel

    cell = {
        "model": "local-model",
        "harness": "terminus-2",
        "accuracy": 0.5,
        "dollars": 1.0,
        "dollars_alt": 2.0,
        "dollars_basis": "gpu-hours",
        "dollars_complete": True,
        "dollars_accounting_view": "provider_billed",
        "dollars_snapshot_id": "provider-settlement-2026-08",
        "dollars_provenance": "provider_invoice",
        "dollars_api_charge_kind": "provider_reported_or_settled",
        "n_tasks": 1,
        "utility_complete": True,
        "utility_provenance": "task_native_verifier",
        "utility_verifier_id": "fixture-native-verifier",
        "utility_verifier_version": "1.0.0",
        "utility_task_manifest_id": "fixture-task-manifest-v1",
        "utility_membership_complete": True,
        "utility_task_manifest_size": 1,
    }
    fig, ax = plt.subplots()
    _draw_panel(ax, [cell], "dollars", allow_incomplete_cost=False)

    assert len(ax.lines) == 1
    plt.close(fig)


@pytest.mark.parametrize(
    ("field", "different_value"),
    [
        ("dollars_accounting_view", "fully_loaded_electricity"),
        ("dollars_snapshot_id", "different-cost-snapshot"),
        ("dollars_api_charge_kind", "catalog_quoted"),
        ("utility_task_manifest_id", "different-task-manifest"),
        ("dataset", "different-dataset"),
    ],
)
def test_strict_cost_frontier_rejects_mixed_comparison_bases(
    tmp_path: Path, field: str, different_value: object
) -> None:
    evidence = {
        "dollars_complete": True,
        "dollars_accounting_view": "marginal_electricity",
        "dollars_snapshot_id": "provider-settlement-2026-08",
        "dollars_provenance": "provider_invoice",
        "dollars_api_charge_kind": "not_applicable",
        "n_tasks": 4,
        "dataset": "terminalbench-v2.1-fixture",
        "utility_complete": True,
        "utility_provenance": "task_native_verifier",
        "utility_verifier_id": "fixture-native-verifier",
        "utility_verifier_version": "1.0.0",
        "utility_task_manifest_id": "fixture-task-manifest-v1",
        "utility_membership_complete": True,
        "utility_task_manifest_size": 4,
    }
    cells = [
        {
            **evidence,
            "model": "model-a",
            "accuracy": 0.5,
            "dollars": 1.0,
            "dollars_basis": "gpu-hours",
        },
        {
            **evidence,
            field: different_value,
            "model": "model-b",
            "accuracy": 0.6,
            "dollars": 2.0,
            "dollars_basis": "api-billed",
        },
    ]

    with pytest.raises(ValueError, match="strict"):
        plot_cost_frontier(
            cells,
            tmp_path / "mixed.png",
            x="dollars",
            allow_incomplete_cost=False,
        )


def test_strict_energy_frontier_rejects_mixed_task_manifests(
    tmp_path: Path,
) -> None:
    evidence = {
        "joules_complete": True,
        "joules_provenance": "measured_dedicated",
        "joules_scope": "local-accelerator",
        "accelerator_device_coverage_complete": True,
        "n_tasks": 4,
        "dataset": "terminalbench-v2.1-fixture",
        "utility_complete": True,
        "utility_provenance": "task_native_verifier",
        "utility_verifier_id": "fixture-native-verifier",
        "utility_verifier_version": "1.0.0",
        "utility_task_manifest_id": "fixture-task-manifest-v1",
        "utility_membership_complete": True,
        "utility_task_manifest_size": 4,
    }
    cells = [
        {**evidence, "model": "model-a", "accuracy": 0.5, "joules": 10.0},
        {
            **evidence,
            "utility_task_manifest_id": "different-task-manifest",
            "model": "model-b",
            "accuracy": 0.6,
            "joules": 20.0,
        },
    ]

    with pytest.raises(ValueError, match="strict comparison"):
        plot_cost_frontier(
            cells,
            tmp_path / "mixed-energy.png",
            x="joules",
            allow_incomplete_energy=False,
        )
