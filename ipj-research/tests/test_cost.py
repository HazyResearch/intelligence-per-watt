from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from ipj.cost import (
    PRICES,
    ApiChargeKind,
    CostAccountingView,
    CostComponents,
    TaskCostObservation,
    dollars_local,
    dollars_tokens,
    dollars_tokens_decimal,
    workload_ipd,
)


def _strict_cost(**overrides):
    values = {
        "snapshot_id": "cost-fixture-v2",
        "provenance": "fixture://ipd-v2",
        "accounting_view": CostAccountingView.PROVIDER_BILLED,
        "causal_components_complete": True,
        "api_charge_kind": ApiChargeKind.PROVIDER_REPORTED_OR_SETTLED,
    }
    values.update(overrides)
    return CostComponents(**values)


def _verified_observation(*, utility, cost, membership_id: str = "fixture-task-0", manifest_size: int = 1):
    return TaskCostObservation(
        utility=utility,
        cost=cost,
        utility_complete=True,
        utility_provenance="task_native_verifier",
        utility_verifier_id="fixture-native-verifier",
        utility_verifier_version="1.0.0",
        utility_task_manifest_id="fixture-task-manifest-v1",
        utility_task_membership_id=membership_id,
        utility_task_manifest_size=manifest_size,
    )


def test_prices_fable_shape():
    # PRICES is a deliverable consumed downstream; lock its shape/order.
    assert "anthropic/claude-fable-5" in PRICES
    price_in, price_out = PRICES["anthropic/claude-fable-5"]
    assert (price_in, price_out) == (10.0, 50.0)


def test_tokens_cost_fable_one_million_each():
    # $10/M in + $50/M out
    assert dollars_tokens(1_000_000, 1_000_000, 10.0, 50.0) == 60.0


def test_tokens_cost_zero():
    assert dollars_tokens(0, 0, 10.0, 50.0) == 0.0


def test_local_cost_one_hour_one_gpu():
    assert dollars_local(3600.0, 2.5, n_gpus=1) == 2.5


def test_local_cost_scales_with_gpus_and_time():
    assert dollars_local(1800.0, 2.0, n_gpus=4) == 4.0


def test_exact_token_cost_avoids_binary_float_arithmetic():
    assert dollars_tokens_decimal(1, 1, "0.1", "0.2") == Decimal("0.0000003")


def test_exact_token_cost_rejects_float_prices():
    with pytest.raises(TypeError, match="binary floats"):
        dollars_tokens_decimal(1, 1, 0.1, "0.2")  # type: ignore[arg-type]


def test_cost_components_preserve_separate_exact_values():
    components = CostComponents(
        api_usd=Decimal("0.125"),
        electricity_usd=Decimal("0.025"),
        hardware_amortization_usd=Decimal("0.350"),
        snapshot_id="cost-fixture-v1",
    )

    assert components.total_usd == Decimal("0.500")


def test_provider_billed_requires_provider_reported_or_settled_charge() -> None:
    catalog = _strict_cost(
        api_usd=Decimal("1"),
        api_charge_kind=ApiChargeKind.CATALOG_QUOTED,
    )

    assert catalog.headline_eligible is False


def test_not_applicable_api_charge_requires_zero_api_cost() -> None:
    with pytest.raises(ValueError, match="api_usd == 0"):
        _strict_cost(
            api_usd=Decimal("1"),
            accounting_view=CostAccountingView.MARGINAL_ELECTRICITY,
            api_charge_kind=ApiChargeKind.NOT_APPLICABLE,
        )


def test_workload_ipd_rejects_mixed_api_charge_kinds() -> None:
    result = workload_ipd(
        [
            _verified_observation(
                utility=Decimal(1),
                membership_id="fixture-task-0",
                manifest_size=2,
                cost=_strict_cost(
                    electricity_usd=Decimal("0.1"),
                    accounting_view=CostAccountingView.MARGINAL_ELECTRICITY,
                    api_charge_kind=ApiChargeKind.NOT_APPLICABLE,
                ),
            ),
            _verified_observation(
                utility=Decimal(1),
                membership_id="fixture-task-1",
                manifest_size=2,
                cost=_strict_cost(
                    electricity_usd=Decimal("0.1"),
                    accounting_view=CostAccountingView.MARGINAL_ELECTRICITY,
                    api_charge_kind=ApiChargeKind.CATALOG_QUOTED,
                ),
            ),
        ]
    )

    assert result.intelligence_per_dollar is None
    assert result.unavailable_reason == ("mixed accounting views, snapshots, or API charge kinds")


def test_workload_ipd_is_ratio_of_totals_not_mean_of_ratios():
    result = workload_ipd(
        [
            _verified_observation(
                utility=Decimal(1),
                cost=_strict_cost(api_usd=Decimal(1)),
                membership_id="fixture-task-0",
                manifest_size=2,
            ),
            _verified_observation(
                utility=Decimal(0),
                cost=_strict_cost(api_usd=Decimal(3)),
                membership_id="fixture-task-1",
                manifest_size=2,
            ),
        ]
    )

    # sum(utility) / sum(cost) = 1 / 4. A mean of per-task ratios would be 1 / 2.
    assert result.intelligence_per_dollar == Decimal("0.25")
    assert result.utility_total == Decimal(1)
    assert result.cost_total_usd == Decimal(4)
    assert result.complete is True


def test_workload_ipd_fails_closed_for_unknown_task_cost():
    result = workload_ipd(
        [
            _verified_observation(utility=Decimal(1), cost=_strict_cost(api_usd=Decimal("0.5"))),
            _verified_observation(utility=Decimal(1), cost=None),
        ]
    )

    assert result.intelligence_per_dollar is None
    assert result.cost_total_usd is None
    assert result.known_cost_tasks == 1
    assert result.task_count == 2
    assert result.complete is False


def test_workload_ipd_zero_denominator_is_unavailable():
    result = workload_ipd([_verified_observation(utility=Decimal(1), cost=_strict_cost())])

    assert result.cost_total_usd == Decimal(0)
    assert result.intelligence_per_dollar is None
    assert result.complete is True


def test_workload_ipd_rejects_unpinned_or_unverified_observation():
    result = workload_ipd([TaskCostObservation(utility=Decimal(1), cost=CostComponents(api_usd=Decimal(1)))])

    assert result.intelligence_per_dollar is None
    assert result.complete is False
    assert result.eligible_tasks == 0


def test_workload_ipd_rejects_mixed_accounting_views():
    result = workload_ipd(
        [
            _verified_observation(
                utility=Decimal(1),
                membership_id="fixture-task-0",
                manifest_size=2,
                cost=_strict_cost(
                    electricity_usd=Decimal(1),
                    accounting_view=CostAccountingView.MARGINAL_ELECTRICITY,
                ),
            ),
            _verified_observation(
                utility=Decimal(1),
                membership_id="fixture-task-1",
                manifest_size=2,
                cost=_strict_cost(
                    electricity_usd=Decimal(1),
                    accounting_view=CostAccountingView.FULLY_LOADED_ELECTRICITY,
                ),
            ),
        ]
    )

    assert result.intelligence_per_dollar is None
    assert result.unavailable_reason == "mixed accounting views, snapshots, or API charge kinds"


def test_workload_ipd_rejects_component_excluded_by_view():
    result = workload_ipd(
        [
            _verified_observation(
                utility=Decimal(1),
                cost=_strict_cost(
                    electricity_usd=Decimal(1),
                    hardware_amortization_usd=Decimal(100),
                    accounting_view=CostAccountingView.MARGINAL_ELECTRICITY,
                ),
            )
        ]
    )

    assert result.intelligence_per_dollar is None
    assert result.complete is False
    assert result.eligible_tasks == 0


def test_workload_ipd_requires_calibration_id_for_judge_utility():
    result = workload_ipd(
        [
            TaskCostObservation(
                utility=Decimal(1),
                cost=_strict_cost(api_usd=Decimal(1)),
                utility_complete=True,
                utility_provenance="calibrated_judge",
                utility_verifier_id="fixture-judge",
                utility_verifier_version="1.0.0",
                utility_task_manifest_id="fixture-task-manifest-v1",
                utility_task_membership_id="fixture-task-0",
                utility_task_manifest_size=1,
            )
        ]
    )

    assert result.intelligence_per_dollar is None
    assert result.complete is False


@pytest.mark.parametrize(
    "missing_field",
    [
        "utility_verifier_id",
        "utility_verifier_version",
        "utility_task_manifest_id",
        "utility_task_membership_id",
        "utility_task_manifest_size",
    ],
)
def test_workload_ipd_requires_verifier_and_manifest_bindings(
    missing_field: str,
) -> None:
    evidence = {
        "utility_verifier_id": "fixture-native-verifier",
        "utility_verifier_version": "1.0.0",
        "utility_task_manifest_id": "fixture-task-manifest-v1",
        "utility_task_membership_id": "fixture-task-0",
        "utility_task_manifest_size": 1,
    }
    evidence[missing_field] = ""
    result = workload_ipd(
        [
            TaskCostObservation(
                utility=Decimal(1),
                cost=_strict_cost(api_usd=Decimal(1)),
                utility_complete=True,
                utility_provenance="task_native_verifier",
                **evidence,
            )
        ]
    )

    assert result.intelligence_per_dollar is None
    assert result.eligible_tasks == 0


def test_workload_ipd_rejects_mixed_utility_manifests() -> None:
    first = _verified_observation(
        utility=Decimal(1),
        cost=_strict_cost(api_usd=Decimal(1)),
        membership_id="fixture-task-0",
        manifest_size=2,
    )
    second = TaskCostObservation(
        utility=Decimal(1),
        cost=_strict_cost(api_usd=Decimal(1)),
        utility_complete=True,
        utility_provenance="task_native_verifier",
        utility_verifier_id="fixture-native-verifier",
        utility_verifier_version="1.0.0",
        utility_task_manifest_id="different-task-manifest",
        utility_task_membership_id="fixture-task-1",
        utility_task_manifest_size=2,
    )

    result = workload_ipd([first, second])

    assert result.intelligence_per_dollar is None
    assert result.unavailable_reason == ("mixed utility verifier, manifest, or calibration basis")


@pytest.mark.parametrize("membership_ids", [["task-0"], ["task-0", "task-0"]])
def test_workload_ipd_rejects_manifest_subsets_and_duplicates(
    membership_ids: list[str],
) -> None:
    observations = [
        _verified_observation(
            utility=Decimal(1),
            cost=_strict_cost(api_usd=Decimal(1)),
            membership_id=membership_id,
            manifest_size=2,
        )
        for membership_id in membership_ids
    ]

    result = workload_ipd(observations)

    assert result.intelligence_per_dollar is None
    assert result.unavailable_reason == ("incomplete or duplicate task-manifest population")


def test_workload_ipd_rejects_empty_workload():
    with pytest.raises(ValueError, match="at least one"):
        workload_ipd([])


def test_ipd_v2_conformance_corpus():
    fixture_path = Path(__file__).parent / "fixtures" / "accounting" / "ipd-v2.json"
    corpus = json.loads(fixture_path.read_text())
    assert corpus["schema_version"] == "ipj.ipd.conformance.v2"

    for case in corpus["cases"]:
        observations = []
        for raw in case["observations"]:
            raw_cost = raw["cost"]
            components = None if raw_cost is None else CostComponents(**raw_cost)
            observations.append(
                TaskCostObservation(
                    utility=raw["utility"],
                    cost=components,
                    utility_complete=raw["utility_complete"],
                    utility_provenance=raw["utility_provenance"],
                    utility_calibration_id=raw.get("utility_calibration_id"),
                    utility_verifier_id=raw.get("utility_verifier_id"),
                    utility_verifier_version=raw.get("utility_verifier_version"),
                    utility_task_manifest_id=raw.get("utility_task_manifest_id"),
                    utility_task_membership_id=raw.get("utility_task_membership_id"),
                    utility_task_manifest_size=raw.get("utility_task_manifest_size"),
                )
            )

        result = workload_ipd(observations)
        expected = case["expected"]
        expected_cost = expected["cost_total_usd"]
        expected_ipd = expected["intelligence_per_dollar"]
        assert result.utility_total == Decimal(expected["utility_total"]), case["name"]
        assert result.cost_total_usd == (None if expected_cost is None else Decimal(expected_cost)), case["name"]
        assert result.intelligence_per_dollar == (None if expected_ipd is None else Decimal(expected_ipd)), case["name"]
        assert result.known_cost_tasks == expected["known_cost_tasks"], case["name"]
        assert result.eligible_tasks == expected["eligible_tasks"], case["name"]
        if result.complete:
            first_cost = next(raw["cost"] for raw in case["observations"] if raw["cost"] is not None)
            assert result.api_charge_kind == ApiChargeKind(first_cost["api_charge_kind"]), case["name"]
        else:
            assert result.api_charge_kind is None, case["name"]
