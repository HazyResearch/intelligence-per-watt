from __future__ import annotations

from typing import NoReturn

import pytest

from trafficbench.datasets.trafficbench import TrafficBenchDataset


def _skip_missing_dataset(exc: FileNotFoundError) -> NoReturn:
    pytest.skip(str(exc))
    raise AssertionError("pytest.skip is expected to abort the test")


@pytest.fixture(scope="module")
def dataset() -> TrafficBenchDataset:
    try:
        return TrafficBenchDataset()
    except FileNotFoundError as exc:
        _skip_missing_dataset(exc)
    # The skip helper never returns; this line satisfies the type checker.
    raise AssertionError("unreachable")


def test_dataset_iterates_records(dataset: TrafficBenchDataset) -> None:
    first = next(iter(dataset.iter_records()))
    assert first.problem
    assert first.answer


def test_dataset_size_nonzero(dataset: TrafficBenchDataset) -> None:
    assert dataset.size() > 0
