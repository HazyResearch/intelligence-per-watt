from __future__ import annotations

import pytest

from mindi.datasets.trafficbench import TrafficBenchDataset


@pytest.fixture(scope="module")
def dataset() -> TrafficBenchDataset:
    try:
        return TrafficBenchDataset()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


def test_dataset_iterates_records(dataset: TrafficBenchDataset) -> None:
    first = next(iter(dataset.iter_records()))
    assert first.problem
    assert first.answer


def test_dataset_size_nonzero(dataset: TrafficBenchDataset) -> None:
    assert dataset.size() > 0

