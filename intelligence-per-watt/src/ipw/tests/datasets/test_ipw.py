from __future__ import annotations

from typing import NoReturn

import pytest
from ipw.datasets.ipw import IPWDataset


def _skip_missing_dataset(exc: FileNotFoundError) -> NoReturn:
    pytest.skip(str(exc))
    raise AssertionError("pytest.skip is expected to abort the test")


@pytest.fixture(scope="module")
def dataset() -> IPWDataset:
    try:
        return IPWDataset()
    except FileNotFoundError as exc:
        _skip_missing_dataset(exc)
    # The skip helper never returns; this line satisfies the type checker.
    raise AssertionError("unreachable")


def test_dataset_iterates_records(dataset: IPWDataset) -> None:
    first = next(iter(dataset.iter_records()))
    assert first.problem
    assert first.answer


def test_dataset_size_nonzero(dataset: IPWDataset) -> None:
    assert dataset.size() > 0
