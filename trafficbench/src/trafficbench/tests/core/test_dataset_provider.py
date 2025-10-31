from __future__ import annotations

from typing import Iterable

from trafficbench.core.types import DatasetRecord
from trafficbench.datasets.base import DatasetProvider


class ExampleProvider(DatasetProvider):
    dataset_id = "example"
    dataset_name = "Example Dataset"

    def __init__(self, records: Iterable[DatasetRecord]) -> None:
        self._records = tuple(records)

    def iter_records(self) -> Iterable[DatasetRecord]:
        for record in self._records:
            yield record

    def size(self) -> int:
        return len(self._records)


def test_dataset_provider_iter_delegates() -> None:
    records = (
        DatasetRecord(problem="p1", answer="a1", subject="s1"),
        DatasetRecord(problem="p2", answer="a2", subject="s2"),
    )

    provider = ExampleProvider(records)

    assert list(provider) == list(records)
    assert provider.size() == 2
