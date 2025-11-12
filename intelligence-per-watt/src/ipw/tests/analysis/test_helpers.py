"""Tests for analysis helper functions."""

from __future__ import annotations

from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict
from ipw.analysis.helpers import (
    collect_run_metadata,
    iter_model_entries,
    load_metrics_dataset,
    resolve_model_name,
)


class TestLoadMetricsDataset:
    """Test dataset loading."""

    def test_loads_single_dataset(self, tmp_path: Path) -> None:
        # Create a simple dataset
        dataset = Dataset.from_list([{"id": 1}, {"id": 2}])
        dataset.save_to_disk(str(tmp_path / "data"))

        loaded = load_metrics_dataset(tmp_path / "data")
        assert isinstance(loaded, Dataset)
        assert len(loaded) == 2

    def test_loads_first_split_from_dataset_dict(self, tmp_path: Path) -> None:
        # Create a DatasetDict
        dataset = Dataset.from_list([{"id": 1}])
        dataset_dict = DatasetDict({"train": dataset})
        dataset_dict.save_to_disk(str(tmp_path / "data"))

        loaded = load_metrics_dataset(tmp_path / "data")
        assert isinstance(loaded, Dataset)
        assert len(loaded) == 1

    def test_raises_on_empty_dataset_dict(self, tmp_path: Path) -> None:
        dataset_dict = DatasetDict({})
        dataset_dict.save_to_disk(str(tmp_path / "data"))

        with pytest.raises(RuntimeError, match="No splits found"):
            load_metrics_dataset(tmp_path / "data")

    def test_raises_on_empty_dataset(self, tmp_path: Path) -> None:
        from datasets import Features, Value

        # Create empty dataset with schema
        dataset = Dataset.from_dict(
            {"id": []},
            features=Features({"id": Value("int64")}),
        )
        dataset.save_to_disk(str(tmp_path / "data"))

        with pytest.raises(RuntimeError, match="Empty dataset"):
            load_metrics_dataset(tmp_path / "data")


class TestResolveModelName:
    """Test model name resolution."""

    def test_returns_requested_model_when_available(self) -> None:
        dataset = Dataset.from_list(
            [
                {"model_metrics": {"llama": {"value": 1}}},
            ]
        )

        model = resolve_model_name(dataset, "llama", Path("/fake"))
        assert model == "llama"

    def test_raises_when_requested_model_not_found(self) -> None:
        dataset = Dataset.from_list(
            [
                {"model_metrics": {"llama": {"value": 1}}},
            ]
        )

        with pytest.raises(RuntimeError, match="Model 'gpt4' not found"):
            resolve_model_name(dataset, "gpt4", Path("/fake"))

    def test_infers_single_model_automatically(self) -> None:
        dataset = Dataset.from_list(
            [
                {"model_metrics": {"llama": {"value": 1}}},
                {"model_metrics": {"llama": {"value": 2}}},
            ]
        )

        model = resolve_model_name(dataset, None, Path("/fake"))
        assert model == "llama"

    def test_raises_when_multiple_models_and_none_requested(self) -> None:
        dataset = Dataset.from_list(
            [
                {"model_metrics": {"llama": {"value": 1}}},
                {"model_metrics": {"gpt4": {"value": 2}}},
            ]
        )

        with pytest.raises(RuntimeError, match="multiple models"):
            resolve_model_name(dataset, None, Path("/fake"))

    def test_raises_when_no_models_found(self) -> None:
        dataset = Dataset.from_list(
            [
                {"other_field": "value"},
            ]
        )

        with pytest.raises(RuntimeError, match="Could not infer model name"):
            resolve_model_name(dataset, None, Path("/fake"))

    def test_handles_empty_model_metrics(self) -> None:
        dataset = Dataset.from_list(
            [
                {"model_metrics": {}},
                {"model_metrics": {"llama": {"value": 1}}},
            ]
        )

        model = resolve_model_name(dataset, None, Path("/fake"))
        assert model == "llama"

    def test_preserves_model_order(self) -> None:
        # First non-empty model should be returned when single model
        dataset = Dataset.from_list(
            [
                {"model_metrics": {}},
                {"model_metrics": {"alpha": {"value": 1}}},
                {"model_metrics": {"alpha": {"value": 2}}},
            ]
        )

        model = resolve_model_name(dataset, None, Path("/fake"))
        assert model == "alpha"


class TestIterModelEntries:
    """Test model entry iteration."""

    def test_yields_entries_for_matching_model(self) -> None:
        dataset = Dataset.from_list(
            [
                {"model_metrics": {"llama": {"tokens": 100}}},
                {"model_metrics": {"llama": {"tokens": 200}}},
            ]
        )

        entries = list(iter_model_entries(dataset, "llama"))
        assert len(entries) == 2
        assert entries[0]["tokens"] == 100
        assert entries[1]["tokens"] == 200

    def test_skips_entries_without_model(self) -> None:
        dataset = Dataset.from_list(
            [
                {"model_metrics": {"llama": {"tokens": 100}}},
                {"model_metrics": {"gpt4": {"tokens": 200}}},
                {"model_metrics": {"llama": {"tokens": 300}}},
            ]
        )

        entries = list(iter_model_entries(dataset, "llama"))
        assert len(entries) == 2
        assert entries[0]["tokens"] == 100
        assert entries[1]["tokens"] == 300

    def test_skips_records_without_model_metrics(self) -> None:
        dataset = Dataset.from_list(
            [
                {"model_metrics": None, "other_field": "value"},
                {"model_metrics": {"llama": {"tokens": 100}}, "other_field": None},
            ]
        )

        entries = list(iter_model_entries(dataset, "llama"))
        assert len(entries) == 1

    def test_skips_non_mapping_model_metrics(self) -> None:
        # Note: HuggingFace Dataset will normalize types, so we can't mix string and dict
        # Instead, test with None and dict
        dataset = Dataset.from_list(
            [
                {"model_metrics": None},
                {"model_metrics": {"llama": {"tokens": 100}}},
            ]
        )

        entries = list(iter_model_entries(dataset, "llama"))
        assert len(entries) == 1

    def test_returns_empty_when_no_matches(self) -> None:
        dataset = Dataset.from_list(
            [
                {"model_metrics": {"gpt4": {"tokens": 100}}},
            ]
        )

        entries = list(iter_model_entries(dataset, "llama"))
        assert len(entries) == 0


class TestCollectRunMetadata:
    """Test metadata collection."""

    def test_extracts_system_info_from_first_entry(self) -> None:
        entries = [
            {"system_info": {"cpu": "Intel"}, "gpu_info": {}},
            {"system_info": {"cpu": "AMD"}, "gpu_info": {}},
        ]

        system_info, _ = collect_run_metadata(entries)
        assert system_info == {"cpu": "Intel"}

    def test_extracts_gpu_info_from_first_entry(self) -> None:
        entries = [
            {"system_info": {}, "gpu_info": {"name": "RTX 3090"}},
            {"system_info": {}, "gpu_info": {"name": "RTX 4090"}},
        ]

        _, gpu_info = collect_run_metadata(entries)
        assert gpu_info == {"name": "RTX 3090"}

    def test_returns_empty_when_no_metadata(self) -> None:
        entries = [
            {"other_field": "value"},
        ]

        system_info, gpu_info = collect_run_metadata(entries)
        assert system_info == {}
        assert gpu_info == {}

    def test_skips_non_mapping_metadata(self) -> None:
        entries = [
            {"system_info": "not a dict", "gpu_info": "not a dict"},
            {"system_info": {"cpu": "Intel"}, "gpu_info": {"name": "RTX 3090"}},
        ]

        system_info, gpu_info = collect_run_metadata(entries)
        assert system_info == {"cpu": "Intel"}
        assert gpu_info == {"name": "RTX 3090"}

    def test_stops_early_when_both_found(self) -> None:
        # Create a mock iterator to verify early stopping
        def make_entries():
            yield {"system_info": {"cpu": "Intel"}, "gpu_info": {"name": "RTX"}}
            # This shouldn't be accessed
            raise RuntimeError("Should not iterate past first complete entry")

        system_info, gpu_info = collect_run_metadata(make_entries())
        assert system_info == {"cpu": "Intel"}
        assert gpu_info == {"name": "RTX"}

    def test_continues_until_both_found(self) -> None:
        entries = [
            {"system_info": {"cpu": "Intel"}},
            {"gpu_info": {"name": "RTX"}},
        ]

        system_info, gpu_info = collect_run_metadata(entries)
        assert system_info == {"cpu": "Intel"}
        assert gpu_info == {"name": "RTX"}
