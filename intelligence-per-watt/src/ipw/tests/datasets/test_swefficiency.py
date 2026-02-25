"""Tests for datasets/swefficiency.py — SWEfficiencyDataset."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ipw.core.types import DatasetRecord


class TestSWEfficiencyDataset:
    """Test SWEfficiencyDataset with mocked HuggingFace loading."""

    @patch("ipw.datasets.swefficiency.load_dataset")
    def test_iter_records_yields_dataset_records(
        self, mock_load_dataset: MagicMock
    ) -> None:
        from ipw.datasets.swefficiency import SWEfficiencyDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [
            {
                "instance_id": "swe_eff_001",
                "repo": "numpy/numpy",
                "problem_statement": "Optimize array operations",
                "workload": "matrix multiply benchmark",
                "speedup": 2.5,
                "patch": "diff content",
                "covering_tests": '["test_perf"]',
            },
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = SWEfficiencyDataset()
        records = list(dataset.iter_records())

        assert len(records) == 1
        assert isinstance(records[0], DatasetRecord)
        assert "numpy/numpy" in records[0].problem
        assert "Optimize array operations" in records[0].problem
        assert records[0].subject == "numpy/numpy"

    @patch("ipw.datasets.swefficiency.load_dataset")
    def test_size(self, mock_load_dataset: MagicMock) -> None:
        from ipw.datasets.swefficiency import SWEfficiencyDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [
            {"instance_id": f"id_{i}", "repo": "r", "problem_statement": f"P{i}"}
            for i in range(3)
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = SWEfficiencyDataset()
        assert dataset.size() == 3

    @patch("ipw.datasets.swefficiency.load_dataset")
    def test_metadata_fields(self, mock_load_dataset: MagicMock) -> None:
        from ipw.datasets.swefficiency import SWEfficiencyDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [
            {
                "instance_id": "swe_eff_001",
                "repo": "proj/proj",
                "problem_statement": "Optimize",
                "speedup": 3.0,
                "workload": "benchmark",
                "patch": "diff",
            },
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = SWEfficiencyDataset()
        record = list(dataset.iter_records())[0]
        meta = record.dataset_metadata
        assert meta["dataset_name"] == "SWEfficiency"
        assert meta["instance_id"] == "swe_eff_001"
        assert meta["speedup"] == 3.0

    @patch("ipw.datasets.swefficiency.load_dataset")
    def test_speedup_in_prompt(self, mock_load_dataset: MagicMock) -> None:
        from ipw.datasets.swefficiency import SWEfficiencyDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [
            {
                "instance_id": "id1",
                "repo": "r",
                "problem_statement": "P",
                "speedup": 2.5,
                "workload": "W",
            },
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = SWEfficiencyDataset()
        record = list(dataset.iter_records())[0]
        assert "2.5x" in record.problem

    @patch("ipw.datasets.swefficiency.load_dataset")
    def test_skips_empty_records(self, mock_load_dataset: MagicMock) -> None:
        from ipw.datasets.swefficiency import SWEfficiencyDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [
            {"instance_id": "", "repo": "r", "problem_statement": "P"},
            {"instance_id": "valid", "repo": "r", "problem_statement": ""},
            {"instance_id": "good", "repo": "r", "problem_statement": "Good problem"},
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = SWEfficiencyDataset()
        assert dataset.size() == 1
