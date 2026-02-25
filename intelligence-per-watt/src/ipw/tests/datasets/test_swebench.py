"""Tests for datasets/swebench.py — SWEBenchDataset."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ipw.core.types import DatasetRecord


class TestSWEBenchDataset:
    """Test SWEBenchDataset with mocked HuggingFace loading."""

    @patch("ipw.datasets.swebench.load_dataset")
    def test_iter_records_yields_dataset_records(
        self, mock_load_dataset: MagicMock
    ) -> None:
        from ipw.datasets.swebench import SWEBenchDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [
            {
                "instance_id": "django__django-12345",
                "repo": "django/django",
                "problem_statement": "Fix bug in QuerySet",
                "patch": "diff --git a/foo.py b/foo.py\n",
                "base_commit": "abc123",
                "FAIL_TO_PASS": '["test_query"]',
                "PASS_TO_PASS": '["test_other"]',
            },
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = SWEBenchDataset()
        records = list(dataset.iter_records())

        assert len(records) == 1
        assert isinstance(records[0], DatasetRecord)
        assert "Fix bug in QuerySet" in records[0].problem
        assert records[0].subject == "django/django"

    @patch("ipw.datasets.swebench.load_dataset")
    def test_size(self, mock_load_dataset: MagicMock) -> None:
        from ipw.datasets.swebench import SWEBenchDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [
            {"instance_id": f"id_{i}", "repo": "r", "problem_statement": f"P{i}"}
            for i in range(5)
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = SWEBenchDataset()
        assert dataset.size() == 5

    @patch("ipw.datasets.swebench.load_dataset")
    def test_metadata_fields(self, mock_load_dataset: MagicMock) -> None:
        from ipw.datasets.swebench import SWEBenchDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [
            {
                "instance_id": "proj__proj-999",
                "repo": "proj/proj",
                "problem_statement": "Problem",
                "patch": "diff content",
                "base_commit": "abc",
                "FAIL_TO_PASS": '["test1"]',
                "PASS_TO_PASS": '["test2"]',
            },
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = SWEBenchDataset()
        record = list(dataset.iter_records())[0]
        meta = record.dataset_metadata
        assert meta["dataset_name"] == "SWE-bench"
        assert meta["instance_id"] == "proj__proj-999"
        assert meta["repo"] == "proj/proj"
        assert meta["fail_to_pass"] == ["test1"]

    def test_unknown_variant_raises(self) -> None:
        from ipw.datasets.swebench import SWEBenchDataset

        with pytest.raises(ValueError, match="Unknown SWE-bench variant"):
            SWEBenchDataset(variant="nonexistent")

    @patch("ipw.datasets.swebench.load_dataset")
    def test_skips_records_without_instance_id(
        self, mock_load_dataset: MagicMock
    ) -> None:
        from ipw.datasets.swebench import SWEBenchDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [
            {"instance_id": "", "repo": "r", "problem_statement": "P"},
            {"instance_id": "valid_id", "repo": "r", "problem_statement": "Valid P"},
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = SWEBenchDataset()
        assert dataset.size() == 1
