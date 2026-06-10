"""Tests for datasets/swebench.py — SWEBenchDataset."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
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

    def test_score_applies_patch_and_runs_tests(self, tmp_path: Path) -> None:
        from ipw.datasets.swebench import SWEBenchDataset

        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE)
        (repo / "calc.py").write_text("def add_one(x):\n    return x\n", encoding="utf-8")
        subprocess.run(["git", "add", "calc.py"], cwd=repo, check=True)

        record = DatasetRecord(
            problem="Fix add_one",
            answer="",
            subject="local/repo",
            dataset_metadata={
                "instance_id": "local__repo-1",
                "workspace_path": str(repo),
                "test_cmd": (
                    f"{sys.executable} -c "
                    "'import calc; assert calc.add_one(1) == 2'"
                ),
            },
        )
        patch = """diff --git a/calc.py b/calc.py
--- a/calc.py
+++ b/calc.py
@@ -1,2 +1,2 @@
 def add_one(x):
-    return x
+    return x + 1
"""

        dataset = object.__new__(SWEBenchDataset)
        ok, meta = dataset.score(record, patch)

        assert ok is True
        assert meta["match_type"] == "test_execution"

    def test_score_rejects_missing_patch_and_workspace_diff(self, tmp_path: Path) -> None:
        from ipw.datasets.swebench import SWEBenchDataset

        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE)
        record = DatasetRecord(
            problem="Fix bug",
            answer="",
            subject="local/repo",
            dataset_metadata={"instance_id": "local__repo-2", "workspace_path": str(repo)},
        )

        dataset = object.__new__(SWEBenchDataset)
        ok, meta = dataset.score(record, "No patch here")

        assert ok is False
        assert meta["reason"] == "no_patch_or_workspace_diff"
