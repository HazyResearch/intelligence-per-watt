from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from ipw.core.types import DatasetRecord
from ipw.datasets.browsecomp import BrowseCompDataset, _decrypt, _derive_key


def _encrypt(plain: str, password: str) -> str:
    import base64

    data = plain.encode()
    key = _derive_key(password, len(data))
    return base64.b64encode(bytes(a ^ b for a, b in zip(data, key))).decode()


class TestLiveCodeBenchDataset:
    @patch("ipw.datasets.livecodebench.load_dataset")
    def test_convert_and_score_executes_tests(self, mock_load_dataset: MagicMock) -> None:
        from ipw.datasets.livecodebench import LiveCodeBenchDataset

        mock_dataset = MagicMock()
        mock_dataset.to_list.return_value = [
            {
                "question_id": "p1",
                "question_content": "Read an integer and print it plus one.",
                "public_test_cases": [{"input": "1\n", "output": "2"}],
                "hidden_test_cases": [{"input": "41\n", "output": "42"}],
            }
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = LiveCodeBenchDataset()
        record = list(dataset.iter_records())[0]
        assert isinstance(record, DatasetRecord)
        assert record.dataset_metadata["test_inputs"] == ["1", "41"]

        ok, meta = dataset.score(
            record,
            "```python\nx=int(input())\nprint(x+1)\n```",
        )
        assert ok is True
        assert meta["tests_passed"] == 2

    def test_extracts_code_from_shell_heredoc_transcript(self) -> None:
        from ipw.datasets.livecodebench import _extract_code

        transcript = """root@box:/workspace# cat << 'EOF' > solution.py
> import sys
> x = int(sys.stdin.readline())
> print(x + 1)
> EOF
root@box:/workspace# python solution.py
"""
        code = _extract_code(transcript)
        assert code == "import sys\nx = int(sys.stdin.readline())\nprint(x + 1)"

    def test_extracts_code_from_noisy_terminus_heredoc(self) -> None:
        from ipw.datasets.livecodebench import _extract_code

        transcript = """root@box:/# cat << 'EOF' > solution.py
> import sys
xe>
> x = int(sys.stdin.readline())
y>
st>         return
> print(x + 1)
> EOF
"""
        code = _extract_code(transcript)
        assert code == "import sys\n\nx = int(sys.stdin.readline())\n\n        return\nprint(x + 1)"


class TestArenaHardAutoV2Dataset:
    def test_loads_question_jsonl(self, tmp_path: Path) -> None:
        from ipw.datasets.arena_hard_auto import ArenaHardAutoV2Dataset

        path = tmp_path / "question.jsonl"
        path.write_text(
            json.dumps({"question_id": 7, "category": "math", "turns": ["Solve 2+2"]}) + "\n",
            encoding="utf-8",
        )
        dataset = ArenaHardAutoV2Dataset(question_path=str(path))
        record = list(dataset.iter_records())[0]
        assert "Solve 2+2" in record.problem
        assert record.dataset_metadata["unscorable_reason"] == "missing_reference_answer"

    def test_loads_uid_and_reference_answer(self, tmp_path: Path) -> None:
        from ipw.datasets.arena_hard_auto import ArenaHardAutoV2Dataset

        question_path = tmp_path / "question.jsonl"
        question_path.write_text(
            json.dumps({"uid": "arena-1", "category": "math", "turns": ["Solve 2+2"]}) + "\n",
            encoding="utf-8",
        )
        reference_path = tmp_path / "reference.jsonl"
        reference_path.write_text(
            json.dumps({"uid": "arena-1", "answer": "4"}) + "\n",
            encoding="utf-8",
        )

        dataset = ArenaHardAutoV2Dataset(
            question_path=str(question_path),
            reference_answer_path=str(reference_path),
        )
        record = list(dataset.iter_records())[0]
        assert record.dataset_metadata["question_id"] == "arena-1"
        assert record.answer == "4"
        assert "unscorable_reason" not in record.dataset_metadata

    def test_verdict_parser_rejects_instruction_template(self) -> None:
        from ipw.datasets.arena_hard_auto import _parse_verdict

        assert _parse_verdict("verdict: candidate") == "candidate"
        assert _parse_verdict("verdict: candidate|reference|tie") is None


class TestBrowseCompDataset:
    def test_decrypt_roundtrip(self) -> None:
        encrypted = _encrypt("secret answer", "canary")
        assert _decrypt(encrypted, "canary") == "secret answer"

    def test_loads_local_csv(self, tmp_path: Path) -> None:
        path = tmp_path / "browsecomp.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["canary", "problem", "answer"])
            writer.writeheader()
            writer.writerow(
                {
                    "canary": "pw",
                    "problem": _encrypt("What is hidden?", "pw"),
                    "answer": _encrypt("the answer", "pw"),
                }
            )
        dataset = BrowseCompDataset(csv_path=str(path))
        record = list(dataset.iter_records())[0]
        assert "What is hidden?" in record.problem
        assert record.answer == "the answer"

    @patch("ipw.datasets.browsecomp.EvaluationRegistry.create")
    @patch("ipw.datasets.browsecomp.ClientRegistry.create")
    def test_score_uses_default_eval_config(
        self,
        mock_client_create: MagicMock,
        mock_eval_create: MagicMock,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "browsecomp.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["canary", "problem", "answer"])
            writer.writeheader()
            writer.writerow(
                {
                    "canary": "pw",
                    "problem": _encrypt("What is hidden?", "pw"),
                    "answer": _encrypt("the answer", "pw"),
                }
            )
        handler = MagicMock()
        handler.evaluate.return_value = (True, {"match_type": "judge"})
        mock_client_create.return_value = object()
        mock_eval_create.return_value = handler

        dataset = BrowseCompDataset(csv_path=str(path))
        record = list(dataset.iter_records())[0]
        ok, meta = dataset.score(record, "candidate answer")

        assert ok is True
        assert meta["match_type"] == "judge"
        mock_client_create.assert_called_once_with(
            "openai",
            base_url="https://api.openai.com/v1",
            model="gpt-5-nano-2025-08-07",
        )


class TestResearchDatasets:
    def test_deepresearchbench_local_jsonl(self, tmp_path: Path) -> None:
        from ipw.datasets.deepresearchbench import DeepResearchBenchDataset

        report = tmp_path / "reports.jsonl"
        report.write_text(
            json.dumps({"id": 1, "prompt": "Research batteries", "article": "Reference report"}) + "\n",
            encoding="utf-8",
        )
        dataset = DeepResearchBenchDataset(data_dir=str(tmp_path), reference_report="reports.jsonl")
        record = list(dataset.iter_records())[0]
        assert "Research batteries" in record.problem
        assert record.answer == "Reference report"
        assert dataset.eval_client is None
        assert dataset.eval_base_url is None
        assert dataset.eval_model is None

    def test_liveresearchbench_local_repo_shape(self, tmp_path: Path) -> None:
        from ipw.datasets.liveresearchbench import LiveResearchBenchDataset

        query_dir = tmp_path / "data" / "prompt_data"
        query_dir.mkdir(parents=True)
        (query_dir / "query.jsonl").write_text(
            json.dumps({"id": 3, "topic": "AI", "language": "en", "prompt": "Research AI"}) + "\n",
            encoding="utf-8",
        )
        dataset = LiveResearchBenchDataset(path=str(tmp_path))
        record = list(dataset.iter_records())[0]
        assert "Research AI" in record.problem
        assert record.answer == "__research_report_rubric_judge__"
        assert dataset.eval_client is None
        assert dataset.eval_base_url is None
        assert dataset.eval_model is None
