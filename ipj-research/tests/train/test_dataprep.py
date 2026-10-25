from __future__ import annotations

import sys
from pathlib import Path

from ipj.train.dataprep import (
    IGNORE_INDEX,
    build_labels,
    load_examples,
    render_message,
    split_examples,
    tokenize_example,
)

FIXTURE = Path(__file__).parent / "fixtures" / "dummy_sft.jsonl"


def test_build_labels_masks_non_loss_spans():
    spans = [([1, 2], False), ([3, 4, 5], True), ([6], False)]
    input_ids, labels = build_labels(spans)
    assert input_ids == [1, 2, 3, 4, 5, 6]
    assert labels == [IGNORE_INDEX, IGNORE_INDEX, 3, 4, 5, IGNORE_INDEX]
    assert len(labels) == len(input_ids)


def test_build_labels_all_masked_when_no_loss_span():
    input_ids, labels = build_labels([([7, 8, 9], False)])
    assert input_ids == [7, 8, 9]
    assert labels == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]


class FakeTokenizer:
    """Char-code tokenizer: deterministic, torch-free, lets CI test the masking end-to-end."""

    pad_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]


def test_tokenize_example_masks_assistant_only():
    messages = [
        {"role": "system", "content": "S", "loss": False},
        {"role": "user", "content": "U", "loss": False},
        {"role": "assistant", "content": "A", "loss": True},
    ]
    out = tokenize_example(messages, FakeTokenizer())
    assert out["input_ids"] == [ord(c) for m in messages for c in render_message(m)]
    assert len(out["labels"]) == len(out["input_ids"])
    ctx_len = len(render_message(messages[0])) + len(render_message(messages[1]))
    assert out["labels"][:ctx_len] == [-100] * ctx_len
    assert out["labels"][ctx_len:] == [ord(c) for c in render_message(messages[2])]


def test_load_examples_reads_all_when_under_limit():
    examples = load_examples(FIXTURE, FakeTokenizer(), max_seq_len=10_000)
    assert len(examples) == 4
    assert all("input_ids" in e and "labels" in e for e in examples)


def test_load_examples_drops_over_length(tmp_path):
    p = tmp_path / "mix.jsonl"
    short = '{"messages": [{"role": "assistant", "content": "x", "loss": true}]}'
    long = '{"messages": [{"role": "assistant", "content": "%s", "loss": true}]}' % ("y" * 500)
    p.write_text(short + "\n" + long + "\n")
    examples = load_examples(p, FakeTokenizer(), max_seq_len=50)
    assert len(examples) == 1  # the 500-char line is dropped


def test_split_examples_is_deterministic_and_sized():
    examples = [{"input_ids": [i], "labels": [i]} for i in range(10)]
    t1, v1 = split_examples(examples, val_frac=0.2, seed=0)
    t2, v2 = split_examples(examples, val_frac=0.2, seed=0)
    assert (t1, v1) == (t2, v2)
    assert len(v1) == 2 and len(t1) == 8
    assert {tuple(e["input_ids"]) for e in t1 + v1} == {(i,) for i in range(10)}  # partition, no loss/dupe


def test_split_examples_single_example_stays_in_train():
    # With <2 examples there is no val hold-out — the lone example must stay trainable (not empty train).
    examples = [{"input_ids": [0], "labels": [0]}]
    train, val = split_examples(examples, val_frac=0.1, seed=0)
    assert train == examples and val == []


def test_split_examples_two_examples_one_each():
    examples = [{"input_ids": [0], "labels": [0]}, {"input_ids": [1], "labels": [1]}]
    train, val = split_examples(examples, val_frac=0.1, seed=0)
    assert len(train) == 1 and len(val) == 1  # never empties train even at a tiny val_frac


def test_load_examples_skips_malformed_lines(tmp_path):
    p = tmp_path / "dirty.jsonl"
    good = '{"messages": [{"role": "assistant", "content": "ok", "loss": true}]}'
    p.write_text("\n".join([good, "{not json", '{"metadata": {}}', good]) + "\n")
    examples = load_examples(p, FakeTokenizer(), max_seq_len=10_000)
    assert len(examples) == 2  # 2 good lines kept; bad-JSON and missing-'messages' lines skipped


def test_importing_train_module_does_not_import_torch():
    # Heavy deps must be lazy (inside functions) so the package import + dataprep tests stay light.
    sys.modules.pop("torch", None)
    import ipj.train.train  # noqa: F401

    assert "torch" not in sys.modules


def test_split_examples_val_frac_zero_keeps_all_in_train():
    examples = [{"input_ids": [i], "labels": [i]} for i in range(4)]
    train, val = split_examples(examples, val_frac=0.0)
    assert len(train) == 4 and val == []  # val_frac<=0 -> no held-out val (train on everything)
