from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Protocol, runtime_checkable

from ipj.orchestrate.harness.prompt import render_message  # noqa: F401  (re-export: one render source)

IGNORE_INDEX = -100  # HF label-ignore sentinel: tokens with this label do not contribute to loss.


def build_labels(spans: list[tuple[list[int], bool]]) -> tuple[list[int], list[int]]:
    """Concatenate span token-ids into input_ids; label each span with its ids if loss else IGNORE_INDEX."""
    input_ids: list[int] = []
    labels: list[int] = []
    for token_ids, loss in spans:
        input_ids.extend(token_ids)
        labels.extend(token_ids if loss else [IGNORE_INDEX] * len(token_ids))
    return input_ids, labels


@runtime_checkable
class Tokenizerish(Protocol):
    """Minimal tokenizer surface dataprep needs — satisfied by a transformers tokenizer and by a test fake."""

    pad_token_id: int

    def encode(self, text: str) -> list[int]: ...


def tokenize_example(messages: list[dict], tok: Tokenizerish) -> dict[str, list[int]]:
    """Render+encode each message, pair with its loss flag, and build masked labels."""
    spans = [(tok.encode(render_message(m)), bool(m.get("loss", False))) for m in messages]
    input_ids, labels = build_labels(spans)
    return {"input_ids": input_ids, "labels": labels}


logger = logging.getLogger(__name__)


def load_examples(path: str | Path, tok: Tokenizerish, *, max_seq_len: int) -> list[dict]:
    """Read sft.jsonl, tokenize each line, drop over-length examples; skip malformed lines (all logged).

    A single bad line (invalid JSON, missing ``messages``, or a message missing ``role``/``content``)
    is skipped with a warning rather than aborting the whole load — matching the Step-3 loader's policy.
    """
    examples: list[dict] = []
    dropped = 0
    skipped = 0
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ex = tokenize_example(json.loads(line)["messages"], tok)
        except (json.JSONDecodeError, KeyError, TypeError):
            skipped += 1
            continue
        if len(ex["input_ids"]) > max_seq_len:
            dropped += 1
            continue
        examples.append(ex)
    if dropped:
        logger.warning("dropped %d over-length examples (> %d tokens); kept %d", dropped, max_seq_len, len(examples))
    if skipped:
        logger.warning("skipped %d malformed lines (bad JSON or missing 'messages'/'role'/'content')", skipped)
    return examples


def split_examples(examples: list[dict], *, val_frac: float = 0.1, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Deterministic shuffle (by seed) + train/val split. Holds out >=1 val example, but never empties
    the train set: with <2 examples (or val_frac<=0) there is no val split (all go to train)."""
    n = len(examples)
    if n < 2 or val_frac <= 0:
        return list(examples), []
    order = list(range(n))
    random.Random(seed).shuffle(order)
    n_val = min(n - 1, max(1, round(n * val_frac)))  # 1 <= n_val <= n-1, so both splits are non-empty
    val_pos = set(order[:n_val])
    train = [examples[i] for i in range(n) if i not in val_pos]
    val = [examples[i] for i in range(n) if i in val_pos]
    return train, val
