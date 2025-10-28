"""Result sinks for profiling runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


class ResultSink:
    """Abstract append-only sink for profiler events."""

    def append(self, event: Mapping[str, object]) -> None:  # pragma: no cover - interface only
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - interface only
        raise NotImplementedError


@dataclass
class JsonlResultSink(ResultSink):
    """Simple JSONLines sink writing one event per line."""

    path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def append(self, event: Mapping[str, object]) -> None:
        json.dump(event, self._handle, ensure_ascii=False)
        self._handle.write("\n")
        self._handle.flush()

    def extend(self, events: Iterable[Mapping[str, object]]) -> None:
        for event in events:
            self.append(event)

    def close(self) -> None:
        self._handle.close()

