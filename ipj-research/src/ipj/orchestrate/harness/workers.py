"""Worker pool: capability cards, provider bindings, and per-trace anonymized sampling."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class WorkerCard:
    """Anonymized capability card the brain sees: alias + capability text + price/latency/energy estimates (no model id)."""

    alias: str
    capability: str
    price_in: float
    price_out: float
    est_latency_s: float
    est_energy_j_per_ktok: float | None = None


@dataclass(frozen=True, slots=True)
class WorkerBinding:
    """Private provider wiring for an alias (provider, model_id, base_url, is_local) — never shown to the brain."""

    alias: str
    provider: str
    model_id: str
    base_url: str | None = None
    is_local: bool = False


@dataclass(slots=True)
class WorkerPool:
    """A set of cards + bindings, with sampling for per-trace pool randomization."""

    cards: dict[str, WorkerCard]
    bindings: dict[str, WorkerBinding]

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> "WorkerPool":
        cards: dict[str, WorkerCard] = {}
        bindings: dict[str, WorkerBinding] = {}
        for alias, w in spec["workers"].items():
            cards[alias] = WorkerCard(
                alias=alias,
                capability=w["capability"],
                price_in=float(w["price_in"]),
                price_out=float(w["price_out"]),
                est_latency_s=float(w["est_latency_s"]),
                est_energy_j_per_ktok=w.get("est_energy_j_per_ktok"),
            )
            bindings[alias] = WorkerBinding(
                alias=alias,
                provider=w["provider"],
                model_id=w["model_id"],
                base_url=w.get("base_url"),
                is_local=bool(w.get("is_local", False)),
            )
        return cls(cards=cards, bindings=bindings)

    @classmethod
    def from_json(cls, path: str | Path) -> "WorkerPool":
        return cls.from_spec(json.loads(Path(path).read_text()))

    def card_catalog(self) -> list[WorkerCard]:
        return [self.cards[a] for a in sorted(self.cards)]

    def binding(self, alias: str) -> WorkerBinding:
        if alias not in self.bindings:
            raise KeyError(f"unknown worker alias {alias!r}; available: {sorted(self.bindings)}")
        return self.bindings[alias]

    def sample(self, rng: random.Random, *, min_aliases: int = 2, price_jitter: float = 0.2) -> "WorkerPool":
        """Return an anonymized, price-jittered random subset of the pool for per-trace card randomization."""
        aliases = sorted(self.cards)
        k = rng.randint(min(min_aliases, len(aliases)), len(aliases))
        chosen = sorted(rng.sample(aliases, k))
        cards: dict[str, WorkerCard] = {}
        bindings: dict[str, WorkerBinding] = {}
        for al in chosen:
            base = self.cards[al]
            jin = 1.0 + rng.uniform(-price_jitter, price_jitter)
            jout = 1.0 + rng.uniform(-price_jitter, price_jitter)
            cards[al] = WorkerCard(
                alias=al,
                capability=base.capability,
                price_in=round(base.price_in * jin, 6),
                price_out=round(base.price_out * jout, 6),
                est_latency_s=base.est_latency_s,
                est_energy_j_per_ktok=base.est_energy_j_per_ktok,
            )
            bindings[al] = self.bindings[al]
        return WorkerPool(cards=cards, bindings=bindings)

    def snapshot(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for al, card in self.cards.items():
            b = self.bindings[al]
            out[al] = {
                "capability": card.capability,
                "price_in": card.price_in,
                "price_out": card.price_out,
                "provider": b.provider,
                "model_id": b.model_id,
                "is_local": b.is_local,
            }
        return out


DEFAULT_POOL_SPEC: dict[str, Any] = {
    "workers": {
        "local-fast": {
            "capability": "Fast small local model. Good at routine bash, file ops, simple edits; "
            "weaker on hard multi-step debugging.",
            "price_in": 0.0,
            "price_out": 0.0,
            "est_latency_s": 2.0,
            "est_energy_j_per_ktok": 8.0,
            "provider": "vllm",
            "model_id": "Qwen/Qwen3.5-9B",
            "base_url": "http://localhost:8000",
            "is_local": True,
        },
        "open-large": {
            "capability": "Large open model. Strong coding/reasoning; higher latency and energy.",
            "price_in": 0.385,
            "price_out": 2.45,
            "est_latency_s": 8.0,
            "est_energy_j_per_ktok": 40.0,
            "provider": "openrouter",
            "model_id": "qwen/qwen3.5-397b-a17b",
            "is_local": False,
        },
        "cloud-mini": {
            "capability": "Cheap fast cloud model. Solid general competence at low price; "
            "not the strongest on the hardest tasks.",
            "price_in": 0.25,
            "price_out": 2.0,
            "est_latency_s": 6.0,
            "provider": "openai",
            "model_id": "gpt-5.4-mini",
            "is_local": False,
        },
        "cloud-frontier": {
            "capability": "Top-tier frontier model. Best accuracy on the hardest tasks; most expensive.",
            "price_in": 5.0,
            "price_out": 25.0,
            "est_latency_s": 20.0,
            "provider": "anthropic",
            "model_id": "claude-opus-4-8",
            "is_local": False,
        },
    }
}
