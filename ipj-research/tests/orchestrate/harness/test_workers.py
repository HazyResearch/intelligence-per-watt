from __future__ import annotations

import json
import random

from ipj.orchestrate.harness.workers import DEFAULT_POOL_SPEC, WorkerPool


def _pool():
    return WorkerPool.from_spec(DEFAULT_POOL_SPEC)


def test_card_catalog_hides_real_model_ids():
    pool = _pool()
    catalog = pool.card_catalog()
    blob = json.dumps([c.__dict__ if hasattr(c, "__dict__") else c.__getstate__() for c in catalog], default=str)
    for binding in pool.bindings.values():
        assert binding.model_id not in blob  # anonymization invariant
    assert {c.alias for c in catalog} == set(pool.bindings)


def test_binding_lookup_and_local_flag():
    pool = _pool()
    assert pool.binding("local-fast").is_local is True
    assert pool.binding("cloud-frontier").is_local is False


def test_sample_is_deterministic_and_perturbs_prices():
    pool = _pool()
    a = pool.sample(random.Random(7))
    b = pool.sample(random.Random(7))
    assert set(a.cards) == set(b.cards)  # same seed -> same subset
    assert len(a.cards) >= 2
    # at least one price moved vs the unperturbed pool
    moved = any(a.cards[al].price_in != pool.cards[al].price_in for al in a.cards)
    assert moved


def test_snapshot_roundtrip_contains_binding_and_card():
    snap = _pool().snapshot()
    assert "cloud-frontier" in snap
    assert "model_id" in snap["cloud-frontier"] and "capability" in snap["cloud-frontier"]


def test_from_json(tmp_path):
    p = tmp_path / "pool.json"
    p.write_text(json.dumps(DEFAULT_POOL_SPEC))
    pool = WorkerPool.from_json(p)
    assert set(pool.cards) == set(DEFAULT_POOL_SPEC["workers"])
