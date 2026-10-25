from __future__ import annotations

from ipj.registry.record import RunRecord


def test_to_dict_from_dict_roundtrip(make_rec):
    rec = make_rec()
    rebuilt = RunRecord.from_dict(rec.to_dict())
    assert rebuilt == rec


def test_to_dict_is_json_safe_and_nested(make_rec):
    d = make_rec().to_dict()
    assert d["model"]["id"] == "openai/Qwen3.6-27B"
    assert d["metrics"]["accuracy"] == 0.76
    assert d["status"] == "ok"
