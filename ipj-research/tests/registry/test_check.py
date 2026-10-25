from __future__ import annotations

from ipj.registry.check import check_spec


def _row(rec, **over):
    d = rec.to_dict()
    d.update(over)
    return d


def test_exact_match_when_energy_sig_present(make_rec):
    rec = make_rec()
    rec.accuracy_sig = "a1"
    rec.energy_sig = "e1"
    index = [_row(rec, accuracy_sig="a1", energy_sig="e1")]
    v = check_spec(rec, index)
    assert v.tier == "exact"
    assert len(v.matches) == 1


def test_accuracy_only_when_energy_differs(make_rec):
    rec = make_rec()
    rec.accuracy_sig = "a1"
    rec.energy_sig = "e1"
    index = [_row(rec, accuracy_sig="a1", energy_sig="e2")]
    v = check_spec(rec, index)
    assert v.tier == "accuracy"


def test_none_when_no_overlap(make_rec):
    rec = make_rec()
    rec.accuracy_sig = "a1"
    rec.energy_sig = "e1"
    # Create index row with different model to ensure no "near" match
    index = [_row(rec, accuracy_sig="zzz", energy_sig="yyy", model={"id": "different-model", "hf_revision": None, "quant": None})]
    assert check_spec(rec, index).tier == "none"


def test_dirty_and_backfilled_rows_are_not_authoritative(make_rec):
    rec = make_rec()
    rec.accuracy_sig = "a1"
    rec.energy_sig = "e1"
    index = [
        _row(rec, accuracy_sig="a1", energy_sig="e1", git_dirty=True),
        _row(rec, accuracy_sig="a1", energy_sig="e1", status="backfilled"),
    ]
    assert check_spec(rec, index).tier == "none"


def test_partial_row_is_not_authoritative(make_rec):
    """A row with status='partial' must not count as an authoritative match."""
    rec = make_rec()
    rec.accuracy_sig = "a1"
    rec.energy_sig = "e1"
    index = [_row(rec, accuracy_sig="a1", energy_sig="e1", status="partial")]
    assert check_spec(rec, index).tier == "none"


def test_exact_verdict_message_contains_sampling_caveat(make_rec):
    """Exact-tier message must warn that sampling/serve config is not captured."""
    rec = make_rec()
    rec.accuracy_sig = "a1"
    rec.energy_sig = "e1"
    index = [_row(rec, accuracy_sig="a1", energy_sig="e1")]
    v = check_spec(rec, index)
    assert v.tier == "exact"
    assert "not captured" in v.message.lower()


def test_accuracy_verdict_message_contains_sampling_caveat(make_rec):
    """Accuracy-tier message must also warn that sampling/serve config is not captured."""
    rec = make_rec()
    rec.accuracy_sig = "a1"
    rec.energy_sig = "e1"
    index = [_row(rec, accuracy_sig="a1", energy_sig="e2")]
    v = check_spec(rec, index)
    assert v.tier == "accuracy"
    assert "not captured" in v.message.lower()
