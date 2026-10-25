from __future__ import annotations

from ipj.registry.check import Verdict
from ipj.registry.cli import spec_matches_block


def test_exact_blocks_unless_forced():
    v = Verdict("exact", [{"energy_sig": "e1"}])
    assert spec_matches_block(v, force=False, allow_dup=False) is True
    assert spec_matches_block(v, force=True, allow_dup=False) is False
    assert spec_matches_block(v, force=False, allow_dup=True) is False


def test_non_exact_never_blocks():
    for tier in ("accuracy", "near", "none"):
        assert spec_matches_block(Verdict(tier, []), force=False, allow_dup=False) is False
