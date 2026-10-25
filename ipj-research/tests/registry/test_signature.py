from __future__ import annotations

import dataclasses

from ipj.registry.record import HardwareSpec, SamplingSpec
from ipj.registry.signature import accuracy_signature, energy_signature


def test_same_config_same_signature(make_rec):
    assert accuracy_signature(make_rec()) == accuracy_signature(make_rec())


def test_non_semantic_fields_do_not_change_accuracy_sig(make_rec):
    a = make_rec()
    b = make_rec()
    b.host = "other-box"
    b.run_dir = "/somewhere/else"
    b.created_at = "2099-01-01T00:00:00"
    b.metrics = dataclasses.replace(b.metrics, accuracy=0.01)
    assert accuracy_signature(a) == accuracy_signature(b)


def test_sampling_change_changes_accuracy_sig(make_rec):
    a = make_rec()
    b = make_rec()
    b.sampling = SamplingSpec(temperature=0.2)
    assert accuracy_signature(a) != accuracy_signature(b)


def test_hardware_change_keeps_accuracy_sig_but_changes_energy_sig(make_rec):
    a = make_rec()
    b = make_rec()
    b.hardware = HardwareSpec(gpu="MI350X", count=8, tp=8)
    assert accuracy_signature(a) == accuracy_signature(b)
    assert energy_signature(a) != energy_signature(b)


def test_energy_sig_embeds_accuracy_sig(make_rec):
    a = make_rec()
    b = make_rec()
    b.sampling = SamplingSpec(temperature=0.2)  # changes accuracy inputs
    assert energy_signature(a) != energy_signature(b)
