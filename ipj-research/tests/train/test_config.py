from __future__ import annotations

import pytest

from ipj.train import config


def test_presets_use_only_allowed_families():
    assert config.pilot().base_model == "Qwen/Qwen3.5-0.8B"
    assert config.pilot_gemma().base_model == "google/gemma-4-E2B-it"
    assert config.real().base_model == "Qwen/Qwen3.5-27B"
    for make in config.PRESETS.values():
        c = make()
        assert c.base_model.startswith(("Qwen/Qwen3.5", "google/gemma-4"))
        assert c.lora_r > 0 and c.max_seq_len > 0 and c.epochs > 0


def test_build_config_applies_overrides_and_ignores_none():
    c = config.build_config("pilot", data_path="runs/x/sft.jsonl", out_dir=None)
    assert c.base_model == "Qwen/Qwen3.5-0.8B"
    assert c.data_path == "runs/x/sft.jsonl"
    assert c.out_dir == config.pilot().out_dir  # None override left the default


def test_build_config_rejects_unknown_preset():
    with pytest.raises(SystemExit):
        config.build_config("nope")


def test_gradient_checkpointing_default_off_and_settable():
    assert config.pilot().gradient_checkpointing is False
    assert config.build_config("pilot", gradient_checkpointing=True).gradient_checkpointing is True
