from __future__ import annotations

from ipj.orchestrate.harness.prompt import render_message


def test_render_message_role_tagged():
    assert render_message({"role": "assistant", "content": "X"}) == "<|assistant|>\nX\n"


def test_train_dataprep_reexports_same_function():
    from ipj.train.dataprep import render_message as dp_render
    assert dp_render is render_message  # ONE source of truth for inference parity
