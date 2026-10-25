"""Shared message->prompt rendering: the single source used by both SFT data prep and trained-brain inference."""

from __future__ import annotations


def render_message(message: dict) -> str:
    """Explicit, deterministic role-tagged render for one message.

    The one source of inference parity: ipj.train.dataprep (SFT tokenization) and the trained-brain
    inference client (trained_client) both render messages through THIS function.
    """
    return f"<|{message['role']}|>\n{message['content']}\n"
