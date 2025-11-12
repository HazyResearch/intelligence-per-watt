"""Inference client implementations.

Clients register themselves with ``ipw.core.ClientRegistry``.
"""

from .base import InferenceClient


def ensure_registered() -> None:
    """Import built-in client implementations to populate the registry."""
    from . import ollama  # noqa: F401
    from . import vllm  # noqa: F401


__all__ = ["InferenceClient", "ensure_registered"]
