"""Inference client implementations.

Clients register themselves with ``mindi.core.ClientRegistry``.
"""

from .ollama import OllamaClient

__all__ = ["OllamaClient"]
