"""Inference client implementations.

Clients register themselves with ``ipw.core.ClientRegistry``.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Dict

from ..core.registry import ClientRegistry
from .base import InferenceClient

MISSING_CLIENTS: Dict[str, str] = {}
_CLIENT_CLASS_MAP = (
    ("openai", "ipw.clients.openai", "OpenAIClient", None),
    ("openai-server", "ipw.clients.openai_server", "OpenAIServerClient", None),
    ("ollama", "ipw.clients.ollama", "OllamaClient", "ollama"),
    ("vllm", "ipw.clients.vllm", "VLLMClient", "vllm"),
)


def ensure_registered() -> None:
    """Import built-in client implementations to populate the registry."""
    for client_id, module_name, class_name, extra in _CLIENT_CLASS_MAP:
        if extra:
            MISSING_CLIENTS.pop(client_id, None)
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            missing_root = exc.name.split(".", 1)[0] if exc.name else None
            if extra is None or missing_root != extra:
                raise
            MISSING_CLIENTS[client_id] = (
                f"Requires optional dependency '{extra}'. "
                f"Install from the repo root via "
                f"`uv pip install -e 'intelligence-per-watt[{extra}]'`."
            )
            continue
        except ImportError:
            # Native extension load failures (e.g. broken .so ABI).
            if extra is None:
                raise
            MISSING_CLIENTS[client_id] = (
                f"Failed to load native extensions for '{extra}'."
            )
            continue

        _register_if_missing(client_id, module, class_name)


def _register_if_missing(client_id: str, module: ModuleType, class_name: str) -> None:
    if ClientRegistry.has(client_id):
        return

    client_cls = getattr(module, class_name, None)
    if client_cls is None:
        return

    # Re-register without re-importing the module (important after ClientRegistry.clear()).
    ClientRegistry.register_value(client_id, client_cls)


__all__ = ["InferenceClient", "MISSING_CLIENTS", "ensure_registered"]
