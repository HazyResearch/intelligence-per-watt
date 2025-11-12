"""Inference client implementations.

Clients register themselves with ``ipw.core.ClientRegistry``.
"""

from __future__ import annotations

import importlib
from typing import Dict

from .base import InferenceClient

MISSING_CLIENTS: Dict[str, str] = {}
_OPTIONAL_CLIENTS = (
    ("ipw.clients.ollama", "ollama", "ollama"),
    ("ipw.clients.vllm", "vllm", "vllm"),
)


def ensure_registered() -> None:
    """Import built-in client implementations to populate the registry."""
    for module_name, client_id, extra in _OPTIONAL_CLIENTS:
        _import_optional_client(module_name, client_id, extra)


def _import_optional_client(module_name: str, client_id: str, extra: str) -> None:
    MISSING_CLIENTS.pop(client_id, None)
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing_root = exc.name.split(".", 1)[0] if exc.name else None
        if missing_root != extra:
            raise
        MISSING_CLIENTS[client_id] = (
            f"Requires optional dependency '{extra}'. "
            f"Install from the repo root via `uv pip install -e 'intelligence-per-watt[{extra}]'` "
            f"(or from inside the package directory via `uv pip install -e '.[{extra}]'`)."
        )


__all__ = ["InferenceClient", "MISSING_CLIENTS", "ensure_registered"]
