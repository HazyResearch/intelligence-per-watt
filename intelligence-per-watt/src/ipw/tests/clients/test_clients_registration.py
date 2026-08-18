import builtins
import sys

import pytest

from ipw.clients import MISSING_CLIENTS, ensure_registered
from ipw.core.registry import ClientRegistry


def test_ensure_registered_registers_openai_client() -> None:
    # Ensure a clean registry for the test run
    ClientRegistry.clear()
    try:
        ensure_registered()
        client_cls = ClientRegistry.get("openai")
        assert getattr(client_cls, "client_id", None) == "openai"
    finally:
        ClientRegistry.clear()


@pytest.mark.parametrize("client_id", ["afm", "mlx", "vllm", "ollama"])
def test_optional_clients_register_or_report_missing(client_id: str) -> None:
    """An optional backend either registers or explains itself -- never raises."""
    ClientRegistry.clear()
    try:
        ensure_registered()
        assert ClientRegistry.has(client_id) or client_id in MISSING_CLIENTS
    finally:
        ClientRegistry.clear()


def test_missing_afm_sdk_does_not_break_other_clients(monkeypatch) -> None:
    """Regression: the ``afm`` extra installs a package whose import root differs.

    ``ensure_registered`` matches a failed import's root module against the
    extra name and re-raises on a mismatch. The ``afm`` extra installs
    ``apple-fm-sdk``, which imports as ``apple_fm_sdk``, so without the explicit
    import-root mapping this would propagate and take every client down with it
    on any host lacking the SDK -- which is all of CI.
    """
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "apple_fm_sdk" or name.startswith("apple_fm_sdk."):
            raise ModuleNotFoundError(f"No module named {name!r}", name="apple_fm_sdk")
        return real_import(name, *args, **kwargs)

    # Drop cached copies so the patched import is actually exercised.
    monkeypatch.delitem(sys.modules, "apple_fm_sdk", raising=False)
    monkeypatch.delitem(sys.modules, "ipw.clients.afm", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    ClientRegistry.clear()
    try:
        ensure_registered()  # must not raise
        assert "afm" in MISSING_CLIENTS
        assert "afm" in MISSING_CLIENTS["afm"]
        # Unrelated clients still land.
        assert ClientRegistry.has("openai")
        assert ClientRegistry.has("openai-server")
    finally:
        ClientRegistry.clear()
        MISSING_CLIENTS.pop("afm", None)
