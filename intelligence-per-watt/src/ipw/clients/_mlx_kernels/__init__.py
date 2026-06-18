"""In-process kernel-level model modifications for the MLX client.

Each kernel mod is a separate module exposing two functions:
    install_<name>(model) -> handle
    uninstall_<name>(handle) -> None

This package is private to ``ipw.clients.mlx``. Adding a new mod is one new
file plus one entry in ``_REGISTRY`` below — the MLX client itself doesn't
need to change.

Composition: the client applies opts left-to-right and uninstalls in reverse
(LIFO), so each swap sees the already-modified module hierarchy as its base.
The ``Swapped_<...>_`` class-name prefix used by every mod's ``_build_class``
makes re-install on the same model a no-op rather than a stack of nested
subclasses.

Note: the ``spec_decode`` submodule does NOT follow the install/uninstall
contract — speculative decoding is a generation-strategy change rather than
a model mutation, and is wired through the MLX client's constructor
(``draft_model=`` / ``num_draft_tokens=``) instead of ``opts``. It lives in
this package because it is the only other MLX-specific knob and we want all
MLX-specific code in one place.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

from . import qkv_fuse

# name -> (install_fn, uninstall_fn). Each install_fn returns an opaque handle;
# uninstall_fn(handle) reverses the change.
_REGISTRY: dict[str, Tuple[Callable, Callable]] = {
    "qkv_fuse": (qkv_fuse.install_qkv_fuse, qkv_fuse.uninstall_qkv_fuse),
}


def available() -> List[str]:
    """Return the names of installable kernel mods in registration order."""
    return sorted(_REGISTRY)


def install(name: str, model) -> Tuple[Callable, object]:
    """Apply the named kernel mod to ``model``.

    Returns ``(uninstall_fn, handle)`` so the caller can restore later via
    ``uninstall_fn(handle)``. Raises ``ValueError`` for unknown names.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown mlx kernel mod {name!r}; available: {available()}"
        )
    install_fn, uninstall_fn = _REGISTRY[name]
    return uninstall_fn, install_fn(model)


__all__ = ["available", "install"]
