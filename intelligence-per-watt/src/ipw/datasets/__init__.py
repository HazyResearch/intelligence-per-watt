"""Dataset implementations bundled with Intelligence Per Watt.

Datasets register themselves with ``ipw.core.DatasetRegistry``.
"""

from .base import DatasetProvider


def ensure_registered() -> None:
    """Import built-in dataset providers to populate the registry."""
    from . import (  # noqa: F401
        frames,
        gaia,
        hle,
        ipw,
        mmlu_pro,
        simpleqa,
        supergpqa,
        swebench,
        swefficiency,
    )

    # Optional-dependency datasets: import errors are silently ignored so
    # that missing packages don't prevent the rest of the registry from
    # being populated.
    try:
        from . import terminalbench  # noqa: F401
    except ImportError:
        pass


__all__ = ["DatasetProvider", "ensure_registered"]
