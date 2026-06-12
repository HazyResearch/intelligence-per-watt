"""Dataset implementations bundled with Intelligence Per Watt.

Datasets register themselves with ``ipw.core.DatasetRegistry``.
"""

from .base import DatasetProvider


def ensure_registered() -> None:
    """Import built-in dataset providers to populate the registry."""
    import ipw.evaluation  # noqa: F401  # register evaluation handlers used by datasets

    from . import (  # noqa: F401
        fixed_length,
        arena_hard_auto,
        browsecomp,
        deepresearchbench,
        frames,
        gaia,
        gdpval,
        gpqa,
        hle,
        ipw,
        livecodebench,
        liveresearchbench,
        math500,
        mmlu_pro,
        natural_reasoning,
        simpleqa,
        supergpqa,
        swebench,
        swefficiency,
        wildchat,
    )

    # Optional-dependency datasets: import errors are silently ignored so
    # that missing packages don't prevent the rest of the registry from
    # being populated.
    try:
        from . import terminalbench  # noqa: F401
    except ImportError:
        pass

    try:
        from . import terminalbench_native  # noqa: F401
    except ImportError:
        pass


__all__ = ["DatasetProvider", "ensure_registered"]
