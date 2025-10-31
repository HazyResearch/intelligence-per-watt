"""Dataset implementations bundled with Mindi.

Datasets register themselves with ``mindi.core.DatasetRegistry``.
"""

from . import trafficbench

__all__ = ["trafficbench"]
