"""Dataset implementations bundled with TrafficBench.

Datasets register themselves with ``trafficbench.core.DatasetRegistry``.
"""

from . import trafficbench

__all__ = ["trafficbench"]
