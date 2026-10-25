"""Intelligence per Joule measurement and orchestration harness."""

__version__ = "0.0.1"

# Register IPJ-side additions to the IPW pricing tables at import time so any
# ``ipw.cost.pricing.calculate_cost`` call from IPJ code resolves them.
# (This is why ``intelligence-per-watt/src/ipw/cost/pricing.py`` can stay
# byte-identical with public upstream — see ADR-0004 when written.)
from . import cost as _cost  # noqa: E402,F401
