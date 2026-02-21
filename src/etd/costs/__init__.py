"""Cost function registry."""

from etd.costs.euclidean import squared_euclidean_cost
from etd.costs.normalize import median_normalize

COSTS = {
    "euclidean": squared_euclidean_cost,
}


def get_cost_fn(name: str):
    """Look up a cost function by name.

    Raises:
        KeyError: If *name* is not in the registry.
    """
    if name not in COSTS:
        raise KeyError(f"Unknown cost function '{name}'. Available: {list(COSTS)}")
    return COSTS[name]


__all__ = [
    "COSTS",
    "get_cost_fn",
    "squared_euclidean_cost",
    "median_normalize",
]
