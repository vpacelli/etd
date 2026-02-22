"""Cost function registry."""

import functools

from etd.costs.euclidean import squared_euclidean_cost
from etd.costs.imq import imq_cost
from etd.costs.langevin import langevin_residual_cost
from etd.costs.linf import linf_cost
from etd.costs.mahalanobis import mahalanobis_cost
from etd.costs.normalize import mean_normalize, median_normalize, normalize_cost

COSTS = {
    "euclidean": squared_euclidean_cost,
    "mahalanobis": mahalanobis_cost,
    "linf": linf_cost,
    "imq": imq_cost,
    "langevin": langevin_residual_cost,
}


def build_cost_fn(name: str, params: tuple = ()):
    """Build a cost function, binding any extra parameters via partial.

    Args:
        name: Cost function name (key in :data:`COSTS`).
        params: Sorted ``(key, value)`` pairs to bind, e.g.
            ``(("c", 1.0),)``.  Empty tuple means no extra params.

    Returns:
        Callable ``(positions, proposals, *, preconditioner=None) → (N, P)``.

    Raises:
        KeyError: If *name* is not in the registry.
    """
    if name not in COSTS:
        raise KeyError(f"Unknown cost function '{name}'. Available: {list(COSTS)}")
    base_fn = COSTS[name]
    if params:
        return functools.partial(base_fn, **dict(params))
    return base_fn


def get_cost_fn(name: str):
    """Look up a cost function by name (backward-compatible).

    Raises:
        KeyError: If *name* is not in the registry.
    """
    return build_cost_fn(name)


__all__ = [
    "COSTS",
    "build_cost_fn",
    "get_cost_fn",
    "squared_euclidean_cost",
    "mahalanobis_cost",
    "linf_cost",
    "imq_cost",
    "langevin_residual_cost",
    "mean_normalize",
    "median_normalize",
    "normalize_cost",
]
