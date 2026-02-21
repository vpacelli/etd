"""Cost matrix normalization (median or mean heuristic)."""

from typing import Tuple

import jax.numpy as jnp

# Maximum number of elements to sort for the approximate median.
# jnp.median compiles to a full O(n log n) sort in XLA.  For a
# (100, 2500) cost matrix the full sort dominates wall-clock time
# (~31 ms out of ~33 ms per step).  Subsampling to ~2048 elements
# via a fixed stride gives a reliable scale estimate at <1 ms.
_MAX_MEDIAN_SAMPLES = 2048


def median_normalize(
    C: jnp.ndarray,   # (N, P)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Normalize cost matrix so that its median equals 1.

    This makes ε interpretable across problems and iterations:
    ε = 0.1 always means "10% of the typical pairwise cost."

    Uses a strided subsample when the cost matrix exceeds
    ``_MAX_MEDIAN_SAMPLES`` elements so that the sort inside
    ``jnp.median`` stays cheap.

    Guards against a zero or near-zero median.

    Args:
        C: Raw cost matrix, shape ``(N, P)``.

    Returns:
        Tuple of (normalized cost, median scalar).
    """
    flat = C.ravel()
    n = flat.shape[0]
    # stride is a compile-time constant (shapes are concrete under JIT)
    stride = max(1, n // _MAX_MEDIAN_SAMPLES)
    median = jnp.median(flat[::stride])
    median = jnp.maximum(median, 1e-8)
    return C / median, median


def mean_normalize(
    C: jnp.ndarray,   # (N, P)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Normalize cost matrix so that its mean equals 1.

    Cheaper than median (O(n) vs O(n log n) sort) but less robust
    to outliers.

    Guards against a zero or near-zero mean.

    Args:
        C: Raw cost matrix, shape ``(N, P)``.

    Returns:
        Tuple of (normalized cost, mean scalar).
    """
    mean = jnp.maximum(jnp.mean(C), 1e-8)
    return C / mean, mean


def normalize_cost(
    C: jnp.ndarray,        # (N, P)
    method: str = "median",
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Normalize cost matrix using the specified method.

    Dispatches to :func:`median_normalize` or :func:`mean_normalize`.

    Args:
        C: Raw cost matrix, shape ``(N, P)``.
        method: ``"median"`` (default) or ``"mean"``.

    Returns:
        Tuple of (normalized cost, scale scalar).

    Raises:
        ValueError: If *method* is not recognized.
    """
    if method == "median":
        return median_normalize(C)
    elif method == "mean":
        return mean_normalize(C)
    else:
        raise ValueError(f"Unknown cost normalization method '{method}'.")
