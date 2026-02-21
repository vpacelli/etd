"""Cost matrix normalization via the median heuristic."""

from typing import Tuple

import jax.numpy as jnp


def median_normalize(
    C: jnp.ndarray,   # (N, P)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Normalize cost matrix so that its median equals 1.

    This makes ε interpretable across problems and iterations:
    ε = 0.1 always means "10% of the typical pairwise cost."

    Guards against a zero or near-zero median.

    Args:
        C: Raw cost matrix, shape ``(N, P)``.

    Returns:
        Tuple of (normalized cost, median scalar).
    """
    median = jnp.median(C)
    median = jnp.maximum(median, 1e-8)
    return C / median, median
