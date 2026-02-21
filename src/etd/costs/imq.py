"""Positive multiquadric (IMQ) cost matrix."""

import jax.numpy as jnp


def imq_cost(
    positions: jnp.ndarray,   # (N, d)
    proposals: jnp.ndarray,   # (P, d)
    *,
    preconditioner: jnp.ndarray = None,  # unused, kept for uniform interface
    c: float = 1.0,
) -> jnp.ndarray:             # (N, P)
    """Compute C_ij = sqrt(c² + ‖x_i - y_j‖²) - c.

    Positive multiquadric cost with sub-linear growth: O(‖x‖) vs O(‖x‖²)
    for squared Euclidean.  Avoids high-dimensional concentration while
    remaining a proper cost (zero at coincidence, monotonically increasing).

    Uses dot-product expansion to avoid an (N, P, d) intermediate:
        ‖x - y‖² = ‖x‖² + ‖y‖² - 2 x·y

    Args:
        positions: Particle positions, shape ``(N, d)``.
        proposals: Proposal positions, shape ``(P, d)``.
        preconditioner: Unused.  Accepted for interface uniformity.
        c: Offset parameter (default 1.0).  Must be non-negative.

    Returns:
        Cost matrix, shape ``(N, P)``.  Non-negative.
    """
    xx = jnp.sum(positions ** 2, axis=1)
    yy = jnp.sum(proposals ** 2, axis=1)
    xy = positions @ proposals.T

    sq_dists = jnp.maximum(xx[:, None] + yy[None, :] - 2.0 * xy, 0.0)
    return jnp.sqrt(c ** 2 + sq_dists) - c
