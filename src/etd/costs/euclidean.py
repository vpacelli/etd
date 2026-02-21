"""Squared Euclidean cost matrix."""

import jax.numpy as jnp


def squared_euclidean_cost(
    positions: jnp.ndarray,   # (N, d)
    proposals: jnp.ndarray,   # (P, d)
) -> jnp.ndarray:             # (N, P)
    """Compute C_ij = ‖x_i - y_j‖² / 2.

    Uses the dot-product expansion to avoid an (N, P, d) intermediate:
        ‖x - y‖² = ‖x‖² + ‖y‖² - 2 x·y

    Floors at zero to prevent negative values from float cancellation.

    Args:
        positions: Particle positions, shape ``(N, d)``.
        proposals: Proposal positions, shape ``(P, d)``.

    Returns:
        Cost matrix, shape ``(N, P)``.  Non-negative.
    """
    # (N,) and (P,) — squared norms
    xx = jnp.sum(positions ** 2, axis=1)
    yy = jnp.sum(proposals ** 2, axis=1)

    # (N, P) — cross term
    xy = positions @ proposals.T

    C = 0.5 * (xx[:, None] + yy[None, :] - 2.0 * xy)
    return jnp.maximum(C, 0.0)
