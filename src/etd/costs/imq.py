"""Positive multiquadric (IMQ) cost matrix."""

import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular


def imq_cost(
    positions: jnp.ndarray,   # (N, d)
    proposals: jnp.ndarray,   # (P, d)
    *,
    preconditioner: jnp.ndarray = None,  # (d,) optional whitening
    cholesky_factor: jnp.ndarray = None,  # (d, d) optional whitening
    c: float = 1.0,
) -> jnp.ndarray:             # (N, P)
    """Compute C_ij = sqrt(c² + ‖x_i - y_j‖²) - c, optionally whitened.

    Positive multiquadric cost with sub-linear growth: O(‖x‖) vs O(‖x‖²)
    for squared Euclidean.  Avoids high-dimensional concentration while
    remaining a proper cost (zero at coincidence, monotonically increasing).

    When *preconditioner* is provided, distances are computed in whitened
    coordinates: ‖P⁻¹(x - y)‖².  When *cholesky_factor* is provided,
    distances are computed in ‖L⁻¹(x - y)‖².

    Uses dot-product expansion to avoid an (N, P, d) intermediate:
        ‖x - y‖² = ‖x‖² + ‖y‖² - 2 x·y

    Args:
        positions: Particle positions, shape ``(N, d)``.
        proposals: Proposal positions, shape ``(P, d)``.
        preconditioner: Diagonal preconditioner P = 1/sqrt(G + δ),
            shape ``(d,)``.  When provided, coordinates are whitened.
        cholesky_factor: Cholesky factor ``L``, shape ``(d, d)``.
            Overrides ``preconditioner`` when provided.
        c: Offset parameter (default 1.0).  Must be non-negative.

    Returns:
        Cost matrix, shape ``(N, P)``.  Non-negative.
    """
    if cholesky_factor is not None:
        d = positions.shape[1]
        L_inv = solve_triangular(
            cholesky_factor, jnp.eye(d), lower=True,
        )
        positions = positions @ L_inv.T
        proposals = proposals @ L_inv.T
    elif preconditioner is not None:
        inv_P = 1.0 / preconditioner   # (d,)
        positions = positions * inv_P   # (N, d)
        proposals = proposals * inv_P   # (P, d)

    xx = jnp.sum(positions ** 2, axis=1)
    yy = jnp.sum(proposals ** 2, axis=1)
    xy = positions @ proposals.T

    sq_dists = jnp.maximum(xx[:, None] + yy[None, :] - 2.0 * xy, 0.0)
    return jnp.sqrt(c ** 2 + sq_dists) - c
