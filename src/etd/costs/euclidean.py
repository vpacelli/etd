"""Squared Euclidean cost matrix."""

import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular


def squared_euclidean_cost(
    positions: jnp.ndarray,   # (N, d)
    proposals: jnp.ndarray,   # (P, d)
    *,
    preconditioner: jnp.ndarray = None,  # (d,) optional whitening
    cholesky_factor: jnp.ndarray = None,  # (d, d) optional whitening
) -> jnp.ndarray:             # (N, P)
    """Compute C_ij = ‖x_i - y_j‖² / 2, optionally in whitened coordinates.

    When *preconditioner* is provided, computes the diagonal Mahalanobis cost:
        C_ij = (1/2)(x_i - y_j)^T diag(P)^{-2} (x_i - y_j)
    by pre-scaling coordinates by 1/P.

    When *cholesky_factor* is provided, computes the full Mahalanobis cost:
        C_ij = (1/2)(x_i - y_j)^T (LLᵀ)⁻¹ (x_i - y_j)
    by pre-whitening coordinates via L⁻¹.

    Uses the dot-product expansion to avoid an (N, P, d) intermediate:
        ‖x - y‖² = ‖x‖² + ‖y‖² - 2 x·y

    Floors at zero to prevent negative values from float cancellation.

    Args:
        positions: Particle positions, shape ``(N, d)``.
        proposals: Proposal positions, shape ``(P, d)``.
        preconditioner: Diagonal preconditioner P = 1/sqrt(G + δ),
            shape ``(d,)``.  When provided, coordinates are whitened.
        cholesky_factor: Lower-triangular Cholesky factor ``L``,
            shape ``(d, d)``.  When provided, overrides ``preconditioner``.

    Returns:
        Cost matrix, shape ``(N, P)``.  Non-negative.
    """
    if cholesky_factor is not None:
        d = positions.shape[1]
        L_inv = solve_triangular(
            cholesky_factor, jnp.eye(d), lower=True,
        )  # (d, d)
        positions = positions @ L_inv.T   # (N, d)
        proposals = proposals @ L_inv.T   # (P, d)
    elif preconditioner is not None:
        inv_P = 1.0 / preconditioner   # (d,)
        positions = positions * inv_P   # (N, d)
        proposals = proposals * inv_P   # (P, d)

    # (N,) and (P,) — squared norms
    xx = jnp.sum(positions ** 2, axis=1)
    yy = jnp.sum(proposals ** 2, axis=1)

    # (N, P) — cross term
    xy = positions @ proposals.T

    C = 0.5 * (xx[:, None] + yy[None, :] - 2.0 * xy)
    return jnp.maximum(C, 0.0)
