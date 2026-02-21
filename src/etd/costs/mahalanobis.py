"""Diagonal Mahalanobis cost matrix.

Uses the RMSProp preconditioner to whiten distances:

    C_ij = (1/2) (x_i - y_j)^T diag(P)^{-2} (x_i - y_j)

where P = 1/sqrt(G + delta) is the diagonal preconditioner.
Pre-scaling by 1/P reduces this to standard squared Euclidean,
keeping memory at O(NP) via the dot-product expansion.
"""

import jax.numpy as jnp


def mahalanobis_cost(
    positions: jnp.ndarray,   # (N, d)
    proposals: jnp.ndarray,   # (P, d)
    *,
    preconditioner: jnp.ndarray = None,  # (d,) — required
) -> jnp.ndarray:             # (N, P)
    """Diagonal Mahalanobis cost.

    C_ij = 0.5 * (x_i - y_j)^T diag(P)^{-2} (x_i - y_j)

    Args:
        positions: Particle positions, shape ``(N, d)``.
        proposals: Proposal positions, shape ``(P, d)``.
        preconditioner: Diagonal preconditioner P = 1/sqrt(G + δ),
            shape ``(d,)``.  **Required** — raises ``ValueError`` if None.

    Returns:
        Cost matrix, shape ``(N, P)``.  Non-negative.
    """
    if preconditioner is None:
        raise ValueError("Mahalanobis cost requires a preconditioner")

    # Scale coordinates by 1/P so that ||x_scaled - y_scaled||^2 = (x-y)^T P^{-2} (x-y)
    inv_P = 1.0 / preconditioner                   # (d,)
    x_scaled = positions * inv_P                    # (N, d)
    y_scaled = proposals * inv_P                    # (P, d)

    # Dot-product expansion: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    xx = jnp.sum(x_scaled ** 2, axis=1)            # (N,)
    yy = jnp.sum(y_scaled ** 2, axis=1)            # (P,)
    xy = x_scaled @ y_scaled.T                     # (N, P)

    C = 0.5 * (xx[:, None] + yy[None, :] - 2.0 * xy)
    return jnp.maximum(C, 0.0)
