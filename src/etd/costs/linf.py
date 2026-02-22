"""L-infinity cost matrix.

    C_ij = ||x_i - y_j||_inf = max_k |x_ik - y_jk|

L-infinity avoids the concentration-of-measure problem of L2 in high
dimensions: L2 distances cluster near sqrt(d) * sigma while L-inf
spreads more evenly, keeping the Gibbs kernel informative.
"""

import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular


def linf_cost(
    positions: jnp.ndarray,   # (N, d)
    proposals: jnp.ndarray,   # (P, d)
    *,
    preconditioner: jnp.ndarray = None,  # (d,) optional whitening
    cholesky_factor: jnp.ndarray = None,  # (d, d) optional whitening
) -> jnp.ndarray:             # (N, P)
    """L-infinity cost, optionally in whitened coordinates.

    When *preconditioner* is provided, computes max_k |Δ_k / P_k|
    where Δ_k = x_ik - y_jk.  When *cholesky_factor* is provided,
    computes max_k |(L⁻¹ Δ)_k|.

    Args:
        positions: Particle positions, shape ``(N, d)``.
        proposals: Proposal positions, shape ``(P, d)``.
        preconditioner: Diagonal preconditioner P = 1/sqrt(G + δ),
            shape ``(d,)``.  When provided, coordinates are whitened.
        cholesky_factor: Cholesky factor ``L``, shape ``(d, d)``.
            Overrides ``preconditioner`` when provided.

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

    # (N, 1, d) - (1, P, d) -> (N, P, d) -> max over d -> (N, P)
    diff = jnp.abs(positions[:, None, :] - proposals[None, :, :])
    return jnp.max(diff, axis=-1)
