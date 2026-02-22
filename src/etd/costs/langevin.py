"""Langevin-residual cost matrix for LRET.

Measures deviation from where Langevin dynamics would take each
particle, rather than deviation from staying put (Euclidean).

    c_L(x, y) = ||y - x - ε·s(x)||² / (4ε)

where s(x) = ∇log π(x).  The non-separable cross-term s(x)^T(y-x)
survives balanced OT and directly shapes the coupling.
"""

import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular


def langevin_residual_cost(
    positions: jnp.ndarray,    # (N, d)
    proposals: jnp.ndarray,    # (P, d)
    scores: jnp.ndarray,       # (N, d)
    epsilon: float,
    *,
    cholesky_factor: jnp.ndarray = None,  # (d, d)
    whiten: bool = False,
) -> jnp.ndarray:              # (N, P)
    """Compute Langevin-residual cost matrix.

    Non-whitened (default)::

        m_i = x_i + ε·s_i               (Langevin mean)
        C_ij = ||y_j - m_i||² / (4ε)

    Whitened (whiten=True, cholesky_factor=L)::

        m_i = x_i + ε·(LL^T)·s_i        (preconditioned Langevin mean)
        C_ij = ||L^{-1}(y_j - m_i)||² / (4ε)

    Uses dot-product expansion to avoid (N, P, d) intermediate::

        ||y - m||² = ||y||² + ||m||² - 2 y·m

    Args:
        positions: Particle positions, shape ``(N, d)``.
        proposals: Proposal positions, shape ``(P, d)``.
        scores: Score vectors ∇log π(x_i), shape ``(N, d)``.
        epsilon: Langevin step size / SB temperature (positive scalar).
        cholesky_factor: Lower-triangular Cholesky ``L``, shape ``(d, d)``.
            Required when ``whiten=True``.
        whiten: If True, use preconditioned Langevin mean and Mahalanobis
            distance.  Requires ``cholesky_factor``.

    Returns:
        Cost matrix, shape ``(N, P)``.  Non-negative.
    """
    # --- Compute Langevin means ---
    if whiten and cholesky_factor is not None:
        # Preconditioned drift: Σ·s = (LL^T)·s
        drift = (scores @ cholesky_factor) @ cholesky_factor.T  # (N, d)
        means = positions + epsilon * drift                      # (N, d)

        # Whiten: L^{-1} applied to both means and proposals
        d = positions.shape[1]
        L_inv = solve_triangular(
            cholesky_factor, jnp.eye(d), lower=True,
        )  # (d, d)
        means = means @ L_inv.T       # (N, d)
        proposals = proposals @ L_inv.T  # (P, d)
    else:
        # Isotropic drift: s directly
        means = positions + epsilon * scores  # (N, d)

    # --- Dot-product expansion: ||y - m||² = ||y||² + ||m||² - 2 y·m ---
    mm = jnp.sum(means ** 2, axis=1)       # (N,)
    yy = jnp.sum(proposals ** 2, axis=1)   # (P,)
    my = means @ proposals.T               # (N, P)

    C = (mm[:, None] + yy[None, :] - 2.0 * my) / (4.0 * epsilon)
    return jnp.maximum(C, 0.0)
