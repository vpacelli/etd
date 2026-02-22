"""Importance-corrected target weights.

Always on — there is no toggle.  The IS correction eliminates the
entropic bias in the balanced coupling case (Proposition 6.3 in the
draft).
"""

from typing import Optional

import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import logsumexp


def _log_proposal_density(
    proposals: jnp.ndarray,                        # (P, d)
    means: jnp.ndarray,                            # (N, d)
    sigma: float,
    preconditioner: Optional[jnp.ndarray] = None,  # (d,)  P = 1/√(G+δ)
    cholesky_factor: Optional[jnp.ndarray] = None,  # (d, d)
) -> jnp.ndarray:                                   # (P,)
    """Evaluate log q_μ(y) for the Gaussian mixture proposal.

    .. math::
        q_\\mu(y) = \\frac{1}{N} \\sum_{i=1}^{N}
            \\mathcal{N}(y;\\, \\mu_i,\\, \\Sigma)

    where Σ = σ²I (isotropic), Σ = diag(σ² P²) (diagonal), or
    Σ = σ² L Lᵀ (Cholesky).

    Args:
        proposals: Proposal positions, shape ``(P, d)``.
        means: Per-particle proposal means, shape ``(N, d)``.
        sigma: Noise scale.
        preconditioner: Diagonal preconditioner ``P = 1/√(G+δ)``,
            shape ``(d,)``.  If None, isotropic covariance is used.
        cholesky_factor: Cholesky factor ``L``, shape ``(d, d)``.
            When provided, overrides ``preconditioner``.

    Returns:
        Log mixture density at each proposal, shape ``(P,)``.
    """
    d = proposals.shape[1]
    N = means.shape[0]

    # --- Compute per-component Mahalanobis distances ---
    # Uses ||a-b||² = ||a||² + ||b||² - 2⟨a,b⟩ to avoid (P,N,d) intermediate.
    # Floors at zero to prevent negative values from float cancellation.
    if cholesky_factor is not None:
        # Full covariance: Σ = σ² L Lᵀ
        # Whiten: z = (σL)⁻¹ x, then Mahalanobis = ||z_y - z_μ||²
        sigma_L = sigma * cholesky_factor                              # (d, d)
        sigma_L_inv = solve_triangular(
            sigma_L, jnp.eye(d), lower=True,
        )                                                              # (d, d)
        z_y = proposals @ sigma_L_inv.T                                # (P, d)
        z_mu = means @ sigma_L_inv.T                                   # (N, d)
        yy = jnp.sum(z_y ** 2, axis=1)                                # (P,)
        mm = jnp.sum(z_mu ** 2, axis=1)                               # (N,)
        ym = z_y @ z_mu.T                                             # (P, N)
        maha_sq = jnp.maximum(yy[:, None] + mm[None, :] - 2.0 * ym, 0.0)
        # Log-det: -½ (d log(2π) + 2 Σ_k log(|diag(σL)_k|))
        log_det_term = -0.5 * (
            d * jnp.log(2.0 * jnp.pi)
            + 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(sigma_L))))
        )
    elif preconditioner is not None:
        # Preconditioned covariance: Σ = diag(σ² P²)
        # Pre-whiten: z = x / (σP), then Mahalanobis = ||z_y - z_μ||²
        inv_std = 1.0 / (sigma * preconditioner)                      # (d,)
        z_y = proposals * inv_std[None, :]                             # (P, d)
        z_mu = means * inv_std[None, :]                                # (N, d)
        yy = jnp.sum(z_y ** 2, axis=1)                                # (P,)
        mm = jnp.sum(z_mu ** 2, axis=1)                               # (N,)
        ym = z_y @ z_mu.T                                             # (P, N)
        maha_sq = jnp.maximum(yy[:, None] + mm[None, :] - 2.0 * ym, 0.0)  # (P, N)
        # Log-determinant: -½ Σ_k log(2π σ² P_k²)
        log_det_term = -0.5 * jnp.sum(jnp.log(2.0 * jnp.pi * (sigma * preconditioner) ** 2))
    else:
        # Isotropic: Σ = σ²I
        yy = jnp.sum(proposals ** 2, axis=1)                          # (P,)
        mm = jnp.sum(means ** 2, axis=1)                              # (N,)
        ym = proposals @ means.T                                      # (P, N)
        sq_dist = jnp.maximum(yy[:, None] + mm[None, :] - 2.0 * ym, 0.0)
        maha_sq = sq_dist / (sigma ** 2)                              # (P, N)
        log_det_term = -0.5 * d * jnp.log(2.0 * jnp.pi * sigma ** 2)

    # log N(y; μ_n, Σ) = log_det_term - 0.5 * maha_sq
    log_components = log_det_term - 0.5 * maha_sq   # (P, N)

    # Mixture: q(y) = (1/N) Σ_n N(y; μ_n, Σ)
    # log q(y) = logsumexp over n  - log N
    log_q = logsumexp(log_components, axis=1) - jnp.log(N)   # (P,)

    return log_q


def importance_weights(
    proposals: jnp.ndarray,                        # (P, d)
    means: jnp.ndarray,                            # (N, d)
    target: object,                                 # Target protocol
    sigma: float,
    preconditioner: Optional[jnp.ndarray] = None,  # (d,)
    cholesky_factor: Optional[jnp.ndarray] = None,  # (d, d)
) -> jnp.ndarray:                                   # (P,)
    """Compute IS-corrected log target weights.

    .. math::
        \\log b_j = \\log \\pi(y_j) - \\log q_\\mu(y_j)

    Weights are floored (to prevent extreme ratios from outlier
    proposals) and normalized to a log-probability vector.

    Args:
        proposals: Pooled proposals, shape ``(P, d)``.
        means: Per-particle proposal means, shape ``(N, d)``.
        target: Target distribution with ``log_prob(x)`` method.
        sigma: Proposal noise scale.
        preconditioner: Diagonal preconditioner, shape ``(d,)``.
        cholesky_factor: Cholesky factor ``L``, shape ``(d, d)``.

    Returns:
        Normalized log weights, shape ``(P,)``.
        In log domain: ``logsumexp(result) ≈ 0``.
    """
    log_pi = target.log_prob(proposals)   # (P,)
    log_q = _log_proposal_density(
        proposals, means, sigma, preconditioner, cholesky_factor,
    )   # (P,)

    log_b = log_pi - log_q

    # Floor: clip proposal density from below to prevent blowup
    log_b = jnp.maximum(log_b, jnp.max(log_b) - 30.0)

    # Normalize to a log-probability vector
    log_b = log_b - logsumexp(log_b)

    return log_b
