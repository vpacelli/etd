"""Importance-corrected target weights.

Always on — there is no toggle.  The IS correction eliminates the
entropic bias in the balanced coupling case (Proposition 6.3 in the
draft).
"""

from typing import Optional

import jax.numpy as jnp
from jax.scipy.special import logsumexp


def _log_proposal_density(
    proposals: jnp.ndarray,                        # (P, d)
    means: jnp.ndarray,                            # (N, d)
    sigma: float,
    preconditioner: Optional[jnp.ndarray] = None,  # (d,)  P = 1/√(G+δ)
) -> jnp.ndarray:                                   # (P,)
    """Evaluate log q_μ(y) for the Gaussian mixture proposal.

    .. math::
        q_\\mu(y) = \\frac{1}{N} \\sum_{i=1}^{N}
            \\mathcal{N}(y;\\, \\mu_i,\\, \\Sigma)

    where Σ = σ²I (isotropic) or Σ = diag(σ² P²) (preconditioned).

    Args:
        proposals: Proposal positions, shape ``(P, d)``.
        means: Per-particle proposal means, shape ``(N, d)``.
        sigma: Noise scale.
        preconditioner: Diagonal preconditioner ``P = 1/√(G+δ)``,
            shape ``(d,)``.  If None, isotropic covariance is used.

    Returns:
        Log mixture density at each proposal, shape ``(P,)``.
    """
    P, d = proposals.shape
    N = means.shape[0]

    # --- Compute per-component log densities ---
    # diff[p, n, :] = y_p - μ_n
    diff = proposals[:, None, :] - means[None, :, :]   # (P, N, d)

    if preconditioner is not None:
        # Preconditioned covariance: Σ = diag(σ² P²)
        # Mahalanobis: (y - μ)ᵀ Σ⁻¹ (y - μ) = Σ_k (diff_k / (σ P_k))²
        inv_std = 1.0 / (sigma * preconditioner)                     # (d,)
        maha_sq = jnp.sum((diff * inv_std[None, None, :]) ** 2, axis=-1)  # (P, N)
        # Log-determinant: -½ Σ_k log(2π σ² P_k²)
        log_det_term = -0.5 * jnp.sum(jnp.log(2.0 * jnp.pi * (sigma * preconditioner) ** 2))
    else:
        # Isotropic: Σ = σ²I
        maha_sq = jnp.sum(diff ** 2, axis=-1) / (sigma ** 2)   # (P, N)
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

    Returns:
        Normalized log weights, shape ``(P,)``.
        In log domain: ``logsumexp(result) ≈ 0``.
    """
    log_pi = target.log_prob(proposals)   # (P,)
    log_q = _log_proposal_density(proposals, means, sigma, preconditioner)   # (P,)

    log_b = log_pi - log_q

    # Floor: clip proposal density from below to prevent blowup
    log_b = jnp.maximum(log_b, jnp.max(log_b) - 30.0)

    # Normalize to a log-probability vector
    log_b = log_b - logsumexp(log_b)

    return log_b
