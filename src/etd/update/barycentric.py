"""Barycentric update rule.

Moves each particle toward the coupling-weighted mean of proposals,
optionally dampened by a step size:

    x_new[i] = (1 - eta) * x[i] + eta * sum_j gamma_ij * y_j

This is the deterministic alternative to categorical resampling.
It produces smoother trajectories but cannot make the discrete jumps
that resampling allows.
"""

from typing import Optional

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


def barycentric_update(
    key: jax.Array,
    log_gamma: jnp.ndarray,        # (N, P)
    proposals: jnp.ndarray,        # (P, d)
    step_size: float = 1.0,
    positions: Optional[jnp.ndarray] = None,  # (N, d)
) -> tuple:
    """Barycentric (weighted mean) update.

    For each particle *i*, computes the coupling-weighted mean of proposals:

    .. math::
        \\bar{y}_i = \\sum_j \\gamma_{ij} \\, y_j

    Then applies step-size damping:

    .. math::
        x_i^{\\text{new}} = (1 - \\eta) \\, x_i + \\eta \\, \\bar{y}_i

    The ``key`` argument is unused (kept for interface uniformity with
    ``systematic_resample``).

    Args:
        key: JAX PRNG key (unused — kept for uniform interface).
        log_gamma: Log coupling matrix, shape ``(N, P)``.
        proposals: Proposal positions, shape ``(P, d)``.
        step_size: Damping factor eta in (0, 1].  Default 1.0 (pure mean).
        positions: Current positions, shape ``(N, d)``.  Required when
            ``step_size < 1.0``.

    Returns:
        Tuple ``(new_positions, aux)`` where:
        - ``new_positions``: shape ``(N, d)``
        - ``aux``: dict with ``"weights"`` → row-normalized coupling ``(N, P)``
    """
    # Row-normalize to get conditional P(j|i)
    log_weights = log_gamma - logsumexp(log_gamma, axis=1, keepdims=True)  # (N, P)
    weights = jnp.exp(log_weights)  # (N, P)

    # Barycentric mean: (N, P) @ (P, d) → (N, d)
    bary_mean = weights @ proposals

    # Apply step-size damping
    if step_size < 1.0:
        return (1.0 - step_size) * positions + step_size * bary_mean, {"weights": weights}

    return bary_mean, {"weights": weights}
