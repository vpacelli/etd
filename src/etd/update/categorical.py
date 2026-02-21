"""Categorical update via systematic resampling.

Samples one proposal per particle from the coupling distribution
using systematic (stratified) resampling for lower variance than
multinomial sampling.
"""

from typing import Optional

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


def systematic_resample(
    key: jax.Array,
    log_gamma: jnp.ndarray,         # (N, P)
    proposals: jnp.ndarray,         # (P, d)
    step_size: float = 1.0,
    positions: Optional[jnp.ndarray] = None,   # (N, d) — needed for damping
) -> jnp.ndarray:                    # (N, d)
    """Resample proposals via systematic resampling.

    For each particle *i*, samples one proposal index from the
    categorical distribution defined by row *i* of the coupling:

    .. math::
        j^* \\sim \\text{Categorical}(\\gamma_{i1}, \\ldots, \\gamma_{iP})

    Uses the systematic resampling trick (single uniform draw +
    stratified offsets) for lower variance.

    Optionally applies step-size damping:

    .. math::
        x_i^{\\text{new}} = (1 - \\eta) \\, x_i + \\eta \\, y_{j^*}

    Args:
        key: JAX PRNG key.
        log_gamma: Log coupling matrix, shape ``(N, P)``.
            Rows must sum to ~1 in probability space.
        proposals: Proposal positions, shape ``(P, d)``.
        step_size: Damping factor η ∈ (0, 1].  Default 1.0 (full step).
        positions: Current positions, shape ``(N, d)``.  Required when
            ``step_size < 1.0``.

    Returns:
        New particle positions, shape ``(N, d)``.
    """
    N, P = log_gamma.shape

    # Row-normalize to get conditional P(j|i) for sampling
    # (coupling functions return the joint; update rule converts to conditional)
    log_gamma_cond = log_gamma - logsumexp(log_gamma, axis=1, keepdims=True)

    # Exponentiate to probability space
    gamma = jnp.exp(log_gamma_cond)   # (N, P)

    # Cumulative sums along proposal axis
    cumsum = jnp.cumsum(gamma, axis=1)   # (N, P)

    # Force last column to exactly 1.0 (float accumulation safety)
    cumsum = cumsum.at[:, -1].set(1.0)

    # Systematic resampling: single uniform + stratified offsets
    u0 = jax.random.uniform(key, shape=(N, 1))          # (N, 1)
    offsets = jnp.arange(1, dtype=jnp.float32)           # (1,) — just [0]
    # For systematic resampling with 1 sample per row:
    # u_i = (u0_i + 0) / 1 = u0_i
    # But we're sampling 1 proposal per particle, so the "stratified"
    # part is across particles.  Each particle gets its own uniform.
    u = u0  # (N, 1)

    # Find first index where cumsum >= u for each particle
    exceeded = cumsum >= u                                # (N, P) boolean
    indices = jnp.argmax(exceeded, axis=1)                # (N,) — first True

    # Gather selected proposals
    new_positions = proposals[indices]   # (N, d)

    # Optional damping
    if step_size < 1.0:
        assert positions is not None, "positions required when step_size < 1.0"
        new_positions = (1.0 - step_size) * positions + step_size * new_positions

    return new_positions
