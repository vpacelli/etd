"""Langevin-based proposal generation.

Generates the pooled proposal set {y_j}_{j=1}^{N*M} via Langevin
dynamics around current particle positions.  Four modes:

- **Score-guided** (default): y_ij = x_i + α·clip(∇logπ(x_i)) + σ·ξ
- **Score-free**: y_ij = x_i + σ·ξ
- **Preconditioned (diagonal)**: y_ij = x_i + α·P⊙s_i + σ·P⊙ξ
- **Preconditioned (Cholesky)**: y_ij = x_i + α·(LLᵀ)s_i + σ·Lξ
"""

from typing import Tuple

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Score clipping
# ---------------------------------------------------------------------------

def clip_scores(
    scores: jnp.ndarray,   # (N, d)
    max_norm: float = 5.0,
) -> jnp.ndarray:           # (N, d)
    """Clip score vectors to a maximum Euclidean norm.

    Essential for stability on targets with heavy tails or sharp
    gradients (e.g., BLR).

    Args:
        scores: Score vectors, shape ``(N, d)``.
        max_norm: Maximum allowed norm (default 5.0).

    Returns:
        Clipped scores with ``‖s_i‖ ≤ max_norm`` for all *i*.
    """
    norms = jnp.linalg.norm(scores, axis=-1, keepdims=True)   # (N, 1)
    scale = jnp.minimum(1.0, max_norm / jnp.maximum(norms, 1e-8))
    return scores * scale


# ---------------------------------------------------------------------------
# Preconditioner update
# ---------------------------------------------------------------------------

def update_preconditioner(
    accum: jnp.ndarray,     # (d,)
    scores: jnp.ndarray,    # (N, d)
    beta: float = 0.9,
) -> jnp.ndarray:           # (d,)
    """RMSProp-style preconditioner update.

    .. math::
        G_{t} = \\beta \\, G_{t-1} + (1 - \\beta) \\, \\text{mean}_i(s_i^2)

    Args:
        accum: Previous accumulator, shape ``(d,)``.
        scores: Score vectors from current particles, shape ``(N, d)``.
        beta: Exponential moving average decay (default 0.9).

    Returns:
        Updated accumulator, shape ``(d,)``.
    """
    mean_sq = jnp.mean(scores ** 2, axis=0)   # (d,)
    return beta * accum + (1.0 - beta) * mean_sq


# ---------------------------------------------------------------------------
# Proposal generation
# ---------------------------------------------------------------------------

def langevin_proposals(
    key: jax.Array,
    positions: jnp.ndarray,      # (N, d)
    target: object,               # Target protocol (dim, log_prob, score)
    alpha: float,
    sigma: float,
    n_proposals: int,             # M
    use_score: bool = True,
    score_clip_val: float = 5.0,
    precondition: bool = False,
    precond_accum: jnp.ndarray | None = None,   # (d,)
    precond_delta: float = 1e-8,
    cholesky_factor: jnp.ndarray | None = None,  # (d, d)
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate the pooled Langevin proposal set.

    Args:
        key: JAX PRNG key.
        positions: Current particle positions, shape ``(N, d)``.
        target: Target distribution with ``score(x)`` method.
        alpha: Score step size.
        sigma: Proposal noise scale.
        n_proposals: Number of proposals per particle (*M*).
        use_score: If True, use score-guided proposals; else score-free.
        score_clip_val: Maximum score norm (default 5.0).
        precondition: Whether to apply the diagonal preconditioner.
        precond_accum: RMSProp accumulator, shape ``(d,)``.  Required
            when ``precondition=True``.
        precond_delta: Regularization for preconditioner (default 1e-8).
        cholesky_factor: Lower-triangular Cholesky factor ``L``,
            shape ``(d, d)``.  When provided, overrides diagonal
            preconditioner.  Drift uses ``Σ@s = (LLᵀ)s``, noise
            uses ``σ·L·ξ``.

    Returns:
        Tuple ``(proposals, means, scores)`` where:
        - ``proposals``: Pooled proposal set, shape ``(N*M, d)``.
        - ``means``: Per-particle proposal means, shape ``(N, d)``.
        - ``scores``: Clipped score vectors, shape ``(N, d)``.
          Zeros when ``use_score=False``.
    """
    N, d = positions.shape
    M = n_proposals

    # --- Compute scores ---
    if use_score:
        raw_scores = target.score(positions)           # (N, d)
        scores = clip_scores(raw_scores, score_clip_val)
    else:
        scores = jnp.zeros_like(positions)             # (N, d)

    # --- Compute per-particle means ---
    if cholesky_factor is not None:
        # Cholesky: drift = x + α * (L @ L.T @ s)  =  x + α * Σ @ s
        # scores: (N, d), L: (d, d)
        # (L.T @ s.T): (d, N) → (L @ ...): (d, N) → transpose: (N, d)
        LLt_scores = (cholesky_factor @ (cholesky_factor.T @ scores.T)).T  # (N, d)
        means = positions + alpha * LLt_scores                              # (N, d)
    elif precondition:
        assert precond_accum is not None, "precond_accum required when precondition=True"
        P = 1.0 / jnp.sqrt(precond_accum + precond_delta)   # (d,)
        means = positions + alpha * (P * scores)              # (N, d)
    else:
        means = positions + alpha * scores                    # (N, d)

    # --- Sample noise ---
    noise = jax.random.normal(key, shape=(N, M, d))          # (N, M, d)

    if cholesky_factor is not None:
        # σ · L @ ξ: einsum over last axis
        scaled_noise = jnp.einsum('ij,nmj->nmi', cholesky_factor, noise)  # (N, M, d)
        proposals = means[:, None, :] + sigma * scaled_noise
    elif precondition:
        # σ · P ⊙ ξ
        proposals = means[:, None, :] + sigma * P[None, None, :] * noise
    else:
        proposals = means[:, None, :] + sigma * noise         # (N, M, d)

    # --- Pool: (N, M, d) → (N*M, d) ---
    proposals = proposals.reshape(N * M, d)

    return proposals, means, scores


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PROPOSALS = {
    "langevin": langevin_proposals,
}


def get_proposal_fn(name: str):
    """Look up a proposal function by name."""
    if name not in PROPOSALS:
        raise KeyError(f"Unknown proposal '{name}'. Available: {list(PROPOSALS)}")
    return PROPOSALS[name]
