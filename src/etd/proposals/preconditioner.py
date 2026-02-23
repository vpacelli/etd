"""Preconditioner computation — diagonal (RMSProp) and full (Cholesky).

Provides:
- ``update_rmsprop_accum``: RMSProp-style diagonal accumulator update.
- ``compute_diagonal_P``: Convert accumulator to diagonal preconditioner.
- ``compute_ensemble_cholesky``: Cholesky factor of shrunk ensemble covariance.
"""

import jax.numpy as jnp

from etd.types import PreconditionerConfig


# ---------------------------------------------------------------------------
# Diagonal (RMSProp) preconditioner
# ---------------------------------------------------------------------------

def update_rmsprop_accum(
    accum: jnp.ndarray,     # (d,)
    scores: jnp.ndarray,    # (N, d)
    beta: float = 0.9,
) -> jnp.ndarray:           # (d,)
    """RMSProp-style preconditioner accumulator update.

    .. math::
        G_{t} = \\beta \\, G_{t-1} + (1 - \\beta) \\, \\text{mean}_i(s_i^2)

    Args:
        accum: Previous accumulator, shape ``(d,)``.
        scores: Score vectors, shape ``(N, d)``.
        beta: EMA decay (default 0.9).

    Returns:
        Updated accumulator, shape ``(d,)``.
    """
    mean_sq = jnp.mean(scores ** 2, axis=0)   # (d,)
    return beta * accum + (1.0 - beta) * mean_sq


def compute_diagonal_P(
    accum: jnp.ndarray,   # (d,)
    delta: float = 1e-8,
) -> jnp.ndarray:         # (d,)
    """Convert RMSProp accumulator to diagonal preconditioner.

    .. math::
        P = 1 / \\sqrt{G + \\delta}

    Args:
        accum: Accumulator, shape ``(d,)``.
        delta: Regularization floor.

    Returns:
        Diagonal preconditioner, shape ``(d,)``.
    """
    return 1.0 / jnp.sqrt(accum + delta)


# ---------------------------------------------------------------------------
# Full (Cholesky) preconditioner
# ---------------------------------------------------------------------------

def compute_ensemble_cholesky(
    data: jnp.ndarray,              # (N, d) — scores or positions
    prev_L: jnp.ndarray,            # (d, d) — previous Cholesky factor
    config: PreconditionerConfig,
) -> jnp.ndarray:                    # (d, d)
    """Cholesky factor of shrunk ensemble covariance.

    Steps:
        1. ``Σ_hat = Cov(data)``  — sample covariance
        2. ``Σ = (1-s)*Σ_hat + s*diag(diag(Σ_hat))``  — Ledoit-Wolf shrinkage
        3. ``Σ_reg = Σ + jitter*I``  — PD guarantee
        4. If ``ema_beta > 0``: blend with previous covariance on Σ level
        5. ``L = cholesky(Σ_reg)``

    Args:
        data: Ensemble data, shape ``(N, d)``.  Either scores or positions.
        prev_L: Previous Cholesky factor, shape ``(d, d)``.
        config: Preconditioner configuration.

    Returns:
        Lower-triangular Cholesky factor, shape ``(d, d)``.
    """
    d = data.shape[1]

    # 1. Sample covariance
    mean = jnp.mean(data, axis=0)                     # (d,)
    centered = data - mean[None, :]                    # (N, d)
    cov_hat = (centered.T @ centered) / (data.shape[0] - 1)  # (d, d)

    # 2. Ledoit-Wolf shrinkage toward diagonal
    s = config.shrinkage
    cov = (1.0 - s) * cov_hat + s * jnp.diag(jnp.diag(cov_hat))  # (d, d)

    # 3. Diagonal jitter for PD
    cov_reg = cov + config.jitter * jnp.eye(d)        # (d, d)

    # 4. EMA blending (on covariance, not on L — preserves PD)
    if config.ema > 0.0:
        prev_cov = prev_L @ prev_L.T                  # (d, d)
        cov_reg = config.ema * prev_cov + (1.0 - config.ema) * cov_reg

    # 5. Cholesky factorization
    L = jnp.linalg.cholesky(cov_reg)                  # (d, d)

    return L
