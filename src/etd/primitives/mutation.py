"""MCMC mutation kernels for post-transport local refinement.

Provides MALA and Random-Walk MH kernels that operate on batches of
particles.  These complete the SMC structure: reweight → resample → mutate.
The MH correction guarantees π-invariance, so mutation can only improve
or maintain approximation quality.

Three public functions:

- :func:`mala_kernel` — single preconditioned MALA step (requires scores)
- :func:`rwm_kernel` — single preconditioned RWM step (score-free)
- :func:`mutate` — dispatcher, runs K kernel steps via ``lax.scan``
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg

from etd.proposals.langevin import clip_scores
from etd.types import MutationConfig


# ---------------------------------------------------------------------------
# MALA kernel
# ---------------------------------------------------------------------------

def mala_kernel(
    key: jax.Array,
    x: jnp.ndarray,            # (N, d)
    log_pi_x: jnp.ndarray,     # (N,)
    scores_x: jnp.ndarray,     # (N, d)
    target,                     # Target protocol
    h: float,                   # step size
    L: Optional[jnp.ndarray],  # (d, d) lower-triangular or None
    score_clip: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Single MALA step for all N particles.

    Cholesky mode (``L`` provided, lower-triangular):
        - Forward mean: ``μ_fwd = x + (h/2) · Σ · s``  where ``Σ = LLᵀ``
        - Proposal: ``y = μ_fwd + √h · L · ξ``,  ``ξ ~ N(0, I)``
        - Forward log-density via ``L⁻¹(y - μ_fwd)``

    Isotropic mode (``L is None``): same with ``Σ = I``.

    Args:
        key: JAX PRNG key.
        x: Current positions, shape ``(N, d)``.
        log_pi_x: Cached log π(x), shape ``(N,)``.
        scores_x: Cached clipped scores, shape ``(N, d)``.
        target: Target distribution with ``log_prob`` and ``score``.
        h: Step size.
        L: Lower-triangular Cholesky factor ``(d, d)`` or None.
        score_clip: Maximum score norm for clipping.

    Returns:
        Tuple ``(new_x, new_log_pi, new_scores, accepted)`` with shapes
        ``(N, d), (N,), (N, d), (N,)``.
    """
    N, d = x.shape
    k_noise, k_accept = jax.random.split(key)
    xi = jax.random.normal(k_noise, (N, d))

    if L is not None:
        # --- Cholesky mode: Σ = LLᵀ ---
        # Forward mean: μ_fwd = x + (h/2) · (LLᵀ) · s
        LLt_s = (L @ (L.T @ scores_x.T)).T          # (N, d)
        mu_fwd = x + (h / 2.0) * LLt_s

        # Proposal: y = μ_fwd + √h · L · ξ
        y = mu_fwd + jnp.sqrt(h) * (L @ xi.T).T     # (N, d)

        # Forward log-density: -1/(2h) · ‖L⁻¹(y - μ_fwd)‖²
        diff_fwd = y - mu_fwd                        # (N, d)
        z_fwd = jax.scipy.linalg.solve_triangular(
            L, diff_fwd.T, lower=True,
        )                                             # (d, N)
        log_q_fwd = -1.0 / (2.0 * h) * jnp.sum(z_fwd ** 2, axis=0)  # (N,)

        # Evaluate target at proposals
        log_pi_y = target.log_prob(y)                 # (N,)
        s_y = clip_scores(target.score(y), score_clip)  # (N, d)

        # Reverse mean: μ_rev = y + (h/2) · Σ · s_y
        LLt_sy = (L @ (L.T @ s_y.T)).T               # (N, d)
        mu_rev = y + (h / 2.0) * LLt_sy

        # Reverse log-density
        diff_rev = x - mu_rev                         # (N, d)
        z_rev = jax.scipy.linalg.solve_triangular(
            L, diff_rev.T, lower=True,
        )                                             # (d, N)
        log_q_rev = -1.0 / (2.0 * h) * jnp.sum(z_rev ** 2, axis=0)  # (N,)

    else:
        # --- Isotropic mode: Σ = I ---
        mu_fwd = x + (h / 2.0) * scores_x
        y = mu_fwd + jnp.sqrt(h) * xi

        diff_fwd = y - mu_fwd
        log_q_fwd = -1.0 / (2.0 * h) * jnp.sum(diff_fwd ** 2, axis=-1)

        log_pi_y = target.log_prob(y)
        s_y = clip_scores(target.score(y), score_clip)

        mu_rev = y + (h / 2.0) * s_y
        diff_rev = x - mu_rev
        log_q_rev = -1.0 / (2.0 * h) * jnp.sum(diff_rev ** 2, axis=-1)

    # --- MH acceptance ---
    log_alpha = log_pi_y - log_pi_x + log_q_rev - log_q_fwd  # (N,)
    log_alpha = jnp.minimum(log_alpha, 0.0)

    log_u = jnp.log(jax.random.uniform(k_accept, shape=(N,)))
    accept = log_u < log_alpha                        # (N,)

    new_x = jnp.where(accept[:, None], y, x)
    new_log_pi = jnp.where(accept, log_pi_y, log_pi_x)
    new_scores = jnp.where(accept[:, None], s_y, scores_x)

    return new_x, new_log_pi, new_scores, accept.astype(jnp.float32)


# ---------------------------------------------------------------------------
# RWM kernel
# ---------------------------------------------------------------------------

def rwm_kernel(
    key: jax.Array,
    x: jnp.ndarray,            # (N, d)
    log_pi_x: jnp.ndarray,     # (N,)
    target,                     # Target protocol
    h: float,                   # step size
    L: Optional[jnp.ndarray],  # (d, d) lower-triangular or None
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Single Random-Walk MH step for all N particles (score-free).

    Symmetric proposal: ``y = x + √h · L · ξ`` (or ``√h · ξ`` when isotropic).
    MH ratio simplifies to ``exp(log π(y) - log π(x))``.

    Args:
        key: JAX PRNG key.
        x: Current positions, shape ``(N, d)``.
        log_pi_x: Cached log π(x), shape ``(N,)``.
        target: Target distribution with ``log_prob``.
        h: Step size.
        L: Lower-triangular Cholesky factor ``(d, d)`` or None.

    Returns:
        Tuple ``(new_x, new_log_pi, accepted)`` with shapes
        ``(N, d), (N,), (N,)``.
    """
    N, d = x.shape
    k_noise, k_accept = jax.random.split(key)
    xi = jax.random.normal(k_noise, (N, d))

    if L is not None:
        y = x + jnp.sqrt(h) * (L @ xi.T).T
    else:
        y = x + jnp.sqrt(h) * xi

    log_pi_y = target.log_prob(y)  # (N,)

    # Symmetric proposal → MH ratio is just the density ratio
    log_alpha = log_pi_y - log_pi_x
    log_alpha = jnp.minimum(log_alpha, 0.0)

    log_u = jnp.log(jax.random.uniform(k_accept, shape=(N,)))
    accept = log_u < log_alpha

    new_x = jnp.where(accept[:, None], y, x)
    new_log_pi = jnp.where(accept, log_pi_y, log_pi_x)

    return new_x, new_log_pi, accept.astype(jnp.float32)


# ---------------------------------------------------------------------------
# Dispatcher: run K mutation steps via lax.scan
# ---------------------------------------------------------------------------

def mutate(
    key: jax.Array,
    positions: jnp.ndarray,         # (N, d)
    target,                          # Target protocol
    mutation_config: MutationConfig,
    cholesky_factor=None,            # (d, d) or None
    score_clip: float = 5.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Run K MCMC mutation steps via ``lax.scan``.

    Computes initial ``log_prob`` fresh (post-transport positions have
    no cache).  For MALA, also computes initial clipped scores.

    Uses ``jax.random.fold_in(key, step_idx)`` for per-step key
    derivation inside the scan body.

    Args:
        key: JAX PRNG key.
        positions: Particle positions, shape ``(N, d)``.
        target: Target distribution.
        mutation_config: Mutation configuration (static for JIT —
            ``kernel`` selects the branch at trace time).
        cholesky_factor: Cholesky factor ``(d, d)`` or None.
        score_clip: Score clipping threshold (used for MALA).

    Returns:
        Tuple ``(new_positions, log_prob, scores, info)`` where:
        - ``new_positions``: Updated positions, shape ``(N, d)``.
        - ``log_prob``: Final log π values, shape ``(N,)``.
        - ``scores``: Final clipped scores ``(N, d)`` (zeros for RWM).
        - ``info``: Dict with ``"acceptance_rate"`` (mean over K steps
          and N particles).
    """
    N, d = positions.shape
    h = mutation_config.step_size
    K = mutation_config.n_steps

    # Fresh log-prob evaluation (post-transport positions have no cache)
    log_pi = target.log_prob(positions)  # (N,)

    # Python if — resolved at trace time (config is static)
    if mutation_config.kernel == "mala":
        scores = clip_scores(target.score(positions), score_clip)

        def scan_body(carry, step_idx):
            x, lp, s = carry
            key_step = jax.random.fold_in(key, step_idx)
            new_x, new_lp, new_s, accepted = mala_kernel(
                key_step, x, lp, s, target, h, cholesky_factor, score_clip,
            )
            return (new_x, new_lp, new_s), accepted

        (new_pos, new_lp, new_s), all_accepted = jax.lax.scan(
            scan_body, (positions, log_pi, scores), jnp.arange(K),
        )

        # all_accepted: (K, N) — mean over steps and particles
        info = {"acceptance_rate": jnp.mean(all_accepted)}
        return new_pos, new_lp, new_s, info

    else:  # rwm
        def scan_body(carry, step_idx):
            x, lp = carry
            key_step = jax.random.fold_in(key, step_idx)
            new_x, new_lp, accepted = rwm_kernel(
                key_step, x, lp, target, h, cholesky_factor,
            )
            return (new_x, new_lp), accepted

        (new_pos, new_lp), all_accepted = jax.lax.scan(
            scan_body, (positions, log_pi), jnp.arange(K),
        )

        info = {"acceptance_rate": jnp.mean(all_accepted)}
        return new_pos, new_lp, jnp.zeros((N, d)), info
