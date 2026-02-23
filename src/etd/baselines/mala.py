"""Metropolis-Adjusted Langevin Algorithm (MALA) baseline.

MALA = ULA + Metropolis-Hastings correction.  Each particle proposes
via a Langevin step and accepts/rejects using the MH ratio, which
accounts for the asymmetric proposal density.

    Propose:   y = x + h * s(x) + sqrt(2h) * noise
    Accept:    alpha = min(1, pi(y) q(x|y) / (pi(x) q(y|x)))

This eliminates ULA's O(h) discretization bias at the cost of
occasional rejections.  An optional RMSProp preconditioner scales
both drift and noise by P = 1/sqrt(G + delta).

When ``preconditioner.is_cholesky`` is True, a full Cholesky-preconditioned
MALA is used instead, delegating to :func:`etd.primitives.mutation.mala_kernel`.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, NamedTuple

import jax
import jax.numpy as jnp

from etd.primitives.mutation import mala_kernel
from etd.proposals.langevin import clip_scores, update_preconditioner
from etd.proposals.preconditioner import compute_ensemble_cholesky
from etd.types import PreconditionerConfig


# ---------------------------------------------------------------------------
# Config & State
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MALAConfig:
    """Configuration for MALA.

    Attributes:
        n_particles: Number of particles (*N*).
        n_iterations: Total iterations.
        stepsize: Langevin step size (*h*).
        clip_score: Maximum score norm (default 5.0).
        preconditioner: Preconditioner config (diagonal RMSProp or Cholesky).
    """
    n_particles: int = 100
    n_iterations: int = 500
    stepsize: float = 0.01
    clip_score: float = 5.0
    preconditioner: PreconditionerConfig = field(
        default_factory=PreconditionerConfig,
    )


class MALAState(NamedTuple):
    """MALA state.

    Attributes:
        positions: Particle positions, shape ``(N, d)``.
        log_prob: Cached log pi(x) for each particle, shape ``(N,)``.
        scores: Cached clipped scores, shape ``(N, d)``.
        precond_accum: RMSProp accumulator, shape ``(d,)``.
        cholesky_factor: Lower-triangular Cholesky factor, shape ``(d, d)``.
        step: Iteration counter.
    """
    positions: jnp.ndarray      # (N, d)
    log_prob: jnp.ndarray       # (N,)
    scores: jnp.ndarray         # (N, d)
    precond_accum: jnp.ndarray  # (d,)
    cholesky_factor: jnp.ndarray  # (d, d)
    step: int                    # scalar


# ---------------------------------------------------------------------------
# Init / Step
# ---------------------------------------------------------------------------

def init(
    key: jax.Array,
    target: object,
    config: MALAConfig,
    init_positions: Optional[jnp.ndarray] = None,
) -> MALAState:
    """Initialize MALA state.

    Caches ``log_prob`` and clipped scores at initial positions so the
    first ``step()`` does not redundantly evaluate them.

    Args:
        key: JAX PRNG key.
        target: Target distribution (``dim``, ``log_prob``, ``score``).
        config: MALA configuration.
        init_positions: Optional starting positions, shape ``(N, d)``.
            If None, samples from ``N(0, 4I)``.

    Returns:
        Initial :class:`MALAState`.
    """
    if init_positions is None:
        positions = jax.random.normal(key, (config.n_particles, target.dim)) * 2.0
    else:
        positions = init_positions

    d = positions.shape[-1]
    log_p = target.log_prob(positions)                         # (N,)
    scores = clip_scores(target.score(positions), config.clip_score)  # (N, d)
    precond_accum = jnp.ones(d)                                # (d,)

    pc = config.preconditioner
    if pc.is_cholesky:
        cholesky_factor = compute_ensemble_cholesky(
            positions, jnp.eye(d), pc,
        )
    else:
        cholesky_factor = jnp.eye(d)

    return MALAState(
        positions=positions,
        log_prob=log_p,
        scores=scores,
        precond_accum=precond_accum,
        cholesky_factor=cholesky_factor,
        step=0,
    )


def step(
    key: jax.Array,
    state: MALAState,
    target: object,
    config: MALAConfig,
) -> Tuple[MALAState, Dict[str, jnp.ndarray]]:
    """Execute one MALA iteration.

    1. Propose via preconditioned Langevin step.
    2. Compute forward / reverse proposal log-densities.
    3. Accept / reject each particle independently (MH).
    4. Update preconditioner if enabled.

    When ``config.preconditioner.is_cholesky`` is True, delegates to
    :func:`mala_kernel` for the full Cholesky-preconditioned step.

    Args:
        key: JAX PRNG key (consumed; do not reuse).
        state: Current MALA state.
        target: Target with ``log_prob(x)`` and ``score(x)`` methods.
        config: MALA configuration.

    Returns:
        Tuple ``(new_state, info)`` where info contains
        ``"acceptance_rate"`` and ``"score_norm"``.
    """
    h = config.stepsize
    x = state.positions                # (N, d)
    log_pi_x = state.log_prob          # (N,)
    s_x = state.scores                 # (N, d)
    pc = config.preconditioner

    if pc.is_cholesky:
        # --- Cholesky-preconditioned MALA via shared kernel ---
        new_positions, new_log_prob, new_scores, accept_float = mala_kernel(
            key, x, log_pi_x, s_x, target,
            h=h, L=state.cholesky_factor, score_clip=config.clip_score,
        )

        # Update Cholesky from post-step positions
        new_cholesky = compute_ensemble_cholesky(
            new_positions, state.cholesky_factor, pc,
        )

        acceptance_rate = jnp.mean(accept_float)
        score_norm = jnp.mean(jnp.linalg.norm(new_scores, axis=-1))

        new_state = MALAState(
            positions=new_positions,
            log_prob=new_log_prob,
            scores=new_scores,
            precond_accum=state.precond_accum,
            cholesky_factor=new_cholesky,
            step=state.step + 1,
        )
        info = {"acceptance_rate": acceptance_rate, "score_norm": score_norm}
        return new_state, info

    # --- Diagonal path (RMSProp or identity) ---

    if pc.is_rmsprop:
        P = 1.0 / jnp.sqrt(state.precond_accum + pc.delta)  # (d,)
    else:
        P = jnp.ones(x.shape[-1])      # (d,)  -- identity

    # P^2 for the quadratic form in proposal density
    P2 = P * P                          # (d,)

    # --- Propose ---
    k_noise, k_accept = jax.random.split(key)
    noise = jax.random.normal(k_noise, x.shape)    # (N, d)

    # Proposal mean: mu_fwd = x + h * P * s_x
    mu_fwd = x + h * (P * s_x)                     # (N, d)
    y = mu_fwd + jnp.sqrt(2.0 * h) * P * noise     # (N, d)

    # --- Evaluate target at proposals ---
    log_pi_y = target.log_prob(y)                              # (N,)
    s_y = clip_scores(target.score(y), config.clip_score)      # (N, d)

    # --- Proposal log-densities (normalization cancels) ---
    # Forward: log q(y | x)
    diff_fwd = y - mu_fwd                                     # (N, d)
    log_q_fwd = -0.5 * jnp.sum(diff_fwd ** 2 / (2.0 * h * P2), axis=-1)  # (N,)

    # Reverse mean: mu_rev = y + h * P * s_y
    mu_rev = y + h * (P * s_y)                                # (N, d)
    diff_rev = x - mu_rev                                     # (N, d)
    log_q_rev = -0.5 * jnp.sum(diff_rev ** 2 / (2.0 * h * P2), axis=-1)  # (N,)

    # --- MH acceptance ---
    log_alpha = log_pi_y - log_pi_x + log_q_rev - log_q_fwd   # (N,)
    log_alpha = jnp.minimum(log_alpha, 0.0)                    # clamp to 0

    log_u = jnp.log(jax.random.uniform(k_accept, shape=log_alpha.shape))
    accept = log_u < log_alpha                                 # (N,)

    # --- Accept / reject ---
    new_positions = jnp.where(accept[:, None], y, x)           # (N, d)
    new_log_prob = jnp.where(accept, log_pi_y, log_pi_x)      # (N,)
    new_scores = jnp.where(accept[:, None], s_y, s_x)         # (N, d)

    # --- Update preconditioner ---
    if pc.is_rmsprop:
        new_accum = update_preconditioner(
            state.precond_accum, new_scores, pc.beta
        )
    else:
        new_accum = state.precond_accum

    # --- Info ---
    acceptance_rate = jnp.mean(accept.astype(jnp.float32))
    score_norm = jnp.mean(jnp.linalg.norm(new_scores, axis=-1))
    info = {"acceptance_rate": acceptance_rate, "score_norm": score_norm}

    new_state = MALAState(
        positions=new_positions,
        log_prob=new_log_prob,
        scores=new_scores,
        precond_accum=new_accum,
        cholesky_factor=state.cholesky_factor,
        step=state.step + 1,
    )
    return new_state, info
