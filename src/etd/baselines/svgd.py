"""Stein Variational Gradient Descent (SVGD) baseline.

Deterministic interacting particle system: RBF kernel drives particles
toward the target via a combination of score-weighted kernel transport
and repulsive kernel gradients.

SVGD is deterministic — the key argument is accepted for interface
uniformity but unused.

Reference: Liu & Wang, "Stein Variational Gradient Descent: A General
Purpose Bayesian Inference Algorithm", NeurIPS 2016.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, NamedTuple

import jax
import jax.numpy as jnp

from etd.proposals.langevin import clip_scores


# ---------------------------------------------------------------------------
# Config & State
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SVGDConfig:
    """Configuration for SVGD.

    Attributes:
        n_particles: Number of particles (*N*).
        n_iterations: Total iterations.
        stepsize: Adam learning rate.
        bandwidth: Kernel bandwidth. ``-1.0`` triggers median heuristic.
        clip_score: Maximum score norm (default 5.0).
        adam_b1: Adam beta_1 (default 0.9).
        adam_b2: Adam beta_2 (default 0.999).
        adam_eps: Adam epsilon (default 1e-8).
    """
    n_particles: int = 100
    n_iterations: int = 500
    stepsize: float = 0.01
    bandwidth: float = -1.0
    clip_score: float = 5.0
    adam_b1: float = 0.9
    adam_b2: float = 0.999
    adam_eps: float = 1e-8


class SVGDState(NamedTuple):
    """SVGD state with Adam optimizer moments.

    Attributes:
        positions: Particle positions, shape ``(N, d)``.
        adam_m: First moment estimate, shape ``(N, d)``.
        adam_v: Second moment estimate, shape ``(N, d)``.
        step: Iteration counter.
    """
    positions: jnp.ndarray  # (N, d)
    adam_m: jnp.ndarray      # (N, d)
    adam_v: jnp.ndarray      # (N, d)
    step: int                # scalar


# ---------------------------------------------------------------------------
# Kernel helpers
# ---------------------------------------------------------------------------

def _squared_distances(positions: jnp.ndarray) -> jnp.ndarray:
    """Pairwise squared Euclidean distances via dot-product expansion.

    Args:
        positions: Particle positions, shape ``(N, d)``.

    Returns:
        Squared distance matrix, shape ``(N, N)``.
    """
    xx = jnp.sum(positions ** 2, axis=1)  # (N,)
    sq_dists = xx[:, None] + xx[None, :] - 2.0 * positions @ positions.T
    return jnp.maximum(sq_dists, 0.0)


def _rbf_kernel_and_grad(
    positions: jnp.ndarray,  # (N, d)
    bandwidth: float,        # h (positive scalar)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute RBF kernel matrix and aggregated kernel gradients.

    Args:
        positions: Particle positions, shape ``(N, d)``.
        bandwidth: Kernel bandwidth *h* (the kernel is exp(-||x-y||²/h)).

    Returns:
        Tuple ``(K, grad_K_agg)`` where:
        - ``K``: Kernel matrix, shape ``(N, N)``.
        - ``grad_K_agg``: Sum of kernel-weighted displacements per
          particle, shape ``(N, d)``.
          ``grad_K_agg[i] = (2/h) * sum_j K[j,i] * (x_i - x_j)``.
    """
    sq_dists = _squared_distances(positions)  # (N, N)
    K = jnp.exp(-sq_dists / bandwidth)       # (N, N)

    # Kernel gradient aggregate:
    # ∇_{x_i} K(x_j, x_i) = (2/h) * K(x_j, x_i) * (x_i - x_j)
    # Sum over j: grad_K_agg[i] = sum_j ∇_{x_i} K(x_j, x_i)
    # diff[j, i, :] = x_i - x_j
    diff = positions[None, :, :] - positions[:, None, :]  # (N, N, d)
    grad_K_agg = (2.0 / bandwidth) * jnp.einsum("ji,jid->id", K, diff)

    return K, grad_K_agg


# ---------------------------------------------------------------------------
# Init / Step
# ---------------------------------------------------------------------------

def init(
    key: jax.Array,
    target: object,
    config: SVGDConfig,
    init_positions: Optional[jnp.ndarray] = None,
) -> SVGDState:
    """Initialize SVGD state.

    Args:
        key: JAX PRNG key.
        target: Target distribution (used for ``dim``).
        config: SVGD configuration.
        init_positions: Optional starting positions, shape ``(N, d)``.
            If None, samples from ``N(0, 4I)``.

    Returns:
        Initial :class:`SVGDState`.
    """
    N = config.n_particles
    d = target.dim

    if init_positions is None:
        positions = jax.random.normal(key, (N, d)) * 2.0
    else:
        positions = init_positions

    return SVGDState(
        positions=positions,
        adam_m=jnp.zeros((N, d)),
        adam_v=jnp.zeros((N, d)),
        step=0,
    )


def step(
    key: jax.Array,
    state: SVGDState,
    target: object,
    config: SVGDConfig,
) -> Tuple[SVGDState, Dict[str, jnp.ndarray]]:
    """Execute one SVGD iteration with Adam optimizer.

    SVGD is deterministic — ``key`` is unused.

    Args:
        key: JAX PRNG key (unused; accepted for interface uniformity).
        state: Current SVGD state.
        target: Target distribution with ``score(x)`` method.
        config: SVGD configuration.

    Returns:
        Tuple ``(new_state, info)`` where info contains
        ``"bandwidth"`` and ``"phi_norm"`` (mean SVGD direction norm).
    """
    N = config.n_particles
    positions = state.positions

    # 1. Clipped scores
    scores = clip_scores(target.score(positions), config.clip_score)

    # 2. Bandwidth: median heuristic or fixed
    sq_dists = _squared_distances(positions)
    if config.bandwidth < 0:
        # Median heuristic: h = median(sq_dists) / log(N+1)
        h = jnp.median(sq_dists) / jnp.log(N + 1.0)
        h = jnp.maximum(h, 1e-8)
    else:
        h = config.bandwidth

    # 3. RBF kernel and gradient
    K, grad_K_agg = _rbf_kernel_and_grad(positions, h)

    # 4. SVGD direction: phi = (1/N) * (K^T @ scores + grad_K_agg)
    phi = (K.T @ scores + grad_K_agg) / N  # (N, d)

    # 5. Adam update
    t = state.step + 1
    new_m = config.adam_b1 * state.adam_m + (1.0 - config.adam_b1) * phi
    new_v = config.adam_b2 * state.adam_v + (1.0 - config.adam_b2) * phi ** 2

    # Bias correction
    m_hat = new_m / (1.0 - config.adam_b1 ** t)
    v_hat = new_v / (1.0 - config.adam_b2 ** t)

    new_positions = positions + config.stepsize * m_hat / (jnp.sqrt(v_hat) + config.adam_eps)

    phi_norm = jnp.mean(jnp.linalg.norm(phi, axis=-1))
    info = {"bandwidth": h, "phi_norm": phi_norm}

    return SVGDState(
        positions=new_positions,
        adam_m=new_m,
        adam_v=new_v,
        step=t,
    ), info
