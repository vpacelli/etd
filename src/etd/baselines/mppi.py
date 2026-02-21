"""Model Predictive Path Integral (MPPI) baseline.

Per-particle importance-weighted averaging — proposals are generated
independently for each particle (N, M, d), NOT pooled.  This is the
critical difference from ETD: no inter-particle interaction.

    proposals = x_i + sigma * noise       per particle
    w_ij = softmax(log_pi(y_ij) / T)      per particle
    x_i' = sum_j w_ij * y_ij              barycentric update
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, NamedTuple

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Config & State
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MPPIConfig:
    """Configuration for MPPI.

    Attributes:
        n_particles: Number of particles (*N*).
        n_iterations: Total iterations.
        n_proposals: Proposals per particle (*M*).
        temperature: Softmax temperature (*T*).
        sigma: Proposal noise scale.
    """
    n_particles: int = 100
    n_iterations: int = 500
    n_proposals: int = 25
    temperature: float = 1.0
    sigma: float = 0.316


class MPPIState(NamedTuple):
    """MPPI state.

    Attributes:
        positions: Particle positions, shape ``(N, d)``.
        step: Iteration counter.
    """
    positions: jnp.ndarray  # (N, d)
    step: int                # scalar


# ---------------------------------------------------------------------------
# Init / Step
# ---------------------------------------------------------------------------

def init(
    key: jax.Array,
    target: object,
    config: MPPIConfig,
    init_positions: Optional[jnp.ndarray] = None,
) -> MPPIState:
    """Initialize MPPI state.

    Args:
        key: JAX PRNG key.
        target: Target distribution (used for ``dim``).
        config: MPPI configuration.
        init_positions: Optional starting positions, shape ``(N, d)``.
            If None, samples from ``N(0, 4I)``.

    Returns:
        Initial :class:`MPPIState`.
    """
    if init_positions is None:
        positions = jax.random.normal(key, (config.n_particles, target.dim)) * 2.0
    else:
        positions = init_positions

    return MPPIState(positions=positions, step=0)


def step(
    key: jax.Array,
    state: MPPIState,
    target: object,
    config: MPPIConfig,
) -> Tuple[MPPIState, Dict[str, jnp.ndarray]]:
    """Execute one MPPI iteration.

    Per-particle proposals (N, M, d) — NOT pooled across particles.
    Uses importance-weighted barycentric update.

    Args:
        key: JAX PRNG key (consumed; do not reuse).
        state: Current MPPI state.
        target: Target distribution with ``log_prob(x)`` method.
        config: MPPI configuration.

    Returns:
        Tuple ``(new_state, info)`` where info contains
        ``"log_w_max"`` (mean max log-weight per particle).
    """
    N = config.n_particles
    M = config.n_proposals
    d = state.positions.shape[1]

    # 1. Per-particle proposals: (N, M, d)
    noise = jax.random.normal(key, (N, M, d))
    proposals = state.positions[:, None, :] + config.sigma * noise  # (N, M, d)

    # 2. Evaluate log-density at all proposals
    log_pi = target.log_prob(proposals.reshape(N * M, d)).reshape(N, M)

    # 3. Softmax weights (log-domain for stability)
    log_w = jax.nn.log_softmax(log_pi / config.temperature, axis=1)  # (N, M)
    w = jnp.exp(log_w)  # (N, M)

    # 4. Barycentric update
    new_positions = jnp.einsum("nm,nmd->nd", w, proposals)  # (N, d)

    log_w_max = jnp.mean(jnp.max(log_w, axis=1))
    info = {"log_w_max": log_w_max}

    return MPPIState(positions=new_positions, step=state.step + 1), info
