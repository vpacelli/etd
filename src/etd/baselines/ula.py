"""Unadjusted Langevin Algorithm (ULA) baseline.

N independent Langevin chains — no inter-particle interaction.
The simplest score-based sampler: each particle follows noisy
gradient ascent on the log-density.

    x_{t+1} = x_t + h * score(x_t) + sqrt(2h) * noise
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
class ULAConfig:
    """Configuration for ULA.

    Attributes:
        n_particles: Number of particles (*N*).
        n_iterations: Total iterations.
        stepsize: Langevin step size (*h*).
        clip_score: Maximum score norm (default 5.0).
    """
    n_particles: int = 100
    n_iterations: int = 500
    stepsize: float = 0.01
    clip_score: float = 5.0


class ULAState(NamedTuple):
    """ULA state.

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
    config: ULAConfig,
    init_positions: Optional[jnp.ndarray] = None,
) -> ULAState:
    """Initialize ULA state.

    Args:
        key: JAX PRNG key.
        target: Target distribution (used for ``dim``).
        config: ULA configuration.
        init_positions: Optional starting positions, shape ``(N, d)``.
            If None, samples from ``N(0, 4I)``.

    Returns:
        Initial :class:`ULAState`.
    """
    if init_positions is None:
        positions = jax.random.normal(key, (config.n_particles, target.dim)) * 2.0
    else:
        positions = init_positions

    return ULAState(positions=positions, step=0)


def step(
    key: jax.Array,
    state: ULAState,
    target: object,
    config: ULAConfig,
) -> Tuple[ULAState, Dict[str, jnp.ndarray]]:
    """Execute one ULA iteration.

    Args:
        key: JAX PRNG key (consumed; do not reuse).
        state: Current ULA state.
        target: Target distribution with ``score(x)`` method.
        config: ULA configuration.

    Returns:
        Tuple ``(new_state, info)`` where info contains
        ``"score_norm"`` (mean score norm).
    """
    h = config.stepsize
    scores = clip_scores(target.score(state.positions), config.clip_score)
    noise = jax.random.normal(key, state.positions.shape)

    new_positions = state.positions + h * scores + jnp.sqrt(2.0 * h) * noise

    score_norm = jnp.mean(jnp.linalg.norm(scores, axis=-1))
    info = {"score_norm": score_norm}

    return ULAState(positions=new_positions, step=state.step + 1), info
