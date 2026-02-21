"""ETD step function — the core algorithm loop.

Wires Phase 0 primitives (propose, cost, couple, update) into a
single iteration of Entropic Transport Descent:

    propose -> IS weights -> cost -> normalize -> couple -> update

Both :func:`init` and :func:`step` are designed for use with
``jax.lax.scan`` or a simple Python loop.
"""

from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from etd.costs import get_cost_fn, median_normalize
from etd.coupling import gibbs_coupling, sinkhorn_log_domain
from etd.proposals.langevin import langevin_proposals, update_preconditioner
from etd.types import ETDConfig, ETDState, Target
from etd.update import systematic_resample
from etd.weights import importance_weights


def init(
    key: jax.Array,
    target: Target,
    config: ETDConfig,
    init_positions: Optional[jnp.ndarray] = None,
) -> ETDState:
    """Initialize the ETD state.

    Args:
        key: JAX PRNG key.
        target: Target distribution (used for ``dim``).
        config: ETD configuration.
        init_positions: Optional starting positions, shape ``(N, d)``.
            If None, samples from ``N(0, 4I)``.

    Returns:
        Initial :class:`ETDState`.
    """
    N = config.n_particles
    d = target.dim
    M = config.n_proposals

    if init_positions is None:
        positions = jax.random.normal(key, (N, d)) * 2.0
    else:
        positions = init_positions

    return ETDState(
        positions=positions,
        dual_f=jnp.zeros(N),
        dual_g=jnp.zeros(N * M),
        precond_accum=jnp.ones(d),
        step=0,
    )


def step(
    key: jax.Array,
    state: ETDState,
    target: Target,
    config: ETDConfig,
) -> Tuple[ETDState, Dict[str, jnp.ndarray]]:
    """Execute one ETD iteration.

    Pipeline: propose -> IS weights -> cost -> normalize -> couple -> update.

    Args:
        key: JAX PRNG key (consumed; do not reuse).
        state: Current ETD state.
        target: Target distribution.
        config: ETD configuration (frozen; used as static arg for JIT).

    Returns:
        Tuple ``(new_state, info)`` where:
        - ``new_state``: Updated :class:`ETDState`.
        - ``info``: Dict with diagnostic keys:
          ``"sinkhorn_iters"``, ``"cost_median"``.
    """
    N = config.n_particles
    key_propose, key_update = jax.random.split(key)

    # --- Preconditioner ---
    preconditioner = None
    if config.precondition:
        P = 1.0 / jnp.sqrt(state.precond_accum + config.precond_delta)
        preconditioner = P

    # --- 1. Propose ---
    proposals, means, scores = langevin_proposals(
        key_propose,
        state.positions,
        target,
        alpha=config.alpha,
        sigma=config.resolved_sigma,
        n_proposals=config.n_proposals,
        use_score=config.use_score,
        score_clip_val=config.score_clip,
        precondition=config.precondition,
        precond_accum=state.precond_accum if config.precondition else None,
        precond_delta=config.precond_delta,
    )

    # --- 2. IS-corrected target weights ---
    log_b = importance_weights(
        proposals, means, target,
        sigma=config.resolved_sigma,
        preconditioner=preconditioner,
    )

    # --- 3. Cost matrix ---
    cost_fn = get_cost_fn(config.cost)
    C = cost_fn(state.positions, proposals)  # (N, N*M)

    # --- 4. Median normalize ---
    C, cost_median = median_normalize(C)

    # --- 5. Source marginal (uniform) ---
    log_a = -jnp.log(N) * jnp.ones(N)

    # --- 6. Coupling ---
    # Python if — resolved at trace time (config is static).
    if config.coupling == "balanced":
        log_gamma, dual_f, dual_g, sinkhorn_iters = sinkhorn_log_domain(
            C, log_a, log_b,
            eps=config.epsilon,
            max_iter=config.sinkhorn_max_iter,
            tol=config.sinkhorn_tol,
            dual_f_init=state.dual_f,
            dual_g_init=state.dual_g,
        )
    elif config.coupling == "gibbs":
        log_gamma, dual_f, dual_g = gibbs_coupling(
            C, log_a, log_b,
            eps=config.epsilon,
        )
        sinkhorn_iters = jnp.int32(0)
    else:
        raise ValueError(f"Unknown coupling '{config.coupling}'")

    # --- 7. Update (systematic resampling) ---
    new_positions = systematic_resample(
        key_update,
        log_gamma,
        proposals,
        step_size=config.step_size,
        positions=state.positions if config.step_size < 1.0 else None,
    )

    # --- 8. Preconditioner accumulator update ---
    if config.precondition:
        new_precond = update_preconditioner(
            state.precond_accum, scores, beta=config.precond_beta,
        )
    else:
        new_precond = state.precond_accum

    # --- Assemble new state ---
    new_state = ETDState(
        positions=new_positions,
        dual_f=dual_f,
        dual_g=dual_g,
        precond_accum=new_precond,
        step=state.step + 1,
    )

    info = {
        "sinkhorn_iters": sinkhorn_iters,
        "cost_median": cost_median,
    }

    return new_state, info
