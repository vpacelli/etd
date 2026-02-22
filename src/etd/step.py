"""ETD step function — the core algorithm loop.

Wires Phase 0 primitives (propose, cost, couple, update) into a
single iteration of Entropic Transport Descent:

    propose -> IS weights -> cost -> normalize -> couple -> update

Both :func:`init` and :func:`step` are designed for use with
``jax.lax.scan`` or a simple Python loop.
"""

import warnings
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from etd.costs import build_cost_fn, normalize_cost
from etd.coupling import gibbs_coupling, sinkhorn_log_domain, sinkhorn_unbalanced
from etd.proposals.langevin import langevin_proposals, update_preconditioner
from etd.schedule import resolve_param
from etd.types import ETDConfig, ETDState, Target
from etd.update import get_update_fn
from etd.weights import importance_weights


def _resolve_sigma(config: ETDConfig, step: int):
    """Resolve sigma, respecting FDR coupling with alpha.

    When ``fdr=True``, σ = √(2α) — if α is scheduled, σ tracks it.
    When ``fdr=False``, σ is resolved independently (may be scheduled).

    Args:
        config: ETD configuration (static for JIT).
        step: Current iteration (may be traced).

    Returns:
        Scalar sigma value.
    """
    if config.fdr:
        alpha = resolve_param(config, "alpha", step)
        return jnp.sqrt(2.0 * alpha)
    return resolve_param(config, "sigma", step)


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
        dv_potential=jnp.zeros(N),
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
          ``"sinkhorn_iters"``, ``"cost_scale"``, ``"coupling_ess"``.
    """
    # --- Resolve cost name + whiten flag (mahalanobis alias) ---
    cost_name = config.cost
    whiten = config.whiten
    if cost_name == "mahalanobis":
        warnings.warn(
            "cost='mahalanobis' is deprecated; use cost='euclidean' with "
            "whiten=True instead.",
            FutureWarning,
            stacklevel=2,
        )
        cost_name = "euclidean"
        whiten = True

    N = config.n_particles
    key_propose, key_update = jax.random.split(key)

    # --- Preconditioner (needed when either precondition or whiten) ---
    preconditioner = None
    if config.needs_precond_accum or whiten:
        P = 1.0 / jnp.sqrt(state.precond_accum + config.precond_delta)
        preconditioner = P

    # --- Resolve scheduled parameters ---
    alpha = resolve_param(config, "alpha", state.step)
    sigma = _resolve_sigma(config, state.step)
    eps = resolve_param(config, "epsilon", state.step)
    step_size = resolve_param(config, "step_size", state.step)

    # --- 1. Propose ---
    proposals, means, scores = langevin_proposals(
        key_propose,
        state.positions,
        target,
        alpha=alpha,
        sigma=sigma,
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
        sigma=sigma,
        preconditioner=preconditioner if config.precondition else None,
    )

    # --- 2b. DV feedback: augment target weights with dual potential ---
    # Uses g from *previous* iteration; on first step dual_g = zeros → no effect.
    if config.dv_feedback:
        dv_weight = resolve_param(config, "dv_weight", state.step)
        log_b = log_b - dv_weight * state.dual_g
        log_b = log_b - logsumexp(log_b)

    # --- 3. Cost matrix (whiten gates preconditioner for cost only) ---
    cost_fn = build_cost_fn(cost_name, config.cost_params)
    C = cost_fn(
        state.positions, proposals,
        preconditioner=preconditioner if whiten else None,
    )  # (N, N*M)

    # --- 4. Normalize cost ---
    C, cost_scale = normalize_cost(C, config.cost_normalize)

    # --- 5. Source marginal (uniform) ---
    log_a = -jnp.log(N) * jnp.ones(N)

    # --- 6. Coupling ---
    # Python if — resolved at trace time (config is static).
    if config.coupling == "balanced":
        log_gamma, dual_f, dual_g, sinkhorn_iters = sinkhorn_log_domain(
            C, log_a, log_b,
            eps=eps,
            max_iter=config.sinkhorn_max_iter,
            tol=config.sinkhorn_tol,
            dual_f_init=state.dual_f,
            dual_g_init=state.dual_g,
        )
    elif config.coupling == "unbalanced":
        log_gamma, dual_f, dual_g, sinkhorn_iters = sinkhorn_unbalanced(
            C, log_a, log_b,
            eps=eps,
            rho=config.rho,
            max_iter=config.sinkhorn_max_iter,
            tol=config.sinkhorn_tol,
            dual_f_init=state.dual_f,
            dual_g_init=state.dual_g,
        )
    elif config.coupling == "gibbs":
        log_gamma, dual_f, dual_g = gibbs_coupling(
            C, log_a, log_b,
            eps=eps,
        )
        sinkhorn_iters = jnp.int32(0)
    else:
        raise ValueError(f"Unknown coupling '{config.coupling}'")

    # --- 7. Update (dispatch by config.update) ---
    update_fn = get_update_fn(config.update)
    new_positions, update_aux = update_fn(
        key_update,
        log_gamma,
        proposals,
        step_size=step_size,
        positions=state.positions,
    )

    # --- 8. Preconditioner accumulator update ---
    # Update when either precondition or whiten is active (or mahalanobis alias).
    if config.needs_precond_accum or whiten:
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
        dv_potential=jnp.zeros(N),  # placeholder — computed in Phase 3
        precond_accum=new_precond,
        step=state.step + 1,
    )

    # --- Coupling ESS: effective proposals per particle ---
    log_gamma_row = log_gamma - logsumexp(log_gamma, axis=1, keepdims=True)
    ess_per_row = 1.0 / jnp.sum(jnp.exp(2.0 * log_gamma_row), axis=1)  # (N,)
    coupling_ess = jnp.mean(ess_per_row)

    info = {
        "sinkhorn_iters": sinkhorn_iters,
        "cost_scale": cost_scale,
        "coupling_ess": coupling_ess,
    }

    return new_state, info
