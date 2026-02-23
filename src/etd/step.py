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
from jax.scipy.special import logsumexp

from etd.costs import build_cost_fn, normalize_cost
from etd.costs.langevin import langevin_residual_cost
from etd.coupling import gibbs_coupling, sinkhorn_log_domain, sinkhorn_unbalanced
from etd.primitives.mutation import mutate
from etd.proposals.langevin import (
    clip_scores,
    langevin_proposals,
)
from etd.proposals.preconditioner import (
    compute_ensemble_cholesky,
    update_rmsprop_accum,
)
from etd.schedule import resolve_param
from etd.types import ETDConfig, ETDState, PreconditionerConfig, Target
from etd.update import get_update_fn
from etd.weights import importance_weights


def _resolve_sigma(config: ETDConfig, step: int):
    """Resolve sigma, respecting FDR coupling with alpha.

    When ``fdr=True``, sigma = sqrt(2*alpha) — if alpha is scheduled, sigma tracks it.
    When ``fdr=False``, sigma is resolved independently (may be scheduled).

    Args:
        config: ETD configuration (static for JIT).
        step: Current iteration (may be traced).

    Returns:
        Scalar sigma value.
    """
    if config.proposal.fdr:
        alpha = resolve_param(config, "proposal.alpha", step)
        return jnp.sqrt(2.0 * alpha)
    return resolve_param(config, "proposal.sigma", step)


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
    M = config.proposal.count

    if init_positions is None:
        positions = jax.random.normal(key, (N, d)) * 2.0
    else:
        positions = init_positions

    # Compute initial Cholesky factor from particle positions.
    # Always use positions at init (not scores): particles start far from the
    # target, so score covariance is unreliable.  The source setting governs
    # updates during step().
    pc = config.preconditioner
    if pc.is_cholesky:
        cholesky_factor = compute_ensemble_cholesky(
            positions, jnp.eye(d), pc,
        )
    elif config.needs_cholesky:
        # Mutation needs Cholesky but preconditioner doesn't provide it
        _auto_pc = PreconditionerConfig(
            type="cholesky", shrinkage=0.1, jitter=1e-6,
        )
        cholesky_factor = compute_ensemble_cholesky(
            positions, jnp.eye(d), _auto_pc,
        )
    else:
        cholesky_factor = jnp.eye(d)

    return ETDState(
        positions=positions,
        dual_f=jnp.zeros(N),
        dual_g=jnp.zeros(N * M),
        dv_potential=jnp.zeros(N),
        log_prob=jnp.zeros(N),
        scores=jnp.zeros((N, d)),
        precond_accum=jnp.ones(d),
        cholesky_factor=cholesky_factor,
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
    N = config.n_particles
    pc = config.preconditioner   # shorthand
    key_propose, key_update, key_mutate = jax.random.split(key, 3)

    # --- Resolve scheduled parameters ---
    alpha = resolve_param(config, "proposal.alpha", state.step)
    sigma = _resolve_sigma(config, state.step)
    eps = resolve_param(config, "epsilon", state.step)
    damping = resolve_param(config, "update.damping", state.step)

    # --- Preconditioner dispatch (trace-time branching) ---
    cholesky_for_proposals = None
    cholesky_for_cost = None
    diag_for_proposals = None
    diag_for_cost = None

    if pc.is_cholesky:
        if pc.proposals:
            cholesky_for_proposals = state.cholesky_factor
        if pc.cost:
            cholesky_for_cost = state.cholesky_factor
    elif pc.is_rmsprop:
        P_new = 1.0 / jnp.sqrt(state.precond_accum + pc.delta)
        if pc.proposals:
            diag_for_proposals = P_new
        if pc.cost:
            diag_for_cost = P_new

    # --- 1. Propose ---
    proposals, means, scores = langevin_proposals(
        key_propose,
        state.positions,
        target,
        alpha=alpha,
        sigma=sigma,
        n_proposals=config.proposal.count,
        use_score=config.proposal.use_score,
        score_clip_val=config.proposal.clip_score,
        cholesky_factor=cholesky_for_proposals,
    )

    # --- 2. IS-corrected target weights ---
    if pc.is_cholesky and pc.proposals:
        log_b = importance_weights(
            proposals, means, target, sigma=sigma,
            cholesky_factor=cholesky_for_proposals,
        )
    elif pc.is_rmsprop and pc.proposals:
        log_b = importance_weights(
            proposals, means, target, sigma=sigma,
            preconditioner=diag_for_proposals,
        )
    else:
        log_b = importance_weights(
            proposals, means, target, sigma=sigma,
        )

    # --- 2b. DV feedback: augment target weights with clean per-particle signal ---
    # Uses dv_potential from *previous* iteration (zeros on first step -> no-op).
    # dv_potential is (N,); expand to (N*M,) matching proposal ordering.
    if config.feedback.enabled:
        dv_weight = resolve_param(config, "feedback.weight", state.step)
        dv_expanded = jnp.repeat(state.dv_potential, config.proposal.count)  # (N*M,)
        log_b = log_b - dv_weight * dv_expanded
        log_b = log_b - logsumexp(log_b)

    # --- 3. Cost matrix ---
    cost_name = config.cost.type
    if cost_name == "langevin":
        # Inline Langevin-residual cost (approach B).
        cost_whiten = config.cost.whiten
        chol = cholesky_for_cost if cost_whiten else None
        C = langevin_residual_cost(
            state.positions, proposals, scores, eps,
            cholesky_factor=chol, whiten=cost_whiten,
        )
    else:
        cost_fn = build_cost_fn(cost_name, config.cost.params)
        if cholesky_for_cost is not None:
            C = cost_fn(
                state.positions, proposals,
                cholesky_factor=cholesky_for_cost,
            )
        elif diag_for_cost is not None:
            C = cost_fn(
                state.positions, proposals,
                preconditioner=diag_for_cost,
            )
        else:
            C = cost_fn(state.positions, proposals)
    # C shape: (N, N*M)

    # --- 4. Normalize cost ---
    C, cost_scale = normalize_cost(C, config.cost.normalize)

    # --- 5. Source marginal (uniform) ---
    log_a = -jnp.log(N) * jnp.ones(N)

    # --- 6. Coupling ---
    # Python if -- resolved at trace time (config is static).
    if config.coupling.type == "balanced":
        log_gamma, dual_f, dual_g, sinkhorn_iters = sinkhorn_log_domain(
            C, log_a, log_b,
            eps=eps,
            max_iter=config.coupling.iterations,
            tol=config.coupling.tolerance,
            dual_f_init=state.dual_f,
            dual_g_init=state.dual_g,
        )
    elif config.coupling.type == "unbalanced":
        log_gamma, dual_f, dual_g, sinkhorn_iters = sinkhorn_unbalanced(
            C, log_a, log_b,
            eps=eps,
            rho=config.coupling.rho,
            max_iter=config.coupling.iterations,
            tol=config.coupling.tolerance,
            dual_f_init=state.dual_f,
            dual_g_init=state.dual_g,
        )
    elif config.coupling.type == "gibbs":
        log_gamma, dual_f, dual_g = gibbs_coupling(
            C, log_a, log_b,
            eps=eps,
        )
        sinkhorn_iters = jnp.int32(0)
    else:
        raise ValueError(f"Unknown coupling '{config.coupling.type}'")

    # --- 7. Update (dispatch by config.update.type) ---
    update_fn = get_update_fn(config.update.type)
    new_positions, update_aux = update_fn(
        key_update,
        log_gamma,
        proposals,
        step_size=damping,
        positions=state.positions,
    )

    # --- 7b. Compute per-particle DV potential (c-transform + interpolation) ---
    if config.feedback.enabled:
        if config.coupling.type == "balanced":
            g_tilde = dual_g - eps * log_b
        elif config.coupling.type == "unbalanced":
            lam = config.coupling.rho / (1.0 + config.coupling.rho)
            g_tilde = dual_g - eps * lam * log_b
        else:  # gibbs
            g_tilde = jnp.zeros_like(dual_g)

        if config.update.type == "categorical":
            selected_g = g_tilde[update_aux["indices"]]          # (N,)
        else:  # barycentric
            selected_g = update_aux["weights"] @ g_tilde         # (N,)

        dv_potential = (1.0 - damping) * dual_f + damping * selected_g
    else:
        dv_potential = jnp.zeros(N)

    # --- 8. MCMC Mutation (post-transport local refinement) ---
    mut = config.mutation
    if mut.active:
        # Resolve Cholesky for mutation
        if mut.cholesky and not pc.is_cholesky:
            _auto_pc = PreconditionerConfig(
                type="cholesky", shrinkage=0.1, jitter=1e-6,
            )
            mut_cholesky = compute_ensemble_cholesky(
                new_positions, state.cholesky_factor, _auto_pc,
            )
        elif mut.cholesky:
            mut_cholesky = state.cholesky_factor
        else:
            mut_cholesky = None

        mut_clip = (
            mut.clip_score if mut.clip_score is not None
            else config.proposal.clip_score
        )

        new_positions, new_log_prob, new_scores, mut_info = mutate(
            key_mutate, new_positions, target, mut,
            cholesky_factor=mut_cholesky, score_clip=mut_clip,
        )
    else:
        new_log_prob = jnp.zeros(N)
        new_scores = jnp.zeros((N, target.dim))
        mut_info = {}

    # --- 9. Preconditioner update ---
    # Use post-mutation positions so Cholesky reflects latest particle spread
    new_precond = state.precond_accum
    new_cholesky = state.cholesky_factor

    if pc.is_cholesky:
        # Select data source for covariance: scores or positions
        if pc.source == "scores":
            if pc.use_raw_scores and config.proposal.use_score:
                # Compute raw (unclipped) scores for preconditioner
                precond_data = target.score(new_positions)  # (N, d)
            else:
                precond_data = scores  # clipped scores from proposal step
        else:  # source == "positions"
            precond_data = new_positions

        new_cholesky = compute_ensemble_cholesky(
            precond_data, state.cholesky_factor, pc,
        )
    elif pc.is_rmsprop:
        new_precond = update_rmsprop_accum(
            state.precond_accum, scores, beta=pc.beta,
        )

    # --- Assemble new state ---
    new_state = ETDState(
        positions=new_positions,
        dual_f=dual_f,
        dual_g=dual_g,
        dv_potential=dv_potential,
        log_prob=new_log_prob,
        scores=new_scores,
        precond_accum=new_precond,
        cholesky_factor=new_cholesky,
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
    if mut.active:
        info["mutation_acceptance_rate"] = mut_info["acceptance_rate"]

    return new_state, info
