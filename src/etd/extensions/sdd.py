"""Stein Discrepancy Descent with Barycentric coupling (SDD-RB).

SDD combines cross-coupling (like ETD) with self-coupling between
particles.  The cross-coupling transports particles toward high-density
proposals; the self-coupling provides repulsion that preserves diversity.

SDD update:
    x_new[i] = x[i] + eta * (y_cross[i] - x_bar_self[i])

where:
    y_cross[i]    : resampled proposal from cross-coupling
    x_bar_self[i] : barycentric mean from self-coupling (N x N)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from etd.costs import build_cost_fn, normalize_cost
from etd.costs.langevin import langevin_residual_cost
from etd.coupling import sinkhorn_log_domain
from etd.primitives.mutation import mutate
from etd.proposals.langevin import langevin_proposals
from etd.proposals.preconditioner import (
    compute_ensemble_cholesky,
    update_rmsprop_accum,
)
from etd.schedule import resolve_param
from etd.types import (
    CostConfig,
    CouplingConfig,
    FeedbackConfig,
    MutationConfig,
    PreconditionerConfig,
    ProposalConfig,
    SelfCouplingConfig,
    Target,
    UpdateConfig,
)
from etd.update import systematic_resample
from etd.weights import importance_weights


# ---------------------------------------------------------------------------
# State and config
# ---------------------------------------------------------------------------

class SDDState(NamedTuple):
    """Mutable state for SDD-RB.

    All fields are concrete arrays for pytree stability.

    Attributes:
        positions: Particle positions, shape ``(N, d)``.
        dual_f_cross: Cross-coupling source duals, shape ``(N,)``.
        dual_g_cross: Cross-coupling target duals, shape ``(N*M,)``.
        dual_f_self: Self-coupling source duals, shape ``(N,)``.
        dual_g_self: Self-coupling target duals, shape ``(N,)``.
        dv_potential: Per-particle DV feedback signal, shape ``(N,)``.
        log_prob: Cached log pi(x) from mutation, shape ``(N,)``.
            Zeros when mutation is off.
        scores: Cached clipped scores from mutation, shape ``(N, d)``.
            Zeros when mutation is off or using RWM.
        precond_accum: RMSProp accumulator, shape ``(d,)``.
        cholesky_factor: Lower-triangular Cholesky factor, shape ``(d, d)``.
        step: Iteration counter.
    """

    positions: jnp.ndarray       # (N, d)
    dual_f_cross: jnp.ndarray    # (N,)
    dual_g_cross: jnp.ndarray    # (N*M,)
    dual_f_self: jnp.ndarray     # (N,)
    dual_g_self: jnp.ndarray     # (N,)
    dv_potential: jnp.ndarray    # (N,)
    log_prob: jnp.ndarray        # (N,)
    scores: jnp.ndarray          # (N, d)
    precond_accum: jnp.ndarray   # (d,)
    cholesky_factor: jnp.ndarray  # (d, d)
    step: int


@dataclass(frozen=True)
class SDDConfig:
    """Configuration for SDD-RB.

    Mirrors core ETD sub-configs plus SDD-specific parameters.
    """

    # --- Scale ---
    n_particles: int = 100
    n_iterations: int = 500

    # --- Core ---
    epsilon: float = 0.1

    # --- Nested sub-configs ---
    proposal: ProposalConfig = field(default_factory=ProposalConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    coupling: CouplingConfig = field(default_factory=CouplingConfig)
    update: UpdateConfig = field(default_factory=UpdateConfig)
    preconditioner: PreconditionerConfig = field(
        default_factory=PreconditionerConfig,
    )
    mutation: MutationConfig = field(default_factory=MutationConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)

    # --- SDD-specific ---
    self_coupling: SelfCouplingConfig = field(
        default_factory=SelfCouplingConfig,
    )
    eta: float = 0.5  # SDD displacement step size

    # --- Schedules ---
    schedules: tuple = ()

    def __post_init__(self):
        if (
            self.mutation.active
            and self.mutation.cholesky
            and not self.preconditioner.is_cholesky
        ):
            warnings.warn(
                "mutation.cholesky=True but preconditioner.type is not "
                "'cholesky'. A Cholesky factor will be auto-computed for "
                "mutation each step (O(Nd^2 + d^3)). Consider enabling "
                "Cholesky preconditioning to share the computation.",
                UserWarning,
                stacklevel=2,
            )

    @property
    def needs_cholesky(self) -> bool:
        """Whether a Cholesky factor is needed (preconditioner or mutation)."""
        return self.preconditioner.is_cholesky or (
            self.mutation.active and self.mutation.cholesky
        )

    @property
    def needs_precond_accum(self) -> bool:
        """Whether the RMSProp accumulator is needed."""
        return self.preconditioner.active


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

def init(
    key: jax.Array,
    target: Target,
    config: SDDConfig,
    init_positions: Optional[jnp.ndarray] = None,
) -> SDDState:
    """Initialize SDD state.

    Args:
        key: JAX PRNG key.
        target: Target distribution.
        config: SDD configuration.
        init_positions: Optional starting positions ``(N, d)``.

    Returns:
        Initial :class:`SDDState`.
    """
    N = config.n_particles
    d = target.dim
    M = config.proposal.count

    if init_positions is None:
        positions = jax.random.normal(key, (N, d)) * 2.0
    else:
        positions = init_positions

    # Compute initial Cholesky factor from particle positions.
    pc = config.preconditioner
    if pc.is_cholesky:
        cholesky_factor = compute_ensemble_cholesky(
            positions, jnp.eye(d), pc,
        )
    elif config.needs_cholesky:
        _auto_pc = PreconditionerConfig(
            type="cholesky", shrinkage=0.1, jitter=1e-6,
        )
        cholesky_factor = compute_ensemble_cholesky(
            positions, jnp.eye(d), _auto_pc,
        )
    else:
        cholesky_factor = jnp.eye(d)

    return SDDState(
        positions=positions,
        dual_f_cross=jnp.zeros(N),
        dual_g_cross=jnp.zeros(N * M),
        dual_f_self=jnp.zeros(N),
        dual_g_self=jnp.zeros(N),
        dv_potential=jnp.zeros(N),
        log_prob=jnp.zeros(N),
        scores=jnp.zeros((N, d)),
        precond_accum=jnp.ones(d),
        cholesky_factor=cholesky_factor,
        step=0,
    )


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

def _resolve_sigma(config: SDDConfig, step: int):
    """Resolve sigma, respecting FDR."""
    if config.proposal.fdr:
        alpha = resolve_param(config, "proposal.alpha", step)
        return jnp.sqrt(2.0 * alpha)
    return resolve_param(config, "proposal.sigma", step)


def step(
    key: jax.Array,
    state: SDDState,
    target: Target,
    config: SDDConfig,
) -> Tuple[SDDState, Dict[str, jnp.ndarray]]:
    """Execute one SDD-RB iteration.

    Pipeline:
        1. Cross-coupling (identical to ETD):
           propose -> IS weights -> cost -> Sinkhorn -> resample -> y_cross
        2. Self-coupling:
           N x N cost between particles -> balanced Sinkhorn -> barycentric mean -> x_bar_self
        3. SDD displacement:
           x_new[i] = x[i] + eta * (y_cross[i] - x_bar_self[i])

    Args:
        key: JAX PRNG key.
        state: Current SDD state.
        target: Target distribution.
        config: SDD configuration.

    Returns:
        Tuple ``(new_state, info)``.
    """
    N = config.n_particles
    pc = config.preconditioner
    key_propose, key_resample, key_mutate = jax.random.split(key, 3)

    # --- Preconditioner dispatch ---
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

    # --- Resolve scheduled parameters ---
    alpha = resolve_param(config, "proposal.alpha", state.step)
    sigma = _resolve_sigma(config, state.step)
    eps = resolve_param(config, "epsilon", state.step)
    sdd_eta = config.eta

    # ===================================================================
    # 1. Cross-coupling (identical to ETD balanced)
    # ===================================================================

    # 1a. Propose
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

    # 1b. IS-corrected target weights
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

    # 1c. DV feedback
    if config.feedback.enabled:
        dv_weight = resolve_param(config, "feedback.weight", state.step)
        dv_expanded = jnp.repeat(state.dv_potential, config.proposal.count)
        log_b = log_b - dv_weight * dv_expanded
        log_b = log_b - logsumexp(log_b)

    # 1d. Cost matrix (cross)
    cost_name = config.cost.type
    if cost_name == "langevin":
        cost_whiten = config.cost.whiten
        chol = cholesky_for_cost if cost_whiten else None
        C_cross = langevin_residual_cost(
            state.positions, proposals, scores, eps,
            cholesky_factor=chol, whiten=cost_whiten,
        )
    else:
        cost_fn = build_cost_fn(cost_name, config.cost.params)
        if cholesky_for_cost is not None:
            C_cross = cost_fn(
                state.positions, proposals, cholesky_factor=cholesky_for_cost,
            )
        elif diag_for_cost is not None:
            C_cross = cost_fn(
                state.positions, proposals, preconditioner=diag_for_cost,
            )
        else:
            C_cross = cost_fn(state.positions, proposals)
    C_cross, cost_scale_cross = normalize_cost(C_cross, config.cost.normalize)

    # 1e. Source marginal (uniform)
    log_a = -jnp.log(N) * jnp.ones(N)

    # 1f. Balanced Sinkhorn (cross)
    log_gamma_cross, dual_f_cross, dual_g_cross, cross_iters = sinkhorn_log_domain(
        C_cross, log_a, log_b,
        eps=eps,
        max_iter=config.coupling.iterations,
        tol=config.coupling.tolerance,
        dual_f_init=state.dual_f_cross,
        dual_g_init=state.dual_g_cross,
    )

    # 1g. Resample: y_cross (one proposal per particle)
    y_cross, cross_aux = systematic_resample(
        key_resample,
        log_gamma_cross,
        proposals,
        step_size=config.update.damping,
        positions=state.positions,
    )

    # 1h. DV potential
    if config.feedback.enabled:
        g_tilde_cross = dual_g_cross - eps * log_b
        selected_g = g_tilde_cross[cross_aux["indices"]]
        cross_damping = resolve_param(config, "update.damping", state.step)
        dv_potential = (
            (1.0 - cross_damping) * dual_f_cross + cross_damping * selected_g
        )
    else:
        dv_potential = jnp.zeros(N)

    # ===================================================================
    # 2. Self-coupling (N x N between particles)
    # ===================================================================

    # 2a. Self-cost matrix (Euclidean for self — Langevin has no self reference)
    self_cost_name = "euclidean" if cost_name == "langevin" else cost_name
    self_cost_params = () if cost_name == "langevin" else config.cost.params
    self_cost_fn = build_cost_fn(self_cost_name, self_cost_params)
    if cholesky_for_cost is not None:
        C_self = self_cost_fn(
            state.positions, state.positions, cholesky_factor=cholesky_for_cost,
        )
    elif diag_for_cost is not None:
        C_self = self_cost_fn(
            state.positions, state.positions, preconditioner=diag_for_cost,
        )
    else:
        C_self = self_cost_fn(state.positions, state.positions)
    C_self, cost_scale_self = normalize_cost(C_self, config.cost.normalize)

    # 2b. Uniform marginals for self-coupling
    log_a_self = -jnp.log(N) * jnp.ones(N)
    log_b_self = -jnp.log(N) * jnp.ones(N)

    # 2c. Balanced Sinkhorn (self)
    log_gamma_self, dual_f_self, dual_g_self, self_iters = sinkhorn_log_domain(
        C_self, log_a_self, log_b_self,
        eps=config.self_coupling.epsilon,
        max_iter=config.self_coupling.iterations,
        tol=config.self_coupling.tolerance,
        dual_f_init=state.dual_f_self,
        dual_g_init=state.dual_g_self,
    )

    # 2d. Barycentric self-mean
    log_weights_self = log_gamma_self - logsumexp(
        log_gamma_self, axis=1, keepdims=True,
    )
    weights_self = jnp.exp(log_weights_self)
    x_bar_self = weights_self @ state.positions

    # ===================================================================
    # 3. SDD displacement
    # ===================================================================
    new_positions = state.positions + sdd_eta * (y_cross - x_bar_self)

    # --- MCMC Mutation (post-transport local refinement) ---
    mut = config.mutation
    if mut.active:
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

    # --- Preconditioner update ---
    new_precond = state.precond_accum
    new_cholesky = state.cholesky_factor

    if pc.is_cholesky:
        if pc.source == "scores":
            if pc.use_raw_scores and config.proposal.use_score:
                precond_data = target.score(new_positions)
            else:
                precond_data = scores
        else:
            precond_data = new_positions
        new_cholesky = compute_ensemble_cholesky(
            precond_data, state.cholesky_factor, pc,
        )
    elif pc.is_rmsprop:
        new_precond = update_rmsprop_accum(
            state.precond_accum, scores, beta=pc.beta,
        )

    # --- Assemble new state ---
    new_state = SDDState(
        positions=new_positions,
        dual_f_cross=dual_f_cross,
        dual_g_cross=dual_g_cross,
        dual_f_self=dual_f_self,
        dual_g_self=dual_g_self,
        dv_potential=dv_potential,
        log_prob=new_log_prob,
        scores=new_scores,
        precond_accum=new_precond,
        cholesky_factor=new_cholesky,
        step=state.step + 1,
    )

    info = {
        "sinkhorn_iters_cross": cross_iters,
        "sinkhorn_iters_self": self_iters,
        "cost_scale_cross": cost_scale_cross,
        "cost_scale_self": cost_scale_self,
    }
    if mut.active:
        info["mutation_acceptance_rate"] = mut_info["acceptance_rate"]

    return new_state, info
