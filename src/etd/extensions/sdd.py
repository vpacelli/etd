"""Stein Discrepancy Descent with Barycentric coupling (SDD-RB).

SDD combines cross-coupling (like ETD) with self-coupling between
particles.  The cross-coupling transports particles toward high-density
proposals; the self-coupling provides repulsion that preserves diversity.

SDD update:
    x_new[i] = x[i] + eta * (y_cross[i] - x_bar_self[i])

where:
    y_cross[i]    : resampled proposal from cross-coupling
    x_bar_self[i] : barycentric mean from self-coupling (N×N)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from etd.costs import build_cost_fn, normalize_cost
from etd.coupling import sinkhorn_log_domain
from etd.proposals.langevin import langevin_proposals, update_preconditioner
from etd.proposals.preconditioner import (
    compute_ensemble_cholesky,
    update_rmsprop_accum,
)
from etd.schedule import Schedule, resolve_param
from etd.types import MutationConfig, PreconditionerConfig, Target
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
    precond_accum: jnp.ndarray   # (d,)
    cholesky_factor: jnp.ndarray  # (d, d)
    step: int


@dataclass(frozen=True)
class SDDConfig:
    """Configuration for SDD-RB.

    Mirrors core ETD fields plus SDD-specific parameters.
    """

    # --- Scale ---
    n_particles: int = 100
    n_iterations: int = 500
    n_proposals: int = 25

    # --- Composable axes ---
    cost: str = "euclidean"
    cost_params: tuple = ()
    cost_normalize: str = "median"   # "median" or "mean"

    # --- Core ---
    epsilon: float = 0.1
    alpha: float = 0.05
    fdr: bool = True
    sigma: float = 0.0
    use_score: bool = True
    score_clip: float = 5.0

    # --- Cross-coupling (balanced Sinkhorn) ---
    sinkhorn_max_iter: int = 50
    sinkhorn_tol: float = 1e-2

    # --- Self-coupling ---
    self_epsilon: float = 0.1
    self_sinkhorn_max_iter: int = 50
    self_sinkhorn_tol: float = 1e-2

    # --- SDD update ---
    sdd_step_size: float = 0.5   # eta for displacement

    # --- Preconditioner ---
    preconditioner: PreconditionerConfig = field(
        default_factory=PreconditionerConfig,
    )

    # --- Mutation (MCMC post-transport) ---
    mutation: MutationConfig = field(
        default_factory=MutationConfig,
    )

    precondition: bool = False
    whiten: bool = False
    precond_beta: float = 0.9
    precond_delta: float = 1e-8

    # --- DV feedback ---
    dv_feedback: bool = False
    dv_weight: float = 1.0

    # --- Update ---
    step_size: float = 1.0  # damping for cross-coupling categorical

    # --- Schedules ---
    schedules: tuple = ()

    def __post_init__(self):
        if (
            self.mutation.active
            and self.mutation.use_cholesky
            and not self.preconditioner.is_cholesky
        ):
            warnings.warn(
                "mutation.use_cholesky=True but preconditioner.type is not "
                "'cholesky'. A Cholesky factor will be auto-computed for "
                "mutation each step (O(Nd² + d³)). Consider enabling "
                "Cholesky preconditioning to share the computation.",
                UserWarning,
                stacklevel=2,
            )

    @property
    def needs_cholesky(self) -> bool:
        """Whether a Cholesky factor is needed (preconditioner or mutation)."""
        return self.preconditioner.is_cholesky or (
            self.mutation.active and self.mutation.use_cholesky
        )

    @property
    def needs_precond_accum(self) -> bool:
        """Whether any preconditioner is active."""
        return self.preconditioner.active or self.precondition or self.whiten

    @property
    def resolved_sigma(self) -> float:
        if self.fdr:
            return (2.0 * self.alpha) ** 0.5
        if self.sigma > 0.0:
            return self.sigma
        raise ValueError("sigma must be > 0 when fdr=False")


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
    M = config.n_proposals

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
    else:
        cholesky_factor = jnp.eye(d)

    return SDDState(
        positions=positions,
        dual_f_cross=jnp.zeros(N),
        dual_g_cross=jnp.zeros(N * M),
        dual_f_self=jnp.zeros(N),
        dual_g_self=jnp.zeros(N),
        dv_potential=jnp.zeros(N),
        precond_accum=jnp.ones(d),
        cholesky_factor=cholesky_factor,
        step=0,
    )


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

def _resolve_sigma(config: SDDConfig, step: int):
    """Resolve sigma, respecting FDR."""
    if config.fdr:
        alpha = resolve_param(config, "alpha", step)
        return jnp.sqrt(2.0 * alpha)
    return resolve_param(config, "sigma", step)


def step(
    key: jax.Array,
    state: SDDState,
    target: Target,
    config: SDDConfig,
) -> Tuple[SDDState, Dict[str, jnp.ndarray]]:
    """Execute one SDD-RB iteration.

    Pipeline:
        1. Cross-coupling (identical to ETD):
           propose → IS weights → cost → Sinkhorn → resample → y_cross
        2. Self-coupling:
           N×N cost between particles → balanced Sinkhorn → barycentric mean → x̄_self
        3. SDD displacement:
           x_new[i] = x[i] + eta * (y_cross[i] - x̄_self[i])

    Args:
        key: JAX PRNG key.
        state: Current SDD state.
        target: Target distribution.
        config: SDD configuration.

    Returns:
        Tuple ``(new_state, info)``.
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
    pc = config.preconditioner
    key_propose, key_resample = jax.random.split(key)

    # --- Legacy preconditioner path ---
    preconditioner = None
    legacy_precond = (not pc.active) and (config.precondition or whiten)
    if legacy_precond:
        P = 1.0 / jnp.sqrt(state.precond_accum + config.precond_delta)
        preconditioner = P

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
    alpha = resolve_param(config, "alpha", state.step)
    sigma = _resolve_sigma(config, state.step)
    eps = resolve_param(config, "epsilon", state.step)
    sdd_eta = config.sdd_step_size

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
        n_proposals=config.n_proposals,
        use_score=config.use_score,
        score_clip_val=config.score_clip,
        precondition=config.precondition if not pc.active else False,
        precond_accum=(
            state.precond_accum if config.precondition and not pc.active else None
        ),
        precond_delta=config.precond_delta,
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
    elif legacy_precond and config.precondition:
        log_b = importance_weights(
            proposals, means, target, sigma=sigma,
            preconditioner=preconditioner,
        )
    else:
        log_b = importance_weights(
            proposals, means, target, sigma=sigma,
        )

    # 1c. DV feedback
    if config.dv_feedback:
        dv_weight = resolve_param(config, "dv_weight", state.step)
        dv_expanded = jnp.repeat(state.dv_potential, config.n_proposals)
        log_b = log_b - dv_weight * dv_expanded
        log_b = log_b - logsumexp(log_b)

    # 1d. Cost matrix (cross)
    cost_fn = build_cost_fn(cost_name, config.cost_params)
    if cholesky_for_cost is not None:
        C_cross = cost_fn(
            state.positions, proposals, cholesky_factor=cholesky_for_cost,
        )
    elif diag_for_cost is not None:
        C_cross = cost_fn(
            state.positions, proposals, preconditioner=diag_for_cost,
        )
    elif legacy_precond and whiten:
        C_cross = cost_fn(
            state.positions, proposals, preconditioner=preconditioner,
        )
    else:
        C_cross = cost_fn(state.positions, proposals)
    C_cross, cost_scale_cross = normalize_cost(C_cross, config.cost_normalize)

    # 1e. Source marginal (uniform)
    log_a = -jnp.log(N) * jnp.ones(N)

    # 1f. Balanced Sinkhorn (cross)
    log_gamma_cross, dual_f_cross, dual_g_cross, cross_iters = sinkhorn_log_domain(
        C_cross, log_a, log_b,
        eps=eps,
        max_iter=config.sinkhorn_max_iter,
        tol=config.sinkhorn_tol,
        dual_f_init=state.dual_f_cross,
        dual_g_init=state.dual_g_cross,
    )

    # 1g. Resample: y_cross (one proposal per particle)
    y_cross, cross_aux = systematic_resample(
        key_resample,
        log_gamma_cross,
        proposals,
        step_size=config.step_size,
        positions=state.positions,
    )

    # 1h. DV potential
    if config.dv_feedback:
        g_tilde_cross = dual_g_cross - eps * log_b
        selected_g = g_tilde_cross[cross_aux["indices"]]
        cross_step_size = resolve_param(config, "step_size", state.step)
        dv_potential = (
            (1.0 - cross_step_size) * dual_f_cross + cross_step_size * selected_g
        )
    else:
        dv_potential = jnp.zeros(N)

    # ===================================================================
    # 2. Self-coupling (N×N between particles)
    # ===================================================================

    # 2a. Self-cost matrix (same whitening as cross)
    if cholesky_for_cost is not None:
        C_self = cost_fn(
            state.positions, state.positions, cholesky_factor=cholesky_for_cost,
        )
    elif diag_for_cost is not None:
        C_self = cost_fn(
            state.positions, state.positions, preconditioner=diag_for_cost,
        )
    elif legacy_precond and whiten:
        C_self = cost_fn(
            state.positions, state.positions, preconditioner=preconditioner,
        )
    else:
        C_self = cost_fn(state.positions, state.positions)
    C_self, cost_scale_self = normalize_cost(C_self, config.cost_normalize)

    # 2b. Uniform marginals for self-coupling
    log_a_self = -jnp.log(N) * jnp.ones(N)
    log_b_self = -jnp.log(N) * jnp.ones(N)

    # 2c. Balanced Sinkhorn (self)
    log_gamma_self, dual_f_self, dual_g_self, self_iters = sinkhorn_log_domain(
        C_self, log_a_self, log_b_self,
        eps=config.self_epsilon,
        max_iter=config.self_sinkhorn_max_iter,
        tol=config.self_sinkhorn_tol,
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

    # --- Preconditioner update ---
    new_precond = state.precond_accum
    new_cholesky = state.cholesky_factor

    if pc.is_cholesky:
        if pc.source == "scores":
            if pc.use_unclipped_scores and config.use_score:
                precond_data = target.score(state.positions)
            else:
                precond_data = scores
        else:
            precond_data = state.positions
        new_cholesky = compute_ensemble_cholesky(
            precond_data, state.cholesky_factor, pc,
        )
    elif pc.is_rmsprop:
        new_precond = update_rmsprop_accum(
            state.precond_accum, scores, beta=pc.beta,
        )
    elif legacy_precond:
        new_precond = update_preconditioner(
            state.precond_accum, scores, beta=config.precond_beta,
        )

    # --- Assemble new state ---
    new_state = SDDState(
        positions=new_positions,
        dual_f_cross=dual_f_cross,
        dual_g_cross=dual_g_cross,
        dual_f_self=dual_f_self,
        dual_g_self=dual_g_self,
        dv_potential=dv_potential,
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

    return new_state, info
