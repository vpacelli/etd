"""Core types for Entropic Transport Descent.

Defines the Target protocol, sub-config dataclasses, ETDState, and ETDConfig
used throughout the algorithm. These are the foundational types that all
other modules reference.

Sub-config hierarchy::

    ETDConfig
    ├── ProposalConfig     — proposal generation (count, drift, noise, clipping)
    ├── CostConfig         — transport cost (type, normalization, params)
    ├── CouplingConfig     — Sinkhorn coupling (type, iterations, tolerance)
    ├── UpdateConfig       — particle update (type, damping)
    ├── PreconditionerConfig — diagonal/Cholesky preconditioning
    ├── MutationConfig     — post-transport MCMC mutation
    └── FeedbackConfig     — Donsker-Varadhan feedback signal
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import NamedTuple, Optional, Protocol, runtime_checkable

import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Target protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Target(Protocol):
    """Protocol for target distributions.

    Any class implementing ``dim``, ``log_prob``, and ``score`` satisfies
    this protocol — no inheritance required.
    """

    dim: int

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate log-density.

        Args:
            x: Positions, shape ``(N, d)``.

        Returns:
            Log-probabilities, shape ``(N,)``.
        """
        ...

    def score(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate score function nabla log pi(x).

        Args:
            x: Positions, shape ``(N, d)``.

        Returns:
            Score vectors, shape ``(N, d)``.
        """
        ...


# ---------------------------------------------------------------------------
# Proposal config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProposalConfig:
    """Configuration for Langevin proposal generation.

    Controls the drift-diffusion process that generates proposal
    particles:  ``y = x + alpha * s(x) + sigma * noise``.

    Attributes:
        type: Proposal type: ``"score"`` (score-guided) or
            ``"score_free"`` (pure noise).
        count: Number of proposals *M* per iteration (pooled across
            all particles).
        alpha: Drift step size for Langevin dynamics.
        fdr: Fluctuation-dissipation relation: ``sigma = sqrt(2*alpha)``.
        sigma: Explicit noise scale; required when ``score_free`` or
            ``fdr=False``.  ``0.0`` is a sentinel meaning "use FDR".
        clip_score: Maximum score norm for gradient clipping.
            ``0.0`` or ``inf`` disables clipping.
    """

    type: str = "score"          # "score" | "score_free"
    count: int = 25              # M proposals
    alpha: float = 0.05          # drift step size
    fdr: bool = True             # sigma = sqrt(2*alpha)
    sigma: float = 0.0           # explicit; required when score_free or fdr=False
    clip_score: float = 5.0      # max score norm; 0 or inf = no clipping

    @property
    def use_score(self) -> bool:
        """Whether proposals use score information."""
        return self.type == "score"

    @property
    def resolved_sigma(self) -> float:
        """Return the proposal noise scale.

        When ``fdr=True``, sigma = sqrt(2*alpha).
        Otherwise, ``sigma`` must have been set explicitly.
        """
        if self.fdr:
            return (2.0 * self.alpha) ** 0.5
        if self.sigma > 0.0:
            return self.sigma
        raise ValueError("sigma must be > 0 when fdr=False")


# ---------------------------------------------------------------------------
# Cost config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CostConfig:
    """Configuration for the transport cost function.

    Attributes:
        type: Cost type: ``"euclidean"``, ``"linf"``, ``"imq"``,
            or ``"langevin"``.
        normalize: Normalization strategy: ``"median"`` or ``"mean"``.
        params: Extra parameters as sorted ``(key, value)`` pairs,
            e.g. ``(("c", 1.0),)`` for IMQ or ``(("whiten", True),)``
            for Langevin cost whitening.
    """

    type: str = "euclidean"      # "euclidean" | "linf" | "imq" | "langevin"
    normalize: str = "median"    # "median" | "mean"
    params: tuple = ()           # sorted (key, value) pairs

    @property
    def whiten(self) -> bool:
        """Whether cost whitening is enabled (from params dict)."""
        return dict(self.params).get("whiten", False)


# ---------------------------------------------------------------------------
# Coupling config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CouplingConfig:
    """Configuration for the Sinkhorn coupling.

    Attributes:
        type: Coupling type: ``"balanced"``, ``"unbalanced"``,
            or ``"gibbs"`` (closed-form, testing only).
        iterations: Maximum Sinkhorn iterations.
        tolerance: Sinkhorn convergence tolerance.
        rho: Unbalanced only: tau/epsilon ratio.
    """

    type: str = "balanced"       # "balanced" | "unbalanced" | "gibbs"
    iterations: int = 50         # Sinkhorn max iters
    tolerance: float = 1e-2      # Sinkhorn convergence
    rho: float = 1.0             # unbalanced only: tau/epsilon


# ---------------------------------------------------------------------------
# Update config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UpdateConfig:
    """Configuration for the particle update rule.

    Attributes:
        type: Update type: ``"categorical"`` (systematic resampling)
            or ``"barycentric"`` (weighted mean).
        damping: Step size damping in ``(0, 1]``.  Always applied
            (identity when 1.0, branchless).
    """

    type: str = "categorical"    # "categorical" | "barycentric"
    damping: float = 1.0         # (0, 1]


# ---------------------------------------------------------------------------
# Feedback config (Donsker-Varadhan)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeedbackConfig:
    """Configuration for DV feedback signal.

    Attributes:
        enabled: Whether DV feedback is active.
        weight: Scaling weight for the feedback signal.
    """

    enabled: bool = False
    weight: float = 1.0


# ---------------------------------------------------------------------------
# Self-coupling config (SDD only)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SelfCouplingConfig:
    """Configuration for SDD self-coupling (N x N between particles).

    Attributes:
        epsilon: Regularization for self-coupling Sinkhorn.
        iterations: Maximum Sinkhorn iterations.
        tolerance: Sinkhorn convergence tolerance.
    """

    epsilon: float = 0.1
    iterations: int = 50
    tolerance: float = 1e-2


# ---------------------------------------------------------------------------
# Preconditioner config — frozen dataclass (static arg for JIT)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PreconditionerConfig:
    """Configuration for the preconditioner.

    Selects between no preconditioner, diagonal RMSProp, or full-covariance
    Cholesky.  Frozen so it can be embedded in ``ETDConfig`` as a static arg.

    Attributes:
        type: Preconditioner type: ``"none"``, ``"rmsprop"``, or ``"cholesky"``.
        proposals: Apply preconditioner to proposal drift and noise.
        cost: Apply preconditioner to cost matrix whitening.
        source: Data source for covariance: ``"scores"`` or ``"positions"``.
        clip_score: Score clipping for preconditioner data.
            ``0.0`` (default) = inherit parent config's clipping.
            ``inf`` = use raw (unclipped) scores.
            Positive value = clip to that threshold.
        beta: RMSProp EMA decay for the diagonal accumulator.
        delta: Regularization floor for diagonal ``P = 1/sqrt(G + delta)``.
        shrinkage: Ledoit-Wolf shrinkage toward the diagonal (Cholesky only).
        jitter: Diagonal jitter for positive-definiteness (Cholesky only).
        ema: EMA on covariance: 0.0 = fresh each step (Cholesky only).
    """

    type: str = "none"                # "none" | "rmsprop" | "cholesky"
    proposals: bool = False           # apply to proposal drift + noise
    cost: bool = False                # apply to cost matrix whitening
    source: str = "scores"            # "scores" | "positions"
    clip_score: float = 0.0           # 0 = inherit parent; inf = raw/unclipped
    # RMSProp
    beta: float = 0.9
    delta: float = 1e-8
    # Cholesky
    shrinkage: float = 0.1           # toward diagonal
    jitter: float = 1e-6             # PD guarantee
    ema: float = 0.0                 # 0.0 = fresh each step

    def __post_init__(self):
        if self.is_rmsprop and self.source != "scores":
            warnings.warn(
                f"source={self.source!r} is ignored for type='rmsprop' "
                "(RMSProp always uses scores). Set source='scores' to "
                "silence this warning.",
                UserWarning,
                stacklevel=2,
            )

    @property
    def active(self) -> bool:
        """Whether any preconditioner is active."""
        return self.type != "none"

    @property
    def is_cholesky(self) -> bool:
        return self.type == "cholesky"

    @property
    def is_rmsprop(self) -> bool:
        return self.type == "rmsprop"

    @property
    def use_raw_scores(self) -> bool:
        """Whether to use raw (unclipped) scores for preconditioner data.

        - ``clip_score == 0.0`` (default): inherit parent's clipping → **False**
        - ``clip_score == inf``: explicit raw/unclipped → **True**
        - ``clip_score > 0``: clip to that threshold (future use) → **False**
        """
        return self.clip_score == float("inf")


# ---------------------------------------------------------------------------
# Mutation config — frozen dataclass (static arg for JIT)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MutationConfig:
    """Configuration for post-transport MCMC mutation.

    Adds MCMC mutation steps after transport resampling, completing the
    SMC structure: reweight -> resample -> mutate.  The mutation is
    pi-invariant (MH correction), so it can only improve or maintain
    approximation quality.

    Attributes:
        kernel: MCMC kernel type: ``"none"``, ``"mala"``, or ``"rwm"``.
        steps: Number of MCMC sub-steps per ETD iteration (static for
            ``lax.scan``).
        stepsize: MALA/RWM step size *h*.
        cholesky: Use ensemble Cholesky factor for proposal covariance.
        clip_score: Score clipping threshold for MALA.  ``None`` inherits
            from the parent config's ``proposal.clip_score``.
    """

    kernel: str = "none"              # "none" | "mala" | "rwm"
    steps: int = 5                    # MCMC sub-steps per ETD iteration
    stepsize: float = 0.01            # MALA/RWM step size h
    cholesky: bool = True             # Use ensemble Cholesky for proposal cov
    clip_score: Optional[float] = None  # None -> inherit from parent config

    @property
    def active(self) -> bool:
        """Whether mutation is enabled."""
        return self.kernel != "none"

    @property
    def needs_score(self) -> bool:
        """Whether the kernel requires score evaluations."""
        return self.kernel == "mala"


# ---------------------------------------------------------------------------
# ETD state — a JAX-compatible NamedTuple (native pytree)
# ---------------------------------------------------------------------------

class ETDState(NamedTuple):
    """Mutable state carried across ETD iterations.

    All fields are concrete arrays — no ``None`` values — so the pytree
    structure is stable across JIT recompilations.

    Attributes:
        positions: Particle positions, shape ``(N, d)``.
        dual_f: Source-side Sinkhorn dual potentials, shape ``(N,)``.
        dual_g: Target-side Sinkhorn dual potentials, shape ``(N*M,)``
            (global proposal pool).
        dv_potential: Per-particle DV feedback signal, shape ``(N,)``.
            Clean c-transformed potential (without ``log_b`` contamination)
            interpolated between source and target potentials.
        log_prob: Cached log pi(x) from mutation, shape ``(N,)``.
            Zeros when mutation is off.
        scores: Cached clipped scores from mutation, shape ``(N, d)``.
            Zeros when mutation is off or using RWM.
        precond_accum: RMSProp-style accumulator for diagonal
            preconditioner, shape ``(d,)``.  Initialized to **ones**
            (it appears as a denominator via ``1 / sqrt(G + delta)``).
        cholesky_factor: Lower-triangular Cholesky factor of the ensemble
            covariance, shape ``(d, d)``.  Initialized to ``eye(d)``.
            Only updated when ``preconditioner.type == "cholesky"``.
        step: Scalar iteration counter.
    """

    positions: jnp.ndarray      # (N, d)
    dual_f: jnp.ndarray         # (N,)
    dual_g: jnp.ndarray         # (N*M,)
    dv_potential: jnp.ndarray   # (N,)
    log_prob: jnp.ndarray       # (N,)
    scores: jnp.ndarray         # (N, d)
    precond_accum: jnp.ndarray  # (d,)
    cholesky_factor: jnp.ndarray  # (d, d)
    step: int                   # scalar


# ---------------------------------------------------------------------------
# ETD config — frozen dataclass (static arg for JIT)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ETDConfig:
    """Configuration for an ETD run.

    Frozen so it can be used as a ``static_argnums`` argument to
    ``jax.jit``.  Each distinct config triggers XLA recompilation,
    so Python ``if`` on config fields is resolved at trace time.

    Sub-configs group related parameters by algorithmic component:
    proposal, cost, coupling, update, preconditioner, mutation, feedback.
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

    # --- Schedules ---
    schedules: tuple = ()  # (("epsilon", Schedule(...)), ("proposal.alpha", Schedule(...)), ...)

    def __post_init__(self):
        # cost="langevin" + proposal="score_free" -> ERROR
        if self.cost.type == "langevin" and not self.proposal.use_score:
            raise ValueError(
                "cost.type='langevin' requires score-guided proposals "
                "(proposal.type='score')"
            )

        # proposal="score_free" + no sigma -> ERROR
        if (
            not self.proposal.use_score
            and not self.proposal.fdr
            and self.proposal.sigma <= 0
        ):
            raise ValueError(
                "proposal.type='score_free' with fdr=False requires "
                "explicit sigma > 0"
            )

        # mutation.cholesky + preconditioner="none" -> WARNING
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
