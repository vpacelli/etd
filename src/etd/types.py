"""Core types for Entropic Transport Descent.

Defines the Target protocol, ETDState, and ETDConfig used throughout the
algorithm. These are the foundational types that all other modules reference.
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
        """Evaluate score function ∇ log π(x).

        Args:
            x: Positions, shape ``(N, d)``.

        Returns:
            Score vectors, shape ``(N, d)``.
        """
        ...


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
        use_unclipped_scores: Use raw (unclipped) scores when ``source="scores"``.
        beta: RMSProp EMA decay for the diagonal accumulator.
        delta: Regularization floor for diagonal ``P = 1/sqrt(G + delta)``.
        shrinkage: Ledoit-Wolf shrinkage toward the diagonal (Cholesky only).
        jitter: Diagonal jitter for positive-definiteness (Cholesky only).
        ema_beta: EMA on covariance: 0.0 = fresh each step (Cholesky only).
    """

    type: str = "none"                # "none" | "rmsprop" | "cholesky"
    proposals: bool = False           # apply to proposal drift + noise
    cost: bool = False                # apply to cost matrix whitening
    source: str = "scores"            # "scores" | "positions"
    use_unclipped_scores: bool = False
    # RMSProp
    beta: float = 0.9
    delta: float = 1e-8
    # Cholesky
    shrinkage: float = 0.1           # toward diagonal
    jitter: float = 1e-6             # PD guarantee
    ema_beta: float = 0.0            # 0.0 = fresh each step

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


def make_preconditioner_config(
    precondition: bool = False,
    whiten: bool = False,
    precond_beta: float = 0.9,
    precond_delta: float = 1e-8,
) -> PreconditionerConfig:
    """Build a PreconditionerConfig from legacy flat fields.

    Args:
        precondition: Apply RMSProp to proposals.
        whiten: Apply RMSProp to cost whitening.
        precond_beta: RMSProp EMA decay.
        precond_delta: RMSProp regularization.

    Returns:
        Equivalent :class:`PreconditionerConfig`.
    """
    if not precondition and not whiten:
        return PreconditionerConfig()
    return PreconditionerConfig(
        type="rmsprop",
        proposals=precondition,
        cost=whiten,
        beta=precond_beta,
        delta=precond_delta,
    )


# ---------------------------------------------------------------------------
# Mutation config — frozen dataclass (static arg for JIT)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MutationConfig:
    """Configuration for post-transport MCMC mutation.

    Adds MCMC mutation steps after transport resampling, completing the
    SMC structure: reweight → resample → mutate.  The mutation is
    π-invariant (MH correction), so it can only improve or maintain
    approximation quality.

    Attributes:
        kernel: MCMC kernel type: ``"none"``, ``"mala"``, or ``"rwm"``.
        n_steps: Number of MCMC sub-steps per ETD iteration (static for
            ``lax.scan``).
        step_size: MALA/RWM step size *h*.
        use_cholesky: Use ensemble Cholesky factor for proposal covariance.
        score_clip: Score clipping threshold for MALA.  ``None`` inherits
            from the parent config's ``score_clip``.
    """

    kernel: str = "none"              # "none" | "mala" | "rwm"
    n_steps: int = 5                  # MCMC sub-steps per ETD iteration
    step_size: float = 0.01           # MALA/RWM step size h
    use_cholesky: bool = True         # Use ensemble Cholesky for proposal cov
    score_clip: Optional[float] = None  # None → inherit from parent config

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
        log_prob: Cached log π(x) from mutation, shape ``(N,)``.
            Zeros when mutation is off.
        scores: Cached clipped scores from mutation, shape ``(N, d)``.
            Zeros when mutation is off or using RWM.
        precond_accum: RMSProp-style accumulator for diagonal
            preconditioner, shape ``(d,)``.  Initialized to **ones**
            (it appears as a denominator via ``1 / sqrt(G + δ)``).
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
    """

    # --- Scale ---
    n_particles: int = 100
    n_iterations: int = 500
    n_proposals: int = 25

    # --- Composable axes (string names → resolved to functions at build) ---
    cost: str = "euclidean"
    cost_params: tuple = ()   # sorted (key, value) pairs, e.g. (("c", 1.0),)
    cost_normalize: str = "median"   # "median" or "mean"
    coupling: str = "balanced"
    update: str = "categorical"

    # --- Core ---
    epsilon: float = 0.1
    alpha: float = 0.05
    fdr: bool = True
    sigma: float = 0.0          # only used when fdr=False; 0.0 sentinel
    use_score: bool = True
    score_clip: float = 5.0

    # --- Coupling ---
    rho: float = 1.0            # unbalanced only: τ/ε
    sinkhorn_max_iter: int = 50
    sinkhorn_tol: float = 1e-2

    # --- Update ---
    step_size: float = 1.0      # damping in (0, 1]

    # --- Preconditioner ---
    preconditioner: PreconditionerConfig = field(
        default_factory=PreconditionerConfig,
    )

    # --- Mutation (MCMC post-transport) ---
    mutation: MutationConfig = field(
        default_factory=MutationConfig,
    )

    # Legacy compat — populated by make_preconditioner_config() in the runner.
    # Deprecated: use ``preconditioner`` instead.
    precondition: bool = False
    whiten: bool = False
    precond_beta: float = 0.9
    precond_delta: float = 1e-8

    # --- DV feedback ---
    dv_feedback: bool = False
    dv_weight: float = 1.0

    # --- Schedules ---
    schedules: tuple = ()  # (("dv_weight", Schedule(...)), ("epsilon", Schedule(...)), ...)

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
        """Whether any preconditioner is active.

        Returns True when the new ``preconditioner`` config is active
        **or** the legacy ``precondition``/``whiten`` flags are set.
        """
        return self.preconditioner.active or self.precondition or self.whiten

    @property
    def resolved_sigma(self) -> float:
        """Return the proposal noise scale.

        When ``fdr=True``, σ = √(2α) (fluctuation-dissipation relation).
        Otherwise, ``sigma`` must have been set explicitly.
        """
        if self.fdr:
            return (2.0 * self.alpha) ** 0.5
        if self.sigma > 0.0:
            return self.sigma
        raise ValueError("sigma must be > 0 when fdr=False")
