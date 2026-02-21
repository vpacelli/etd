"""Core types for Entropic Transport Descent.

Defines the Target protocol, ETDState, and ETDConfig used throughout the
algorithm. These are the foundational types that all other modules reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Protocol, runtime_checkable

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
        precond_accum: RMSProp-style accumulator for diagonal
            preconditioner, shape ``(d,)``.  Initialized to **ones**
            (it appears as a denominator via ``1 / sqrt(G + δ)``).
        step: Scalar iteration counter.
    """

    positions: jnp.ndarray      # (N, d)
    dual_f: jnp.ndarray         # (N,)
    dual_g: jnp.ndarray         # (N*M,)
    precond_accum: jnp.ndarray  # (d,)
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
    precondition: bool = False
    precond_beta: float = 0.9
    precond_delta: float = 1e-8

    # --- DV feedback ---
    dv_feedback: bool = False
    dv_weight: float = 1.0

    # --- Schedules ---
    schedules: tuple = ()  # (("dv_weight", Schedule(...)), ("epsilon", Schedule(...)), ...)

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
