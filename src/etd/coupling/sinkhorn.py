"""Log-domain balanced Sinkhorn with warm-start.

Computes the balanced entropic optimal transport coupling where both
marginal constraints (source and target) are enforced as hard constraints
via alternating Bregman projections in log space.
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


def sinkhorn_log_domain(
    C: jnp.ndarray,                          # (N, P)
    log_a: jnp.ndarray,                      # (N,)
    log_b: jnp.ndarray,                      # (P,)
    eps: float,
    max_iter: int = 50,
    tol: float = 1e-5,
    dual_f_init: Optional[jnp.ndarray] = None,   # (N,)  warm-start
    dual_g_init: Optional[jnp.ndarray] = None,   # (P,)  warm-start
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    """Balanced Sinkhorn in log domain.

    Iterates the dual updates:

    .. math::
        f \\leftarrow \\varepsilon \\bigl(\\log a - \\mathrm{logsumexp}_j(\\log K_{\\cdot j} + g_j/\\varepsilon)\\bigr)
        g \\leftarrow \\varepsilon \\bigl(\\log b - \\mathrm{logsumexp}_i(\\log K_{i\\cdot} + f_i/\\varepsilon)\\bigr)

    with dynamic termination via ``lax.while_loop``.

    Args:
        C: Cost matrix, shape ``(N, P)``.
        log_a: Log source marginal, shape ``(N,)``.
        log_b: Log target marginal, shape ``(P,)``.
        eps: Entropic regularization.
        max_iter: Maximum Sinkhorn iterations (default 50).
        tol: Convergence tolerance on max |f_new − f_old| (default 1e-5).
        dual_f_init: Warm-start for source dual, shape ``(N,)``.
        dual_g_init: Warm-start for target dual, shape ``(P,)``.

    Returns:
        Tuple ``(log_gamma, dual_f, dual_g, iterations)`` where:
        - ``log_gamma``: Row-normalized log coupling, shape ``(N, P)``.
        - ``dual_f``: Converged source dual, shape ``(N,)``.
        - ``dual_g``: Converged target dual, shape ``(P,)``.
        - ``iterations``: Number of Sinkhorn iterations executed.
    """
    N, P = C.shape
    log_K = -C / eps  # (N, P)

    # --- Initialization (warm-start or zeros) ---
    f = dual_f_init if dual_f_init is not None else jnp.zeros(N)
    g = dual_g_init if dual_g_init is not None else jnp.zeros(P)

    # Init f_prev so the first convergence check always fails.
    # Using inf guarantees |f - f_prev| = inf > any finite tol,
    # unlike the old f + 1.0 trick which broke for tol >= 1.0 or
    # |f| > 2^24 (float32 precision loss).
    f_prev = jnp.full_like(f, jnp.inf)

    # --- While-loop body ---
    def cond_fn(state):
        f, g, f_prev, iteration = state
        converged = jnp.max(jnp.abs(f - f_prev)) < tol
        return (~converged) & (iteration < max_iter)

    def body_fn(state):
        f, g, _f_prev, iteration = state
        f_prev = f

        # f-update: project onto source marginal (sum over columns j)
        f_new = eps * (log_a - logsumexp(log_K + g[None, :] / eps, axis=1))

        # g-update: project onto target marginal (sum over rows i)
        # log_K.T is (P, N), f_new is (N,) → broadcast via f_new[None, :] → (1, N)
        g_new = eps * (log_b - logsumexp(log_K.T + f_new[None, :] / eps, axis=1))

        return (f_new, g_new, f_prev, iteration + 1)

    init_state = (f, g, f_prev, jnp.int32(0))
    f, g, _, iterations = jax.lax.while_loop(cond_fn, body_fn, init_state)

    # --- Joint log coupling (NOT row-normalized) ---
    # log_gamma[i,j] = log_K[i,j] + f[i]/eps + g[j]/eps
    # Row sums ≈ a, column sums ≈ b  (balanced constraints).
    # The update rule row-normalizes before sampling.
    log_gamma = log_K + f[:, None] / eps + g[None, :] / eps

    return log_gamma, f, g, iterations
