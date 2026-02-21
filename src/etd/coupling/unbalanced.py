"""Log-domain unbalanced Sinkhorn with warm-start.

Enforces the source marginal exactly and applies a soft KL penalty
on the target marginal.  The parameter rho = tau/eps controls
marginal softness:

    rho -> 0  :  Gibbs (semi-relaxed, target free)
    rho -> inf:  balanced Sinkhorn (both marginals exact)
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


def sinkhorn_unbalanced(
    C: jnp.ndarray,                          # (N, P)
    log_a: jnp.ndarray,                      # (N,)
    log_b: jnp.ndarray,                      # (P,)
    eps: float,
    rho: float = 1.0,
    max_iter: int = 50,
    tol: float = 1e-5,
    dual_f_init: Optional[jnp.ndarray] = None,   # (N,)  warm-start
    dual_g_init: Optional[jnp.ndarray] = None,   # (P,)  warm-start
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    """Unbalanced Sinkhorn in log domain.

    Source marginal is enforced exactly; target marginal is soft
    via KL penalty with strength tau = rho * eps.

    The proximal parameter lambda = rho / (1 + rho) interpolates
    the g-update between no constraint (lambda=0, Gibbs) and full
    constraint (lambda=1, balanced).

    Args:
        C: Cost matrix, shape ``(N, P)``.
        log_a: Log source marginal, shape ``(N,)``.
        log_b: Log target marginal, shape ``(P,)``.
        eps: Entropic regularization.
        rho: Marginal softness tau/eps.  Default 1.0.
        max_iter: Maximum Sinkhorn iterations (default 50).
        tol: Convergence tolerance on max |f_new - f_old| (default 1e-5).
        dual_f_init: Warm-start for source dual, shape ``(N,)``.
        dual_g_init: Warm-start for target dual, shape ``(P,)``.

    Returns:
        Tuple ``(log_gamma, dual_f, dual_g, iterations)`` where:
        - ``log_gamma``: Log coupling, shape ``(N, P)``.
        - ``dual_f``: Converged source dual, shape ``(N,)``.
        - ``dual_g``: Converged target dual, shape ``(P,)``.
        - ``iterations``: Number of iterations executed.
    """
    N, P = C.shape
    log_K = -C / eps  # (N, P)

    # Proximal parameter: lambda = rho / (1 + rho)
    lam = rho / (1.0 + rho)

    # --- Initialization (warm-start or zeros) ---
    f = dual_f_init if dual_f_init is not None else jnp.zeros(N)
    g = dual_g_init if dual_g_init is not None else jnp.zeros(P)

    # Ensure first convergence check fails
    f_prev = f + 1.0

    # --- While-loop ---
    def cond_fn(state):
        f, g, f_prev, iteration = state
        converged = jnp.max(jnp.abs(f - f_prev)) < tol
        return (~converged) & (iteration < max_iter)

    def body_fn(state):
        f, g, _f_prev, iteration = state
        f_prev = f

        # f-update: exact source marginal (same as balanced)
        f_new = eps * (log_a - logsumexp(log_K + g[None, :] / eps, axis=1))

        # g-update: soft target marginal via proximal scaling
        g_unreg = logsumexp(log_K.T + f_new[None, :] / eps, axis=1)
        g_new = eps * lam * (log_b - g_unreg)

        return (f_new, g_new, f_prev, iteration + 1)

    init_state = (f, g, f_prev, jnp.int32(0))
    f, g, _, iterations = jax.lax.while_loop(cond_fn, body_fn, init_state)

    # --- Log coupling ---
    log_gamma = log_K + f[:, None] / eps + g[None, :] / eps

    return log_gamma, f, g, iterations
