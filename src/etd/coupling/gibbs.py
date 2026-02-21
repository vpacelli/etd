"""Closed-form Gibbs (semi-relaxed) coupling.

The semi-relaxed case (τ = 0) enforces only the source marginal:
each row of γ sums to 1.  This is a single softmax — no Sinkhorn
iterations required.
"""

from typing import Tuple

import jax
import jax.numpy as jnp


def gibbs_coupling(
    C: jnp.ndarray,        # (N, P)
    log_a: jnp.ndarray,    # (N,)   — source marginal (unused, kept for interface)
    log_b: jnp.ndarray,    # (P,)   — target marginal / IS weights
    eps: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute the Gibbs (semi-relaxed) coupling in log domain.

    .. math::
        \\log \\gamma_{ij} = \\text{log\\_softmax}_j(-C_{ij}/\\varepsilon + \\log b_j)

    Args:
        C: Cost matrix, shape ``(N, P)``.
        log_a: Log source marginal, shape ``(N,)``.  Unused — the source
            constraint is enforced by the softmax normalization.
        log_b: Log target weights (IS-corrected), shape ``(P,)``.
        eps: Entropic regularization strength.

    Returns:
        Tuple ``(log_gamma, dual_f, dual_g)`` where:
        - ``log_gamma`` has shape ``(N, P)`` with rows summing to 1 in
          probability space.
        - ``dual_f`` is zeros ``(N,)`` (interface compatibility).
        - ``dual_g`` is zeros ``(P,)`` (interface compatibility).
    """
    # Joint coupling: log_gamma[i, :] = log_a[i] + log_softmax(-C[i,:]/eps + log_b)
    # Row sums = a_i (source marginal enforced), column sums are free.
    log_gamma = log_a[:, None] + jax.nn.log_softmax(-C / eps + log_b[None, :], axis=1)

    N, P = C.shape
    dual_f = jnp.zeros(N)
    dual_g = jnp.zeros(P)

    return log_gamma, dual_f, dual_g
