"""L-infinity cost matrix.

    C_ij = ||x_i - y_j||_inf = max_k |x_ik - y_jk|

L-infinity avoids the concentration-of-measure problem of L2 in high
dimensions: L2 distances cluster near sqrt(d) * sigma while L-inf
spreads more evenly, keeping the Gibbs kernel informative.
"""

import jax.numpy as jnp


def linf_cost(
    positions: jnp.ndarray,   # (N, d)
    proposals: jnp.ndarray,   # (P, d)
    *,
    preconditioner: jnp.ndarray = None,  # unused, kept for uniform interface
) -> jnp.ndarray:             # (N, P)
    """L-infinity cost.

    Args:
        positions: Particle positions, shape ``(N, d)``.
        proposals: Proposal positions, shape ``(P, d)``.
        preconditioner: Unused.  Accepted for interface uniformity.

    Returns:
        Cost matrix, shape ``(N, P)``.  Non-negative.
    """
    # (N, 1, d) - (1, P, d) -> (N, P, d) -> max over d -> (N, P)
    diff = jnp.abs(positions[:, None, :] - proposals[None, :, :])
    return jnp.max(diff, axis=-1)
