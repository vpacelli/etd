"""Neal's funnel target distribution.

A challenging hierarchical target where the first dimension *v* controls
the variance of all remaining dimensions:

    v       ~ N(0, sigma_v^2)
    x_k | v ~ N(0, exp(v))    for k = 1, ..., d-1

The "funnel" shape arises because small v compresses x_k to near zero
while large v allows wide dispersion.  This is notoriously difficult
for both MCMC (NUTS requires careful tuning) and VI (Gaussians can't
capture the varying scale).

Numerical note
--------------
The score contains ``exp(-v)`` which overflows for v << 0 (deep in the
funnel neck).  This is intrinsic to the geometry — score clipping at 5.0
(already in the ETD pipeline) handles it.  We do NOT add extra guards
here to keep the target mathematically exact.
"""

import jax
import jax.numpy as jnp


class FunnelTarget:
    """Neal's funnel distribution.

    Satisfies the :class:`~etd.types.Target` protocol.

    Attributes:
        dim: Dimensionality (>= 2).
        sigma_v: Standard deviation of the first (scale) dimension.
    """

    def __init__(self, dim: int = 10, sigma_v: float = 3.0):
        """
        Args:
            dim: Dimensionality of the target (>= 2).
            sigma_v: Standard deviation of the v (first) dimension.
        """
        if dim < 2:
            raise ValueError(f"FunnelTarget requires dim >= 2, got {dim}")
        self.dim = dim
        self.sigma_v = sigma_v

    @property
    def mean(self) -> jnp.ndarray:
        """Distribution mean, shape ``(d,)``.  All zeros."""
        return jnp.zeros(self.dim)

    @property
    def variance(self) -> jnp.ndarray:
        """Per-dimension marginal variance, shape ``(d,)``.

        Var[v] = sigma_v^2.
        Var[x_k] = E[exp(v)] = exp(sigma_v^2 / 2)  (log-normal mean).
        """
        var_v = self.sigma_v ** 2
        var_x = jnp.exp(var_v / 2.0)
        return jnp.concatenate([
            jnp.array([var_v]),
            jnp.full(self.dim - 1, var_x),
        ])

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate log-density.

        log p(v, x_{1:d-1}) = log N(v; 0, sigma_v^2)
                             + sum_k log N(x_k; 0, exp(v))

        Args:
            x: Positions, shape ``(N, d)``.

        Returns:
            Log-probabilities, shape ``(N,)``.
        """
        v = x[:, 0]            # (N,)
        x_tail = x[:, 1:]      # (N, d-1)
        d_tail = self.dim - 1

        # log p(v) = -0.5 * v^2 / sigma_v^2 - 0.5 * log(2*pi*sigma_v^2)
        log_pv = -0.5 * v ** 2 / self.sigma_v ** 2 \
                 - 0.5 * jnp.log(2.0 * jnp.pi * self.sigma_v ** 2)

        # log p(x_k | v) = -0.5 * x_k^2 * exp(-v) - 0.5 * v - 0.5 * log(2*pi)
        # Sum over k:
        log_px = -0.5 * jnp.sum(x_tail ** 2, axis=1) * jnp.exp(-v) \
                 - 0.5 * d_tail * v \
                 - 0.5 * d_tail * jnp.log(2.0 * jnp.pi)

        return log_pv + log_px

    def score(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate score function nabla log pi(x).

        Analytic gradient:

        d/dv = -v / sigma_v^2
               + 0.5 * sum_k x_k^2 * exp(-v)
               - 0.5 * (d-1)

        d/dx_k = -x_k * exp(-v)

        Warning: ``exp(-v)`` overflows for v << 0. Rely on score clipping
        in the ETD pipeline (default threshold 5.0).

        Args:
            x: Positions, shape ``(N, d)``.

        Returns:
            Score vectors, shape ``(N, d)``.
        """
        v = x[:, 0]            # (N,)
        x_tail = x[:, 1:]      # (N, d-1)
        d_tail = self.dim - 1

        exp_neg_v = jnp.exp(-v)  # (N,)

        # Score for v
        score_v = -v / self.sigma_v ** 2 \
                  + 0.5 * jnp.sum(x_tail ** 2, axis=1) * exp_neg_v \
                  - 0.5 * d_tail

        # Score for x_k
        score_tail = -x_tail * exp_neg_v[:, None]  # (N, d-1)

        return jnp.concatenate([score_v[:, None], score_tail], axis=1)

    def sample(self, key: jax.Array, n: int) -> jnp.ndarray:
        """Draw exact samples via ancestral sampling.

        Args:
            key: JAX PRNG key.
            n: Number of samples.

        Returns:
            Samples, shape ``(n, d)``.
        """
        k1, k2 = jax.random.split(key)

        # v ~ N(0, sigma_v^2)
        v = jax.random.normal(k1, (n,)) * self.sigma_v  # (n,)

        # x_k | v ~ N(0, exp(v))  →  std = exp(v/2)
        std = jnp.exp(v / 2.0)  # (n,)
        noise = jax.random.normal(k2, (n, self.dim - 1))
        x_tail = noise * std[:, None]  # (n, d-1)

        return jnp.concatenate([v[:, None], x_tail], axis=1)
