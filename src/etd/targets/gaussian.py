"""Gaussian target distribution.

Isotropic or anisotropic (axis-aligned) Gaussian with controllable
condition number.  Useful as the simplest benchmark: particles should
converge to the known mean and covariance.
"""

import jax
import jax.numpy as jnp


class GaussianTarget:
    """Axis-aligned Gaussian with controllable condition number.

    Satisfies the :class:`~etd.types.Target` protocol.

    Attributes:
        dim: Dimensionality.
        mean: Mean vector, shape ``(d,)``.
        variance: Per-dimension variances, shape ``(d,)``.
    """

    def __init__(self, dim: int = 2, condition_number: float = 1.0):
        """
        Args:
            dim: Dimensionality of the target.
            condition_number: Ratio of largest to smallest eigenvalue.
                When 1.0, the target is isotropic (standard normal).
        """
        self.dim = dim

        if condition_number > 1.0:
            self.variance = jnp.logspace(
                0.0, jnp.log10(condition_number), dim
            )
        else:
            self.variance = jnp.ones(dim)

        self.mean = jnp.zeros(dim)
        self._precisions = 1.0 / self.variance
        self._log_normalizer = (
            0.5 * dim * jnp.log(2.0 * jnp.pi)
            + 0.5 * jnp.sum(jnp.log(self.variance))
        )

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate log-density.

        Args:
            x: Positions, shape ``(N, d)``.

        Returns:
            Log-probabilities, shape ``(N,)``.
        """
        diff = x - self.mean  # (N, d)
        return (
            -0.5 * jnp.sum(diff ** 2 * self._precisions, axis=-1)
            - self._log_normalizer
        )

    def score(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate score function nabla log pi(x).

        Args:
            x: Positions, shape ``(N, d)``.

        Returns:
            Score vectors, shape ``(N, d)``.
        """
        return -(x - self.mean) * self._precisions

    def sample(self, key: jax.Array, n: int) -> jnp.ndarray:
        """Draw exact samples.

        Args:
            key: JAX PRNG key.
            n: Number of samples.

        Returns:
            Samples, shape ``(n, d)``.
        """
        return self.mean + jax.random.normal(key, (n, self.dim)) * jnp.sqrt(
            self.variance
        )
