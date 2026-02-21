"""Banana (Rosenbrock twist) target distribution.

A non-Gaussian benchmark with curved, banana-shaped contours.
The distribution is constructed via an affine twist of a Gaussian:

    x_1   ~ N(0, sigma_1^2)
    x_k | x_1  ~ N(b * (x_1^2 - a), sigma_2^2)   for k = 2, ..., d

This creates a multimodal-ish curved ridge that challenges particle
methods relying on Gaussian approximations.
"""

import jax
import jax.numpy as jnp


class BananaTarget:
    """Banana (Rosenbrock twist) distribution.

    Satisfies the :class:`~etd.types.Target` protocol.

    Attributes:
        dim: Dimensionality.
        curvature: Twist parameter *b*.
        offset: Offset parameter *a*.
        sigma1: Marginal std of x_1.
        sigma2: Conditional std of x_2, ..., x_d.
    """

    def __init__(
        self,
        dim: int = 2,
        curvature: float = 0.1,
        offset: float = 100.0,
        sigma1: float = 10.0,
        sigma2: float = 1.0,
    ):
        """
        Args:
            dim: Dimensionality of the target (>= 2).
            curvature: Twist parameter *b* controlling banana curvature.
            offset: Offset parameter *a* (mean shift in conditional).
            sigma1: Marginal standard deviation of x_1.
            sigma2: Conditional standard deviation of x_2, ..., x_d | x_1.
        """
        if dim < 2:
            raise ValueError(f"BananaTarget requires dim >= 2, got {dim}")
        self.dim = dim
        self.curvature = curvature
        self.offset = offset
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    @property
    def mean(self) -> jnp.ndarray:
        """Distribution mean, shape ``(d,)``.

        E[x_1] = 0, E[x_k] = b * (sigma_1^2 - a).
        """
        b, a = self.curvature, self.offset
        mu_tail = b * (self.sigma1 ** 2 - a)
        return jnp.concatenate([
            jnp.zeros(1),
            jnp.full(self.dim - 1, mu_tail),
        ])

    @property
    def variance(self) -> jnp.ndarray:
        """Per-dimension marginal variance, shape ``(d,)``.

        Var[x_1] = sigma_1^2.
        Var[x_k] = sigma_2^2 + 2 * b^2 * sigma_1^4.
        """
        b = self.curvature
        var_tail = self.sigma2 ** 2 + 2.0 * b ** 2 * self.sigma1 ** 4
        return jnp.concatenate([
            jnp.array([self.sigma1 ** 2]),
            jnp.full(self.dim - 1, var_tail),
        ])

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate log-density.

        Args:
            x: Positions, shape ``(N, d)``.

        Returns:
            Log-probabilities, shape ``(N,)``.
        """
        b, a = self.curvature, self.offset
        x1 = x[:, 0]                      # (N,)
        x_tail = x[:, 1:]                  # (N, d-1)

        # Marginal: x_1 ~ N(0, sigma_1^2)
        log_p1 = -0.5 * x1 ** 2 / self.sigma1 ** 2 \
                 - 0.5 * jnp.log(2.0 * jnp.pi * self.sigma1 ** 2)

        # Conditional: x_k | x_1 ~ N(b*(x_1^2 - a), sigma_2^2)
        cond_mean = b * (x1[:, None] ** 2 - a)  # (N, 1) broadcast
        residuals = x_tail - cond_mean           # (N, d-1)
        log_p_tail = -0.5 * jnp.sum(residuals ** 2, axis=1) / self.sigma2 ** 2 \
                     - 0.5 * (self.dim - 1) * jnp.log(2.0 * jnp.pi * self.sigma2 ** 2)

        return log_p1 + log_p_tail

    def score(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate score function nabla log pi(x).

        Analytic gradient (avoids autodiff overhead):

        d/dx_1 = -x_1 / sigma_1^2
                 + sum_k 2*b*x_1*(x_k - b*(x_1^2 - a)) / sigma_2^2

        d/dx_k = -(x_k - b*(x_1^2 - a)) / sigma_2^2

        Args:
            x: Positions, shape ``(N, d)``.

        Returns:
            Score vectors, shape ``(N, d)``.
        """
        b, a = self.curvature, self.offset
        x1 = x[:, 0]                        # (N,)
        x_tail = x[:, 1:]                    # (N, d-1)

        cond_mean = b * (x1[:, None] ** 2 - a)  # (N, 1) broadcast
        residuals = x_tail - cond_mean           # (N, d-1)

        # Score for x_k (k >= 2): -residual / sigma_2^2
        score_tail = -residuals / self.sigma2 ** 2  # (N, d-1)

        # Score for x_1: marginal + coupling from conditionals
        score_x1 = -x1 / self.sigma1 ** 2 \
                   + 2.0 * b * x1 * jnp.sum(residuals, axis=1) / self.sigma2 ** 2

        return jnp.concatenate([score_x1[:, None], score_tail], axis=1)

    def sample(self, key: jax.Array, n: int) -> jnp.ndarray:
        """Draw exact samples via ancestral sampling.

        Args:
            key: JAX PRNG key.
            n: Number of samples.

        Returns:
            Samples, shape ``(n, d)``.
        """
        b, a = self.curvature, self.offset
        k1, k2 = jax.random.split(key)

        # x_1 ~ N(0, sigma_1^2)
        x1 = jax.random.normal(k1, (n,)) * self.sigma1  # (n,)

        # x_k | x_1 ~ N(b*(x_1^2 - a), sigma_2^2)
        cond_mean = b * (x1[:, None] ** 2 - a)  # (n, 1)
        noise = jax.random.normal(k2, (n, self.dim - 1)) * self.sigma2
        x_tail = cond_mean + noise  # (n, d-1)

        return jnp.concatenate([x1[:, None], x_tail], axis=1)
