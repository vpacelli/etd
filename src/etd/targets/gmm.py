"""Gaussian mixture model target distribution.

Supports grid and ring arrangements of isotropic components in the
first two dimensions.  A standard benchmark for multi-modal inference:
the algorithm must discover all modes without collapsing.
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


class GMMTarget:
    """Mixture of isotropic Gaussians with configurable layout.

    Satisfies the :class:`~etd.types.Target` protocol.

    Attributes:
        dim: Dimensionality.
        n_modes: Number of mixture components.
        means: Component means, shape ``(K, d)``.
        log_weights: Log mixing weights, shape ``(K,)``.
        component_std: Standard deviation of each component.
    """

    def __init__(
        self,
        dim: int = 2,
        n_modes: int = 4,
        arrangement: str = "grid",
        separation: float = 6.0,
        component_std: float = 1.0,
    ):
        """
        Args:
            dim: Dimensionality of the target.
            n_modes: Number of mixture components (*K*).
            arrangement: ``"grid"`` or ``"ring"``.
            separation: Distance scale between components.
            component_std: Std dev of each isotropic component.
        """
        self.dim = dim
        self.n_modes = n_modes
        self.component_std = component_std
        self.log_weights = -jnp.log(n_modes) * jnp.ones(n_modes)

        if arrangement == "grid":
            self.means = self._grid_means(n_modes, dim, separation)
        elif arrangement == "ring":
            self.means = self._ring_means(n_modes, dim, separation)
        else:
            raise ValueError(
                f"Unknown arrangement '{arrangement}'. Use 'grid' or 'ring'."
            )

    # -----------------------------------------------------------------
    # Layout helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _grid_means(K: int, d: int, separation: float) -> jnp.ndarray:
        """Place K components on a 2D grid, zeros in remaining dims.

        Returns:
            Component means, shape ``(K, d)``.
        """
        import math

        n_side = math.ceil(math.sqrt(K))
        coords = jnp.linspace(-separation / 2, separation / 2, n_side)
        gx, gy = jnp.meshgrid(coords, coords)
        grid_2d = jnp.stack([gx.ravel(), gy.ravel()], axis=-1)[:K]  # (K, 2)

        if d > 2:
            padding = jnp.zeros((K, d - 2))
            return jnp.concatenate([grid_2d, padding], axis=-1)
        return grid_2d

    @staticmethod
    def _ring_means(K: int, d: int, separation: float) -> jnp.ndarray:
        """Place K components on a ring in the first 2 dims.

        Returns:
            Component means, shape ``(K, d)``.
        """
        radius = separation / 2.0
        angles = jnp.linspace(0, 2 * jnp.pi, K, endpoint=False)
        ring_2d = radius * jnp.stack(
            [jnp.cos(angles), jnp.sin(angles)], axis=-1
        )  # (K, 2)

        if d > 2:
            padding = jnp.zeros((K, d - 2))
            return jnp.concatenate([ring_2d, padding], axis=-1)
        return ring_2d

    # -----------------------------------------------------------------
    # Target protocol
    # -----------------------------------------------------------------

    def _log_component_densities(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute log density of each component at each point.

        Args:
            x: Positions, shape ``(N, d)``.

        Returns:
            Log component densities, shape ``(N, K)``.
        """
        diff = x[:, None, :] - self.means[None, :, :]  # (N, K, d)
        maha = jnp.sum(diff ** 2, axis=-1) / (self.component_std ** 2)  # (N, K)
        log_norm = -0.5 * self.dim * jnp.log(
            2.0 * jnp.pi * self.component_std ** 2
        )
        return self.log_weights + log_norm - 0.5 * maha  # (N, K)

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate log-density of the mixture.

        Args:
            x: Positions, shape ``(N, d)``.

        Returns:
            Log-probabilities, shape ``(N,)``.
        """
        log_comp = self._log_component_densities(x)  # (N, K)
        return logsumexp(log_comp, axis=1)

    def score(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate score function nabla log pi(x).

        Uses analytic responsibilities (softmax) to avoid autodiff overhead.

        Args:
            x: Positions, shape ``(N, d)``.

        Returns:
            Score vectors, shape ``(N, d)``.
        """
        log_comp = self._log_component_densities(x)  # (N, K)
        r = jax.nn.softmax(log_comp, axis=1)  # (N, K) — responsibilities

        diff = x[:, None, :] - self.means[None, :, :]  # (N, K, d)
        comp_scores = -diff / (self.component_std ** 2)  # (N, K, d)

        return jnp.sum(r[:, :, None] * comp_scores, axis=1)  # (N, d)

    def sample(self, key: jax.Array, n: int) -> jnp.ndarray:
        """Draw exact samples from the mixture.

        Args:
            key: JAX PRNG key.
            n: Number of samples.

        Returns:
            Samples, shape ``(n, d)``.
        """
        k1, k2 = jax.random.split(key)

        # Select component for each sample
        indices = jax.random.categorical(k1, self.log_weights, shape=(n,))
        chosen_means = self.means[indices]  # (n, d)

        # Add Gaussian noise
        noise = jax.random.normal(k2, (n, self.dim)) * self.component_std
        return chosen_means + noise
