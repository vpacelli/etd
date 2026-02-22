"""Bayesian logistic regression target distribution.

The posterior for logistic regression with a Gaussian prior:

    theta ~ N(0, prior_std^2 * I)
    y_i | x_i, theta ~ Bernoulli(sigmoid(x_i . theta))

This is the standard BLR benchmark for variational inference, where the
target is the unnormalized posterior log p(theta | X, y).

Numerical note
--------------
Uses ``jax.nn.log_sigmoid`` for stable log-likelihood computation
(avoids log(0) when sigmoid saturates).  The analytic score bypasses
autodiff: ``grad = X.T @ (y - sigmoid(X @ theta)) - theta / prior_std^2``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


class BLRTarget:
    """Bayesian logistic regression posterior.

    Satisfies the :class:`~etd.types.Target` protocol.

    Attributes:
        dim: Number of features (= number of regression coefficients).
        X: Design matrix, shape ``(n_data, dim)``.
        y: Binary labels, shape ``(n_data,)``.
        prior_std: Prior standard deviation for each coefficient.
    """

    def __init__(
        self,
        X: jnp.ndarray | None = None,
        y: jnp.ndarray | None = None,
        dataset: str | None = None,
        prior_std: float = 5.0,
    ):
        """
        Args:
            X: Design matrix, shape ``(n_data, d)``.  If None, loads from DuckDB.
            y: Binary labels, shape ``(n_data,)``.  If None, loads from DuckDB.
            dataset: DuckDB table name (e.g. ``"german_credit"``).
                Used when X/y are None.
            prior_std: Prior standard deviation for each coefficient.
        """
        if X is None and dataset is not None:
            from experiments.datasets import load_dataset
            X_np, y_np = load_dataset(dataset)
            X = X_np
            y = y_np
        elif X is None:
            raise ValueError("Must provide either (X, y) or dataset name")

        self.X = jnp.asarray(X, dtype=jnp.float32)   # (n_data, d)
        self.y = jnp.asarray(y, dtype=jnp.float32)    # (n_data,)
        self.dim = self.X.shape[1]
        self.prior_std = prior_std

    @property
    def mean(self) -> jnp.ndarray:
        """Prior mean (used for initialization reference), shape ``(d,)``."""
        return jnp.zeros(self.dim)

    @property
    def variance(self) -> jnp.ndarray:
        """Prior variance, shape ``(d,)``."""
        return jnp.full(self.dim, self.prior_std ** 2)

    def log_prob(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Evaluate unnormalized log-posterior.

        log p(theta | X, y) propto log p(y | X, theta) + log p(theta)

        Uses ``jax.nn.log_sigmoid`` for numerical stability.

        Args:
            theta: Parameter vectors, shape ``(N, d)``.

        Returns:
            Log-posterior values, shape ``(N,)``.
        """
        # Logits: (n_data, N)
        logits = self.X @ theta.T

        # Log-likelihood: sum_i [y_i * log(sig) + (1-y_i) * log(1-sig)]
        # = sum_i [y_i * log_sigmoid(logit) + (1-y_i) * log_sigmoid(-logit)]
        log_lik = jnp.sum(
            self.y[:, None] * jax.nn.log_sigmoid(logits)
            + (1.0 - self.y[:, None]) * jax.nn.log_sigmoid(-logits),
            axis=0,
        )  # (N,)

        # Log-prior: -0.5 * ||theta||^2 / prior_std^2 + const
        log_prior = -0.5 * jnp.sum(theta ** 2, axis=1) / self.prior_std ** 2

        return log_lik + log_prior

    def score(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Evaluate score function nabla log p(theta | X, y).

        Analytic gradient (avoids autodiff overhead):

        grad_lik  = X.T @ (y - sigmoid(X @ theta))  per particle
        grad_prior = -theta / prior_std^2

        Args:
            theta: Parameter vectors, shape ``(N, d)``.

        Returns:
            Score vectors, shape ``(N, d)``.
        """
        logits = self.X @ theta.T             # (n_data, N)
        residuals = self.y[:, None] - jax.nn.sigmoid(logits)  # (n_data, N)

        grad_lik = (self.X.T @ residuals).T   # (N, d)
        grad_prior = -theta / self.prior_std ** 2  # (N, d)

        return grad_lik + grad_prior
