"""Tests for Cholesky preconditioner computation.

Covers: known covariance recovery, shrinkage, EMA, jitter, PD guarantee.
"""

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from etd.proposals.preconditioner import (
    compute_diagonal_P,
    compute_ensemble_cholesky,
    update_rmsprop_accum,
)
from etd.types import PreconditionerConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_correlated(key, n, d, cov):
    """Sample from N(0, cov) using Cholesky factoring."""
    L = jnp.linalg.cholesky(cov)
    z = jax.random.normal(key, (n, d))
    return z @ L.T


# ---------------------------------------------------------------------------
# RMSProp backward compat
# ---------------------------------------------------------------------------

class TestRMSPropCompat:
    """update_rmsprop_accum should match the original update_preconditioner."""

    def test_matches_original(self):
        key = jax.random.key(0)
        scores = jax.random.normal(key, (50, 5))
        accum = jnp.ones(5)
        new_accum = update_rmsprop_accum(accum, scores, beta=0.9)
        expected = 0.9 * accum + 0.1 * jnp.mean(scores ** 2, axis=0)
        npt.assert_allclose(new_accum, expected, atol=1e-6)

    def test_compute_diagonal_P(self):
        accum = jnp.array([1.0, 4.0, 9.0])
        P = compute_diagonal_P(accum, delta=0.0)
        npt.assert_allclose(P, jnp.array([1.0, 0.5, 1.0 / 3.0]), atol=1e-6)


# ---------------------------------------------------------------------------
# Cholesky computation
# ---------------------------------------------------------------------------

class TestComputeEnsembleCholesky:
    """Unit tests for compute_ensemble_cholesky."""

    def test_known_covariance_recovery(self):
        """L @ L.T should approximate the true covariance (large N)."""
        key = jax.random.key(42)
        d = 3
        true_cov = jnp.array([
            [2.0, 0.5, 0.0],
            [0.5, 1.0, 0.3],
            [0.0, 0.3, 1.5],
        ])
        data = _sample_correlated(key, 5000, d, true_cov)
        config = PreconditionerConfig(
            type="cholesky", shrinkage=0.0, jitter=0.0, ema_beta=0.0,
        )
        L = compute_ensemble_cholesky(data, jnp.eye(d), config)
        recovered_cov = L @ L.T
        npt.assert_allclose(recovered_cov, true_cov, atol=0.15)

    def test_shrinkage_one_gives_diagonal(self):
        """shrinkage=1.0 should produce a diagonal Cholesky factor."""
        key = jax.random.key(1)
        d = 4
        data = jax.random.normal(key, (200, d))
        config = PreconditionerConfig(
            type="cholesky", shrinkage=1.0, jitter=1e-6, ema_beta=0.0,
        )
        L = compute_ensemble_cholesky(data, jnp.eye(d), config)
        # Off-diagonal should be ~0
        off_diag = L - jnp.diag(jnp.diag(L))
        assert jnp.max(jnp.abs(off_diag)) < 1e-4

    def test_ema_zero_ignores_prev(self):
        """ema_beta=0.0 should ignore the previous L entirely."""
        key = jax.random.key(2)
        d = 3
        data = jax.random.normal(key, (100, d))
        prev_L = 10.0 * jnp.eye(d)  # very different scale

        config = PreconditionerConfig(
            type="cholesky", shrinkage=0.1, jitter=1e-6, ema_beta=0.0,
        )
        L = compute_ensemble_cholesky(data, prev_L, config)
        # L should not be anywhere near 10*I
        assert jnp.max(jnp.abs(jnp.diag(L))) < 5.0

    def test_ema_blends_previous(self):
        """ema_beta=0.9 should heavily weight the previous covariance."""
        key = jax.random.key(3)
        d = 2
        data = jax.random.normal(key, (100, d)) * 0.1  # small variance data

        # Previous L encodes large variance
        prev_L = 5.0 * jnp.eye(d)

        config_no_ema = PreconditionerConfig(
            type="cholesky", shrinkage=0.1, jitter=1e-6, ema_beta=0.0,
        )
        config_ema = PreconditionerConfig(
            type="cholesky", shrinkage=0.1, jitter=1e-6, ema_beta=0.9,
        )
        L_fresh = compute_ensemble_cholesky(data, prev_L, config_no_ema)
        L_ema = compute_ensemble_cholesky(data, prev_L, config_ema)

        # EMA result should have much larger diagonal than fresh
        assert jnp.mean(jnp.diag(L_ema)) > 3.0 * jnp.mean(jnp.diag(L_fresh))

    def test_output_lower_triangular(self):
        """Output L should be lower-triangular."""
        key = jax.random.key(4)
        d = 5
        data = jax.random.normal(key, (100, d))
        config = PreconditionerConfig(
            type="cholesky", shrinkage=0.1, jitter=1e-6, ema_beta=0.0,
        )
        L = compute_ensemble_cholesky(data, jnp.eye(d), config)
        # Upper triangle (above diagonal) should be zero
        upper = jnp.triu(L, k=1)
        assert jnp.max(jnp.abs(upper)) < 1e-10

    def test_output_positive_diagonal(self):
        """Diagonal entries of L should be positive (PD)."""
        key = jax.random.key(5)
        d = 4
        data = jax.random.normal(key, (100, d))
        config = PreconditionerConfig(
            type="cholesky", shrinkage=0.1, jitter=1e-6, ema_beta=0.0,
        )
        L = compute_ensemble_cholesky(data, jnp.eye(d), config)
        assert jnp.all(jnp.diag(L) > 0)

    def test_jitter_prevents_singularity(self):
        """Degenerate input (constant) should still produce valid L."""
        d = 3
        data = jnp.ones((50, d))  # rank-0 — Σ_hat = 0
        config = PreconditionerConfig(
            type="cholesky", shrinkage=0.0, jitter=1e-4, ema_beta=0.0,
        )
        L = compute_ensemble_cholesky(data, jnp.eye(d), config)
        # Should get sqrt(jitter) * I
        assert jnp.all(jnp.isfinite(L))
        assert jnp.all(jnp.diag(L) > 0)

    def test_shape_matches_input(self):
        """Output shape should be (d, d)."""
        key = jax.random.key(6)
        for d in [2, 5, 10]:
            data = jax.random.normal(key, (100, d))
            config = PreconditionerConfig(
                type="cholesky", shrinkage=0.1, jitter=1e-6, ema_beta=0.0,
            )
            L = compute_ensemble_cholesky(data, jnp.eye(d), config)
            assert L.shape == (d, d)
