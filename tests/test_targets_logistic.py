"""Tests for LogisticRegressionTarget (BLR)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from etd.targets.logistic import LogisticRegressionTarget
from etd.types import Target


# ---------------------------------------------------------------------------
# Fixtures: synthetic data
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_data():
    """Small synthetic dataset (50 samples, 5 features)."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 5)).astype(np.float32)
    w_true = rng.standard_normal(5).astype(np.float32)
    logits = X @ w_true
    y = (logits > 0).astype(np.float32)
    return X, y


@pytest.fixture
def blr_target(synthetic_data):
    X, y = synthetic_data
    return LogisticRegressionTarget(X=X, y=y, prior_std=5.0)


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

class TestShapes:
    def test_log_prob_shape(self, blr_target):
        theta = jnp.zeros((10, 5))
        lp = blr_target.log_prob(theta)
        assert lp.shape == (10,)

    def test_score_shape(self, blr_target):
        theta = jnp.zeros((10, 5))
        s = blr_target.score(theta)
        assert s.shape == (10, 5)

    def test_dim(self, blr_target):
        assert blr_target.dim == 5

    def test_mean_shape(self, blr_target):
        assert blr_target.mean.shape == (5,)

    def test_variance_shape(self, blr_target):
        assert blr_target.variance.shape == (5,)


# ---------------------------------------------------------------------------
# Score correctness
# ---------------------------------------------------------------------------

class TestScore:
    def test_score_matches_autodiff(self, blr_target):
        """Analytic score must match vmap(grad(log_prob))."""
        key = jax.random.PRNGKey(42)
        theta = jax.random.normal(key, (8, 5)) * 0.5

        analytic = blr_target.score(theta)
        autodiff = jax.vmap(
            jax.grad(lambda t: blr_target.log_prob(t[None])[0])
        )(theta)

        np.testing.assert_allclose(analytic, autodiff, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------

class TestStability:
    def test_log_prob_finite_at_extremes(self, blr_target):
        """log_prob should remain finite even for large theta."""
        theta_large = jnp.ones((5, 5)) * 10.0
        lp = blr_target.log_prob(theta_large)
        assert jnp.all(jnp.isfinite(lp)), f"Non-finite log_prob: {lp}"

    def test_log_prob_finite_at_zero(self, blr_target):
        theta_zero = jnp.zeros((3, 5))
        lp = blr_target.log_prob(theta_zero)
        assert jnp.all(jnp.isfinite(lp))

    def test_score_finite(self, blr_target):
        key = jax.random.PRNGKey(0)
        theta = jax.random.normal(key, (10, 5)) * 2.0
        s = blr_target.score(theta)
        assert jnp.all(jnp.isfinite(s))


# ---------------------------------------------------------------------------
# Protocol check
# ---------------------------------------------------------------------------

class TestProtocol:
    def test_is_target(self, blr_target):
        assert isinstance(blr_target, Target)

    def test_registry(self):
        from etd.targets import TARGETS
        assert "logistic" in TARGETS


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_requires_data_or_dataset(self):
        with pytest.raises(ValueError, match="Must provide"):
            LogisticRegressionTarget()

    def test_prior_variance(self, blr_target):
        """Variance should be prior_std^2."""
        np.testing.assert_allclose(blr_target.variance, 25.0)
