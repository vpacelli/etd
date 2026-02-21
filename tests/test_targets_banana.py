"""Tests for BananaTarget (Rosenbrock twist)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from etd.targets.banana import BananaTarget
from etd.types import Target


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def banana_2d():
    return BananaTarget(dim=2, curvature=0.1, offset=100.0, sigma1=10.0, sigma2=1.0)


@pytest.fixture
def banana_5d():
    return BananaTarget(dim=5, curvature=0.05, offset=50.0, sigma1=5.0, sigma2=2.0)


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

class TestShapes:
    def test_log_prob_shape_2d(self, banana_2d):
        x = jnp.ones((50, 2))
        lp = banana_2d.log_prob(x)
        assert lp.shape == (50,)

    def test_log_prob_shape_5d(self, banana_5d):
        x = jnp.ones((30, 5))
        lp = banana_5d.log_prob(x)
        assert lp.shape == (30,)

    def test_score_shape_2d(self, banana_2d):
        x = jnp.ones((50, 2))
        s = banana_2d.score(x)
        assert s.shape == (50, 2)

    def test_score_shape_5d(self, banana_5d):
        x = jnp.ones((30, 5))
        s = banana_5d.score(x)
        assert s.shape == (30, 5)

    def test_sample_shape_2d(self, banana_2d):
        key = jax.random.PRNGKey(0)
        samples = banana_2d.sample(key, 100)
        assert samples.shape == (100, 2)

    def test_sample_shape_5d(self, banana_5d):
        key = jax.random.PRNGKey(0)
        samples = banana_5d.sample(key, 100)
        assert samples.shape == (100, 5)

    def test_mean_shape(self, banana_2d):
        assert banana_2d.mean.shape == (2,)

    def test_variance_shape(self, banana_5d):
        assert banana_5d.variance.shape == (5,)


# ---------------------------------------------------------------------------
# Score correctness
# ---------------------------------------------------------------------------

class TestScore:
    def test_score_matches_autodiff_2d(self, banana_2d):
        """Analytic score must match vmap(grad(log_prob)) to ~1e-5."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (20, 2)) * 3.0

        analytic = banana_2d.score(x)
        autodiff = jax.vmap(jax.grad(lambda xi: banana_2d.log_prob(xi[None])[0]))(x)

        np.testing.assert_allclose(analytic, autodiff, atol=1e-5, rtol=1e-5)

    def test_score_matches_autodiff_5d(self, banana_5d):
        key = jax.random.PRNGKey(123)
        x = jax.random.normal(key, (15, 5)) * 2.0

        analytic = banana_5d.score(x)
        autodiff = jax.vmap(jax.grad(lambda xi: banana_5d.log_prob(xi[None])[0]))(x)

        np.testing.assert_allclose(analytic, autodiff, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Sample statistics
# ---------------------------------------------------------------------------

class TestSampleStatistics:
    def test_sample_mean_2d(self, banana_2d):
        """Sample mean should match analytic mean (10k samples)."""
        key = jax.random.PRNGKey(7)
        samples = banana_2d.sample(key, 10_000)
        emp_mean = jnp.mean(samples, axis=0)

        np.testing.assert_allclose(emp_mean, banana_2d.mean, atol=0.5)

    def test_sample_variance_2d(self, banana_2d):
        """Sample variance should match analytic variance (10k samples)."""
        key = jax.random.PRNGKey(8)
        samples = banana_2d.sample(key, 10_000)
        emp_var = jnp.var(samples, axis=0)

        np.testing.assert_allclose(emp_var, banana_2d.variance, rtol=0.15)

    def test_sample_mean_5d(self, banana_5d):
        key = jax.random.PRNGKey(9)
        samples = banana_5d.sample(key, 10_000)
        emp_mean = jnp.mean(samples, axis=0)

        np.testing.assert_allclose(emp_mean, banana_5d.mean, atol=0.5)


# ---------------------------------------------------------------------------
# Protocol check
# ---------------------------------------------------------------------------

class TestProtocol:
    def test_is_target(self, banana_2d):
        assert isinstance(banana_2d, Target)

    def test_dim_attribute(self, banana_2d):
        assert banana_2d.dim == 2

    def test_dim_attribute_5d(self, banana_5d):
        assert banana_5d.dim == 5


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_log_prob_finite(self, banana_2d):
        """log_prob should be finite for reasonable inputs."""
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (50, 2)) * 5.0
        lp = banana_2d.log_prob(x)
        assert jnp.all(jnp.isfinite(lp))

    def test_dim_validation(self):
        with pytest.raises(ValueError, match="dim >= 2"):
            BananaTarget(dim=1)

    def test_registry(self):
        from etd.targets import TARGETS
        assert "banana" in TARGETS
