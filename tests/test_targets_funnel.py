"""Tests for FunnelTarget (Neal's funnel)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from etd.targets.funnel import FunnelTarget
from etd.types import Target


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def funnel_10d():
    return FunnelTarget(dim=10, sigma_v=3.0)


@pytest.fixture
def funnel_3d():
    return FunnelTarget(dim=3, sigma_v=2.0)


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

class TestShapes:
    def test_log_prob_shape(self, funnel_10d):
        x = jnp.ones((50, 10))
        lp = funnel_10d.log_prob(x)
        assert lp.shape == (50,)

    def test_score_shape(self, funnel_10d):
        x = jnp.ones((50, 10))
        s = funnel_10d.score(x)
        assert s.shape == (50, 10)

    def test_sample_shape(self, funnel_10d):
        key = jax.random.PRNGKey(0)
        samples = funnel_10d.sample(key, 100)
        assert samples.shape == (100, 10)

    def test_mean_shape(self, funnel_10d):
        assert funnel_10d.mean.shape == (10,)

    def test_variance_shape(self, funnel_10d):
        assert funnel_10d.variance.shape == (10,)


# ---------------------------------------------------------------------------
# Score correctness
# ---------------------------------------------------------------------------

class TestScore:
    def test_score_matches_autodiff(self, funnel_3d):
        """Analytic score must match vmap(grad(log_prob))."""
        key = jax.random.PRNGKey(42)
        # Use moderate v values to avoid exp(-v) overflow
        x = jax.random.normal(key, (20, 3)) * 1.5

        analytic = funnel_3d.score(x)
        autodiff = jax.vmap(jax.grad(lambda xi: funnel_3d.log_prob(xi[None])[0]))(x)

        np.testing.assert_allclose(analytic, autodiff, atol=1e-5, rtol=1e-5)

    def test_score_matches_autodiff_10d(self, funnel_10d):
        key = jax.random.PRNGKey(123)
        x = jax.random.normal(key, (15, 10)) * 1.0

        analytic = funnel_10d.score(x)
        autodiff = jax.vmap(jax.grad(lambda xi: funnel_10d.log_prob(xi[None])[0]))(x)

        np.testing.assert_allclose(analytic, autodiff, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Sample statistics
# ---------------------------------------------------------------------------

class TestSampleStatistics:
    def test_sample_mean(self, funnel_10d):
        """Sample mean should be near zero (large sample)."""
        key = jax.random.PRNGKey(7)
        samples = funnel_10d.sample(key, 20_000)
        emp_mean = jnp.mean(samples, axis=0)

        # All dimensions have mean 0
        np.testing.assert_allclose(emp_mean, jnp.zeros(10), atol=0.5)

    def test_sample_v_variance(self, funnel_10d):
        """Var[v] should match sigma_v^2."""
        key = jax.random.PRNGKey(8)
        samples = funnel_10d.sample(key, 20_000)
        v_var = jnp.var(samples[:, 0])

        np.testing.assert_allclose(v_var, 9.0, rtol=0.1)  # sigma_v=3 → var=9


# ---------------------------------------------------------------------------
# Funnel geometry
# ---------------------------------------------------------------------------

class TestFunnelGeometry:
    def test_std_increases_with_v(self, funnel_10d):
        """For bins of v, std(x_k) should increase with v."""
        key = jax.random.PRNGKey(42)
        samples = funnel_10d.sample(key, 50_000)
        v = samples[:, 0]
        x_tail = samples[:, 1:]

        # Bin by v: low (v < -2) vs high (v > 2)
        mask_low = v < -2.0
        mask_high = v > 2.0

        std_low = jnp.std(x_tail[mask_low], axis=0)
        std_high = jnp.std(x_tail[mask_high], axis=0)

        # In the funnel mouth (high v), std should be much larger
        assert jnp.all(std_high > std_low), \
            f"std_high={std_high}, std_low={std_low}"

    def test_neck_and_mouth_populated(self, funnel_10d):
        """Exact samples should populate both neck and mouth."""
        key = jax.random.PRNGKey(0)
        samples = funnel_10d.sample(key, 10_000)
        v = samples[:, 0]

        n_neck = jnp.sum(v < -2.0)
        n_mouth = jnp.sum(v > 2.0)

        # With sigma_v=3 and 10k samples, both regions should have many points
        assert n_neck > 100, f"Only {n_neck} samples in neck"
        assert n_mouth > 100, f"Only {n_mouth} samples in mouth"


# ---------------------------------------------------------------------------
# Score finiteness
# ---------------------------------------------------------------------------

class TestScoreFiniteness:
    def test_score_finite_moderate_v(self, funnel_10d):
        """Score should be finite for v in [-5, 5]."""
        key = jax.random.PRNGKey(0)
        n = 50
        v = jax.random.uniform(key, (n, 1), minval=-5.0, maxval=5.0)
        x_tail = jax.random.normal(jax.random.PRNGKey(1), (n, 9)) * 0.5
        x = jnp.concatenate([v, x_tail], axis=1)

        s = funnel_10d.score(x)
        assert jnp.all(jnp.isfinite(s)), f"Non-finite scores at v in [-5,5]"


# ---------------------------------------------------------------------------
# Protocol and registry
# ---------------------------------------------------------------------------

class TestProtocol:
    def test_is_target(self, funnel_10d):
        assert isinstance(funnel_10d, Target)

    def test_dim_attribute(self, funnel_10d):
        assert funnel_10d.dim == 10

    def test_dim_validation(self):
        with pytest.raises(ValueError, match="dim >= 2"):
            FunnelTarget(dim=1)

    def test_registry(self):
        from etd.targets import TARGETS
        assert "funnel" in TARGETS
