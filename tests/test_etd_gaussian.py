"""Phase 1 gate tests and supporting tests.

Gate tests (from ROADMAP.md):
  - ETD on GaussianTarget(dim=2) converges: mean_error < 0.1
  - Variance ratio in [0.8, 1.2] for all dimensions.

Supporting tests cover init/step shapes, determinism, targets,
and diagnostic metrics.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from etd.diagnostics.metrics import (
    energy_distance,
    mean_error,
    mode_coverage,
    variance_ratio,
)
from etd.step import init, step
from etd.targets.gaussian import GaussianTarget
from etd.targets.gmm import GMMTarget
from etd.types import ETDConfig, ETDState, Target


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gaussian_target():
    return GaussianTarget(dim=2)


@pytest.fixture
def gmm_target():
    return GMMTarget(dim=2, n_modes=4, arrangement="grid", separation=6.0)


@pytest.fixture
def balanced_config():
    return ETDConfig(
        coupling="balanced",
        epsilon=0.1,
        alpha=0.05,
        n_particles=100,
        n_proposals=25,
        n_iterations=200,
    )


@pytest.fixture
def converged_state(gaussian_target, balanced_config):
    """Run 200 iterations on Gaussian and return final state."""
    key = jax.random.PRNGKey(42)
    k_init, k_run = jax.random.split(key)
    state = init(k_init, gaussian_target, balanced_config)

    for i in range(balanced_config.n_iterations):
        k_run, k_step = jax.random.split(k_run)
        state, _ = step(k_step, state, gaussian_target, balanced_config)

    return state


# ---------------------------------------------------------------------------
# Gate tests
# ---------------------------------------------------------------------------

class TestETDGaussianGate:
    """Phase 1 convergence gate tests."""

    def test_mean_error(self, converged_state, gaussian_target):
        """Particle centroid should be close to origin after 200 iters."""
        err = mean_error(converged_state.positions, gaussian_target.mean)
        assert float(err) < 0.1, f"mean_error = {float(err):.4f} >= 0.1"

    def test_variance_ratio(self, converged_state, gaussian_target):
        """Per-dimension variance should be within [0.8, 1.2] of truth."""
        ratios = variance_ratio(
            converged_state.positions, gaussian_target.variance
        )
        ratios_np = np.array(ratios)
        assert np.all(ratios_np > 0.8), (
            f"variance_ratio too low: {ratios_np}"
        )
        assert np.all(ratios_np < 1.2), (
            f"variance_ratio too high: {ratios_np}"
        )


# ---------------------------------------------------------------------------
# Init tests
# ---------------------------------------------------------------------------

class TestInit:
    def test_shapes(self, gaussian_target, balanced_config):
        """All state fields should have correct shapes."""
        key = jax.random.PRNGKey(0)
        state = init(key, gaussian_target, balanced_config)

        N = balanced_config.n_particles
        d = gaussian_target.dim
        M = balanced_config.n_proposals

        assert state.positions.shape == (N, d)
        assert state.dual_f.shape == (N,)
        assert state.dual_g.shape == (N * M,)
        assert state.precond_accum.shape == (d,)
        assert int(state.step) == 0

    def test_custom_positions(self, gaussian_target, balanced_config):
        """Explicit init_positions should be used verbatim."""
        key = jax.random.PRNGKey(1)
        custom = jnp.ones((100, 2)) * 3.14
        state = init(key, gaussian_target, balanced_config, init_positions=custom)

        np.testing.assert_array_equal(np.array(state.positions), np.array(custom))

    def test_determinism(self, gaussian_target, balanced_config):
        """Same key should produce identical initial state."""
        key = jax.random.PRNGKey(42)
        s1 = init(key, gaussian_target, balanced_config)
        s2 = init(key, gaussian_target, balanced_config)

        np.testing.assert_array_equal(
            np.array(s1.positions), np.array(s2.positions)
        )


# ---------------------------------------------------------------------------
# Step tests
# ---------------------------------------------------------------------------

class TestStep:
    def test_returns_valid_state(self, gaussian_target, balanced_config):
        """Step output should have correct shapes, incremented step, all finite."""
        key = jax.random.PRNGKey(0)
        k_init, k_step = jax.random.split(key)
        state = init(k_init, gaussian_target, balanced_config)

        new_state, info = step(k_step, state, gaussian_target, balanced_config)

        N = balanced_config.n_particles
        d = gaussian_target.dim
        M = balanced_config.n_proposals

        assert new_state.positions.shape == (N, d)
        assert new_state.dual_f.shape == (N,)
        assert new_state.dual_g.shape == (N * M,)
        assert new_state.precond_accum.shape == (d,)
        assert int(new_state.step) == 1

        assert jnp.all(jnp.isfinite(new_state.positions))
        assert jnp.all(jnp.isfinite(new_state.dual_f))
        assert jnp.all(jnp.isfinite(new_state.dual_g))

        assert "sinkhorn_iters" in info
        assert "cost_median" in info

    def test_determinism(self, gaussian_target, balanced_config):
        """Same key + state should produce identical output."""
        key = jax.random.PRNGKey(0)
        k_init, k_step = jax.random.split(key)
        state = init(k_init, gaussian_target, balanced_config)

        s1, _ = step(k_step, state, gaussian_target, balanced_config)
        s2, _ = step(k_step, state, gaussian_target, balanced_config)

        np.testing.assert_array_equal(
            np.array(s1.positions), np.array(s2.positions)
        )

    def test_gibbs_coupling(self, gaussian_target):
        """Step with Gibbs coupling should run without error."""
        config = ETDConfig(
            coupling="gibbs",
            epsilon=0.1,
            alpha=0.05,
            n_particles=50,
            n_proposals=20,
        )
        key = jax.random.PRNGKey(0)
        k_init, k_step = jax.random.split(key)
        state = init(k_init, gaussian_target, config)

        new_state, info = step(k_step, state, gaussian_target, config)

        assert new_state.positions.shape == (50, 2)
        assert int(info["sinkhorn_iters"]) == 0
        assert jnp.all(jnp.isfinite(new_state.positions))


# ---------------------------------------------------------------------------
# Target tests
# ---------------------------------------------------------------------------

class TestTargets:
    def test_gaussian_log_prob_shape(self, gaussian_target):
        x = jnp.zeros((10, 2))
        lp = gaussian_target.log_prob(x)
        assert lp.shape == (10,)

    def test_gaussian_score_shape(self, gaussian_target):
        x = jnp.zeros((10, 2))
        s = gaussian_target.score(x)
        assert s.shape == (10, 2)

    def test_gaussian_score_matches_autodiff(self, gaussian_target):
        """Analytic score should match JAX autograd."""
        key = jax.random.PRNGKey(99)
        x = jax.random.normal(key, (20, 2))

        analytic = gaussian_target.score(x)
        # vmap grad over batch dimension
        autodiff = jax.vmap(jax.grad(lambda xi: gaussian_target.log_prob(xi[None])[0]))(x)

        np.testing.assert_allclose(
            np.array(analytic), np.array(autodiff), atol=1e-5
        )

    def test_gaussian_sample_shape(self, gaussian_target):
        key = jax.random.PRNGKey(0)
        samples = gaussian_target.sample(key, 50)
        assert samples.shape == (50, 2)

    def test_gmm_log_prob_shape(self, gmm_target):
        x = jnp.zeros((10, 2))
        lp = gmm_target.log_prob(x)
        assert lp.shape == (10,)

    def test_gmm_score_matches_autodiff(self, gmm_target):
        """GMM analytic score should match JAX autograd."""
        key = jax.random.PRNGKey(99)
        x = jax.random.normal(key, (20, 2))

        analytic = gmm_target.score(x)
        autodiff = jax.vmap(jax.grad(lambda xi: gmm_target.log_prob(xi[None])[0]))(x)

        np.testing.assert_allclose(
            np.array(analytic), np.array(autodiff), atol=1e-5
        )

    def test_gmm_ring_arrangement(self):
        """Ring arrangement: all means should be at radius = separation/2."""
        sep = 8.0
        target = GMMTarget(dim=2, n_modes=6, arrangement="ring", separation=sep)
        expected_radius = sep / 2.0

        radii = jnp.linalg.norm(target.means, axis=-1)
        np.testing.assert_allclose(
            np.array(radii), expected_radius, atol=1e-5
        )

    def test_gmm_sample_shape(self, gmm_target):
        key = jax.random.PRNGKey(0)
        samples = gmm_target.sample(key, 50)
        assert samples.shape == (50, 2)

    def test_gmm_satisfies_target_protocol(self, gmm_target):
        """GMMTarget should satisfy the runtime-checkable Target protocol."""
        assert isinstance(gmm_target, Target)

    def test_gaussian_satisfies_target_protocol(self, gaussian_target):
        """GaussianTarget should satisfy the runtime-checkable Target protocol."""
        assert isinstance(gaussian_target, Target)


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_energy_distance_self_near_zero(self):
        """Energy distance of a distribution with itself should be ~0."""
        key = jax.random.PRNGKey(0)
        X = jax.random.normal(key, (200, 3))

        ed = energy_distance(X, X)
        # V-statistic with diagonal is exactly 0 for same array
        assert float(ed) < 0.01, f"energy_distance(X, X) = {float(ed)}"

    def test_energy_distance_positive_for_different(self):
        """Energy distance should be positive for different distributions."""
        key = jax.random.PRNGKey(1)
        k1, k2 = jax.random.split(key)
        X = jax.random.normal(k1, (200, 3))
        Y = jax.random.normal(k2, (200, 3)) + 5.0  # shifted

        ed = energy_distance(X, Y)
        assert float(ed) > 0.5, f"energy_distance should be large, got {float(ed)}"

    def test_mean_error_known_offset(self):
        """Mean error for particles at known offset."""
        positions = jnp.ones((50, 3)) * 2.0
        err = mean_error(positions, jnp.zeros(3))

        expected = jnp.sqrt(3.0) * 2.0  # ||[2,2,2]||
        np.testing.assert_allclose(float(err), float(expected), atol=1e-5)

    def test_mode_coverage_partial(self):
        """Mode coverage should detect partial coverage."""
        # 2 of 4 modes have nearby particles
        modes = jnp.array([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=jnp.float32)
        particles = jnp.array([
            [0.1, 0.1],   # covers mode 0
            [10.1, 0.1],  # covers mode 1
            [5.0, 5.0],   # doesn't cover any mode
        ], dtype=jnp.float32)

        cov = mode_coverage(particles, modes, tolerance=2.0)
        assert float(cov) == pytest.approx(0.5, abs=0.01)
