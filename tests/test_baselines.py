"""Tests for baseline inference algorithms (ULA, SVGD, MPPI).

Covers init shapes, step shapes, determinism, convergence, registry,
per-particle proposals (MPPI), and bandwidth options (SVGD).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from etd.baselines import (
    BASELINES,
    get_baseline,
    MPPIConfig, MPPIState, mppi_init, mppi_step,
    SVGDConfig, SVGDState, svgd_init, svgd_step,
    ULAConfig, ULAState, ula_init, ula_step,
)
from etd.diagnostics.metrics import mean_error
from etd.targets.gaussian import GaussianTarget


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gaussian_target():
    return GaussianTarget(dim=2)


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


# ---------------------------------------------------------------------------
# Init tests
# ---------------------------------------------------------------------------

class TestInit:
    """Each baseline: shapes correct, step counter = 0, all finite."""

    @pytest.mark.parametrize("method", ["ula", "svgd", "mppi"])
    def test_shapes_and_step(self, method, gaussian_target, key):
        bl = get_baseline(method)
        cfg = bl["config"](n_particles=50)
        state = bl["init"](key, gaussian_target, cfg)

        assert state.positions.shape == (50, 2)
        assert int(state.step) == 0
        assert jnp.all(jnp.isfinite(state.positions))

    def test_svgd_adam_zeros(self, gaussian_target, key):
        """SVGD Adam moments should be initialized to zeros."""
        state = svgd_init(key, gaussian_target, SVGDConfig(n_particles=30))
        assert state.adam_m.shape == (30, 2)
        assert state.adam_v.shape == (30, 2)
        np.testing.assert_array_equal(np.array(state.adam_m), 0.0)
        np.testing.assert_array_equal(np.array(state.adam_v), 0.0)

    @pytest.mark.parametrize("method", ["ula", "svgd", "mppi"])
    def test_custom_positions(self, method, gaussian_target, key):
        """Explicit init_positions should be used verbatim."""
        bl = get_baseline(method)
        custom = jnp.ones((50, 2)) * 3.14
        cfg = bl["config"](n_particles=50)
        state = bl["init"](key, gaussian_target, cfg, init_positions=custom)
        np.testing.assert_array_equal(np.array(state.positions), np.array(custom))


# ---------------------------------------------------------------------------
# Step tests
# ---------------------------------------------------------------------------

class TestStep:
    """Output shapes, step increments, all finite, info keys present."""

    def test_ula_step(self, gaussian_target, key):
        cfg = ULAConfig(n_particles=30)
        k1, k2 = jax.random.split(key)
        state = ula_init(k1, gaussian_target, cfg)
        new_state, info = ula_step(k2, state, gaussian_target, cfg)

        assert new_state.positions.shape == (30, 2)
        assert int(new_state.step) == 1
        assert jnp.all(jnp.isfinite(new_state.positions))
        assert "score_norm" in info

    def test_svgd_step(self, gaussian_target, key):
        cfg = SVGDConfig(n_particles=30)
        state = svgd_init(key, gaussian_target, cfg)
        new_state, info = svgd_step(key, state, gaussian_target, cfg)

        assert new_state.positions.shape == (30, 2)
        assert new_state.adam_m.shape == (30, 2)
        assert new_state.adam_v.shape == (30, 2)
        assert int(new_state.step) == 1
        assert jnp.all(jnp.isfinite(new_state.positions))
        assert "bandwidth" in info
        assert "phi_norm" in info

    def test_mppi_step(self, gaussian_target, key):
        cfg = MPPIConfig(n_particles=30, n_proposals=10)
        k1, k2 = jax.random.split(key)
        state = mppi_init(k1, gaussian_target, cfg)
        new_state, info = mppi_step(k2, state, gaussian_target, cfg)

        assert new_state.positions.shape == (30, 2)
        assert int(new_state.step) == 1
        assert jnp.all(jnp.isfinite(new_state.positions))
        assert "log_w_max" in info


# ---------------------------------------------------------------------------
# Determinism tests
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Same key → same output (ULA, MPPI); SVGD deterministic regardless."""

    def test_ula_determinism(self, gaussian_target, key):
        cfg = ULAConfig(n_particles=20)
        k1, k2 = jax.random.split(key)
        state = ula_init(k1, gaussian_target, cfg)
        s1, _ = ula_step(k2, state, gaussian_target, cfg)
        s2, _ = ula_step(k2, state, gaussian_target, cfg)
        np.testing.assert_array_equal(np.array(s1.positions), np.array(s2.positions))

    def test_mppi_determinism(self, gaussian_target, key):
        cfg = MPPIConfig(n_particles=20, n_proposals=10)
        k1, k2 = jax.random.split(key)
        state = mppi_init(k1, gaussian_target, cfg)
        s1, _ = mppi_step(k2, state, gaussian_target, cfg)
        s2, _ = mppi_step(k2, state, gaussian_target, cfg)
        np.testing.assert_array_equal(np.array(s1.positions), np.array(s2.positions))

    def test_svgd_deterministic(self, gaussian_target, key):
        """SVGD is deterministic: different keys yield same output."""
        cfg = SVGDConfig(n_particles=20)
        state = svgd_init(key, gaussian_target, cfg)
        k1, k2 = jax.random.split(key)
        s1, _ = svgd_step(k1, state, gaussian_target, cfg)
        s2, _ = svgd_step(k2, state, gaussian_target, cfg)
        np.testing.assert_array_equal(np.array(s1.positions), np.array(s2.positions))


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_get_baseline_unknown(self):
        with pytest.raises(KeyError, match="Unknown baseline"):
            get_baseline("nonexistent")

    @pytest.mark.parametrize("method", ["svgd", "ula", "mppi"])
    def test_get_baseline_returns_dict(self, method):
        bl = get_baseline(method)
        assert "config" in bl
        assert "init" in bl
        assert "step" in bl

    def test_baselines_count(self):
        assert len(BASELINES) == 3


# ---------------------------------------------------------------------------
# Convergence tests
# ---------------------------------------------------------------------------

class TestULAConvergence:
    """ULA on isotropic Gaussian(d=2) should converge in 200 steps."""

    def test_mean_error(self, gaussian_target):
        cfg = ULAConfig(n_particles=100, step_size=0.01, n_iterations=200)
        key = jax.random.PRNGKey(0)
        k_init, k_run = jax.random.split(key)
        state = ula_init(k_init, gaussian_target, cfg)

        for _ in range(200):
            k_run, k_step = jax.random.split(k_run)
            state, _ = ula_step(k_step, state, gaussian_target, cfg)

        err = mean_error(state.positions, gaussian_target.mean)
        assert float(err) < 0.3, f"ULA mean_error = {float(err):.4f} >= 0.3"


class TestSVGDConvergence:
    """SVGD on isotropic Gaussian(d=2) should converge in 200 steps."""

    def test_mean_error(self, gaussian_target):
        cfg = SVGDConfig(n_particles=100, learning_rate=0.1, n_iterations=200)
        key = jax.random.PRNGKey(0)
        state = svgd_init(key, gaussian_target, cfg)

        for i in range(200):
            state, _ = svgd_step(key, state, gaussian_target, cfg)

        err = mean_error(state.positions, gaussian_target.mean)
        assert float(err) < 0.2, f"SVGD mean_error = {float(err):.4f} >= 0.2"


# ---------------------------------------------------------------------------
# Algorithm-specific tests
# ---------------------------------------------------------------------------

class TestMPPIPerParticle:
    """Verify MPPI proposals are per-particle (N, M, d) not pooled."""

    def test_proposals_shape(self, gaussian_target):
        """MPPI step should not pool proposals — each particle gets M."""
        # We verify this by checking that particles move independently:
        # set one particle far from target, others at origin.
        N, M = 5, 20
        cfg = MPPIConfig(n_particles=N, n_proposals=M, sigma=0.1, temperature=1.0)
        positions = jnp.zeros((N, 2))
        positions = positions.at[0].set(jnp.array([10.0, 10.0]))
        state = MPPIState(positions=positions, step=0)

        key = jax.random.PRNGKey(7)
        new_state, _ = mppi_step(key, state, gaussian_target, cfg)

        # The outlier particle should move more than the central ones
        displacement = jnp.linalg.norm(
            new_state.positions - positions, axis=-1
        )
        assert float(displacement[0]) > float(jnp.mean(displacement[1:]))


class TestSVGDBandwidth:
    """Fixed bandwidth vs median heuristic both produce finite results."""

    def test_median_heuristic(self, gaussian_target, key):
        cfg = SVGDConfig(n_particles=30, bandwidth=-1.0)
        state = svgd_init(key, gaussian_target, cfg)
        new_state, info = svgd_step(key, state, gaussian_target, cfg)
        assert jnp.all(jnp.isfinite(new_state.positions))
        assert float(info["bandwidth"]) > 0

    def test_fixed_bandwidth(self, gaussian_target, key):
        cfg = SVGDConfig(n_particles=30, bandwidth=1.0)
        state = svgd_init(key, gaussian_target, cfg)
        new_state, info = svgd_step(key, state, gaussian_target, cfg)
        assert jnp.all(jnp.isfinite(new_state.positions))
        assert float(info["bandwidth"]) == pytest.approx(1.0)
