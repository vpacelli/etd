"""Tests for SDD-RB extension."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from etd.extensions.sdd import SDDConfig, SDDState, init, step
from etd.targets.gaussian import GaussianTarget


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def target():
    return GaussianTarget(dim=2)


@pytest.fixture
def config():
    return SDDConfig(
        n_particles=20,
        n_proposals=10,
        n_iterations=50,
        epsilon=0.1,
        self_epsilon=0.1,
        alpha=0.05,
        sdd_step_size=0.5,
    )


@pytest.fixture
def state(target, config):
    key = jax.random.PRNGKey(42)
    return init(key, target, config)


# ---------------------------------------------------------------------------
# Init tests
# ---------------------------------------------------------------------------

class TestInit:
    def test_positions_shape(self, state, config):
        assert state.positions.shape == (config.n_particles, 2)

    def test_dual_f_cross_shape(self, state, config):
        assert state.dual_f_cross.shape == (config.n_particles,)

    def test_dual_g_cross_shape(self, state, config):
        assert state.dual_g_cross.shape == (config.n_particles * config.n_proposals,)

    def test_dual_f_self_shape(self, state, config):
        assert state.dual_f_self.shape == (config.n_particles,)

    def test_dual_g_self_shape(self, state, config):
        assert state.dual_g_self.shape == (config.n_particles,)

    def test_precond_accum_shape(self, state):
        assert state.precond_accum.shape == (2,)

    def test_step_counter_zero(self, state):
        assert state.step == 0

    def test_custom_init_positions(self, target, config):
        positions = jnp.ones((20, 2))
        s = init(jax.random.PRNGKey(0), target, config, init_positions=positions)
        np.testing.assert_array_equal(s.positions, positions)


# ---------------------------------------------------------------------------
# Step tests
# ---------------------------------------------------------------------------

class TestStep:
    def test_step_preserves_shapes(self, state, target, config):
        key = jax.random.PRNGKey(0)
        new_state, info = step(key, state, target, config)

        assert new_state.positions.shape == state.positions.shape
        assert new_state.dual_f_cross.shape == state.dual_f_cross.shape
        assert new_state.dual_g_cross.shape == state.dual_g_cross.shape
        assert new_state.dual_f_self.shape == state.dual_f_self.shape
        assert new_state.dual_g_self.shape == state.dual_g_self.shape
        assert new_state.step == 1

    def test_step_all_finite(self, state, target, config):
        key = jax.random.PRNGKey(0)
        new_state, _ = step(key, state, target, config)

        assert jnp.all(jnp.isfinite(new_state.positions))
        assert jnp.all(jnp.isfinite(new_state.dual_f_cross))
        assert jnp.all(jnp.isfinite(new_state.dual_g_cross))
        assert jnp.all(jnp.isfinite(new_state.dual_f_self))
        assert jnp.all(jnp.isfinite(new_state.dual_g_self))

    def test_step_info_keys(self, state, target, config):
        key = jax.random.PRNGKey(0)
        _, info = step(key, state, target, config)

        assert "sinkhorn_iters_cross" in info
        assert "sinkhorn_iters_self" in info
        assert "cost_scale_cross" in info
        assert "cost_scale_self" in info


# ---------------------------------------------------------------------------
# SDD-specific behavior
# ---------------------------------------------------------------------------

class TestSDDBehavior:
    def test_zero_step_size_no_movement(self, target):
        """sdd_step_size=0 → positions unchanged."""
        config = SDDConfig(
            n_particles=20,
            n_proposals=10,
            epsilon=0.1,
            self_epsilon=0.1,
            alpha=0.05,
            sdd_step_size=0.0,
        )
        key = jax.random.PRNGKey(42)
        state = init(key, target, config)

        k_step = jax.random.PRNGKey(99)
        new_state, _ = step(k_step, state, target, config)

        np.testing.assert_allclose(
            new_state.positions, state.positions, atol=1e-6,
        )

    def test_deterministic(self, state, target, config):
        """Same key → same result."""
        key = jax.random.PRNGKey(123)
        s1, _ = step(key, state, target, config)
        s2, _ = step(key, state, target, config)

        np.testing.assert_array_equal(s1.positions, s2.positions)


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------

class TestConvergence:
    def test_gaussian_convergence(self, target):
        """SDD should converge on Gaussian(d=2) in 200 iters."""
        config = SDDConfig(
            n_particles=50,
            n_proposals=15,
            n_iterations=200,
            epsilon=0.1,
            self_epsilon=0.1,
            alpha=0.05,
            sdd_step_size=0.5,
        )
        key = jax.random.PRNGKey(42)
        k_init, k_run = jax.random.split(key)
        state = init(k_init, target, config)

        for i in range(200):
            k_run, k_step = jax.random.split(k_run)
            state, _ = step(k_step, state, target, config)

        mean_err = jnp.linalg.norm(jnp.mean(state.positions, axis=0) - target.mean)
        assert mean_err < 0.5, f"Mean error {mean_err} too high"
