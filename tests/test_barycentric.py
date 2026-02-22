"""Tests for barycentric update rule."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from etd.update.barycentric import barycentric_update


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

class TestShapes:
    def test_output_shape(self):
        N, P, d = 10, 20, 3
        key = jax.random.PRNGKey(0)
        log_gamma = jnp.zeros((N, P))
        proposals = jax.random.normal(key, (P, d))
        positions = jnp.zeros((N, d))

        result, aux = barycentric_update(key, log_gamma, proposals, positions=positions)
        assert result.shape == (N, d)


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

class TestCorrectness:
    def test_step_size_one_pure_barycentric(self):
        """step_size=1.0 → pure coupling-weighted mean (no damping)."""
        N, P, d = 5, 8, 2
        key = jax.random.PRNGKey(42)
        log_gamma = jax.random.normal(key, (N, P))
        proposals = jax.random.normal(jax.random.PRNGKey(1), (P, d))
        positions = jnp.ones((N, d)) * 100.0  # should be ignored

        result, aux = barycentric_update(key, log_gamma, proposals, step_size=1.0,
                                        positions=positions)

        # Manual computation
        from jax.scipy.special import logsumexp
        log_w = log_gamma - logsumexp(log_gamma, axis=1, keepdims=True)
        expected = jnp.exp(log_w) @ proposals

        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_step_size_zero_positions_unchanged(self):
        """step_size=0.0 → positions unchanged."""
        N, P, d = 5, 8, 2
        key = jax.random.PRNGKey(0)
        log_gamma = jnp.zeros((N, P))
        proposals = jax.random.normal(key, (P, d))
        positions = jax.random.normal(jax.random.PRNGKey(1), (N, d))

        result, aux = barycentric_update(key, log_gamma, proposals, step_size=0.0,
                                        positions=positions)

        np.testing.assert_allclose(result, positions, atol=1e-7)

    def test_uniform_coupling_gives_proposal_mean(self):
        """Uniform coupling → barycentric mean = mean of all proposals."""
        N, P, d = 4, 10, 3
        key = jax.random.PRNGKey(0)
        log_gamma = jnp.zeros((N, P))  # uniform
        proposals = jax.random.normal(key, (P, d))
        positions = jnp.zeros((N, d))

        result, aux = barycentric_update(key, log_gamma, proposals, positions=positions)

        expected_mean = jnp.mean(proposals, axis=0)
        for i in range(N):
            np.testing.assert_allclose(result[i], expected_mean, atol=1e-6)

    def test_deterministic(self):
        """Barycentric is deterministic — different keys give same result."""
        N, P, d = 5, 8, 2
        log_gamma = jnp.zeros((N, P))
        proposals = jnp.ones((P, d))
        positions = jnp.zeros((N, d))

        r1, _ = barycentric_update(jax.random.PRNGKey(0), log_gamma, proposals,
                                    positions=positions)
        r2, _ = barycentric_update(jax.random.PRNGKey(999), log_gamma, proposals,
                                    positions=positions)

        np.testing.assert_array_equal(r1, r2)

    def test_intermediate_step_size(self):
        """step_size=0.5 → halfway between positions and barycentric mean."""
        N, P, d = 3, 6, 2
        key = jax.random.PRNGKey(42)
        log_gamma = jnp.zeros((N, P))  # uniform
        proposals = jnp.ones((P, d)) * 4.0
        positions = jnp.zeros((N, d))

        result, aux = barycentric_update(key, log_gamma, proposals, step_size=0.5,
                                        positions=positions)

        # Expected: 0.5 * 0 + 0.5 * 4 = 2
        np.testing.assert_allclose(result, 2.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Aux dict tests
# ---------------------------------------------------------------------------

class TestAuxDict:
    def test_aux_weights_shape(self):
        """aux['weights'] should have shape (N, P)."""
        N, P, d = 10, 20, 3
        key = jax.random.PRNGKey(0)
        log_gamma = jax.random.normal(key, (N, P))
        proposals = jax.random.normal(jax.random.PRNGKey(1), (P, d))
        positions = jnp.zeros((N, d))

        _, aux = barycentric_update(key, log_gamma, proposals, positions=positions)
        assert aux["weights"].shape == (N, P)

    def test_aux_weights_sum_to_one(self):
        """Each row of aux['weights'] should sum to 1.0."""
        N, P, d = 10, 20, 3
        key = jax.random.PRNGKey(42)
        log_gamma = jax.random.normal(key, (N, P))
        proposals = jax.random.normal(jax.random.PRNGKey(1), (P, d))
        positions = jnp.zeros((N, d))

        _, aux = barycentric_update(key, log_gamma, proposals, positions=positions)
        row_sums = jnp.sum(aux["weights"], axis=1)  # (N,)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_get_update_fn_barycentric(self):
        from etd.update import get_update_fn
        fn = get_update_fn("barycentric")
        assert fn is barycentric_update

    def test_updates_dict(self):
        from etd.update import UPDATES
        assert "barycentric" in UPDATES
        assert "categorical" in UPDATES


# ---------------------------------------------------------------------------
# Integration with ETD step
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_etd_with_barycentric_runs(self):
        """10 iterations of ETD with barycentric update, no errors."""
        from etd.step import init, step
        from etd.targets.gaussian import GaussianTarget
        from etd.types import ETDConfig

        target = GaussianTarget(dim=2)
        config = ETDConfig(
            n_particles=20,
            n_proposals=10,
            n_iterations=10,
            coupling="balanced",
            update="barycentric",
            epsilon=0.1,
            alpha=0.05,
        )

        key = jax.random.PRNGKey(42)
        k_init, k_run = jax.random.split(key)
        state = init(k_init, target, config)

        for i in range(10):
            k_run, k_step = jax.random.split(k_run)
            state, info = step(k_step, state, target, config)

        assert state.positions.shape == (20, 2)
        assert jnp.all(jnp.isfinite(state.positions))
