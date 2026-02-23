"""Tests for the MALA (Metropolis-Adjusted Langevin Algorithm) baseline.

Covers init, step, determinism, accept/reject mechanics,
preconditioning, convergence, and MALA-vs-ULA bias comparison.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from etd.baselines.mala import MALAConfig, MALAState, init, step
from etd.diagnostics.metrics import mean_error
from etd.targets.gaussian import GaussianTarget
from etd.types import PreconditionerConfig


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
    """State initialization: shapes, caches, defaults."""

    def test_shapes(self, gaussian_target, key):
        cfg = MALAConfig(n_particles=50)
        state = init(key, gaussian_target, cfg)

        assert state.positions.shape == (50, 2)
        assert state.log_prob.shape == (50,)
        assert state.scores.shape == (50, 2)
        assert state.precond_accum.shape == (2,)
        assert int(state.step) == 0

    def test_custom_positions(self, gaussian_target, key):
        custom = jnp.ones((50, 2)) * 3.14
        cfg = MALAConfig(n_particles=50)
        state = init(key, gaussian_target, cfg, init_positions=custom)
        np.testing.assert_array_equal(np.array(state.positions), np.array(custom))

    def test_cached_log_prob_correct(self, gaussian_target, key):
        """Cached log_prob should match target.log_prob(positions)."""
        cfg = MALAConfig(n_particles=20)
        state = init(key, gaussian_target, cfg)
        expected = gaussian_target.log_prob(state.positions)
        np.testing.assert_allclose(
            np.array(state.log_prob), np.array(expected), atol=1e-5
        )

    def test_cached_scores_finite(self, gaussian_target, key):
        cfg = MALAConfig(n_particles=20)
        state = init(key, gaussian_target, cfg)
        assert jnp.all(jnp.isfinite(state.scores))

    def test_precond_accum_ones(self, gaussian_target, key):
        """Preconditioner accumulator initialized to ones."""
        cfg = MALAConfig(n_particles=20)
        state = init(key, gaussian_target, cfg)
        np.testing.assert_array_equal(np.array(state.precond_accum), 1.0)


# ---------------------------------------------------------------------------
# Step tests
# ---------------------------------------------------------------------------

class TestStep:
    """Step output contract: shapes, step increment, info keys."""

    def test_shapes(self, gaussian_target, key):
        cfg = MALAConfig(n_particles=30)
        k1, k2 = jax.random.split(key)
        state = init(k1, gaussian_target, cfg)
        new_state, info = step(k2, state, gaussian_target, cfg)

        assert new_state.positions.shape == (30, 2)
        assert new_state.log_prob.shape == (30,)
        assert new_state.scores.shape == (30, 2)
        assert new_state.precond_accum.shape == (2,)

    def test_step_increment(self, gaussian_target, key):
        cfg = MALAConfig(n_particles=20)
        k1, k2 = jax.random.split(key)
        state = init(k1, gaussian_target, cfg)
        new_state, _ = step(k2, state, gaussian_target, cfg)
        assert int(new_state.step) == 1

    def test_info_keys(self, gaussian_target, key):
        cfg = MALAConfig(n_particles=20)
        k1, k2 = jax.random.split(key)
        state = init(k1, gaussian_target, cfg)
        _, info = step(k2, state, gaussian_target, cfg)
        assert "acceptance_rate" in info
        assert "score_norm" in info

    def test_acceptance_rate_bounded(self, gaussian_target, key):
        """Acceptance rate should be in [0, 1]."""
        cfg = MALAConfig(n_particles=50)
        k1, k2 = jax.random.split(key)
        state = init(k1, gaussian_target, cfg)
        _, info = step(k2, state, gaussian_target, cfg)
        ar = float(info["acceptance_rate"])
        assert 0.0 <= ar <= 1.0

    def test_all_finite(self, gaussian_target, key):
        cfg = MALAConfig(n_particles=30)
        k1, k2 = jax.random.split(key)
        state = init(k1, gaussian_target, cfg)
        new_state, _ = step(k2, state, gaussian_target, cfg)
        assert jnp.all(jnp.isfinite(new_state.positions))
        assert jnp.all(jnp.isfinite(new_state.log_prob))
        assert jnp.all(jnp.isfinite(new_state.scores))


# ---------------------------------------------------------------------------
# Determinism tests
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Same key → same output."""

    def test_same_key_same_output(self, gaussian_target, key):
        cfg = MALAConfig(n_particles=20)
        k1, k2 = jax.random.split(key)
        state = init(k1, gaussian_target, cfg)
        s1, i1 = step(k2, state, gaussian_target, cfg)
        s2, i2 = step(k2, state, gaussian_target, cfg)
        np.testing.assert_array_equal(np.array(s1.positions), np.array(s2.positions))
        np.testing.assert_array_equal(np.array(s1.log_prob), np.array(s2.log_prob))


# ---------------------------------------------------------------------------
# Accept / reject tests
# ---------------------------------------------------------------------------

class TestAcceptReject:
    """MH acceptance mechanics."""

    def test_tiny_step_high_acceptance(self, gaussian_target, key):
        """Very small h → proposals are close → nearly 100% acceptance."""
        cfg = MALAConfig(n_particles=100, stepsize=1e-5)
        k1, k_run = jax.random.split(key)
        state = init(k1, gaussian_target, cfg)

        # Run a few steps to get away from initialization effects
        total_ar = 0.0
        n_steps = 10
        for _ in range(n_steps):
            k_run, k_step = jax.random.split(k_run)
            state, info = step(k_step, state, gaussian_target, cfg)
            total_ar += float(info["acceptance_rate"])

        avg_ar = total_ar / n_steps
        assert avg_ar > 0.9, f"Expected high acceptance with tiny h, got {avg_ar:.3f}"

    def test_huge_step_low_acceptance(self, gaussian_target, key):
        """Very large h → proposals jump far → low acceptance."""
        cfg = MALAConfig(n_particles=100, stepsize=100.0)
        k1, k2 = jax.random.split(key)
        state = init(k1, gaussian_target, cfg)
        _, info = step(k2, state, gaussian_target, cfg)
        ar = float(info["acceptance_rate"])
        assert ar < 0.5, f"Expected low acceptance with huge h, got {ar:.3f}"

    def test_rejection_preserves_cache(self, gaussian_target, key):
        """Rejected particles should keep their original log_prob/scores."""
        cfg = MALAConfig(n_particles=200, stepsize=100.0)
        k1, k2 = jax.random.split(key)
        state = init(k1, gaussian_target, cfg)
        new_state, _ = step(k2, state, gaussian_target, cfg)

        # For rejected particles, log_prob should match the old value
        # (with huge h, most should be rejected)
        unchanged = jnp.isclose(
            new_state.positions, state.positions, atol=1e-10
        ).all(axis=-1)  # (N,) — True if particle didn't move

        if jnp.any(unchanged):
            # log_prob of unmoved particles should match original
            np.testing.assert_allclose(
                np.array(new_state.log_prob[unchanged]),
                np.array(state.log_prob[unchanged]),
                atol=1e-6,
            )


# ---------------------------------------------------------------------------
# Preconditioner tests
# ---------------------------------------------------------------------------

class TestPreconditioner:
    """Preconditioning with RMSProp accumulator."""

    def test_accum_updates(self, gaussian_target, key):
        """With rmsprop preconditioner, accumulator should change after a step."""
        cfg = MALAConfig(
            n_particles=50,
            preconditioner=PreconditionerConfig(type="rmsprop", proposals=True),
        )
        k1, k2 = jax.random.split(key)
        state = init(k1, gaussian_target, cfg)
        new_state, _ = step(k2, state, gaussian_target, cfg)

        # Accumulator should differ from initial ones
        assert not jnp.allclose(new_state.precond_accum, state.precond_accum)

    def test_accum_unchanged_without_precond(self, gaussian_target, key):
        """Without preconditioner, accumulator should remain ones."""
        cfg = MALAConfig(n_particles=50)
        k1, k2 = jax.random.split(key)
        state = init(k1, gaussian_target, cfg)
        new_state, _ = step(k2, state, gaussian_target, cfg)
        np.testing.assert_array_equal(
            np.array(new_state.precond_accum), np.array(state.precond_accum)
        )

    def test_anisotropic_target_benefit(self):
        """Preconditioned MALA should handle anisotropic targets better."""
        # condition_number=1000 → variances span [1, 1000]
        target = GaussianTarget(dim=2, condition_number=1000.0)

        key = jax.random.PRNGKey(123)
        n_steps = 300

        # Without preconditioning
        cfg_no = MALAConfig(n_particles=100, stepsize=0.0001)
        k1, k_run = jax.random.split(key)
        state_no = init(k1, target, cfg_no)
        for _ in range(n_steps):
            k_run, k_step = jax.random.split(k_run)
            state_no, _ = step(k_step, state_no, target, cfg_no)
        err_no = float(mean_error(state_no.positions, target.mean))

        # With preconditioning
        cfg_yes = MALAConfig(
            n_particles=100, stepsize=0.0001,
            preconditioner=PreconditionerConfig(type="rmsprop", proposals=True),
        )
        k1, k_run = jax.random.split(key)
        state_yes = init(k1, target, cfg_yes)
        for _ in range(n_steps):
            k_run, k_step = jax.random.split(k_run)
            state_yes, _ = step(k_step, state_yes, target, cfg_yes)
        err_yes = float(mean_error(state_yes.positions, target.mean))

        # Both should be reasonable; preconditioned should be no worse
        assert err_yes < err_no + 0.5, (
            f"Preconditioned err={err_yes:.3f} much worse than plain err={err_no:.3f}"
        )


# ---------------------------------------------------------------------------
# Convergence test
# ---------------------------------------------------------------------------

class TestConvergence:
    """MALA on isotropic Gaussian(d=2) should converge in 500 steps."""

    def test_mean_error(self, gaussian_target):
        cfg = MALAConfig(n_particles=100, stepsize=0.01, n_iterations=500)
        key = jax.random.PRNGKey(0)
        k_init, k_run = jax.random.split(key)
        state = init(k_init, gaussian_target, cfg)

        for _ in range(500):
            k_run, k_step = jax.random.split(k_run)
            state, _ = step(k_step, state, gaussian_target, cfg)

        err = mean_error(state.positions, gaussian_target.mean)
        assert float(err) < 0.3, f"MALA mean_error = {float(err):.4f} >= 0.3"


# ---------------------------------------------------------------------------
# MALA vs ULA: bias comparison
# ---------------------------------------------------------------------------

class TestMALAvsULA:
    """MALA should have lower bias than ULA at the same step size."""

    def test_variance_closer_to_truth(self):
        """With moderate h, MALA variance should be closer to truth than ULA."""
        from etd.baselines.ula import ULAConfig, init as ula_init, step as ula_step

        target = GaussianTarget(dim=2)
        h = 0.05  # moderate step size — ULA will show O(h) bias
        n_steps = 1000
        N = 200

        key = jax.random.PRNGKey(7)

        # --- ULA ---
        ula_cfg = ULAConfig(n_particles=N, stepsize=h)
        k1, k_run = jax.random.split(key)
        ula_state = ula_init(k1, target, ula_cfg)
        for _ in range(n_steps):
            k_run, k_step = jax.random.split(k_run)
            ula_state, _ = ula_step(k_step, ula_state, target, ula_cfg)
        ula_var = jnp.var(ula_state.positions, axis=0)  # (d,)

        # --- MALA ---
        mala_cfg = MALAConfig(n_particles=N, stepsize=h)
        k1, k_run = jax.random.split(key)
        mala_state = init(k1, target, mala_cfg)
        for _ in range(n_steps):
            k_run, k_step = jax.random.split(k_run)
            mala_state, _ = step(k_step, mala_state, target, mala_cfg)
        mala_var = jnp.var(mala_state.positions, axis=0)  # (d,)

        # True variance is 1.0 for isotropic Gaussian
        true_var = 1.0
        ula_err = float(jnp.mean(jnp.abs(ula_var - true_var)))
        mala_err = float(jnp.mean(jnp.abs(mala_var - true_var)))

        # MALA should be closer to truth (or at worst similar)
        assert mala_err < ula_err + 0.1, (
            f"MALA var error ({mala_err:.3f}) not better than "
            f"ULA var error ({ula_err:.3f})"
        )


# ---------------------------------------------------------------------------
# Cholesky-preconditioned MALA
# ---------------------------------------------------------------------------

class TestCholesky:
    """Tests for the Cholesky-preconditioned MALA path."""

    def test_cholesky_init(self):
        """Cholesky preconditioner → cholesky_factor is not identity."""
        target = GaussianTarget(dim=4, condition_number=10.0)
        key = jax.random.PRNGKey(50)
        cfg = MALAConfig(
            n_particles=50,
            preconditioner=PreconditionerConfig(type="cholesky"),
        )
        state = init(key, target, cfg)

        assert state.cholesky_factor.shape == (4, 4)
        # Should differ from identity due to ensemble covariance
        assert not jnp.allclose(state.cholesky_factor, jnp.eye(4), atol=0.01)

    def test_cholesky_step_shapes(self):
        """After 1 step, all state fields have correct shapes."""
        target = GaussianTarget(dim=4)
        key = jax.random.PRNGKey(51)
        cfg = MALAConfig(
            n_particles=30,
            preconditioner=PreconditionerConfig(type="cholesky"),
        )
        k1, k2 = jax.random.split(key)
        state = init(k1, target, cfg)
        new_state, info = step(k2, state, target, cfg)

        assert new_state.positions.shape == (30, 4)
        assert new_state.log_prob.shape == (30,)
        assert new_state.scores.shape == (30, 4)
        assert new_state.cholesky_factor.shape == (4, 4)
        assert "acceptance_rate" in info
        assert "score_norm" in info

    def test_cholesky_convergence(self):
        """Cholesky MALA on anisotropic Gaussian(dim=4) converges."""
        target = GaussianTarget(dim=4, condition_number=50.0)
        key = jax.random.PRNGKey(52)
        cfg = MALAConfig(
            n_particles=100, n_iterations=500,
            stepsize=0.05,
            preconditioner=PreconditionerConfig(type="cholesky"),
        )
        k_init, k_run = jax.random.split(key)
        state = init(k_init, target, cfg)

        for _ in range(500):
            k_run, k_step = jax.random.split(k_run)
            state, _ = step(k_step, state, target, cfg)

        err = float(jnp.mean(jnp.abs(jnp.mean(state.positions, axis=0))))
        assert err < 0.5, f"Cholesky MALA mean_error = {err:.4f} >= 0.5"

    def test_cholesky_vs_diagonal(self):
        """Cholesky MALA >= diagonal MALA on anisotropic target."""
        target = GaussianTarget(dim=4, condition_number=100.0)
        key = jax.random.PRNGKey(53)
        n_steps = 300
        N = 100

        # Diagonal MALA
        cfg_diag = MALAConfig(
            n_particles=N, stepsize=0.001,
            preconditioner=PreconditionerConfig(type="rmsprop", proposals=True),
        )
        k_init, k_run = jax.random.split(key)
        state_diag = init(k_init, target, cfg_diag)
        for _ in range(n_steps):
            k_run, k_step = jax.random.split(k_run)
            state_diag, _ = step(k_step, state_diag, target, cfg_diag)
        err_diag = float(jnp.mean(jnp.abs(jnp.mean(state_diag.positions, axis=0))))

        # Cholesky MALA
        cfg_chol = MALAConfig(
            n_particles=N, stepsize=0.001,
            preconditioner=PreconditionerConfig(type="cholesky"),
        )
        k_init, k_run = jax.random.split(key)
        state_chol = init(k_init, target, cfg_chol)
        for _ in range(n_steps):
            k_run, k_step = jax.random.split(k_run)
            state_chol, _ = step(k_step, state_chol, target, cfg_chol)
        err_chol = float(jnp.mean(jnp.abs(jnp.mean(state_chol.positions, axis=0))))

        # Cholesky should be no worse (allowing some tolerance)
        assert err_chol < err_diag + 0.5, (
            f"Cholesky err ({err_chol:.3f}) much worse than "
            f"diagonal err ({err_diag:.3f})"
        )
