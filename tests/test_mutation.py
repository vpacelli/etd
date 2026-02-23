"""Tests for MCMC mutation kernels (mala_kernel, rwm_kernel, mutate).

Covers shapes, acceptance behavior, score-free guarantees, cache
preservation, and Cholesky vs identity comparison.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import make_test_config, make_test_sdd_config

from etd.primitives.mutation import mala_kernel, mutate, rwm_kernel
from etd.extensions.sdd import init as sdd_init, step as sdd_step
from etd.step import init as etd_init, step as etd_step
from etd.types import MutationConfig
from etd.targets.gaussian import GaussianTarget


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gaussian_target():
    return GaussianTarget(dim=4)


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

class TestMALAKernelShapes:
    """Output shapes of mala_kernel match (N,d), (N,), (N,d), (N,)."""

    def test_shapes_isotropic(self, gaussian_target, key):
        N, d = 20, 4
        x = jax.random.normal(key, (N, d))
        log_pi = gaussian_target.log_prob(x)
        scores = gaussian_target.score(x)

        new_x, new_lp, new_s, accepted = mala_kernel(
            key, x, log_pi, scores, gaussian_target,
            h=0.01, L=None, score_clip=5.0,
        )

        assert new_x.shape == (N, d)
        assert new_lp.shape == (N,)
        assert new_s.shape == (N, d)
        assert accepted.shape == (N,)

    def test_shapes_cholesky(self, gaussian_target, key):
        N, d = 20, 4
        x = jax.random.normal(key, (N, d))
        log_pi = gaussian_target.log_prob(x)
        scores = gaussian_target.score(x)
        L = jnp.eye(d) * 0.5

        new_x, new_lp, new_s, accepted = mala_kernel(
            key, x, log_pi, scores, gaussian_target,
            h=0.01, L=L, score_clip=5.0,
        )

        assert new_x.shape == (N, d)
        assert new_lp.shape == (N,)
        assert new_s.shape == (N, d)
        assert accepted.shape == (N,)


class TestRWMKernelShapes:
    """Output shapes of rwm_kernel match (N,d), (N,), (N,)."""

    def test_shapes_isotropic(self, gaussian_target, key):
        N, d = 20, 4
        x = jax.random.normal(key, (N, d))
        log_pi = gaussian_target.log_prob(x)

        new_x, new_lp, accepted = rwm_kernel(
            key, x, log_pi, gaussian_target, h=0.01, L=None,
        )

        assert new_x.shape == (N, d)
        assert new_lp.shape == (N,)
        assert accepted.shape == (N,)

    def test_shapes_cholesky(self, gaussian_target, key):
        N, d = 20, 4
        x = jax.random.normal(key, (N, d))
        log_pi = gaussian_target.log_prob(x)
        L = jnp.eye(d)

        new_x, new_lp, accepted = rwm_kernel(
            key, x, log_pi, gaussian_target, h=0.01, L=L,
        )

        assert new_x.shape == (N, d)
        assert new_lp.shape == (N,)
        assert accepted.shape == (N,)


# ---------------------------------------------------------------------------
# Acceptance behavior
# ---------------------------------------------------------------------------

class TestMALAAcceptance:
    """MALA acceptance rate depends on step size h."""

    def test_tiny_h_high_acceptance(self, gaussian_target, key):
        """h=1e-6 → nearly all proposals accepted."""
        N, d = 100, 4
        x = gaussian_target.sample(key, N)
        log_pi = gaussian_target.log_prob(x)
        scores = gaussian_target.score(x)

        _, _, _, accepted = mala_kernel(
            key, x, log_pi, scores, gaussian_target,
            h=1e-6, L=None, score_clip=5.0,
        )
        assert float(jnp.mean(accepted)) > 0.95

    def test_huge_h_low_acceptance(self, gaussian_target, key):
        """h=100 → most proposals rejected."""
        N, d = 100, 4
        x = gaussian_target.sample(key, N)
        log_pi = gaussian_target.log_prob(x)
        scores = gaussian_target.score(x)

        _, _, _, accepted = mala_kernel(
            key, x, log_pi, scores, gaussian_target,
            h=100.0, L=None, score_clip=5.0,
        )
        assert float(jnp.mean(accepted)) < 0.30


# ---------------------------------------------------------------------------
# Score-free guarantee
# ---------------------------------------------------------------------------

class TestRWMScoreFree:
    """RWM kernel must never call target.score."""

    def test_no_score_call(self, key):
        """Mock target with broken score → RWM still works."""

        class ScoreFreeTarget:
            dim = 4

            def log_prob(self, x):
                return -0.5 * jnp.sum(x ** 2, axis=-1)

            def score(self, x):
                raise RuntimeError("score should not be called for RWM")

        target = ScoreFreeTarget()
        N, d = 20, 4
        x = jax.random.normal(key, (N, d))
        log_pi = target.log_prob(x)

        # Should not raise
        new_x, new_lp, accepted = rwm_kernel(
            key, x, log_pi, target, h=0.01, L=None,
        )
        assert new_x.shape == (N, d)


# ---------------------------------------------------------------------------
# Cache preservation on rejection
# ---------------------------------------------------------------------------

class TestCachePreservation:
    """Rejected particles keep original log_prob and scores."""

    def test_rejection_preserves_cache(self, gaussian_target, key):
        N, d = 200, 4
        x = gaussian_target.sample(key, N)
        log_pi = gaussian_target.log_prob(x)
        scores = gaussian_target.score(x)

        # Large h → many rejections
        new_x, new_lp, new_s, accepted = mala_kernel(
            key, x, log_pi, scores, gaussian_target,
            h=100.0, L=None, score_clip=5.0,
        )

        # Rejected particles: positions unchanged → log_prob unchanged
        rejected = accepted < 0.5  # (N,) bool
        if jnp.any(rejected):
            np.testing.assert_allclose(
                np.array(new_lp[rejected]),
                np.array(log_pi[rejected]),
                atol=1e-6,
            )
            np.testing.assert_allclose(
                np.array(new_s[rejected]),
                np.array(scores[rejected]),
                atol=1e-6,
            )


# ---------------------------------------------------------------------------
# mutate() integration
# ---------------------------------------------------------------------------

class TestMutate:
    """mutate() dispatcher: shapes, info, and correct kernel routing."""

    def test_mala_5_steps(self, gaussian_target, key):
        N, d = 50, 4
        x = gaussian_target.sample(key, N)
        cfg = MutationConfig(kernel="mala", steps=5, stepsize=0.01)

        new_pos, new_lp, new_s, info = mutate(
            key, x, gaussian_target, cfg, score_clip=5.0,
        )

        assert new_pos.shape == (N, d)
        assert new_lp.shape == (N,)
        assert new_s.shape == (N, d)
        assert "acceptance_rate" in info
        ar = float(info["acceptance_rate"])
        assert 0.0 <= ar <= 1.0

    def test_rwm_5_steps(self, gaussian_target, key):
        N, d = 50, 4
        x = gaussian_target.sample(key, N)
        cfg = MutationConfig(kernel="rwm", steps=5, stepsize=0.01)

        new_pos, new_lp, new_s, info = mutate(
            key, x, gaussian_target, cfg, score_clip=5.0,
        )

        assert new_pos.shape == (N, d)
        assert new_lp.shape == (N,)
        assert new_s.shape == (N, d)
        # RWM returns zero scores
        np.testing.assert_array_equal(np.array(new_s), 0.0)
        ar = float(info["acceptance_rate"])
        assert 0.0 <= ar <= 1.0

    def test_all_finite(self, gaussian_target, key):
        N, d = 50, 4
        x = gaussian_target.sample(key, N)
        cfg = MutationConfig(kernel="mala", steps=5, stepsize=0.01)

        new_pos, new_lp, new_s, info = mutate(
            key, x, gaussian_target, cfg, score_clip=5.0,
        )

        assert jnp.all(jnp.isfinite(new_pos))
        assert jnp.all(jnp.isfinite(new_lp))
        assert jnp.all(jnp.isfinite(new_s))


# ---------------------------------------------------------------------------
# Cholesky vs identity comparison
# ---------------------------------------------------------------------------

class TestCholeskyBenefit:
    """Cholesky MALA should handle anisotropic targets better."""

    def test_cholesky_vs_identity_acceptance(self):
        """On anisotropic target, Cholesky MALA has better acceptance."""
        target = GaussianTarget(dim=4, condition_number=100.0)
        key = jax.random.PRNGKey(123)

        N = 100
        # Start from samples near the target
        k_init, k_iso, k_chol = jax.random.split(key, 3)
        x = target.sample(k_init, N)

        h = 0.5  # moderate step size where preconditioning matters

        # Build true Cholesky from target covariance (diagonal)
        L_true = jnp.diag(jnp.sqrt(target.variance))

        # --- Identity MALA: many steps ---
        cfg = MutationConfig(kernel="mala", steps=20, stepsize=h)
        _, _, _, info_iso = mutate(
            k_iso, x, target, cfg,
            cholesky_factor=None, score_clip=5.0,
        )

        # --- Cholesky MALA: same step count ---
        _, _, _, info_chol = mutate(
            k_chol, x, target, cfg,
            cholesky_factor=L_true, score_clip=5.0,
        )

        ar_iso = float(info_iso["acceptance_rate"])
        ar_chol = float(info_chol["acceptance_rate"])

        # Cholesky should have better (or at least comparable) acceptance
        assert ar_chol >= ar_iso - 0.1, (
            f"Cholesky acceptance ({ar_chol:.3f}) much worse than "
            f"identity ({ar_iso:.3f}) on anisotropic target"
        )


# ---------------------------------------------------------------------------
# ETD step integration
# ---------------------------------------------------------------------------

class TestETDMutation:
    """ETD step with mutation enabled: shapes, info, and noop."""

    def test_etd_with_mala_mutation(self):
        target = GaussianTarget(dim=2)
        key = jax.random.PRNGKey(0)
        mut = MutationConfig(kernel="mala", steps=3, stepsize=0.01)
        cfg = make_test_config(
            n_particles=30, n_iterations=10, n_proposals=10,
            mutation=mut,
        )
        k_init, k_run = jax.random.split(key)
        state = etd_init(k_init, target, cfg)

        for i in range(10):
            k_run, k_step = jax.random.split(k_run)
            state, info = etd_step(k_step, state, target, cfg)

        assert state.positions.shape == (30, 2)
        assert state.log_prob.shape == (30,)
        assert state.scores.shape == (30, 2)
        assert jnp.all(jnp.isfinite(state.positions))
        assert "mutation_acceptance_rate" in info
        ar = float(info["mutation_acceptance_rate"])
        assert 0.0 <= ar <= 1.0

    def test_etd_with_rwm_mutation(self):
        target = GaussianTarget(dim=2)
        key = jax.random.PRNGKey(1)
        mut = MutationConfig(kernel="rwm", steps=3, stepsize=0.01)
        cfg = make_test_config(
            n_particles=30, n_iterations=10, n_proposals=10,
            mutation=mut,
        )
        k_init, k_run = jax.random.split(key)
        state = etd_init(k_init, target, cfg)

        for i in range(10):
            k_run, k_step = jax.random.split(k_run)
            state, info = etd_step(k_step, state, target, cfg)

        assert state.positions.shape == (30, 2)
        assert jnp.all(jnp.isfinite(state.positions))
        assert "mutation_acceptance_rate" in info

    def test_etd_mutation_off_noop(self):
        """Default MutationConfig → log_prob/scores are zeros."""
        target = GaussianTarget(dim=2)
        key = jax.random.PRNGKey(2)
        cfg = make_test_config(
            n_particles=30, n_iterations=10, n_proposals=10,
        )
        k_init, k_run = jax.random.split(key)
        state = etd_init(k_init, target, cfg)
        k_run, k_step = jax.random.split(k_run)
        state, info = etd_step(k_step, state, target, cfg)

        # When mutation is off, log_prob and scores should be zeros
        np.testing.assert_array_equal(np.array(state.log_prob), 0.0)
        np.testing.assert_array_equal(np.array(state.scores), 0.0)
        assert "mutation_acceptance_rate" not in info


# ---------------------------------------------------------------------------
# SDD step integration
# ---------------------------------------------------------------------------

class TestSDDMutation:
    """SDD step with mutation enabled."""

    def test_sdd_with_mala_mutation(self):
        target = GaussianTarget(dim=2)
        key = jax.random.PRNGKey(10)
        mut = MutationConfig(kernel="mala", steps=3, stepsize=0.01)
        cfg = make_test_sdd_config(
            n_particles=30, n_iterations=10, n_proposals=10,
            mutation=mut,
        )
        k_init, k_run = jax.random.split(key)
        state = sdd_init(k_init, target, cfg)

        for i in range(10):
            k_run, k_step = jax.random.split(k_run)
            state, info = sdd_step(k_step, state, target, cfg)

        assert state.positions.shape == (30, 2)
        assert state.log_prob.shape == (30,)
        assert state.scores.shape == (30, 2)
        assert jnp.all(jnp.isfinite(state.positions))
        assert "mutation_acceptance_rate" in info

    def test_sdd_mutation_off_noop(self):
        """Default MutationConfig → log_prob/scores are zeros."""
        target = GaussianTarget(dim=2)
        key = jax.random.PRNGKey(11)
        cfg = make_test_sdd_config(
            n_particles=30, n_iterations=10, n_proposals=10,
        )
        k_init, k_run = jax.random.split(key)
        state = sdd_init(k_init, target, cfg)
        k_run, k_step = jax.random.split(k_run)
        state, info = sdd_step(k_step, state, target, cfg)

        np.testing.assert_array_equal(np.array(state.log_prob), 0.0)
        np.testing.assert_array_equal(np.array(state.scores), 0.0)
        assert "mutation_acceptance_rate" not in info
