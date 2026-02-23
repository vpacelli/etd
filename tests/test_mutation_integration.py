"""Integration tests for MCMC mutation with ETD and SDD.

End-to-end convergence tests on meaningful targets proving mutation
works correctly through transport + mutation pipeline.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import make_test_config, make_test_sdd_config

from etd.extensions.sdd import init as sdd_init, step as sdd_step
from etd.step import init as etd_init, step as etd_step
from etd.types import MutationConfig, PreconditionerConfig
from etd.targets.gaussian import GaussianTarget


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_etd(key, target, config, n_iters):
    """Run ETD for n_iters and return final state."""
    k_init, k_run = jax.random.split(key)
    state = etd_init(k_init, target, config)
    for i in range(n_iters):
        k_run, k_step = jax.random.split(k_run)
        state, info = etd_step(k_step, state, target, config)
    return state, info


# ---------------------------------------------------------------------------
# No-degradation test
# ---------------------------------------------------------------------------

class TestNoDegradation:
    """ETD+MALA should not be worse than ETD alone on a Gaussian."""

    def test_gaussian_no_degradation(self):
        target = GaussianTarget(dim=2)
        key = jax.random.PRNGKey(100)

        # ETD without mutation
        cfg_plain = make_test_config(
            n_particles=100, n_iterations=200, n_proposals=25,
            coupling="balanced", epsilon=0.1, alpha=0.05,
        )
        state_plain, _ = _run_etd(key, target, cfg_plain, 200)
        err_plain = float(jnp.mean(jnp.abs(
            jnp.mean(state_plain.positions, axis=0)
        )))

        # ETD with MALA mutation
        mut = MutationConfig(kernel="mala", steps=3, stepsize=0.01)
        cfg_mut = make_test_config(
            n_particles=100, n_iterations=200, n_proposals=25,
            coupling="balanced", epsilon=0.1, alpha=0.05,
            mutation=mut,
        )
        state_mut, _ = _run_etd(key, target, cfg_mut, 200)
        err_mut = float(jnp.mean(jnp.abs(
            jnp.mean(state_mut.positions, axis=0)
        )))

        # Mutation should not degrade convergence
        assert err_mut < err_plain + 0.5, (
            f"Mutation err ({err_mut:.3f}) much worse than "
            f"plain ({err_plain:.3f})"
        )


# ---------------------------------------------------------------------------
# Anisotropic variance improvement
# ---------------------------------------------------------------------------

class TestAnisotropicImprovement:
    """ETD+MALA+Cholesky should improve variance ratio on anisotropic target."""

    def test_anisotropic_variance_improvement(self):
        target = GaussianTarget(dim=4, condition_number=10.0)
        key = jax.random.PRNGKey(200)

        pc = PreconditionerConfig(
            type="cholesky", proposals=True, cost=True, shrinkage=0.1,
        )
        mut = MutationConfig(kernel="mala", steps=5, stepsize=0.01)
        cfg = make_test_config(
            n_particles=100, n_iterations=300, n_proposals=25,
            coupling="balanced", epsilon=0.1, alpha=0.05,
            preconditioner=pc, mutation=mut,
        )

        state, info = _run_etd(key, target, cfg, 300)

        # Variance ratio: particle variance / true variance
        var_ratio = jnp.var(state.positions, axis=0) / target.variance
        mean_ratio = float(jnp.mean(var_ratio))

        # Should be reasonably close to 1.0 (not collapsed or exploded)
        assert 0.05 < mean_ratio < 5.0, (
            f"Variance ratio mean = {mean_ratio:.3f} out of [0.05, 5.0]"
        )


# ---------------------------------------------------------------------------
# Score-free RWM convergence
# ---------------------------------------------------------------------------

class TestScoreFreeRWM:
    """Score-free ETD + RWM should still converge."""

    def test_score_free_rwm(self):
        target = GaussianTarget(dim=2)
        key = jax.random.PRNGKey(300)

        mut = MutationConfig(
            kernel="rwm", steps=5, stepsize=0.1, cholesky=False,
        )
        cfg = make_test_config(
            n_particles=100, n_iterations=300, n_proposals=25,
            coupling="balanced", epsilon=0.1, alpha=0.05,
            use_score=False, mutation=mut,
        )

        state, _ = _run_etd(key, target, cfg, 300)
        err = float(jnp.mean(jnp.abs(jnp.mean(state.positions, axis=0))))
        assert err < 1.0, f"Score-free ETD+RWM mean_error = {err:.3f} >= 1.0"


# ---------------------------------------------------------------------------
# SDD + MALA
# ---------------------------------------------------------------------------

class TestSDDMutation:
    """SDD+MALA on Gaussian should converge."""

    def test_sdd_mala(self):
        target = GaussianTarget(dim=2)
        key = jax.random.PRNGKey(400)

        mut = MutationConfig(kernel="mala", steps=3, stepsize=0.01)
        cfg = make_test_sdd_config(
            n_particles=50, n_iterations=200, n_proposals=15,
            epsilon=0.1, alpha=0.05, mutation=mut,
        )

        k_init, k_run = jax.random.split(key)
        state = sdd_init(k_init, target, cfg)
        for i in range(200):
            k_run, k_step = jax.random.split(k_run)
            state, info = sdd_step(k_step, state, target, cfg)

        err = float(jnp.mean(jnp.abs(jnp.mean(state.positions, axis=0))))
        assert err < 1.5, f"SDD+MALA mean_error = {err:.3f} >= 1.5"
        assert jnp.all(jnp.isfinite(state.positions))


# ---------------------------------------------------------------------------
# Acceptance rate sanity
# ---------------------------------------------------------------------------

class TestAcceptanceRate:
    """Acceptance rate should be in a reasonable range."""

    def test_acceptance_rate_reasonable(self):
        target = GaussianTarget(dim=2)
        key = jax.random.PRNGKey(500)

        mut = MutationConfig(kernel="mala", steps=5, stepsize=0.01)
        cfg = make_test_config(
            n_particles=100, n_iterations=50, n_proposals=25,
            coupling="balanced", epsilon=0.1, alpha=0.05,
            mutation=mut,
        )

        state, info = _run_etd(key, target, cfg, 50)
        ar = float(info["mutation_acceptance_rate"])
        assert 0.1 <= ar <= 1.0, (
            f"Acceptance rate {ar:.3f} outside [0.1, 1.0]"
        )


# ---------------------------------------------------------------------------
# JIT + lax.scan compatibility
# ---------------------------------------------------------------------------

class TestJITCompatibility:
    """Verify mutation works inside lax.scan (nested scan)."""

    def test_jit_lax_scan_nested(self):
        """Run via lax.scan for 10 steps with mutation active."""
        target = GaussianTarget(dim=2)
        key = jax.random.PRNGKey(600)

        mut = MutationConfig(
            kernel="mala", steps=3, stepsize=0.01, cholesky=False,
        )
        cfg = make_test_config(
            n_particles=30, n_iterations=10, n_proposals=10,
            coupling="balanced", epsilon=0.1, alpha=0.05,
            mutation=mut,
        )

        k_init, k_run = jax.random.split(key)
        state = etd_init(k_init, target, cfg)

        def scan_body(carry, t):
            key_step = jax.random.fold_in(k_run, t)
            new_state, info = etd_step(key_step, carry, target, cfg)
            return new_state, info

        final_state, stacked_info = jax.lax.scan(
            scan_body, state, jnp.arange(10),
        )

        assert final_state.positions.shape == (30, 2)
        assert jnp.all(jnp.isfinite(final_state.positions))
        assert stacked_info["mutation_acceptance_rate"].shape == (10,)

    def test_pytree_shape_stability(self):
        """init → scan 5 → scan 5 more: no recompilation error."""
        target = GaussianTarget(dim=2)
        key = jax.random.PRNGKey(700)

        mut = MutationConfig(
            kernel="mala", steps=3, stepsize=0.01, cholesky=False,
        )
        cfg = make_test_config(
            n_particles=20, n_iterations=10, n_proposals=10,
            coupling="balanced", epsilon=0.1, alpha=0.05,
            mutation=mut,
        )

        k_init, k_run = jax.random.split(key)
        state = etd_init(k_init, target, cfg)

        def scan_body(carry, t):
            key_step = jax.random.fold_in(k_run, t)
            return etd_step(key_step, carry, target, cfg)

        # First scan
        state1, _ = jax.lax.scan(scan_body, state, jnp.arange(5))

        # Second scan (same shapes, no recompilation needed)
        state2, _ = jax.lax.scan(scan_body, state1, jnp.arange(5, 10))

        assert state2.positions.shape == (20, 2)
        assert jnp.all(jnp.isfinite(state2.positions))
        assert int(state2.step) == 10
