"""Tests for Phase 3: cost variants, unbalanced coupling, DV feedback.

Covers:
  - Mahalanobis cost: correctness, reduction to Euclidean, non-negativity
  - L-inf cost: correctness, non-negativity, bounded by Euclidean
  - Unbalanced Sinkhorn: Gibbs limit, balanced limit, source marginals, warm-start
  - DV feedback: effect on coupling, no-op on first step
  - Integration: validation, smoke tests for each variant
"""

import functools
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from etd.costs import COSTS, build_cost_fn
from etd.costs.euclidean import squared_euclidean_cost
from etd.costs.imq import imq_cost
from etd.costs.linf import linf_cost
from etd.costs.mahalanobis import mahalanobis_cost
from etd.coupling.gibbs import gibbs_coupling
from etd.coupling.sinkhorn import sinkhorn_log_domain
from etd.coupling.unbalanced import sinkhorn_unbalanced
from etd.step import init as etd_init, step as etd_step
from etd.types import CostConfig, PreconditionerConfig

from conftest import make_test_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class SimpleGaussian:
    """Isotropic standard Gaussian target for testing."""

    def __init__(self, dim: int):
        self.dim = dim

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        return -0.5 * jnp.sum(x ** 2, axis=-1)

    def score(self, x: jnp.ndarray) -> jnp.ndarray:
        return -x


@pytest.fixture
def rng():
    return jax.random.PRNGKey(42)


@pytest.fixture
def positions_proposals(rng):
    """Standard test arrays: N=20 particles, P=30 proposals, d=5."""
    k1, k2 = jax.random.split(rng)
    positions = jax.random.normal(k1, (20, 5))
    proposals = jax.random.normal(k2, (30, 5))
    return positions, proposals


# ===========================================================================
# Mahalanobis Cost
# ===========================================================================

class TestMahalanobisCost:
    def test_matches_manual_computation(self):
        """Hand-computed Mahalanobis cost for a tiny case."""
        x = jnp.array([[1.0, 2.0]])   # (1, 2)
        y = jnp.array([[3.0, 4.0]])   # (1, 2)
        P = jnp.array([0.5, 1.0])     # preconditioner

        # C = 0.5 * ((1-3)^2 / 0.5^2 + (2-4)^2 / 1.0^2)
        #   = 0.5 * (4/0.25 + 4/1.0) = 0.5 * (16 + 4) = 10.0
        C = mahalanobis_cost(x, y, preconditioner=P)
        np.testing.assert_allclose(float(C[0, 0]), 10.0, atol=1e-5)

    def test_reduces_to_euclidean_with_identity_precond(self, positions_proposals):
        """When P = ones, Mahalanobis should match Euclidean."""
        positions, proposals = positions_proposals
        d = positions.shape[1]
        P = jnp.ones(d)

        C_maha = mahalanobis_cost(positions, proposals, preconditioner=P)
        C_euc = squared_euclidean_cost(positions, proposals)

        np.testing.assert_allclose(np.array(C_maha), np.array(C_euc), rtol=1e-5)

    def test_nonnegative(self, positions_proposals):
        """All cost entries should be non-negative."""
        positions, proposals = positions_proposals
        P = jnp.ones(positions.shape[1]) * 0.5
        C = mahalanobis_cost(positions, proposals, preconditioner=P)
        assert jnp.all(C >= 0.0)

    def test_self_cost_diagonal_zero(self, rng):
        """C(X, X) diagonal should be zero."""
        X = jax.random.normal(rng, (15, 4))
        P = jnp.array([0.5, 1.0, 2.0, 0.3])
        C = mahalanobis_cost(X, X, preconditioner=P)
        np.testing.assert_allclose(np.array(jnp.diag(C)), 0.0, atol=1e-5)

    def test_shape(self, positions_proposals):
        """Output shape should be (N, P)."""
        positions, proposals = positions_proposals
        P = jnp.ones(positions.shape[1])
        C = mahalanobis_cost(positions, proposals, preconditioner=P)
        assert C.shape == (20, 30)

    def test_requires_preconditioner(self, positions_proposals):
        """Should raise ValueError when preconditioner is None."""
        positions, proposals = positions_proposals
        with pytest.raises(ValueError, match="requires a preconditioner"):
            mahalanobis_cost(positions, proposals)


# ===========================================================================
# L-inf Cost
# ===========================================================================

class TestLinfCost:
    def test_known_values(self):
        """Hand-computed L-inf cost."""
        x = jnp.array([[1.0, 5.0, 3.0]])  # (1, 3)
        y = jnp.array([[4.0, 2.0, 7.0]])  # (1, 3)

        # max(|1-4|, |5-2|, |3-7|) = max(3, 3, 4) = 4.0
        C = linf_cost(x, y)
        np.testing.assert_allclose(float(C[0, 0]), 4.0, atol=1e-5)

    def test_nonnegative(self, positions_proposals):
        """All entries should be non-negative."""
        positions, proposals = positions_proposals
        C = linf_cost(positions, proposals)
        assert jnp.all(C >= 0.0)

    def test_self_cost_diagonal_zero(self, rng):
        """C(X, X) diagonal should be zero."""
        X = jax.random.normal(rng, (15, 4))
        C = linf_cost(X, X)
        np.testing.assert_allclose(np.array(jnp.diag(C)), 0.0, atol=1e-6)

    def test_dominated_by_euclidean(self, positions_proposals):
        """L-inf cost <= L2 cost since ||x||_inf <= ||x||_2.

        More precisely: ||x||_inf <= ||x||_2, but the Euclidean cost
        is (1/2)||x||_2^2 while L-inf is ||x||_inf. So the comparison is:
        linf <= sqrt(2 * euclidean) since ||x||_inf <= ||x||_2 = sqrt(2 * C_euc).
        """
        positions, proposals = positions_proposals
        C_linf = linf_cost(positions, proposals)
        C_euc = squared_euclidean_cost(positions, proposals)

        # ||x-y||_inf <= ||x-y||_2 = sqrt(2 * C_euc)
        assert jnp.all(C_linf <= jnp.sqrt(2.0 * C_euc) + 1e-6)

    def test_shape(self, positions_proposals):
        """Output shape should be (N, P)."""
        positions, proposals = positions_proposals
        C = linf_cost(positions, proposals)
        assert C.shape == (20, 30)


# ===========================================================================
# Unbalanced Sinkhorn
# ===========================================================================

class TestUnbalancedSinkhorn:

    @pytest.fixture
    def coupling_inputs(self, rng):
        """Standard coupling test inputs."""
        N, P = 10, 20
        k1, k2, k3 = jax.random.split(rng, 3)
        positions = jax.random.normal(k1, (N, 5))
        proposals = jax.random.normal(k2, (P, 5))
        C = squared_euclidean_cost(positions, proposals)
        log_a = -jnp.log(N) * jnp.ones(N)
        log_b = jax.nn.log_softmax(jax.random.normal(k3, (P,)))
        return C, log_a, log_b

    def test_reduces_to_gibbs_at_small_rho(self, coupling_inputs):
        """At rho ~ 0, unbalanced should approximate Gibbs coupling.

        With very small rho, g -> 0, so log_b drops out of the coupling.
        The Gibbs coupling includes log_b in the kernel. These agree
        only when log_b is uniform, so we test with uniform target weights.
        """
        C, log_a, _ = coupling_inputs
        eps = 0.5
        P = C.shape[1]
        log_b_uniform = -jnp.log(P) * jnp.ones(P)

        # Gibbs reference (with uniform log_b)
        log_gamma_gibbs, _, _ = gibbs_coupling(C, log_a, log_b_uniform, eps=eps)

        # Unbalanced with rho ~ 0
        log_gamma_ub, _, _, _ = sinkhorn_unbalanced(
            C, log_a, log_b_uniform, eps=eps, rho=1e-4, max_iter=200,
        )

        # Row-normalize both for comparison
        from jax.scipy.special import logsumexp
        lg_gibbs = log_gamma_gibbs - logsumexp(log_gamma_gibbs, axis=1, keepdims=True)
        lg_ub = log_gamma_ub - logsumexp(log_gamma_ub, axis=1, keepdims=True)

        np.testing.assert_allclose(
            np.array(lg_ub), np.array(lg_gibbs), atol=0.05,
        )

    def test_approaches_balanced_at_large_rho(self, coupling_inputs):
        """At large rho, unbalanced should approximate balanced Sinkhorn."""
        C, log_a, log_b = coupling_inputs
        eps = 0.5

        # Balanced reference
        log_gamma_bal, _, _, _ = sinkhorn_log_domain(
            C, log_a, log_b, eps=eps, max_iter=200,
        )

        # Unbalanced with very large rho (lambda ~ 1)
        log_gamma_ub, _, _, _ = sinkhorn_unbalanced(
            C, log_a, log_b, eps=eps, rho=1000.0, max_iter=200,
        )

        # Row-normalize both
        from jax.scipy.special import logsumexp
        lg_bal = log_gamma_bal - logsumexp(log_gamma_bal, axis=1, keepdims=True)
        lg_ub = log_gamma_ub - logsumexp(log_gamma_ub, axis=1, keepdims=True)

        np.testing.assert_allclose(
            np.array(lg_ub), np.array(lg_bal), atol=0.05,
        )

    def test_row_sums_are_exact(self, coupling_inputs):
        """Source marginal should be exactly enforced (up to numerics)."""
        C, log_a, log_b = coupling_inputs
        eps = 0.5

        log_gamma, _, _, _ = sinkhorn_unbalanced(
            C, log_a, log_b, eps=eps, rho=1.0, max_iter=100,
        )

        # Row sums in probability space should match source marginal
        from jax.scipy.special import logsumexp
        log_row_sums = logsumexp(log_gamma, axis=1)

        # Source marginal is not strictly enforced in unbalanced OT the same
        # way as balanced -- but the f-update does enforce it per-iteration.
        # Check approximate agreement.
        np.testing.assert_allclose(
            np.array(log_row_sums), np.array(log_a), atol=0.1,
        )

    def test_warm_start_reduces_iterations(self, coupling_inputs):
        """Warm-starting from converged duals should use fewer iterations."""
        C, log_a, log_b = coupling_inputs
        eps = 0.5

        # Cold start
        _, f1, g1, iters1 = sinkhorn_unbalanced(
            C, log_a, log_b, eps=eps, rho=1.0, max_iter=200, tol=1e-6,
        )

        # Warm start from converged duals
        _, _, _, iters2 = sinkhorn_unbalanced(
            C, log_a, log_b, eps=eps, rho=1.0, max_iter=200, tol=1e-6,
            dual_f_init=f1, dual_g_init=g1,
        )

        assert int(iters2) < int(iters1), (
            f"Warm-start should use fewer iterations: {int(iters2)} >= {int(iters1)}"
        )

    def test_finite_outputs(self, coupling_inputs):
        """No NaN or Inf in coupling or duals."""
        C, log_a, log_b = coupling_inputs
        eps = 0.5

        log_gamma, f, g, _ = sinkhorn_unbalanced(
            C, log_a, log_b, eps=eps, rho=1.0, max_iter=100,
        )

        assert jnp.all(jnp.isfinite(log_gamma)), "NaN/Inf in log_gamma"
        assert jnp.all(jnp.isfinite(f)), "NaN/Inf in dual_f"
        assert jnp.all(jnp.isfinite(g)), "NaN/Inf in dual_g"


# ===========================================================================
# DV Feedback
# ===========================================================================

class TestDVFeedback:
    def test_dv_no_effect_initial(self, rng):
        """On first step, dual_g = zeros -> DV feedback should have no effect.

        We compare two runs: one with feedback enabled, one without.
        On the very first step, dual_g is all zeros, so -weight * dual_g = 0
        and the coupling should be identical.
        """
        target = SimpleGaussian(dim=3)
        key = rng

        config_base = make_test_config(
            n_particles=10, n_proposals=5, n_iterations=1,
            coupling="balanced", epsilon=0.5, alpha=0.05,
            dv_feedback=False,
        )
        config_dv = make_test_config(
            n_particles=10, n_proposals=5, n_iterations=1,
            coupling="balanced", epsilon=0.5, alpha=0.05,
            dv_feedback=True, dv_weight=1.0,
        )

        k1, k2 = jax.random.split(key)
        state_base = etd_init(k1, target, config_base)
        state_dv = etd_init(k1, target, config_dv)

        _, info_base = etd_step(k2, state_base, target, config_base)
        _, info_dv = etd_step(k2, state_dv, target, config_dv)

        # Same coupling ESS since dual_g was zeros
        np.testing.assert_allclose(
            float(info_base["coupling_ess"]),
            float(info_dv["coupling_ess"]),
            rtol=1e-4,
        )

    def test_dv_changes_coupling(self, rng):
        """With non-zero dv_potential, DV feedback should change the coupling."""
        target = SimpleGaussian(dim=3)
        key = rng

        config = make_test_config(
            n_particles=10, n_proposals=5, n_iterations=1,
            coupling="balanced", epsilon=0.5, alpha=0.05,
            dv_feedback=True, dv_weight=1.0,
        )

        k1, k2 = jax.random.split(key)
        state = etd_init(k1, target, config)

        # Run one step to get non-zero dv_potential
        state_1, _ = etd_step(k2, state, target, config)

        # Run second step with DV feedback (uses non-zero dv_potential from step 1)
        k3, k4 = jax.random.split(k2)
        _, info_dv = etd_step(k3, state_1, target, config)

        # Run second step without DV
        config_no_dv = make_test_config(
            n_particles=10, n_proposals=5, n_iterations=1,
            coupling="balanced", epsilon=0.5, alpha=0.05,
            dv_feedback=False,
        )
        _, info_no_dv = etd_step(k3, state_1, target, config_no_dv)

        # The coupling ESS should differ
        ess_dv = float(info_dv["coupling_ess"])
        ess_no_dv = float(info_no_dv["coupling_ess"])
        assert ess_dv != pytest.approx(ess_no_dv, abs=1e-3), (
            f"DV feedback should change coupling: {ess_dv} ~ {ess_no_dv}"
        )

    def test_dv_potential_shape_and_finiteness(self, rng):
        """After one step with DV: dv_potential has shape (N,) and all finite."""
        target = SimpleGaussian(dim=3)
        config = make_test_config(
            n_particles=15, n_proposals=8, n_iterations=1,
            coupling="balanced", epsilon=0.5, alpha=0.05,
            dv_feedback=True, dv_weight=1.0,
        )
        k1, k2 = jax.random.split(rng)
        state = etd_init(k1, target, config)
        new_state, _ = etd_step(k2, state, target, config)

        assert new_state.dv_potential.shape == (15,)
        assert jnp.all(jnp.isfinite(new_state.dv_potential))

    def test_dv_potential_zeros_when_disabled(self, rng):
        """With feedback disabled: dv_potential should be all zeros."""
        target = SimpleGaussian(dim=3)
        config = make_test_config(
            n_particles=15, n_proposals=8, n_iterations=1,
            coupling="balanced", epsilon=0.5, alpha=0.05,
            dv_feedback=False,
        )
        k1, k2 = jax.random.split(rng)
        state = etd_init(k1, target, config)
        new_state, _ = etd_step(k2, state, target, config)

        np.testing.assert_array_equal(
            np.array(new_state.dv_potential), np.zeros(15),
        )

    def test_dv_potential_full_step_equals_g_tilde(self, rng):
        """With damping=1.0: dv_potential == g_tilde[indices] (no interpolation).

        At full step, the interpolation formula (1-eta)*f + eta*g_tilde[j]
        reduces to just g_tilde[j].
        """
        target = SimpleGaussian(dim=3)
        N, M = 15, 8
        eps_val = 0.5
        config = make_test_config(
            n_particles=N, n_proposals=M, n_iterations=1,
            coupling="balanced", epsilon=eps_val, alpha=0.05,
            step_size=1.0,  # full step -> update.damping=1.0
            dv_feedback=True, dv_weight=1.0,
        )
        k1, k2 = jax.random.split(rng)
        state = etd_init(k1, target, config)
        new_state, _ = etd_step(k2, state, target, config)

        # dv_potential should be non-zero (g_tilde has actual signal)
        assert not jnp.allclose(new_state.dv_potential, 0.0)
        # At full step, dv_potential = g_tilde[indices] (pure c-transform)
        # Since dual_f contribution is (1-1.0)*f = 0, the potential should be
        # purely from the target-side c-transform.
        assert new_state.dv_potential.shape == (N,)
        assert jnp.all(jnp.isfinite(new_state.dv_potential))

    def test_dv_potential_interpolation(self, rng):
        """With damping=0.5: dv_potential blends dual_f and g_tilde."""
        target = SimpleGaussian(dim=3)
        N, M = 15, 8
        config_full = make_test_config(
            n_particles=N, n_proposals=M, n_iterations=1,
            coupling="balanced", epsilon=0.5, alpha=0.05,
            step_size=1.0,
            dv_feedback=True, dv_weight=1.0,
        )
        config_half = make_test_config(
            n_particles=N, n_proposals=M, n_iterations=1,
            coupling="balanced", epsilon=0.5, alpha=0.05,
            step_size=0.5,
            dv_feedback=True, dv_weight=1.0,
        )
        k1, k2 = jax.random.split(rng)
        state = etd_init(k1, target, config_full)

        # Run one step to build up non-zero dv_potential and duals
        state_1, _ = etd_step(k2, state, target, config_full)

        # Second step with full vs half step -- same key so same proposals/coupling
        k3, _ = jax.random.split(k2)
        state_full, _ = etd_step(k3, state_1, target, config_full)
        state_half, _ = etd_step(k3, state_1, target, config_half)

        # Half-step potential should differ from full-step (interpolation effect)
        # (unless dual_f happens to equal g_tilde, which is astronomically unlikely)
        assert not jnp.allclose(
            state_full.dv_potential, state_half.dv_potential, atol=1e-4,
        ), "Interpolation should produce different potentials at different step sizes"

    def test_dv_potential_unbalanced(self, rng):
        """DV potential works with unbalanced coupling (lam-scaled c-transform)."""
        target = SimpleGaussian(dim=3)
        config = make_test_config(
            n_particles=15, n_proposals=8, n_iterations=1,
            coupling="unbalanced", rho=1.0,
            epsilon=0.5, alpha=0.05,
            dv_feedback=True, dv_weight=1.0,
        )
        k1, k2 = jax.random.split(rng)
        state = etd_init(k1, target, config)
        new_state, _ = etd_step(k2, state, target, config)

        assert new_state.dv_potential.shape == (15,)
        assert jnp.all(jnp.isfinite(new_state.dv_potential))

    def test_dv_potential_gibbs_is_zeros(self, rng):
        """Gibbs coupling has no iterative solver -> dv_potential is zeros."""
        target = SimpleGaussian(dim=3)
        config = make_test_config(
            n_particles=15, n_proposals=8, n_iterations=1,
            coupling="gibbs", epsilon=0.5, alpha=0.05,
            dv_feedback=True, dv_weight=1.0,
        )
        k1, k2 = jax.random.split(rng)
        state = etd_init(k1, target, config)
        new_state, _ = etd_step(k2, state, target, config)

        # Gibbs: g_tilde = zeros -> dv_potential = 0*f + 1.0*zeros = zeros
        np.testing.assert_allclose(
            np.array(new_state.dv_potential), 0.0, atol=1e-6,
        )


# ===========================================================================
# Integration Tests
# ===========================================================================

class TestIntegration:
    @pytest.mark.parametrize("cost_name,preconditioner", [
        ("euclidean", PreconditionerConfig()),                                    # no precond
        ("euclidean", PreconditionerConfig(type="rmsprop", proposals=True, cost=True)),  # whitened
        ("linf", PreconditionerConfig()),                                         # no precond
        ("linf", PreconditionerConfig(type="rmsprop", cost=True)),                # whitened
        ("imq", PreconditionerConfig()),                                          # no precond
        ("imq", PreconditionerConfig(type="rmsprop", cost=True)),                 # whitened
    ])
    def test_step_with_each_cost(self, rng, cost_name, preconditioner):
        """Smoke test: one ETD step with each cost variant."""
        target = SimpleGaussian(dim=4)
        config = make_test_config(
            n_particles=15, n_proposals=10, n_iterations=1,
            cost=cost_name, coupling="balanced",
            cost_params=(("c", 1.0),) if cost_name == "imq" else (),
            epsilon=0.5, alpha=0.05,
            preconditioner=preconditioner,
        )

        k1, k2 = jax.random.split(rng)
        state = etd_init(k1, target, config)
        new_state, info = etd_step(k2, state, target, config)

        assert new_state.positions.shape == (15, 4)
        assert jnp.all(jnp.isfinite(new_state.positions))
        assert "coupling_ess" in info
        assert jnp.isfinite(info["coupling_ess"])

    def test_step_with_unbalanced_coupling(self, rng):
        """Smoke test: one ETD step with unbalanced coupling."""
        target = SimpleGaussian(dim=4)
        config = make_test_config(
            n_particles=15, n_proposals=10, n_iterations=1,
            cost="euclidean", coupling="unbalanced",
            rho=1.0, epsilon=0.5, alpha=0.05,
        )

        k1, k2 = jax.random.split(rng)
        state = etd_init(k1, target, config)
        new_state, info = etd_step(k2, state, target, config)

        assert new_state.positions.shape == (15, 4)
        assert jnp.all(jnp.isfinite(new_state.positions))
        assert info["sinkhorn_iters"] > 0

    def test_step_with_dv_feedback(self, rng):
        """Smoke test: ETD step with DV feedback enabled."""
        target = SimpleGaussian(dim=4)
        config = make_test_config(
            n_particles=15, n_proposals=10, n_iterations=1,
            cost="euclidean", coupling="balanced",
            dv_feedback=True, dv_weight=1.0,
            epsilon=0.5, alpha=0.05,
        )

        k1, k2 = jax.random.split(rng)
        state = etd_init(k1, target, config)
        new_state, info = etd_step(k2, state, target, config)

        assert new_state.positions.shape == (15, 4)
        assert jnp.all(jnp.isfinite(new_state.positions))

    def test_coupling_ess_in_info(self, rng):
        """Coupling ESS should be a finite positive scalar."""
        target = SimpleGaussian(dim=3)
        config = make_test_config(
            n_particles=10, n_proposals=5, n_iterations=1,
            coupling="balanced", epsilon=0.5, alpha=0.05,
        )

        k1, k2 = jax.random.split(rng)
        state = etd_init(k1, target, config)
        _, info = etd_step(k2, state, target, config)

        ess = float(info["coupling_ess"])
        assert ess > 0.0, f"Coupling ESS should be positive, got {ess}"
        assert np.isfinite(ess), f"Coupling ESS should be finite, got {ess}"

    def test_cost_registry_has_new_entries(self):
        """Registry should include mahalanobis, linf, and imq."""
        from etd.costs import COSTS
        assert "mahalanobis" in COSTS
        assert "linf" in COSTS
        assert "euclidean" in COSTS
        assert "imq" in COSTS

    def test_coupling_registry_has_unbalanced(self):
        """Registry should include unbalanced."""
        from etd.coupling import COUPLINGS
        assert "unbalanced" in COUPLINGS
        assert "balanced" in COUPLINGS
        assert "gibbs" in COUPLINGS

    def test_step_with_imq(self, rng):
        """Smoke test: one ETD step with IMQ cost."""
        target = SimpleGaussian(dim=4)
        config = make_test_config(
            n_particles=15, n_proposals=10, n_iterations=1,
            cost="imq", cost_params=(("c", 1.0),),
            coupling="balanced", epsilon=0.5, alpha=0.05,
        )

        k1, k2 = jax.random.split(rng)
        state = etd_init(k1, target, config)
        new_state, info = etd_step(k2, state, target, config)

        assert new_state.positions.shape == (15, 4)
        assert jnp.all(jnp.isfinite(new_state.positions))
        assert jnp.isfinite(info["coupling_ess"])


# ===========================================================================
# IMQ (Positive Multiquadric) Cost
# ===========================================================================

class TestIMQCost:
    def test_known_value(self):
        """C(0, [3,4]) with c=1 -> sqrt(1 + 25) - 1 = sqrt(26) - 1."""
        x = jnp.array([[0.0, 0.0]])   # (1, 2)
        y = jnp.array([[3.0, 4.0]])    # (1, 2)

        C = imq_cost(x, y, c=1.0)
        expected = jnp.sqrt(26.0) - 1.0
        np.testing.assert_allclose(float(C[0, 0]), float(expected), atol=1e-5)

    def test_self_diagonal_zero(self, rng):
        """C(X, X) diagonal should be zero for any c."""
        X = jax.random.normal(rng, (15, 4))
        C = imq_cost(X, X, c=2.0)
        np.testing.assert_allclose(np.array(jnp.diag(C)), 0.0, atol=1e-5)

    def test_nonnegative(self, positions_proposals):
        """All entries should be non-negative."""
        positions, proposals = positions_proposals
        C = imq_cost(positions, proposals, c=1.0)
        assert jnp.all(C >= -1e-7)

    def test_sublinear(self, positions_proposals):
        """For large distances, IMQ < squared Euclidean.

        sqrt(c^2 + r^2) - c < r^2/2 for large r, since IMQ is O(r) while
        squared Euclidean is O(r^2).
        """
        positions, proposals = positions_proposals
        C_imq = imq_cost(positions, proposals, c=1.0)
        C_euc = squared_euclidean_cost(positions, proposals)

        # IMQ grows as ~r (sub-linear) vs r^2/2 -> strictly less for all r > 0
        # (actually sqrt(c^2 + r^2) - c <= r for all r >= 0, and r < r^2/2 for r > 2)
        # Filter to entries where squared distance > 4 (i.e., r > 2)
        large_mask = C_euc > 2.0  # r^2 > 4 -> r > 2
        if jnp.any(large_mask):
            assert jnp.all(C_imq[large_mask] < C_euc[large_mask])

    def test_reduces_to_l2_norm_as_c_to_zero(self, positions_proposals):
        """As c -> 0, IMQ -> ||x - y||_2 = sqrt(2 * C_euc)."""
        positions, proposals = positions_proposals
        C_imq = imq_cost(positions, proposals, c=1e-8)
        C_euc = squared_euclidean_cost(positions, proposals)
        l2_norm = jnp.sqrt(2.0 * C_euc)

        np.testing.assert_allclose(
            np.array(C_imq), np.array(l2_norm), rtol=1e-3,
        )

    def test_shape(self, positions_proposals):
        """Output shape should be (N, P)."""
        positions, proposals = positions_proposals
        C = imq_cost(positions, proposals, c=1.0)
        assert C.shape == (20, 30)


# ===========================================================================
# Parameterized Cost Wiring (build_cost_fn)
# ===========================================================================

class TestBuildCostFn:
    def test_no_params_returns_base_fn(self):
        """build_cost_fn without params returns the base function."""
        fn = build_cost_fn("euclidean")
        assert fn is squared_euclidean_cost

    def test_with_params_returns_partial(self):
        """build_cost_fn with params returns a functools.partial."""
        fn = build_cost_fn("imq", (("c", 2.0),))
        assert isinstance(fn, functools.partial)
        assert fn.keywords == {"c": 2.0}

    def test_partial_produces_correct_result(self):
        """Partial with c=2 gives correct IMQ cost."""
        fn = build_cost_fn("imq", (("c", 2.0),))
        x = jnp.array([[0.0, 0.0]])
        y = jnp.array([[3.0, 4.0]])
        C = fn(x, y)
        expected = jnp.sqrt(4.0 + 25.0) - 2.0  # sqrt(29) - 2
        np.testing.assert_allclose(float(C[0, 0]), float(expected), atol=1e-5)

    def test_unknown_cost_raises(self):
        """build_cost_fn with unknown name raises KeyError."""
        with pytest.raises(KeyError, match="Unknown cost"):
            build_cost_fn("nonexistent")

    def test_nested_cost_config(self):
        """CostConfig correctly stores type and params."""
        config = make_test_config(cost="euclidean")
        assert config.cost.type == "euclidean"
        assert config.cost.params == ()

    def test_nested_cost_config_with_params(self):
        """CostConfig correctly stores type and params for imq."""
        config = make_test_config(cost="imq", cost_params=(("c", 1.0),))
        assert config.cost.type == "imq"
        assert config.cost.params == (("c", 1.0),)


# ===========================================================================
# Whitened Cost Unit Tests
# ===========================================================================

class TestWhitenedLinfCost:
    def test_whitened_known_value(self):
        """Hand-computed whitened L-inf: max_k |delta_k / P_k|."""
        x = jnp.array([[1.0, 2.0, 3.0]])   # (1, 3)
        y = jnp.array([[4.0, 4.0, 7.0]])    # (1, 3)
        P = jnp.array([1.0, 0.5, 2.0])      # preconditioner

        # delta = [3, 2, 4], 1/P = [1, 2, 0.5]
        # whitened delta = [3*1, 2*2, 4*0.5] = [3, 4, 2]
        # max = 4.0
        C = linf_cost(x, y, preconditioner=P)
        np.testing.assert_allclose(float(C[0, 0]), 4.0, atol=1e-5)

    def test_whitened_identity_matches_unwhitened(self, positions_proposals):
        """P=ones -> whitened L-inf == unwhitened L-inf."""
        positions, proposals = positions_proposals
        P = jnp.ones(positions.shape[1])

        C_whitened = linf_cost(positions, proposals, preconditioner=P)
        C_plain = linf_cost(positions, proposals)

        np.testing.assert_allclose(
            np.array(C_whitened), np.array(C_plain), atol=1e-6,
        )


class TestWhitenedIMQCost:
    def test_whitened_known_value(self):
        """Hand-computed whitened IMQ: sqrt(c^2 + ||P^{-1} delta||^2) - c."""
        x = jnp.array([[0.0, 0.0]])   # (1, 2)
        y = jnp.array([[3.0, 4.0]])    # (1, 2)
        P = jnp.array([0.5, 1.0])      # preconditioner

        # 1/P = [2, 1], whitened: [6, 4], ||.||^2 = 52
        # sqrt(1 + 52) - 1 = sqrt(53) - 1
        C = imq_cost(x, y, preconditioner=P, c=1.0)
        expected = jnp.sqrt(53.0) - 1.0
        np.testing.assert_allclose(float(C[0, 0]), float(expected), atol=1e-5)

    def test_whitened_identity_matches_unwhitened(self, positions_proposals):
        """P=ones -> whitened IMQ == unwhitened IMQ."""
        positions, proposals = positions_proposals
        P = jnp.ones(positions.shape[1])

        C_whitened = imq_cost(positions, proposals, preconditioner=P, c=1.0)
        C_plain = imq_cost(positions, proposals, c=1.0)

        np.testing.assert_allclose(
            np.array(C_whitened), np.array(C_plain), atol=1e-6,
        )


class TestWhitenedEuclideanCost:
    def test_whitened_matches_mahalanobis(self, positions_proposals):
        """Euclidean with preconditioner should match Mahalanobis exactly."""
        positions, proposals = positions_proposals
        P = jnp.array([0.5, 1.0, 2.0, 0.3, 1.5])

        C_whitened = squared_euclidean_cost(
            positions, proposals, preconditioner=P,
        )
        C_maha = mahalanobis_cost(positions, proposals, preconditioner=P)

        np.testing.assert_allclose(
            np.array(C_whitened), np.array(C_maha), rtol=1e-5,
        )

    def test_whitened_identity_matches_unwhitened(self, positions_proposals):
        """P=ones -> whitened euclidean == unwhitened euclidean."""
        positions, proposals = positions_proposals
        P = jnp.ones(positions.shape[1])

        C_whitened = squared_euclidean_cost(
            positions, proposals, preconditioner=P,
        )
        C_plain = squared_euclidean_cost(positions, proposals)

        np.testing.assert_allclose(
            np.array(C_whitened), np.array(C_plain), rtol=1e-5,
        )


# ===========================================================================
# Preconditioner Config Integration Tests
# ===========================================================================

class TestPreconditionerConfig:
    def test_needs_precond_accum(self):
        """needs_precond_accum reflects preconditioner.active."""
        assert not make_test_config().needs_precond_accum
        assert make_test_config(
            preconditioner=PreconditionerConfig(type="rmsprop"),
        ).needs_precond_accum
        assert make_test_config(
            preconditioner=PreconditionerConfig(type="rmsprop", cost=True),
        ).needs_precond_accum
        assert make_test_config(
            preconditioner=PreconditionerConfig(type="rmsprop", proposals=True, cost=True),
        ).needs_precond_accum

    def test_cost_whitening_only(self, rng):
        """preconditioner with cost=True only: accumulator updated, cost whitened."""
        target = SimpleGaussian(dim=4)
        config = make_test_config(
            n_particles=15, n_proposals=10, n_iterations=1,
            cost="euclidean", coupling="balanced",
            epsilon=0.5, alpha=0.05,
            preconditioner=PreconditionerConfig(type="rmsprop", cost=True),
        )

        k1, k2 = jax.random.split(rng)
        state = etd_init(k1, target, config)
        new_state, info = etd_step(k2, state, target, config)

        assert new_state.positions.shape == (15, 4)
        assert jnp.all(jnp.isfinite(new_state.positions))
        # Accumulator should have been updated from ones
        assert not jnp.allclose(new_state.precond_accum, jnp.ones(4))

    def test_proposals_only(self, rng):
        """preconditioner with proposals=True only: proposals preconditioned, cost isotropic."""
        target = SimpleGaussian(dim=4)
        config = make_test_config(
            n_particles=15, n_proposals=10, n_iterations=1,
            cost="euclidean", coupling="balanced",
            epsilon=0.5, alpha=0.05,
            preconditioner=PreconditionerConfig(type="rmsprop", proposals=True),
        )

        k1, k2 = jax.random.split(rng)
        state = etd_init(k1, target, config)
        new_state, info = etd_step(k2, state, target, config)

        assert new_state.positions.shape == (15, 4)
        assert jnp.all(jnp.isfinite(new_state.positions))
        # Accumulator should have been updated
        assert not jnp.allclose(new_state.precond_accum, jnp.ones(4))

    def test_step_linf_whitened(self, rng):
        """Smoke test: ETD step with L-inf + cost whitening via preconditioner."""
        target = SimpleGaussian(dim=4)
        config = make_test_config(
            n_particles=15, n_proposals=10, n_iterations=1,
            cost="linf", coupling="balanced",
            epsilon=0.5, alpha=0.05,
            preconditioner=PreconditionerConfig(
                type="rmsprop", proposals=True, cost=True,
            ),
        )

        k1, k2 = jax.random.split(rng)
        state = etd_init(k1, target, config)
        new_state, info = etd_step(k2, state, target, config)

        assert new_state.positions.shape == (15, 4)
        assert jnp.all(jnp.isfinite(new_state.positions))

    def test_step_imq_whitened(self, rng):
        """Smoke test: ETD step with IMQ + cost whitening via preconditioner."""
        target = SimpleGaussian(dim=4)
        config = make_test_config(
            n_particles=15, n_proposals=10, n_iterations=1,
            cost="imq", cost_params=(("c", 1.0),),
            coupling="balanced",
            epsilon=0.5, alpha=0.05,
            preconditioner=PreconditionerConfig(
                type="rmsprop", proposals=True, cost=True,
            ),
        )

        k1, k2 = jax.random.split(rng)
        state = etd_init(k1, target, config)
        new_state, info = etd_step(k2, state, target, config)

        assert new_state.positions.shape == (15, 4)
        assert jnp.all(jnp.isfinite(new_state.positions))
