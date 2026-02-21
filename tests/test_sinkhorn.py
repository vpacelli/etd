"""Phase 0 gate tests for Sinkhorn and coupling solvers.

Required gate (ROADMAP.md):
  - Balanced (50,200): row sums ≈ a, col sums ≈ b  (atol=1e-4)
  - Warm-start ≤ 10 iterations  (cold-start ~40)
  - ε → ∞: max row entropy → log P  (atol=0.1)
  - ε → 0: min row entropy → 0  (mean < 0.5)
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from etd.coupling.gibbs import gibbs_coupling
from etd.coupling.sinkhorn import sinkhorn_log_domain


def _row_entropy(log_gamma: jnp.ndarray) -> jnp.ndarray:
    """Compute Shannon entropy of each row (in nats).

    H_i = -Σ_j γ_ij log(γ_ij)
    """
    gamma = jnp.exp(log_gamma)
    # Avoid log(0) by using log_gamma directly where gamma > 0
    return -jnp.sum(gamma * log_gamma, axis=1)


# ---------------------------------------------------------------------------
# Gate: Balanced marginals
# ---------------------------------------------------------------------------

class TestBalancedMarginals:
    def test_row_and_col_sums(self):
        """Row sums ≈ a and col sums ≈ b for balanced Sinkhorn."""
        key = jax.random.PRNGKey(0)
        N, P = 50, 200

        # Random cost matrix
        k1, k2, k3 = jax.random.split(key, 3)
        C = jnp.abs(jax.random.normal(k1, (N, P)))

        # Uniform marginals
        log_a = -jnp.log(N) * jnp.ones(N)
        log_b = -jnp.log(P) * jnp.ones(P)

        log_gamma, f, g, iters = sinkhorn_log_domain(
            C, log_a, log_b, eps=0.1, max_iter=200, tol=1e-6,
        )

        gamma = jnp.exp(log_gamma)

        # Row sums should equal a = 1/N
        row_sums = gamma.sum(axis=1)
        np.testing.assert_allclose(
            np.array(row_sums), np.array(jnp.exp(log_a)), atol=1e-4,
        )

        # Column sums should equal b = 1/P
        col_sums = gamma.sum(axis=0)
        np.testing.assert_allclose(
            np.array(col_sums), np.array(jnp.exp(log_b)), atol=1e-4,
        )

    def test_nonuniform_marginals(self):
        """Sinkhorn should handle non-uniform marginals correctly."""
        key = jax.random.PRNGKey(1)
        N, P = 20, 30

        k1, k2, k3 = jax.random.split(key, 3)
        C = jnp.abs(jax.random.normal(k1, (N, P)))

        # Non-uniform marginals (still sum to 1)
        a_raw = jax.random.uniform(k2, (N,)) + 0.1
        b_raw = jax.random.uniform(k3, (P,)) + 0.1
        log_a = jnp.log(a_raw) - logsumexp(jnp.log(a_raw))
        log_b = jnp.log(b_raw) - logsumexp(jnp.log(b_raw))

        log_gamma, f, g, iters = sinkhorn_log_domain(
            C, log_a, log_b, eps=0.1, max_iter=200, tol=1e-6,
        )

        gamma = jnp.exp(log_gamma)

        row_sums = gamma.sum(axis=1)
        col_sums = gamma.sum(axis=0)

        np.testing.assert_allclose(
            np.array(row_sums), np.array(jnp.exp(log_a)), atol=1e-4,
        )
        np.testing.assert_allclose(
            np.array(col_sums), np.array(jnp.exp(log_b)), atol=1e-4,
        )


# ---------------------------------------------------------------------------
# Gate: Warm-start
# ---------------------------------------------------------------------------

class TestWarmStart:
    def test_warm_start_fewer_iterations(self):
        """Warm-started solve should use ≤ 10 iterations."""
        key = jax.random.PRNGKey(2)
        N, P = 50, 200

        k1, _ = jax.random.split(key)
        C = jnp.abs(jax.random.normal(k1, (N, P)))
        log_a = -jnp.log(N) * jnp.ones(N)
        log_b = -jnp.log(P) * jnp.ones(P)

        # Cold start
        _, f_cold, g_cold, iters_cold = sinkhorn_log_domain(
            C, log_a, log_b, eps=0.1, max_iter=200, tol=1e-5,
        )

        # Warm start from cold solution
        _, _, _, iters_warm = sinkhorn_log_domain(
            C, log_a, log_b, eps=0.1, max_iter=200, tol=1e-5,
            dual_f_init=f_cold, dual_g_init=g_cold,
        )

        assert int(iters_warm) <= 10, (
            f"Warm start used {int(iters_warm)} iterations (expected ≤ 10); "
            f"cold start used {int(iters_cold)}"
        )


# ---------------------------------------------------------------------------
# Gate: Entropic limits
# ---------------------------------------------------------------------------

class TestEntropicLimits:
    def test_large_eps_max_entropy(self):
        """As ε → ∞, coupling → product measure, row entropy → log P."""
        key = jax.random.PRNGKey(3)
        N, P = 20, 50

        C = jnp.abs(jax.random.normal(key, (N, P)))
        log_a = -jnp.log(N) * jnp.ones(N)
        log_b = -jnp.log(P) * jnp.ones(P)

        log_gamma, _, _, _ = sinkhorn_log_domain(
            C, log_a, log_b, eps=1000.0, max_iter=200, tol=1e-6,
        )

        # Row-normalize to get conditional for entropy computation
        log_gamma_cond = log_gamma - logsumexp(log_gamma, axis=1, keepdims=True)
        entropies = _row_entropy(log_gamma_cond)
        max_entropy = jnp.log(P)  # uniform conditional → log(P) nats

        np.testing.assert_allclose(
            np.array(entropies), float(max_entropy), atol=0.1,
        )

    def test_small_eps_low_entropy(self):
        """As ε → 0, coupling concentrates, row entropy → 0.

        Uses a square coupling (N = P) so the balanced constraint allows
        a 1-to-1 assignment at small ε.  With N < P, the minimum
        achievable conditional entropy is bounded below by log(P/N) > 0.
        """
        key = jax.random.PRNGKey(4)
        N, P = 20, 20  # square so 1-to-1 assignment is feasible

        C = jnp.abs(jax.random.normal(key, (N, P))) + 0.1  # avoid zero costs
        log_a = -jnp.log(N) * jnp.ones(N)
        log_b = -jnp.log(P) * jnp.ones(P)

        log_gamma, _, _, _ = sinkhorn_log_domain(
            C, log_a, log_b, eps=0.01, max_iter=500, tol=1e-8,
        )

        # Row-normalize to get conditional for entropy computation
        log_gamma_cond = log_gamma - logsumexp(log_gamma, axis=1, keepdims=True)
        entropies = _row_entropy(log_gamma_cond)
        assert float(jnp.mean(entropies)) < 0.5, (
            f"Mean row entropy = {float(jnp.mean(entropies)):.3f}, expected < 0.5"
        )


# ---------------------------------------------------------------------------
# Additional: Gibbs coupling
# ---------------------------------------------------------------------------

class TestGibbsCoupling:
    def test_row_sums_equal_source_marginal(self):
        """Gibbs coupling row sums should equal the source marginal a."""
        key = jax.random.PRNGKey(5)
        N, P = 20, 50

        C = jnp.abs(jax.random.normal(key, (N, P)))
        log_a = -jnp.log(N) * jnp.ones(N)
        log_b = -jnp.log(P) * jnp.ones(P)

        log_gamma, dual_f, dual_g = gibbs_coupling(C, log_a, log_b, eps=0.1)

        row_sums = jnp.exp(log_gamma).sum(axis=1)
        np.testing.assert_allclose(
            np.array(row_sums), np.array(jnp.exp(log_a)), atol=1e-5,
        )

        # Duals should be zeros (interface compatibility)
        np.testing.assert_array_equal(np.array(dual_f), np.zeros(N))
        np.testing.assert_array_equal(np.array(dual_g), np.zeros(P))

    def test_gibbs_matches_sinkhorn_large_eps(self):
        """At very large ε, Gibbs and Sinkhorn should produce similar couplings.

        Both approach the product measure, so they should agree on the
        row-normalized coupling.
        """
        key = jax.random.PRNGKey(6)
        N, P = 10, 30

        C = jnp.abs(jax.random.normal(key, (N, P)))
        log_a = -jnp.log(N) * jnp.ones(N)
        log_b = -jnp.log(P) * jnp.ones(P)

        eps = 100.0  # very large

        log_gamma_gibbs, _, _ = gibbs_coupling(C, log_a, log_b, eps)
        log_gamma_sink, _, _, _ = sinkhorn_log_domain(
            C, log_a, log_b, eps, max_iter=200, tol=1e-6,
        )

        np.testing.assert_allclose(
            np.array(jnp.exp(log_gamma_gibbs)),
            np.array(jnp.exp(log_gamma_sink)),
            atol=0.05,
        )


# ---------------------------------------------------------------------------
# Additional: Sinkhorn validity
# ---------------------------------------------------------------------------

class TestSinkhornValidity:
    def test_duals_finite(self):
        """Returned dual potentials should be finite (no NaN/Inf)."""
        key = jax.random.PRNGKey(7)
        N, P = 30, 100

        C = jnp.abs(jax.random.normal(key, (N, P)))
        log_a = -jnp.log(N) * jnp.ones(N)
        log_b = -jnp.log(P) * jnp.ones(P)

        _, f, g, _ = sinkhorn_log_domain(
            C, log_a, log_b, eps=0.1, max_iter=100, tol=1e-5,
        )

        assert jnp.all(jnp.isfinite(f)), "dual_f has non-finite values"
        assert jnp.all(jnp.isfinite(g)), "dual_g has non-finite values"

    def test_log_gamma_valid(self):
        """log_gamma should be finite and row sums should equal source marginal."""
        key = jax.random.PRNGKey(8)
        N, P = 30, 100

        C = jnp.abs(jax.random.normal(key, (N, P)))
        log_a = -jnp.log(N) * jnp.ones(N)
        log_b = -jnp.log(P) * jnp.ones(P)

        log_gamma, _, _, _ = sinkhorn_log_domain(
            C, log_a, log_b, eps=0.1, max_iter=100, tol=1e-5,
        )

        assert jnp.all(jnp.isfinite(log_gamma)), "log_gamma has non-finite values"

        # Joint coupling: row sums should equal source marginal
        gamma = jnp.exp(log_gamma)
        row_sums = gamma.sum(axis=1)
        np.testing.assert_allclose(
            np.array(row_sums), np.array(jnp.exp(log_a)), atol=1e-4,
        )
