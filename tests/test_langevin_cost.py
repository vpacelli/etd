"""Tests for Langevin-residual cost (LRET).

Covers:
  - Unit tests: manual computation, score-free degeneracy, non-negativity,
    whitened Mahalanobis, shape contract
  - FDR wiring: default alpha=epsilon, alpha override, use_score enforcement
  - Integration: LRET-B convergence on GMM 2D, LRET-SDD convergence,
    LRET vs ETD comparison
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from etd.costs.euclidean import squared_euclidean_cost
from etd.costs.langevin import langevin_residual_cost
from etd.step import init as etd_init, step as etd_step
from etd.types import ETDConfig
from etd.targets.gmm import GMMTarget


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
def gmm_2d_4():
    return GMMTarget(dim=2, n_modes=4, arrangement="grid", separation=6.0)


# ===========================================================================
# 5a. Unit tests for langevin_residual_cost
# ===========================================================================

class TestLangevinResidualCost:

    def test_correctness_vs_manual(self):
        """Small case (3 particles, 5 proposals, 2D): match manual computation."""
        positions = jnp.array([
            [1.0, 2.0],
            [3.0, 0.0],
            [-1.0, 1.0],
        ])  # (3, 2)
        proposals = jnp.array([
            [1.5, 2.5],
            [3.2, 0.1],
            [0.0, 0.0],
            [-0.5, 1.5],
            [2.0, 1.0],
        ])  # (5, 2)
        scores = jnp.array([
            [-1.0, -2.0],
            [-3.0, 0.0],
            [1.0, -1.0],
        ])  # (3, 2)
        eps = 0.1

        C = langevin_residual_cost(positions, proposals, scores, eps)
        assert C.shape == (3, 5)

        # Manual: m_i = x_i + eps * s_i
        # m_0 = [1.0, 2.0] + 0.1*[-1.0, -2.0] = [0.9, 1.8]
        # m_1 = [3.0, 0.0] + 0.1*[-3.0,  0.0] = [2.7, 0.0]
        # m_2 = [-1.0, 1.0] + 0.1*[1.0, -1.0] = [-0.9, 0.9]
        means = positions + eps * scores
        for i in range(3):
            for j in range(5):
                diff = proposals[j] - means[i]
                expected = float(jnp.sum(diff ** 2)) / (4.0 * eps)
                np.testing.assert_allclose(
                    float(C[i, j]), expected, atol=1e-5,
                    err_msg=f"Mismatch at ({i}, {j})",
                )

    def test_score_free_degeneracy(self, rng):
        """When scores=0, Langevin cost = ||y-x||² / (4ε).

        Standard Euclidean is ||y-x||² / 2, so ratio is 1/(2ε).
        """
        k1, k2 = jax.random.split(rng)
        N, P, d = 10, 20, 5
        positions = jax.random.normal(k1, (N, d))
        proposals = jax.random.normal(k2, (P, d))
        scores = jnp.zeros((N, d))
        eps = 0.25

        C_lang = langevin_residual_cost(positions, proposals, scores, eps)
        C_eucl = squared_euclidean_cost(positions, proposals)
        # C_eucl = ||y-x||² / 2, C_lang = ||y-x||² / (4ε) = C_eucl / (2ε)
        expected = C_eucl / (2.0 * eps)
        np.testing.assert_allclose(C_lang, expected, atol=1e-5)

    def test_non_negativity(self, rng):
        """Cost matrix should be >= 0 for random inputs."""
        k1, k2, k3 = jax.random.split(rng, 3)
        N, P, d = 15, 30, 8
        positions = jax.random.normal(k1, (N, d))
        proposals = jax.random.normal(k2, (P, d))
        scores = jax.random.normal(k3, (N, d))

        C = langevin_residual_cost(positions, proposals, scores, 0.1)
        assert jnp.all(C >= 0.0)

    def test_whitened_correctness(self, rng):
        """Whitened cost matches manual Mahalanobis computation."""
        k1, k2, k3 = jax.random.split(rng, 3)
        N, P, d = 3, 4, 3
        positions = jax.random.normal(k1, (N, d))
        proposals = jax.random.normal(k2, (P, d))
        scores = jax.random.normal(k3, (N, d))
        eps = 0.2

        # Build a known PD Cholesky factor
        L = jnp.array([
            [2.0, 0.0, 0.0],
            [0.5, 1.5, 0.0],
            [0.1, 0.3, 1.0],
        ])

        C = langevin_residual_cost(
            positions, proposals, scores, eps,
            cholesky_factor=L, whiten=True,
        )
        assert C.shape == (N, P)

        # Manual: Σ = LL^T, drift = Σ·s, m = x + ε·Σ·s
        Sigma = L @ L.T
        drift = scores @ Sigma  # (N, d) — note: s @ Σ = s @ (LL^T) = (s @ L) @ L^T
        means = positions + eps * drift

        # L^{-1} applied to both
        L_inv = jnp.linalg.inv(L)
        means_w = means @ L_inv.T
        proposals_w = proposals @ L_inv.T

        for i in range(N):
            for j in range(P):
                diff = proposals_w[j] - means_w[i]
                expected = float(jnp.sum(diff ** 2)) / (4.0 * eps)
                np.testing.assert_allclose(
                    float(C[i, j]), expected, atol=1e-4,
                    err_msg=f"Whitened mismatch at ({i}, {j})",
                )

    def test_shape_contract(self, rng):
        """(N,d) × (P,d) × (N,d) × scalar → (N,P)."""
        k1, k2, k3 = jax.random.split(rng, 3)
        N, P, d = 7, 13, 4
        positions = jax.random.normal(k1, (N, d))
        proposals = jax.random.normal(k2, (P, d))
        scores = jax.random.normal(k3, (N, d))

        C = langevin_residual_cost(positions, proposals, scores, 0.05)
        assert C.shape == (N, P)
        assert C.dtype == jnp.float32

    def test_whiten_without_cholesky_falls_back(self, rng):
        """whiten=True but cholesky_factor=None → isotropic (no crash)."""
        k1, k2, k3 = jax.random.split(rng, 3)
        N, P, d = 5, 10, 3
        positions = jax.random.normal(k1, (N, d))
        proposals = jax.random.normal(k2, (P, d))
        scores = jax.random.normal(k3, (N, d))

        C_whiten_none = langevin_residual_cost(
            positions, proposals, scores, 0.1,
            whiten=True, cholesky_factor=None,
        )
        C_isotropic = langevin_residual_cost(
            positions, proposals, scores, 0.1,
        )
        np.testing.assert_allclose(C_whiten_none, C_isotropic, atol=1e-6)


# ===========================================================================
# 5b. FDR wiring tests
# ===========================================================================

class TestLangevinFDRWiring:

    def test_default_fdr_alpha_equals_epsilon(self):
        """YAML {cost: langevin, epsilon: 0.2} → alpha=0.2, fdr=True."""
        from experiments.run import build_algo_config

        entry = {
            "label": "LRET-B",
            "cost": {"type": "langevin"},
            "coupling": "balanced",
            "update": "categorical",
            "epsilon": 0.2,
        }
        shared = {"n_particles": 50, "n_iterations": 100}
        config, _, _, _ = build_algo_config(entry, shared)

        assert config.cost == "langevin"
        assert config.alpha == pytest.approx(0.2)
        assert config.fdr is True
        assert config.use_score is True

    def test_alpha_override(self):
        """Explicit alpha=0.03 is respected alongside cost: langevin."""
        from experiments.run import build_algo_config

        entry = {
            "label": "LRET-B-Custom",
            "cost": {"type": "langevin"},
            "coupling": "balanced",
            "update": "categorical",
            "epsilon": 0.1,
            "alpha": 0.03,
        }
        shared = {"n_particles": 50, "n_iterations": 100}
        config, _, _, _ = build_algo_config(entry, shared)

        assert config.alpha == pytest.approx(0.03)

    def test_use_score_enforcement(self):
        """cost: langevin sets use_score=True even without explicit flag."""
        from experiments.run import build_algo_config

        entry = {
            "label": "LRET-B",
            "cost": {"type": "langevin"},
            "coupling": "balanced",
            "update": "categorical",
            "epsilon": 0.1,
        }
        shared = {"n_particles": 50, "n_iterations": 100}
        config, _, _, _ = build_algo_config(entry, shared)

        assert config.use_score is True

    def test_cost_params_contain_whiten(self):
        """cost: {type: langevin, whiten: true} → cost_params has whiten."""
        from experiments.run import build_algo_config

        entry = {
            "label": "LRET-B-W",
            "cost": {"type": "langevin", "whiten": True},
            "coupling": "balanced",
            "update": "categorical",
            "epsilon": 0.1,
        }
        shared = {"n_particles": 50, "n_iterations": 100}
        config, _, _, _ = build_algo_config(entry, shared)

        params_dict = dict(config.cost_params)
        assert params_dict.get("whiten") is True


# ===========================================================================
# 5c. Integration tests (GMM 2D)
# ===========================================================================

class TestLangevinIntegration:

    @pytest.mark.slow
    def test_lret_balanced_converges_gmm_2d(self, rng, gmm_2d_4):
        """LRET-B on GMM 2D 4 modes: all modes covered after 300 iters."""
        target = gmm_2d_4
        config = ETDConfig(
            n_particles=100,
            n_iterations=300,
            n_proposals=25,
            cost="langevin",
            coupling="balanced",
            update="categorical",
            use_score=True,
            epsilon=0.1,
            alpha=0.1,   # = epsilon for LRET
            fdr=True,
            score_clip=5.0,
        )

        k_init, k_run = jax.random.split(rng)
        state = etd_init(k_init, target, config)

        for i in range(1, 301):
            k_step = jax.random.fold_in(k_run, i)
            state, info = etd_step(k_step, state, target, config)

        # Check all 4 modes are covered
        positions = state.positions
        means = target.means  # (4, 2)
        for mode_idx in range(4):
            dists = jnp.sqrt(jnp.sum(
                (positions - means[mode_idx]) ** 2, axis=1,
            ))
            n_near = jnp.sum(dists < 2.0)
            assert n_near >= 3, (
                f"Mode {mode_idx} at {means[mode_idx]} has only "
                f"{int(n_near)} particles within r=2.0"
            )

    @pytest.mark.slow
    def test_lret_sdd_converges_gmm_2d(self, rng, gmm_2d_4):
        """LRET-SDD on GMM 2D 4 modes: cross-cost is Langevin."""
        from etd.extensions.sdd import (
            SDDConfig,
            init as sdd_init,
            step as sdd_step,
        )

        target = gmm_2d_4
        config = SDDConfig(
            n_particles=100,
            n_iterations=300,
            n_proposals=25,
            cost="langevin",
            use_score=True,
            epsilon=0.1,
            alpha=0.1,
            fdr=True,
            score_clip=5.0,
            sdd_step_size=0.5,
        )

        k_init, k_run = jax.random.split(rng)
        state = sdd_init(k_init, target, config)

        for i in range(1, 301):
            k_step = jax.random.fold_in(k_run, i)
            state, info = sdd_step(k_step, state, target, config)

        # Verify self-cost uses Euclidean (reported separately from cross)
        assert "cost_scale_self" in info
        assert "cost_scale_cross" in info

        # Check mode coverage
        positions = state.positions
        means = target.means
        modes_hit = 0
        for mode_idx in range(4):
            dists = jnp.sqrt(jnp.sum(
                (positions - means[mode_idx]) ** 2, axis=1,
            ))
            if jnp.sum(dists < 2.5) >= 3:
                modes_hit += 1
        assert modes_hit >= 3, f"Only {modes_hit}/4 modes covered"

    def test_lret_vs_etd_sanity(self, rng, gmm_2d_4):
        """LRET and ETD should both work on GMM 2D (sanity, not comparison).

        Quick test: 50 iterations, just verify no crashes and positions
        are finite.
        """
        target = gmm_2d_4

        for cost_name, alpha in [("langevin", 0.1), ("euclidean", 0.05)]:
            config = ETDConfig(
                n_particles=50,
                n_iterations=50,
                n_proposals=25,
                cost=cost_name,
                coupling="balanced",
                update="categorical",
                use_score=True,
                epsilon=0.1,
                alpha=alpha,
                fdr=True,
                score_clip=5.0,
            )
            k_init, k_run = jax.random.split(rng)
            state = etd_init(k_init, target, config)

            for i in range(1, 51):
                k_step = jax.random.fold_in(k_run, i)
                state, info = etd_step(k_step, state, target, config)

            assert jnp.all(jnp.isfinite(state.positions)), (
                f"Non-finite positions with cost={cost_name}"
            )

    def test_lret_etd_step_function_runs_with_scan(self, rng, gmm_2d_4):
        """Verify LRET works inside jax.lax.scan (JIT compatibility)."""
        target = gmm_2d_4
        config = ETDConfig(
            n_particles=50,
            n_iterations=10,
            n_proposals=15,
            cost="langevin",
            coupling="balanced",
            update="categorical",
            use_score=True,
            epsilon=0.1,
            alpha=0.1,
            fdr=True,
            score_clip=5.0,
        )

        k_init, k_run = jax.random.split(rng)
        state = etd_init(k_init, target, config)

        @jax.jit
        def scan_body(state, t):
            key_step = jax.random.fold_in(k_run, t)
            new_state, info = etd_step(key_step, state, target, config)
            return new_state, info

        final_state, _ = jax.lax.scan(scan_body, state, jnp.arange(10))
        assert jnp.all(jnp.isfinite(final_state.positions))
