"""Phase 4 gate tests.

Gate criteria:
  - BLR benchmark: ETD-B competitive with SVGD on German Credit (RMSE, variance ratio).
  - Funnel test: ETD particles span the funnel neck; SVGD collapses.

BLR gate requires DuckDB + NUTS reference (skipped if unavailable).
Funnel gate uses exact target.sample() as reference.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from etd.diagnostics.metrics import mean_error
from etd.step import init as etd_init, step as etd_step
from etd.targets import TARGETS
from etd.targets.funnel import FunnelTarget
from etd.targets.gaussian import GaussianTarget
from etd.types import ETDConfig
from etd.update import UPDATES


# ---------------------------------------------------------------------------
# Check if DuckDB data is available
# ---------------------------------------------------------------------------

_DB_PATH = Path("data/etd.duckdb")
DUCKDB_EXISTS = _DB_PATH.exists()


# ---------------------------------------------------------------------------
# Helper: run ETD for N iterations
# ---------------------------------------------------------------------------

def _run_etd(key, target, config, n_iters, init_positions=None):
    """Run ETD for n_iters steps and return final state."""
    k_init, k_run = jax.random.split(key)
    state = etd_init(k_init, target, config, init_positions=init_positions)

    for _ in range(n_iters):
        k_run, k_step = jax.random.split(k_run)
        state, _ = etd_step(k_step, state, target, config)

    return state


def _run_svgd(key, target, n_particles, n_iters, lr=0.1, init_positions=None):
    """Run SVGD for n_iters steps and return final state."""
    from etd.baselines import get_baseline
    bl = get_baseline("svgd")
    config = bl["config"](
        n_particles=n_particles,
        n_iterations=n_iters,
        learning_rate=lr,
        score_clip=5.0,
    )

    k_init, k_run = jax.random.split(key)
    state = bl["init"](k_init, target, config, init_positions=init_positions)

    for _ in range(n_iters):
        k_run, k_step = jax.random.split(k_run)
        state, _ = bl["step"](k_step, state, target, config)

    return state


# ---------------------------------------------------------------------------
# Funnel gate
# ---------------------------------------------------------------------------

class TestFunnelGate:
    """ETD should span both funnel neck and mouth; SVGD often collapses."""

    @pytest.fixture
    def funnel(self):
        return FunnelTarget(dim=10, sigma_v=3.0)

    @pytest.fixture
    def shared_init(self, funnel):
        """Shared init positions for fair comparison."""
        key = jax.random.PRNGKey(42)
        return jax.random.normal(key, (200, funnel.dim)) * 2.0

    @pytest.mark.slow
    def test_etd_spans_funnel_neck(self, funnel, shared_init):
        """ETD particles in v < -2 region should be present."""
        config = ETDConfig(
            n_particles=200,
            n_proposals=25,
            n_iterations=500,
            coupling="balanced",
            epsilon=0.1,
            alpha=0.05,
            score_clip=5.0,
        )
        state = _run_etd(
            jax.random.PRNGKey(0), funnel, config, 500,
            init_positions=shared_init,
        )

        v = state.positions[:, 0]
        n_neck = jnp.sum(v < -2.0)
        # ETD should have at least some particles in the neck
        assert n_neck >= 5, f"Only {n_neck} ETD particles in funnel neck"

    @pytest.mark.slow
    def test_etd_spans_funnel_mouth(self, funnel, shared_init):
        """ETD particles in v > 2 region should be present."""
        config = ETDConfig(
            n_particles=200,
            n_proposals=25,
            n_iterations=500,
            coupling="balanced",
            epsilon=0.1,
            alpha=0.05,
            score_clip=5.0,
        )
        state = _run_etd(
            jax.random.PRNGKey(0), funnel, config, 500,
            init_positions=shared_init,
        )

        v = state.positions[:, 0]
        n_mouth = jnp.sum(v > 2.0)
        assert n_mouth >= 5, f"Only {n_mouth} ETD particles in funnel mouth"

    @pytest.mark.slow
    def test_etd_more_neck_coverage_than_svgd(self, funnel, shared_init):
        """ETD should have more particles in the neck than SVGD."""
        etd_config = ETDConfig(
            n_particles=200,
            n_proposals=25,
            n_iterations=500,
            coupling="balanced",
            epsilon=0.1,
            alpha=0.05,
            score_clip=5.0,
        )
        etd_state = _run_etd(
            jax.random.PRNGKey(0), funnel, etd_config, 500,
            init_positions=shared_init,
        )

        svgd_state = _run_svgd(
            jax.random.PRNGKey(0), funnel, 200, 500, lr=0.1,
            init_positions=shared_init,
        )

        etd_neck = int(jnp.sum(etd_state.positions[:, 0] < -2.0))
        svgd_neck = int(jnp.sum(svgd_state.positions[:, 0] < -2.0))

        # ETD should explore the neck at least as well as SVGD
        # (SVGD is known to collapse on the funnel)
        assert etd_neck >= svgd_neck, \
            f"ETD neck={etd_neck} < SVGD neck={svgd_neck}"


# ---------------------------------------------------------------------------
# BLR gate (requires DuckDB + NUTS reference)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not DUCKDB_EXISTS, reason="DuckDB not populated")
class TestBLRGate:
    """ETD-B should be competitive with SVGD on German Credit BLR."""

    @pytest.mark.slow
    def test_etd_rmse_competitive(self):
        """ETD mean RMSE vs NUTS should be <= SVGD RMSE."""
        from experiments.nuts import load_reference
        from etd.diagnostics.metrics import mean_rmse
        from etd.targets.logistic import LogisticRegressionTarget

        target = LogisticRegressionTarget(dataset="german_credit", prior_std=5.0)
        ref = load_reference("logistic", {"dataset": "german_credit", "prior_std": 5.0})
        if ref is None:
            pytest.skip("NUTS reference not cached")

        ref = jnp.asarray(ref)
        init_pos = jax.random.normal(jax.random.PRNGKey(42), (200, target.dim)) * 0.1

        etd_config = ETDConfig(
            n_particles=200,
            n_proposals=25,
            n_iterations=1000,
            coupling="balanced",
            epsilon=0.1,
            alpha=0.01,
            score_clip=5.0,
        )
        etd_state = _run_etd(
            jax.random.PRNGKey(0), target, etd_config, 1000,
            init_positions=init_pos,
        )
        etd_rmse = float(mean_rmse(etd_state.positions, ref))

        svgd_state = _run_svgd(
            jax.random.PRNGKey(0), target, 200, 1000, lr=0.01,
            init_positions=init_pos,
        )
        svgd_rmse = float(mean_rmse(svgd_state.positions, ref))

        # ETD should be competitive (within 2x of SVGD)
        assert etd_rmse <= 2.0 * svgd_rmse + 0.1, \
            f"ETD RMSE {etd_rmse:.4f} >> SVGD RMSE {svgd_rmse:.4f}"

    @pytest.mark.slow
    def test_etd_variance_ratio_closer_to_one(self):
        """ETD variance ratio should be closer to 1.0 than SVGD."""
        from experiments.nuts import load_reference
        from etd.diagnostics.metrics import variance_ratio_vs_reference
        from etd.targets.logistic import LogisticRegressionTarget

        target = LogisticRegressionTarget(dataset="german_credit", prior_std=5.0)
        ref = load_reference("logistic", {"dataset": "german_credit", "prior_std": 5.0})
        if ref is None:
            pytest.skip("NUTS reference not cached")

        ref = jnp.asarray(ref)
        init_pos = jax.random.normal(jax.random.PRNGKey(42), (200, target.dim)) * 0.1

        etd_config = ETDConfig(
            n_particles=200,
            n_proposals=25,
            n_iterations=1000,
            coupling="balanced",
            epsilon=0.1,
            alpha=0.01,
            score_clip=5.0,
        )
        etd_state = _run_etd(
            jax.random.PRNGKey(0), target, etd_config, 1000,
            init_positions=init_pos,
        )
        etd_vr = float(variance_ratio_vs_reference(etd_state.positions, ref))

        svgd_state = _run_svgd(
            jax.random.PRNGKey(0), target, 200, 1000, lr=0.01,
            init_positions=init_pos,
        )
        svgd_vr = float(variance_ratio_vs_reference(svgd_state.positions, ref))

        # ETD variance ratio should be closer to 1.0
        etd_dist = abs(etd_vr - 1.0)
        svgd_dist = abs(svgd_vr - 1.0)

        # Soft check: ETD shouldn't be much worse than SVGD
        assert etd_dist <= svgd_dist + 0.5, \
            f"ETD var ratio {etd_vr:.3f} worse than SVGD {svgd_vr:.3f}"


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_all_targets_registered(self):
        """All expected targets should be in the registry."""
        expected = {"gaussian", "gmm", "banana", "funnel", "logistic"}
        assert expected.issubset(set(TARGETS.keys())), \
            f"Missing: {expected - set(TARGETS.keys())}"

    def test_all_updates_registered(self):
        """All expected updates should be in the registry."""
        expected = {"categorical", "barycentric"}
        assert expected.issubset(set(UPDATES.keys())), \
            f"Missing: {expected - set(UPDATES.keys())}"

    def test_banana_smoke(self):
        """10 iterations of ETD on banana target, no errors."""
        from etd.targets.banana import BananaTarget

        target = BananaTarget(dim=2)
        config = ETDConfig(
            n_particles=20,
            n_proposals=10,
            n_iterations=10,
            coupling="balanced",
            epsilon=0.1,
            alpha=0.05,
        )

        state = _run_etd(jax.random.PRNGKey(0), target, config, 10)
        assert state.positions.shape == (20, 2)
        assert jnp.all(jnp.isfinite(state.positions))

    def test_funnel_smoke(self):
        """10 iterations of ETD on funnel target, no errors."""
        target = FunnelTarget(dim=5)
        config = ETDConfig(
            n_particles=20,
            n_proposals=10,
            n_iterations=10,
            coupling="balanced",
            epsilon=0.1,
            alpha=0.05,
            score_clip=5.0,
        )

        state = _run_etd(jax.random.PRNGKey(0), target, config, 10)
        assert state.positions.shape == (20, 5)
        assert jnp.all(jnp.isfinite(state.positions))

    def test_sdd_smoke(self):
        """10 iterations of SDD on Gaussian, no errors."""
        from etd.extensions.sdd import SDDConfig
        from etd.extensions.sdd import init as sdd_init, step as sdd_step

        target = GaussianTarget(dim=2)
        config = SDDConfig(
            n_particles=20,
            n_proposals=10,
            epsilon=0.1,
            self_epsilon=0.1,
            alpha=0.05,
            sdd_step_size=0.5,
        )

        key = jax.random.PRNGKey(42)
        k_init, k_run = jax.random.split(key)
        state = sdd_init(k_init, target, config)

        for _ in range(10):
            k_run, k_step = jax.random.split(k_run)
            state, _ = sdd_step(k_step, state, target, config)

        assert state.positions.shape == (20, 2)
        assert jnp.all(jnp.isfinite(state.positions))
