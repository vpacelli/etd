"""Tests for NUTS reference sampler."""

import tempfile
from pathlib import Path
from unittest import mock

import jax.numpy as jnp
import numpy as np
import pytest

from etd.targets.gaussian import GaussianTarget


# ---------------------------------------------------------------------------
# NUTS run (small scale for speed)
# ---------------------------------------------------------------------------

class TestRunNuts:
    @pytest.mark.slow
    def test_gaussian_shapes(self):
        """run_nuts on Gaussian(d=2), 500 samples — correct shapes."""
        from experiments.nuts import run_nuts

        target = GaussianTarget(dim=2)
        result = run_nuts(target, n_samples=500, n_warmup=200, n_chains=2, seed=0)

        assert result["samples"].shape == (1000, 2)  # 2 chains × 500
        assert result["r_hat"].shape == (2,)
        assert result["ess"].shape == (2,)
        assert isinstance(result["n_divergent"], int)

    @pytest.mark.slow
    def test_gaussian_r_hat(self):
        """R-hat should be close to 1 for a well-behaved target."""
        from experiments.nuts import run_nuts

        target = GaussianTarget(dim=2)
        result = run_nuts(target, n_samples=500, n_warmup=200, n_chains=2, seed=42)

        assert np.max(result["r_hat"]) < 1.05, \
            f"R-hat too high: {result['r_hat']}"


# ---------------------------------------------------------------------------
# Convergence check
# ---------------------------------------------------------------------------

class TestCheckConvergence:
    def test_good_result_no_warnings(self):
        from experiments.nuts import check_convergence

        result = {
            "samples": np.zeros((4000, 5)),
            "r_hat": np.array([1.001, 1.002, 1.003, 1.004, 1.005]),
            "ess": np.array([500, 600, 700, 800, 900]),
            "n_divergent": 0,
        }
        warnings = check_convergence(result)
        assert warnings == []

    def test_bad_r_hat_warns(self):
        from experiments.nuts import check_convergence

        result = {
            "samples": np.zeros((4000, 2)),
            "r_hat": np.array([1.05, 1.001]),
            "ess": np.array([1000, 1000]),
            "n_divergent": 0,
        }
        warnings = check_convergence(result)
        assert len(warnings) == 1
        assert "R-hat" in warnings[0]

    def test_bad_ess_warns(self):
        from experiments.nuts import check_convergence

        result = {
            "samples": np.zeros((4000, 2)),
            "r_hat": np.array([1.001, 1.001]),
            "ess": np.array([50, 1000]),
            "n_divergent": 0,
        }
        warnings = check_convergence(result)
        assert len(warnings) == 1
        assert "ESS" in warnings[0]

    def test_divergences_warns(self):
        from experiments.nuts import check_convergence

        result = {
            "samples": np.zeros((1000, 2)),
            "r_hat": np.array([1.001, 1.001]),
            "ess": np.array([500, 500]),
            "n_divergent": 50,  # 5% of 1000
        }
        warnings = check_convergence(result)
        assert len(warnings) == 1
        assert "divergent" in warnings[0]


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        from experiments.nuts import save_reference, load_reference

        with mock.patch("experiments.nuts.REFERENCE_DIR", tmp_path):
            result = {
                "samples": np.random.randn(100, 3),
                "r_hat": np.array([1.001, 1.002, 1.003]),
                "ess": np.array([500, 600, 700]),
                "n_divergent": 0,
            }

            path = save_reference("gaussian", {"dim": 3}, result)
            assert Path(path).exists()

            loaded = load_reference("gaussian", {"dim": 3})
            np.testing.assert_array_equal(loaded, result["samples"])

    def test_load_missing_returns_none(self, tmp_path):
        from experiments.nuts import load_reference

        with mock.patch("experiments.nuts.REFERENCE_DIR", tmp_path):
            assert load_reference("nonexistent", {"dim": 2}) is None


# ---------------------------------------------------------------------------
# Hash determinism
# ---------------------------------------------------------------------------

class TestHash:
    def test_deterministic(self):
        from experiments.nuts import _target_hash

        h1 = _target_hash("gaussian", {"dim": 2, "condition_number": 1.0})
        h2 = _target_hash("gaussian", {"dim": 2, "condition_number": 1.0})
        assert h1 == h2

    def test_order_invariant(self):
        from experiments.nuts import _target_hash

        h1 = _target_hash("gaussian", {"dim": 2, "a": 1})
        h2 = _target_hash("gaussian", {"a": 1, "dim": 2})
        assert h1 == h2

    def test_different_params_different_hash(self):
        from experiments.nuts import _target_hash

        h1 = _target_hash("gaussian", {"dim": 2})
        h2 = _target_hash("gaussian", {"dim": 3})
        assert h1 != h2


# ---------------------------------------------------------------------------
# New metrics
# ---------------------------------------------------------------------------

class TestNewMetrics:
    def test_mean_rmse(self):
        from etd.diagnostics.metrics import mean_rmse

        particles = jnp.zeros((100, 3))
        reference = jnp.ones((200, 3))

        rmse = mean_rmse(particles, reference)
        # particle mean = 0, ref mean = 1 → rmse = 1.0
        np.testing.assert_allclose(rmse, 1.0, atol=0.01)

    def test_variance_ratio_vs_reference(self):
        from etd.diagnostics.metrics import variance_ratio_vs_reference

        key = jnp.array([0, 1], dtype=jnp.uint32)
        particles = np.random.default_rng(0).standard_normal((500, 3)).astype(np.float32)
        reference = np.random.default_rng(1).standard_normal((1000, 3)).astype(np.float32)

        ratio = variance_ratio_vs_reference(
            jnp.asarray(particles), jnp.asarray(reference),
        )
        # Both are standard normals → ratio should be near 1
        assert 0.7 < float(ratio) < 1.3
