"""Tests for the experiment runner and tuner.

Covers config loading, sweep expansion, algo config building,
metric computation, single-run execution, and results I/O.
"""

import json
import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from experiments.run import (
    _compute_segments,
    build_algo_config,
    build_algo_label,
    compute_metrics,
    expand_algo_sweeps,
    get_reference_data,
    load_config,
    make_init_positions,
    run_single,
    save_results,
    maybe_jit,
)
from etd.targets import get_target
from etd.targets.gmm import GMMTarget
from etd.types import ETDConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gmm_target():
    return GMMTarget(dim=2, n_modes=4, arrangement="grid", separation=6.0)


@pytest.fixture
def ref_data(gmm_target):
    key = jax.random.PRNGKey(99999)
    return gmm_target.sample(key, 1000)


# ---------------------------------------------------------------------------
# Sweep expansion
# ---------------------------------------------------------------------------

class TestExpandSweeps:
    def test_no_lists(self):
        """No list params → returns [entry] unchanged."""
        entry = {"label": "ETD-B", "epsilon": 0.1, "alpha": 0.05}
        result = expand_algo_sweeps(entry)
        assert len(result) == 1
        assert result[0] == entry

    def test_single_list(self):
        """Single list param → len(list) entries."""
        entry = {"label": "ETD-B", "epsilon": [0.1, 0.2, 0.3]}
        result = expand_algo_sweeps(entry)
        assert len(result) == 3
        assert result[0]["epsilon"] == 0.1
        assert result[2]["epsilon"] == 0.3

    def test_cartesian_product(self):
        """Two list params → Cartesian product."""
        entry = {
            "label": "ETD-B",
            "epsilon": [0.1, 0.2],
            "alpha": [0.01, 0.02],
        }
        result = expand_algo_sweeps(entry)
        assert len(result) == 4


# ---------------------------------------------------------------------------
# Label building
# ---------------------------------------------------------------------------

class TestBuildLabel:
    def test_no_sweep(self):
        """No list in original → label unchanged."""
        entry = {"label": "ETD-B", "epsilon": 0.1}
        label = build_algo_label("ETD-B", entry, entry)
        assert label == "ETD-B"

    def test_with_sweep(self):
        """Sweep suffix appended correctly."""
        original = {"label": "ETD-B", "epsilon": [0.1, 0.2]}
        concrete = {"label": "ETD-B", "epsilon": 0.1}
        label = build_algo_label("ETD-B", concrete, original)
        assert "eps=0.1" in label


# ---------------------------------------------------------------------------
# Config building
# ---------------------------------------------------------------------------

class TestBuildConfig:
    def test_etd_config(self):
        """Dict → ETDConfig with correct fields."""
        entry = {
            "label": "ETD-B",
            "cost": "euclidean",
            "coupling": "balanced",
            "update": "categorical",
            "epsilon": 0.1,
            "alpha": 0.05,
            "use_score": True,
        }
        shared = {"n_particles": 50, "n_iterations": 100}
        config, init_fn, step_fn, is_bl = build_algo_config(entry, shared)

        assert isinstance(config, ETDConfig)
        assert config.n_particles == 50
        assert config.epsilon == 0.1
        assert not is_bl

    def test_baseline_config(self):
        """Dict with type=baseline → correct baseline config."""
        entry = {
            "label": "SVGD",
            "type": "baseline",
            "method": "svgd",
            "learning_rate": 0.1,
        }
        shared = {"n_particles": 50, "n_iterations": 100}
        config, init_fn, step_fn, is_bl = build_algo_config(entry, shared)

        assert is_bl
        assert config.n_particles == 50
        assert config.learning_rate == 0.1


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_gmm_metrics(self, gmm_target, ref_data):
        """energy_distance and mode_coverage compute without error."""
        particles = gmm_target.sample(jax.random.PRNGKey(0), 100)
        result = compute_metrics(
            particles, gmm_target,
            ["energy_distance", "mode_coverage", "mean_error"],
            ref_data,
        )
        assert "energy_distance" in result
        assert "mode_coverage" in result
        assert "mean_error" in result
        assert not np.isnan(result["energy_distance"])
        assert 0 <= result["mode_coverage"] <= 1

    def test_unknown_metric(self, gmm_target, ref_data):
        """Unknown metric → NaN."""
        particles = jnp.zeros((10, 2))
        result = compute_metrics(
            particles, gmm_target, ["nonexistent"], ref_data,
        )
        assert np.isnan(result["nonexistent"])


# ---------------------------------------------------------------------------
# Run single
# ---------------------------------------------------------------------------

class TestRunSingle:
    def test_etd_run(self, gmm_target, ref_data):
        """10-iteration ETD run returns correct dict structure."""
        from etd.step import init as etd_init, step as etd_step

        config = ETDConfig(
            n_particles=20, n_iterations=10, n_proposals=10,
            epsilon=0.1, alpha=0.05,
        )
        key = jax.random.PRNGKey(0)
        init_pos = jax.random.normal(key, (20, 2)) * 2.0

        m, p, wc = run_single(
            key, gmm_target, config, etd_init, etd_step, False,
            init_pos, 10, [5, 10],
            ["energy_distance", "mode_coverage"], ref_data,
        )

        assert 5 in m and 10 in m
        assert 5 in p and 10 in p
        assert p[10].shape == (20, 2)
        assert "energy_distance" in m[10]
        assert wc > 0

    def test_svgd_run(self, gmm_target, ref_data):
        """10-iteration SVGD run returns correct dict structure."""
        from etd.baselines.svgd import SVGDConfig, init, step

        config = SVGDConfig(n_particles=20, n_iterations=10, learning_rate=0.1)
        key = jax.random.PRNGKey(0)
        init_pos = jax.random.normal(key, (20, 2)) * 2.0

        m, p, wc = run_single(
            key, gmm_target, config, init, step, True,
            init_pos, 10, [10],
            ["energy_distance"], ref_data,
        )

        assert 10 in m
        assert 10 in p
        assert p[10].shape == (20, 2)
        assert wc > 0


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_roundtrip(self, gmm_target, ref_data):
        """Round-trip: save → load metrics.json matches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = {"experiment": {"name": "test"}}
            all_metrics = {
                0: {
                    "ETD-B": {
                        100: {"energy_distance": 0.05, "mode_coverage": 1.0}
                    }
                }
            }
            all_particles = {
                0: {
                    "ETD-B": {100: np.random.randn(20, 2)}
                }
            }

            save_results(tmpdir, cfg, all_metrics, all_particles)

            # Check files exist
            assert os.path.exists(os.path.join(tmpdir, "config.yaml"))
            assert os.path.exists(os.path.join(tmpdir, "metrics.json"))
            assert os.path.exists(os.path.join(tmpdir, "particles.npz"))

            # Check metrics roundtrip
            with open(os.path.join(tmpdir, "metrics.json")) as f:
                loaded = json.load(f)

            assert loaded["seed0"]["ETD-B"]["100"]["energy_distance"] == pytest.approx(0.05)
            assert loaded["seed0"]["ETD-B"]["100"]["mode_coverage"] == pytest.approx(1.0)

            # Check particles roundtrip
            particles = np.load(os.path.join(tmpdir, "particles.npz"))
            key = "seed0__ETD-B__iter100"
            assert key in particles
            assert particles[key].shape == (20, 2)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_load_gmm_config(self):
        """Load the main GMM config file."""
        cfg = load_config("experiments/configs/gmm_2d_4.yaml")
        assert cfg["experiment"]["name"] == "gmm-2d-4"
        assert len(cfg["experiment"]["seeds"]) == 5
        assert len(cfg["experiment"]["algorithms"]) == 7


# ---------------------------------------------------------------------------
# Scan runner segments
# ---------------------------------------------------------------------------

class TestComputeSegments:
    def test_basic(self):
        """Two checkpoints produce two contiguous segments."""
        segs = _compute_segments([5, 10], n_iterations=10)
        assert segs == [(1, 5, 5), (6, 5, 10)]

    def test_single_checkpoint(self):
        """Single checkpoint at the end."""
        segs = _compute_segments([100], n_iterations=100)
        assert segs == [(1, 100, 100)]

    def test_checkpoint_zero_filtered(self):
        """Checkpoint 0 is filtered out (handled separately)."""
        segs = _compute_segments([0, 50], n_iterations=50)
        assert segs == [(1, 50, 50)]

    def test_beyond_n_iter_dropped(self):
        """Checkpoints beyond n_iterations are dropped."""
        segs = _compute_segments([5, 10, 200], n_iterations=10)
        assert segs == [(1, 5, 5), (6, 5, 10)]

    def test_empty_checkpoints(self):
        """Empty checkpoint list returns empty segments."""
        segs = _compute_segments([], n_iterations=100)
        assert segs == []

    def test_contiguity(self):
        """Segments are contiguous: sum of n_steps == last checkpoint."""
        ckpts = [50, 100, 200, 500]
        segs = _compute_segments(ckpts, n_iterations=500)
        total_steps = sum(n for _, n, _ in segs)
        assert total_steps == 500
        # Verify contiguity
        for i, (start, n, ckpt) in enumerate(segs):
            assert start + n - 1 == ckpt
            if i > 0:
                prev_end = segs[i - 1][2]
                assert start == prev_end + 1

    def test_unsorted_input(self):
        """Unsorted checkpoints are handled correctly."""
        segs = _compute_segments([10, 5], n_iterations=10)
        assert segs == [(1, 5, 5), (6, 5, 10)]
