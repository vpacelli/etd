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
    _resolve_mutation_config,
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
from etd.types import MutationConfig
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
        """energy_distance, mode_proximity, mode_balance compute without error."""
        particles = gmm_target.sample(jax.random.PRNGKey(0), 100)
        result = compute_metrics(
            particles, gmm_target,
            ["energy_distance", "mode_proximity", "mode_balance", "mean_error"],
            ref_data,
        )
        assert "energy_distance" in result
        assert "mode_proximity" in result
        assert "mode_balance" in result
        assert "mean_error" in result
        assert not np.isnan(result["energy_distance"])
        assert result["mode_proximity"] >= 0
        assert 0 <= result["mode_balance"] <= np.log(2) + 1e-6

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
            ["energy_distance", "mode_proximity"], ref_data,
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
                        100: {"energy_distance": 0.05, "mode_proximity": 1.0}
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
            assert loaded["seed0"]["ETD-B"]["100"]["mode_proximity"] == pytest.approx(1.0)

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


# ---------------------------------------------------------------------------
# Mutation config resolution
# ---------------------------------------------------------------------------

class TestResolveMutationConfig:
    """_resolve_mutation_config converts dict → MutationConfig."""

    def test_dict_to_mutation_config(self):
        kwargs = {
            "mutation": {
                "kernel": "mala",
                "n_steps": 5,
                "step_size": 0.01,
                "use_cholesky": True,
            },
        }
        _resolve_mutation_config(kwargs)
        mc = kwargs["mutation"]
        assert isinstance(mc, MutationConfig)
        assert mc.kernel == "mala"
        assert mc.n_steps == 5
        assert mc.step_size == 0.01
        assert mc.use_cholesky is True

    def test_build_algo_config_with_mutation(self):
        """ETD entry with mutation block → correct ETDConfig."""
        entry = {
            "label": "test",
            "coupling": "balanced",
            "epsilon": 0.1,
            "alpha": 0.05,
            "n_proposals": 10,
            "mutation": {
                "kernel": "mala",
                "n_steps": 3,
                "step_size": 0.02,
            },
        }
        shared = {"n_particles": 50, "n_iterations": 100}
        config, init_fn, step_fn, is_bl = build_algo_config(entry, shared)
        assert not is_bl
        assert config.mutation.kernel == "mala"
        assert config.mutation.n_steps == 3
        assert config.mutation.step_size == 0.02

    def test_mutation_default_off(self):
        """Entry without mutation → MutationConfig(kernel='none')."""
        entry = {
            "label": "test",
            "coupling": "balanced",
            "epsilon": 0.1,
            "alpha": 0.05,
            "n_proposals": 10,
        }
        shared = {"n_particles": 50, "n_iterations": 100}
        config, _, _, _ = build_algo_config(entry, shared)
        assert config.mutation.kernel == "none"
        assert not config.mutation.active


# ---------------------------------------------------------------------------
# Enabled flag
# ---------------------------------------------------------------------------

class TestEnabledFlag:
    """The ``enabled`` field controls whether an algorithm entry is included."""

    def test_enabled_default_true(self):
        """Missing ``enabled`` field defaults to True (entry included)."""
        entry = {"label": "ETD-B", "epsilon": 0.1}
        assert entry.get("enabled", True) is True

    def test_enabled_false_skips(self):
        """Entry with ``enabled: false`` is excluded from expansion."""
        entries = [
            {"label": "ETD-B", "epsilon": 0.1},
            {"label": "ULA", "type": "baseline", "method": "ula",
             "step_size": 0.01, "enabled": False},
        ]
        active = [e for e in entries if e.get("enabled", True)]
        assert len(active) == 1
        assert active[0]["label"] == "ETD-B"


# ---------------------------------------------------------------------------
# Sublabel
# ---------------------------------------------------------------------------

class TestSublabel:
    """Sublabel composes into label as ``"Label (sublabel)"``."""

    def test_sublabel_parenthetical(self):
        """Sublabel produces parenthetical display name."""
        base = "ETD-B"
        sublabel = "whitened"
        result = f"{base} ({sublabel})"
        assert result == "ETD-B (whitened)"

    def test_no_sublabel_unchanged(self):
        """Missing sublabel leaves label as-is."""
        entry = {"label": "ETD-B", "epsilon": 0.1}
        base = entry.get("label", "unnamed")
        sublabel = entry.get("sublabel")
        if sublabel:
            base = f"{base} ({sublabel})"
        assert base == "ETD-B"

    def test_sublabel_with_sweep(self):
        """Sweep suffix comes after sublabel: ``"ETD-B (wh)_eps=0.1"``."""
        original = {"label": "ETD-B", "sublabel": "wh", "epsilon": [0.1, 0.5]}
        expanded = expand_algo_sweeps(original)
        labels = []
        for concrete in expanded:
            base = concrete.get("label", "unnamed")
            sublabel = concrete.get("sublabel")
            if sublabel:
                base = f"{base} ({sublabel})"
            label = build_algo_label(base, concrete, original)
            labels.append(label)
        assert labels[0] == "ETD-B (wh)_eps=0.1"
        assert labels[1] == "ETD-B (wh)_eps=0.5"


# ---------------------------------------------------------------------------
# Display metadata filtering
# ---------------------------------------------------------------------------

class TestDisplayMetaFiltering:
    """Display/sublabel/enabled fields must not leak into algorithm configs."""

    def test_display_excluded_from_etd_config(self):
        """``display`` block doesn't leak into ETDConfig."""
        entry = {
            "label": "ETD-B",
            "coupling": "balanced",
            "epsilon": 0.1,
            "alpha": 0.05,
            "display": {"family": "etd", "group": "transport"},
        }
        shared = {"n_particles": 50, "n_iterations": 100}
        config, _, _, is_bl = build_algo_config(entry, shared)
        assert not is_bl
        assert not hasattr(config, "display")

    def test_display_excluded_from_baseline_config(self):
        """``display`` block doesn't leak into baseline configs."""
        entry = {
            "label": "SVGD",
            "type": "baseline",
            "method": "svgd",
            "learning_rate": 0.1,
            "display": {"linestyle": "--"},
        }
        shared = {"n_particles": 50, "n_iterations": 100}
        config, _, _, is_bl = build_algo_config(entry, shared)
        assert is_bl
        assert not hasattr(config, "display")

    def test_sublabel_excluded_from_config(self):
        """``sublabel`` doesn't leak into ETDConfig."""
        entry = {
            "label": "ETD-B",
            "sublabel": "whitened",
            "coupling": "balanced",
            "epsilon": 0.1,
            "alpha": 0.05,
        }
        shared = {"n_particles": 50, "n_iterations": 100}
        config, _, _, _ = build_algo_config(entry, shared)
        assert not hasattr(config, "sublabel")


# ---------------------------------------------------------------------------
# Display not swept
# ---------------------------------------------------------------------------

class TestDisplayNotSwept:
    """``display`` block must not create sweep combinations."""

    def test_display_dict_not_expanded(self):
        """Display dict is not treated as a nested sweep axis."""
        entry = {
            "label": "ETD-B",
            "epsilon": 0.1,
            "display": {"family": "etd", "color": "#DC143C"},
        }
        expanded = expand_algo_sweeps(entry)
        assert len(expanded) == 1

    def test_display_preserved_in_expansion(self):
        """Display block is copied to all expanded entries."""
        entry = {
            "label": "ETD-B",
            "epsilon": [0.1, 0.5],
            "display": {"family": "etd", "group": "transport"},
        }
        expanded = expand_algo_sweeps(entry)
        assert len(expanded) == 2
        for e in expanded:
            assert e["display"]["family"] == "etd"
            assert e["display"]["group"] == "transport"


# ---------------------------------------------------------------------------
# resolve_algo_styles
# ---------------------------------------------------------------------------

class TestResolveAlgoStyles:
    """Tests for the family-palette color resolver."""

    def test_family_inference_etd(self):
        """ETD prefix → etd family."""
        from figures.style import _infer_family
        assert _infer_family({"label": "ETD-B"}) == "etd"

    def test_family_inference_lret(self):
        """LRET prefix → etd family."""
        from figures.style import _infer_family
        assert _infer_family({"label": "LRET-B"}) == "etd"

    def test_family_inference_baseline(self):
        """is_baseline flag → baseline family."""
        from figures.style import _infer_family
        assert _infer_family({"label": "SVGD", "is_baseline": True}) == "baseline"

    def test_explicit_family_overrides(self):
        """Explicit ``family`` takes precedence over label prefix."""
        from figures.style import _infer_family
        assert _infer_family({"label": "ETD-B", "family": "custom"}) == "custom"

    def test_explicit_color_overrides_palette(self):
        """Explicit ``color`` bypasses palette assignment."""
        from figures.style import resolve_algo_styles
        meta = [{"label": "X", "color": "#FF0000", "is_baseline": False}]
        result = resolve_algo_styles(meta)
        assert result["X"]["color"] == "#FF0000"

    def test_algo_colors_fallback(self):
        """Known ALGO_COLORS label gets its registered color."""
        from figures.style import resolve_algo_styles, ALGO_COLORS
        meta = [{"label": "SVGD", "is_baseline": True}]
        result = resolve_algo_styles(meta)
        assert result["SVGD"]["color"] == ALGO_COLORS["SVGD"]

    def test_yaml_order_palette_assignment(self):
        """Two etd-family algos get consecutive palette colors."""
        from figures.style import resolve_algo_styles, FAMILY_PALETTES
        meta = [
            {"label": "AlgoA", "is_baseline": False},
            {"label": "AlgoB", "is_baseline": False},
        ]
        result = resolve_algo_styles(meta)
        assert result["AlgoA"]["color"] == FAMILY_PALETTES["etd"][0]
        assert result["AlgoB"]["color"] == FAMILY_PALETTES["etd"][1]

    def test_linestyle_default_solid(self):
        """Default linestyle is ``"-"``."""
        from figures.style import resolve_algo_styles
        meta = [{"label": "X", "is_baseline": False}]
        result = resolve_algo_styles(meta)
        assert result["X"]["linestyle"] == "-"

    def test_group_preserved(self):
        """Group string passes through unchanged."""
        from figures.style import resolve_algo_styles
        meta = [{"label": "X", "group": "transport", "is_baseline": False}]
        result = resolve_algo_styles(meta)
        assert result["X"]["group"] == "transport"

    def test_legend_order_matches_yaml_order(self):
        """Dict keys preserve YAML insertion order."""
        from figures.style import resolve_algo_styles
        meta = [
            {"label": "C", "is_baseline": False},
            {"label": "A", "is_baseline": True},
            {"label": "B", "is_baseline": False},
        ]
        result = resolve_algo_styles(meta)
        assert list(result.keys()) == ["C", "A", "B"]


# ---------------------------------------------------------------------------
# Metadata save / load
# ---------------------------------------------------------------------------

class TestMetadataSave:
    """Tests for metadata.json persistence."""

    def test_metadata_json_saved(self):
        """Round-trip: save + load metadata.json."""
        from figures.style import load_display_metadata
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = {"experiment": {"name": "test"}}
            all_metrics = {0: {"X": {10: {"ed": 0.1}}}}
            all_particles = {0: {"X": {10: np.random.randn(5, 2)}}}
            display_meta = {
                "X": {"family": "etd", "color": "#DC143C",
                       "linestyle": "-", "group": None},
            }
            save_results(
                tmpdir, cfg, all_metrics, all_particles,
                display_metadata=display_meta,
            )
            assert os.path.exists(os.path.join(tmpdir, "metadata.json"))
            loaded = load_display_metadata(tmpdir)
            assert loaded["X"]["color"] == "#DC143C"
            assert loaded["X"]["family"] == "etd"

    def test_metadata_json_not_saved_when_none(self):
        """No metadata.json when display_metadata is None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = {"experiment": {"name": "test"}}
            all_metrics = {0: {"X": {10: {"ed": 0.1}}}}
            all_particles = {0: {"X": {10: np.random.randn(5, 2)}}}
            save_results(tmpdir, cfg, all_metrics, all_particles)
            assert not os.path.exists(os.path.join(tmpdir, "metadata.json"))
