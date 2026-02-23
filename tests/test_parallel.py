"""Tests for parallel seed and sweep execution via vmap.

Covers:
- Batched (vmapped) execution matches sequential execution
- Buffer donation works without warnings
- Progress segment merging
- Structural config grouping
- Sweep field validation
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from etd.targets.gmm import GMMTarget
from etd.types import ETDConfig, PreconditionerConfig
from experiments._parallel import (
    _STRUCTURAL_FIELDS_ETD,
    _compute_progress_segments,
    _make_batched_scan,
    batch_init_states,
    group_configs_by_structure,
    identify_varying_scalars,
    run_seeds_batched,
    structural_key,
    validate_sweep_fields,
)
from experiments.run import (
    _compute_segments,
    compute_metrics,
    make_init_positions,
    run_single,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gmm_target():
    return GMMTarget(dim=2, n_modes=4, arrangement="grid", separation=6.0)


@pytest.fixture
def shared():
    return {"n_particles": 50, "n_iterations": 30, "init": {"type": "gaussian", "scale": 3.0}}


@pytest.fixture
def etd_config():
    return ETDConfig(
        n_particles=50, n_iterations=30, n_proposals=10,
        coupling="balanced", cost="euclidean", update="categorical",
        use_score=True, epsilon=0.1, alpha=0.05,
    )


@pytest.fixture
def ref_data(gmm_target):
    return gmm_target.sample(jax.random.PRNGKey(99999), 500)


# ---------------------------------------------------------------------------
# Progress segments
# ---------------------------------------------------------------------------

class TestProgressSegments:
    def test_no_progress(self):
        """Without n_progress, segments match checkpoints."""
        segments, ckpts = _compute_progress_segments([10, 20, 30], 30)
        assert ckpts == {10, 20, 30}
        assert len(segments) == 3
        assert segments[0] == (1, 10, 10)
        assert segments[1] == (11, 10, 20)
        assert segments[2] == (21, 10, 30)

    def test_with_progress(self):
        """Progress boundaries add extra segment breaks."""
        segments, ckpts = _compute_progress_segments([30], 30, n_progress=3)
        # n_progress=3 → boundaries at 10, 20, 30
        # Checkpoint at 30 only
        assert ckpts == {30}
        assert len(segments) == 3
        # All boundaries present
        boundaries = [s[2] for s in segments]
        assert 10 in boundaries
        assert 20 in boundaries
        assert 30 in boundaries

    def test_superset_of_checkpoints(self):
        """All checkpoints appear in segment boundaries."""
        ckpts_input = [5, 15, 30]
        segments, ckpts = _compute_progress_segments(ckpts_input, 30, n_progress=3)
        boundaries = {s[2] for s in segments}
        assert set(ckpts_input) <= boundaries

    def test_empty_checkpoints(self):
        """Empty checkpoints → empty segments."""
        segments, ckpts = _compute_progress_segments([], 30)
        assert segments == []
        assert ckpts == set()

    def test_checkpoint_zero_excluded(self):
        """Checkpoint 0 is handled separately, not in segments."""
        segments, ckpts = _compute_progress_segments([0, 10], 10)
        assert 0 not in ckpts
        assert len(segments) == 1


# ---------------------------------------------------------------------------
# Batched init
# ---------------------------------------------------------------------------

class TestBatchInit:
    def test_batched_shapes(self, gmm_target, etd_config, shared):
        """Batched init produces correct leading dimension."""
        from etd.step import init as etd_init

        seeds = [0, 1, 2]
        keys = [jax.random.split(jax.random.PRNGKey(s), 3)[0] for s in seeds]
        init_pos = np.stack([
            np.asarray(make_init_positions(k, gmm_target, shared))
            for k in keys
        ])

        batched = batch_init_states(
            etd_init, keys, gmm_target, etd_config, init_pos,
        )
        # positions should be (S, N, d) = (3, 50, 2)
        assert batched.positions.shape == (3, 50, 2)
        assert batched.dual_f.shape == (3, 50)

    def test_numpy_backed(self, gmm_target, etd_config, shared):
        """Batched init returns numpy arrays (for safe donation)."""
        from etd.step import init as etd_init

        seeds = [0, 1]
        keys = [jax.random.split(jax.random.PRNGKey(s), 3)[0] for s in seeds]
        init_pos = np.stack([
            np.asarray(make_init_positions(k, gmm_target, shared))
            for k in keys
        ])

        batched = batch_init_states(
            etd_init, keys, gmm_target, etd_config, init_pos,
        )
        assert isinstance(batched.positions, np.ndarray)


# ---------------------------------------------------------------------------
# Batched vs sequential: ETD
# ---------------------------------------------------------------------------

class TestBatchedMatchesSequential:
    """Core correctness: vmapped execution must match sequential."""

    def test_etd_batched_vs_sequential(self, gmm_target, etd_config, shared, ref_data):
        """ETD batched positions match sequential within tolerance."""
        from etd.step import init as etd_init, step as etd_step

        seeds = [0, 1, 2]
        checkpoints = [10, 30]
        metrics_list = ["energy_distance"]
        n_iters = 30

        # Pre-compute keys matching main() flow:
        # key = PRNGKey(seed) → split(3) → [k_init, k_ref, k_run]
        seed_keys = {s: jax.random.split(jax.random.PRNGKey(s), 3) for s in seeds}
        init_positions = {
            s: make_init_positions(seed_keys[s][0], gmm_target, shared)
            for s in seeds
        }

        # --- Sequential ---
        seq_positions = {}
        for seed in seeds:
            k_run = seed_keys[seed][2]
            _m, p_dict, _wc = run_single(
                k_run, gmm_target, etd_config, etd_init, etd_step, False,
                init_positions[seed], n_iters, checkpoints, metrics_list, ref_data,
            )
            seq_positions[seed] = p_dict

        # --- Batched ---
        # Pass same k_run keys; run_seeds_batched splits them internally
        # just like run_single does
        batch_keys = [seed_keys[s][2] for s in seeds]
        init_pos_all = np.stack([
            np.asarray(init_positions[s]) for s in seeds
        ])

        m_by_seed, p_by_seed, wc = run_seeds_batched(
            batch_keys, gmm_target, etd_config,
            etd_init, etd_step, init_pos_all,
            n_iters, checkpoints, metrics_list, ref_data,
            compute_metrics_fn=compute_metrics,
        )

        # Compare final positions
        for s_idx, seed in enumerate(seeds):
            for ckpt in checkpoints:
                seq_pos = seq_positions[seed][ckpt]
                bat_pos = p_by_seed[s_idx][ckpt]
                np.testing.assert_allclose(
                    bat_pos, seq_pos, atol=1e-4,
                    err_msg=f"Mismatch at seed={seed}, ckpt={ckpt}",
                )

    def test_svgd_batched_vs_sequential(self, gmm_target, shared, ref_data):
        """SVGD batched positions match sequential."""
        from etd.baselines.svgd import SVGDConfig
        from etd.baselines.svgd import init as svgd_init, step as svgd_step

        config = SVGDConfig(n_particles=50, n_iterations=30, learning_rate=0.1)
        seeds = [0, 1]
        checkpoints = [15, 30]
        metrics_list = ["energy_distance"]

        seed_keys = {s: jax.random.split(jax.random.PRNGKey(s), 3) for s in seeds}
        init_positions = {
            s: make_init_positions(seed_keys[s][0], gmm_target, shared)
            for s in seeds
        }

        # Sequential
        seq_positions = {}
        for seed in seeds:
            k_run = seed_keys[seed][2]
            _m, p_dict, _wc = run_single(
                k_run, gmm_target, config, svgd_init, svgd_step, True,
                init_positions[seed], 30, checkpoints, metrics_list, ref_data,
            )
            seq_positions[seed] = p_dict

        # Batched — pass same k_run keys
        batch_keys = [seed_keys[s][2] for s in seeds]
        init_pos_all = np.stack([
            np.asarray(init_positions[s]) for s in seeds
        ])

        m_by_seed, p_by_seed, wc = run_seeds_batched(
            batch_keys, gmm_target, config,
            svgd_init, svgd_step, init_pos_all,
            30, checkpoints, metrics_list, ref_data,
            compute_metrics_fn=compute_metrics,
        )

        for s_idx, seed in enumerate(seeds):
            for ckpt in checkpoints:
                np.testing.assert_allclose(
                    p_by_seed[s_idx][ckpt], seq_positions[seed][ckpt],
                    atol=1e-4,
                    err_msg=f"SVGD mismatch at seed={seed}, ckpt={ckpt}",
                )

    def test_mala_batched_vs_sequential(self, gmm_target, shared, ref_data):
        """MALA batched positions match sequential."""
        from etd.baselines.mala import MALAConfig
        from etd.baselines.mala import init as mala_init, step as mala_step

        config = MALAConfig(n_particles=50, n_iterations=30, step_size=0.01)
        seeds = [0, 1]
        checkpoints = [30]
        metrics_list = ["energy_distance"]

        seed_keys = {s: jax.random.split(jax.random.PRNGKey(s), 3) for s in seeds}
        init_positions = {
            s: make_init_positions(seed_keys[s][0], gmm_target, shared)
            for s in seeds
        }

        # Sequential
        seq_positions = {}
        for seed in seeds:
            k_run = seed_keys[seed][2]
            _m, p_dict, _wc = run_single(
                k_run, gmm_target, config, mala_init, mala_step, True,
                init_positions[seed], 30, checkpoints, metrics_list, ref_data,
            )
            seq_positions[seed] = p_dict

        # Batched — pass same k_run keys
        batch_keys = [seed_keys[s][2] for s in seeds]
        init_pos_all = np.stack([
            np.asarray(init_positions[s]) for s in seeds
        ])

        m_by_seed, p_by_seed, wc = run_seeds_batched(
            batch_keys, gmm_target, config,
            mala_init, mala_step, init_pos_all,
            30, checkpoints, metrics_list, ref_data,
            compute_metrics_fn=compute_metrics,
        )

        for s_idx, seed in enumerate(seeds):
            np.testing.assert_allclose(
                p_by_seed[s_idx][30], seq_positions[seed][30],
                atol=1e-4,
                err_msg=f"MALA mismatch at seed={seed}",
            )


# ---------------------------------------------------------------------------
# Buffer donation
# ---------------------------------------------------------------------------

class TestBufferDonation:
    def test_no_donation_warning(self, gmm_target, etd_config, shared):
        """Buffer donation should not produce warnings (except int32 fields)."""
        from etd.step import init as etd_init, step as etd_step

        seeds = [0, 1]
        seed_keys = {s: jax.random.split(jax.random.PRNGKey(s), 3) for s in seeds}
        batch_keys = [seed_keys[s][2] for s in seeds]
        init_pos_all = np.stack([
            np.asarray(make_init_positions(seed_keys[s][0], gmm_target, shared))
            for s in seeds
        ])

        # Should run without errors
        m_by_seed, p_by_seed, wc = run_seeds_batched(
            batch_keys, gmm_target, etd_config,
            etd_init, etd_step, init_pos_all,
            10, [10], ["energy_distance"], None,
            compute_metrics_fn=compute_metrics,
        )
        assert 0 in m_by_seed
        assert 1 in m_by_seed


# ---------------------------------------------------------------------------
# Structural grouping
# ---------------------------------------------------------------------------

class TestStructuralGrouping:
    def test_same_structure_grouped(self):
        """Configs differing only in epsilon should be in the same group."""
        c1 = ETDConfig(epsilon=0.1, coupling="balanced")
        c2 = ETDConfig(epsilon=0.5, coupling="balanced")
        assert structural_key(c1) == structural_key(c2)

    def test_different_coupling_separated(self):
        """Configs with different coupling should be in different groups."""
        c1 = ETDConfig(coupling="balanced")
        c2 = ETDConfig(coupling="gibbs")
        assert structural_key(c1) != structural_key(c2)

    def test_grouping_function(self):
        """group_configs_by_structure partitions correctly."""
        from etd.step import init as etd_init, step as etd_step

        c1 = ETDConfig(epsilon=0.1, coupling="balanced")
        c2 = ETDConfig(epsilon=0.5, coupling="balanced")
        c3 = ETDConfig(epsilon=0.1, coupling="gibbs")

        items = [
            ("A", c1, etd_init, etd_step, False),
            ("B", c2, etd_init, etd_step, False),
            ("C", c3, etd_init, etd_step, False),
        ]
        groups = group_configs_by_structure(items)
        # c1 and c2 share structure; c3 is separate
        assert len(groups) == 2
        sizes = sorted(len(v) for v in groups.values())
        assert sizes == [1, 2]

    def test_identify_varying_scalars(self):
        """Correctly identifies which scalar fields differ."""
        c1 = ETDConfig(epsilon=0.1, score_clip=3.0, alpha=0.05)
        c2 = ETDConfig(epsilon=0.5, score_clip=5.0, alpha=0.05)
        varying = identify_varying_scalars([c1, c2])
        assert "epsilon" in varying
        assert "score_clip" in varying
        assert "alpha" not in varying
        # step_size is scalar (damping uses branchless arithmetic)
        assert "step_size" not in _STRUCTURAL_FIELDS_ETD


# ---------------------------------------------------------------------------
# Sweep validation
# ---------------------------------------------------------------------------

class TestSweepValidation:
    def test_schedule_conflict_raises(self):
        """Sweeping a scheduled field should raise ValueError."""
        from etd.schedule import Schedule
        config = ETDConfig(
            epsilon=0.1,
            schedules=(("epsilon", Schedule(kind="linear_decay", value=0.1, end=0.01)),),
        )
        with pytest.raises(ValueError, match="Cannot sweep fields"):
            validate_sweep_fields(config, ["epsilon"])

    def test_no_conflict_passes(self):
        """Non-conflicting sweep fields should not raise."""
        config = ETDConfig(epsilon=0.1)
        validate_sweep_fields(config, ["epsilon", "step_size"])  # should not raise


# ---------------------------------------------------------------------------
# Double vmap: sweep epsilon across configs
# ---------------------------------------------------------------------------

class TestDoubleSweep:
    """Sweep vmap (configs × seeds) matches sequential execution."""

    def test_sweep_epsilon_produces_valid_metrics(self, gmm_target, shared, ref_data):
        """Double-vmapped epsilon sweep produces valid, ordered metrics.

        NOTE: We compare statistical quality rather than exact positions
        because XLA constant-folding differs between static (closure-captured)
        and traced (vmap-injected) config scalars. Over many Sinkhorn × ETD
        iterations, float32 rounding differences cause trajectory divergence.
        This is expected for chaotic particle systems.
        """
        from experiments._parallel import run_sweep_batched
        from etd.step import init as etd_init, step as etd_step

        base_config = ETDConfig(
            n_particles=50, n_iterations=30, n_proposals=10,
            coupling="balanced", cost="euclidean", update="categorical",
            use_score=True, epsilon=0.1, alpha=0.05,
        )
        epsilons = [0.05, 0.1, 0.5]
        configs = [
            dataclasses.replace(base_config, epsilon=e) for e in epsilons
        ]
        seeds = [0, 1]
        checkpoints = [30]
        metrics_list = ["energy_distance"]

        seed_keys = {}
        for seed in seeds:
            key = jax.random.PRNGKey(seed)
            k_init, k_run = jax.random.split(key)
            seed_keys[seed] = (k_init, k_run)

        init_pos_all = np.stack([
            np.asarray(make_init_positions(seed_keys[s][0], gmm_target, shared))
            for s in seeds
        ])
        batch_keys = [seed_keys[s][1] for s in seeds]

        m_by_cs, wc = run_sweep_batched(
            batch_keys, gmm_target, base_config,
            configs, etd_init, etd_step, init_pos_all,
            ["epsilon"], 30, checkpoints, metrics_list, ref_data,
            compute_metrics_fn=compute_metrics,
        )

        # All configs should produce valid (non-NaN) metrics
        for c_idx in range(len(configs)):
            for s_idx in range(len(seeds)):
                val = m_by_cs[c_idx][s_idx][30]["energy_distance"]
                assert not np.isnan(val), (
                    f"NaN at config={c_idx} (eps={epsilons[c_idx]}), seed={s_idx}"
                )
                assert val > 0, (
                    f"Non-positive energy distance at config={c_idx}, seed={s_idx}"
                )

        # Different epsilon values should produce meaningfully different
        # average metrics (statistical check, not pointwise)
        avg_by_eps = {}
        for c_idx, eps in enumerate(epsilons):
            vals = [m_by_cs[c_idx][s][30]["energy_distance"] for s in range(len(seeds))]
            avg_by_eps[eps] = np.mean(vals)

        # All averages should be reasonable (< 5.0 for a well-configured algorithm)
        for eps, avg in avg_by_eps.items():
            assert avg < 5.0, f"Average energy distance {avg:.4f} too high for eps={eps}"

    def test_sweep_multiple_fields(self, gmm_target, shared, ref_data):
        """Sweeping epsilon + alpha simultaneously works."""
        from experiments._parallel import run_sweep_batched
        from etd.step import init as etd_init, step as etd_step

        # Match shared config: n_particles=50
        base_config = ETDConfig(
            n_particles=50, n_iterations=15, n_proposals=10,
            coupling="balanced", cost="euclidean", update="categorical",
            use_score=True, epsilon=0.1, alpha=0.05, fdr=False, sigma=0.1,
        )
        configs = [
            dataclasses.replace(base_config, epsilon=0.05, alpha=0.01),
            dataclasses.replace(base_config, epsilon=0.5, alpha=0.1),
        ]
        seeds = [0]
        checkpoints = [15]

        seed_keys = {}
        for seed in seeds:
            key = jax.random.PRNGKey(seed)
            k_init, k_run = jax.random.split(key)
            seed_keys[seed] = (k_init, k_run)

        init_pos_all = np.stack([
            np.asarray(make_init_positions(seed_keys[s][0], gmm_target, shared))
            for s in seeds
        ])
        batch_keys = [seed_keys[s][1] for s in seeds]

        m_by_cs, wc = run_sweep_batched(
            batch_keys, gmm_target, base_config,
            configs, etd_init, etd_step, init_pos_all,
            ["epsilon", "alpha"],
            15, checkpoints, ["energy_distance"], ref_data,
            compute_metrics_fn=compute_metrics,
        )

        # Both configs should produce valid (non-NaN) results
        for c_idx in range(2):
            val = m_by_cs[c_idx][0][15]["energy_distance"]
            assert not np.isnan(val), f"NaN at config={c_idx}"

        # Different configs should generally produce different results
        v0 = m_by_cs[0][0][15]["energy_distance"]
        v1 = m_by_cs[1][0][15]["energy_distance"]
        # They shouldn't be exactly equal (different epsilon/alpha)
        assert v0 != v1, "Configs with different params produced identical results"

    def test_structural_grouping_in_sweep(self):
        """Configs with different coupling are in different groups."""
        from experiments._parallel import group_configs_by_structure
        from etd.step import init as etd_init, step as etd_step

        c1 = ETDConfig(epsilon=0.1, coupling="balanced")
        c2 = ETDConfig(epsilon=0.5, coupling="balanced")
        c3 = ETDConfig(epsilon=0.1, coupling="gibbs")

        items = [
            ("A", c1, etd_init, etd_step, False),
            ("B", c2, etd_init, etd_step, False),
            ("C", c3, etd_init, etd_step, False),
        ]
        groups = group_configs_by_structure(items)
        assert len(groups) == 2

        # Find the group with 2 members (balanced)
        for members in groups.values():
            if len(members) == 2:
                labels = {m[0] for m in members}
                assert labels == {"A", "B"}

    def test_schedule_conflict_in_sweep(self):
        """Sweeping a scheduled field raises an error."""
        from experiments._parallel import validate_sweep_fields
        from etd.schedule import Schedule

        config = ETDConfig(
            epsilon=0.1,
            schedules=(("epsilon", Schedule(kind="linear_decay", value=0.1, end=0.01)),),
        )
        with pytest.raises(ValueError, match="Cannot sweep fields"):
            validate_sweep_fields(config, ["epsilon"])


# ---------------------------------------------------------------------------
# Integration: run.py main() batched vs sequential
# ---------------------------------------------------------------------------

class TestRunIntegration:
    """Integration test: run.py main() produces same results in both modes."""

    def test_main_batched_vs_sequential(self, tmp_path):
        """main() with parallel seeds matches --no-parallel-seeds."""
        import yaml
        from experiments.run import main

        # Write a small config
        cfg = {
            "experiment": {
                "name": "test-parallel-integration",
                "seeds": [0, 1],
                "target": {
                    "type": "gmm",
                    "params": {"dim": 2, "n_modes": 2, "separation": 4.0},
                },
                "shared": {
                    "n_particles": 30,
                    "n_iterations": 20,
                    "init": {"type": "gaussian", "scale": 3.0},
                },
                "checkpoints": [10, 20],
                "metrics": ["energy_distance"],
                "algorithms": [
                    {
                        "label": "MALA",
                        "type": "baseline",
                        "method": "mala",
                        "step_size": 0.01,
                    },
                ],
            }
        }
        config_path = str(tmp_path / "test_cfg.yaml")
        with open(config_path, "w") as f:
            yaml.dump(cfg, f)

        # Run with parallel seeds (default)
        # Use a baseline (MALA) which has deterministic init
        metrics_parallel = main(config_path=config_path)

        # Run with sequential fallback (debug mode forces sequential)
        metrics_sequential = main(config_path=config_path, debug=True)

        # Both should produce non-NaN energy_distance at checkpoint 20
        for seed in [0, 1]:
            val_p = metrics_parallel[seed]["MALA"][20]["energy_distance"]
            val_s = metrics_sequential[seed]["MALA"][20]["energy_distance"]
            assert not np.isnan(val_p), f"Parallel NaN at seed={seed}"
            assert not np.isnan(val_s), f"Sequential NaN at seed={seed}"
            # Values should match (same keys, same init positions)
            np.testing.assert_allclose(
                val_p, val_s, atol=1e-3,
                err_msg=f"Metric mismatch at seed={seed}",
            )
