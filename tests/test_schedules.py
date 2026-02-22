"""Tests for parameter schedules.

Covers:
  - Schedule evaluation (linear_warmup, linear_decay, cosine_decay)
  - resolve_param with and without schedules
  - Integration with ETD step function
  - YAML parsing of schedule dicts
  - DV warmup validation on 8-mode GMM
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from etd.schedule import Schedule, eval_schedule, resolve_param
from etd.step import init, step
from etd.targets.gmm import GMMTarget
from etd.types import ETDConfig


# ---------------------------------------------------------------------------
# Schedule evaluation
# ---------------------------------------------------------------------------

class TestEvalSchedule:
    """Test eval_schedule for all schedule kinds."""

    def test_linear_warmup_ramp(self):
        """Linear warmup ramps from end to value over warmup iters."""
        sched = Schedule(kind="linear_warmup", value=1.0, end=0.2, warmup=100)
        assert float(eval_schedule(sched, 0, 500)) == pytest.approx(0.2, abs=1e-6)
        assert float(eval_schedule(sched, 50, 500)) == pytest.approx(0.6, abs=1e-6)
        assert float(eval_schedule(sched, 100, 500)) == pytest.approx(1.0, abs=1e-6)
        # Clamps after warmup
        assert float(eval_schedule(sched, 200, 500)) == pytest.approx(1.0, abs=1e-6)

    def test_linear_warmup_default_end_zero(self):
        """Default end=0.0 → ramps from 0 to value."""
        sched = Schedule(kind="linear_warmup", value=2.0, warmup=50)
        assert float(eval_schedule(sched, 0, 500)) == pytest.approx(0.0, abs=1e-6)
        assert float(eval_schedule(sched, 25, 500)) == pytest.approx(1.0, abs=1e-6)
        assert float(eval_schedule(sched, 50, 500)) == pytest.approx(2.0, abs=1e-6)

    def test_linear_warmup_zero_warmup(self):
        """warmup=0 → constant at value (no division by zero)."""
        sched = Schedule(kind="linear_warmup", value=3.0, warmup=0)
        assert float(eval_schedule(sched, 0, 500)) == pytest.approx(3.0, abs=1e-6)

    def test_linear_decay_endpoints(self):
        """Linear decay: step=0 → value, step=n_iterations → end."""
        sched = Schedule(kind="linear_decay", value=1.0, end=0.01)
        assert float(eval_schedule(sched, 0, 500)) == pytest.approx(1.0, abs=1e-6)
        assert float(eval_schedule(sched, 500, 500)) == pytest.approx(0.01, abs=1e-6)
        # Midpoint
        assert float(eval_schedule(sched, 250, 500)) == pytest.approx(0.505, abs=1e-3)

    def test_cosine_decay_endpoints(self):
        """Cosine decay: step=0 → value, step=n_iterations → end."""
        sched = Schedule(kind="cosine_decay", value=0.5, end=0.01)
        assert float(eval_schedule(sched, 0, 500)) == pytest.approx(0.5, abs=1e-6)
        assert float(eval_schedule(sched, 500, 500)) == pytest.approx(0.01, abs=1e-6)
        # Midpoint of cosine ≈ (value + end) / 2
        mid = float(eval_schedule(sched, 250, 500))
        assert mid == pytest.approx(0.255, abs=1e-2)

    def test_cosine_decay_monotonic(self):
        """Cosine decay is monotonically decreasing."""
        sched = Schedule(kind="cosine_decay", value=1.0, end=0.0)
        steps = jnp.arange(0, 501)
        vals = jax.vmap(lambda s: eval_schedule(sched, s, 500))(steps)
        diffs = jnp.diff(vals)
        assert jnp.all(diffs <= 1e-6), "Cosine decay should be monotonically decreasing"

    def test_unknown_kind_raises(self):
        """Unknown schedule kind raises ValueError."""
        sched = Schedule(kind="exponential", value=1.0)
        with pytest.raises(ValueError, match="Unknown schedule kind"):
            eval_schedule(sched, 0, 500)


# ---------------------------------------------------------------------------
# resolve_param
# ---------------------------------------------------------------------------

class TestResolveParam:
    """Test resolve_param with various config/schedule combos."""

    def test_resolve_no_schedule(self):
        """Without schedules, returns the static config value."""
        config = ETDConfig(epsilon=0.05, dv_weight=0.5)
        assert resolve_param(config, "epsilon", 0) == 0.05
        assert resolve_param(config, "dv_weight", 100) == 0.5

    def test_resolve_with_schedule(self):
        """With a schedule, returns the evaluated value at the given step."""
        sched = Schedule(kind="linear_warmup", value=1.0, warmup=200)
        config = ETDConfig(
            dv_weight=1.0,
            n_iterations=500,
            schedules=(("dv_weight", sched),),
        )
        val_0 = float(resolve_param(config, "dv_weight", 0))
        val_100 = float(resolve_param(config, "dv_weight", 100))
        val_200 = float(resolve_param(config, "dv_weight", 200))
        assert val_0 == pytest.approx(0.0, abs=1e-6)
        assert val_100 == pytest.approx(0.5, abs=1e-6)
        assert val_200 == pytest.approx(1.0, abs=1e-6)

    def test_resolve_multiple_schedules(self):
        """Multiple scheduled params resolve independently."""
        config = ETDConfig(
            dv_weight=1.0,
            epsilon=0.5,
            n_iterations=500,
            schedules=(
                ("dv_weight", Schedule(kind="linear_warmup", value=1.0, warmup=100)),
                ("epsilon", Schedule(kind="cosine_decay", value=0.5, end=0.01)),
            ),
        )
        # dv_weight at step 50 → 0.5
        assert float(resolve_param(config, "dv_weight", 50)) == pytest.approx(0.5, abs=1e-6)
        # epsilon at step 0 → 0.5 (start of cosine)
        assert float(resolve_param(config, "epsilon", 0)) == pytest.approx(0.5, abs=1e-6)
        # epsilon at step 500 → 0.01 (end of cosine)
        assert float(resolve_param(config, "epsilon", 500)) == pytest.approx(0.01, abs=1e-6)
        # Unscheduled param falls through
        assert resolve_param(config, "step_size", 0) == 1.0

    def test_resolve_is_jit_safe(self):
        """resolve_param works inside a JIT-compiled function."""
        sched = Schedule(kind="linear_warmup", value=2.0, warmup=100)
        config = ETDConfig(
            dv_weight=2.0,
            n_iterations=500,
            schedules=(("dv_weight", sched),),
        )

        @jax.jit
        def f(step):
            return resolve_param(config, "dv_weight", step)

        assert float(f(0)) == pytest.approx(0.0, abs=1e-6)
        assert float(f(50)) == pytest.approx(1.0, abs=1e-6)
        assert float(f(100)) == pytest.approx(2.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Integration with ETD step
# ---------------------------------------------------------------------------

class TestStepIntegration:
    """Integration tests: schedules inside the ETD step function."""

    @pytest.fixture
    def gmm_target_2d(self):
        return GMMTarget(dim=2, n_modes=4, arrangement="grid", separation=6.0)

    def test_step_with_dv_warmup(self, gmm_target_2d):
        """DV warmup: step 0 → dv_weight ≈ 0 (no feedback), step 200 → full."""
        sched = Schedule(kind="linear_warmup", value=1.0, warmup=200)
        config = ETDConfig(
            coupling="balanced",
            epsilon=0.1,
            alpha=0.05,
            n_particles=20,
            n_proposals=10,
            n_iterations=500,
            dv_feedback=True,
            dv_weight=1.0,
            schedules=(("dv_weight", sched),),
        )
        key = jax.random.PRNGKey(0)
        k_init, k_step = jax.random.split(key)
        state = init(k_init, gmm_target_2d, config)

        # Run one step — should not raise
        new_state, info = step(k_step, state, gmm_target_2d, config)
        assert new_state.positions.shape == (20, 2)
        assert new_state.step == 1

    def test_step_with_epsilon_decay(self, gmm_target_2d):
        """Epsilon decay produces valid coupling at both early and late steps."""
        sched = Schedule(kind="cosine_decay", value=0.5, end=0.01)
        config = ETDConfig(
            coupling="balanced",
            epsilon=0.5,
            alpha=0.05,
            n_particles=20,
            n_proposals=10,
            n_iterations=200,
            schedules=(("epsilon", sched),),
        )
        key = jax.random.PRNGKey(42)
        k_init, k_run = jax.random.split(key)
        state = init(k_init, gmm_target_2d, config)

        # Run a few steps
        for _ in range(5):
            k_run, k_step = jax.random.split(k_run)
            state, info = step(k_step, state, gmm_target_2d, config)

        assert state.step == 5
        assert jnp.isfinite(info["coupling_ess"])

    def test_no_schedule_backward_compat(self, gmm_target_2d):
        """Config with schedules=() behaves identically to before."""
        config = ETDConfig(
            coupling="balanced",
            epsilon=0.1,
            alpha=0.05,
            n_particles=20,
            n_proposals=10,
            n_iterations=100,
        )
        assert config.schedules == ()

        key = jax.random.PRNGKey(7)
        k_init, k_step = jax.random.split(key)
        state = init(k_init, gmm_target_2d, config)
        new_state, info = step(k_step, state, gmm_target_2d, config)
        assert new_state.positions.shape == (20, 2)


# ---------------------------------------------------------------------------
# YAML parsing
# ---------------------------------------------------------------------------

class TestYAMLParsing:
    """Test that schedule dicts are correctly parsed in build_algo_config."""

    def test_yaml_schedule_parsing(self):
        """Dict with 'schedule' key → ETDConfig with populated schedules."""
        from experiments.run import build_algo_config

        entry = {
            "cost": "euclidean",
            "coupling": "balanced",
            "dv_feedback": True,
            "dv_weight": {"schedule": "linear_warmup", "value": 1.0, "warmup": 200},
            "n_proposals": 25,
            "epsilon": 0.1,
            "alpha": 0.05,
        }
        shared = {"n_particles": 50, "n_iterations": 500}
        config, init_fn, step_fn, is_bl = build_algo_config(entry, shared)

        assert not is_bl
        assert isinstance(config, ETDConfig)
        assert len(config.schedules) == 1
        name, sched = config.schedules[0]
        assert name == "dv_weight"
        assert sched.kind == "linear_warmup"
        assert sched.value == 1.0
        assert sched.warmup == 200
        # Base field holds target value
        assert config.dv_weight == 1.0

    def test_yaml_scalar_backward_compat(self):
        """Plain scalar → schedules = ()."""
        from experiments.run import build_algo_config

        entry = {
            "cost": "euclidean",
            "coupling": "balanced",
            "dv_feedback": True,
            "dv_weight": 1.0,
            "n_proposals": 25,
            "epsilon": 0.1,
            "alpha": 0.05,
        }
        shared = {"n_particles": 50, "n_iterations": 500}
        config, _, _, _ = build_algo_config(entry, shared)
        assert config.schedules == ()
        assert config.dv_weight == 1.0

    def test_yaml_multiple_schedules(self):
        """Multiple scheduled params parse correctly."""
        from experiments.run import build_algo_config

        entry = {
            "cost": "euclidean",
            "coupling": "balanced",
            "dv_feedback": True,
            "dv_weight": {"schedule": "linear_warmup", "value": 1.0, "warmup": 200},
            "epsilon": {"schedule": "cosine_decay", "value": 0.5, "end": 0.01},
            "n_proposals": 25,
            "alpha": 0.05,
        }
        shared = {"n_particles": 50, "n_iterations": 500}
        config, _, _, _ = build_algo_config(entry, shared)
        assert len(config.schedules) == 2
        sched_names = {s[0] for s in config.schedules}
        assert sched_names == {"dv_weight", "epsilon"}


# ---------------------------------------------------------------------------
# DV warmup validation
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestDVWarmupValidation:
    """Validate that DV warmup stabilizes the 8-mode ring GMM."""

    def test_dv_warmup_improves_mode_proximity(self):
        """DV warmup on 8-mode ring GMM → better mode proximity than no warmup.

        Runs multiple seeds and compares average mode proximity with warmup
        vs. without warmup over 500 iterations.  Lower proximity is better.
        """
        from etd.diagnostics.metrics import mode_proximity

        target = GMMTarget(
            dim=2, n_modes=8, arrangement="ring",
            separation=10.0, component_std=1.0,
        )

        def run_seeds(config, seeds, n_iters):
            proximities = []
            for seed in seeds:
                key = jax.random.PRNGKey(seed)
                k_init, k_run = jax.random.split(key)
                positions = jax.random.normal(k_init, (config.n_particles, 2)) * 5.0
                state = init(k_init, target, config, init_positions=positions)
                for _ in range(n_iters):
                    k_run, k_step = jax.random.split(k_run)
                    state, _ = step(k_step, state, target, config)
                prox = float(mode_proximity(
                    state.positions, target.means,
                    component_std=target.component_std, dim=target.dim,
                ))
                proximities.append(prox)
            return proximities

        # Config WITHOUT warmup
        config_no_warmup = ETDConfig(
            coupling="balanced",
            epsilon=0.1,
            alpha=0.05,
            n_particles=100,
            n_proposals=25,
            n_iterations=500,
            dv_feedback=True,
            dv_weight=1.0,
        )

        # Config WITH warmup
        config_warmup = ETDConfig(
            coupling="balanced",
            epsilon=0.1,
            alpha=0.05,
            n_particles=100,
            n_proposals=25,
            n_iterations=500,
            dv_feedback=True,
            dv_weight=1.0,
            schedules=(("dv_weight", Schedule(kind="linear_warmup", value=1.0, warmup=200)),),
        )

        seeds = [0, 1, 2, 3, 4]
        n_iters = 500

        prox_no_warmup = run_seeds(config_no_warmup, seeds, n_iters)
        prox_warmup = run_seeds(config_warmup, seeds, n_iters)

        avg_no_warmup = np.mean(prox_no_warmup)
        avg_warmup = np.mean(prox_warmup)

        # Warmup should improve (lower) or match mode proximity on average
        assert avg_warmup <= avg_no_warmup + 0.1, (
            f"Warmup avg={avg_warmup:.4f} vs no-warmup avg={avg_no_warmup:.4f}"
        )
