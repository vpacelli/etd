"""Experiment runner for ETD and baselines.

Loads a YAML config, expands sweeps, runs algorithms on a target,
records metrics at checkpoints, and saves results.

Usage:
    python -m experiments.run configs/gmm_2d_4.yaml
    python -m experiments.run configs/gmm_2d_4.yaml --debug   # no JIT
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import functools
import itertools
import json
import os
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from rich import box
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from etd.baselines import BASELINES, get_baseline
from etd.diagnostics.metrics import (
    energy_distance, mean_error, mean_rmse, mode_coverage,
    sliced_wasserstein, variance_ratio_vs_reference,
)
from etd.schedule import Schedule
from etd.step import init as etd_init, step as etd_step
from etd.targets import get_target
from etd.types import ETDConfig

console = Console()

# ---------------------------------------------------------------------------
# JIT strategy
# ---------------------------------------------------------------------------

DEBUG = os.environ.get("ETD_DEBUG", "0") == "1"


def maybe_jit(fn, **kwargs):
    """Wrap ``fn`` with ``jax.jit`` unless debug mode is active."""
    return fn if DEBUG else jax.jit(fn, **kwargs)


# ---------------------------------------------------------------------------
# Scan runner infrastructure
# ---------------------------------------------------------------------------

def _compute_segments(
    checkpoints: List[int],
    n_iterations: int,
) -> List[Tuple[int, int, int]]:
    """Convert checkpoint list to contiguous scan segments.

    Each segment is ``(start, n_steps, checkpoint)`` where
    ``start`` is the 1-based iteration index of the first step,
    ``n_steps`` is how many steps to run, and ``checkpoint`` is
    the iteration at which to record metrics (end of segment).

    Args:
        checkpoints: Sorted iteration numbers at which to record.
        n_iterations: Total iterations.

    Returns:
        List of ``(start, n_steps, checkpoint)`` tuples.
    """
    # Filter out 0 (handled separately) and anything beyond n_iterations
    ckpts = sorted(c for c in checkpoints if 0 < c <= n_iterations)
    if not ckpts:
        return []

    segments = []
    prev = 0
    for c in ckpts:
        n_steps = c - prev
        if n_steps > 0:
            segments.append((prev + 1, n_steps, c))
        prev = c
    return segments


def _make_jit_scan(
    step_fn: Callable,
    target: object,
    config: Any,
) -> Callable:
    """Build a JIT-compiled scan runner for a given step_fn/target/config.

    The returned function runs ``n_steps`` iterations of ``step_fn`` fused
    into a single ``jax.lax.scan``, with buffer donation for in-place state
    reuse and ``fold_in`` keying for deterministic PRNG.

    Args:
        step_fn: Algorithm step function ``(key, state, target, config) → (state, info)``.
        target: Target distribution (captured by closure, static for JIT).
        config: Algorithm config (captured by closure, static for JIT).

    Returns:
        ``run_segment(state, key_base, start_iter, n_steps) → final_state``
    """
    # Suppress JAX warning about donating int32 fields (e.g. state.step)
    warnings.filterwarnings(
        "ignore", message=".*Some donated buffers were not usable.*"
    )

    @functools.partial(jax.jit, static_argnums=(2, 3), donate_argnums=(0,))
    def run_segment(state, key_base, start_iter, n_steps):
        """Run n_steps of step_fn as a fused scan.

        Args:
            state: Algorithm state (donated — buffer reused in-place).
            key_base: Base PRNG key; per-step keys derived via fold_in.
            start_iter: 1-based iteration index of the first step.
            n_steps: Number of steps to run (static).

        Returns:
            Tuple ``(final_state, last_info)`` where ``last_info`` is the
            info dict from the final step (for checkpoint diagnostics).
        """
        def scan_body(carry, t_offset):
            t = start_iter + t_offset
            key_step = jax.random.fold_in(key_base, t)
            new_state, info = step_fn(key_step, carry, target, config)
            return new_state, info

        final_state, stacked_info = jax.lax.scan(
            scan_body, state, jnp.arange(n_steps),
        )
        # Extract last step's info from the stacked outputs
        last_info = jax.tree.map(lambda x: x[-1], stacked_info)
        return final_state, last_info

    return run_segment


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load a YAML experiment config from disk.

    Args:
        path: Path to a ``.yaml`` file.

    Returns:
        Parsed config dict with an ``experiment`` top-level key.
    """
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Sweep expansion
# ---------------------------------------------------------------------------

def _collect_sweep_axes(entry: dict, prefix: str = ""):
    """Collect sweepable (list-valued) parameters, including nested dicts.

    Args:
        entry: Config dict (may contain nested dicts, e.g. ``cost``).
        prefix: Dot-separated key prefix for nested dicts.

    Returns:
        List of ``(dotted_key, values_list)`` pairs.
    """
    axes = []
    for k, v in entry.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            axes.extend(_collect_sweep_axes(v, prefix=full_key))
        elif isinstance(v, list) and k not in ("label",):
            axes.append((full_key, v))
    return axes


def _set_nested(d: dict, dotted_key: str, value):
    """Set a value in a nested dict using a dotted key path.

    Intermediate dicts are copied so the original is not mutated.

    Args:
        d: Root dict.
        dotted_key: Key path like ``"cost.c"``.
        value: Value to set.
    """
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        cur[k] = dict(cur[k])  # shallow copy intermediate
        cur = cur[k]
    cur[keys[-1]] = value


def expand_algo_sweeps(entry: dict) -> List[dict]:
    """Expand list-valued parameters into a Cartesian product of configs.

    Supports nested dicts (e.g. ``cost: {type: imq, c: [0.5, 1.0]}``).

    Args:
        entry: A single algorithm entry from the YAML config.

    Returns:
        List of concrete entries (one per sweep point).
    """
    axes = _collect_sweep_axes(entry)

    if not axes:
        return [entry]

    sweep_keys = [k for k, _ in axes]
    sweep_vals = [v for _, v in axes]

    expanded = []
    for combo in itertools.product(*sweep_vals):
        new_entry = dict(entry)
        # Deep-copy any nested dicts that contain sweep axes
        for k, v in entry.items():
            if isinstance(v, dict):
                new_entry[k] = dict(v)
        for k, v in zip(sweep_keys, combo):
            if "." in k:
                _set_nested(new_entry, k, v)
            else:
                new_entry[k] = v
        expanded.append(new_entry)

    return expanded


def build_algo_label(base_label: str, entry: dict, original: dict) -> str:
    """Build a descriptive label with sweep suffix.

    Args:
        base_label: The original ``label`` field.
        entry: The expanded (concrete) entry.
        original: The original entry before expansion.

    Returns:
        Label like ``"ETD-B_eps=0.05"`` or just ``"ETD-B"`` if no sweep.
    """
    abbrevs = {
        "epsilon": "eps",
        "learning_rate": "lr",
        "step_size": "h",
        "temperature": "T",
        "n_proposals": "M",
        "bandwidth": "bw",
    }

    sweep_axes = _collect_sweep_axes(original)
    suffixes = []
    for dotted_key, _ in sweep_axes:
        # Use the leaf key name for abbreviation and value lookup
        leaf = dotted_key.split(".")[-1]
        abbrev = abbrevs.get(leaf, leaf)
        # Resolve value from the expanded entry
        val = entry
        for part in dotted_key.split("."):
            val = val[part]
        suffixes.append(f"{abbrev}={val}")

    if suffixes:
        return f"{base_label}_{'_'.join(suffixes)}"
    return base_label


# ---------------------------------------------------------------------------
# Algorithm config dispatch
# ---------------------------------------------------------------------------

# Parameters that are NOT passed to ETDConfig (they are meta or shared)
_ETD_META_KEYS = {"label", "type", "method"}


def build_algo_config(
    entry: dict,
    shared: dict,
) -> Tuple[Any, Callable, Callable, bool]:
    """Build an algorithm config, init_fn, and step_fn from a YAML entry.

    Args:
        entry: Algorithm entry (with ``type``, ``method``, params).
        shared: Shared experiment settings (``n_particles``, ``n_iterations``).

    Returns:
        Tuple ``(config, init_fn, step_fn, is_baseline)``
    """
    is_baseline = entry.get("type") == "baseline"

    if is_baseline:
        method = entry["method"]
        bl = get_baseline(method)
        config_cls = bl["config"]

        # Build kwargs from entry + shared
        kwargs = {}
        for k, v in entry.items():
            if k in ("label", "type", "method"):
                continue
            kwargs[k] = v

        kwargs["n_particles"] = shared.get("n_particles", 100)
        kwargs["n_iterations"] = shared.get("n_iterations", 500)

        config = config_cls(**kwargs)
        return config, bl["init"], bl["step"], True

    else:
        # ETD or SDD
        is_sdd = entry.get("method") == "sdd"

        kwargs = {}
        for k, v in entry.items():
            if k in _ETD_META_KEYS:
                continue
            kwargs[k] = v

        kwargs["n_particles"] = shared.get("n_particles", 100)
        kwargs["n_iterations"] = shared.get("n_iterations", 500)

        # Normalize dict-style cost: {type: imq, c: 1.0} → cost="imq", cost_params=(("c",1.0),)
        raw_cost = kwargs.get("cost", "euclidean")
        if isinstance(raw_cost, dict):
            raw_cost = dict(raw_cost)  # copy to avoid mutating YAML entry
            kwargs["cost"] = raw_cost.pop("type")
            kwargs["cost_params"] = tuple(sorted(raw_cost.items()))

        # Extract schedule dicts: {schedule: "linear_warmup", value: 1.0, warmup: 200}
        schedules = []
        for k in list(kwargs):
            v = kwargs[k]
            if isinstance(v, dict) and "schedule" in v:
                v = dict(v)  # copy to avoid mutating YAML entry
                kind = v.pop("schedule")
                value = float(v.pop("value"))
                # Coerce remaining kwargs to appropriate types
                sched_kw = {}
                for sk, sv in v.items():
                    if sk == "warmup":
                        sched_kw[sk] = int(sv)
                    else:
                        sched_kw[sk] = float(sv)
                sched = Schedule(kind=kind, value=value, **sched_kw)
                kwargs[k] = value  # base field gets target/start value
                schedules.append((k, sched))
        if schedules:
            kwargs["schedules"] = tuple(schedules)

        if is_sdd:
            from etd.extensions.sdd import SDDConfig
            from etd.extensions.sdd import init as sdd_init
            from etd.extensions.sdd import step as sdd_step

            _COERCE = {"float": float, "int": int, "bool": bool}
            field_types = {f.name: f.type for f in dataclasses.fields(SDDConfig)}
            for k, v in list(kwargs.items()):
                coerce_fn = _COERCE.get(field_types.get(k, ""))
                if coerce_fn is not None and not isinstance(v, coerce_fn):
                    kwargs[k] = coerce_fn(v)

            # Remove keys not in SDDConfig
            valid_fields = {f.name for f in dataclasses.fields(SDDConfig)}
            kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}

            config = SDDConfig(**kwargs)
            return config, sdd_init, sdd_step, False

        # Coerce YAML values to match ETDConfig field types.
        # YAML parses '1e-3' as a string; this casts it to float.
        _COERCE = {"float": float, "int": int, "bool": bool}
        field_types = {f.name: f.type for f in dataclasses.fields(ETDConfig)}
        for k, v in list(kwargs.items()):
            coerce_fn = _COERCE.get(field_types.get(k, ""))
            if coerce_fn is not None and not isinstance(v, coerce_fn):
                kwargs[k] = coerce_fn(v)

        config = ETDConfig(**kwargs)
        return config, etd_init, etd_step, False


# ---------------------------------------------------------------------------
# Init positions
# ---------------------------------------------------------------------------

def make_init_positions(
    key: jax.Array,
    target: object,
    shared: dict,
) -> jnp.ndarray:
    """Generate shared starting positions for all algorithms.

    Args:
        key: JAX PRNG key.
        target: Target distribution (used for ``dim``).
        shared: Shared settings dict (``n_particles``, ``init``).

    Returns:
        Positions array, shape ``(N, d)``.
    """
    N = shared.get("n_particles", 100)
    d = target.dim
    init_cfg = shared.get("init", {})
    scale = init_cfg.get("scale", 2.0)

    init_type = init_cfg.get("type", "gaussian")
    if init_type == "gaussian":
        return jax.random.normal(key, (N, d)) * scale
    elif init_type == "prior" and hasattr(target, "sample"):
        return target.sample(key, N)
    else:
        return jax.random.normal(key, (N, d)) * scale


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

def get_reference_data(
    target: object,
    cfg: dict,
    key: jax.Array,
    n_ref: int = 5000,
) -> Optional[jnp.ndarray]:
    """Get reference samples for computing metrics.

    Tries NUTS cache first (for real-data targets), then ``target.sample()``
    for synthetic targets, then None.

    Args:
        target: Target distribution.
        cfg: Experiment config dict.
        key: JAX PRNG key.
        n_ref: Number of reference samples.

    Returns:
        Reference samples ``(n_ref, d)`` or None.
    """
    # Try NUTS cache first
    try:
        from experiments.nuts import load_reference
        target_cfg = cfg.get("experiment", {}).get("target", {})
        target_name = target_cfg.get("type", "")
        target_params = target_cfg.get("params", {})
        ref = load_reference(target_name, target_params)
        if ref is not None:
            return jnp.asarray(ref)
    except (ImportError, Exception):
        pass

    # Fall back to exact sampling
    if hasattr(target, "sample"):
        return target.sample(key, n_ref)
    return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

METRIC_DISPATCH = {
    "energy_distance": lambda p, t, ref: float(energy_distance(p, ref))
        if ref is not None else float("nan"),
    "sliced_wasserstein": lambda p, t, ref: float(
        sliced_wasserstein(p, ref, jax.random.key(0))
    ) if ref is not None else float("nan"),
    "mode_coverage": lambda p, t, ref: float(
        mode_coverage(p, t.means, tolerance=2.0)
    ) if hasattr(t, "means") else float("nan"),
    "mean_error": lambda p, t, ref: float(mean_error(p, t.mean))
        if hasattr(t, "mean") else float("nan"),
    "mean_rmse": lambda p, t, ref: float(mean_rmse(p, ref))
        if ref is not None else float("nan"),
    "variance_ratio_ref": lambda p, t, ref: float(
        variance_ratio_vs_reference(p, ref)
    ) if ref is not None else float("nan"),
}


def compute_metrics(
    particles: jnp.ndarray,
    target: object,
    metrics_list: List[str],
    ref_data: Optional[jnp.ndarray],
) -> Dict[str, float]:
    """Compute requested metrics on a particle set.

    Args:
        particles: Current particle positions, shape ``(N, d)``.
        target: Target distribution.
        metrics_list: List of metric names.
        ref_data: Reference samples or None.

    Returns:
        Dict ``{metric_name: value}`` with Python floats.
    """
    results = {}
    for name in metrics_list:
        fn = METRIC_DISPATCH.get(name)
        if fn is None:
            results[name] = float("nan")
        else:
            results[name] = fn(particles, target, ref_data)
    return results


# ---------------------------------------------------------------------------
# Core run loop
# ---------------------------------------------------------------------------

def run_single(
    key: jax.Array,
    target: object,
    config: Any,
    init_fn: Callable,
    step_fn: Callable,
    is_baseline: bool,
    init_positions: jnp.ndarray,
    n_iterations: int,
    checkpoints: List[int],
    metrics_list: List[str],
    ref_data: Optional[jnp.ndarray],
    step_jit: Optional[Callable] = None,
    debug: bool = False,
) -> Tuple[Dict[int, Dict[str, float]], Dict[int, np.ndarray], float]:
    """Run a single algorithm to completion.

    Uses ``jax.lax.scan`` with buffer donation for fused execution unless
    ``debug=True``, in which case falls back to a Python loop (no JIT).

    Args:
        key: JAX PRNG key.
        target: Target distribution.
        config: Algorithm config (ETDConfig or baseline config).
        init_fn: Initialization function.
        step_fn: Step function.
        is_baseline: Whether this is a baseline algorithm.
        init_positions: Starting positions ``(N, d)``.
        n_iterations: Total iterations.
        checkpoints: Iterations at which to record metrics/particles.
        metrics_list: Which metrics to compute.
        ref_data: Reference samples for metrics (or None).
        step_jit: Pre-compiled step function (if None, uses step_fn directly).
        debug: If True, use Python loop instead of lax.scan.

    Returns:
        Tuple ``(metrics_dict, particles_dict, wall_clock)`` where:
        - ``metrics_dict``: ``{checkpoint → {metric → value}}``
        - ``particles_dict``: ``{checkpoint → (N, d) ndarray}``
        - ``wall_clock``: Total wall-clock seconds.
    """
    k_init, k_run = jax.random.split(key)

    metrics_dict = {}
    particles_dict = {}

    checkpoint_set = set(checkpoints)

    use_scan = not debug and not DEBUG

    if use_scan:
        # --- Scan path: fused lax.scan with buffer donation ---
        segments = _compute_segments(checkpoints, n_iterations)
        scan_runner = _make_jit_scan(step_fn, target, config)

        # Use numpy positions so each init_fn call creates a fresh JAX
        # buffer — donation consumes the warmup buffer and we need a
        # separate allocation for the real run.
        init_positions_np = np.asarray(init_positions)

        # Warm-up: compile scan with a 1-step segment, then discard.
        # This excludes JIT compilation time from the wall-clock measurement.
        warmup_state = init_fn(k_init, target, config, init_positions=init_positions_np)
        if segments:
            _, _wi = scan_runner(warmup_state, k_run, 1, 1)
            jax.block_until_ready(_)
            del warmup_state, _, _wi

        # Re-init state (warm-up consumed the buffer via donation)
        state = init_fn(k_init, target, config, init_positions=init_positions_np)

        # Checkpoint 0
        if 0 in checkpoint_set:
            particles_dict[0] = np.array(state.positions)
            metrics_dict[0] = compute_metrics(
                state.positions, target, metrics_list, ref_data,
            )

        t_start = time.perf_counter()

        for start, n_steps, ckpt in segments:
            state, info = scan_runner(state, k_run, start, n_steps)
            jax.block_until_ready(state)

            positions_np = np.array(state.positions)
            particles_dict[ckpt] = positions_np
            step_metrics = compute_metrics(
                state.positions, target, metrics_list, ref_data,
            )
            for diag_name in ("coupling_ess", "sinkhorn_iters"):
                if diag_name in metrics_list and diag_name in info:
                    step_metrics[diag_name] = float(info[diag_name])
            metrics_dict[ckpt] = step_metrics

        wall_clock = time.perf_counter() - t_start

    else:
        # --- Debug path: Python loop, no JIT ---
        state = init_fn(k_init, target, config, init_positions=init_positions)
        step_fn_actual = step_jit if step_jit is not None else step_fn

        if 0 in checkpoint_set:
            particles_dict[0] = np.array(state.positions)
            metrics_dict[0] = compute_metrics(
                state.positions, target, metrics_list, ref_data,
            )

        t_start = time.perf_counter()

        for i in range(1, n_iterations + 1):
            k_step = jax.random.fold_in(k_run, i)
            state, info = step_fn_actual(k_step, state, target, config)

            if i in checkpoint_set:
                positions_np = np.array(state.positions)
                particles_dict[i] = positions_np
                step_metrics = compute_metrics(
                    state.positions, target, metrics_list, ref_data,
                )
                for diag_name in ("coupling_ess", "sinkhorn_iters"):
                    if diag_name in metrics_list and diag_name in info:
                        step_metrics[diag_name] = float(info[diag_name])
                metrics_dict[i] = step_metrics

        wall_clock = time.perf_counter() - t_start

    return metrics_dict, particles_dict, wall_clock


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------

def save_results(
    out_dir: str,
    cfg: dict,
    all_metrics: dict,
    all_particles: dict,
    ref_data=None,
) -> str:
    """Save experiment results to disk.

    Args:
        out_dir: Output directory path.
        cfg: Original config dict (written as config.yaml).
        all_metrics: ``{seed → {algo → {ckpt → {metric → val}}}}``.
        all_particles: ``{seed → {algo → {ckpt → (N,d) ndarray}}}``.
        ref_data: Reference samples ``(M, d)`` to bundle for portability.

    Returns:
        Output directory path.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Save frozen config
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # Save metrics (convert int keys to strings for JSON)
    metrics_json = {}
    for seed, algos in all_metrics.items():
        seed_key = f"seed{seed}"
        metrics_json[seed_key] = {}
        for algo, ckpts in algos.items():
            metrics_json[seed_key][algo] = {
                str(ckpt): vals for ckpt, vals in ckpts.items()
            }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics_json, f, indent=2)

    # Save particles with flat keys: "seed0__ETD-B__iter100"
    particles_flat = {}
    for seed, algos in all_particles.items():
        for algo, ckpts in algos.items():
            for ckpt, arr in ckpts.items():
                flat_key = f"seed{seed}__{algo}__iter{ckpt}"
                particles_flat[flat_key] = arr

    np.savez(os.path.join(out_dir, "particles.npz"), **particles_flat)

    # Bundle reference samples for portable results
    if ref_data is not None:
        np.savez(
            os.path.join(out_dir, "reference.npz"),
            samples=np.asarray(ref_data),
        )

    return out_dir


# ---------------------------------------------------------------------------
# Summary display
# ---------------------------------------------------------------------------

def _print_summary(
    all_metrics: dict,
    seeds: List[int],
    algo_labels: List[str],
    checkpoints: List[int],
    metrics_list: List[str],
    wall_clocks: dict,
) -> None:
    """Print a Rich summary table of seed-averaged results."""
    final_ckpt = max(checkpoints)

    table = Table(
        title="Results (seed avg ± std)",
        box=box.ROUNDED,
        show_lines=False,
    )
    table.add_column("Algorithm", style="bold")
    for m in metrics_list:
        table.add_column(m.replace("_", " ").title(), justify="right")
    table.add_column("Wall Clock", justify="right")

    for label in algo_labels:
        row = [label]
        for metric in metrics_list:
            vals = []
            for seed in seeds:
                v = all_metrics.get(seed, {}).get(label, {}).get(final_ckpt, {}).get(metric)
                if v is not None and not np.isnan(v):
                    vals.append(v)
            if vals:
                avg = np.mean(vals)
                std = np.std(vals)
                row.append(f"{avg:.4f} ± {std:.4f}")
            else:
                row.append("N/A")

        # Wall clock
        wc_vals = [wall_clocks.get((seed, label), 0.0) for seed in seeds]
        wc_avg = np.mean(wc_vals)
        wc_std = np.std(wc_vals)
        row.append(f"{wc_avg:.1f} ± {wc_std:.1f}s")

        table.add_row(*row)

    console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: Optional[str] = None, debug: bool = False) -> dict:
    """Run an experiment from a YAML config.

    Args:
        config_path: Path to YAML config. If None, parses from CLI.
        debug: If True, disable JIT.

    Returns:
        ``all_metrics`` dict for programmatic use.
    """
    global DEBUG

    if config_path is None:
        parser = argparse.ArgumentParser(description="ETD experiment runner")
        parser.add_argument("config", help="Path to YAML config file")
        parser.add_argument("--debug", action="store_true", help="Disable JIT")
        args = parser.parse_args()
        config_path = args.config
        debug = args.debug

    if debug:
        DEBUG = True

    # --- Load config ---
    cfg = load_config(config_path)
    exp = cfg["experiment"]

    name = exp["name"]
    seeds = exp["seeds"]
    target_cfg = exp["target"]
    shared = exp.get("shared", {})
    checkpoints = exp["checkpoints"]
    metrics_list = exp["metrics"]
    algo_entries = exp["algorithms"]

    # --- Build target ---
    target = get_target(target_cfg["type"], **target_cfg.get("params", {}))

    # --- Header ---
    now = datetime.now().strftime("%H:%M:%S")
    console.print(f"[{now}] Loading config from {config_path}")
    console.print(
        f"[{now}] Target: {target_cfg['type']}, d={target.dim}"
    )

    # --- Expand sweeps and build configs ---
    algo_configs = []  # list of (label, config, init_fn, step_fn, is_baseline)
    for entry in algo_entries:
        original = dict(entry)
        expanded = expand_algo_sweeps(entry)
        for concrete in expanded:
            label = build_algo_label(
                concrete.get("label", "unnamed"), concrete, original
            )
            config, init_fn, step_fn, is_bl = build_algo_config(concrete, shared)
            algo_configs.append((label, config, init_fn, step_fn, is_bl))

    algo_labels = [ac[0] for ac in algo_configs]
    n_iters = shared.get("n_iterations", 500)

    console.print(
        f"[{now}] Running {len(algo_configs)} algorithms × {len(seeds)} seeds"
    )

    # --- Reference data ---
    ref_key = jax.random.PRNGKey(99999)
    ref_data = get_reference_data(target, cfg, ref_key)

    # --- Pre-compile step functions (debug mode only) ---
    # In scan mode, _make_jit_scan creates its own JIT closure.
    compiled_steps = {}
    if DEBUG:
        for label, config, init_fn, step_fn, is_bl in algo_configs:
            compiled_steps[label] = step_fn  # raw, no JIT in debug

    # --- Run ---
    all_metrics = {}   # seed → algo → ckpt → {metric: val}
    all_particles = {} # seed → algo → ckpt → (N,d) ndarray
    wall_clocks = {}   # (seed, algo) → seconds

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        for label, config, init_fn, step_fn, is_bl in algo_configs:
            task_id = progress.add_task(
                f"  {label}", total=len(seeds)
            )

            for seed in seeds:
                key = jax.random.PRNGKey(seed)
                k_init, k_ref, k_run = jax.random.split(key, 3)

                init_positions = make_init_positions(k_init, target, shared)

                m_dict, p_dict, wc = run_single(
                    k_run, target, config, init_fn, step_fn, is_bl,
                    init_positions, n_iters, checkpoints,
                    metrics_list, ref_data,
                    step_jit=compiled_steps.get(label),
                    debug=debug,
                )

                all_metrics.setdefault(seed, {})[label] = m_dict
                all_particles.setdefault(seed, {})[label] = p_dict
                wall_clocks[(seed, label)] = wc

                progress.advance(task_id)

    # --- Summary ---
    _print_summary(
        all_metrics, seeds, algo_labels, checkpoints,
        metrics_list, wall_clocks,
    )

    # --- Save ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = os.path.join("results", name, timestamp)
    save_results(out_dir, cfg, all_metrics, all_particles, ref_data=ref_data)
    console.print(f"\n✓ Results saved to {out_dir}/")

    return all_metrics


if __name__ == "__main__":
    main()
