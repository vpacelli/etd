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
import itertools
import json
import os
import time
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
from etd.diagnostics.metrics import energy_distance, mean_error, mode_coverage
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

def expand_algo_sweeps(entry: dict) -> List[dict]:
    """Expand list-valued parameters into a Cartesian product of configs.

    Args:
        entry: A single algorithm entry from the YAML config.

    Returns:
        List of concrete entries (one per sweep point).
    """
    sweep_keys = []
    sweep_vals = []

    for k, v in entry.items():
        if isinstance(v, list) and k not in ("label",):
            sweep_keys.append(k)
            sweep_vals.append(v)

    if not sweep_keys:
        return [entry]

    expanded = []
    for combo in itertools.product(*sweep_vals):
        new_entry = dict(entry)
        for k, v in zip(sweep_keys, combo):
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
    suffixes = []
    for k, v in original.items():
        if isinstance(v, list) and k not in ("label",):
            # Abbreviate parameter names for readability
            abbrev = {
                "epsilon": "eps",
                "learning_rate": "lr",
                "step_size": "h",
                "temperature": "T",
                "n_proposals": "M",
                "bandwidth": "bw",
            }.get(k, k)
            suffixes.append(f"{abbrev}={entry[k]}")

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
        # ETD
        kwargs = {}
        for k, v in entry.items():
            if k in _ETD_META_KEYS:
                continue
            kwargs[k] = v

        kwargs["n_particles"] = shared.get("n_particles", 100)
        kwargs["n_iterations"] = shared.get("n_iterations", 500)

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

    For synthetic targets with ``sample()``, draws ``n_ref`` samples.

    Args:
        target: Target distribution.
        cfg: Experiment config dict.
        key: JAX PRNG key.
        n_ref: Number of reference samples.

    Returns:
        Reference samples ``(n_ref, d)`` or None.
    """
    if hasattr(target, "sample"):
        return target.sample(key, n_ref)
    return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

METRIC_DISPATCH = {
    "energy_distance": lambda p, t, ref: float(energy_distance(p, ref))
        if ref is not None else float("nan"),
    "mode_coverage": lambda p, t, ref: float(
        mode_coverage(p, t.means, tolerance=2.0)
    ) if hasattr(t, "means") else float("nan"),
    "mean_error": lambda p, t, ref: float(mean_error(p, t.mean))
        if hasattr(t, "mean") else float("nan"),
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
) -> Tuple[Dict[int, Dict[str, float]], Dict[int, np.ndarray], float]:
    """Run a single algorithm to completion.

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

    Returns:
        Tuple ``(metrics_dict, particles_dict, wall_clock)`` where:
        - ``metrics_dict``: ``{checkpoint → {metric → value}}``
        - ``particles_dict``: ``{checkpoint → (N, d) ndarray}``
        - ``wall_clock``: Total wall-clock seconds.
    """
    k_init, k_run = jax.random.split(key)

    state = init_fn(k_init, target, config, init_positions=init_positions)

    step_fn_actual = step_jit if step_jit is not None else step_fn

    metrics_dict = {}
    particles_dict = {}

    checkpoint_set = set(checkpoints)

    t_start = time.perf_counter()

    for i in range(1, n_iterations + 1):
        k_run, k_step = jax.random.split(k_run)

        if is_baseline:
            state, info = step_fn_actual(k_step, state, target, config)
        else:
            state, info = step_fn_actual(k_step, state, target, config)

        if i in checkpoint_set:
            positions_np = np.array(state.positions)
            particles_dict[i] = positions_np
            metrics_dict[i] = compute_metrics(
                state.positions, target, metrics_list, ref_data,
            )

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
) -> str:
    """Save experiment results to disk.

    Args:
        out_dir: Output directory path.
        cfg: Original config dict (written as config.yaml).
        all_metrics: ``{seed → {algo → {ckpt → {metric → val}}}}``.
        all_particles: ``{seed → {algo → {ckpt → (N,d) ndarray}}}``.

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

    # --- Pre-compile step functions ---
    compiled_steps = {}
    for label, config, init_fn, step_fn, is_bl in algo_configs:
        if is_bl:
            compiled_steps[label] = maybe_jit(step_fn)
        else:
            compiled_steps[label] = maybe_jit(
                step_fn, static_argnums=(2, 3)
            )

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
                    step_jit=compiled_steps[label],
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
    save_results(out_dir, cfg, all_metrics, all_particles)
    console.print(f"\n✓ Results saved to {out_dir}/")

    return all_metrics


if __name__ == "__main__":
    main()
