"""Grid-search tuner for hyperparameter selection.

Runs sweep configs, ranks results by a specified metric, and
saves the best configuration to ``best.yaml``.

Usage:
    python -m experiments.tune experiments/configs/sweeps/eps_sensitivity.yaml
    python -m experiments.tune experiments/configs/sweeps/eps_sensitivity.yaml --debug
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

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

from experiments.run import (
    build_algo_config,
    build_algo_label,
    compute_metrics,
    expand_algo_sweeps,
    get_reference_data,
    load_config,
    make_init_positions,
    maybe_jit,
    run_single,
    save_results,
)
from etd.targets import get_target

console = Console()

# ---------------------------------------------------------------------------
# Metric direction
# ---------------------------------------------------------------------------

LOWER_IS_BETTER = {"energy_distance", "sliced_wasserstein", "mean_error"}
HIGHER_IS_BETTER = {"mode_coverage"}


def _is_better(val: float, best: float, metric: str) -> bool:
    """Return True if val is better than best for the given metric."""
    if metric in LOWER_IS_BETTER:
        return val < best
    return val > best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: Optional[str] = None, debug: bool = False) -> dict:
    """Run a grid search and print ranked results.

    Args:
        config_path: Path to YAML config with sweep parameters.
        debug: If True, disable JIT.

    Returns:
        Dict with ``"best_label"``, ``"best_config"``, ``"rankings"``.
    """
    import experiments.run as run_module

    if config_path is None:
        parser = argparse.ArgumentParser(description="ETD hyperparameter tuner")
        parser.add_argument("config", help="Path to sweep YAML config file")
        parser.add_argument("--debug", action="store_true", help="Disable JIT")
        parser.add_argument(
            "--metric", default=None,
            help="Metric to rank by (default: first in metrics list)",
        )
        args = parser.parse_args()
        config_path = args.config
        debug = args.debug
        rank_metric = args.metric
    else:
        rank_metric = None

    if debug:
        run_module.DEBUG = True

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

    if rank_metric is None:
        rank_metric = metrics_list[0]

    # --- Build target ---
    target = get_target(target_cfg["type"], **target_cfg.get("params", {}))

    console.print(f"[bold]Tuning:[/bold] {name}")
    console.print(f"  Target: {target_cfg['type']}, d={target.dim}")
    console.print(f"  Ranking by: {rank_metric}")

    # --- Expand sweeps ---
    algo_configs = []
    for entry in algo_entries:
        original = dict(entry)
        expanded = expand_algo_sweeps(entry)
        for concrete in expanded:
            label = build_algo_label(
                concrete.get("label", "unnamed"), concrete, original,
            )
            config, init_fn, step_fn, is_bl = build_algo_config(concrete, shared)
            algo_configs.append((label, config, init_fn, step_fn, is_bl, concrete))

    console.print(f"  {len(algo_configs)} configs × {len(seeds)} seeds\n")

    # --- Reference data ---
    ref_key = jax.random.PRNGKey(99999)
    ref_data = get_reference_data(target, cfg, ref_key)

    # --- Run all configs ---
    n_iters = shared.get("n_iterations", 500)
    final_ckpt = max(checkpoints)
    results = {}  # label → list of metric values across seeds

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        for label, config, init_fn, step_fn, is_bl, entry in algo_configs:
            task_id = progress.add_task(f"  {label}", total=len(seeds))
            seed_vals = []
            step_compiled = maybe_jit(
                step_fn, static_argnums=(2, 3),
            ) if not debug else None

            for seed in seeds:
                key = jax.random.PRNGKey(seed)
                k_init, k_run = jax.random.split(key)
                init_pos = make_init_positions(k_init, target, shared)

                m_dict, _, wc = run_single(
                    k_run, target, config, init_fn, step_fn, is_bl,
                    init_pos, n_iters, [final_ckpt],
                    [rank_metric], ref_data,
                    step_jit=step_compiled,
                    debug=debug,
                )

                val = m_dict.get(final_ckpt, {}).get(rank_metric, float("nan"))
                seed_vals.append(val)
                progress.advance(task_id)

            results[label] = seed_vals

    # --- Rank ---
    rankings = []
    for label, vals in results.items():
        valid = [v for v in vals if not np.isnan(v)]
        avg = np.mean(valid) if valid else float("nan")
        std = np.std(valid) if valid else float("nan")
        rankings.append((label, avg, std))

    reverse = rank_metric in HIGHER_IS_BETTER
    rankings.sort(key=lambda x: x[1] if not np.isnan(x[1]) else float("inf"),
                  reverse=reverse)

    # --- Display ---
    table = Table(
        title=f"Tuning Results — {rank_metric}",
        box=box.ROUNDED,
    )
    table.add_column("Rank", justify="right", style="bold")
    table.add_column("Configuration")
    table.add_column(rank_metric.replace("_", " ").title(), justify="right")
    table.add_column("Std", justify="right")

    for i, (label, avg, std) in enumerate(rankings):
        style = "bold green" if i == 0 else ""
        table.add_row(
            str(i + 1), label, f"{avg:.6f}", f"{std:.6f}", style=style,
        )

    console.print(table)

    # --- Save best config ---
    best_label = rankings[0][0]
    best_entry = None
    for label, _, _, _, _, entry in algo_configs:
        if label == best_label:
            best_entry = entry
            break

    if best_entry is not None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out_dir = os.path.join("results", name, timestamp)
        os.makedirs(out_dir, exist_ok=True)
        best_path = os.path.join(out_dir, "best.yaml")
        with open(best_path, "w") as f:
            yaml.dump(
                {"best": {"label": best_label, "config": best_entry}},
                f,
                default_flow_style=False,
            )
        console.print(f"\n✓ Best config saved to {best_path}")

    return {
        "best_label": rankings[0][0],
        "best_value": rankings[0][1],
        "rankings": rankings,
    }


if __name__ == "__main__":
    main()
