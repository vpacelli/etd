"""Shared terminal output helpers for experiment scripts.

Provides formatting, metric-direction knowledge, summary tables,
and standardized header/footer output used by run.py, tune.py, and show.py.
"""

from __future__ import annotations

from math import floor, log10
from typing import Any, Dict, List, Optional, Set

import numpy as np
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
# Metric direction (single source of truth)
# ---------------------------------------------------------------------------

LOWER_IS_BETTER: Set[str] = {
    "energy_distance",
    "sliced_wasserstein",
    "mean_error",
    "mean_rmse",
    "mode_proximity",
    "mode_balance",
}

HIGHER_IS_BETTER: Set[str] = {
    "variance_ratio_ref",
}


def is_better(val: float, best: float, metric: str) -> bool:
    """Return True if *val* is better than *best* for *metric*."""
    if metric in LOWER_IS_BETTER:
        return val < best
    return val > best


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------

def fmt_sigfig(x: float, n: int = 3) -> str:
    """Format a float to *n* significant figures.

    Uses scientific notation when |x| < 0.001 or |x| > 99999.

    Args:
        x: Value to format.
        n: Number of significant figures.

    Returns:
        Formatted string.
    """
    if not np.isfinite(x):
        if np.isnan(x):
            return "NaN"
        return "inf" if x > 0 else "-inf"
    if x == 0:
        return "0"
    magnitude = floor(log10(abs(x)))
    if magnitude < -3 or magnitude > 4:
        return f"{x:.{n - 1}e}"
    precision = max(0, n - 1 - magnitude)
    return f"{x:.{precision}f}"


def fmt_mean_std(mean: float, std: float, sig: int = 3) -> str:
    """Format ``mean +/- std`` with precision matched to std magnitude.

    Args:
        mean: Mean value.
        std: Standard deviation.
        sig: Number of significant figures.

    Returns:
        Formatted string like ``"0.023 +/- 0.004"``.
    """
    if np.isnan(mean):
        return "N/A"
    if not np.isfinite(mean) or not np.isfinite(std):
        return f"{fmt_sigfig(mean, sig)} \u00b1 {fmt_sigfig(std, sig)}"
    if std == 0:
        return fmt_sigfig(mean, sig)
    # Scientific notation for very small values
    if abs(mean) < 0.001 and mean != 0:
        return f"{mean:.{sig - 1}e} \u00b1 {std:.1e}"
    # Show enough decimals to display 2 sig figs of std,
    # but at least enough to show the mean's precision too.
    std_magnitude = floor(log10(abs(std)))
    # Decimals to show 2 sig figs of std
    decimals = max(0, -std_magnitude + 1)
    # Cap at 6 to avoid absurd precision
    decimals = min(decimals, 6)
    return f"{mean:.{decimals}f} \u00b1 {std:.{decimals}f}"


def fmt_wallclock(seconds: float) -> str:
    """Format wall-clock seconds as a human-readable string.

    Returns:
        ``"0.4s"``, ``"12.3s"``, ``"2m 34s"``, or ``"1h 12m"``.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s:02d}s"
    h, rem = divmod(int(seconds), 3600)
    m = rem // 60
    return f"{h}h {m:02d}m"


# ---------------------------------------------------------------------------
# Header / footer
# ---------------------------------------------------------------------------

def log_header(
    console: Console,
    config_path: str,
    target_type: str,
    dim: int,
    n_algos: int,
    n_seeds: int,
    mode: str = "parallel",
    *,
    hparams: Optional[Dict[str, Any]] = None,
) -> None:
    """Print a standardized header with auto-timestamps.

    Args:
        console: Rich console.
        config_path: Path to the YAML config file.
        target_type: Target distribution name.
        dim: Dimensionality of the target.
        n_algos: Number of algorithms (or configs).
        n_seeds: Number of seeds.
        mode: ``"parallel"`` or ``"sequential"``.
        hparams: Optional dict of key hyperparameters to display on
            a dim line (e.g. ``{"N": 100, "T": 500}``).
    """
    console.log(f"Loading config from {config_path}")
    console.log(f"Target: {target_type}, d={dim}")
    console.log(f"Running {n_algos} algorithms \u00d7 {n_seeds} seeds ({mode})")
    if hparams:
        parts = [f"{k}={v}" for k, v in hparams.items()]
        console.print(f"  {'  '.join(parts)}", style="dim")


def log_footer(console: Console, path: str, message: str = "Results saved to") -> None:
    """Print a standardized footer with green checkmark.

    Args:
        console: Rich console.
        path: Output path to display.
        message: Message prefix before the path.
    """
    console.log(f"[green]\u2713[/green] {message} {path}")


# ---------------------------------------------------------------------------
# Per-algorithm completion line
# ---------------------------------------------------------------------------

def print_algo_result(
    console: Console,
    label: str,
    metric_name: str,
    value: float,
) -> None:
    """Print a dim one-liner after an algorithm completes.

    Args:
        console: Rich console.
        label: Algorithm label.
        metric_name: Primary metric name.
        value: Metric value.
    """
    formatted = fmt_sigfig(value) if not np.isnan(value) else "N/A"
    console.log(
        f"  [dim]{label:<16s}[/dim] \u2192 "
        f"[dim]{metric_name}: {formatted}[/dim]"
    )


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def build_summary_table(
    all_metrics: Dict[int, Dict[str, Dict[int, Dict[str, float]]]],
    seeds: List[int],
    algo_labels: List[str],
    final_ckpt: int,
    metrics_list: List[str],
    wall_clocks: Optional[Dict] = None,
    title: str = "Results (seed avg \u00b1 std)",
) -> Table:
    """Build a Rich Table with best-result highlighting.

    Args:
        all_metrics: ``{seed -> {algo -> {ckpt -> {metric -> val}}}}``.
        seeds: List of seed values.
        algo_labels: Ordered algorithm labels.
        final_ckpt: Checkpoint to evaluate (typically ``max(checkpoints)``).
        metrics_list: Ordered metric names.
        wall_clocks: Optional ``{(seed, algo) -> seconds}``.
        title: Table title.

    Returns:
        Rich :class:`Table` ready for ``console.print()``.
    """
    # --- Collect per-algo aggregates ---
    algo_stats: Dict[str, Dict[str, tuple]] = {}  # label -> metric -> (mean, std)
    for label in algo_labels:
        stats = {}
        for metric in metrics_list:
            vals = []
            for seed in seeds:
                v = (all_metrics.get(seed, {})
                     .get(label, {})
                     .get(final_ckpt, {})
                     .get(metric))
                if v is not None and not np.isnan(v):
                    vals.append(v)
            if vals:
                stats[metric] = (float(np.mean(vals)), float(np.std(vals)))
            else:
                stats[metric] = (float("nan"), float("nan"))
        algo_stats[label] = stats

    # --- Find best per metric ---
    best_algo: Dict[str, str] = {}
    for metric in metrics_list:
        if metric not in LOWER_IS_BETTER and metric not in HIGHER_IS_BETTER:
            continue
        best_label = None
        best_val = float("inf") if metric in LOWER_IS_BETTER else float("-inf")
        for label in algo_labels:
            mean, _ = algo_stats[label][metric]
            if np.isnan(mean):
                continue
            if is_better(mean, best_val, metric):
                best_val = mean
                best_label = label
        if best_label is not None:
            best_algo[metric] = best_label

    # --- Build table ---
    table = Table(title=title, box=box.ROUNDED, show_lines=False)
    table.add_column("Algorithm", style="bold")
    for m in metrics_list:
        table.add_column(m.replace("_", " ").title(), justify="right")
    if wall_clocks is not None:
        table.add_column("Wall Clock", justify="right", style="dim")

    for label in algo_labels:
        row: list = [label]
        for metric in metrics_list:
            mean, std = algo_stats[label][metric]
            cell_str = fmt_mean_std(mean, std)
            cell = Text(cell_str)
            if best_algo.get(metric) == label:
                cell.stylize("bold green")
            row.append(cell)

        if wall_clocks is not None:
            wc_vals = [wall_clocks.get((seed, label), 0.0) for seed in seeds]
            wc_mean = float(np.mean(wc_vals))
            wc_std = float(np.std(wc_vals))
            if wc_std > 0 and len(seeds) > 1:
                row.append(f"{fmt_wallclock(wc_mean)} \u00b1 {fmt_wallclock(wc_std)}")
            else:
                row.append(fmt_wallclock(wc_mean))

        table.add_row(*row)

    return table


def build_summary_table_flat(
    results: Dict[int, Dict[str, Dict[str, float]]],
    metrics_list: List[str],
    title: str = "Results (seed avg \u00b1 std)",
) -> Table:
    """Build a summary table from a flat ``{seed -> {algo -> {metric -> val}}}`` dict.

    Used by :mod:`experiments.show` where metrics are already at a single
    checkpoint and don't have a checkpoint sub-dict.

    Args:
        results: ``{seed_int -> {algo -> {metric -> val}}}``.
        metrics_list: Ordered metric names.
        title: Table title.

    Returns:
        Rich :class:`Table` ready for ``console.print()``.
    """
    # Discover algo labels (preserve insertion order from first seed)
    algo_labels: List[str] = []
    for seed_data in results.values():
        for label in seed_data:
            if label not in algo_labels:
                algo_labels.append(label)

    seeds = sorted(results.keys())

    # --- Collect per-algo aggregates ---
    algo_stats: Dict[str, Dict[str, tuple]] = {}
    for label in algo_labels:
        stats = {}
        for metric in metrics_list:
            vals = []
            for seed in seeds:
                v = results.get(seed, {}).get(label, {}).get(metric)
                if v is not None and not np.isnan(v):
                    vals.append(v)
            if vals:
                stats[metric] = (float(np.mean(vals)), float(np.std(vals)))
            else:
                stats[metric] = (float("nan"), float("nan"))
        algo_stats[label] = stats

    # --- Find best per metric ---
    best_algo: Dict[str, str] = {}
    for metric in metrics_list:
        if metric not in LOWER_IS_BETTER and metric not in HIGHER_IS_BETTER:
            continue
        best_label = None
        best_val = float("inf") if metric in LOWER_IS_BETTER else float("-inf")
        for label in algo_labels:
            mean, _ = algo_stats[label][metric]
            if np.isnan(mean):
                continue
            if is_better(mean, best_val, metric):
                best_val = mean
                best_label = label
        if best_label is not None:
            best_algo[metric] = best_label

    # --- Build table ---
    table = Table(title=title, box=box.ROUNDED, show_lines=False)
    table.add_column("Algorithm", style="bold")
    for m in metrics_list:
        table.add_column(m.replace("_", " ").title(), justify="right")

    for label in algo_labels:
        row: list = [label]
        for metric in metrics_list:
            mean, std = algo_stats[label][metric]
            cell_str = fmt_mean_std(mean, std)
            cell = Text(cell_str)
            if best_algo.get(metric) == label:
                cell.stylize("bold green")
            row.append(cell)
        table.add_row(*row)

    return table
