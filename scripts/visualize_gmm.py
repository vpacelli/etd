"""Visualize ETD benchmark results on 2D 4-mode Gaussian mixture.

Produces two figures:
  1. Final particle scatter plots (one panel per algorithm)
  2. Convergence curves (energy distance + mode coverage vs iteration)

Usage:
    python scripts/visualize_gmm.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from etd.targets.gmm import GMMTarget
from figures.style import (
    ALGO_COLORS,
    FULL_WIDTH,
    plot_contours,
    plot_particles,
    savefig_paper,
    setup_style,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = ROOT / "results" / "gmm-2d-4" / "2026-02-21_003354"
METRICS_PATH = RESULTS_DIR / "metrics.json"
PARTICLES_PATH = RESULTS_DIR / "particles.npz"

ALGOS = ["ETD-B", "ETD-SR", "SVGD", "ULA", "MPPI"]
CHECKPOINTS = [1, 5, 10, 25, 50, 100, 200, 500]
SEEDS = [f"seed{i}" for i in range(5)]

AXIS_LIM = (-6.5, 6.5)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_metrics():
    """Load metrics.json into nested dict.

    Returns:
        dict: {seed: {algo: {iter_int: {metric: value}}}}
    """
    with open(METRICS_PATH) as f:
        raw = json.load(f)
    # Convert checkpoint string keys to ints for easier handling
    out = {}
    for seed, algo_dict in raw.items():
        out[seed] = {}
        for algo, ckpt_dict in algo_dict.items():
            out[seed][algo] = {
                int(k): v for k, v in ckpt_dict.items()
            }
    return out


def load_particles():
    """Load particles.npz.

    Returns:
        dict: Flat dict with keys like "seed0__ETD-B__iter500" -> (N, 2) array.
    """
    return dict(np.load(PARTICLES_PATH))


# ---------------------------------------------------------------------------
# Figure 1: Final particle scatter plots
# ---------------------------------------------------------------------------

def make_scatter_figure(target, particles):
    """One panel per algorithm showing contours + final particles (seed 0)."""
    n_algos = len(ALGOS)
    fig, axes = plt.subplots(
        1, n_algos,
        figsize=(FULL_WIDTH, FULL_WIDTH / n_algos + 0.15),
        sharex=True,
        sharey=True,
    )

    for ax, algo in zip(axes, ALGOS):
        # Background contours
        plot_contours(ax, target.log_prob, AXIS_LIM, AXIS_LIM)

        # Particles
        key = f"seed0__{algo}__iter500"
        pts = particles[key]
        plot_particles(ax, pts, color=ALGO_COLORS[algo], s=18)

        # Panel title
        ax.set_title(algo, fontsize=9)
        ax.set_xlim(AXIS_LIM)
        ax.set_ylim(AXIS_LIM)
        ax.set_aspect("equal")

        # Minimal tick labels: only on leftmost panel
        if ax is not axes[0]:
            ax.tick_params(labelleft=False)

    fig.subplots_adjust(wspace=0.08)
    return fig


# ---------------------------------------------------------------------------
# Figure 2: Convergence curves
# ---------------------------------------------------------------------------

def _gather_metric(metrics, metric_name):
    """Gather metric across seeds and checkpoints for each algorithm.

    Returns:
        dict: {algo: (median_array, q25_array, q75_array)} using
        median and interquartile range for robustness to outlier seeds.
    """
    out = {}
    for algo in ALGOS:
        vals = []  # shape will be (n_seeds, n_checkpoints)
        for seed in SEEDS:
            row = []
            for ckpt in CHECKPOINTS:
                entry = metrics.get(seed, {}).get(algo, {}).get(ckpt, {})
                row.append(entry.get(metric_name, np.nan))
            vals.append(row)
        vals = np.array(vals)  # (n_seeds, n_checkpoints)
        median = np.nanmedian(vals, axis=0)
        q25 = np.nanpercentile(vals, 25, axis=0)
        q75 = np.nanpercentile(vals, 75, axis=0)
        out[algo] = (median, q25, q75)
    return out


def make_convergence_figure(metrics):
    """Energy distance and mode coverage vs iteration."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(FULL_WIDTH, 2.2),
    )

    iters = np.array(CHECKPOINTS)

    # --- Energy distance ---
    ed = _gather_metric(metrics, "energy_distance")
    for algo in ALGOS:
        median, q25, q75 = ed[algo]
        color = ALGO_COLORS[algo]
        ax1.plot(iters, median, color=color, linewidth=1.2, label=algo)
        ax1.fill_between(iters, q25, q75, color=color, alpha=0.15)

    ax1.set_xscale("log")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Energy distance")
    ax1.set_xticks(CHECKPOINTS)
    ax1.set_xticklabels([str(c) for c in CHECKPOINTS], fontsize=7)
    ax1.minorticks_off()

    # --- Mode coverage ---
    mc = _gather_metric(metrics, "mode_coverage")
    for algo in ALGOS:
        median, q25, q75 = mc[algo]
        color = ALGO_COLORS[algo]
        ax2.plot(iters, median, color=color, linewidth=1.2, label=algo)
        ax2.fill_between(
            iters,
            np.clip(q25, 0, 1),
            np.clip(q75, 0, 1),
            color=color,
            alpha=0.15,
        )

    ax2.set_xscale("log")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Mode coverage")
    ax2.set_ylim(-0.05, 1.08)
    ax2.set_xticks(CHECKPOINTS)
    ax2.set_xticklabels([str(c) for c in CHECKPOINTS], fontsize=7)
    ax2.minorticks_off()

    # Legend — placed once, outside the panels
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=len(ALGOS),
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 1.06),
    )

    fig.subplots_adjust(wspace=0.35)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    setup_style()

    # Target for contour computation
    target = GMMTarget(
        dim=2,
        n_modes=4,
        arrangement="grid",
        separation=6.0,
        component_std=1.0,
    )

    metrics = load_metrics()
    particles = load_particles()

    # Figure 1: scatter panels
    fig1 = make_scatter_figure(target, particles)
    path1 = savefig_paper(fig1, "gmm_2d_4_scatter")
    print(f"Saved: {path1}")

    # Figure 2: convergence curves
    fig2 = make_convergence_figure(metrics)
    path2 = savefig_paper(fig2, "gmm_2d_4_convergence")
    print(f"Saved: {path2}")

    plt.show()


if __name__ == "__main__":
    main()
