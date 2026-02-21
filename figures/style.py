"""Plot aesthetics for ETD figures.

Defines the crimson palette, NeurIPS-sized layouts, and reusable
plotting helpers.  Call :func:`setup_style` once before creating
any figures.
"""

import os
from typing import Callable, Optional

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

CRIMSON = "#DC143C"
DARK_CRIMSON = "#8B0A1A"
LIGHT_CRIMSON = "#F08080"
STEEL_BLUE = "#4682B4"
SLATE_GRAY = "#708090"
TEAL = "#2E8B8B"

CRIMSON_SEQ = [
    "#F5C6CB",  # lightest
    "#F08080",
    "#DC143C",
    "#B22222",
    "#8B0A1A",  # darkest
]

ALGO_COLORS = {
    "ETD": CRIMSON,
    "SDD": STEEL_BLUE,
    "SVGD": TEAL,
    "MPPI": SLATE_GRAY,
    "EKS": DARK_CRIMSON,
}

# ---------------------------------------------------------------------------
# NeurIPS sizing
# ---------------------------------------------------------------------------

COL_WIDTH = 3.25    # inches — single NeurIPS column
FULL_WIDTH = 6.75   # inches — full NeurIPS page width


# ---------------------------------------------------------------------------
# Style setup
# ---------------------------------------------------------------------------

def setup_style() -> None:
    """Configure matplotlib rcParams for paper figures.

    Sets font sizes, white background, no grid, top/right spines off,
    and publication-quality DPI.
    """
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


# ---------------------------------------------------------------------------
# Reusable plotting helpers
# ---------------------------------------------------------------------------

def plot_contours(
    ax: plt.Axes,
    log_prob_fn: Callable,
    xlim: tuple,
    ylim: tuple,
    n_grid: int = 200,
    levels: int = 6,
) -> None:
    """Draw unfilled contour lines from a log-density on a 2D grid.

    Uses quantile-based level selection for robustness across targets.

    Args:
        ax: Matplotlib axes.
        log_prob_fn: Callable taking ``(N, 2)`` array, returning ``(N,)``.
        xlim: ``(xmin, xmax)`` for the grid.
        ylim: ``(ymin, ymax)`` for the grid.
        n_grid: Grid resolution per axis.
        levels: Number of contour levels.
    """
    xs = np.linspace(xlim[0], xlim[1], n_grid)
    ys = np.linspace(ylim[0], ylim[1], n_grid)
    Xg, Yg = np.meshgrid(xs, ys)
    grid_pts = jnp.array(np.stack([Xg.ravel(), Yg.ravel()], axis=-1))

    log_vals = np.array(log_prob_fn(grid_pts)).reshape(n_grid, n_grid)

    # Quantile-based levels for robustness
    finite_vals = log_vals[np.isfinite(log_vals)]
    if len(finite_vals) == 0:
        return
    quantiles = np.linspace(0.05, 0.95, levels)
    level_vals = np.quantile(finite_vals, quantiles)

    ax.contour(
        Xg, Yg, log_vals,
        levels=level_vals,
        colors=SLATE_GRAY,
        linewidths=0.8,
        alpha=0.6,
    )


def plot_particles(
    ax: plt.Axes,
    positions: jnp.ndarray,
    color: str = CRIMSON,
    s: float = 65,
    label: Optional[str] = None,
) -> None:
    """Scatter plot of 2D particle positions.

    Args:
        ax: Matplotlib axes.
        positions: Particle positions, shape ``(N, 2)`` or ``(N, d)``
            (only first 2 dims plotted).
        color: Marker face color.
        s: Marker size.
        label: Optional legend label.
    """
    pts = np.array(positions[:, :2])
    ax.scatter(
        pts[:, 0], pts[:, 1],
        c=color,
        s=s,
        edgecolors="white",
        linewidths=0.7,
        zorder=5,
        label=label,
    )


def savefig_paper(
    fig: Figure,
    name: str,
    output_dir: str = "figures/output",
    fmt: str = "pdf",
) -> str:
    """Save a figure for the paper.

    Args:
        fig: Matplotlib figure.
        name: Filename (without extension).
        output_dir: Directory for output files.
        fmt: File format (default ``"pdf"``).

    Returns:
        Full path to the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.{fmt}")
    fig.savefig(path, bbox_inches="tight")
    return path
