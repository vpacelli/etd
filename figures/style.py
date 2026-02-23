"""Plot aesthetics for ETD figures.

Defines the crimson palette, NeurIPS-sized layouts, and reusable
plotting helpers.  Call :func:`setup_style` once before creating
any figures.
"""

import json
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
    # ETD variants (crimson family)
    "ETD": CRIMSON,
    "ETD-B": CRIMSON,
    "ETD-UB": "#A10E2B",
    "ETD-B-Maha": "#E89DA3",
    "ETD-SR": "#E89DA3",
    "ETD-B-SF": "#B22222",
    # LRET variants (same coupling types, different name)
    "LRET-B": "#B22222",
    "LRET-B-W": "#E89DA3",
    # Mutation variants
    "ETD-B+MALA": "#8B0A1A",
    "LRET-B+MALA": "#7A2E2E",
    # Other algorithms
    "SDD": STEEL_BLUE,
    "SVGD": TEAL,
    "ULA": SLATE_GRAY,
    "MALA": SLATE_GRAY,
    "MPPI": "#6A5ACD",      # Slate blue — distinct from ULA's gray
    "EKS": DARK_CRIMSON,
}

# Deterministic linestyle cycle for algorithms sharing similar colors.
ALGO_LINESTYLES = {
    "ETD-B": "-",
    "LRET-B": "--",
    "ETD-B+MALA": "-.",
    "LRET-B+MALA": (0, (3, 1, 1, 1, 1, 1)),  # dash-dot-dot
    "LRET-B-W": ":",
}

# Family-based palettes for automatic color assignment.
# Within each family, algorithms are assigned consecutive palette colors
# in YAML order.
FAMILY_PALETTES = {
    "etd": [
        "#DC143C",  # crimson (primary)
        "#A10E2B",  # dark crimson
        "#B22222",  # firebrick
        "#E89DA3",  # light crimson
        "#8B0A1A",  # deepest
        "#F08080",  # light coral
    ],
    "sdd": [
        "#4682B4",  # steel blue (primary)
        "#2C5F8A",  # dark steel
        "#6A9FD0",  # medium blue
        "#1A3D5C",  # navy
        "#87CEEB",  # sky blue
    ],
    "baseline": [
        "#708090",  # slate gray
        "#2E8B8B",  # teal
        "#6A5ACD",  # slate blue
        "#8B4513",  # saddle brown
        "#556B2F",  # dark olive
        "#9370DB",  # medium purple
    ],
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


def frame_panel(ax: plt.Axes, linewidth: float = 0.5) -> None:
    """Style an axes as a thin-bordered frame with no ticks or labels.

    Turns on all four spines and removes ticks. Useful for contour /
    particle panels where coordinates are not meaningful.

    Args:
        ax: Matplotlib axes.
        linewidth: Spine line width in points.
    """
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(linewidth)
    ax.set_xticks([])
    ax.set_yticks([])


def ref_line(
    ax: plt.Axes,
    value: float,
    orientation: str = "horizontal",
    **kwargs,
) -> None:
    """Draw a reference line (e.g., ideal share, zero baseline).

    Defaults are tuned for visibility over colored fills: dark gray,
    solid weight 0.8, zorder behind data but above fills.

    Args:
        ax: Matplotlib axes.
        value: Position on the y-axis (horizontal) or x-axis (vertical).
        orientation: ``"horizontal"`` or ``"vertical"``.
        **kwargs: Overrides passed to ``ax.axhline`` / ``ax.axvline``.
    """
    defaults = dict(color="#444444", linewidth=0.8, linestyle="--", zorder=1)
    defaults.update(kwargs)
    if orientation == "vertical":
        ax.axvline(value, **defaults)
    else:
        ax.axhline(value, **defaults)


import math


def facet_grid(
    n_panels: int,
    *,
    total_width: float = FULL_WIDTH,
    panel_size: float = 1.6,
    sharex: bool = True,
    sharey: bool = True,
    square: bool = True,
) -> tuple:
    """Create a figure with wrapped rows of equal-sized panels.

    Computes a grid that keeps each panel close to *panel_size* inches,
    wrapping into multiple rows when needed.  Unused trailing axes are
    hidden automatically.

    Args:
        n_panels: Number of axes needed.
        total_width: Figure width in inches (default ``FULL_WIDTH``).
        panel_size: Target panel width/height in inches.
        sharex: Share x-axes within rows.
        sharey: Share y-axes across panels.
        square: If True, panel height equals panel width.

    Returns:
        ``(fig, axes)`` where *axes* is a flat array of length *n_panels*
        (unused axes already hidden).
    """
    ncols = max(1, min(n_panels, int(total_width / panel_size)))
    nrows = math.ceil(n_panels / ncols)
    panel_w = total_width / ncols
    panel_h = panel_w if square else panel_size
    fig_h = panel_h * nrows

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(total_width, fig_h),
        sharex=sharex, sharey=sharey,
        constrained_layout=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()

    # Hide unused trailing axes
    for i in range(n_panels, len(axes_flat)):
        axes_flat[i].set_visible(False)

    return fig, axes_flat[:n_panels]


def metric_facet_grid(
    n_metrics: int,
    n_algos: int,
    *,
    total_width: float = FULL_WIDTH,
    panel_size: float = 1.6,
    panel_height: float = 1.2,
) -> tuple:
    """Create a wrapped grid for (metric rows) x (algorithm columns).

    Like :func:`facet_grid` but preserves a 2D structure: algorithms
    wrap into column groups, and each metric gets its own row within
    each group.  Y-axes are shared across all panels for the same
    metric, so scales stay comparable.

    Args:
        n_metrics: Number of metric rows (e.g., 2 for energy + mean error).
        n_algos: Number of algorithm columns.
        total_width: Figure width in inches.
        panel_size: Target panel width in inches.
        panel_height: Panel height in inches.

    Returns:
        ``(fig, axes_map)`` where *axes_map* is a 2D array of shape
        ``(n_metrics, n_algos)`` with one axis per (metric, algorithm) pair.
        Unused grid cells are hidden.
    """
    ncols = max(1, min(n_algos, int(total_width / panel_size)))
    algo_nrows = math.ceil(n_algos / ncols)
    total_rows = n_metrics * algo_nrows

    fig, all_axes = plt.subplots(
        total_rows, ncols,
        figsize=(total_width, panel_height * total_rows),
        squeeze=False,
        constrained_layout=True,
    )

    # Build a (n_metrics, n_algos) map into the raw grid
    axes_map = np.empty((n_metrics, n_algos), dtype=object)
    for m in range(n_metrics):
        for a in range(n_algos):
            row = m + (a // ncols) * n_metrics
            col = a % ncols
            axes_map[m, a] = all_axes[row, col]

    # Share y within each metric, share x across all
    for m in range(n_metrics):
        anchor = axes_map[m, 0]
        for a in range(1, n_algos):
            axes_map[m, a].sharey(anchor)
    for row in range(total_rows):
        anchor = all_axes[row, 0]
        for col in range(1, ncols):
            all_axes[row, col].sharex(anchor)

    # Hide unused trailing cells
    for a in range(n_algos, algo_nrows * ncols):
        for m in range(n_metrics):
            row = m + (a // ncols) * n_metrics
            col = a % ncols
            all_axes[row, col].set_visible(False)

    return fig, axes_map


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


# ---------------------------------------------------------------------------
# Display metadata resolution
# ---------------------------------------------------------------------------

def _infer_family(meta: dict) -> str:
    """Infer family from label prefix and baseline flag.

    Args:
        meta: Dict with keys ``label``, ``family``, ``is_baseline``.

    Returns:
        Family string: ``"etd"``, ``"sdd"``, or ``"baseline"``.
    """
    if meta.get("family"):
        return meta["family"]
    label = meta.get("label", "")
    if label.startswith("ETD") or label.startswith("LRET"):
        return "etd"
    if label.startswith("SDD"):
        return "sdd"
    if meta.get("is_baseline"):
        return "baseline"
    return "etd"


def resolve_algo_styles(algo_display_meta: list[dict]) -> dict[str, dict]:
    """Resolve display metadata into concrete plot styles.

    Assigns family-palette colors by YAML order within each family.
    Explicit ``color`` overrides palette. ``ALGO_COLORS`` is a
    secondary fallback for known labels.

    Args:
        algo_display_meta: List of dicts (YAML order), each with keys:
            ``label``, ``family`` (or None), ``color`` (or None),
            ``linestyle``, ``group``, ``is_baseline``.

    Returns:
        Ordered dict ``{label: {family, color, linestyle, group}}``.
    """
    # Pass 1: infer families
    families = [_infer_family(m) for m in algo_display_meta]

    # Pass 2: assign colors in YAML order, per-family counter
    family_counters: dict[str, int] = {}
    result: dict[str, dict] = {}

    for meta, family in zip(algo_display_meta, families):
        label = meta["label"]
        linestyle = meta.get("linestyle", "-")
        group = meta.get("group")

        # Color priority: explicit > ALGO_COLORS > family palette
        explicit_color = meta.get("color")
        if explicit_color:
            color = explicit_color
        elif label in ALGO_COLORS:
            color = ALGO_COLORS[label]
        else:
            palette = FAMILY_PALETTES.get(family, FAMILY_PALETTES["etd"])
            idx = family_counters.get(family, 0)
            color = palette[idx % len(palette)]
            family_counters[family] = idx + 1

        result[label] = {
            "family": family,
            "color": color,
            "linestyle": linestyle,
            "group": group,
        }

    return result


def format_iteration_axis(
    ax: plt.Axes,
    checkpoints: list | np.ndarray,
    *,
    axis: str = "x",
) -> None:
    """Format an iteration axis with log-scale and clean tick labels.

    Replaces the categorical x-position pattern (``X_POS = arange(...)``)
    with a proper log-scale axis using the actual checkpoint values.
    Checkpoint 0 is mapped to 1 so log-scale works.

    Args:
        ax: Matplotlib axes.
        checkpoints: Sorted list of iteration checkpoints (e.g., [0, 10, 50, ...]).
        axis: ``"x"`` or ``"y"`` — which axis to format.
    """
    from matplotlib.ticker import ScalarFormatter

    ckpts = np.asarray(checkpoints, dtype=float)
    # Show at most ~6 ticks to avoid crowding
    if len(ckpts) > 6:
        step = max(1, len(ckpts) // 5)
        idx = sorted(set(list(range(0, len(ckpts), step)) + [len(ckpts) - 1]))
        tick_vals = ckpts[idx]
    else:
        tick_vals = ckpts

    if axis == "x":
        ax.set_xscale("log")
        ax.set_xticks(tick_vals)
        fmt = ScalarFormatter()
        fmt.set_scientific(False)
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.tick_params(axis="x", which="minor", length=0)
    else:
        ax.set_yscale("log")
        ax.set_yticks(tick_vals)
        fmt = ScalarFormatter()
        fmt.set_scientific(False)
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
        ax.tick_params(axis="y", which="minor", length=0)


def load_display_metadata(results_dir: str) -> dict[str, dict]:
    """Load metadata.json from a results directory.

    Args:
        results_dir: Path to the results directory.

    Returns:
        Dict ``{label: {family, color, linestyle, group}}`` or empty dict
        if no metadata.json exists.
    """
    path = os.path.join(results_dir, "metadata.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)
