"""Analyze per-seed variance of ETD-B-DV on the 8-mode ring GMM.

Produces a table of final metrics and a 2x2 diagnostic figure comparing
the best and worst seeds.
"""

import json
import numpy as np
import matplotlib.pyplot as plt

# ---- paths ----------------------------------------------------------------
RESULTS_DIR = "/Users/vpacelli/Code/etd/results/gmm-2d-8/2026-02-21_022844"
METRICS_PATH = f"{RESULTS_DIR}/metrics.json"
PARTICLES_PATH = f"{RESULTS_DIR}/particles.npz"
OUTPUT_PATH = "/Users/vpacelli/Code/etd/figures/dv_seed_analysis.png"

# ---- target geometry ------------------------------------------------------
N_MODES = 8
SEPARATION = 10.0
COMPONENT_STD = 1.0
RADIUS = SEPARATION / (2 * np.sin(np.pi / N_MODES))
ANGLES = np.linspace(0, 2 * np.pi, N_MODES, endpoint=False)
MODE_CENTERS = np.stack([RADIUS * np.cos(ANGLES), RADIUS * np.sin(ANGLES)], axis=-1)

# ---- checkpoints ----------------------------------------------------------
CHECKPOINTS = [0, 1, 5, 10, 25, 50, 100, 200, 500]
SEEDS = list(range(10))

# ---- load data -------------------------------------------------------------
with open(METRICS_PATH) as f:
    metrics = json.load(f)

particles = np.load(PARTICLES_PATH)


# ---- helpers ---------------------------------------------------------------
def get_final_metrics(algo: str) -> dict:
    """Return {seed: {metric: value}} at checkpoint 500."""
    out = {}
    for seed in SEEDS:
        key = f"seed{seed}"
        if key in metrics and algo in metrics[key]:
            out[seed] = metrics[key][algo]["500"]
    return out


def get_trajectory(algo: str, metric: str) -> dict:
    """Return {seed: [values at each checkpoint]}."""
    out = {}
    for seed in SEEDS:
        key = f"seed{seed}"
        if key in metrics and algo in metrics[key]:
            vals = []
            for cp in CHECKPOINTS:
                vals.append(metrics[key][algo][str(cp)][metric])
            out[seed] = vals
    return out


def print_table(algo: str, final: dict) -> None:
    """Print a formatted table of final metrics per seed."""
    header = f"{'Seed':>4}  {'energy_dist':>12}  {'mode_prox':>10}  {'mode_bal':>9}  {'mean_err':>10}"
    print(f"\n{'=' * len(header)}")
    print(f"  {algo}")
    print(f"{'=' * len(header)}")
    print(header)
    print("-" * len(header))

    ed_vals = []
    for seed in sorted(final.keys()):
        m = final[seed]
        ed = m["energy_distance"]
        mp = m.get("mode_proximity", float("nan"))
        mb = m.get("mode_balance", float("nan"))
        me = m["mean_error"]
        ed_vals.append(ed)
        print(f"{seed:>4}  {ed:>12.4f}  {mp:>10.4f}  {mb:>9.4f}  {me:>10.4f}")

    ed_arr = np.array(ed_vals)
    print("-" * len(header))
    print(f"{'mean':>4}  {ed_arr.mean():>12.4f}  "
          f"{'':>9}  {'':>10}")
    print(f"{'std':>4}  {ed_arr.std():>12.4f}  "
          f"{'':>9}  {'':>10}")
    print(f"{'med':>4}  {np.median(ed_arr):>12.4f}  "
          f"{'':>9}  {'':>10}")


# ---- print tables ----------------------------------------------------------
final_dv = get_final_metrics("ETD-B-DV")
final_b = get_final_metrics("ETD-B")

print_table("ETD-B-DV", final_dv)
print_table("ETD-B", final_b)

# ---- identify best / worst seeds ------------------------------------------
ed_by_seed = {s: final_dv[s]["energy_distance"] for s in final_dv}
best_seed = min(ed_by_seed, key=ed_by_seed.get)
worst_seed = max(ed_by_seed, key=ed_by_seed.get)

ed_vals = np.array(list(ed_by_seed.values()))
median_ed = np.median(ed_vals)
iqr = np.percentile(ed_vals, 75) - np.percentile(ed_vals, 25)
threshold = median_ed + 1.5 * iqr
outlier_seeds = [s for s, v in ed_by_seed.items() if v > threshold]

print(f"\nBest seed:    {best_seed}  (energy_distance = {ed_by_seed[best_seed]:.4f})")
print(f"Worst seed:   {worst_seed}  (energy_distance = {ed_by_seed[worst_seed]:.4f})")
print(f"Outlier seeds (>{threshold:.4f}): {outlier_seeds}")
print(f"Median: {median_ed:.4f}, IQR: {iqr:.4f}")

# ---- load particles --------------------------------------------------------
pts_best_dv = particles[f"seed{best_seed}__ETD-B-DV__iter500"]
pts_worst_dv = particles[f"seed{worst_seed}__ETD-B-DV__iter500"]

# ---- trajectories ----------------------------------------------------------
traj_dv = get_trajectory("ETD-B-DV", "energy_distance")
traj_b = get_trajectory("ETD-B", "energy_distance")

# ---- figure ----------------------------------------------------------------
plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

fig, axes = plt.subplots(2, 2, figsize=(7, 6))

C0 = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]  # blue
C1 = plt.rcParams["axes.prop_cycle"].by_key()["color"][1]  # orange
C2 = plt.rcParams["axes.prop_cycle"].by_key()["color"][2]  # green
C_MODE = plt.rcParams["axes.prop_cycle"].by_key()["color"][3]  # red

for row, (seed, pts, label) in enumerate([
    (best_seed, pts_best_dv, "Best"),
    (worst_seed, pts_worst_dv, "Worst"),
]):
    ax_scatter = axes[row, 0]
    ax_traj = axes[row, 1]

    # ---- scatter: particles + mode centers ----
    ax_scatter.scatter(
        pts[:, 0], pts[:, 1],
        c=C0, s=20, alpha=0.7, edgecolors="white", linewidths=0.4,
        label="ETD-B-DV particles", zorder=5,
    )
    ax_scatter.scatter(
        MODE_CENTERS[:, 0], MODE_CENTERS[:, 1],
        marker="x", c=C_MODE, s=60, linewidths=2.0, zorder=10,
        label="Mode centers",
    )
    # Draw circles at 2*std around each mode for reference
    for cx, cy in MODE_CENTERS:
        circle = plt.Circle(
            (cx, cy), 2 * COMPONENT_STD,
            fill=False, color=C_MODE, alpha=0.25, linewidth=0.8,
        )
        ax_scatter.add_patch(circle)

    ax_scatter.set_aspect("equal")
    pad = RADIUS + 4 * COMPONENT_STD
    ax_scatter.set_xlim(-pad, pad)
    ax_scatter.set_ylim(-pad, pad)
    ax_scatter.set_xlabel("$x_1$")
    ax_scatter.set_ylabel("$x_2$")
    ax_scatter.set_title(
        f"{label} seed (seed {seed})\n"
        f"mode_prox={final_dv[seed].get('mode_proximity', float('nan')):.4f}",
    )
    if row == 0:
        ax_scatter.legend(loc="upper right", framealpha=0.8)

    # ---- trajectory: energy distance over checkpoints ----
    ax_traj.plot(
        CHECKPOINTS, traj_dv[seed],
        "o-", color=C0, markersize=4, linewidth=1.5,
        label="ETD-B-DV",
    )
    ax_traj.plot(
        CHECKPOINTS, traj_b[seed],
        "s--", color=C1, markersize=4, linewidth=1.5,
        label="ETD-B",
    )
    ax_traj.set_xlabel("Iteration")
    ax_traj.set_ylabel("Energy distance")
    ax_traj.set_title(f"Seed {seed} — energy distance trajectory")
    ax_traj.set_xscale("symlog", linthresh=1)
    ax_traj.legend(loc="upper right", framealpha=0.8)

fig.suptitle(
    f"ETD-B-DV seed analysis (8-mode ring GMM)\n"
    f"Best seed={best_seed} vs Worst seed={worst_seed}",
    fontsize=11, fontweight="bold", y=1.02,
)
fig.tight_layout()
fig.savefig(OUTPUT_PATH, bbox_inches="tight")
print(f"\nFigure saved to {OUTPUT_PATH}")
