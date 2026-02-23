"""Visual sanity check: 5-panel GMM convergence figure.

Usage:
    python scripts/visual_sanity.py

Produces ``figures/output/gmm_sanity_check.pdf`` showing particle
evolution at iterations {1, 50, 100, 200, 500} overlaid on the
GMM contour.
"""

import jax
import jax.numpy as jnp

from etd.diagnostics.metrics import mode_proximity
from etd.step import init, step
from etd.targets.gmm import GMMTarget
from etd.types import (
    CouplingConfig,
    ETDConfig,
    ProposalConfig,
)
from figures.style import (
    FULL_WIDTH,
    plot_contours,
    plot_particles,
    savefig_paper,
    setup_style,
)


def main():
    setup_style()
    import matplotlib.pyplot as plt

    # --- Setup ---
    target = GMMTarget(dim=2, n_modes=4, arrangement="grid", separation=6.0)
    config = ETDConfig(
        n_particles=100,
        n_iterations=500,
        epsilon=0.1,
        proposal=ProposalConfig(count=25, alpha=0.05),
        coupling=CouplingConfig(type="balanced"),
    )

    key = jax.random.PRNGKey(42)
    k_init, k_run = jax.random.split(key)
    init_positions = jax.random.normal(k_init, (100, 2)) * 5.0
    state = init(k_init, target, config, init_positions=init_positions)

    # --- Run and collect snapshots ---
    checkpoints = {1, 50, 100, 200, 500}
    snapshots = {}

    for i in range(1, config.n_iterations + 1):
        k_run, k_step = jax.random.split(k_run)
        state, info = step(k_step, state, target, config)

        if i in checkpoints:
            prox = mode_proximity(
                state.positions, target.means,
                component_std=target.component_std, dim=target.dim,
            )
            print(
                f"  iter {i:4d} | "
                f"sinkhorn_iters={int(info['sinkhorn_iters']):3d} | "
                f"mode_proximity={float(prox):.4f}"
            )
            snapshots[i] = state.positions.copy()

    # --- Plot ---
    fig, axes = plt.subplots(1, 5, figsize=(FULL_WIDTH * 2, 2.5))
    xlim = (-6, 6)
    ylim = (-6, 6)

    for ax, t in zip(axes, sorted(snapshots.keys())):
        plot_contours(ax, target.log_prob, xlim, ylim, levels=8)
        plot_particles(ax, snapshots[t], s=20)
        ax.set_title(f"t = {t}")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")

    fig.suptitle("ETD on 4-mode GMM (balanced Sinkhorn)", fontsize=11)
    fig.tight_layout()

    path = savefig_paper(fig, "gmm_sanity_check")
    print(f"\nSaved to {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
