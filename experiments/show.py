"""Recompute and display metrics from saved experiment results.

Loads particles from a results directory, reconstructs the target from
the saved config, and (re)computes all requested metrics — including any
added after the original run.

Usage:
    python -m experiments.show results/gmm-10d-5-cost-variants/2026-02-21_092309/
    python -m experiments.show results/gmm-10d-5-cost-variants/2026-02-21_092309/ --metrics sliced_wasserstein mode_proximity
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from rich.console import Console

from experiments._display import build_summary_table_flat

from etd.diagnostics.metrics import (
    energy_distance,
    mean_error,
    mode_balance,
    mode_proximity,
    sliced_wasserstein,
)
from etd.targets import get_target

try:
    from experiments.nuts import load_reference
except ImportError:
    load_reference = None

console = Console()

# Same dispatch table as run.py — add new metrics here.
METRIC_DISPATCH = {
    "energy_distance": lambda p, t, ref: float(energy_distance(p, ref))
    if ref is not None else float("nan"),
    "sliced_wasserstein": lambda p, t, ref: float(
        sliced_wasserstein(p, ref, jax.random.key(0))
    ) if ref is not None else float("nan"),
    "mode_proximity": lambda p, t, ref: float(
        mode_proximity(
            p, t.means,
            component_std=getattr(t, "component_std", 1.0),
            dim=p.shape[1],
        )
    ) if hasattr(t, "means") else float("nan"),
    "mode_balance": lambda p, t, ref: float(
        mode_balance(p, t.means)
    ) if hasattr(t, "means") else float("nan"),
    "mean_error": lambda p, t, ref: float(mean_error(p, t.mean))
    if hasattr(t, "mean") else float("nan"),
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_results(results_dir: str):
    """Load config, metrics, and particles from a results directory.

    Args:
        results_dir: Path to a timestamped results directory containing
            ``config.yaml``, ``metrics.json``, and ``particles.npz``.

    Returns:
        Tuple of ``(cfg, saved_metrics, particles_npz)``.
    """
    with open(os.path.join(results_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(results_dir, "metrics.json")) as f:
        saved_metrics = json.load(f)
    particles_npz = np.load(os.path.join(results_dir, "particles.npz"))
    return cfg, saved_metrics, particles_npz


def reconstruct_target(cfg: dict):
    """Reconstruct the target distribution from a saved config.

    Args:
        cfg: Parsed config dict with ``experiment.target``.

    Returns:
        Target instance.
    """
    target_cfg = cfg["experiment"]["target"]
    return get_target(target_cfg["type"], **target_cfg.get("params", {}))


# ---------------------------------------------------------------------------
# Reference data lookup chain
# ---------------------------------------------------------------------------

def _load_reference(
    args,
    cfg: dict,
    target,
    results_dir: str,
) -> Optional[jnp.ndarray]:
    """Load reference samples using a prioritized lookup chain.

    Tries, in order:
      1. ``--reference`` CLI path (explicit override)
      2. ``reference.npz`` bundled in the results directory
      3. NUTS cache via ``load_reference()``
      4. ``target.sample()`` (synthetic targets)
      5. None — prints which metrics will be unavailable

    Args:
        args: Parsed CLI arguments (with ``.reference`` and ``.n_ref``).
        cfg: Experiment config dict.
        target: Reconstructed target distribution.
        results_dir: Path to the results directory.

    Returns:
        Reference samples ``(M, d)`` or None.
    """
    # 1. Explicit --reference flag
    if args.reference is not None:
        ref_path = args.reference
        if not os.path.exists(ref_path):
            console.print(f"[red]Error: --reference path not found: {ref_path}[/]")
            return None
        data = np.load(ref_path)
        ref = jnp.asarray(data["samples"])
        console.print(f"Reference: loaded from --reference ({ref_path})")
        return ref

    # 2. Bundled reference.npz in results directory
    bundled_path = os.path.join(results_dir, "reference.npz")
    if os.path.exists(bundled_path):
        data = np.load(bundled_path)
        ref = jnp.asarray(data["samples"])
        console.print(f"Reference: loaded from bundled reference.npz")
        return ref

    # 3. NUTS cache
    if load_reference is not None:
        try:
            target_cfg = cfg.get("experiment", {}).get("target", {})
            target_name = target_cfg.get("type", "")
            target_params = target_cfg.get("params", {})
            ref = load_reference(target_name, target_params)
            if ref is not None:
                console.print(f"Reference: loaded from NUTS cache")
                return jnp.asarray(ref)
        except Exception:
            pass

    # 4. Synthetic sampling
    if hasattr(target, "sample"):
        ref = target.sample(jax.random.PRNGKey(99999), args.n_ref)
        console.print(f"Reference: sampled from target ({args.n_ref} samples)")
        return ref

    # 5. None — explain what's missing
    console.print(
        "[yellow]No reference data available. "
        "Metrics requiring reference samples (energy_distance, "
        "sliced_wasserstein, mean_rmse, variance_ratio_ref) will show N/A.\n"
        "  Hint: re-run the experiment to bundle reference.npz, "
        "or use --reference <path>.[/]"
    )
    return None


# ---------------------------------------------------------------------------
# Metric recomputation
# ---------------------------------------------------------------------------

def recompute_metrics(
    particles_npz,
    saved_metrics: dict,
    target,
    ref_data: Optional[jnp.ndarray],
    metrics_list: List[str],
    checkpoint: Optional[int] = None,
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Recompute metrics from saved particles.

    Args:
        particles_npz: Loaded ``.npz`` archive.
        saved_metrics: Original ``metrics.json`` structure (for key discovery).
        target: Reconstructed target distribution.
        ref_data: Reference samples ``(M, d)`` or None.
        metrics_list: Metrics to compute.
        checkpoint: If given, only compute at this checkpoint.
            Defaults to the final checkpoint.

    Returns:
        ``{seed_int → {algo → {metric → val}}}`` at the chosen checkpoint.
    """
    results = {}

    for seed_key, algos in saved_metrics.items():
        seed_int = int(seed_key.replace("seed", ""))
        results[seed_int] = {}

        for algo, ckpts in algos.items():
            ckpt_keys = sorted(int(k) for k in ckpts.keys())
            ckpt = checkpoint if checkpoint is not None else max(ckpt_keys)
            npz_key = f"{seed_key}__{algo}__iter{ckpt}"

            if npz_key not in particles_npz:
                console.print(
                    f"[yellow]Warning: {npz_key} not in particles.npz, skipping[/]"
                )
                continue

            particles = jnp.array(particles_npz[npz_key])
            row = {}
            for metric in metrics_list:
                fn = METRIC_DISPATCH.get(metric)
                if fn is None:
                    # Fall back to saved value if available
                    row[metric] = ckpts.get(str(ckpt), {}).get(metric, float("nan"))
                else:
                    row[metric] = fn(particles, target, ref_data)

            # Carry over non-recomputable metrics (e.g. coupling_ess, wall_clock)
            saved_ckpt = ckpts.get(str(ckpt), {})
            for k, v in saved_ckpt.items():
                if k not in row and k not in metrics_list:
                    row[k] = v

            results[seed_int][algo] = row

    return results


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_summary(
    results: dict,
    metrics_list: List[str],
    wall_clocks: Optional[dict] = None,
) -> None:
    """Print a Rich summary table of seed-averaged results.

    Args:
        results: ``{seed → {algo → {metric → val}}}``.
        metrics_list: Ordered list of metrics to display.
        wall_clocks: Optional ``{(seed, algo) → seconds}`` for timing column.
    """
    table = build_summary_table_flat(results, metrics_list)
    console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Recompute and display metrics from saved results",
    )
    parser.add_argument(
        "results_dir",
        help="Path to a timestamped results directory",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=None,
        help="Metrics to compute (default: all available in METRIC_DISPATCH)",
    )
    parser.add_argument(
        "--checkpoint", type=int, default=None,
        help="Checkpoint to evaluate (default: final)",
    )
    parser.add_argument(
        "--n-ref", type=int, default=5000,
        help="Number of reference samples (default: 5000)",
    )
    parser.add_argument(
        "--reference", default=None,
        help="Path to reference samples .npz (key: 'samples')",
    )
    args = parser.parse_args()

    # --- Load ---
    cfg, saved_metrics, particles_npz = load_results(args.results_dir)
    target = reconstruct_target(cfg)

    exp = cfg["experiment"]
    console.print(
        f"Loaded results for [bold]{exp['name']}[/] "
        f"(target={exp['target']['type']}, d={target.dim})"
    )

    # --- Metrics to compute ---
    metrics_list = args.metrics or list(METRIC_DISPATCH.keys())
    console.print(f"Metrics: {', '.join(metrics_list)}")

    # --- Reference data (prioritized lookup chain) ---
    ref_data = _load_reference(args, cfg, target, args.results_dir)

    # --- Recompute ---
    results = recompute_metrics(
        particles_npz, saved_metrics, target, ref_data,
        metrics_list, args.checkpoint,
    )

    # --- Display ---
    print_summary(results, metrics_list)


if __name__ == "__main__":
    main()
