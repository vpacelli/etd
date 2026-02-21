"""NUTS reference sampler for ground-truth evaluation.

Uses NumPyro's NUTS with the target's log_prob as a potential function,
then caches results for reuse across experiments.

Usage:
    python -m experiments.nuts --target gaussian --dim 2
    python -m experiments.nuts --target funnel --dim 10 --n_samples 10000
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np

REFERENCE_DIR = Path("results/reference")


# ---------------------------------------------------------------------------
# NUTS runner
# ---------------------------------------------------------------------------

def run_nuts(
    target: object,
    n_samples: int = 10000,
    n_warmup: int = 5000,
    n_chains: int = 4,
    seed: int = 0,
) -> dict:
    """Run NUTS via NumPyro's potential_fn interface.

    Args:
        target: Target distribution (must have ``log_prob`` and ``dim``).
        n_samples: Number of post-warmup samples per chain.
        n_warmup: Number of warmup/adaptation samples.
        n_chains: Number of MCMC chains.
        seed: Random seed.

    Returns:
        Dict with keys:
        - ``"samples"``: ``(n_chains * n_samples, d)`` array.
        - ``"r_hat"``: ``(d,)`` R-hat convergence diagnostic.
        - ``"ess"``: ``(d,)`` effective sample size per dimension.
        - ``"n_divergent"``: Total number of divergent transitions.
    """
    import numpyro
    from numpyro.infer import MCMC, NUTS

    d = target.dim

    # Wrap target.log_prob as a potential function (negative log-density).
    # NumPyro expects potential_fn(theta) → scalar (not batched).
    def potential_fn(theta):
        return -target.log_prob(theta[None])[0]

    kernel = NUTS(potential_fn=potential_fn)
    mcmc = MCMC(
        kernel,
        num_warmup=n_warmup,
        num_samples=n_samples,
        num_chains=n_chains,
        progress_bar=True,
    )

    key = jax.random.PRNGKey(seed)
    init_params = jnp.zeros(d)
    mcmc.run(key, init_params=init_params)

    # Get samples: dict with a single key or direct array
    samples_dict = mcmc.get_samples(group_by_chain=True)  # (n_chains, n_samples, d)
    # NumPyro returns dict keyed by param name. With potential_fn, key is "auto"
    if isinstance(samples_dict, dict):
        # Get the single value from the dict
        samples_by_chain = next(iter(samples_dict.values()))
    else:
        samples_by_chain = samples_dict

    samples_by_chain = np.asarray(samples_by_chain)  # (n_chains, n_samples, d)

    # Flatten across chains for output
    samples_flat = samples_by_chain.reshape(-1, d)  # (n_chains * n_samples, d)

    # Compute R-hat and ESS
    r_hat = _compute_r_hat(samples_by_chain)
    ess = _compute_ess(samples_by_chain)

    # Count divergences
    extra_fields = mcmc.get_extra_fields(group_by_chain=False)
    if "diverging" in extra_fields:
        n_divergent = int(np.sum(np.asarray(extra_fields["diverging"])))
    else:
        n_divergent = 0

    return {
        "samples": samples_flat,
        "r_hat": r_hat,
        "ess": ess,
        "n_divergent": n_divergent,
    }


def _compute_r_hat(chains: np.ndarray) -> np.ndarray:
    """Split R-hat convergence diagnostic.

    Args:
        chains: ``(n_chains, n_samples, d)`` array.

    Returns:
        R-hat per dimension, ``(d,)`` array.
    """
    n_chains, n_samples, d = chains.shape

    # Split each chain in half
    half = n_samples // 2
    split_chains = np.concatenate([
        chains[:, :half, :],
        chains[:, half:2*half, :],
    ], axis=0)  # (2*n_chains, half, d)

    m = split_chains.shape[0]  # number of split chains
    n = split_chains.shape[1]  # samples per split chain

    # Per-chain means and variances
    chain_means = split_chains.mean(axis=1)  # (m, d)
    chain_vars = split_chains.var(axis=1, ddof=1)  # (m, d)

    # Between-chain variance
    overall_mean = chain_means.mean(axis=0)  # (d,)
    B = n * np.var(chain_means, axis=0, ddof=1)  # (d,)

    # Within-chain variance
    W = np.mean(chain_vars, axis=0)  # (d,)

    # R-hat
    var_hat = (1.0 - 1.0 / n) * W + B / n
    r_hat = np.sqrt(var_hat / np.maximum(W, 1e-10))

    return r_hat


def _compute_ess(chains: np.ndarray) -> np.ndarray:
    """Bulk effective sample size (simple estimate).

    Args:
        chains: ``(n_chains, n_samples, d)`` array.

    Returns:
        ESS per dimension, ``(d,)`` array.
    """
    n_chains, n_samples, d = chains.shape
    total = n_chains * n_samples

    # Flatten chains
    flat = chains.reshape(-1, d)  # (total, d)

    # Simple ESS estimate via autocorrelation at lag 1
    mean = flat.mean(axis=0)  # (d,)
    centered = flat - mean

    # Lag-0 and lag-1 autocorrelation per dimension
    var = np.sum(centered ** 2, axis=0) / total  # (d,)
    lag1 = np.sum(centered[:-1] * centered[1:], axis=0) / total  # (d,)

    rho1 = lag1 / np.maximum(var, 1e-10)

    # ESS ≈ total / (1 + 2*rho1)  (Geyer's initial positive estimate)
    ess = total / np.maximum(1.0 + 2.0 * np.abs(rho1), 1.0)

    return ess


# ---------------------------------------------------------------------------
# Convergence check
# ---------------------------------------------------------------------------

def check_convergence(result: dict) -> List[str]:
    """Return warnings if NUTS diagnostics indicate poor convergence.

    Checks:
    - R-hat > 1.01
    - ESS < 400
    - > 1% divergent transitions

    Args:
        result: Dict returned by :func:`run_nuts`.

    Returns:
        List of warning strings (empty if all OK).
    """
    warnings = []

    r_hat_max = float(np.max(result["r_hat"]))
    if r_hat_max > 1.01:
        warnings.append(f"R-hat max {r_hat_max:.4f} > 1.01")

    ess_min = float(np.min(result["ess"]))
    if ess_min < 400:
        warnings.append(f"ESS min {ess_min:.0f} < 400")

    n_total = result["samples"].shape[0]
    div_pct = result["n_divergent"] / max(n_total, 1) * 100
    if div_pct > 1.0:
        warnings.append(f"{result['n_divergent']} divergent ({div_pct:.1f}%)")

    return warnings


# ---------------------------------------------------------------------------
# Save / load reference
# ---------------------------------------------------------------------------

def _target_hash(target_name: str, params: dict) -> str:
    """Deterministic hash for caching."""
    key_str = json.dumps({"name": target_name, "params": dict(sorted(params.items()))})
    return hashlib.sha256(key_str.encode()).hexdigest()[:12]


def save_reference(
    target_name: str,
    params: dict,
    result: dict,
    force: bool = False,
) -> str:
    """Save NUTS reference to disk.

    Args:
        target_name: Target name string.
        params: Target parameters dict.
        result: Dict from :func:`run_nuts`.
        force: Overwrite existing reference.

    Returns:
        Path to saved file.
    """
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    h = _target_hash(target_name, params)
    path = REFERENCE_DIR / f"{target_name}_{h}.npz"

    if path.exists() and not force:
        raise FileExistsError(f"Reference already exists: {path}. Use force=True.")

    np.savez(
        str(path),
        samples=result["samples"],
        r_hat=result["r_hat"],
        ess=result["ess"],
        n_divergent=np.array(result["n_divergent"]),
    )

    return str(path)


def load_reference(target_name: str, params: dict) -> Optional[np.ndarray]:
    """Load cached NUTS reference samples.

    Args:
        target_name: Target name string.
        params: Target parameters dict.

    Returns:
        Reference samples ``(n_total, d)`` or None if not cached.
    """
    h = _target_hash(target_name, params)
    path = REFERENCE_DIR / f"{target_name}_{h}.npz"

    if not path.exists():
        return None

    data = np.load(str(path))
    return data["samples"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NUTS reference sampler")
    parser.add_argument("--target", required=True, help="Target name")
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--n_warmup", type=int, default=5000)
    parser.add_argument("--n_chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")

    # Extra target params
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--prior_std", type=float, default=5.0)

    args = parser.parse_args()

    from etd.targets import get_target

    # Build target params — logistic gets dim from the dataset, not CLI
    target_params: Dict[str, Any] = {}
    if args.target == "logistic":
        if args.dataset:
            target_params["dataset"] = args.dataset
        target_params["prior_std"] = args.prior_std
    else:
        target_params["dim"] = args.dim

    target = get_target(args.target, **target_params)

    print(f"Running NUTS on {args.target} (d={target.dim})")
    print(f"  n_samples={args.n_samples}, n_warmup={args.n_warmup}, n_chains={args.n_chains}")

    result = run_nuts(
        target,
        n_samples=args.n_samples,
        n_warmup=args.n_warmup,
        n_chains=args.n_chains,
        seed=args.seed,
    )

    # Check convergence
    warnings = check_convergence(result)
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("Convergence OK")

    print(f"  R-hat: max={float(np.max(result['r_hat'])):.4f}")
    print(f"  ESS: min={float(np.min(result['ess'])):.0f}")
    print(f"  Divergent: {result['n_divergent']}")

    # Save
    path = save_reference(args.target, target_params, result, force=args.force)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
