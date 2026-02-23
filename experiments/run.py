"""Experiment runner for ETD and baselines.

Loads a YAML config, expands sweeps, runs algorithms on a target,
records metrics at checkpoints, and saves results.

Usage:
    python -m experiments.run experiments/configs/gmm/2d_4.yaml
    python -m experiments.run experiments/configs/gmm/2d_4.yaml --debug   # no JIT
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import functools
import itertools
import json
import os
import time
import warnings
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
from etd.diagnostics.metrics import (
    energy_distance, mean_error, mean_rmse, mode_balance,
    mode_proximity, sliced_wasserstein, variance_ratio_vs_reference,
)
from etd.schedule import Schedule
from etd.step import init as etd_init, step as etd_step
from etd.targets import get_target
from etd.types import (
    CostConfig,
    CouplingConfig,
    ETDConfig,
    FeedbackConfig,
    MutationConfig,
    PreconditionerConfig,
    ProposalConfig,
    UpdateConfig,
)

console = Console()

# ---------------------------------------------------------------------------
# JIT strategy
# ---------------------------------------------------------------------------

DEBUG = os.environ.get("ETD_DEBUG", "0") == "1"


def maybe_jit(fn, **kwargs):
    """Wrap ``fn`` with ``jax.jit`` unless debug mode is active."""
    return fn if DEBUG else jax.jit(fn, **kwargs)


# ---------------------------------------------------------------------------
# Scan runner infrastructure
# ---------------------------------------------------------------------------

def _compute_segments(
    checkpoints: List[int],
    n_iterations: int,
) -> List[Tuple[int, int, int]]:
    """Convert checkpoint list to contiguous scan segments.

    Each segment is ``(start, n_steps, checkpoint)`` where
    ``start`` is the 1-based iteration index of the first step,
    ``n_steps`` is how many steps to run, and ``checkpoint`` is
    the iteration at which to record metrics (end of segment).

    Args:
        checkpoints: Sorted iteration numbers at which to record.
        n_iterations: Total iterations.

    Returns:
        List of ``(start, n_steps, checkpoint)`` tuples.
    """
    # Filter out 0 (handled separately) and anything beyond n_iterations
    ckpts = sorted(c for c in checkpoints if 0 < c <= n_iterations)
    if not ckpts:
        return []

    segments = []
    prev = 0
    for c in ckpts:
        n_steps = c - prev
        if n_steps > 0:
            segments.append((prev + 1, n_steps, c))
        prev = c
    return segments


def _make_jit_scan(
    step_fn: Callable,
    target: object,
    config: Any,
) -> Callable:
    """Build a JIT-compiled scan runner for a given step_fn/target/config.

    The returned function runs ``n_steps`` iterations of ``step_fn`` fused
    into a single ``jax.lax.scan``, with buffer donation for in-place state
    reuse and ``fold_in`` keying for deterministic PRNG.

    Args:
        step_fn: Algorithm step function ``(key, state, target, config) -> (state, info)``.
        target: Target distribution (captured by closure, static for JIT).
        config: Algorithm config (captured by closure, static for JIT).

    Returns:
        ``run_segment(state, key_base, start_iter, n_steps) -> final_state``
    """
    # Suppress JAX warning about donating int32 fields (e.g. state.step)
    warnings.filterwarnings(
        "ignore", message=".*Some donated buffers were not usable.*"
    )

    @functools.partial(jax.jit, static_argnums=(2, 3), donate_argnums=(0,))
    def run_segment(state, key_base, start_iter, n_steps):
        """Run n_steps of step_fn as a fused scan.

        Args:
            state: Algorithm state (donated -- buffer reused in-place).
            key_base: Base PRNG key; per-step keys derived via fold_in.
            start_iter: 1-based iteration index of the first step.
            n_steps: Number of steps to run (static).

        Returns:
            Tuple ``(final_state, last_info)`` where ``last_info`` is the
            info dict from the final step (for checkpoint diagnostics).
        """
        def scan_body(carry, t_offset):
            t = start_iter + t_offset
            key_step = jax.random.fold_in(key_base, t)
            new_state, info = step_fn(key_step, carry, target, config)
            return new_state, info

        final_state, stacked_info = jax.lax.scan(
            scan_body, state, jnp.arange(n_steps),
        )
        # Extract last step's info from the stacked outputs
        last_info = jax.tree.map(lambda x: x[-1], stacked_info)
        return final_state, last_info

    return run_segment


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

def _collect_sweep_axes(entry: dict, prefix: str = ""):
    """Collect sweepable (list-valued) parameters, including nested dicts.

    Args:
        entry: Config dict (may contain nested dicts, e.g. ``cost``).
        prefix: Dot-separated key prefix for nested dicts.

    Returns:
        List of ``(dotted_key, values_list)`` pairs.
    """
    _SWEEP_EXCLUDE = {"label", "sublabel", "display", "enabled"}
    axes = []
    for k, v in entry.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict) and k != "display":
            axes.extend(_collect_sweep_axes(v, prefix=full_key))
        elif isinstance(v, list) and k not in _SWEEP_EXCLUDE:
            axes.append((full_key, v))
    return axes


def _set_nested(d: dict, dotted_key: str, value):
    """Set a value in a nested dict using a dotted key path.

    Intermediate dicts are copied so the original is not mutated.

    Args:
        d: Root dict.
        dotted_key: Key path like ``"cost.c"``.
        value: Value to set.
    """
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        cur[k] = dict(cur[k])  # shallow copy intermediate
        cur = cur[k]
    cur[keys[-1]] = value


def expand_algo_sweeps(entry: dict) -> List[dict]:
    """Expand list-valued parameters into a Cartesian product of configs.

    Supports nested dicts (e.g. ``cost: {type: imq, c: [0.5, 1.0]}``).

    Args:
        entry: A single algorithm entry from the YAML config.

    Returns:
        List of concrete entries (one per sweep point).
    """
    axes = _collect_sweep_axes(entry)

    if not axes:
        return [entry]

    sweep_keys = [k for k, _ in axes]
    sweep_vals = [v for _, v in axes]

    expanded = []
    for combo in itertools.product(*sweep_vals):
        new_entry = dict(entry)
        # Deep-copy any nested dicts that contain sweep axes
        for k, v in entry.items():
            if isinstance(v, dict):
                new_entry[k] = dict(v)
        for k, v in zip(sweep_keys, combo):
            if "." in k:
                _set_nested(new_entry, k, v)
            else:
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
    abbrevs = {
        "epsilon": "eps",
        "stepsize": "h",
        "temperature": "T",
        "bandwidth": "bw",
        # Nested sub-config abbreviations
        "proposal.alpha": "alpha",
        "proposal.count": "M",
        "proposal.clip_score": "clip",
        "coupling.tolerance": "tol",
        "coupling.rho": "rho",
        "update.damping": "damp",
        "mutation.steps": "mut.steps",
        "mutation.stepsize": "mut.h",
        "feedback.weight": "dv",
    }

    sweep_axes = _collect_sweep_axes(original)
    suffixes = []
    for dotted_key, _ in sweep_axes:
        # Use the leaf key name for abbreviation and value lookup
        abbrev = abbrevs.get(dotted_key, dotted_key.split(".")[-1])
        # Resolve value from the expanded entry
        val = entry
        for part in dotted_key.split("."):
            val = val[part]
        suffixes.append(f"{abbrev}={val}")

    if suffixes:
        return f"{base_label}_{'_'.join(suffixes)}"
    return base_label


# ---------------------------------------------------------------------------
# Sub-config expansion helpers
# ---------------------------------------------------------------------------

# Parameters that are NOT passed to algorithm configs (meta or shared)
_META_KEYS = {"label", "type", "method", "sublabel", "display", "enabled"}


def _expand_proposal(raw) -> ProposalConfig:
    """Expand proposal config from YAML value.

    Accepts:
    - str: ``"score"`` or ``"score_free"`` -> defaults
    - dict: ``{type: "score", count: 25, alpha: 0.05, ...}``
    - None: defaults

    Args:
        raw: YAML value for ``proposal``.

    Returns:
        :class:`ProposalConfig` instance.
    """
    if raw is None:
        return ProposalConfig()
    if isinstance(raw, str):
        return ProposalConfig(type=raw)
    if isinstance(raw, ProposalConfig):
        return raw
    # dict
    d = dict(raw)
    return ProposalConfig(
        type=d.pop("type", "score"),
        count=int(d.pop("count", 25)),
        alpha=float(d.pop("alpha", 0.05)),
        fdr=bool(d.pop("fdr", True)),
        sigma=float(d.pop("sigma", 0.0)),
        clip_score=float(d.pop("clip_score", 5.0)),
    )


def _expand_cost(raw) -> CostConfig:
    """Expand cost config from YAML value.

    Accepts:
    - str: ``"euclidean"`` -> defaults
    - dict: ``{type: "imq", c: 1.0, normalize: "median"}``
      Extra keys beyond type/normalize become ``params``.
    - None: defaults
    """
    if raw is None:
        return CostConfig()
    if isinstance(raw, str):
        return CostConfig(type=raw)
    if isinstance(raw, CostConfig):
        return raw
    d = dict(raw)
    cost_type = d.pop("type", "euclidean")
    normalize = str(d.pop("normalize", "median"))
    # Remaining keys are extra params (e.g. c=1.0, whiten=True)
    params = tuple(sorted(d.items())) if d else ()
    return CostConfig(type=cost_type, normalize=normalize, params=params)


def _expand_coupling(raw) -> CouplingConfig:
    """Expand coupling config from YAML value.

    Accepts:
    - str: ``"balanced"`` -> defaults
    - dict: ``{type: "balanced", tolerance: 1e-5}``
    - None: defaults
    """
    if raw is None:
        return CouplingConfig()
    if isinstance(raw, str):
        return CouplingConfig(type=raw)
    if isinstance(raw, CouplingConfig):
        return raw
    d = dict(raw)
    return CouplingConfig(
        type=d.pop("type", "balanced"),
        iterations=int(d.pop("iterations", 50)),
        tolerance=float(d.pop("tolerance", 1e-2)),
        rho=float(d.pop("rho", 1.0)),
    )


def _expand_update(raw) -> UpdateConfig:
    """Expand update config from YAML value.

    Accepts:
    - str: ``"categorical"`` -> defaults
    - dict: ``{type: "categorical", damping: 0.5}``
    - None: defaults
    """
    if raw is None:
        return UpdateConfig()
    if isinstance(raw, str):
        return UpdateConfig(type=raw)
    if isinstance(raw, UpdateConfig):
        return raw
    d = dict(raw)
    return UpdateConfig(
        type=d.pop("type", "categorical"),
        damping=float(d.pop("damping", 1.0)),
    )


def _expand_preconditioner(raw) -> PreconditionerConfig:
    """Expand preconditioner config from YAML value.

    Accepts:
    - str: ``"none"``, ``"rmsprop"``, ``"cholesky"``
    - dict: full config
    - None: defaults
    """
    if raw is None:
        return PreconditionerConfig()
    if isinstance(raw, str):
        _defaults = {
            "none": PreconditionerConfig(),
            "rmsprop": PreconditionerConfig(type="rmsprop", proposals=True),
            "cholesky": PreconditionerConfig(
                type="cholesky", proposals=True, cost=True,
            ),
        }
        return _defaults.get(raw, PreconditionerConfig())
    if isinstance(raw, PreconditionerConfig):
        return raw
    d = dict(raw)
    # Coerce types
    _BOOL_KEYS = {"proposals", "cost"}
    _FLOAT_KEYS = {"clip_score", "beta", "delta", "shrinkage", "jitter", "ema"}
    for k in _BOOL_KEYS:
        if k in d:
            d[k] = bool(d[k])
    for k in _FLOAT_KEYS:
        if k in d:
            d[k] = float(d[k])
    return PreconditionerConfig(**d)


def _expand_mutation(raw) -> MutationConfig:
    """Expand mutation config from YAML value.

    Accepts:
    - str: ``"none"``, ``"mala"``, ``"rwm"``
    - dict: full config
    - None: defaults
    """
    if raw is None:
        return MutationConfig()
    if isinstance(raw, str):
        if raw == "none":
            return MutationConfig()
        return MutationConfig(kernel=raw)
    if isinstance(raw, MutationConfig):
        return raw
    d = dict(raw)
    _BOOL_KEYS = {"cholesky"}
    _FLOAT_KEYS = {"stepsize", "clip_score"}
    _INT_KEYS = {"steps"}
    for k in _BOOL_KEYS:
        if k in d:
            d[k] = bool(d[k])
    for k in _FLOAT_KEYS:
        if k in d and d[k] is not None:
            d[k] = float(d[k])
    for k in _INT_KEYS:
        if k in d:
            d[k] = int(d[k])
    return MutationConfig(**d)


def _expand_feedback(raw) -> FeedbackConfig:
    """Expand feedback config from YAML value.

    Accepts:
    - bool: ``true`` -> enabled
    - dict: ``{enabled: true, weight: 0.5}``
    - None: defaults
    """
    if raw is None:
        return FeedbackConfig()
    if isinstance(raw, bool):
        return FeedbackConfig(enabled=raw)
    if isinstance(raw, FeedbackConfig):
        return raw
    d = dict(raw)
    return FeedbackConfig(
        enabled=bool(d.pop("enabled", False)),
        weight=float(d.pop("weight", 1.0)),
    )


# ---------------------------------------------------------------------------
# Schedule extraction
# ---------------------------------------------------------------------------

def _extract_schedules(kwargs: dict) -> tuple:
    """Extract schedule dicts from kwargs, supporting nested sub-config dicts.

    A schedule is any dict value with a ``"schedule"`` key::

        epsilon: {schedule: "linear_warmup", value: 0.1, warmup: 100}
        proposal:
          alpha: {schedule: "cosine_decay", value: 0.05, end: 0.001}

    Top-level schedules use simple keys (``"epsilon"``).
    Nested schedules use dotted keys (``"proposal.alpha"``).

    Modifies ``kwargs`` in-place: schedule dicts are replaced with
    their ``value`` field (the target/start value).

    Args:
        kwargs: Mutable dict of algorithm kwargs.

    Returns:
        Tuple of ``(dotted_key, Schedule)`` pairs.
    """
    schedules = []

    # Top-level schedules
    for k in list(kwargs):
        v = kwargs[k]
        if isinstance(v, dict) and "schedule" in v:
            v = dict(v)  # copy
            kind = v.pop("schedule")
            value = float(v.pop("value"))
            sched_kw = {}
            for sk, sv in v.items():
                if sk == "warmup":
                    sched_kw[sk] = int(sv)
                else:
                    sched_kw[sk] = float(sv)
            sched = Schedule(kind=kind, value=value, **sched_kw)
            kwargs[k] = value
            schedules.append((k, sched))

    # Nested schedules (inside sub-config dicts)
    _SUB_CONFIG_KEYS = {
        "proposal", "cost", "coupling", "update", "feedback",
    }
    for sub_key in _SUB_CONFIG_KEYS:
        sub = kwargs.get(sub_key)
        if isinstance(sub, dict):
            for k in list(sub):
                v = sub[k]
                if isinstance(v, dict) and "schedule" in v:
                    v = dict(v)
                    kind = v.pop("schedule")
                    value = float(v.pop("value"))
                    sched_kw = {}
                    for sk, sv in v.items():
                        if sk == "warmup":
                            sched_kw[sk] = int(sv)
                        else:
                            sched_kw[sk] = float(sv)
                    sched = Schedule(kind=kind, value=value, **sched_kw)
                    sub[k] = value
                    schedules.append((f"{sub_key}.{k}", sched))

    return tuple(schedules)


# ---------------------------------------------------------------------------
# Baseline field translation
# ---------------------------------------------------------------------------

_BASELINE_RENAMES = {
    "svgd": {"learning_rate": "stepsize", "score_clip": "clip_score"},
    "ula": {"step_size": "stepsize", "score_clip": "clip_score"},
    "mala": {"step_size": "stepsize", "score_clip": "clip_score"},
    "mppi": {},
    "eks": {},
}


def _translate_baseline_fields(method: str, kwargs: dict) -> dict:
    """Apply field name translations for baseline configs.

    Supports both old and new field names in YAML for backward compat.
    """
    renames = _BASELINE_RENAMES.get(method, {})
    for old_name, new_name in renames.items():
        if old_name in kwargs and new_name not in kwargs:
            kwargs[new_name] = kwargs.pop(old_name)
    return kwargs


# ---------------------------------------------------------------------------
# Algorithm config dispatch
# ---------------------------------------------------------------------------

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
            if k in _META_KEYS:
                continue
            kwargs[k] = v

        kwargs["n_particles"] = shared.get("n_particles", 100)
        kwargs["n_iterations"] = shared.get("n_iterations", 500)

        # Apply field name translations
        _translate_baseline_fields(method, kwargs)

        # Handle MALA preconditioner
        if method == "mala":
            raw_pc = kwargs.pop("preconditioner", None)
            if raw_pc is not None:
                kwargs["preconditioner"] = _expand_preconditioner(raw_pc)

        config = config_cls(**kwargs)
        return config, bl["init"], bl["step"], True

    else:
        # ETD or SDD
        is_sdd = entry.get("method") == "sdd"

        kwargs = {}
        for k, v in entry.items():
            if k in _META_KEYS:
                continue
            kwargs[k] = v

        n_particles = shared.get("n_particles", 100)
        n_iterations = shared.get("n_iterations", 500)

        # Extract schedules (modifies kwargs in-place)
        schedules = _extract_schedules(kwargs)

        # Expand sub-configs
        proposal = _expand_proposal(kwargs.pop("proposal", None))
        cost = _expand_cost(kwargs.pop("cost", None))
        coupling = _expand_coupling(kwargs.pop("coupling", None))
        update = _expand_update(kwargs.pop("update", None))
        preconditioner = _expand_preconditioner(kwargs.pop("preconditioner", None))
        mutation = _expand_mutation(kwargs.pop("mutation", None))
        feedback = _expand_feedback(kwargs.pop("feedback", None))

        epsilon = float(kwargs.pop("epsilon", 0.1))

        # Langevin cost FDR defaults: alpha = epsilon unless overridden
        if cost.type == "langevin":
            if proposal.fdr:
                # Check if alpha was explicitly set in the proposal sub-config
                raw_proposal = entry.get("proposal", {})
                alpha_explicit = (
                    isinstance(raw_proposal, dict) and "alpha" in raw_proposal
                )
                if not alpha_explicit:
                    proposal = dataclasses.replace(proposal, alpha=epsilon)

        if is_sdd:
            from etd.extensions.sdd import SDDConfig, SelfCouplingConfig
            from etd.extensions.sdd import init as sdd_init
            from etd.extensions.sdd import step as sdd_step

            # Self-coupling config
            raw_self = kwargs.pop("self_coupling", None)
            if raw_self is None:
                self_coupling = SelfCouplingConfig()
            elif isinstance(raw_self, dict):
                d = dict(raw_self)
                self_coupling = SelfCouplingConfig(
                    epsilon=float(d.pop("epsilon", 0.1)),
                    iterations=int(d.pop("iterations", 50)),
                    tolerance=float(d.pop("tolerance", 1e-2)),
                )
            else:
                self_coupling = raw_self

            eta = float(kwargs.pop("eta", 0.5))

            config = SDDConfig(
                n_particles=n_particles,
                n_iterations=n_iterations,
                epsilon=epsilon,
                proposal=proposal,
                cost=cost,
                coupling=coupling,
                update=update,
                preconditioner=preconditioner,
                mutation=mutation,
                feedback=feedback,
                self_coupling=self_coupling,
                eta=eta,
                schedules=schedules,
            )
            return config, sdd_init, sdd_step, False

        config = ETDConfig(
            n_particles=n_particles,
            n_iterations=n_iterations,
            epsilon=epsilon,
            proposal=proposal,
            cost=cost,
            coupling=coupling,
            update=update,
            preconditioner=preconditioner,
            mutation=mutation,
            feedback=feedback,
            schedules=schedules,
        )
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

    Tries NUTS cache first (for real-data targets), then ``target.sample()``
    for synthetic targets, then None.

    Args:
        target: Target distribution.
        cfg: Experiment config dict.
        key: JAX PRNG key.
        n_ref: Number of reference samples.

    Returns:
        Reference samples ``(n_ref, d)`` or None.
    """
    # Try NUTS cache first
    try:
        from experiments.nuts import load_reference
        target_cfg = cfg.get("experiment", {}).get("target", {})
        target_name = target_cfg.get("type", "")
        target_params = target_cfg.get("params", {})
        ref = load_reference(target_name, target_params)
        if ref is not None:
            return jnp.asarray(ref)
    except (ImportError, Exception):
        pass

    # Fall back to exact sampling
    if hasattr(target, "sample"):
        return target.sample(key, n_ref)
    return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

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
    "mean_rmse": lambda p, t, ref: float(mean_rmse(p, ref))
        if ref is not None else float("nan"),
    "variance_ratio_ref": lambda p, t, ref: float(
        variance_ratio_vs_reference(p, ref)
    ) if ref is not None else float("nan"),
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
    debug: bool = False,
) -> Tuple[Dict[int, Dict[str, float]], Dict[int, np.ndarray], float]:
    """Run a single algorithm to completion.

    Uses ``jax.lax.scan`` with buffer donation for fused execution unless
    ``debug=True``, in which case falls back to a Python loop (no JIT).

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
        debug: If True, use Python loop instead of lax.scan.

    Returns:
        Tuple ``(metrics_dict, particles_dict, wall_clock)`` where:
        - ``metrics_dict``: ``{checkpoint -> {metric -> value}}``
        - ``particles_dict``: ``{checkpoint -> (N, d) ndarray}``
        - ``wall_clock``: Total wall-clock seconds.
    """
    k_init, k_run = jax.random.split(key)

    metrics_dict = {}
    particles_dict = {}

    checkpoint_set = set(checkpoints)

    use_scan = not debug and not DEBUG

    if use_scan:
        # --- Scan path: fused lax.scan with buffer donation ---
        segments = _compute_segments(checkpoints, n_iterations)
        scan_runner = _make_jit_scan(step_fn, target, config)

        # Use numpy positions so each init_fn call creates a fresh JAX
        # buffer -- donation consumes the warmup buffer and we need a
        # separate allocation for the real run.
        init_positions_np = np.asarray(init_positions)

        # Warm-up: compile scan with a 1-step segment, then discard.
        # This excludes JIT compilation time from the wall-clock measurement.
        warmup_state = init_fn(k_init, target, config, init_positions=init_positions_np)
        if segments:
            _, _wi = scan_runner(warmup_state, k_run, 1, 1)
            jax.block_until_ready(_)
            del warmup_state, _, _wi

        # Re-init state (warm-up consumed the buffer via donation)
        state = init_fn(k_init, target, config, init_positions=init_positions_np)

        # Checkpoint 0
        if 0 in checkpoint_set:
            particles_dict[0] = np.array(state.positions)
            metrics_dict[0] = compute_metrics(
                state.positions, target, metrics_list, ref_data,
            )

        t_start = time.perf_counter()

        for start, n_steps, ckpt in segments:
            state, info = scan_runner(state, k_run, start, n_steps)
            jax.block_until_ready(state)

            positions_np = np.array(state.positions)
            particles_dict[ckpt] = positions_np
            step_metrics = compute_metrics(
                state.positions, target, metrics_list, ref_data,
            )
            for diag_name in ("coupling_ess", "sinkhorn_iters"):
                if diag_name in metrics_list and diag_name in info:
                    step_metrics[diag_name] = float(info[diag_name])
            metrics_dict[ckpt] = step_metrics

        wall_clock = time.perf_counter() - t_start

    else:
        # --- Debug path: Python loop, no JIT ---
        state = init_fn(k_init, target, config, init_positions=init_positions)
        step_fn_actual = step_jit if step_jit is not None else step_fn

        if 0 in checkpoint_set:
            particles_dict[0] = np.array(state.positions)
            metrics_dict[0] = compute_metrics(
                state.positions, target, metrics_list, ref_data,
            )

        t_start = time.perf_counter()

        for i in range(1, n_iterations + 1):
            k_step = jax.random.fold_in(k_run, i)
            state, info = step_fn_actual(k_step, state, target, config)

            if i in checkpoint_set:
                positions_np = np.array(state.positions)
                particles_dict[i] = positions_np
                step_metrics = compute_metrics(
                    state.positions, target, metrics_list, ref_data,
                )
                for diag_name in ("coupling_ess", "sinkhorn_iters"):
                    if diag_name in metrics_list and diag_name in info:
                        step_metrics[diag_name] = float(info[diag_name])
                metrics_dict[i] = step_metrics

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
    ref_data=None,
    display_metadata=None,
) -> str:
    """Save experiment results to disk.

    Args:
        out_dir: Output directory path.
        cfg: Original config dict (written as config.yaml).
        all_metrics: ``{seed -> {algo -> {ckpt -> {metric -> val}}}}``.
        all_particles: ``{seed -> {algo -> {ckpt -> (N,d) ndarray}}}``.
        ref_data: Reference samples ``(M, d)`` to bundle for portability.
        display_metadata: Resolved display styles ``{label -> {family, color, ...}}``
            from :func:`figures.style.resolve_algo_styles`.

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

    # Bundle reference samples for portable results
    if ref_data is not None:
        np.savez(
            os.path.join(out_dir, "reference.npz"),
            samples=np.asarray(ref_data),
        )

    # Save display metadata for downstream plotting
    if display_metadata is not None:
        with open(os.path.join(out_dir, "metadata.json"), "w") as f:
            json.dump(display_metadata, f, indent=2)

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
        title="Results (seed avg +/- std)",
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
                row.append(f"{avg:.4f} +/- {std:.4f}")
            else:
                row.append("N/A")

        # Wall clock
        wc_vals = [wall_clocks.get((seed, label), 0.0) for seed in seeds]
        wc_avg = np.mean(wc_vals)
        wc_std = np.std(wc_vals)
        row.append(f"{wc_avg:.1f} +/- {wc_std:.1f}s")

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

    parallel_seeds = True

    if config_path is None:
        parser = argparse.ArgumentParser(description="ETD experiment runner")
        parser.add_argument("config", help="Path to YAML config file")
        parser.add_argument("--debug", action="store_true", help="Disable JIT")
        parser.add_argument(
            "--no-parallel-seeds", action="store_true",
            help="Disable vmapped seed parallelism (use sequential loop)",
        )
        args = parser.parse_args()
        config_path = args.config
        debug = args.debug
        parallel_seeds = not args.no_parallel_seeds

    if debug:
        DEBUG = True
        parallel_seeds = False  # vmap requires JIT

    # --- JAX compilation cache ---
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "jax", "etd")
    os.makedirs(cache_dir, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", cache_dir)

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
    algo_display_meta = []  # display metadata per algo (YAML order)
    for entry in algo_entries:
        if not entry.get("enabled", True):
            continue
        original = dict(entry)
        expanded = expand_algo_sweeps(entry)
        for concrete in expanded:
            # Compose sublabel into base label before building sweep suffix
            base = concrete.get("label", "unnamed")
            sublabel = concrete.get("sublabel")
            if sublabel:
                base = f"{base} ({sublabel})"
            label = build_algo_label(base, concrete, original)

            config, init_fn, step_fn, is_bl = build_algo_config(concrete, shared)
            algo_configs.append((label, config, init_fn, step_fn, is_bl))

            # Collect display metadata
            display = concrete.get("display", {})
            algo_display_meta.append({
                "label": label,
                "family": display.get("family"),
                "color": display.get("color"),
                "linestyle": display.get("linestyle", "-"),
                "group": display.get("group"),
                "is_baseline": is_bl,
            })

    algo_labels = [ac[0] for ac in algo_configs]
    n_iters = shared.get("n_iterations", 500)
    n_progress = shared.get("progress_segments")

    mode_str = "parallel" if parallel_seeds else "sequential"
    console.print(
        f"[{now}] Running {len(algo_configs)} algorithms x {len(seeds)} seeds ({mode_str})"
    )

    # --- Reference data ---
    ref_key = jax.random.PRNGKey(99999)
    ref_data = get_reference_data(target, cfg, ref_key)

    # --- Pre-compute init positions (shared across algorithms) ---
    # Init positions depend only on (seed, target, shared), not on the
    # algorithm, so we compute them once and reuse for all algorithms.
    seed_run_keys = {}  # seed -> k_run (passed to run_single / batched)
    seed_init_positions = {}  # seed -> (N, d) ndarray
    for seed in seeds:
        key = jax.random.PRNGKey(seed)
        k_init, _k_ref, k_run = jax.random.split(key, 3)
        seed_run_keys[seed] = k_run
        seed_init_positions[seed] = np.asarray(
            make_init_positions(k_init, target, shared)
        )

    # Pre-stack for batched path
    all_init_pos = np.stack([seed_init_positions[s] for s in seeds])
    batch_run_keys = [seed_run_keys[s] for s in seeds]

    # --- Pre-compile step functions (debug mode only) ---
    # In scan mode, _make_jit_scan creates its own JIT closure.
    compiled_steps = {}
    if DEBUG:
        for label, config, init_fn, step_fn, is_bl in algo_configs:
            compiled_steps[label] = step_fn  # raw, no JIT in debug

    # --- Run ---
    all_metrics = {}   # seed -> algo -> ckpt -> {metric: val}
    all_particles = {} # seed -> algo -> ckpt -> (N,d) ndarray
    wall_clocks = {}   # (seed, algo) -> seconds

    if parallel_seeds and not DEBUG:
        # --- Batched path: vmap over seeds ---
        from experiments._parallel import run_seeds_batched

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            for label, config, init_fn, step_fn, is_bl in algo_configs:
                # Progress: one tick per segment (not per seed)
                segments_count = len(
                    _compute_segments(checkpoints, n_iters)
                )
                if n_progress is not None:
                    from experiments._parallel import _compute_progress_segments
                    segs, _ = _compute_progress_segments(
                        checkpoints, n_iters, n_progress,
                    )
                    segments_count = len(segs)

                task_id = progress.add_task(
                    f"  {label}", total=segments_count or 1,
                )

                m_by_seed, p_by_seed, wc = run_seeds_batched(
                    batch_run_keys, target, config,
                    init_fn, step_fn, all_init_pos,
                    n_iters, checkpoints, metrics_list, ref_data,
                    compute_metrics_fn=compute_metrics,
                    n_progress=n_progress,
                    progress_callback=lambda: progress.advance(task_id),
                )

                # Unpack batched results into all_metrics / all_particles
                for s_idx, seed in enumerate(seeds):
                    all_metrics.setdefault(seed, {})[label] = m_by_seed[s_idx]
                    all_particles.setdefault(seed, {})[label] = p_by_seed[s_idx]
                    wall_clocks[(seed, label)] = wc / len(seeds)  # amortized

    else:
        # --- Sequential path (unchanged) ---
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
                    m_dict, p_dict, wc = run_single(
                        seed_run_keys[seed], target, config,
                        init_fn, step_fn, is_bl,
                        seed_init_positions[seed], n_iters, checkpoints,
                        metrics_list, ref_data,
                        step_jit=compiled_steps.get(label),
                        debug=debug,
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

    # --- Resolve display metadata ---
    from figures.style import resolve_algo_styles
    resolved_styles = resolve_algo_styles(algo_display_meta)

    # --- Save ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = os.path.join("results", name, timestamp)
    save_results(
        out_dir, cfg, all_metrics, all_particles,
        ref_data=ref_data, display_metadata=resolved_styles,
    )
    console.print(f"\nResults saved to {out_dir}/")

    return all_metrics


if __name__ == "__main__":
    main()
