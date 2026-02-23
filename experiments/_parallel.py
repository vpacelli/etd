"""Parallel seed and sweep execution via vmap.

Provides infrastructure to batch algorithm execution across seeds (and
sweep configs) using ``jax.vmap``, saturating the GPU with a single
fused kernel per segment instead of sequential Python loops.

Key functions:

    ``batch_init_states``
        Initialize states for all seeds via Python loop (init has
        Python-level branching on config fields that prevents vmap).

    ``_make_batched_scan``
        Build a JIT-compiled, vmapped scan runner over seeds.

    ``run_seeds_batched``
        Full orchestrator: init → warm-up → segment loop → metrics.

    ``_compute_progress_segments``
        Merge user checkpoints with progress-reporting boundaries.

For sweep execution (tune.py):

    ``structural_key``
        Hash structural config fields to group configs that share
        the same JIT trace.

    ``group_configs_by_structure``
        Partition sweep configs into structural groups.

    ``_make_sweep_scan``
        Double vmap: configs (outer) × seeds (inner).

    ``run_sweep_batched``
        Orchestrator for double-vmapped sweep execution.
"""

from __future__ import annotations

import dataclasses
import functools
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Progress segment computation
# ---------------------------------------------------------------------------

def _compute_progress_segments(
    checkpoints: List[int],
    n_iterations: int,
    n_progress: Optional[int] = None,
) -> Tuple[List[Tuple[int, int, int]], set]:
    """Merge user checkpoints with evenly-spaced progress boundaries.

    Returns segments similar to ``run._compute_segments`` but with
    additional segment breaks for progress reporting.  Metrics are
    only computed at real checkpoints; progress ticks at all boundaries.

    Args:
        checkpoints: Sorted iteration numbers at which to record metrics.
        n_iterations: Total iterations.
        n_progress: Number of evenly-spaced progress boundaries to add.
            If None, no extra boundaries (segments match checkpoints).

    Returns:
        Tuple ``(segments, checkpoint_set)`` where:
        - ``segments``: List of ``(start, n_steps, boundary)`` tuples.
        - ``checkpoint_set``: Set of real checkpoint iterations (metrics
          should only be computed at these).
    """
    # Real checkpoints (excluding 0, handled separately)
    ckpts = sorted(c for c in checkpoints if 0 < c <= n_iterations)
    checkpoint_set = set(ckpts)

    # Add progress boundaries
    if n_progress is not None and n_progress > 0:
        progress_iters = set()
        for i in range(1, n_progress + 1):
            boundary = int(round(i * n_iterations / n_progress))
            if 0 < boundary <= n_iterations:
                progress_iters.add(boundary)
        all_boundaries = sorted(checkpoint_set | progress_iters)
    else:
        all_boundaries = ckpts

    if not all_boundaries:
        return [], checkpoint_set

    segments = []
    prev = 0
    for b in all_boundaries:
        n_steps = b - prev
        if n_steps > 0:
            segments.append((prev + 1, n_steps, b))
        prev = b

    return segments, checkpoint_set


# ---------------------------------------------------------------------------
# Batched initialization
# ---------------------------------------------------------------------------

def batch_init_states(
    init_fn: Callable,
    keys: Sequence[jax.Array],
    target: Any,
    config: Any,
    init_positions_all: np.ndarray,
) -> Any:
    """Initialize states for all seeds via Python loop.

    We cannot vmap ``init_fn`` because it has Python-level branching on
    config fields (e.g. ``config.preconditioner.is_cholesky``).  Instead
    we loop over seeds and ``jax.tree.map(np.stack, ...)`` the results.

    Args:
        init_fn: Algorithm init function ``(key, target, config, init_positions) → state``.
        keys: PRNG keys, one per seed — length ``S``.
        target: Target distribution (shared across seeds).
        config: Algorithm config (shared across seeds).
        init_positions_all: Starting positions, shape ``(S, N, d)``
            (numpy array so each call creates a fresh JAX buffer).

    Returns:
        Batched state pytree with leading dimension ``S``.
    """
    states = []
    for i, k in enumerate(keys):
        state = init_fn(k, target, config, init_positions=init_positions_all[i])
        # Convert to numpy immediately to avoid accumulating JAX buffers
        state_np = jax.tree.map(np.asarray, state)
        states.append(state_np)

    # Stack into batched pytree: each leaf gets shape (S, ...)
    batched = jax.tree.map(lambda *leaves: np.stack(leaves, axis=0), *states)
    return batched


# ---------------------------------------------------------------------------
# Batched scan runner (vmap over seeds)
# ---------------------------------------------------------------------------

def _make_batched_scan(
    step_fn: Callable,
    target: Any,
    config: Any,
) -> Callable:
    """Build a JIT-compiled, vmapped scan runner for parallel seed execution.

    The scan body is built without JIT, vmapped over the leading (seed)
    axis, then JIT-compiled as a single fused kernel.

    Args:
        step_fn: Algorithm step function ``(key, state, target, config) → (state, info)``.
        target: Target distribution (captured in closure).
        config: Algorithm config (captured in closure, static for JIT).

    Returns:
        ``run_segment_batched(states, keys, start_iter, n_steps) → (final_states, last_infos)``
        where states has leading seed dimension ``S``.
    """
    # Suppress donation warnings for int32 fields (e.g. state.step)
    warnings.filterwarnings(
        "ignore", message=".*Some donated buffers were not usable.*"
    )

    def _single_seed_scan(state, key_base, start_iter, n_steps):
        """Run n_steps of step_fn for a single seed as a fused scan.

        Args:
            state: Algorithm state (single seed).
            key_base: Base PRNG key for this seed.
            start_iter: 1-based iteration index.
            n_steps: Number of steps (static).

        Returns:
            Tuple ``(final_state, last_info)``.
        """
        def scan_body(carry, t_offset):
            t = start_iter + t_offset
            key_step = jax.random.fold_in(key_base, t)
            new_state, info = step_fn(key_step, carry, target, config)
            return new_state, info

        final_state, stacked_info = jax.lax.scan(
            scan_body, state, jnp.arange(n_steps),
        )
        last_info = jax.tree.map(lambda x: x[-1], stacked_info)
        return final_state, last_info

    # vmap over seeds: state batch dim 0, key batch dim 0,
    # start_iter and n_steps are broadcast (same for all seeds)
    _vmapped = jax.vmap(
        _single_seed_scan,
        in_axes=(0, 0, None, None),
    )

    @functools.partial(jax.jit, static_argnums=(2, 3), donate_argnums=(0,))
    def run_segment_batched(states, keys, start_iter, n_steps):
        """Run a segment for all seeds in parallel.

        Args:
            states: Batched state pytree, leading dim ``S`` (donated).
            keys: PRNG keys, shape ``(S, 2)`` — one per seed.
            start_iter: 1-based iteration index (static, broadcast).
            n_steps: Number of steps (static, broadcast).

        Returns:
            Tuple ``(final_states, last_infos)`` with leading dim ``S``.
        """
        return _vmapped(states, keys, start_iter, n_steps)

    return run_segment_batched


# ---------------------------------------------------------------------------
# Full batched execution orchestrator
# ---------------------------------------------------------------------------

def run_seeds_batched(
    keys: Sequence[jax.Array],
    target: Any,
    config: Any,
    init_fn: Callable,
    step_fn: Callable,
    init_positions_all: np.ndarray,
    n_iterations: int,
    checkpoints: List[int],
    metrics_list: List[str],
    ref_data: Optional[jnp.ndarray],
    compute_metrics_fn: Callable,
    n_progress: Optional[int] = None,
    progress_callback: Optional[Callable] = None,
) -> Tuple[Dict[int, Dict[int, Dict[str, float]]], Dict[int, Dict[int, np.ndarray]], float]:
    """Run all seeds in parallel via vmapped scan.

    Key management matches ``run_single``: each key is split into
    ``(k_init, k_run)`` internally, so the caller should pass the
    same key they would pass to ``run_single``.

    Orchestrates:
    1. ``batch_init_states`` → ``(S, N, d)`` numpy-backed states
    2. Warm-up: 1-step compile, ``block_until_ready``, discard
    3. Re-init (donation consumed warm-up buffers)
    4. Segment loop: ``run_segment_batched`` → unbatch → metrics per seed
    5. Return results

    Args:
        keys: PRNG keys, one per seed (same key as passed to ``run_single``).
        target: Target distribution.
        config: Algorithm config.
        init_fn: Algorithm init function.
        step_fn: Algorithm step function.
        init_positions_all: Init positions, shape ``(S, N, d)`` (numpy).
        n_iterations: Total iterations.
        checkpoints: Iterations at which to record metrics/particles.
        metrics_list: Which metrics to compute.
        ref_data: Reference samples for metrics (or None).
        compute_metrics_fn: ``(particles, target, metrics_list, ref_data) → dict``.
        n_progress: Number of progress ticks (None = ticks at checkpoints only).
        progress_callback: Called after each segment completes.

    Returns:
        Tuple ``(metrics_by_seed, particles_by_seed, wall_clock)`` where:
        - ``metrics_by_seed``: ``{seed_idx → {ckpt → {metric → val}}}``
        - ``particles_by_seed``: ``{seed_idx → {ckpt → (N,d) ndarray}}``
        - ``wall_clock``: Total wall-clock seconds (excludes compilation).
    """
    S = len(keys)
    checkpoint_set = set(checkpoints)

    # Match run_single's key split: k_init, k_run = jax.random.split(key)
    init_keys = [jax.random.split(k)[0] for k in keys]
    run_keys = [jax.random.split(k)[1] for k in keys]

    # Compute segments with progress boundaries
    segments, real_checkpoints = _compute_progress_segments(
        checkpoints, n_iterations, n_progress,
    )

    # Build vmapped scan runner
    scan_runner = _make_batched_scan(step_fn, target, config)

    # Stack keys for vmap: (S, 2)
    run_keys_arr = jnp.stack([jnp.asarray(k) for k in run_keys])

    # --- Warm-up: compile with 1-step segment, then discard ---
    warmup_states = batch_init_states(
        init_fn, init_keys, target, config, init_positions_all,
    )
    if segments:
        _ws, _wi = scan_runner(warmup_states, run_keys_arr, 1, 1)
        jax.block_until_ready(_ws)
        del warmup_states, _ws, _wi

    # --- Re-init (warm-up consumed buffers via donation) ---
    states = batch_init_states(
        init_fn, init_keys, target, config, init_positions_all,
    )

    # --- Checkpoint 0 ---
    metrics_by_seed: Dict[int, Dict[int, Dict[str, float]]] = {}
    particles_by_seed: Dict[int, Dict[int, np.ndarray]] = {}

    for s_idx in range(S):
        metrics_by_seed[s_idx] = {}
        particles_by_seed[s_idx] = {}

    if 0 in checkpoint_set:
        for s_idx in range(S):
            pos_s = np.asarray(states.positions[s_idx])  # type: ignore[attr-defined]
            particles_by_seed[s_idx][0] = pos_s
            metrics_by_seed[s_idx][0] = compute_metrics_fn(
                jnp.asarray(pos_s), target, metrics_list, ref_data,
            )

    # --- Segment loop ---
    t_start = time.perf_counter()

    for start, n_steps, boundary in segments:
        states, infos = scan_runner(states, run_keys_arr, start, n_steps)
        jax.block_until_ready(states)

        # Only compute metrics at real checkpoints
        if boundary in real_checkpoints:
            for s_idx in range(S):
                # Extract positions for this seed
                pos_s = np.asarray(states.positions[s_idx])  # type: ignore[attr-defined]
                particles_by_seed[s_idx][boundary] = pos_s
                step_metrics = compute_metrics_fn(
                    jnp.asarray(pos_s), target, metrics_list, ref_data,
                )
                # Add diagnostic metrics from info
                for diag_name in ("coupling_ess", "sinkhorn_iters"):
                    if diag_name in metrics_list and diag_name in infos:
                        step_metrics[diag_name] = float(
                            np.asarray(infos[diag_name][s_idx])
                        )
                metrics_by_seed[s_idx][boundary] = step_metrics

        if progress_callback is not None:
            progress_callback()

    wall_clock = time.perf_counter() - t_start

    return metrics_by_seed, particles_by_seed, wall_clock


# ---------------------------------------------------------------------------
# Structural config grouping (for sweep parallelism in tune.py)
# ---------------------------------------------------------------------------

# Fields that affect JIT trace (Python-level branching, array shapes).
# Sweeping these requires separate JIT compilations.
_STRUCTURAL_FIELDS_ETD = frozenset({
    "coupling", "cost", "cost_params", "cost_normalize", "update",
    "n_particles", "n_proposals", "use_score", "fdr", "dv_feedback",
    "sinkhorn_max_iter", "preconditioner", "mutation",
    "precondition", "whiten",
})

# Fields that are purely arithmetic (safe as JAX tracers under vmap).
_SCALAR_FIELDS_ETD = frozenset({
    "epsilon", "alpha", "sigma", "score_clip",
    "rho", "dv_weight", "sinkhorn_tol",
})


def structural_key(config: Any) -> tuple:
    """Hash the structural fields of a config for grouping.

    Two configs with the same structural key share the same JIT trace
    and can be vmapped together (differing only in scalar parameters).

    Args:
        config: Algorithm config (ETDConfig, SDDConfig, or baseline config).

    Returns:
        Hashable tuple of structural field values.
    """
    parts = []
    for f in dataclasses.fields(config):
        if f.name in _STRUCTURAL_FIELDS_ETD:
            val = getattr(config, f.name)
            # Dataclass sub-configs (PreconditionerConfig, MutationConfig)
            # are frozen and hashable
            parts.append((f.name, val))
        elif f.name == "schedules":
            # Schedules are structural: they affect trace-time branching
            parts.append((f.name, val))
    return tuple(parts)


def group_configs_by_structure(
    algo_configs: List[Tuple[str, Any, Callable, Callable, bool]],
) -> Dict[tuple, List[Tuple[str, Any, Callable, Callable, bool]]]:
    """Partition sweep configs into groups that share JIT traces.

    Configs in the same group differ only in scalar parameters and can
    be double-vmapped (configs × seeds) with a single JIT compilation.

    Args:
        algo_configs: List of ``(label, config, init_fn, step_fn, is_baseline)``.

    Returns:
        Dict mapping structural key → list of matching configs.
    """
    groups: Dict[tuple, List] = {}
    for item in algo_configs:
        _label, config, _init, _step, _is_bl = item
        try:
            key = structural_key(config)
        except (TypeError, AttributeError):
            # Baseline configs may not have all ETD fields — give each its own group
            key = (id(config),)
        groups.setdefault(key, []).append(item)
    return groups


def identify_varying_scalars(
    configs: List[Any],
) -> List[str]:
    """Identify scalar fields that vary across a group of configs.

    Args:
        configs: List of algorithm configs (same structural group).

    Returns:
        List of field names that differ across configs and are in
        ``_SCALAR_FIELDS_ETD``.
    """
    if len(configs) <= 1:
        return []

    varying = []
    first = configs[0]
    for f in dataclasses.fields(first):
        if f.name not in _SCALAR_FIELDS_ETD:
            continue
        val0 = getattr(first, f.name)
        for c in configs[1:]:
            if getattr(c, f.name) != val0:
                varying.append(f.name)
                break
    return varying


def validate_sweep_fields(
    config: Any,
    sweep_fields: List[str],
) -> None:
    """Validate that swept fields are not also scheduled.

    A scheduled field uses ``resolve_param`` which branches on static
    schedule metadata — sweeping it as a tracer would conflict.

    Args:
        config: Algorithm config with ``schedules`` attribute.
        sweep_fields: Field names being swept via vmap.

    Raises:
        ValueError: If any swept field has an active schedule.
    """
    scheduled_names = {name for name, _sched in getattr(config, "schedules", ())}
    conflicts = set(sweep_fields) & scheduled_names
    if conflicts:
        raise ValueError(
            f"Cannot sweep fields that have active schedules: {conflicts}. "
            f"Schedule takes precedence in resolve_param, making the field "
            f"non-traceable under vmap."
        )


# ---------------------------------------------------------------------------
# Double vmap: configs × seeds (for tune.py)
# ---------------------------------------------------------------------------

def _make_sweep_scan(
    step_fn: Callable,
    target: Any,
    base_config: Any,
    sweep_field_names: List[str],
) -> Callable:
    """Build a double-vmapped scan: configs (outer) × seeds (inner).

    Each swept field becomes a tracer via ``dataclasses.replace``.
    The base config provides all structural (static) fields.

    Args:
        step_fn: Algorithm step function.
        target: Target distribution (captured in closure).
        base_config: Base config with structural fields fixed.
        sweep_field_names: Names of scalar fields being swept.

    Returns:
        ``run_sweep(states, keys, scalar_vals, start_iter, n_steps)``
        where:
        - ``states``: ``(C, S, ...)`` batched pytree
        - ``keys``: ``(S, 2)`` broadcast across configs
        - ``scalar_vals``: tuple of ``(C,)`` arrays, one per swept field
        - ``start_iter``, ``n_steps``: static ints
    """
    warnings.filterwarnings(
        "ignore", message=".*Some donated buffers were not usable.*"
    )

    def _single(state, key_base, scalar_vals, start_iter, n_steps):
        """Single config × single seed scan.

        Args:
            state: Algorithm state (single config, single seed).
            key_base: Base PRNG key.
            scalar_vals: Tuple of scalar tracer values for swept fields.
            start_iter: 1-based iteration index.
            n_steps: Number of steps (static).
        """
        # Replace scalar fields with tracer values
        overrides = dict(zip(sweep_field_names, scalar_vals))
        config = dataclasses.replace(base_config, **overrides)

        def scan_body(carry, t_offset):
            t = start_iter + t_offset
            key_step = jax.random.fold_in(key_base, t)
            new_state, info = step_fn(key_step, carry, target, config)
            return new_state, info

        final_state, stacked_info = jax.lax.scan(
            scan_body, state, jnp.arange(n_steps),
        )
        last_info = jax.tree.map(lambda x: x[-1], stacked_info)
        return final_state, last_info

    # Inner vmap: over seeds (axis 0 of state and key)
    _seed_vmapped = jax.vmap(
        _single,
        in_axes=(0, 0, None, None, None),
    )

    # Outer vmap: over configs (axis 0 of state and scalar_vals)
    # scalar_vals is a tuple of (C,) arrays — vmap over axis 0 of each
    _sweep_vmapped = jax.vmap(
        _seed_vmapped,
        in_axes=(0, None, 0, None, None),
    )

    @functools.partial(jax.jit, static_argnums=(3, 4), donate_argnums=(0,))
    def run_sweep(states, keys, scalar_vals, start_iter, n_steps):
        """Run a segment for all configs × seeds in parallel.

        Args:
            states: ``(C, S, ...)`` batched pytree (donated).
            keys: ``(S, 2)`` broadcast across configs.
            scalar_vals: Tuple of ``(C,)`` arrays for swept fields.
            start_iter: 1-based iteration index (static).
            n_steps: Number of steps (static).

        Returns:
            ``(final_states, last_infos)`` with shape ``(C, S, ...)``.
        """
        return _sweep_vmapped(states, keys, scalar_vals, start_iter, n_steps)

    return run_sweep


def run_sweep_batched(
    keys: Sequence[jax.Array],
    target: Any,
    base_config: Any,
    configs: List[Any],
    init_fn: Callable,
    step_fn: Callable,
    init_positions_all: np.ndarray,
    sweep_field_names: List[str],
    n_iterations: int,
    checkpoints: List[int],
    metrics_list: List[str],
    ref_data: Optional[jnp.ndarray],
    compute_metrics_fn: Callable,
    n_progress: Optional[int] = None,
    progress_callback: Optional[Callable] = None,
    chunk_size: int = 8,
) -> Tuple[Dict[int, Dict[int, Dict[int, Dict[str, float]]]], float]:
    """Run all configs × seeds in parallel via double-vmapped scan.

    Key management matches ``run_single``: each key is split into
    ``(k_init, k_run)`` internally, same as ``run_seeds_batched``.

    For large sweeps (C > chunk_size), chunks configs into batches
    for memory safety and more frequent progress updates.

    Args:
        keys: PRNG keys, one per seed (same as passed to ``run_single``).
        target: Target distribution.
        base_config: Base config (structural fields).
        configs: List of concrete configs (one per sweep point).
        init_fn: Algorithm init function.
        step_fn: Algorithm step function.
        init_positions_all: ``(S, N, d)`` numpy init positions.
        sweep_field_names: Names of scalar fields being swept.
        n_iterations: Total iterations.
        checkpoints: Iterations at which to record metrics.
        metrics_list: Which metrics to compute.
        ref_data: Reference samples (or None).
        compute_metrics_fn: Metrics computation function.
        n_progress: Number of progress ticks.
        progress_callback: Called after each segment per chunk.
        chunk_size: Max configs per double-vmap batch.

    Returns:
        Tuple ``(metrics_by_config_seed, wall_clock)`` where:
        - ``metrics_by_config_seed``: ``{config_idx → {seed_idx → {ckpt → {metric → val}}}}``
        - ``wall_clock``: Total wall-clock seconds.
    """
    S = len(keys)
    C = len(configs)

    # Match run_single's key split: k_init, k_run = jax.random.split(key)
    init_keys = [jax.random.split(k)[0] for k in keys]
    run_keys = [jax.random.split(k)[1] for k in keys]

    # Validate sweep fields
    validate_sweep_fields(base_config, sweep_field_names)

    segments, real_checkpoints = _compute_progress_segments(
        checkpoints, n_iterations, n_progress,
    )

    run_keys_arr = jnp.stack([jnp.asarray(k) for k in run_keys])

    # Build scalar value arrays: tuple of (C,) arrays
    scalar_vals_all = tuple(
        jnp.array([float(getattr(c, name)) for c in configs])
        for name in sweep_field_names
    )

    # Chunk configs for memory safety
    chunks = list(range(0, C, chunk_size))

    metrics_by_config_seed: Dict[int, Dict[int, Dict[int, Dict[str, float]]]] = {}
    total_wall = 0.0

    for chunk_start in chunks:
        chunk_end = min(chunk_start + chunk_size, C)
        chunk_C = chunk_end - chunk_start

        # Slice scalar vals for this chunk
        chunk_scalar_vals = tuple(
            sv[chunk_start:chunk_end] for sv in scalar_vals_all
        )

        # Build sweep scan for this chunk (all share same structural config)
        sweep_runner = _make_sweep_scan(
            step_fn, target, base_config, sweep_field_names,
        )

        # Init states: use base_config (structural fields are shared)
        # Shape: (S, ...) → broadcast to (C_chunk, S, ...)
        seed_states = batch_init_states(
            init_fn, init_keys, target, base_config, init_positions_all,
        )
        # Broadcast: tile the seed states across chunk configs
        chunk_states = jax.tree.map(
            lambda x: np.broadcast_to(
                x[np.newaxis], (chunk_C,) + x.shape
            ).copy(),
            seed_states,
        )

        # Warm-up
        if segments:
            _ws, _wi = sweep_runner(
                chunk_states, run_keys_arr, chunk_scalar_vals, 1, 1,
            )
            jax.block_until_ready(_ws)
            del _ws, _wi

            # Re-init after warmup (donation consumed buffers)
            seed_states = batch_init_states(
                init_fn, init_keys, target, base_config, init_positions_all,
            )
            chunk_states = jax.tree.map(
                lambda x: np.broadcast_to(
                    x[np.newaxis], (chunk_C,) + x.shape
                ).copy(),
                seed_states,
            )

        # Init metrics storage for this chunk
        for c_offset in range(chunk_C):
            c_idx = chunk_start + c_offset
            metrics_by_config_seed[c_idx] = {}
            for s_idx in range(S):
                metrics_by_config_seed[c_idx][s_idx] = {}

        # Segment loop
        t_start = time.perf_counter()

        for start, n_steps, boundary in segments:
            chunk_states, infos = sweep_runner(
                chunk_states, run_keys_arr, chunk_scalar_vals, start, n_steps,
            )
            jax.block_until_ready(chunk_states)

            if boundary in real_checkpoints:
                for c_offset in range(chunk_C):
                    c_idx = chunk_start + c_offset
                    for s_idx in range(S):
                        pos = np.asarray(
                            chunk_states.positions[c_offset, s_idx]  # type: ignore
                        )
                        step_metrics = compute_metrics_fn(
                            jnp.asarray(pos), target, metrics_list, ref_data,
                        )
                        for diag_name in ("coupling_ess", "sinkhorn_iters"):
                            if diag_name in metrics_list and diag_name in infos:
                                step_metrics[diag_name] = float(
                                    np.asarray(infos[diag_name][c_offset, s_idx])
                                )
                        metrics_by_config_seed[c_idx][s_idx][boundary] = step_metrics

            if progress_callback is not None:
                progress_callback()

        total_wall += time.perf_counter() - t_start

    return metrics_by_config_seed, total_wall
