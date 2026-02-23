"""Parameter schedules for ETD hyperparameters.

Provides a frozen :class:`Schedule` dataclass and a JIT-safe
:func:`eval_schedule` function.  Schedules live inside ``ETDConfig``
as static data, so the ``kind`` branch is resolved at trace time
while ``step`` remains a traced JAX integer.

Supported schedule kinds:
    ``linear_warmup``
        Ramps linearly from ``end`` (default 0) to ``value`` over
        ``warmup`` iterations, then holds ``value``.
    ``linear_decay``
        Decays linearly from ``value`` to ``end`` over ``n_iterations``.
    ``cosine_decay``
        Decays from ``value`` to ``end`` via a half-cosine over
        ``n_iterations``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class Schedule:
    """Frozen schedule descriptor — hashable for JIT static args.

    Attributes:
        kind: Schedule type (``"linear_warmup"``, ``"linear_decay"``,
            ``"cosine_decay"``).
        value: Target/start value (warmup ramps *to* this; decay starts
            *from* this).
        end: Final value for decay schedules; start value for warmup.
            Defaults to 0.0.
        warmup: Number of iterations for warmup ramp.
    """

    kind: str
    value: float
    end: float = 0.0
    warmup: int = 0


def eval_schedule(schedule: Schedule, step: int, n_iterations: int) -> jnp.ndarray:
    """Evaluate a schedule at the given iteration.

    Args:
        schedule: Schedule descriptor (static for JIT).
        step: Current iteration (may be a traced JAX int).
        n_iterations: Total iterations (static, from config).

    Returns:
        Scalar schedule value (traced JAX array).
    """
    if schedule.kind == "linear_warmup":
        if schedule.warmup == 0:
            # No warmup -> constant at value.
            return jnp.float32(schedule.value)
        t = jnp.clip(step / schedule.warmup, 0.0, 1.0)
        return schedule.end + (schedule.value - schedule.end) * t

    elif schedule.kind == "linear_decay":
        t = jnp.clip(step / jnp.maximum(n_iterations, 1), 0.0, 1.0)
        return schedule.value + (schedule.end - schedule.value) * t

    elif schedule.kind == "cosine_decay":
        t = jnp.clip(step / jnp.maximum(n_iterations, 1), 0.0, 1.0)
        return schedule.end + 0.5 * (schedule.value - schedule.end) * (1.0 + jnp.cos(jnp.pi * t))

    else:
        raise ValueError(f"Unknown schedule kind '{schedule.kind}'")


def _resolve_dotted(obj, dotted_name: str):
    """Resolve a dotted attribute path on a config object.

    Args:
        obj: Root config object.
        dotted_name: Attribute path, e.g. ``"proposal.alpha"``
            or ``"epsilon"`` (no dots).

    Returns:
        The resolved attribute value.
    """
    for part in dotted_name.split("."):
        obj = getattr(obj, part)
    return obj


def resolve_param(config, name: str, step: int):
    """Resolve a parameter, applying its schedule if one exists.

    Looks up ``name`` in ``config.schedules`` (a tuple of
    ``(param_name, Schedule)`` pairs).  If found, evaluates the
    schedule at ``step``; otherwise returns the static config value.

    Supports dotted paths for nested sub-configs:
    ``"proposal.alpha"`` resolves to ``config.proposal.alpha``.

    Args:
        config: ETDConfig (frozen, static for JIT).
        name: Parameter name (e.g. ``"epsilon"`` or ``"proposal.alpha"``).
        step: Current iteration (may be traced).

    Returns:
        Python float (no schedule) or traced JAX scalar (scheduled).
    """
    for param_name, sched in config.schedules:
        if param_name == name:
            return eval_schedule(sched, step, config.n_iterations)
    return _resolve_dotted(config, name)
