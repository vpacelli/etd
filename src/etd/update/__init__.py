"""Update rule registry."""

from etd.update.barycentric import barycentric_update
from etd.update.categorical import systematic_resample

UPDATES = {
    "categorical": systematic_resample,
    "barycentric": barycentric_update,
}


def get_update_fn(name: str):
    """Look up an update function by name.

    Raises:
        KeyError: If *name* is not in the registry.
    """
    if name not in UPDATES:
        raise KeyError(f"Unknown update '{name}'. Available: {list(UPDATES)}")
    return UPDATES[name]


__all__ = [
    "UPDATES",
    "get_update_fn",
    "barycentric_update",
    "systematic_resample",
]
