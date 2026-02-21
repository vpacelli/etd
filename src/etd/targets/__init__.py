"""Target distribution registry."""

from etd.targets.gaussian import GaussianTarget
from etd.targets.gmm import GMMTarget

TARGETS = {
    "gaussian": GaussianTarget,
    "gmm": GMMTarget,
}


def get_target(name: str, **params):
    """Instantiate a target distribution by name.

    Args:
        name: Target name (``"gaussian"`` or ``"gmm"``).
        **params: Forwarded to the target constructor.

    Returns:
        A target instance satisfying the :class:`~etd.types.Target` protocol.

    Raises:
        KeyError: If *name* is not in the registry.
    """
    if name not in TARGETS:
        raise KeyError(f"Unknown target '{name}'. Available: {list(TARGETS)}")
    return TARGETS[name](**params)


__all__ = [
    "TARGETS",
    "get_target",
    "GaussianTarget",
    "GMMTarget",
]
