"""Target distribution registry."""

from etd.targets.banana import BananaTarget
from etd.targets.funnel import FunnelTarget
from etd.targets.gaussian import GaussianTarget
from etd.targets.gmm import GMMTarget

TARGETS = {
    "gaussian": GaussianTarget,
    "gmm": GMMTarget,
    "banana": BananaTarget,
    "funnel": FunnelTarget,
}


def get_target(name: str, **params):
    """Instantiate a target distribution by name.

    Args:
        name: Target name (e.g. ``"gaussian"``, ``"gmm"``, ``"banana"``).
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
    "BananaTarget",
    "FunnelTarget",
    "GaussianTarget",
    "GMMTarget",
]
