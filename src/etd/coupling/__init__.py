"""Coupling solver registry."""

from etd.coupling.gibbs import gibbs_coupling
from etd.coupling.sinkhorn import sinkhorn_log_domain
from etd.coupling.unbalanced import sinkhorn_unbalanced

COUPLINGS = {
    "gibbs": gibbs_coupling,
    "balanced": sinkhorn_log_domain,
    "unbalanced": sinkhorn_unbalanced,
}


def get_coupling_fn(name: str):
    """Look up a coupling solver by name.

    Raises:
        KeyError: If *name* is not in the registry.
    """
    if name not in COUPLINGS:
        raise KeyError(f"Unknown coupling '{name}'. Available: {list(COUPLINGS)}")
    return COUPLINGS[name]


__all__ = [
    "COUPLINGS",
    "get_coupling_fn",
    "gibbs_coupling",
    "sinkhorn_log_domain",
    "sinkhorn_unbalanced",
]
