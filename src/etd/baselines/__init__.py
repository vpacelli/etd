"""Baseline inference algorithms: SVGD, ULA, MPPI."""

from etd.baselines.mppi import MPPIConfig, MPPIState, init as mppi_init, step as mppi_step
from etd.baselines.svgd import SVGDConfig, SVGDState, init as svgd_init, step as svgd_step
from etd.baselines.ula import ULAConfig, ULAState, init as ula_init, step as ula_step

BASELINES = {
    "svgd": {"config": SVGDConfig, "init": svgd_init, "step": svgd_step},
    "ula":  {"config": ULAConfig,  "init": ula_init,  "step": ula_step},
    "mppi": {"config": MPPIConfig, "init": mppi_init, "step": mppi_step},
}


def get_baseline(method: str) -> dict:
    """Return ``{"config": Class, "init": fn, "step": fn}`` for a baseline.

    Args:
        method: Baseline name (``"svgd"``, ``"ula"``, or ``"mppi"``).

    Raises:
        KeyError: If *method* is not in the registry.
    """
    if method not in BASELINES:
        raise KeyError(
            f"Unknown baseline '{method}'. Available: {list(BASELINES)}"
        )
    return BASELINES[method]


__all__ = [
    "BASELINES",
    "get_baseline",
    "SVGDConfig", "SVGDState", "svgd_init", "svgd_step",
    "ULAConfig", "ULAState", "ula_init", "ula_step",
    "MPPIConfig", "MPPIState", "mppi_init", "mppi_step",
]
