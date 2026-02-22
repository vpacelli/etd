"""Proposal generation registry."""

from etd.proposals.langevin import (
    PROPOSALS,
    clip_scores,
    get_proposal_fn,
    langevin_proposals,
    update_preconditioner,
)
from etd.proposals.preconditioner import (
    compute_diagonal_P,
    compute_ensemble_cholesky,
    update_rmsprop_accum,
)

__all__ = [
    "PROPOSALS",
    "clip_scores",
    "compute_diagonal_P",
    "compute_ensemble_cholesky",
    "get_proposal_fn",
    "langevin_proposals",
    "update_preconditioner",
    "update_rmsprop_accum",
]
