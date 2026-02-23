"""Shared test fixtures and helpers.

Provides ``make_test_config`` / ``make_test_sdd_config`` helpers that
accept flat kwargs (old-style names) and construct nested sub-configs,
making test migration mechanical.
"""

from etd.types import (
    CostConfig,
    CouplingConfig,
    ETDConfig,
    FeedbackConfig,
    MutationConfig,
    PreconditionerConfig,
    ProposalConfig,
    SelfCouplingConfig,
    UpdateConfig,
)


def make_test_config(**kw) -> ETDConfig:
    """Build ETDConfig from flat kwargs for test convenience.

    Accepts both old-style flat field names and new nested config
    objects.  Old-style names are mapped to the appropriate sub-config.

    Examples::

        # Old-style (flat kwargs):
        make_test_config(coupling="balanced", n_proposals=25, use_score=True)

        # New-style (nested config objects):
        make_test_config(proposal=ProposalConfig(count=25))

        # Mix:
        make_test_config(coupling="balanced", proposal=ProposalConfig(count=25))
    """
    # --- Proposal ---
    if "proposal" in kw and isinstance(kw["proposal"], ProposalConfig):
        proposal = kw.pop("proposal")
    else:
        proposal_type = "score" if kw.pop("use_score", True) else "score_free"
        proposal = ProposalConfig(
            type=proposal_type,
            count=kw.pop("n_proposals", kw.pop("count", 25)),
            alpha=kw.pop("alpha", 0.05),
            fdr=kw.pop("fdr", True),
            sigma=kw.pop("sigma", 0.0),
            clip_score=kw.pop("score_clip", kw.pop("clip_score", 5.0)),
        )

    # --- Cost ---
    if "cost" in kw and isinstance(kw["cost"], CostConfig):
        cost = kw.pop("cost")
    else:
        cost_val = kw.pop("cost", "euclidean")
        cost = CostConfig(
            type=cost_val,
            normalize=kw.pop("cost_normalize", kw.pop("normalize", "median")),
            params=kw.pop("cost_params", kw.pop("params", ())),
        )

    # --- Coupling ---
    if "coupling" in kw and isinstance(kw["coupling"], CouplingConfig):
        coupling = kw.pop("coupling")
    else:
        coupling = CouplingConfig(
            type=kw.pop("coupling", "balanced"),
            iterations=kw.pop("sinkhorn_max_iter", kw.pop("iterations", 50)),
            tolerance=kw.pop("sinkhorn_tol", kw.pop("tolerance", 1e-2)),
            rho=kw.pop("rho", 1.0),
        )

    # --- Update ---
    if "update" in kw and isinstance(kw["update"], UpdateConfig):
        update = kw.pop("update")
    else:
        update = UpdateConfig(
            type=kw.pop("update", "categorical"),
            damping=kw.pop("step_size", kw.pop("damping", 1.0)),
        )

    # --- Preconditioner ---
    preconditioner = kw.pop("preconditioner", PreconditionerConfig())

    # --- Mutation ---
    mutation = kw.pop("mutation", MutationConfig())

    # --- Feedback ---
    if "feedback" in kw and isinstance(kw["feedback"], FeedbackConfig):
        feedback = kw.pop("feedback")
    else:
        feedback = FeedbackConfig(
            enabled=kw.pop("dv_feedback", kw.pop("enabled", False)),
            weight=kw.pop("dv_weight", kw.pop("weight", 1.0)),
        )

    # --- Legacy flat fields to discard silently ---
    kw.pop("precondition", None)
    kw.pop("whiten", None)
    kw.pop("precond_beta", None)
    kw.pop("precond_delta", None)

    return ETDConfig(
        n_particles=kw.pop("n_particles", 100),
        n_iterations=kw.pop("n_iterations", 500),
        epsilon=kw.pop("epsilon", 0.1),
        proposal=proposal,
        cost=cost,
        coupling=coupling,
        update=update,
        preconditioner=preconditioner,
        mutation=mutation,
        feedback=feedback,
        schedules=kw.pop("schedules", ()),
    )


def make_test_sdd_config(**kw):
    """Build SDDConfig from flat kwargs for test convenience."""
    from etd.extensions.sdd import SDDConfig

    # --- Proposal ---
    if "proposal" in kw and isinstance(kw["proposal"], ProposalConfig):
        proposal = kw.pop("proposal")
    else:
        proposal_type = "score" if kw.pop("use_score", True) else "score_free"
        proposal = ProposalConfig(
            type=proposal_type,
            count=kw.pop("n_proposals", kw.pop("count", 25)),
            alpha=kw.pop("alpha", 0.05),
            fdr=kw.pop("fdr", True),
            sigma=kw.pop("sigma", 0.0),
            clip_score=kw.pop("score_clip", kw.pop("clip_score", 5.0)),
        )

    # --- Cost ---
    if "cost" in kw and isinstance(kw["cost"], CostConfig):
        cost = kw.pop("cost")
    else:
        cost_val = kw.pop("cost", "euclidean")
        cost = CostConfig(
            type=cost_val,
            normalize=kw.pop("cost_normalize", kw.pop("normalize", "median")),
            params=kw.pop("cost_params", kw.pop("params", ())),
        )

    # --- Coupling ---
    if "coupling" in kw and isinstance(kw["coupling"], CouplingConfig):
        coupling = kw.pop("coupling")
    else:
        coupling = CouplingConfig(
            type=kw.pop("coupling", "balanced"),
            iterations=kw.pop("sinkhorn_max_iter", kw.pop("iterations", 50)),
            tolerance=kw.pop("sinkhorn_tol", kw.pop("tolerance", 1e-2)),
            rho=kw.pop("rho", 1.0),
        )

    # --- Update ---
    if "update" in kw and isinstance(kw["update"], UpdateConfig):
        update = kw.pop("update")
    else:
        update = UpdateConfig(
            type=kw.pop("update", "categorical"),
            damping=kw.pop("step_size", kw.pop("damping", 1.0)),
        )

    # --- Preconditioner ---
    preconditioner = kw.pop("preconditioner", PreconditionerConfig())

    # --- Mutation ---
    mutation = kw.pop("mutation", MutationConfig())

    # --- Feedback ---
    if "feedback" in kw and isinstance(kw["feedback"], FeedbackConfig):
        feedback = kw.pop("feedback")
    else:
        feedback = FeedbackConfig(
            enabled=kw.pop("dv_feedback", kw.pop("enabled", False)),
            weight=kw.pop("dv_weight", kw.pop("weight", 1.0)),
        )

    # --- Self-coupling ---
    if "self_coupling" in kw and isinstance(kw["self_coupling"], SelfCouplingConfig):
        self_coupling = kw.pop("self_coupling")
    else:
        self_coupling = SelfCouplingConfig(
            epsilon=kw.pop("self_epsilon", 0.1),
            iterations=kw.pop("self_sinkhorn_max_iter", 50),
            tolerance=kw.pop("self_sinkhorn_tol", 1e-2),
        )

    # --- Legacy flat fields to discard silently ---
    kw.pop("precondition", None)
    kw.pop("whiten", None)
    kw.pop("precond_beta", None)
    kw.pop("precond_delta", None)

    return SDDConfig(
        n_particles=kw.pop("n_particles", 100),
        n_iterations=kw.pop("n_iterations", 500),
        epsilon=kw.pop("epsilon", 0.1),
        proposal=proposal,
        cost=cost,
        coupling=coupling,
        update=update,
        preconditioner=preconditioner,
        mutation=mutation,
        feedback=feedback,
        self_coupling=self_coupling,
        eta=kw.pop("sdd_step_size", kw.pop("eta", 0.5)),
        schedules=kw.pop("schedules", ()),
    )
