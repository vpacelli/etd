# Architecture

## Overview

Particle-based variational inference via entropic optimal transport. The
algorithm — Entropic Transport Descent (ETD) — minimizes an entropic-regularized
optimal transport objective to evolve a particle ensemble toward a target
distribution. The goal is a NeurIPS paper showing that ETD's transport coupling
addresses mode collapse better than SVGD's kernel repulsion while providing
uncertainty quantification via the fitted covariance.

## Design Principles

| Principle | Rationale |
|-----------|-----------|
| **Composition over configuration** | The axes of variation (cost, coupling, update, proposals) are separate composable functions, not fields in a monolithic config. |
| **Correctness first** | Wrong results are worthless. |
| **One way to do things** | No V1/V2 migration, no legacy wrappers. |
| **Config-driven runs** | Every result is reproducible from YAML + git commit. |
| **Flat over nested** | Simple modules, no abstract factory patterns. |
| **JAX contained** | Pure Python for configs and I/O; JAX only in numerics. |

This is research code for a paper, not production software.

## Directory Structure

```
etd/
├── src/etd/
│   ├── __init__.py
│   ├── types.py                    # Particles, Target protocol, ETDState
│   │
│   ├── costs/                      # Cost matrix construction
│   │   ├── __init__.py             # get_cost_fn() registry
│   │   ├── euclidean.py            # c(x,y) = ||x-y||²/2
│   │   ├── mahalanobis.py          # c(x,y) = (x-y)ᵀ Σ⁻¹ (x-y) / 2
│   │   ├── linf.py                 # c(x,y) = ||x-y||_∞
│   │   └── normalize.py            # Median heuristic, shared utility
│   │
│   ├── coupling/                   # Entropic OT solvers
│   │   ├── __init__.py             # get_coupling_fn() registry
│   │   ├── sinkhorn.py             # Log-domain balanced Sinkhorn
│   │   ├── unbalanced.py           # Log-domain unbalanced (KL penalty)
│   │   └── gibbs.py                # Closed-form softmax (SR, 0 iterations)
│   │
│   ├── proposals/                  # Proposal generation
│   │   ├── __init__.py
│   │   └── langevin.py             # Score-guided + score-free, optional precond
│   │
│   ├── update/                     # Particle update rules
│   │   ├── __init__.py
│   │   ├── categorical.py          # Systematic resampling from coupling
│   │   └── barycentric.py          # Weighted mean with step size η
│   │
│   ├── weights.py                  # IS-corrected target weights (always on)
│   ├── step.py                     # The ~20-line ETD step function
│   │
│   ├── targets/                    # Target distributions
│   │   ├── __init__.py             # get_target() registry
│   │   ├── gaussian.py             # Single Gaussian (variable κ)
│   │   ├── gmm.py                  # Mixture of Gaussians
│   │   ├── banana.py               # Banana-shaped posterior
│   │   ├── funnel.py               # Neal's funnel
│   │   └── logistic.py             # Bayesian logistic regression
│   │
│   ├── baselines/                  # Comparison methods
│   │   ├── __init__.py
│   │   ├── svgd.py
│   │   ├── ula.py
│   │   └── mppi.py
│   │
│   ├── diagnostics/
│   │   └── metrics.py              # Energy dist, mode coverage, MMD, etc.
│   │
│   └── extensions/                 # Optional algorithm wrappers
│       └── sdd.py                  # Sinkhorn divergence debiasing
│
├── experiments/
│   ├── run.py                      # YAML → run → save
│   ├── configs/                    # YAML experiment definitions
│   │   └── sweeps/                 # Multi-parameter sweeps
│   └── scripts/                    # One-off analysis scripts
│
├── figures/
│   ├── style.py                    # Crimson palette, contour defaults
│   └── paper/                      # One script per figure
│
├── results/                        # gitignored
│   └── {experiment}/{timestamp}/
│       ├── config.yaml
│       ├── metrics.json
│       └── particles.npz
│
├── tests/
│   ├── test_primitives.py
│   ├── test_sinkhorn.py
│   ├── test_etd_gaussian.py
│   └── test_determinism.py
│
├── docs/                           # These documents
├── pyproject.toml
└── .gitignore
```

## Layer Diagram

```
┌──────────────────────────────────────────────────────┐
│  EXPERIMENTS  (volatile)                             │
│  configs, runner, analysis, figures                  │
└────────────────────┬─────────────────────────────────┘
                     │ imports
┌────────────────────▼─────────────────────────────────┐
│  ALGORITHMS  src/etd/  (stable)                      │
│  step, baselines, targets, diagnostics, extensions   │
└────────────────────┬─────────────────────────────────┘
                     │ imports
┌────────────────────▼─────────────────────────────────┐
│  PRIMITIVES  src/etd/{costs,coupling,proposals,...}   │
│  costs, coupling solvers, proposals, update rules    │
└──────────────────────────────────────────────────────┘
```

Dependencies flow downward only. Never import upward. The primitives
layer has no knowledge of `ETDConfig`, `ETDState`, or experiment structure.

## Composable Axes

ETD's variation is along four orthogonal axes. Each axis is a **directory**
containing interchangeable functions with the same signature:

| Axis | Directory | Options | Signature |
|------|-----------|---------|-----------|
| **Cost** | `costs/` | euclidean, mahalanobis, linf, *(extensible)* | `(positions, proposals, aux) → C` |
| **Coupling** | `coupling/` | gibbs, sinkhorn, unbalanced | `(C, log_a, log_b, eps, ...) → (log_γ, f, g)` |
| **Update** | `update/` | categorical, barycentric | `(key, positions, proposals, log_γ) → new_positions` |
| **Proposals** | `proposals/` | langevin (score-guided/free, ±precond) | `(key, positions, target, config, accum) → proposals` |

Adding a new variant (e.g., a multi-bandwidth kernel cost) means writing
one function and registering it. No other code changes.

## Dependencies

```
# Core
jax >= 0.4.20
jaxlib >= 0.4.20
numpy
matplotlib

# Config
pyyaml

# Terminal output
rich

# Testing
chex
pytest

# Reference sampler
numpyro

# Data (for BLR)
scikit-learn
```
