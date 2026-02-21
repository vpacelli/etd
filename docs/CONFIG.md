# Configuration

## Overview

YAML configs drive all experiments. Each config specifies a target, shared
settings, algorithm variants, metrics, and checkpoint schedule. The runner
expands sweeps, instantiates configs, and produces reproducible results.

---

## YAML Schema

### Top Level

```yaml
experiment:
  name: str                    # Identifier (e.g., "gmm-2d-4")
  seeds: list[int]             # Random seeds for replication

  target:
    type: str                  # "gaussian" | "gmm" | "banana" | "funnel" | "logistic"
    params: dict               # Type-specific (see §Targets below)

  shared:
    n_particles: int           # N (default: 100)
    n_iterations: int          # T (default: 500)
    init:
      type: str                # "gaussian" | "prior"
      scale: float             # std for gaussian init (default: 2.0)

  checkpoints: list[int]       # Iterations to record metrics
  metrics: list[str]           # Which metrics to compute

  algorithms: list             # Algorithm entries (see below)
```

---

## Algorithm Entries

### ETD

```yaml
- label: "ETD-B"
  # --- Composable axes (select functions) ---
  cost: "euclidean"            # "euclidean" | "mahalanobis" | "linf"
  coupling: "balanced"         # "balanced" | "unbalanced" | "gibbs"
  update: "categorical"        # "categorical" | "barycentric"

  # --- Core parameters ---
  epsilon: 0.1                 # Entropic regularization
  alpha: 0.05                  # Score step size
  fdr: true                    # Tie σ = √(2α). When true, sigma is derived.
  # sigma: null                # Only used when fdr: false
  n_proposals: 25              # M per particle
  use_score: true              # Score-guided proposals
  score_clip: 5.0              # Score clipping threshold

  # --- Coupling-specific ---
  # rho: 1.0                   # Unbalanced only: τ/ε
  # sinkhorn_max_iter: 50
  # sinkhorn_tol: 1.0e-5

  # --- Update-specific ---
  # step_size: 1.0             # (0, 1], damping factor

  # --- Optional features ---
  # precondition: false
  # precond_beta: 0.9
  # dv_feedback: false
  # dv_weight: 1.0
  # sdd: false                 # Sinkhorn divergence debiasing
```

### Baselines

```yaml
- label: "SVGD"
  type: "baseline"
  method: "svgd"
  learning_rate: 0.01
  # bandwidth: "median"
  # score_clip: 5.0

- label: "ULA"
  type: "baseline"
  method: "ula"
  step_size: 0.01
  # score_clip: 5.0

- label: "MPPI"
  type: "baseline"
  method: "mppi"
  temperature: 1.0
  sigma: 0.316
  n_proposals: 25
```

---

## Target Types

### Gaussian
```yaml
target:
  type: "gaussian"
  params:
    dim: 10
    condition_number: 100      # eigenvalues log-spaced 1 to κ
```

### Mixture of Gaussians
```yaml
target:
  type: "gmm"
  params:
    dim: 2
    n_modes: 4
    arrangement: "grid"        # "grid" | "ring"
    separation: 6.0            # δ/σ ratio
    component_std: 1.0
```

### Banana
```yaml
target:
  type: "banana"
  params:
    curvature: 0.1             # b parameter
    scale: 3.0                 # s parameter
```

### Neal's Funnel
```yaml
target:
  type: "funnel"
  params:
    dim: 10
    scale: 3.0                 # controls funnel width
```

### Bayesian Logistic Regression
```yaml
target:
  type: "logistic"
  params:
    dataset: "german_credit"   # "german_credit" | "australian"
    prior_std: 5.0
    nuts_samples: 10000        # for ground truth
    nuts_warmup: 5000
```

---

## Dataclasses

### ETDConfig

```python
@dataclass(frozen=True)
class ETDConfig:
    # Scale
    n_particles: int = 100
    n_iterations: int = 500
    n_proposals: int = 25

    # Composable axes (string names → resolved to functions by build())
    cost: str = "euclidean"
    coupling: str = "balanced"
    update: str = "categorical"

    # Core
    epsilon: float = 0.1
    alpha: float = 0.05
    fdr: bool = True           # σ = √(2α) when True
    sigma: float | None = None # explicit σ when fdr=False
    use_score: bool = True
    score_clip: float = 5.0

    # Coupling
    rho: float = 1.0           # UB only
    sinkhorn_max_iter: int = 50
    sinkhorn_tol: float = 1e-5

    # Update
    step_size: float = 1.0

    # Preconditioner
    precondition: bool = False
    precond_beta: float = 0.9
    precond_delta: float = 1e-8

    # DV feedback
    dv_feedback: bool = False
    dv_weight: float = 1.0

    # SDD
    sdd: bool = False

    @property
    def resolved_sigma(self) -> float:
        if self.fdr:
            return (2 * self.alpha) ** 0.5
        if self.sigma is not None:
            return self.sigma
        raise ValueError("sigma must be set when fdr=False")
```

### Baseline Configs

```python
@dataclass(frozen=True)
class SVGDConfig:
    n_particles: int = 100
    n_iterations: int = 500
    learning_rate: float = 0.01
    bandwidth: str = "median"
    score_clip: float = 5.0
    adam_b1: float = 0.9
    adam_b2: float = 0.999

@dataclass(frozen=True)
class ULAConfig:
    n_particles: int = 100
    n_iterations: int = 500
    step_size: float = 0.01
    score_clip: float = 5.0

@dataclass(frozen=True)
class MPPIConfig:
    n_particles: int = 100
    n_iterations: int = 500
    temperature: float = 1.0
    sigma: float = 0.1
    n_proposals: int = 25
```

---

## Sweep Expansion

When a parameter value is a list, the runner expands the Cartesian product:

```yaml
- label: "ETD-B"
  cost: "euclidean"
  coupling: "balanced"
  update: "categorical"
  epsilon: [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
  use_score: true
```

Expands to 6 runs: `ETD-B_eps=0.01`, `ETD-B_eps=0.05`, ...

Multi-parameter sweeps:
```yaml
- label: "ETD-UB"
  coupling: "unbalanced"
  epsilon: [0.05, 0.1, 0.25]
  rho: [0.1, 1.0, 10.0]
```

Expands to 9 runs (3 × 3). Sweep expansion happens in the runner, not
the config dataclass. The config is always a single point.

---

## Metrics

```yaml
metrics:
  # Synthetic targets (ground truth available)
  - "energy_distance"
  - "mode_coverage"
  - "sliced_w2"
  - "mean_error"
  - "cov_error"

  # Bayesian inference (NUTS reference)
  - "mean_rmse"
  - "variance_ratio"
  - "mmd"

  # Always recorded automatically
  # - "wall_clock"
  # - "target_evals"
  # - "score_evals"
```

---

## Output Format

Results saved to `results/{experiment_name}/{timestamp}/`:

```
results/gmm-2d-4/2026-02-20_143022/
├── config.yaml          # frozen copy of input config
├── metrics.json         # {seed → {algorithm → {checkpoint → {metric → value}}}}
└── particles.npz        # {seed → {algorithm → {checkpoint → (N, d) array}}}
```

---

## Complete Example

```yaml
experiment:
  name: "gmm-2d-4"
  seeds: [0, 1, 2, 3, 4]

  target:
    type: "gmm"
    params:
      dim: 2
      n_modes: 4
      arrangement: "grid"
      separation: 6.0
      component_std: 1.0

  shared:
    n_particles: 100
    n_iterations: 500
    init:
      type: "gaussian"
      scale: 5.0

  checkpoints: [1, 5, 10, 25, 50, 100, 200, 500]

  metrics:
    - "energy_distance"
    - "mode_coverage"
    - "mean_error"

  algorithms:
    - label: "ETD-B"
      cost: "euclidean"
      coupling: "balanced"
      update: "categorical"
      use_score: true
      epsilon: 0.1
      alpha: 0.05

    - label: "ETD-UB"
      cost: "euclidean"
      coupling: "unbalanced"
      update: "categorical"
      use_score: true
      epsilon: 0.1
      alpha: 0.05
      rho: 1.0

    - label: "ETD-B-Maha"
      cost: "mahalanobis"
      coupling: "balanced"
      update: "categorical"
      use_score: true
      epsilon: 0.1
      alpha: 0.05
      precondition: true

    - label: "ETD-B-SF"
      cost: "euclidean"
      coupling: "balanced"
      update: "categorical"
      use_score: false
      epsilon: 0.1
      alpha: 0.0

    - label: "SVGD"
      type: "baseline"
      method: "svgd"
      learning_rate: 0.01

    - label: "ULA"
      type: "baseline"
      method: "ula"
      step_size: 0.01

    - label: "MPPI"
      type: "baseline"
      method: "mppi"
      temperature: 1.0
      sigma: 0.316
      n_proposals: 25
```
