# Configuration

## Overview

YAML configs drive all experiments. Each config specifies a target, shared
settings, algorithm variants, metrics, and checkpoint schedule. The runner
expands sweeps, instantiates configs, and produces reproducible results.

Configs are organized by target type:

```
experiments/configs/
├── gmm/
│   ├── 2d_4.yaml
│   ├── 2d_8.yaml
│   ├── 10d_5.yaml
│   └── 20d_5.yaml
├── banana/
│   └── 2d.yaml
├── funnel/
│   └── 10d.yaml
├── blr/
│   ├── german.yaml
│   ├── ionosphere.yaml
│   ├── ionosphere_chol.yaml
│   ├── ionosphere_inf.yaml
│   ├── ionosphere_rms.yaml
│   ├── mutation.yaml
│   └── sonar.yaml
├── sweeps/
│   └── eps_sensitivity.yaml
└── template.yaml
```

---

## YAML Schema

### Top Level

```yaml
experiment:
  name: str                    # Identifier (e.g., "gmm-2d-4")
  seeds: list[int]             # Random seeds for replication

  target:
    type: str                  # "gaussian" | "gmm" | "banana" | "funnel" | "blr"
    params: dict               # Type-specific (see Targets below)

  shared:
    n_particles: int           # N (default: 100)
    n_iterations: int          # T (default: 500)
    progress_segments: int     # Optional: extra progress ticks between checkpoints
    init:
      type: str                # "gaussian" | "prior"
      scale: float             # std for gaussian init (default: 2.0)

  checkpoints: list[int]       # Iterations to record metrics
  metrics: list[str]           # Which metrics to compute

  algorithms: list             # Algorithm entries (see below)
```

---

## Algorithm Entries

### Dispatch Rules

Each algorithm entry is dispatched by its `type` and `method` fields:

| `type` | `method` | Algorithm |
|--------|----------|-----------|
| *(omitted)* | *(omitted)* | ETD |
| *(omitted)* | `"sdd"` | SDD |
| `"baseline"` | `"svgd"` | SVGD |
| `"baseline"` | `"ula"` | ULA |
| `"baseline"` | `"mala"` | MALA |
| `"baseline"` | `"mppi"` | MPPI |

### Meta Fields

These fields control display and filtering but are **not** passed to configs:

- `label`: Algorithm name for display and results keying.
- `sublabel`: Appended to label: `"ETD-B (MALA)"`.
- `display`: Plotting metadata `{family, color, linestyle, group}`.
- `enabled`: Set `false` to skip (default: `true`).

---

### ETD / SDD

ETD entries use **nested sub-configs** for each algorithmic component.
String shorthands are accepted where sensible.

```yaml
- label: "ETD-B"
  # --- Core ---
  epsilon: 0.1                 # Entropic regularization

  # --- Proposal (score-guided drift-diffusion) ---
  proposal:                    # or just omit for defaults
    type: "score"              # "score" | "score_free"
    count: 25                  # M proposals (pooled across particles)
    alpha: 0.05                # Drift step size
    fdr: true                  # sigma = sqrt(2*alpha)
    # sigma: 0.0              # Explicit noise; required when fdr=false
    clip_score: 5.0            # Score clipping threshold

  # --- Cost (transport cost function) ---
  cost: "euclidean"            # String shorthand
  # cost:                      # Or dict for extra params:
  #   type: "imq"
  #   c: 1.0
  #   normalize: "median"      # "median" | "mean"

  # --- Coupling (Sinkhorn solver) ---
  coupling: "balanced"         # String shorthand
  # coupling:                  # Or dict for tuning:
  #   type: "balanced"         # "balanced" | "unbalanced" | "gibbs"
  #   iterations: 50
  #   tolerance: 1e-2
  #   rho: 1.0                # Unbalanced only: tau/epsilon

  # --- Update (particle update rule) ---
  update: "categorical"        # String shorthand
  # update:                    # Or dict:
  #   type: "categorical"     # "categorical" | "barycentric"
  #   damping: 1.0            # (0, 1], step size damping

  # --- Preconditioner ---
  # preconditioner: "cholesky" # String shorthand
  # preconditioner:            # Or dict:
  #   type: "cholesky"        # "none" | "rmsprop" | "cholesky"
  #   proposals: true          # Apply to proposal drift + noise
  #   cost: true               # Apply to cost whitening
  #   source: "positions"      # "scores" | "positions" (Cholesky only)
  #   ema: 0.8                 # EMA on covariance (0 = fresh; Cholesky only)
  #   shrinkage: 0.1           # Ledoit-Wolf shrinkage (Cholesky only)
  #   jitter: 1e-6             # PD guarantee (Cholesky only)
  #   clip_score: 0.0          # 0 = inherit parent; inf = raw scores
  #   beta: 0.9                # RMSProp EMA decay
  #   delta: 1e-8              # RMSProp floor

  # --- Mutation (post-transport MCMC) ---
  # mutation:
  #   kernel: "mala"           # "none" | "mala" | "rwm"
  #   steps: 5                 # MCMC sub-steps per ETD iteration
  #   stepsize: 0.01           # MALA/RWM step size h
  #   cholesky: true           # Use ensemble Cholesky for proposal covariance
  #   clip_score: null          # null -> inherit from parent

  # --- Feedback (Donsker-Varadhan) ---
  # feedback:
  #   enabled: true
  #   weight: 1.0
```

#### SDD-Specific Fields

SDD entries additionally accept:

```yaml
- label: "SDD"
  method: "sdd"
  # ... all ETD fields above ...
  self_coupling:
    epsilon: 0.1
    iterations: 50
    tolerance: 1e-2
  eta: 0.5                    # SDD step size
```

#### Langevin Cost (LRET) Auto-Defaults

When `cost: "langevin"` (or `cost: {type: "langevin"}`):
- If `proposal.fdr == true` and `proposal.alpha` is not explicitly set,
  `alpha` defaults to `epsilon` (FDR relation for the Langevin residual).
- `proposal.type` must be `"score"` (score-guided proposals required).
- Use `cost: {type: "langevin", whiten: true}` with a Cholesky preconditioner
  for whitened Mahalanobis LRET.

#### Schedules

Any numeric parameter can be annealed by replacing the value with a
schedule dict:

```yaml
epsilon: {schedule: "linear_warmup", value: 0.1, warmup: 100}
proposal:
  alpha: {schedule: "cosine_decay", value: 0.05, end: 0.001}
```

Schedule keys use dotted paths internally: `"epsilon"`, `"proposal.alpha"`,
`"feedback.weight"`, etc.

Supported schedule types: `"linear_warmup"`, `"cosine_decay"`, `"exponential_decay"`.

---

### Baselines

```yaml
- label: "SVGD"
  type: "baseline"
  method: "svgd"
  stepsize: 0.01               # was: learning_rate
  clip_score: 5.0              # was: score_clip

- label: "ULA"
  type: "baseline"
  method: "ula"
  stepsize: 0.01               # was: step_size
  clip_score: 5.0              # was: score_clip

- label: "MALA"
  type: "baseline"
  method: "mala"
  stepsize: 0.01               # was: step_size
  clip_score: 5.0              # was: score_clip
  preconditioner: "cholesky"   # string shorthand; or full dict

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
    condition_number: 100      # eigenvalues log-spaced 1 to kappa
```

### Mixture of Gaussians
```yaml
target:
  type: "gmm"
  params:
    dim: 2
    n_modes: 4
    arrangement: "grid"        # "grid" | "ring"
    separation: 6.0
    component_std: 1.0
```

### Banana
```yaml
target:
  type: "banana"
  params:
    dim: 2
    curvature: 0.1
    offset: 100.0
    sigma1: 10.0
    sigma2: 1.0
```

### Neal's Funnel
```yaml
target:
  type: "funnel"
  params:
    dim: 10
    sigma_v: 3.0
```

### Bayesian Logistic Regression (BLR)
```yaml
target:
  type: "blr"
  params:
    dataset: "german_credit"   # "german_credit" | "ionosphere" | "sonar"
    prior_std: 5.0
```

---

## Dataclasses

### Sub-Config Hierarchy

```
ETDConfig
├── ProposalConfig     — proposal generation (count, drift, noise, clipping)
├── CostConfig         — transport cost (type, normalization, params)
├── CouplingConfig     — Sinkhorn coupling (type, iterations, tolerance)
├── UpdateConfig       — particle update (type, damping)
├── PreconditionerConfig — diagonal/Cholesky preconditioning
├── MutationConfig     — post-transport MCMC mutation
└── FeedbackConfig     — Donsker-Varadhan feedback signal
```

### ETDConfig

```python
@dataclass(frozen=True)
class ETDConfig:
    n_particles: int = 100
    n_iterations: int = 500
    epsilon: float = 0.1
    proposal: ProposalConfig
    cost: CostConfig
    coupling: CouplingConfig
    update: UpdateConfig
    preconditioner: PreconditionerConfig
    mutation: MutationConfig
    feedback: FeedbackConfig
    schedules: tuple = ()      # ((dotted_key, Schedule), ...)
```

### ProposalConfig

```python
@dataclass(frozen=True)
class ProposalConfig:
    type: str = "score"        # "score" | "score_free"
    count: int = 25
    alpha: float = 0.05
    fdr: bool = True           # sigma = sqrt(2*alpha)
    sigma: float = 0.0         # explicit; required when score_free or fdr=False
    clip_score: float = 5.0
```

### CostConfig

```python
@dataclass(frozen=True)
class CostConfig:
    type: str = "euclidean"    # "euclidean" | "linf" | "imq" | "langevin"
    normalize: str = "median"  # "median" | "mean"
    params: tuple = ()         # sorted (key, value) pairs
```

### CouplingConfig

```python
@dataclass(frozen=True)
class CouplingConfig:
    type: str = "balanced"     # "balanced" | "unbalanced" | "gibbs"
    iterations: int = 50
    tolerance: float = 1e-2
    rho: float = 1.0           # unbalanced only: tau/epsilon
```

### UpdateConfig

```python
@dataclass(frozen=True)
class UpdateConfig:
    type: str = "categorical"  # "categorical" | "barycentric"
    damping: float = 1.0       # (0, 1]
```

### PreconditionerConfig

```python
@dataclass(frozen=True)
class PreconditionerConfig:
    type: str = "none"         # "none" | "rmsprop" | "cholesky"
    proposals: bool = False    # apply to proposal drift + noise
    cost: bool = False         # apply to cost whitening
    source: str = "scores"     # "scores" | "positions"
    clip_score: float = 0.0    # 0 = inherit parent; inf = raw/unclipped
    beta: float = 0.9          # RMSProp EMA decay
    delta: float = 1e-8        # RMSProp floor
    shrinkage: float = 0.1     # Cholesky: Ledoit-Wolf shrinkage
    jitter: float = 1e-6       # Cholesky: PD guarantee
    ema: float = 0.0           # Cholesky: EMA on covariance (0 = fresh)
```

### MutationConfig

```python
@dataclass(frozen=True)
class MutationConfig:
    kernel: str = "none"       # "none" | "mala" | "rwm"
    steps: int = 5
    stepsize: float = 0.01
    cholesky: bool = True
    clip_score: Optional[float] = None  # None -> inherit parent
```

### FeedbackConfig

```python
@dataclass(frozen=True)
class FeedbackConfig:
    enabled: bool = False
    weight: float = 1.0
```

### Baseline Configs

```python
@dataclass(frozen=True)
class SVGDConfig:
    n_particles: int = 100
    n_iterations: int = 500
    stepsize: float = 0.01
    bandwidth: str = "median"
    clip_score: float = 5.0
    adam_b1: float = 0.9
    adam_b2: float = 0.999

@dataclass(frozen=True)
class ULAConfig:
    n_particles: int = 100
    n_iterations: int = 500
    stepsize: float = 0.01
    clip_score: float = 5.0

@dataclass(frozen=True)
class MALAConfig:
    n_particles: int = 100
    n_iterations: int = 500
    stepsize: float = 0.01
    clip_score: float = 5.0
    preconditioner: PreconditionerConfig = PreconditionerConfig()

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

When a parameter value is a list, the runner expands the Cartesian product.
Nested sub-config fields are supported:

```yaml
- label: "ETD-B"
  cost: "euclidean"
  coupling: "balanced"
  epsilon: [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
  proposal:
    alpha: 0.05
```

Expands to 6 runs: `ETD-B_eps=0.01`, `ETD-B_eps=0.05`, ...

Multi-parameter sweeps (including nested):
```yaml
- label: "ETD-UB"
  coupling:
    type: "unbalanced"
    rho: [0.1, 1.0, 10.0]
  epsilon: [0.05, 0.1, 0.25]
```

Expands to 9 runs (3 x 3). Sweep expansion happens in the runner, not
the config dataclass. The config is always a single point.

Label abbreviations for sweep suffixes:

| Dotted key | Abbreviation |
|-----------|-------------|
| `epsilon` | `eps` |
| `proposal.alpha` | `alpha` |
| `proposal.count` | `M` |
| `proposal.clip_score` | `clip` |
| `coupling.tolerance` | `tol` |
| `coupling.rho` | `rho` |
| `update.damping` | `damp` |
| `mutation.steps` | `mut.steps` |
| `mutation.stepsize` | `mut.h` |
| `feedback.weight` | `dv` |

---

## Metrics

```yaml
metrics:
  # Synthetic targets (ground truth available)
  - "energy_distance"
  - "sliced_wasserstein"
  - "mode_proximity"
  - "mode_balance"
  - "mean_error"

  # Bayesian inference (NUTS reference)
  - "mean_rmse"
  - "variance_ratio_ref"

  # Diagnostic (from step info)
  - "coupling_ess"
  - "sinkhorn_iters"
```

---

## Output Format

Results saved to `results/{experiment_name}/{timestamp}/`:

```
results/gmm-2d-4/2026-02-20_143022/
├── config.yaml          # frozen copy of input config
├── metrics.json         # {seed -> {algorithm -> {checkpoint -> {metric -> value}}}}
├── particles.npz        # flat keys: "seed0__ETD-B__iter100" -> (N, d) array
├── reference.npz        # reference samples (if available)
└── metadata.json        # display styles for downstream plotting
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

  checkpoints: [0, 1, 10, 50, 100, 200, 500]

  metrics:
    - "energy_distance"
    - "sliced_wasserstein"
    - "mode_proximity"
    - "mode_balance"

  algorithms:
    - label: "ETD-B"
      epsilon: 0.1
      proposal:
        alpha: 0.05
        count: 25
      cost: "euclidean"
      coupling: "balanced"
      update: "categorical"

    - label: "ETD-B-Chol"
      epsilon: 0.1
      proposal:
        alpha: 0.05
      cost: "euclidean"
      coupling: "balanced"
      preconditioner:
        type: "cholesky"
        proposals: true
        cost: true
        source: "positions"
        ema: 0.8
        shrinkage: 0.1

    - label: "LRET-B"
      epsilon: 0.1
      cost: "langevin"
      coupling: "balanced"
      # alpha auto-set to epsilon via FDR

    - label: "SVGD"
      type: "baseline"
      method: "svgd"
      stepsize: 0.1
      clip_score: 5.0

    - label: "MALA"
      type: "baseline"
      method: "mala"
      stepsize: 0.01
      clip_score: 5.0

    - label: "MPPI"
      type: "baseline"
      method: "mppi"
      temperature: 1.0
      sigma: 0.316
      n_proposals: 25
```
