# Algorithm

## The ETD Step

ETD iterates four stages. Each is a composable function.

```
propose  →  cost  →  couple  →  update
```

### Pseudocode

```
Given: particles X = {x_i}, target π, config
  1. proposals Y = propose(X, π, config)             # (P, d), P = N*M
  2. log_b    = importance_weights(Y, X, π, config)   # (P,)
  3. C        = cost_fn(X, Y, aux)                    # (N, P)
  4. C, med   = normalize(C)                          # median heuristic
  5. log_γ, f, g = coupling_fn(C, log_a, log_b, ε)   # (N,P), (N,), (P,)
  6. X_new    = update_fn(key, X, Y, log_γ)           # (N, d)
```

The step function in code should be ~20 lines. Each composable piece is
described below.

---

## 1. Proposals

Generate $NM$ candidate positions around current particles via Langevin
dynamics:

**Score-guided** (default, requires `score`):
$$y_{ij} = x_i + \alpha \, \nabla \log \pi(x_i) + \sigma \, \xi_{ij}, \quad \xi_{ij} \sim \mathcal{N}(0, I)$$

**Score-free** (`use_score: false`, requires only `log_prob`):
$$y_{ij} = x_i + \sigma \, \xi_{ij}$$

**Preconditioned** (`precondition: true`):
$$y_{ij} = x_i + \alpha \, P \odot \nabla \log \pi(x_i) + \sqrt{2\alpha\rho} \, P \odot \xi_{ij}$$

where $P = 1/\sqrt{G_t + \delta}$ is a diagonal preconditioner from an
RMSProp-style accumulator (see §Preconditioner below).

All $NM$ proposals are collected into a single shared pool $\{y_j\}_{j=1}^{P}$.
This pooling is what makes ETD an interacting particle system: a particle
near mode A can couple to proposals generated near mode B.

### Fluctuation-Dissipation Relation (FDR)

When `fdr: true` (default), the noise scale is tied to the score step:
$$\sigma = \sqrt{2\alpha}$$

This ensures the proposal distribution is a proper Langevin discretization.
The parameter $\alpha$ is set directly. When `fdr: false`, $\alpha$ and
$\sigma$ are independent.

### Score Clipping

Clip score vectors to maximum norm $c$ (default 5.0):
```python
scale = min(1.0, c / max(||s||, 1e-8))
s_clipped = scale * s
```

This is essential. Without clipping, BLR diverges.

---

## 2. Target Weights (Importance Correction)

Compute importance-corrected target weights for each proposal:

$$\log b_j = \log \pi(y_j) - \log q_\mu(y_j)$$

where the pooled proposal mixture density is:

$$q_\mu(y) = \frac{1}{N}\sum_{i=1}^{N} \mathcal{N}(y;\; x_i + \alpha \nabla\log\pi(x_i),\; \sigma^2 I)$$

Then normalize $b$ to a probability vector via `logsumexp`.

**Why always on:** Without IS correction, the standard target weights
$b_j \propto \pi(y_j)$ produce a biased stationary distribution
$\pi_\varepsilon^*(x) \propto \pi(x) \cdot [\pi * G_{\varepsilon_\text{eff}}](x)$
(Theorem 6.1 in the draft). The importance correction eliminates this
bias in the balanced coupling case (Proposition 6.3). This was discovered
after the initial implementation and empirically confirmed. There is no
reason to disable it.

**Floor:** Set $q_\mu(y_j) \geq e^{-30} \cdot \max_k q_\mu(y_k)$ to
prevent division blowup from outlier proposals.

**Preconditioned case:** When proposals use a diagonal preconditioner,
the proposal density must use the preconditioned covariance
$\text{diag}(2\alpha\rho \, P^2)$ per component, including the correct
log-determinant. A mismatch between the proposal distribution used for
generation and the density used for IS weighting invalidates the correction.

---

## 3. Cost Functions

Each cost function maps particle-proposal pairs to a scalar transport cost.

### Euclidean (default)

$$C_{ij} = \frac{1}{2}\|x_i - y_j\|^2$$

Standard quadratic cost. In high dimensions, squared Euclidean distances
concentrate: all entries $C_{ij} \approx \mu_d$ with fluctuations
$O(\sqrt{d})$, so the signal-to-noise ratio in the Gibbs kernel scales as
$O(1/\sqrt{d})$. This is the dimensional scaling problem that motivates
alternative costs.

### Mahalanobis (preconditioned)

$$C_{ij} = \frac{1}{2}(x_i - y_j)^\top \text{diag}(P)^{-2} (x_i - y_j)$$

Uses the diagonal preconditioner $P$ from the RMSProp accumulator.
Whitens the distance distribution, reducing concentration. Justified by
covariance-modulated OT theory (Burger, Erbar, Hoffmann, Matthes, and
Schlichting, 2024): modulating the kinetic energy by the ensemble covariance
yields convergence rates independent of the target's condition number.

### $L_\infty$

$$C_{ij} = \|x_i - y_j\|_\infty$$

The max-coordinate distance. Does not concentrate as badly as $L_2$ in high
dimensions (controlled by max rather than sum). The Gibbs kernel
$\exp(-\|x-y\|_\infty / \varepsilon)$ produces softer, more informative
couplings in high $d$.

### Langevin-Residual (LRET)

$$C_{ij} = \frac{\|y_j - x_i - \varepsilon\, s(x_i)\|^2}{4\varepsilon}$$

where $s(x) = \nabla\log\pi(x)$.  Replaces the Brownian reference process
with Langevin diffusion in the Schrödinger bridge formulation.  The cost
measures *how far is $y_j$ from where Langevin would take $x_i$ in time
$\varepsilon$?*

**Why it matters:** Expanding the cost gives three terms:

| Term | Role | Separability |
|------|------|-------------|
| $\|y - x\|^2 / 4\varepsilon$ | Geometric proximity (standard) | Non-separable |
| $-\frac{1}{2} s(x)^\top(y - x)$ | Score-displacement coupling | **Non-separable** — survives in balanced OT |
| $\frac{\varepsilon}{4}\|s(x)\|^2$ | Score magnitude (absorbed by source dual) | Separable in $x$ |

The cross-term $-\frac{1}{2}s(x)^\top(y-x)$ is non-separable: it
couples source and target positions through the score.  In balanced
Sinkhorn, separable terms are absorbed by dual variables and become
invisible to the coupling.  This cross-term survives and directly shapes
which proposals each particle couples to — encoding target geometry into
the transport plan.

**Score-free degeneracy:** Setting $s = 0$ recovers the standard
Euclidean cost $\|y-x\|^2/(4\varepsilon)$, making LRET a strict
generalization of ETD.

**Graceful degradation:** When the coupling flattens toward product
measure (high $d$), LRET's default is independent Langevin dynamics,
not importance sampling — a qualitatively better floor than standard ETD.

**FDR coupling:** With LRET, $\varepsilon$ plays a dual role: Langevin
step size in proposals AND SB temperature in the cost.  The default
wiring sets $\alpha = \varepsilon$ (one fewer hyperparameter).

**Whitened mode:** With a Cholesky preconditioner $L$:

$$m_i = x_i + \varepsilon\, (LL^\top)\, s_i, \quad
C_{ij} = \frac{\|L^{-1}(y_j - m_i)\|^2}{4\varepsilon}$$

**YAML usage:**

```yaml
# Minimal LRET (FDR defaults: α = ε, σ = √(2ε))
- label: "LRET-B"
  cost: {type: langevin}
  coupling: "balanced"
  epsilon: 0.1

# LRET with whitened cost
- label: "LRET-B-W"
  cost: {type: langevin, whiten: true}
  coupling: "balanced"
  epsilon: 0.1
  preconditioner:
    type: cholesky
    proposals: true
    cost: true

# LRET with explicit alpha override
- label: "LRET-B-Custom"
  cost: {type: langevin}
  coupling: "balanced"
  epsilon: 0.1
  alpha: 0.03
```

**SDD integration:** The Langevin cost applies to the cross-coupling
only.  Self-coupling (particle→particle) uses Euclidean cost — there is
no Langevin reference for self-transport.

### Cost Normalization

After computing the raw cost matrix, normalize to a unit statistic so that
$\varepsilon$ is interpretable across problems and iterations:
$\varepsilon = 0.1$ always means "10% of the typical pairwise cost."

Two methods are available, selected by ``cost_normalize`` in the config:

| Method | Default? | Statistic | Complexity | Notes |
|--------|----------|-----------|------------|-------|
| ``"median"`` | Yes | median(C) | O(n log n) sort | Robust to outliers; uses strided subsample for speed |
| ``"mean"`` | No | mean(C) | O(n) | Cheaper; less robust to heavy tails |

```python
# median (default)
C_med = median(C)
C = C / max(C_med, 1e-8)

# mean
C_mean = mean(C)
C = C / max(C_mean, 1e-8)
```

Both methods guard against zero or near-zero scales. The info dict reports
the computed scale as ``"cost_scale"`` (ETD) or ``"cost_scale_cross"`` /
``"cost_scale_self"`` (SDD).

**YAML usage:**

```yaml
algorithms:
  - label: "ETD-B"
    coupling: "balanced"
    cost_normalize: "mean"     # optional, defaults to "median"
    epsilon: 0.1
```

### Future Cost Functions

The cost function interface is designed for extension. Planned additions:

- **DV-augmented cost:** $C_{ij} = C_{ij}^{\text{geo}} + \lambda g_j$,
  adding the target-side Sinkhorn dual potential. Implements the
  $+\log\rho(y)$ repulsion term from the DV-optimal cost decomposition.
  (See §DV Feedback for implementation notes.)
- **Energy-augmented cost:** $C_{ij} = \|x_i - y_j\|^2/2 - \lambda \log\pi(y_j)$,
  putting target log-likelihood into the transport cost directly.
  UOT with this cost doesn't need separate target weights.
- **Multi-bandwidth kernel cost:** $C_{ij} = \sum_k w_k \|x_i - y_j\|^2 / h_k^2$
  with geometrically spaced bandwidths, inspired by the SVGD literature.
- **IMQ (Inverse Multiquadric):** $C_{ij} = (c^2 + \|x_i - y_j\|^2)^{-1/2}$,
  heavy-tailed kernel from the S-SVGD literature.

---

## 4. Coupling

Compute the entropic optimal coupling between particles and proposals.
The coupling solver takes the cost matrix and marginals and returns the
conditional coupling (row-normalized log-probabilities) plus dual potentials.

### Gibbs / Semi-Relaxed ($\tau = 0$)

Closed-form, no Sinkhorn iterations:
$$\gamma_{ij} = \frac{b_j \exp(-C_{ij}/\varepsilon)}{\sum_k b_k \exp(-C_{ik}/\varepsilon)}$$

Equivalently: `log_gamma[i] = softmax(-C[i] / eps + log_b)`.

Only the source marginal is enforced (hard constraint: each particle's
coupling row sums to 1). The target marginal enters only through the
reference measure in the KL regularization term. This is proper
semi-relaxed entropic OT.

### Balanced ($\tau \to \infty$)

Both marginals enforced as hard constraints. Full Sinkhorn iteration in
log domain:

```python
log_K = -C / eps
f = zeros(N)
g = zeros(P)
for _ in range(max_iter):
    f = eps * (log_a - logsumexp(log_K + g / eps, axis=1))
    g = eps * (log_b - logsumexp(log_K.T + f / eps, axis=1))
```

Warm-start dual potentials $(f, g)$ across ETD iterations. This typically
reduces inner iterations from ~50 to ~5–10 after the first few steps.

The balanced marginal constraint $\Gamma^\top \mathbf{1} = b$ forces
particles to *compete* for proposal mass. This is the implicit
$+\log\rho$ repulsion identified in the DV analysis: once proposal $j$
has used up its allocation $b_j$, other particles cannot pile onto it.
Different particles are forced to couple to different proposals, maintaining
ensemble diversity.

### Unbalanced ($0 < \tau < \infty$)

Soft target marginal via KL penalty. Strength controlled by $\rho = \tau/\varepsilon$.

```python
log_K = -C / eps
lam = rho / (1 + rho)   # damping exponent
f = zeros(N)
g = zeros(P)
for _ in range(max_iter):
    f = -eps * logsumexp(log_K + g / eps, axis=1)    # exact source
    g_unreg = logsumexp(log_K.T + f / eps, axis=1)
    g = eps * lam * (log_b - g_unreg)                 # soft target
```

Interpolates: $\rho \to 0$ recovers Gibbs/SR, $\rho \to \infty$ approaches
balanced. Default: $\rho = 1.0$.

**DV analysis interpretation:** The UOT marginal penalty parameter $\rho$
controls how much of the $+\log\rho$ repulsion is handled implicitly (via
soft marginals) vs. would need to be added explicitly (via cost
augmentation). At $\rho = \Theta(\varepsilon)$, partial implicit repulsion
via soft marginals combined with explicit $-\log\pi$ in the target weights
provides a good balance.

---

## 5. Update Rules

### Categorical (default)

Sample one proposal per particle from the coupling:
$$j^* \sim \text{Categorical}(\gamma_{i1}, \ldots, \gamma_{iP}), \quad x_i^{\text{new}} = y_{j^*}$$

Use **systematic resampling** (stratified single uniform draw) rather than
multinomial for lower variance:

```python
u = (random.uniform(key) + arange(N)) / N
cumsum_gamma = cumsum(exp(log_gamma), axis=1)
indices = searchsorted(cumsum_gamma, u[:, None]).squeeze()
```

**Why categorical is primary:** Clean stationary distribution theory
(Theorem 6.5 in the draft). Robust to flat couplings (if coupling is
uninformative, degrades gracefully to importance resampling rather than
collapsing). Does not average across modes. Enables exact dual potential
propagation: after resampling, particle $i$ at $y_{j(i)}$ inherits the
target-side potential $g_{j(i)}$ for free.

### Barycentric

Deterministic weighted mean with step size $\eta$:
$$x_i^{\text{new}} = (1-\eta) x_i + \eta \sum_j \gamma_{ij} y_j$$

Advantages: lower variance, differentiable through Sinkhorn (for future
learned-cost extensions), provides covariance estimate. Disadvantages:
averages across modes when coupling is multimodal → collapses variance
(Remark 5.7). Requires the $+\log\rho$ dual potential feedback to prevent
collapse (this was the missing ingredient in early experiments).

### Step-Size Damping (optional, both updates)

Interpolate between old and new positions after the update:
$$x^{(t+1)} = x^{(t)} + \eta \cdot (x_{\text{new}} - x^{(t)})$$

For categorical: stabilizes oscillation on stiff targets.
For barycentric: controls the fraction of transport executed per step.
Default $\eta = 1.0$ (full step).

---

## 6. MCMC Mutation (ETD-SMC)

ETD converges rapidly in the global phase — particles find the right basins
within a few iterations — but local refinement is slow. The coupling flattens
as particles approach the target, and categorical resampling introduces noise
when positions are already approximately correct. The variance ratio on BLR
(~0.38 vs MALA's ~0.90) confirms the ensemble is underdispersed.

The fix completes the SMC structure: **reweight → resample → mutate**. After
transport resampling, apply a $\pi$-invariant MCMC kernel independently to
each particle. The mutation targets $\pi$ directly — no IS correction,
density tracking, or trajectory storage is required. The MH acceptance
ratio ensures exact $\pi$-invariance regardless of step size, so the mutation
can only improve or maintain approximation quality.

### Why mutation doesn't need IS correction

After categorical resampling, particle $i$ sits at position $y_{j(i)}$. This
is an approximate sample from the transported distribution $\rho_{k+1}$.
Applying a $\pi$-invariant kernel $K$ (e.g., MALA with MH correction) yields
$x_i' \sim K(\cdot | y_{j(i)})$. If $\rho_{k+1} = \pi$, then $K$ preserves
$\pi$ exactly. If $\rho_{k+1} \neq \pi$, then $K$ moves $\rho_{k+1}$ closer
to $\pi$ by ergodicity. The mutation kernel only needs access to $\pi$ (via
scores or density ratios), not the transported distribution.

### Pseudocode

```
After step 6 (categorical resample), for each particle i independently:

  Compute log π(x_i) and (for MALA) ∇log π(x_i)

  For t = 1..K:
    # Preconditioned MALA proposal
    μ_fwd = x_i + (h/2) Σ ∇log π(x_i)
    x_prop = μ_fwd + √h L ξ,    ξ ~ N(0, I)

    # Evaluate proposal
    log π(x_prop), ∇log π(x_prop)
    μ_rev = x_prop + (h/2) Σ ∇log π(x_prop)

    # MH acceptance
    log α = log π(x_prop) - log π(x_i)
            + log N(x_i; μ_rev, h Σ) - log N(x_prop; μ_fwd, h Σ)
    Accept x_i ← x_prop with prob min(1, exp(log α))
```

where $L$ is the ensemble Cholesky factor ($\Sigma = LL^\top$), $h$ is the
mutation step size, and $K$ is the number of MCMC sub-steps. Score clipping
is applied (inherits `score_clip` from the parent config by default).

### MALA kernel (Cholesky mode)

The preconditioned MALA proposal for particle $i$:

$$x^{\text{prop}} = x_i + \frac{h}{2}\, \Sigma\, \nabla\log\pi(x_i) + \sqrt{h}\, L\, \xi, \quad \xi \sim \mathcal{N}(0, I)$$

The forward/reverse log-densities use triangular solve for numerical stability:
$\log q(y|x) = -\frac{1}{2h}\|L^{-1}(y - \mu)\|^2 + \text{const}$, computed
via `jax.scipy.linalg.solve_triangular`.

When `use_cholesky: false`, the kernel falls back to isotropic mode
($\Sigma = I$, $L = I$), simplifying to standard MALA.

### Random-Walk MH (score-free)

For targets where scores are unavailable or expensive:

$$x^{\text{prop}} = x_i + \sqrt{h}\, L\, \xi$$

The symmetric proposal gives a MH ratio that depends only on density ratios:
$\log\alpha = \log\pi(x^{\text{prop}}) - \log\pi(x_i)$. Zero score evaluations.

### Two-phase convergence

ETD-SMC exhibits two distinct phases:

1. **Transport-dominated:** The coupling is informative; particles make large
   coordinated moves. Energy distance drops rapidly. Mutation contributes
   marginally (particles are far from $\pi$).

2. **Mutation-dominated:** Particles are in the right basins but underdispersed.
   The coupling softens. MALA steps with MH correction explore local basins
   while respecting $\pi$ exactly. Transport maintains global coordination
   (preventing mode collapse) but contributes less to distributional improvement.

### Ordering constraints

- **Mutation after resampling:** The transport step provides globally
  coordinated placement; mutation refines locally. Reversing would apply MCMC
  to stale positions and discard refinement via resampling.
- **Cholesky after mutation:** Update $\Sigma$ from post-mutation positions
  (closer to $\pi$), giving a better covariance estimate for the next step.

### Mutation hyperparameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `mutation.kernel` | `"none"` | `"none"` / `"mala"` / `"rwm"` |
| `mutation.n_steps` | 5 | MCMC sub-steps per ETD iteration |
| `mutation.step_size` | 0.01 | MALA/RWM step size $h$ |
| `mutation.use_cholesky` | true | Use ensemble Cholesky $\Sigma$ for proposal |
| `mutation.score_clip` | None | None → inherit from parent config |

---

## DV Feedback (Dual Potential Augmentation)

### Theory

The Donsker-Varadhan dual of the JKO free energy reveals that the optimal
transport cost has three terms:

$$c^*(x,y) = \frac{\|x-y\|^2}{2\tau} + \underbrace{\log\rho^*(y)}_{\text{repulsion}} - \underbrace{\log\pi(y)}_{\text{attraction}}$$

Standard ETD has the geometric cost (term 1) and attraction via target
weights (term 3). The repulsion term ($+\log\rho$) is handled implicitly
by balanced marginal constraints but is missing from the cost.

### Implementation

The target-side Sinkhorn dual potential $g_j$ is a proxy for $\log\rho(y_j)$.
After entropic OT, the first-marginal constraint gives
$f(x) = \varepsilon\log\rho(x) - \psi(x)$, where $\psi$ is a softmin
correction that is approximately constant at large $\varepsilon$ or high $d$.

**Conceptually**, this is a cost augmentation: $C_{ij} \leftarrow C_{ij} + \lambda g_j$.
**Implementationally**, it is equivalent and cleaner to add $-\lambda \tilde{g}_j$ to
the log target weights `log_b` before the coupling solve. This avoids
polluting the median heuristic with non-geometric terms and keeps cost
functions as pure geometry. The Sinkhorn solver sees identical log-domain
arithmetic either way.

### c-Transform (Double-Counting Fix)

The raw Sinkhorn dual $g_j$ **cannot** be fed back directly. The g-update
bakes in `log_b_j` (the IS-corrected target weights), so feeding raw $g$
into `log_b` double-counts $\log\pi - \log q$. The fix is a
coupling-dependent c-transform:

- **Balanced**: $\tilde{g}_j = g_j - \varepsilon \cdot \log b_j$
- **Unbalanced**: $\tilde{g}_j = g_j - \varepsilon \lambda \cdot \log b_j$, where $\lambda = \rho/(1+\rho)$
- **Gibbs**: no iterative solver, $\tilde{g} = 0$ (no meaningful signal)

The cleaned potential $\tilde{g}_j$ reflects pure geometry-weighted repulsion.

### Per-Particle Potential and Interpolation

After the coupling solve, each particle receives a scalar `dv_potential`
that is carried to the next iteration:

- **Categorical update**: $\text{dv}_i = \tilde{g}_{j(i)}$ (index into selected proposal)
- **Barycentric update**: $\text{dv}_i = \sum_j w_{ij} \tilde{g}_j$ (weighted mean)

When `step_size` $\eta < 1$, the particle moves only partway to its target.
The carried potential interpolates source and target duals:

$$\text{dv}_i = (1 - \eta) \, f_i + \eta \, \tilde{g}_{j(i)}$$

At $\eta = 1$ (full step), this reduces to $\tilde{g}_{j(i)}$.

### Warmup

DV feedback is most effective after the coupling structure stabilizes
(first ~50 iterations). Use a linear warmup schedule:

```yaml
dv_feedback: true
dv_weight:
  schedule: linear_warmup
  value: 1.0
  warmup: 50
```

Config: `dv_feedback: true`, `dv_weight: 1.0`. Off by default until
validated experimentally.

---

## Preconditioner

ETD supports two preconditioner types via a nested `PreconditionerConfig`:

| Type | State | Shape | Captures |
|------|-------|-------|----------|
| `"rmsprop"` | `precond_accum` | $(d,)$ | Per-axis curvature (diagonal) |
| `"cholesky"` | `cholesky_factor` | $(d, d)$ | Full covariance (correlations) |
| `"none"` | — | — | No preconditioning |

Both state fields always exist for pytree stability; only the relevant one
is updated each step. The config type is static (frozen dataclass), so
Python `if` resolves at JIT trace time with zero runtime overhead.

Each type has two independent application axes:

- **`proposals: true`** — Scale proposal drift and noise.
- **`cost: true`** — Whiten cost matrix distances.

### RMSProp (Diagonal)

Accumulator update:

$$G_t = \beta \, G_{t-1} + (1 - \beta) \, \text{diag}\!\left(\frac{1}{N}\sum_i s_i \odot s_i\right)$$

where $s_i = \nabla \log \pi(x_i)$ and $\beta = 0.9$. Initialize $G_0 = \mathbf{1}_d$.

The inverse square root $P = 1/\sqrt{G_t + \delta}$ is a $(d,)$ vector.

**Proposals:** Mean $x_i + \alpha \, P \odot s_i$, variance $\text{diag}(\sigma^2 P^2)$.
Stretches proposals along low-curvature directions.

**Cost whitening:** Distances in whitened coordinates $\tilde{x} = P^{-1} x$,
e.g. $C_{ij} = \|\tilde{x}_i - \tilde{y}_j\|^2 / 2$.

### Cholesky (Full Covariance)

Computes a full-covariance Cholesky factor $L$ from the particle ensemble
each step, enabling proposals and costs that capture inter-dimensional
correlations.

**Computation (once per step, before proposals):**

1. Sample covariance: $\hat{\Sigma} = \text{Cov}(\text{data})$ — shape $(d, d)$
2. Ledoit-Wolf shrinkage: $\Sigma = (1-s)\hat{\Sigma} + s\,\text{diag}(\text{diag}(\hat{\Sigma}))$
3. Jitter: $\Sigma_{\text{reg}} = \Sigma + \delta I$
4. Optional EMA: $\Sigma_{\text{reg}} = \beta_{\text{ema}} \Sigma_{\text{prev}} + (1-\beta_{\text{ema}}) \Sigma_{\text{reg}}$
5. Cholesky: $L = \text{chol}(\Sigma_{\text{reg}})$

The `source` field controls what data the covariance is computed from
(**Cholesky only** — RMSProp always uses scores):

| `source` | Data | Captures |
|----------|------|----------|
| `"scores"` (default) | $\nabla \log \pi(x_i)$ | Target curvature |
| `"positions"` | $x_i$ | Ensemble spread |

Setting `source` to anything other than `"scores"` with `type="rmsprop"`
emits a warning — RMSProp's accumulator $G_t = \mathbb{E}[s \odot s]$ is
inherently a score statistic and computing it from positions is not meaningful.

When `source="scores"`, the `use_unclipped_scores` flag controls whether
raw or clipped scores are used (raw scores better capture the true Hessian
structure but may be noisy on stiff targets). This flag is also Cholesky-only.

**EMA note:** EMA is applied on the *covariance matrix* before Cholesky
factoring, not on $L$ directly. Direct EMA on $L$ does not preserve
positive-definiteness.

**Proposals:**
$$y_{ij} = x_i + \alpha \, (LL^\top s_i) + \sigma \, L \xi_{ij}, \quad \xi_{ij} \sim \mathcal{N}(0, I)$$

This gives $y_{ij} \sim \mathcal{N}(x_i + \alpha \Sigma s_i,\; \sigma^2 \Sigma)$.
The IS correction uses the corresponding full-covariance Gaussian density
with log-determinant from $L$.

**Cost whitening:** $\tilde{x} = L^{-1} x$ via triangular solve, then
standard distance in whitened space.

**Cost:** $O(Nd^2)$ for covariance, $O(d^3)$ for Cholesky. Negligible for $d < 100$.

### Config

```yaml
preconditioner:
  type: "cholesky"        # "none" | "rmsprop" | "cholesky"
  proposals: true          # apply to proposal drift + noise
  cost: true               # apply to cost whitening
  source: "scores"         # "scores" | "positions"
  use_unclipped_scores: false
  # RMSProp params
  beta: 0.9
  delta: 1.0e-8
  # Cholesky params
  shrinkage: 0.1           # Ledoit-Wolf shrinkage toward diagonal
  jitter: 1.0e-6           # diagonal jitter for PD guarantee
  ema_beta: 0.0            # 0 = fresh each step; >0 = EMA smoothing
```

### Legacy Flat Fields

The flat fields `precondition`, `whiten`, `precond_beta`, `precond_delta`
are still supported for backward compatibility and map to
`PreconditionerConfig(type="rmsprop", ...)`. The nested config takes
precedence when both are present.

**Mahalanobis alias:** `cost: "mahalanobis"` is a deprecated alias for
`cost: "euclidean"` with `whiten: true`. It emits a `FutureWarning`.

### Adaptive Score Clipping

When the preconditioner is enabled, score clipping is preconditioner-aware:
clip the preconditioned score $P \odot s$ to a maximum norm of
`score_clip * sigma`, then invert: $s_{\text{clipped}} = (P \odot s)_{\text{clipped}} / P$.

---

## SDD Extension (Sinkhorn Divergence Descent)

SDD replaces the entropic OT objective with the Sinkhorn divergence:

$$S_\varepsilon(\mu, \pi) = \text{OT}_\varepsilon(\mu, \pi) - \tfrac{1}{2}\text{OT}_\varepsilon(\mu, \mu) - \tfrac{1}{2}\text{OT}_\varepsilon(\pi, \pi)$$

This is a proper divergence ($S_\varepsilon(\pi, \pi) = 0$), eliminating
the entropic bias at any $\varepsilon$.

### SDD-RB Update

SDD-RB (Rao-Blackwellized) replaces the cross-coupling with a categorical
sample while keeping the self-coupling barycentric:

$$x_i \leftarrow x_i + \eta\bigl(y_{j_i^*}^{\text{cross}} - \bar{x}_i^{\text{self}}\bigr)$$

where $\bar{x}_i^{\text{self}} = \sum_j \gamma_{ij}^{\text{self}} x_j$ is
the barycentric mean under the self-coupling.

SDD is a **step-level wrapper**, not a coupling variant. It composes with
any coupling/cost combination. See `extensions/sdd.py`.

---

## Hyperparameters

### ETD

| Parameter | Symbol | Default | Effect |
|-----------|--------|---------|--------|
| `epsilon` | $\varepsilon$ | 0.1 | Entropic regularization. Larger = more exploration. |
| `alpha` | $\alpha$ | 0.05 | Score step size for proposals. |
| `fdr` | — | true | Tie $\sigma = \sqrt{2\alpha}$. |
| `sigma` | $\sigma$ | $\sqrt{2\alpha}$ | Proposal noise (when `fdr: false`). |
| `n_proposals` | $M$ | 25 | Proposals per particle. |
| `score_clip` | $c$ | 5.0 | Score clipping threshold. |
| `rho` | $\rho$ | 1.0 | UB marginal strength ($\tau/\varepsilon$). |
| `step_size` | $\eta$ | 1.0 | Damping in $(0, 1]$. |
| `sinkhorn_max_iter` | — | 50 | Max inner Sinkhorn iterations. |
| `precondition` | — | false | Enable diagonal preconditioner. |
| `dv_feedback` | — | false | Enable dual potential augmentation. |

### Rules of Thumb

- Start with $\varepsilon = 0.1$. If particles cluster, increase. If they
  diffuse, decrease.
- Always clip scores.
- `step_size < 1.0` (e.g., 0.3–0.5) helps when particles oscillate or
  diverge on stiff targets.
- For BLR experiments, $\varepsilon \in [0.05, 0.5]$ typically works.
- `n_proposals = 25` is a good default. Increase to 50–100 for high $d$.

### Baselines

**SVGD:** RBF kernel with median heuristic bandwidth, Adam optimizer.
Tune learning rate $\in \{10^{-4}, 3 \times 10^{-4}, \ldots, 3 \times 10^{-1}\}$.

**ULA:** Tune step size over the same grid as SVGD.

**MPPI:** Use same $\sigma$ as best ETD variant. Tune temperature
$\beta \in \{0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0\}$.
