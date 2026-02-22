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
**Implementationally**, it is equivalent and cleaner to add $-\lambda g_j$ to
the log target weights `log_b` before the coupling solve. This avoids
polluting the median heuristic with non-geometric terms and keeps cost
functions as pure geometry. The Sinkhorn solver sees identical log-domain
arithmetic either way.

With categorical resampling, dual potentials propagate for free: particle
$i$ resampled to position $y_{j(i)}$ inherits $g_{j(i)}$ from the current
solve. No interpolation, no c-transform, no staleness.

Config: `dv_feedback: true`, `dv_weight: 1.0`. Off by default until
validated experimentally.

---

## Preconditioner

Diagonal RMSProp-style accumulator built from score evaluations.

### Accumulator Update

$$G_t = \beta \, G_{t-1} + (1 - \beta) \, \text{diag}\!\left(\frac{1}{N}\sum_i s_i \odot s_i\right)$$

where $s_i = \nabla \log \pi(x_i)$ and $\beta = 0.9$. Initialize $G_0 = \mathbf{1}_d$.

The inverse square root $P = 1/\sqrt{G_t + \delta}$ is a $(d,)$ vector.

### Usage

The preconditioner affects two things:

1. **Proposals:** Mean $x_i + \alpha \, P \odot s_i$, variance
   $\text{diag}(2\alpha\rho \, P^2)$. Stretches proposals along
   low-curvature directions.
2. **Mahalanobis cost** (if enabled): $C_{ij} = (x_i - y_j)^\top \text{diag}(P)^{-2} (x_i - y_j) / 2$.
   Makes the coupling sensitive to displacements along high-curvature
   directions.

The preconditioner lives in `ETDState.precond_accum` as a $(d,)$ array.
Updated during proposal generation. No separate `Preconditioner` class.

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
