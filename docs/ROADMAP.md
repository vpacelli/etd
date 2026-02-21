# Roadmap

Implementation phases for the ETD redesign. Each phase has a concrete
test/experiment gate. Do not proceed to the next phase until the gate passes.

---

## Phase 0: Primitives

**Goal:** Implement the building blocks that everything else calls.

### Deliverables

- [ ] `types.py`: `Target` protocol, `ETDState` NamedTuple, `ETDConfig` dataclass
- [ ] `costs/euclidean.py`: Squared Euclidean cost matrix, `(N,d) × (P,d) → (N,P)`
- [ ] `costs/normalize.py`: Median heuristic normalization
- [ ] `coupling/sinkhorn.py`: Log-domain balanced Sinkhorn with warm-start
- [ ] `coupling/gibbs.py`: Closed-form Gibbs coupling (semi-relaxed)
- [ ] `proposals/langevin.py`: Score-guided + score-free proposals, score clipping
- [ ] `update/categorical.py`: Systematic resampling from log-coupling
- [ ] `weights.py`: IS-corrected target weights (always on)
- [ ] `weights.py`: IS-corrected target weights with proposal density evaluation and floor

### Gate

```
test_sinkhorn.py:
  - Balanced Sinkhorn on random (50, 200) cost matrix.
    Row sums ≈ a, column sums ≈ b to tolerance 1e-4.
  - Warm-started solve converges in ≤ 10 iterations
    (cold-start baseline: ~40 iterations).
  - As ε → ∞, coupling → product measure (max row entropy → log P).
  - As ε → 0, coupling concentrates (min row entropy → 0).

test_primitives.py:
  - Euclidean cost matches scipy.spatial.distance.cdist.
  - Median normalization: output median ≈ 1.0.
  - Score clipping: output norms ≤ max_norm.
  - Systematic resampling: empirical distribution matches
    coupling rows over 10k samples (KS test p > 0.01).
```

---

## Phase 1: Minimal ETD

**Goal:** Wire the primitives into a working ETD loop. Validate on a
trivial target.

### Deliverables

- [ ] `step.py`: The ~20-line step function (propose → cost → couple → update)
- [ ] `targets/gaussian.py`: Single isotropic Gaussian
- [ ] `targets/gmm.py`: 2D mixture of Gaussians (grid arrangement)
- [ ] `diagnostics/metrics.py`: Energy distance, mode coverage, mean error
- [ ] `figures/style.py`: Crimson palette, contour plot helpers

### Gate

```
test_etd_gaussian.py:
  - ETD-B (balanced, categorical, Euclidean cost) on N(0, I) in d=2.
    N=100 particles, 200 iterations, ε=0.1, α=0.05.
    Mean error < 0.1, variance ratio in [0.8, 1.2].

Visual sanity check:
  - ETD-B on 2D 4-mode GMM (separation=6).
    Scatter plot at iterations {1, 50, 100, 200, 500}.
    All 4 modes covered by iteration 100.
```

---

## Phase 2: Baselines + Runner

**Goal:** Implement comparison methods and the experiment runner
so we can produce reproducible multi-algorithm benchmarks.

### Deliverables

- [ ] `baselines/svgd.py`: SVGD with RBF kernel, median heuristic, Adam
- [ ] `baselines/ula.py`: Unadjusted Langevin (N independent chains)
- [ ] `baselines/mppi.py`: Importance-weighted averaging
- [ ] `experiments/run.py`: YAML → config → run → save metrics + particles
- [ ] Sweep expansion (list params → Cartesian product)
- [ ] Rich terminal output (progress bars, summary tables)
- [ ] `experiments/run.py`: main runner with sweep expansion, Rich output, `run_single()` factored out
- [ ] `experiments/tune.py`: grid-search tuner reusing `run_single()`
- [ ] `experiments/datasets.py`: DuckDB pipeline for UCI datasets (German Credit, Australian)
- [ ] `data/etd.duckdb` created and populated.

### Gate

```
First benchmark:
  - Run gmm-2d-4.yaml with ETD-B, SVGD, ULA, MPPI.
  - 5 seeds, 500 iterations, N=100.
  - ETD-B energy distance < SVGD energy distance (averaged over seeds).
  - ETD-B mode coverage = 4/4, SVGD mode coverage ≤ 3/4.
  - Results saved to results/ with frozen config.
```

---

## Phase 3: Cost Variants (the paper's contribution)

**Goal:** Implement the alternative cost functions and coupling variants
that are the paper's main experimental axis. Test dimensional scaling.

### Deliverables

- [ ] `costs/mahalanobis.py`: Diagonal Mahalanobis from preconditioner
- [ ] `costs/linf.py`: $L_\infty$ cost
- [ ] `coupling/unbalanced.py`: Log-domain unbalanced Sinkhorn
- [ ] `proposals/langevin.py`: Add diagonal preconditioning path
- [ ] DV feedback in `step.py` ($g$-potential → target weight augmentation)

### Gate

```
Dimensional scaling experiment:
  - 10D GMM, 5 modes, N=200, M=50, 500 iterations.
  - Compare: Euclidean, Mahalanobis, L∞, DV-augmented.
  - 5 seeds each.
  - At least one cost variant beats Euclidean on energy distance
    by > 20% (averaged over seeds).
  - Coupling ESS diagnostic: Mahalanobis ESS > Euclidean ESS
    at iteration 100.
```

---

## Phase 4: Additional Targets + Extensions

**Goal:** Add the real-data and non-log-convex targets needed for the
paper. Implement SDD for comparison.

### Deliverables

- [ ] `targets/banana.py`: Banana-shaped posterior
- [ ] `targets/funnel.py`: Neal's funnel
- [ ] `targets/logistic.py`: Bayesian logistic regression (German Credit, Australian)
- [ ] `update/barycentric.py`: Barycentric projection with step size
- [ ] `extensions/sdd.py`: SDD-RB (self-coupling subtraction)
- [ ] NUTS reference sampler integration (numpyro)
- [ ] `experiments/nuts.py`: NUTS reference sampler with convergence gate (R-hat, ESS, divergences)
[ ] `experiments/datasets.py`: Statcast pipeline (if pitching target is in scope)
[ ] `results/reference/{target}.npz` cached for all paper targets

### Gate

```
BLR benchmark:
  - German Credit (d≈25), N=200, 1000 iterations.
  - ETD-B vs SVGD vs ULA.
  - Mean RMSE (vs NUTS): ETD-B competitive with or better than SVGD.
  - Variance ratio: ETD-B closer to 1.0 than SVGD.

Funnel test:
  - Neal's funnel d=10.
  - ETD-B vs SVGD, 500 iterations.
  - Visual: ETD particles span the funnel neck; SVGD collapses.
```

---

## Phase 5: Progressive Scheduling

**Goal:** Replace fixed hyperparameters with ε-annealing and epoch
structure (ProgOT-inspired).

### Deliverables

- [ ] ε-annealing: geometric decay from ε₀ to ε_K
- [ ] Epoch structure: fixed proposals with periodic refresh every L steps
- [ ] ε₀ calibration from initial cost geometry
- [ ] Warm-starting dual potentials across iterations (already in place
      from Phase 0, but test with epochs)

### Gate

```
Annealing comparison:
  - 20D GMM, N=200, 1000 iterations.
  - Fixed ε=0.1 vs geometric anneal (ε₀=1.0 → ε_K=0.01).
  - Annealed version achieves lower final energy distance.
  - Annealed version's coupling ESS decreases smoothly
    (not stuck at ceiling or floor).
```

---

## Phase 6: Paper Figures

**Goal:** Generate publication-quality figures for the NeurIPS submission.

### Deliverables

- [ ] **Figure 1** (hero): Dimensional scaling — energy distance vs d,
      Euclidean vs Mahalanobis vs L∞ vs DV. Two panels: fixed ε and annealed.
- [ ] **Figure 2**: Convergence trajectories on 2D GMM — particles at
      {1, 50, 200, 500} iterations, ETD-B vs SVGD.
- [ ] **Figure 3**: Coupling diagnostics — ESS and dual potential variation
      vs iteration for different cost functions.
- [ ] **Figure 4**: BLR comparison table — mean RMSE and variance ratio
      for ETD variants, SVGD, ULA on German Credit + Australian.
- [ ] **Figure 5** (optional): Funnel — particle scatter in (θ₁, log τ)
      plane, ETD vs SVGD.
- [ ] **Table 1**: Algorithm summary — cost, coupling, update, score
      requirement, key property.

### Gate

```
All figures render cleanly at NeurIPS column width (3.25 in).
Font sizes readable. Colors match the crimson palette.
No clipped labels or overlapping text.
```

---

## What If Things Go Wrong

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| All particles collapse (spread → 0) | Step size too large, or missing repulsion | Decrease η to 0.3; enable DV feedback; try balanced coupling |
| No convergence (energy distance plateaus) | ε too large, coupling uninformative | Check coupling entropy — if near log(NP), decrease ε |
| Mode coverage stuck below target | Proposals don't reach all modes | Increase σ or M; use score-guided proposals |
| Sinkhorn NaN/Inf | Cost matrix has extreme entries | Verify median normalization is active; use log-domain |
| SVGD outperforms ETD on everything | Bug in IS weights or coupling | Check: are IS weights computed with correct proposal density? |
| Mahalanobis no better than Euclidean | Preconditioner not converged | Let preconditioner warm up 50 iterations before comparing |
