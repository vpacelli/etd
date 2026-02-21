# Development

## Priority Order

1. **Correctness** — wrong results are worthless
2. **Reproducibility** — same seed → same output
3. **Readability** — future-you must understand it
4. **Speed** — optimize last

---

## Code Style

### General

- Line length: 100 characters.
- Imports: stdlib → third-party → local, alphabetized within groups.
- Type hints on function signatures.
- Docstrings: document shapes, assumptions, and *why*, not just *what*.

### Naming

| Type | Convention | Example |
|------|------------|---------|
| Functions | `verb_noun` | `compute_coupling`, `sample_proposals` |
| Config classes | `*Config` | `ETDConfig` |
| State classes | `*State` | `ETDState` |
| Constants | `UPPER_SNAKE` | `DEFAULT_EPSILON` |

Use semantic variable names. Not `alpha` — use `score_step` or leave as
`config.alpha` with the config field being self-documenting. Exception:
universally understood notation like `N`, `d`, `eps`.

### Shape Documentation

Every function that takes arrays must document their shapes:

```python
def compute_coupling(
    positions: jnp.ndarray,    # (N, d)
    proposals: jnp.ndarray,    # (P, d)  where P = N*M
    log_pi: jnp.ndarray,       # (P,)
    eps: float,
) -> jnp.ndarray:              # (N, P) rows sum to 1
    """Compute semi-relaxed coupling weights."""
```

---

## JAX Patterns

### RNG Management

```python
# CORRECT: explicit key splitting
def step(key, state, target, config):
    key_propose, key_update = jax.random.split(key)
    proposals = sample_proposals(key_propose, ...)
    new_positions = update(key_update, ...)
    return state._replace(positions=new_positions, step=state.step + 1)

# WRONG: reusing keys
x = jax.random.normal(key, (10,))
y = jax.random.normal(key, (10,))  # same as x!
```

### JIT Strategy

```python
# Development: no JIT, print debugging works
def step(key, state, target, config):
    ...
    print(f"mean log_prob = {log_probs.mean()}")  # works
    ...

# Production: JIT at outer level
step_jit = jax.jit(step, static_argnums=(2, 3))
```

Apply JIT as far up the call stack as possible. Debug mode:
`ETD_DEBUG=1 python -m experiments.run config.yaml` disables JIT.

```python
import os, jax
DEBUG = os.environ.get("ETD_DEBUG", "0") == "1"

def maybe_jit(fn, **kwargs):
    return fn if DEBUG else jax.jit(fn, **kwargs)
```

### Static-Config Contract

`target` and `config` must be `static_argnums` in JIT because they
contain Python objects. Each distinct `(target, config)` pair triggers
XLA recompilation. Python `if/elif/else` on config fields are resolved
at trace time — this is intentional.

### Init-Time Pytree Contract

The pytree structure (shapes, None vs array) is fixed at `init()`.
**Never use `None` for optional arrays.** Initialize dual potentials as
zeros so the pytree shape is stable:

```python
# In init():
dual_f = jnp.zeros(n_particles)
dual_g = jnp.zeros(n_particles * n_proposals)
precond_accum = jnp.ones(dim)
```

### Flatten-Evaluate-Reshape

When evaluating `log_prob` at all proposals:

```python
# proposals: (N, M, d) — reshape to batch call
proposals_flat = proposals.reshape(-1, d)      # (N*M, d)
log_pi = target.log_prob(proposals_flat)        # (N*M,)
```

### Avoid `vmap` When Broadcasting Suffices

Use broadcasting + `einsum` for uniform batch operations. Reserve `vmap`
for per-element logic that genuinely differs.

```python
# GOOD: broadcasting
C = 0.5 * jnp.sum((positions[:, None, :] - proposals[None, :, :]) ** 2, axis=-1)

# UNNECESSARY: vmap for the same thing
C = jax.vmap(lambda x: 0.5 * jnp.sum((x - proposals) ** 2, axis=-1))(positions)
```

### `lax.scan` for Sinkhorn

Use `lax.scan` (not a Python for-loop) for Sinkhorn iterations to enable
JIT compilation with a fixed iteration count:

```python
def body(carry, _):
    f, g = carry
    f_new = eps * (log_a - logsumexp(log_K + g / eps, axis=1))
    g_new = eps * (log_b - logsumexp((log_K + f_new[:, None] / eps), axis=0))
    return (f_new, g_new), None

(f, g), _ = jax.lax.scan(body, (f_init, g_init), None, length=max_iter)
```

---

## Numerical Hygiene

### Log-Domain Everything

```python
from jax.scipy.special import logsumexp

# GOOD: stable
log_gamma = -C / eps + log_b
gamma = jax.nn.softmax(log_gamma, axis=-1)

# BAD: overflow/underflow
gamma = jnp.exp(-C / eps) * b
gamma = gamma / gamma.sum(axis=-1, keepdims=True)
```

### Median Heuristic

```python
def normalize_cost(C: jnp.ndarray) -> tuple[jnp.ndarray, float]:
    """Normalize cost matrix to unit median. Makes ε interpretable."""
    C_med = jnp.median(C)
    C_med = jnp.maximum(C_med, 1e-8)
    return C / C_med, C_med
```

### Score Clipping

```python
def clip_scores(scores: jnp.ndarray, max_norm: float = 5.0) -> jnp.ndarray:
    """Clip score vectors to maximum norm. Essential for stability."""
    norms = jnp.linalg.norm(scores, axis=-1, keepdims=True)
    scale = jnp.minimum(1.0, max_norm / jnp.maximum(norms, 1e-8))
    return scores * scale
```

### Importance Weight Floor

```python
def compute_log_proposal_density(proposals, positions, alpha, sigma, scores):
    """Evaluate log q_mu(y) = logsumexp_i log N(y; mu_i, sigma²I) - log N.

    Floor at max - 30 to prevent division blowup.
    """
    # ... compute log_q (P,)
    log_q = jnp.maximum(log_q, jnp.max(log_q) - 30.0)
    return log_q
```

### Covariance Stability (for Gaussian update / diagnostics)

```python
def fit_weighted_gaussian(weights, points, reg=1e-4):
    """Fit Gaussian to weighted points. Always regularize."""
    mu = jnp.sum(weights[:, None] * points, axis=0)
    diff = points - mu
    Sigma = jnp.einsum('j,ji,jk->ik', weights, diff, diff)
    Sigma = Sigma + reg * jnp.eye(points.shape[1])
    return mu, Sigma
```

---

## Testing

### What to Test

1. **Sinkhorn marginals:** `exp(log_gamma).sum(axis=1) ≈ a`,
   `exp(log_gamma).sum(axis=0) ≈ b` (for balanced).
2. **Score correctness:** autodiff matches finite differences.
3. **Determinism:** same seed → same output.
4. **Convergence:** ETD-B on isotropic Gaussian, mean error < 0.1 after
   200 steps.
5. **Cost functions:** Euclidean cost matches `cdist`. Mahalanobis with
   identity matrix matches Euclidean.
6. **Coupling limits:** As $\varepsilon \to \infty$, coupling → product
   measure. As $\varepsilon \to 0$, coupling → nearest-neighbor.

### When to Test

- After implementing each phase (see ROADMAP.md).
- After modifying any primitive.
- NOT for exploratory scripts.

---

## Terminal Output

Use **`rich`** for all terminal output: progress bars, tables, logging.

```python
from rich.console import Console
from rich.table import Table
from rich.progress import track, Progress
from rich.traceback import install
from rich import box

install(show_locals=True)    # readable JAX errors
console = Console()
```

### Pattern

```
[14:30:22] Loading config from configs/mog_2d_4.yaml
[14:30:22] Target: gmm, d=2, 4 modes
[14:30:22] Running 5 algorithms × 5 seeds
           ε = 0.1    N = 100    M = 25

  ETD-B      ━━━━━━━━━━━━━━━━━━━━ 100% 0:00:12
  SVGD       ━━━━━━━━━━━━━━━━━━━━ 100% 0:00:18

╭──────────────── Results (seed avg ± std) ─────────────────╮
│ Algorithm │ Energy Dist  │ Modes │ Wall Clock              │
│ ETD-B     │ 0.023 ± 0.01 │   4/4 │  1.2 ± 0.1s            │
│ SVGD      │ 0.187 ± 0.04 │   2/4 │  1.8 ± 0.2s            │
╰───────────────────────────────────────────────────────────╯

✓ Results saved to results/mog-2-4/2026-02-20/
```
