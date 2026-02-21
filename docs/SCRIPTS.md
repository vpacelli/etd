# Scripts

Entry-point scripts for running experiments, tuning, generating reference
samples, and managing datasets. All live in `experiments/`.

```
experiments/
├── run.py              # Main: YAML → run → save
├── tune.py             # Sweep-based hyperparameter tuning
├── nuts.py             # Generate NUTS reference samples
├── datasets.py         # Download, preprocess, store in DuckDB
└── configs/
    ├── gmm_2d_4.yaml
    ├── logistic_german.yaml
    └── sweeps/
        └── eps_sensitivity.yaml
```

---

## run.py — Experiment Runner

The main entry point. Reads a YAML config, expands sweeps, runs all
algorithm × seed combinations, and saves results.

### Usage

```bash
# Standard run
python -m experiments.run configs/gmm_2d_4.yaml

# Debug mode (no JIT, print statements work)
ETD_DEBUG=1 python -m experiments.run configs/gmm_2d_4.yaml

# Override output directory
python -m experiments.run configs/gmm_2d_4.yaml --output results/scratch/
```

### What It Does

1. Load YAML config.
2. Expand sweep parameters (list values → Cartesian product).
3. For each seed × algorithm combination:
   a. Initialize target and particles.
   b. Resolve config strings to functions (`cost: "euclidean"` → `euclidean_cost`).
   c. Run the algorithm loop with Rich progress bar.
   d. Record metrics at checkpoint iterations.
   e. Save particle snapshots.
4. Print summary table (Rich).
5. Save to `results/{name}/{timestamp}/`:
   - `config.yaml` — frozen copy of input
   - `metrics.json` — nested dict: `{seed → {algo → {checkpoint → {metric → value}}}}`
   - `particles.npz` — arrays: `{seed → {algo → {checkpoint → (N, d)}}}`

### Sweep Expansion

When a YAML parameter value is a list, the runner expands the Cartesian
product into separate runs. Each expanded run gets a label suffix:

```yaml
- label: "ETD-B"
  epsilon: [0.05, 0.1, 0.25]
  alpha: [0.025, 0.05]
```

Expands to 6 runs: `ETD-B_eps=0.05_alpha=0.025`, etc.

Expansion happens in the runner, not the config dataclass. Individual
algorithm configs are always single-valued.

### Infrastructure Shared with tune.py

The runner's core loop (initialize → step → checkpoint → save) is
factored into a reusable function:

```python
def run_single(key, target, config, checkpoints, metrics_list):
    """Run one algorithm on one seed. Returns metrics dict + particles."""
    ...
```

Both `run.py` and `tune.py` call this. Neither duplicates the loop.

---

## tune.py — Hyperparameter Tuning

Grid search over a small number of parameters. Evaluates each
configuration on a single tuning target and reports the best.

### Usage

```bash
# Grid sweep
python -m experiments.tune configs/sweeps/eps_sensitivity.yaml

# Specify tuning metric
python -m experiments.tune configs/sweeps/eps_sensitivity.yaml --metric energy_distance
```

### Tuning Config

A tuning config is a standard experiment config where the algorithm
entry has list-valued parameters (same sweep syntax as `run.py`).
The tuning script:

1. Expands the sweep.
2. Runs each configuration (fewer seeds than a full benchmark — typically 3).
3. Ranks by the specified metric at the final checkpoint.
4. Prints a ranked table and saves the best config.

```yaml
experiment:
  name: "tune-eps"
  seeds: [0, 1, 2]

  target:
    type: "gmm"
    params:
      dim: 2
      n_modes: 4
      separation: 6.0

  shared:
    n_particles: 100
    n_iterations: 200          # shorter than full benchmark

  checkpoints: [200]           # only final
  metrics: ["energy_distance"]

  algorithms:
    - label: "ETD-B"
      cost: "euclidean"
      coupling: "balanced"
      update: "categorical"
      use_score: true
      epsilon: [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
      alpha: 0.05
```

### Output

```
╭──────────────── Tuning: ETD-B on gmm-2d-4 ──────────────╮
│ Rank │ ε      │ Energy Dist (mean ± std)                 │
│    1 │ 0.10   │ 0.023 ± 0.008                            │
│    2 │ 0.25   │ 0.031 ± 0.012                            │
│    3 │ 0.05   │ 0.045 ± 0.015                            │
│  ... │        │                                           │
╰──────────────────────────────────────────────────────────╯
✓ Best config saved to results/tune-eps/best.yaml
```

### Optuna (future)

If grid search proves insufficient for higher-dimensional parameter
spaces, a separate `optuna_tune.py` can wrap the same `run_single()`
infrastructure with Optuna's TPE sampler. This is not needed for the
initial paper experiments (grid over 1–2 parameters suffices) but the
shared `run_single()` function makes it straightforward to add.

---

## nuts.py — NUTS Reference Samples

Generate ground-truth samples via NumPyro's NUTS sampler. These are used
as the reference distribution for metrics like energy distance, mean RMSE,
and variance ratio.

### Usage

```bash
# Generate reference for a specific target
python -m experiments.nuts --target logistic --dataset german_credit --n_samples 10000

# Generate reference for a config's target
python -m experiments.nuts --from-config configs/logistic_german.yaml
```

### What It Does

1. Instantiate the target distribution.
2. Run NUTS with specified warmup and sample count.
3. Run basic convergence diagnostics (R-hat, ESS, divergences).
4. Save samples to `results/reference/{target_name}.npz`.

### Convergence Gate

The script **refuses to save** if convergence is suspect:

- Any parameter with R-hat > 1.01 → warning + flag.
- More than 1% divergent transitions → warning + flag.
- Minimum ESS < 400 → warning + flag.

If any flag is raised, the script prints a warning and requires `--force`
to save. This prevents contaminating benchmarks with unreliable reference
samples.

### Caching

Reference samples are cached by target name + parameter hash. If a
matching file exists, `nuts.py` skips re-running and prints the cached
path. Force regeneration with `--regenerate`.

The `Target` protocol includes a `reference_params_hash()` method that
returns a deterministic hash of the target's parameters. For BLR targets,
this includes the dataset name, prior std, and data hash.

---

## datasets.py — Data Management

Download, preprocess, and store datasets in a DuckDB database. All
experiment code reads from this database rather than scattered files.

### Usage

```bash
# Download and preprocess all datasets
python -m experiments.datasets --all

# Specific dataset
python -m experiments.datasets --dataset german_credit
python -m experiments.datasets --dataset statcast --year 2024

# Check what's available
python -m experiments.datasets --list
```

### Database

All data lives in a single DuckDB file: `data/etd.duckdb`.

```
data/
└── etd.duckdb              # single database file
```

### Tables

#### UCI datasets (for BLR)

```sql
CREATE TABLE german_credit (
    id INTEGER PRIMARY KEY,
    -- features (standardized)
    x1 DOUBLE, x2 DOUBLE, ..., x24 DOUBLE,
    -- binary outcome
    y INTEGER,
    -- metadata
    feature_names JSON
);

CREATE TABLE australian (
    id INTEGER PRIMARY KEY,
    x1 DOUBLE, ..., x14 DOUBLE,
    y INTEGER,
    feature_names JSON
);
```

#### Statcast (for pitching models)

```sql
CREATE TABLE statcast_pitches (
    pitch_id INTEGER PRIMARY KEY,
    pitcher_id INTEGER,
    pitcher_name VARCHAR,
    pitch_type VARCHAR,
    -- features
    release_speed DOUBLE,
    pfx_z DOUBLE,
    pfx_x DOUBLE,
    release_extension DOUBLE,
    release_spin_rate DOUBLE,
    -- outcome
    is_whiff BOOLEAN,
    -- metadata
    game_date DATE,
    season INTEGER
);

-- Derived: per-pitcher aggregates
CREATE VIEW pitcher_ff_summary AS
SELECT
    pitcher_id, pitcher_name,
    COUNT(*) as n_ff,
    AVG(release_speed) as avg_velo,
    AVG(CAST(is_whiff AS DOUBLE)) as whiff_rate
FROM statcast_pitches
WHERE pitch_type = 'FF'
GROUP BY pitcher_id, pitcher_name;
```

### Why DuckDB

- **Single file**, no server, no setup. Just `pip install duckdb`.
- **SQL queries** for filtering and joining. Want all pitchers with
  200+ fastballs? `SELECT ... WHERE n_ff >= 200`.
- **Columnar storage** — efficient for the read-heavy, append-rare
  access pattern of research data.
- **Reusable across targets.** The pitching data feeds both the
  hierarchical funnel model (STUFF.md) and potential future models
  without duplicating CSVs or NPZ files.

### Loading Data in Target Code

```python
import duckdb

def load_german_credit():
    """Load German Credit dataset from DuckDB."""
    con = duckdb.connect("data/etd.duckdb", read_only=True)
    df = con.execute("SELECT * FROM german_credit").fetchdf()
    con.close()
    X = df.filter(like='x').values.astype(np.float64)
    y = df['y'].values.astype(np.int32)
    return X, y

def load_statcast_ff(min_pitches=200, season=2024):
    """Load filtered Statcast fastball data."""
    con = duckdb.connect("data/etd.duckdb", read_only=True)
    df = con.execute(f"""
        SELECT p.*
        FROM statcast_pitches p
        JOIN pitcher_ff_summary s ON p.pitcher_id = s.pitcher_id
        WHERE p.pitch_type = 'FF'
          AND p.season = {season}
          AND s.n_ff >= {min_pitches}
    """).fetchdf()
    con.close()
    return df
```

### Data Pipeline

Each dataset has a `download_and_store_{name}()` function in
`datasets.py`:

1. Download raw data (UCI via `sklearn.datasets`, Statcast via
   `pybaseball`).
2. Preprocess: standardize features, encode outcomes, handle missing
   values.
3. Insert into DuckDB table (idempotent: drops and recreates if exists).

```python
def download_and_store_german_credit(con):
    """Download German Credit from sklearn, preprocess, store."""
    from sklearn.datasets import fetch_openml
    data = fetch_openml('credit-g', version=1, as_frame=True)
    # ... preprocess ...
    con.execute("DROP TABLE IF EXISTS german_credit")
    con.execute("CREATE TABLE german_credit AS SELECT * FROM df")
```

---

## Script Dependency Summary

```
datasets.py  →  data/etd.duckdb    (write)
nuts.py      →  data/etd.duckdb    (read, for BLR targets)
             →  results/reference/  (write)
tune.py      →  run_single()       (shared loop)
             →  results/            (write)
run.py       →  run_single()       (shared loop)
             →  results/reference/  (read, for metrics)
             →  results/            (write)
```

All scripts share:
- `run_single()` for the algorithm loop
- Rich console for output
- The same Target / Config infrastructure from `src/etd/`