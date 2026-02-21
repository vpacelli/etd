"""DuckDB pipeline for UCI benchmark datasets.

Downloads, preprocesses, and stores datasets used for Bayesian logistic
regression benchmarks.  Categorical features are ordinal-encoded (via
``pd.factorize``), then all features are standardized (mean=0, std=1).

Usage:
    python -m experiments.datasets --all          # download all datasets
    python -m experiments.datasets --dataset german_credit
    python -m experiments.datasets --list         # list stored tables
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import duckdb
import numpy as np

DB_PATH = Path("data/etd.duckdb")


def get_connection(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection, creating the data directory if needed.

    Args:
        read_only: If True, open in read-only mode.

    Returns:
        DuckDB connection.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(DB_PATH), read_only=read_only)


def download_and_store_german_credit(con: duckdb.DuckDBPyConnection) -> int:
    """Fetch German Credit from OpenML, ordinal-encode, standardize, store.

    Returns:
        Number of rows stored.
    """
    from sklearn.datasets import fetch_openml

    data = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
    df = data.frame

    # Separate target
    y = (df["class"] == "good").astype(int).values  # 1=good, 0=bad

    X_df = df.drop(columns=["class"])

    # Ordinal-encode categoricals
    import pandas as pd
    for col in X_df.select_dtypes(include=["category", "object"]).columns:
        X_df[col], _ = pd.factorize(X_df[col])
        X_df[col] = X_df[col].astype(float)

    # Standardize all features
    X_arr = X_df.values.astype(float)
    mu = X_arr.mean(axis=0)
    std = X_arr.std(axis=0)
    std = np.where(std < 1e-10, 1.0, std)  # avoid division by zero
    X_arr = (X_arr - mu) / std

    n, d = X_arr.shape

    # Store in DuckDB
    con.execute("DROP TABLE IF EXISTS german_credit")
    con.execute(
        f"CREATE TABLE german_credit (y INTEGER, {', '.join(f'x{i} DOUBLE' for i in range(d))})"
    )

    # Insert rows
    for i in range(n):
        vals = [int(y[i])] + X_arr[i].tolist()
        placeholders = ", ".join(["?"] * (d + 1))
        con.execute(f"INSERT INTO german_credit VALUES ({placeholders})", vals)

    return n


def download_and_store_australian(con: duckdb.DuckDBPyConnection) -> int:
    """Fetch Australian Credit from OpenML, standardize, store.

    Returns:
        Number of rows stored.
    """
    from sklearn.datasets import fetch_openml

    data = fetch_openml("Australian", version=4, as_frame=True, parser="auto")
    df = data.frame

    # Target
    import pandas as pd
    y_col = df.columns[-1]
    y = df[y_col].astype(int).values

    X_df = df.drop(columns=[y_col])

    # Ordinal-encode categoricals
    for col in X_df.select_dtypes(include=["category", "object"]).columns:
        X_df[col], _ = pd.factorize(X_df[col])
        X_df[col] = X_df[col].astype(float)

    X_arr = X_df.values.astype(float)
    mu = X_arr.mean(axis=0)
    std = X_arr.std(axis=0)
    std = np.where(std < 1e-10, 1.0, std)
    X_arr = (X_arr - mu) / std

    n, d = X_arr.shape

    con.execute("DROP TABLE IF EXISTS australian")
    con.execute(
        f"CREATE TABLE australian (y INTEGER, {', '.join(f'x{i} DOUBLE' for i in range(d))})"
    )

    for i in range(n):
        vals = [int(y[i])] + X_arr[i].tolist()
        placeholders = ", ".join(["?"] * (d + 1))
        con.execute(f"INSERT INTO australian VALUES ({placeholders})", vals)

    return n


def download_and_store_ionosphere(con: duckdb.DuckDBPyConnection) -> int:
    """Fetch UCI Ionosphere from OpenML, standardize, drop constant cols, store.

    The ionosphere dataset has 351 instances and 34 numeric features.  Feature
    ``a02`` is constant zeros and is dropped, leaving 33 features.

    Returns:
        Number of rows stored.
    """
    from sklearn.datasets import fetch_openml

    data = fetch_openml("ionosphere", version=1, as_frame=True, parser="auto")
    df = data.frame

    # Identify target column (may be "Class" or "class" depending on version)
    target_col = [c for c in df.columns if c.lower() == "class"][0]
    y = (df[target_col] == "g").astype(int).values  # 1=good, 0=bad

    X_df = df.drop(columns=[target_col])

    # All features are numeric — convert to float array
    X_arr = X_df.values.astype(float)

    # Standardize, dropping constant columns (std < 1e-10)
    mu = X_arr.mean(axis=0)
    std = X_arr.std(axis=0)
    keep = std >= 1e-10
    X_arr = (X_arr[:, keep] - mu[keep]) / std[keep]

    n, d = X_arr.shape

    # Store in DuckDB
    con.execute("DROP TABLE IF EXISTS ionosphere")
    con.execute(
        f"CREATE TABLE ionosphere (y INTEGER, {', '.join(f'x{i} DOUBLE' for i in range(d))})"
    )

    for i in range(n):
        vals = [int(y[i])] + X_arr[i].tolist()
        placeholders = ", ".join(["?"] * (d + 1))
        con.execute(f"INSERT INTO ionosphere VALUES ({placeholders})", vals)

    return n


_DOWNLOADERS = {
    "german_credit": download_and_store_german_credit,
    "australian": download_and_store_australian,
    "ionosphere": download_and_store_ionosphere,
}


def load_dataset(name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load (X, y) from DuckDB.

    Args:
        name: Table name (e.g. ``"german_credit"``).

    Returns:
        Tuple ``(X, y)`` where X is ``(n, d)`` float64 and y is ``(n,)`` int32.

    Raises:
        KeyError: If the table does not exist.
    """
    if not DB_PATH.exists():
        raise KeyError(
            f"Database {DB_PATH} does not exist. "
            f"Run: python -m experiments.datasets --all"
        )
    con = get_connection(read_only=True)
    try:
        tables = list_tables(con)
        if name not in tables:
            raise KeyError(
                f"Table '{name}' not found in {DB_PATH}. "
                f"Available: {tables}. "
                f"Run: python -m experiments.datasets --dataset {name}"
            )
        result = con.execute(f"SELECT * FROM {name}").fetchnumpy()
    finally:
        con.close()

    y = np.asarray(result["y"], dtype=np.int32)
    # Feature columns are x0, x1, ..., xd
    feature_cols = sorted(
        [k for k in result.keys() if k.startswith("x")],
        key=lambda k: int(k[1:]),
    )
    X = np.column_stack([np.asarray(result[c], dtype=np.float64) for c in feature_cols])

    return X, y


def list_tables(con: duckdb.DuckDBPyConnection) -> list[str]:
    """List all table names in the DuckDB database.

    Args:
        con: DuckDB connection.

    Returns:
        List of table name strings.
    """
    rows = con.execute("SHOW TABLES").fetchall()
    return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ETD dataset pipeline")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--dataset", type=str, help="Download a specific dataset")
    parser.add_argument("--list", action="store_true", help="List stored tables")
    args = parser.parse_args()

    con = get_connection()

    if args.list:
        tables = list_tables(con)
        print(f"Tables in {DB_PATH}: {tables}")
        con.close()
        return

    if args.all:
        for name, fn in _DOWNLOADERS.items():
            n = fn(con)
            print(f"Stored {name}: {n} rows")
    elif args.dataset:
        if args.dataset not in _DOWNLOADERS:
            print(f"Unknown dataset '{args.dataset}'. Available: {list(_DOWNLOADERS)}")
            con.close()
            return
        n = _DOWNLOADERS[args.dataset](con)
        print(f"Stored {args.dataset}: {n} rows")
    else:
        parser.print_help()

    con.close()


if __name__ == "__main__":
    main()
