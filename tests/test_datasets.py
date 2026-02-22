"""Tests for the DuckDB dataset pipeline."""

import os
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# We patch DB_PATH to a temp file for all tests in this module
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary DuckDB database path."""
    db_path = tmp_path / "test.duckdb"
    with mock.patch("experiments.datasets.DB_PATH", db_path):
        yield db_path


# ---------------------------------------------------------------------------
# Download + store + load roundtrip
# ---------------------------------------------------------------------------

class TestGermanCredit:
    @pytest.mark.slow
    def test_download_store_load(self, temp_db):
        """Full roundtrip: download → store → load."""
        from experiments.datasets import (
            download_and_store_german_credit,
            get_connection,
            load_dataset,
        )

        con = get_connection()
        n = download_and_store_german_credit(con)
        con.close()

        assert n == 1000  # German Credit has 1000 samples

        X, y = load_dataset("german_credit")
        assert X.shape[0] == 1000
        assert X.ndim == 2
        assert y.shape == (1000,)
        assert X.dtype == np.float64
        assert y.dtype == np.int32

        # Standardized: mean ~0, std ~1
        np.testing.assert_allclose(X.mean(axis=0), 0.0, atol=0.1)

        # Labels are binary
        assert set(np.unique(y)).issubset({0, 1})


class TestAustralian:
    @pytest.mark.slow
    def test_download_store_load(self, temp_db):
        from experiments.datasets import (
            download_and_store_australian,
            get_connection,
            load_dataset,
        )

        con = get_connection()
        n = download_and_store_australian(con)
        con.close()

        assert n == 690  # Australian Credit has 690 samples

        X, y = load_dataset("australian")
        assert X.shape[0] == 690
        assert y.shape == (690,)


class TestIonosphere:
    @pytest.mark.slow
    def test_download_store_load(self, temp_db):
        """Full roundtrip: download → store → load."""
        from experiments.datasets import (
            download_and_store_ionosphere,
            get_connection,
            load_dataset,
        )

        con = get_connection()
        n = download_and_store_ionosphere(con)
        con.close()

        assert n == 351  # Ionosphere has 351 samples

        X, y = load_dataset("ionosphere")
        assert X.shape == (351, 33)  # 34 features minus 1 constant (a02)
        assert y.shape == (351,)
        assert X.dtype == np.float64
        assert y.dtype == np.int32

        # Standardized: mean ~0
        np.testing.assert_allclose(X.mean(axis=0), 0.0, atol=0.1)

        # No constant columns remain
        assert np.all(X.std(axis=0) > 0.1)

        # Labels are binary
        assert set(np.unique(y)).issubset({0, 1})


# ---------------------------------------------------------------------------
# Table listing and missing table
# ---------------------------------------------------------------------------

class TestSonar:
    @pytest.mark.slow
    def test_download_store_load(self, temp_db):
        """Full roundtrip: download → store → load."""
        from experiments.datasets import (
            download_and_store_sonar,
            get_connection,
            load_dataset,
        )

        con = get_connection()
        n = download_and_store_sonar(con)
        con.close()

        assert n == 208  # Sonar has 208 samples

        X, y = load_dataset("sonar")
        assert X.shape == (208, 60)  # 60 frequency-band features
        assert y.shape == (208,)
        assert X.dtype == np.float64
        assert y.dtype == np.int32

        # Standardized: mean ~0
        np.testing.assert_allclose(X.mean(axis=0), 0.0, atol=0.1)

        # Labels are binary
        assert set(np.unique(y)).issubset({0, 1})


class TestHelpers:
    def test_list_tables_empty(self, temp_db):
        from experiments.datasets import get_connection, list_tables

        con = get_connection()
        tables = list_tables(con)
        con.close()
        assert tables == []

    def test_load_missing_table(self, temp_db):
        from experiments.datasets import load_dataset

        with pytest.raises(KeyError, match="(not found|does not exist)"):
            load_dataset("nonexistent_table")
