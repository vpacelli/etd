"""Tests for mode_proximity and mode_balance diagnostics."""

import jax.numpy as jnp
import numpy as np
import pytest

from etd.diagnostics.metrics import mode_balance, mode_proximity


class TestModeProximity:
    def test_perfect_placement(self):
        """Particles exactly at modes → proximity ≈ 0."""
        modes = jnp.array([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=jnp.float32)
        particles = modes.copy()
        prox = mode_proximity(particles, modes, component_std=1.0, dim=2)
        assert float(prox) == pytest.approx(0.0, abs=1e-5)

    def test_distant_particles(self):
        """Particles far from modes → proximity > 0."""
        modes = jnp.array([[0, 0], [10, 0]], dtype=jnp.float32)
        particles = jnp.array([[50, 50], [60, 60]], dtype=jnp.float32)
        prox = mode_proximity(particles, modes, component_std=1.0, dim=2)
        assert float(prox) > 1.0

    def test_normalization(self):
        """Doubling σ halves proximity; quadrupling d halves it."""
        modes = jnp.array([[0.0, 0.0]], dtype=jnp.float32)
        particles = jnp.array([[1.0, 0.0]], dtype=jnp.float32)

        # Distance is 1.0 in all cases
        p1 = float(mode_proximity(particles, modes, component_std=1.0, dim=2))
        p2 = float(mode_proximity(particles, modes, component_std=2.0, dim=2))
        p3 = float(mode_proximity(particles, modes, component_std=1.0, dim=8))

        # Doubling sigma should halve the proximity
        assert p2 == pytest.approx(p1 / 2.0, rel=1e-5)
        # Quadrupling dim: scale = sqrt(8)/sqrt(2) = 2, so proximity halved
        assert p3 == pytest.approx(p1 / 2.0, rel=1e-5)

    def test_nearest_particle_used(self):
        """Only the nearest particle to each mode matters."""
        modes = jnp.array([[0.0, 0.0]], dtype=jnp.float32)
        # One close, one far — only the close one should count
        particles = jnp.array([
            [0.1, 0.0],
            [100.0, 100.0],
        ], dtype=jnp.float32)
        prox = mode_proximity(particles, modes, component_std=1.0, dim=2)
        # Should be 0.1 / sqrt(2) ≈ 0.0707
        expected = 0.1 / np.sqrt(2.0)
        assert float(prox) == pytest.approx(expected, rel=1e-4)


class TestModeBalance:
    def test_uniform_assignment(self):
        """Equal particles per mode → JSD ≈ 0."""
        modes = jnp.array([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=jnp.float32)
        # 25 particles near each mode
        particles = jnp.concatenate([
            modes[k:k+1] + 0.01 * jnp.ones((25, 2))
            for k in range(4)
        ])
        bal = mode_balance(particles, modes)
        assert float(bal) == pytest.approx(0.0, abs=1e-4)

    def test_collapsed_to_one_mode(self):
        """All particles on one mode → JSD > 0."""
        modes = jnp.array([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=jnp.float32)
        particles = jnp.zeros((100, 2))  # all near mode 0
        bal = mode_balance(particles, modes)
        assert float(bal) > 0.3  # ~0.38 for K=4 all-on-one-mode

    def test_bounded_by_ln2(self):
        """JSD is bounded in [0, ln2]."""
        modes = jnp.array([[0, 0], [10, 0]], dtype=jnp.float32)
        particles = jnp.zeros((100, 2))
        bal = float(mode_balance(particles, modes))
        assert 0 <= bal <= np.log(2) + 1e-6

    def test_two_modes_half_half(self):
        """50-50 split between two modes → JSD ≈ 0."""
        modes = jnp.array([[0, 0], [10, 0]], dtype=jnp.float32)
        particles = jnp.concatenate([
            jnp.zeros((50, 2)),
            jnp.ones((50, 2)) * 10,
        ])
        bal = mode_balance(particles, modes)
        assert float(bal) == pytest.approx(0.0, abs=1e-4)
