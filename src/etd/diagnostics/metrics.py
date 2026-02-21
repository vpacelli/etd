"""Convergence diagnostics for particle-based inference.

All functions are pure JAX operations — JIT-compatible with no Python
control flow on array values.
"""

import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _pairwise_distances(
    X: jnp.ndarray,  # (N, d)
    Y: jnp.ndarray,  # (M, d)
) -> jnp.ndarray:     # (N, M)
    """Euclidean distance matrix via dot-product expansion.

    Uses ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x . y, then sqrt.
    Floors at zero before sqrt to avoid NaN from float cancellation.

    Args:
        X: First point set, shape ``(N, d)``.
        Y: Second point set, shape ``(M, d)``.

    Returns:
        Distance matrix, shape ``(N, M)``.
    """
    xx = jnp.sum(X ** 2, axis=1)  # (N,)
    yy = jnp.sum(Y ** 2, axis=1)  # (M,)
    xy = X @ Y.T                   # (N, M)
    sq_dists = xx[:, None] + yy[None, :] - 2.0 * xy
    return jnp.sqrt(jnp.maximum(sq_dists, 0.0))


# ---------------------------------------------------------------------------
# Public metrics
# ---------------------------------------------------------------------------

def energy_distance(
    particles: jnp.ndarray,   # (N, d)
    reference: jnp.ndarray,   # (M, d)
) -> jnp.ndarray:              # scalar
    """V-statistic energy distance between two point clouds.

    .. math::
        \\mathcal{E}(P, R) = 2\\,\\mathbb{E}\\|X - Y\\|
            - \\mathbb{E}\\|X - X'\\| - \\mathbb{E}\\|Y - Y'\\|

    Includes diagonal terms (bias is O(1/N), negligible for N >= 100).
    Matches the scipy convention.

    Args:
        particles: Particle positions, shape ``(N, d)``.
        reference: Reference samples, shape ``(M, d)``.

    Returns:
        Energy distance (non-negative scalar).
    """
    cross = _pairwise_distances(particles, reference)
    self_p = _pairwise_distances(particles, particles)
    self_r = _pairwise_distances(reference, reference)

    return 2.0 * jnp.mean(cross) - jnp.mean(self_p) - jnp.mean(self_r)


def mode_coverage(
    particles: jnp.ndarray,    # (N, d)
    mode_centers: jnp.ndarray, # (K, d)
    tolerance: float = 2.0,
) -> jnp.ndarray:               # scalar
    """Fraction of modes with at least one nearby particle.

    A mode is "covered" if ``min_i ||x_i - mu_k|| < tolerance``.

    Args:
        particles: Particle positions, shape ``(N, d)``.
        mode_centers: Mode center positions, shape ``(K, d)``.
        tolerance: Maximum distance threshold (default 2.0, i.e. 2 sigma
            for unit-variance components).

    Returns:
        Fraction of covered modes in [0, 1].
    """
    dists = _pairwise_distances(mode_centers, particles)  # (K, N)
    min_dists = jnp.min(dists, axis=1)                     # (K,)
    covered = min_dists < tolerance
    return jnp.mean(covered.astype(jnp.float32))


def mean_error(
    particles: jnp.ndarray,  # (N, d)
    true_mean: jnp.ndarray,  # (d,)
) -> jnp.ndarray:             # scalar
    """L2 error between the particle centroid and the true mean.

    Args:
        particles: Particle positions, shape ``(N, d)``.
        true_mean: True mean of the target, shape ``(d,)``.

    Returns:
        Euclidean distance ``||mean(particles) - true_mean||_2``.
    """
    return jnp.linalg.norm(jnp.mean(particles, axis=0) - true_mean)


def variance_ratio(
    particles: jnp.ndarray,     # (N, d)
    true_variance: jnp.ndarray, # (d,)
) -> jnp.ndarray:                # (d,)
    """Per-dimension ratio of empirical to true variance.

    Should be close to 1.0 when particles have converged.

    Args:
        particles: Particle positions, shape ``(N, d)``.
        true_variance: True per-dimension variance, shape ``(d,)``.

    Returns:
        Variance ratios, shape ``(d,)``.
    """
    empirical_var = jnp.var(particles, axis=0)
    return empirical_var / jnp.maximum(true_variance, 1e-10)
