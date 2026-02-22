"""Convergence diagnostics for particle-based inference.

All functions are pure JAX operations — JIT-compatible with no Python
control flow on array values.
"""

import jax
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

    .. warning::
        Materializes the full ``(N, M)`` matrix.  For large point sets
        (e.g. reference-vs-reference with M > 5000), prefer
        :func:`_mean_pairwise_distance` which computes the mean without
        allocating the full matrix.

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


def _mean_pairwise_distance(
    X: jnp.ndarray,  # (N, d)
    Y: jnp.ndarray,  # (M, d)
    chunk_size: int = 256,
) -> jnp.ndarray:     # scalar
    """Mean Euclidean distance between two point sets, O(chunk·M) memory.

    Computes ``mean(||x_i - y_j||)`` over all (i, j) pairs by scanning
    over chunks of X rows, computing a ``(chunk, M)`` distance block at
    a time and immediately reducing to a scalar sum.  Peak memory is
    O(chunk_size · M) instead of O(N · M).

    Uses ``lax.scan`` (not vmap) to prevent XLA from fusing all rows
    into one large allocation.

    Args:
        X: First point set, shape ``(N, d)``.
        Y: Second point set, shape ``(M, d)``.
        chunk_size: Rows of X to process per scan step (default 256).

    Returns:
        Mean pairwise Euclidean distance (scalar).
    """
    N = X.shape[0]
    M = Y.shape[0]
    yy = jnp.sum(Y ** 2, axis=1)  # (M,) — precompute once

    # Pad X to a multiple of chunk_size so scan has fixed-shape slices
    n_chunks = (N + chunk_size - 1) // chunk_size
    padded_n = n_chunks * chunk_size
    X_pad = jnp.zeros((padded_n, X.shape[1]), dtype=X.dtype)
    X_pad = X_pad.at[:N].set(X)

    # Reshape into (n_chunks, chunk_size, d)
    X_chunks = X_pad.reshape(n_chunks, chunk_size, -1)

    def _chunk_sum(acc, x_chunk):
        """Sum of distances from a chunk of rows to all of Y."""
        # x_chunk: (chunk_size, d)
        xx = jnp.sum(x_chunk ** 2, axis=1)          # (chunk_size,)
        dots = x_chunk @ Y.T                         # (chunk_size, M)
        sq = xx[:, None] + yy[None, :] - 2.0 * dots  # (chunk_size, M)
        dists = jnp.sqrt(jnp.maximum(sq, 0.0))       # (chunk_size, M)
        return acc + jnp.sum(dists), None

    total, _ = jax.lax.scan(_chunk_sum, jnp.float32(0.0), X_chunks)

    # Subtract contribution from padded rows (distance from zero-rows to Y)
    if N < padded_n:
        # Padded rows are zeros, their distances to Y are just ||y_j||
        pad_contrib = (padded_n - N) * jnp.sum(jnp.sqrt(jnp.maximum(yy, 0.0)))
        total = total - pad_contrib

    return total / (N * M)


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
    cross = _mean_pairwise_distance(particles, reference)
    self_p = _mean_pairwise_distance(particles, particles)
    self_r = _mean_pairwise_distance(reference, reference)

    return 2.0 * cross - self_p - self_r


def sliced_wasserstein(
    particles: jnp.ndarray,   # (N, d)
    reference: jnp.ndarray,   # (M, d)
    key: jax.Array,
    n_projections: int = 100,
) -> jnp.ndarray:              # scalar
    """Sliced Wasserstein-2 distance between two point clouds.

    Projects both sets onto ``n_projections`` random 1-D directions, computes
    the exact 1-D W2 (sort-and-compare) on each slice, and returns the mean.

    .. math::
        \\mathrm{SW}_2(P, R) = \\left(\\frac{1}{L}\\sum_{l=1}^{L}
            W_2^2(\\theta_l^\\top P,\\; \\theta_l^\\top R)\\right)^{1/2}

    Args:
        particles: Particle positions, shape ``(N, d)``.
        reference: Reference samples, shape ``(M, d)``.
        key: JAX PRNG key for random projections.
        n_projections: Number of random 1-D slices (default 100).

    Returns:
        Sliced Wasserstein-2 distance (non-negative scalar).
    """
    d = particles.shape[1]
    # Random unit directions: sample from standard normal, then normalize
    directions = jax.random.normal(key, (n_projections, d))       # (L, d)
    directions = directions / jnp.linalg.norm(
        directions, axis=1, keepdims=True,
    )                                                              # (L, d)

    # Project: (L, N) and (L, M)
    proj_p = particles @ directions.T   # (N, L)
    proj_r = reference @ directions.T   # (M, L)

    # Sort along sample axis
    proj_p = jnp.sort(proj_p, axis=0)   # (N, L)
    proj_r = jnp.sort(proj_r, axis=0)   # (M, L)

    # Quantile alignment when N != M: interpolate the smaller set onto the
    # larger set's quantiles so both have the same length along axis 0.
    n, m = proj_p.shape[0], proj_r.shape[0]
    size = max(n, m)
    quantiles = jnp.linspace(0.0, 1.0, size)

    def _interp_cols(sorted_proj, orig_len):
        """Linearly interpolate each projection slice to ``size`` points."""
        orig_q = jnp.linspace(0.0, 1.0, orig_len)
        # vmap over projection columns
        return jax.vmap(
            lambda col: jnp.interp(quantiles, orig_q, col),
            in_axes=1, out_axes=1,
        )(sorted_proj)

    proj_p = _interp_cols(proj_p, n)  # (size, L)
    proj_r = _interp_cols(proj_r, m)  # (size, L)

    # Mean squared difference per slice, then sqrt of mean
    w2_sq_per_slice = jnp.mean((proj_p - proj_r) ** 2, axis=0)   # (L,)
    return jnp.sqrt(jnp.mean(w2_sq_per_slice))


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
        tolerance: Maximum Euclidean distance threshold.  The default
            2.0 is calibrated for **2-D** unit-variance components.
            For higher dimensions, callers should scale as
            ``2 * component_std * sqrt(d / 2)`` to maintain a constant
            coverage probability (~86.5 %).

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


def mean_rmse(
    particles: jnp.ndarray,   # (N, d)
    reference: jnp.ndarray,   # (M, d)
) -> jnp.ndarray:              # scalar
    """RMSE of particle mean vs reference mean.

    .. math::
        \\mathrm{RMSE} = \\sqrt{\\frac{1}{d} \\sum_k
            (\\bar{x}_k - \\bar{r}_k)^2}

    Args:
        particles: Particle positions, shape ``(N, d)``.
        reference: Reference samples, shape ``(M, d)``.

    Returns:
        Root mean squared error (scalar).
    """
    particle_mean = jnp.mean(particles, axis=0)  # (d,)
    ref_mean = jnp.mean(reference, axis=0)        # (d,)
    return jnp.sqrt(jnp.mean((particle_mean - ref_mean) ** 2))


def variance_ratio_vs_reference(
    particles: jnp.ndarray,   # (N, d)
    reference: jnp.ndarray,   # (M, d)
) -> jnp.ndarray:              # scalar
    """Median per-dimension variance ratio vs reference samples.

    Computes ``Var(particles) / Var(reference)`` per dimension, returns
    the median.  Should be close to 1.0 for well-converged particles.

    Args:
        particles: Particle positions, shape ``(N, d)``.
        reference: Reference samples, shape ``(M, d)``.

    Returns:
        Median variance ratio (scalar).
    """
    var_p = jnp.var(particles, axis=0)   # (d,)
    var_r = jnp.var(reference, axis=0)   # (d,)
    ratios = var_p / jnp.maximum(var_r, 1e-10)
    return jnp.median(ratios)
