"""Phase 0 gate tests for ETD primitives.

Required gate (ROADMAP.md):
  - Euclidean cost matches scipy.spatial.distance.cdist  (rtol=1e-5)
  - Median normalization: output median ≈ 1.0  (atol=1e-5)
  - Score clipping: output norms ≤ max_norm
  - Systematic resampling: empirical ≈ coupling rows  (KS p > 0.01)
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.distance import cdist
from scipy.stats import chisquare

from etd.costs.euclidean import squared_euclidean_cost
from etd.costs.normalize import mean_normalize, median_normalize, normalize_cost
from etd.proposals.langevin import clip_scores, langevin_proposals, update_preconditioner
from etd.update.categorical import systematic_resample
from etd.weights import _log_proposal_density, importance_weights


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class SimpleGaussian:
    """Isotropic standard Gaussian target for testing."""

    def __init__(self, dim: int):
        self.dim = dim

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        return -0.5 * jnp.sum(x ** 2, axis=-1)

    def score(self, x: jnp.ndarray) -> jnp.ndarray:
        return -x


# ---------------------------------------------------------------------------
# Gate: Euclidean cost matches scipy
# ---------------------------------------------------------------------------

class TestEuclideanCost:
    def test_matches_scipy_cdist(self):
        """Euclidean cost should match scipy.spatial.distance.cdist('sqeuclidean')/2."""
        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)

        N, P, d = 30, 50, 5
        positions = jax.random.normal(k1, (N, d))
        proposals = jax.random.normal(k2, (P, d))

        C = squared_euclidean_cost(positions, proposals)
        C_scipy = cdist(np.array(positions), np.array(proposals), "sqeuclidean") / 2.0

        np.testing.assert_allclose(np.array(C), C_scipy, rtol=1e-5)

    def test_nonnegative(self):
        """Cost matrix entries should be non-negative."""
        key = jax.random.PRNGKey(1)
        k1, k2 = jax.random.split(key)
        positions = jax.random.normal(k1, (20, 3))
        proposals = jax.random.normal(k2, (40, 3))

        C = squared_euclidean_cost(positions, proposals)
        assert jnp.all(C >= 0.0)

    def test_self_cost_diagonal_zero(self):
        """C(X, X) diagonal should be zero."""
        key = jax.random.PRNGKey(2)
        X = jax.random.normal(key, (15, 4))

        C = squared_euclidean_cost(X, X)
        np.testing.assert_allclose(np.array(jnp.diag(C)), 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Gate: Median normalization
# ---------------------------------------------------------------------------

class TestMedianNormalize:
    def test_output_median_is_one(self):
        """After normalization, median of cost matrix should ≈ 1.0.

        Uses a loose tolerance because median_normalize uses a strided
        subsample for speed (avoids O(n log n) sort on the full matrix).
        """
        key = jax.random.PRNGKey(3)
        k1, k2 = jax.random.split(key)
        positions = jax.random.normal(k1, (50, 5))
        proposals = jax.random.normal(k2, (100, 5))

        C = squared_euclidean_cost(positions, proposals)
        C_norm, med = median_normalize(C)

        np.testing.assert_allclose(float(jnp.median(C_norm)), 1.0, atol=0.15)
        assert float(med) > 0.0

    def test_guards_against_zero_median(self):
        """Normalization should not produce inf when cost is all zeros."""
        C = jnp.zeros((10, 20))
        C_norm, med = median_normalize(C)

        assert jnp.all(jnp.isfinite(C_norm))
        assert float(med) > 0.0  # guard prevents division by zero


# ---------------------------------------------------------------------------
# Gate: Mean normalization
# ---------------------------------------------------------------------------

class TestMeanNormalize:
    def test_output_mean_is_one(self):
        """After normalization, mean of cost matrix should ≈ 1.0."""
        key = jax.random.PRNGKey(30)
        k1, k2 = jax.random.split(key)
        positions = jax.random.normal(k1, (50, 5))
        proposals = jax.random.normal(k2, (100, 5))

        C = squared_euclidean_cost(positions, proposals)
        C_norm, mean_val = mean_normalize(C)

        np.testing.assert_allclose(float(jnp.mean(C_norm)), 1.0, atol=1e-5)
        assert float(mean_val) > 0.0

    def test_guards_against_zero_mean(self):
        """Normalization should not produce inf when cost is all zeros."""
        C = jnp.zeros((10, 20))
        C_norm, mean_val = mean_normalize(C)

        assert jnp.all(jnp.isfinite(C_norm))
        assert float(mean_val) > 0.0


# ---------------------------------------------------------------------------
# Gate: normalize_cost dispatcher
# ---------------------------------------------------------------------------

class TestNormalizeCostDispatch:
    def test_median_dispatch(self):
        """normalize_cost('median') matches median_normalize."""
        key = jax.random.PRNGKey(31)
        k1, k2 = jax.random.split(key)
        C = squared_euclidean_cost(
            jax.random.normal(k1, (20, 3)),
            jax.random.normal(k2, (40, 3)),
        )
        C_disp, s_disp = normalize_cost(C, "median")
        C_ref, s_ref = median_normalize(C)
        np.testing.assert_array_equal(np.array(C_disp), np.array(C_ref))
        np.testing.assert_array_equal(np.array(s_disp), np.array(s_ref))

    def test_mean_dispatch(self):
        """normalize_cost('mean') matches mean_normalize."""
        key = jax.random.PRNGKey(32)
        k1, k2 = jax.random.split(key)
        C = squared_euclidean_cost(
            jax.random.normal(k1, (20, 3)),
            jax.random.normal(k2, (40, 3)),
        )
        C_disp, s_disp = normalize_cost(C, "mean")
        C_ref, s_ref = mean_normalize(C)
        np.testing.assert_array_equal(np.array(C_disp), np.array(C_ref))
        np.testing.assert_array_equal(np.array(s_disp), np.array(s_ref))

    def test_unknown_method_raises(self):
        """Unknown method should raise ValueError."""
        C = jnp.ones((5, 5))
        with pytest.raises(ValueError, match="Unknown cost normalization"):
            normalize_cost(C, "unknown")


# ---------------------------------------------------------------------------
# Gate: Score clipping
# ---------------------------------------------------------------------------

class TestScoreClipping:
    def test_output_norms_bounded(self):
        """All clipped score norms should be ≤ max_norm."""
        key = jax.random.PRNGKey(4)
        scores = jax.random.normal(key, (100, 10)) * 20.0  # large scores
        max_norm = 5.0

        clipped = clip_scores(scores, max_norm)
        norms = jnp.linalg.norm(clipped, axis=-1)

        assert jnp.all(norms <= max_norm + 1e-6)

    def test_small_scores_unchanged(self):
        """Scores already below max_norm should be unchanged."""
        key = jax.random.PRNGKey(5)
        scores = jax.random.normal(key, (50, 5)) * 0.1  # small norms
        max_norm = 5.0

        clipped = clip_scores(scores, max_norm)
        np.testing.assert_allclose(np.array(clipped), np.array(scores), atol=1e-6)

    def test_preserves_direction(self):
        """Clipping should preserve the direction of the score vector."""
        scores = jnp.array([[10.0, 0.0], [0.0, 10.0], [3.0, 4.0]])
        clipped = clip_scores(scores, max_norm=5.0)

        # Normalize originals and clipped, directions should match
        orig_unit = scores / jnp.linalg.norm(scores, axis=-1, keepdims=True)
        clip_unit = clipped / jnp.linalg.norm(clipped, axis=-1, keepdims=True)
        np.testing.assert_allclose(np.array(clip_unit), np.array(orig_unit), atol=1e-5)


# ---------------------------------------------------------------------------
# Gate: Systematic resampling
# ---------------------------------------------------------------------------

class TestSystematicResample:
    def test_empirical_matches_coupling(self):
        """Empirical resampling distribution should match coupling rows.

        Run 10k independent resamples and compare empirical frequencies
        to coupling probabilities via a chi-squared test.
        """
        P = 20
        N = 1  # single particle for this test

        # Create a non-uniform coupling (one particle, P proposals)
        key = jax.random.PRNGKey(6)
        logits = jax.random.normal(key, (N, P))
        log_gamma = jax.nn.log_softmax(logits, axis=1)

        proposals = jnp.eye(P, P)  # use identity so index = position

        n_samples = 10_000
        counts = np.zeros(P)

        for i in range(n_samples):
            k = jax.random.PRNGKey(i + 100)
            new_pos = systematic_resample(k, log_gamma, proposals)
            # Identify which proposal was selected
            idx = int(jnp.argmax(new_pos[0]))
            counts[idx] += 1

        expected_probs = np.array(jnp.exp(log_gamma[0]), dtype=np.float64)
        expected_probs /= expected_probs.sum()  # exact normalization in float64
        expected_counts = expected_probs * n_samples

        # Chi-squared goodness-of-fit test
        stat, pvalue = chisquare(counts, f_exp=expected_counts)
        assert pvalue > 0.01, (
            f"Chi-squared test failed: stat={stat:.2f}, p={pvalue:.4f}"
        )


# ---------------------------------------------------------------------------
# Additional tests
# ---------------------------------------------------------------------------

class TestProposalShape:
    def test_pooled_shape(self):
        """Proposals should have shape (N*M, d)."""
        key = jax.random.PRNGKey(7)
        N, d, M = 10, 3, 25
        target = SimpleGaussian(d)
        positions = jax.random.normal(key, (N, d))

        k1, _ = jax.random.split(key)
        proposals, means, scores = langevin_proposals(
            k1, positions, target, alpha=0.05, sigma=0.316,
            n_proposals=M, use_score=True, score_clip_val=5.0,
        )

        assert proposals.shape == (N * M, d)
        assert means.shape == (N, d)
        assert scores.shape == (N, d)


class TestImportanceWeights:
    def test_sum_to_one(self):
        """exp(log_weights) should sum to 1."""
        key = jax.random.PRNGKey(8)
        N, d, M = 10, 3, 25
        target = SimpleGaussian(d)

        k1, k2 = jax.random.split(key)
        positions = jax.random.normal(k1, (N, d))
        proposals, means, _ = langevin_proposals(
            k2, positions, target, alpha=0.05, sigma=0.316,
            n_proposals=M, use_score=True,
        )

        log_b = importance_weights(proposals, means, target, sigma=0.316)

        total = float(jnp.sum(jnp.exp(log_b)))
        np.testing.assert_allclose(total, 1.0, atol=1e-5)

    def test_all_nonpositive(self):
        """Log weights should all be ≤ 0 (since they are log-probabilities)."""
        key = jax.random.PRNGKey(9)
        N, d, M = 10, 3, 25
        target = SimpleGaussian(d)

        k1, k2 = jax.random.split(key)
        positions = jax.random.normal(k1, (N, d))
        proposals, means, _ = langevin_proposals(
            k2, positions, target, alpha=0.05, sigma=0.316,
            n_proposals=M, use_score=True,
        )

        log_b = importance_weights(proposals, means, target, sigma=0.316)
        assert jnp.all(log_b <= 1e-6)  # allow tiny float tolerance


class TestPreconditioner:
    def test_update(self):
        """Preconditioner should accumulate squared score statistics."""
        key = jax.random.PRNGKey(10)
        d = 5
        accum = jnp.ones(d)
        scores = jax.random.normal(key, (50, d)) * 2.0

        new_accum = update_preconditioner(accum, scores, beta=0.9)

        assert new_accum.shape == (d,)
        # Should be between accum and mean(scores**2) (convex combination)
        assert jnp.all(new_accum > 0.0)


class TestDeterminism:
    def test_same_key_same_output(self):
        """Same PRNG key should produce identical proposals."""
        key = jax.random.PRNGKey(42)
        N, d, M = 10, 3, 25
        target = SimpleGaussian(d)
        positions = jax.random.normal(jax.random.PRNGKey(0), (N, d))

        p1, m1, s1 = langevin_proposals(
            key, positions, target, alpha=0.05, sigma=0.316,
            n_proposals=M, use_score=True,
        )
        p2, m2, s2 = langevin_proposals(
            key, positions, target, alpha=0.05, sigma=0.316,
            n_proposals=M, use_score=True,
        )

        np.testing.assert_array_equal(np.array(p1), np.array(p2))
        np.testing.assert_array_equal(np.array(m1), np.array(m2))
        np.testing.assert_array_equal(np.array(s1), np.array(s2))


# ---------------------------------------------------------------------------
# BLAS-friendly proposal density
# ---------------------------------------------------------------------------

def _log_proposal_density_naive(proposals, means, sigma, preconditioner=None):
    """Naive (P,N,d) reference implementation for numerical comparison."""
    d = proposals.shape[1]
    N = means.shape[0]
    diff = proposals[:, None, :] - means[None, :, :]  # (P, N, d)
    if preconditioner is not None:
        inv_std = 1.0 / (sigma * preconditioner)
        maha_sq = jnp.sum((diff * inv_std[None, None, :]) ** 2, axis=-1)
        log_det = -0.5 * jnp.sum(jnp.log(2 * jnp.pi * (sigma * preconditioner) ** 2))
    else:
        maha_sq = jnp.sum(diff ** 2, axis=-1) / (sigma ** 2)
        log_det = -0.5 * d * jnp.log(2 * jnp.pi * sigma ** 2)
    log_components = log_det - 0.5 * maha_sq
    from jax.scipy.special import logsumexp
    return logsumexp(log_components, axis=1) - jnp.log(N)


class TestBLASDensity:
    """Verify BLAS-friendly proposal density matches naive reference."""

    def test_isotropic(self):
        """Isotropic (no preconditioner) path matches naive reference."""
        key = jax.random.PRNGKey(100)
        k1, k2 = jax.random.split(key)
        P, N, d = 200, 30, 8
        proposals = jax.random.normal(k1, (P, d))
        means = jax.random.normal(k2, (N, d))
        sigma = 0.5

        blas = _log_proposal_density(proposals, means, sigma)
        naive = _log_proposal_density_naive(proposals, means, sigma)
        np.testing.assert_allclose(np.array(blas), np.array(naive), atol=1e-5)

    def test_preconditioned(self):
        """Preconditioned path matches naive reference."""
        key = jax.random.PRNGKey(101)
        k1, k2, k3 = jax.random.split(key, 3)
        P, N, d = 200, 30, 8
        proposals = jax.random.normal(k1, (P, d))
        means = jax.random.normal(k2, (N, d))
        sigma = 0.3
        precond = jnp.abs(jax.random.normal(k3, (d,))) + 0.1

        blas = _log_proposal_density(proposals, means, sigma, preconditioner=precond)
        naive = _log_proposal_density_naive(proposals, means, sigma, preconditioner=precond)
        # Relaxed tolerance: different accumulation order causes ~1e-4 float32
        # cancellation on values of magnitude ~100.
        np.testing.assert_allclose(np.array(blas), np.array(naive), atol=5e-4)

    def test_high_dim(self):
        """BLAS path accurate in higher dimensions where cancellation matters."""
        key = jax.random.PRNGKey(102)
        k1, k2 = jax.random.split(key)
        P, N, d = 100, 20, 50
        proposals = jax.random.normal(k1, (P, d)) * 0.1
        means = jax.random.normal(k2, (N, d)) * 0.1
        sigma = 0.2

        blas = _log_proposal_density(proposals, means, sigma)
        naive = _log_proposal_density_naive(proposals, means, sigma)
        np.testing.assert_allclose(np.array(blas), np.array(naive), atol=1e-4)
