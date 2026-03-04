"""Parity and unit tests for mfe_toolbox.distributions.composite_likelihood.

Tests the composite normal log-likelihood (bivariate pairwise) function that
is migrated from MATLAB ``composite_likelihood.m`` + C MEX
``composite_likelihood.c`` to Python with Numba JIT acceleration.

The function computes the negative composite normal log-likelihood by summing
bivariate normal contributions over all specified index pairs.  Two code paths
are exercised:

1. **Square data** (K × K matrix, e.g. outer product of returns)
2. **Vector data** (K-length vector, e.g. raw returns)

Per AAP Section 0.7.1:
- ``numpy.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-4)``
- Numba ``@jit(nopython=True, cache=True)`` must work without object mode
- Coverage ≥90%, all tests pass with ``pytest -x --tb=short``

Ref: distributions/composite_likelihood.m (51 lines)
Ref: mex_source/composite_likelihood.c (133 lines)
"""

import math
import time

import numba
import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.distributions.composite_likelihood import (
    _composite_likelihood_core_square,
    _composite_likelihood_core_vector,
    composite_likelihood,
)

# Re-export conftest constants locally for clarity.
# ATOL and RTOL are also available via conftest fixtures, but having them
# accessible as module-level constants improves readability.
ATOL: float = 1e-6
RTOL: float = 1e-4

# Precomputed constant used in the composite likelihood formula.
# Equals 2 * log(2 * pi) — Ref: composite_likelihood.m:23
LIK_CONST: float = 3.67575413281869


# ---------------------------------------------------------------------------
# Pytest Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_corr_3() -> np.ndarray:
    """Return a K=3 symmetric positive-definite correlation matrix."""
    S = np.array(
        [[1.0, 0.5, 0.3],
         [0.5, 1.0, 0.2],
         [0.3, 0.2, 1.0]],
        dtype=np.float64,
    )
    return S


@pytest.fixture(scope="module")
def all_pairs_3() -> np.ndarray:
    """Return all unique K=3 index pairs (0-based)."""
    return np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int64)


@pytest.fixture(scope="module")
def rng_local() -> np.random.Generator:
    """Return a locally-seeded RNG for reproducible test data."""
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def square_data_3(rng_local: np.random.Generator) -> np.ndarray:
    """Return a K=3 square data matrix from rng(42)."""
    x = rng_local.standard_normal(3)
    return np.outer(x, x)


@pytest.fixture(scope="module")
def vector_data_3(rng_local: np.random.Generator) -> np.ndarray:
    """Return a K=3 vector data array from rng(42)."""
    # Re-seed to get the same base vector as square_data_3
    rng_fresh = np.random.default_rng(42)
    return rng_fresh.standard_normal(3)


@pytest.fixture(scope="module")
def fixture_data() -> dict:
    """Load the MATLAB-generated composite_likelihood fixture data.

    The fixture is a dict-in-npy with 6 test cases covering matrix/vector
    data for K=3 (all pairs, single pair) and K=4 (all pairs).

    If the fixture file does not exist, tests using this fixture are skipped.
    """
    import os
    from pathlib import Path

    fixture_dir = Path(
        os.environ.get("MFE_FIXTURE_DIR", "")
    ) if os.environ.get("MFE_FIXTURE_DIR") else (
        Path(__file__).resolve().parent.parent / "fixtures" / "distributions"
    )
    path = fixture_dir / "composite_likelihood.npy"
    if not path.exists():
        pytest.skip(f"Fixture file not found: {path}")
    return np.load(path, allow_pickle=True).item()


# ---------------------------------------------------------------------------
# Helper: Hand-compute composite likelihood for a single pair
# ---------------------------------------------------------------------------


def _hand_compute_single_pair_square(
    S: np.ndarray, data: np.ndarray, i: int, j: int, q: int,
) -> float:
    """Hand-compute one pair's contribution (square data path).

    Ref: composite_likelihood.m:26-36
    ll_k = 0.5 * (likConst + log(det) + (s22*x11 - 2*s12*x12 + s11*x22)/det) / q
    """
    s11 = S[i, i]
    s12 = S[i, j]
    s22 = S[j, j]
    det_val = s11 * s22 - s12 * s12
    x11 = data[i, i]
    x12 = data[i, j]
    x22 = data[j, j]
    return 0.5 * (
        LIK_CONST + math.log(det_val) + (s22 * x11 - 2 * s12 * x12 + s11 * x22) / det_val
    ) / q


def _hand_compute_single_pair_vector(
    S: np.ndarray, data: np.ndarray, i: int, j: int, q: int,
) -> float:
    """Hand-compute one pair's contribution (vector data path).

    Ref: composite_likelihood.m:39-49
    """
    s11 = S[i, i]
    s12 = S[i, j]
    s22 = S[j, j]
    det_val = s11 * s22 - s12 * s12
    x11 = data[i] * data[i]
    x12 = data[i] * data[j]
    x22 = data[j] * data[j]
    return 0.5 * (
        LIK_CONST + math.log(det_val) + (s22 * x11 - 2 * s12 * x12 + s11 * x22) / det_val
    ) / q


# =========================================================================
# Phase 1: likConst Validation
# =========================================================================


class TestLikConst:
    """Verify the hardcoded constant used in composite_likelihood."""

    def test_composite_likelihood_lik_const(self) -> None:
        """likConst ≈ 2 * log(2 * pi) within machine precision."""
        expected = 2.0 * np.log(2.0 * np.pi)
        npt.assert_allclose(LIK_CONST, expected, atol=1e-14, rtol=0)

    def test_lik_const_is_positive(self) -> None:
        """Verify the constant is positive (absolute value of -2*log(2*pi))."""
        assert LIK_CONST > 0.0


# =========================================================================
# Phase 2: Square Data Tests
# =========================================================================


class TestSquareData:
    """Tests for the square (K × K) data path.

    Ref: composite_likelihood.m:25-37 — if m==n branch
    """

    def test_composite_likelihood_square_basic(
        self,
        small_corr_3: np.ndarray,
        all_pairs_3: np.ndarray,
    ) -> None:
        """K=3, S=corr matrix, data=K×K outer product, all pairs → scalar."""
        rng = np.random.default_rng(99)
        x = rng.standard_normal(3)
        data_sq = np.outer(x, x)
        result = composite_likelihood(small_corr_3, data_sq, all_pairs_3)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_composite_likelihood_square_known_values(self) -> None:
        """Hand-computed K=2, single pair (0,1) on identity S."""
        # S = I2, data = [[0.25, -0.15], [-0.15, 0.09]]
        S = np.eye(2, dtype=np.float64)
        data = np.array([[0.25, -0.15], [-0.15, 0.09]], dtype=np.float64)
        indices = np.array([[0, 1]], dtype=np.int64)

        # Hand computation:
        # s11=1, s12=0, s22=1, det=1
        # x11=0.25, x12=-0.15, x22=0.09
        # ll = 0.5 * (likConst + log(1) + (1*0.25 - 2*0*(-0.15) + 1*0.09)/1) / 1
        #    = 0.5 * (3.67575413281869 + 0 + 0.34) / 1
        #    = 0.5 * 4.01575413281869
        #    = 2.007877066409345
        expected = 0.5 * (LIK_CONST + math.log(1.0) + (0.25 + 0.09))
        result = composite_likelihood(S, data, indices)
        npt.assert_allclose(result, expected, atol=1e-14, rtol=0)

    def test_composite_likelihood_square_identity_S(self) -> None:
        """S=I_K (zero off-diagonal correlation) simplifies formula.

        When S=I, s12=0, det=1, so:
        ll_k = 0.5 * (likConst + 0 + (x11 + x22)) / q
        """
        K = 3
        S = np.eye(K, dtype=np.float64)
        rng = np.random.default_rng(123)
        x = rng.standard_normal(K)
        data = np.outer(x, x)
        indices = np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int64)

        # Hand-compute expected
        expected = 0.0
        q = len(indices)
        for pair in indices:
            i, j = pair
            expected += 0.5 * (LIK_CONST + data[i, i] + data[j, j]) / q

        result = composite_likelihood(S, data, indices)
        npt.assert_allclose(result, expected, atol=1e-14, rtol=0)

    def test_composite_likelihood_square_subset_indices(
        self,
        small_corr_3: np.ndarray,
    ) -> None:
        """Using only a subset of pairs (not all K*(K-1)/2)."""
        rng = np.random.default_rng(77)
        x = rng.standard_normal(3)
        data = np.outer(x, x)
        # Only 2 out of 3 possible pairs
        indices_subset = np.array([[0, 1], [1, 2]], dtype=np.int64)

        result = composite_likelihood(small_corr_3, data, indices_subset)
        assert isinstance(result, float)
        assert np.isfinite(result)

        # Verify hand-computation matches
        expected = 0.0
        q = len(indices_subset)
        for pair in indices_subset:
            expected += _hand_compute_single_pair_square(
                small_corr_3, data, pair[0], pair[1], q,
            )
        npt.assert_allclose(result, expected, atol=1e-14, rtol=0)

    @pytest.mark.parity
    def test_composite_likelihood_square_parity(self, fixture_data: dict) -> None:
        """Parity test: K=3 matrix data, all pairs — fixture case 1."""
        S = fixture_data["case1_S"]
        data = fixture_data["case1_data"]
        indices = fixture_data["case1_indices"]
        expected = fixture_data["case1_ll"]

        result = composite_likelihood(S, data, indices)
        npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)

    @pytest.mark.parity
    def test_composite_likelihood_square_k4_parity(self, fixture_data: dict) -> None:
        """Parity test: K=4 matrix data, all 6 pairs — fixture case 3."""
        S = fixture_data["case3_S"]
        data = fixture_data["case3_data"]
        indices = fixture_data["case3_indices"]
        expected = fixture_data["case3_ll"]

        result = composite_likelihood(S, data, indices)
        npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)

    @pytest.mark.parity
    def test_composite_likelihood_square_single_pair_parity(
        self, fixture_data: dict,
    ) -> None:
        """Parity test: K=3 matrix data, single pair — fixture case 5."""
        S = fixture_data["case5_S"]
        data = fixture_data["case5_data"]
        indices = fixture_data["case5_indices"]
        expected = fixture_data["case5_ll"]

        result = composite_likelihood(S, data, indices)
        npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)


# =========================================================================
# Phase 3: Vector Data Tests
# =========================================================================


class TestVectorData:
    """Tests for the vector (K,) data path.

    Ref: composite_likelihood.m:38-50 — else (m != n) branch
    """

    def test_composite_likelihood_vector_basic(
        self,
        small_corr_3: np.ndarray,
        all_pairs_3: np.ndarray,
    ) -> None:
        """K=3, S=corr matrix, data=K-length vector, all pairs → scalar."""
        rng = np.random.default_rng(99)
        x = rng.standard_normal(3)
        result = composite_likelihood(small_corr_3, x, all_pairs_3)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_composite_likelihood_vector_known_values(self) -> None:
        """Hand-computed K=2, single pair (0,1) on identity S with vector."""
        S = np.eye(2, dtype=np.float64)
        data = np.array([0.5, -0.3], dtype=np.float64)
        indices = np.array([[0, 1]], dtype=np.int64)

        # Hand computation (vector path):
        # s11=1, s12=0, s22=1, det=1
        # x11=0.5*0.5=0.25, x12=0.5*(-0.3)=-0.15, x22=(-0.3)*(-0.3)=0.09
        # ll = 0.5 * (likConst + log(1) + (1*0.25 - 2*0*(-0.15) + 1*0.09)/1) / 1
        #    = 0.5 * (3.67575... + 0.34)
        expected = 0.5 * (LIK_CONST + 0.25 + 0.09)
        result = composite_likelihood(S, data, indices)
        npt.assert_allclose(result, expected, atol=1e-14, rtol=0)

    def test_composite_likelihood_vector_identity_S(self) -> None:
        """S=I with vector data: formula simplifies to 0.5*(likConst + xi^2+xj^2)/q."""
        K = 3
        S = np.eye(K, dtype=np.float64)
        rng = np.random.default_rng(456)
        x = rng.standard_normal(K)
        indices = np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int64)

        expected = 0.0
        q = len(indices)
        for pair in indices:
            i, j = pair
            expected += 0.5 * (LIK_CONST + x[i] ** 2 + x[j] ** 2) / q

        result = composite_likelihood(S, x, indices)
        npt.assert_allclose(result, expected, atol=1e-14, rtol=0)

    def test_composite_likelihood_vector_matches_square(
        self,
        small_corr_3: np.ndarray,
        all_pairs_3: np.ndarray,
    ) -> None:
        """Vector data and its outer product (square data) yield same result.

        When data_matrix = x @ x.T, both code paths should produce identical
        composite likelihood values.
        """
        rng = np.random.default_rng(88)
        x = rng.standard_normal(3)
        data_sq = np.outer(x, x)

        ll_vec = composite_likelihood(small_corr_3, x, all_pairs_3)
        ll_sq = composite_likelihood(small_corr_3, data_sq, all_pairs_3)
        npt.assert_allclose(ll_vec, ll_sq, atol=1e-14, rtol=0)

    @pytest.mark.parity
    def test_composite_likelihood_vector_parity(self, fixture_data: dict) -> None:
        """Parity test: K=3 vector data, all pairs — fixture case 2."""
        S = fixture_data["case2_S"]
        data = fixture_data["case2_data"]
        indices = fixture_data["case2_indices"]
        expected = fixture_data["case2_ll"]

        result = composite_likelihood(S, data, indices)
        npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)

    @pytest.mark.parity
    def test_composite_likelihood_vector_k4_parity(self, fixture_data: dict) -> None:
        """Parity test: K=4 vector data, all 6 pairs — fixture case 4."""
        S = fixture_data["case4_S"]
        data = fixture_data["case4_data"]
        indices = fixture_data["case4_indices"]
        expected = fixture_data["case4_ll"]

        result = composite_likelihood(S, data, indices)
        npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)

    @pytest.mark.parity
    def test_composite_likelihood_vector_single_pair_parity(
        self, fixture_data: dict,
    ) -> None:
        """Parity test: K=3 vector data, single pair — fixture case 6."""
        S = fixture_data["case6_S"]
        data = fixture_data["case6_data"]
        indices = fixture_data["case6_indices"]
        expected = fixture_data["case6_ll"]

        result = composite_likelihood(S, data, indices)
        npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)

    @pytest.mark.parity
    def test_composite_likelihood_matrix_vector_parity_match(
        self, fixture_data: dict,
    ) -> None:
        """Fixture cases 1 & 2 (matrix vs vector, same data) yield same ll."""
        ll_matrix = fixture_data["case1_ll"]
        ll_vector = fixture_data["case2_ll"]
        npt.assert_allclose(ll_matrix, ll_vector, atol=1e-14, rtol=0)

    @pytest.mark.parity
    def test_composite_likelihood_k4_matrix_vector_parity_match(
        self, fixture_data: dict,
    ) -> None:
        """Fixture cases 3 & 4 (matrix vs vector, K=4) yield same ll."""
        ll_matrix = fixture_data["case3_ll"]
        ll_vector = fixture_data["case4_ll"]
        npt.assert_allclose(ll_matrix, ll_vector, atol=1e-14, rtol=0)


# =========================================================================
# Phase 4: Numba JIT Verification
# =========================================================================


class TestNumbaJIT:
    """Verify that Numba JIT compilation is active and performant."""

    def test_composite_likelihood_numba_compiled_square(self) -> None:
        """_composite_likelihood_core_square is a Numba Dispatcher (nopython)."""
        assert isinstance(
            _composite_likelihood_core_square,
            numba.core.dispatcher.Dispatcher,
        ), (
            f"Expected Numba Dispatcher, got {type(_composite_likelihood_core_square)}"
        )

    def test_composite_likelihood_numba_compiled_vector(self) -> None:
        """_composite_likelihood_core_vector is a Numba Dispatcher (nopython)."""
        assert isinstance(
            _composite_likelihood_core_vector,
            numba.core.dispatcher.Dispatcher,
        ), (
            f"Expected Numba Dispatcher, got {type(_composite_likelihood_core_vector)}"
        )

    @pytest.mark.slow
    def test_composite_likelihood_performance_not_degraded(self) -> None:
        """K=50 with all pairs completes in under 2 seconds.

        This verifies that the Numba JIT compilation provides acceptable
        performance for moderate-sized problems.  The 2-second threshold
        accounts for first-call JIT compilation overhead.
        """
        K = 50
        rng = np.random.default_rng(42)
        # Construct a proper covariance matrix: S = A A^T + I (ensure PD)
        A = rng.standard_normal((K, K)) * 0.1
        S = A @ A.T + np.eye(K, dtype=np.float64)
        data = rng.standard_normal(K)

        # All unique pairs
        pairs = []
        for i in range(K):
            for j in range(i + 1, K):
                pairs.append([i, j])
        indices = np.array(pairs, dtype=np.int64)

        # Warm up JIT if not already cached
        _ = composite_likelihood(S, data, indices)

        # Time the actual computation
        t_start = time.perf_counter()
        result = composite_likelihood(S, data, indices)
        t_elapsed = time.perf_counter() - t_start

        assert isinstance(result, float)
        assert np.isfinite(result)
        assert t_elapsed < 2.0, (
            f"K=50 composite_likelihood took {t_elapsed:.3f}s (expected < 2.0s)"
        )

    def test_composite_likelihood_repeated_calls(self) -> None:
        """Repeated calls produce identical results (cached compilation)."""
        S = np.array([[1.0, 0.4], [0.4, 1.0]], dtype=np.float64)
        data = np.array([0.7, -0.3], dtype=np.float64)
        indices = np.array([[0, 1]], dtype=np.int64)

        results = [composite_likelihood(S, data, indices) for _ in range(5)]

        # All 5 calls must return exactly the same value
        for r in results[1:]:
            assert r == results[0], (
                f"Repeated call mismatch: {r} != {results[0]}"
            )

    def test_composite_likelihood_cached_is_faster(self) -> None:
        """Second call should be faster than first (JIT caching).

        We cannot guarantee this in all environments (e.g., if the cache
        is already warm), so this test verifies functional correctness
        rather than strict timing.
        """
        K = 10
        rng = np.random.default_rng(55)
        S = np.eye(K, dtype=np.float64) + rng.standard_normal((K, K)) * 0.05
        S = (S + S.T) / 2  # symmetrize
        np.fill_diagonal(S, 1.0)
        data = rng.standard_normal(K)
        pairs = [[i, j] for i in range(K) for j in range(i + 1, K)]
        indices = np.array(pairs, dtype=np.int64)

        # Run multiple times — results must all match
        r1 = composite_likelihood(S, data, indices)
        r2 = composite_likelihood(S, data, indices)
        r3 = composite_likelihood(S, data, indices)
        npt.assert_allclose(r1, r2, atol=0, rtol=0)
        npt.assert_allclose(r2, r3, atol=0, rtol=0)


# =========================================================================
# Phase 5: Edge Cases
# =========================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_composite_likelihood_k2_single_pair(self) -> None:
        """Minimal K=2, single pair — simplest valid call."""
        S = np.array([[1.2, 0.3], [0.3, 0.8]], dtype=np.float64)
        data = np.array([1.0, -0.5], dtype=np.float64)
        indices = np.array([[0, 1]], dtype=np.int64)

        result = composite_likelihood(S, data, indices)
        assert isinstance(result, float)
        assert np.isfinite(result)

        # Hand-compute expected
        expected = _hand_compute_single_pair_vector(S, data, 0, 1, 1)
        npt.assert_allclose(result, expected, atol=1e-14, rtol=0)

    def test_composite_likelihood_large_k(self) -> None:
        """K=20 with all pairs — no numeric overflow."""
        K = 20
        rng = np.random.default_rng(789)
        # Generate proper SPD covariance
        A = rng.standard_normal((K, K)) * 0.2
        S = A @ A.T + np.eye(K, dtype=np.float64)
        data = rng.standard_normal(K)

        pairs = [[i, j] for i in range(K) for j in range(i + 1, K)]
        indices = np.array(pairs, dtype=np.int64)

        result = composite_likelihood(S, data, indices)
        assert isinstance(result, float)
        assert np.isfinite(result), f"Result is not finite: {result}"

    def test_composite_likelihood_high_correlation(self) -> None:
        """S entries near ±1 (e.g., 0.99) — test numerical stability.

        The formula contains 1/(s11*s22 - s12^2). When s12 is close to
        sqrt(s11*s22), the determinant is close to zero, testing stability.
        """
        # Correlation matrix with high off-diagonal
        S = np.array([[1.0, 0.99], [0.99, 1.0]], dtype=np.float64)
        data = np.array([0.5, 0.4], dtype=np.float64)
        indices = np.array([[0, 1]], dtype=np.int64)

        result = composite_likelihood(S, data, indices)
        assert isinstance(result, float)
        assert np.isfinite(result), (
            f"High correlation (0.99) produced non-finite result: {result}"
        )

        # Hand-compute to verify
        expected = _hand_compute_single_pair_vector(S, data, 0, 1, 1)
        npt.assert_allclose(result, expected, atol=1e-12, rtol=0)

    def test_composite_likelihood_negative_high_correlation(self) -> None:
        """S entries near -1 (e.g., -0.98) — test numerical stability."""
        S = np.array([[1.0, -0.98], [-0.98, 1.0]], dtype=np.float64)
        data = np.array([0.5, -0.4], dtype=np.float64)
        indices = np.array([[0, 1]], dtype=np.int64)

        result = composite_likelihood(S, data, indices)
        assert isinstance(result, float)
        assert np.isfinite(result)

        expected = _hand_compute_single_pair_vector(S, data, 0, 1, 1)
        npt.assert_allclose(result, expected, atol=1e-12, rtol=0)

    def test_composite_likelihood_returns_scalar(self) -> None:
        """Output is always a Python float scalar, never an ndarray."""
        S = np.eye(3, dtype=np.float64)
        data = np.zeros(3, dtype=np.float64)
        indices = np.array([[0, 1], [0, 2]], dtype=np.int64)

        result = composite_likelihood(S, data, indices)
        assert isinstance(result, float), (
            f"Expected float, got {type(result)}"
        )
        assert not isinstance(result, np.ndarray)

    def test_composite_likelihood_zero_data(self) -> None:
        """All-zero data vector still produces a valid result.

        When data is zero, x11=x12=x22=0 for vector path, so the quadratic
        form vanishes: ll_k = 0.5 * (likConst + log(det)) / q.
        """
        S = np.array([[1.0, 0.3], [0.3, 1.0]], dtype=np.float64)
        data = np.zeros(2, dtype=np.float64)
        indices = np.array([[0, 1]], dtype=np.int64)

        result = composite_likelihood(S, data, indices)
        det_val = 1.0 - 0.3 ** 2
        expected = 0.5 * (LIK_CONST + math.log(det_val))
        npt.assert_allclose(result, expected, atol=1e-14, rtol=0)

    def test_composite_likelihood_empty_indices(self) -> None:
        """Empty indices array returns 0.0."""
        S = np.eye(3, dtype=np.float64)
        data = np.ones(3, dtype=np.float64)
        indices = np.zeros((0, 2), dtype=np.int64)

        result = composite_likelihood(S, data, indices)
        assert result == 0.0

    def test_composite_likelihood_row_vector_data(self) -> None:
        """A (1, K) row vector is squeezed to (K,) and processed."""
        S = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float64)
        data_vec = np.array([0.3, -0.2], dtype=np.float64)
        data_row = data_vec.reshape(1, -1)
        indices = np.array([[0, 1]], dtype=np.int64)

        ll_vec = composite_likelihood(S, data_vec, indices)
        ll_row = composite_likelihood(S, data_row, indices)
        npt.assert_allclose(ll_vec, ll_row, atol=1e-14, rtol=0)

    def test_composite_likelihood_col_vector_data(self) -> None:
        """A (K, 1) column vector is squeezed to (K,) and processed."""
        S = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float64)
        data_vec = np.array([0.3, -0.2], dtype=np.float64)
        data_col = data_vec.reshape(-1, 1)
        indices = np.array([[0, 1]], dtype=np.int64)

        ll_vec = composite_likelihood(S, data_vec, indices)
        ll_col = composite_likelihood(S, data_col, indices)
        npt.assert_allclose(ll_vec, ll_col, atol=1e-14, rtol=0)


# =========================================================================
# Phase 6: Input Validation Tests
# =========================================================================


class TestInputValidation:
    """Test that invalid inputs raise appropriate errors."""

    def test_composite_likelihood_non_square_S_raises(self) -> None:
        """Non-square S raises ValueError."""
        S = np.ones((3, 2), dtype=np.float64)
        data = np.zeros(3, dtype=np.float64)
        indices = np.array([[0, 1]], dtype=np.int64)

        with pytest.raises(ValueError, match="2-D square matrix"):
            composite_likelihood(S, data, indices)

    def test_composite_likelihood_1d_S_raises(self) -> None:
        """1-D S raises ValueError."""
        S = np.ones(3, dtype=np.float64)
        data = np.zeros(3, dtype=np.float64)
        indices = np.array([[0, 1]], dtype=np.int64)

        with pytest.raises(ValueError, match="2-D square matrix"):
            composite_likelihood(S, data, indices)

    def test_composite_likelihood_bad_indices_shape_raises(self) -> None:
        """Indices with ≠2 columns raises ValueError."""
        S = np.eye(3, dtype=np.float64)
        data = np.zeros(3, dtype=np.float64)
        indices = np.array([[0, 1, 2]], dtype=np.int64)

        with pytest.raises(ValueError, match="2 columns"):
            composite_likelihood(S, data, indices)

    def test_composite_likelihood_1d_indices_raises(self) -> None:
        """1-D indices raises ValueError."""
        S = np.eye(3, dtype=np.float64)
        data = np.zeros(3, dtype=np.float64)
        indices = np.array([0, 1], dtype=np.int64)

        with pytest.raises(ValueError, match="2-D array"):
            composite_likelihood(S, data, indices)

    def test_composite_likelihood_non_square_non_vector_data_raises(self) -> None:
        """Non-square, non-vector 2D data raises ValueError."""
        S = np.eye(3, dtype=np.float64)
        data = np.ones((3, 2), dtype=np.float64)
        indices = np.array([[0, 1]], dtype=np.int64)

        with pytest.raises(ValueError, match="square matrix"):
            composite_likelihood(S, data, indices)

    def test_composite_likelihood_3d_data_raises(self) -> None:
        """3-D data array raises ValueError."""
        S = np.eye(3, dtype=np.float64)
        data = np.ones((3, 3, 3), dtype=np.float64)
        indices = np.array([[0, 1]], dtype=np.int64)

        with pytest.raises(ValueError, match="1-D or 2-D"):
            composite_likelihood(S, data, indices)


# =========================================================================
# Phase 7: Type Coercion Tests
# =========================================================================


class TestTypeCoercion:
    """Verify that input coercion works for various dtypes."""

    def test_composite_likelihood_int_data_coerced(self) -> None:
        """Integer data is coerced to float64 without error."""
        S = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float64)
        data = np.array([1, -1], dtype=np.int64)
        indices = np.array([[0, 1]], dtype=np.int64)

        result = composite_likelihood(S, data, indices)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_composite_likelihood_float32_coerced(self) -> None:
        """float32 data is coerced to float64 for Numba compatibility."""
        S = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float32)
        data = np.array([0.5, -0.3], dtype=np.float32)
        indices = np.array([[0, 1]], dtype=np.int64)

        result = composite_likelihood(S, data, indices)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_composite_likelihood_list_inputs(self) -> None:
        """Python list inputs are converted to numpy arrays."""
        S = [[1.0, 0.5], [0.5, 1.0]]
        data = [0.5, -0.3]
        indices = [[0, 1]]

        result = composite_likelihood(S, data, indices)
        assert isinstance(result, float)
        assert np.isfinite(result)


# =========================================================================
# Phase 8: Comprehensive Parity Tests (all 6 fixture cases)
# =========================================================================


class TestComprehensiveParity:
    """Systematic fixture parity for all 6 reference test cases."""

    @pytest.mark.parity
    @pytest.mark.parametrize("case_num", [1, 2, 3, 4, 5, 6])
    def test_composite_likelihood_fixture_case(
        self, fixture_data: dict, case_num: int,
    ) -> None:
        """Parametrized parity test across all 6 fixture cases.

        Cases:
        1: K=3, matrix data (m==n), all pairs (q=3)
        2: K=3, vector data (m!=n), all pairs (q=3)
        3: K=4, matrix data (m==n), all pairs (q=6)
        4: K=4, vector data (m!=n), all pairs (q=6)
        5: K=3, matrix data (m==n), single pair (q=1)
        6: K=3, vector data (m!=n), single pair (q=1)
        """
        prefix = f"case{case_num}"
        S = fixture_data[f"{prefix}_S"]
        data = fixture_data[f"{prefix}_data"]
        indices = fixture_data[f"{prefix}_indices"]
        expected = fixture_data[f"{prefix}_ll"]
        data_type = fixture_data[f"{prefix}_data_type"]
        K = fixture_data[f"{prefix}_K"]
        q = fixture_data[f"{prefix}_q"]

        result = composite_likelihood(S, data, indices)
        npt.assert_allclose(
            result, expected, atol=ATOL, rtol=RTOL,
            err_msg=(
                f"Case {case_num}: K={K}, data_type={data_type}, q={q}, "
                f"result={result}, expected={expected}"
            ),
        )
