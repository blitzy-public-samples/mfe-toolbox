"""
Pytest test suite for mfe_toolbox.timeseries.bkfilter — Baxter-King band-pass filter.

Tests cover:
  Phase 1: Input validation (invalid p, q < p, negative k)
  Phase 2: Output shape and decomposition identity
  Phase 3: Boundary point behaviour at first/last K observations
  Phase 4: Special cases (equal p/q, infinite q low-pass, multivariate, default k)
  Phase 5: Numerical parity against MATLAB/Octave reference fixtures

Reference: timeseries/bkfilter.m — Kevin Sheppard, MFE Toolbox v4.0
"""

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.timeseries.bkfilter import bkfilter

# ---------------------------------------------------------------------------
# Tolerance constants per AAP Section 0.7.1
# ---------------------------------------------------------------------------
ATOL = 1e-6
RTOL = 1e-4


# ---------------------------------------------------------------------------
# Local test fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def bkfilter_fixtures(timeseries_fixture_dir):
    """Load the main bkfilter.npy fixture dict containing 5 MATLAB scenarios."""
    path = timeseries_fixture_dir / "bkfilter.npy"
    if not path.exists():
        pytest.skip("bkfilter.npy fixture not found")
    return np.load(str(path), allow_pickle=True).item()


@pytest.fixture(scope="module")
def bk_trend_fixture(timeseries_fixture_dir):
    """Load the standalone bkfilter_bk_trend.npy trend reference vector."""
    path = timeseries_fixture_dir / "bkfilter_bk_trend.npy"
    if not path.exists():
        pytest.skip("bkfilter_bk_trend.npy fixture not found")
    return np.load(str(path))


@pytest.fixture(scope="module")
def bk_cyclic_fixture(timeseries_fixture_dir):
    """Load the standalone bkfilter_bk_cyclic.npy cyclic reference vector."""
    path = timeseries_fixture_dir / "bkfilter_bk_cyclic.npy"
    if not path.exists():
        pytest.skip("bkfilter_bk_cyclic.npy fixture not found")
    return np.load(str(path))


@pytest.fixture()
def sample_univariate(rng):
    """Generate a univariate random-walk series of length 200 (seed=42)."""
    return np.cumsum(rng.standard_normal(200))


@pytest.fixture()
def sample_multivariate(rng):
    """Generate a T=200, K=3 multivariate random-walk matrix (seed=42)."""
    return np.cumsum(rng.standard_normal((200, 3)), axis=0)


# ===================================================================
# Phase 1 — Input Validation
# ===================================================================
class TestBkfilterInputValidation:
    """Tests that invalid parameters raise ValueError."""

    def test_bkfilter_invalid_p_zero(self, sample_univariate):
        """p <= 0 must raise ValueError."""
        with pytest.raises(ValueError, match="(?i)p.*positive"):
            bkfilter(sample_univariate, 0, 32, 12)

    def test_bkfilter_invalid_p_negative(self, sample_univariate):
        """Negative p must raise ValueError."""
        with pytest.raises(ValueError, match="(?i)p.*positive"):
            bkfilter(sample_univariate, -5, 32, 12)

    def test_bkfilter_q_less_than_p(self, sample_univariate):
        """q < p must raise ValueError."""
        with pytest.raises(ValueError, match="(?i)q.*positive.*q.*>=.*p|q.*>=.*p"):
            bkfilter(sample_univariate, 10, 5, 12)

    def test_bkfilter_negative_k(self, sample_univariate):
        """k <= 0 must raise ValueError."""
        with pytest.raises(ValueError, match="(?i)k.*positive"):
            bkfilter(sample_univariate, 6, 32, 0)

    def test_bkfilter_negative_k_minus_one(self, sample_univariate):
        """k = -1 must raise ValueError."""
        with pytest.raises(ValueError, match="(?i)k.*positive"):
            bkfilter(sample_univariate, 6, 32, -1)


# ===================================================================
# Phase 2 — Output Shape and Structure
# ===================================================================
class TestBkfilterOutputShape:
    """Tests that outputs have correct shape and structure."""

    def test_bkfilter_returns_three(self, sample_univariate):
        """bkfilter must return a 3-tuple (trend, cyclic, noise)."""
        result = bkfilter(sample_univariate, 6, 32, 12)
        assert isinstance(result, tuple), "Return value must be a tuple"
        assert len(result) == 3, "Must return exactly 3 components"

    def test_bkfilter_output_shapes_univariate(self, sample_univariate):
        """All outputs must be (T, 1) for a univariate 1-D input of length T."""
        T = len(sample_univariate)
        trend, cyclic, noise = bkfilter(sample_univariate, 6, 32, 12)
        assert trend.shape == (T, 1), f"trend shape {trend.shape} != ({T}, 1)"
        assert cyclic.shape == (T, 1), f"cyclic shape {cyclic.shape} != ({T}, 1)"
        assert noise.shape == (T, 1), f"noise shape {noise.shape} != ({T}, 1)"

    def test_bkfilter_output_shapes_multivariate(self, sample_multivariate):
        """All outputs must be (T, K) for a T×K input matrix."""
        T, K = sample_multivariate.shape
        trend, cyclic, noise = bkfilter(sample_multivariate, 6, 32, 12)
        assert trend.shape == (T, K), f"trend shape {trend.shape} != ({T}, {K})"
        assert cyclic.shape == (T, K), f"cyclic shape {cyclic.shape} != ({T}, {K})"
        assert noise.shape == (T, K), f"noise shape {noise.shape} != ({T}, {K})"

    def test_bkfilter_decomposition_identity(self, sample_univariate):
        """Fundamental identity: y == trend + cyclic + noise within tolerance.

        The Baxter-King filter enforces this exactly by construction: boundary
        points have trend=y, cyclic=0, noise=0, and interior points compute
        noise = y - trend - cyclic.
        """
        y = sample_univariate
        trend, cyclic, noise = bkfilter(y, 6, 32, 12)
        # The function reshapes 1-D to (T, 1); match shape for comparison
        y_col = y.reshape(-1, 1)
        npt.assert_allclose(trend + cyclic + noise, y_col, atol=ATOL, rtol=RTOL)

    def test_bkfilter_output_dtype(self, sample_univariate):
        """All outputs must be float64 numpy arrays."""
        trend, cyclic, noise = bkfilter(sample_univariate, 6, 32, 12)
        for name, arr in [("trend", trend), ("cyclic", cyclic), ("noise", noise)]:
            assert isinstance(arr, np.ndarray), f"{name} must be ndarray"
            assert arr.dtype == np.float64, f"{name} dtype must be float64"


# ===================================================================
# Phase 3 — Boundary Point Tests
# ===================================================================
class TestBkfilterBoundaryPoints:
    """Tests that first/last K observations are handled correctly.

    By construction, the Baxter-King filter discards the first and last K
    observations from the filter output:
      - trend[:K] = y[:K]  and  trend[-K:] = y[-K:]
      - cyclic[:K] = 0     and  cyclic[-K:] = 0
      - noise[:K] = 0      and  noise[-K:] = 0
    """

    def test_bkfilter_first_k_points_trend(self, sample_univariate):
        """First K trend values must equal original data y[:K]."""
        k = 12
        y = sample_univariate
        trend, _, _ = bkfilter(y, 6, 32, k)
        npt.assert_allclose(trend[:k, 0], y[:k], atol=ATOL, rtol=RTOL)

    def test_bkfilter_last_k_points_trend(self, sample_univariate):
        """Last K trend values must equal original data y[-K:]."""
        k = 12
        y = sample_univariate
        trend, _, _ = bkfilter(y, 6, 32, k)
        npt.assert_allclose(trend[-k:, 0], y[-k:], atol=ATOL, rtol=RTOL)

    def test_bkfilter_first_k_cyclic_zero(self, sample_univariate):
        """First K cyclic values must be exactly zero."""
        k = 12
        _, cyclic, _ = bkfilter(sample_univariate, 6, 32, k)
        npt.assert_allclose(cyclic[:k, 0], np.zeros(k), atol=ATOL, rtol=RTOL)

    def test_bkfilter_last_k_cyclic_zero(self, sample_univariate):
        """Last K cyclic values must be exactly zero."""
        k = 12
        _, cyclic, _ = bkfilter(sample_univariate, 6, 32, k)
        npt.assert_allclose(cyclic[-k:, 0], np.zeros(k), atol=ATOL, rtol=RTOL)

    def test_bkfilter_first_k_noise_zero(self, sample_univariate):
        """First K noise values must be exactly zero."""
        k = 12
        _, _, noise = bkfilter(sample_univariate, 6, 32, k)
        npt.assert_allclose(noise[:k, 0], np.zeros(k), atol=ATOL, rtol=RTOL)

    def test_bkfilter_last_k_noise_zero(self, sample_univariate):
        """Last K noise values must be exactly zero."""
        k = 12
        _, _, noise = bkfilter(sample_univariate, 6, 32, k)
        npt.assert_allclose(noise[-k:, 0], np.zeros(k), atol=ATOL, rtol=RTOL)


# ===================================================================
# Phase 4 — Special Cases
# ===================================================================
class TestBkfilterSpecialCases:
    """Tests for special configurations of the BK filter."""

    def test_bkfilter_equal_pq_no_cyclic(self, sample_univariate):
        """When q == p the band-pass filter b = bp - bq = 0 → cyclic ≡ 0.

        Ref: bkfilter.m — if p == q the two weight vectors are identical,
        so their difference is the zero vector.
        """
        _, cyclic, _ = bkfilter(sample_univariate, 10, 10, 12)
        npt.assert_allclose(cyclic, np.zeros_like(cyclic), atol=ATOL, rtol=RTOL)

    def test_bkfilter_inf_q_lowpass(self, sample_univariate):
        """When q = inf the filter degenerates to a low-pass filter.

        With q = inf, the lower-frequency angular cutoff fq = 2π/q → 0,
        making the low-frequency weight vector bq a uniform 1/(2K+1)
        moving average after mean-preserving normalisation.  The trend
        should therefore approximate a symmetric moving average of the
        data.
        """
        k = 12
        trend, _, _ = bkfilter(sample_univariate, 6, np.inf, k)
        # The trend should be finite and well-defined
        assert not np.any(np.isnan(trend)), "Low-pass trend must not contain NaN"
        assert not np.any(np.isinf(trend)), "Low-pass trend must not contain Inf"
        # Boundary: first/last K points still equal original data
        npt.assert_allclose(trend[:k, 0], sample_univariate[:k], atol=ATOL, rtol=RTOL)
        npt.assert_allclose(trend[-k:, 0], sample_univariate[-k:], atol=ATOL, rtol=RTOL)

    def test_bkfilter_multivariate(self, sample_multivariate):
        """Each column of a T×K matrix must be filtered independently.

        Verify that filtering the full matrix produces the same results
        as filtering each column separately.
        """
        T, K = sample_multivariate.shape
        trend_full, cyclic_full, noise_full = bkfilter(sample_multivariate, 6, 32, 12)

        for col in range(K):
            trend_col, cyclic_col, noise_col = bkfilter(
                sample_multivariate[:, col], 6, 32, 12
            )
            npt.assert_allclose(
                trend_col.ravel(), trend_full[:, col],
                atol=ATOL, rtol=RTOL,
                err_msg=f"Trend mismatch on column {col}",
            )
            npt.assert_allclose(
                cyclic_col.ravel(), cyclic_full[:, col],
                atol=ATOL, rtol=RTOL,
                err_msg=f"Cyclic mismatch on column {col}",
            )
            npt.assert_allclose(
                noise_col.ravel(), noise_full[:, col],
                atol=ATOL, rtol=RTOL,
                err_msg=f"Noise mismatch on column {col}",
            )

    def test_bkfilter_default_k(self, sample_univariate):
        """Default k=12 produces the same result as explicitly passing k=12."""
        trend_default, cyclic_default, noise_default = bkfilter(
            sample_univariate, 6, 32
        )
        trend_explicit, cyclic_explicit, noise_explicit = bkfilter(
            sample_univariate, 6, 32, 12
        )
        npt.assert_allclose(trend_default, trend_explicit, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(cyclic_default, cyclic_explicit, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(noise_default, noise_explicit, atol=ATOL, rtol=RTOL)

    def test_bkfilter_multivariate_decomposition_identity(self, sample_multivariate):
        """Decomposition identity y == trend + cyclic + noise for multivariate."""
        trend, cyclic, noise = bkfilter(sample_multivariate, 6, 32, 12)
        npt.assert_allclose(
            trend + cyclic + noise, sample_multivariate, atol=ATOL, rtol=RTOL
        )


# ===================================================================
# Phase 5 — Parity Tests against MATLAB/Octave Fixtures
# ===================================================================
class TestBkfilterParity:
    """Numerical parity tests comparing Python outputs with MATLAB reference.

    Fixture file: tests/fixtures/timeseries/bkfilter.npy
    Scenarios:
      1: standard_quarterly  (T=200, p=6, q=32, k=12)
      2: monthly             (T=500, p=18, q=96, k=12)
      3: low_pass_only       (T=200, p=40, q=40, k=12)  — equal p/q
      4: multivariate        (T=200, K=3, p=6, q=32, k=12)
      5: custom_k            (T=200, p=6, q=32, k=6)
    """

    @pytest.mark.parity
    def test_bkfilter_parity_quarterly(self, bkfilter_fixtures):
        """Scenario 1: standard quarterly filter (p=6, q=32, k=12) on T=200."""
        data = bkfilter_fixtures
        y = data["scenario_1_y"]
        p = int(data["scenario_1_p"])
        q = int(data["scenario_1_q"])
        k = int(data["scenario_1_k"])
        expected_trend = data["scenario_1_trend"]
        expected_cyclic = data["scenario_1_cyclic"]
        expected_noise = data["scenario_1_noise"]

        trend, cyclic, noise = bkfilter(y, p, q, k)

        npt.assert_allclose(trend, expected_trend, atol=ATOL, rtol=RTOL,
                            err_msg="Quarterly trend parity failed")
        npt.assert_allclose(cyclic, expected_cyclic, atol=ATOL, rtol=RTOL,
                            err_msg="Quarterly cyclic parity failed")
        npt.assert_allclose(noise, expected_noise, atol=ATOL, rtol=RTOL,
                            err_msg="Quarterly noise parity failed")

    @pytest.mark.parity
    def test_bkfilter_parity_monthly(self, bkfilter_fixtures):
        """Scenario 2: monthly filter (p=18, q=96, k=12) on T=500."""
        data = bkfilter_fixtures
        y = data["scenario_2_y"]
        p = int(data["scenario_2_p"])
        q = int(data["scenario_2_q"])
        k = int(data["scenario_2_k"])
        expected_trend = data["scenario_2_trend"]
        expected_cyclic = data["scenario_2_cyclic"]
        expected_noise = data["scenario_2_noise"]

        trend, cyclic, noise = bkfilter(y, p, q, k)

        npt.assert_allclose(trend, expected_trend, atol=ATOL, rtol=RTOL,
                            err_msg="Monthly trend parity failed")
        npt.assert_allclose(cyclic, expected_cyclic, atol=ATOL, rtol=RTOL,
                            err_msg="Monthly cyclic parity failed")
        npt.assert_allclose(noise, expected_noise, atol=ATOL, rtol=RTOL,
                            err_msg="Monthly noise parity failed")

    @pytest.mark.parity
    def test_bkfilter_parity_equal_pq(self, bkfilter_fixtures):
        """Scenario 3: equal p=q=40 (low-pass only, cyclic=0)."""
        data = bkfilter_fixtures
        y = data["scenario_3_y"]
        p = int(data["scenario_3_p"])
        q = int(data["scenario_3_q"])
        k = int(data["scenario_3_k"])
        expected_trend = data["scenario_3_trend"]
        expected_cyclic = data["scenario_3_cyclic"]
        expected_noise = data["scenario_3_noise"]

        trend, cyclic, noise = bkfilter(y, p, q, k)

        npt.assert_allclose(trend, expected_trend, atol=ATOL, rtol=RTOL,
                            err_msg="Equal-pq trend parity failed")
        npt.assert_allclose(cyclic, expected_cyclic, atol=ATOL, rtol=RTOL,
                            err_msg="Equal-pq cyclic parity failed")
        npt.assert_allclose(noise, expected_noise, atol=ATOL, rtol=RTOL,
                            err_msg="Equal-pq noise parity failed")

    @pytest.mark.parity
    def test_bkfilter_parity_multivariate(self, bkfilter_fixtures):
        """Scenario 4: multivariate T=200×K=3 with quarterly settings."""
        data = bkfilter_fixtures
        y = data["scenario_4_y"]
        p = int(data["scenario_4_p"])
        q = int(data["scenario_4_q"])
        k = int(data["scenario_4_k"])
        expected_trend = data["scenario_4_trend"]
        expected_cyclic = data["scenario_4_cyclic"]
        expected_noise = data["scenario_4_noise"]

        trend, cyclic, noise = bkfilter(y, p, q, k)

        npt.assert_allclose(trend, expected_trend, atol=ATOL, rtol=RTOL,
                            err_msg="Multivariate trend parity failed")
        npt.assert_allclose(cyclic, expected_cyclic, atol=ATOL, rtol=RTOL,
                            err_msg="Multivariate cyclic parity failed")
        npt.assert_allclose(noise, expected_noise, atol=ATOL, rtol=RTOL,
                            err_msg="Multivariate noise parity failed")

    @pytest.mark.parity
    def test_bkfilter_parity_custom_k(self, bkfilter_fixtures):
        """Scenario 5: custom k=6 (shorter filter window)."""
        data = bkfilter_fixtures
        y = data["scenario_5_y"]
        p = int(data["scenario_5_p"])
        q = int(data["scenario_5_q"])
        k = int(data["scenario_5_k"])
        expected_trend = data["scenario_5_trend"]
        expected_cyclic = data["scenario_5_cyclic"]
        expected_noise = data["scenario_5_noise"]

        trend, cyclic, noise = bkfilter(y, p, q, k)

        npt.assert_allclose(trend, expected_trend, atol=ATOL, rtol=RTOL,
                            err_msg="Custom-k trend parity failed")
        npt.assert_allclose(cyclic, expected_cyclic, atol=ATOL, rtol=RTOL,
                            err_msg="Custom-k cyclic parity failed")
        npt.assert_allclose(noise, expected_noise, atol=ATOL, rtol=RTOL,
                            err_msg="Custom-k noise parity failed")
