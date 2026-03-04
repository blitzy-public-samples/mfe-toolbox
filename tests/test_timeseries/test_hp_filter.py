"""
Pytest tests for mfe_toolbox.timeseries.hp_filter — Hodrick-Prescott filter.

Migrated from timeseries/hp_filter.m (Kevin Sheppard, MFE Toolbox v4.0).
Tests cover:
  - Input validation (negative / zero / nonscalar lambda)
  - Output shapes (univariate T, multivariate T×K)
  - Decomposition identity: y == trend + cyclic
  - Smoothing property: larger lambda → smoother trend
  - Extreme lambda behaviour (very small, very large)
  - Constant data handling
  - Multivariate column independence
  - Standard parameterisations (lambda=1600, 14400, 6.25)
  - MATLAB parity against Octave-generated fixtures

Numerical tolerances per AAP Section 0.7.1:
    ATOL = 1e-6
    RTOL = 1e-4
"""

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.timeseries.hp_filter import hp_filter

# Conftest helpers — load_fixture_npy and assert_allclose are plain functions
# imported directly; timeseries_fixture_dir is a pytest fixture injected by
# the conftest.py session-scoped fixture.
from tests.conftest import assert_allclose, load_fixture_npy

# ---------------------------------------------------------------------------
# Tolerance constants (match AAP Section 0.7.1)
# ---------------------------------------------------------------------------
ATOL: float = 1e-6
RTOL: float = 1e-4


# ===========================================================================
# Phase 1 — Input Validation Tests
# ===========================================================================


class TestHPFilterInputValidation:
    """Validate that hp_filter raises on invalid inputs."""

    def test_hp_filter_negative_lambda(self) -> None:
        """lambda < 0 must raise ValueError.

        Ref: hp_filter.m:40 — MATLAB error() on ~isscalar || lambda<0.
        """
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        with pytest.raises(ValueError):
            hp_filter(y, -1.0)

    def test_hp_filter_zero_lambda(self) -> None:
        """lambda == 0 must yield trend == y and cyclic == 0.

        Ref: hp_filter.m:40 — MATLAB allows lambda==0 (Gamma = I).
        When lambda is zero the penalty vanishes, so the minimiser
        picks trend = y exactly.
        """
        y = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        trend, cyclic = hp_filter(y, 0.0)
        npt.assert_allclose(trend, y, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(cyclic, np.zeros_like(y), atol=ATOL, rtol=RTOL)

    def test_hp_filter_nonscalar_lambda(self) -> None:
        """Non-scalar lambda (e.g. array) must raise ValueError.

        Ref: hp_filter.m:40 — ~isscalar(lambda) guard.
        """
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        with pytest.raises(ValueError):
            hp_filter(y, np.array([1.0, 2.0]))

    def test_hp_filter_3d_input(self) -> None:
        """3-D array input must raise ValueError.

        Ref: hp_filter.m:37 — ndims(y) > 2 guard.
        """
        y_3d = np.ones((10, 3, 2))
        with pytest.raises(ValueError):
            hp_filter(y_3d, 1600)

    def test_hp_filter_empty_input(self) -> None:
        """Empty array must raise ValueError.

        Ref: hp_filter.m:37 — size(y,1) < 1 guard.
        """
        y_empty = np.array([]).reshape(0, 1)
        with pytest.raises(ValueError):
            hp_filter(y_empty, 1600)


# ===========================================================================
# Phase 2 — Output Tests
# ===========================================================================


class TestHPFilterOutputs:
    """Verify return types, shapes, and the fundamental decomposition."""

    def test_hp_filter_returns_two(self) -> None:
        """hp_filter must return a tuple of two arrays."""
        rng = np.random.default_rng(42)
        y = np.cumsum(rng.standard_normal(100))
        result = hp_filter(y, 1600)
        assert isinstance(result, tuple), "hp_filter must return a tuple"
        assert len(result) == 2, "hp_filter must return exactly 2 elements"

    def test_hp_filter_output_shapes_univariate(self) -> None:
        """Univariate 1-D input must produce 1-D trend and cyclic of length T."""
        rng = np.random.default_rng(42)
        T = 200
        y = np.cumsum(rng.standard_normal(T))
        trend, cyclic = hp_filter(y, 1600)
        assert trend.shape == (T,), f"Trend shape {trend.shape} != ({T},)"
        assert cyclic.shape == (T,), f"Cyclic shape {cyclic.shape} != ({T},)"

    def test_hp_filter_output_shapes_univariate_2d(self) -> None:
        """Univariate 2-D (T,1) input must produce (T,1) outputs."""
        rng = np.random.default_rng(42)
        T = 200
        y = np.cumsum(rng.standard_normal((T, 1)), axis=0)
        trend, cyclic = hp_filter(y, 1600)
        assert trend.shape == (T, 1), f"Trend shape {trend.shape} != ({T}, 1)"
        assert cyclic.shape == (T, 1), f"Cyclic shape {cyclic.shape} != ({T}, 1)"

    def test_hp_filter_output_shapes_multivariate(self) -> None:
        """Multivariate T×K input must produce T×K trend and cyclic."""
        rng = np.random.default_rng(42)
        T, K = 200, 5
        y = np.cumsum(rng.standard_normal((T, K)), axis=0)
        trend, cyclic = hp_filter(y, 1600)
        assert trend.shape == (T, K), f"Trend shape {trend.shape} != ({T}, {K})"
        assert cyclic.shape == (T, K), f"Cyclic shape {cyclic.shape} != ({T}, {K})"

    def test_hp_filter_decomposition_identity(self) -> None:
        """y == trend + cyclic must hold within tolerance.

        This is the fundamental HP filter identity:
            cyclic = y - trend  →  y = trend + cyclic.
        Ref: hp_filter.m:16 — "CYCLIC = Y - TREND".
        """
        rng = np.random.default_rng(42)
        y = np.cumsum(rng.standard_normal(200))
        trend, cyclic = hp_filter(y, 1600)
        npt.assert_allclose(trend + cyclic, y, atol=ATOL, rtol=RTOL)

    def test_hp_filter_decomposition_identity_multivariate(self) -> None:
        """Decomposition identity must hold for multivariate T×K data."""
        rng = np.random.default_rng(42)
        y = np.cumsum(rng.standard_normal((300, 4)), axis=0)
        trend, cyclic = hp_filter(y, 1600)
        npt.assert_allclose(trend + cyclic, y, atol=ATOL, rtol=RTOL)

    def test_hp_filter_output_dtype(self) -> None:
        """Outputs must be float64 numpy arrays."""
        rng = np.random.default_rng(42)
        y = np.cumsum(rng.standard_normal(100))
        trend, cyclic = hp_filter(y, 1600)
        assert isinstance(trend, np.ndarray)
        assert isinstance(cyclic, np.ndarray)
        assert trend.dtype == np.float64
        assert cyclic.dtype == np.float64


# ===========================================================================
# Phase 3 — Smoothing Properties
# ===========================================================================


class TestHPFilterProperties:
    """Validate smoothing and extreme-lambda behaviour."""

    def test_hp_filter_trend_smoother_with_larger_lambda(self) -> None:
        """Larger lambda → smoother trend (lower variance of Δ²trend).

        The HP penalty is proportional to lambda * Σ (Δ²trend)².  A larger
        lambda forces a smaller second-difference variance.
        """
        rng = np.random.default_rng(42)
        y = np.cumsum(rng.standard_normal(200))
        trend_low, _ = hp_filter(y, 100)
        trend_high, _ = hp_filter(y, 10000)
        var_dd_low = np.var(np.diff(np.diff(trend_low)))
        var_dd_high = np.var(np.diff(np.diff(trend_high)))
        assert var_dd_high < var_dd_low, (
            f"Larger lambda must produce smoother trend: "
            f"var(Δ²trend_high)={var_dd_high:.6e} >= var(Δ²trend_low)={var_dd_low:.6e}"
        )

    def test_hp_filter_lambda_very_small_trend_equals_y(self) -> None:
        """Very small lambda (≈0) → trend ≈ y (zero penalty).

        When lambda → 0 the smoothing penalty vanishes and the minimiser
        picks trend = y to minimise Σ (y-trend)².
        """
        rng = np.random.default_rng(42)
        y = np.cumsum(rng.standard_normal(100))
        trend, cyclic = hp_filter(y, 1e-10)
        npt.assert_allclose(trend, y, atol=1e-3)
        npt.assert_allclose(cyclic, np.zeros_like(y), atol=1e-3)

    def test_hp_filter_lambda_very_large_trend_linear(self) -> None:
        """Very large lambda → trend ≈ OLS linear fit.

        Ref: hp_filter.m:83-86 — lambda > 1e10 triggers OLS fallback.
        Even below 1e10, large lambda drives Δ²trend → 0, forcing linearity.
        """
        rng = np.random.default_rng(42)
        y = np.cumsum(rng.standard_normal(100))
        trend, _ = hp_filter(y, 1e12)  # Above 1e10 → OLS fallback path
        # OLS linear fit for comparison
        x = np.arange(len(y), dtype=np.float64)
        coeffs = np.polyfit(x, trend, 1)
        trend_linear = np.polyval(coeffs, x)
        npt.assert_allclose(trend, trend_linear, atol=1e-6)

    def test_hp_filter_constant_data(self) -> None:
        """Constant y → trend = y, cyclic = 0.

        A constant series has zero second-differences, so it minimises
        both the fit and the smoothness penalty simultaneously.
        """
        y_const = np.ones(100) * 5.0
        trend, cyclic = hp_filter(y_const, 1600)
        npt.assert_allclose(trend, y_const, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(cyclic, np.zeros(100), atol=ATOL, rtol=RTOL)

    def test_hp_filter_linear_data(self) -> None:
        """Linear y → trend = y, cyclic = 0 (for any lambda ≥ 0).

        A linear series has zero second-differences, so the trend
        matches exactly regardless of lambda.
        """
        y_linear = np.linspace(1.0, 10.0, 200)
        trend, cyclic = hp_filter(y_linear, 1600)
        npt.assert_allclose(trend, y_linear, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(cyclic, np.zeros(200), atol=ATOL, rtol=RTOL)

    def test_hp_filter_small_t_cases(self) -> None:
        """T = 1 and T = 2: trend must equal y (identity fallback).

        Ref: hp_filter.m:55-56 — Gamma = eye(T) for T ∈ {1, 2}.
        """
        # T = 1
        y1 = np.array([3.14])
        trend1, cyclic1 = hp_filter(y1, 1600)
        npt.assert_allclose(trend1, y1, atol=ATOL)
        npt.assert_allclose(cyclic1, np.zeros(1), atol=ATOL)

        # T = 2
        y2 = np.array([1.0, 4.0])
        trend2, cyclic2 = hp_filter(y2, 1600)
        npt.assert_allclose(trend2, y2, atol=ATOL)
        npt.assert_allclose(cyclic2, np.zeros(2), atol=ATOL)

    def test_hp_filter_t_equals_3(self) -> None:
        """T = 3: must produce valid decomposition via dense 3×3 system.

        Ref: hp_filter.m:57-66 — special 3×3 Gamma matrix construction.
        """
        y3 = np.array([1.0, 3.0, 2.0])
        trend3, cyclic3 = hp_filter(y3, 1600)
        # Decomposition identity
        npt.assert_allclose(trend3 + cyclic3, y3, atol=ATOL)
        # T=3 with large lambda: trend should approximate a line
        _, _ = hp_filter(y3, 1e6)  # Should not raise

    def test_hp_filter_t_equals_4(self) -> None:
        """T = 4: boundary case where sparse pentadiagonal system starts.

        Ref: hp_filter.m:67 — otherwise clause for T ≥ 4.
        """
        rng = np.random.default_rng(42)
        y4 = np.cumsum(rng.standard_normal(4))
        trend4, cyclic4 = hp_filter(y4, 1600)
        npt.assert_allclose(trend4 + cyclic4, y4, atol=ATOL)


# ===========================================================================
# Phase 4 — Multivariate Tests
# ===========================================================================


class TestHPFilterMultivariate:
    """Verify multivariate behaviour and column independence."""

    def test_hp_filter_multivariate_columns_independent(self) -> None:
        """Each column must be filtered independently.

        The HP filter applied to a T×K matrix must produce the same result
        as filtering each column separately.  This verifies no cross-column
        leakage in the sparse solver.
        """
        rng = np.random.default_rng(42)
        y = np.cumsum(rng.standard_normal((200, 3)), axis=0)
        trend_multi, cyclic_multi = hp_filter(y, 1600)
        for k in range(3):
            trend_uni, cyclic_uni = hp_filter(y[:, k], 1600)
            npt.assert_allclose(
                trend_multi[:, k], trend_uni.ravel(), atol=ATOL, rtol=RTOL,
                err_msg=f"Column {k} trend mismatch"
            )
            npt.assert_allclose(
                cyclic_multi[:, k], cyclic_uni.ravel(), atol=ATOL, rtol=RTOL,
                err_msg=f"Column {k} cyclic mismatch"
            )

    def test_hp_filter_single_column_2d(self) -> None:
        """T×1 2-D input must match 1-D input (after squeeze).

        Ensures that the internal 1-D → 2-D conversion logic is consistent.
        """
        rng = np.random.default_rng(42)
        y_1d = np.cumsum(rng.standard_normal(150))
        y_2d = y_1d.reshape(-1, 1)

        trend_1d, cyclic_1d = hp_filter(y_1d, 1600)
        trend_2d, cyclic_2d = hp_filter(y_2d, 1600)

        npt.assert_allclose(
            trend_2d.ravel(), trend_1d, atol=ATOL, rtol=RTOL,
            err_msg="1-D vs 2-D trend mismatch"
        )
        npt.assert_allclose(
            cyclic_2d.ravel(), cyclic_1d, atol=ATOL, rtol=RTOL,
            err_msg="1-D vs 2-D cyclic mismatch"
        )


# ===========================================================================
# Phase 5 — Standard Parameterisations
# ===========================================================================


class TestHPFilterStandardParams:
    """Test standard parameter choices from the literature."""

    def test_hp_filter_quarterly_lambda1600(self) -> None:
        """Quarterly data with lambda=1600 (Hodrick & Prescott, 1997).

        Must produce a valid decomposition and a reasonably smooth trend.
        """
        rng = np.random.default_rng(42)
        # Simulate 80 quarters (20 years) of GDP-like data
        T = 80
        trend_true = np.linspace(100, 200, T)
        cycle_true = 5.0 * np.sin(2.0 * np.pi * np.arange(T) / 32)
        noise = rng.standard_normal(T) * 2.0
        y = trend_true + cycle_true + noise

        trend, cyclic = hp_filter(y, 1600)
        # Decomposition identity
        npt.assert_allclose(trend + cyclic, y, atol=ATOL, rtol=RTOL)
        # Trend should be smoother than original
        var_dd_y = np.var(np.diff(np.diff(y)))
        var_dd_trend = np.var(np.diff(np.diff(trend)))
        assert var_dd_trend < var_dd_y, (
            "Trend second-difference variance should be less than original"
        )

    def test_hp_filter_monthly_lambda14400(self) -> None:
        """Monthly data with lambda=14400.

        Ref: hp_filter.m:17 — "14400 is the recommended value for monthly data."
        """
        rng = np.random.default_rng(42)
        T = 360  # 30 years of monthly data
        trend_true = np.linspace(100, 300, T)
        cycle_true = 3.0 * np.sin(2.0 * np.pi * np.arange(T) / 96)
        noise = rng.standard_normal(T) * 1.5
        y = trend_true + cycle_true + noise

        trend, cyclic = hp_filter(y, 14400)
        npt.assert_allclose(trend + cyclic, y, atol=ATOL, rtol=RTOL)
        var_dd_trend = np.var(np.diff(np.diff(trend)))
        var_dd_y = np.var(np.diff(np.diff(y)))
        assert var_dd_trend < var_dd_y

    def test_hp_filter_annual_lambda6_25(self) -> None:
        """Annual data with lambda=6.25 (Ravn & Uhlig, 2002).

        Lower lambda permits more curvature in the trend for
        low-frequency annual data.
        """
        rng = np.random.default_rng(42)
        T = 100
        trend_true = np.linspace(50, 150, T)
        y = trend_true + rng.standard_normal(T) * 3.0

        trend, cyclic = hp_filter(y, 6.25)
        npt.assert_allclose(trend + cyclic, y, atol=ATOL, rtol=RTOL)


# ===========================================================================
# Phase 6 — MATLAB Fixture Parity Tests
# ===========================================================================


class TestHPFilterParity:
    """Compare Python hp_filter output against MATLAB/Octave reference fixtures."""

    @pytest.mark.parity
    @pytest.mark.requires_fixtures
    def test_hp_filter_parity_scenario_dict(self, timeseries_fixture_dir) -> None:
        """Test all 4 scenarios from hp_filter.npy fixture dict.

        Fixture structure (generated by generate_fixtures.m):
          - scenario_1: T=200, lambda=1600  (quarterly)
          - scenario_2: T=500, lambda=14400 (monthly)
          - scenario_3: T=200×3, lambda=1600 (multivariate quarterly)
          - scenario_4: T=100, lambda=6.25  (annual)
        """
        fixture = load_fixture_npy(timeseries_fixture_dir, "hp_filter")
        data = fixture.item()  # dict with scenario_N_* keys

        for scenario_idx in range(1, 5):
            prefix = f"scenario_{scenario_idx}"
            y_key = f"{prefix}_y"
            lam_key = f"{prefix}_lambda"
            trend_key = f"{prefix}_trend"
            cyclic_key = f"{prefix}_cyclic"

            # Skip if keys are missing for this scenario
            if y_key not in data:
                continue

            y_ref = np.asarray(data[y_key], dtype=np.float64)
            lam_ref = float(data[lam_key])
            trend_ref = np.asarray(data[trend_key], dtype=np.float64)
            cyclic_ref = np.asarray(data[cyclic_key], dtype=np.float64)

            trend_py, cyclic_py = hp_filter(y_ref, lam_ref)

            # Match shapes for comparison — fixture may be (T,1) while
            # 1-D input produces (T,) in Python
            if trend_py.ndim == 1 and trend_ref.ndim == 2 and trend_ref.shape[1] == 1:
                trend_ref = trend_ref.ravel()
                cyclic_ref = cyclic_ref.ravel()

            assert_allclose(
                trend_py, trend_ref,
                err_msg=f"Scenario {scenario_idx} trend parity (lambda={lam_ref})"
            )
            assert_allclose(
                cyclic_py, cyclic_ref,
                err_msg=f"Scenario {scenario_idx} cyclic parity (lambda={lam_ref})"
            )

    @pytest.mark.parity
    @pytest.mark.requires_fixtures
    def test_hp_filter_parity_trend_cyclic_separate(self, timeseries_fixture_dir) -> None:
        """Test against hp_filter_hp_trend.npy and hp_filter_hp_cyclic.npy.

        These are standalone 1-D fixtures (T=1000) generated with a standard
        seed and lambda=1600.
        """
        trend_ref = load_fixture_npy(timeseries_fixture_dir, "hp_filter_hp_trend")
        cyclic_ref = load_fixture_npy(timeseries_fixture_dir, "hp_filter_hp_cyclic")

        # Reconstruct input y from the decomposition identity
        y_ref = trend_ref + cyclic_ref

        trend_py, cyclic_py = hp_filter(y_ref, 1600)

        assert_allclose(
            trend_py, trend_ref,
            err_msg="hp_filter_hp_trend parity"
        )
        assert_allclose(
            cyclic_py, cyclic_ref,
            err_msg="hp_filter_hp_cyclic parity"
        )


# ===========================================================================
# Phase 7 — Edge Cases and Robustness
# ===========================================================================


class TestHPFilterEdgeCases:
    """Additional edge case and robustness tests."""

    def test_hp_filter_integer_input(self) -> None:
        """Integer arrays should be auto-cast to float64 without error."""
        y_int = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        trend, cyclic = hp_filter(y_int, 1600)
        npt.assert_allclose(trend + cyclic, y_int.astype(np.float64), atol=ATOL)

    def test_hp_filter_monotonic_increasing(self) -> None:
        """Monotonically increasing data: trend should also increase.

        For a strongly increasing series with moderate lambda the trend
        should capture the upward trajectory.
        """
        y = np.arange(1.0, 101.0)
        trend, cyclic = hp_filter(y, 1600)
        # Trend must be non-decreasing (allowing for tiny numerical noise)
        trend_diff = np.diff(trend)
        assert np.all(trend_diff > -ATOL), "Trend of increasing data should be non-decreasing"

    def test_hp_filter_warning_large_lambda(self) -> None:
        """Lambda > 1e10 must trigger UserWarning.

        Ref: hp_filter.m:44-46 — MATLAB warning for lambda > lambdaUpperBound.
        """
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        with pytest.warns(UserWarning, match="HP_FILTER"):
            hp_filter(y, 1e11)

    def test_hp_filter_reproducibility(self) -> None:
        """Calling hp_filter twice with the same inputs must produce
        identical results (deterministic computation, no random state).
        """
        rng = np.random.default_rng(42)
        y = np.cumsum(rng.standard_normal(200))

        trend1, cyclic1 = hp_filter(y, 1600)
        trend2, cyclic2 = hp_filter(y, 1600)

        npt.assert_array_equal(trend1, trend2)
        npt.assert_array_equal(cyclic1, cyclic2)
