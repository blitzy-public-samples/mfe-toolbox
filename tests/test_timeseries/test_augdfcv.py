"""
Pytest tests for mfe_toolbox.timeseries.augdfcv — ADF critical value interpolation.

Tests cover:
- Return type and shape validation
- P-value range constraints (0 ≤ pval ≤ 1)
- Critical value shape (6 elements) and monotonicity
- Extreme negative, zero, and positive t-statistics
- Well-known asymptotic critical values (DF with constant, 5% ≈ -2.86)
- All deterministic specifications (p=0,1,2,3)
- Finite-sample size effects on critical values
- MATLAB parity tests against generated fixtures (ATOL=1e-6, RTOL=1e-4)

Source MATLAB reference: timeseries/augdfcv.m — Kevin Sheppard
Signature: [pval, critval] = augdfcv(tstat, p, T)

Per AAP Section 0.7.1: numpy.testing.assert_allclose(actual, expected,
    atol=1e-6, rtol=1e-4) is used for all numerical parity comparisons.
"""

from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.timeseries.augdfcv import augdfcv

# ---------------------------------------------------------------------------
# Numerical parity tolerances per AAP Section 0.7.1
# ---------------------------------------------------------------------------
ATOL: float = 1e-6
RTOL: float = 1e-4


# ---------------------------------------------------------------------------
# Helper to load the augdfcv fixture data
# ---------------------------------------------------------------------------
def _load_augdfcv_fixture(timeseries_fixture_dir: Path) -> dict:
    """Load the augdfcv fixture .npy file, skipping if absent."""
    path = timeseries_fixture_dir / "augdfcv.npy"
    if not path.exists():
        pytest.skip(f"Fixture file not found: {path}")
    data = np.load(path, allow_pickle=True)
    return data.item()


# ===========================================================================
# Basic return type and structure tests
# ===========================================================================


class TestAugdfcvReturnStructure:
    """Verify that augdfcv returns the correct types and shapes."""

    def test_augdfcv_returns_two(self) -> None:
        """augdfcv must return a 2-tuple (pval, critval)."""
        result = augdfcv(tstat=-2.5, p=1, T=100)
        assert isinstance(result, tuple), "Expected a tuple return"
        assert len(result) == 2, "Expected exactly 2 return values (pval, critval)"

    def test_augdfcv_pval_is_float(self) -> None:
        """The first return value (pval) must be a Python float."""
        pval, _ = augdfcv(tstat=-2.5, p=1, T=100)
        # Accept both Python float and numpy scalar float
        assert isinstance(pval, (float, np.floating)), (
            f"pval should be float, got {type(pval)}"
        )

    def test_augdfcv_critval_is_ndarray(self) -> None:
        """The second return value (critval) must be a numpy ndarray."""
        _, critval = augdfcv(tstat=-2.5, p=1, T=100)
        assert isinstance(critval, np.ndarray), (
            f"critval should be np.ndarray, got {type(critval)}"
        )

    def test_augdfcv_critval_shape(self) -> None:
        """critval must have exactly 6 elements [1%, 5%, 10%, 90%, 95%, 99%]."""
        _, critval = augdfcv(tstat=-2.5, p=1, T=100)
        assert critval.shape == (6,), (
            f"critval shape should be (6,), got {critval.shape}"
        )


# ===========================================================================
# P-value range and monotonicity tests
# ===========================================================================


class TestAugdfcvPvalRange:
    """Verify p-value is always in [0, 1]."""

    @pytest.mark.parametrize("tstat", [-10.0, -5.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0])
    @pytest.mark.parametrize("p", [0, 1, 2, 3])
    def test_augdfcv_pval_in_range(self, tstat: float, p: int) -> None:
        """0 ≤ pval ≤ 1 for all valid inputs."""
        pval, _ = augdfcv(tstat=tstat, p=p, T=100)
        assert 0.0 <= pval <= 1.0, (
            f"pval={pval} out of [0,1] for tstat={tstat}, p={p}, T=100"
        )


class TestAugdfcvCritvalMonotonicity:
    """Verify critical values are monotonically increasing."""

    @pytest.mark.parametrize("p", [0, 1, 2, 3])
    @pytest.mark.parametrize("T", [25, 100, 500, 10000])
    def test_augdfcv_critval_ordered(self, p: int, T: int) -> None:
        """critval must be monotonically increasing: cv[i] < cv[i+1]."""
        _, critval = augdfcv(tstat=-2.5, p=p, T=T)
        for i in range(len(critval) - 1):
            assert critval[i] < critval[i + 1], (
                f"critval not monotonic at index {i}: "
                f"critval[{i}]={critval[i]} >= critval[{i+1}]={critval[i+1]} "
                f"for p={p}, T={T}"
            )


# ===========================================================================
# Extreme t-statistic tests
# ===========================================================================


class TestAugdfcvExtremeTstat:
    """Verify behavior at extreme t-statistic values."""

    def test_augdfcv_extreme_negative_tstat(self) -> None:
        """Very negative tstat should yield pval near 0 (strong rejection)."""
        pval, _ = augdfcv(tstat=-20.0, p=1, T=100)
        assert pval == 0.0, (
            f"Expected pval=0 for very negative tstat=-20.0, got pval={pval}"
        )

    def test_augdfcv_zero_tstat(self) -> None:
        """tstat=0 should yield pval near 1 (not rejecting unit root)."""
        # For p=0 (no deterministic terms), tstat=0 means the series looks
        # like a unit root — pval should be large (close to 1 but depends on
        # the quantile grid).
        pval, _ = augdfcv(tstat=0.0, p=0, T=100)
        assert pval > 0.5, (
            f"Expected pval > 0.5 for tstat=0, p=0, T=100, got pval={pval}"
        )

    def test_augdfcv_positive_tstat(self) -> None:
        """Positive tstat should yield pval very close to 1."""
        pval, _ = augdfcv(tstat=3.0, p=0, T=100)
        assert pval > 0.99, (
            f"Expected pval > 0.99 for positive tstat=3.0, p=0, got pval={pval}"
        )

    def test_augdfcv_large_positive_tstat(self) -> None:
        """Very large positive tstat should give pval = 1.0."""
        pval, _ = augdfcv(tstat=10.0, p=1, T=100)
        assert pval == 1.0, (
            f"Expected pval=1.0 for tstat=10.0, p=1, T=100, got pval={pval}"
        )


# ===========================================================================
# Known asymptotic critical value tests
# ===========================================================================


class TestAugdfcvKnownCriticalValues:
    """Verify well-known asymptotic DF critical values."""

    def test_augdfcv_known_cv_at_5pct(self) -> None:
        """For large T, p=1: 5% critical value ≈ -2.86 (well-known DF table).

        The Dickey-Fuller 5% critical value for the case with a constant
        (p=1) is widely tabulated as approximately -2.86 for large samples.
        The Monte Carlo table should match within reasonable tolerance.
        """
        # Ref: augdfcv.m case 2 (p=1), T=10000 (near asymptotic)
        _, critval = augdfcv(tstat=-3.0, p=1, T=10000)
        # critval[1] is the 5% critical value
        assert -3.5 < critval[1] < -2.5, (
            f"5% CV for p=1, T=10000 should be near -2.86, got {critval[1]}"
        )
        # More precise check: should be close to -2.86
        assert abs(critval[1] - (-2.86)) < 0.05, (
            f"5% CV for p=1, T=10000 expected ≈-2.86, got {critval[1]}"
        )

    def test_augdfcv_known_cv_at_1pct_p1(self) -> None:
        """For large T, p=1: 1% critical value ≈ -3.43 (well-known DF table)."""
        _, critval = augdfcv(tstat=-3.0, p=1, T=10000)
        # critval[0] is the 1% critical value
        assert abs(critval[0] - (-3.43)) < 0.05, (
            f"1% CV for p=1, T=10000 expected ≈-3.43, got {critval[0]}"
        )

    def test_augdfcv_known_cv_at_10pct_p1(self) -> None:
        """For large T, p=1: 10% critical value ≈ -2.57 (well-known DF table)."""
        _, critval = augdfcv(tstat=-3.0, p=1, T=10000)
        # critval[2] is the 10% critical value
        assert abs(critval[2] - (-2.57)) < 0.05, (
            f"10% CV for p=1, T=10000 expected ≈-2.57, got {critval[2]}"
        )

    def test_augdfcv_p3_uses_normal_distribution(self) -> None:
        """For p=3 (standard normal), critical values are normal quantiles.

        The 5% normal quantile is -1.645 and the 1% is -2.326.
        These should be independent of T.
        """
        from scipy.stats import norm

        _, critval_100 = augdfcv(tstat=-2.0, p=3, T=100)
        _, critval_1000 = augdfcv(tstat=-2.0, p=3, T=1000)

        # Critical values should be identical for different T when p=3
        npt.assert_allclose(critval_100, critval_1000, atol=ATOL, rtol=RTOL,
                            err_msg="p=3 critical values should not depend on T")

        # Check against known normal quantiles
        expected_5pct = norm.ppf(0.05)  # ≈ -1.6449
        npt.assert_allclose(critval_100[1], expected_5pct, atol=ATOL, rtol=RTOL,
                            err_msg="p=3 5% CV should equal normal 5% quantile")

        expected_1pct = norm.ppf(0.01)  # ≈ -2.3263
        npt.assert_allclose(critval_100[0], expected_1pct, atol=ATOL, rtol=RTOL,
                            err_msg="p=3 1% CV should equal normal 1% quantile")


# ===========================================================================
# All deterministic specifications test
# ===========================================================================


class TestAugdfcvAllPValues:
    """Test that all deterministic specifications (p=0,1,2,3) produce valid output."""

    @pytest.mark.parametrize("p", [0, 1, 2, 3])
    def test_augdfcv_all_p_values(self, p: int) -> None:
        """Each p value should produce valid (pval, critval) without error."""
        pval, critval = augdfcv(tstat=-2.5, p=p, T=500)

        # Basic validity checks
        assert 0.0 <= pval <= 1.0, f"pval out of range for p={p}"
        assert critval.shape == (6,), f"critval shape wrong for p={p}"
        assert np.all(np.isfinite(critval)), f"critval has non-finite values for p={p}"

    def test_augdfcv_invalid_p_raises(self) -> None:
        """p values outside {0,1,2,3} should raise ValueError."""
        with pytest.raises(ValueError, match="p must be"):
            augdfcv(tstat=-2.5, p=4, T=100)

        with pytest.raises(ValueError, match="p must be"):
            augdfcv(tstat=-2.5, p=-1, T=100)


# ===========================================================================
# Sample size effect tests
# ===========================================================================


class TestAugdfcvSampleSizeEffect:
    """Verify that sample size affects critical values (finite-sample correction)."""

    def test_augdfcv_sample_size_effect(self) -> None:
        """Larger T should produce slightly different critical values.

        For the ADF test, critical values converge to asymptotic values
        as T → ∞. Small T has wider (more negative) critical values.
        """
        _, critval_small = augdfcv(tstat=-2.5, p=1, T=25)
        _, critval_large = augdfcv(tstat=-2.5, p=1, T=10000)

        # For p=1, the 1% CV should be more negative for small T
        # (wider distribution → more extreme critical values for small samples)
        assert critval_small[0] < critval_large[0], (
            f"Small-sample 1% CV ({critval_small[0]}) should be more negative "
            f"than large-sample 1% CV ({critval_large[0]})"
        )

        # The values should NOT be identical (different T → different CVs)
        assert not np.allclose(critval_small, critval_large), (
            "Critical values for T=25 and T=10000 should differ"
        )

    def test_augdfcv_asymptotic_convergence(self) -> None:
        """Critical values should be nearly identical for very large T values."""
        _, critval_5000 = augdfcv(tstat=-2.5, p=1, T=5000)
        _, critval_10000 = augdfcv(tstat=-2.5, p=1, T=10000)

        # For very large T, CVs should be close (within the Monte Carlo grid)
        npt.assert_allclose(critval_5000, critval_10000, atol=0.05, rtol=0.01,
                            err_msg="CVs for T=5000 and T=10000 should nearly match")

    def test_augdfcv_t_below_grid_minimum(self) -> None:
        """T smaller than grid minimum (10) should still return valid results.

        Per MATLAB source: T < min(Ts) uses last column (asymptotic).
        """
        pval, critval = augdfcv(tstat=-2.5, p=0, T=5)
        assert 0.0 <= pval <= 1.0
        assert critval.shape == (6,)
        assert np.all(np.isfinite(critval))

    def test_augdfcv_t_above_grid_maximum(self) -> None:
        """T larger than grid maximum (10000) should still return valid results.

        Per MATLAB source: T > max(Ts) uses last column (asymptotic).
        """
        pval, critval = augdfcv(tstat=-2.5, p=0, T=50000)
        assert 0.0 <= pval <= 1.0
        assert critval.shape == (6,)
        assert np.all(np.isfinite(critval))

    def test_augdfcv_t_exact_grid_point(self) -> None:
        """T exactly on a grid point (e.g. T=100) should interpolate cleanly."""
        pval, critval = augdfcv(tstat=-2.5, p=1, T=100)
        assert 0.0 <= pval <= 1.0
        assert critval.shape == (6,)
        assert np.all(np.isfinite(critval))

    def test_augdfcv_t_between_grid_points(self) -> None:
        """T between grid points (e.g. T=75) should interpolate."""
        pval, critval = augdfcv(tstat=-2.5, p=1, T=75)
        assert 0.0 <= pval <= 1.0
        assert critval.shape == (6,)

        # Result should be between the two bracketing grid-point values
        _, critval_50 = augdfcv(tstat=-2.5, p=1, T=50)
        _, critval_100 = augdfcv(tstat=-2.5, p=1, T=100)

        # Each element of critval should be between the two grid values
        for i in range(6):
            lo = min(critval_50[i], critval_100[i])
            hi = max(critval_50[i], critval_100[i])
            assert lo <= critval[i] <= hi, (
                f"critval[{i}]={critval[i]} not between "
                f"T=50 ({critval_50[i]}) and T=100 ({critval_100[i]})"
            )


# ===========================================================================
# MATLAB parity tests — fixture-based comparison
# ===========================================================================


class TestAugdfcvParity:
    """MATLAB parity tests comparing Python augdfcv against Octave-generated fixtures."""

    @pytest.mark.parity
    def test_augdfcv_parity_case1_p0(self, timeseries_fixture_dir: Path) -> None:
        """Parity: p=0 (no deterministic terms), T=1000, tstat=-2.5."""
        fixture = _load_augdfcv_fixture(timeseries_fixture_dir)
        case = fixture["case_1_p0_no_deterministic"]

        pval, critval = augdfcv(
            tstat=float(case["tstat"]),
            p=int(case["p"]),
            T=int(case["T"]),
        )

        npt.assert_allclose(pval, case["pval"], atol=ATOL, rtol=RTOL,
                            err_msg="pval parity failure for case 1 (p=0)")
        npt.assert_allclose(critval, case["critval"], atol=ATOL, rtol=RTOL,
                            err_msg="critval parity failure for case 1 (p=0)")

    @pytest.mark.parity
    def test_augdfcv_parity_case2_p1(self, timeseries_fixture_dir: Path) -> None:
        """Parity: p=1 (constant), T=1000, tstat=-3.0."""
        fixture = _load_augdfcv_fixture(timeseries_fixture_dir)
        case = fixture["case_2_p1_constant"]

        pval, critval = augdfcv(
            tstat=float(case["tstat"]),
            p=int(case["p"]),
            T=int(case["T"]),
        )

        npt.assert_allclose(pval, case["pval"], atol=ATOL, rtol=RTOL,
                            err_msg="pval parity failure for case 2 (p=1)")
        npt.assert_allclose(critval, case["critval"], atol=ATOL, rtol=RTOL,
                            err_msg="critval parity failure for case 2 (p=1)")

    @pytest.mark.parity
    def test_augdfcv_parity_case3_p2(self, timeseries_fixture_dir: Path) -> None:
        """Parity: p=2 (time trend), T=500, tstat=-3.5."""
        fixture = _load_augdfcv_fixture(timeseries_fixture_dir)
        case = fixture["case_3_p2_time_trend"]

        pval, critval = augdfcv(
            tstat=float(case["tstat"]),
            p=int(case["p"]),
            T=int(case["T"]),
        )

        npt.assert_allclose(pval, case["pval"], atol=ATOL, rtol=RTOL,
                            err_msg="pval parity failure for case 3 (p=2)")
        npt.assert_allclose(critval, case["critval"], atol=ATOL, rtol=RTOL,
                            err_msg="critval parity failure for case 3 (p=2)")

    @pytest.mark.parity
    def test_augdfcv_parity_case4_p3(self, timeseries_fixture_dir: Path) -> None:
        """Parity: p=3 (DGP trend, normal distribution), T=200, tstat=-4.0."""
        fixture = _load_augdfcv_fixture(timeseries_fixture_dir)
        case = fixture["case_4_p3_dgp_trend"]

        pval, critval = augdfcv(
            tstat=float(case["tstat"]),
            p=int(case["p"]),
            T=int(case["T"]),
        )

        npt.assert_allclose(pval, case["pval"], atol=ATOL, rtol=RTOL,
                            err_msg="pval parity failure for case 4 (p=3)")
        npt.assert_allclose(critval, case["critval"], atol=ATOL, rtol=RTOL,
                            err_msg="critval parity failure for case 4 (p=3)")

    @pytest.mark.parity
    def test_augdfcv_parity_case5_small_sample(self, timeseries_fixture_dir: Path) -> None:
        """Parity: small sample p=1, T=50, tstat=-2.0."""
        fixture = _load_augdfcv_fixture(timeseries_fixture_dir)
        case = fixture["case_5_small_sample"]

        pval, critval = augdfcv(
            tstat=float(case["tstat"]),
            p=int(case["p"]),
            T=int(case["T"]),
        )

        npt.assert_allclose(pval, case["pval"], atol=ATOL, rtol=RTOL,
                            err_msg="pval parity failure for case 5 (small sample)")
        npt.assert_allclose(critval, case["critval"], atol=ATOL, rtol=RTOL,
                            err_msg="critval parity failure for case 5 (small sample)")

    @pytest.mark.parity
    def test_augdfcv_parity_additional_scenarios(
        self, timeseries_fixture_dir: Path
    ) -> None:
        """Parity: all additional scenarios from the fixture file.

        The fixture contains ~54 additional test scenarios covering a wide
        range of tstat, p, and T combinations across all four deterministic
        specifications.  Each scenario is tested for both pval and critval
        parity against MATLAB/Octave reference outputs.
        """
        fixture = _load_augdfcv_fixture(timeseries_fixture_dir)
        scenarios = fixture["additional_scenarios"]

        for idx, scenario in enumerate(scenarios):
            tstat_val = float(scenario["tstat"])
            p_val = int(scenario["p"])
            T_val = int(scenario["T"])
            expected_pval = float(scenario["pval"])
            expected_critval = np.asarray(scenario["critval"])

            pval, critval = augdfcv(tstat=tstat_val, p=p_val, T=T_val)

            npt.assert_allclose(
                pval, expected_pval, atol=ATOL, rtol=RTOL,
                err_msg=(
                    f"pval parity failure for additional scenario {idx}: "
                    f"tstat={tstat_val}, p={p_val}, T={T_val}"
                ),
            )
            npt.assert_allclose(
                critval, expected_critval, atol=ATOL, rtol=RTOL,
                err_msg=(
                    f"critval parity failure for additional scenario {idx}: "
                    f"tstat={tstat_val}, p={p_val}, T={T_val}"
                ),
            )


# ===========================================================================
# Edge case and robustness tests
# ===========================================================================


class TestAugdfcvEdgeCases:
    """Edge case and robustness tests for augdfcv."""

    def test_augdfcv_tstat_at_exact_cv_boundary(self) -> None:
        """tstat exactly at a critical value boundary should still return valid output."""
        # Get CVs first, then use one as tstat
        _, critval = augdfcv(tstat=-2.5, p=1, T=100)
        cv_5pct = critval[1]  # 5% critical value

        pval, critval2 = augdfcv(tstat=float(cv_5pct), p=1, T=100)
        assert 0.0 <= pval <= 1.0
        # pval should be close to 0.05 if tstat is at the 5% CV
        assert abs(pval - 0.05) < 0.02, (
            f"pval at 5% CV should be near 0.05, got {pval}"
        )

    def test_augdfcv_consistent_critvals_across_tstats(self) -> None:
        """Critical values should be the same for different tstats with same p and T.

        Critical values only depend on p and T, not on tstat.
        """
        _, critval1 = augdfcv(tstat=-5.0, p=1, T=100)
        _, critval2 = augdfcv(tstat=0.0, p=1, T=100)
        _, critval3 = augdfcv(tstat=5.0, p=1, T=100)

        npt.assert_allclose(critval1, critval2, atol=ATOL, rtol=RTOL,
                            err_msg="critval should not depend on tstat")
        npt.assert_allclose(critval2, critval3, atol=ATOL, rtol=RTOL,
                            err_msg="critval should not depend on tstat")

    def test_augdfcv_pval_monotone_in_tstat(self) -> None:
        """pval should be monotonically non-decreasing in tstat.

        More negative tstat → stronger rejection → smaller pval.
        """
        tstats = [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0]
        pvals = [augdfcv(tstat=t, p=1, T=100)[0] for t in tstats]

        for i in range(len(pvals) - 1):
            assert pvals[i] <= pvals[i + 1] + 1e-10, (
                f"pval not monotone: pval({tstats[i]})={pvals[i]} > "
                f"pval({tstats[i+1]})={pvals[i+1]}"
            )

    def test_augdfcv_p0_more_negative_cv_than_p1(self) -> None:
        """For the same T, the 5% CV for p=0 should be less negative than p=1.

        The DF distribution without deterministic terms (p=0) has less
        negative critical values than the case with a constant (p=1).
        """
        _, critval_p0 = augdfcv(tstat=-2.5, p=0, T=1000)
        _, critval_p1 = augdfcv(tstat=-2.5, p=1, T=1000)

        # 5% CV: p=0 should be less negative (closer to 0) than p=1
        assert critval_p0[1] > critval_p1[1], (
            f"5% CV for p=0 ({critval_p0[1]}) should be less negative than "
            f"p=1 ({critval_p1[1]})"
        )

    def test_augdfcv_p2_more_negative_cv_than_p1(self) -> None:
        """For the same T, the 5% CV for p=2 should be more negative than p=1.

        Including a time trend shifts the DF distribution further left.
        """
        _, critval_p1 = augdfcv(tstat=-2.5, p=1, T=1000)
        _, critval_p2 = augdfcv(tstat=-2.5, p=2, T=1000)

        # 5% CV: p=2 should be more negative than p=1
        assert critval_p2[1] < critval_p1[1], (
            f"5% CV for p=2 ({critval_p2[1]}) should be more negative than "
            f"p=1 ({critval_p1[1]})"
        )
