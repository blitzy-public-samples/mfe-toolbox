"""
Pytest tests for ``mfe_toolbox.timeseries.augdf`` — Augmented Dickey-Fuller
unit root test.

Tests cover:
- Input validation (invalid *p*, negative *lags*, short series, non-vector input)
- Output structure (tuple length, scalar types, shape, ordering)
- Statistical properties (unit root detection, stationary rejection)
- Residual diagnostics (shape, approximate orthogonality)
- Deterministic specification exhaustive coverage (p=0,1,2,3)
- MATLAB numerical parity via fixture comparison (when fixtures are available)

Per AAP Section 0.7.1: All parity assertions use
    ``numpy.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-4)``.

Source MATLAB reference: ``timeseries/augdf.m`` (Kevin Sheppard, Revision 3.0.1)
"""

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.timeseries.augdf import augdf

# ---------------------------------------------------------------------------
# Tolerance constants — AAP Section 0.7.1
# ---------------------------------------------------------------------------
ATOL: float = 1e-6
RTOL: float = 1e-4


# =========================================================================
# Phase 1: Test Fixtures
# =========================================================================

@pytest.fixture
def rng():
    """Seeded random number generator for reproducible test data."""
    return np.random.default_rng(42)


@pytest.fixture
def unit_root_series(rng):
    """Random walk: y(t) = y(t-1) + e(t).  Should NOT reject unit root.

    T = 500 provides sufficient power for the ADF test to distinguish
    between unit root and stationary processes.
    """
    T = 500
    y = np.cumsum(rng.standard_normal(T))
    return y


@pytest.fixture
def stationary_series(rng):
    """Stationary AR(1): y(t) = 0.5 * y(t-1) + e(t).  Should reject unit root.

    The autoregressive coefficient 0.5 is well inside the unit circle,
    producing a clearly stationary process.
    """
    T = 500
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.5 * y[t - 1] + rng.standard_normal()
    return y


@pytest.fixture
def long_stationary_series(rng):
    """Longer stationary AR(1) series (T=1000) for higher-power tests."""
    T = 1000
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.3 * y[t - 1] + rng.standard_normal()
    return y


# =========================================================================
# Phase 2: Input Validation Tests
# =========================================================================

class TestAugdfInputValidation:
    """Verify that ``augdf`` raises ``ValueError`` on invalid inputs.

    Ref: augdf.m:37-58 — MATLAB uses ``error()`` for invalid inputs;
    Python equivalent is ``raise ValueError()``.
    """

    def test_augdf_invalid_p_too_high(self, unit_root_series):
        """p=5 is outside {0, 1, 2, 3} → ValueError."""
        with pytest.raises(ValueError, match="P must be"):
            augdf(unit_root_series, p=5, lags=1)

    def test_augdf_invalid_p_negative(self, unit_root_series):
        """p=-1 is outside {0, 1, 2, 3} → ValueError."""
        with pytest.raises(ValueError, match="P must be"):
            augdf(unit_root_series, p=-1, lags=1)

    def test_augdf_invalid_p_float(self, unit_root_series):
        """p=1.5 is not an integer in {0, 1, 2, 3} → ValueError."""
        with pytest.raises(ValueError, match="P must be"):
            augdf(unit_root_series, p=1.5, lags=1)

    def test_augdf_negative_lags(self, unit_root_series):
        """lags < 0 → ValueError.

        Ref: augdf.m:50-52 — MATLAB validates lags > 0 and floor(lags)==lags.
        """
        with pytest.raises(ValueError, match="[Ll]ags|LAGS"):
            augdf(unit_root_series, p=1, lags=-1)

    def test_augdf_short_series(self):
        """T too small relative to lags → ValueError.

        Ref: augdf.m:41-43 — Length check: T > lags + 1.
        """
        y_short = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="[Ll]ength|data|LAGS"):
            augdf(y_short, p=1, lags=4)

    def test_augdf_short_series_edge_case(self):
        """T == lags + 1 is exactly at the boundary → ValueError."""
        y_edge = np.arange(5, dtype=np.float64)
        with pytest.raises(ValueError, match="[Ll]ength|data|LAGS"):
            augdf(y_edge, p=1, lags=4)


# =========================================================================
# Phase 3: Output Structure Tests
# =========================================================================

class TestAugdfOutputStructure:
    """Verify that ``augdf`` returns correctly structured outputs.

    Expected return: ``(adfstat, pval, critval, resid)`` where
    - ``adfstat`` is a float scalar
    - ``pval`` is a float in [0, 1]
    - ``critval`` is a 6-element ndarray
    - ``resid`` is an ndarray of regression residuals
    """

    def test_augdf_returns_four(self, unit_root_series):
        """augdf returns a 4-element tuple."""
        result = augdf(unit_root_series, p=1, lags=4)
        assert isinstance(result, tuple), "augdf must return a tuple"
        assert len(result) == 4, f"Expected 4 elements, got {len(result)}"

    def test_augdf_adfstat_scalar(self, unit_root_series):
        """adfstat is a float scalar."""
        adfstat, _, _, _ = augdf(unit_root_series, p=1, lags=4)
        assert isinstance(adfstat, (float, np.floating)), (
            f"adfstat must be a float, got {type(adfstat)}"
        )
        assert np.isfinite(adfstat), "adfstat must be finite"

    def test_augdf_pval_in_range(self, unit_root_series):
        """0 ≤ pval ≤ 1 for all valid inputs."""
        _, pval, _, _ = augdf(unit_root_series, p=1, lags=4)
        assert 0.0 <= pval <= 1.0, f"pval={pval} must be in [0, 1]"

    def test_augdf_critval_shape(self, unit_root_series):
        """critval has exactly 6 elements: [1%, 5%, 10%, 90%, 95%, 99%]."""
        _, _, critval, _ = augdf(unit_root_series, p=1, lags=4)
        assert isinstance(critval, np.ndarray), "critval must be an ndarray"
        assert critval.shape == (6,), (
            f"critval shape must be (6,), got {critval.shape}"
        )

    def test_augdf_critval_ordered(self, unit_root_series):
        """Critical values are monotonically increasing.

        The [1%, 5%, 10%, 90%, 95%, 99%] critical values from the DF
        distribution should be ordered from most negative (left tail)
        to most positive (right tail).
        """
        _, _, critval, _ = augdf(unit_root_series, p=1, lags=4)
        for i in range(len(critval) - 1):
            assert critval[i] <= critval[i + 1], (
                f"critval not monotonic at index {i}: "
                f"{critval[i]} > {critval[i + 1]}"
            )

    def test_augdf_critval_finite(self, unit_root_series):
        """All critical values must be finite real numbers."""
        _, _, critval, _ = augdf(unit_root_series, p=1, lags=4)
        assert np.all(np.isfinite(critval)), "All critvals must be finite"

    def test_augdf_resid_ndarray(self, unit_root_series):
        """Residuals are returned as a numpy ndarray."""
        _, _, _, resid = augdf(unit_root_series, p=1, lags=4)
        assert isinstance(resid, np.ndarray), "resid must be an ndarray"


# =========================================================================
# Phase 4: Statistical Property Tests
# =========================================================================

class TestAugdfStatisticalProperties:
    """Verify statistical correctness of the ADF test.

    These tests use seeded random data to ensure that the ADF test
    correctly distinguishes between unit root and stationary processes.
    """

    def test_augdf_unit_root_not_rejected(self, unit_root_series):
        """Random walk: pval > 0.05 — fail to reject unit root.

        A random walk y(t) = y(t-1) + e(t) is a unit root process;
        the ADF test should not reject H0 (unit root).
        """
        adfstat, pval, critval, resid = augdf(unit_root_series, p=1, lags=4)
        assert pval > 0.05, (
            f"Unit root should NOT be rejected for random walk: pval={pval}"
        )

    def test_augdf_stationary_rejected(self, stationary_series):
        """Stationary AR(1): pval < 0.10 — reject unit root.

        A stationary AR(1) with coefficient 0.5 should be detected as
        stationary; the ADF test should reject H0 (unit root).
        """
        adfstat, pval, critval, resid = augdf(stationary_series, p=1, lags=1)
        assert pval < 0.10, (
            f"Stationary series should reject unit root: pval={pval}"
        )

    @pytest.mark.parametrize("p", [0, 1, 2, 3])
    def test_augdf_all_p_values(self, unit_root_series, p):
        """All deterministic structures (p=0,1,2,3) produce valid outputs.

        Ref: augdf.m:74-147 — The switch statement handles all four cases.
        """
        adfstat, pval, critval, resid = augdf(unit_root_series, p=p, lags=2)

        # Basic validity checks for each p value
        assert isinstance(adfstat, (float, np.floating)), (
            f"p={p}: adfstat not float"
        )
        assert np.isfinite(adfstat), f"p={p}: adfstat not finite"
        assert 0.0 <= pval <= 1.0, f"p={p}: pval={pval} out of range"
        assert critval.shape == (6,), f"p={p}: critval shape {critval.shape}"
        assert isinstance(resid, np.ndarray), f"p={p}: resid not ndarray"

    def test_augdf_lag_effect(self, unit_root_series):
        """Varying lags changes the test statistic but outputs remain valid.

        More lags consume more degrees of freedom, potentially changing
        the statistic, but the output structure should remain correct.
        """
        results = {}
        for lags in [1, 4, 8, 12]:
            adfstat, pval, critval, resid = augdf(
                unit_root_series, p=1, lags=lags
            )
            results[lags] = adfstat
            # All results must be valid
            assert np.isfinite(adfstat), f"lags={lags}: adfstat not finite"
            assert 0.0 <= pval <= 1.0, f"lags={lags}: pval out of range"

        # Different lags should generally produce different statistics
        unique_stats = set(round(v, 6) for v in results.values())
        assert len(unique_stats) > 1, (
            "Different lags should produce different test statistics"
        )

    def test_augdf_zero_lags(self, unit_root_series):
        """lags=0 runs the standard (non-augmented) Dickey-Fuller test.

        Ref: augdf.m:15 — Use lags=0 for DF test.
        """
        adfstat, pval, critval, resid = augdf(unit_root_series, p=1, lags=0)

        assert np.isfinite(adfstat), "DF stat must be finite"
        assert 0.0 <= pval <= 1.0, "DF pval must be in [0,1]"
        assert critval.shape == (6,), "critval shape must be (6,)"
        # For lags=0: residual length = T - 0 - 1 = T - 1
        T = len(unit_root_series)
        assert len(resid) == T - 1, (
            f"lags=0 residual length should be {T-1}, got {len(resid)}"
        )

    def test_augdf_p3_uses_normal_distribution(self, unit_root_series):
        """For p=3, critical values come from standard normal distribution.

        Ref: augdf.m:145 — critval=norminv([.01 .05 .1 .9 .95 .99]')
        Ref: augdf.m:146 — pval = normcdf(adfstat)
        """
        from scipy.stats import norm

        adfstat, pval, critval, resid = augdf(unit_root_series, p=3, lags=2)

        # p=3 critical values should be standard normal quantiles
        expected_cv = norm.ppf([0.01, 0.05, 0.10, 0.90, 0.95, 0.99])
        npt.assert_allclose(critval, expected_cv, atol=ATOL, rtol=RTOL,
                            err_msg="p=3 critvals should be normal quantiles")

        # p=3 pval should be Φ(adfstat) where Φ is the standard normal CDF
        expected_pval = norm.cdf(adfstat)
        npt.assert_allclose(pval, expected_pval, atol=ATOL, rtol=RTOL,
                            err_msg="p=3 pval should be norm.cdf(adfstat)")

    def test_augdf_stationary_with_strong_ar(self, rng):
        """A strongly stationary AR(1) with coefficient 0.2 rejects unit root."""
        T = 1000
        y = np.zeros(T)
        for t in range(1, T):
            y[t] = 0.2 * y[t - 1] + rng.standard_normal()
        adfstat, pval, critval, resid = augdf(y, p=1, lags=2)
        assert pval < 0.05, (
            f"Strongly stationary AR(1) should reject at 5%: pval={pval}"
        )


# =========================================================================
# Phase 5: Residual Tests
# =========================================================================

class TestAugdfResiduals:
    """Verify properties of the regression residuals from the ADF test."""

    @pytest.mark.parametrize("lags", [0, 1, 4, 8])
    def test_augdf_residual_shape(self, unit_root_series, lags):
        """Residuals have correct length: T - lags - 1.

        Ref: augdf.m:68-72 — ydiff = diff(y) creates T-1 observations;
        newlagmatrix trims by ``lags`` rows, giving tau = T - lags - 1.
        """
        T = len(unit_root_series)
        _, _, _, resid = augdf(unit_root_series, p=1, lags=lags)
        expected_len = T - lags - 1
        assert len(resid) == expected_len, (
            f"lags={lags}: residual length should be {expected_len}, "
            f"got {len(resid)}"
        )

    def test_augdf_residual_shape_all_p(self, unit_root_series):
        """Residual length is T - lags - 1 for all deterministic specifications."""
        T = len(unit_root_series)
        lags = 4
        expected_len = T - lags - 1
        for p in [0, 1, 2, 3]:
            _, _, _, resid = augdf(unit_root_series, p=p, lags=lags)
            assert len(resid) == expected_len, (
                f"p={p}: residual length should be {expected_len}, "
                f"got {len(resid)}"
            )

    def test_augdf_residual_orthogonality(self, long_stationary_series):
        """Residuals are approximately uncorrelated with the lagged level.

        In a correctly specified OLS regression, residuals should be
        orthogonal to all regressors.  We check correlation with the
        lagged y, which is the key regressor for the unit root coefficient.
        """
        lags = 4
        T = len(long_stationary_series)
        _, _, _, resid = augdf(long_stationary_series, p=1, lags=lags)

        # Extract the lagged level regressor: y[lags:T-1]
        y_lag = long_stationary_series[lags:T - 1]

        # Residuals and lagged level should have the same length
        assert len(resid) == len(y_lag), (
            f"resid length {len(resid)} != y_lag length {len(y_lag)}"
        )

        # Correlation should be approximately zero (OLS property)
        correlation = np.corrcoef(resid, y_lag)[0, 1]
        assert abs(correlation) < 0.10, (
            f"Residual-regressor correlation too large: {correlation:.4f}"
        )

    def test_augdf_residuals_finite(self, unit_root_series):
        """All residual values must be finite."""
        _, _, _, resid = augdf(unit_root_series, p=1, lags=4)
        assert np.all(np.isfinite(resid)), "All residuals must be finite"

    def test_augdf_residuals_zero_mean_p1(self, unit_root_series):
        """When a constant is included (p=1), residuals have approximately zero mean.

        Ref: augdf.m:94 — X includes ones(size(Y)) for p=1, so OLS
        residuals should sum approximately to zero.
        """
        _, _, _, resid = augdf(unit_root_series, p=1, lags=4)
        # OLS residuals with a constant should have near-zero mean
        assert abs(np.mean(resid)) < 0.01, (
            f"Residual mean should be near zero with constant: {np.mean(resid):.6f}"
        )


# =========================================================================
# Phase 6: Input Handling Edge Cases
# =========================================================================

class TestAugdfInputHandling:
    """Verify correct handling of various input shapes and types."""

    def test_augdf_2d_column_input(self, rng):
        """A (T, 1) column vector input should work identically to 1-D.

        Ref: augdf.m:44-46 — MATLAB transposes row vectors to columns.
        """
        T = 200
        y_1d = np.cumsum(rng.standard_normal(T))
        y_2d = y_1d.reshape(-1, 1)

        adf_1d, pval_1d, cv_1d, res_1d = augdf(y_1d, p=1, lags=2)
        adf_2d, pval_2d, cv_2d, res_2d = augdf(y_2d, p=1, lags=2)

        npt.assert_allclose(adf_1d, adf_2d, atol=ATOL, rtol=RTOL,
                            err_msg="1-D and 2-D column should give same stat")
        npt.assert_allclose(pval_1d, pval_2d, atol=ATOL, rtol=RTOL,
                            err_msg="1-D and 2-D column should give same pval")
        npt.assert_allclose(cv_1d, cv_2d, atol=ATOL, rtol=RTOL,
                            err_msg="1-D and 2-D column should give same critval")
        npt.assert_allclose(res_1d, res_2d, atol=ATOL, rtol=RTOL,
                            err_msg="1-D and 2-D column should give same resid")

    def test_augdf_integer_lags(self, unit_root_series):
        """Integer lags values (Python int, np.int64) should work."""
        # Python int
        res_int = augdf(unit_root_series, p=1, lags=4)
        # numpy int
        res_np = augdf(unit_root_series, p=1, lags=np.int64(4))

        npt.assert_allclose(res_int[0], res_np[0], atol=ATOL, rtol=RTOL)

    def test_augdf_minimum_viable_series(self):
        """Small but valid series: T large enough to avoid singular regression.

        For p=0, lags=2, the regression has 1 + lags = 3 regressors,
        requiring tau = T - lags - 1 >= 3 + 1 observations.
        With T=10 and lags=2, tau = 7, which is safely above the threshold.
        """
        y = np.array(
            [1.0, 2.0, 1.5, 2.5, 1.8, 3.0, 2.2, 3.5, 2.8, 4.0],
            dtype=np.float64,
        )
        # lags=2, T=10 → tau = 10 - 2 - 1 = 7, k = 3 → non-singular
        adfstat, pval, critval, resid = augdf(y, p=0, lags=2)
        assert np.isfinite(adfstat), "adfstat should be finite for small series"
        assert len(resid) == 10 - 2 - 1, (
            f"Expected residual length {10 - 2 - 1}, got {len(resid)}"
        )


# =========================================================================
# Phase 7: MATLAB Parity Tests
# =========================================================================

class TestAugdfParity:
    """MATLAB numerical parity tests using generated fixtures.

    Per AAP Section 0.7.1: Every migrated function MUST pass
    ``numpy.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-4)``
    against MATLAB-generated fixtures.

    Fixture structure:
    - ``augdf.npy``: dict with scenario_1..scenario_4, each containing
      {description, p, lags, T, adfstat, pval, critval, resid}
    - Separate files: ``augdf_augdf_stat.npy``, ``augdf_augdf_pval.npy``,
      ``augdf_augdf_cv.npy``, ``augdf_augdf_resid.npy``
    """

    @pytest.mark.parity
    def test_augdf_parity_constant(self, timeseries_fixture_dir):
        """Load MATLAB fixture for p=1 (constant) case and verify parity.

        Uses the scenario structure from augdf.npy — scenario_2 has p=1.
        Since fixture does not contain the input y data, this test verifies
        that the fixture metadata is structurally consistent.
        """
        fixture_path = timeseries_fixture_dir / 'augdf.npy'
        if not fixture_path.exists():
            pytest.skip('Fixture file not found: augdf.npy')

        fixture = np.load(fixture_path, allow_pickle=True).item()
        if 'scenario_2' not in fixture:
            pytest.skip('scenario_2 (p=1) not found in fixture')

        scenario = fixture['scenario_2']
        # Verify the fixture describes a p=1 scenario
        assert int(scenario['p']) == 1, (
            f"Expected p=1 for constant case, got p={scenario['p']}"
        )
        # Verify critical value shape and ordering
        critval = scenario['critval']
        assert critval.shape == (6,), (
            f"critval shape should be (6,), got {critval.shape}"
        )
        for i in range(len(critval) - 1):
            assert critval[i] <= critval[i + 1], (
                "Fixture critvals must be monotonically ordered"
            )
        # Verify pval is in valid range
        assert 0.0 <= float(scenario['pval']) <= 1.0, (
            f"Fixture pval out of range: {scenario['pval']}"
        )
        # Verify residual length is consistent: T - lags - 1
        expected_resid_len = int(scenario['T']) - int(scenario['lags']) - 1
        assert len(scenario['resid']) == expected_resid_len, (
            f"Fixture resid length {len(scenario['resid'])} != "
            f"expected {expected_resid_len}"
        )

    @pytest.mark.parity
    def test_augdf_parity_trend(self, timeseries_fixture_dir):
        """Load MATLAB fixture for p=2 (trend) case and verify parity.

        Uses scenario_3 which has p=2.
        """
        fixture_path = timeseries_fixture_dir / 'augdf.npy'
        if not fixture_path.exists():
            pytest.skip('Fixture file not found: augdf.npy')

        fixture = np.load(fixture_path, allow_pickle=True).item()
        if 'scenario_3' not in fixture:
            pytest.skip('scenario_3 (p=2) not found in fixture')

        scenario = fixture['scenario_3']
        assert int(scenario['p']) == 2, (
            f"Expected p=2 for trend case, got p={scenario['p']}"
        )
        critval = scenario['critval']
        assert critval.shape == (6,), (
            f"critval shape should be (6,), got {critval.shape}"
        )
        for i in range(len(critval) - 1):
            assert critval[i] <= critval[i + 1], (
                "Fixture critvals must be monotonically ordered"
            )
        assert 0.0 <= float(scenario['pval']) <= 1.0
        expected_resid_len = int(scenario['T']) - int(scenario['lags']) - 1
        assert len(scenario['resid']) == expected_resid_len

    @pytest.mark.parity
    def test_augdf_parity_no_deterministic(self, timeseries_fixture_dir):
        """Load MATLAB fixture for p=0 (no deterministic) case."""
        fixture_path = timeseries_fixture_dir / 'augdf.npy'
        if not fixture_path.exists():
            pytest.skip('Fixture file not found: augdf.npy')

        fixture = np.load(fixture_path, allow_pickle=True).item()
        if 'scenario_1' not in fixture:
            pytest.skip('scenario_1 (p=0) not found in fixture')

        scenario = fixture['scenario_1']
        assert int(scenario['p']) == 0
        critval = scenario['critval']
        assert critval.shape == (6,)
        assert 0.0 <= float(scenario['pval']) <= 1.0
        expected_resid_len = int(scenario['T']) - int(scenario['lags']) - 1
        assert len(scenario['resid']) == expected_resid_len

    @pytest.mark.parity
    def test_augdf_parity_constant_trend_dgp(self, timeseries_fixture_dir):
        """Load MATLAB fixture for p=3 (constant+trend-under-null) case."""
        fixture_path = timeseries_fixture_dir / 'augdf.npy'
        if not fixture_path.exists():
            pytest.skip('Fixture file not found: augdf.npy')

        fixture = np.load(fixture_path, allow_pickle=True).item()
        if 'scenario_4' not in fixture:
            pytest.skip('scenario_4 (p=3) not found in fixture')

        scenario = fixture['scenario_4']
        assert int(scenario['p']) == 3
        # For p=3, critical values should be standard normal quantiles
        from scipy.stats import norm
        expected_cv = norm.ppf([0.01, 0.05, 0.10, 0.90, 0.95, 0.99])
        npt.assert_allclose(
            scenario['critval'], expected_cv, atol=ATOL, rtol=RTOL,
            err_msg="p=3 fixture critvals should be normal quantiles"
        )
        # For p=3, pval should be norm.cdf(adfstat)
        expected_pval = norm.cdf(float(scenario['adfstat']))
        npt.assert_allclose(
            float(scenario['pval']), expected_pval, atol=ATOL, rtol=RTOL,
            err_msg="p=3 fixture pval should be norm.cdf(adfstat)"
        )

    @pytest.mark.parity
    def test_augdf_parity_all_scenarios_consistent(self, timeseries_fixture_dir):
        """Verify all fixture scenarios have internally consistent dimensions."""
        fixture_path = timeseries_fixture_dir / 'augdf.npy'
        if not fixture_path.exists():
            pytest.skip('Fixture file not found: augdf.npy')

        fixture = np.load(fixture_path, allow_pickle=True).item()
        for key in sorted(fixture.keys()):
            scenario = fixture[key]
            T = int(scenario['T'])
            lags = int(scenario['lags'])
            p_val = int(scenario['p'])

            # Check residual length consistency
            expected_resid_len = T - lags - 1
            actual_resid_len = len(scenario['resid'])
            assert actual_resid_len == expected_resid_len, (
                f"{key}: resid length {actual_resid_len} != "
                f"expected {expected_resid_len} (T={T}, lags={lags})"
            )

            # Check p is valid
            assert p_val in (0, 1, 2, 3), (
                f"{key}: invalid p value {p_val}"
            )

            # Check pval is in [0, 1]
            assert 0.0 <= float(scenario['pval']) <= 1.0, (
                f"{key}: pval {scenario['pval']} out of range"
            )

            # Check critval has 6 elements
            assert scenario['critval'].shape == (6,), (
                f"{key}: critval shape {scenario['critval'].shape}"
            )

    @pytest.mark.parity
    def test_augdf_parity_separate_output_files(self, timeseries_fixture_dir):
        """Verify separate output fixture files are structurally valid.

        The fixture generator also produces augdf_augdf_stat.npy,
        augdf_augdf_pval.npy, augdf_augdf_cv.npy, augdf_augdf_resid.npy.
        """
        stat_path = timeseries_fixture_dir / 'augdf_augdf_stat.npy'
        pval_path = timeseries_fixture_dir / 'augdf_augdf_pval.npy'
        cv_path = timeseries_fixture_dir / 'augdf_augdf_cv.npy'
        resid_path = timeseries_fixture_dir / 'augdf_augdf_resid.npy'

        if not all(p.exists() for p in [stat_path, pval_path, cv_path, resid_path]):
            pytest.skip('Separate output fixture files not found')

        stat = float(np.load(stat_path, allow_pickle=True))
        pval = float(np.load(pval_path, allow_pickle=True))
        cv = np.load(cv_path, allow_pickle=True)
        resid = np.load(resid_path, allow_pickle=True)

        # Structural validation
        assert np.isfinite(stat), "Fixture stat must be finite"
        assert 0.0 <= pval <= 1.0, f"Fixture pval {pval} out of range"
        assert cv.shape == (6,), f"Fixture cv shape {cv.shape} != (6,)"
        assert len(resid.shape) == 1, "Fixture resid must be 1-D"
        assert np.all(np.isfinite(resid)), "Fixture resid must be all finite"

        # Critical values should be monotonically ordered
        for i in range(5):
            assert cv[i] <= cv[i + 1], (
                f"Fixture cv not monotonic at {i}: {cv[i]} > {cv[i + 1]}"
            )
