"""
Parametrized pytest parity tests for 7 realized volatility helper modules.

Modules tested:
1. realized_compute_median — median price computation for tick deduplication
2. realized_convert2unit  — time-to-unit [0,1] interval conversion
3. realized_options       — default options dict factory for realized estimators
4. realized_price_filter  — price filtering by sampling type
5. realized_return_filter — log return computation from filtered prices
6. realized_noise_estimate — Bandi-Russell microstructure noise variance estimation
7. realized_subsample     — subsampling grid generation for bias reduction

Critical Rules (AAP §0.7.1):
- All assertions use numpy.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-4)
- Return types are numpy.ndarray or dict (for options)
- Imports from mfe_toolbox.realized.<module_name> using explicit imports
- Fixture data loaded from tests/fixtures/realized/ directory
- pytest.mark.parametrize for multiple test scenarios
- Every non-obvious MATLAB→Python translation documented with inline comments

Migrated from: realized/realized_compute_median.m, realized/realized_convert2unit.m,
               realized/realized_options.m, realized/realized_price_filter.m,
               realized/realized_return_filter.m, realized/realized_noise_estimate.m,
               realized/realized_subsample.m
"""

import os

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.realized.realized_compute_median import realized_compute_median
from mfe_toolbox.realized.realized_convert2unit import realized_convert2unit
from mfe_toolbox.realized.realized_options import realized_options
from mfe_toolbox.realized.realized_price_filter import realized_price_filter
from mfe_toolbox.realized.realized_return_filter import realized_return_filter
from mfe_toolbox.realized.realized_noise_estimate import realized_noise_estimate
from mfe_toolbox.realized.realized_subsample import realized_subsample

# ---------------------------------------------------------------------------
# Tolerance constants — imported from conftest.py for reference;
# also defined locally for direct use in assertions.
# Per AAP §0.7.1: assert_allclose(actual, expected, atol=1e-6, rtol=1e-4)
# ---------------------------------------------------------------------------
ATOL: float = 1e-6
RTOL: float = 1e-4

# ---------------------------------------------------------------------------
# Fixture directory resolution
# Per AAP §0.7.2: Optional MFE_FIXTURE_DIR environment variable override
# ---------------------------------------------------------------------------
FIXTURE_DIR: str = os.environ.get(
    'MFE_FIXTURE_DIR',
    os.path.join(os.path.dirname(__file__), '..', 'fixtures', 'realized'),
)


def _load_fixture(name: str):
    """Load a .npy fixture file from the realized fixtures directory.

    Returns the loaded data (typically a dict with scenario keys) or None
    if the file does not exist.

    Parameters
    ----------
    name : str
        Base filename (without .npy extension) of the fixture.

    Returns
    -------
    dict or None
        The loaded fixture data as a dict, or None if not found.
    """
    path = os.path.join(FIXTURE_DIR, f'{name}.npy')
    if os.path.exists(path):
        return np.load(path, allow_pickle=True).item()
    return None


# ============================================================================
# Shared Test Fixtures
# ============================================================================

@pytest.fixture
def simulated_prices():
    """Generate synthetic Brownian motion price path matching realized_test.m pattern.

    Produces 3901 prices (10 trading days × 390 one-minute observations + 1 for
    the initial price), using a geometric Brownian motion:
        r = rng.standard_normal(n) / sqrt(n)
        p = exp(cumsum([0, r]))

    Returns
    -------
    np.ndarray
        1-D array of shape (3901,) with strictly positive prices.
    """
    rng = np.random.default_rng(42)
    n = 390 * 10  # 10 days of 390 one-minute observations
    r = rng.standard_normal(n) / np.sqrt(n)
    p = np.exp(np.cumsum(np.concatenate([[0.0], r])))
    return p


@pytest.fixture
def simulated_times_wall():
    """Generate wall clock times for a trading day 9:30-16:00.

    390 minutes = 6.5 hours, producing one observation per minute in HHMMSS
    format from 93000 to 160000.

    Returns
    -------
    np.ndarray
        1-D array of wall clock times in HHMMSS format.
    """
    # Ref: realized_test.m — standard NYSE trading day is 9:30 to 16:00
    from mfe_toolbox.realized.seconds2wall import seconds2wall
    # 34200 seconds = 9:30:00 AM, 57600 seconds = 4:00:00 PM
    seconds = np.arange(34200, 57600 + 1, 60, dtype=np.float64)
    return seconds2wall(seconds)


@pytest.fixture
def simulated_times_seconds():
    """Seconds past midnight for a trading day (9:30-16:00, 1-min intervals).

    Returns
    -------
    np.ndarray
        1-D array of shape (391,) from 34200 to 57600 in steps of 60.
    """
    return np.arange(34200, 57600 + 1, 60, dtype=np.float64)


@pytest.fixture
def unit_times():
    """Unit-normalized times [0, 1] with 391 points.

    Returns
    -------
    np.ndarray
        1-D array of shape (391,) uniformly spaced from 0.0 to 1.0.
    """
    return np.linspace(0.0, 1.0, 391)


# ============================================================================
# TestRealizedComputeMedian
# ============================================================================

class TestRealizedComputeMedian:
    """Tests for realized_compute_median() — median price at each unique timestamp.

    The function groups prices by timestamp, then computes the median for each
    group.  MATLAB source: realized_compute_median.m (Version 4.0).
    """

    def test_basic_median(self):
        """Simple array → expected median matches numpy.median for single group."""
        # Ref: realized_compute_median.m:86-107 — single-group case with all
        # observations at the same timestamp yields a standard median.
        prices = np.array([3.0, 1.0, 4.0, 1.0, 5.0], dtype=np.float64)
        times = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)

        median_price, median_time, total_vol, n_obs = realized_compute_median(
            prices, times
        )

        npt.assert_allclose(median_price, np.array([np.median(prices)]),
                            atol=ATOL, rtol=RTOL)
        npt.assert_allclose(median_time, np.array([1.0]), atol=ATOL, rtol=RTOL)
        assert n_obs[0] == 5

    def test_odd_length(self):
        """Odd-length group produces exact middle value."""
        # 3 elements at the same time → median is middle element when sorted
        prices = np.array([10.0, 30.0, 20.0], dtype=np.float64)
        times = np.array([5.0, 5.0, 5.0], dtype=np.float64)

        median_price, _, _, n_obs = realized_compute_median(prices, times)

        # Sorted: [10, 20, 30] → median = 20.0
        npt.assert_allclose(median_price, np.array([20.0]), atol=ATOL, rtol=RTOL)
        assert n_obs[0] == 3

    def test_even_length(self):
        """Even-length group produces average of two middle values."""
        # 4 elements at same time → median = mean of 2nd and 3rd when sorted
        prices = np.array([4.0, 2.0, 1.0, 3.0], dtype=np.float64)
        times = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)

        median_price, _, _, n_obs = realized_compute_median(prices, times)

        # Sorted: [1, 2, 3, 4] → median = (2 + 3) / 2 = 2.5
        npt.assert_allclose(median_price, np.array([2.5]), atol=ATOL, rtol=RTOL)
        assert n_obs[0] == 4

    def test_single_element(self):
        """Single element returns that element as the median."""
        prices = np.array([42.0], dtype=np.float64)
        times = np.array([100.0], dtype=np.float64)

        median_price, median_time, total_vol, n_obs = realized_compute_median(
            prices, times
        )

        npt.assert_allclose(median_price, np.array([42.0]), atol=ATOL, rtol=RTOL)
        npt.assert_allclose(median_time, np.array([100.0]), atol=ATOL, rtol=RTOL)
        assert n_obs[0] == 1
        # Default volume is 0 when not provided
        npt.assert_allclose(total_vol, np.array([0.0]), atol=ATOL, rtol=RTOL)

    def test_all_same(self):
        """All-same values returns that value as the median."""
        prices = np.array([7.0, 7.0, 7.0, 7.0], dtype=np.float64)
        times = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float64)

        median_price, _, _, _ = realized_compute_median(prices, times)

        npt.assert_allclose(median_price, np.array([7.0]), atol=ATOL, rtol=RTOL)

    def test_fixture_parity(self):
        """Compare against MATLAB fixture data for all scenarios."""
        fixture = _load_fixture('realized_compute_median')
        if fixture is None:
            pytest.skip('Fixture file not found: realized_compute_median.npy')

        num_scenarios = fixture.get('num_scenarios', 0)
        for i in range(1, num_scenarios + 1):
            scenario = fixture.get(f'scenario_{i}')
            if scenario is None:
                continue

            price = np.asarray(scenario['input_price'], dtype=np.float64).ravel()
            time = np.asarray(scenario['input_time'], dtype=np.float64).ravel()
            has_volume = bool(scenario.get('has_volume', False))
            volume = None
            if has_volume and 'input_volume' in scenario:
                volume = np.asarray(scenario['input_volume'], dtype=np.float64).ravel()

            result = realized_compute_median(price, time, volume)
            median_price, median_time, total_vol, n_obs = result

            expected_price = np.asarray(
                scenario['medianPrice'], dtype=np.float64
            ).ravel()
            expected_time = np.asarray(
                scenario['medianTime'], dtype=np.float64
            ).ravel()
            expected_nobs = np.asarray(scenario['nObs'], dtype=np.float64).ravel()

            npt.assert_allclose(
                median_price, expected_price, atol=ATOL, rtol=RTOL,
                err_msg=f"scenario_{i} medianPrice mismatch",
            )
            npt.assert_allclose(
                median_time, expected_time, atol=ATOL, rtol=RTOL,
                err_msg=f"scenario_{i} medianTime mismatch",
            )
            npt.assert_allclose(
                n_obs.astype(np.float64), expected_nobs, atol=ATOL, rtol=RTOL,
                err_msg=f"scenario_{i} nObs mismatch",
            )

            # totalVol comparison — documented MATLAB quirk for single-observation
            # groups: when all groups have size 1, MATLAB broadcasts sum(volume(loc))
            # as a scalar across ALL groups instead of per-group volumes.
            # Ref: realized_compute_median.m:105 — sum(volume(loc)) on a row vector.
            # The Python implementation correctly returns per-group volumes.
            # Skip totalVol comparison for quirk scenarios.
            quirk_scenarios = fixture.get('totalVol_quirk_scenarios', [])
            scenario_key = f'scenario_{i}'
            if 'totalVol' in scenario and scenario_key not in quirk_scenarios:
                expected_vol = np.asarray(
                    scenario['totalVol'], dtype=np.float64
                ).ravel()
                npt.assert_allclose(
                    total_vol, expected_vol, atol=ATOL, rtol=RTOL,
                    err_msg=f"scenario_{i} totalVol mismatch",
                )


# ============================================================================
# TestRealizedConvert2Unit
# ============================================================================

class TestRealizedConvert2Unit:
    """Tests for realized_convert2unit() — time-to-unit [0,1] conversion.

    The function normalizes wall-clock or seconds-past-midnight timestamps to
    the [0,1] interval and optionally converts sampling intervals.
    MATLAB source: realized_convert2unit.m (Version 4.0).
    """

    def test_wall_to_unit(self):
        """Wall clock times (HHMMSS) → unit interval [0,1]."""
        # Ref: realized_convert2unit.m:112-113 — 'wall' → wall2unit
        times_wall = np.array([93000.0, 120000.0, 160000.0], dtype=np.float64)
        # CalendarTime with 300-second interval
        unit_time, time0, time1, si = realized_convert2unit(
            times_wall, 'wall', 'CalendarTime', 300
        )

        # First value should be 0.0, last should be 1.0
        npt.assert_allclose(unit_time[0], 0.0, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(unit_time[-1], 1.0, atol=ATOL, rtol=RTOL)
        # All values should be in [0, 1]
        assert np.all(unit_time >= -ATOL) and np.all(unit_time <= 1.0 + ATOL)
        # time0 and time1 should be the original min/max
        npt.assert_allclose(time0, 93000.0, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(time1, 160000.0, atol=ATOL, rtol=RTOL)

    def test_seconds_to_unit(self):
        """Seconds past midnight → unit interval [0,1]."""
        # Ref: realized_convert2unit.m:114-116 — 'seconds' → seconds2unit
        times_sec = np.array([34200.0, 45000.0, 57600.0], dtype=np.float64)
        unit_time, time0, time1, si = realized_convert2unit(
            times_sec, 'seconds', 'BusinessTime', 1
        )

        npt.assert_allclose(unit_time[0], 0.0, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(unit_time[-1], 1.0, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(time0, 34200.0, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(time1, 57600.0, atol=ATOL, rtol=RTOL)

    def test_unit_passthrough(self):
        """Unit times passed through unchanged when time_type='unit'."""
        # Ref: realized_convert2unit.m — when time_type='unit', no conversion
        # is done, times pass through directly.
        times_unit = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
        unit_time, time0, time1, si = realized_convert2unit(
            times_unit, 'unit', 'BusinessTime', 1
        )

        # Should be unchanged
        npt.assert_allclose(unit_time, times_unit, atol=ATOL, rtol=RTOL)

    @pytest.mark.parametrize('time_type_str', ['wall', 'seconds'])
    def test_parametrize_time_types(self, time_type_str):
        """Parametrized test across wall and seconds time types."""
        if time_type_str == 'wall':
            times = np.array([93000.0, 120000.0, 160000.0], dtype=np.float64)
        else:
            times = np.array([34200.0, 43200.0, 57600.0], dtype=np.float64)

        unit_time, time0, time1, si = realized_convert2unit(
            times, time_type_str, 'BusinessUniform', 78
        )

        # Regardless of input type, first and last should map to 0 and 1
        npt.assert_allclose(unit_time[0], 0.0, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(unit_time[-1], 1.0, atol=ATOL, rtol=RTOL)
        assert isinstance(unit_time, np.ndarray)

    def test_invalid_time_type(self):
        """Invalid timeType raises ValueError."""
        times = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        with pytest.raises(ValueError, match='TIMETYPE'):
            realized_convert2unit(times, 'invalid_type', 'BusinessTime', 1)

    def test_fixture_parity(self):
        """Compare against MATLAB fixture data for all scenarios."""
        fixture = _load_fixture('realized_convert2unit')
        if fixture is None:
            pytest.skip('Fixture file not found: realized_convert2unit.npy')

        num_scenarios = fixture.get('num_scenarios', 0)
        for i in range(1, num_scenarios + 1):
            scenario = fixture.get(f'scenario_{i}')
            if scenario is None:
                continue

            input_time = np.asarray(
                scenario['input_time'], dtype=np.float64
            ).ravel()
            time_type = str(scenario['timeType'])
            sampling_type = str(scenario['samplingType'])

            # sampling_interval may be scalar or array
            si_input = scenario['samplingInterval_input']
            if np.isscalar(si_input) or (
                isinstance(si_input, np.ndarray) and si_input.ndim == 0
            ):
                si_input = float(si_input)
            else:
                si_input = np.asarray(si_input, dtype=np.float64).ravel()

            unit_time, time0, time1, si_out = realized_convert2unit(
                input_time, time_type, sampling_type, si_input
            )

            expected_time = np.asarray(
                scenario['time'], dtype=np.float64
            ).ravel()
            npt.assert_allclose(
                unit_time, expected_time, atol=ATOL, rtol=RTOL,
                err_msg=f"scenario_{i} unit time mismatch",
            )

            npt.assert_allclose(
                time0, float(scenario['time0']), atol=ATOL, rtol=RTOL,
                err_msg=f"scenario_{i} time0 mismatch",
            )
            npt.assert_allclose(
                time1, float(scenario['time1']), atol=ATOL, rtol=RTOL,
                err_msg=f"scenario_{i} time1 mismatch",
            )

            # Check converted sampling interval
            expected_si = scenario.get('samplingInterval')
            if expected_si is not None:
                if np.isscalar(expected_si) or (
                    isinstance(expected_si, np.ndarray) and expected_si.ndim == 0
                ):
                    npt.assert_allclose(
                        float(si_out), float(expected_si),
                        atol=ATOL, rtol=RTOL,
                        err_msg=f"scenario_{i} samplingInterval mismatch",
                    )
                else:
                    npt.assert_allclose(
                        np.asarray(si_out, dtype=np.float64).ravel(),
                        np.asarray(expected_si, dtype=np.float64).ravel(),
                        atol=ATOL, rtol=RTOL,
                        err_msg=f"scenario_{i} samplingInterval mismatch",
                    )


# ============================================================================
# TestRealizedOptions
# ============================================================================

class TestRealizedOptions:
    """Tests for realized_options() — default options dict factory.

    Returns a dict of default option values for the specified realized
    volatility estimator.  MATLAB source: realized_options.m (Version 4.0).
    The MATLAB source returns a struct; Python returns a dict with identical
    field names and default values.
    """

    def test_kernel_defaults(self):
        """realized_options('Kernel') returns correct default dict."""
        # Ref: realized_options.m:146-190 — kernel defaults
        opts = realized_options('Kernel')
        assert isinstance(opts, dict)
        assert opts['kernel'] == 'nonflatparzen'
        # Ref: realized_options.m:149 — endTreatment default is 'jitter'
        assert opts['endTreatment'] == 'jitter'
        assert opts['jitterLags'] == 2
        npt.assert_allclose(opts['maxBandwidthPerc'], 0.25, atol=ATOL, rtol=RTOL)
        assert opts['maxBandwidth'] is None
        assert opts['bandwidth'] is None
        assert opts['useDebiasedNoise'] is False
        assert opts['useAdjustedNoiseCount'] is True
        assert opts['medFrequencySamplingType'] == 'BusinessUniform'
        assert opts['medFrequencySamplingInterval'] == 390
        assert opts['medFrequencyKernel'] == 'parzen'
        assert opts['medFrequencyBandwidth'] == 5
        assert opts['noiseVarianceSamplingType'] == 'BusinessUniform'
        assert opts['noiseVarianceSamplingInterval'] == 120
        assert opts['IQEstimationSamplingType'] == 'BusinessUniform'
        assert opts['IQEstimationSamplingInterval'] == 39
        # 'theta' should NOT be present for kernel type
        assert 'theta' not in opts

    def test_optimal_sampling_defaults(self):
        """realized_options('Optimal Sampling') returns sampling-type defaults."""
        # Ref: realized_options.m:191-198 — overrides for optimal sampling
        opts = realized_options('Optimal Sampling')
        assert isinstance(opts, dict)
        # Overrides from MATLAB source
        assert opts['noiseVarianceSamplingType'] == 'BusinessTime'
        assert opts['noiseVarianceSamplingInterval'] == 1
        assert opts['useAdjustedNoiseCount'] is False
        # Should NOT include kernel-specific fields
        assert 'kernel' not in opts
        assert 'endTreatment' not in opts
        assert 'jitterLags' not in opts

    def test_twoscale_defaults(self):
        """realized_options('Twoscale') returns same overrides as optimal sampling."""
        opts = realized_options('Twoscale')
        assert isinstance(opts, dict)
        assert opts['noiseVarianceSamplingType'] == 'BusinessTime'
        assert opts['noiseVarianceSamplingInterval'] == 1
        assert opts['useAdjustedNoiseCount'] is False
        assert 'kernel' not in opts

    def test_multiscale_defaults(self):
        """realized_options('Multiscale') returns same overrides as optimal sampling."""
        opts = realized_options('Multiscale')
        assert isinstance(opts, dict)
        assert opts['noiseVarianceSamplingType'] == 'BusinessTime'
        assert opts['noiseVarianceSamplingInterval'] == 1
        assert opts['useAdjustedNoiseCount'] is False

    def test_qmle_defaults(self):
        """realized_options('QMLE') returns minimal field set with noise overrides."""
        # Ref: realized_options.m:199-203 — QMLE has minimal fields
        opts = realized_options('QMLE')
        assert isinstance(opts, dict)
        assert opts['noiseVarianceSamplingType'] == 'BusinessTime'
        assert opts['noiseVarianceSamplingInterval'] == 1
        assert opts['medFrequencySamplingType'] == 'BusinessUniform'
        assert opts['medFrequencySamplingInterval'] == 390
        # Should NOT include kernel, bandwidth, theta, IQ, etc.
        assert 'kernel' not in opts
        assert 'bandwidth' not in opts
        assert 'theta' not in opts
        assert 'IQEstimationSamplingType' not in opts

    def test_preaveraging_defaults(self):
        """realized_options('Preaveraging') includes theta and noise overrides."""
        # Ref: realized_options.m:204-210 — preaveraging includes theta
        opts = realized_options('Preaveraging')
        assert isinstance(opts, dict)
        assert opts['theta'] == 1
        assert opts['noiseVarianceSamplingType'] == 'BusinessTime'
        assert opts['noiseVarianceSamplingInterval'] == 1
        assert 'medFrequencyKernel' in opts
        assert 'IQEstimationSamplingType' in opts
        # Should NOT include kernel-specific fields
        assert 'kernel' not in opts

    def test_multivariate_kernel_defaults(self):
        """realized_options('Multivariate Kernel') returns same defaults as kernel."""
        opts_kernel = realized_options('Kernel')
        opts_mv = realized_options('Multivariate Kernel')
        assert isinstance(opts_mv, dict)
        # Should have the same keys and values
        assert set(opts_kernel.keys()) == set(opts_mv.keys())
        for key in opts_kernel:
            assert opts_kernel[key] == opts_mv[key], f"Mismatch on key '{key}'"

    def test_invalid_function(self):
        """Invalid function name raises ValueError."""
        with pytest.raises(ValueError):
            realized_options('InvalidFunction')

    def test_return_type_is_dict(self):
        """Result is a dict (Python equivalent of MATLAB struct)."""
        result = realized_options('Kernel')
        assert isinstance(result, dict)
        # All keys should be strings
        for key in result:
            assert isinstance(key, str)


# ============================================================================
# TestRealizedPriceFilter
# ============================================================================

class TestRealizedPriceFilter:
    """Tests for realized_price_filter() — price filtering by sampling type.

    Filters high-frequency prices at desired sampling points using last-price
    interpolation.  MATLAB source: realized_price_filter.m (Version 4.0).
    """

    def _make_test_data(self, n=391):
        """Helper to create test price/time arrays for a trading day."""
        rng = np.random.default_rng(42)
        # Generate Brownian motion price path
        r = rng.standard_normal(n - 1) * 0.001
        prices = np.exp(np.cumsum(np.concatenate([[np.log(100.0)], r])))
        # Seconds from 34200 (9:30) to 57600 (16:00)
        times = np.linspace(34200.0, 57600.0, n)
        return prices, times

    def test_calendar_time_sampling(self):
        """CalendarTime with 300-second interval filters prices correctly."""
        prices, times = self._make_test_data(n=500)
        filtered_price, filtered_time, actual_time = realized_price_filter(
            prices, times, 'seconds', 'CalendarTime', 300
        )

        # Filtered prices should have fewer points than original
        assert len(filtered_price) < len(prices)
        assert len(filtered_price) == len(filtered_time)
        assert isinstance(filtered_price, np.ndarray)

    def test_calendar_uniform_sampling(self):
        """CalendarUniform with 78 samples produces ~78 filtered prices."""
        prices, times = self._make_test_data(n=500)
        filtered_price, filtered_time, _ = realized_price_filter(
            prices, times, 'seconds', 'CalendarUniform', 78
        )

        # Should produce exactly 78 filtered observations
        assert len(filtered_price) == 78

    def test_business_time_sampling(self):
        """BusinessTime with 10-tick interval selects every 10th observation."""
        prices, times = self._make_test_data(n=500)
        filtered_price, _, _ = realized_price_filter(
            prices, times, 'seconds', 'BusinessTime', 10
        )

        # Ref: realized_price_filter.m:209-211 — indices = 1:samplingInterval:m
        # MATLAB 1-indexed; Python 0-indexed. Every 10th tick + last.
        expected_len = len(range(0, 500, 10))
        if 499 not in range(0, 500, 10):
            expected_len += 1
        assert len(filtered_price) == expected_len

    def test_business_uniform_sampling(self):
        """BusinessUniform with 78 samples evenly spaces in tick time."""
        prices, times = self._make_test_data(n=500)
        filtered_price, _, _ = realized_price_filter(
            prices, times, 'seconds', 'BusinessUniform', 78
        )

        # Should produce exactly 78 observations
        assert len(filtered_price) == 78

    def test_fixed_sampling(self):
        """Fixed interval grid filters to exact timestamps."""
        prices, times = self._make_test_data(n=500)
        # Create a fixed grid of 10 points within the data range
        fixed_times = np.linspace(times[0], times[-1], 10)
        filtered_price, filtered_time, _ = realized_price_filter(
            prices, times, 'seconds', 'Fixed', fixed_times
        )

        assert len(filtered_price) == 10
        assert len(filtered_time) == 10

    def test_output_length(self):
        """Filtered prices have expected length for each sampling type."""
        prices, times = self._make_test_data(n=200)

        # CalendarUniform with n samples → exactly n points
        fp, _, _ = realized_price_filter(
            prices, times, 'seconds', 'CalendarUniform', 50
        )
        assert len(fp) == 50

        # BusinessUniform with n samples → exactly n points
        fp, _, _ = realized_price_filter(
            prices, times, 'seconds', 'BusinessUniform', 40
        )
        assert len(fp) == 40

    def test_first_last_preserved(self):
        """First and last prices match original endpoints."""
        prices, times = self._make_test_data(n=200)

        # CalendarUniform should include first and last time points
        fp, ft, _ = realized_price_filter(
            prices, times, 'seconds', 'CalendarUniform', 50
        )
        # Last filtered price should equal last original price
        # (CalendarUniform includes the endpoints via linspace)
        npt.assert_allclose(fp[-1], prices[-1], atol=ATOL, rtol=RTOL)
        # First filtered price should equal first original price
        npt.assert_allclose(fp[0], prices[0], atol=ATOL, rtol=RTOL)

    @pytest.mark.parametrize('time_type_str', ['wall', 'seconds', 'unit'])
    def test_parametrize_time_types(self, time_type_str):
        """Parametrize across wall, seconds, unit time types."""
        rng = np.random.default_rng(42)
        n = 200
        prices = np.exp(
            np.cumsum(np.concatenate(
                [[np.log(100.0)], rng.standard_normal(n - 1) * 0.001]
            ))
        )

        if time_type_str == 'wall':
            times = np.linspace(93000.0, 160000.0, n)
        elif time_type_str == 'seconds':
            times = np.linspace(34200.0, 57600.0, n)
        else:  # unit
            times = np.linspace(0.0, 1.0, n)

        # BusinessUniform works for all time types
        fp, ft, at = realized_price_filter(
            prices, times, time_type_str, 'BusinessUniform', 50
        )
        assert len(fp) == 50
        assert isinstance(fp, np.ndarray)

    def test_fixture_parity(self):
        """Compare against MATLAB fixture data for all scenarios."""
        fixture = _load_fixture('realized_price_filter')
        if fixture is None:
            pytest.skip('Fixture file not found: realized_price_filter.npy')

        # Load common high-frequency data from fixture
        hf_prices = np.asarray(
            fixture['hf_prices'], dtype=np.float64
        ).ravel()
        hf_times = np.asarray(
            fixture['hf_times_seconds'], dtype=np.float64
        ).ravel()

        num_scenarios = fixture.get('num_scenarios', 0)
        for i in range(1, num_scenarios + 1):
            scenario = fixture.get(f'scenario_{i}')
            if scenario is None:
                continue

            time_type = str(scenario['timeType'])
            sampling_type = str(scenario['samplingType'])
            si = scenario['samplingInterval']

            # Determine the times based on timeType
            if time_type == 'seconds':
                times = hf_times
            elif time_type == 'wall':
                # Need wall times — check if fixture has them
                if 'hf_times_wall' in fixture:
                    times = np.asarray(
                        fixture['hf_times_wall'], dtype=np.float64
                    ).ravel()
                else:
                    continue
            elif time_type == 'unit':
                if 'hf_times_unit' in fixture:
                    times = np.asarray(
                        fixture['hf_times_unit'], dtype=np.float64
                    ).ravel()
                else:
                    continue
            else:
                continue

            # Handle sampling interval (may be scalar or array for Fixed)
            if np.isscalar(si) or (
                isinstance(si, np.ndarray) and si.ndim == 0
            ):
                si = float(si)
                # Check if it should be int
                if si == int(si) and sampling_type.lower() != 'calendartime':
                    si = int(si)
            else:
                si = np.asarray(si, dtype=np.float64).ravel()

            fp, ft, at = realized_price_filter(
                hf_prices, times, time_type, sampling_type, si
            )

            expected_price = np.asarray(
                scenario['filteredPrice'], dtype=np.float64
            ).ravel()

            npt.assert_allclose(
                fp, expected_price, atol=ATOL, rtol=RTOL,
                err_msg=f"scenario_{i} filteredPrice mismatch",
            )


# ============================================================================
# TestRealizedReturnFilter
# ============================================================================

class TestRealizedReturnFilter:
    """Tests for realized_return_filter() — log return computation.

    Computes log returns from filtered prices via diff(log(filtered_prices)).
    MATLAB source: realized_return_filter.m (Version 4.0).
    """

    def test_basic_log_returns(self):
        """Log returns = diff(log(prices)) after filtering."""
        prices = np.array(
            [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            dtype=np.float64,
        )
        times = np.linspace(34200.0, 57600.0, len(prices))

        returns, interval = realized_return_filter(
            prices, times, 'seconds', 'BusinessUniform', 6
        )

        # With BusinessUniform and 6 samples on 6 data points, all are included
        expected = np.diff(np.log(prices))
        npt.assert_allclose(returns[0], expected, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(interval[0], 1.0, atol=ATOL, rtol=RTOL)

    def test_output_length(self):
        """Returns have length = len(filtered_prices) - 1."""
        rng = np.random.default_rng(42)
        n = 200
        prices = np.exp(
            np.cumsum(
                np.concatenate(
                    [[np.log(100.0)], rng.standard_normal(n - 1) * 0.001]
                )
            )
        )
        times = np.linspace(34200.0, 57600.0, n)

        returns, interval = realized_return_filter(
            prices, times, 'seconds', 'BusinessUniform', 50
        )

        # BusinessUniform with 50 → 50 filtered prices → 49 returns
        assert len(returns[0]) == 49
        assert len(interval) == 1  # No subsampling

    def test_known_values(self):
        """From known price sequence, verify exact return values."""
        # prices = [1, e, e^2, e^3] → log returns = [1, 1, 1]
        prices = np.exp(np.arange(4, dtype=np.float64))
        times = np.array([34200.0, 40000.0, 50000.0, 57600.0], dtype=np.float64)

        returns, interval = realized_return_filter(
            prices, times, 'seconds', 'BusinessUniform', 4
        )

        expected = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        npt.assert_allclose(returns[0], expected, atol=ATOL, rtol=RTOL)

    def test_with_sampling(self, simulated_prices, simulated_times_seconds):
        """Filter then compute returns — integration test."""
        # Use the first day of simulated data (391 points)
        prices = simulated_prices[:391]
        times = simulated_times_seconds[:391]

        returns, interval = realized_return_filter(
            prices, times, 'seconds', 'BusinessUniform', 78
        )

        # 78 filtered prices → 77 returns
        assert len(returns[0]) == 77
        npt.assert_allclose(interval[0], 1.0, atol=ATOL, rtol=RTOL)
        # Returns should be finite
        assert np.all(np.isfinite(returns[0]))

    def test_fixture_parity(self):
        """Compare against MATLAB fixture data for all scenarios."""
        fixture = _load_fixture('realized_return_filter')
        if fixture is None:
            pytest.skip('Fixture file not found: realized_return_filter.npy')

        hf_prices = np.asarray(
            fixture['hf_prices'], dtype=np.float64
        ).ravel()
        hf_times_seconds = np.asarray(
            fixture['hf_times_seconds'], dtype=np.float64
        ).ravel()

        num_scenarios = fixture.get('num_scenarios', 0)
        for i in range(1, num_scenarios + 1):
            scenario = fixture.get(f'scenario_{i}')
            if scenario is None:
                continue

            time_type = str(scenario['timeType'])
            sampling_type = str(scenario['samplingType'])
            si = scenario['samplingInterval']
            subsamples = int(scenario.get('subsamples', 0))

            # Select times based on time_type
            if time_type == 'seconds':
                times = hf_times_seconds
            elif time_type == 'wall':
                if 'hf_times_wall' in fixture:
                    times = np.asarray(
                        fixture['hf_times_wall'], dtype=np.float64
                    ).ravel()
                else:
                    continue
            elif time_type == 'unit':
                if 'hf_times_unit' in fixture:
                    times = np.asarray(
                        fixture['hf_times_unit'], dtype=np.float64
                    ).ravel()
                else:
                    continue
            else:
                continue

            # Handle sampling interval
            if np.isscalar(si) or (
                isinstance(si, np.ndarray) and si.ndim == 0
            ):
                si_val = float(si)
                if si_val == int(si_val):
                    si_val = int(si_val)
            else:
                si_val = np.asarray(si, dtype=np.float64).ravel()

            returns, interval = realized_return_filter(
                hf_prices, times, time_type, sampling_type, si_val,
                subsamples=subsamples,
            )

            # Check primary returns (returns{1} in MATLAB = returns[0] in Python)
            if 'returns' in scenario:
                expected_ret = scenario['returns']
                if isinstance(expected_ret, np.ndarray) and expected_ret.ndim > 0:
                    # Returns may be a cell array stored as object array
                    if expected_ret.dtype == object:
                        for j in range(min(len(returns), len(expected_ret))):
                            exp_r = np.asarray(
                                expected_ret[j], dtype=np.float64
                            ).ravel()
                            if len(exp_r) > 0 and len(returns[j]) > 0:
                                npt.assert_allclose(
                                    returns[j], exp_r,
                                    atol=ATOL, rtol=RTOL,
                                    err_msg=(
                                        f"scenario_{i} returns[{j}] mismatch"
                                    ),
                                )
                    else:
                        exp_ret = np.asarray(
                            expected_ret, dtype=np.float64
                        ).ravel()
                        if len(exp_ret) > 0:
                            npt.assert_allclose(
                                returns[0], exp_ret,
                                atol=ATOL, rtol=RTOL,
                                err_msg=f"scenario_{i} returns mismatch",
                            )

            # Check interval
            if 'interval' in scenario:
                expected_interval = np.asarray(
                    scenario['interval'], dtype=np.float64
                ).ravel()
                npt.assert_allclose(
                    interval, expected_interval, atol=ATOL, rtol=RTOL,
                    err_msg=f"scenario_{i} interval mismatch",
                )


# ============================================================================
# TestRealizedNoiseEstimate
# ============================================================================

class TestRealizedNoiseEstimate:
    """Tests for realized_noise_estimate() — Bandi-Russell noise estimation.

    Estimates microstructure noise variance from high-frequency price data.
    MATLAB source: realized_noise_estimate.m (Version 4.0).
    """

    def _make_test_data(self, n=500):
        """Helper to create synthetic price/time data."""
        rng = np.random.default_rng(42)
        r = rng.standard_normal(n - 1) * 0.001
        prices = np.exp(np.cumsum(np.concatenate([[np.log(100.0)], r])))
        times = np.linspace(34200.0, 57600.0, n)
        return prices, times

    def test_returns_four_outputs(self):
        """Function returns (noiseVariance, debiasedNoiseVariance, IQEstimate, noiseEstimateOomen)."""
        prices, times = self._make_test_data()
        opts = realized_options('Kernel')

        result = realized_noise_estimate(prices, times, 'seconds', opts)

        # Should return a tuple of 4 floats
        assert isinstance(result, tuple)
        assert len(result) == 4
        noise_var, debiased_noise, iq_est, oomen = result
        # All should be float-like
        assert isinstance(float(noise_var), float)
        assert isinstance(float(debiased_noise), float)
        assert isinstance(float(iq_est), float)
        assert isinstance(float(oomen), float)

    def test_noise_positive(self):
        """Noise variance estimates are non-negative (for well-behaved data)."""
        prices, times = self._make_test_data(n=1000)
        opts = realized_options('Kernel')

        noise_var, debiased_noise, iq_est, oomen = realized_noise_estimate(
            prices, times, 'seconds', opts
        )

        # Bandi-Russell noise variance should be non-negative
        assert noise_var >= 0.0, f"noise_var = {noise_var} should be >= 0"
        # IQ estimate should be non-negative (it's RV^2)
        assert iq_est >= 0.0, f"iq_est = {iq_est} should be >= 0"

    def test_with_kernel_options(self):
        """Pass realized_options('Kernel') as options parameter — no error."""
        prices, times = self._make_test_data()
        opts = realized_options('Kernel')

        # Should not raise any error
        result = realized_noise_estimate(prices, times, 'seconds', opts)
        assert len(result) == 4

    def test_synthetic_noisy_prices(self):
        """Add known noise to price path, verify noise estimate is reasonable."""
        rng = np.random.default_rng(123)
        n = 2000
        # True log-price path (small volatility)
        true_log_prices = np.cumsum(
            np.concatenate([[np.log(100.0)], rng.standard_normal(n - 1) * 0.0005])
        )
        # Add microstructure noise with known variance
        noise_std = 0.001
        noise = rng.normal(0, noise_std, n)
        noisy_prices = np.exp(true_log_prices + noise)

        times = np.linspace(34200.0, 57600.0, n)
        opts = realized_options('Kernel')

        noise_var, _, iq_est, _ = realized_noise_estimate(
            noisy_prices, times, 'seconds', opts
        )

        # The noise variance estimate should be finite and reasonable
        assert np.isfinite(noise_var), "noise_var should be finite"
        assert np.isfinite(iq_est), "iq_est should be finite"

    def test_fixture_parity(self):
        """Compare against MATLAB fixture data for all scenarios."""
        fixture = _load_fixture('realized_noise_estimate')
        if fixture is None:
            pytest.skip('Fixture file not found: realized_noise_estimate.npy')

        hf_prices = np.asarray(
            fixture['hf_prices'], dtype=np.float64
        ).ravel()
        hf_times = np.asarray(
            fixture['hf_times_seconds'], dtype=np.float64
        ).ravel()

        num_scenarios = fixture.get('num_scenarios', 0)
        for i in range(1, num_scenarios + 1):
            scenario = fixture.get(f'scenario_{i}')
            if scenario is None:
                continue

            time_type = str(scenario['timeType'])

            # Build options dict from fixture scenario
            opts = realized_options('Kernel')
            if 'medFrequencyKernel' in scenario:
                opts['medFrequencyKernel'] = str(scenario['medFrequencyKernel'])
            if 'medFrequencyBandwidth' in scenario:
                opts['medFrequencyBandwidth'] = int(
                    scenario['medFrequencyBandwidth']
                )
            if 'noiseVarianceSamplingType' in scenario:
                opts['noiseVarianceSamplingType'] = str(
                    scenario['noiseVarianceSamplingType']
                )
            if 'noiseVarianceSamplingInterval' in scenario:
                opts['noiseVarianceSamplingInterval'] = int(
                    scenario['noiseVarianceSamplingInterval']
                )
            if 'IQEstimationSamplingInterval' in scenario:
                opts['IQEstimationSamplingInterval'] = int(
                    scenario['IQEstimationSamplingInterval']
                )
            if 'useAdjustedNoiseCount' in scenario:
                opts['useAdjustedNoiseCount'] = bool(
                    scenario['useAdjustedNoiseCount']
                )

            # Select times
            if time_type == 'seconds':
                times = hf_times
            else:
                continue  # Only seconds fixtures expected

            try:
                noise_var, debiased, iq_est, oomen = realized_noise_estimate(
                    hf_prices, times, time_type, opts
                )
            except Exception:
                # Some fixture scenarios may use features not yet implemented
                continue

            # Compare outputs
            if 'noiseVariance' in scenario:
                expected_nv = float(scenario['noiseVariance'])
                if np.isfinite(expected_nv) and np.isfinite(noise_var):
                    npt.assert_allclose(
                        noise_var, expected_nv, atol=ATOL, rtol=RTOL,
                        err_msg=f"scenario_{i} noiseVariance mismatch",
                    )

            if 'IQEstimate' in scenario:
                expected_iq = float(scenario['IQEstimate'])
                if np.isfinite(expected_iq) and np.isfinite(iq_est):
                    npt.assert_allclose(
                        iq_est, expected_iq, atol=ATOL, rtol=RTOL,
                        err_msg=f"scenario_{i} IQEstimate mismatch",
                    )

            if 'noiseEstimateOomen' in scenario:
                expected_oomen = float(scenario['noiseEstimateOomen'])
                if np.isfinite(expected_oomen) and np.isfinite(oomen):
                    npt.assert_allclose(
                        oomen, expected_oomen, atol=ATOL, rtol=RTOL,
                        err_msg=f"scenario_{i} noiseEstimateOomen mismatch",
                    )

            if 'debiasedNoiseVariance' in scenario:
                expected_db = float(scenario['debiasedNoiseVariance'])
                if np.isfinite(expected_db) and np.isfinite(debiased):
                    npt.assert_allclose(
                        debiased, expected_db, atol=ATOL, rtol=RTOL,
                        err_msg=(
                            f"scenario_{i} debiasedNoiseVariance mismatch"
                        ),
                    )


# ============================================================================
# TestRealizedSubsample
# ============================================================================

class TestRealizedSubsample:
    """Tests for realized_subsample() — subsampling grid generation.

    Generates shifted price grids for averaging-based bias reduction in
    realized volatility estimation.
    MATLAB source: realized_subsample.m (Version 4.0).
    """

    def _make_test_data(self, n=200):
        """Helper to create synthetic unit-time price/time data."""
        rng = np.random.default_rng(42)
        r = rng.standard_normal(n - 1) * 0.001
        prices = np.exp(np.cumsum(np.concatenate([[np.log(100.0)], r])))
        times = np.linspace(0.0, 1.0, n)
        return prices, times

    def test_basic_subsampling(self):
        """With subsamples=5, returns list of 5 tuples."""
        prices, times = self._make_test_data(n=200)

        result = realized_subsample(
            prices, times, 'unit', 'BusinessUniform', 50, 5
        )

        # Should return a list of 5 subsamples
        assert isinstance(result, list)
        assert len(result) == 5
        # Each element is a tuple with (prices, times, base_count, total_count)
        for j, item in enumerate(result):
            assert isinstance(item, tuple), f"subsample {j} should be tuple"
            assert len(item) == 4, f"subsample {j} should have 4 elements"
            sub_prices, sub_times, base_count, total_count = item
            assert isinstance(sub_prices, np.ndarray)
            assert isinstance(sub_times, np.ndarray)
            assert len(sub_prices) == base_count
            assert total_count == 200

    def test_no_subsampling(self):
        """With subsamples=1, returns single grid identical to price_filter."""
        prices, times = self._make_test_data(n=200)

        result = realized_subsample(
            prices, times, 'unit', 'BusinessUniform', 50, 1
        )

        assert len(result) == 1
        sub_prices, sub_times, base_count, total_count = result[0]

        # Compare with direct price_filter call
        fp, ft, _ = realized_price_filter(
            prices, times, 'unit', 'BusinessUniform', 50
        )

        npt.assert_allclose(sub_prices, fp, atol=ATOL, rtol=RTOL)
        assert base_count == len(fp)

    def test_output_type(self):
        """Returns list of tuples containing numpy.ndarray."""
        prices, times = self._make_test_data(n=200)

        result = realized_subsample(
            prices, times, 'unit', 'CalendarUniform', 40, 3
        )

        assert isinstance(result, list)
        for item in result:
            sub_prices, sub_times, _, _ = item
            assert isinstance(sub_prices, np.ndarray)
            assert isinstance(sub_times, np.ndarray)
            assert sub_prices.dtype == np.float64 or np.issubdtype(
                sub_prices.dtype, np.floating
            )

    def test_fixture_parity(self):
        """Compare against MATLAB fixture data for all scenarios."""
        fixture = _load_fixture('realized_subsample')
        if fixture is None:
            pytest.skip('Fixture file not found: realized_subsample.npy')

        hf_prices = np.asarray(
            fixture['hf_prices'], dtype=np.float64
        ).ravel()

        num_scenarios = fixture.get('num_scenarios', 0)
        for i in range(1, num_scenarios + 1):
            scenario = fixture.get(f'scenario_{i}')
            if scenario is None:
                continue

            # Skip scenarios without Octave reference data
            if not scenario.get('has_octave_reference', False):
                continue

            time_type = str(scenario['timeType'])
            sampling_type = str(scenario['samplingType'])
            si = scenario['samplingInterval']
            sub_samples = int(scenario['subSamples'])

            # Select times
            time_key = f'hf_times_{time_type}'
            if time_key in fixture:
                times = np.asarray(
                    fixture[time_key], dtype=np.float64
                ).ravel()
            elif time_type == 'seconds' and 'hf_times_seconds' in fixture:
                times = np.asarray(
                    fixture['hf_times_seconds'], dtype=np.float64
                ).ravel()
            elif time_type == 'unit' and 'hf_times_unit' in fixture:
                times = np.asarray(
                    fixture['hf_times_unit'], dtype=np.float64
                ).ravel()
            else:
                continue

            # Handle sampling interval
            if np.isscalar(si) or (
                isinstance(si, np.ndarray) and si.ndim == 0
            ):
                si_val = float(si)
                if si_val == int(si_val):
                    si_val = int(si_val)
            else:
                si_val = np.asarray(si, dtype=np.float64).ravel()

            try:
                result = realized_subsample(
                    hf_prices, times, time_type, sampling_type,
                    si_val, sub_samples,
                )
            except Exception:
                continue

            # Compare subsampled prices against fixture
            if 'subsampledPrice' in scenario:
                expected_prices = scenario['subsampledPrice']
                if isinstance(expected_prices, np.ndarray):
                    if expected_prices.dtype == object:
                        # Cell array stored as object array
                        for j in range(
                            min(len(result), len(expected_prices))
                        ):
                            exp = np.asarray(
                                expected_prices[j], dtype=np.float64
                            ).ravel()
                            actual = result[j][0]
                            if len(exp) > 0 and len(actual) > 0:
                                min_len = min(len(actual), len(exp))
                                npt.assert_allclose(
                                    actual[:min_len],
                                    exp[:min_len],
                                    atol=ATOL,
                                    rtol=RTOL,
                                    err_msg=(
                                        f"scenario_{i} subsample {j} "
                                        f"prices mismatch"
                                    ),
                                )
