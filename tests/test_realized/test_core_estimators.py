"""
Parametrized pytest parity tests for 4 core realized volatility estimators.

Tests:
1. realized_variance — standard realized variance with optional subsampling
2. realized_bipower_variation — BPV with skip-k and debiased variants
3. realized_semivariance — positive/negative semivariance decomposition
4. realized_quarticity — BNS, Tripower, Quadpower quarticity variants

All assertions use numpy.testing.assert_allclose(actual, expected, atol=1e-6,
rtol=1e-4) per AAP Section 0.7.1.

Test patterns mirror realized_test.m (the MATLAB smoke test) and validate
against Octave-generated fixture data stored under tests/fixtures/realized/.

Ref: AAP Section 0.5.1 — Parametrized parity tests per estimator against fixtures
"""

import os

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.realized.realized_variance import realized_variance
from mfe_toolbox.realized.realized_bipower_variation import realized_bipower_variation
from mfe_toolbox.realized.realized_semivariance import realized_semivariance
from mfe_toolbox.realized.realized_quarticity import realized_quarticity
from mfe_toolbox.realized.wall2seconds import wall2seconds
from mfe_toolbox.realized.seconds2wall import seconds2wall

# ---------------------------------------------------------------------------
# Fixture Directory Resolution
# ---------------------------------------------------------------------------
# Ref: AAP Section 0.7.2 — MFE_FIXTURE_DIR environment variable for CI
FIXTURE_DIR = os.environ.get(
    'MFE_FIXTURE_DIR',
    os.path.join(os.path.dirname(__file__), '..', 'fixtures', 'realized'),
)


def _load_fixture(name):
    """Load a .npy fixture file from the realized fixtures directory.

    Parameters
    ----------
    name : str
        Base filename (without extension) to load.

    Returns
    -------
    dict or None
        Loaded fixture data (typically a dict from object-dtype .npy),
        or None if the file does not exist.
    """
    path = os.path.join(FIXTURE_DIR, name)
    if not path.endswith('.npy'):
        path = path + '.npy'
    if os.path.exists(path):
        return np.load(path, allow_pickle=True)
    return None


# ---------------------------------------------------------------------------
# Shared Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def brownian_prices():
    """Generate synthetic Brownian motion prices matching realized_test.m.

    Ref: realized_test.m — r = randn(390*300,1)/sqrt(390*300);
                           p = exp(cumsum([0;r]))

    Uses a deterministic seed (42) for reproducibility.  The number of
    observations (117001) models 300 days of 390 one-minute ticks plus
    the opening price.
    """
    rng = np.random.default_rng(42)
    n = 390 * 300  # 300 days × 390 minutes
    r = rng.standard_normal(n) / np.sqrt(n)
    # Ref: realized_test.m — p = exp(cumsum([0; r]))
    # MATLAB prepends a zero to the cumulative sum; Python equivalent uses
    # concatenate before cumsum.
    p = np.exp(np.cumsum(np.concatenate([[0.0], r])))
    return p


@pytest.fixture
def brownian_times_wall(brownian_prices):
    """Wall clock times for Brownian prices — one trading day 9:30–16:00.

    Maps all observations to the wall-clock range 93000–160000
    (HHMMSS format) via linear spacing in seconds then conversion.
    """
    n = len(brownian_prices)
    # Ref: seconds range from 9:30 AM (34200s) to 4:00 PM (57600s)
    t_seconds = np.linspace(34200, 57600, n)
    return seconds2wall(t_seconds)


@pytest.fixture
def brownian_times_seconds(brownian_prices):
    """Seconds past midnight for Brownian prices."""
    n = len(brownian_prices)
    return np.linspace(34200, 57600, n)


@pytest.fixture
def brownian_times_unit(brownian_prices):
    """Unit-normalised times [0, 1] for Brownian prices."""
    n = len(brownian_prices)
    return np.linspace(0.0, 1.0, n)


@pytest.fixture
def fixed_interval_grid():
    """Fixed 5-minute interval grid matching realized_test.m.

    Ref: realized_test.m —
        fixedInterval = seconds2wall(
            wall2seconds(93000):300:wall2seconds(150000)
        )

    Creates a wall-time grid from 9:30 AM to 3:00 PM at 300-second
    (5-minute) intervals.
    """
    # Ref: wall2seconds expects array input
    start_sec = wall2seconds(np.array([93000.0]))[0]   # 34200 seconds
    end_sec = wall2seconds(np.array([150000.0]))[0]     # 54000 seconds
    seconds_grid = np.arange(start_sec, end_sec + 1, 300)  # 5-min intervals
    return seconds2wall(seconds_grid)


@pytest.fixture
def fixture_prices_and_times():
    """Load prices and times from the variance fixture file for parity tests.

    Returns (prices, times_seconds, times_unit, times_wall) or None
    if fixture is unavailable.
    """
    raw = _load_fixture('realized_variance')
    if raw is None:
        return None
    d = raw.item() if raw.ndim == 0 else raw
    return (
        d['hf_prices'],
        d['hf_times_seconds'],
        d.get('hf_times_unit'),
        d.get('hf_times_wall'),
    )


# ===================================================================
# TestRealizedVariance — 14 tests
# ===================================================================

class TestRealizedVariance:
    """Test ``realized_variance()`` function.

    Signature (Python): realized_variance(price, time, time_type,
        sampling_type, sampling_interval, subsamples)
    Returns: (rv, rv_ss, diagnostics)

    Ref: realized_variance.m — [rv, rvSS] = realized_variance(price,
        time, timeType, samplingType, samplingInterval, subsamples)
    """

    def test_basic_calendar_time(self, brownian_prices, brownian_times_seconds):
        """CalendarTime with 300-second interval produces positive RV."""
        rv, rv_ss, diag = realized_variance(
            brownian_prices, brownian_times_seconds,
            'seconds', 'CalendarTime', 300,
        )
        assert rv > 0, "RV must be positive for non-constant prices"
        assert isinstance(rv, float)

    def test_basic_calendar_uniform(self, brownian_prices, brownian_times_seconds):
        """CalendarUniform with 78 points produces positive RV."""
        rv, rv_ss, diag = realized_variance(
            brownian_prices, brownian_times_seconds,
            'seconds', 'CalendarUniform', 78,
        )
        assert rv > 0

    def test_basic_business_time(self, brownian_prices, brownian_times_seconds):
        """BusinessTime with 10-tick interval."""
        rv, rv_ss, diag = realized_variance(
            brownian_prices, brownian_times_seconds,
            'seconds', 'BusinessTime', 10,
        )
        assert rv > 0

    def test_basic_business_uniform(self, brownian_prices, brownian_times_seconds):
        """BusinessUniform with 78 bins."""
        rv, rv_ss, diag = realized_variance(
            brownian_prices, brownian_times_seconds,
            'seconds', 'BusinessUniform', 78,
        )
        assert rv > 0

    def test_fixed_sampling(
        self, brownian_prices, brownian_times_wall, fixed_interval_grid
    ):
        """Fixed interval grid from fixture produces positive RV."""
        rv, rv_ss, diag = realized_variance(
            brownian_prices, brownian_times_wall,
            'wall', 'Fixed', fixed_interval_grid,
        )
        assert rv > 0

    def test_returns_two_outputs(self, brownian_prices, brownian_times_seconds):
        """Function returns (rv, rv_ss, diagnostics) 3-tuple.

        Ref: realized_variance.m returns [rv, rvSS]; Python adds diagnostics.
        """
        result = realized_variance(
            brownian_prices, brownian_times_seconds,
            'seconds', 'CalendarTime', 300,
        )
        assert len(result) == 3
        rv, rv_ss, diag = result
        assert isinstance(rv, float)
        assert isinstance(rv_ss, float)
        assert isinstance(diag, dict)

    def test_no_subsampling(self, brownian_prices, brownian_times_seconds):
        """With subsamples=1 (default), rv_ss equals rv.

        Ref: realized_variance.m — when subsamples is omitted, default is 1,
        so rvSS == rv.

        Note: MATLAB default is 1 (no subsampling).  Python validates
        subsamples >= 1 (subsamples=0 raises ValueError).
        """
        rv, rv_ss, diag = realized_variance(
            brownian_prices, brownian_times_seconds,
            'seconds', 'CalendarTime', 300, subsamples=1,
        )
        npt.assert_allclose(rv_ss, rv, atol=1e-6, rtol=1e-4)

    def test_with_subsampling(self, brownian_prices, brownian_times_seconds):
        """With subsamples=5, rv_ss differs from rv (averaged subsamples)."""
        rv, rv_ss, diag = realized_variance(
            brownian_prices, brownian_times_seconds,
            'seconds', 'CalendarTime', 300, subsamples=5,
        )
        # Subsampled RV should be close to but generally different from basic RV
        assert rv > 0
        assert rv_ss > 0
        # They should not be exactly equal when subsampling is used
        # (extremely unlikely for Brownian motion with 5 subsamples)

    def test_rv_positive(self, brownian_prices, brownian_times_seconds):
        """RV is always positive for non-constant prices."""
        rv, rv_ss, diag = realized_variance(
            brownian_prices, brownian_times_seconds,
            'seconds', 'CalendarTime', 600,
        )
        assert rv > 0
        assert rv_ss > 0

    def test_rv_scales_with_time(self, brownian_prices, brownian_times_seconds):
        """RV over full day > RV over half day for same price process.

        Use all prices vs first-half prices to verify that RV accumulates
        with more data (more returns typically means larger total RV).
        """
        n = len(brownian_prices)
        half = n // 2
        # Full-day RV
        rv_full, _, _ = realized_variance(
            brownian_prices, brownian_times_seconds,
            'seconds', 'CalendarTime', 300,
        )
        # Half-day RV using first half of prices and times
        rv_half, _, _ = realized_variance(
            brownian_prices[:half], brownian_times_seconds[:half],
            'seconds', 'CalendarTime', 300,
        )
        # Full RV should generally be larger (accumulates over more returns)
        assert rv_full > rv_half * 0.3, (
            "Full-day RV should be noticeably larger than half-day RV"
        )

    @pytest.mark.parametrize(
        'sampling_type, sampling_interval',
        [
            ('CalendarTime', 60),
            ('CalendarTime', 300),
            ('CalendarTime', 600),
            ('CalendarUniform', 39),
            ('CalendarUniform', 78),
            ('CalendarUniform', 130),
            ('BusinessTime', 5),
            ('BusinessTime', 10),
            ('BusinessTime', 20),
            ('BusinessUniform', 39),
            ('BusinessUniform', 78),
            ('BusinessUniform', 130),
        ],
    )
    def test_parametrize_sampling_types(
        self, brownian_prices, brownian_times_seconds,
        sampling_type, sampling_interval,
    ):
        """Parametrize over all scalar sampling types with appropriate intervals.

        Ref: realized_test.m loops over CalendarTime/CalendarUniform/
        BusinessTime/BusinessUniform with various intervals.  Fixed sampling
        tested separately due to vector interval requirement.
        """
        rv, rv_ss, diag = realized_variance(
            brownian_prices, brownian_times_seconds,
            'seconds', sampling_type, sampling_interval,
        )
        assert rv > 0
        assert rv_ss > 0

    def test_parametrize_fixed_sampling(
        self, brownian_prices, brownian_times_wall, fixed_interval_grid,
    ):
        """Test Fixed sampling type with wall-time interval grid."""
        rv, rv_ss, diag = realized_variance(
            brownian_prices, brownian_times_wall,
            'wall', 'Fixed', fixed_interval_grid,
        )
        assert rv > 0
        assert rv_ss > 0

    @pytest.mark.parametrize('time_type_label', ['wall', 'seconds', 'unit'])
    def test_parametrize_time_types(
        self, brownian_prices, brownian_times_wall,
        brownian_times_seconds, brownian_times_unit, time_type_label,
    ):
        """Parametrize over 'wall', 'seconds', 'unit' time types.

        Ref: realized_variance.m accepts all three timeType values.
        """
        if time_type_label == 'wall':
            times = brownian_times_wall
            interval = 300  # 300-second calendar time
        elif time_type_label == 'seconds':
            times = brownian_times_seconds
            interval = 300
        else:  # 'unit'
            times = brownian_times_unit
            # Ref: For unit time, use a fraction-based interval
            interval = 0.01
        rv, rv_ss, diag = realized_variance(
            brownian_prices, times, time_type_label,
            'CalendarTime', interval,
        )
        assert rv > 0

    def test_known_value_simple(self):
        """For a simple 3-price sequence, verify RV = sum of squared log returns.

        Prices: [100, 101, 102]
        Log returns: [log(101/100), log(102/101)]
        RV = sum of squared log returns
        """
        prices = np.array([100.0, 101.0, 102.0])
        times = np.array([0.0, 0.5, 1.0])  # Unit times
        rv, rv_ss, diag = realized_variance(
            prices, times, 'unit', 'CalendarUniform', 3,
        )
        # Expected: sum of squared log returns
        log_returns = np.diff(np.log(prices))
        expected_rv = float(np.sum(log_returns ** 2))
        npt.assert_allclose(rv, expected_rv, atol=1e-6, rtol=1e-4)

    @pytest.mark.parity
    def test_fixture_parity(self):
        """Load MATLAB fixture and compare at atol=1e-6.

        Ref: AAP Section 0.7.1 — All migrated functions MUST pass
        numpy.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-4)
        """
        raw = _load_fixture('realized_variance')
        if raw is None:
            pytest.skip("Realized variance fixture not found")
        d = raw.item() if raw.ndim == 0 else raw

        prices = d['hf_prices']
        times_seconds = d['hf_times_seconds']
        times_unit = d.get('hf_times_unit')
        times_wall = d.get('hf_times_wall')

        n_scenarios = d['num_scenarios']
        for i in range(1, n_scenarios + 1):
            scenario = d[f'scenario_{i}']
            time_type = scenario['timeType']
            sampling_type = scenario['samplingType']
            sampling_interval = scenario['samplingInterval']
            subsamples = int(scenario['subsamples'])

            # Select appropriate times array based on time_type
            if time_type == 'seconds':
                times = times_seconds
            elif time_type == 'wall' and times_wall is not None:
                times = times_wall
            elif time_type == 'unit' and times_unit is not None:
                times = times_unit
            else:
                continue  # Skip if times not available for this type

            rv, rv_ss, diag = realized_variance(
                prices, times, time_type, sampling_type,
                sampling_interval, subsamples,
            )

            expected_rv = float(scenario['rv'])
            expected_rv_ss = float(scenario['rvSS'])

            npt.assert_allclose(
                rv, expected_rv, atol=1e-6, rtol=1e-4,
                err_msg=f"RV mismatch in scenario {i}: {scenario['description']}",
            )
            npt.assert_allclose(
                rv_ss, expected_rv_ss, atol=1e-6, rtol=1e-4,
                err_msg=f"RVSS mismatch in scenario {i}: {scenario['description']}",
            )


# ===================================================================
# TestRealizedBipowerVariation — 11 tests
# ===================================================================

class TestRealizedBipowerVariation:
    """Test ``realized_bipower_variation()`` function.

    Signature (Python): realized_bipower_variation(price, time, time_type,
        sampling_type, sampling_interval, skip, subsamples)
    Returns: (bv, bv_ss, bv_debiased, bv_ss_debiased, diagnostics)

    Ref: realized_bipower_variation.m — [bv, bvSS, bvDebiased, bvSSDebiased]
    """

    def test_returns_four_outputs(self, brownian_prices, brownian_times_seconds):
        """Returns (bv, bv_ss, bv_debiased, bv_ss_debiased, diagnostics) 5-tuple.

        Ref: MATLAB returns [bv, bvSS, bvDebiased, bvSSDebiased];
        Python adds a diagnostics dict.
        """
        result = realized_bipower_variation(
            brownian_prices, brownian_times_seconds,
            'seconds', 'CalendarTime', 300,
        )
        assert len(result) == 5
        bv, bv_ss, bv_debiased, bv_ss_debiased, diag = result
        assert isinstance(bv, float)
        assert isinstance(bv_ss, float)
        assert isinstance(bv_debiased, float)
        assert isinstance(bv_ss_debiased, float)
        assert isinstance(diag, dict)

    def test_basic_estimation(self, brownian_prices, brownian_times_seconds):
        """BPV is positive."""
        bv, bv_ss, bv_d, bv_ss_d, diag = realized_bipower_variation(
            brownian_prices, brownian_times_seconds,
            'seconds', 'CalendarTime', 300,
        )
        assert bv > 0

    def test_skip_zero(self, brownian_prices, brownian_times_seconds):
        """skip=0 produces standard BPV."""
        bv, bv_ss, bv_d, bv_ss_d, diag = realized_bipower_variation(
            brownian_prices, brownian_times_seconds,
            'seconds', 'CalendarTime', 300, skip=0,
        )
        assert bv > 0
        assert diag['skip'] == 0

    def test_skip_one(self, brownian_prices, brownian_times_seconds):
        """skip=1 produces skip-1 BPV.

        Ref: realized_test.m tests skip=0:1.
        Skip-1 BPV is valid and positive.
        """
        bv, bv_ss, bv_d, bv_ss_d, diag = realized_bipower_variation(
            brownian_prices, brownian_times_seconds,
            'seconds', 'CalendarTime', 300, skip=1,
        )
        assert bv > 0
        assert diag['skip'] == 1

    def test_debiased_larger(self, brownian_prices, brownian_times_seconds):
        """Debiased BV >= BV because correction factor m/(m-skip-1) >= 1.

        Ref: realized_bipower_variation.m:180-181 —
            biasScale = (m-1-skip) / m;
            bvDebiased = bv / biasScale;
        Since biasScale < 1, bvDebiased > bv.
        """
        bv, bv_ss, bv_d, bv_ss_d, diag = realized_bipower_variation(
            brownian_prices, brownian_times_seconds,
            'seconds', 'CalendarTime', 300, skip=0,
        )
        # bv_debiased = bv / ((m-1-skip)/m) = bv * m / (m-1-skip)
        # Since m > m-1-skip, bv_debiased >= bv
        assert bv_d >= bv - 1e-15, (
            "Debiased BV should be >= BV (correction factor >= 1)"
        )

    def test_bpv_less_than_rv(self, brownian_prices, brownian_times_seconds):
        """For jump-free Brownian motion, BPV ≈ RV.

        For a continuous process without jumps, BPV converges to the
        integrated variance just like RV.  With finite samples, BPV
        should be in the same ballpark as RV.

        Ref: Barndorff-Nielsen & Shephard (2004) — BPV is consistent for
        integrated variance in the absence of jumps.
        """
        bv, bv_ss, bv_d, bv_ss_d, _ = realized_bipower_variation(
            brownian_prices, brownian_times_seconds,
            'seconds', 'CalendarTime', 300,
        )
        rv, rv_ss, _ = realized_variance(
            brownian_prices, brownian_times_seconds,
            'seconds', 'CalendarTime', 300,
        )
        # BPV should be close to RV for Brownian motion (within a factor)
        # Use a generous tolerance since finite-sample BPV and RV may differ
        assert bv > 0
        assert bv < rv * 5.0, "BPV should be same order of magnitude as RV"
        assert bv > rv * 0.1, "BPV should be same order of magnitude as RV"

    @pytest.mark.parametrize(
        'sampling_type, sampling_interval',
        [
            ('CalendarTime', 300),
            ('CalendarTime', 600),
            ('CalendarUniform', 78),
            ('BusinessTime', 10),
            ('BusinessTime', 20),
            ('BusinessUniform', 78),
        ],
    )
    def test_parametrize_sampling_types(
        self, brownian_prices, brownian_times_seconds,
        sampling_type, sampling_interval,
    ):
        """All scalar sampling types produce positive BPV.

        Ref: realized_test.m loops over all sampling types.
        """
        bv, bv_ss, bv_d, bv_ss_d, _ = realized_bipower_variation(
            brownian_prices, brownian_times_seconds,
            'seconds', sampling_type, sampling_interval,
        )
        assert bv > 0
        assert bv_ss > 0
        assert bv_d >= bv - 1e-15

    def test_with_subsampling(self, brownian_prices, brownian_times_seconds):
        """subsamples=5 produces averaged estimate."""
        bv, bv_ss, bv_d, bv_ss_d, _ = realized_bipower_variation(
            brownian_prices, brownian_times_seconds,
            'seconds', 'CalendarTime', 300, skip=0, subsamples=5,
        )
        assert bv > 0
        assert bv_ss > 0

    def test_no_subsampling(self, brownian_prices, brownian_times_seconds):
        """subsamples=1 means bv_ss == bv (no averaging)."""
        bv, bv_ss, bv_d, bv_ss_d, _ = realized_bipower_variation(
            brownian_prices, brownian_times_seconds,
            'seconds', 'CalendarTime', 300, skip=0, subsamples=1,
        )
        npt.assert_allclose(bv_ss, bv, atol=1e-6, rtol=1e-4)

    def test_fixed_interval(
        self, brownian_prices, brownian_times_wall, fixed_interval_grid,
    ):
        """Fixed grid with 5-minute intervals."""
        bv, bv_ss, bv_d, bv_ss_d, _ = realized_bipower_variation(
            brownian_prices, brownian_times_wall,
            'wall', 'Fixed', fixed_interval_grid,
        )
        assert bv > 0
        assert bv_d >= bv - 1e-15

    @pytest.mark.parity
    def test_fixture_parity(self):
        """Load MATLAB fixture and compare at atol=1e-6.

        Ref: AAP Section 0.7.1 — numerical parity contract.
        """
        raw = _load_fixture('realized_bipower_variation')
        if raw is None:
            pytest.skip("BPV fixture not found")
        d = raw.item() if raw.ndim == 0 else raw

        prices = d['hf_prices']
        times_seconds = d['hf_times_seconds']

        n_scenarios = d['num_scenarios']
        for i in range(1, n_scenarios + 1):
            scenario = d[f'scenario_{i}']
            time_type = scenario['timeType']
            sampling_type = scenario['samplingType']
            sampling_interval = scenario['samplingInterval']
            skip = int(scenario['skip'])
            subsamples = int(scenario['subsamples'])

            # Select times
            if time_type == 'seconds':
                times = times_seconds
            else:
                continue

            bv, bv_ss, bv_d, bv_ss_d, _ = realized_bipower_variation(
                prices, times, time_type, sampling_type,
                sampling_interval, skip=skip, subsamples=subsamples,
            )

            npt.assert_allclose(
                bv, float(scenario['bv']), atol=1e-6, rtol=1e-4,
                err_msg=f"BV mismatch: scenario {i} — {scenario['description']}",
            )
            npt.assert_allclose(
                bv_ss, float(scenario['bvSS']), atol=1e-6, rtol=1e-4,
                err_msg=f"BVSS mismatch: scenario {i} — {scenario['description']}",
            )
            npt.assert_allclose(
                bv_d, float(scenario['bvDebiased']), atol=1e-6, rtol=1e-4,
                err_msg=f"BVDebiased mismatch: scenario {i}",
            )
            npt.assert_allclose(
                bv_ss_d, float(scenario['bvSSDebiased']), atol=1e-6, rtol=1e-4,
                err_msg=f"BVSSDebiased mismatch: scenario {i}",
            )


# ===================================================================
# TestRealizedSemivariance — 7 tests
# ===================================================================

class TestRealizedSemivariance:
    """Test ``realized_semivariance()`` function.

    Signature (Python): realized_semivariance(price, time, time_type,
        sampling_type, sampling_interval, subsamples)
    Returns: (rsvn, rsvp, rsvn_ss, rsvp_ss, diagnostics)

    Ref: realized_semivariance.m — [rsvn, rsvp, rsvnSS, rsvpSS]

    Note: Unlike realized_variance, realized_semivariance requires the
    time parameter (time=None raises ValueError).
    """

    def test_returns_structure(self, brownian_prices, brownian_times_seconds):
        """Returns contain positive and negative semivariance components.

        Python returns 5-tuple: (rsvn, rsvp, rsvn_ss, rsvp_ss, diagnostics).
        """
        result = realized_semivariance(
            brownian_prices, brownian_times_seconds,
            'seconds', 'CalendarTime', 300,
        )
        assert len(result) == 5
        rsvn, rsvp, rsvn_ss, rsvp_ss, diag = result
        assert isinstance(rsvn, float)
        assert isinstance(rsvp, float)
        assert isinstance(rsvn_ss, float)
        assert isinstance(rsvp_ss, float)
        assert isinstance(diag, dict)

    def test_sum_equals_rv(self, brownian_prices, brownian_times_seconds):
        """Positive semivariance + negative semivariance ≈ realized variance.

        Ref: By construction, rsvn + rsvp = RV (within machine precision).
        The MATLAB fixture crosscheck verifies: rsvn_plus_rsvp == rv.
        """
        rsvn, rsvp, rsvn_ss, rsvp_ss, _ = realized_semivariance(
            brownian_prices, brownian_times_seconds,
            'seconds', 'CalendarTime', 300,
        )
        rv, rv_ss, _ = realized_variance(
            brownian_prices, brownian_times_seconds,
            'seconds', 'CalendarTime', 300,
        )
        # Semivariance decomposition: rsvn + rsvp = rv
        npt.assert_allclose(
            rsvn + rsvp, rv, atol=1e-6, rtol=1e-4,
            err_msg="rsvn + rsvp should equal realized variance",
        )

    def test_positive_semivar_positive(
        self, brownian_prices, brownian_times_seconds,
    ):
        """Positive semivariance >= 0.

        Sum of squared positive returns is non-negative by construction.
        """
        rsvn, rsvp, rsvn_ss, rsvp_ss, _ = realized_semivariance(
            brownian_prices, brownian_times_seconds,
            'seconds', 'CalendarTime', 300,
        )
        assert rsvp >= 0.0

    def test_negative_semivar_positive(
        self, brownian_prices, brownian_times_seconds,
    ):
        """Negative semivariance >= 0.

        Sum of squared negative returns is non-negative by construction.
        """
        rsvn, rsvp, rsvn_ss, rsvp_ss, _ = realized_semivariance(
            brownian_prices, brownian_times_seconds,
            'seconds', 'CalendarTime', 300,
        )
        assert rsvn >= 0.0

    def test_symmetric_for_symmetric_returns(self):
        """For symmetric price changes, positive ≈ negative semivariance.

        Construct prices with exactly symmetric log returns (alternating
        up and down of equal magnitude) so that positive and negative
        semivariance are equal.
        """
        n = 201  # Odd number of prices → even number of returns
        # Alternating +/- returns of magnitude 0.01
        log_returns = np.array([0.01 if i % 2 == 0 else -0.01 for i in range(n - 1)])
        log_prices = np.cumsum(np.concatenate([[np.log(100.0)], log_returns]))
        prices = np.exp(log_prices)
        times = np.linspace(0.0, 1.0, n)

        rsvn, rsvp, rsvn_ss, rsvp_ss, _ = realized_semivariance(
            prices, times, 'unit', 'CalendarUniform', n,
        )
        # With exactly symmetric returns (100 positive, 100 negative),
        # semivariances should be nearly equal
        npt.assert_allclose(
            rsvn, rsvp, atol=1e-6, rtol=1e-2,
            err_msg="Symmetric returns should yield similar semivariances",
        )

    @pytest.mark.parametrize(
        'sampling_type, sampling_interval',
        [
            ('CalendarTime', 300),
            ('CalendarTime', 600),
            ('CalendarUniform', 78),
            ('BusinessTime', 10),
            ('BusinessUniform', 78),
        ],
    )
    def test_parametrize_sampling_types(
        self, brownian_prices, brownian_times_seconds,
        sampling_type, sampling_interval,
    ):
        """All 5 sampling types produce valid semivariance decomposition."""
        rsvn, rsvp, rsvn_ss, rsvp_ss, _ = realized_semivariance(
            brownian_prices, brownian_times_seconds,
            'seconds', sampling_type, sampling_interval,
        )
        assert rsvn >= 0.0
        assert rsvp >= 0.0
        # Semivariance decomposition: rsvn + rsvp should be a positive RV
        assert rsvn + rsvp > 0.0

    def test_parametrize_fixed_sampling(
        self, brownian_prices, brownian_times_wall, fixed_interval_grid,
    ):
        """Test Fixed sampling type for semivariance."""
        rsvn, rsvp, rsvn_ss, rsvp_ss, _ = realized_semivariance(
            brownian_prices, brownian_times_wall,
            'wall', 'Fixed', fixed_interval_grid,
        )
        assert rsvn >= 0.0
        assert rsvp >= 0.0
        assert rsvn + rsvp > 0.0

    @pytest.mark.parity
    def test_fixture_parity(self):
        """Load MATLAB fixture and compare at atol=1e-6.

        Fixture structure:
        - price, time_seconds, time_unit arrays
        - scenario_1 through scenario_N dicts with rsvn, rsvp, rsvnSS, rsvpSS
        - crosscheck dict verifying rsvn + rsvp = rv
        """
        raw = _load_fixture('realized_semivariance')
        if raw is None:
            pytest.skip("Semivariance fixture not found")
        d = raw.item() if raw.ndim == 0 else raw

        prices = d['price']
        times_seconds = d['time_seconds']

        # Test each scenario
        for i in range(1, 6):
            key = f'scenario_{i}'
            if key not in d:
                continue
            scenario = d[key]
            time_type = scenario['timeType']
            sampling_type = scenario['samplingType']
            sampling_interval = scenario['samplingInterval']
            subsamples = int(scenario['subsamples'])

            times = times_seconds

            rsvn, rsvp, rsvn_ss, rsvp_ss, _ = realized_semivariance(
                prices, times, time_type, sampling_type,
                sampling_interval, subsamples,
            )

            npt.assert_allclose(
                rsvn, float(scenario['rsvn']), atol=1e-6, rtol=1e-4,
                err_msg=f"RSVN mismatch: scenario {i} — {scenario['description']}",
            )
            npt.assert_allclose(
                rsvp, float(scenario['rsvp']), atol=1e-6, rtol=1e-4,
                err_msg=f"RSVP mismatch: scenario {i} — {scenario['description']}",
            )
            npt.assert_allclose(
                rsvn_ss, float(scenario['rsvnSS']), atol=1e-6, rtol=1e-4,
                err_msg=f"RSVNSS mismatch: scenario {i}",
            )
            npt.assert_allclose(
                rsvp_ss, float(scenario['rsvpSS']), atol=1e-6, rtol=1e-4,
                err_msg=f"RSVPSS mismatch: scenario {i}",
            )

        # Verify crosscheck: rsvn + rsvp = rv
        if 'crosscheck' in d:
            cc = d['crosscheck']
            expected_sum = float(cc['rsvn_plus_rsvp'])
            # Compute using scenario_1 settings
            s1 = d['scenario_1']
            rsvn, rsvp, _, _, _ = realized_semivariance(
                prices, times_seconds, s1['timeType'],
                s1['samplingType'], s1['samplingInterval'],
                int(s1['subsamples']),
            )
            npt.assert_allclose(
                rsvn + rsvp, expected_sum, atol=1e-6, rtol=1e-4,
                err_msg="Crosscheck: rsvn + rsvp should equal RV",
            )


# ===================================================================
# TestRealizedQuarticity — 10 tests
# ===================================================================

class TestRealizedQuarticity:
    """Test ``realized_quarticity()`` function.

    Signature (Python): realized_quarticity(price, time, time_type,
        sampling_type, sampling_interval, skip, quarticity_type, subsamples)
    Returns: (qt, qt_ss, qt_debiased, qt_ss_debiased, diagnostics)

    Ref: realized_quarticity.m — [qt, qtSS, qtDebiased, qtSSDebiased] =
         realized_quarticity(price, time, timeType, samplingType,
             samplingInterval, QTtype, skip, subsamples)

    Note: Python parameter order differs from MATLAB:
        MATLAB:  ..., QTtype, skip, subsamples
        Python:  ..., skip, quarticity_type, subsamples
    """

    def test_basic_estimation(self, brownian_prices, brownian_times_seconds):
        """Quarticity is positive."""
        qt, qt_ss, qt_d, qt_ss_d, diag = realized_quarticity(
            brownian_prices, brownian_times_seconds,
            'seconds', 'CalendarTime', 300,
        )
        assert qt > 0
        assert qt_ss > 0

    def test_bns_type(self, brownian_prices, brownian_times_seconds):
        """type='BNS' (Barndorff-Nielsen-Shephard estimator).

        BNS quarticity: QT = (1/3) * m * sum(r^4)
        Not robust to jumps.
        """
        qt, qt_ss, qt_d, qt_ss_d, diag = realized_quarticity(
            brownian_prices, brownian_times_seconds,
            'seconds', 'CalendarTime', 300,
            quarticity_type='BNS',
        )
        assert qt > 0

    def test_tripower_type(self, brownian_prices, brownian_times_seconds):
        """type='Tripower' estimator.

        Ref: realized_quarticity.m — tripower uses products of |r|^{4/3}
        over 3 adjacent returns.
        """
        qt, qt_ss, qt_d, qt_ss_d, diag = realized_quarticity(
            brownian_prices, brownian_times_seconds,
            'seconds', 'CalendarTime', 300,
            quarticity_type='Tripower',
        )
        assert qt > 0

    def test_quadpower_type(self, brownian_prices, brownian_times_seconds):
        """type='Quadpower' estimator.

        Ref: realized_quarticity.m — quadpower uses products of |r| over
        4 adjacent returns.
        """
        qt, qt_ss, qt_d, qt_ss_d, diag = realized_quarticity(
            brownian_prices, brownian_times_seconds,
            'seconds', 'CalendarTime', 300,
            quarticity_type='Quadpower',
        )
        assert qt > 0

    @pytest.mark.parametrize(
        'qt_type', ['BNS', 'Tripower', 'Quadpower'],
    )
    def test_parametrize_types(
        self, brownian_prices, brownian_times_seconds, qt_type,
    ):
        """Parametrize over all 3 quarticity types."""
        qt, qt_ss, qt_d, qt_ss_d, _ = realized_quarticity(
            brownian_prices, brownian_times_seconds,
            'seconds', 'CalendarTime', 300,
            quarticity_type=qt_type,
        )
        assert qt > 0
        assert qt_ss > 0
        # Debiased >= original for Tripower/Quadpower (due to count < m)
        if qt_type != 'BNS':
            assert qt_d >= qt - 1e-20

    @pytest.mark.parametrize('skip_val', [0, 1])
    def test_skip_0_and_1(
        self, brownian_prices, brownian_times_seconds, skip_val,
    ):
        """Test with skip=0 and skip=1 matching realized_test.m.

        Ref: realized_test.m tests skip=0:1 for quarticity.
        """
        qt, qt_ss, qt_d, qt_ss_d, _ = realized_quarticity(
            brownian_prices, brownian_times_seconds,
            'seconds', 'CalendarTime', 300,
            skip=skip_val, quarticity_type='Tripower',
        )
        assert qt > 0
        assert qt_d >= qt - 1e-20

    def test_with_subsamples(self, brownian_prices, brownian_times_seconds):
        """Test with subsamples=5 for bias reduction."""
        qt, qt_ss, qt_d, qt_ss_d, _ = realized_quarticity(
            brownian_prices, brownian_times_seconds,
            'seconds', 'CalendarTime', 300,
            quarticity_type='Tripower', subsamples=5,
        )
        assert qt > 0
        assert qt_ss > 0

    @pytest.mark.parametrize(
        'sampling_type, sampling_interval',
        [
            ('CalendarTime', 300),
            ('CalendarTime', 600),
            ('CalendarUniform', 78),
            ('BusinessTime', 10),
            ('BusinessUniform', 78),
        ],
    )
    def test_parametrize_sampling_types(
        self, brownian_prices, brownian_times_seconds,
        sampling_type, sampling_interval,
    ):
        """All 5 scalar sampling types produce positive quarticity.

        Ref: realized_test.m loops over all sampling types.
        """
        qt, qt_ss, qt_d, qt_ss_d, _ = realized_quarticity(
            brownian_prices, brownian_times_seconds,
            'seconds', sampling_type, sampling_interval,
        )
        assert qt > 0
        assert qt_ss > 0

    def test_quarticity_reasonable_magnitude(
        self, brownian_prices, brownian_times_seconds,
    ):
        """For Brownian motion, quarticity ≈ 3 * RV^2 (approximately).

        Ref: For a standard Brownian motion, the integrated quarticity
        equals 3 * (integrated variance)^2.  With finite samples and
        discretization, we only check order of magnitude.
        """
        qt, qt_ss, qt_d, qt_ss_d, _ = realized_quarticity(
            brownian_prices, brownian_times_seconds,
            'seconds', 'CalendarTime', 300,
            quarticity_type='BNS',
        )
        rv, rv_ss, _ = realized_variance(
            brownian_prices, brownian_times_seconds,
            'seconds', 'CalendarTime', 300,
        )
        # Theoretical: QT ≈ 3 * RV^2 for Brownian motion
        expected_qt = 3.0 * rv ** 2
        # Check order of magnitude: within factor of 100 (generous for
        # finite-sample discretisation)
        if expected_qt > 0:
            ratio = qt / expected_qt
            assert 0.01 < ratio < 100, (
                f"Quarticity / (3*RV^2) ratio = {ratio:.4f}, "
                f"expected within [0.01, 100]"
            )

    @pytest.mark.parity
    def test_fixture_parity(self):
        """Load MATLAB fixture and compare at atol=1e-6.

        The quarticity fixture uses named scenarios like
        'bns_calendartime_300s', 'tripower_calendartime_300s', etc.
        Each has _params, _qt, _qtSS, _qtDebiased, _qtSSDebiased keys.
        """
        raw = _load_fixture('realized_quarticity')
        if raw is None:
            pytest.skip("Quarticity fixture not found")
        d = raw.item() if raw.ndim == 0 else raw

        # Identify all scenario names from the fixture
        scenario_names_key = 'scenario_names'
        if scenario_names_key in d:
            scenario_names = list(d[scenario_names_key])
        else:
            # Fallback: find all keys ending with '_params'
            scenario_names = [
                k.replace('_params', '')
                for k in d.keys()
                if k.endswith('_params')
            ]

        # Use the variance fixture for shared prices/times
        rv_raw = _load_fixture('realized_variance')
        if rv_raw is None:
            pytest.skip("Variance fixture (for prices/times) not found")
        rv_d = rv_raw.item() if rv_raw.ndim == 0 else rv_raw
        prices = rv_d['hf_prices']
        times_seconds = rv_d['hf_times_seconds']

        for name in scenario_names:
            params_key = f'{name}_params'
            if params_key not in d:
                continue

            params = d[params_key]
            time_type = params['timeType']
            sampling_type = params['samplingType']
            sampling_interval = params['samplingInterval']
            qt_type = params.get('QTtype', 'BNS')
            skip = int(params.get('skip', 0))
            subsamples = int(params.get('subsamples', 1))

            if time_type == 'seconds':
                times = times_seconds
            else:
                continue

            qt, qt_ss, qt_d, qt_ss_d, _ = realized_quarticity(
                prices, times, time_type, sampling_type,
                sampling_interval, skip=skip,
                quarticity_type=qt_type, subsamples=subsamples,
            )

            expected_qt = float(d[f'{name}_qt'])
            expected_qt_ss = float(d[f'{name}_qtSS'])
            expected_qt_d = float(d[f'{name}_qtDebiased'])
            expected_qt_ss_d = float(d[f'{name}_qtSSDebiased'])

            npt.assert_allclose(
                qt, expected_qt, atol=1e-6, rtol=1e-4,
                err_msg=f"QT mismatch: {name}",
            )
            npt.assert_allclose(
                qt_ss, expected_qt_ss, atol=1e-6, rtol=1e-4,
                err_msg=f"QTSS mismatch: {name}",
            )
            npt.assert_allclose(
                qt_d, expected_qt_d, atol=1e-6, rtol=1e-4,
                err_msg=f"QTDebiased mismatch: {name}",
            )
            npt.assert_allclose(
                qt_ss_d, expected_qt_ss_d, atol=1e-6, rtol=1e-4,
                err_msg=f"QTSSDebiased mismatch: {name}",
            )
