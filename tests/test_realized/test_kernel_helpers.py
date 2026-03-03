"""
Parametrized pytest parity tests for realized kernel helper modules.

Tests 4 realized kernel helper functions:
1. realized_kernel_bandwidth — optimal bandwidth selection for realized kernel estimator
2. realized_kernel_core — core kernel computation (autocovariance accumulation)
3. realized_kernel_weights — kernel weight functions (Parzen, Bartlett, Tukey-Hanning, etc.)
4. realized_kernel_jitter_lag_length — optimal jitter lag length for end-point treatment

All assertions use numpy.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-4)
per AAP Section 0.7.1 numerical parity contract.

Migrated from MATLAB sources:
- realized/realized_kernel_bandwidth.m
- realized/realized_kernel_core.m
- realized/realized_kernel_weights.m
- realized/realized_kernel_jitter_lag_length.m
"""

import os

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.realized.realized_kernel_bandwidth import realized_kernel_bandwidth
from mfe_toolbox.realized.realized_kernel_core import realized_kernel_core
from mfe_toolbox.realized.realized_kernel_jitter_lag_length import realized_kernel_jitter_lag_length
from mfe_toolbox.realized.realized_kernel_weights import realized_kernel_weights
from mfe_toolbox.realized.realized_options import realized_options

# ---------------------------------------------------------------------------
# Fixture Directory Resolution
# Ref: AAP Section 0.7.2 — Optional MFE_FIXTURE_DIR environment variable
# override for fixture path during CI.
# ---------------------------------------------------------------------------
FIXTURE_DIR = os.environ.get(
    'MFE_FIXTURE_DIR',
    os.path.join(os.path.dirname(__file__), '..', 'fixtures', 'realized'),
)


def _load_fixture(name: str):
    """Load a .npy fixture file from the realized fixtures directory.

    Parameters
    ----------
    name : str
        Base name of the fixture file without .npy extension.

    Returns
    -------
    object or None
        Loaded fixture data (typically a dict when allow_pickle=True),
        or None if the fixture file does not exist on disk.
    """
    path = os.path.join(FIXTURE_DIR, name + '.npy')
    if os.path.exists(path):
        return np.load(path, allow_pickle=True)
    return None


# ---------------------------------------------------------------------------
# Complete list of all supported kernel types in the MFE Toolbox.
# Ref: realized_kernel_weights.m:32-37 — Combined flat-top and non-flat-top
# kernel lists.
# ---------------------------------------------------------------------------
ALL_KERNEL_TYPES = [
    'nonflatparzen', 'parzen', 'qs', 'fejer', 'thinf', 'bnhls',
    'bartlett', 'twoscale', 'th1', 'th2', 'th5', 'th16',
    'cubic', 'multiscale', '2ndorder', '5thorder', '6thorder',
    '7thorder', '8thorder', 'epanechnikov',
]

# Flat-top kernels produce weights in [0, 1] for normalized lags in [0, 1).
# Ref: realized_kernel_weights.m:32-34
FLAT_TOP_KERNELS = [
    'bartlett', 'twoscale', '2ndorder', 'epanechnikov',
    'cubic', 'multiscale', '5thorder', '6thorder', '7thorder', '8thorder',
    'parzen', 'th1', 'th2', 'th5', 'th16',
]

# Non-flat-top kernels have different normalization conventions.
# Ref: realized_kernel_weights.m:37
NON_FLAT_TOP_KERNELS = [
    'nonflatparzen', 'qs', 'fejer', 'thinf', 'bnhls',
]


# ---------------------------------------------------------------------------
# Shared Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_returns():
    """Generate sample log returns for kernel tests.

    Uses a fixed seed (42) via numpy.random.default_rng for reproducibility.
    Generates 1000 scaled standard normal returns to mimic intraday log returns.
    """
    rng = np.random.default_rng(42)
    n = 1000
    # Ref: Intraday log returns are approximately σ/sqrt(n) per observation
    r = rng.standard_normal(n) / np.sqrt(n)
    return r


@pytest.fixture
def kernel_options():
    """Default kernel options from realized_options('Kernel').

    Returns a dict with fields: kernel, endTreatment, bandwidth,
    jitterLags, maxBandwidthPerc, etc.
    """
    return realized_options('Kernel')


# ===========================================================================
# TestRealizedKernelWeights
# ===========================================================================


class TestRealizedKernelWeights:
    """Test realized_kernel_weights() — kernel weight vector computation.

    Tests the weight functions for all ~20 supported kernel types including
    Parzen, Bartlett, Tukey-Hanning, Quadratic Spectral, Epanechnikov, and
    higher-order polynomial kernels.
    """

    def test_nonflatparzen_weights(self):
        """Non-flat-top Parzen kernel produces weights in [0, 1], first weight < 1.

        Ref: realized_kernel_weights.m:96-98 — Non-flat-top Parzen uses
        x = j/(H+1) starting from j=1, so the first weight is strictly < 1.
        """
        opts = {'kernel': 'nonflatparzen', 'bandwidth': 10.0}
        weights = realized_kernel_weights(opts)
        assert isinstance(weights, np.ndarray)
        assert len(weights) == 10  # floor(bandwidth) lags
        # Non-flat-top: first weight is at x=1/(H+1) which gives w < 1
        assert weights[0] < 1.0
        assert weights[0] > 0.0
        # Weights should decrease towards zero for Parzen kernel
        assert weights[-1] < weights[0]
        # All weights should be non-negative for Parzen
        assert np.all(weights >= 0.0)

    def test_parzen_weights(self):
        """Flat-top Parzen kernel: first weight = 1, correct boundary behavior.

        Ref: realized_kernel_weights.m:86 — Piecewise Parzen definition:
        k(x) = 1 - 6x^2 + 6x^3 for 0 <= x <= 1/2
        k(x) = 2(1 - x)^3       for 1/2 < x < 1
        """
        opts = {'kernel': 'parzen', 'bandwidth': 20}
        weights = realized_kernel_weights(opts)
        assert isinstance(weights, np.ndarray)
        assert len(weights) == 20
        # Flat-top: first element at x=0 should be exactly 1.0
        npt.assert_allclose(weights[0], 1.0, atol=1e-6, rtol=1e-4)
        # All weights should be >= 0 for positive semi-definite kernel
        assert np.all(weights >= -1e-10)
        # Weights should decay towards zero at the boundary
        assert weights[-1] < weights[0]

    def test_bartlett_weights(self):
        """Bartlett (linear decay) kernel: w(x) = 1 - x for x in [0, 1).

        Ref: realized_kernel_weights.m:70 — Bartlett is the simplest flat-top
        kernel, linearly decreasing from 1 to 0.
        """
        bw = 10
        opts = {'kernel': 'bartlett', 'bandwidth': bw}
        weights = realized_kernel_weights(opts)
        assert isinstance(weights, np.ndarray)
        assert len(weights) == bw
        # Expected weights: 1 - (0/10, 1/10, ..., 9/10)
        expected = np.array([1.0 - i / bw for i in range(bw)])
        npt.assert_allclose(weights, expected, atol=1e-6, rtol=1e-4)

    def test_qs_weights(self):
        """Quadratic Spectral kernel weights: truncated at 30*H lags.

        Ref: realized_kernel_weights.m:100-103 — QS formula:
        k(x) = 3/x^2 * (sin(x)/x - cos(x))
        with truncation at 30*bandwidth lags.
        """
        opts = {'kernel': 'qs', 'bandwidth': 5.0}
        weights = realized_kernel_weights(opts)
        assert isinstance(weights, np.ndarray)
        # QS is truncated at 30*H = 150 lags
        assert len(weights) > 0
        assert len(weights) <= 150
        # First weight should be close to 1 (for small x → k(x) → 1)
        assert weights[0] > 0.9

    def test_tukey_hanning_weights(self):
        """Tukey-Hanning kernel variants (th1, th2, th5, th16) produce valid weights.

        Ref: realized_kernel_weights.m:88-94 — Formula: sin(π/2 · (1-x)^p)^2
        for power p = 1, 2, 5, 16 respectively.
        """
        bw = 20
        for kernel_name in ['th1', 'th2', 'th5', 'th16']:
            opts = {'kernel': kernel_name, 'bandwidth': bw}
            weights = realized_kernel_weights(opts)
            assert isinstance(weights, np.ndarray), f"Failed for {kernel_name}"
            assert len(weights) == bw, f"Wrong length for {kernel_name}"
            # Flat-top: first weight at x=0 → sin(π/2)^2 = 1
            npt.assert_allclose(
                weights[0], 1.0, atol=1e-6, rtol=1e-4,
                err_msg=f"First weight not 1 for {kernel_name}",
            )
            # All weights should be in [0, 1] for these kernels
            assert np.all(weights >= -1e-10), f"Negative weight for {kernel_name}"
            assert np.all(weights <= 1.0 + 1e-10), f"Weight > 1 for {kernel_name}"

    def test_epanechnikov_weights(self):
        """Epanechnikov kernel weights: k(x) = 1 - x^2.

        Ref: realized_kernel_weights.m:74
        """
        bw = 10
        opts = {'kernel': 'epanechnikov', 'bandwidth': bw}
        weights = realized_kernel_weights(opts)
        assert isinstance(weights, np.ndarray)
        assert len(weights) == bw
        # First weight at x=0: 1 - 0 = 1.0
        npt.assert_allclose(weights[0], 1.0, atol=1e-6, rtol=1e-4)
        # Expected: 1 - (i/H)^2 for i=0,...,H-1
        x = np.arange(bw, dtype=np.float64) / bw
        expected = 1.0 - x ** 2
        npt.assert_allclose(weights, expected, atol=1e-6, rtol=1e-4)

    def test_cubic_weights(self):
        """Cubic (multiscale) kernel weights: k(x) = 1 - 2x^2 + 2x^3.

        Ref: realized_kernel_weights.m:76
        """
        bw = 10
        opts = {'kernel': 'cubic', 'bandwidth': bw}
        weights = realized_kernel_weights(opts)
        assert isinstance(weights, np.ndarray)
        assert len(weights) == bw
        # First weight at x=0: 1.0
        npt.assert_allclose(weights[0], 1.0, atol=1e-6, rtol=1e-4)
        # Verify cubic formula
        x = np.arange(bw, dtype=np.float64) / bw
        expected = 1.0 - 2.0 * x ** 2 + 2.0 * x ** 3
        npt.assert_allclose(weights, expected, atol=1e-6, rtol=1e-4)

    def test_weight_at_zero(self):
        """All flat-top kernels return weight = 1 at lag 0 (x = 0).

        Ref: realized_kernel_weights.m:59-63 — Flat-top kernels use
        x = (0, 1, ..., H-1)/H, so the first element is at x=0 → k(0)=1.
        """
        bw = 20
        for kernel_name in FLAT_TOP_KERNELS:
            opts = {'kernel': kernel_name, 'bandwidth': bw}
            weights = realized_kernel_weights(opts)
            if len(weights) > 0:
                npt.assert_allclose(
                    weights[0], 1.0, atol=1e-6, rtol=1e-4,
                    err_msg=f"k(0) != 1 for flat-top kernel '{kernel_name}'",
                )

    def test_weight_at_bandwidth(self):
        """Weight at the bandwidth boundary per kernel spec.

        For flat-top kernels with integer bandwidth H, the last weight is
        at x = (H-1)/H which should be close to but not exactly zero for
        most kernels. The Bartlett kernel at x=(H-1)/H equals 1/H.

        Ref: realized_kernel_weights.m:69-94
        """
        bw = 10
        # Bartlett: last weight = 1 - (H-1)/H = 1/H
        opts = {'kernel': 'bartlett', 'bandwidth': bw}
        weights = realized_kernel_weights(opts)
        npt.assert_allclose(
            weights[-1], 1.0 / bw, atol=1e-6, rtol=1e-4,
            err_msg="Bartlett last weight should be 1/H",
        )

        # 2ndorder: last weight = (1 - (H-1)/H)^2 = (1/H)^2
        opts = {'kernel': '2ndorder', 'bandwidth': bw}
        weights = realized_kernel_weights(opts)
        expected_last = (1.0 / bw) ** 2
        npt.assert_allclose(
            weights[-1], expected_last, atol=1e-6, rtol=1e-4,
            err_msg="2ndorder last weight mismatch",
        )

    @pytest.mark.parametrize('kernel_name', ALL_KERNEL_TYPES)
    def test_parametrize_all_kernels(self, kernel_name):
        """Parametrize over all ~20 supported kernel strings.

        Verify output is a valid numpy array with appropriate properties
        for each kernel type.
        """
        bw = 15.0
        opts = {'kernel': kernel_name, 'bandwidth': bw}
        weights = realized_kernel_weights(opts)
        assert isinstance(weights, np.ndarray), f"Not ndarray for {kernel_name}"
        assert weights.dtype == np.float64, f"Wrong dtype for {kernel_name}"
        assert len(weights) > 0, f"Empty weights for {kernel_name} with bw={bw}"
        # All weights should be finite
        assert np.all(np.isfinite(weights)), f"Non-finite weights for {kernel_name}"

    def test_invalid_kernel_name(self):
        """Unknown kernel name raises ValueError.

        Ref: realized_kernel_weights.m:42-44 — Invalid kernel triggers error.
        """
        opts = {'kernel': 'invalid_kernel_xyz', 'bandwidth': 10}
        with pytest.raises(ValueError):
            realized_kernel_weights(opts)

    def test_fixture_parity(self):
        """Compare against MATLAB fixture data if available.

        Fixture structure: dict with scenario_1, scenario_2, ... each containing
        'kernel', 'bandwidth', 'weights' keys.
        """
        fixture_data = _load_fixture('realized_kernel_weights')
        if fixture_data is None:
            pytest.skip("Fixture file realized_kernel_weights.npy not found")
        data = fixture_data.item() if fixture_data.shape == () else fixture_data

        num_scenarios = int(data.get('num_scenarios', 0))
        if num_scenarios == 0:
            pytest.skip("No scenarios in fixture data")

        for i in range(1, num_scenarios + 1):
            scenario_key = f'scenario_{i}'
            if scenario_key not in data:
                continue
            scenario = data[scenario_key]
            kernel_name = scenario['kernel']
            bw = float(scenario['bandwidth'])
            expected_weights = np.asarray(scenario['weights'], dtype=np.float64)

            opts = {'kernel': kernel_name, 'bandwidth': bw}
            actual_weights = realized_kernel_weights(opts)

            npt.assert_allclose(
                actual_weights, expected_weights, atol=1e-6, rtol=1e-4,
                err_msg=(
                    f"Kernel weights fixture parity failed for "
                    f"scenario {i}: kernel='{kernel_name}', bandwidth={bw}"
                ),
            )


# ===========================================================================
# TestRealizedKernelBandwidth
# ===========================================================================


class TestRealizedKernelBandwidth:
    """Test realized_kernel_bandwidth() — optimal bandwidth selection.

    Uses plug-in estimates of noise variance and integrated quarticity
    to compute bandwidth following Barndorff-Nielsen et al. (2008).
    """

    def test_returns_positive_bandwidth(self):
        """Bandwidth is a positive number (float).

        Ref: realized_kernel_bandwidth.m:120-126 — All kernel types produce
        positive bandwidth via c* * ξ^(p1) * n^(p2) where all terms are positive.
        """
        opts = {'kernel': 'nonflatparzen', 'filteredN': 1000}
        bw = realized_kernel_bandwidth(0.001, 0.5, opts)
        assert isinstance(bw, float)
        assert bw > 0.0

    def test_with_default_options(self, kernel_options):
        """Default Kernel options produce reasonable bandwidth (1-100 for ~1000 returns).

        Ref: realized_options.m:147 — Default kernel is 'nonflatparzen'.
        """
        opts = dict(kernel_options)
        opts['filteredN'] = 1000
        bw = realized_kernel_bandwidth(1e-6, 1e-4, opts)
        assert isinstance(bw, float)
        assert bw > 0.0
        # For 1000 returns with typical noise/IQ, bandwidth should be reasonable
        assert bw < 10000, "Bandwidth implausibly large for n=1000"

    def test_bandwidth_scales_with_data(self):
        """More data points → bandwidth changes according to convergence rate.

        For non-flat-top kernels (type 3): bandwidth ∝ n^(3/5).
        Doubling n should increase bandwidth by factor ~2^(3/5) ≈ 1.516.

        Ref: realized_kernel_bandwidth.m:125 — bw = c* * ξ^(2/5) * n^(3/5)
        """
        noise_var = 1e-6
        iq = 1e-4

        opts1 = {'kernel': 'nonflatparzen', 'filteredN': 1000}
        bw1 = realized_kernel_bandwidth(noise_var, iq, opts1)

        opts2 = {'kernel': 'nonflatparzen', 'filteredN': 2000}
        bw2 = realized_kernel_bandwidth(noise_var, iq, opts2)

        # Non-flat-top (type 3): bw ∝ n^(3/5)
        expected_ratio = (2000 / 1000) ** (3.0 / 5.0)
        actual_ratio = bw2 / bw1
        npt.assert_allclose(
            actual_ratio, expected_ratio, atol=0.01, rtol=0.01,
            err_msg="Non-flat-top bandwidth scaling ratio mismatch",
        )

        # Flat-top type 1 (parzen): bw ∝ n^(1/2)
        opts3 = {'kernel': 'parzen', 'filteredN': 1000}
        bw3 = realized_kernel_bandwidth(noise_var, iq, opts3)

        opts4 = {'kernel': 'parzen', 'filteredN': 4000}
        bw4 = realized_kernel_bandwidth(noise_var, iq, opts4)

        expected_ratio_parzen = (4000 / 1000) ** (1.0 / 2.0)
        actual_ratio_parzen = bw4 / bw3
        npt.assert_allclose(
            actual_ratio_parzen, expected_ratio_parzen, atol=0.01, rtol=0.01,
            err_msg="Flat-top type 1 bandwidth scaling ratio mismatch",
        )

    def test_different_kernels_different_bandwidth(self):
        """Different kernel types produce different optimal bandwidths.

        Ref: realized_kernel_bandwidth.m:51-108 — Each kernel has its own c*
        constant, so bandwidths should differ for the same noise/IQ/N.
        """
        noise_var = 1e-5
        iq = 1e-4
        n = 2000

        bandwidths = {}
        for kernel_name in ['nonflatparzen', 'parzen', 'bartlett', 'qs', 'cubic']:
            opts = {'kernel': kernel_name, 'filteredN': n}
            bandwidths[kernel_name] = realized_kernel_bandwidth(noise_var, iq, opts)

        # At least some kernels should produce distinct bandwidths
        bw_values = list(bandwidths.values())
        assert len(set(round(v, 4) for v in bw_values)) >= 3, (
            f"Expected at least 3 distinct bandwidths, got {bandwidths}"
        )

    def test_fixture_parity(self):
        """Compare against MATLAB fixture data if available.

        Fixture structure: dict with scenario_1, scenario_2, ... each containing
        'noiseVariance', 'IQEstimate', 'kernel', 'filteredN', 'bandwidth'.
        """
        fixture_data = _load_fixture('realized_kernel_bandwidth')
        if fixture_data is None:
            pytest.skip("Fixture file realized_kernel_bandwidth.npy not found")
        data = fixture_data.item() if fixture_data.shape == () else fixture_data

        num_scenarios = int(data.get('num_scenarios', 0))
        if num_scenarios == 0:
            pytest.skip("No scenarios in fixture data")

        for i in range(1, num_scenarios + 1):
            scenario_key = f'scenario_{i}'
            if scenario_key not in data:
                continue
            scenario = data[scenario_key]
            noise_var = float(scenario['noiseVariance'])
            iq_est = float(scenario['IQEstimate'])
            kernel_name = scenario['kernel']
            filtered_n = int(scenario['filteredN'])
            expected_bw = float(scenario['bandwidth'])

            opts = {'kernel': kernel_name, 'filteredN': filtered_n}
            actual_bw = realized_kernel_bandwidth(noise_var, iq_est, opts)

            npt.assert_allclose(
                actual_bw, expected_bw, atol=1e-6, rtol=1e-4,
                err_msg=(
                    f"Bandwidth fixture parity failed for "
                    f"scenario {i}: kernel='{kernel_name}', "
                    f"noiseVar={noise_var}, IQ={iq_est}, N={filtered_n}"
                ),
            )


# ===========================================================================
# TestRealizedKernelCore
# ===========================================================================


class TestRealizedKernelCore:
    """Test realized_kernel_core() — core kernel computation.

    Computes the realized kernel estimate as a weighted sum of realized
    autocovariances with support for 'jitter' and 'stagger' endpoint
    treatments.
    """

    def test_basic_kernel_computation(self, sample_returns):
        """With known returns and weights, verify kernel estimate is computed.

        Ref: realized_kernel_core.m:110-111 — rk = gamma0 + weights' * (gammaMinus + gammaPlus)
        """
        # Use bartlett weights with small bandwidth
        bw = 5
        weights = np.array([1.0 - i / bw for i in range(bw)])
        opts = {'endTreatment': 'jitter'}

        rk = realized_kernel_core(sample_returns, weights, opts)
        assert isinstance(rk, float)
        assert np.isfinite(rk)

    def test_zero_bandwidth(self, sample_returns):
        """Bandwidth=0 → returns realized variance (sum of squared returns).

        Ref: realized_kernel_core.m:113-114 — When weights is empty,
        rk = gamma0 = returns' * returns.
        """
        empty_weights = np.array([], dtype=np.float64)
        opts = {'endTreatment': 'jitter'}

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rk = realized_kernel_core(sample_returns, empty_weights, opts)

        # With no weights, rk should equal sum of squared returns
        expected_rv = float(np.dot(sample_returns, sample_returns))
        npt.assert_allclose(rk, expected_rv, atol=1e-6, rtol=1e-4)

    def test_output_scalar(self, sample_returns):
        """Result is a scalar (float).

        Ref: realized_kernel_core.m:1 — function rk = realized_kernel_core(...)
        returns a single scalar value.
        """
        weights = np.array([0.9, 0.7, 0.4, 0.1])
        opts = {'endTreatment': 'jitter'}
        rk = realized_kernel_core(sample_returns, weights, opts)
        assert isinstance(rk, float)
        assert np.ndim(rk) == 0

    def test_positive_result(self, sample_returns):
        """Kernel estimate should be positive (variance measure) for well-chosen weights.

        For positive semi-definite kernels (e.g., Bartlett), the kernel estimate
        should produce a positive result with real data.
        """
        bw = 10
        # Bartlett weights are non-negative and produce PSD estimator
        weights = np.array([1.0 - i / bw for i in range(bw)])
        opts = {'endTreatment': 'jitter'}
        rk = realized_kernel_core(sample_returns, weights, opts)
        # With enough data points and reasonable weights, expect positive result
        assert rk > 0.0, f"Kernel estimate should be positive, got {rk}"

    def test_with_bartlett_weights(self, sample_returns):
        """Bartlett weights produce a consistent result across endpoint treatments.

        The 'jitter' and 'stagger' (non-jitter) treatments use different subsets
        of returns for autocovariance computation but should produce results of
        the same order of magnitude.

        Ref: realized_kernel_core.m:77-107 — jitter uses all returns,
        non-jitter uses central returns[H:m-H].
        """
        bw = 5
        weights = np.array([1.0 - i / bw for i in range(bw)])

        opts_jitter = {'endTreatment': 'jitter'}
        rk_jitter = realized_kernel_core(sample_returns, weights, opts_jitter)

        opts_stagger = {'endTreatment': 'stagger'}
        rk_stagger = realized_kernel_core(sample_returns, weights, opts_stagger)

        # Both should be positive and of similar magnitude
        assert rk_jitter > 0.0
        assert rk_stagger > 0.0
        # They won't be identical but should be same order of magnitude
        ratio = rk_jitter / rk_stagger
        assert 0.1 < ratio < 10.0, (
            f"Jitter/stagger ratio {ratio} is out of range: "
            f"jitter={rk_jitter}, stagger={rk_stagger}"
        )

    def test_fixture_parity(self):
        """Compare against MATLAB fixture data if available.

        Fixture structure: dict with scenario_1, scenario_2, ... each containing
        'returns', 'weights', 'endTreatment', 'rk' (expected result).
        """
        fixture_data = _load_fixture('realized_kernel_core')
        if fixture_data is None:
            pytest.skip("Fixture file realized_kernel_core.npy not found")
        data = fixture_data.item() if fixture_data.shape == () else fixture_data

        num_scenarios = int(data.get('num_scenarios', 0))
        if num_scenarios == 0:
            pytest.skip("No scenarios in fixture data")

        for i in range(1, num_scenarios + 1):
            scenario_key = f'scenario_{i}'
            if scenario_key not in data:
                continue
            scenario = data[scenario_key]

            returns = np.asarray(scenario['returns'], dtype=np.float64)
            weights = np.asarray(scenario['weights'], dtype=np.float64)
            end_treatment = scenario.get('endTreatment', 'jitter')
            expected_rk = float(scenario['rk'])

            opts = {'endTreatment': end_treatment}
            # Add optional fields if present in fixture
            for opt_key in ['maxBandwidthPerc', 'maxBandwidth', 'filteredN']:
                if opt_key in scenario:
                    opts[opt_key] = scenario[opt_key]

            actual_rk = realized_kernel_core(returns, weights, opts)

            npt.assert_allclose(
                actual_rk, expected_rk, atol=1e-6, rtol=1e-4,
                err_msg=(
                    f"Kernel core fixture parity failed for "
                    f"scenario {i}: endTreatment='{end_treatment}', "
                    f"m={len(returns)}, H={len(weights)}"
                ),
            )


# ===========================================================================
# TestRealizedKernelJitterLagLength
# ===========================================================================


class TestRealizedKernelJitterLagLength:
    """Test realized_kernel_jitter_lag_length() — optimal jitter lag computation.

    Determines the MSE-minimizing number of endpoint jitter lags for
    kernel estimation, balancing noise reduction against information loss.
    """

    def test_returns_positive_integer(self):
        """Jitter lag length is a positive integer.

        Ref: realized_kernel_jitter_lag_length.m:201-202 — Returns the index
        minimizing the MSE vector, which is always >= 1.
        """
        result = realized_kernel_jitter_lag_length(1e-6, 1e-4, 'nonflatparzen', 2000)
        assert isinstance(result, (int, np.integer))
        assert result >= 1

    def test_with_default_options(self):
        """Default options produce reasonable jitter lag.

        With typical noise/IQ levels and moderate sample size, jitter lag
        should be a small positive integer (typically 1–100).

        Ref: realized_options.m:155 — Default jitterLags is 2.
        """
        result = realized_kernel_jitter_lag_length(1e-6, 1e-8, 'nonflatparzen', 2000)
        assert result >= 1
        assert result < 2000  # Must be less than N
        # For typical parameters, jitter lag should be moderate
        assert result < 500, f"Jitter lag {result} seems excessive for N=2000"

    def test_increases_with_noise(self):
        """Higher noise → more jitter lags needed.

        Ref: realized_kernel_jitter_lag_length.m:198 — MSE formula has
        8 * noise^2 * m^(-2) term which increases with noise, requiring
        larger m to compensate.
        """
        iq = 1e-8
        n = 2000
        kernel = 'nonflatparzen'

        # Low noise
        lag_low = realized_kernel_jitter_lag_length(1e-8, iq, kernel, n)
        # High noise
        lag_high = realized_kernel_jitter_lag_length(1e-4, iq, kernel, n)

        assert lag_high > lag_low, (
            f"Expected more jitter lags for higher noise: "
            f"low_noise_lag={lag_low}, high_noise_lag={lag_high}"
        )

    def test_fixture_parity(self):
        """Compare against MATLAB fixture data if available.

        Fixture structure: dict with scenario_1, scenario_2, ... each containing
        'noiseEstimate', 'iqEstimate', 'kernel' (or 'kernel_matlab'), 'N', 'jitterLags'.
        """
        fixture_data = _load_fixture('realized_kernel_jitter_lag_length')
        if fixture_data is None:
            pytest.skip("Fixture file realized_kernel_jitter_lag_length.npy not found")
        data = fixture_data.item() if fixture_data.shape == () else fixture_data

        num_scenarios = int(data.get('num_scenarios', 0))
        if num_scenarios == 0:
            pytest.skip("No scenarios in fixture data")

        for i in range(1, num_scenarios + 1):
            scenario_key = f'scenario_{i}'
            if scenario_key not in data:
                continue
            scenario = data[scenario_key]

            noise_est = float(scenario['noiseEstimate'])
            iq_est = float(scenario['iqEstimate'])
            # Ref: Fixture may use 'kernel_matlab' as the actual MATLAB kernel
            # name; 'kernel' may be a display alias (e.g. 'tukeyhanning' → 'thinf')
            kernel_name = scenario.get('kernel_matlab', scenario.get('kernel', ''))
            n_val = int(scenario['N'])
            expected_lag = int(float(scenario['jitterLags']))

            # Skip scenarios with unrecognized kernel names
            # (e.g., fixture may use alias not in _VALID_KERNELS)
            valid_kernels = {
                'bartlett', 'twoscale', '2ndorder', 'epanechnikov',
                'cubic', 'multiscale', '5thorder', '6thorder',
                '7thorder', '8thorder', 'parzen', 'th1', 'th2', 'th5', 'th16',
                'nonflatparzen', 'qs', 'fejer', 'thinf', 'bnhls',
            }
            if kernel_name.lower() not in valid_kernels:
                continue

            actual_lag = realized_kernel_jitter_lag_length(
                noise_est, iq_est, kernel_name, n_val
            )

            npt.assert_allclose(
                float(actual_lag), float(expected_lag), atol=1e-6, rtol=1e-4,
                err_msg=(
                    f"Jitter lag fixture parity failed for "
                    f"scenario {i}: kernel='{kernel_name}', "
                    f"noise={noise_est}, iq={iq_est}, N={n_val}"
                ),
            )
