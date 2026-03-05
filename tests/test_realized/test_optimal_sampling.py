"""
Parametrized pytest parity tests for the ``realized_variance_optimal_sampling``
module — Bandi-Russell optimal sampling frequency estimation for realized
variance.

This module exercises the Python translation of
``realized/realized_variance_optimal_sampling.m`` from the MFE Toolbox
(Kevin Sheppard, Version 4.0).  Tests cover:

* Basic execution and return-type validation
* All three time-type input paths (``'wall'``, ``'seconds'``, ``'unit'``)
* Options-structure pass-through from ``realized_options('Optimal Sampling')``
* Noise sensitivity (noisier data → coarser optimal frequency)
* Consistency with naïve realized variance at reasonable frequencies
* Diagnostics output structure verification
* MATLAB fixture parity at ``atol=1e-6, rtol=1e-4``

Per AAP Section 0.7.1: All assertions use
``numpy.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-4)``.

Ref: realized_test.m — ``rvOS = realized_variance_optimal_sampling(data, ...)``
"""

from __future__ import annotations

import os

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.realized.realized_variance_optimal_sampling import (
    realized_variance_optimal_sampling,
)
from mfe_toolbox.realized.realized_options import realized_options
from mfe_toolbox.realized.wall2seconds import wall2seconds
from mfe_toolbox.realized.seconds2wall import seconds2wall


# ---------------------------------------------------------------------------
# Fixture Directory Resolution
# ---------------------------------------------------------------------------
# Per AAP Section 0.7.2: Optional MFE_FIXTURE_DIR env var for CI override.
FIXTURE_DIR: str = os.environ.get(
    'MFE_FIXTURE_DIR',
    os.path.join(os.path.dirname(__file__), '..', 'fixtures', 'realized'),
)


def _load_fixture(name: str):
    """Load a ``.npy`` fixture file from the realized fixture directory.

    Returns the loaded array/object if the file exists, or ``None`` otherwise.
    This allows tests to gracefully skip when fixtures have not been generated.
    """
    path = os.path.join(FIXTURE_DIR, name)
    if os.path.exists(path):
        return np.load(path, allow_pickle=True)
    return None


# ---------------------------------------------------------------------------
# Shared Pytest Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def hf_prices():
    """High-frequency price data suitable for optimal sampling analysis.

    Generates a synthetic price path of length 5001 with microstructure noise,
    using a reproducible seed (42) for deterministic test behaviour.

    Ref: AAP Section 0.7.3 — ``randn`` replaced by ``rng.standard_normal``.
    """
    rng = np.random.default_rng(42)
    n = 5000
    # Generate true log-returns and cumulate to log-price
    r_true = rng.standard_normal(n) / np.sqrt(n)
    log_p = np.cumsum(np.concatenate([[0.0], r_true]))
    # Add microstructure noise (bid-ask bounce proxy)
    noise = rng.normal(0, 0.0005, n + 1)
    return np.exp(log_p + noise)


@pytest.fixture
def hf_times_seconds(hf_prices):
    """Uniformly-spaced time grid in seconds-past-midnight from 9:30 to 16:00.

    Ref: realized_variance_optimal_sampling.m:97 — default time grid uses
    wall2seconds(93000) to wall2seconds(160000), i.e. 34200 to 57600 seconds.
    """
    return np.linspace(34200, 57600, len(hf_prices))


@pytest.fixture
def hf_times_wall(hf_times_seconds):
    """Wall-clock (HHMMSS) time vector converted from seconds via seconds2wall.

    Used by ``test_with_wall_time`` and ``test_parametrize_time_types``.
    """
    return seconds2wall(hf_times_seconds)


@pytest.fixture
def optimal_sampling_options():
    """Default options dict for the 'Optimal Sampling' estimator type.

    Ref: realized_options.m — returns dict with sampling-specific defaults
    including ``noiseVarianceSamplingType='BusinessTime'``,
    ``noiseVarianceSamplingInterval=1``, and ``useAdjustedNoiseCount=False``.
    """
    return realized_options('Optimal Sampling')


# ---------------------------------------------------------------------------
# Test Class
# ---------------------------------------------------------------------------


class TestRealizedVarianceOptimalSampling:
    """Tests for :func:`realized_variance_optimal_sampling`.

    Covers execution, return values, all time-type paths, options pass-through,
    noise sensitivity, consistency, diagnostics, and MATLAB fixture parity.
    """

    # ------------------------------------------------------------------
    # 1. test_basic_execution
    # ------------------------------------------------------------------
    def test_basic_execution(self, hf_prices, hf_times_seconds):
        """Function returns a 5-tuple (rv, rvD, rvSS, rvDSS, diagnostics)."""
        result = realized_variance_optimal_sampling(
            hf_prices,
            hf_times_seconds,
            time_type='seconds',
            sampling_type='BusinessTime',
            sampling_interval=1,
        )
        assert isinstance(result, tuple), 'Expected a tuple return'
        assert len(result) == 5, 'Expected 5-element tuple'
        rv, rv_debiased, rv_ss, rv_debiased_ss, diagnostics = result
        # First four are scalars (float), last is dict
        assert isinstance(rv, (float, np.floating)), 'rv should be float'
        assert isinstance(rv_debiased, (float, np.floating)), 'rv_debiased should be float'
        assert isinstance(rv_ss, (float, np.floating)), 'rv_ss should be float'
        assert isinstance(rv_debiased_ss, (float, np.floating)), 'rv_debiased_ss should be float'
        assert isinstance(diagnostics, dict), 'diagnostics should be dict'

    # ------------------------------------------------------------------
    # 2. test_returns_positive_rv
    # ------------------------------------------------------------------
    def test_returns_positive_rv(self, hf_prices, hf_times_seconds):
        """Optimal-sampling RV (standard) is strictly positive.

        Realized variance of non-constant prices must be positive.
        The debiased variant may be negative due to noise subtraction,
        so only the standard RV is checked here.
        """
        rv, _rv_d, rv_ss, _rv_dss, _diag = realized_variance_optimal_sampling(
            hf_prices,
            hf_times_seconds,
            time_type='seconds',
            sampling_type='BusinessTime',
            sampling_interval=1,
        )
        assert rv > 0.0, f'Standard RV should be positive, got {rv}'
        assert rv_ss > 0.0, f'Subsampled RV should be positive, got {rv_ss}'

    # ------------------------------------------------------------------
    # 3. test_optimal_frequency_reasonable
    # ------------------------------------------------------------------
    def test_optimal_frequency_reasonable(self, hf_prices, hf_times_seconds):
        """Optimal number of samples lies in a reasonable range [2, len(prices)-1].

        Ref: realized_variance_optimal_sampling.m:141-149 — clamped to [2, m].
        """
        _rv, _rvd, _rvss, _rvdss, diagnostics = realized_variance_optimal_sampling(
            hf_prices,
            hf_times_seconds,
            time_type='seconds',
            sampling_type='BusinessTime',
            sampling_interval=1,
        )
        opt_n = diagnostics['optimalSamples']
        samples_used = diagnostics['samples']
        m = diagnostics['m']
        # optimalSamples is the raw Bandi-Russell estimate (may exceed m)
        # samples is the clamped value used for RV computation
        assert samples_used >= 2, f'Samples used should be >= 2, got {samples_used}'
        assert samples_used <= m, f'Samples used should be <= m={m}, got {samples_used}'
        # The optimal samples estimate should be a positive integer
        assert opt_n >= 1, f'Optimal samples should be >= 1, got {opt_n}'

    # ------------------------------------------------------------------
    # 4. test_with_wall_time
    # ------------------------------------------------------------------
    def test_with_wall_time(self, hf_prices, hf_times_wall):
        """Function works correctly with timeType='wall' input.

        Wall-clock HHMMSS times are converted internally by the function.
        """
        rv, rv_d, rv_ss, rv_dss, diag = realized_variance_optimal_sampling(
            hf_prices,
            hf_times_wall,
            time_type='wall',
            sampling_type='BusinessTime',
            sampling_interval=1,
        )
        assert isinstance(rv, (float, np.floating))
        assert isinstance(diag, dict)
        assert diag['samples'] >= 2

    # ------------------------------------------------------------------
    # 5. test_with_seconds_time
    # ------------------------------------------------------------------
    def test_with_seconds_time(self, hf_prices, hf_times_seconds):
        """Function works correctly with timeType='seconds' input."""
        rv, rv_d, rv_ss, rv_dss, diag = realized_variance_optimal_sampling(
            hf_prices,
            hf_times_seconds,
            time_type='seconds',
            sampling_type='BusinessTime',
            sampling_interval=1,
        )
        assert isinstance(rv, (float, np.floating))
        assert isinstance(diag, dict)
        assert diag['samples'] >= 2

    # ------------------------------------------------------------------
    # 6. test_with_unit_time
    # ------------------------------------------------------------------
    def test_with_unit_time(self, hf_prices):
        """Function accepts timeType='unit' input.

        Ref: MATLAB fixture scenario_4 — ``realized_noise_estimate.m``
        does not support 'unit' timeType, resulting in an error in
        MATLAB/Octave.  The Python implementation may raise ValueError or
        produce a result depending on how realized_noise_estimate.py was
        translated.  We verify that the function either succeeds
        (returning a valid tuple) or raises a ValueError — but never
        silently corrupts data.
        """
        n = len(hf_prices)
        hf_times_unit = np.linspace(0.0, 1.0, n)
        try:
            result = realized_variance_optimal_sampling(
                hf_prices,
                hf_times_unit,
                time_type='unit',
                sampling_type='BusinessTime',
                sampling_interval=1,
            )
            # If it succeeds, verify shape of return
            assert len(result) == 5, 'Expected 5-element return tuple'
            rv = result[0]
            assert isinstance(rv, (float, np.floating))
        except (ValueError, RuntimeError):
            # Acceptable — mirrors MATLAB error for 'unit' timeType
            pass

    # ------------------------------------------------------------------
    # 7. test_parametrize_time_types
    # ------------------------------------------------------------------
    @pytest.mark.parametrize('time_type_val', ['wall', 'seconds'])
    def test_parametrize_time_types(
        self, hf_prices, hf_times_seconds, hf_times_wall, time_type_val,
    ):
        """Parametrize across 'wall' and 'seconds' time types.

        Note: 'unit' is excluded from parametrization because the MATLAB
        fixture shows it can error inside realized_noise_estimate.
        Ref: fixture scenario_4.

        Both 'wall' and 'seconds' produce finite positive standard RV.
        """
        if time_type_val == 'wall':
            t = hf_times_wall
        else:
            t = hf_times_seconds

        rv, _rv_d, rv_ss, _rv_dss, diag = realized_variance_optimal_sampling(
            hf_prices,
            t,
            time_type=time_type_val,
            sampling_type='BusinessTime',
            sampling_interval=1,
        )
        assert np.isfinite(rv), f'rv should be finite for time_type={time_type_val}'
        assert rv > 0.0, f'rv should be positive for time_type={time_type_val}'
        assert isinstance(diag, dict)
        assert 'optimalSamples' in diag

    # ------------------------------------------------------------------
    # 8. test_with_options
    # ------------------------------------------------------------------
    def test_with_options(self, hf_prices, hf_times_seconds, optimal_sampling_options):
        """Pass explicit options from realized_options('Optimal Sampling').

        Verifies the options dict does not cause errors and produces the same
        result as the default options (since defaults are identical).
        """
        # Run with default options (None)
        rv_def, rvd_def, rvss_def, rvdss_def, diag_def = (
            realized_variance_optimal_sampling(
                hf_prices,
                hf_times_seconds,
                time_type='seconds',
                sampling_type='BusinessTime',
                sampling_interval=1,
                options=None,
            )
        )
        # Run with explicit options
        rv_opt, rvd_opt, rvss_opt, rvdss_opt, diag_opt = (
            realized_variance_optimal_sampling(
                hf_prices,
                hf_times_seconds,
                time_type='seconds',
                sampling_type='BusinessTime',
                sampling_interval=1,
                options=optimal_sampling_options,
            )
        )
        # Results should be identical since default options match
        npt.assert_allclose(rv_opt, rv_def, atol=1e-6, rtol=1e-4,
                            err_msg='Options pass-through: rv mismatch')
        npt.assert_allclose(rvd_opt, rvd_def, atol=1e-6, rtol=1e-4,
                            err_msg='Options pass-through: rv_debiased mismatch')
        npt.assert_allclose(rvss_opt, rvss_def, atol=1e-6, rtol=1e-4,
                            err_msg='Options pass-through: rv_ss mismatch')
        npt.assert_allclose(rvdss_opt, rvdss_def, atol=1e-6, rtol=1e-4,
                            err_msg='Options pass-through: rv_debiased_ss mismatch')
        # Diagnostics should match
        assert diag_opt['optimalSamples'] == diag_def['optimalSamples']
        assert diag_opt['samples'] == diag_def['samples']

    # ------------------------------------------------------------------
    # 9. test_noise_impact
    # ------------------------------------------------------------------
    def test_noise_impact(self, hf_times_seconds):
        """Noisier data leads to a different (typically coarser) optimal frequency.

        Per Bandi-Russell (2008): optimal N ~ (IQ / sigma_u^4)^(1/3).
        Higher noise variance (sigma_u^2) decreases the optimal N, leading
        to fewer (coarser) samples.

        Ref: realized_variance_optimal_sampling.m:140 — optimal N formula.
        """
        rng = np.random.default_rng(123)
        n = len(hf_times_seconds) - 1
        r_true = rng.standard_normal(n) / np.sqrt(n)
        log_p = np.cumsum(np.concatenate([[0.0], r_true]))

        # Low noise scenario
        noise_low = rng.normal(0, 0.0001, n + 1)
        prices_low = np.exp(log_p + noise_low)

        # High noise scenario (10x noise std dev)
        noise_high = rng.normal(0, 0.001, n + 1)
        prices_high = np.exp(log_p + noise_high)

        _, _, _, _, diag_low = realized_variance_optimal_sampling(
            prices_low,
            hf_times_seconds,
            time_type='seconds',
            sampling_type='BusinessTime',
            sampling_interval=1,
        )
        _, _, _, _, diag_high = realized_variance_optimal_sampling(
            prices_high,
            hf_times_seconds,
            time_type='seconds',
            sampling_type='BusinessTime',
            sampling_interval=1,
        )
        # Higher noise should yield fewer optimal samples (coarser frequency)
        # or at minimum a detectably different optimal count
        opt_low = diag_low['optimalSamples']
        opt_high = diag_high['optimalSamples']
        assert opt_low != opt_high, (
            f'Expected different optimal N for different noise levels, '
            f'both gave {opt_low}'
        )

    # ------------------------------------------------------------------
    # 10. test_consistent_with_standard_rv
    # ------------------------------------------------------------------
    def test_consistent_with_standard_rv(self, hf_prices, hf_times_seconds):
        """Optimal sampling RV is broadly consistent with naïve standard RV.

        The optimal-sampling RV should be within a reasonable order of
        magnitude of a naïve sum-of-squared-log-returns at the same frequency.
        This is a sanity check, not a strict numerical parity test.
        """
        rv, _rvd, _rvss, _rvdss, diag = realized_variance_optimal_sampling(
            hf_prices,
            hf_times_seconds,
            time_type='seconds',
            sampling_type='BusinessTime',
            sampling_interval=1,
        )
        # Compute naïve RV from all log returns
        log_returns = np.diff(np.log(hf_prices))
        naive_rv = np.sum(log_returns ** 2)
        # The optimal sampling RV may differ substantially from the naïve RV
        # (that's the whole point — it corrects for noise), but they should
        # be within ~2 orders of magnitude for well-behaved data
        ratio = rv / naive_rv if naive_rv > 0 else float('inf')
        assert 0.001 < ratio < 1000.0, (
            f'Optimal RV ({rv:.6e}) is too far from naïve RV '
            f'({naive_rv:.6e}), ratio={ratio:.4f}'
        )

    # ------------------------------------------------------------------
    # 11. test_diagnostics_output
    # ------------------------------------------------------------------
    def test_diagnostics_output(self, hf_prices, hf_times_seconds):
        """Diagnostics dict contains all expected fields with valid values.

        Ref: realized_variance_optimal_sampling.m:185-192 — diagnostics struct.
        """
        _, _, _, _, diagnostics = realized_variance_optimal_sampling(
            hf_prices,
            hf_times_seconds,
            time_type='seconds',
            sampling_type='BusinessTime',
            sampling_interval=1,
        )
        # Required fields per MATLAB source (lines 185-192)
        required_fields = [
            'm',
            'optimalSamples',
            'samples',
            'optimalNumberOfSamplesDebiased',
            'samplesDebiased',
            'noiseVariance',
            'debiasedNoiseVariance',
            'IQEstimate',
        ]
        for field in required_fields:
            assert field in diagnostics, (
                f"Missing diagnostics field: '{field}'"
            )

        # Validate field types and basic constraints
        # 'm' — number of filtered returns, must be positive int
        assert isinstance(diagnostics['m'], (int, np.integer))
        assert diagnostics['m'] > 0

        # 'optimalSamples' — positive integer
        assert isinstance(diagnostics['optimalSamples'], (int, np.integer))
        assert diagnostics['optimalSamples'] >= 1

        # 'samples' — integer in [2, m]
        assert isinstance(diagnostics['samples'], (int, np.integer))
        assert 2 <= diagnostics['samples'] <= diagnostics['m']

        # 'noiseVariance' — non-negative scalar
        assert np.isfinite(diagnostics['noiseVariance'])
        assert diagnostics['noiseVariance'] >= 0.0

        # 'IQEstimate' — non-negative scalar (integrated quarticity)
        assert np.isfinite(diagnostics['IQEstimate'])
        assert diagnostics['IQEstimate'] >= 0.0

        # 'debiasedNoiseVariance' — non-negative scalar
        assert np.isfinite(diagnostics['debiasedNoiseVariance'])
        assert diagnostics['debiasedNoiseVariance'] >= 0.0

    # ------------------------------------------------------------------
    # 12. test_fixture_parity
    # ------------------------------------------------------------------
    @pytest.mark.parity
    @pytest.mark.requires_fixtures
    def test_fixture_parity(self):
        """Load MATLAB fixture and compare Python outputs at atol=1e-6.

        The fixture file ``realized_variance_optimal_sampling.npy``
        contains price data, time vectors, and multiple scenario outputs
        generated by GNU Octave from the original MATLAB source.

        Per AAP Section 0.7.1: numpy.testing.assert_allclose(actual,
        expected, atol=1e-6, rtol=1e-4).
        """
        fixture_data = _load_fixture('realized_variance_optimal_sampling.npy')
        if fixture_data is None:
            pytest.skip(
                'Fixture file realized_variance_optimal_sampling.npy not found'
            )

        # The .npy file stores a 0-d object array wrapping a dict
        data = fixture_data.item() if fixture_data.ndim == 0 else fixture_data

        # Extract common inputs
        price = np.asarray(data['price'], dtype=np.float64)
        time_seconds = np.asarray(data['time_seconds'], dtype=np.float64)

        # Iterate over available scenarios
        scenario_keys = sorted(
            k for k in data.keys() if k.startswith('scenario_')
        )
        for skey in scenario_keys:
            scenario = data[skey]
            desc = scenario.get('description', skey)

            # Skip error scenarios — these represent MATLAB-side failures
            # that may or may not occur in the Python implementation
            if scenario.get('error', False):
                continue

            # Extract scenario parameters
            # Ref: MATLAB fixture uses camelCase parameter names
            time_type = scenario['timeType']
            sampling_type = scenario['samplingType']
            sampling_interval = int(scenario['samplingInterval'])
            subsamples = int(scenario['subsamples'])

            # Select appropriate time vector
            if time_type == 'seconds':
                time_vec = time_seconds
            elif time_type == 'wall':
                time_vec = seconds2wall(time_seconds)
            elif time_type == 'unit':
                time_vec = np.asarray(data.get(
                    'time_unit', np.linspace(0, 1, len(price)),
                ), dtype=np.float64)
            else:
                pytest.fail(f'Unknown timeType in fixture: {time_type}')

            # Expected outputs from MATLAB/Octave
            expected_rv = float(scenario['rv'])
            expected_rvd = float(scenario['rvDebiased'])
            expected_rvss = float(scenario['rvSS'])
            expected_rvdss = float(scenario['rvDebiasedSS'])
            expected_diag = scenario.get('diagnostics', {})

            # Run Python implementation
            rv, rvd, rvss, rvdss, diag = realized_variance_optimal_sampling(
                price,
                time_vec,
                time_type=time_type,
                sampling_type=sampling_type,
                sampling_interval=sampling_interval,
                subsamples=subsamples,
            )

            # Assert numerical parity for all scalar outputs
            npt.assert_allclose(
                rv, expected_rv, atol=1e-6, rtol=1e-4,
                err_msg=f'{desc}: rv mismatch',
            )
            npt.assert_allclose(
                rvd, expected_rvd, atol=1e-6, rtol=1e-4,
                err_msg=f'{desc}: rv_debiased mismatch',
            )
            npt.assert_allclose(
                rvss, expected_rvss, atol=1e-6, rtol=1e-4,
                err_msg=f'{desc}: rv_ss mismatch',
            )
            npt.assert_allclose(
                rvdss, expected_rvdss, atol=1e-6, rtol=1e-4,
                err_msg=f'{desc}: rv_debiased_ss mismatch',
            )

            # Assert diagnostics parity where available
            if expected_diag:
                if 'optimalSamples' in expected_diag:
                    assert diag['optimalSamples'] == int(
                        expected_diag['optimalSamples']
                    ), f'{desc}: optimalSamples mismatch'
                if 'samples' in expected_diag:
                    assert diag['samples'] == int(
                        expected_diag['samples']
                    ), f'{desc}: samples mismatch'
                if 'm' in expected_diag:
                    assert diag['m'] == int(
                        expected_diag['m']
                    ), f'{desc}: m mismatch'
                if 'noiseVariance' in expected_diag:
                    npt.assert_allclose(
                        diag['noiseVariance'],
                        float(expected_diag['noiseVariance']),
                        atol=1e-6, rtol=1e-4,
                        err_msg=f'{desc}: noiseVariance mismatch',
                    )
                if 'IQEstimate' in expected_diag:
                    npt.assert_allclose(
                        diag['IQEstimate'],
                        float(expected_diag['IQEstimate']),
                        atol=1e-6, rtol=1e-4,
                        err_msg=f'{desc}: IQEstimate mismatch',
                    )
                if 'debiasedNoiseVariance' in expected_diag:
                    npt.assert_allclose(
                        diag['debiasedNoiseVariance'],
                        float(expected_diag['debiasedNoiseVariance']),
                        atol=1e-6, rtol=1e-4,
                        err_msg=f'{desc}: debiasedNoiseVariance mismatch',
                    )
                if 'optimalNumberOfSamplesDebiased' in expected_diag:
                    assert diag['optimalNumberOfSamplesDebiased'] == int(
                        expected_diag['optimalNumberOfSamplesDebiased']
                    ), f'{desc}: optimalNumberOfSamplesDebiased mismatch'
                if 'samplesDebiased' in expected_diag:
                    assert diag['samplesDebiased'] == int(
                        expected_diag['samplesDebiased']
                    ), f'{desc}: samplesDebiased mismatch'
