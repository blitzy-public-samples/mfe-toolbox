"""
Optimal sampling frequency selection for realized variance (Bandi-Russell).

Implements the Bandi-Russell (2008) method for selecting the optimal number
of observations to use when computing realized variance in the presence of
microstructure noise.  Two variants are provided: a standard optimal sampling
estimator and a debiased optimal sampling estimator that corrects for finite-
sample noise bias.

The optimal number of intraday samples that minimises the mean-squared error
of the RV estimator is:

.. math::

    n^* = \\left( \\frac{IQ}{\\sigma_u^4} \\right)^{1/3}

where *IQ* is the integrated quarticity and :math:`\\sigma_u^2` is the noise
variance of log returns (not log prices).  The debiased variant uses Eq. (18)
from Bandi-Russell (2008):

.. math::

    n^*_D = \\left( \\frac{2 \\cdot IQ}{2 \\mu_4 - 3 \\sigma_u^4} \\right)^{1/2}

Migrated from: ``realized/realized_variance_optimal_sampling.m``
(MFE Toolbox Version 4.0, Kevin Sheppard)

See Also
--------
realized_variance : Standard realized variance estimator.
realized_kernel : Realized kernel quadratic variation estimator.
realized_quantile_variance : Quantile-based realized variance.
realized_range : Range-based volatility estimator.
realized_price_filter : Core price filtering function.
realized_threshold_variance : Threshold realized variance.

References
----------
.. [1] Bandi, F. M. and Russell, J. R. (2008). "Microstructure noise,
   realized variance, and optimal sampling." *Review of Economic Studies*,
   75(2), 339-369.
"""

from __future__ import annotations

import warnings

import numpy as np

from mfe_toolbox.realized.realized_noise_estimate import realized_noise_estimate
from mfe_toolbox.realized.realized_options import realized_options
from mfe_toolbox.realized.realized_price_filter import realized_price_filter
from mfe_toolbox.realized.realized_variance import realized_variance
from mfe_toolbox.realized.wall2seconds import wall2seconds


def realized_variance_optimal_sampling(
    price,
    time=None,
    time_type: str = 'seconds',
    sampling_type: str = 'businessuniform',
    sampling_interval=1,
    subsamples: int = 1,
    options: dict | None = None,
) -> tuple[float, float, float, float, dict]:
    """Estimate realized variance using Bandi-Russell optimal sampling selection.

    Computes the optimal number of intraday price observations to use when
    computing realized variance by minimizing the mean-squared error in the
    presence of microstructure noise.  Both a standard and a debiased variant
    are returned.

    Parameters
    ----------
    price : array_like
        m-by-1 vector of high-frequency prices.
    time : array_like or None, optional
        m-by-1 vector of observation times corresponding to *price*.  When
        ``None`` (default), a uniformly spaced grid from 9:30 AM to 4:00 PM
        in seconds-past-midnight is generated.
    time_type : str, optional
        Format of *time* values.  One of ``'wall'`` (HHMMSS), ``'seconds'``
        (seconds past midnight), or ``'unit'`` (normalised [0, 1]).
        Default is ``'seconds'``.
    sampling_type : str, optional
        Sampling scheme for the **maximum** frequency of returns.
        One of ``'CalendarTime'``, ``'CalendarUniform'``, ``'BusinessTime'``,
        ``'BusinessUniform'``, or ``'Fixed'``.  Default is ``'businessuniform'``.
    sampling_interval : int, float, or array_like, optional
        Interpretation depends on *sampling_type*.  Default is ``1``.
    subsamples : int, optional
        Number of subsample realized variance estimators to average with the
        original.  If larger than the computed optimal sample count, clamped
        to the optimal count.  Default is ``1``.
    options : dict or None, optional
        Options dictionary (see :func:`realized_options`).  When ``None``,
        defaults for ``'optimal sampling'`` are used.

    Returns
    -------
    rv : float
        Realized variance at the optimal sampling frequency.
    rv_debiased : float
        Debiased realized variance at the optimal debiased sampling frequency.
    rv_ss : float
        Subsampled realized variance at the optimal sampling frequency.
    rv_debiased_ss : float
        Subsampled debiased realized variance.
    diagnostics : dict
        Dictionary with estimation details:

        - ``'m'``                               : int   — number of filtered returns
        - ``'optimalSamples'``                  : int   — optimal N (standard)
        - ``'samples'``                         : int   — samples actually used (standard)
        - ``'optimalNumberOfSamplesDebiased'``  : int   — optimal N (debiased)
        - ``'samplesDebiased'``                 : int   — samples actually used (debiased)
        - ``'noiseVariance'``                   : float — noise variance (recalculated)
        - ``'debiasedNoiseVariance'``           : float — debiased noise variance (doubled)
        - ``'IQEstimate'``                      : float — integrated quarticity estimate

    Raises
    ------
    ValueError
        If any input fails validation (see parameter-check helper).

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.realized.realized_variance_optimal_sampling import (
    ...     realized_variance_optimal_sampling,
    ... )
    >>> rng = np.random.default_rng(42)
    >>> prices = 100.0 * np.exp(np.cumsum(rng.standard_normal(500) * 0.001))
    >>> rv, rvd, rvss, rvssd, diag = realized_variance_optimal_sampling(prices)

    Notes
    -----
    The role of *sampling_type* and *sampling_interval* is to set the
    **maximum** frequency of returns.  For example, to cap at 15-second
    frequency, use ``sampling_type='CalendarTime'`` and
    ``sampling_interval=15``.  To use all available data, use
    ``sampling_type='BusinessTime'`` and ``sampling_interval=1``.

    Copyright: Kevin Sheppard
    kevin.sheppard@economics.ox.ac.uk
    Revision: 1    Date: 5/1/2008
    """
    # ==================================================================
    # Default Argument Handling — replaces MATLAB nargin switch
    # Ref: realized_variance_optimal_sampling.m:95-115
    # ==================================================================

    # Ref: realized_variance_optimal_sampling.m:96-102 — nargin == 1
    if time is None:
        # Generate default time grid from 9:30 AM to 4:00 PM
        # Ref: realized_variance_optimal_sampling.m:97
        start_sec = float(wall2seconds(93000))
        end_sec = float(wall2seconds(160000))
        price_arr = np.asarray(price, dtype=np.float64).ravel()
        time = np.linspace(start_sec, end_sec, len(price_arr))
        time_type = 'seconds'
        sampling_type = 'businessuniform'
        sampling_interval = 1
        subsamples = 1
        options = realized_options('optimal sampling')
    else:
        # Ref: realized_variance_optimal_sampling.m:103-112
        # Fill in defaults for optional arguments
        if subsamples is None:
            # Ref: realized_variance_optimal_sampling.m:109-110
            subsamples = 1
        if options is None:
            # Ref: realized_variance_optimal_sampling.m:107
            options = realized_options('optimal sampling')

    # Make a working copy to avoid mutating caller's dict
    options = dict(options)

    # ==================================================================
    # Input Checking
    # Ref: realized_variance_optimal_sampling.m:117-120
    # ==================================================================
    error_message = _realized_variance_optimal_sampling_parameter_check(
        price, time, time_type, sampling_type, sampling_interval,
        subsamples, options,
    )
    if error_message is not None:
        raise ValueError(error_message)

    # ==================================================================
    # Core Computation
    # ==================================================================

    # Ref: realized_variance_optimal_sampling.m:125 — protect against integer times
    time = np.asarray(time, dtype=np.float64).ravel()
    price = np.asarray(price, dtype=np.float64).ravel()

    # ------------------------------------------------------------------
    # Step 1: Filter prices at user-specified maximum frequency
    # Ref: realized_variance_optimal_sampling.m:127
    # ------------------------------------------------------------------
    filter_result = realized_price_filter(
        price, time, time_type, sampling_type, sampling_interval,
    )
    filtered_price = np.asarray(filter_result[0], dtype=np.float64).ravel()
    filtered_time = np.asarray(filter_result[1], dtype=np.float64).ravel()

    # Ref: realized_variance_optimal_sampling.m:130 — m = size(filteredPrice,1)
    m = filtered_price.shape[0]

    # ------------------------------------------------------------------
    # Step 2: Estimate IQ and noise variance
    # Ref: realized_variance_optimal_sampling.m:133
    # realized_noise_estimate returns (noise_var, debiased_noise_var, iq, oomen)
    # ------------------------------------------------------------------
    noise_variance, debiased_noise_variance, iq_estimate, _ = (
        realized_noise_estimate(filtered_price, filtered_time, time_type, options)
    )

    # ------------------------------------------------------------------
    # Step 3: Double noise variances — these are the variance of the
    # error to returns, not to prices, which is what
    # realized_noise_estimate returns.
    # Ref: realized_variance_optimal_sampling.m:136-137
    # ------------------------------------------------------------------
    noise_variance = 2.0 * noise_variance
    debiased_noise_variance = 2.0 * debiased_noise_variance

    # ------------------------------------------------------------------
    # Step 4: Optimal number of samples — Eq. (16) from Bandi-Russell
    # Ref: realized_variance_optimal_sampling.m:140
    # ------------------------------------------------------------------
    if noise_variance != 0.0:
        optimal_n = int(np.round((iq_estimate / (noise_variance ** 2)) ** (1.0 / 3.0)))
    else:
        # If noise variance is zero, use all available data
        optimal_n = m

    # Ref: realized_variance_optimal_sampling.m:141-149 — clamp to [2, m]
    if optimal_n > m:
        # Ref: realized_variance_optimal_sampling.m:142
        warnings.warn(
            'Computed optimal number of samples exceeds the number of '
            'filtered prices. Using all available data.',
            stacklevel=2,
        )
        samples_used = m
    elif optimal_n < 2:
        # Ref: realized_variance_optimal_sampling.m:145
        warnings.warn(
            'Computed optimal number of samples is less than 2. '
            'Using minimum of 2.',
            stacklevel=2,
        )
        samples_used = 2
    else:
        samples_used = optimal_n

    # ------------------------------------------------------------------
    # Step 5: Compute RV at optimal sampling using BusinessTime
    # Ref: realized_variance_optimal_sampling.m:152-153
    # ------------------------------------------------------------------
    actual_subsamples = min(subsamples, samples_used)
    # realized_variance returns (rv, rv_ss, diagnostics) — extract first two
    rv_result = realized_variance(
        filtered_price, filtered_time, time_type,
        'BusinessTime', samples_used, actual_subsamples,
    )
    rv = float(rv_result[0])
    rv_ss = float(rv_result[1])

    # ==================================================================
    # Debiased Variant
    # Ref: realized_variance_optimal_sampling.m:155-183
    # ==================================================================

    # ------------------------------------------------------------------
    # Step 6: Filter prices for noise estimation at noise sampling freq
    # Ref: realized_variance_optimal_sampling.m:155
    # ------------------------------------------------------------------
    noise_filter_result = realized_price_filter(
        filtered_price, filtered_time, time_type,
        options['noiseVarianceSamplingType'],
        options['noiseVarianceSamplingInterval'],
    )
    filtered_price_noise = np.asarray(
        noise_filter_result[0], dtype=np.float64,
    ).ravel()

    # ------------------------------------------------------------------
    # Step 7: Compute log returns for noise 4th moment estimation
    # Ref: realized_variance_optimal_sampling.m:156
    # ------------------------------------------------------------------
    returns = np.diff(np.log(filtered_price_noise))

    # Ref: realized_variance_optimal_sampling.m:157-160
    if options.get('useAdjustedNoiseCount', False):
        # Ref: realized_variance_optimal_sampling.m:158 — only count non-zero returns
        n = int(np.sum(returns != 0))
    else:
        # Ref: realized_variance_optimal_sampling.m:160
        n = len(returns)

    # Protect against division by zero
    if n <= 0:
        n = 1

    # ------------------------------------------------------------------
    # Step 8: Noise 4th power and recalculated noise variance
    # Ref: realized_variance_optimal_sampling.m:162-163
    # NOTE: This OVERWRITES the earlier noise_variance variable, matching
    #       the MATLAB source exactly.
    # ------------------------------------------------------------------
    noise4th_power = float(np.sum(returns ** 4) / n)
    # Ref: realized_variance_optimal_sampling.m:163
    noise_variance = float(np.sum(returns ** 2) / n)

    # ------------------------------------------------------------------
    # Step 9: Optimal debiased number of samples — Eq. (18) Bandi-Russell
    # Ref: realized_variance_optimal_sampling.m:166
    # ------------------------------------------------------------------
    denominator = 2.0 * noise4th_power - 3.0 * noise_variance ** 2
    if denominator > 0.0:
        optimal_n_debiased = int(
            np.round((2.0 * iq_estimate / denominator) ** 0.5)
        )
    else:
        # Denominator non-positive: degenerate case, use all data
        optimal_n_debiased = m

    # Ref: realized_variance_optimal_sampling.m:167-175 — clamp to [2, m]
    if optimal_n_debiased > m:
        # Ref: realized_variance_optimal_sampling.m:168
        warnings.warn(
            'Computed optimal debiased number of samples exceeds the number '
            'of filtered prices. Using all available data.',
            stacklevel=2,
        )
        samples_debiased = m
    elif optimal_n_debiased < 2:
        # Ref: realized_variance_optimal_sampling.m:170
        warnings.warn(
            'Computed optimal debiased number of samples is less than 2. '
            'Using minimum of 2.',
            stacklevel=2,
        )
        samples_debiased = 2
    else:
        samples_debiased = optimal_n_debiased

    # ------------------------------------------------------------------
    # Step 10: Compute debiased RV at optimal debiased sampling
    # Ref: realized_variance_optimal_sampling.m:178-183
    # ------------------------------------------------------------------
    actual_subsamples_debiased = min(subsamples, samples_debiased)
    # Ref: realized_variance_optimal_sampling.m:179
    rv_debiased_result = realized_variance(
        filtered_price, filtered_time, time_type,
        sampling_type, samples_debiased, actual_subsamples_debiased,
    )
    rv_debiased = float(rv_debiased_result[0])
    rv_debiased_ss = float(rv_debiased_result[1])

    # Ref: realized_variance_optimal_sampling.m:180-181 — debias by removing noise
    rv_debiased = rv_debiased - optimal_n_debiased * noise_variance
    rv_debiased_ss = rv_debiased_ss - optimal_n_debiased * noise_variance

    # Ref: realized_variance_optimal_sampling.m:182-183 — scale by (1 - n*/m)
    scale_factor = 1.0 - optimal_n_debiased / m
    if abs(scale_factor) > 1e-12:
        rv_debiased = rv_debiased / scale_factor
        rv_debiased_ss = rv_debiased_ss / scale_factor

    # ==================================================================
    # Diagnostics
    # Ref: realized_variance_optimal_sampling.m:185-192
    # ==================================================================
    diagnostics: dict = {
        # Ref: realized_variance_optimal_sampling.m:185 — m = length(filteredPrice) - 1
        'm': len(filtered_price) - 1,
        # Ref: realized_variance_optimal_sampling.m:186
        'optimalSamples': optimal_n,
        # Ref: realized_variance_optimal_sampling.m:187
        'samples': samples_used,
        # Ref: realized_variance_optimal_sampling.m:188
        'optimalNumberOfSamplesDebiased': optimal_n_debiased,
        # Ref: realized_variance_optimal_sampling.m:189
        'samplesDebiased': samples_debiased,
        # Ref: realized_variance_optimal_sampling.m:190 — recalculated noiseVariance
        'noiseVariance': noise_variance,
        # Ref: realized_variance_optimal_sampling.m:191 — doubled debiased noise
        'debiasedNoiseVariance': debiased_noise_variance,
        # Ref: realized_variance_optimal_sampling.m:192
        'IQEstimate': iq_estimate,
    }

    return rv, rv_debiased, rv_ss, rv_debiased_ss, diagnostics


# ======================================================================
# Private helper: parameter validation
# ======================================================================


def _realized_variance_optimal_sampling_parameter_check(
    price,
    time,
    time_type: str,
    sampling_type: str,
    sampling_interval,
    subsamples,
    options: dict,
) -> str | None:
    """Validate inputs for :func:`realized_variance_optimal_sampling`.

    Parameters
    ----------
    price, time, time_type, sampling_type, sampling_interval, subsamples, options
        See :func:`realized_variance_optimal_sampling`.

    Returns
    -------
    str or None
        Error message string if validation fails; ``None`` if all inputs
        are valid.

    Notes
    -----
    Ref: realized_variance_optimal_sampling.m:194-356
    """
    # ------------------------------------------------------------------
    # PRICE validation
    # Ref: realized_variance_optimal_sampling.m:217-223
    # ------------------------------------------------------------------
    price_arr = np.asarray(price, dtype=np.float64)
    if price_arr.ndim == 2 and price_arr.shape[1] > price_arr.shape[0]:
        # Ref: realized_variance_optimal_sampling.m:218 — auto-transpose row vector
        price_arr = price_arr.T
    price_arr = price_arr.ravel()
    if price_arr.ndim != 1 or (price_arr.ndim == 1 and len(price_arr.shape) > 0
                               and price_arr.shape[0] == 0):
        pass  # Already 1-D at this point
    # Ref: realized_variance_optimal_sampling.m:220-222
    # Check that it's truly a vector (not a matrix)
    if np.asarray(price, dtype=np.float64).ndim == 2:
        orig = np.asarray(price, dtype=np.float64)
        if orig.shape[1] > orig.shape[0]:
            orig = orig.T
        if orig.shape[1] > 1:
            return 'PRICE must be a m by 1 vector.'

    # ------------------------------------------------------------------
    # TIME validation
    # Ref: realized_variance_optimal_sampling.m:224-234
    # ------------------------------------------------------------------
    time_arr = np.asarray(time, dtype=np.float64)
    if time_arr.ndim == 2 and time_arr.shape[1] > time_arr.shape[0]:
        # Ref: realized_variance_optimal_sampling.m:225 — auto-transpose
        time_arr = time_arr.T
    time_arr = time_arr.ravel()

    # Ref: realized_variance_optimal_sampling.m:227-229 — sorted and increasing
    if np.any(np.diff(time_arr) < 0):
        return 'TIME must be sorted and increasing'

    # Ref: realized_variance_optimal_sampling.m:231-233
    if np.asarray(time, dtype=np.float64).ndim == 2:
        orig_t = np.asarray(time, dtype=np.float64)
        if orig_t.shape[1] > orig_t.shape[0]:
            orig_t = orig_t.T
        if orig_t.shape[1] > 1:
            return 'TIME must be a m by 1 vector.'
    if len(time_arr) != len(price_arr):
        return 'TIME must be a m by 1 vector.'

    # ------------------------------------------------------------------
    # TIMETYPE validation
    # Ref: realized_variance_optimal_sampling.m:236-240
    # ------------------------------------------------------------------
    time_type_lower = time_type.lower()
    if time_type_lower not in ('wall', 'seconds', 'unit'):
        return "TIMETYPE must be one of 'wall', 'seconds' or 'unit'."

    # ------------------------------------------------------------------
    # SAMPLINGTYPE validation
    # Ref: realized_variance_optimal_sampling.m:242-246
    # ------------------------------------------------------------------
    sampling_type_lower = sampling_type.lower()
    valid_sampling = (
        'calendartime', 'calendaruniform',
        'businesstime', 'businessuniform', 'fixed',
    )
    if sampling_type_lower not in valid_sampling:
        return (
            "SAMPLINGTYPE must be one of 'CalendarTime', 'CalendarUniform', "
            "'BusinessTime', 'BusinessUniform' or 'Fixed'."
        )

    # ------------------------------------------------------------------
    # SUBSAMPLES validation
    # Ref: realized_variance_optimal_sampling.m:249-253
    # ------------------------------------------------------------------
    if subsamples is not None:
        if (not np.isscalar(subsamples)
                or subsamples < 0
                or np.floor(subsamples) != subsamples):
            return 'SUBSAMPLES must be a non-negative scalar.'

    # ------------------------------------------------------------------
    # SAMPLINGINTERVAL validation
    # Ref: realized_variance_optimal_sampling.m:257-282
    # ------------------------------------------------------------------
    m_price = len(price_arr)
    t0 = time_arr[0]
    tT = time_arr[-1] if m_price > 0 else 0.0

    if sampling_type_lower in ('calendartime', 'calendaruniform',
                               'businesstime', 'businessuniform'):
        if time_type_lower in ('wall', 'seconds'):
            # Ref: realized_variance_optimal_sampling.m:264-266
            if (not np.isscalar(sampling_interval)
                    or np.floor(float(sampling_interval)) != float(sampling_interval)
                    or float(sampling_interval) < 1):
                return (
                    'SAMPLINGINTERVAL must be a positive integer for the '
                    "SAMPLINGTYPE selected when using 'wall' or 'seconds' "
                    'as TIMETYPE.'
                )
        else:
            # Ref: realized_variance_optimal_sampling.m:268-270
            if not np.isscalar(sampling_interval) or float(sampling_interval) < 0:
                return (
                    'SAMPLINGINTERVAL must be a positive value for the '
                    "SAMPLINGTYPE selected when using 'unit' as TIMETYPE."
                )
    else:
        # Ref: realized_variance_optimal_sampling.m:273-281 — 'fixed' type
        si_arr = np.asarray(sampling_interval, dtype=np.float64).ravel()
        if si_arr.ndim == 2 and si_arr.shape[1] > si_arr.shape[0]:
            si_arr = si_arr.T.ravel()
        if not (np.any(si_arr >= t0) and np.any(si_arr <= tT)):
            return (
                'At least one sampling interval must be between min(TIME) '
                "and max(TIME) when using 'Fixed' as SAMPLINGTYPE."
            )
        if np.any(np.diff(si_arr) <= 0):
            return (
                "When using 'Fixed' as SAMPLINGTYPE the vector of sampling "
                'times in SAMPLINGINTERVAL must be sorted and strictly '
                'increasing.'
            )

    # ------------------------------------------------------------------
    # OPTIONS validation
    # Ref: realized_variance_optimal_sampling.m:288-355
    # ------------------------------------------------------------------

    # Ref: realized_variance_optimal_sampling.m:289-291 — flat-top kernel list
    flat_top_kernel_list = [
        'bartlett', 'twoscale', '2ndorder', 'epanechnikov',
        'cubic', 'multiscale', '5thorder', '6thorder', '7thorder',
        '8thorder', 'parzen', 'th1', 'th2', 'th5', 'th16',
    ]

    # Ref: realized_variance_optimal_sampling.m:294 — non-flat-top kernel list
    non_flat_top_kernel_list = [
        'nonflatparzen', 'qs', 'fejer', 'thinf', 'bnhls',
    ]

    # Ref: realized_variance_optimal_sampling.m:297 — combined list
    kernel_list = flat_top_kernel_list + non_flat_top_kernel_list

    # Ref: realized_variance_optimal_sampling.m:300-309 — merge missing fields
    options_field_names = list(options.keys())
    default_options = realized_options('Optimal Sampling')
    for key, value in default_options.items():
        if key not in options:
            options[key] = value

    # Ref: realized_variance_optimal_sampling.m:311-355 — validate each field
    for field_name in options_field_names:
        field_value = options[field_name]

        # Ref: realized_variance_optimal_sampling.m:314-317 — lowercase strings
        if isinstance(field_value, str):
            field_value = field_value.lower()
            # Note: We update the dict so downstream uses lowered values
            # (MATLAB does this on a local copy; we do it on our working copy)

        if field_name == 'medFrequencyKernel':
            # Ref: realized_variance_optimal_sampling.m:321-325
            if field_value not in kernel_list:
                return f'OPTIONS.{field_name} must be one of the listed types.'

        elif field_name == 'medFrequencyBandwidth':
            # Ref: realized_variance_optimal_sampling.m:327-331
            if field_value is not None and not _is_nonnegative_scalar(field_value):
                return f'OPTIONS.{field_name} must a non-negative scalar.'

        elif field_name in ('useDebiasedNoise', 'useAdjustedNoiseCount'):
            # Ref: realized_variance_optimal_sampling.m:333-337
            if not isinstance(field_value, bool) and field_value not in (0, 1):
                return 'OPTIONS.useDebiasedNoise must be a logical value.'

        elif field_name in ('IQEstimationSamplingType',
                            'medFrequencySamplingType',
                            'noiseVarianceSamplingType'):
            # Ref: realized_variance_optimal_sampling.m:339-343
            valid_st = (
                'calendartime', 'calendaruniform',
                'businesstime', 'businessuniform', 'fixed',
            )
            if field_value not in valid_st:
                return (
                    f"OPTIONS.{field_name} must be one of 'CalendarTime', "
                    f"'CalendarUniform', 'BusinessTime', 'BusinessUniform' "
                    f"or 'Fixed'."
                )

        elif field_name in ('medFrequencySamplingInterval',
                            'noiseVarianceSamplingInterval',
                            'IQEstimationSamplingInterval'):
            # Ref: realized_variance_optimal_sampling.m:345-353
            if field_value is None or not _is_nonnegative_scalar(field_value):
                return (
                    f'OPTIONS.{field_name} must be a non-negative scalar '
                    f'between 0 and 1.'
                )
            if time_type_lower == 'unit' and float(field_value) > 1:
                return (
                    f"OPTIONS.{field_name} must be less than 1 if TIMETYPE "
                    f"when 'unit'."
                )

    return None


# ======================================================================
# Private helper: non-negative scalar check
# ======================================================================


def _is_nonnegative_scalar(x) -> bool:
    """Return ``True`` if *x* is a non-empty scalar >= 0.

    Ref: realized_variance_optimal_sampling.m:357-359

    Parameters
    ----------
    x : any
        Value to test.

    Returns
    -------
    bool
        ``True`` if *x* is a scalar and ``x >= 0``.
    """
    # Ref: realized_variance_optimal_sampling.m:358-359
    # condition = ~isempty(x) && isscalar(x) && x>=0;
    return x is not None and np.isscalar(x) and float(x) >= 0
