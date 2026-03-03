"""
Realized kernel estimator for quadratic variation.

Implements the Barndorff-Nielsen, Hansen, Lunde and Shephard (BNHLS)
realized kernel estimator, which provides a consistent, rate-efficient
estimate of integrated variance in the presence of market microstructure
noise.  This module is an orchestrator that delegates to six helper modules
for noise estimation, bandwidth selection, jitter lag selection, price
filtering, weight computation, and the kernel core calculation.

Migrated from: realized/realized_kernel.m (414 lines)
Original author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
Revision: 1    Date: 5/1/2008

See Also
--------
realized_options : Default option factory for kernel configuration.
realized_noise_estimate : Microstructure noise variance estimation.
realized_kernel_bandwidth : Optimal bandwidth selection.
realized_kernel_jitter_lag_length : Endpoint jitter lag computation.
realized_price_filter : Price sampling/filtering.
realized_kernel_weights : Kernel weight computation.
realized_kernel_core : Core weighted autocovariance summation.
"""

from __future__ import annotations

import warnings

import numpy as np

from mfe_toolbox.realized.realized_options import realized_options
from mfe_toolbox.realized.realized_noise_estimate import realized_noise_estimate
from mfe_toolbox.realized.realized_kernel_bandwidth import realized_kernel_bandwidth
from mfe_toolbox.realized.realized_kernel_jitter_lag_length import (
    realized_kernel_jitter_lag_length,
)
from mfe_toolbox.realized.realized_price_filter import realized_price_filter
from mfe_toolbox.realized.realized_kernel_weights import realized_kernel_weights
from mfe_toolbox.realized.realized_kernel_core import realized_kernel_core


# ---------------------------------------------------------------------------
# Kernel classification lists (shared by validation)
# Ref: realized_kernel.m:312-321
# ---------------------------------------------------------------------------

# Ref: realized_kernel.m:313-315 — flat-top kernels
_FLAT_TOP_KERNELS: list[str] = [
    'bartlett', 'twoscale', '2ndorder', 'epanechnikov',
    'cubic', 'multiscale', '5thorder', '6thorder', '7thorder', '8thorder',
    'parzen', 'th1', 'th2', 'th5', 'th16',
]

# Ref: realized_kernel.m:318 — non-flat-top kernels
_NON_FLAT_TOP_KERNELS: list[str] = [
    'nonflatparzen', 'qs', 'fejer', 'thinf', 'bnhls',
]

# Ref: realized_kernel.m:321 — combined list
_KERNEL_LIST: list[str] = _FLAT_TOP_KERNELS + _NON_FLAT_TOP_KERNELS

# Valid sampling types for options validation
# Ref: realized_kernel.m:280-283
_VALID_SAMPLING_TYPES: frozenset[str] = frozenset({
    'calendartime', 'calendaruniform', 'businesstime',
    'businessuniform', 'fixed',
})


# =========================================================================
# Private helper functions — translated from MATLAB subfunctions
# =========================================================================


def _is_nonnegative_scalar_integer(x: object) -> bool:
    """Return True if *x* is a non-empty, scalar, non-negative integer.

    Translates the MATLAB ``isnonnegativescalarinteger`` subfunction.

    Parameters
    ----------
    x : object
        Value to test.

    Returns
    -------
    bool

    Notes
    -----
    Ref: realized_kernel.m:408-410
    """
    if x is None:
        return False
    try:
        val = float(x)
    except (TypeError, ValueError):
        return False
    if not np.isscalar(x):
        return False
    return bool(val >= 0.0 and val == float(np.floor(val)))


def _is_nonnegative_scalar(x: object) -> bool:
    """Return True if *x* is a non-empty, scalar, non-negative value.

    Translates the MATLAB ``isnonnegativescalar`` subfunction.

    Parameters
    ----------
    x : object
        Value to test.

    Returns
    -------
    bool

    Notes
    -----
    Ref: realized_kernel.m:413-415
    """
    if x is None:
        return False
    try:
        val = float(x)
    except (TypeError, ValueError):
        return False
    if not np.isscalar(x):
        return False
    return bool(val >= 0.0)


# =========================================================================
# Parameter validation
# =========================================================================


def _realized_kernel_parameter_check(
    price: np.ndarray,
    time: np.ndarray,
    time_type: str,
    sampling_type: str,
    sampling_interval,
    options: dict,
) -> dict:
    """Validate inputs and fill missing options for :func:`realized_kernel`.

    Translates the MATLAB ``realized_kernel_parameter_check`` subfunction
    (lines 220-405).

    Parameters
    ----------
    price : np.ndarray
        1-D array of prices.
    time : np.ndarray
        1-D array of observation times.
    time_type : str
        One of ``'wall'``, ``'seconds'``, ``'unit'``.
    sampling_type : str
        One of ``'CalendarTime'``, ``'CalendarUniform'``, ``'BusinessTime'``,
        ``'BusinessUniform'``, ``'Fixed'``.
    sampling_interval : int, float, or np.ndarray
        Sampling interval specification.
    options : dict
        Realized kernel options dictionary.

    Returns
    -------
    dict
        Validated and completed options dictionary (with ``'filteredN'``
        added).

    Raises
    ------
    ValueError
        If any input fails validation.
    """
    # ------------------------------------------------------------------
    # Price validation
    # Ref: realized_kernel.m:251-258
    # ------------------------------------------------------------------
    # Ref: realized_kernel.m:252-254 — auto-transpose row vector
    if price.ndim == 2 and price.shape[1] > price.shape[0]:
        price = price.ravel()
    # Ref: realized_kernel.m:255-258 — reject multi-column
    if price.ndim == 2 and price.shape[1] > 1:
        raise ValueError('PRICE must be a m by 1 vector.')
    price = np.asarray(price, dtype=np.float64).ravel()

    # ------------------------------------------------------------------
    # Time validation
    # Ref: realized_kernel.m:260-271
    # ------------------------------------------------------------------
    # Ref: realized_kernel.m:261-263 — auto-transpose row vector
    if time.ndim == 2 and time.shape[1] > time.shape[0]:
        time = time.ravel()
    time = np.asarray(time, dtype=np.float64).ravel()

    # Ref: realized_kernel.m:264-266 — sorted check
    if np.any(np.diff(time) < 0):
        raise ValueError('TIME must be sorted and increasing')
    # Ref: realized_kernel.m:268-271 — shape and length match
    if time.ndim != 1 or len(time) != len(price):
        raise ValueError('TIME must be a m by 1 vector the same size as PRICE.')

    # ------------------------------------------------------------------
    # Time type validation
    # Ref: realized_kernel.m:272-277
    # ------------------------------------------------------------------
    time_type = time_type.lower()
    if time_type not in ('wall', 'seconds', 'unit'):
        raise ValueError(
            "TIMETYPE must be one of 'wall', 'seconds' or 'unit'."
        )

    # ------------------------------------------------------------------
    # Sampling type validation
    # Ref: realized_kernel.m:278-283
    # ------------------------------------------------------------------
    sampling_type = sampling_type.lower()
    if sampling_type not in _VALID_SAMPLING_TYPES:
        raise ValueError(
            "SAMPLINGTYPE must be one of 'CalendarTime', 'CalendarUniform', "
            "'BusinessTime', 'BusinessUniform' or 'Fixed'."
        )

    # ------------------------------------------------------------------
    # Sampling interval validation
    # Ref: realized_kernel.m:285-306
    # ------------------------------------------------------------------
    m = len(price)
    t0 = time[0]  # Ref: realized_kernel.m:286 — time(1) → Python time[0]
    tT = time[m - 1]  # Ref: realized_kernel.m:287 — time(m) → Python time[m-1]

    if sampling_type in ('calendartime', 'calendaruniform',
                         'businesstime', 'businessuniform'):
        # Ref: realized_kernel.m:289-293 — must be positive scalar integer
        if (not np.isscalar(sampling_interval)
                or np.floor(float(sampling_interval)) != float(sampling_interval)
                or float(sampling_interval) < 1):
            raise ValueError(
                'SAMPLINGINTERVAL must be a positive integer for the '
                'SAMPLINGTYPE selected.'
            )
    else:
        # 'fixed' case
        # Ref: realized_kernel.m:294-306
        sampling_interval_arr = np.asarray(sampling_interval, dtype=np.float64).ravel()
        # Ref: realized_kernel.m:298-301
        if not (np.any(sampling_interval_arr >= t0)
                and np.any(sampling_interval_arr <= tT)):
            raise ValueError(
                "At least one sampling interval must be between min(TIME) "
                "and max(TIME) when using 'Fixed' as SAMPLINGTYPE."
            )
        # Ref: realized_kernel.m:302-305
        if np.any(np.diff(sampling_interval_arr) <= 0):
            raise ValueError(
                "When using 'Fixed' as SAMPLINGTYPE the vector of sampling "
                "times in SAMPLINGINTERVAL must be sorted and strictly "
                "increasing."
            )

    # ------------------------------------------------------------------
    # Options validation
    # Ref: realized_kernel.m:309-396
    # ------------------------------------------------------------------

    # Ref: realized_kernel.m:324-333 — fill missing fields from defaults
    default_options = realized_options('Kernel')
    for key, default_val in default_options.items():
        if key not in options:
            options[key] = default_val

    # Ref: realized_kernel.m:335-396 — validate each field
    for field_name in list(options.keys()):
        field_value = options[field_name]

        # Ref: realized_kernel.m:338-341 — lowercase string fields
        if isinstance(field_value, str):
            field_value = field_value.lower()
            options[field_name] = field_value

        if field_name in ('kernel', 'medFrequencyKernel'):
            # Ref: realized_kernel.m:344-349 — must be in kernel list
            if field_value not in _KERNEL_LIST:
                raise ValueError(
                    f'OPTIONS.{field_name} must be one of the listed types.'
                )

        elif field_name in ('bandwidth', 'medFrequencyBandwidth'):
            # Ref: realized_kernel.m:350-355 — None or non-negative scalar
            if field_value is not None and not _is_nonnegative_scalar(field_value):
                raise ValueError(
                    f'OPTIONS.{field_name} must a non-negative scalar.'
                )

        elif field_name == 'endTreatment':
            # Ref: realized_kernel.m:356-361
            if field_value not in ('jitter', 'stagger'):
                raise ValueError(
                    "OPTIONS.endTreatment must be either 'Jitter' or 'Stagger'."
                )

        elif field_name in ('jitterLags', 'maxBandwidth'):
            # Ref: realized_kernel.m:362-367 — None or non-negative scalar integer
            if field_value is not None and not _is_nonnegative_scalar_integer(field_value):
                raise ValueError(
                    f'OPTIONS.{field_name} must be a non-negative scalar integer.'
                )

        elif field_name in ('useDebiasedNoise', 'useAdjustedNoiseCount'):
            # Ref: realized_kernel.m:368-373 — boolean or 0/1
            if not isinstance(field_value, bool) and field_value not in (0, 1):
                raise ValueError(
                    'OPTIONS.useDebiasedNoise must be a logical value.'
                )

        elif field_name == 'maxBandwidthPerc':
            # Ref: realized_kernel.m:374-379
            # NOTE: The MATLAB code uses && for the last condition, which is a
            # known permissive bug — it only errors if fieldValue is
            # simultaneously not-a-non-negative-scalar AND > 1.  We replicate
            # the MATLAB behavior exactly.
            if (field_value is not None
                    and not _is_nonnegative_scalar(field_value)
                    and field_value > 1):
                raise ValueError(
                    'OPTIONS.maxBandwidthPerc must be a non-negative scalar '
                    'between 0 and 1.'
                )

        elif field_name in ('IQEstimationSamplingType',
                            'medFrequencySamplingType',
                            'noiseVarianceSamplingType'):
            # Ref: realized_kernel.m:380-385 — must be a valid sampling type
            if field_value not in _VALID_SAMPLING_TYPES:
                raise ValueError(
                    f"OPTIONS.{field_name} must be one of 'CalendarTime', "
                    f"'CalendarUniform', 'BusinessTime', 'BusinessUniform' "
                    f"or 'Fixed'."
                )

        elif field_name in ('medFrequencySamplingInterval',
                            'noiseVarianceSamplingInterval',
                            'IQEstimationSamplingInterval'):
            # Ref: realized_kernel.m:386-395 — non-negative scalar
            if field_value is None or not _is_nonnegative_scalar(field_value):
                raise ValueError(
                    f'OPTIONS.{field_name} must be a non-negative scalar '
                    f'between 0 and 1.'
                )
            # Ref: realized_kernel.m:392-394 — if time_type is 'unit', must be <= 1
            if time_type == 'unit' and float(field_value) > 1:
                raise ValueError(
                    f"OPTIONS.{field_name} must be less than 1 if TIMETYPE "
                    f"when 'unit'."
                )

    # ------------------------------------------------------------------
    # Compute filteredN — the effective sample size after filtering
    # Ref: realized_kernel.m:399-405
    # ------------------------------------------------------------------
    # Ref: realized_kernel.m:400 — ~strcmpi(samplingType,'businesstime') || samplingInterval~=1
    if sampling_type != 'businesstime' or float(sampling_interval if np.isscalar(sampling_interval) else 1) != 1:
        # Ref: realized_kernel.m:401
        temp_result = realized_price_filter(
            price, time, time_type, sampling_type, sampling_interval,
        )
        # realized_price_filter returns (filtered_price, filtered_time, actual_time)
        temp_filtered_price = temp_result[0]
        options['filteredN'] = len(temp_filtered_price)
    else:
        # Ref: realized_kernel.m:404 — filteredN = length(price)
        options['filteredN'] = len(price)

    return options


# =========================================================================
# Main public function
# =========================================================================


def realized_kernel(
    price: np.ndarray,
    time: np.ndarray | None = None,
    time_type: str = 'unit',
    sampling_type: str = 'BusinessTime',
    sampling_interval: int | float | np.ndarray = 1,
    options: dict | None = None,
) -> tuple[float, float, dict]:
    """Estimate quadratic variation using realized kernels.

    Implements the Barndorff-Nielsen, Hansen, Lunde and Shephard (BNHLS)
    realized kernel estimator.  This function is an orchestrator that
    calls six helper modules to validate inputs, estimate noise, select
    bandwidth, perform endpoint jittering, filter prices, compute kernel
    weights, and compute the final realized kernel estimate.

    Parameters
    ----------
    price : np.ndarray
        1-D array of *m* high-frequency prices.
    time : np.ndarray or None, optional
        1-D array of *m* observation times corresponding to ``price``.
        If ``None``, a default time grid from 9:30 to 16:00 in seconds
        is generated automatically and ``time_type`` is set to
        ``'seconds'``.
    time_type : str, optional
        Time format descriptor:

        * ``'wall'``    — 24-hour clock HHMMSS (e.g. 101543)
        * ``'seconds'`` — seconds past midnight
        * ``'unit'``    — unit-normalized date format
    sampling_type : str, optional
        Sampling scheme for price filtering:

        * ``'CalendarTime'``     — calendar-time sampling
        * ``'CalendarUniform'``  — uniform calendar-time sampling
        * ``'BusinessTime'``     — tick-time sampling (default)
        * ``'BusinessUniform'``  — uniform business-time sampling
        * ``'Fixed'``            — fixed-time sampling
    sampling_interval : int, float, or np.ndarray, optional
        Sampling interval.  Interpretation depends on ``sampling_type``.
    options : dict or None, optional
        Realized kernel options dictionary, typically obtained from
        ``realized_options('Kernel')``.  If ``None``, defaults are used.

    Returns
    -------
    rk : float
        Realized kernel estimate of integrated variance.
    rk_adjusted : float
        Bias-adjusted realized kernel, correcting for the fraction of
        the sample used in estimation.
    diagnostics : dict
        Diagnostic information with keys:

        * ``'kernel'`` — kernel type used
        * ``'bandwidth'`` — bandwidth value
        * ``'filteredPrice'`` — filtered prices after sampling
        * ``'filteredTime'`` — time stamps of filtered prices
        * ``'weights'`` — kernel weight vector
        * ``'adjustment'`` — de-biasing adjustment factor
        * ``'noiseVariance'`` — Bandi-Russell noise variance
          (only if computed)
        * ``'debiasedNoiseVariance'`` — debiased noise variance
          (only if computed)
        * ``'jitterLags'`` — number of jitter lags (only if jittered)

    Raises
    ------
    ValueError
        If inputs fail validation (invalid shapes, unsorted times,
        unrecognized options, etc.).

    Notes
    -----
    For best results, use prices sampled at close to their highest
    frequency.  Use business-time sampling with a small number of ticks
    between samples (preferably 1, but usually < 15).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> prices = 100.0 * np.exp(np.cumsum(rng.standard_normal(1000) * 0.001))
    >>> rk, rk_adj, diag = realized_kernel(prices)

    See Also
    --------
    realized_options, realized_noise_estimate, realized_variance,
    realized_variance_optimal_sampling, realized_range,
    realized_quantile_variance
    """
    # ==================================================================
    # 1. Input defaulting
    # Ref: realized_kernel.m:100-114 — nargin dispatch
    # ==================================================================
    price = np.asarray(price, dtype=np.float64).ravel()

    if time is None:
        # Ref: realized_kernel.m:101-107 — nargin==1 default path
        m = len(price)
        # Ref: realized_kernel.m:103 — linspace(9.5*3600, 16*3600, m)'
        time = np.linspace(9.5 * 3600, 16.0 * 3600, m)
        time_type = 'seconds'
        sampling_type = 'businesstime'
        sampling_interval = 1
        options = realized_options('kernel')
    else:
        time = np.asarray(time, dtype=np.float64).ravel()
        if options is None:
            # Ref: realized_kernel.m:108-109 — nargin==5 default path
            options = realized_options('kernel')

    # Make a mutable copy of options to avoid side-effects on caller's dict
    options = dict(options)

    # ==================================================================
    # 2. Parameter validation
    # Ref: realized_kernel.m:116-120
    # ==================================================================
    options = _realized_kernel_parameter_check(
        price, time, time_type, sampling_type, sampling_interval, options,
    )

    # ==================================================================
    # 3. Jitter flag
    # Ref: realized_kernel.m:125
    # ==================================================================
    is_jittered: bool = options.get('endTreatment', '').lower() == 'jitter'

    # ==================================================================
    # 4. Ensure float64 time (protect against integer times)
    # Ref: realized_kernel.m:128
    # ==================================================================
    time = np.asarray(time, dtype=np.float64)

    # ==================================================================
    # 5. Noise estimation (optional)
    # Ref: realized_kernel.m:130-142
    # ==================================================================
    diagnostics: dict = {}
    noise_estimation_needed = (
        options.get('bandwidth') is None
        or (is_jittered and options.get('jitterLags') is None)
    )

    if noise_estimation_needed:
        # Ref: realized_kernel.m:133
        # realized_noise_estimate returns 4 values; MATLAB captures first 3.
        noise_result = realized_noise_estimate(price, time, time_type, options)
        noise_variance = noise_result[0]
        debiased_noise_variance = noise_result[1]
        IQ_estimate = noise_result[2]

        # Ref: realized_kernel.m:134-135
        diagnostics['noiseVariance'] = noise_variance
        diagnostics['debiasedNoiseVariance'] = debiased_noise_variance

        # Ref: realized_kernel.m:137-141 — select noise variance
        if options.get('useDebiasedNoise', False):
            selected_noise_variance = debiased_noise_variance
        else:
            selected_noise_variance = noise_variance

    # ==================================================================
    # 6. Bandwidth computation (if not pre-specified)
    # Ref: realized_kernel.m:146-150
    # ==================================================================
    if options.get('bandwidth') is None:
        # Ref: realized_kernel.m:148
        options['bandwidth'] = realized_kernel_bandwidth(
            selected_noise_variance, IQ_estimate, options,
        )

    # ==================================================================
    # 7. Endpoint jittering
    # Ref: realized_kernel.m:152-167
    # ==================================================================
    if is_jittered:
        # Ref: realized_kernel.m:154-157 — auto-compute jitterLags if needed
        if options.get('jitterLags') is None:
            jitter_lags = realized_kernel_jitter_lag_length(
                selected_noise_variance, IQ_estimate,
                options['kernel'], options['filteredN'],
            )
            options['jitterLags'] = jitter_lags

        # Ref: realized_kernel.m:159 — m = length(price)
        m = len(price)
        jl = int(options['jitterLags'])

        # Ref: realized_kernel.m:161 — p0 = mean(price(1:jl))
        # MATLAB 1:jl → Python 0:jl (0-based slicing: price[:jl])
        p0 = np.mean(price[:jl])

        # Ref: realized_kernel.m:162 — p1 = mean(price(m-jl+1:m))
        # MATLAB m-jl+1:m → Python m-jl:m (0-based slicing: price[m-jl:])
        p1 = np.mean(price[m - jl:])

        # Ref: realized_kernel.m:163 — t0 = time(ceil(mean(1:jl)))
        # MATLAB 1:jl = [1,2,...,jl], mean = (jl+1)/2
        # ceil gives 1-based index; subtract 1 for 0-based
        t0_idx = int(np.ceil(np.mean(np.arange(1, jl + 1)))) - 1
        t0_val = time[t0_idx]

        # Ref: realized_kernel.m:164 — t1 = time(floor(mean(m-jl+1:m)))
        # MATLAB m-jl+1:m = [m-jl+1,...,m], mean = m - (jl-1)/2
        # floor gives 1-based index; subtract 1 for 0-based
        t1_idx = int(np.floor(np.mean(np.arange(m - jl + 1, m + 1)))) - 1
        t1_val = time[t1_idx]

        # Ref: realized_kernel.m:165-166 — reconstruct jittered arrays
        # MATLAB: price = [p0; price(jl+1:m-jl); p1]
        # Python: price[jl:m-jl] corresponds to MATLAB price(jl+1:m-jl)
        price = np.concatenate(([p0], price[jl:m - jl], [p1]))
        time = np.concatenate(([t0_val], time[jl:m - jl], [t1_val]))

    # ==================================================================
    # 8. Price filtering
    # Ref: realized_kernel.m:170-171
    # ==================================================================
    filter_result = realized_price_filter(
        price, time, time_type, sampling_type, sampling_interval,
    )
    # realized_price_filter returns (filtered_price, filtered_time, actual_time)
    filtered_price = filter_result[0]
    filtered_time = filter_result[1]

    # Ref: realized_kernel.m:171 — returns = diff(log(filteredPrice))
    returns = np.diff(np.log(filtered_price))

    # ==================================================================
    # 9. Weight computation
    # Ref: realized_kernel.m:174
    # ==================================================================
    weights = realized_kernel_weights(options)

    # ==================================================================
    # 10. Kernel computation
    # Ref: realized_kernel.m:177
    # ==================================================================
    rk = realized_kernel_core(returns, weights, options)

    # ==================================================================
    # 11. Diagnostics construction
    # Ref: realized_kernel.m:181-200
    # ==================================================================
    # Ref: realized_kernel.m:181-182
    n = len(filtered_price)

    # Ref: realized_kernel.m:185-189
    diagnostics['kernel'] = options['kernel']
    diagnostics['bandwidth'] = options['bandwidth']
    diagnostics['filteredPrice'] = filtered_price
    diagnostics['filteredTime'] = filtered_time
    diagnostics['weights'] = weights

    # Ref: realized_kernel.m:190-192
    if is_jittered:
        diagnostics['jitterLags'] = options['jitterLags']

    # Ref: realized_kernel.m:194-200 — adjustment computation
    if is_jittered:
        # Ref: realized_kernel.m:196 — adjustment = 1 for jittered case
        diagnostics['adjustment'] = 1.0
    else:
        # Ref: realized_kernel.m:199
        # MATLAB: filteredTime(n-round(options.bandwidth)) is 1-based
        #   → Python 0-based: filtered_time[n - 1 - bw_round]
        # MATLAB: filteredTime(1+round(options.bandwidth)) is 1-based
        #   → Python 0-based: filtered_time[bw_round]
        bw_round = int(round(options['bandwidth']))
        # Guard against edge cases where bw_round >= n
        if bw_round < n and bw_round >= 0:
            numerator = (
                filtered_time[n - 1 - bw_round] - filtered_time[bw_round]
            )
        else:
            # Fallback: use full range if bw_round is out of range
            # Ref: realized_kernel.m — edge case not in MATLAB; defensive Python guard
            warnings.warn(
                f"Bandwidth ({bw_round}) is too large relative to the number of "
                f"filtered observations ({n}). Using full time range for adjustment.",
                stacklevel=2,
            )
            numerator = filtered_time[-1] - filtered_time[0]

        denominator = time[-1] - time[0]
        if denominator != 0.0:
            diagnostics['adjustment'] = float(numerator / denominator)
        else:
            diagnostics['adjustment'] = 1.0

    # ==================================================================
    # 12. Adjusted RK
    # Ref: realized_kernel.m:203
    # ==================================================================
    rk_adjusted = rk / diagnostics['adjustment'] if diagnostics['adjustment'] != 0.0 else rk

    return float(rk), float(rk_adjusted), diagnostics
