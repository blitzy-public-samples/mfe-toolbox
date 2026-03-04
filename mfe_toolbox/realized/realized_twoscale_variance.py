"""
Two-scale realized variance (TSRV) estimator.

Implements the Ait-Sahalia, Mykland and Zhang (2005) two-scale realized
variance estimator for quadratic variation estimation from noisy
high-frequency price data.  The TSRV combines a slow-scale overlapping
realized variance with a bias-correcting fast-scale realized variance
to produce a consistent estimator under market microstructure noise.

The estimator uses the relationship:

    TSRV = RV_slow - (n_bar / n) * RV_fast

where ``RV_slow`` is computed on overlapping subgrids of bandwidth *K*
and ``RV_fast`` is the all-returns realized variance.  The debiased variant
applies the small-sample correction factor ``(1 - n_bar/n)^{-1}``.

Migrated from: ``realized/realized_twoscale_variance.m`` (MFE Toolbox v4.0)
Original author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
Revision: 1    Date: 5/1/2008

See Also
--------
mfe_toolbox.realized.realized_options : Default option factory.
mfe_toolbox.realized.realized_noise_estimate : Noise variance estimation.
mfe_toolbox.realized.realized_variance : Standard realized variance.
mfe_toolbox.realized.realized_price_filter : Price filtering.
mfe_toolbox.realized.realized_subsample : Subsampling infrastructure.

References
----------
.. [1] Ait-Sahalia, Y., Mykland, P. A. and Zhang, L. (2005). "How often
   to sample a continuous-time process in the presence of market
   microstructure noise." *Review of Financial Studies*, 18(2), 351-416.
.. [2] Zhang, L., Mykland, P. A. and Ait-Sahalia, Y. (2005). "A tale of
   two time scales: Determining integrated volatility with noisy
   high-frequency data." *Journal of the American Statistical Association*,
   100(472), 1394-1411.
"""

from __future__ import annotations

import warnings

import numpy as np

from mfe_toolbox.realized.realized_convert2unit import realized_convert2unit
from mfe_toolbox.realized.realized_noise_estimate import realized_noise_estimate
from mfe_toolbox.realized.realized_options import realized_options
from mfe_toolbox.realized.realized_price_filter import realized_price_filter
from mfe_toolbox.realized.realized_subsample import realized_subsample


# ---------------------------------------------------------------------------
# Internal helper: overlapping realized variance
# Ref: realized_twoscale_variance.m:189-197
# ---------------------------------------------------------------------------

def _overlap_realized_variance(
    price: np.ndarray,
    skip: int,
) -> tuple[float, int, float]:
    """Compute overlapping realized variance using skip-length returns.

    Constructs non-overlapping returns of length *skip* from the price
    array, computes the sum of squared returns, and reports the overlap
    scaling factors needed for bias correction.

    Parameters
    ----------
    price : np.ndarray
        1-D array of (log) prices with *m* observations.
    skip : int
        Number of price indices to skip when computing returns.
        ``skip=1`` gives the all-returns RV; ``skip=K`` gives the
        slow-scale RV.

    Returns
    -------
    rv : float
        Sum of squared returns (inner product of the return vector
        with itself): ``returns' * returns``.
    count : int
        Number of returns computed: ``m - skip``.
    natural_count : float
        Number of non-overlapping returns that would exist:
        ``(m - 1) / skip``.  This is a float because the division
        is generally not exact.

    Notes
    -----
    Ref: realized_twoscale_variance.m:189-197 — MATLAB ``overlap_realized_variance``
    inner function.

    MATLAB uses ``price(1+skip:m) - price(1:m-skip)`` with 1-based indexing.
    Python uses ``price[skip:] - price[:m-skip]`` with 0-based indexing.
    The inner product ``returns' * returns`` is a scalar dot product.
    """
    m = len(price)
    # Ref: realized_twoscale_variance.m:191 — naturalCount = (length(price)-1)/skip
    natural_count = (m - 1) / skip
    # Ref: realized_twoscale_variance.m:193 — returns = price(1+skip:m) - price(1:m-skip)
    # MATLAB 1-based: price(1+skip) .. price(m)  minus  price(1) .. price(m-skip)
    # Python 0-based: price[skip:]  minus  price[:m-skip]
    returns = price[skip:] - price[:m - skip]
    # Ref: realized_twoscale_variance.m:194
    count = len(returns)
    # Ref: realized_twoscale_variance.m:195 — rv = returns' * returns (dot product)
    rv = float(np.dot(returns, returns))
    return rv, count, natural_count


# ---------------------------------------------------------------------------
# Internal helper: non-negative scalar check
# Ref: realized_twoscale_variance.m:349-351
# ---------------------------------------------------------------------------

def _isnonnegativescalar(x) -> bool:
    """Return True if *x* is a non-empty scalar >= 0.

    Replicates MATLAB's local ``isnonnegativescalar`` helper
    (realized_twoscale_variance.m:349-351).
    """
    if x is None:
        return False
    try:
        return bool(np.isscalar(x) and float(x) >= 0)
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Internal helper: parameter validation
# Ref: realized_twoscale_variance.m:200-351
# ---------------------------------------------------------------------------

def _realized_twoscale_variance_parameter_check(
    price: np.ndarray,
    time: np.ndarray,
    time_type: str,
    sampling_type: str,
    sampling_interval,
    subsamples: int,
    options: dict,
) -> dict:
    """Validate inputs and merge default options for TSRV estimation.

    Parameters
    ----------
    price : np.ndarray
        1-D array of prices (already ravelled by caller).
    time : np.ndarray
        1-D array of times (already ravelled by caller).
    time_type : str
        Lowercased time type string.
    sampling_type : str
        Lowercased sampling type string.
    sampling_interval : int, float, or np.ndarray
        Sampling interval or fixed time vector.
    subsamples : int
        Number of subsamples.
    options : dict
        User-provided options dict.

    Returns
    -------
    dict
        Options dict with missing fields filled from
        ``realized_options('optimal sampling')`` defaults.

    Raises
    ------
    ValueError
        If any input validation check fails.

    Notes
    -----
    Ref: realized_twoscale_variance.m:200-351 — MATLAB
    ``realized_twoscale_variance_parameter_check`` inner function.
    """
    # ---------------------------------------------------------------
    # Price validation
    # Ref: realized_twoscale_variance.m:223-228
    # ---------------------------------------------------------------
    if price.ndim != 1:
        raise ValueError('PRICE must be a m by 1 vector.')

    # ---------------------------------------------------------------
    # Time validation
    # Ref: realized_twoscale_variance.m:230-240
    # ---------------------------------------------------------------
    if time.ndim != 1:
        raise ValueError('TIME must be a m by 1 vector.')
    # Ref: realized_twoscale_variance.m:233
    if len(time) > 1 and np.any(np.diff(time) < 0):
        raise ValueError('TIME must be sorted and increasing')
    # Ref: realized_twoscale_variance.m:237
    if len(time) != len(price):
        raise ValueError('TIME must be a m by 1 vector.')

    # ---------------------------------------------------------------
    # Time type validation
    # Ref: realized_twoscale_variance.m:242-246
    # ---------------------------------------------------------------
    if time_type not in ('wall', 'seconds', 'unit'):
        raise ValueError(
            "TIMETYPE must be one of 'wall', 'seconds' or 'unit'."
        )

    # ---------------------------------------------------------------
    # Sampling type and interval validation
    # Ref: realized_twoscale_variance.m:249-268
    # ---------------------------------------------------------------
    valid_sampling_types = (
        'calendartime', 'calendaruniform',
        'businesstime', 'businessuniform', 'fixed',
    )
    if sampling_type not in valid_sampling_types:
        raise ValueError(
            "SAMPLINGTYPE must be one of 'CalendarTime', 'CalendarUniform', "
            "'BusinessTime', 'BusinessUniform' or 'Fixed'."
        )

    if sampling_type in ('calendartime', 'calendaruniform',
                         'businesstime', 'businessuniform'):
        # Ref: realized_twoscale_variance.m:252-255
        # Must be a scalar positive integer
        si_val = float(sampling_interval)
        if (not np.isscalar(sampling_interval)
                or np.floor(si_val) != si_val
                or si_val < 1):
            raise ValueError(
                'SAMPLINGINTERVAL must be a positive integer for the '
                'SAMPLINGTYPE selected.'
            )
    else:
        # Ref: realized_twoscale_variance.m:257-267 — 'fixed' sampling type
        si_arr = np.asarray(sampling_interval, dtype=np.float64).ravel()
        t0 = time[0]
        tT = time[-1]
        # Ref: realized_twoscale_variance.m:260
        if not (np.any(si_arr >= t0) and np.any(si_arr <= tT)):
            raise ValueError(
                "At least one sampling interval must be between min(TIME) "
                "and max(TIME) when using 'Fixed' as SAMPLINGTYPE."
            )
        # Ref: realized_twoscale_variance.m:264
        if len(si_arr) > 1 and np.any(np.diff(si_arr) <= 0):
            raise ValueError(
                "When using 'Fixed' as SAMPLINGTYPE the vector of sampling "
                "times in SAMPLINGINTERVAL must be sorted and strictly "
                "increasing."
            )

    # ---------------------------------------------------------------
    # Options merging and validation
    # Ref: realized_twoscale_variance.m:270-340
    # ---------------------------------------------------------------

    # Work on a copy to avoid mutating the caller's dict
    options = dict(options)

    # Ref: realized_twoscale_variance.m:274-279 — kernel lists
    flat_top_kernel_list = [
        'bartlett', 'twoscale', '2ndorder', 'epanechnikov',
        'cubic', 'multiscale', '5thorder', '6thorder', '7thorder',
        '8thorder', 'parzen', 'th1', 'th2', 'th5', 'th16',
    ]
    non_flat_top_kernel_list = [
        'nonflatparzen', 'qs', 'fejer', 'thinf', 'bnhls',
    ]
    kernel_list = flat_top_kernel_list + non_flat_top_kernel_list

    # Ref: realized_twoscale_variance.m:285-294 — merge missing fields
    # from realized_options('Optimal Sampling') defaults
    default_options = realized_options('optimal sampling')
    for key, value in default_options.items():
        if key not in options:
            options[key] = value

    # Ref: realized_twoscale_variance.m:296-340 — validate option fields
    for field_name, field_value in list(options.items()):
        # Ref: realized_twoscale_variance.m:299-301 — lowercase strings
        if isinstance(field_value, str):
            field_value = field_value.lower()
            options[field_name] = field_value

        if field_name == 'medFrequencyKernel':
            # Ref: realized_twoscale_variance.m:305-310
            if field_value not in kernel_list:
                raise ValueError(
                    f'OPTIONS.{field_name} must be one of the listed types.'
                )

        elif field_name == 'medFrequencyBandwidth':
            # Ref: realized_twoscale_variance.m:311-316
            if field_value is not None and not _isnonnegativescalar(field_value):
                raise ValueError(
                    f'OPTIONS.{field_name} must a non-negative scalar.'
                )

        elif field_name in ('useDebiasedNoise', 'useAdjustedNoiseCount'):
            # Ref: realized_twoscale_variance.m:317-322
            if not isinstance(field_value, (bool, int, np.bool_)):
                raise ValueError(
                    'OPTIONS.useDebiasedNoise must be a logical value.'
                )
            if isinstance(field_value, int) and field_value not in (0, 1):
                raise ValueError(
                    'OPTIONS.useDebiasedNoise must be a logical value.'
                )

        elif field_name in ('IQEstimationSamplingType',
                            'medFrequencySamplingType',
                            'noiseVarianceSamplingType'):
            # Ref: realized_twoscale_variance.m:323-328
            valid_st = (
                'calendartime', 'calendaruniform',
                'businesstime', 'businessuniform', 'fixed',
            )
            if field_value not in valid_st:
                raise ValueError(
                    f"OPTIONS.{field_name} must be one of 'CalendarTime', "
                    f"'CalendarUniform', 'BusinessTime', 'BusinessUniform' "
                    f"or 'Fixed'."
                )

        elif field_name in ('medFrequencySamplingInterval',
                            'noiseVarianceSamplingInterval',
                            'IQEstimationSamplingInterval'):
            # Ref: realized_twoscale_variance.m:329-338
            if field_value is None or not _isnonnegativescalar(field_value):
                raise ValueError(
                    f'OPTIONS.{field_name} must be a non-negative scalar '
                    f'between 0 and 1.'
                )
            if time_type == 'unit' and float(field_value) > 1:
                raise ValueError(
                    f"OPTIONS.{field_name} must be less than 1 if TIMETYPE "
                    f"when 'unit'."
                )

    # ---------------------------------------------------------------
    # Subsamples validation
    # Ref: realized_twoscale_variance.m:342-346
    # ---------------------------------------------------------------
    if subsamples is not None:
        sub_val = float(subsamples)
        if (not np.isscalar(subsamples)
                or sub_val < 0
                or np.floor(sub_val) != sub_val):
            raise ValueError('SUBSAMPLES must be a non-negative scalar.')

    return options


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def realized_twoscale_variance(
    price: np.ndarray,
    time: np.ndarray | None = None,
    time_type: str = 'unit',
    sampling_type: str = 'CalendarTime',
    sampling_interval: int | float | np.ndarray = 1,
    subsamples: int = 1,
    options: dict | None = None,
) -> tuple[float, float, dict]:
    """Estimate quadratic variation using the two-scale realized variance.

    Computes the Ait-Sahalia, Mykland and Zhang (2005) two-scale
    realized variance (TSRV) estimator, which combines a slow-scale
    overlapping realized variance with a fast-scale bias correction to
    obtain a consistent estimator under market microstructure noise.

    Parameters
    ----------
    price : array_like
        1-D array of *m* high-frequency prices.  If a row vector is
        provided, it is automatically transposed.
    time : array_like or None, optional
        1-D array of *m* times corresponding to *price*.  When ``None``
        (default), a synthetic time grid from 9:30 AM to 4:00 PM in
        seconds is generated and ``time_type`` / ``sampling_type`` are
        overridden to ``'seconds'`` / ``'businesstime'``.
    time_type : str, optional
        Time format descriptor.  Case-insensitive.  One of:

        * ``'wall'``    — 24-hour clock in HHMMSS format.
        * ``'seconds'`` — Seconds past midnight.
        * ``'unit'``    — Unit-normalised [0, 1] interval.

        Default is ``'unit'``.
    sampling_type : str, optional
        Sampling scheme for filtering the fast-scale price grid.
        Case-insensitive.  One of:

        * ``'CalendarTime'``     — Fixed calendar-time interval.
        * ``'CalendarUniform'``  — Uniform calendar-time spacing.
        * ``'BusinessTime'``     — Every *N*-th tick.
        * ``'BusinessUniform'``  — Uniform tick spacing.
        * ``'Fixed'``            — User-supplied sampling times.

        Default is ``'CalendarTime'``.
    sampling_interval : int, float, or array_like, optional
        Meaning depends on *sampling_type*.  Default is ``1``.
    subsamples : int, optional
        Number of subsamples for the subsampled TSRV variant.
        Default is ``1`` (no additional subsampling).  Set to ``0``
        to skip subsampled computation (subsampled results will be
        ``float('nan')``).
    options : dict or None, optional
        Realized two-scale option dictionary.  Use
        ``realized_options('twoscale')`` to create defaults.  Key fields:

        * ``'bandwidth'`` — Slow-scale bandwidth *K*.  ``None`` → auto
          via noise estimation.
        * ``'useDebiasedNoise'`` — If ``True``, use debiased noise
          variance for bandwidth selection.

        When ``None``, defaults are created via
        ``realized_options('twoscale')``.

    Returns
    -------
    tsrv : float
        Two-scale realized variance estimate (Eq. 55 in Zhang et al.).
    tsrv_debiased : float
        Debiased two-scale realized variance (Eq. 64).
    diagnostics : dict
        Dictionary of diagnostic information with keys:

        * ``'bandwidth'`` — int, the slow-scale bandwidth used.
        * ``'noiseVariance'`` — float, Bandi-Russell noise variance
          estimate (present only if bandwidth was auto-selected).
        * ``'debiasedNoiseVariance'`` — float, debiased noise variance
          (present only if bandwidth was auto-selected).
        * ``'IQEstimate'`` — float, integrated quarticity estimate
          (present only if bandwidth was auto-selected).
        * ``'tsrv_ss'`` — float, subsampled TSRV.
        * ``'tsrv_ss_debiased'`` — float, subsampled debiased TSRV.

    Raises
    ------
    ValueError
        If any input validation check fails (wrong shapes, unsorted
        time, invalid options, etc.).

    Notes
    -----
    **Core algorithm:**

    1. Compute filtered log prices at the specified sampling frequency.
    2. If bandwidth *K* is not user-supplied, estimate the optimal
       bandwidth from the microstructure noise estimate.
    3. Compute the fast-scale RV from all returns (``skip=1``).
    4. Compute the slow-scale overlapping RV from ``skip=K`` returns,
       normalised by the overlap scaling factor.
    5. TSRV = slow - (n_bar / n) * fast, where n_bar = (n - K + 1) / K.
    6. Debiased TSRV = (1 - n_bar / n)^{-1} * TSRV.
    7. Optionally, repeat with subsampled price grids and average.

    **MATLAB → Python translation notes:**

    * MATLAB 1-based indexing → Python 0-based indexing throughout.
    * ``ceil(x)`` → ``int(np.ceil(x))``.
    * ``error()`` → ``raise ValueError()``.
    * ``warning()`` → ``warnings.warn()``.
    * ``nargin`` switch → Python keyword defaults with ``None`` checks.
    * Five MATLAB outputs ``[rvts, rvtsSS, rvtsD, rvtsSSD, diagnostics]``
      are mapped to three Python outputs ``(tsrv, tsrv_debiased, diagnostics)``
      with subsampled values included in the diagnostics dict.

    Examples
    --------
    >>> import numpy as np
    >>> # Default: BusinessTime sampling with interval 1, auto bandwidth
    >>> tsrv, tsrv_d, diag = realized_twoscale_variance(prices)

    >>> # 5-tick TSRV with auto bandwidth
    >>> tsrv, tsrv_d, diag = realized_twoscale_variance(
    ...     prices, times, 'wall', 'BusinessTime', 5)

    >>> # Fixed bandwidth of 30
    >>> from mfe_toolbox.realized.realized_options import realized_options
    >>> opts = realized_options('twoscale')
    >>> opts['bandwidth'] = 30
    >>> tsrv, tsrv_d, diag = realized_twoscale_variance(
    ...     prices, times, 'wall', 'BusinessTime', 1, 0, opts)

    See Also
    --------
    realized_options, realized_noise_estimate, realized_variance,
    realized_variance_optimal_sampling, realized_range,
    realized_quantile_variance
    """
    # ==================================================================
    # Input Defaults
    # Ref: realized_twoscale_variance.m:88-106 — nargin switch
    # ==================================================================

    # Ensure price is a 1-D float64 array
    price = np.asarray(price, dtype=np.float64)
    # Ref: realized_twoscale_variance.m:223-225 — transpose row to column
    if price.ndim == 2 and price.shape[1] > price.shape[0]:
        price = price.ravel()
    elif price.ndim >= 2:
        price = price.ravel()

    # Ref: realized_twoscale_variance.m:88-96 — nargin==1 defaults
    if time is None:
        m = len(price)
        # Ref: realized_twoscale_variance.m:91 — linspace(9.5*3600,16*3600,m)
        time = np.linspace(9.5 * 3600.0, 16.0 * 3600.0, m)
        time_type = 'seconds'
        sampling_type = 'businesstime'
        sampling_interval = 1
        subsamples = 1
        options = realized_options('twoscale')
    else:
        time = np.asarray(time, dtype=np.float64)
        # Ref: realized_twoscale_variance.m:230-231 — transpose row to col
        if time.ndim == 2 and time.shape[1] > time.shape[0]:
            time = time.ravel()
        elif time.ndim >= 2:
            time = time.ravel()

    # Ref: realized_twoscale_variance.m:101 — default options
    if options is None:
        options = realized_options('twoscale')

    # Ref: realized_twoscale_variance.m:107-108 — lowercase
    sampling_type = sampling_type.lower()
    time_type = time_type.lower()

    # Ensure subsamples is an integer
    subsamples = int(subsamples)

    # ==================================================================
    # Input Validation
    # Ref: realized_twoscale_variance.m:109-113
    # ==================================================================
    options = _realized_twoscale_variance_parameter_check(
        price, time, time_type, sampling_type, sampling_interval,
        subsamples, options
    )

    # ==================================================================
    # Core Computation
    # ==================================================================

    # Ref: realized_twoscale_variance.m:118
    log_price = np.log(price)

    # Ref: realized_twoscale_variance.m:119 — filter log prices
    # realized_price_filter returns (filtered_price, filtered_time, actual_time)
    # MATLAB captures only the first output
    filtered_log_price = realized_price_filter(
        log_price, time, time_type, sampling_type, sampling_interval
    )[0]

    diagnostics: dict = {}

    # ------------------------------------------------------------------
    # Bandwidth estimation (if not user-provided)
    # Ref: realized_twoscale_variance.m:121-137
    # ------------------------------------------------------------------
    if options.get('bandwidth') is None:
        # Ref: realized_twoscale_variance.m:123
        # realized_noise_estimate returns (noiseVar, debiasedNoiseVar, IQ, oomenNoise)
        # MATLAB captures only first 3
        noise_result = realized_noise_estimate(price, time, time_type, options)
        noise_variance = float(noise_result[0])
        debiased_noise_variance = float(noise_result[1])
        iq_estimate = float(noise_result[2])

        # Ref: realized_twoscale_variance.m:124-126
        diagnostics['noiseVariance'] = noise_variance
        diagnostics['debiasedNoiseVariance'] = debiased_noise_variance
        diagnostics['IQEstimate'] = iq_estimate

        # Ref: realized_twoscale_variance.m:128-132 — select noise estimate
        if options.get('useDebiasedNoise', False):
            selected_noise_variance = debiased_noise_variance
        else:
            selected_noise_variance = noise_variance

        # Ref: realized_twoscale_variance.m:134-136 — compute optimal bandwidth
        n = len(filtered_log_price) - 1
        # Ref: realized_twoscale_variance.m:135 — cOpt = ((12*noise^2)/IQ)^(1/3)
        c_opt = ((12.0 * selected_noise_variance ** 2) / iq_estimate) ** (1.0 / 3.0)
        # Ref: realized_twoscale_variance.m:136 — bandwidth = ceil(cOpt * n^(2/3))
        options['bandwidth'] = int(np.ceil(c_opt * n ** (2.0 / 3.0)))

    # Ref: realized_twoscale_variance.m:139
    bandwidth = int(np.ceil(options['bandwidth']))

    # Ref: realized_twoscale_variance.m:140-143 — warn if bandwidth < 2
    if bandwidth < 2:
        warnings.warn(
            'The selected bandwidth is less then 2, and RTVS is not well '
            'defined in this case.  Setting bandwidth to 2 and proceeding.',
            stacklevel=2,
        )
        bandwidth = 2

    # Ref: realized_twoscale_variance.m:144
    diagnostics['bandwidth'] = bandwidth

    # ------------------------------------------------------------------
    # Non-subsampled TSRV
    # Ref: realized_twoscale_variance.m:147-157
    # ------------------------------------------------------------------

    # Ref: realized_twoscale_variance.m:147
    n = len(filtered_log_price) - 1
    # Ref: realized_twoscale_variance.m:148
    K = bandwidth
    # Ref: realized_twoscale_variance.m:149 — nbar = (n - K + 1) / K
    nbar = (n - K + 1) / K

    # Ref: realized_twoscale_variance.m:150
    rv_k, count, natural_count = _overlap_realized_variance(filtered_log_price, K)
    # Ref: realized_twoscale_variance.m:151
    overlap_scale = count / natural_count
    # Ref: realized_twoscale_variance.m:152 — RVq = RV / overlapScale
    rv_q = rv_k / overlap_scale

    # Ref: realized_twoscale_variance.m:153 — RV1 = overlap_realized_variance(filteredLogPrice,1)
    rv_1, _, _ = _overlap_realized_variance(filtered_log_price, 1)

    # Ref: realized_twoscale_variance.m:155 — Eq. 55: rvts = RVq - nbar/n * RV1
    tsrv = rv_q - nbar / n * rv_1

    # Ref: realized_twoscale_variance.m:157 — Eq. 64: rvtsD = (1-nbar/n)^(-1)*rvts
    tsrv_debiased = (1.0 - nbar / n) ** (-1) * tsrv

    # ------------------------------------------------------------------
    # Subsampled TSRV
    # Ref: realized_twoscale_variance.m:159-185
    # ------------------------------------------------------------------

    if subsamples >= 1:
        # Ref: realized_twoscale_variance.m:160
        subsampled_log_prices = realized_subsample(
            log_price, time, time_type, sampling_type,
            sampling_interval, subsamples
        )

        # Ref: realized_twoscale_variance.m:161-162
        rvqs = np.zeros(subsamples, dtype=np.float64)
        rv1s = np.zeros(subsamples, dtype=np.float64)

        # Ref: realized_twoscale_variance.m:163-164
        total_count = 0
        total_count0 = 0
        base_count = 0
        base_count0 = 0

        # Ref: realized_twoscale_variance.m:165-178
        # MATLAB: for i=1:subsamples  (1-based)
        # Python: for i in range(subsamples)  (0-based)
        for i in range(subsamples):
            # Ref: realized_twoscale_variance.m:166
            # MATLAB: filteredLogPrice = subsampledLogPrices{i}
            # Python: subsampledLogPrices[i] is a tuple (prices, times, base_count, total_count)
            # Access the prices array at index [0]
            filtered_lp = subsampled_log_prices[i][0]

            # Ref: realized_twoscale_variance.m:167
            rv_val, cnt, _ = _overlap_realized_variance(filtered_lp, K)
            # Ref: realized_twoscale_variance.m:168
            rvqs[i] = rv_val
            # Ref: realized_twoscale_variance.m:169
            total_count += cnt
            # Ref: realized_twoscale_variance.m:170-172
            # MATLAB: if i==1 → Python: if i==0
            if i == 0:
                base_count = cnt

            # Ref: realized_twoscale_variance.m:173
            rv1_val, cnt1, _ = _overlap_realized_variance(filtered_lp, 1)
            # Ref: realized_twoscale_variance.m:173 (second assignment to rv1s)
            rv1s[i] = rv1_val
            # Ref: realized_twoscale_variance.m:174
            total_count0 += cnt1
            # Ref: realized_twoscale_variance.m:175-177
            if i == 0:
                base_count0 = cnt1

        # Ref: realized_twoscale_variance.m:179
        rv_q_ss = float(np.sum(rvqs)) * (base_count / total_count) if total_count > 0 else 0.0
        # Ref: realized_twoscale_variance.m:182 — normalize by overlap scale
        rv_q_ss = rv_q_ss / overlap_scale
        # Ref: realized_twoscale_variance.m:183
        rv_1_ss = float(np.sum(rv1s)) * (base_count0 / total_count0) if total_count0 > 0 else 0.0

        # Ref: realized_twoscale_variance.m:184 — rvtsSS = RVqSS - nbar/n * RV1SS
        tsrv_ss = rv_q_ss - nbar / n * rv_1_ss
        # Ref: realized_twoscale_variance.m:185 — rvtsSSD = (1-nbar/n)^(-1)*rvtsSS
        tsrv_ss_debiased = (1.0 - nbar / n) ** (-1) * tsrv_ss
    else:
        # subsamples == 0: skip subsampled computation
        # MATLAB would error due to undefined baseCount; Python handles gracefully
        tsrv_ss = float('nan')
        tsrv_ss_debiased = float('nan')

    # Store subsampled results in diagnostics
    diagnostics['tsrv_ss'] = float(tsrv_ss)
    diagnostics['tsrv_ss_debiased'] = float(tsrv_ss_debiased)

    return float(tsrv), float(tsrv_debiased), diagnostics
