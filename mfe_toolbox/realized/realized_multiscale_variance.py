"""
Multiscale realized variance (MSRV) estimator of Zhang (2006).

Implements the multi-scale realized variance estimator which combines
realized variances computed at multiple subsampling frequencies with
kernel-based weights to produce a consistent, rate-optimal estimator
of integrated variance in the presence of market microstructure noise.

Migrated from: ``realized/realized_multiscale_variance.m`` (MFE Toolbox v4.0)
Original author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
Revision: 1    Date: 5/1/2008

See Also
--------
realized_options : Default options factory for realized volatility estimators.
realized_noise_estimate : Microstructure noise variance estimation.
realized_variance : Standard realized variance estimator.
realized_twoscale_variance : Two-scale realized variance estimator.
realized_variance_optimal_sampling : Optimal sampling frequency selection.

References
----------
.. [1] Zhang, L. (2006). "Efficient estimation of stochastic volatility using
   noisy observations: A multi-scale approach." *Bernoulli*, 12(6), 1019–1043.
"""

from __future__ import annotations

import warnings

import numpy as np


# ---------------------------------------------------------------------------
# Internal imports from the mfe_toolbox.realized subpackage
# ---------------------------------------------------------------------------
from mfe_toolbox.realized.realized_options import realized_options
from mfe_toolbox.realized.realized_noise_estimate import realized_noise_estimate
from mfe_toolbox.realized.realized_price_filter import realized_price_filter
from mfe_toolbox.realized.realized_convert2unit import realized_convert2unit
from mfe_toolbox.realized.realized_subsample import realized_subsample


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _overlap_realized_variance(
    price: np.ndarray,
    skip: int,
) -> tuple[float, int, float]:
    """Compute the overlapping realized variance at a given scale.

    For a vector of (log) prices of length *m*, the overlapping RV at
    scale *skip* is defined as the sum of squared overlapping returns
    of span ``skip + 1`` observations.

    Parameters
    ----------
    price : np.ndarray
        1-D array of (log) prices of length m.
    skip : int
        Sub-grid size (positive integer ≥ 1).  Determines the return
        span: each return covers ``skip + 1`` consecutive prices.

    Returns
    -------
    rv : float
        Overlapping realized variance (sum of squared overlapping returns).
    count : int
        Number of overlapping returns actually computed.
    natural_count : float
        Theoretical number of non-overlapping returns at this scale,
        ``(m - 1) / skip``.

    Notes
    -----
    Ref: realized_multiscale_variance.m:210-218

    MATLAB indexing translation:
      ``returns = price(2+skip : m) - price(1 : m-skip-1)``
      Python:  ``returns = price[skip+1:] - price[:m-skip-1]``
    """
    m = len(price)
    # Ref: realized_multiscale_variance.m:212
    natural_count = float(m - 1) / float(skip)

    # Ref: realized_multiscale_variance.m:214 — MATLAB 1-based to Python 0-based
    # MATLAB: returns = price(2+skip:m) - price(1:m-skip-1)
    upper_end = m  # inclusive in MATLAB, exclusive in Python slice
    lower_end = m - skip - 1  # number of elements
    if lower_end <= 0 or skip + 1 >= m:
        return 0.0, 0, natural_count

    returns = price[skip + 1:] - price[:m - skip - 1]
    count = len(returns)

    if count == 0:
        return 0.0, 0, natural_count

    # Ref: realized_multiscale_variance.m:216 — rv = returns' * returns
    rv = float(np.dot(returns, returns))

    return rv, count, natural_count


def _parameter_check(
    price: np.ndarray,
    time: np.ndarray,
    time_type: str,
    sampling_type: str,
    sampling_interval,
    subsamples: int,
    options: dict,
) -> tuple[str | None, dict]:
    """Validate inputs for :func:`realized_multiscale_variance`.

    Mirrors ``realized_multiscale_variance_parameter_check`` in the MATLAB
    source (lines 222–373).

    Returns
    -------
    error_message : str or None
        Description of the first validation error found, or ``None`` if all
        inputs are valid.
    options : dict
        Options dict with any missing fields populated from defaults.
    """
    # Ref: realized_multiscale_variance.m:243-250 — price validation
    if price.ndim >= 2 and min(price.shape) > 1:
        return 'PRICE must be a m by 1 vector.', options

    # Ref: realized_multiscale_variance.m:252-257 — time validation
    if np.any(np.diff(time) < 0):
        return 'TIME must be sorted and increasing', options
    if len(time) != len(price):
        return 'TIME must be a m by 1 vector.', options

    # Ref: realized_multiscale_variance.m:264-267 — timeType validation
    time_type_lower = time_type.lower()
    if time_type_lower not in ('wall', 'seconds', 'unit'):
        return "TIMETYPE must be one of 'wall', 'seconds' or 'unit'.", options

    # Ref: realized_multiscale_variance.m:271-289 — samplingType validation
    sampling_type_lower = sampling_type.lower()
    if sampling_type_lower in (
        'calendartime', 'calendaruniform', 'businesstime', 'businessuniform'
    ):
        # Ref: realized_multiscale_variance.m:274-276
        if (
            not np.isscalar(sampling_interval)
            or np.floor(float(sampling_interval)) != float(sampling_interval)
            or float(sampling_interval) < 1
        ):
            return (
                'SAMPLINGINTERVAL must be a positive integer for the '
                'SAMPLINGTYPE selected.'
            ), options
    else:
        # Fixed sampling type — validate vector
        si_arr = np.asarray(sampling_interval, dtype=np.float64).ravel()
        t0 = time[0]
        tT = time[-1]
        # Ref: realized_multiscale_variance.m:282-284
        if not (np.any(si_arr >= t0) and np.any(si_arr <= tT)):
            return (
                'At least one sampling interval must be between min(TIME) '
                "and max(TIME) when using 'Fixed' as SAMPLINGTYPE."
            ), options
        # Ref: realized_multiscale_variance.m:286-289
        if len(si_arr) > 1 and np.any(np.diff(si_arr) <= 0):
            return (
                "When using 'Fixed' as SAMPLINGTYPE the vector of sampling "
                'times in SAMPLINGINTERVAL must be sorted and strictly '
                'increasing.'
            ), options

    # Ref: realized_multiscale_variance.m:306-316 — fill missing options fields
    # The MATLAB code uses 'Optimal Sampling' defaults; Python uses 'multiscale'
    # which produces the same _SAMPLING_FIELDS set.
    default_options = realized_options('multiscale')
    for key, value in default_options.items():
        if key not in options:
            options[key] = value

    # Ref: realized_multiscale_variance.m:318-324 — lowercase string fields
    for key in list(options.keys()):
        if isinstance(options[key], str):
            options[key] = options[key].lower()

    # ----------------------------------------------------------------
    # Validate individual options fields
    # Ref: realized_multiscale_variance.m:296-361
    # ----------------------------------------------------------------
    flat_top_kernel_list = [
        'bartlett', 'multiscale', '2ndorder', 'epanechnikov',
        'cubic', '5thorder', '6thorder', '7thorder', '8thorder',
        'parzen', 'th1', 'th2', 'th5', 'th16',
    ]
    non_flat_top_kernel_list = ['nonflatparzen', 'qs', 'fejer', 'thinf', 'bnhls']
    kernel_list = flat_top_kernel_list + non_flat_top_kernel_list

    valid_sampling_types = (
        'calendartime', 'calendaruniform', 'businesstime',
        'businessuniform', 'fixed',
    )

    for field_name, field_value in options.items():
        if field_name == 'medFrequencyKernel':
            if field_value not in kernel_list:
                return (
                    f'OPTIONS.{field_name} must be one of the listed types.',
                    options,
                )
        elif field_name == 'medFrequencyBandwidth':
            # Ref: realized_multiscale_variance.m:333-337
            if field_value is not None and (
                not np.isscalar(field_value) or float(field_value) < 0
            ):
                return (
                    f'OPTIONS.{field_name} must a non-negative scalar.',
                    options,
                )
        elif field_name in ('useDebiasedNoise', 'useAdjustedNoiseCount'):
            # Ref: realized_multiscale_variance.m:339-343
            if not isinstance(field_value, (bool, int)):
                return (
                    'OPTIONS.useDebiasedNoise must be a logical value.',
                    options,
                )
            if isinstance(field_value, int) and field_value not in (0, 1):
                return (
                    'OPTIONS.useDebiasedNoise must be a logical value.',
                    options,
                )
        elif field_name in (
            'IQEstimationSamplingType',
            'medFrequencySamplingType',
            'noiseVarianceSamplingType',
        ):
            # Ref: realized_multiscale_variance.m:345-349
            if field_value not in valid_sampling_types:
                return (
                    f"OPTIONS.{field_name} must be one of 'CalendarTime', "
                    "'CalendarUniform', 'BusinessTime', 'BusinessUniform' "
                    "or 'Fixed'.",
                    options,
                )
        elif field_name in (
            'medFrequencySamplingInterval',
            'noiseVarianceSamplingInterval',
            'IQEstimationSamplingInterval',
        ):
            # Ref: realized_multiscale_variance.m:351-360
            if field_value is None or not np.isscalar(field_value) or float(field_value) < 0:
                return (
                    f'OPTIONS.{field_name} must be a non-negative scalar '
                    'between 0 and 1.',
                    options,
                )
            if time_type_lower == 'unit' and float(field_value) > 1:
                return (
                    f"OPTIONS.{field_name} must be less than 1 if TIMETYPE "
                    "when 'unit'.",
                    options,
                )

    # Ref: realized_multiscale_variance.m:364-368 — subsamples validation
    if subsamples is not None:
        if (
            not np.isscalar(subsamples)
            or float(subsamples) < 0
            or np.floor(float(subsamples)) != float(subsamples)
        ):
            return 'SUBSAMPLES must be a non-negative scalar.', options

    return None, options


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def realized_multiscale_variance(
    price,
    time=None,
    time_type: str = 'unit',
    sampling_type: str = 'CalendarTime',
    sampling_interval=1,
    subsamples: int = 1,
    options: dict | None = None,
) -> tuple[float, float, dict]:
    """Estimate quadratic variation using the Multi-Scale estimator of Zhang (2006).

    The multiscale realized variance (MSRV) estimator combines realized
    variances computed at multiple subsampling frequencies with kernel-based
    weights to produce a consistent, rate-optimal estimator of integrated
    variance in the presence of market microstructure noise.

    Parameters
    ----------
    price : array_like
        1-D array of *m* high-frequency prices (not log-prices).
    time : array_like or None, optional
        1-D array of *m* observation times corresponding to *price*.
        Must be sorted in non-decreasing order.  If ``None`` (default),
        a default grid from 09:30 to 16:00 in seconds-past-midnight is
        generated and ``time_type`` is overridden to ``'seconds'``,
        ``sampling_type`` to ``'businesstime'``, ``sampling_interval`` to 1.
    time_type : str, optional
        Format of *time*.  One of:

        * ``'wall'``    — 24-hour clock HHMMSS (e.g. 101543).
        * ``'seconds'`` — Seconds past midnight.
        * ``'unit'``    — Unit-normalized [0, 1].

        Default is ``'unit'``.
    sampling_type : str, optional
        Sampling scheme for the fast-scale estimator.  One of:

        * ``'CalendarTime'``     — Fixed calendar-time interval.
        * ``'CalendarUniform'``  — Uniform calendar-time spacing.
        * ``'BusinessTime'``     — Every N-th tick.
        * ``'BusinessUniform'``  — Uniform tick spacing.
        * ``'Fixed'``            — User-specified time points.

        Default is ``'CalendarTime'``.
    sampling_interval : int, float, or array_like, optional
        Meaning depends on *sampling_type*.  Default is 1.
    subsamples : int, optional
        Number of subsamples for averaging.  Must be a non-negative integer.
        Default is 1 (no subsampling averaging).
    options : dict or None, optional
        Options dict as returned by ``realized_options('Multiscale')``.
        If ``None``, default options are used.  Key fields:

        * ``'bandwidth'`` — Ratio of slow-to-fast scale.  ``None`` for
          automatic MSE-optimal selection.
        * ``'useDebiasedNoise'`` — Whether to use the debiased noise
          variance for bandwidth selection.

    Returns
    -------
    rvms : float
        Realized multi-scale variance estimate.
    rvms_ss : float
        Realized multi-scale variance estimate constructed by averaging
        across multiple initial observations (subsampled version).
        Equals *rvms* when *subsamples* ≤ 1 or when subsampling is not
        possible (e.g. ``'BusinessTime'`` with ``sampling_interval=1``).
    diagnostics : dict
        Dictionary of diagnostic information:

        * ``'bandwidth'``             — Bandwidth (J) actually used.
        * ``'noiseVariance'``         — Bandi-Russell noise variance
          estimate (``None`` if user supplied bandwidth).
        * ``'debiasedNoiseVariance'`` — Debiased noise variance
          (``None`` if user supplied bandwidth).
        * ``'IQEstimate'``            — Integrated quarticity estimate
          (``None`` if user supplied bandwidth).

    Raises
    ------
    ValueError
        If any input fails validation.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> price = 100.0 * np.exp(np.cumsum(0.001 * rng.standard_normal(1000)))
    >>> rvms, rvms_ss, diag = realized_multiscale_variance(price)
    >>> isinstance(rvms, float)
    True

    Notes
    -----
    For best results:

    * Use prices sampled close to the highest frequency available if not
      using ``'BusinessTime'``.
    * When sampling less frequently (5–30 min), standard realized variance
      is probably more appropriate.
    """
    # ==================================================================
    # Input Checking
    # Ref: realized_multiscale_variance.m:84-112
    # ==================================================================

    # Coerce to float64 1-D arrays
    price = np.asarray(price, dtype=np.float64).ravel()

    # Ref: realized_multiscale_variance.m:84-102 — nargin-based defaults
    if time is None:
        # Ref: realized_multiscale_variance.m:86-92
        m = len(price)
        # Ref: linspace(9.5*3600, 16*3600, m)  (9:30 AM to 4:00 PM in seconds)
        time = np.linspace(9.5 * 3600.0, 16.0 * 3600.0, m)
        time_type = 'seconds'
        sampling_type = 'businesstime'
        sampling_interval = 1
        subsamples = 1
        options = realized_options('multiscale')
    else:
        time = np.asarray(time, dtype=np.float64).ravel()
        if options is None:
            options = realized_options('multiscale')
        else:
            # Make a copy to avoid mutating the caller's dict
            options = dict(options)

    # Ref: realized_multiscale_variance.m:103-104 — lowercase
    sampling_type = sampling_type.lower()
    time_type = time_type.lower()

    # Ref: realized_multiscale_variance.m:105-109 — parameter validation
    error_message, options = _parameter_check(
        price, time, time_type, sampling_type, sampling_interval,
        subsamples, options,
    )
    if error_message is not None:
        raise ValueError(error_message)

    # ==================================================================
    # Core Computation
    # ==================================================================

    # Ref: realized_multiscale_variance.m:114 — log-transform prices
    log_price = np.log(price)

    # Ref: realized_multiscale_variance.m:115 — filter log-prices
    # Python realized_price_filter returns (filtered_price, filtered_time, actual_time)
    filtered_log_price, filtered_times, _ = realized_price_filter(
        log_price, time, time_type, sampling_type, sampling_interval,
    )

    # Ref: realized_multiscale_variance.m:117 — number of returns (m)
    m = len(filtered_log_price) - 1

    # Initialize diagnostics with None for noise fields
    diagnostics: dict = {
        'noiseVariance': None,
        'debiasedNoiseVariance': None,
        'IQEstimate': None,
    }

    # ==================================================================
    # Bandwidth Selection
    # Ref: realized_multiscale_variance.m:120-161
    # ==================================================================
    if options.get('bandwidth') is None:
        # Ref: realized_multiscale_variance.m:122 — estimate noise and IQ
        # Python realized_noise_estimate returns 4 values; MATLAB captures 3.
        noise_variance, debiased_noise_variance, iq_estimate, _ = (
            realized_noise_estimate(price, time, time_type, options)
        )
        diagnostics['noiseVariance'] = noise_variance
        diagnostics['debiasedNoiseVariance'] = debiased_noise_variance
        diagnostics['IQEstimate'] = iq_estimate

        # Ref: realized_multiscale_variance.m:127-131 — select noise variance
        if options.get('useDebiasedNoise', False):
            selected_noise_variance = debiased_noise_variance
        else:
            selected_noise_variance = noise_variance

        # Ref: realized_multiscale_variance.m:133 — recalculate m
        m = len(filtered_log_price) - 1

        # Ref: realized_multiscale_variance.m:137-161 — bisection search
        if iq_estimate > 0:
            # Ref: realized_multiscale_variance.m:138-141
            lb = 2.0 / np.sqrt(float(m))
            ub = np.sqrt(float(m))

            # Ref: realized_multiscale_variance.m:142-143 — evaluate deriv at UB
            c = ub
            ub_deriv = (
                2.0 * (52.0 / 35.0) * iq_estimate * c ** 4
                - (48.0 / 5.0) * c ** 2 * (
                    selected_noise_variance * (
                        np.sqrt(iq_estimate) + selected_noise_variance / 2.0
                    )
                )
                - 144.0 * selected_noise_variance ** 2
            )

            # Ref: realized_multiscale_variance.m:144-155 — 32 bisection iterations
            for _ in range(32):
                mp = (ub + lb) / 2.0
                c = mp
                mp_deriv = (
                    2.0 * (52.0 / 35.0) * iq_estimate * c ** 4
                    - (48.0 / 5.0) * c ** 2 * (
                        selected_noise_variance * (
                            np.sqrt(iq_estimate) + selected_noise_variance / 2.0
                        )
                    )
                    - 144.0 * selected_noise_variance ** 2
                )

                if np.sign(mp_deriv) == np.sign(ub_deriv):
                    ub = mp
                    ub_deriv = mp_deriv
                else:
                    lb = mp

            # Ref: realized_multiscale_variance.m:156-157
            c = (ub + lb) / 2.0
            options['bandwidth'] = c * np.sqrt(float(m))
        else:
            # Ref: realized_multiscale_variance.m:159-160
            warnings.warn(
                'IQ Estimate is 0 so optimal bandwidth selection is not '
                'possible.',
                stacklevel=2,
            )
            options['bandwidth'] = 5.0

    # Ref: realized_multiscale_variance.m:164 — ceil bandwidth
    bandwidth = int(np.ceil(float(options['bandwidth'])))

    # Ref: realized_multiscale_variance.m:165-168 — clamp to minimum 2
    if bandwidth < 2:
        warnings.warn(
            'The selected bandwidth is less then 2, and RVMS is not well '
            'defined in this case.  Setting bandwidth to 2 and proceeding.',
            stacklevel=2,
        )
        bandwidth = 2

    # Ref: realized_multiscale_variance.m:169
    diagnostics['bandwidth'] = bandwidth

    # ==================================================================
    # Non-Subsampled MSRV
    # Ref: realized_multiscale_variance.m:173-184
    # ==================================================================
    M = bandwidth

    # Ref: realized_multiscale_variance.m:173-177 — compute overlap RV at each scale
    rv_q = np.zeros(M, dtype=np.float64)
    for skip in range(1, M + 1):
        # Ref: realized_multiscale_variance.m:175 — MATLAB skip is 1-based
        rv, count, natural_count = _overlap_realized_variance(
            filtered_log_price, skip,
        )
        # Ref: realized_multiscale_variance.m:176 — scale by naturalCount/count
        if count > 0:
            rv_q[skip - 1] = rv * (natural_count / float(count))
        else:
            rv_q[skip - 1] = 0.0

    # Ref: realized_multiscale_variance.m:179-183 — weights from Eq. 20 of Zhang
    weights = np.zeros(M, dtype=np.float64)
    for i in range(M):
        # Ref: realized_multiscale_variance.m:182 — MATLAB i is 1-based
        i_1 = float(i + 1)
        M_f = float(M)
        # Ref: weights(i) = 12*(i/M^2)*(i/M - 1/2 - 1/(2*M)) / (1 - 1/M^2)
        weights[i] = (
            12.0 * (i_1 / (M_f ** 2))
            * (i_1 / M_f - 0.5 - 1.0 / (2.0 * M_f))
            / (1.0 - 1.0 / (M_f ** 2))
        )

    # Ref: realized_multiscale_variance.m:184 — rvms = weights'*RVq
    rvms = float(np.dot(weights, rv_q))

    # ==================================================================
    # Subsampled MSRV
    # Ref: realized_multiscale_variance.m:190-205
    # ==================================================================
    if subsamples >= 1:
        # Ref: realized_multiscale_variance.m:190 — get subsampled log prices
        subsampled_result = realized_subsample(
            log_price, time, time_type, sampling_type, sampling_interval,
            subsamples,
        )

        # Ref: realized_multiscale_variance.m:191-193 — preallocate arrays
        rv_qs = np.zeros((subsamples, M), dtype=np.float64)
        count_arr = np.zeros((subsamples, M), dtype=np.float64)
        natural_count_arr = np.zeros((subsamples, M), dtype=np.float64)

        # Ref: realized_multiscale_variance.m:194-199
        for i in range(subsamples):
            # Ref: realized_multiscale_variance.m:195
            # MATLAB: filteredLogPrice = subsampledLogPrices{i}
            # Python: subsampled_result[i] is (prices, times, base_count, total)
            filtered_lp_sub = subsampled_result[i][0]

            for skip_idx in range(M):
                skip = skip_idx + 1  # 1-based skip value
                # Ref: realized_multiscale_variance.m:197
                rv, cnt, ncnt = _overlap_realized_variance(
                    filtered_lp_sub, skip,
                )
                rv_qs[i, skip_idx] = rv
                count_arr[i, skip_idx] = float(cnt)
                natural_count_arr[i, skip_idx] = ncnt

        # Ref: realized_multiscale_variance.m:200 — sum counts across subsamples
        count_sum = np.sum(count_arr, axis=0)

        # Ref: realized_multiscale_variance.m:201 — scale using first subsample
        # Handle potential division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            scale = np.where(
                count_sum > 0,
                natural_count_arr[0, :] / count_sum,
                0.0,
            )

        # Ref: realized_multiscale_variance.m:202 — sum RVqs across subsamples
        rv_qs_sum = np.sum(rv_qs, axis=0)

        # Ref: realized_multiscale_variance.m:203 — scale the summed RVqs
        rv_q_ss = rv_qs_sum * scale

        # Ref: realized_multiscale_variance.m:205 — rvmsSS = weights'*RVq
        rvms_ss = float(np.dot(weights, rv_q_ss))
    else:
        # subsamples == 0: subsampled version not well-defined; fallback
        rvms_ss = rvms

    return rvms, rvms_ss, diagnostics
