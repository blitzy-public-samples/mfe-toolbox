"""
Realized variance and subsampled realized variance estimator.

This module provides the :func:`realized_variance` function, which computes
the standard realized variance (RV) estimator by summing squared intraday
log returns over a filtered price grid, as well as the subsampled realized
variance (RVSS) that averages multiple RV estimates computed on shifted grids
for finite-sample bias reduction.

The realized variance is defined as:

.. math::

    RV = \\sum_{i=1}^{n} r_i^2

where :math:`r_i = \\log(P_{t_i}) - \\log(P_{t_{i-1}})` are log returns
computed on filtered (sampled) prices.

Migrated from: ``realized/realized_variance.m`` (MFE Toolbox Version 4.0)

See Also
--------
mfe_toolbox.realized.realized_kernel : Realized kernel estimator.
mfe_toolbox.realized.realized_quantile_variance : Quantile-based RV.
mfe_toolbox.realized.realized_range : Range-based estimator.
mfe_toolbox.realized.realized_price_filter : Core price filtering function.
mfe_toolbox.realized.realized_subsample : Subsampling grid generation.

Notes
-----
Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 1    Date: 5/1/2008
"""

import warnings

import numpy as np

from mfe_toolbox.realized.realized_price_filter import realized_price_filter
from mfe_toolbox.realized.realized_subsample import realized_subsample
from mfe_toolbox.realized.realized_convert2unit import realized_convert2unit


def realized_variance(
    price,
    time=None,
    time_type: str = 'unit',
    sampling_type: str = 'CalendarTime',
    sampling_interval=1,
    subsamples: int = 1,
) -> tuple[float, float, dict]:
    """Estimate realized variance and subsampled realized variance.

    Computes the standard realized variance (sum of squared log returns)
    on a filtered price grid, plus an optional subsampled (averaged)
    estimator for bias reduction.

    Parameters
    ----------
    price : array_like
        An m-element 1-D vector of high-frequency prices.  If a row
        vector is provided, it is transposed to a column vector.
        Must contain at least 2 elements and all positive values.
    time : array_like or None, optional
        An m-element 1-D vector of observation times corresponding to
        *price*.  Must be sorted in non-decreasing order.  The format
        must match *time_type*:

        * ``'wall'``    — 24-hour clock in HHMMSS format (e.g. 93000).
        * ``'seconds'`` — Seconds past midnight (e.g. 34200).
        * ``'unit'``    — Unit-normalised on [0, 1].

        If ``None`` (default), a uniformly spaced time grid on [0, 1]
        is generated and *time_type* is set to ``'unit'``.
    time_type : str, optional
        Time format descriptor.  Case-insensitive.  One of ``'wall'``,
        ``'seconds'``, or ``'unit'``.  Default is ``'unit'``.
    sampling_type : str, optional
        Sampling scheme.  Case-insensitive.  One of:

        * ``'CalendarTime'``     — Fixed calendar-time interval.
        * ``'CalendarUniform'``  — Uniform spacing in calendar time.
        * ``'BusinessTime'``     — Every N-th tick.
        * ``'BusinessUniform'``  — Uniform spacing in tick time.
        * ``'Fixed'``            — Sample at specific user-supplied times.

        Default is ``'CalendarTime'``.
    sampling_interval : int, float, or array_like, optional
        Interpretation depends on *sampling_type*.  See
        :func:`realized_price_filter` for full details.  Default is 1.
    subsamples : int, optional
        Number of subsample realized variance estimators to average with
        the original.  ``1`` (default) uses a single sample (no
        subsampling).  Values > 1 compute multiple shifted grids for
        bias reduction.

    Returns
    -------
    tuple[float, float, dict]
        A 3-tuple ``(rv, rv_ss, diagnostics)`` where:

        * **rv** — Standard realized variance (sum of squared returns).
        * **rv_ss** — Subsampled realized variance (weighted average of
          RV estimates on shifted grids).  When *subsamples* = 1,
          ``rv_ss`` equals ``rv``.
        * **diagnostics** — Dictionary with estimation details:

          - ``'num_prices'``          : int — Original price vector length.
          - ``'num_filtered_prices'`` : int — Number of filtered prices.
          - ``'num_returns'``         : int — Number of log returns.
          - ``'rv'``                  : float — Standard RV.
          - ``'rv_ss'``              : float — Subsampled RV.
          - ``'subsamples'``         : int — Number of subsamples used.
          - ``'sampling_type'``      : str — Sampling scheme.
          - ``'sampling_interval'``  : sampling interval value.
          - ``'time_type'``          : str — Time format used.
          - ``'base_count'``         : int — Returns in first subsample.
          - ``'total_count'``        : int — Total returns across subsamples.

    Raises
    ------
    ValueError
        If *price* has fewer than 2 elements.
        If any *price* value is non-positive.
        If *time* is not sorted and increasing.
        If *time* length does not match *price* length.
        If *time_type* is not a recognized format.
        If *sampling_type* is not a recognized scheme.
        If *sampling_interval* is invalid for the given *sampling_type*
        and *time_type* combination.
        If *subsamples* is not a positive integer.

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.realized.realized_variance import realized_variance
    >>> prices = np.array([100.0, 100.5, 101.0, 100.8, 101.2, 101.5])
    >>> times = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    >>> rv, rv_ss, diag = realized_variance(prices, times, 'unit',
    ...                                      'CalendarTime', 0.5)
    >>> rv > 0
    True

    Notes
    -----
    **Core algorithm (from realized_variance.m):**

    1. Compute log prices: ``logPrice = log(price)``
    2. Filter log prices: ``filteredLogPrice = realized_price_filter(
       logPrice, time, timeType, samplingType, samplingInterval)``
    3. Compute log returns: ``returns = diff(filteredLogPrice)``
    4. Realized variance: ``rv = returns' * returns``
    5. For subsampled RV:

       a. Generate shifted grids: ``realized_subsample(logPrice, time,
          timeType, samplingType, samplingInterval, subsamples)``
       b. Compute RV on each shifted grid.
       c. Weight by base count / total count:
          ``rvSS = sum(rvs) * (baseCount / totalCount)``

    **MATLAB → Python translation notes:**

    * ``returns' * returns`` → ``returns @ returns`` (dot product = sum
      of squared returns).
    * ``diff(logPrice)`` → ``np.diff(log_price)``
    * MATLAB cell array → Python ``list[tuple]`` from
      :func:`realized_subsample`.
    * Ref: realized_variance.m:139 — MATLAB ``log(price)`` → Python
      ``np.log(price)``
    * Ref: realized_variance.m:143 — MATLAB ``returns' * returns`` →
      Python ``returns @ returns``
    """
    # ==================================================================
    # Input Validation
    # Ref: realized_variance.m:68-132
    # ==================================================================

    # Ref: realized_variance.m:71-73 — Transpose row to column
    price = np.asarray(price, dtype=np.float64).ravel()
    m = len(price)

    # Ref: realized_variance.m:68 — At least 5 inputs required
    if m < 2:
        raise ValueError("PRICE must have at least 2 elements.")

    # Ref: realized_variance.m:74-76 — PRICE must be a vector
    # (Handled by ravel above; 2-D inputs would also fail in price filter)

    # Default time array when time is not supplied
    # Ref: matches convention in other realized estimators
    if time is None:
        time = np.linspace(0.0, 1.0, m)
        time_type = 'unit'

    # Ref: realized_variance.m:77-79 — Transpose time to column
    time = np.asarray(time, dtype=np.float64).ravel()

    # Ref: realized_variance.m:80-82 — TIME must be sorted and increasing
    if len(time) > 1 and np.any(np.diff(time) < 0):
        raise ValueError("TIME must be sorted and increasing")

    # Ref: realized_variance.m:83-85 — TIME length must match PRICE
    if len(time) != m:
        raise ValueError("TIME must be a m by 1 vector.")

    # Ref: realized_variance.m:87 — Cast to double (protect against ints)
    # Already handled by np.asarray(..., dtype=np.float64) above.

    # Ref: realized_variance.m:89-92 — Validate timeType
    time_type_lower = time_type.lower()
    if time_type_lower not in ('wall', 'seconds', 'unit'):
        raise ValueError(
            "TIMETYPE must be one of 'wall', 'seconds' or 'unit'."
        )

    # Ref: realized_variance.m:93-96 — Validate samplingType
    sampling_type_lower = sampling_type.lower()
    _valid_sampling = (
        'calendartime', 'calendaruniform',
        'businesstime', 'businessuniform', 'fixed',
    )
    if sampling_type_lower not in _valid_sampling:
        raise ValueError(
            "SAMPLINGTYPE must be one of 'CalendarTime', "
            "'CalendarUniform', 'BusinessTime', "
            "'BusinessUniform' or 'Fixed'."
        )

    # Ref: realized_variance.m:98-100 — Compute t0, tT for validation
    t0 = time[0]
    tT = time[-1]

    # Ref: realized_variance.m:101-122 — Validate samplingInterval
    if sampling_type_lower in ('calendartime', 'calendaruniform',
                                'businesstime', 'businessuniform'):
        if time_type_lower in ('wall', 'seconds'):
            # Ref: realized_variance.m:103-106 — positive scalar integer
            if (not np.isscalar(sampling_interval)
                    or np.floor(float(sampling_interval)) != float(sampling_interval)
                    or float(sampling_interval) < 1):
                raise ValueError(
                    'SAMPLINGINTERVAL must be a positive integer for '
                    'the SAMPLINGTYPE selected when using '
                    "'wall' or 'seconds' as TIMETYPE."
                )
        else:
            # Ref: realized_variance.m:108-110 — positive scalar (unit)
            if not np.isscalar(sampling_interval) or float(sampling_interval) < 0:
                raise ValueError(
                    'SAMPLINGINTERVAL must be a positive value for '
                    'the SAMPLINGTYPE selected when using '
                    "'unit' as TIMETYPE."
                )
    else:
        # Ref: realized_variance.m:112-122 — Fixed sampling type
        si_arr = np.asarray(sampling_interval, dtype=np.float64).ravel()
        if not (np.any(si_arr >= t0) and np.any(si_arr <= tT)):
            raise ValueError(
                'At least one sampling interval must be between '
                "min(TIME) and max(TIME) when using 'Fixed' "
                'as SAMPLINGTYPE.'
            )
        if len(si_arr) > 1 and np.any(np.diff(si_arr) <= 0):
            raise ValueError(
                "When using 'Fixed' as SAMPLINGTYPE the vector of "
                'sampling times in SAMPLINGINTERVAL must be sorted '
                'and strictly increasing.'
            )

    # Ref: realized_variance.m:124-132 — Validate subsamples
    if subsamples is None:
        subsamples = 1
    subsamples = int(subsamples)
    if subsamples < 1:
        raise ValueError('SUBSAMPLES must be a non-negative scalar.')

    # ==================================================================
    # Core Computation
    # Ref: realized_variance.m:139-157
    # ==================================================================

    # Ref: realized_variance.m:139 — logPrice = log(price)
    log_price = np.log(price)

    # Ref: realized_variance.m:141 — Filter log prices
    # realized_price_filter returns (filtered_price, filtered_time,
    # actual_time).  We need only the filtered prices.
    filter_result = realized_price_filter(
        log_price, time, time_type, sampling_type, sampling_interval
    )
    if isinstance(filter_result, tuple):
        filtered_log_price = np.asarray(
            filter_result[0], dtype=np.float64
        ).ravel()
    else:
        filtered_log_price = np.asarray(
            filter_result, dtype=np.float64
        ).ravel()

    # Ref: realized_variance.m:142 — returns = diff(filteredLogPrice)
    returns = np.diff(filtered_log_price)

    # Ref: realized_variance.m:143 — rv = returns' * returns
    rv = float(returns @ returns)

    # ==================================================================
    # Subsampled Realized Variance
    # Ref: realized_variance.m:146-157
    # ==================================================================

    # Ref: realized_variance.m:146 — Generate shifted price grids
    subsample_results = realized_subsample(
        log_price, time, time_type, sampling_type,
        sampling_interval, subsamples,
    )

    # Ref: realized_variance.m:147-156 — Compute RV on each grid
    rvs = np.zeros(subsamples)
    total_count = 0
    base_count = 0
    for i in range(subsamples):
        # realized_subsample returns list of tuples:
        # (subsampled_prices, subsampled_times, base_count, total_count)
        # Ref: realized_variance.m:150 — returns = diff(subsampledLogPrices{i})
        sub_log_prices = np.asarray(
            subsample_results[i][0], dtype=np.float64
        ).ravel()
        sub_returns = np.diff(sub_log_prices)

        # Ref: realized_variance.m:151 — rvs(i) = returns' * returns
        rvs[i] = float(sub_returns @ sub_returns)

        # Ref: realized_variance.m:152-154 — Track base and total counts
        if i == 0:
            base_count = len(sub_returns)
        total_count += len(sub_returns)

    # Ref: realized_variance.m:157 — rvSS = sum(rvs) * (baseCount / totalCount)
    if total_count > 0:
        rv_ss = float(np.sum(rvs) * (base_count / total_count))
    else:
        # Fallback: if no returns at all, use the basic RV
        rv_ss = rv
        warnings.warn(
            "Subsampled realized variance has zero total returns; "
            "falling back to standard RV.",
            stacklevel=2,
        )

    # ==================================================================
    # Diagnostics
    # ==================================================================
    diagnostics = {
        'num_prices': m,
        'num_filtered_prices': len(filtered_log_price),
        'num_returns': len(returns),
        'rv': rv,
        'rv_ss': rv_ss,
        'subsamples': subsamples,
        'sampling_type': sampling_type,
        'sampling_interval': sampling_interval,
        'time_type': time_type,
        'base_count': base_count,
        'total_count': total_count,
    }

    return rv, rv_ss, diagnostics
