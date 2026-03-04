"""
Minimum and median realized variance estimators (Andersen, Dobrev, Schaumburg 2012).

Implements the MinRV and MedRV truncated realized variance estimators for
jump-robust volatility estimation from high-frequency financial price data.
Both estimators are consistent for integrated variance in the presence of
finite-activity jumps, complementing the bipower variation estimator with
different truncation strategies:

* **MedRV** uses three-point rolling medians of squared (scaled) returns,
  making it robust to isolated jumps affecting at most one return in each
  triple.
* **MinRV** uses two-point rolling minimums of squared (scaled) returns,
  providing an alternative truncation with a different efficiency-robustness
  trade-off.

Both estimators include finite-sample bias corrections and subsampled
variants for microstructure-noise reduction.

Migrated from: ``realized/realized_min_med_variance.m`` (MFE Toolbox v4.0)

See Also
--------
mfe_toolbox.realized.realized_bipower_variation : Bipower variation estimator.
mfe_toolbox.realized.realized_kernel : Kernel-based realized volatility.
mfe_toolbox.realized.realized_quantile_variance : Quantile-based RV.
mfe_toolbox.realized.realized_range : Range-based estimator.
mfe_toolbox.realized.realized_price_filter : Core price filtering function.

Notes
-----
Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 1    Date: 10/24/2011
"""

import warnings

import numpy as np

from mfe_toolbox.realized.realized_convert2unit import realized_convert2unit
from mfe_toolbox.realized.realized_price_filter import realized_price_filter
from mfe_toolbox.realized.realized_subsample import realized_subsample


def realized_min_med_variance(
    price,
    time=None,
    time_type: str = 'unit',
    sampling_type: str = 'CalendarTime',
    sampling_interval=1,
    subsamples: int = 1,
) -> tuple[float, float, float, float, float, float, float, float, dict]:
    """Compute minimum and median realized variance (MinRV and MedRV).

    Estimates truncated realized variance using the median and minimum
    methods of Andersen, Dobrev, and Schaumburg (2012).  Returns base
    estimates, subsampled versions, and finite-sample debiased variants
    for both MedRV and MinRV.

    Parameters
    ----------
    price : array_like
        An m-element 1-D vector of high-frequency prices (raw, **not**
        log prices).  If a 2-D row vector is provided it is automatically
        transposed to a column vector.
    time : array_like or None, optional
        An m-element 1-D vector of observation times where ``time[i]``
        corresponds to ``price[i]``.  Must be sorted in non-decreasing
        order.  When ``None``, evenly spaced unit-interval times
        ``np.linspace(0, 1, len(price))`` are generated and *time_type*
        is forced to ``'unit'``.

        Accepted formats depend on *time_type*:

        * ``'wall'``    — 24-hour clock in HHMMSS format (e.g. 101543).
        * ``'seconds'`` — Seconds past midnight (e.g. 36943).
        * ``'unit'``    — Unit-normalised [0, 1] interval.
    time_type : str, optional
        Time format descriptor.  Case-insensitive.  One of ``'wall'``,
        ``'seconds'``, or ``'unit'``.  Default ``'unit'``.
    sampling_type : str, optional
        Sampling scheme.  Case-insensitive.  One of:

        * ``'CalendarTime'``     — fixed calendar-time interval.
        * ``'CalendarUniform'``  — uniform spacing in calendar time.
        * ``'BusinessTime'``     — every *sampling_interval*-th tick.
        * ``'BusinessUniform'``  — uniform spacing in tick time.
        * ``'Fixed'``            — sample at specific user-supplied times.

        Default ``'CalendarTime'``.
    sampling_interval : int, float, or array_like, optional
        Interpretation depends on *sampling_type*:

        * CalendarTime / CalendarUniform — positive integer (seconds or
          unit fraction for ``'unit'``).
        * BusinessTime / BusinessUniform — positive integer (tick gap).
        * Fixed — sorted, strictly increasing 1-D vector of times in the
          same format as *time*.

        Default ``1``.
    subsamples : int, optional
        Positive integer indicating the number of subsample estimators to
        average.  Subsample estimators are based on prices uniformly
        spaced between the sampling points (calendar) or ticks (business).
        ``subsamples=1`` (default) means no subsampling.

    Returns
    -------
    medrv : float
        Median realized variance estimate from the primary sampling grid.
    medrv_ss : float
        Subsampled median realized variance estimate.  Averages MedRV
        contributions across all subsample grids.
    medrv_debiased : float
        Finite-sample debiased MedRV, equal to
        ``medrv * m / (m - 2)`` where *m* is the number of returns.
        Ref: realized_bipower_variation.m:43-44 — same debiasing pattern
        as BPV ``m / (m - skip - 1)``.
    medrv_ss_debiased : float
        Debiased version of the subsampled MedRV.
    minrv : float
        Minimum realized variance estimate from the primary sampling grid.
    minrv_ss : float
        Subsampled minimum realized variance estimate.
    minrv_debiased : float
        Finite-sample debiased MinRV, equal to
        ``minrv * m / (m - 1)`` where *m* is the number of returns.
    minrv_ss_debiased : float
        Debiased version of the subsampled MinRV.
    diagnostics : dict
        Dictionary containing diagnostic information:

        * ``'num_returns'`` — number of returns from the primary filter.
        * ``'num_med_windows'`` — number of 3-point median windows (m-2).
        * ``'num_min_windows'`` — number of 2-point minimum windows (m-1).
        * ``'med_rv_scale'`` — MedRV scaling constant π/(6−4√3+π).
        * ``'min_rv_scale'`` — MinRV scaling constant π/(π−2).
        * ``'bias_scale_med'`` — finite-sample bias factor (m-2)/m.
        * ``'bias_scale_min'`` — finite-sample bias factor (m-1)/m.
        * ``'subsamples'`` — number of subsamples used.

    Raises
    ------
    ValueError
        If *price* is not a 1-D vector.
        If *time* is not sorted and increasing.
        If *time* length does not match *price* length.
        If *time_type* is not a recognised format.
        If *sampling_type* is not a recognised scheme.
        If *sampling_interval* is invalid for the chosen sampling/time type.
        If *subsamples* is not a non-negative integer.
        If the number of filtered returns is insufficient for MedRV (< 3)
        or MinRV (< 2).

    Examples
    --------
    Using all 5-minute returns on unit interval:

    >>> import numpy as np
    >>> from mfe_toolbox.realized.realized_min_med_variance import (
    ...     realized_min_med_variance,
    ... )
    >>> prices = np.array([100.0, 100.2, 100.5, 100.3, 100.8, 101.0])
    >>> times = np.linspace(0.0, 1.0, len(prices))
    >>> result = realized_min_med_variance(
    ...     prices, times, 'unit', 'CalendarUniform', 3
    ... )
    >>> medrv, medrv_ss, medrv_d, medrv_ss_d, minrv, minrv_ss, minrv_d, minrv_ss_d, diag = result

    Notes
    -----
    **Core algorithm (from realized_min_med_variance.m):**

    1. Convert all times to the unit [0, 1] interval via
       :func:`realized_convert2unit`.
    2. Filter log-prices at the desired sampling grid via
       :func:`realized_price_filter`.
    3. Compute log-returns: ``returns = np.diff(filtered_log_prices)``.
    4. Scale returns: ``returns2 = m * returns**2`` where *m* is the number
       of returns (equivalent to MATLAB ``(sqrt(m)*returns).^2``).
       Ref: realized_min_med_variance.m:154
    5. **MedRV** — three-point rolling median windows:

       .. math::

           MedRV = c_{med} \\cdot \\frac{1}{m-2} \\sum_{i=1}^{m-2}
           \\mathrm{median}(m \\cdot r_i^2, m \\cdot r_{i+1}^2, m \\cdot r_{i+2}^2)

       where :math:`c_{med} = \\pi / (6 - 4\\sqrt{3} + \\pi)`.
       Ref: realized_min_med_variance.m:155-158

    6. **MinRV** — two-point rolling minimum windows:

       .. math::

           MinRV = c_{min} \\cdot \\frac{1}{m-1} \\sum_{i=1}^{m-1}
           \\min(m \\cdot r_i^2, m \\cdot r_{i+1}^2)

       where :math:`c_{min} = \\pi / (\\pi - 2)`.
       Ref: realized_min_med_variance.m:160-163

    7. Subsampled variants average median/minimum values across all shifted
       grids from :func:`realized_subsample`.
       Ref: realized_min_med_variance.m:167-191

    **MATLAB → Python translation notes:**

    * ``(sqrt(m)*returns).^2`` → ``m * returns ** 2``.
      Ref: realized_min_med_variance.m:154 — equivalent algebraic form.
    * MATLAB ``median(r2, 2)`` (row-wise median) → ``np.median(r2, axis=1)``.
    * MATLAB ``min(r2, [], 2)`` (row-wise minimum) → ``np.minimum(r2[:, 0], r2[:, 1])``
      or ``np.min(r2, axis=1)``.
    * MATLAB 1-based indexing → Python 0-based indexing throughout.
    * ``error()`` → ``raise ValueError()``.
    * MATLAB cell array access ``subsampledLogPrices{i}`` →
      Python tuple access ``subsampled_result[i][0]``.
    """
    # ==================================================================
    # Input Validation
    # Ref: realized_min_med_variance.m:68-132
    # ==================================================================

    # --- price ---
    # Ref: realized_min_med_variance.m:71-76
    # Coerce to numpy ndarray with float64 dtype for consistent handling.
    if not isinstance(price, np.ndarray):
        price = np.array(price, dtype=np.float64)
    else:
        price = np.asarray(price, dtype=np.float64)
    if price.ndim == 2:
        # Ref: m:71-72 — if size(price,2)>size(price,1), price=price'
        if price.shape[1] > price.shape[0]:
            price = price.T
        price = price.ravel()
    elif price.ndim > 2:
        raise ValueError('PRICE must be a m by 1 vector.')
    if price.ndim == 0:
        price = price.reshape(1)

    if price.size < 2:
        raise ValueError(
            'PRICE must contain at least 2 observations to compute returns.'
        )

    # --- time ---
    # Ref: realized_min_med_variance.m:77-85
    if time is None:
        # Auto-generate evenly spaced unit times when time is not provided.
        # Matches the bipower_variation Python implementation pattern.
        time = np.linspace(0.0, 1.0, len(price))
        time_type = 'unit'
    else:
        time = np.asarray(time, dtype=np.float64)
        if time.ndim == 2:
            # Ref: m:77-78 — transpose row vector
            if time.shape[1] > time.shape[0]:
                time = time.T
            time = time.ravel()
        elif time.ndim > 2:
            time = time.ravel()
        if time.ndim == 0:
            time = time.reshape(1)

    # Ref: m:80-82 — TIME must be sorted and increasing
    if len(time) > 1 and np.any(np.diff(time) < 0):
        raise ValueError('TIME must be sorted and increasing')

    # Ref: m:83-85 — TIME must have same length as PRICE
    if len(time) != len(price):
        raise ValueError('TIME must be a m by 1 vector.')

    # Ref: m:87 — Cast to double (protect against integer-typed times)
    # Already handled by np.asarray(..., dtype=np.float64) above.

    # --- time_type ---
    # Ref: realized_min_med_variance.m:89-92
    time_type = time_type.lower()
    if time_type not in ('wall', 'seconds', 'unit'):
        raise ValueError(
            "TIMETYPE must be one of 'wall', 'seconds' or 'unit'."
        )

    # --- sampling_type ---
    # Ref: realized_min_med_variance.m:93-96
    sampling_type = sampling_type.lower()
    valid_sampling_types = (
        'calendartime', 'calendaruniform',
        'businesstime', 'businessuniform', 'fixed',
    )
    if sampling_type not in valid_sampling_types:
        raise ValueError(
            "SAMPLINGTYPE must be one of 'CalendarTime', 'CalendarUniform', "
            "'BusinessTime', 'BusinessUniform' or 'Fixed'."
        )

    # --- sampling_interval ---
    # Ref: realized_min_med_variance.m:98-122
    m_price = len(price)
    t0 = time[0]
    tT = time[m_price - 1]

    if sampling_type in ('calendartime', 'calendaruniform',
                         'businesstime', 'businessuniform'):
        # Ref: m:101-111 — scalar validation branching on time_type
        if time_type in ('wall', 'seconds'):
            # Ref: m:103-106 — positive scalar integer required
            if (not np.isscalar(sampling_interval)
                    or float(np.floor(float(sampling_interval)))
                    != float(sampling_interval)
                    or float(sampling_interval) < 1):
                raise ValueError(
                    'SAMPLINGINTERVAL must be a positive integer for the '
                    "SAMPLINGTYPE selected when using 'wall' or 'seconds' "
                    'as TIMETYPE.'
                )
        else:
            # Ref: m:108-110 — positive scalar for unit time type
            if not np.isscalar(sampling_interval) or float(sampling_interval) < 0:
                raise ValueError(
                    'SAMPLINGINTERVAL must be a positive value for the '
                    "SAMPLINGTYPE selected when using 'unit' as TIMETYPE."
                )
    else:
        # Ref: m:113-122 — 'fixed' sampling type: vector validation
        sampling_interval = np.asarray(
            sampling_interval, dtype=np.float64
        ).ravel()
        # Ref: m:113-114 — transpose if row vector (handled by ravel above)

        # Ref: m:116-118
        if not (np.any(sampling_interval >= t0)
                and np.any(sampling_interval <= tT)):
            raise ValueError(
                'At least one sampling interval must be between min(TIME) '
                "and max(TIME) when using 'Fixed' as SAMPLINGTYPE."
            )

        # Ref: m:119-121
        if (len(sampling_interval) > 1
                and np.any(np.diff(sampling_interval) <= 0)):
            raise ValueError(
                "When using 'Fixed' as SAMPLINGTYPE the vector of sampling "
                'times in SAMPLINGINTERVAL must be sorted and strictly '
                'increasing.'
            )

    # --- subsamples ---
    # Ref: realized_min_med_variance.m:124-132
    if subsamples is None:
        subsamples = 1
    subsamples = int(subsamples)
    if subsamples < 0:
        raise ValueError('SUBSAMPLES must be a non-negative scalar.')

    # ==================================================================
    # Convert times to unit [0, 1] interval if not already unit
    # Ref: realized_min_med_variance.m:139-145
    # ==================================================================

    # Ref: m:139 — logPrice = log(price)
    log_price = np.log(price)

    # Ref: m:141-142 — Convert time if not unit
    if time_type != 'unit':
        time, _, _, sampling_interval = realized_convert2unit(
            time, time_type, sampling_type, sampling_interval
        )
    # Ref: m:145 — Set timeType to unit after conversion
    time_type = 'unit'

    # ==================================================================
    # Filter prices and compute returns
    # Ref: realized_min_med_variance.m:149-152
    # ==================================================================

    # Ref: m:149 — filteredLogPrice = realized_price_filter(...)
    filtered_log_price, _filtered_time, _actual_time = realized_price_filter(
        log_price, time, time_type, sampling_type, sampling_interval
    )

    # Ref: m:151 — returns = diff(filteredLogPrice)
    returns = np.diff(filtered_log_price)

    # Ref: m:152 — m = length(returns)
    m = len(returns)

    # Validate sufficient returns for MedRV and MinRV computation
    if m < 3:
        warnings.warn(
            'Fewer than 3 returns available after filtering. '
            'MedRV requires at least 3 returns (for 3-point median windows). '
            'Results may be unreliable or undefined.',
            stacklevel=2,
        )
    if m < 2:
        warnings.warn(
            'Fewer than 2 returns available after filtering. '
            'MinRV requires at least 2 returns (for 2-point minimum windows). '
            'Returning NaN for all estimates.',
            stacklevel=2,
        )
        # Build empty diagnostics and return NaN for all estimates
        diagnostics = {
            'num_returns': int(m),
            'num_med_windows': 0,
            'num_min_windows': 0,
            'med_rv_scale': float(np.pi / (6.0 - 4.0 * np.sqrt(3.0) + np.pi)),
            'min_rv_scale': float(np.pi / (np.pi - 2.0)),
            'bias_scale_med': np.nan,
            'bias_scale_min': np.nan,
            'subsamples': int(subsamples),
        }
        return (
            np.nan, np.nan, np.nan, np.nan,
            np.nan, np.nan, np.nan, np.nan,
            diagnostics,
        )

    # ==================================================================
    # Compute MedRV and MinRV — primary (unshifted) grid
    # Ref: realized_min_med_variance.m:154-163
    # ==================================================================

    # Ref: m:154 — returns2 = (sqrt(m)*returns).^2  ≡  m * returns.^2
    # The sqrt(m) scaling followed by squaring is algebraically identical
    # to multiplying by m.  This normalises each squared return to be
    # approximately O(1) under constant volatility.
    # Use np.abs for clarity: |r|^2 == r^2 for real returns, but makes
    # the truncation intent explicit per the ADS (2012) formulation.
    abs_returns = np.abs(returns)
    returns2 = float(m) * abs_returns ** 2

    # ---- MedRV ----
    # Ref: m:155-158
    # Build (m-2) × 3 matrix of consecutive scaled squared returns,
    # compute row-wise median, then take the mean and scale.
    # Ref: m:155 — r2 = [returns2(1:m-2) returns2(2:m-1) returns2(3:m)]
    # MATLAB 1-based → Python 0-based:
    #   returns2(1:m-2) → returns2[:m-2]
    #   returns2(2:m-1) → returns2[1:m-1]
    #   returns2(3:m)   → returns2[2:m]   = returns2[2:]
    if m >= 3:
        r2_med = np.column_stack([
            returns2[: m - 2],
            returns2[1: m - 1],
            returns2[2:],
        ])
        # Ref: m:156 — medRV = mean(median(r2, 2))
        med_values = np.median(r2_med, axis=1)
        medrv_raw = float(np.mean(med_values))
    else:
        medrv_raw = np.nan

    # Ref: m:157 — medRVScale = pi/(6-4*sqrt(3)+pi)
    # Computed from exact expressions, not hardcoded approximations.
    med_rv_scale = float(np.pi / (6.0 - 4.0 * np.sqrt(3.0) + np.pi))

    # Ref: m:158 — medRV = medRV * medRVScale
    medrv = float(medrv_raw * med_rv_scale)

    # ---- MinRV ----
    # Ref: m:160-163
    # Build (m-1) × 2 matrix (or use np.minimum for efficiency),
    # compute row-wise minimum, then take the mean and scale.
    if m >= 2:
        # Ref: m:160 — r2 = [returns2(1:m-1) returns2(2:m)]
        # Using np.minimum for efficient pairwise min (avoids matrix build)
        min_values = np.minimum(returns2[: m - 1], returns2[1:])
        minrv_raw = float(np.mean(min_values))
    else:
        minrv_raw = np.nan

    # Ref: m:162 — minRVscale = pi/(pi-2)
    min_rv_scale = float(np.pi / (np.pi - 2.0))

    # Ref: m:163 — minRV = minRV * minRVscale
    minrv = float(minrv_raw * min_rv_scale)

    # ==================================================================
    # Finite-sample debiased variants
    # Ref: realized_bipower_variation.m:180-181 — same debiasing pattern:
    #   biasScale = (m-1-skip)/m; bvDebiased = bv / biasScale
    # For MedRV: 3-point windows lose 2 returns → bias_scale = (m-2)/m
    # For MinRV: 2-point windows lose 1 return  → bias_scale = (m-1)/m
    # ==================================================================
    if m >= 3:
        bias_scale_med = float(m - 2) / float(m)
        medrv_debiased = float(medrv / bias_scale_med) if bias_scale_med > 0 else np.nan
    else:
        bias_scale_med = np.nan
        medrv_debiased = np.nan

    if m >= 2:
        bias_scale_min = float(m - 1) / float(m)
        minrv_debiased = float(minrv / bias_scale_min) if bias_scale_min > 0 else np.nan
    else:
        bias_scale_min = np.nan
        minrv_debiased = np.nan

    # ==================================================================
    # Subsampled MedRV and MinRV
    # Ref: realized_min_med_variance.m:167-191
    #
    # Strategy: accumulate all individual median/minimum values from each
    # subsample into flat arrays, then compute a single grand mean and
    # scale.  This differs from BPV subsampling (which computes BPV per
    # subsample then weights by baseCount/totalCount) — here we average
    # the raw window values across all subsamples before scaling.
    # ==================================================================

    # Ref: m:167 — subsampledLogPrices = realized_subsample(...)
    subsampled_result = realized_subsample(
        log_price, time, time_type, sampling_type,
        sampling_interval, subsamples
    )

    # Ref: m:168-171 — Pre-allocate accumulator arrays
    # MATLAB pre-allocates nan(m*subsamples, 1); we use flat pre-allocated
    # numpy arrays with a count index, matching the MATLAB approach closely.
    max_size = m * subsamples  # upper bound on total window values
    all_med_values_arr = np.zeros(max_size, dtype=np.float64)
    all_min_values_arr = np.zeros(max_size, dtype=np.float64)

    med_rv_count = 0
    min_rv_count = 0

    # Ref: m:172-187 — for i = 1:subsamples
    for i in range(subsamples):
        # Ref: m:173 — filteredLogPrice = subsampledLogPrices{i}
        # Python: subsampled_result[i] is a tuple; [0] is the price array
        sub_filtered = subsampled_result[i][0]

        # Ref: m:174 — returns = diff(filteredLogPrice)
        sub_returns = np.diff(sub_filtered)

        # Ref: m:175 — m = size(returns, 1)
        # NOTE: MATLAB reassigns m here; we use m_sub to avoid shadowing
        m_sub = len(sub_returns)

        if m_sub < 2:
            # Not enough returns for either estimator in this subsample
            continue

        # Ref: m:176 — returns2 = m * returns.^2
        # Ref: realized_min_med_variance.m:176 — uses local m (m_sub)
        sub_returns2 = float(m_sub) * sub_returns ** 2

        # ---- MedRV contribution ----
        # Ref: m:178-181
        if m_sub >= 3:
            # Ref: m:178 — r2 = [returns2(1:m-2) returns2(2:m-1) returns2(3:m)]
            sub_r2_med = np.column_stack([
                sub_returns2[: m_sub - 2],
                sub_returns2[1: m_sub - 1],
                sub_returns2[2:],
            ])
            # Ref: m:180 — medRVs(medRVcount+(1:n)) = median(r2, 2)
            sub_med = np.median(sub_r2_med, axis=1)
            n_med = len(sub_med)
            # Store in pre-allocated array, extending if needed
            if med_rv_count + n_med > len(all_med_values_arr):
                all_med_values_arr = np.array(
                    np.concatenate([all_med_values_arr,
                                    np.zeros(n_med, dtype=np.float64)]),
                    dtype=np.float64,
                )
            all_med_values_arr[med_rv_count: med_rv_count + n_med] = sub_med
            # Ref: m:181 — medRVcount = medRVcount + n
            med_rv_count += n_med

        # ---- MinRV contribution ----
        # Ref: m:183-186
        if m_sub >= 2:
            # Ref: m:183 — r2 = [returns2(1:m-1) returns2(2:m)]
            sub_min = np.minimum(sub_returns2[: m_sub - 1], sub_returns2[1:])
            n_min = len(sub_min)
            # Store in pre-allocated array, extending if needed
            if min_rv_count + n_min > len(all_min_values_arr):
                all_min_values_arr = np.array(
                    np.concatenate([all_min_values_arr,
                                    np.zeros(n_min, dtype=np.float64)]),
                    dtype=np.float64,
                )
            all_min_values_arr[min_rv_count: min_rv_count + n_min] = sub_min
            # Ref: m:186 — minRVcount = minRVcount + n
            min_rv_count += n_min

    # Ref: m:188-189 — medRVSS = mean(medRVs(1:medRVcount)) * medRVScale
    if med_rv_count > 0:
        medrv_ss = float(np.mean(all_med_values_arr[:med_rv_count]) * med_rv_scale)
    else:
        medrv_ss = np.nan

    # Ref: m:190-191 — minRVSS = mean(minRVs(1:minRVcount)) * minRVscale
    if min_rv_count > 0:
        minrv_ss = float(np.mean(all_min_values_arr[:min_rv_count]) * min_rv_scale)
    else:
        minrv_ss = np.nan

    # ==================================================================
    # Debiased subsampled variants
    # Ref: realized_bipower_variation.m:196 — bvSSDebiased = bvSS / biasScale
    # Uses the same bias_scale computed from the primary (first) filter pass,
    # consistent with the BPV pattern.
    # ==================================================================
    if not np.isnan(bias_scale_med) and bias_scale_med > 0:
        medrv_ss_debiased = float(medrv_ss / bias_scale_med) if not np.isnan(medrv_ss) else np.nan
    else:
        medrv_ss_debiased = np.nan

    if not np.isnan(bias_scale_min) and bias_scale_min > 0:
        minrv_ss_debiased = float(minrv_ss / bias_scale_min) if not np.isnan(minrv_ss) else np.nan
    else:
        minrv_ss_debiased = np.nan

    # ==================================================================
    # Build diagnostics dictionary
    # ==================================================================
    diagnostics = {
        'num_returns': int(m),
        'num_med_windows': int(m - 2) if m >= 3 else 0,
        'num_min_windows': int(m - 1) if m >= 2 else 0,
        'med_rv_scale': float(med_rv_scale),
        'min_rv_scale': float(min_rv_scale),
        'bias_scale_med': float(bias_scale_med) if not np.isnan(bias_scale_med) else np.nan,
        'bias_scale_min': float(bias_scale_min) if not np.isnan(bias_scale_min) else np.nan,
        'subsamples': int(subsamples),
    }

    return (
        medrv, medrv_ss, medrv_debiased, medrv_ss_debiased,
        minrv, minrv_ss, minrv_debiased, minrv_ss_debiased,
        diagnostics,
    )
