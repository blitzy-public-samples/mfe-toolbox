"""
Realized quarticity estimator with BNS, tripower, and quadpower variants.

This module provides the :func:`realized_quarticity` function, which computes
realized quarticity — a measure of the variation of variation (integrated
quarticity) — using three estimator types:

* **BNS** (Barndorff-Nielsen & Shephard): standard realized quarticity based
  on the fourth power of returns, :math:`QT = (n/3) \\sum r_i^4`.  Not robust
  to jumps.
* **Tripower**: jump-robust estimator using three adjacent returns raised to
  the 4/3 power, separated by *skip* + 1 positions.
* **Quadpower**: jump-robust estimator using four adjacent absolute returns,
  separated by *skip* + 1 positions.

Skip-*k* variants allow separation of products by *k* + 1 positions to
attenuate the effects of microstructure noise.  Debiased versions apply a
finite-sample correction equal to *m* / (*m* − *k_terms* × *skip* − *k_terms*)
where *k_terms* is the number of return terms in the product (0 for BNS,
2 for Tripower, 3 for Quadpower).

Subsampled quarticity is computed by averaging quarticity estimates over
multiple shifted price grids for bias reduction.

Migrated from: ``realized/realized_quarticity.m`` (MFE Toolbox Version 4.0)

See Also
--------
mfe_toolbox.realized.realized_variance : Realized variance estimator.
mfe_toolbox.realized.realized_kernel : Realized kernel estimator.
mfe_toolbox.realized.realized_quantile_variance : Quantile-based RV.
mfe_toolbox.realized.realized_range : Range-based estimator.
mfe_toolbox.realized.realized_bipower_variation : Bipower variation estimator.
mfe_toolbox.realized.realized_semivariance : Realized semivariance.

Notes
-----
Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 1    Date: 5/1/2008
"""

import warnings

import numpy as np
from scipy import special

from mfe_toolbox.realized.realized_convert2unit import realized_convert2unit
from mfe_toolbox.realized.realized_price_filter import realized_price_filter
from mfe_toolbox.realized.realized_subsample import realized_subsample


# ---------------------------------------------------------------------------
# Internal helper: core quarticity computation on a return vector
# Ref: realized_quarticity.m:210-234 — nested function realized_quarticity_core
# ---------------------------------------------------------------------------


def _realized_quarticity_core(
    returns: np.ndarray,
    qt_type: str,
    skip: int,
) -> tuple[float, int]:
    """Compute quarticity for a single return vector.

    Parameters
    ----------
    returns : np.ndarray
        1-D array of log returns.
    qt_type : str
        One of ``'bns'``, ``'tripower'``, ``'quadpower'`` (lower-case).
    skip : int
        Non-negative integer for skip-k separation.

    Returns
    -------
    qt : float
        Quarticity estimate.
    count : int
        Number of return products used (for debiasing).

    Raises
    ------
    ValueError
        If *skip* is too large relative to the number of returns.

    Notes
    -----
    Ref: realized_quarticity.m:210-234 — MATLAB nested function
    ``realized_quarticity_core``.

    MATLAB source line 212 contains a variable-name bug: the guard
    condition references ``QTType`` (capital T) instead of ``QTtype``
    (lowercase t), causing an undefined-variable runtime error in MATLAB.
    This Python version uses the corrected variable name (``qt_type``)
    and applies the intended tripower/quadpower length checks.
    """
    m = len(returns)

    # Ref: realized_quarticity.m:212-213 — Guard against skip being too large.
    # Original MATLAB has a variable-name typo (QTType vs QTtype); this
    # implementation corrects it and applies the proper check for each type.
    if qt_type == 'tripower' and (m - 2 - 2 * skip) < 1:
        raise ValueError(
            'The value of SKIP is too large for the sampling method used. '
            'SKIP should be small relative to the number of prices computed '
            'using the chosen sampling method.'
        )
    if qt_type == 'quadpower' and (m - 3 - 3 * skip) < 1:
        raise ValueError(
            'The value of SKIP is too large for the sampling method used. '
            'SKIP should be small relative to the number of prices computed '
            'using the chosen sampling method.'
        )

    if qt_type == 'bns':
        # Ref: realized_quarticity.m:216-219
        # BNS realized quarticity: QT = (1/3) * m * sum(r^4)
        # constant = 3, so qt = constant^{-1} * m * sum(r^4)
        constant = 3.0
        qt = (1.0 / constant) * m * float(np.sum(returns ** 4))
        count = len(returns)

    elif qt_type == 'tripower':
        # Ref: realized_quarticity.m:220-226
        # mu_{4/3} = 2^{2/3} * gamma(7/6) / gamma(1/2)
        # Tripower quarticity uses products of |r_i|^{4/3} over 3 terms
        # separated by (skip+1) positions.
        constant = (2.0 ** (2.0 / 3.0)) * special.gamma(7.0 / 6.0) / special.gamma(0.5)

        # Ref: realized_quarticity.m:222-225 — MATLAB 1-based indexing:
        #   returns(1 : m-2-2*skip)       -> Python returns[0 : m-2-2*skip]
        #   returns(2+skip : m-1-skip)     -> Python returns[1+skip : m-1-skip]
        #   returns(3+2*skip : m)          -> Python returns[2+2*skip : m]
        end1 = m - 2 - 2 * skip  # Ref: m:223 — upper bound of first slice
        start2 = 1 + skip        # Ref: m:224 — start of second slice (0-based)
        end2 = m - 1 - skip      # Ref: m:224 — upper bound of second slice
        start3 = 2 + 2 * skip    # Ref: m:225 — start of third slice (0-based)

        r1 = np.abs(returns[0:end1]) ** (4.0 / 3.0)
        r2 = np.abs(returns[start2:end2]) ** (4.0 / 3.0)
        r3 = np.abs(returns[start3:m]) ** (4.0 / 3.0)

        qt = float((constant ** (-3)) * m * np.sum(r1 * r2 * r3))
        count = m - 2 - 2 * skip

    elif qt_type == 'quadpower':
        # Ref: realized_quarticity.m:227-234
        # mu_1 = sqrt(2/pi)
        # Quadpower quarticity uses products of |r_i| over 4 terms
        # separated by (skip+1) positions.
        constant = np.sqrt(2.0 / np.pi)

        # Ref: realized_quarticity.m:229-233 — MATLAB 1-based indexing:
        #   returns(1 : m-3-3*skip)       -> Python returns[0 : m-3-3*skip]
        #   returns(2+skip : m-2-2*skip)   -> Python returns[1+skip : m-2-2*skip]
        #   returns(3+2*skip : m-1-skip)   -> Python returns[2+2*skip : m-1-skip]
        #   returns(4+3*skip : m)          -> Python returns[3+3*skip : m]
        end1 = m - 3 - 3 * skip
        start2 = 1 + skip
        end2 = m - 2 - 2 * skip
        start3 = 2 + 2 * skip
        end3 = m - 1 - skip
        start4 = 3 + 3 * skip

        r1 = np.abs(returns[0:end1])
        r2 = np.abs(returns[start2:end2])
        r3 = np.abs(returns[start3:end3])
        r4 = np.abs(returns[start4:m])

        qt = float((constant ** (-4)) * m * np.sum(r1 * r2 * r3 * r4))
        count = m - 3 - 3 * skip

    else:
        # Should never reach here due to validation in the main function
        raise ValueError(
            "QTTYPE must be 'BNS', 'Tripower' or 'Quadpower'."
        )

    return qt, count


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def realized_quarticity(
    price,
    time=None,
    time_type: str = 'unit',
    sampling_type: str = 'CalendarTime',
    sampling_interval=1,
    skip: int = 0,
    quarticity_type: str = 'BNS',
    subsamples: int = 1,
) -> tuple[float, float, float, float, dict]:
    """Estimate realized quarticity and subsampled/debiased variants.

    Computes realized quarticity — a measure of integrated quarticity —
    using one of three estimator types (BNS, Tripower, Quadpower) on
    a filtered price grid.  Optionally computes subsampled and debiased
    variants.

    Parameters
    ----------
    price : array_like
        An m-element 1-D vector of high-frequency prices.  If a row
        vector is provided it is transposed.  Must contain at least 2
        positive values.
    time : array_like or None, optional
        An m-element 1-D vector of observation times corresponding to
        *price*.  Must be sorted in non-decreasing order.  The format
        must match *time_type*:

        * ``'wall'``    — 24-hour clock in HHMMSS format (e.g. 93000).
        * ``'seconds'`` — Seconds past midnight (e.g. 34200).
        * ``'unit'``    — Unit-normalised on [0, 1].

        If ``None`` (default), a uniformly spaced grid on [0, 1] is
        generated and *time_type* is set to ``'unit'``.
    time_type : str, optional
        Time format descriptor.  Case-insensitive.  One of ``'wall'``,
        ``'seconds'``, or ``'unit'``.  Default ``'unit'``.
    sampling_type : str, optional
        Sampling scheme.  Case-insensitive.  One of:

        * ``'CalendarTime'``     — Fixed calendar-time interval.
        * ``'CalendarUniform'``  — Uniform spacing in calendar time.
        * ``'BusinessTime'``     — Every N-th tick.
        * ``'BusinessUniform'``  — Uniform spacing in tick time.
        * ``'Fixed'``            — Sample at user-supplied times.

        Default ``'CalendarTime'``.
    sampling_interval : int, float, or array_like, optional
        Interpretation depends on *sampling_type*.  See
        :func:`realized_price_filter` for full details.  Default 1.
    skip : int, optional
        Non-negative integer indicating the number of returns to skip
        when computing the quarticity product.  The skip-*k* tripower
        quarticity is:
        ``|r_m|^{4/3} |r_{m-(k+1)}|^{4/3} |r_{m-2(k+1)}|^{4/3}``
        When ``skip=0`` (default) the usual estimator is produced.
        For BNS, *skip* is ignored.
    quarticity_type : str, optional
        Estimator variant.  Case-insensitive.  One of:

        * ``'BNS'`` (default) — Barndorff-Nielsen & Shephard standard
          quarticity (not robust to jumps).
        * ``'Tripower'`` — Jump-robust tripower quarticity using three
          adjacent returns raised to the 4/3 power.
        * ``'Quadpower'`` — Jump-robust quadpower quarticity using four
          adjacent absolute returns.
    subsamples : int, optional
        Number of subsample estimates to average.  ``1`` (default) uses
        a single grid.  Values > 1 compute shifted grids for bias
        reduction.

    Returns
    -------
    tuple[float, float, float, float, dict]
        A 5-tuple ``(qt, qt_ss, qt_debiased, qt_ss_debiased, diagnostics)``
        where:

        * **qt** — Realized quarticity estimate.
        * **qt_ss** — Subsampled realized quarticity (weighted average
          over shifted grids).  Equals *qt* when *subsamples* = 1.
        * **qt_debiased** — Debiased quarticity:
          ``qt * m / (m - k_terms * skip - k_terms)`` where *k_terms*
          is 0 (BNS), 2 (Tripower), or 3 (Quadpower).
        * **qt_ss_debiased** — Debiased subsampled quarticity.
        * **diagnostics** — Dictionary with estimation details:

          - ``'num_prices'``           : int — Original price vector length.
          - ``'num_filtered_prices'``  : int — Filtered prices count.
          - ``'num_returns'``          : int — Number of log returns.
          - ``'qt'``                   : float — Standard QT.
          - ``'qt_ss'``               : float — Subsampled QT.
          - ``'qt_debiased'``         : float — Debiased QT.
          - ``'qt_ss_debiased'``      : float — Debiased subsampled QT.
          - ``'quarticity_type'``     : str — Estimator variant used.
          - ``'skip'``                : int — Skip parameter used.
          - ``'subsamples'``          : int — Number of subsamples.
          - ``'sampling_type'``       : str — Sampling scheme.
          - ``'sampling_interval'``   : Sampling interval value.
          - ``'time_type'``           : str — Time format used.
          - ``'bias_scale'``          : float — Ratio count/m for debiasing.
          - ``'base_count'``          : int — Returns in first subsample.
          - ``'total_count'``         : int — Total returns across subsamples.

    Raises
    ------
    ValueError
        If *price* is not a 1-D vector with at least 2 elements.
        If *time* is not sorted and increasing.
        If *time* length does not match *price* length.
        If *time_type* is not a recognized format.
        If *sampling_type* is not a recognized scheme.
        If *sampling_interval* is invalid for the given scheme.
        If *quarticity_type* is not ``'BNS'``, ``'Tripower'``, or
        ``'Quadpower'``.
        If *skip* is not a non-negative integer.
        If *subsamples* is not a positive integer.
        If *skip* is too large relative to the number of filtered returns.

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.realized.realized_quarticity import realized_quarticity
    >>> prices = np.array([100.0, 100.5, 101.0, 100.8, 101.2, 101.5])
    >>> times = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    >>> qt, qt_ss, qt_d, qt_ss_d, diag = realized_quarticity(
    ...     prices, times, 'unit', 'CalendarTime', 0.5)
    >>> qt > 0
    True

    Notes
    -----
    **Core algorithm (from realized_quarticity.m):**

    1. Compute log prices: ``logPrice = log(price)``
    2. Filter log prices via :func:`realized_price_filter`.
    3. Compute log returns: ``returns = diff(filteredLogPrice)``
    4. Compute quarticity on filtered returns using the selected type.
    5. Compute debiased quarticity: ``qt_debiased = qt / bias_scale``
       where ``bias_scale = count / len(returns)``.
    6. For subsampled quarticity:

       a. Generate shifted grids via :func:`realized_subsample`.
       b. Compute quarticity on each shifted grid.
       c. Weight: ``qt_ss = sum(qts) * base_count / total_count``
       d. ``qt_ss_debiased = qt_ss / bias_scale``

    **Constants:**

    * BNS: ``constant = 3``, ``QT = (1/3) * m * sum(r^4)``
    * Tripower: ``mu_{4/3} = 2^{2/3} * gamma(7/6) / gamma(1/2)``
    * Quadpower: ``mu_1 = sqrt(2/pi) = 2^{1/2} * gamma(1) / gamma(1/2)``

    **MATLAB → Python translation notes:**

    * ``gamma(x)`` → ``scipy.special.gamma(x)``
    * MATLAB 1-based indexing → Python 0-based indexing for return slices.
    * ``error()`` → ``raise ValueError()``
    * ``warning()`` → ``warnings.warn()``
    * Ref: realized_quarticity.m:221 — tripower constant uses
      ``gamma(7/6)`` and ``gamma(1/2)`` directly.
    * Ref: realized_quarticity.m:228 — quadpower constant ``sqrt(2/pi)``.
    """
    # ==================================================================
    # Input Validation
    # Ref: realized_quarticity.m:88-171
    # ==================================================================

    # Ref: realized_quarticity.m:91-93 — Transpose row to column
    price = np.asarray(price, dtype=np.float64).ravel()
    m_orig = len(price)

    if m_orig < 2:
        raise ValueError('PRICE must have at least 2 elements.')

    # Ref: realized_quarticity.m:94-95 — PRICE must be a vector
    # (Handled by ravel above)

    # Default time array when time is not supplied
    # Ref: consistent pattern with realized_variance.py
    if time is None:
        time = np.linspace(0.0, 1.0, m_orig)
        time_type = 'unit'

    # Ref: realized_quarticity.m:97-99 — Transpose time to column
    time = np.asarray(time, dtype=np.float64).ravel()

    # Ref: realized_quarticity.m:100-101 — TIME must be sorted and increasing
    if len(time) > 1 and np.any(np.diff(time) < 0):
        raise ValueError('TIME must be sorted and increasing')

    # Ref: realized_quarticity.m:103-104 — TIME length must match PRICE
    if len(time) != m_orig:
        raise ValueError('TIME must be a m by 1 vector.')

    # Ref: realized_quarticity.m:107-109 — Cast time to double
    # Already handled by np.asarray(..., dtype=np.float64) above.

    # Ref: realized_quarticity.m:112-114 — Validate time_type
    time_type = time_type.lower()
    if time_type not in ('wall', 'seconds', 'unit'):
        raise ValueError(
            "TIMETYPE must be one of 'wall', 'seconds' or 'unit'."
        )

    # Ref: realized_quarticity.m:116-118 — Validate sampling_type
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

    # Ref: realized_quarticity.m:121-123 — Compute t0, tT for validation
    t0 = time[0]
    tT = time[-1]

    # Ref: realized_quarticity.m:124-145 — Validate samplingInterval
    if sampling_type_lower in ('calendartime', 'calendaruniform',
                               'businesstime', 'businessuniform'):
        if time_type in ('wall', 'seconds'):
            # Ref: realized_quarticity.m:127-128 — positive scalar integer
            if (not np.isscalar(sampling_interval)
                    or np.floor(float(sampling_interval)) != float(sampling_interval)
                    or float(sampling_interval) < 1):
                raise ValueError(
                    'SAMPLINGINTERVAL must be a positive integer for '
                    'the SAMPLINGTYPE selected when using '
                    "'wall' or 'seconds' as TIMETYPE."
                )
        else:
            # Ref: realized_quarticity.m:131-132 — positive scalar for unit
            if not np.isscalar(sampling_interval) or float(sampling_interval) < 0:
                raise ValueError(
                    'SAMPLINGINTERVAL must be a positive value for '
                    'the SAMPLINGTYPE selected when using '
                    "'unit' as TIMETYPE."
                )
    else:
        # Ref: realized_quarticity.m:136-144 — Fixed sampling type
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

    # Ref: realized_quarticity.m:147-155 — Validate QTtype
    quarticity_type_lower = quarticity_type.lower()
    if quarticity_type_lower not in ('bns', 'tripower', 'quadpower'):
        raise ValueError(
            "QTTYPE must be 'BNS', 'Tripower' or 'Quadpower'."
        )

    # Ref: realized_quarticity.m:157-163 — Validate skip
    if not np.isscalar(skip):
        raise ValueError('SKIP must be a non-negative scalar.')
    skip = int(skip)
    if skip < 0 or np.floor(float(skip)) != float(skip):
        raise ValueError('SKIP must be a non-negative scalar.')

    # Ref: realized_quarticity.m:165-171 — Validate subsamples
    if subsamples is None:
        subsamples = 1
    subsamples = int(subsamples)
    if subsamples < 1:
        raise ValueError('SUBSAMPLES must be a non-negative scalar.')

    # ==================================================================
    # Core Computation
    # Ref: realized_quarticity.m:176-206
    # ==================================================================

    # Ref: realized_quarticity.m:177 — logPrice = log(price)
    log_price = np.log(price)

    # Ref: realized_quarticity.m:178 — Filter log prices
    # realized_price_filter returns (filtered_price, filtered_time, actual_time).
    # We need only the filtered prices.
    filter_result = realized_price_filter(
        log_price, time, time_type, sampling_type_lower, sampling_interval
    )
    if isinstance(filter_result, tuple):
        filtered_log_price = np.asarray(
            filter_result[0], dtype=np.float64
        ).ravel()
    else:
        filtered_log_price = np.asarray(
            filter_result, dtype=np.float64
        ).ravel()

    # Ref: realized_quarticity.m:179 — returns = diff(filteredLogPrice)
    returns = np.diff(filtered_log_price)
    m = len(returns)

    # Ref: realized_quarticity.m:181-186 — Check skip validity against returns
    if (m - 1 - skip) < 1 or (2 + skip) > m:
        raise ValueError(
            'The value of SKIP is too large for the sampling method used. '
            'SKIP should be small relative to the number of prices computed '
            'using the chosen sampling method.'
        )
    elif (skip / m) > 0.1:
        # Ref: realized_quarticity.m:183-186 — Warning for large skip ratio
        warnings.warn(
            'The value chosen for skip is large relative to the number of '
            'prices returned for the chosen SAMPLINGTYPE and '
            'SAMPLINGINTERVAL - (skip / #returns) > .1 indicating more '
            'than 10% of the returns are being dropped. Interpret the '
            'results with caution.',
            stacklevel=2,
        )

    # Ref: realized_quarticity.m:189 — Compute quarticity on filtered returns
    qt, count = _realized_quarticity_core(returns, quarticity_type_lower, skip)

    # Ref: realized_quarticity.m:190-191 — Debiased quarticity
    # bias_scale = count / length(returns), so qt_debiased = qt / bias_scale
    bias_scale = count / len(returns)
    qt_debiased = qt / bias_scale if bias_scale > 0 else qt

    # ==================================================================
    # Subsampled Quarticity
    # Ref: realized_quarticity.m:193-206
    # ==================================================================

    # Ref: realized_quarticity.m:193 — Generate shifted price grids
    subsample_results = realized_subsample(
        log_price, time, time_type, sampling_type_lower,
        sampling_interval, subsamples,
    )

    # Ref: realized_quarticity.m:194-203 — Compute QT on each subsample
    qts = np.zeros(subsamples)
    total_count = 0
    base_count = 0
    for i in range(subsamples):
        # realized_subsample returns list of tuples:
        # (subsampled_prices, subsampled_times, base_count_i, total_count_i)
        # Ref: realized_quarticity.m:197 — returns = diff(subsampledLogPrices{i})
        sub_log_prices = np.asarray(
            subsample_results[i][0], dtype=np.float64
        ).ravel()
        sub_returns = np.diff(sub_log_prices)

        # Ref: realized_quarticity.m:198 — [qts(i),count] = realized_quarticity_core(...)
        qts[i], sub_count = _realized_quarticity_core(
            sub_returns, quarticity_type_lower, skip
        )

        # Ref: realized_quarticity.m:199-200 — Track base count from first subsample
        if i == 0:
            base_count = sub_count

        # Ref: realized_quarticity.m:202 — totalCount = totalCount + count
        total_count += sub_count

    # Ref: realized_quarticity.m:205 — Weighted average:
    # qtSS = sum(qts) * baseCount / totalCount
    if total_count > 0:
        qt_ss = float(np.sum(qts) * base_count / total_count)
    else:
        # Fallback: if no returns in subsamples, use main quarticity
        qt_ss = qt
        warnings.warn(
            'Subsampled realized quarticity has zero total returns; '
            'falling back to standard quarticity.',
            stacklevel=2,
        )

    # Ref: realized_quarticity.m:206 — qtSSDebiased = qtSS / biasScale
    qt_ss_debiased = qt_ss / bias_scale if bias_scale > 0 else qt_ss

    # ==================================================================
    # Diagnostics
    # ==================================================================
    diagnostics = {
        'num_prices': m_orig,
        'num_filtered_prices': len(filtered_log_price),
        'num_returns': m,
        'qt': qt,
        'qt_ss': qt_ss,
        'qt_debiased': qt_debiased,
        'qt_ss_debiased': qt_ss_debiased,
        'quarticity_type': quarticity_type_lower,
        'skip': skip,
        'subsamples': subsamples,
        'sampling_type': sampling_type_lower,
        'sampling_interval': sampling_interval,
        'time_type': time_type,
        'bias_scale': bias_scale,
        'base_count': base_count,
        'total_count': total_count,
    }

    return qt, qt_ss, qt_debiased, qt_ss_debiased, diagnostics
