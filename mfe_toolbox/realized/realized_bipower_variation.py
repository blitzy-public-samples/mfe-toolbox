"""
Bipower variation estimator (Barndorff-Nielsen & Shephard, 2004).

Computes bipower variation (BPV), skip-k bipower variation, and subsample
versions of these for jump-robust volatility estimation from high-frequency
financial price data.  The bipower variation converges to the integrated
variance in the presence of finite-activity jumps, making it a core tool
for separating continuous and jump components in asset price dynamics.

The skip-k extension replaces adjacent return products with returns separated
by k+1 positions, reducing the microstructure-noise-induced bias at the cost
of (slightly) fewer product terms.  The finite-sample bias correction factor
``m / (m - k - 1)`` compensates for the reduced sample.

Subsampled bipower variation averages BPV estimates from multiple shifted
sampling grids, reducing finite-sample bias analogous to the Zhang, Mykland
& Aït-Sahalia (2005) subsampled realized variance.

Migrated from: ``realized/realized_bipower_variation.m`` (MFE Toolbox v4.0)

See Also
--------
mfe_toolbox.realized.realized_variance : Standard realized variance estimator.
mfe_toolbox.realized.realized_quarticity : Realized quarticity estimator.
mfe_toolbox.realized.realized_kernel : Kernel-based realized volatility.
mfe_toolbox.realized.realized_quantile_variance : Quantile-based RV.
mfe_toolbox.realized.realized_range : Range-based estimator.
mfe_toolbox.realized.realized_price_filter : Core price filtering function.
mfe_toolbox.realized.realized_kernel_weights : Kernel weight functions.

Notes
-----
Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 1    Date: 5/1/2008
"""

import warnings

import numpy as np

from mfe_toolbox.realized.realized_convert2unit import realized_convert2unit
from mfe_toolbox.realized.realized_price_filter import realized_price_filter
from mfe_toolbox.realized.realized_subsample import realized_subsample


def realized_bipower_variation(
    price,
    time=None,
    time_type: str = 'unit',
    sampling_type: str = 'CalendarTime',
    sampling_interval=1,
    skip: int = 0,
    subsamples: int = 1,
) -> tuple[float, float, float, float, dict]:
    """Compute bipower variation (BPV), skip-k BPV, and subsample versions.

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
    skip : int, optional
        Non-negative integer indicating the number of returns to skip when
        computing BPV.  The skip-k BPV uses products
        ``|r_i| * |r_{i+k+1}|``, so ``skip=0`` (default) produces the
        standard BPV estimator.
    subsamples : int, optional
        Positive integer indicating the number of subsample BPV estimators
        to average.  Subsample BPV is based on prices uniformly spaced
        between the sampling points (calendar) or ticks (business).
        ``subsamples=1`` (default) means no subsampling.

    Returns
    -------
    bv : float
        Bipower variation estimate from the primary (unshifted) sampling
        grid.
    bv_ss : float
        Bipower variation including subsample averaging.  Equal to *bv*
        when ``subsamples=1``.
    bv_debiased : float
        Finite-sample debiased BPV, equal to
        ``bv * m / (m - skip - 1)`` where *m* is the number of returns.
    bv_ss_debiased : float
        Debiased version of the subsampled bipower variation.
    diagnostics : dict
        Dictionary containing diagnostic information:

        * ``'num_returns'`` — number of returns from the primary filter.
        * ``'num_products'`` — number of product terms used (m - 1 - skip).
        * ``'mu_1'`` — the constant ``sqrt(2 / pi)``.
        * ``'bias_scale'`` — finite-sample bias factor ``(m - 1 - skip) / m``.
        * ``'skip'`` — skip value used.
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
        If *skip* is not a non-negative integer.
        If *subsamples* is not a positive integer.
        If *skip* is too large relative to the number of filtered returns.

    Warns
    -----
    UserWarning
        When ``skip / num_returns > 0.1``, indicating that more than 10%
        of returns are dropped.

    Examples
    --------
    5-minute bipower variation using fixed wall-time intervals:

    >>> import numpy as np
    >>> from mfe_toolbox.realized.realized_bipower_variation import (
    ...     realized_bipower_variation,
    ... )
    >>> prices = np.array([100.0, 100.2, 100.5, 100.3, 100.8, 101.0])
    >>> times = np.linspace(0.0, 1.0, len(prices))
    >>> bv, bv_ss, bv_d, bv_ss_d, diag = realized_bipower_variation(
    ...     prices, times, 'unit', 'CalendarUniform', 3
    ... )

    Skip-1 bipower variation:

    >>> bv, bv_ss, bv_d, bv_ss_d, diag = realized_bipower_variation(
    ...     prices, times, 'unit', 'CalendarUniform', 3, skip=1
    ... )

    Notes
    -----
    **Core algorithm (from realized_bipower_variation.m):**

    1. Convert all times to the unit [0, 1] interval via
       :func:`realized_convert2unit`.
    2. Filter log-prices at the desired sampling grid via
       :func:`realized_price_filter`.
    3. Compute log-returns: ``returns = np.diff(filtered_log_prices)``.
    4. Standard BPV:

       .. math::

           BPV = \\frac{\\pi}{2} \\sum_{i=0}^{m-2-k} |r_i| \\, |r_{i+k+1}|

       where *m* = number of returns, *k* = skip.

    5. Finite-sample bias correction:

       .. math::

           BPV_{debiased} = BPV \\times \\frac{m}{m - k - 1}

    6. Subsampled BPV averages BPV across shifted grids from
       :func:`realized_subsample`, weighted by the number of product
       terms in each subsample relative to the first subsample.

    **MATLAB → Python translation notes:**

    * ``abs(returns(1:m-1-skip))' * abs(returns(2+skip:m))`` →
      ``np.sum(np.abs(returns[:m-1-skip]) * np.abs(returns[1+skip:]))``.
      Ref: realized_bipower_variation.m:179 — MATLAB 1-based indexing
      converted to Python 0-based.
    * ``mu1 = sqrt(2/pi)`` → ``mu_1 = np.sqrt(2.0 / np.pi)``.
    * ``error()`` → ``raise ValueError()``.
    * ``warning()`` → ``warnings.warn()``.
    * MATLAB cell array access ``subsampledLogPrices{i}`` →
      Python tuple access ``subsampled_result[i][0]``.
    """
    # ==================================================================
    # Input Validation
    # Ref: realized_bipower_variation.m:79-152
    # ==================================================================

    # --- price ---
    # Ref: realized_bipower_variation.m:82-87 — transpose row, error if matrix
    price = np.asarray(price, dtype=np.float64)
    if price.ndim == 2:
        # Ref: m:82-83 — if size(price,2)>size(price,1), price=price'
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
    # Ref: realized_bipower_variation.m:88-101
    if time is None:
        # Auto-generate evenly spaced unit times when time is not provided
        time = np.linspace(0.0, 1.0, len(price))
        time_type = 'unit'
    else:
        time = np.asarray(time, dtype=np.float64)
        if time.ndim == 2:
            # Ref: m:88-89 — transpose row vector
            if time.shape[1] > time.shape[0]:
                time = time.T
            time = time.ravel()
        elif time.ndim > 2:
            time = time.ravel()
        if time.ndim == 0:
            time = time.reshape(1)

    # Ref: m:91-92 — TIME must be sorted and increasing
    if len(time) > 1 and np.any(np.diff(time) < 0):
        raise ValueError('TIME must be sorted and increasing')

    # Ref: m:94-96 — TIME must have same length as PRICE
    if len(time) != len(price):
        raise ValueError('TIME must be a m by 1 vector.')

    # Ref: m:98-101 — Cast to double (protect against integer-typed times)
    # Already handled by np.asarray(..., dtype=np.float64) above.

    # --- time_type ---
    # Ref: realized_bipower_variation.m:103-106
    time_type = time_type.lower()
    if time_type not in ('wall', 'seconds', 'unit'):
        raise ValueError(
            "TIMETYPE must be one of 'wall', 'seconds' or 'unit'."
        )

    # --- sampling_type ---
    # Ref: realized_bipower_variation.m:107-110
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
    # Ref: realized_bipower_variation.m:112-136
    m_price = len(price)
    t0 = time[0]
    tT = time[m_price - 1]

    if sampling_type in ('calendartime', 'calendaruniform',
                         'businesstime', 'businessuniform'):
        # Ref: m:115-125 — scalar validation branching on time_type
        if time_type in ('wall', 'seconds'):
            # Ref: m:118-120 — positive scalar integer required
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
            # Ref: m:122-124 — positive scalar for unit time type
            if not np.isscalar(sampling_interval) or float(sampling_interval) < 0:
                raise ValueError(
                    'SAMPLINGINTERVAL must be a positive value for the '
                    "SAMPLINGTYPE selected when using 'unit' as TIMETYPE."
                )
    else:
        # Ref: m:127-136 — 'fixed' sampling type: vector validation
        sampling_interval = np.asarray(
            sampling_interval, dtype=np.float64
        ).ravel()
        if not (np.any(sampling_interval >= t0)
                and np.any(sampling_interval <= tT)):
            raise ValueError(
                'At least one sampling interval must be between min(TIME) '
                "and max(TIME) when using 'Fixed' as SAMPLINGTYPE."
            )
        if (len(sampling_interval) > 1
                and np.any(np.diff(sampling_interval) <= 0)):
            raise ValueError(
                "When using 'Fixed' as SAMPLINGTYPE the vector of sampling "
                'times in SAMPLINGINTERVAL must be sorted and strictly '
                'increasing.'
            )

    # --- skip ---
    # Ref: realized_bipower_variation.m:138-144
    if skip is None:
        skip = 0
    skip = int(skip)
    if skip < 0:
        raise ValueError('SKIP must be a non-negative scalar.')

    # --- subsamples ---
    # Ref: realized_bipower_variation.m:146-152
    if subsamples is None:
        subsamples = 1
    subsamples = int(subsamples)
    if subsamples < 1:
        raise ValueError('SUBSAMPLES must be a positive integer (>= 1).')

    # ==================================================================
    # Convert times to unit [0, 1] interval if not already unit
    # Ref: realized_bipower_variation.m:159-164
    # ==================================================================
    if time_type != 'unit':
        # Ref: m:161 — [time,~,~,samplingInterval] =
        #   realized_convert2unit(time,timeType,samplingType,samplingInterval)
        time, _, _, sampling_interval = realized_convert2unit(
            time, time_type, sampling_type, sampling_interval
        )
    # Ref: m:164 — Since everything is converted to unit, set timeType='unit'
    time_type = 'unit'

    # ==================================================================
    # Filter prices and compute BPV
    # Ref: realized_bipower_variation.m:167-181
    # ==================================================================

    # Ref: m:167 — logPrice = log(price)
    log_price = np.log(price)

    # Ref: m:168 — filteredLogPrice = realized_price_filter(...)
    filtered_log_price, _filtered_time, _actual_time = realized_price_filter(
        log_price, time, time_type, sampling_type, sampling_interval
    )

    # Ref: m:169 — returns = diff(filteredLogPrice)
    returns = np.diff(filtered_log_price)

    # Ref: m:170 — m = length(returns)
    m = len(returns)

    # Ref: m:171-177 — Validate skip against number of returns
    # Ref: m:171 — if (m-1-skip)<1 || (2+skip)>m
    if (m - 1 - skip) < 1 or (2 + skip) > m:
        raise ValueError(
            'The value of SKIP is too large for the sampling method used. '
            'SKIP should be small relative to the number of prices computed '
            'using the chosen sampling method.'
        )
    # Ref: m:173-176 — warning if skip/m > 0.1
    elif (skip / m) > 0.1:
        warnings.warn(
            'The value chosen for skip is large relative to the number of '
            'prices returned for the chosen SAMPLINGTYPE and '
            'SAMPLINGINTERVAL - (skip / #returns) > .1 indicating more than '
            '10% of the returns are being dropped. Interpret the results '
            'with caution.',
            stacklevel=2,
        )

    # Ref: m:179 — bv = (pi/2) * abs(returns(1:m-1-skip))' * abs(returns(2+skip:m))
    # MATLAB 1-based: returns(1:m-1-skip) → Python 0-based: returns[:m-1-skip]
    # MATLAB 1-based: returns(2+skip:m)   → Python 0-based: returns[1+skip:]
    # The MATLAB ' * operation is the inner product (dot product) of two vectors.
    bv = float(
        (np.pi / 2.0)
        * np.sum(
            np.abs(returns[: m - 1 - skip])
            * np.abs(returns[1 + skip :])
        )
    )

    # Ref: m:180 — biasScale = (m-1-skip) / m
    bias_scale = float(m - 1 - skip) / float(m)

    # Ref: m:181 — bvDebiased = bv / biasScale
    bv_debiased = float(bv / bias_scale)

    # ==================================================================
    # Subsampled BPV
    # Ref: realized_bipower_variation.m:183-196
    # ==================================================================

    # Ref: m:183 — subsampledLogPrices = realized_subsample(...)
    subsampled_result = realized_subsample(
        log_price, time, time_type, sampling_type,
        sampling_interval, subsamples
    )

    # Ref: m:184 — bvs = zeros(subsamples, 1)
    bvs = np.zeros(subsamples, dtype=np.float64)

    # Ref: m:185 — totalCount = 0
    total_count = 0
    base_count = 0

    # Ref: m:186-194 — for i=1:subsamples
    for i in range(subsamples):
        # Ref: m:187 — returns = diff(subsampledLogPrices{i})
        # Python: subsampled_result[i] is a tuple; [0] is the price array
        sub_returns = np.diff(subsampled_result[i][0])
        # Ref: m:188 — m = length(returns)
        m_sub = len(sub_returns)

        # Compute BPV for this subsample using the same skip-k formula
        # Ref: m:189 — bvs(i) = (pi/2) * abs(returns(1:m-1-skip))' * abs(returns(2+skip:m))
        n_products = m_sub - 1 - skip
        if n_products >= 1 and (1 + skip) <= m_sub:
            bvs[i] = (np.pi / 2.0) * np.sum(
                np.abs(sub_returns[: m_sub - 1 - skip])
                * np.abs(sub_returns[1 + skip :])
            )
        else:
            # Edge case: subsample has too few returns for the given skip.
            # MATLAB would produce 0 from an empty dot product.
            bvs[i] = 0.0

        # Ref: m:190-192 — if i==1, baseCount = m-1-skip
        if i == 0:
            base_count = m_sub - 1 - skip

        # Ref: m:193 — totalCount = totalCount + m-1-skip
        total_count += m_sub - 1 - skip

    # Ref: m:195 — bvSS = sum(bvs) * (baseCount / totalCount)
    if total_count > 0:
        bv_ss = float(np.sum(bvs) * (float(base_count) / float(total_count)))
    else:
        # Degenerate case: no valid product terms across all subsamples
        bv_ss = 0.0

    # Ref: m:196 — bvSSDebiased = bvSS / biasScale
    bv_ss_debiased = float(bv_ss / bias_scale)

    # ==================================================================
    # Build diagnostics dictionary
    # ==================================================================
    # Ref: realized_bipower_variation.m — mu_1 = sqrt(2/pi) constant
    # used implicitly via pi/2 = 1/mu_1^2
    mu_1 = float(np.sqrt(2.0 / np.pi))

    diagnostics = {
        'num_returns': int(m),
        'num_products': int(m - 1 - skip),
        'mu_1': mu_1,
        'bias_scale': float(bias_scale),
        'skip': int(skip),
        'subsamples': int(subsamples),
    }

    return bv, bv_ss, bv_debiased, bv_ss_debiased, diagnostics
