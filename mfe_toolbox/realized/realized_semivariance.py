"""
Realized semivariance estimation (Barndorff-Nielsen, Kinnebrock, Shephard).

Computes positive and negative realized semivariance from high-frequency
price data, decomposing realized variance into upside and downside
components.  Supports multiple sampling schemes and optional subsampling
for bias reduction.

The negative semivariance ``rsvn`` captures variation from price decreases,
while the positive semivariance ``rsvp`` captures variation from price
increases.  By construction, ``rsvn + rsvp`` equals the total realized
variance.

Migrated from: ``realized/realized_semivariance.m`` (MFE Toolbox Version 4.0)

See Also
--------
mfe_toolbox.realized.realized_variance : Total realized variance.
mfe_toolbox.realized.realized_kernel : Kernel-based realized variance estimator.
mfe_toolbox.realized.realized_quantile_variance : Quantile-based realized variance.
mfe_toolbox.realized.realized_range : Range-based realized variance estimator.
mfe_toolbox.realized.realized_price_filter : Core price filtering function.

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


def realized_semivariance(
    price,
    time=None,
    time_type: str = 'unit',
    sampling_type: str = 'CalendarTime',
    sampling_interval=1,
    subsamples: int = 1,
) -> tuple[float, float, float, float, dict]:
    """
    Compute realized semivariance from high-frequency price data.

    Estimates positive and negative realized semivariance, decomposing
    realized variance into upside and downside components.  Also computes
    subsampled versions for bias reduction using the Barndorff-Nielsen,
    Kinnebrock, and Shephard (2010) averaging scheme.

    Parameters
    ----------
    price : array_like
        An m-element 1-D vector of high frequency prices.  If a row vector
        is provided it is automatically transposed to a column vector.
    time : array_like
        An m-element 1-D vector of times where ``time[i]`` corresponds to
        ``price[i]``.  Must be sorted and increasing.  The format must
        match *time_type*:

        * ``'wall'``    — 24-hour clock in HHMMSS format (e.g. 101543).
        * ``'seconds'`` — Seconds past midnight (e.g. 36943).
        * ``'unit'``    — Unit normalized [0, 1].
    time_type : str, optional
        String describing the way times are measured.  Case-insensitive.
        One of ``'wall'``, ``'seconds'``, or ``'unit'``.
        Default is ``'unit'``.
    sampling_type : str, optional
        String describing the type of sampling to use when filtering prices.
        Case-insensitive.  One of:

        * ``'CalendarTime'``     — Sample in calendar time using observations
          separated by *sampling_interval* seconds.
        * ``'CalendarUniform'``  — *sampling_interval* observations spread
          uniformly between ``time[0]`` and ``time[-1]``.
        * ``'BusinessTime'``     — Every *sampling_interval*-th tick.
        * ``'BusinessUniform'``  — *sampling_interval* ticks uniformly
          spaced in business time.
        * ``'Fixed'``            — Sample at specific points in time given
          by *sampling_interval*.

        Default is ``'CalendarTime'``.
    sampling_interval : scalar or array_like, optional
        Scalar integer or 1-D vector whose meaning depends on *sampling_type*.
        For ``'Fixed'``, must be a sorted, strictly increasing 1-D array of
        times in the same format as *time*.  Default is ``1``.
    subsamples : int, optional
        Number of subsample realized semivariance estimators to average with
        the original.  Subsample estimators are based on prices uniformly
        shifted between the original sample points.  Must be a non-negative
        integer.  Default is ``1``.

    Returns
    -------
    rsvn : float
        Negative-part realized semivariance estimate.  Equals the sum of
        squared returns for all strictly negative returns.
    rsvp : float
        Positive-part realized semivariance estimate.  Equals the sum of
        squared returns for all strictly positive returns.
    rsvn_ss : float
        Subsample-averaged negative-part realized semivariance estimate.
    rsvp_ss : float
        Subsample-averaged positive-part realized semivariance estimate.
    diagnostics : dict
        Dictionary with diagnostic information:

        * ``'num_returns'``  — Number of returns in the base (non-subsampled)
          sample.
        * ``'num_negative'`` — Count of strictly negative returns.
        * ``'num_positive'`` — Count of strictly positive returns.
        * ``'num_zero'``     — Count of exactly zero returns.
        * ``'subsamples'``   — Number of subsamples used.
        * ``'rv'``           — Total realized variance (``rsvn + rsvp``).
        * ``'rv_ss'``        — Total subsampled realized variance
          (``rsvn_ss + rsvp_ss``).

    Raises
    ------
    ValueError
        If *time* is ``None`` (5 inputs are required).
        If *price* is not a 1-D vector.
        If *time* is not sorted and increasing.
        If *time* length does not match *price* length.
        If *time_type* is not a recognized format.
        If *sampling_type* is not a recognized scheme.
        If *sampling_interval* is invalid for the selected *sampling_type*
        and *time_type* combination.
        If *subsamples* is not a non-negative scalar integer.

    Notes
    -----
    By construction, ``rsvn + rsvp`` equals the total realized variance
    (within machine precision) since the semivariance decomposes squared
    returns into those from strictly negative and strictly positive price
    movements.  Returns that are exactly zero do not contribute to either
    semivariance component, matching the MATLAB implementation.

    The subsampled estimators use an averaging scheme where the weighting
    factor is ``base_count / total_count``, where ``base_count`` is the
    number of returns in the first subsample grid and ``total_count`` is
    the sum of return counts across all subsample grids.

    Examples
    --------
    5-minute realized semivariance using unit time:

    >>> import numpy as np
    >>> prices = np.array([100.0, 100.5, 101.0, 100.8, 101.2])
    >>> times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    >>> rsvn, rsvp, rsvn_ss, rsvp_ss, diag = realized_semivariance(
    ...     prices, times, 'unit', 'CalendarTime', 0.5)

    10-tick realized semivariance:

    >>> rsvn, rsvp, rsvn_ss, rsvp_ss, diag = realized_semivariance(
    ...     prices, times, 'unit', 'BusinessTime', 2)
    """
    # ==================================================================
    # Input Checking
    # Ref: realized_semivariance.m:64-133
    # ==================================================================

    # Ref: realized_semivariance.m:67-69 — Five or Six inputs required.
    # In Python, time=None default signals that the caller forgot to
    # supply the mandatory time argument.
    if time is None:
        raise ValueError('Five or Six inputs required.')

    # Ref: realized_semivariance.m:70-71 — If row vector, transpose to column.
    # np.asarray + ravel guarantees a 1-D float64 array.
    price = np.asarray(price, dtype=np.float64).ravel()

    # Ref: realized_semivariance.m:73-75 — PRICE must be a m by 1 vector.
    # After ravel() the array is guaranteed 1-D.  Check it is non-empty.
    if price.size == 0:
        raise ValueError('PRICE must be a m by 1 vector.')

    # Ref: realized_semivariance.m:76-78 — Transpose time row to column.
    time = np.asarray(time, dtype=np.float64).ravel()

    # Ref: realized_semivariance.m:79-81 — TIME must be sorted and increasing.
    if len(time) > 1 and np.any(np.diff(time) < 0):
        raise ValueError('TIME must be sorted and increasing')

    # Ref: realized_semivariance.m:82-84 — TIME must be m by 1 and match
    # price length.
    if len(time) != len(price):
        raise ValueError('TIME must be a m by 1 vector.')

    # Ref: realized_semivariance.m:86 — Cast time to double.
    # Already handled by np.asarray(..., dtype=np.float64) above.

    # Ref: realized_semivariance.m:88-91 — Validate timeType.
    time_type = time_type.lower()
    if time_type not in ('wall', 'seconds', 'unit'):
        raise ValueError(
            "TIMETYPE must be one of 'wall', 'seconds' or 'unit'."
        )

    # Ref: realized_semivariance.m:92-95 — Validate samplingType.
    sampling_type_lower = sampling_type.lower()
    _valid_sampling_types = (
        'calendartime', 'calendaruniform',
        'businesstime', 'businessuniform', 'fixed',
    )
    if sampling_type_lower not in _valid_sampling_types:
        raise ValueError(
            "SAMPLINGTYPE must be one of 'CalendarTime', 'CalendarUniform', "
            "'BusinessTime', 'BusinessUniform' or 'Fixed'."
        )

    # Ref: realized_semivariance.m:97-99 — Compute m, t0, tT.
    m = len(price)
    t0 = time[0]
    tT = time[m - 1]

    # Ref: realized_semivariance.m:100-121 — Validate samplingInterval
    # based on samplingType and timeType combination.
    if sampling_type_lower in (
        'calendartime', 'calendaruniform',
        'businesstime', 'businessuniform',
    ):
        # Ref: realized_semivariance.m:102-110 — Must be scalar; integer
        # check depends on timeType.
        if time_type in ('wall', 'seconds'):
            # Ref: realized_semivariance.m:103-105 — positive scalar integer
            if (
                not np.isscalar(sampling_interval)
                or np.floor(float(sampling_interval)) != float(sampling_interval)
                or float(sampling_interval) < 1
            ):
                raise ValueError(
                    'SAMPLINGINTERVAL must be a positive integer for the '
                    "SAMPLINGTYPE selected when using 'wall' or 'seconds' "
                    'as TIMETYPE.'
                )
        else:
            # Ref: realized_semivariance.m:107-109 — positive scalar (unit)
            if not np.isscalar(sampling_interval) or float(sampling_interval) < 0:
                raise ValueError(
                    'SAMPLINGINTERVAL must be a positive value for the '
                    "SAMPLINGTYPE selected when using 'unit' as TIMETYPE."
                )
    else:
        # Ref: realized_semivariance.m:112-121 — Fixed sampling type.
        # samplingInterval must be a vector with at least one value in range.
        sampling_interval = np.asarray(
            sampling_interval, dtype=np.float64
        ).ravel()

        # Ref: realized_semivariance.m:115-117
        if not (
            np.any(sampling_interval >= t0)
            and np.any(sampling_interval <= tT)
        ):
            raise ValueError(
                'At least one sampling interval must be between min(TIME) '
                "and max(TIME) when using 'Fixed' as SAMPLINGTYPE."
            )

        # Ref: realized_semivariance.m:118-120 — strictly increasing
        if len(sampling_interval) > 1 and np.any(
            np.diff(sampling_interval) <= 0
        ):
            raise ValueError(
                "When using 'Fixed' as SAMPLINGTYPE the vector of sampling "
                'times in SAMPLINGINTERVAL must be sorted and strictly '
                'increasing.'
            )

    # Ref: realized_semivariance.m:123-130 — Validate subsamples.
    if subsamples is None:
        subsamples = 1
    if (
        not np.isscalar(subsamples)
        or float(subsamples) < 0
        or np.floor(float(subsamples)) != float(subsamples)
    ):
        raise ValueError('SUBSAMPLES must be a non-negative scalar.')
    subsamples = int(subsamples)

    # ==================================================================
    # Time Conversion
    # Ref: realized_semivariance.m:135-140
    # ==================================================================

    # Ref: realized_semivariance.m:136-138 — Convert times to unit
    # interval if not already in unit format.
    if time_type != 'unit':
        # Ref: realized_semivariance.m:137
        time, _time0, _time1, sampling_interval = realized_convert2unit(
            time, time_type, sampling_type_lower, sampling_interval
        )

    # Ref: realized_semivariance.m:140 — Since everything is now in unit
    # interval, set timeType to 'unit' for all downstream calls.
    time_type = 'unit'

    # ==================================================================
    # Filter prices and compute the base realized semivariance
    # Ref: realized_semivariance.m:142-147
    # ==================================================================

    # Ref: realized_semivariance.m:143 — Compute log prices from raw prices.
    log_price = np.log(price)

    # Ref: realized_semivariance.m:144 — Filter log prices at the sampling
    # grid using last-price interpolation.
    # realized_price_filter returns (filtered_price, filtered_time, actual_time);
    # we only need filtered_price here.
    filtered_log_price, _filtered_time, _actual_time = realized_price_filter(
        log_price, time, time_type, sampling_type_lower, sampling_interval
    )

    # Ref: realized_semivariance.m:145 — Compute returns as first differences
    # of the filtered log prices.
    returns = np.diff(filtered_log_price)

    # Ref: realized_semivariance.m:146 — Negative semivariance:
    # sum of squared returns weighted by the indicator (returns < 0).
    # MATLAB: rsvn = sum((returns.^2).*(returns<0))
    # Note: returns exactly equal to zero do NOT contribute.
    rsvn = float(np.sum((returns ** 2) * (returns < 0)))

    # Ref: realized_semivariance.m:147 — Positive semivariance:
    # sum of squared returns weighted by the indicator (returns > 0).
    # MATLAB: rsvp = sum((returns.^2).*(returns>0))
    rsvp = float(np.sum((returns ** 2) * (returns > 0)))

    # ==================================================================
    # Subsampled semivariance
    # Ref: realized_semivariance.m:149-163
    # ==================================================================

    # Ref: realized_semivariance.m:149 — Generate subsampled log price
    # grids.  Each element of the returned list is a tuple
    # (subsampled_prices, subsampled_times, base_count, total_count).
    subsampled_log_prices = realized_subsample(
        log_price, time, time_type, sampling_type_lower,
        sampling_interval, subsamples
    )

    # Ref: realized_semivariance.m:150-151 — Initialise accumulators.
    rsvns = np.zeros(subsamples, dtype=np.float64)
    rsvps = np.zeros(subsamples, dtype=np.float64)
    total_count = 0
    base_count = 0

    # Ref: realized_semivariance.m:153-161 — Loop over subsamples.
    # MATLAB loop: for i = 1:subsamples (1-based)
    # Python loop: for i in range(subsamples) (0-based)
    for i in range(subsamples):
        # Ref: realized_semivariance.m:154 — Extract log prices for this
        # subsample from the cell array / list.
        # MATLAB: returns = diff(subsampledLogPrices{i})
        sub_log_prices = subsampled_log_prices[i][0]
        sub_returns = np.diff(sub_log_prices)

        # Ref: realized_semivariance.m:155-156 — Compute semivariances
        # for this subsample using the same indicator scheme.
        rsvns[i] = np.sum((sub_returns ** 2) * (sub_returns < 0))
        rsvps[i] = np.sum((sub_returns ** 2) * (sub_returns > 0))

        # Ref: realized_semivariance.m:157-159 — Record the return count
        # of the first subsample as the base count.
        # MATLAB: if i==1, baseCount = length(returns), end
        if i == 0:
            base_count = len(sub_returns)
        total_count = total_count + len(sub_returns)

    # Ref: realized_semivariance.m:162-163 — Weighted average of subsampled
    # semivariances.  The weighting factor (baseCount / totalCount) rescales
    # so that the subsampled estimator has the same expected magnitude as
    # the base estimator.
    if total_count > 0:
        weight = float(base_count) / float(total_count)
        rsvn_ss = float(np.sum(rsvns) * weight)
        rsvp_ss = float(np.sum(rsvps) * weight)
    else:
        # Edge case: no returns in any subsample (degenerate input).
        rsvn_ss = 0.0
        rsvp_ss = 0.0

    # ==================================================================
    # Build diagnostics dictionary
    # ==================================================================
    num_negative = int(np.sum(returns < 0))
    num_positive = int(np.sum(returns > 0))
    num_zero = int(np.sum(returns == 0))

    diagnostics = {
        'num_returns': int(len(returns)),
        'num_negative': num_negative,
        'num_positive': num_positive,
        'num_zero': num_zero,
        'subsamples': subsamples,
        'rv': rsvn + rsvp,
        'rv_ss': rsvn_ss + rsvp_ss,
    }

    return rsvn, rsvp, rsvn_ss, rsvp_ss, diagnostics
