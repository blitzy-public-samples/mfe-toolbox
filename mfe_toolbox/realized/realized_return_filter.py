"""
Return filtering for realized volatility estimators.

Filters high-frequency prices to compute log returns and time intervals at
a specified sampling frequency.  Supports multiple sampling schemes
(CalendarTime, CalendarUniform, BusinessTime, BusinessUniform, Fixed) and
optional subsampling for bias-adjusted realized estimators.

Migrated from: realized/realized_return_filter.m
Original author: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 1    Date: 5/1/2008

Notes
-----
The original MATLAB source contained an erroneous docstring header
("THESE COMMENTS ARE WRONG") referencing Quantile Realized Variance.
The *actual* implementation performs return filtering, which is what this
Python migration faithfully reproduces.
"""

import numpy as np

from mfe_toolbox.realized.realized_price_filter import realized_price_filter
from mfe_toolbox.realized.wall2seconds import wall2seconds


def realized_return_filter(
    price: np.ndarray,
    time: np.ndarray,
    time_type: str,
    sampling_type: str,
    sampling_interval,
    subsamples: int = 0,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Filter high-frequency prices to produce log returns and time intervals.

    Applies ``realized_price_filter`` to obtain filtered prices at the
    requested sampling scheme, then computes log returns as
    ``diff(log(filtered_price))`` and time intervals for bias adjustment
    in subsampled realized estimators.

    Parameters
    ----------
    price : array_like
        1-D array of high-frequency prices (m,).
    time : array_like
        1-D array of observation times corresponding to *price* (m,).
        Must be sorted and non-decreasing.
    time_type : str
        Time format descriptor.  One of:

        * ``'wall'``    — 24-hour clock HHMMSS (e.g. 93000, 160000).
        * ``'seconds'`` — Seconds past midnight.
        * ``'unit'``    — Unit-normalised [0, 1].
    sampling_type : str
        Sampling scheme.  One of:

        * ``'CalendarTime'``     — Calendar-time intervals.
        * ``'CalendarUniform'``  — Uniformly spaced in calendar time.
        * ``'BusinessTime'``     — Tick-time intervals.
        * ``'BusinessUniform'``  — Uniformly spaced in business (tick) time.
        * ``'Fixed'``            — Sample at specific user-provided times.
    sampling_interval : int, float, or np.ndarray
        Meaning depends on *sampling_type* (see ``realized_price_filter``).
    subsamples : int, optional
        Number of jittered subsamples to compute.  When ``subsamples > 0``,
        additional return vectors are produced by shifting the sampling grid,
        enabling bias-adjusted (subsampled) realized estimators.  Default is 0
        (no subsampling; a single return vector is computed).

    Returns
    -------
    returns : list of np.ndarray
        A list of ``subsamples + 1`` 1-D arrays, each containing filtered
        log returns.  ``returns[0]`` is the primary (un-shifted) return
        vector.  Subsequent entries are jittered subsample returns.
    interval : np.ndarray
        1-D array of length ``subsamples + 1`` giving the time-interval
        ratio for each subsample relative to the primary sample.
        ``interval[0]`` is always ``1.0``.

    Raises
    ------
    ValueError
        If inputs fail validation (wrong shapes, unsorted times,
        invalid *time_type* or *sampling_type*, etc.).

    See Also
    --------
    mfe_toolbox.realized.realized_price_filter.realized_price_filter :
        Core price filtering engine.
    mfe_toolbox.realized.wall2seconds.wall2seconds :
        Wall-clock to seconds conversion.

    Examples
    --------
    >>> import numpy as np
    >>> prices = np.array([100.0, 100.5, 101.0, 100.8, 101.2, 101.5])
    >>> times  = np.array([93000., 93500., 100000., 110000., 140000., 160000.])
    >>> rets, ival = realized_return_filter(prices, times, 'wall',
    ...                                     'CalendarTime', 3600)
    """
    # ==================================================================
    # Input coercion
    # Ref: realized_return_filter.m:62-78
    # ==================================================================
    price = np.asarray(price, dtype=np.float64).ravel()
    time = np.asarray(time, dtype=np.float64).ravel()

    # Ref: realized_return_filter.m:58-60 — nargin check (5 or 6 args)
    # In Python all arguments except subsamples are required by signature.

    # ------------------------------------------------------------------
    # price validation
    # Ref: realized_return_filter.m:62-66
    # ------------------------------------------------------------------
    if price.ndim != 1 or price.size == 0:
        raise ValueError("PRICE must be a non-empty 1-D vector.")

    # ------------------------------------------------------------------
    # time validation
    # Ref: realized_return_filter.m:68-76
    # ------------------------------------------------------------------
    if time.ndim != 1 or time.size != price.size:
        raise ValueError(
            "TIME must be a 1-D vector with the same length as PRICE."
        )
    if np.any(np.diff(time) < 0):
        raise ValueError("TIME must be sorted and increasing.")

    # Ref: realized_return_filter.m:78 — cast to float64 (already done above)

    # ------------------------------------------------------------------
    # time_type validation
    # Ref: realized_return_filter.m:80-83
    # ------------------------------------------------------------------
    time_type_lower = time_type.lower()
    if time_type_lower not in ("wall", "seconds", "unit"):
        raise ValueError(
            "time_type must be one of 'wall', 'seconds' or 'unit'."
        )

    # ------------------------------------------------------------------
    # sampling_type validation
    # Ref: realized_return_filter.m:84-87
    # ------------------------------------------------------------------
    sampling_type_lower = sampling_type.lower()
    valid_sampling = (
        "calendartime",
        "calendaruniform",
        "businesstime",
        "businessuniform",
        "fixed",
    )
    if sampling_type_lower not in valid_sampling:
        raise ValueError(
            "sampling_type must be one of 'CalendarTime', 'CalendarUniform', "
            "'BusinessTime', 'BusinessUniform' or 'Fixed'."
        )

    # ------------------------------------------------------------------
    # sampling_interval validation
    # Ref: realized_return_filter.m:89-107
    # ------------------------------------------------------------------
    m = price.shape[0]
    t0 = time[0]
    tT = time[m - 1]

    if sampling_type_lower in (
        "calendartime",
        "calendaruniform",
        "businesstime",
        "businessuniform",
    ):
        # Ref: realized_return_filter.m:93-96 — must be positive scalar int
        _si_scalar = np.isscalar(sampling_interval) or (
            isinstance(sampling_interval, np.ndarray)
            and sampling_interval.ndim == 0
        )
        _si_val = float(sampling_interval)
        if not _si_scalar or np.floor(_si_val) != _si_val or _si_val < 1:
            raise ValueError(
                "sampling_interval must be a positive integer for the "
                "selected sampling_type."
            )
    else:
        # 'fixed' — vector of sampling times
        # Ref: realized_return_filter.m:98-107
        sampling_interval = np.asarray(
            sampling_interval, dtype=np.float64
        ).ravel()
        if not (np.any(sampling_interval >= t0) and np.any(sampling_interval <= tT)):
            raise ValueError(
                "At least one sampling interval must be between min(TIME) and "
                "max(TIME) when using 'Fixed' as sampling_type."
            )
        if sampling_interval.size > 1 and np.any(np.diff(sampling_interval) <= 0):
            raise ValueError(
                "When using 'Fixed' as sampling_type the vector of sampling "
                "times in sampling_interval must be sorted and strictly "
                "increasing."
            )

    # ------------------------------------------------------------------
    # subsamples validation
    # Ref: realized_return_filter.m:110-116
    # ------------------------------------------------------------------
    if subsamples is None:
        subsamples = 0
    subsamples = int(subsamples)
    if subsamples < 0 or float(subsamples) != float(int(subsamples)):
        raise ValueError("subsamples must be a non-negative integer.")

    # ==================================================================
    # Time conversion: wall → seconds
    # Ref: realized_return_filter.m:122-126
    # This is done *before* calling realized_price_filter so that
    # subsampling interval arithmetic operates on seconds, not HHMMSS.
    # ==================================================================
    if time_type_lower == "wall":
        # Ref: realized_return_filter.m:124 — time = wall2seconds(time)
        time = wall2seconds(time)
        time_type_lower = "seconds"

    # ==================================================================
    # Initialise output containers
    # Ref: realized_return_filter.m:129-130
    # MATLAB uses a cell array for returns and a vector for interval.
    # Python: list of np.ndarray and a 1-D np.ndarray.
    # ==================================================================
    returns: list[np.ndarray] = [np.empty(0)] * (subsamples + 1)
    interval = np.zeros(subsamples + 1, dtype=np.float64)

    # ==================================================================
    # Primary filtering
    # Ref: realized_return_filter.m:132-135
    # ==================================================================
    log_price = np.log(price)

    # Ref: realized_return_filter.m:133
    # realized_price_filter returns (filtered_price, filtered_time, actual_time)
    filtered_log_price, filtered_times, _ = realized_price_filter(
        log_price, time, time_type_lower, sampling_type, sampling_interval
    )

    # Ref: realized_return_filter.m:134 — returns{1} = diff(filteredLogPrice)
    returns[0] = np.diff(filtered_log_price)
    # Ref: realized_return_filter.m:135 — interval(1) = 1
    interval[0] = 1.0

    # ==================================================================
    # Subsampling
    # Ref: realized_return_filter.m:138-196
    # ==================================================================
    if subsamples > 0:
        # Ref: realized_return_filter.m:141 — n = number of filtered times
        n = filtered_times.shape[0]
        # Ref: realized_return_filter.m:144
        base_difference = filtered_times[n - 1] - filtered_times[0]

        if sampling_type_lower in ("calendartime", "calendaruniform", "fixed"):
            # --------------------------------------------------------
            # Calendar-time / Fixed subsampling
            # Ref: realized_return_filter.m:146-168
            # --------------------------------------------------------
            # Ref: realized_return_filter.m:148 — gap = diff(filteredTimes)
            gap = np.diff(filtered_times)
            # Ref: realized_return_filter.m:150 — step = gap / (subsamples+1)
            step = gap / (subsamples + 1)

            # Ref: realized_return_filter.m:151-155
            # Append one extra step element for the last interval.
            if sampling_type_lower in ("calendartime", "calendaruniform"):
                # Ref: realized_return_filter.m:152 — step = [step; mean(step)]
                step = np.append(step, np.mean(step))
            else:
                # 'fixed' — Ref: realized_return_filter.m:154
                # step = [step; step(length(step))]
                step = np.append(step, step[-1])

            for i in range(1, subsamples + 1):
                # Ref: realized_return_filter.m:157 — thisSampleTime = filteredTimes
                this_sample_time = filtered_times.copy()
                # Ref: realized_return_filter.m:159
                # thisSampleTime = thisSampleTime(1:n) + i*step
                this_sample_time = this_sample_time[:n] + i * step[:n]

                # Ref: realized_return_filter.m:161 — MATLAB passes 'unit' as
                # time_type so that realized_price_filter skips all time
                # conversion.  The raw time/thisSampleTime values (in seconds
                # or unit format) are compared directly by the interpolation
                # engine, which is unit-agnostic.
                filtered_log_price_sub, _, _ = realized_price_filter(
                    log_price, time, "unit", "fixed", this_sample_time
                )

                # Ref: realized_return_filter.m:163
                returns[i] = np.diff(filtered_log_price_sub)

                # Ref: realized_return_filter.m:167
                # interval(i+1) = (thisSampleTime(n-1)-thisSampleTime(1))/baseDifference
                # MATLAB n-1 is 0-based n-2 in Python
                interval[i] = (
                    (this_sample_time[n - 2] - this_sample_time[0])
                    / base_difference
                )

        elif sampling_type_lower in ("businesstime", "businessuniform"):
            # --------------------------------------------------------
            # Business-time subsampling
            # Ref: realized_return_filter.m:169-195
            # --------------------------------------------------------
            if sampling_type_lower == "businesstime":
                # Ref: realized_return_filter.m:173
                # originalIndices = 1:samplingInterval:m  (MATLAB 1-based)
                # Python 0-based: 0, si, 2*si, ...
                si_int = int(sampling_interval)
                original_indices = np.arange(0, m, si_int)
                # Ref: realized_return_filter.m:174-176 — ensure last obs included
                if original_indices[-1] != m - 1:
                    original_indices = np.append(original_indices, m - 1)
                bt_gap = float(si_int)
            else:
                # 'businessuniform'
                # Ref: realized_return_filter.m:180
                # originalIndices = floor(linspace(1,m,samplingInterval))  (MATLAB 1-based)
                # Python 0-based equivalent:
                si_int = int(sampling_interval)
                original_indices = np.floor(
                    np.linspace(0, m - 1, si_int)
                ).astype(np.intp)
                # Ref: realized_return_filter.m:181-183
                if original_indices[-1] != m - 1:
                    original_indices = np.append(original_indices, m - 1)
                # Ref: realized_return_filter.m:184 — gap = m/samplingInterval
                bt_gap = float(m) / float(si_int)

            # Ref: realized_return_filter.m:187 — step = gap/(subsamples+1)
            bt_step = bt_gap / (subsamples + 1)

            for i in range(1, subsamples + 1):
                # Ref: realized_return_filter.m:191
                # indices = floor(originalIndices + step*i)
                # MATLAB originalIndices is 1-based, so offset conversion:
                #   MATLAB: floor(1-based + step*i) → 1-based index
                #   Python: floor(0-based + step*i) → 0-based index
                indices = np.floor(
                    original_indices.astype(np.float64) + bt_step * i
                ).astype(np.intp)

                # Ref: realized_return_filter.m:192 — indices(indices>m) = m
                # MATLAB 1-based: cap at m. Python 0-based: cap at m-1.
                indices[indices >= m] = m - 1

                # Ref: realized_return_filter.m:193
                returns[i] = np.diff(log_price[indices])

                # Ref: realized_return_filter.m:194
                # interval(i+1) = (time(max(indices))-time(min(indices)))/baseDifference
                interval[i] = (
                    (time[np.max(indices)] - time[np.min(indices)])
                    / base_difference
                )

    return returns, interval
