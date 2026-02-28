"""
Bivariate refresh time synchronization for two asset price series.

Implements the refresh time sampling algorithm from Barndorff-Nielsen, Hansen,
Lunde, and Shephard (BNHLS, 2008) "Multivariate Realised Kernels". This module
synchronizes two asynchronously observed price series onto a common set of
refresh time instants, which is a prerequisite for computing multivariate
realized kernels and realized covariances.

The core algorithm is a dual-pointer scan that walks through both time series
simultaneously, aligning observations at the earliest moment both assets have
been observed at or after the previous refresh time.

Migrated from: realized/realized_refresh_time_bivariate.m
Original author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
Original revision: 1, Date: 7/1/2008
"""

import warnings

import numpy as np


def realized_refresh_time_bivariate(
    time_type: str,
    price1: np.ndarray,
    time1: np.ndarray,
    price2: np.ndarray,
    time2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute refresh time synchronized prices for two asset series.

    Implements the bivariate refresh time synchronization from BNHLS (2008).
    Given two price series observed at potentially different (asynchronous)
    times, this function finds the set of refresh time instants and the
    corresponding synchronized prices.

    Parameters
    ----------
    time_type : str
        String describing the time measurement format. Must be one of:

        - ``'wall'`` : 24-hour clock format HHMMSS (e.g. 101543 or 153217).
        - ``'seconds'`` : Seconds past midnight on the first day.
        - ``'unit'`` : Unit-normalized time in [0, 1] (e.g. 0.1, 0.234, 0.9).
          Unit-normalized times are more general and can span multiple
          calendar days.

    price1 : np.ndarray
        (m1,) or (m1, 1) vector of prices for the first asset. Observations
        must correspond 1:1 with ``time1``.
    time1 : np.ndarray
        (m1,) or (m1, 1) vector of observation times for the first asset.
        Must be strictly ascending (monotonically increasing) and unique.
    price2 : np.ndarray
        (m2,) or (m2, 1) vector of prices for the second asset. Observations
        must correspond 1:1 with ``time2``.
    time2 : np.ndarray
        (m2,) or (m2, 1) vector of observation times for the second asset.
        Must be strictly ascending (monotonically increasing) and unique.

    Returns
    -------
    prices : np.ndarray
        (N, 2) array of refresh time synchronized prices, where column 0
        corresponds to ``price1`` and column 1 to ``price2``. N is the number
        of refresh time instants found (N <= min(m1, m2)).
    refresh_times : np.ndarray
        (N,) vector of refresh time instants. Each refresh time is the
        row-wise maximum of ``actual_times``, representing the moment at which
        both assets have been observed.
    actual_times : np.ndarray
        (N, 2) array of the actual observation times used to construct each
        row of ``prices``. Column 0 holds times from ``time1``, column 1 from
        ``time2``.

    Raises
    ------
    ValueError
        If ``time_type`` is not one of 'wall', 'seconds', or 'unit'.
    ValueError
        If ``price1`` and ``time1`` have different lengths, or ``price2`` and
        ``time2`` have different lengths.
    ValueError
        If ``time1`` or ``time2`` are not strictly ascending.

    Notes
    -----
    The price series inputs should generally be in tick time. It is possible
    to compute a calendar-time refresh time price by first sampling the more
    liquid asset in calendar time using ``realized_price_filter`` and then
    running this function on the filtered prices of the more liquid asset
    and the original prices of the less liquid asset.

    The algorithm uses a dual-pointer scan that walks both time vectors
    forward, at each step finding the earliest pair of observations such
    that both assets have been observed at or after the current pointer
    positions. This produces the Refresh Time grid described in BNHLS (2008).

    References
    ----------
    Barndorff-Nielsen, O. E., Hansen, P. R., Lunde, A., & Shephard, N. (2008).
    "Multivariate Realised Kernels." Working paper.

    See Also
    --------
    realized_refresh_time : General N-asset refresh time synchronization.
    realized_multivariate_kernel : Multivariate realized kernel estimator.
    realized_variance : Univariate realized variance estimator.

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.realized.realized_refresh_time_bivariate import (
    ...     realized_refresh_time_bivariate,
    ... )
    >>> t1 = np.array([1.0, 2.0, 3.0, 5.0])
    >>> p1 = np.array([100.0, 101.0, 102.0, 103.0])
    >>> t2 = np.array([1.5, 3.0, 4.0])
    >>> p2 = np.array([200.0, 201.0, 202.0])
    >>> prices, rt, at = realized_refresh_time_bivariate('unit', p1, t1, p2, t2)
    """
    # ------------------------------------------------------------------
    # Input Validation
    # ------------------------------------------------------------------

    # Ref: realized_refresh_time_bivariate.m:49 — convert timeType to lowercase
    time_type = time_type.lower()

    # Ref: realized_refresh_time_bivariate.m:50-51 — validate time_type
    valid_time_types = {"wall", "seconds", "unit"}
    if time_type not in valid_time_types:
        raise ValueError(
            f"time_type must be one of 'wall', 'seconds' or 'unit'. "
            f"Got '{time_type}'."
        )

    # Ensure inputs are numpy arrays for consistent indexing and operations
    price1 = np.asarray(price1, dtype=np.float64).ravel()
    time1 = np.asarray(time1, dtype=np.float64).ravel()
    price2 = np.asarray(price2, dtype=np.float64).ravel()
    time2 = np.asarray(time2, dtype=np.float64).ravel()

    # Validate matching lengths between prices and times
    if price1.shape[0] != time1.shape[0]:
        raise ValueError(
            f"price1 and time1 must have the same length. "
            f"Got {price1.shape[0]} and {time1.shape[0]}."
        )
    if price2.shape[0] != time2.shape[0]:
        raise ValueError(
            f"price2 and time2 must have the same length. "
            f"Got {price2.shape[0]} and {time2.shape[0]}."
        )

    # Validate that times are strictly ascending (monotonically increasing)
    if time1.shape[0] > 1 and not np.all(np.diff(time1) > 0):
        raise ValueError("time1 must be strictly ascending (monotonically increasing).")
    if time2.shape[0] > 1 and not np.all(np.diff(time2) > 0):
        raise ValueError("time2 must be strictly ascending (monotonically increasing).")

    # Warn about degenerate cases with very short series
    if price1.shape[0] == 0 or price2.shape[0] == 0:
        warnings.warn(
            "One or both price series are empty. Returning empty arrays.",
            stacklevel=2,
        )
        empty_prices = np.zeros((0, 2), dtype=np.float64)
        empty_refresh = np.zeros((0,), dtype=np.float64)
        empty_actual = np.zeros((0, 2), dtype=np.float64)
        return empty_prices, empty_refresh, empty_actual

    # ------------------------------------------------------------------
    # Core Algorithm — Dual-Pointer Scan
    # Ref: realized_refresh_time_bivariate.m:54-98
    # ------------------------------------------------------------------

    # Ref: realized_refresh_time_bivariate.m:54-55 — get series lengths
    m1: int = price1.shape[0]
    m2: int = price2.shape[0]

    # Ref: realized_refresh_time_bivariate.m:58 — preallocate output prices
    max_pairs: int = min(m1, m2)
    prices_out: np.ndarray = np.zeros((max_pairs, 2), dtype=np.float64)

    # Ref: realized_refresh_time_bivariate.m:59 — preallocate actual times
    actual_times_out: np.ndarray = np.zeros((max_pairs, 2), dtype=np.float64)

    # Ref: realized_refresh_time_bivariate.m:61-63 — initialize counters
    # CRITICAL: MATLAB uses 1-based indexing (count=1, ind1=1, ind2=1)
    # Python uses 0-based indexing (count=0, ind1=0, ind2=0)
    count: int = 0
    ind1: int = 0
    ind2: int = 0

    # Ref: realized_refresh_time_bivariate.m:64 — main dual-pointer loop
    # MATLAB: while ind1<=m1 && ind2<=m2
    # Python: while ind1 < m1 and ind2 < m2 (0-based indexing)
    while ind1 < m1 and ind2 < m2:
        # Ref: realized_refresh_time_bivariate.m:65 — check if time1 is behind time2
        if time1[ind1] < time2[ind2]:
            # Ref: realized_refresh_time_bivariate.m:67-68
            # Advance ind1 to find the largest time1 <= time2[ind2]
            while ind1 < m1 and time1[ind1] < time2[ind2]:
                ind1 += 1

            # Ref: realized_refresh_time_bivariate.m:70-71
            # Clamp to last valid index if overshot past end
            # MATLAB: if ind1>m1, ind1=m1  →  Python: if ind1>=m1, ind1=m1-1
            if ind1 >= m1:
                ind1 = m1 - 1

            # Ref: realized_refresh_time_bivariate.m:73-75
            # Step back one if we overshot past the target time
            if time1[ind1] > time2[ind2]:
                ind1 -= 1

        # Ref: realized_refresh_time_bivariate.m:76 — check if time2 is behind time1
        elif time2[ind2] < time1[ind1]:
            # Ref: realized_refresh_time_bivariate.m:77-78
            # Advance ind2 to find the largest time2 <= time1[ind1]
            while ind2 < m2 and time2[ind2] < time1[ind1]:
                ind2 += 1

            # Ref: realized_refresh_time_bivariate.m:80-81
            # Clamp to last valid index if overshot past end
            # MATLAB: if ind2>m2, ind2=m2  →  Python: if ind2>=m2, ind2=m2-1
            if ind2 >= m2:
                ind2 = m2 - 1

            # Ref: realized_refresh_time_bivariate.m:83-85
            # Step back one if we overshot past the target time
            if time2[ind2] > time1[ind1]:
                ind2 -= 1

        # Ref: realized_refresh_time_bivariate.m:87-90
        # Record the synchronized pair at the current refresh instant
        actual_times_out[count, 0] = time1[ind1]
        actual_times_out[count, 1] = time2[ind2]
        prices_out[count, 0] = price1[ind1]
        prices_out[count, 1] = price2[ind2]

        # Ref: realized_refresh_time_bivariate.m:91-93
        # Advance both pointers and the output counter
        ind1 += 1
        ind2 += 1
        count += 1

    # ------------------------------------------------------------------
    # Trim and Compute Refresh Times
    # ------------------------------------------------------------------

    # Ref: realized_refresh_time_bivariate.m:96 — trim to actual number of pairs
    # MATLAB: prices = prices(1:count-1,:) — because MATLAB count started at 1
    # Python: prices = prices[:count] — because Python count started at 0
    prices_out = prices_out[:count]

    # Ref: realized_refresh_time_bivariate.m:97 — trim actual times
    actual_times_out = actual_times_out[:count]

    # Ref: realized_refresh_time_bivariate.m:98 — refresh times are row-wise max
    # refreshTimes = max(actualTimes,[],2) → np.max(actual_times, axis=1)
    refresh_times: np.ndarray = np.max(actual_times_out, axis=1)

    return prices_out, refresh_times, actual_times_out
