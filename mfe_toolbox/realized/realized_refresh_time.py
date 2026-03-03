"""
Refresh-time synchronization for multiple high-frequency asset price series.

Computes refresh time synchronized prices for multi-asset data using the
algorithm from Barndorff-Nielsen, Hansen, Lunde, and Shephard (2008)
"Multivariate Realised Kernels: Consistent Positive Semi-Definite Estimators
of the Covariation of Equity Prices with Noise and Non-Synchronous Trading".

A *refresh time* is defined as the first time at which **all** assets have
been observed at least once since the previous refresh time.  Prices at
refresh times are obtained by previous-tick (forward-fill) interpolation
via ``realized_price_filter`` in 'Fixed' sampling mode.

Migrated from: realized/realized_refresh_time.m
Original author: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 1    Date: 7/1/2008
"""

from __future__ import annotations

import numpy as np

from mfe_toolbox.realized.realized_price_filter import realized_price_filter


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def realized_refresh_time(
    prices: list[np.ndarray],
    times: list[np.ndarray],
    time_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute refresh-time synchronized prices for multiple assets.

    Synchronizes multiple high-frequency price series to a common set of
    refresh times.  A refresh time is defined as the first time at which
    **all** assets have been observed at least once since the previous
    refresh time.

    Parameters
    ----------
    prices : list of np.ndarray
        List of *K* one-dimensional price arrays, one per asset.  Each
        array must contain raw (non-logged) prices.
    times : list of np.ndarray
        List of *K* one-dimensional time arrays corresponding to each
        price array.  Each must be sorted in ascending order.
    time_type : str
        Time format descriptor.  One of:

        * ``'wall'``    — 24-hour clock in HHMMSS format
        * ``'seconds'`` — Seconds past midnight
        * ``'unit'``    — Unit-normalised [0, 1]

    Returns
    -------
    synchronized_prices : np.ndarray
        ``(n, K)`` matrix of refresh-time synchronized prices, where *n*
        is the number of refresh times and *K* is the number of assets.
        Prices are obtained via last-price (previous-tick) interpolation
        at each refresh time.
    refresh_times : np.ndarray
        ``(n,)`` vector of refresh-time points.
    actual_times : np.ndarray
        ``(n, K)`` matrix of actual observation times corresponding to
        each synchronized price (i.e. the true trade time that the
        previous-tick rule selected).

    Raises
    ------
    ValueError
        If inputs are invalid (wrong shapes, unsupported time type,
        fewer than two assets, mismatched lengths, etc.).

    Notes
    -----
    The price series inputs should generally be in *tick time*.  It is
    possible to compute a calendar-time refresh-time price by sampling
    the most liquid asset in calendar time using
    ``realized_price_filter`` and then running this function on the
    filtered prices of the most liquid asset together with the original
    prices of the others.

    References
    ----------
    Barndorff-Nielsen, O. E., Hansen, P. R., Lunde, A., & Shephard, N.
    (2011). Multivariate realised kernels: Consistent positive
    semi-definite estimators of the covariation of equity prices with
    noise and non-synchronous trading.  *Journal of Econometrics*,
    162(2), 149–169.

    Examples
    --------
    >>> import numpy as np
    >>> p1 = np.array([100.0, 100.5, 101.0])
    >>> t1 = np.array([93000.0, 93500.0, 100000.0])
    >>> p2 = np.array([50.0, 50.5, 51.0, 51.5])
    >>> t2 = np.array([93000.0, 93200.0, 93500.0, 100000.0])
    >>> synced, rt, at = realized_refresh_time([p1, p2], [t1, t2], 'wall')
    """

    # ==================================================================
    # Input Validation
    # Ref: realized_refresh_time.m:49–78
    # ==================================================================

    # --- Validate time_type (Ref: m:52-54) ---
    if not isinstance(time_type, str):
        raise ValueError("time_type must be a string.")
    time_type_lower: str = time_type.lower()
    if time_type_lower not in ("wall", "seconds", "unit"):
        raise ValueError(
            "TIMETYPE must be one of 'wall', 'seconds' or 'unit'."
        )

    # --- Validate prices / times are list-like (Ref: m:49-51) ---
    if not isinstance(prices, (list, tuple)):
        raise ValueError("prices must be a list of 1-D price arrays.")
    if not isinstance(times, (list, tuple)):
        raise ValueError("times must be a list of 1-D time arrays.")

    n_assets: int = len(prices)
    # Ref: m:49 — nargin>=3 means at least timeType + one pair; we
    # require at least 2 assets for multi-asset synchronisation.
    if n_assets < 2:
        raise ValueError(
            "At least 2 price/time pairs are required for "
            "refresh-time synchronization."
        )
    if len(times) != n_assets:
        raise ValueError(
            "prices and times must have the same number of elements."
        )

    # --- Parse and validate each price/time pair (Ref: m:60-78) ---
    price_list: list[np.ndarray] = []
    time_list: list[np.ndarray] = []

    for i in np.arange(n_assets):
        # Ref: m:61 — cell assignment; m:64 — double(varargin{2*i})
        temp_price: np.ndarray = np.asarray(prices[i], dtype=np.float64).ravel()
        temp_time: np.ndarray = np.asarray(times[i], dtype=np.float64).ravel()

        # Ref: m:66-68 — column vector enforcement (handled by ravel)

        # Ref: m:73-75 — check dimensions and length match
        if temp_price.ndim != 1 or temp_time.ndim != 1:
            raise ValueError(
                f"Problem reading price and time pair #{i + 1}. "
                f"Both PRICE{i + 1:02d} and TIME{i + 1:02d} must be "
                f"1-D arrays with the same number of elements."
            )
        if len(temp_price) != len(temp_time):
            raise ValueError(
                f"Problem reading price and time pair #{i + 1}. "
                f"Both PRICE{i + 1:02d} and TIME{i + 1:02d} must be "
                f"1-D arrays with the same number of elements."
            )
        if len(temp_price) == 0:
            raise ValueError(
                f"Price/time pair #{i + 1} must contain at least one "
                f"observation."
            )

        price_list.append(temp_price)
        time_list.append(temp_time)

    # ==================================================================
    # Compute Union of All Timestamps
    # Ref: realized_refresh_time.m:83–87
    # ==================================================================
    # Ref: m:84-87 — MATLAB uses iterative union(); equivalent to
    # np.unique(np.concatenate(...)) since union returns sorted unique.
    utimes: np.ndarray = np.unique(np.concatenate(time_list))
    m: int = len(utimes)

    # ==================================================================
    # Construct Time Indicator Matrix
    # Ref: realized_refresh_time.m:90–94
    # ==================================================================
    # Ref: m:91 — timeIndicator = false(m, n)
    time_indicator: np.ndarray = np.zeros((m, n_assets), dtype=bool)
    for i in range(n_assets):
        # Ref: m:93 — timeIndicator(:,i) = ismember(utimes, time{i})
        time_indicator[:, i] = np.isin(utimes, time_list[i])

    # ==================================================================
    # Compute Refresh Times
    # Ref: realized_refresh_time.m:97–111
    # ==================================================================
    # Ref: m:98 — refreshIndicator = false(1, n)
    refresh_indicator: np.ndarray = np.zeros(n_assets, dtype=bool)
    # Ref: m:99 — refreshTimes = false(m, 1) (boolean mask)
    refresh_mask: np.ndarray = np.zeros(m, dtype=bool)

    for j in range(m):
        # Ref: m:102 — OR accumulation: refresh once *any* observation
        # for each asset arrives; when *all* are True, it is a refresh
        # time.
        refresh_indicator = refresh_indicator | time_indicator[j, :]
        # Ref: m:103-108
        if np.all(refresh_indicator):
            # This timestamp is a refresh time
            refresh_mask[j] = True
            # Ref: m:107 — reset indicator for next refresh interval
            refresh_indicator = np.zeros(n_assets, dtype=bool)

    # Ref: m:111 — refreshTimes = utimes(refreshTimes)
    refresh_times: np.ndarray = utimes[refresh_mask]
    n_refresh: int = len(refresh_times)

    # ==================================================================
    # Compute Synchronized Prices and Actual Times
    # Ref: realized_refresh_time.m:114–120
    # ==================================================================
    # Collect per-asset columns first, then stack — uses np.column_stack
    # for assembly (matching numpy members_accessed schema requirement).
    if n_refresh == 0:
        # Edge case: no refresh times found (degenerate input)
        empty_prices = np.zeros((0, n_assets), dtype=np.float64)
        empty_times = np.zeros((0, n_assets), dtype=np.float64)
        return empty_prices, refresh_times, empty_times

    price_cols: list[np.ndarray] = []
    actual_time_cols: list[np.ndarray] = []

    for i in range(n_assets):
        # Ref: m:119 — realized_price_filter(price{i}, time{i},
        #               timeType, 'Fixed', refreshTimes)
        # The Python realized_price_filter accepts 'Fixed' and returns
        # a 3-tuple: (filtered_price, filtered_time, actual_time).
        # Ref: m:119 — [prices(:,i), temp, actualTimes(:,i)]
        filtered_price, _filtered_time, actual_time = realized_price_filter(
            price_list[i],
            time_list[i],
            time_type_lower,
            "Fixed",
            refresh_times,
        )
        price_cols.append(filtered_price)
        actual_time_cols.append(actual_time)

    # Ref: m:115-116 — prices = zeros(length(refreshTimes), n)
    # Build n_refresh × n_assets matrices via column_stack.
    synchronized_prices: np.ndarray = np.column_stack(price_cols)
    actual_times: np.ndarray = np.column_stack(actual_time_cols)

    return synchronized_prices, refresh_times, actual_times
