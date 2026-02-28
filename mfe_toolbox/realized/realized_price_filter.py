"""
Price filtering for realized variance computation.

Applies last-price interpolation to sample high-frequency prices at desired
points in time. Supports calendar-time sampling, business-time (tick) sampling,
and fixed-time sampling with automatic time format conversion.

Migrated from: realized/realized_price_filter.m
Original author: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 1    Date: 5/1/2008
"""

import numpy as np
import warnings


# ---------------------------------------------------------------------------
# Local helper functions for time conversion
# These replicate functionality from wall2seconds.m, seconds2wall.m,
# wall2unit.m, seconds2unit.m, unit2wall.m, unit2seconds.m.
# Implemented locally to keep this module self-contained per schema
# (depends_on_files is empty).
# ---------------------------------------------------------------------------


def _wall2seconds(wall: np.ndarray) -> np.ndarray:
    """Convert wall clock times (HHMMSS.SS format) to seconds past midnight.

    Parameters
    ----------
    wall : np.ndarray
        Wall clock times where HH*10000 + MM*100 + SS.SS.

    Returns
    -------
    np.ndarray
        Seconds past midnight.

    Notes
    -----
    Ref: wall2seconds.m:36-48
    """
    wall = np.asarray(wall, dtype=np.float64)
    hr = np.floor(wall / 10000.0)
    # Ref: wall2seconds.m:39 — mm = floor(wall/100) - hr*100
    mm = np.floor(wall / 100.0) - hr * 100.0
    # Ref: wall2seconds.m:40 — ss = rem(wall, 100)
    ss = np.fmod(wall, 100.0)
    return 3600.0 * hr + 60.0 * mm + ss


def _seconds2wall(seconds: np.ndarray) -> np.ndarray:
    """Convert seconds past midnight to wall clock times (HHMMSS.SS format).

    Parameters
    ----------
    seconds : np.ndarray
        Seconds past midnight values.

    Returns
    -------
    np.ndarray
        Wall clock times in HHMMSS.SS format.

    Notes
    -----
    Ref: seconds2wall.m:38-41
    """
    seconds = np.asarray(seconds, dtype=np.float64)
    hr = np.floor(seconds / 3600.0)
    mm = np.floor((seconds - hr * 3600.0) / 60.0)
    ss = np.fmod(seconds, 60.0)
    return hr * 10000.0 + mm * 100.0 + ss


def _wall2unit(wall: np.ndarray, wall0: float, wall1: float) -> np.ndarray:
    """Convert wall clock times to unit interval [0, 1].

    Parameters
    ----------
    wall : np.ndarray
        Array of wall clock times in HHMMSS format.
    wall0 : float
        Start wall time (maps to 0.0 in unit interval).
    wall1 : float
        End wall time (maps to 1.0 in unit interval).

    Returns
    -------
    np.ndarray
        Times in [0, 1] unit interval.

    Notes
    -----
    Ref: wall2unit.m:73-76 — converts via seconds as intermediate step.
    """
    wall0_sec = _wall2seconds(np.asarray(wall0, dtype=np.float64))
    wall1_sec = _wall2seconds(np.asarray(wall1, dtype=np.float64))
    seconds = _wall2seconds(wall)
    return (seconds - wall0_sec) / (wall1_sec - wall0_sec)


def _seconds2unit(
    seconds: np.ndarray, seconds0: float, seconds1: float
) -> np.ndarray:
    """Convert seconds past midnight to unit interval [0, 1].

    Parameters
    ----------
    seconds : np.ndarray
        Array of seconds past midnight.
    seconds0 : float
        Start seconds (maps to 0.0).
    seconds1 : float
        End seconds (maps to 1.0).

    Returns
    -------
    np.ndarray
        Times in [0, 1] unit interval.

    Notes
    -----
    Ref: seconds2unit.m:65
    """
    seconds = np.asarray(seconds, dtype=np.float64)
    return (seconds - float(seconds0)) / (float(seconds1) - float(seconds0))


def _unit2wall(unit: np.ndarray, wall0: float, wall1: float) -> np.ndarray:
    """Convert unit interval times back to wall clock times (HHMMSS).

    Parameters
    ----------
    unit : np.ndarray
        Array of unit interval times [0, 1].
    wall0 : float
        Start wall time.
    wall1 : float
        End wall time.

    Returns
    -------
    np.ndarray
        Wall clock times in HHMMSS format.

    Notes
    -----
    Ref: unit2wall.m:62-66 — chain: wall→seconds→unit→seconds→wall, then round.
    """
    wall0_sec = _wall2seconds(np.asarray(wall0, dtype=np.float64))
    wall1_sec = _wall2seconds(np.asarray(wall1, dtype=np.float64))
    seconds = wall0_sec + (wall1_sec - wall0_sec) * np.asarray(
        unit, dtype=np.float64
    )
    wall = _seconds2wall(seconds)
    # Ref: unit2wall.m:66 — round(100000*seconds2wall(seconds))/100000
    return np.round(wall * 100000.0) / 100000.0


def _unit2seconds(
    unit: np.ndarray, seconds0: float, seconds1: float
) -> np.ndarray:
    """Convert unit interval times back to seconds past midnight.

    Parameters
    ----------
    unit : np.ndarray
        Array of unit interval times [0, 1].
    seconds0 : float
        Start seconds.
    seconds1 : float
        End seconds.

    Returns
    -------
    np.ndarray
        Seconds past midnight values.

    Notes
    -----
    Ref: unit2seconds.m:63
    """
    unit = np.asarray(unit, dtype=np.float64)
    return float(seconds0) + (float(seconds1) - float(seconds0)) * unit


# ---------------------------------------------------------------------------
# Fast time filter (last-price interpolation)
# ---------------------------------------------------------------------------


def _fasttimefilter(
    price: np.ndarray,
    time: np.ndarray,
    filtered_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast last-price interpolation filter using a two-index algorithm.

    For each point in *filtered_time*, finds the last observation in
    (*price*, *time*) at or before that point.  This implements "last price
    interpolation" — if a requested sample time falls between observations,
    the most recent observed price is used.  If requested times are before
    the first observation, backfills with ``price[0]``.

    Parameters
    ----------
    price : np.ndarray
        1-D array of observed prices.
    time : np.ndarray
        1-D array of observation times (must be sorted ascending).
    filtered_time : np.ndarray
        1-D array of desired sample times (must be sorted ascending).

    Returns
    -------
    filtered_price : np.ndarray
        Prices sampled at *filtered_time* via last-price interpolation.
    actual_time : np.ndarray
        The actual observation time used for each entry in *filtered_price*.

    Notes
    -----
    Ref: realized_price_filter.m:241-281 — two-index walk-through algorithm.
    MATLAB uses 1-based indexing; this implementation uses 0-based indexing.
    """
    m = len(price)
    n = len(filtered_time)

    # Ref: realized_price_filter.m:250 — pl holds index values.
    # MATLAB initialises to ones(n,1) = index 1 (1-based first element).
    # Python equivalent: initialise to 0 (0-based first element) for backfill.
    pl = np.zeros(n, dtype=np.intp)

    time_idx = 0   # Ref: m:247 — timeIndex = 1 (1-based)
    ft_idx = 0     # Ref: m:248 — filteredTimeIndex = 1 (1-based)

    # Ref: realized_price_filter.m:251-271 — two-index walk-through
    while time_idx < m and ft_idx < n:
        if time[time_idx] <= filtered_time[ft_idx]:
            # Ref: m:252-254 — time observation falls at or before sample point
            pl[ft_idx] = time_idx
            time_idx += 1
        elif ft_idx < n - 1:
            # Ref: m:255-266 — advance ft_idx until filtered_time >= time
            while ft_idx < n - 1 and filtered_time[ft_idx] < time[time_idx]:
                ft_idx += 1
                # Last-price interpolation: carry forward previous index
                # Ref: m:261
                pl[ft_idx] = pl[ft_idx - 1]
            # Assign current time observation if not at the last filtered point
            # Ref: m:264-265
            if ft_idx < n - 1:
                pl[ft_idx] = time_idx
        elif time[time_idx] > filtered_time[n - 1]:
            # Ref: m:267-269 — no more filtered times to fill
            break

    # Ref: realized_price_filter.m:274-277 — clean up trailing inconsistencies.
    # Find positions where pl decreases (indicates an unfilled tail).
    diffs = np.diff(pl)
    neg_indices = np.where(diffs < 0)[0]
    if len(neg_indices) > 0:
        # Ref: m:275 — starter = find(diff(pl)<0) + 1  (MATLAB 1-based)
        # Python: neg_indices[0] is 0-based position in diffs;
        # the problematic index in pl is neg_indices[0] + 1.
        starter = neg_indices[0] + 1
        pl[starter:] = pl[starter - 1]

    # Ref: realized_price_filter.m:279-280
    filtered_price = price[pl]
    actual_time = time[pl]

    return filtered_price, actual_time


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def realized_price_filter(
    price: np.ndarray,
    time: np.ndarray,
    time_type: str,
    sampling_type: str,
    sampling_interval,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter prices according to a sampling scheme for realized estimation.

    Applies last-price interpolation to sample high-frequency prices at
    desired points in time.  Supports calendar-time, business-time (tick),
    and fixed-time sampling with automatic time format conversion.

    Parameters
    ----------
    price : np.ndarray
        1-D array of observed prices.
    time : np.ndarray
        1-D array of observation times corresponding to *price*.
        Must be sorted and non-decreasing.
    time_type : str
        Time format descriptor.  One of:

        * ``'wall'``    — 24-hour wall clock in HHMMSS format (e.g. 93000).
        * ``'seconds'`` — Seconds past midnight [0, 86400).
        * ``'unit'``    — Unit-normalised [0, 1].
    sampling_type : str
        Sampling scheme.  One of:

        * ``'CalendarTime'``     — Fixed calendar-time interval in seconds
          (or unit fraction when *time_type* = ``'unit'``).
        * ``'CalendarUniform'``  — *sampling_interval* uniformly-spaced
          points between first and last observation.
        * ``'BusinessTime'``     — Every *sampling_interval*-th tick.
        * ``'BusinessUniform'``  — *sampling_interval* ticks uniformly
          spread in business time.
        * ``'Fixed'``            — Sample at the specific times given in
          *sampling_interval*.
    sampling_interval : int, float, or np.ndarray
        Interpretation depends on *sampling_type*:

        * CalendarTime      — seconds between samples (or unit fraction).
        * CalendarUniform   — number of sample points.
        * BusinessTime      — ticks between samples.
        * BusinessUniform   — total number of samples.
        * Fixed             — 1-D array of sampling times.

    Returns
    -------
    filtered_price : np.ndarray
        Sampled prices.
    filtered_time : np.ndarray
        Sampling times in the original time format.
    actual_time : np.ndarray
        Actual observation times used for each sampled price.

    Raises
    ------
    ValueError
        If any input fails validation (invalid shapes, unsorted times, etc.).

    Notes
    -----
    Migrated from ``realized/realized_price_filter.m``.
    This is a helper function for ``realized_kernel`` and other realised
    volatility estimators.

    Examples
    --------
    >>> import numpy as np
    >>> p = np.array([100.0, 100.5, 101.0, 100.8, 101.2])
    >>> t = np.array([93000.0, 93500.0, 100000.0, 110000.0, 150000.0])
    >>> fp, ft, at = realized_price_filter(p, t, 'wall', 'CalendarTime', 3600)
    """
    # ==================================================================
    # Input Validation
    # Ref: realized_price_filter.m:92-152
    # ==================================================================

    # --- price ---
    price = np.asarray(price, dtype=np.float64)
    # Ref: m:93-96 — transpose row vector; error if matrix
    if price.ndim == 2:
        if min(price.shape) > 1:
            raise ValueError("PRICE must be a 1-D array (m,).")
        price = price.ravel()
    elif price.ndim > 2:
        raise ValueError("PRICE must be a 1-D array (m,).")
    if price.size == 0:
        raise ValueError("PRICE must be a non-empty 1-D array.")

    # --- time ---
    time = np.asarray(time, dtype=np.float64)
    if time.ndim == 2:
        if min(time.shape) > 1:
            raise ValueError("TIME must be a 1-D array with the same length as PRICE.")
        time = time.ravel()
    elif time.ndim > 2:
        raise ValueError("TIME must be a 1-D array with the same length as PRICE.")
    if time.size != price.size:
        raise ValueError("TIME must be a 1-D array with the same length as PRICE.")

    # Ref: m:104-108 — check sorting
    time_diff = np.diff(time)
    if np.any(time_diff < 0):
        raise ValueError("TIME must be sorted and increasing.")
    if np.any(time_diff == 0):
        # Ref: m:107-108 — MATLAB warning about duplicate timestamps
        warnings.warn(
            "TIME contains multiple entries with the same value. This creates "
            "an ambiguity and filtered_price will contain the last price if "
            "TIME does not contain only unique elements.",
            stacklevel=2,
        )

    # Ref: m:112-115 — validate time_type
    time_type_lower = time_type.lower()
    if time_type_lower not in ("wall", "seconds", "unit"):
        raise ValueError("time_type must be one of 'wall', 'seconds', or 'unit'.")

    # Ref: m:117 — ensure float64 (already done above)
    # Ref: m:119-122 — validate sampling_type
    sampling_type_lower = sampling_type.lower()
    valid_sampling_types = (
        "calendartime",
        "calendaruniform",
        "businesstime",
        "businessuniform",
        "fixed",
    )
    if sampling_type_lower not in valid_sampling_types:
        raise ValueError(
            "sampling_type must be one of 'CalendarTime', 'CalendarUniform', "
            "'BusinessTime', 'BusinessUniform', or 'Fixed'."
        )

    m = price.shape[0]
    t0_original = time[0]   # Ref: m:125
    tT_original = time[-1]  # Ref: m:126

    # Ref: m:127-142 — validate sampling_interval
    if sampling_type_lower in (
        "calendartime",
        "calendaruniform",
        "businesstime",
        "businessuniform",
    ):
        # Ref: m:129-131 — must be scalar positive integer (unless 'unit')
        _si_scalar = np.isscalar(sampling_interval) or (
            isinstance(sampling_interval, np.ndarray)
            and sampling_interval.ndim == 0
        )
        _si_val = float(sampling_interval)
        _is_integer = np.floor(_si_val) == _si_val
        if (
            not _si_scalar or not _is_integer or _si_val < 1
        ) and time_type_lower != "unit":
            raise ValueError(
                "sampling_interval must be a positive integer for the selected "
                "sampling_type when time_type is not 'unit'."
            )
    else:
        # 'fixed' — sampling_interval is a vector of times
        # Ref: m:133-141
        sampling_interval = np.asarray(sampling_interval, dtype=np.float64).ravel()
        if not (
            np.any(sampling_interval >= t0_original)
            and np.any(sampling_interval <= tT_original)
        ):
            raise ValueError(
                "At least one sampling interval must be between min(TIME) and "
                "max(TIME) when using 'Fixed' as sampling_type."
            )
        if sampling_interval.size > 1 and np.any(np.diff(sampling_interval) <= 0):
            raise ValueError(
                "When using 'Fixed' as sampling_type, the vector of sampling "
                "times in sampling_interval must be sorted and strictly increasing."
            )

    # Ref: m:144-149 — unit + CalendarTime constraint
    if time_type_lower == "unit" and sampling_type_lower == "calendartime":
        if float(sampling_interval) > 1:
            raise ValueError(
                "When time_type is 'unit' and sampling_type is 'CalendarTime', "
                "sampling_interval must be in 'unit' terms (between 0 and 1)."
            )

    # ==================================================================
    # Time Conversion to Unit Interval
    # Ref: realized_price_filter.m:154-177
    # ==================================================================

    # Ref: m:158-164 — determine time0 and time1
    if sampling_type_lower == "fixed":
        time0 = float(np.min(sampling_interval))
        time1 = float(np.max(sampling_interval))
    else:
        time0 = float(np.min(time))
        time1 = float(np.max(time))

    # Ref: m:165-169 — convert time to unit interval
    if time_type_lower == "wall":
        time = _wall2unit(time, time0, time1)
    elif time_type_lower == "seconds":
        time = _seconds2unit(time, time0, time1)
    # If 'unit', time is already in unit format — no conversion needed.

    # Ref: m:171-177 — convert fixed sampling times to unit
    if sampling_type_lower == "fixed":
        if time_type_lower == "wall":
            sampling_interval = _wall2unit(sampling_interval, time0, time1)
        elif time_type_lower == "seconds":
            sampling_interval = _seconds2unit(sampling_interval, time0, time1)

    # ==================================================================
    # Filtering Logic
    # Ref: realized_price_filter.m:180-228
    # ==================================================================

    t0 = time[0]       # Ref: m:181
    tT = time[-1]      # Ref: m:182 — time(m) in MATLAB (1-based)

    if sampling_type_lower == "calendartime":
        # Ref: m:185-201 — convert sampling interval to unit fraction
        if time_type_lower == "wall":
            # Ref: m:189-191
            time_base = _wall2seconds(np.array([time0, time1]))
            si = float(sampling_interval) / float(np.diff(time_base)[0])
        elif time_type_lower == "seconds":
            # Ref: m:192-194
            si = float(sampling_interval) / (time1 - time0)
        else:
            # 'unit' — already in unit terms
            si = float(sampling_interval)

        # Ref: m:195 — generate evenly-spaced filtered times: (t0:si:tT)
        # Mimic MATLAB colon operator with floating-point tolerance
        tol = 2.0 * np.finfo(np.float64).eps * max(abs(t0), abs(tT), 1.0)
        n_steps = int(np.floor((tT - t0 + tol) / si))
        filtered_time = t0 + np.arange(n_steps + 1) * si

        # Ref: m:197-199 — append final observation if not present
        if not np.any(np.abs(filtered_time - tT) < tol):
            filtered_time = np.append(filtered_time, tT)

        # Ref: m:201
        filtered_price, actual_time = _fasttimefilter(price, time, filtered_time)

    elif sampling_type_lower == "calendaruniform":
        # Ref: m:202-207 — sampling_interval = number of uniform samples
        filtered_time = np.linspace(t0, tT, int(sampling_interval))
        filtered_price, actual_time = _fasttimefilter(price, time, filtered_time)

    elif sampling_type_lower == "businesstime":
        # Ref: m:208-215 — every sampling_interval-th tick
        si_int = int(sampling_interval)
        # Ref: m:209 — MATLAB: indices=(1:samplingInterval:m)' (1-based)
        # Python 0-based equivalent:
        indices = np.arange(0, m, si_int)
        # Ref: m:210-212 — ensure last observation is included
        if indices[-1] != m - 1:
            indices = np.append(indices, m - 1)
        filtered_price = price[indices]
        filtered_time = time[indices]
        actual_time = filtered_time.copy()

    elif sampling_type_lower == "businessuniform":
        # Ref: m:216-221 — sampling_interval uniformly-spaced samples
        # Ref: m:218 — MATLAB: floor(linspace(1,m,samplingInterval)) (1-based)
        # Python 0-based: floor(linspace(0, m-1, samplingInterval))
        indices = np.floor(
            np.linspace(0, m - 1, int(sampling_interval))
        ).astype(np.intp)
        filtered_price = price[indices]
        filtered_time = time[indices]
        actual_time = filtered_time.copy()

    elif sampling_type_lower == "fixed":
        # Ref: m:222-225 — use given fixed times
        filtered_time = sampling_interval.copy()
        filtered_price, actual_time = _fasttimefilter(price, time, filtered_time)

    else:
        # Defensive — should never be reached after validation
        raise ValueError(f"Unrecognised sampling_type: {sampling_type}")

    # ==================================================================
    # Convert Times Back to Original Format
    # Ref: realized_price_filter.m:231-237
    # ==================================================================

    if time_type_lower == "wall":
        filtered_time = _unit2wall(filtered_time, time0, time1)
        actual_time = _unit2wall(actual_time, time0, time1)
    elif time_type_lower == "seconds":
        filtered_time = _unit2seconds(filtered_time, time0, time1)
        actual_time = _unit2seconds(actual_time, time0, time1)
    # If 'unit', times are already in unit format — no conversion.

    return filtered_price, filtered_time, actual_time
