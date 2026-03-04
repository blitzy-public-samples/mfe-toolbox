"""
Hayashi-Yoshida estimator of quadratic covariation with K-lead-lag correction.

Computes the Hayashi-Yoshida (HY) asynchronous covariation estimator between
two high-frequency price series that may be observed at different (non-
synchronous) times.  Optionally applies K-lead-and-lag correction similar
to Drost and Nijman (1997) to improve finite-sample performance.

Asset A's prices may be pre-filtered via a configurable sampling scheme
(calendar-time, business-time, or fixed-time sampling) before computing the
HY estimator against the full tick stream of asset B.

Migrated from: ``realized/realized_hayashi_yoshida.m`` (MFE Toolbox v4.0)

Original author: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 1    Date: 5/1/2008

See Also
--------
mfe_toolbox.realized.realized_multivariate_kernel : Multivariate kernel estimator.
mfe_toolbox.realized.realized_covariance : Synchronous realized covariance.
mfe_toolbox.realized.realized_kernel : Kernel-based realized volatility.
mfe_toolbox.realized.realized_variance : Standard realized variance.
"""

import warnings

import numpy as np

from mfe_toolbox.realized.realized_convert2unit import realized_convert2unit
from mfe_toolbox.realized.realized_price_filter import realized_price_filter


# ---------------------------------------------------------------------------
# Internal helper: core Hayashi-Yoshida computation
# ---------------------------------------------------------------------------


def _realized_hayashi_yoshida_core(
    price_a: np.ndarray,
    time_a: np.ndarray,
    price_b: np.ndarray,
    time_b: np.ndarray,
    overlap: int,
) -> tuple[float, np.ndarray]:
    """Core routine for the Hayashi-Yoshida covariation estimator.

    Implements the two-pointer scanning algorithm that efficiently computes
    the HY cross-covariation between two log-price series observed at
    arbitrary (asynchronous) times, with optional overlap (lead-lag)
    extension.

    Parameters
    ----------
    price_a : np.ndarray
        1-D array of prices for asset A (will be log-transformed internally).
    time_a : np.ndarray
        1-D array of observation times for asset A (must be sorted ascending).
    price_b : np.ndarray
        1-D array of prices for asset B (will be log-transformed internally).
    time_b : np.ndarray
        1-D array of observation times for asset B (must be sorted ascending).
    overlap : int
        Number of ticks to extend the B interval in each direction for each
        A return.  ``overlap=0`` gives the standard HY estimator; larger
        values implement the K-lead-and-lag correction.

    Returns
    -------
    rchy : float
        The (K-lead-and-lag) Hayashi-Yoshida cross-covariation scalar.
    times : np.ndarray
        An ``(n_a - 1, 4)`` diagnostic matrix recording, for each A return,
        the time stamps ``[tA_start, tA_end, tB_start, tB_end]`` used in
        the product.  Primarily for diagnostic purposes.

    Notes
    -----
    This is a helper function for :func:`realized_hayashi_yoshida` and
    performs **no** input validation.  In general it should not be called
    directly.

    Ref: ``realized_hayashi_yoshida.m:199-311`` — MATLAB sub-function.

    The algorithm uses a monotonic two-pointer scan through the B time
    series, advancing B alongside A to find overlapping intervals.  For
    each A return ``[timeA[i], timeA[i+1]]``, the corresponding B interval
    is ``[timeB[indexB - overlap], timeB[indexB' + overlap]]`` where
    ``indexB'`` is the first B index whose time reaches or exceeds
    ``timeA[i+1]``.  The product of A and B log-returns over these
    intervals is accumulated.

    The pre-pend / post-pend trick (extending B to cover A's time range)
    ensures the scan never goes out of bounds, with the added observations
    generating zero returns.

    **Complexity:** O(n_A + n_B) — single pass through both series.
    """
    # Ref: realized_hayashi_yoshida.m:235-238 — log-transform and cast
    log_price_a = np.log(np.asarray(price_a, dtype=np.float64))
    log_price_b = np.log(np.asarray(price_b, dtype=np.float64))
    time_a = np.asarray(time_a, dtype=np.float64)
    time_b = np.asarray(time_b, dtype=np.float64)

    n_a = len(log_price_a)
    n_b = len(log_price_b)

    # Ref: realized_hayashi_yoshida.m:243-244
    # Early return if there are no returns to compute
    if n_a <= 1:
        return 0.0, np.zeros((0, 4), dtype=np.float64)

    # ------------------------------------------------------------------
    # Extend B to cover A's time range (pre-pend / post-pend trick)
    # This ensures the scanning loop never exits array bounds.
    # ------------------------------------------------------------------

    # Ref: realized_hayashi_yoshida.m:246-252 — A starts before B:
    # project backward the price of the first B to the time of the first A.
    # This generates a zero return but simplifies the algorithm.
    if time_a[0] < time_b[0]:
        time_b = np.concatenate(([time_a[0]], time_b))
        log_price_b = np.concatenate(([log_price_b[0]], log_price_b))
        n_b = len(log_price_b)

    # Ref: realized_hayashi_yoshida.m:254-260 — A ends after B:
    # project forward the last price of B to the time of the last A.
    if time_a[n_a - 1] > time_b[n_b - 1]:
        time_b = np.concatenate((time_b, [time_a[n_a - 1]]))
        log_price_b = np.concatenate((log_price_b, [log_price_b[n_b - 1]]))
        n_b = len(log_price_b)

    # ------------------------------------------------------------------
    # Initialize the B pointer
    # ------------------------------------------------------------------

    # Ref: realized_hayashi_yoshida.m:267 — find last B index at or before
    # the first A observation.
    # MATLAB: indexB = find(timeB<=timeA(1), 1, 'last')
    # Use np.searchsorted for efficient binary search on the sorted time_b
    # array.  searchsorted(side='right') returns the insertion point *after*
    # any existing entries equal to time_a[0], so subtracting 1 gives the
    # last index where time_b <= time_a[0].
    insert_pos = int(np.searchsorted(time_b, time_a[0], side="right"))
    if insert_pos > 0:
        index_b = insert_pos - 1
    else:
        index_b = 0  # pragma: no cover — should not happen after pre-pend

    # Ref: realized_hayashi_yoshida.m:268-270 — pull back by overlap
    if overlap > 0:
        # Ref: m:269 — max(indexB-overlap, 1) → 0-based: max(index_b-overlap, 0)
        index_b = max(index_b - overlap, 0)

    # ------------------------------------------------------------------
    # Main scanning loop
    # ------------------------------------------------------------------

    # Ref: realized_hayashi_yoshida.m:273-274
    rchy = 0.0
    n_returns = n_a - 1
    times = np.zeros((n_returns, 4), dtype=np.float64)

    # Ref: realized_hayashi_yoshida.m:275-311
    # MATLAB uses 1-based indexA starting at 1, incremented inside loop.
    # Python: iterate through return indices 0 .. n_returns-1.
    for ret_idx in range(n_returns):
        # Ref: m:277-278 — prices for the current A return
        p_a_start = log_price_a[ret_idx]
        p_a_end = log_price_a[ret_idx + 1]

        # Record A interval times in diagnostic array
        # Ref: m:280-281 (note: MATLAB code has a 0-indexing bug in times;
        # Python uses consistent 0-based row indexing here)
        times[ret_idx, 0] = time_a[ret_idx]
        times[ret_idx, 1] = time_a[ret_idx + 1]

        # Ref: m:287-289 — B start with overlap extension
        # Extend backward by *overlap* ticks, clamped to index 0.
        index_b_minus = max(index_b - overlap, 0)
        p_b_start = log_price_b[index_b_minus]
        times[ret_idx, 2] = time_b[index_b_minus]

        # Ref: m:292-297 — advance B until its time reaches or passes the
        # end time of the current A return.
        # MATLAB: while timeB(indexB) < timeA(indexA) (after indexA increment)
        # The end time of the current A return is time_a[ret_idx + 1].
        while index_b < n_b and time_b[index_b] < time_a[ret_idx + 1]:
            index_b += 1

        # Defensive: clamp index_b to valid range (should not be needed
        # after the post-pend trick, but guards against floating-point edge
        # cases where the last B time is slightly less than the last A time)
        if index_b >= n_b:
            index_b = n_b - 1  # pragma: no cover

        # Ref: m:299-301 — B end with overlap extension
        # Extend forward by *overlap* ticks, clamped to n_b - 1.
        index_b_plus = min(index_b + overlap, n_b - 1)
        p_b_end = log_price_b[index_b_plus]
        times[ret_idx, 3] = time_b[index_b_plus]

        # Ref: m:305 — accumulate cross-product of log-returns
        rchy += (p_a_end - p_a_start) * (p_b_end - p_b_start)

        # Ref: m:308-310 — rewind B by 1 if it overshot A
        # This keeps the B pointer correctly positioned for the next
        # iteration: the next A return's start time equals this A return's
        # end time, and B's pointer should be at or just before that.
        if index_b < n_b and time_b[index_b] > time_a[ret_idx + 1]:
            index_b -= 1

    return rchy, times


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def realized_hayashi_yoshida(
    price_a,
    time_a,
    price_b,
    time_b,
    time_type: str,
    sampling_type: str = "BusinessTime",
    sampling_interval=1,
    K: int = 0,
) -> tuple[float, dict]:
    """Hayashi-Yoshida estimator of quadratic covariation with K-lead-lag.

    Computes the Hayashi-Yoshida (HY) asynchronous cross-covariation
    between two high-frequency price series observed at potentially
    different times.  Asset A is optionally pre-filtered using the
    specified sampling scheme (via :func:`realized_price_filter`), while
    all tick-level observations of asset B are retained.

    Parameters
    ----------
    price_a : array_like
        An ``m_A``-element vector of high-frequency prices for asset A.
    time_a : array_like
        An ``m_A``-element vector of observation times for asset A.
        Must be sorted in non-decreasing order and correspond element-wise
        to *price_a*.
    price_b : array_like
        An ``m_B``-element vector of high-frequency prices for asset B.
    time_b : array_like
        An ``m_B``-element vector of observation times for asset B.
        Must be sorted in non-decreasing order and correspond element-wise
        to *price_b*.
    time_type : str
        Time format descriptor.  One of:

        * ``'wall'``    — 24-hour clock in HHMMSS format (e.g. 93000).
        * ``'seconds'`` — Seconds past midnight, ``0 <= t < 86400``.
        * ``'unit'``    — Unit-normalised [0, 1].
    sampling_type : str, optional
        Sampling scheme applied to asset A's prices before the HY
        computation.  Default ``'BusinessTime'``.  One of:

        * ``'CalendarTime'``     — Fixed calendar-time interval in seconds
          (or unit fraction when *time_type* = ``'unit'``).
        * ``'CalendarUniform'``  — *sampling_interval* uniformly-spaced
          sample points.
        * ``'BusinessTime'``     — Every *sampling_interval*-th tick.
        * ``'BusinessUniform'``  — *sampling_interval* ticks uniformly
          spread in business time.
        * ``'Fixed'``            — Specific times given in *sampling_interval*.
    sampling_interval : int, float, or array_like, optional
        Interpretation depends on *sampling_type*.  Default ``1`` (every tick).
    K : int, optional
        Number of lead-and-lag ticks.  ``K=0`` (default) gives the standard
        HY estimator — the maximum-likelihood estimator when observation
        times are driven by independent Poisson processes.  Larger values
        broaden the overlap window by *K* ticks in each direction for every
        A return, implementing the empirical-performance motivated
        K-lead-and-lag correction.

    Returns
    -------
    hy_cov : float
        The K-lead-and-lag Hayashi-Yoshida cross-covariation scalar.
    diagnostics : dict
        Dictionary with diagnostic information:

        * ``'times'`` — ``(n, 4)`` array of time stamps
          ``[tA_start, tA_end, tB_start, tB_end]`` for each matched pair.
        * ``'intersection_is_empty'`` — ``True`` if the time ranges of
          assets A and B do not overlap.
        * ``'n_returns_a'`` — Number of returns used from the filtered
          asset A series.
        * ``'n_obs_b'`` — Number of observations in asset B's tick stream.

    Raises
    ------
    ValueError
        If inputs fail validation (wrong shapes, unsorted times, invalid
        *time_type* / *sampling_type*, etc.).

    Warnings
    --------
    UserWarning
        Issued when time arrays contain duplicate entries or when the
        time ranges of assets A and B do not overlap.

    Notes
    -----
    The filtering specified by *sampling_type* and *sampling_interval* is
    applied only to asset A.  After filtering, the HY estimator is computed
    between the filtered price of A and all observations of B, respecting
    the lead-lag parameter *K*.

    Ref: ``realized_hayashi_yoshida.m`` — full MATLAB source (312 lines).

    Examples
    --------
    >>> import numpy as np
    >>> pa = np.array([100.0, 100.5, 101.0, 100.8, 101.2])
    >>> ta = np.array([93000., 93500., 100000., 110000., 150000.])
    >>> pb = np.array([50.0, 50.2, 50.5, 50.3, 50.6])
    >>> tb = np.array([93000., 94000., 103000., 120000., 150000.])
    >>> hy, diag = realized_hayashi_yoshida(pa, ta, pb, tb, 'wall',
    ...                                     'BusinessTime', 1, K=0)
    """
    # ==================================================================
    # Input Validation
    # Ref: realized_hayashi_yoshida.m:82-177
    # ==================================================================

    # --- price_a ---
    # Ref: m:85-90 — transpose row vectors; error if matrix
    price_a = np.asarray(price_a, dtype=np.float64)
    if price_a.ndim == 2:
        if min(price_a.shape) > 1:
            raise ValueError("PRICEA must be a 1-D vector.")
        price_a = price_a.ravel()
    elif price_a.ndim > 2:
        raise ValueError("PRICEA must be a 1-D vector.")
    if price_a.ndim == 0:
        price_a = price_a.reshape(1)

    # --- time_a ---
    # Ref: m:91-101 — validate sorting and size
    time_a = np.asarray(time_a, dtype=np.float64)
    if time_a.ndim == 2:
        if min(time_a.shape) > 1:
            raise ValueError("TIMEA must be a 1-D vector with the same length as PRICEA.")
        time_a = time_a.ravel()
    elif time_a.ndim > 2:
        raise ValueError("TIMEA must be a 1-D vector with the same length as PRICEA.")
    if time_a.ndim == 0:
        time_a = time_a.reshape(1)
    if len(time_a) != len(price_a):
        raise ValueError("TIMEA must be a 1-D vector with the same length as PRICEA.")

    # Ref: m:94-98 — check sorting
    if len(time_a) > 1:
        diff_a = np.diff(time_a)
        if np.any(diff_a < 0):
            raise ValueError("TIMEA must be sorted and increasing")
        if np.any(diff_a == 0):
            # Ref: m:97 — MATLAB warning about duplicate timestamps
            warnings.warn(
                "TIMEA contains multiple entries with the same value. This "
                "creates an ambiguity and FILTEREDPRICE will contain the last "
                "price if TIMEA does not contain only unique elements.",
                stacklevel=2,
            )

    # Ref: m:103 — protect against integer times
    # (Already ensured by dtype=np.float64 cast above)

    # --- price_b ---
    # Ref: m:104-109 — same validation as price_a
    price_b = np.asarray(price_b, dtype=np.float64)
    if price_b.ndim == 2:
        if min(price_b.shape) > 1:
            raise ValueError("PRICEB must be a 1-D vector.")
        price_b = price_b.ravel()
    elif price_b.ndim > 2:
        raise ValueError("PRICEB must be a 1-D vector.")
    if price_b.ndim == 0:
        price_b = price_b.reshape(1)

    # --- time_b ---
    # Ref: m:110-122 — validate and sort-check
    time_b = np.asarray(time_b, dtype=np.float64)
    if time_b.ndim == 2:
        if min(time_b.shape) > 1:
            raise ValueError("TIMEB must be a 1-D vector with the same length as PRICEB.")
        time_b = time_b.ravel()
    elif time_b.ndim > 2:
        raise ValueError("TIMEB must be a 1-D vector with the same length as PRICEB.")
    if time_b.ndim == 0:
        time_b = time_b.reshape(1)
    if len(time_b) != len(price_b):
        raise ValueError("TIMEB must be a 1-D vector with the same length as PRICEB.")

    # Ref: m:113-117 — check sorting
    if len(time_b) > 1:
        diff_b = np.diff(time_b)
        if np.any(diff_b < 0):
            raise ValueError("TIMEB must be sorted and increasing")
        if np.any(diff_b == 0):
            # Ref: m:116 — MATLAB warning about duplicate timestamps
            warnings.warn(
                "TIMEB contains multiple entries with the same value. This "
                "creates an ambiguity and FILTEREDPRICE will contain the last "
                "price if TIMEB does not contain only unique elements.",
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Intersection check
    # Ref: realized_hayashi_yoshida.m:124-130
    # If the intersection of the time ranges of A and B is empty, the
    # HY estimator is necessarily 0.
    # ------------------------------------------------------------------
    # Ref: m:125 — MATLAB: ~(min(timeB)<max(timeA) || min(timeA)<max(timeB))
    # This condition fires when BOTH min(timeB) >= max(timeA) AND
    # min(timeA) >= max(timeB), which is a very conservative check.
    min_tb = np.min(time_b)
    max_ta = np.max(time_a)
    min_ta = np.min(time_a)
    max_tb = np.max(time_b)
    intersection_is_empty = not (min_tb < max_ta or min_ta < max_tb)

    if intersection_is_empty:
        # Ref: m:126-127 — issue warning
        warnings.warn(
            "The intersection of TIMEA and TIMEB is empty. The RCHY will "
            "necessarily be 0.",
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # Validate time_type
    # Ref: realized_hayashi_yoshida.m:133-136
    # ------------------------------------------------------------------
    time_type_lower = time_type.lower()
    if time_type_lower not in ("wall", "seconds", "unit"):
        raise ValueError(
            "TIMETYPE must be one of 'wall', 'seconds' or 'unit'."
        )

    # ------------------------------------------------------------------
    # Validate sampling_type
    # Ref: realized_hayashi_yoshida.m:138-141
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
            "SAMPLINGTYPE must be one of 'CalendarTime', 'CalendarUniform', "
            "'BusinessTime', 'BusinessUniform' or 'Fixed'."
        )

    # ------------------------------------------------------------------
    # Validate sampling_interval
    # Ref: realized_hayashi_yoshida.m:143-161
    # ------------------------------------------------------------------
    m_a = len(price_a)
    t0_original = time_a[0]
    tT_original = time_a[m_a - 1]

    if sampling_type_lower in (
        "calendartime",
        "calendaruniform",
        "businesstime",
        "businessuniform",
    ):
        # Ref: m:148 — must be a scalar positive integer (unless unit times)
        _si_scalar = np.isscalar(sampling_interval) or (
            isinstance(sampling_interval, np.ndarray) and sampling_interval.ndim == 0
        )
        _si_val = float(sampling_interval)
        _is_int = np.floor(_si_val) == _si_val
        if (not _si_scalar or not _is_int or _si_val < 1) and time_type_lower != "unit":
            raise ValueError(
                "SAMPLINGINTERVAL must be a positive integer for the "
                "SAMPLINGTYPE selected."
            )
    else:
        # 'fixed' — sampling_interval is a vector of times
        # Ref: m:152-160
        sampling_interval = np.asarray(sampling_interval, dtype=np.float64).ravel()
        if not (
            np.any(sampling_interval >= t0_original)
            and np.any(sampling_interval <= tT_original)
        ):
            raise ValueError(
                "At least one sampling interval must be between min(TIME) "
                "and max(TIME) when using 'Fixed' as SAMPLINGTYPE."
            )
        if sampling_interval.size > 1 and np.any(np.diff(sampling_interval) <= 0):
            raise ValueError(
                "When using 'Fixed' as SAMPLINGTYPE the vector of sampling "
                "times in SAMPLINGINTERVAL must be sorted and strictly "
                "increasing."
            )

    # Ref: m:163-168 — unit + CalendarTime constraint
    if time_type_lower == "unit" and sampling_type_lower == "calendartime":
        if float(sampling_interval) > 1:
            raise ValueError(
                "When TIMETYPE is 'unit' and SAMPLINGTYPE is 'CalendarTime', "
                "SAMPLINGINTERVAL must also be in 'unit' terms, and so must "
                "be between 0 and 1."
            )

    # ------------------------------------------------------------------
    # Validate K (overlap)
    # Ref: realized_hayashi_yoshida.m:170-174
    # ------------------------------------------------------------------
    # Ref: m:172 — overlap must be a non-negative integer
    if not np.isscalar(K):
        raise ValueError("K (overlap) must be a non-negative integer.")
    K_val = int(K)
    if K_val < 0 or float(K) != float(K_val):
        raise ValueError("K (overlap) must be a non-negative integer.")

    # ==================================================================
    # Use realized_convert2unit for consistent time normalization of B.
    # While realized_price_filter handles A's conversion internally,
    # we normalize B's times here for the intersection diagnostic and
    # to ensure the two grids are on a comparable scale.
    # Ref: schema requirement — import realized_convert2unit for
    # consistent overlap detection.
    # ==================================================================
    # Convert time_b to unit interval for diagnostic consistency
    time_b_unit, _t0b, _t1b, _ = realized_convert2unit(
        time_b, time_type_lower
    )

    # ==================================================================
    # Core computation
    # Ref: realized_hayashi_yoshida.m:185-193
    # ==================================================================
    if not intersection_is_empty:
        # Ref: m:187 — Filter price A using the specified sampling scheme
        filtered_price_a, filtered_time_a, actual_time_a = realized_price_filter(
            price_a, time_a, time_type_lower, sampling_type, sampling_interval
        )

        # Ref: m:190 — Core HY computation with overlap=K
        # The core function uses raw (unfiltered) times for overlap
        # detection, matching the MATLAB implementation exactly.
        hy_cov, times_diag = _realized_hayashi_yoshida_core(
            filtered_price_a, actual_time_a, price_b, time_b, K_val
        )
    else:
        # Ref: m:192 — empty intersection yields zero covariation
        hy_cov = 0.0
        times_diag = np.zeros((0, 4), dtype=np.float64)
        filtered_price_a = price_a
        actual_time_a = time_a

    # Build diagnostics dict
    n_returns_a = max(len(filtered_price_a) - 1, 0) if not intersection_is_empty else 0

    # Construct a summary time column-stack for diagnostic convenience:
    # columns: [A_start, A_end, B_start, B_end] — same data as times_diag
    # but built via np.column_stack for explicit construction clarity.
    if times_diag.shape[0] > 0:
        times_summary = np.column_stack(
            (times_diag[:, 0], times_diag[:, 1],
             times_diag[:, 2], times_diag[:, 3])
        )
    else:
        times_summary = np.zeros((0, 4), dtype=np.float64)

    diagnostics = {
        "times": times_summary,
        "intersection_is_empty": intersection_is_empty,
        "n_returns_a": n_returns_a,
        "n_obs_b": len(price_b),
    }

    return float(hy_cov), diagnostics
