"""
Multivariate realized covariance estimation.

Estimates Realized Covariance and subsampled Realized Covariance for
multiple assets using synchronized sampling and optional subsampling for
bias reduction.  The primary asset's price grid determines the sampling
times; all secondary assets are aligned to that grid using last-price
interpolation via ``'Fixed'`` sampling.

Migrated from: ``realized/realized_covariance.m`` (MFE Toolbox Version 4.0)

See Also
--------
mfe_toolbox.realized.realized_kernel : Realized kernel estimator.
mfe_toolbox.realized.realized_hayashi_yoshida : Hayashi-Yoshida asynchronous covariance.
mfe_toolbox.realized.realized_variance : Univariate realized variance.
mfe_toolbox.realized.realized_quantile_variance : Quantile-based realized variance.
mfe_toolbox.realized.realized_range : Range-based realized estimator.

Notes
-----
Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 1    Date: 7/1/2008
"""

import warnings

import numpy as np

from mfe_toolbox.realized.realized_convert2unit import realized_convert2unit
from mfe_toolbox.realized.realized_price_filter import realized_price_filter
from mfe_toolbox.realized.realized_subsample import realized_subsample


# ---------------------------------------------------------------------------
# Local helper functions
# ---------------------------------------------------------------------------


def _wall2seconds(wall: np.ndarray) -> np.ndarray:
    """Convert wall-clock times (HHMMSS.SS format) to seconds past midnight.

    This is a self-contained local helper used by
    :func:`_multivariate_convert2unit` to avoid importing the sibling
    module ``wall2seconds`` (which is not in this file's dependency list).

    Parameters
    ----------
    wall : np.ndarray
        Wall-clock times encoded as HH*10000 + MM*100 + SS.SS.

    Returns
    -------
    np.ndarray
        Seconds past midnight.

    Notes
    -----
    Ref: wall2seconds.m:36-48 — direct numerical translation.
    """
    wall = np.asarray(wall, dtype=np.float64)
    hr = np.floor(wall / 10000.0)
    # Ref: wall2seconds.m:39 — mm = floor(wall/100) - hr*100
    mm = np.floor(wall / 100.0) - hr * 100.0
    # Ref: wall2seconds.m:40 — ss = rem(wall, 100)
    ss = np.fmod(wall, 100.0)
    return 3600.0 * hr + 60.0 * mm + ss


def _multivariate_convert2unit(
    time: np.ndarray,
    time_type: str,
    time0: float,
    time1: float,
) -> np.ndarray:
    """Convert a secondary asset's time array to the unit [0, 1] interval.

    Uses the primary asset's time endpoints (*time0*, *time1*) as the
    reference bounds for the linear mapping, so that all assets share a
    consistent unit-interval coordinate system.

    This is a direct translation of the local MATLAB subfunction
    ``realized_multivariate_convert2unit`` found at the bottom of
    ``realized_covariance.m`` (lines 209-216).

    Parameters
    ----------
    time : np.ndarray
        1-D array of observation times for a secondary asset,
        in the **original** time format (*time_type*).
    time_type : str
        Time encoding format.  ``'wall'`` for HHMMSS wall-clock,
        ``'seconds'`` for seconds past midnight.  Any other value
        (including ``'unit'``) returns *time* unchanged.
    time0 : float
        Start time of the primary asset in the original format.
        Maps to 0.0 in the unit interval.
    time1 : float
        End time of the primary asset in the original format.
        Maps to 1.0 in the unit interval.

    Returns
    -------
    np.ndarray
        Times mapped to the [0, 1] unit interval.

    Notes
    -----
    Ref: realized_covariance.m:209-216 — local subfunction
    ``realized_multivariate_convert2unit``.

    For ``'wall'`` time type, the mapping is::

        wall → seconds → unit = (seconds - seconds0) / (seconds1 - seconds0)

    where ``seconds0 = wall2seconds(time0)`` and
    ``seconds1 = wall2seconds(time1)``.

    For ``'seconds'`` time type, the mapping is a direct linear rescale::

        unit = (time - time0) / (time1 - time0)
    """
    time = np.asarray(time, dtype=np.float64)
    if time_type == 'wall':
        # Ref: realized_covariance.m:211 — time = wall2unit(time, time0, time1)
        wall0_sec = float(_wall2seconds(np.asarray(time0, dtype=np.float64)))
        wall1_sec = float(_wall2seconds(np.asarray(time1, dtype=np.float64)))
        seconds = _wall2seconds(time)
        return (seconds - wall0_sec) / (wall1_sec - wall0_sec)
    elif time_type == 'seconds':
        # Ref: realized_covariance.m:213 — time = seconds2unit(time, time0, time1)
        return (time - float(time0)) / (float(time1) - float(time0))
    else:
        # time_type is 'unit' or unrecognised — return unchanged
        return time


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def realized_covariance(
    prices: list,
    times: list,
    time_type: str,
    sampling_type: str,
    sampling_interval,
    subsamples: int = 1,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Estimate Realized Covariance and subsampled Realized Covariance.

    Computes a *k* × *k* realized covariance matrix for *k* assets whose
    high-frequency prices are observed on potentially different (asynchronous)
    time grids.  The primary asset (first element) determines the sampling
    grid; all secondary assets are aligned to that grid using last-price
    interpolation (``'Fixed'`` sampling via :func:`realized_price_filter`).

    An optional subsampling procedure averages covariance matrices computed
    over multiple shifted grids, which can reduce finite-sample bias caused
    by microstructure noise.

    Parameters
    ----------
    prices : list of array_like
        A list of *k* one-dimensional price arrays, one per asset.
        ``prices[0]`` is treated as the primary asset whose sampling grid
        determines the filtered time stamps for all assets.
    times : list of array_like
        A list of *k* one-dimensional time arrays.  ``times[i]`` must have
        the same length as ``prices[i]``, be sorted in non-decreasing
        order, and be encoded according to *time_type*.
    time_type : str
        Time encoding descriptor (case-insensitive):

        * ``'wall'``    — 24-hour wall clock as HHMMSS (e.g. ``93000``).
        * ``'seconds'`` — Seconds past midnight (e.g. ``34200``).
        * ``'unit'``    — Unit-normalised [0, 1].
    sampling_type : str
        Sampling scheme for the primary asset (case-insensitive):

        * ``'CalendarTime'``     — Observations separated by
          *sampling_interval* seconds (or unit fraction).
        * ``'CalendarUniform'``  — *sampling_interval* observations
          uniformly spread between first and last time.
        * ``'Fixed'``            — Sample at specific times given by
          *sampling_interval*.
    sampling_interval : int, float, or array_like
        Meaning depends on *sampling_type*:

        * CalendarTime      — seconds between samples (or unit fraction).
        * CalendarUniform   — number of sample points.
        * Fixed             — 1-D sorted, strictly increasing vector of
          sample times in the same format as *times*.
    subsamples : int, optional
        Number of subsampled realized covariance estimators to average
        (default ``1``).  When *subsamples* ≤ 1 or is ``0``, the
        subsampled estimate equals the plain realized covariance.

        * ``subsamples=1`` — single subsample at the primary grid
          (identity; ``rc_ss == rc``).
        * ``subsamples=2`` — average over the primary grid and one
          midpoint-shifted grid.
        * ``subsamples=N`` — average over *N* uniformly shifted grids.

    Returns
    -------
    rc : numpy.ndarray
        *k* × *k* realized covariance matrix.
    rc_ss : numpy.ndarray
        *k* × *k* subsampled realized covariance matrix.  If
        *subsamples* ≤ 1, ``rc_ss`` equals ``rc``.
    diagnostics : dict
        Diagnostic information with keys:

        * ``'num_assets'``       — *k*, number of assets.
        * ``'num_filtered_obs'`` — number of synchronised observations
          after filtering.
        * ``'num_returns'``      — number of log-returns used for ``rc``.
        * ``'base_count'``       — returns in the first (base) subsample.
        * ``'total_count'``      — total returns across all subsamples.
        * ``'subsamples'``       — number of subsamples used.

    Raises
    ------
    ValueError
        If *prices* and *times* are not lists of equal length.
        If any price/time pair has mismatched lengths.
        If any time array is not sorted and increasing.
        If *time_type* is not one of ``'wall'``, ``'seconds'``, ``'unit'``.
        If *sampling_type* is not one of ``'CalendarTime'``,
        ``'CalendarUniform'``, ``'Fixed'``.
        If *sampling_interval* violates constraints for the selected
        *sampling_type* and *time_type* combination.
        If *subsamples* is not a non-negative integer.

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.realized.realized_covariance import realized_covariance
    >>> t = np.linspace(0.0, 1.0, 101)
    >>> rng = np.random.default_rng(42)
    >>> p1 = 100.0 * np.exp(np.cumsum(rng.standard_normal(101) * 0.001))
    >>> p2 = 50.0 * np.exp(np.cumsum(rng.standard_normal(101) * 0.001))
    >>> rc, rc_ss, info = realized_covariance(
    ...     [p1, p2], [t, t], 'unit', 'CalendarUniform', 20
    ... )
    >>> rc.shape
    (2, 2)
    >>> info['num_assets']
    2

    Notes
    -----
    **Core algorithm (from realized_covariance.m):**

    1. Convert all asset times to the unit [0, 1] interval via
       :func:`realized_convert2unit` (primary asset) and a local linear
       mapping (secondary assets using the primary asset's time bounds).
    2. Filter the primary asset's prices at the requested sampling grid
       using :func:`realized_price_filter`.
    3. Align all secondary assets to the primary's filtered time stamps
       using ``'Fixed'`` sampling in :func:`realized_price_filter`.
    4. Compute log-returns for each column of the synchronised price
       matrix: ``returns = np.diff(np.log(filtered_price), axis=0)``.
    5. The plain RC is the outer-product sum: ``rc = returns.T @ returns``.
    6. For subsampled RC, shift the sampling grid *subsamples* times via
       :func:`realized_subsample`, compute the RC on each shifted grid,
       and average with a bias-correction factor:
       ``rc_ss = sum(rcs) * (base_count / total_count)``.

    **MATLAB → Python translation notes:**

    * MATLAB ``varargin`` with interleaved price/time pairs → Python
      ``list`` arguments *prices* and *times*.
    * MATLAB ``cell2mat(...)`` → ``np.column_stack(...)`` or direct
      column assignment.
    * ``diff(log(x))`` → ``np.diff(np.log(x))``.
    * ``returns' * returns`` → ``returns.T @ returns``.
    * ``error(...)`` → ``raise ValueError(...)``.
    * Ref: realized_covariance.m:42 — MATLAB 1-indexed loops; Python
      0-indexed loops throughout.
    """
    # ==================================================================
    # Input Validation
    # Ref: realized_covariance.m:59-154
    # ==================================================================

    # --- Validate prices / times are lists ---
    # Ref: realized_covariance.m:63-75 — determine k from varargin.
    # In Python, k is simply len(prices).
    if not isinstance(prices, (list, tuple)):
        raise ValueError(
            'PRICES must be a list of 1-D price arrays, one per asset.'
        )
    if not isinstance(times, (list, tuple)):
        raise ValueError(
            'TIMES must be a list of 1-D time arrays, one per asset.'
        )
    k = len(prices)
    if len(times) != k:
        raise ValueError(
            'PRICES and TIMES must have the same number of elements.'
        )
    if k < 1:
        # Ref: realized_covariance.m:64-66 — 'At least 5 inputs required.'
        raise ValueError('At least one price-time pair is required.')

    # --- Parse and validate each price-time pair ---
    # Ref: realized_covariance.m:78-97
    price: list[np.ndarray] = []
    time: list[np.ndarray] = []
    for i in range(k):
        # Ref: realized_covariance.m:82-83 — extract and cast to double
        p_i = np.asarray(prices[i], dtype=np.float64).ravel()
        t_i = np.asarray(times[i], dtype=np.float64).ravel()

        # Ref: realized_covariance.m:84-89 — transpose row to column
        # Already handled by ravel() above.

        # Ref: realized_covariance.m:90-92 — length compatibility
        if p_i.shape[0] != t_i.shape[0]:
            raise ValueError(
                f'Problem with PRICE{i + 1} and TIME{i + 1}.  '
                f'Their lengths are not compatible or they are not '
                f'column vectors.'
            )

        # Ref: realized_covariance.m:93-95 — sorted and increasing
        if len(t_i) > 1 and np.any(np.diff(t_i) < 0):
            raise ValueError(
                f'TIME{i + 1} must be sorted and increasing.'
            )

        price.append(p_i)
        time.append(t_i)

    # --- Validate time_type ---
    # Ref: realized_covariance.m:108-111
    time_type = time_type.lower()
    if time_type not in ('wall', 'seconds', 'unit'):
        raise ValueError(
            "TIMETYPE must be one of 'wall', 'seconds' or 'unit'."
        )

    # --- Validate sampling_type ---
    # Ref: realized_covariance.m:113-116
    sampling_type_lower = sampling_type.lower()
    if sampling_type_lower not in ('calendartime', 'calendaruniform', 'fixed'):
        raise ValueError(
            "SAMPLINGTYPE must be one of 'CalendarTime', "
            "'CalendarUniform' or 'Fixed'."
        )

    # --- Validate sampling_interval ---
    # Ref: realized_covariance.m:119-145
    if sampling_type_lower in ('calendartime', 'calendaruniform'):
        # Ref: realized_covariance.m:121-129 — scalar validation
        if time_type in ('wall', 'seconds'):
            if (not np.isscalar(sampling_interval)
                    or np.floor(float(sampling_interval))
                    != float(sampling_interval)
                    or float(sampling_interval) < 1):
                raise ValueError(
                    'SAMPLINGINTERVAL must be a positive integer for '
                    'the SAMPLINGTYPE selected when using '
                    "'wall' or 'seconds' as TIMETYPE."
                )
        else:
            # Ref: realized_covariance.m:126-128 — 'unit' time type
            if (not np.isscalar(sampling_interval)
                    or float(sampling_interval) < 0):
                raise ValueError(
                    'SAMPLINGINTERVAL must be a positive value for '
                    'the SAMPLINGTYPE selected when using '
                    "'unit' as TIMETYPE."
                )
    else:
        # Ref: realized_covariance.m:131-144 — 'Fixed' sampling type
        sampling_interval = np.asarray(
            sampling_interval, dtype=np.float64
        ).ravel()

        for i in range(k):
            t0_i = time[i][0]
            tT_i = time[i][len(time[i]) - 1]
            # Ref: realized_covariance.m:138-139
            if not (np.any(sampling_interval >= t0_i)
                    and np.any(sampling_interval <= tT_i)):
                raise ValueError(
                    f'At least one sampling interval must be between '
                    f'min(TIME{i + 1}) and max(TIME{i + 1}) when using '
                    f"'Fixed' as SAMPLINGTYPE."
                )

        # Ref: realized_covariance.m:142-144 — strictly increasing
        if (len(sampling_interval) > 1
                and np.any(np.diff(sampling_interval) <= 0)):
            raise ValueError(
                "When using 'Fixed' as SAMPLINGTYPE the vector of "
                'sampling times in SAMPLINGINTERVAL must be sorted '
                'and strictly increasing.'
            )

    # --- Validate subsamples ---
    # Ref: realized_covariance.m:147-151
    if subsamples is None:
        subsamples = 1
    else:
        if (not np.isscalar(subsamples)
                or float(subsamples) < 0
                or np.floor(float(subsamples)) != float(subsamples)):
            raise ValueError(
                'SUBSAMPLES must be a non-negative scalar.'
            )
        subsamples = int(subsamples)

    # ==================================================================
    # Time Conversion to Unit Interval
    # Ref: realized_covariance.m:159-166
    # ==================================================================
    if time_type != 'unit':
        # Ref: realized_covariance.m:161 — convert primary asset
        time[0], time0, time1, sampling_interval = realized_convert2unit(
            time[0], time_type, sampling_type_lower, sampling_interval
        )
        # Ref: realized_covariance.m:162-165 — convert secondary assets
        # using the primary asset's time bounds (time0, time1)
        for j in range(1, k):
            # Ref: realized_covariance.m:164 — local subfunction call
            time[j] = _multivariate_convert2unit(
                time[j], time_type, time0, time1
            )

    # ==================================================================
    # Filter Prices and Compute Realized Covariance
    # Ref: realized_covariance.m:168-180
    # ==================================================================

    # Ref: realized_covariance.m:169 — filter primary asset prices
    filtered_price_0, filtered_times, _actual_times = realized_price_filter(
        price[0], time[0], 'unit', sampling_type_lower, sampling_interval
    )

    # Ref: realized_covariance.m:172 — number of filtered observations
    n = len(filtered_price_0)

    # Ref: realized_covariance.m:173 — build n × k filtered price matrix
    filtered_price = np.zeros((n, k), dtype=np.float64)
    filtered_price[:, 0] = filtered_price_0

    # Ref: realized_covariance.m:174-176 — align secondary assets to
    # primary asset's filtered time grid using 'Fixed' sampling
    for j in range(1, k):
        # Ref: realized_covariance.m:175
        filtered_price[:, j] = realized_price_filter(
            price[j], time[j], 'unit', 'fixed', filtered_times
        )[0]

    # Ref: realized_covariance.m:178 — log-returns
    returns = np.diff(np.log(filtered_price), axis=0)

    # Ref: realized_covariance.m:179 — RC = returns' * returns
    rc = returns.T @ returns

    # Ref: realized_covariance.m:180 — adjustment factor (diagnostic only)
    adj_factor = returns.shape[0]

    # ==================================================================
    # Subsampled Realized Covariance
    # Ref: realized_covariance.m:183-205
    # ==================================================================
    if subsamples < 1:
        # Ref: realized_covariance.m output comment (line 45-46):
        # "If SUBSAMPLE = 0 or is omitted, RCSUBSAMPLE = RC"
        rc_ss = rc.copy()
        base_count = adj_factor
        total_count = adj_factor
    else:
        # Ref: realized_covariance.m:183-187 — subsample each asset's
        # log prices using the primary asset's filtered time grid.
        subsampled_log_prices: list[list] = []
        for j in range(k):
            # Ref: realized_covariance.m:185 — logPrice = log(price{j})
            log_price_j = np.log(price[j])
            # Ref: realized_covariance.m:186 — call realized_subsample
            # NOTE: time_type is the ORIGINAL (lowercased) time type,
            # not 'unit'.  This matches the MATLAB code which passes the
            # un-modified timeType variable.  See detailed analysis in
            # module-level notes.  For values already in [0,1], the
            # wall2seconds / seconds2unit conversions inside
            # realized_subsample are effectively linear rescalings that
            # preserve relative positions — producing correct results.
            subsampled_j = realized_subsample(
                log_price_j, time[j], time_type,
                'fixed', filtered_times, subsamples
            )
            subsampled_log_prices.append(subsampled_j)

        # Ref: realized_covariance.m:189 — initialize 3-D covariance array
        rcs = np.zeros((k, k, subsamples), dtype=np.float64)
        total_count = 0
        base_count = 0

        # Ref: realized_covariance.m:191-204 — iterate over subsamples
        for i in range(subsamples):
            # Ref: realized_covariance.m:192 — number of obs in subsample i
            # realized_subsample returns list of tuples:
            #   (subsampled_prices, subsampled_times, base_count, total_count)
            # We access [0] for the price array.
            m_i = len(subsampled_log_prices[0][i][0])
            all_log_prices = np.zeros((m_i, k), dtype=np.float64)

            # Ref: realized_covariance.m:194-196 — assemble multi-asset
            # subsampled log-price matrix
            for j in range(k):
                all_log_prices[:, j] = subsampled_log_prices[j][i][0]

            # Ref: realized_covariance.m:198 — log-returns from subsampled
            # log prices (prices are already in log form, so diff gives
            # log-returns directly)
            returns_ss = np.diff(all_log_prices, axis=0)

            # Ref: realized_covariance.m:199-201 — track counts
            if i == 0:
                base_count = returns_ss.shape[0]
            total_count += returns_ss.shape[0]

            # Ref: realized_covariance.m:203 — covariance for subsample i
            rcs[:, :, i] = returns_ss.T @ returns_ss

        # Ref: realized_covariance.m:205 — bias-corrected average
        if total_count > 0:
            rc_ss = np.sum(rcs, axis=2) * (base_count / total_count)
        else:
            # Defensive: if no observations in any subsample, fall back
            rc_ss = rc.copy()

    # ==================================================================
    # Build diagnostics dictionary
    # ==================================================================
    diagnostics: dict = {
        'num_assets': k,
        'num_filtered_obs': n,
        'num_returns': adj_factor,
        'base_count': base_count,
        'total_count': total_count,
        'subsamples': subsamples,
    }

    return rc, rc_ss, diagnostics
