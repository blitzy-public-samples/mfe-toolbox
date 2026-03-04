"""
Subsampling grid generation for realized volatility estimators.

This module provides the :func:`realized_subsample` function, which generates
shifted price grids for averaging-based bias reduction in realized volatility
estimation.  The subsampling scheme creates multiple price/time grids, each
starting from a different offset within the sampling interval.

For *K* subsamples, the original filtered grid is subsample 0.  Subsamples
1 through *K*-1 shift the sampling grid by ``k / K`` of each inter-sample
gap (calendar/uniform/fixed) or by ``floor(gap / K)`` ticks (business time).
Averaging the realized variance across all *K* subsamples yields the
subsampled (or "average") realized variance estimator, which reduces
finite-sample bias due to microstructure noise.

Migrated from: ``realized/realized_subsample.m`` (MFE Toolbox Version 4.0)

See Also
--------
mfe_toolbox.realized.realized_variance : Realized variance estimator.
mfe_toolbox.realized.realized_twoscale_variance : Two-scale RV estimator.
mfe_toolbox.realized.realized_price_filter : Core price filtering function.
mfe_toolbox.realized.realized_convert2unit : Time unit conversion helper.

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
from mfe_toolbox.realized.unit2wall import unit2wall
from mfe_toolbox.realized.unit2seconds import unit2seconds


def realized_subsample(
    price,
    time,
    time_type: str,
    sampling_type: str,
    sampling_interval,
    sub_samples: int,
) -> list[tuple[np.ndarray, np.ndarray, int, int]]:
    """Generate shifted price grids for subsampled realized volatility estimation.

    Creates multiple subsampled price/time grids by shifting the sampling
    points within each sampling interval.  This enables averaging across
    multiple starting points for bias reduction in realized volatility
    estimation.

    Parameters
    ----------
    price : array_like
        An m-element 1-D vector of observed (log) prices.  If a row vector
        is provided, it is transposed to a column vector.
    time : array_like
        An m-element 1-D vector of observation times corresponding to
        *price*.  Must be sorted in non-decreasing order.  The format
        must match *time_type*:

        * ``'wall'``    — 24-hour clock in HHMMSS format (e.g. 93000).
        * ``'seconds'`` — Seconds past midnight (e.g. 34200).
        * ``'unit'``    — Unit-normalised [0, 1].
    time_type : str
        Time format descriptor.  Case-insensitive.  One of ``'wall'``,
        ``'seconds'``, or ``'unit'``.
    sampling_type : str
        Sampling scheme.  Case-insensitive.  One of:

        * ``'CalendarTime'``     — Fixed calendar-time interval.
        * ``'CalendarUniform'``  — Uniform spacing in calendar time.
        * ``'BusinessTime'``     — Every N-th tick.
        * ``'BusinessUniform'``  — Uniform spacing in tick (business) time.
        * ``'Fixed'``            — Sample at specific user-supplied times.
    sampling_interval : int, float, or array_like
        Interpretation depends on *sampling_type*:

        * CalendarTime / CalendarUniform — positive integer (seconds or
          ticks between samples, or unit fraction for ``'unit'``).
        * BusinessTime / BusinessUniform — positive integer (tick gap or
          number of samples).
        * Fixed — 1-D sorted, strictly increasing vector of sample times
          in the same format as *time*.

        See :func:`realized_price_filter` for full details.
    sub_samples : int
        Number of subsamples to generate.  Must be a positive integer (≥ 1).

    Returns
    -------
    list of tuple[numpy.ndarray, numpy.ndarray, int, int]
        A list of length *sub_samples*.  Each element is a tuple
        ``(subsampled_prices, subsampled_times, base_count, total_count)``
        where:

        * ``subsampled_prices`` — 1-D ``float64`` array of filtered prices
          for this subsample.
        * ``subsampled_times`` — 1-D ``float64`` array of filtered times
          for this subsample (in the original *time_type* format when
          *sub_samples* > 1, or in unit interval when *sub_samples* = 1).
        * ``base_count`` — Number of filtered observations in this
          subsample (``len(subsampled_prices)``).
        * ``total_count`` — Total number of original price observations
          (``len(price)``).

    Raises
    ------
    ValueError
        If *price* is not a 1-D vector.
        If *time* is not sorted and increasing.
        If *time* length does not match *price* length.
        If *time_type* is not a recognized format.
        If *sampling_type* is not a recognized scheme.
        If *sampling_interval* is invalid for the given *sampling_type*
        and *time_type* combination.
        If *sub_samples* is not a positive integer.

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.realized.realized_subsample import realized_subsample
    >>> prices = np.array([100.0, 100.5, 101.0, 100.8, 101.2, 101.5])
    >>> times = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    >>> result = realized_subsample(prices, times, 'unit', 'CalendarTime', 0.5, 2)
    >>> len(result)
    2
    >>> result[0][2]  # base_count for first subsample
    3
    >>> result[0][3]  # total_count (original observations)
    6

    Notes
    -----
    **Core algorithm (from realized_subsample.m):**

    For calendar-time / calendar-uniform / fixed sampling types:

    1. Filter prices at the primary (unshifted) grid.
    2. Compute the vector of inter-sample gaps:
       ``gap = diff(filtered_times)``.
    3. For each subsample offset *k* = 1, …, *sub_samples* − 1:

       * Shift the time grid by ``k * gap / sub_samples``.
       * Re-filter prices at the shifted grid using ``'fixed'`` sampling.

    For business-time / business-uniform sampling types:

    1. Filter prices at the primary (unshifted) grid.
    2. Compute the tick gap and step:
       ``step = floor(gap / sub_samples)``.
    3. For each subsample offset *k* = 1, …, *sub_samples* − 1:

       * Compute shifted tick indices starting at ``step * k``.
       * Extract prices and times at those indices.

    **MATLAB → Python translation notes:**

    * MATLAB cell arrays → Python ``list[tuple]``.
    * ``logPrice{i}`` → ``result[i][0]``.
    * ``for i = 1:subsamples`` → ``for i in range(sub_samples)``.
    * ``error()`` → ``raise ValueError()``.
    * MATLAB 1-based indexing → Python 0-based indexing.

    Ref: realized_subsample.m:71 — MATLAB ``cell(1, length(subSamples+1))``
    has a latent sizing bug (``length(scalar+1) == 1``); MATLAB auto-expands
    cell arrays on assignment.  Python list avoids this entirely.

    Ref: realized_subsample.m:94-96 — When *sub_samples* = 1, the MATLAB
    code returns early before the time-format post-processing loop.  This
    behaviour is preserved for strict numerical parity.
    """
    # ==================================================================
    # Input Validation
    # Ref: realized_subsample.m:6-64
    # ==================================================================

    # Ref: realized_subsample.m:6-8 — Six inputs required
    # Python enforces this via the function signature.

    # Ref: realized_subsample.m:9-11 — Transpose row to column
    price = np.asarray(price, dtype=np.float64).ravel()

    # Ref: realized_subsample.m:12-14 — PRICE must be m by 1 vector
    # After ravel(), the array is guaranteed 1-D; no further shape check.

    # Ref: realized_subsample.m:15-17 — Transpose row to column for time
    time = np.asarray(time, dtype=np.float64).ravel()

    # Ref: realized_subsample.m:18-20 — TIME must be sorted and increasing
    if len(time) > 1 and np.any(np.diff(time) < 0):
        raise ValueError('TIME must be sorted and increasing')

    # Ref: realized_subsample.m:21-23 — TIME length must match PRICE length
    if len(time) != len(price):
        raise ValueError('TIME must be a m by 1 vector.')

    # Ref: realized_subsample.m:25 — Cast to double (protect against ints)
    # Already handled by np.asarray(..., dtype=np.float64) above.

    # Ref: realized_subsample.m:27-30 — Validate timeType
    time_type = time_type.lower()
    if time_type not in ('wall', 'seconds', 'unit'):
        raise ValueError(
            "TIMETYPE must be one of 'wall', 'seconds' or 'unit'."
        )

    # Ref: realized_subsample.m:31-34 — Validate samplingType
    sampling_type = sampling_type.lower()
    _valid_sampling_types = (
        'calendartime', 'calendaruniform',
        'businesstime', 'businessuniform', 'fixed',
    )
    if sampling_type not in _valid_sampling_types:
        raise ValueError(
            "SAMPLINGTYPE must be one of 'CalendarTime', 'CalendarUniform', "
            "'BusinessTime', 'BusinessUniform' or 'Fixed'."
        )

    # Ref: realized_subsample.m:36-38 — Compute m, t0, tT
    m = len(price)
    t0 = time[0]
    tT = time[m - 1]

    # Ref: realized_subsample.m:39-60 — Validate samplingInterval
    if sampling_type in ('calendartime', 'calendaruniform',
                         'businesstime', 'businessuniform'):
        # Ref: realized_subsample.m:41-49 — Scalar validation branching on
        # timeType for wall/seconds vs unit.
        if time_type in ('wall', 'seconds'):
            # Ref: realized_subsample.m:42-44 — positive scalar integer
            if (not np.isscalar(sampling_interval)
                    or np.floor(float(sampling_interval)) != float(sampling_interval)
                    or float(sampling_interval) < 1):
                raise ValueError(
                    'SAMPLINGINTERVAL must be a positive integer for the '
                    "SAMPLINGTYPE selected when using 'wall' or 'seconds' "
                    'as TIMETYPE.'
                )
        else:
            # Ref: realized_subsample.m:46-48 — positive scalar (unit)
            if not np.isscalar(sampling_interval) or float(sampling_interval) < 0:
                raise ValueError(
                    'SAMPLINGINTERVAL must be a positive value for the '
                    "SAMPLINGTYPE selected when using 'unit' as TIMETYPE."
                )
    else:
        # Ref: realized_subsample.m:51-59 — Fixed sampling type
        sampling_interval_arr = np.asarray(
            sampling_interval, dtype=np.float64
        ).ravel()

        # Ref: realized_subsample.m:54-56 — At least one value in range
        if not (np.any(sampling_interval_arr >= t0)
                and np.any(sampling_interval_arr <= tT)):
            raise ValueError(
                'At least one sampling interval must be between min(TIME) '
                "and max(TIME) when using 'Fixed' as SAMPLINGTYPE."
            )

        # Ref: realized_subsample.m:57-59 — Strictly increasing
        if (len(sampling_interval_arr) > 1
                and np.any(np.diff(sampling_interval_arr) <= 0)):
            raise ValueError(
                "When using 'Fixed' as SAMPLINGTYPE the vector of sampling "
                'times in SAMPLINGINTERVAL must be sorted and strictly '
                'increasing.'
            )

    # Ref: realized_subsample.m:62-64 — Validate subSamples
    _sub_samples_valid = True
    if sub_samples is None:
        _sub_samples_valid = False
    elif not np.isscalar(sub_samples):
        _sub_samples_valid = False
    else:
        _ss_float = float(sub_samples)
        if _ss_float < 1 or np.floor(_ss_float) != _ss_float:
            _sub_samples_valid = False

    if not _sub_samples_valid:
        raise ValueError('SUBSAMPLES must be a positive scalar.')
    sub_samples = int(sub_samples)

    # ==================================================================
    # Initialize output list
    # Ref: realized_subsample.m:71-72 — cell array initialisation
    # MATLAB ``cell(1, length(subSamples+1))`` has a latent bug:
    # ``length(scalar+1)`` evaluates to 1, creating a 1×1 cell.
    # MATLAB auto-expands the cell array on later ``{i}`` assignment.
    # Python list avoids this entirely.
    # ==================================================================
    result: list[tuple[np.ndarray, np.ndarray, int, int]] = []

    # ==================================================================
    # Convert times to unit [0,1] if not already unit
    # Ref: realized_subsample.m:74-77
    # ==================================================================
    time0: float | None = None
    time1: float | None = None

    if time_type != 'unit':
        # Ref: realized_subsample.m:76
        time, time0, time1, sampling_interval = realized_convert2unit(
            time, time_type, sampling_type, sampling_interval
        )

    # Ref: realized_subsample.m:79-80 — Save original timeType; set to unit
    original_time_type = time_type
    time_type = 'unit'

    # Ref: realized_subsample.m:82 — m is the length of price
    m = len(price)

    # ==================================================================
    # First (unshifted) filter pass
    # Ref: realized_subsample.m:86-89
    # ==================================================================
    # Ref: realized_subsample.m:86 — MATLAB returns 2 values; Python
    # realized_price_filter returns 3 (filtered_price, filtered_time,
    # actual_time).  The third output (actual observation times used) is
    # not needed here, so it is discarded.
    filtered_price, filtered_times, _actual = realized_price_filter(
        price, time, time_type, sampling_type, sampling_interval
    )

    # Ref: realized_subsample.m:87 — Preserve unshifted times for shift
    # computation in the calendar/fixed branch.
    full_filtered_times = filtered_times.copy()

    # Ref: realized_subsample.m:88-89 — Store first subsample
    result.append((
        filtered_price.copy(),
        filtered_times.copy(),
        int(len(filtered_price)),
        int(m),
    ))

    # Ref: realized_subsample.m:92 — Number of filtered time points
    n = len(filtered_times)

    # Ref: realized_subsample.m:94-96 — Early return for single subsample.
    # NOTE: The MATLAB code returns *before* the post-processing loop that
    # converts times from unit back to the original format.  This means
    # that for sub_samples=1 with time_type='wall' or 'seconds', the
    # returned times are in unit [0,1] interval rather than the original
    # format.  We preserve this behaviour for strict numerical parity.
    if sub_samples == 1:
        return result

    # ==================================================================
    # Compute shifted grids for subsamples 2..sub_samples
    # Ref: realized_subsample.m:98-133
    # ==================================================================

    if sampling_type in ('calendartime', 'calendaruniform', 'fixed'):
        # ---------------------------------------------------------------
        # Calendar-time / calendar-uniform / fixed case
        # Ref: realized_subsample.m:99-112
        #
        # Uniform sampling in calendar time: the gap vector is the
        # difference between consecutive filtered times.  Each subsample
        # shifts the grid by an equal fraction of the gap.
        # ---------------------------------------------------------------

        # Ref: realized_subsample.m:101 — gap = diff(filteredTimes)
        gap = np.diff(filtered_times)

        # Ref: realized_subsample.m:103 — step = gap / subSamples
        # Element-wise division: each inter-sample interval is divided
        # into sub_samples equal parts.
        step = gap / float(sub_samples)

        # Ref: realized_subsample.m:104-112
        # MATLAB loop: for i = 2:subSamples → (i-1) = 1..(subSamples-1)
        # Python loop: for i in range(1, sub_samples) → i = 1..(sub_samples-1)
        for i in range(1, sub_samples):
            # Ref: realized_subsample.m:105-107
            # thisSampleTime = fullFilteredTimes(1:n-1) + (i-1)*step
            # MATLAB (i-1) maps to Python i since Python loop starts at 1.
            # MATLAB (1:n-1) selects elements 1 through n-1 (1-based) =
            # first n-1 elements; Python [:n-1] = elements 0 through n-2.
            this_sample_time = full_filtered_times[:n - 1] + i * step

            # Ref: realized_subsample.m:109 — Re-filter at shifted grid
            # using 'fixed' sampling type with the shifted time points.
            fp, ft, _at = realized_price_filter(
                price, time, 'unit', 'fixed', this_sample_time
            )

            # Ref: realized_subsample.m:110-111
            result.append((
                fp.copy(),
                ft.copy(),
                int(len(fp)),
                int(m),
            ))

    elif sampling_type in ('businesstime', 'businessuniform'):
        # ---------------------------------------------------------------
        # Business-time / business-uniform case
        # Ref: realized_subsample.m:113-133
        #
        # Sampling is linear in ticks.  The gap is the number of ticks
        # between consecutive samples.  Each subsample shifts the
        # starting tick index by an equal offset.
        # ---------------------------------------------------------------

        if sampling_type == 'businesstime':
            # Ref: realized_subsample.m:115-117 — gap = samplingInterval
            # (number of ticks to skip)
            gap = float(sampling_interval)
        else:
            # Ref: realized_subsample.m:118-120 — gap = m / samplingInterval
            # (average gap between uniform samples)
            gap = float(m) / float(sampling_interval)

        # Ref: realized_subsample.m:123 — step = floor(gap / subSamples)
        step = int(np.floor(gap / float(sub_samples)))

        # Warn if step is zero (degenerate subsampling)
        if step == 0 and sub_samples > 1:
            warnings.warn(
                'Subsampling step size is zero because '
                'floor(gap / sub_samples) = 0.  All business-time '
                'subsamples beyond the first will start at the same '
                'offset, producing identical or near-identical grids.',
                stacklevel=2,
            )

        # Ref: realized_subsample.m:124-133
        # MATLAB loop: for i = 2:subSamples → (i-1) = 1..(subSamples-1)
        # Python loop: for i in range(1, sub_samples) → i = 1..(sub_samples-1)
        for i in range(1, sub_samples):
            # Ref: realized_subsample.m:127
            # MATLAB: indices = floor(step*(i-1) : gap : m)
            # With MATLAB i starting at 2, (i-1) ranges 1..(subSamples-1).
            # Python i ranges 1..(sub_samples-1), matching MATLAB (i-1).
            #
            # The MATLAB colon operator a:b:c generates values from a,
            # incrementing by b, up to and including c.  These are 1-based
            # indices into the price/time arrays.
            matlab_offset = float(step * i)

            if gap > 0 and matlab_offset <= m:
                # Number of elements in the MATLAB range offset:gap:m
                n_elem = int(np.floor((float(m) - matlab_offset) / gap)) + 1
                if n_elem > 0:
                    raw = matlab_offset + np.arange(n_elem) * gap
                    # Ref: realized_subsample.m:127 — floor() for float gap
                    indices_1based = np.floor(raw).astype(np.intp)

                    # Ref: realized_subsample.m:128
                    # indices = indices(indices>0 & indices<=m)
                    valid_mask = (indices_1based > 0) & (indices_1based <= m)
                    indices_1based = indices_1based[valid_mask]

                    if len(indices_1based) > 0:
                        # Convert MATLAB 1-based to Python 0-based
                        indices_0based = indices_1based - 1

                        # Ref: realized_subsample.m:129-130
                        fp = price[indices_0based].copy()
                        ft = time[indices_0based].copy()
                    else:
                        fp = np.array([], dtype=np.float64)
                        ft = np.array([], dtype=np.float64)
                else:
                    fp = np.array([], dtype=np.float64)
                    ft = np.array([], dtype=np.float64)
            else:
                fp = np.array([], dtype=np.float64)
                ft = np.array([], dtype=np.float64)

            # Ref: realized_subsample.m:131-132
            result.append((
                fp,
                ft,
                int(len(fp)),
                int(m),
            ))

    # ==================================================================
    # Post-processing: convert times from unit back to original format
    # Ref: realized_subsample.m:137-148
    # ==================================================================
    # Ref: realized_subsample.m:137 — Restore original timeType
    time_type = original_time_type

    eps_val = np.finfo(np.float64).eps

    processed: list[tuple[np.ndarray, np.ndarray, int, int]] = []
    for prices_i, times_i, base_count_i, total_count_i in result:
        # Ref: realized_subsample.m:139-143 — Convert unit→wall or
        # unit→seconds if the original timeType was not 'unit'.
        if time_type == 'wall' and time0 is not None and time1 is not None:
            # Ref: realized_subsample.m:140
            # unit2wall requires at least 2 elements.  For robustness,
            # handle edge cases where a subsample has fewer than 2 points.
            if len(times_i) >= 2:
                times_i = unit2wall(times_i, time0, time1)
            elif len(times_i) == 1:
                # Inline single-element conversion using the same formula
                # as unit2wall: wall→seconds→linear map→seconds→wall→round.
                # Ref: unit2wall.m:62-66
                _w0 = float(time0)
                _w1 = float(time1)
                _hr0 = np.floor(_w0 / 10000.0)
                _mm0 = np.floor(_w0 / 100.0) - _hr0 * 100.0
                _ss0 = np.fmod(_w0, 100.0)
                _sec0 = 3600.0 * _hr0 + 60.0 * _mm0 + _ss0

                _hr1 = np.floor(_w1 / 10000.0)
                _mm1 = np.floor(_w1 / 100.0) - _hr1 * 100.0
                _ss1 = np.fmod(_w1, 100.0)
                _sec1 = 3600.0 * _hr1 + 60.0 * _mm1 + _ss1

                _secs = _sec0 + (_sec1 - _sec0) * times_i
                _hr = np.floor(_secs / 3600.0)
                _mm_val = np.floor((_secs - _hr * 3600.0) / 60.0)
                _ss_val = np.fmod(_secs, 60.0)
                _wall = _hr * 10000.0 + _mm_val * 100.0 + _ss_val
                times_i = np.round(_wall * 100000.0) / 100000.0
            # len == 0: no conversion needed (empty array)

        elif time_type == 'seconds' and time0 is not None and time1 is not None:
            # Ref: realized_subsample.m:142
            if len(times_i) > 0:
                times_i = unit2seconds(times_i, time0, time1)

        # Ref: realized_subsample.m:144-148 — Digit rounding to clean up
        # floating-point noise from the unit→original conversion.
        if time_type != 'unit' and len(times_i) > 0:
            max_time = np.max(times_i)
            if max_time > 0:
                # Ref: realized_subsample.m:145
                # digits = -log10(max(subsampledTimes{i}) * eps) - 1
                digits = -np.log10(max_time * eps_val) - 1.0
                # Ref: realized_subsample.m:146
                digits = max(digits, 0.0)
                # Ref: realized_subsample.m:147
                scale = 10.0 ** digits
                times_i = np.floor(times_i * scale) / scale

        processed.append((prices_i, times_i, base_count_i, total_count_i))

    return processed
