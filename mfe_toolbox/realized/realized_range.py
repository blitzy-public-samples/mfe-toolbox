"""
Realized range estimator of quadratic variation.

Computes the realized range (RR) and optionally the subsampled realized range
(RR_SS) for a set of high-frequency prices.  The range estimator uses the
per-block high-low spread of log-prices, scaled by Monte Carlo-derived
constants to achieve an unbiased estimate of integrated variance.

Supports both overlapping and non-overlapping block constructions, multiple
sampling schemes (calendar-time, business-time, fixed), and subsampled
averaging for microstructure-noise reduction.

Migrated from: ``realized/realized_range.m`` (MFE Toolbox Version 4.0)

See Also
--------
mfe_toolbox.realized.realized_kernel : Realized kernel estimator.
mfe_toolbox.realized.realized_variance : Standard realized variance.
mfe_toolbox.realized.realized_quantile_variance : Quantile realized variance.
mfe_toolbox.realized.realized_price_filter : Price filtering utility.

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
# Internal helper: Monte Carlo scale factor lookup
# ---------------------------------------------------------------------------


def _realized_range_scale(samples_per_bin: int) -> float:
    """Return Monte Carlo-derived scale factor for the realized range estimator.

    The scale factors correct the raw squared range to produce an unbiased
    estimate of integrated variance.  They were computed from 1,000,000 Monte
    Carlo simulations and then smoothed using concave regression to enforce
    the theoretical concavity property.

    For very large block sizes the asymptotic value ``4 * ln(2)`` is used
    (the theoretical limit of the Brownian-motion range variance).

    Parameters
    ----------
    samples_per_bin : int
        Number of price observations in each block window.

    Returns
    -------
    float
        Scale factor ``c(m)`` such that ``E[range^2 / c(m)] = sigma^2 * dt``.

    Notes
    -----
    Ref: realized_range.m:235-276 — ``realized_range_scale`` sub-function.
    The lookup table ``m`` and ``scale`` are hardcoded from the file
    ``realized_range_simulation_results.mat`` (concave-regression-smoothed
    values).

    The original MATLAB code has an assignment bug on line 273 where the
    ``elseif`` branch (23401 <= spb < 100000) calls ``interp1`` but does
    not assign the result to ``scale``.  This Python version corrects the
    bug by using :func:`numpy.interp` over the full lookup table including
    the asymptotic point at ``m = 1e6``, which gives the intended behaviour
    for all block sizes.
    """
    # Ref: realized_range.m:238-240 — Block sizes used in Monte Carlo
    m_values: np.ndarray = np.array([
        2, 3, 4, 5, 6, 7, 9, 10, 11, 13, 14, 16, 19, 21, 25, 26,
        27, 31, 37, 40, 41, 46, 51, 53,
        61, 66, 73, 76, 79, 91, 101, 105, 118, 121, 131, 151, 157, 181,
        196, 201, 226, 235, 261, 301, 313, 326, 361, 391,
        451, 469, 521, 586, 601, 651, 781, 901, 937, 976, 1171, 1301,
        1561, 1801, 1951, 2341, 2601, 2926, 3901, 4681, 5851, 7801,
        11701, 23401,
    ], dtype=np.float64)

    # Ref: realized_range.m:255-262 — Smoothed (concave-regression) scale
    # These values were filtered through a concave regression to enforce the
    # concavity of the actual values from Monte Carlo integration with
    # 1,000,000 simulations.
    scale_values: np.ndarray = np.array([
        1.000000, 1.228112, 1.382860, 1.494147, 1.581668, 1.669189,
        1.756710, 1.804702, 1.851427,
        1.898152, 1.942636, 1.987119, 2.031603, 2.070209, 2.108815,
        2.134815, 2.160815, 2.186815,
        2.212815, 2.238815, 2.257754, 2.276692, 2.295631, 2.314570,
        2.333509, 2.352447, 2.366642,
        2.380837, 2.395032, 2.409227, 2.423421, 2.435522, 2.447622,
        2.459722, 2.471822, 2.483922,
        2.496022, 2.508122, 2.517765, 2.526830, 2.535895, 2.544959,
        2.554024, 2.563089, 2.570665,
        2.578241, 2.585817, 2.593393, 2.600969, 2.607880, 2.614791,
        2.621570, 2.628348, 2.635126,
        2.641904, 2.648682, 2.654688, 2.660694, 2.666700, 2.672706,
        2.678712, 2.684718, 2.690724,
        2.696730, 2.702736, 2.708742, 2.714748, 2.720754, 2.726760,
        2.732766, 2.738772, 2.744779,
    ], dtype=np.float64)

    # Ref: realized_range.m:268-269 — Append asymptotic point (4*ln(2))
    asymptotic_value = 4.0 * np.log(2.0)
    m_values = np.append(m_values, np.float64(1e6))
    scale_values = np.append(scale_values, np.float64(asymptotic_value))

    # Ref: realized_range.m:270-276 — Interpolation / clamping
    # For very large block sizes (>= 100000), return the asymptotic limit.
    # Otherwise, linearly interpolate from the lookup table.  np.interp
    # automatically clamps at the boundary values for extrapolation beyond
    # the table range.
    spb_float = float(samples_per_bin)
    if spb_float >= 100000.0:
        # Ref: realized_range.m:274-275 — asymptotic branch
        return float(asymptotic_value)
    else:
        # Ref: realized_range.m:271 — interp1(m, scale, samplesperbin)
        return float(np.interp(spb_float, m_values, scale_values))


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def realized_range(
    price,
    time=None,
    time_type: str = 'unit',
    sampling_type: str = 'CalendarTime',
    sampling_interval=1,
    samples_per_bin: int = 2,
    overlap: bool = True,
    subsamples: int = 1,
) -> tuple[float, float, dict]:
    """Compute realized range estimate of quadratic variation.

    Estimates quadratic variation (integrated variance) from high-frequency
    prices by summing scaled squared per-block ranges.  For each block of
    ``samples_per_bin`` consecutive log-prices, the range (max - min) is
    computed and squared, then divided by a Monte Carlo-derived scale factor
    ``c(m)`` to produce an unbiased variance estimate.

    Parameters
    ----------
    price : array_like
        An m-element vector of high-frequency prices.  Must be positive
        (log-prices are computed internally).
    time : array_like or None, optional
        An m-element vector of observation times corresponding to *price*.
        Must be sorted in non-decreasing order.  Format depends on
        *time_type*.  If ``None`` and ``time_type='unit'``, a uniformly
        spaced grid on [0, 1] is generated.
    time_type : str, optional
        Time format descriptor.  Case-insensitive.  One of:

        * ``'wall'``    -- 24-hour clock in HHMMSS format (e.g. 93000).
        * ``'seconds'`` -- Seconds past midnight.
        * ``'unit'``    -- Unit-normalised [0, 1] interval.

        Default is ``'unit'``.
    sampling_type : str, optional
        Sampling scheme used for price filtering.  Case-insensitive.  One of:

        * ``'CalendarTime'``     -- Fixed calendar-time interval.
        * ``'CalendarUniform'``  -- Uniform spacing in calendar time.
        * ``'BusinessTime'``     -- Every N-th tick.
        * ``'BusinessUniform'``  -- Uniform in business time.
        * ``'Fixed'``            -- Sample at user-specified times.

        Default is ``'CalendarTime'``.
    sampling_interval : int, float, or array_like, optional
        Interpretation depends on *sampling_type*.  See
        :func:`~mfe_toolbox.realized.realized_price_filter.realized_price_filter`
        for details.  Default is ``1``.
    samples_per_bin : int, optional
        Number of price observations in each block window for computing
        the high-low range.  Must be a positive integer >= 1.  The number
        of filtered prices *n* returned by the price filter must satisfy
        ``(n - 1) % (samples_per_bin - 1) == 0`` when ``overlap=False``.
        Default is ``2``.
    overlap : bool, optional
        If ``True`` (default), use overlapping blocks (sliding window).
        If ``False``, use disjoint (non-overlapping) blocks.
    subsamples : int, optional
        Number of subsamples for bias reduction.  Must satisfy
        ``0 < subsamples < samples_per_bin``.  Subsampling is only active
        when ``overlap=True``; a warning is issued if ``subsamples > 1``
        with ``overlap=False``.  Default is ``1`` (no subsampling).

    Returns
    -------
    rr : float
        Realized range estimate of quadratic variation.
    rr_ss : float
        Subsampled realized range estimate.  If ``subsamples == 1`` or
        ``overlap == False``, this equals *rr*.
    diagnostics : dict
        Dictionary containing diagnostic information:

        * ``'n_filtered'`` -- Number of filtered price observations.
        * ``'bins'`` -- Number of bins used in the range computation.
        * ``'expected_bins'`` -- Theoretical number of non-overlapping bins.
        * ``'overlap_scale'`` -- Overlap correction factor.
        * ``'scale_factor'`` -- Monte Carlo scale factor c(m).
        * ``'subsamples'`` -- Number of subsamples used.

    Raises
    ------
    ValueError
        If *price* is not a 1-D vector of positive values.
        If *time* is not sorted and increasing.
        If *time* length does not match *price* length.
        If *time_type* is not recognized.
        If *sampling_type* is not recognized.
        If *sampling_interval* is invalid for the given *sampling_type*.
        If *samples_per_bin* is not a positive integer.
        If *subsamples* is not a valid non-negative integer.
        If the filtered price count is incompatible with *samples_per_bin*
        when ``overlap=False``.

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.realized.realized_range import realized_range
    >>> np.random.seed(42)
    >>> prices = 100.0 * np.exp(np.cumsum(0.001 * np.random.randn(391)))
    >>> times = np.linspace(0, 1, 391)
    >>> rr, rr_ss, info = realized_range(
    ...     prices, times, 'unit', 'BusinessTime', 1, 31)
    >>> rr > 0
    True

    Notes
    -----
    **Algorithm (from realized_range.m):**

    1. Compute log-prices: ``log_price = log(price)``.
    2. Filter at the specified sampling grid via
       :func:`realized_price_filter`.
    3. Partition the *n* filtered log-prices into blocks of size
       *samples_per_bin*:

       * Overlap mode: sliding windows starting at indices 0, 1, ...,
         n - samples_per_bin - 1.
       * Non-overlap mode: disjoint windows starting at indices
         0, samples_per_bin - 1, 2*(samples_per_bin - 1), etc.

    4. For each block compute ``range_k = max(block) - min(block)``.
    5. Sum scaled squared ranges:
       ``RR = sum(range_k^2 / scale) / overlap_scale``
       where *scale* is the Monte Carlo-derived constant from
       :func:`_realized_range_scale` and *overlap_scale* adjusts for
       the higher count of overlapping bins.
    6. (Optional) Subsample: generate shifted grids via
       :func:`realized_subsample`, compute the range for each, and
       average with count weighting.

    Ref: realized_range.m:1-276 -- Full MATLAB source.
    """
    # ==================================================================
    # Input validation
    # Ref: realized_range.m:76-154
    # ==================================================================

    # --- Price validation ---
    # Ref: realized_range.m:79-84
    price = np.asarray(price, dtype=np.float64)
    if price.ndim == 2:
        # Ref: realized_range.m:79-80 — transpose row vector to column
        if price.shape[1] > price.shape[0]:
            price = price.T
        if price.shape[1] > 1:
            raise ValueError('PRICE must be a m by 1 vector.')
        price = price.ravel()
    elif price.ndim > 2:
        raise ValueError('PRICE must be a m by 1 vector.')
    else:
        price = price.ravel()

    if price.size == 0:
        raise ValueError('PRICE must be a non-empty vector.')

    # Check for NaN values in the price array
    if np.any(np.isnan(price)):
        raise ValueError('PRICE must not contain NaN values.')

    m: np.int64 = np.int64(price.shape[0])

    # --- Time validation ---
    # Ref: realized_range.m:85-95
    if time is None:
        # Python extension: generate default unit-interval times
        if time_type.lower() == 'unit':
            time = np.linspace(0.0, 1.0, int(m))
        else:
            raise ValueError(
                "TIME must be provided when time_type is not 'unit'."
            )

    time = np.asarray(time, dtype=np.float64)
    if time.ndim == 2:
        # Ref: realized_range.m:85-86 — transpose row to column
        if time.shape[1] > time.shape[0]:
            time = time.T
        if time.shape[1] > 1:
            raise ValueError('TIME must be a m by 1 vector.')
        time = time.ravel()
    elif time.ndim > 2:
        raise ValueError('TIME must be a m by 1 vector.')
    else:
        time = time.ravel()

    # Ref: realized_range.m:88-89 — TIME must be sorted and increasing
    if len(time) > 1 and np.any(np.diff(time) < 0):
        raise ValueError('TIME must be sorted and increasing')

    # Ref: realized_range.m:91-92 — Length match
    if len(time) != int(m):
        raise ValueError('TIME must be a m by 1 vector.')

    # Ref: realized_range.m:95 — Protect against integer-typed times
    time = time.astype(np.float64)

    # --- time_type validation ---
    # Ref: realized_range.m:97-100
    time_type = time_type.lower()
    if time_type not in ('wall', 'seconds', 'unit'):
        raise ValueError(
            "TIMETYPE must be one of 'wall', 'seconds' or 'unit'."
        )

    # --- sampling_type validation ---
    # Ref: realized_range.m:101-104
    sampling_type_lower = sampling_type.lower()
    if sampling_type_lower not in (
        'calendartime', 'calendaruniform',
        'businesstime', 'businessuniform', 'fixed',
    ):
        raise ValueError(
            "SAMPLINGTYPE must be one of 'CalendarTime', 'CalendarUniform', "
            "'BusinessTime', 'BusinessUniform' or 'Fixed'."
        )

    # --- sampling_interval validation ---
    # Ref: realized_range.m:106-124
    t0 = time[0]
    tT = time[int(m) - 1]

    if sampling_type_lower in (
        'calendartime', 'calendaruniform',
        'businesstime', 'businessuniform',
    ):
        # Ref: realized_range.m:111-113 — Must be a scalar positive integer
        if not np.isscalar(sampling_interval):
            raise ValueError(
                'SAMPLINGINTERVAL must be a positive integer for the '
                'SAMPLINGTYPE selected.'
            )
        si_val = float(sampling_interval)
        if np.floor(si_val) != si_val or si_val < 1:
            raise ValueError(
                'SAMPLINGINTERVAL must be a positive integer for the '
                'SAMPLINGTYPE selected.'
            )
    else:
        # Ref: realized_range.m:115-123 — 'Fixed' sampling type
        sampling_interval = np.asarray(
            sampling_interval, dtype=np.float64
        ).ravel()
        # Ref: realized_range.m:115-116 — Transpose if row vector
        if sampling_interval.ndim == 0:
            sampling_interval = sampling_interval.reshape(1)

        # Ref: realized_range.m:118-119 — At least one in [t0, tT]
        if not (np.any(sampling_interval >= t0)
                and np.any(sampling_interval <= tT)):
            raise ValueError(
                'At least one sampling interval must be between min(TIME) '
                "and max(TIME) when using 'Fixed' as SAMPLINGTYPE."
            )
        # Ref: realized_range.m:121-123 — Strictly increasing
        if len(sampling_interval) > 1 and np.any(
            np.diff(sampling_interval) <= 0
        ):
            raise ValueError(
                "When using 'Fixed' as SAMPLINGTYPE the vector of sampling "
                'times in SAMPLINGINTERVAL must be sorted and strictly '
                'increasing.'
            )

    # --- samples_per_bin validation ---
    # Ref: realized_range.m:126-128
    # MATLAB check: ~isscalar(samplesperbin) || samplesperbin<1 ||
    #               floor(samplesperbin)~=samplesperbin
    if (not np.isscalar(samples_per_bin)
            or float(samples_per_bin) < 1
            or np.floor(float(samples_per_bin)) != float(samples_per_bin)):
        raise ValueError('SAMPLESPERBIN must be a positive integer >= 2.')
    samples_per_bin = int(samples_per_bin)

    # --- overlap validation ---
    # Ref: realized_range.m:131-140
    if overlap is None:
        # Ref: realized_range.m:132-134
        overlap = True
    if not isinstance(overlap, (bool, int, float, np.bool_, np.integer)):
        raise ValueError('OVERLAP must be a scalar boolean value.')
    overlap = bool(overlap)

    # --- subsamples validation ---
    # Ref: realized_range.m:143-150
    if subsamples is None:
        subsamples = 1
    sub_float = float(subsamples)
    if (not np.isscalar(subsamples)
            or sub_float < 0
            or np.floor(sub_float) != sub_float
            or int(subsamples) >= samples_per_bin):
        raise ValueError(
            'SUBSAMPLES must be a non-negative integer less than '
            'SAMPLESPERBIN.'
        )
    subsamples = int(subsamples)

    # Ref: realized_range.m:148-149 — Default to 1 if 0 or empty
    if subsamples < 1:
        subsamples = 1

    # Ref: realized_range.m:152-153 — Warning for subsamples>1 with no overlap
    if subsamples > 1 and not overlap:
        warnings.warn(
            'SUBSAMPLE can only be used if OVERLAP = true',
            stacklevel=2,
        )

    # ==================================================================
    # Core computation
    # Ref: realized_range.m:160-198
    # ==================================================================

    # Ref: realized_range.m:161 — Log the price
    log_price = np.log(price)

    # Ref: realized_range.m:163 — Filter the log-price at the sampling grid
    filtered_log_price, _filt_time, _actual_time = realized_price_filter(
        log_price, time, time_type, sampling_type, sampling_interval
    )

    # Ref: realized_range.m:165 — Number of filtered prices
    n = len(filtered_log_price)

    # Ref: realized_range.m:166-170 — Check compatibility for non-overlap
    if not overlap:
        if samples_per_bin > 1:
            ratio = (n - 1) / (samples_per_bin - 1)
            # Ref: realized_range.m:167 — floor check for integer multiple
            if np.floor(ratio) != ratio:
                raise ValueError(
                    'The number of prices returned from '
                    'realized_price_filter(PRICE,TIME,TIMETYPE,SAMPLINGTYPE,'
                    'SAMPLINGINTERVAL) minus 1 must be an integer multiple '
                    'of SAMPLESPERBIN-1.'
                )

    # Ref: realized_range.m:173 — Expected number of non-overlapping bins
    if samples_per_bin > 1:
        expected_bins = (n - 1) / (samples_per_bin - 1)
    else:
        # Edge case: samples_per_bin == 1 → trivially n bins, each of size 1
        expected_bins = float(n)

    # Ref: realized_range.m:174-180 — Compute bin count and starting indices
    if overlap:
        # Ref: realized_range.m:175-176
        bins = n - samples_per_bin
        # Ref: realized_range.m:176 — binStart = 1:n-samplesperbin (MATLAB 1-based)
        # Python 0-based: 0, 1, ..., n-samples_per_bin-1
        bin_start = np.arange(bins, dtype=np.int64)
    else:
        # Ref: realized_range.m:178-179
        bins = int(expected_bins)
        # Ref: realized_range.m:179 — binStart = 1:(samplesperbin-1):(n-1)
        # Python 0-based: 0, (spb-1), 2*(spb-1), ...
        bin_start = np.arange(
            0, n - 1, samples_per_bin - 1, dtype=np.int64
        )

    # Handle degenerate case: no valid bins
    if bins <= 0:
        diagnostics = {
            'n_filtered': n,
            'bins': 0,
            'expected_bins': expected_bins,
            'overlap_scale': 1.0,
            'scale_factor': _realized_range_scale(samples_per_bin),
            'subsamples': subsamples,
        }
        return 0.0, 0.0, diagnostics

    # Ref: realized_range.m:181 — Overlap correction scale
    if expected_bins > 0:
        overlap_scale = float(bins) / float(expected_bins)
    else:
        overlap_scale = 1.0

    # Ref: realized_range.m:185 — Get Monte Carlo scale factor
    scale = _realized_range_scale(samples_per_bin)

    # Ref: realized_range.m:182-183 — Allocate hi/lo arrays
    hi = np.zeros(bins, dtype=np.float64)
    lo = np.zeros(bins, dtype=np.float64)

    # Ref: realized_range.m:187-193 — Loop over bins
    # MATLAB uses 1-based count variable; Python uses 0-based enumerate.
    for count, i_start in enumerate(bin_start):
        # Ref: realized_range.m:189 — pl = i:i+samplesperbin-1 (MATLAB 1-based)
        # Python 0-based: slice [i_start : i_start + samples_per_bin]
        block = filtered_log_price[int(i_start):int(i_start) + samples_per_bin]
        hi[count] = np.max(block)
        lo[count] = np.min(block)

    # Ref: realized_range.m:195 — Compute the range per bin
    range_vals = hi - lo

    # Ref: realized_range.m:197-198 — Sum scaled squared ranges and normalise
    rr = float(np.sum(range_vals ** 2 / scale))
    if overlap_scale > 0:
        rr = rr / overlap_scale

    # ==================================================================
    # Subsampling
    # Ref: realized_range.m:200-232
    # ==================================================================

    if not overlap:
        # Ref: realized_range.m:200-203 — No subsampling in non-overlap mode
        rr_ss = rr
    else:
        # Ref: realized_range.m:206 — Generate subsampled log-price grids
        subsampled_data = realized_subsample(
            log_price, time, time_type, sampling_type,
            sampling_interval, subsamples,
        )

        # Ref: realized_range.m:207 — Allocate per-subsample RR
        rrs = np.zeros(subsamples, dtype=np.float64)
        total_count: int = 0
        base_count: int = 0

        # Ref: realized_range.m:209-230 — Loop over all subsamples
        for i in range(subsamples):
            # Ref: realized_range.m:210 — Extract filtered log-prices
            # Python realized_subsample returns list of tuples:
            # (prices, times, base_count_sub, total_count_sub)
            sub_log_price: np.ndarray = subsampled_data[i][0]
            n_sub = len(sub_log_price)

            # Ref: realized_range.m:212 — Bin count for this subsample
            # Subsampling always uses overlap mode
            bins_sub = n_sub - samples_per_bin
            if bins_sub <= 0:
                # Degenerate subsample: not enough observations for a block
                warnings.warn(
                    f'Subsample {i} has only {n_sub} filtered prices, '
                    f'which is fewer than samples_per_bin={samples_per_bin}. '
                    'Skipping this subsample.',
                    stacklevel=2,
                )
                continue

            # Ref: realized_range.m:213 — binStart = 1:n-samplesperbin
            # Allocate fresh arrays per subsample (fixes MATLAB array-reuse
            # issue where hi/lo were not re-initialised between subsamples)
            sub_hi = np.zeros(bins_sub, dtype=np.float64)
            sub_lo = np.zeros(bins_sub, dtype=np.float64)

            # Ref: realized_range.m:215-220 — Inner loop over bins
            for j in range(bins_sub):
                # Ref: realized_range.m:216-218 — pl = j:j+samplesperbin-1
                block = sub_log_price[j:j + samples_per_bin]
                sub_hi[j] = np.max(block)
                sub_lo[j] = np.min(block)

            # Ref: realized_range.m:222 — Compute per-block range
            sub_range = sub_hi - sub_lo

            # Ref: realized_range.m:224 — Sum scaled squared ranges
            rrs[i] = float(np.sum(sub_range ** 2 / scale))

            # Ref: realized_range.m:226-228 — Base count from first subsample
            if i == 0:
                base_count = bins_sub

            # Ref: realized_range.m:229
            total_count += bins_sub

        # Ref: realized_range.m:231-232 — Weighted average across subsamples
        if total_count > 0 and base_count > 0:
            rr_ss = float(np.sum(rrs) * base_count / total_count)
            rr_ss = rr_ss / overlap_scale
        else:
            # Fallback if all subsamples were degenerate
            rr_ss = rr

    # ==================================================================
    # Build diagnostics dictionary
    # ==================================================================
    diagnostics: dict = {
        'n_filtered': n,
        'bins': bins,
        'expected_bins': expected_bins,
        'overlap_scale': overlap_scale,
        'scale_factor': scale,
        'subsamples': subsamples,
    }

    return rr, rr_ss, diagnostics
