"""
Convert time arrays to unit [0, 1] interval for realized volatility estimators.

This module provides the :func:`realized_convert2unit` function, which maps
time arrays from wall-clock (HHMMSS) or seconds-past-midnight format to the
unit [0, 1] interval.  It also converts sampling intervals to the same unit
scale when the sampling type is ``'CalendarTime'`` or ``'Fixed'``.

This is **critical infrastructure** used by virtually all realized volatility
estimators in the toolbox (e.g., :func:`realized_kernel`,
:func:`realized_variance`, :func:`realized_price_filter`).

Migrated from: ``realized/realized_convert2unit.m`` (MFE Toolbox Version 4.0)

See Also
--------
mfe_toolbox.realized.wall2unit : Wall clock HHMMSS to unit [0, 1].
mfe_toolbox.realized.seconds2unit : Seconds past midnight to unit [0, 1].
mfe_toolbox.realized.wall2seconds : Wall clock HHMMSS to seconds past midnight.
mfe_toolbox.realized.realized_kernel : Primary consumer of time conversion.
mfe_toolbox.realized.realized_variance : Realized variance estimator.
mfe_toolbox.realized.realized_price_filter : Price filtering with time normalization.

Notes
-----
Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 1    Date: 5/1/2008
"""

import warnings

import numpy as np

from mfe_toolbox.realized.seconds2unit import seconds2unit
from mfe_toolbox.realized.wall2seconds import wall2seconds
from mfe_toolbox.realized.wall2unit import wall2unit


def realized_convert2unit(
    time,
    time_type: str,
    sampling_type: str | None = None,
    sampling_interval=None,
) -> tuple[np.ndarray, float, float, np.ndarray | float | None]:
    """
    Convert time arrays from wall-clock or seconds format to unit [0, 1] interval.

    Helper function that converts wall and seconds times to unit times,
    assigning 0 to the smallest time and 1 to the largest.  Also converts
    sampling intervals to the unit scale when applicable.

    Parameters
    ----------
    time : array_like
        An m-element vector of times where ``time[i]`` corresponds to
        ``price[i]``.  Must be sorted in non-decreasing order.  The
        format must match *time_type*:

        * ``'wall'`` — 24-hour clock in HHMMSS format (e.g. 101543)
        * ``'seconds'`` — seconds past midnight (e.g. 36943)
        * ``'unit'`` — already in [0, 1] (pass-through)
    time_type : str
        Describes how *time* values are encoded.  Case-insensitive.
        Accepted values: ``'wall'``, ``'seconds'``, ``'unit'``.
    sampling_type : str or None, optional
        Describes the sampling strategy applied during price filtering.
        Case-insensitive.  Accepted values:

        * ``'CalendarTime'`` — sample in calendar time using observations
          separated by *sampling_interval* seconds.
        * ``'CalendarUniform'`` — *sampling_interval* observations
          uniformly spread between ``time[0]`` and ``time[-1]``.
        * ``'BusinessTime'`` — sample every *sampling_interval* ticks.
        * ``'BusinessUniform'`` — uniform spacing in tick time.
        * ``'Fixed'`` — sample at specific time points supplied in
          *sampling_interval*.

        If ``None``, no sampling-interval conversion is performed.
    sampling_interval : scalar, array_like, or None, optional
        Scalar integer or vector whose semantics depend on *sampling_type*:

        * For ``'CalendarTime'``/``'CalendarUniform'``/``'BusinessTime'``/
          ``'BusinessUniform'``: a positive integer.
        * For ``'Fixed'``: a sorted, strictly increasing 1-D vector of
          times in the same format as *time*.

        If ``None``, no sampling-interval conversion is performed.

    Returns
    -------
    unit_time : numpy.ndarray
        The original times mapped to the [0, 1] interval.
    time0 : float
        The base time in the original format (minimum of *time* or
        *sampling_interval* for ``'Fixed'``).  Required to invert the
        transformation.
    time1 : float
        The final time in the original format (maximum of *time* or
        *sampling_interval* for ``'Fixed'``).  Required to invert the
        transformation.
    converted_sampling_interval : numpy.ndarray, float, or None
        The sampling interval converted to the unit scale.  For
        ``'CalendarTime'`` and ``'Fixed'`` types the interval is
        rescaled; for other types it is returned unchanged.

    Raises
    ------
    ValueError
        If *time* is not sorted in non-decreasing order.
        If *time_type* is not ``'wall'``, ``'seconds'``, or ``'unit'``.
        If *sampling_type* is not a recognized sampling strategy.
        If *sampling_interval* is invalid for the given *sampling_type*.

    Warnings
    --------
    UserWarning
        Issued when *time* contains duplicate entries (same as MATLAB
        ``warning`` on source line 64).

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.realized.realized_convert2unit import realized_convert2unit
    >>> t = np.array([93000.0, 100000.0, 120000.0, 160000.0])
    >>> ut, t0, t1, si = realized_convert2unit(t, 'wall', 'BusinessTime', 1)
    >>> t0
    93000.0
    >>> t1
    160000.0

    Notes
    -----
    Ref: realized_convert2unit.m — Full source (134 lines) implements
    validation, time-type branching, and sampling-interval normalization.

    Ref: realized_convert2unit.m:55 — MATLAB requires ``nargin == 4``.
    The Python version makes *sampling_type* and *sampling_interval*
    optional so that the ``'unit'`` pass-through case can be invoked
    with fewer arguments.
    """
    # ----------------------------------------------------------------
    # Input coercion
    # Ref: realized_convert2unit.m:58-60 — if row vector, transpose
    # Ref: realized_convert2unit.m:70 — time = double(time)
    # Protects against integer-typed inputs by casting to float64.
    # ----------------------------------------------------------------
    time = np.asarray(time, dtype=np.float64).ravel()

    # ----------------------------------------------------------------
    # Monotonicity check
    # Ref: realized_convert2unit.m:61-65
    #   if any(diff(time)<0) → error
    #   elseif any(diff(time)==0) → warning
    # ----------------------------------------------------------------
    if len(time) > 1:
        diffs = np.diff(time)
        if np.any(diffs < 0):
            raise ValueError('TIME must be sorted and increasing')
        if np.any(diffs == 0):
            # Ref: realized_convert2unit.m:64 — MATLAB warning id
            # 'oxfordRealized:realizedPriceFilter'
            warnings.warn(
                'TIME contains multiple entries with the same value. '
                'This creates an ambiguity and FILTEREDPRICE will '
                'contain the last price if TIME does not only contain '
                'unique elements.',
                stacklevel=2,
            )

    # Ref: realized_convert2unit.m:66-68 — TIME must be m by 1 vector
    # After ravel() the array is guaranteed 1-D; no further check needed.

    # ----------------------------------------------------------------
    # Validate time_type
    # Ref: realized_convert2unit.m:72-75
    # MATLAB only accepts 'wall'/'seconds'.  The Python API additionally
    # supports 'unit' for pass-through (time already in [0,1]).
    # ----------------------------------------------------------------
    time_type = time_type.lower()
    if time_type not in {'wall', 'seconds', 'unit'}:
        raise ValueError(
            "TIMETYPE must be one of 'wall', 'seconds', or 'unit'."
        )

    # ----------------------------------------------------------------
    # Validate sampling_type (case-insensitive)
    # Ref: realized_convert2unit.m:76-79
    # ----------------------------------------------------------------
    _valid_sampling_types = {
        'calendartime', 'calendaruniform',
        'businesstime', 'businessuniform', 'fixed',
    }
    if sampling_type is not None:
        sampling_type = sampling_type.lower()
        if sampling_type not in _valid_sampling_types:
            raise ValueError(
                "SAMPLINGTYPE must be one of 'CalendarTime', "
                "'CalendarUniform', 'BusinessTime', 'BusinessUniform' "
                "or 'Fixed'."
            )

    # ----------------------------------------------------------------
    # Store original time endpoints (before any conversion)
    # Ref: realized_convert2unit.m:81-83
    #   m = size(time, 1);
    #   t0Original = time(1);
    #   tTOriginal = time(m);
    # ----------------------------------------------------------------
    m = len(time)
    t0_original = time[0] if m > 0 else np.float64(0.0)
    t_t_original = time[m - 1] if m > 0 else np.float64(0.0)

    # ----------------------------------------------------------------
    # Validate sampling_interval based on sampling_type
    # Ref: realized_convert2unit.m:84-98
    # ----------------------------------------------------------------
    if sampling_type is not None and sampling_interval is not None:
        if sampling_type in {
            'calendartime', 'calendaruniform',
            'businesstime', 'businessuniform',
        }:
            # Ref: realized_convert2unit.m:85-87 — scalar positive integer
            if not np.isscalar(sampling_interval):
                raise ValueError(
                    'SAMPLINGINTERVAL must be a positive integer for '
                    'the SAMPLINGTYPE selected.'
                )
            si_val = float(sampling_interval)
            if np.floor(si_val) != si_val or si_val < 1:
                raise ValueError(
                    'SAMPLINGINTERVAL must be a positive integer for '
                    'the SAMPLINGTYPE selected.'
                )
        elif sampling_type == 'fixed':
            # Ref: realized_convert2unit.m:89-91 — transpose if row
            sampling_interval = np.asarray(
                sampling_interval, dtype=np.float64
            ).ravel()

            # Ref: realized_convert2unit.m:92-94
            # At least one value must be between min(TIME) and max(TIME)
            if not (
                np.any(sampling_interval >= t0_original)
                and np.any(sampling_interval <= t_t_original)
            ):
                raise ValueError(
                    'At least one sampling interval must be between '
                    "min(TIME) and max(TIME) when using 'Fixed' as "
                    'SAMPLINGTYPE.'
                )

            # Ref: realized_convert2unit.m:95-97 — strictly increasing
            if len(sampling_interval) > 1 and np.any(
                np.diff(sampling_interval) <= 0
            ):
                raise ValueError(
                    "When using 'Fixed' as SAMPLINGTYPE the vector of "
                    'sampling times in SAMPLINGINTERVAL must be sorted '
                    'and strictly increasing.'
                )

    # ----------------------------------------------------------------
    # Determine time bounds (time0, time1) for conversion
    # Ref: realized_convert2unit.m:105-111
    #   if strcmp(samplingType,'fixed')
    #       time0 = min(samplingInterval);
    #       time1 = max(samplingInterval);
    #   else
    #       time0 = min(time);
    #       time1 = max(time);
    #   end
    # ----------------------------------------------------------------
    if (
        sampling_type == 'fixed'
        and sampling_interval is not None
        and isinstance(sampling_interval, np.ndarray)
        and len(sampling_interval) > 0
    ):
        # Ref: realized_convert2unit.m:106-107
        time0 = float(np.min(sampling_interval))
        time1 = float(np.max(sampling_interval))
    else:
        # Ref: realized_convert2unit.m:109-110
        time0 = float(np.min(time)) if m > 0 else 0.0
        time1 = float(np.max(time)) if m > 0 else 0.0

    # ----------------------------------------------------------------
    # Convert time to unit [0, 1] interval
    # Ref: realized_convert2unit.m:112-116
    #   if strcmp(timeType,'wall')
    #       time = wall2unit(time, time0, time1);
    #   elseif strcmp(timeType,'seconds')
    #       time = seconds2unit(time, time0, time1);
    #   end
    # ----------------------------------------------------------------
    if time_type == 'wall':
        # Ref: realized_convert2unit.m:113
        time = wall2unit(time, time0, time1)
    elif time_type == 'seconds':
        # Ref: realized_convert2unit.m:115
        time = seconds2unit(time, time0, time1)
    # time_type == 'unit': pass through — already in [0, 1]

    # ----------------------------------------------------------------
    # Convert Fixed sampling interval to unit scale
    # Ref: realized_convert2unit.m:118-124
    #   if strcmp(samplingType,'fixed')
    #       if strcmp(timeType,'wall')
    #           samplingInterval = wall2unit(samplingInterval,time0,time1);
    #       elseif strcmp(timeType,'seconds')
    #           samplingInterval = seconds2unit(...);
    #       end
    #   end
    # ----------------------------------------------------------------
    if (
        sampling_type == 'fixed'
        and sampling_interval is not None
        and isinstance(sampling_interval, np.ndarray)
    ):
        if time_type == 'wall':
            # Ref: realized_convert2unit.m:120
            sampling_interval = wall2unit(sampling_interval, time0, time1)
        elif time_type == 'seconds':
            # Ref: realized_convert2unit.m:122
            sampling_interval = seconds2unit(
                sampling_interval, time0, time1
            )
        # time_type == 'unit': Fixed interval already in unit scale

    # ----------------------------------------------------------------
    # Convert CalendarTime sampling interval to unit fraction
    # Ref: realized_convert2unit.m:125-132
    #   if strcmp(samplingType, 'calendartime')
    #       if strcmp(timeType,'wall')
    #           timeBase = wall2seconds([time0 time1]);
    #           samplingInterval = samplingInterval / diff(timeBase);
    #       elseif strcmp(timeType,'seconds')
    #           samplingInterval = samplingInterval / (time1 - time0);
    #       end
    #   end
    # ----------------------------------------------------------------
    if sampling_type == 'calendartime' and sampling_interval is not None:
        if time_type == 'wall':
            # Ref: realized_convert2unit.m:127-128
            time_base = wall2seconds(np.array([time0, time1]))
            # Ref: realized_convert2unit.m:128 — divide by diff(timeBase)
            diff_base = float(np.diff(time_base)[0])
            sampling_interval = float(sampling_interval) / diff_base
        elif time_type == 'seconds':
            # Ref: realized_convert2unit.m:130
            sampling_interval = float(sampling_interval) / (time1 - time0)
        # time_type == 'unit': CalendarTime interval is ambiguous in unit
        # scale; leave sampling_interval unchanged to avoid silent errors.

    return time, time0, time1, sampling_interval
