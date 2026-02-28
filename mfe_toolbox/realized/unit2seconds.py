"""
Convert unit interval [0,1] times to seconds past midnight.

This module provides the inverse conversion of seconds2unit, mapping fractional
trading day times back to seconds-past-midnight representation.

Migrated from realized/unit2seconds.m (MFE Toolbox, Version 4.0, Kevin Sheppard)

See Also
--------
seconds2unit : Inverse operation (seconds to unit interval)
wall2unit : Wall clock string to unit interval
wall2seconds : Wall clock string to seconds
seconds2wall : Seconds to wall clock string
realized_kernel : Primary consumer of time conversion utilities
realized_variance : Realized variance estimator
"""

import numpy as np


def unit2seconds(unit: np.ndarray, seconds0: float, seconds1: float) -> np.ndarray:
    """Convert unit interval [0,1] times to seconds past midnight.

    Maps fractional times in the unit interval [0, 1] to seconds past midnight
    using a linear mapping where 0 maps to seconds0 and 1 maps to seconds1.
    This is the inverse operation of seconds2unit.

    Parameters
    ----------
    unit : array_like
        m-element array of times measured as fraction of time between
        seconds0 and seconds1. Values should be in [0, 1] for meaningful
        results within the trading day.
    seconds0 : float
        Base time in seconds past midnight that maps to 0 in the unit interval.
        Must be non-negative.
    seconds1 : float
        End time in seconds past midnight that maps to 1 in the unit interval.
        Must be non-negative and strictly greater than seconds0.

    Returns
    -------
    np.ndarray
        1D array of float64 times expressed as seconds past midnight
        (e.g., 1:00:00 AM is 3600.0, 12:00:15 PM is 43215.0).

    Raises
    ------
    ValueError
        If seconds0 is negative, seconds1 is negative, seconds1 is not
        strictly greater than seconds0, or unit is not a column vector
        (1D array or 2D single-column array).

    Notes
    -----
    This is a helper function for realized_kernel and other realized
    volatility estimators that require time conversion between different
    representations.

    The conversion formula is:

        seconds = seconds0 + (seconds1 - seconds0) * unit

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.realized.unit2seconds import unit2seconds
    >>> # NYSE trading hours: 9:30 AM (34200s) to 4:00 PM (57600s)
    >>> unit = np.array([0.0, 0.5, 1.0])
    >>> seconds = unit2seconds(unit, 34200.0, 57600.0)
    >>> seconds
    array([34200., 45900., 57600.])
    """
    # ----------------------------------------------------------------
    # Input Validation
    # Ref: unit2seconds.m:34-35 — SECONDS0 must be non-negative
    # ----------------------------------------------------------------
    seconds0 = np.float64(seconds0)
    seconds1 = np.float64(seconds1)

    if seconds0 < 0:
        raise ValueError('SECONDS0 must be non-negative')

    # Ref: unit2seconds.m:37-38 — SECONDS1 must be non-negative
    if seconds1 < 0:
        raise ValueError('SECONDS1 must be non-negative')

    # Ref: unit2seconds.m:41-43 — SECONDS1 must be strictly larger than SECONDS0
    # Uses machine epsilon for floating-point comparison, matching MATLAB's eps
    if (seconds1 - seconds0) < np.finfo(np.float64).eps:
        raise ValueError('SECONDS1 must be larger than SECONDS0.')

    # Ref: unit2seconds.m:46-48 — If input is a row vector (wider than tall),
    # transpose to column vector. MATLAB uses size(unit,2)>size(unit,1).
    unit = np.asarray(unit, dtype=np.float64)

    if unit.ndim == 2 and unit.shape[1] > unit.shape[0]:
        # Ref: unit2seconds.m:47 — unit=unit' (transpose row to column)
        unit = unit.T

    # Ref: unit2seconds.m:54-56 — After potential transpose, enforce column vector.
    # MATLAB checks size(unit,2)>1 to reject multi-column matrices.
    if unit.ndim == 2 and unit.shape[1] > 1:
        raise ValueError('UNIT must be an m by 1 column vector')

    # Flatten 2D column vector (m,1) to 1D array (m,) for consistent output
    if unit.ndim == 2:
        unit = unit.ravel()

    # ----------------------------------------------------------------
    # Conversion
    # Ref: unit2seconds.m:63 — seconds=seconds0+(seconds1-seconds0)*unit
    # Linear mapping: unit=0 → seconds=seconds0, unit=1 → seconds=seconds1
    # ----------------------------------------------------------------
    seconds: np.ndarray = seconds0 + (seconds1 - seconds0) * unit

    return seconds
