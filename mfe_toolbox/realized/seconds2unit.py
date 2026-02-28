"""
Convert seconds past midnight to unit [0, 1] interval times.

This module provides the seconds2unit function that maps absolute
seconds-past-midnight timestamps to fractional positions within a
specified trading window, where the start time maps to 0 and the
end time maps to 1.

This is a helper function primarily used by realized_kernel and
related realized volatility estimators.

Migrated from: realized/seconds2unit.m (MFE Toolbox, Kevin Sheppard)

See Also
--------
wall2unit, wall2seconds, seconds2wall, realized_kernel, realized_variance
"""

import numpy as np


def seconds2unit(
    seconds: np.ndarray,
    seconds0: float,
    seconds1: float,
) -> np.ndarray:
    """
    Convert seconds past midnight to unit [0, 1] interval times.

    Maps an array of absolute seconds-past-midnight values to the unit
    interval [0, 1] based on a specified trading window defined by
    ``seconds0`` (maps to 0.0) and ``seconds1`` (maps to 1.0).

    Parameters
    ----------
    seconds : array_like
        1-D array of times expressed as seconds past midnight (e.g.,
        1:00:00 AM is 3600, 12:00:15 PM is 43215). Must contain at
        least 2 elements and all values must be non-negative.
    seconds0 : float
        Base time in seconds past midnight that maps to 0 on the unit
        interval (e.g., 34200 for 9:30 AM).
    seconds1 : float
        End time in seconds past midnight that maps to 1 on the unit
        interval (e.g., 57600 for 4:00 PM). Must be strictly greater
        than ``seconds0``.

    Returns
    -------
    numpy.ndarray
        1-D array (dtype float64) of times measured as fractions of the
        interval [seconds0, seconds1]. Values at seconds0 yield 0.0
        and values at seconds1 yield 1.0.

    Raises
    ------
    ValueError
        If ``seconds`` contains negative values.
        If ``seconds0`` is negative.
        If ``seconds1`` is negative.
        If ``seconds1`` is not strictly greater than ``seconds0``.
        If ``seconds`` is not a 1-D array after flattening.
        If ``seconds`` contains fewer than 2 elements.

    Notes
    -----
    The conversion formula is:

    .. math::

        \\text{unit} = \\frac{\\text{seconds} - \\text{seconds0}}{\\text{seconds1} - \\text{seconds0}}

    Ref: seconds2unit.m — MATLAB uses column vectors; Python flattens to 1-D.
    Ref: seconds2unit.m:44 — MATLAB uses eps for the seconds1 > seconds0 check;
         Python uses np.finfo(np.float64).eps for identical comparison behavior.

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.realized.seconds2unit import seconds2unit
    >>> times = np.array([34200.0, 45900.0, 57600.0])
    >>> seconds2unit(times, 34200.0, 57600.0)
    array([0. , 0.5, 1. ])
    """
    # ------------------------------------------------------------------
    # Input coercion to float64 numpy arrays
    # Ref: seconds2unit.m:52-54 — MATLAB casts all inputs to double
    # ------------------------------------------------------------------
    seconds = np.asarray(seconds, dtype=np.float64)
    seconds0 = np.float64(seconds0)
    seconds1 = np.float64(seconds1)

    # ------------------------------------------------------------------
    # Flatten to 1-D (mirrors MATLAB column vector transpose logic)
    # Ref: seconds2unit.m:48-50 — if size(seconds,2) > size(seconds,1),
    #   transpose. In Python, we simply flatten to 1-D.
    # ------------------------------------------------------------------
    seconds = seconds.flatten()

    # ------------------------------------------------------------------
    # Validation: non-negative checks
    # Ref: seconds2unit.m:34-42
    # ------------------------------------------------------------------
    if np.any(seconds < 0):
        raise ValueError('SECONDS must be non-negative')

    if seconds0 < 0:
        raise ValueError('SECONDS0 must be non-negative')

    if seconds1 < 0:
        raise ValueError('SECONDS1 must be non-negative')

    # ------------------------------------------------------------------
    # Validation: seconds1 must be strictly greater than seconds0
    # Ref: seconds2unit.m:44 — uses MATLAB eps (≈2.22e-16)
    # ------------------------------------------------------------------
    if (seconds1 - seconds0) < np.finfo(np.float64).eps:
        raise ValueError('SECONDS1 must be larger than SECONDS0.')

    # ------------------------------------------------------------------
    # Validation: must be 1-D after flattening
    # Ref: seconds2unit.m:56-58 — ensures column vector
    # ------------------------------------------------------------------
    if seconds.ndim != 1:
        raise ValueError('SECONDS must be an m by 1 column vector')

    # ------------------------------------------------------------------
    # Validation: at least 2 elements
    # Ref: seconds2unit.m:59-61
    # ------------------------------------------------------------------
    if len(seconds) < 2:
        raise ValueError('SECONDS must contain at least 2 elements')

    # ------------------------------------------------------------------
    # Core computation
    # Ref: seconds2unit.m:65 — unit = (seconds - seconds0) / (seconds1 - seconds0)
    # ------------------------------------------------------------------
    unit: np.ndarray = (seconds - seconds0) / (seconds1 - seconds0)

    return unit
