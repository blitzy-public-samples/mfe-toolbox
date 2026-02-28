"""
Convert seconds past midnight to wall time (HHMMSS.SS format).

Migrated from: realized/seconds2wall.m
Original author: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 1    Date: 5/1/2008

This is a helper function for REALIZED_KERNEL and other realized
volatility estimators requiring wall-clock time representation.

See Also
--------
wall2seconds : Inverse conversion from wall time to seconds past midnight.
realized_kernel : Realized kernel estimator using wall time inputs.
realized_variance : Realized variance estimator.
"""

import numpy as np


def seconds2wall(seconds) -> np.ndarray:
    """Convert seconds past midnight to wall time in HHMMSS.SS format.

    Converts an array of times expressed as seconds since midnight (00:00:00)
    into the HHMMSS.SS wall-clock format used by financial data feeds and
    the MFE Toolbox realized volatility estimators.

    Parameters
    ----------
    seconds : array_like
        An array of times measured as seconds past midnight.
        Values must satisfy ``0 <= seconds < 86400`` (one full day).
        Scalar inputs are also accepted and will be converted to a 1-D array.

    Returns
    -------
    wall : np.ndarray
        A 1-D array of wall times in HHMMSS.SS format.
        For example, 9:30:00.00 AM is represented as ``93000.00``,
        and 4:00:00.00 PM is represented as ``160000.00``.
        Fractional seconds are preserved in the SS.SS portion.

    Raises
    ------
    ValueError
        If any element of ``seconds`` is negative or >= 86400.

    Notes
    -----
    The HHMMSS.SS format encodes:

    - Hours (HH) in the ten-thousands place: ``HH * 10000``
    - Minutes (MM) in the hundreds place: ``MM * 100``
    - Seconds (SS.SS) in the units and decimal places

    The conversion formula is:

    .. math::

        \\text{wall} = \\lfloor s / 3600 \\rfloor \\times 10000
                     + \\lfloor (s \\bmod 3600) / 60 \\rfloor \\times 100
                     + (s \\bmod 60)

    where ``s`` is seconds past midnight.

    This function is the inverse of :func:`wall2seconds`.

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.realized.seconds2wall import seconds2wall
    >>> seconds2wall(np.array([0.0, 34200.0, 57600.0]))
    array([     0., 93000., 160000.])

    >>> seconds2wall(np.array([34230.5]))
    array([93030.5])

    References
    ----------
    Ref: realized/seconds2wall.m — MFE Toolbox Version 4.0
    """
    # ----------------------------------------------------------------
    # Input Checking
    # Ref: seconds2wall.m:26-32 — Validate input and cast to float64
    # ----------------------------------------------------------------
    seconds = np.asarray(seconds, dtype=np.float64)

    # Flatten to 1-D for consistent handling
    # Ref: seconds2wall.m:32 — MATLAB enforces column vector via double()
    seconds = seconds.ravel()

    # Ref: seconds2wall.m:29 — Validate range: 0 <= seconds < 86400
    if np.any(seconds >= 24 * 3600) or np.any(seconds < 0):
        raise ValueError(
            'SECONDS does not contain valid numerical times. '
            'Numerical times must satisfy 0 <= SECONDS < 86400.'
        )

    # ----------------------------------------------------------------
    # Parse and recombine
    # Ref: seconds2wall.m:38-41 — Extract hours, minutes, seconds
    # components and recombine into HHMMSS.SS format
    # ----------------------------------------------------------------

    # Ref: seconds2wall.m:38 — hr = floor(seconds/3600)
    hr = np.floor(seconds / 3600.0)

    # Ref: seconds2wall.m:39 — mm = floor((seconds - hr*3600)/60)
    mm = np.floor((seconds - hr * 3600.0) / 60.0)

    # Ref: seconds2wall.m:40 — ss = rem(seconds, 60)
    # MATLAB rem(a, b) is equivalent to np.fmod(a, b) for positive values.
    # Using np.fmod to match MATLAB's rem() exactly for numerical parity.
    ss = np.fmod(seconds, 60.0)

    # Ref: seconds2wall.m:41 — wall = hr*10000 + mm*100 + ss
    wall = hr * 10000.0 + mm * 100.0 + ss

    return wall
