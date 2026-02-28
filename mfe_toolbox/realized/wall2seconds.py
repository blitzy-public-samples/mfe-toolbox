"""
Convert wall-clock times in HHMMSS format to seconds past midnight.

This module provides the :func:`wall2seconds` function, which converts
wall-clock timestamps encoded as numeric values in HHMMSS.SS format
(e.g., 93000.00 for 9:30:00 AM) into floating-point seconds past midnight.

Migrated from ``realized/wall2seconds.m`` (MFE Toolbox Version 4.0).

See Also
--------
mfe_toolbox.realized.seconds2wall : Inverse conversion (seconds to HHMMSS).
mfe_toolbox.realized.realized_kernel : Primary consumer of time conversion helpers.
mfe_toolbox.realized.realized_variance : Uses seconds-based time representation.

Notes
-----
Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 1    Date: 5/1/2008
"""

import numpy as np


def wall2seconds(wall) -> np.ndarray:
    """
    Convert wall-clock times in HHMMSS.SS format to seconds past midnight.

    Parses numeric wall-clock timestamps where the integer part encodes
    hours, minutes, and seconds as ``HH * 10000 + MM * 100 + SS``, with
    optional fractional seconds in the decimal portion.

    Parameters
    ----------
    wall : array_like
        A scalar or 1-D array of wall-clock times in HHMMSS.SS format.
        For example, ``93000.00`` represents 09:30:00.00 and ``134529.50``
        represents 13:45:29.50.  Values must satisfy ``0 <= wall < 240000``,
        with minutes in ``[0, 60)`` and seconds in ``[0, 60)``.

    Returns
    -------
    seconds : numpy.ndarray
        A 1-D ``float64`` array of the same length as *wall*, giving
        the equivalent times as seconds past midnight.

    Raises
    ------
    ValueError
        If any element of *wall* is negative or ``>= 240000``, or if
        the decoded minutes or seconds exceed 60.

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.realized.wall2seconds import wall2seconds
    >>> wall2seconds(np.array([93000.0, 160000.0]))
    array([34200., 57600.])

    >>> wall2seconds(np.array([0.0]))
    array([0.])

    Notes
    -----
    The conversion formula mirrors the original MATLAB implementation
    (``realized/wall2seconds.m``) exactly:

    .. math::

        \\text{hr} &= \\lfloor \\text{wall} / 10000 \\rfloor \\\\
        \\text{mm} &= \\lfloor \\text{wall} / 100 \\rfloor - \\text{hr} \\times 100 \\\\
        \\text{ss} &= \\operatorname{fmod}(\\text{wall}, 100) \\\\
        \\text{seconds} &= 3600 \\times \\text{hr} + 60 \\times \\text{mm} + \\text{ss}

    Fractional seconds (e.g., ``93000.50``) are preserved through the
    ``fmod`` remainder and carried into the output.

    MATLAB 1-based indexing is irrelevant here because the function
    operates element-wise on the entire array.
    """
    # ----------------------------------------------------------------
    # Input coercion
    # Ref: wall2seconds.m:34 — wall = double(wall);
    # Ensures integer-typed inputs are promoted to float64 so that
    # floor / fmod produce correct floating-point results.
    # ----------------------------------------------------------------
    wall = np.asarray(wall, dtype=np.float64)

    # ----------------------------------------------------------------
    # Input validation — bounds check
    # Ref: wall2seconds.m:30-31 — if any(wall>=240000) || any(wall<0)
    # ----------------------------------------------------------------
    if np.any(wall >= 240000) or np.any(wall < 0):
        raise ValueError(
            "WALL does not contain valid wall times.  "
            "Wall times should be of the form HHMMSS (e.g. 101534)."
        )

    # ----------------------------------------------------------------
    # HHMMSS digit extraction
    # Ref: wall2seconds.m:36 — hr = floor(wall/10000);
    # Extract the HH portion by integer-dividing by 10000.
    # ----------------------------------------------------------------
    hr = np.floor(wall / 10000)

    # Ref: wall2seconds.m:37 — mm = floor(wall/100) - hr*100;
    # Extract the MM portion: total "HHMM" minus hours contribution.
    mm = np.floor(wall / 100) - hr * 100

    # Ref: wall2seconds.m:38 — ss = rem(wall, 100);
    # Extract the SS.SS portion using floating-point remainder.
    # MATLAB rem(a,b) maps to numpy.fmod(a,b) for non-negative inputs.
    ss = np.fmod(wall, 100)

    # ----------------------------------------------------------------
    # Input validation — decoded minute / second range check
    # Ref: wall2seconds.m:40-41 — if any(mm>60) || any(ss>60)
    # ----------------------------------------------------------------
    if np.any(mm > 60) or np.any(ss > 60):
        raise ValueError(
            "WALL does not contain valid wall times.  "
            "Wall times should be of the form HHMMSS (e.g. 101534)."
        )

    # ----------------------------------------------------------------
    # Conversion to seconds past midnight
    # Ref: wall2seconds.m:48 — seconds = 3600 * hr + 60 * mm + ss;
    # ----------------------------------------------------------------
    seconds: np.ndarray = 3600 * hr + 60 * mm + ss

    return seconds
