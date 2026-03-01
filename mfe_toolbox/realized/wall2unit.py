"""
Convert wall-clock times (HHMMSS format) to unit interval [0, 1].

This module provides the :func:`wall2unit` function, which maps wall-clock
timestamps in HHMMSS.SS numeric format to the unit interval [0, 1], where
the start-of-day time (``wall0``) maps to 0.0 and the end-of-day time
(``wall1``) maps to 1.0.

The conversion is implemented as a chain of two existing conversions:

1. Wall clock → seconds past midnight  (via :func:`wall2seconds`)
2. Seconds → unit interval             (via :func:`seconds2unit`)

Migrated from: ``realized/wall2unit.m`` (MFE Toolbox Version 4.0)

See Also
--------
mfe_toolbox.realized.unit2wall : Inverse conversion (unit interval to wall clock).
mfe_toolbox.realized.wall2seconds : Wall clock to seconds past midnight.
mfe_toolbox.realized.seconds2unit : Seconds to unit interval.
mfe_toolbox.realized.realized_kernel : Primary consumer of time conversion helpers.
mfe_toolbox.realized.realized_variance : Realized variance estimator.

Notes
-----
Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 1    Date: 5/1/2008
"""

import numpy as np

from mfe_toolbox.realized.wall2seconds import wall2seconds
from mfe_toolbox.realized.seconds2unit import seconds2unit


def wall2unit(wall, wall0, wall1) -> np.ndarray:
    """
    Convert wall-clock times in HHMMSS format to unit [0, 1] interval.

    Maps an array of wall-clock timestamps to the unit interval based on
    a trading window defined by ``wall0`` (maps to 0.0) and ``wall1``
    (maps to 1.0).  The conversion chains two intermediate steps:
    wall → seconds (via :func:`wall2seconds`) and seconds → unit (via
    :func:`seconds2unit`).

    Parameters
    ----------
    wall : array_like
        An m-element array (m >= 2) of wall-clock times in HHMMSS.SS
        format.  For example, ``93047`` represents 09:30:47 and
        ``134529`` represents 13:45:29.  Values must satisfy
        ``0 <= wall < 240000`` with valid minutes (< 60) and seconds
        (< 60).
    wall0 : float
        Base wall time that maps to 0 in the unit interval.  Must be
        a valid HHMMSS time (e.g., ``93000`` for 09:30:00).
    wall1 : float
        End wall time that maps to 1 in the unit interval.  Must be
        a valid HHMMSS time greater than ``wall0``
        (e.g., ``160000`` for 16:00:00).

    Returns
    -------
    numpy.ndarray
        A 1-D ``float64`` array of the same length as *wall*, with
        values in the unit interval [0, 1] representing fractional
        positions within the [wall0, wall1] trading window.

    Raises
    ------
    ValueError
        If any element of *wall* is outside [0, 240000), or contains
        invalid minutes or seconds (>= 60).
        If *wall0* or *wall1* is not a valid HHMMSS time.
        If *wall* contains fewer than 2 elements or is not 1-D.

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.realized.wall2unit import wall2unit
    >>> wall = np.array([93000.0, 123000.0, 160000.0])
    >>> wall2unit(wall, 93000.0, 160000.0)
    array([0.        , 0.46153846, 1.        ])

    Notes
    -----
    Ref: wall2unit.m:34-37 — MATLAB uses ``wall = double(wall)`` to protect
    against integer inputs.  Python equivalent is ``np.asarray(...,
    dtype=np.float64)``.

    Ref: wall2unit.m:53-58 — MATLAB extracts hr, mm, ss from HHMMSS format
    using ``floor`` and ``rem`` and validates mm < 60, ss < 60.  Here
    these validations are delegated to :func:`wall2seconds`, which performs
    the identical checks.

    Ref: wall2unit.m:73-76 — The core algorithm:
    ``wall0=wall2seconds(wall0); wall1=wall2seconds(wall1);
    seconds=wall2seconds(wall); unit=(seconds-wall0)/(wall1-wall0);``
    This is exactly reproduced by chaining :func:`wall2seconds` then
    :func:`seconds2unit`.
    """
    # ----------------------------------------------------------------
    # Input coercion
    # Ref: wall2unit.m:51 — wall = double(wall);
    # Protects against integer-typed inputs for correct floor/fmod.
    # ----------------------------------------------------------------
    wall = np.asarray(wall, dtype=np.float64).ravel()

    # ----------------------------------------------------------------
    # Input validation — array size and shape
    # Ref: wall2unit.m:38-40 — transpose row vector to column
    # Ref: wall2unit.m:42-44 — WALL must be column vector
    # Ref: wall2unit.m:46-48 — WALL must have at least 2 elements
    # ----------------------------------------------------------------
    if wall.ndim != 1:
        raise ValueError('WALL must be an m by 1 column vector')
    if len(wall) < 2:
        raise ValueError('WALL must contain at least 2 elements')

    # ----------------------------------------------------------------
    # Input validation — wall time range
    # Ref: wall2unit.m:34-36 — any(wall>=240000) || any(wall<0)
    # ----------------------------------------------------------------
    if np.any(wall >= 240000) or np.any(wall < 0):
        raise ValueError(
            'WALL does not contain valid numerical times.  '
            'Numerical times should be of the form HHMMSS (e.g. 101534).'
        )

    # ----------------------------------------------------------------
    # Input validation — minutes and seconds
    # Ref: wall2unit.m:53-59 — extract hr, mm, ss and validate
    # ----------------------------------------------------------------
    hr = np.floor(wall / 10000.0)
    mm = np.floor(wall / 100.0) - hr * 100.0
    ss = np.fmod(wall, 100.0)

    if np.any(mm > 60) or np.any(ss > 60):
        raise ValueError(
            'WALL does not contain valid numerical times.  '
            'Numerical times should be 24-hour and of the form HHMMSS '
            '(e.g. 101534, 221313).'
        )

    # ----------------------------------------------------------------
    # Input validation — wall0 and wall1
    # Ref: wall2unit.m:61-67 — validate wall0 and wall1 as valid HHMMSS
    # ----------------------------------------------------------------
    wall0 = float(wall0)
    wall1 = float(wall1)

    if (wall0 < 0 or wall0 >= 240000
            or np.fmod(wall0, 100) >= 60
            or ((np.fmod(wall0, 10000) - np.fmod(wall0, 100)) / 100) >= 60):
        raise ValueError('WALL0 must be a valid numerical time')

    if (wall1 < 0 or wall1 >= 240000
            or np.fmod(wall1, 100) >= 60
            or ((np.fmod(wall1, 10000) - np.fmod(wall1, 100)) / 100) >= 60):
        raise ValueError('WALL1 must be a valid numerical time')

    # ----------------------------------------------------------------
    # Core conversion: wall → seconds → unit
    # Ref: wall2unit.m:73-76
    #   wall0 = wall2seconds(wall0)
    #   wall1 = wall2seconds(wall1)
    #   seconds = wall2seconds(wall)
    #   unit = (seconds - wall0) / (wall1 - wall0)
    #
    # This is equivalent to:
    #   seconds0 = wall2seconds(wall0)  [scalar]
    #   seconds1 = wall2seconds(wall1)  [scalar]
    #   seconds  = wall2seconds(wall)   [array]
    #   unit = seconds2unit(seconds, seconds0, seconds1)
    # ----------------------------------------------------------------
    seconds0 = wall2seconds(np.array([wall0]))[0]
    seconds1 = wall2seconds(np.array([wall1]))[0]
    seconds = wall2seconds(wall)

    # Ref: wall2unit.m:76 — unit = (seconds - wall0) / (wall1 - wall0)
    # seconds2unit performs: (seconds - seconds0) / (seconds1 - seconds0)
    unit = seconds2unit(seconds, seconds0, seconds1)

    return unit
