"""
Convert unit interval [0, 1] times to wall-clock times (HHMMSS format).

This module provides the :func:`unit2wall` function, which maps fractional
trading-day positions in the unit interval [0, 1] back to wall-clock
timestamps in HHMMSS.SS numeric format.  This is the inverse operation of
:func:`wall2unit`.

The conversion is implemented as a chain of three existing conversions:

1. Wall clock boundaries → seconds  (via :func:`wall2seconds`)
2. Unit → seconds past midnight     (via :func:`unit2seconds`)
3. Seconds → wall clock             (via :func:`seconds2wall`)

Migrated from: ``realized/unit2wall.m`` (MFE Toolbox Version 4.0)

See Also
--------
mfe_toolbox.realized.wall2unit : Inverse conversion (wall clock to unit interval).
mfe_toolbox.realized.wall2seconds : Wall clock to seconds past midnight.
mfe_toolbox.realized.unit2seconds : Unit interval to seconds.
mfe_toolbox.realized.seconds2wall : Seconds past midnight to wall clock.
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
from mfe_toolbox.realized.unit2seconds import unit2seconds
from mfe_toolbox.realized.seconds2wall import seconds2wall


def unit2wall(unit, wall0, wall1) -> np.ndarray:
    """
    Convert unit interval [0, 1] times to wall-clock HHMMSS format.

    Maps an array of fractional trading-day positions to wall-clock
    timestamps in HHMMSS.SS format, based on a trading window defined
    by ``wall0`` (corresponding to unit = 0) and ``wall1`` (corresponding
    to unit = 1).

    Parameters
    ----------
    unit : array_like
        An m-element array (m >= 2) of times measured as fractions of the
        interval [wall0, wall1].  Values at 0 correspond to ``wall0`` and
        values at 1 correspond to ``wall1``.
    wall0 : float
        Base wall time in HHMMSS format that corresponds to unit = 0.
        Must be a valid HHMMSS time (e.g., ``93000`` for 09:30:00).
    wall1 : float
        End wall time in HHMMSS format that corresponds to unit = 1.
        Must be a valid HHMMSS time (e.g., ``160000`` for 16:00:00).

    Returns
    -------
    numpy.ndarray
        A 1-D ``float64`` array of wall-clock times in HHMMSS.SS format,
        rounded to 5 decimal places to match the MATLAB implementation.

    Raises
    ------
    ValueError
        If *unit* is not 1-D after flattening.
        If *unit* contains fewer than 2 elements.
        If *wall0* or *wall1* is not a valid HHMMSS time.

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.realized.unit2wall import unit2wall
    >>> unit = np.array([0.0, 0.5, 1.0])
    >>> unit2wall(unit, 93000.0, 160000.0)
    array([ 93000., 124500., 160000.])

    Notes
    -----
    Ref: unit2wall.m:35-37 — MATLAB transposes row vectors to column.
    Python equivalent is ``np.ravel()``.

    Ref: unit2wall.m:46-48 — MATLAB casts inputs to ``double()``.
    Python uses ``np.asarray(..., dtype=np.float64)``.

    Ref: unit2wall.m:62-65 — The core algorithm:
    ``wall0=wall2seconds(wall0); wall1=wall2seconds(wall1);
    seconds=wall0+(wall1-wall0)*unit;
    wall = round(100000*seconds2wall(seconds))/100000;``

    The rounding to 5 decimal places matches MATLAB's explicit
    ``round(100000*x)/100000`` pattern, ensuring sub-second precision
    is preserved while preventing floating-point noise from creating
    spurious microsecond digits.
    """
    # ----------------------------------------------------------------
    # Input coercion
    # Ref: unit2wall.m:46-48 — unit = double(unit); wall0 = double(wall0);
    #   wall1 = double(wall1);
    # ----------------------------------------------------------------
    unit = np.asarray(unit, dtype=np.float64).ravel()
    wall0 = float(wall0)
    wall1 = float(wall1)

    # ----------------------------------------------------------------
    # Input validation — array size and shape
    # Ref: unit2wall.m:35-37 — transpose row to column
    # Ref: unit2wall.m:39-41 — UNIT must be column vector
    # Ref: unit2wall.m:42-44 — UNIT must have at least 2 elements
    # ----------------------------------------------------------------
    if unit.ndim != 1:
        raise ValueError('UNIT must be an m by 1 column vector')
    if len(unit) < 2:
        raise ValueError('UNIT must contain at least 2 elements')

    # ----------------------------------------------------------------
    # Input validation — wall0 and wall1 as valid HHMMSS
    # Ref: unit2wall.m:50-56 — validate wall0 and wall1
    # ----------------------------------------------------------------
    if (wall0 < 0 or wall0 >= 240000
            or np.fmod(wall0, 100) >= 60
            or ((np.fmod(wall0, 10000) - np.fmod(wall0, 100)) / 100) >= 60):
        raise ValueError('WALL0 must be a valid numerical time')

    if (wall1 < 0 or wall1 >= 240000
            or np.fmod(wall1, 100) >= 60
            or ((np.fmod(wall1, 10000) - np.fmod(wall1, 100)) / 100) >= 60):
        raise ValueError('WALL1 must be a valid numerical time')

    # ----------------------------------------------------------------
    # Core conversion: unit → seconds → wall
    # Ref: unit2wall.m:62-66
    #   wall0 = wall2seconds(wall0)
    #   wall1 = wall2seconds(wall1)
    #   seconds = wall0 + (wall1 - wall0) * unit
    #   wall = round(100000 * seconds2wall(seconds)) / 100000
    # ----------------------------------------------------------------
    seconds0 = wall2seconds(np.array([wall0]))[0]
    seconds1 = wall2seconds(np.array([wall1]))[0]

    # Ref: unit2wall.m:65 — seconds = wall0 + (wall1-wall0)*unit
    # unit2seconds performs: seconds0 + (seconds1 - seconds0) * unit
    seconds = unit2seconds(unit, seconds0, seconds1)

    # Ref: unit2wall.m:66 — Convert seconds back to wall time
    wall = seconds2wall(seconds)

    # Ref: unit2wall.m:66 — round(100000*seconds2wall(seconds))/100000
    # Round to 5 decimal places to match MATLAB's explicit rounding
    # This prevents floating-point noise from creating spurious digits
    wall = np.round(wall * 100000.0) / 100000.0

    return wall
