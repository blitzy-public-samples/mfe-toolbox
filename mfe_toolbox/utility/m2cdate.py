"""
MATLAB datenum to CRSP date conversion.

Migrated from utility/m2cdate.m — Author: Kevin Sheppard
Copyright: Kevin Sheppard, kevin.sheppard@economics.ox.ac.uk
Revision: 2    Date: 10/17/2009

Converts MATLAB datenum values (days since year 0000, Jan 1, where datenum(0,1,1)=1)
to CRSP dates (YYYYMMDD integer format) used by WRDS (Wharton Research Data Services).

The MATLAB datenum for the Unix epoch (1970-01-01) is 719529. This constant is
used to bridge between MATLAB's datenum representation and Python's datetime system
via pandas.to_datetime with unit='D'.

See also
--------
c2mdate : Inverse conversion (CRSP date → MATLAB datenum)
"""

import datetime

import numpy as np
import pandas as pd

# MATLAB datenum value for the Unix epoch (1970-01-01 00:00:00).
# MATLAB datenum counts days since 0000-Jan-01 with datenum(0,1,1) = 1.
# datenum(1970,1,1) = 719529 in MATLAB.
_MATLAB_DATENUM_UNIX_EPOCH: int = 719529


def m2cdate(mldate):
    """
    Convert MATLAB datenum values to CRSP dates in YYYYMMDD integer format.

    Provides a simple method to convert between MATLAB serial date numbers
    and CRSP dates provided by WRDS (Wharton Research Data Services).

    Parameters
    ----------
    mldate : numpy.ndarray, pandas.DatetimeIndex, datetime.datetime, or scalar
        A scalar, 1-D array, or pandas DatetimeIndex of MATLAB datenum values
        (float). MATLAB datenums count days since the fictitious date
        January 0, year 0000, so ``datenum(0,1,1) = 1`` and
        ``datenum(1970,1,1) = 719529``.

        Also accepts Python ``datetime.datetime`` objects and pandas
        ``DatetimeIndex`` / ``Timestamp`` objects, which are converted
        directly without the datenum offset.

    Returns
    -------
    crspdate : numpy.ndarray
        An array of integers in YYYYMMDD format (e.g., 19951028 for
        October 28, 1995). The output array has the same length as the
        input. The dtype is ``int64``.

    Raises
    ------
    ValueError
        If ``mldate`` is a string or contains strings.
    ValueError
        If ``mldate`` has more than one dimension (must be scalar or 1-D).

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.utility.m2cdate import m2cdate
    >>> mldate = np.array([728960, 733960, 734960])
    >>> m2cdate(mldate)
    array([19951028, 20090706, 20120401])

    Notes
    -----
    Ref: m2cdate.m — The MATLAB implementation uses ``datevec(mldate)`` to
    decompose datenum values into ``[year, month, day, hour, minute, second]``
    vectors, then computes ``10000 * year + 100 * month + day``.

    The Python implementation uses ``pandas.to_datetime`` with ``unit='D'``
    after subtracting the MATLAB-to-Unix-epoch offset of 719529 days.

    See Also
    --------
    c2mdate : Convert CRSP dates to MATLAB datenum values.
    """
    # ------------------------------------------------------------------
    # Handle datetime-like inputs directly (pandas DatetimeIndex, Timestamp,
    # or Python datetime.datetime) — no datenum conversion needed.
    # ------------------------------------------------------------------
    if isinstance(mldate, pd.DatetimeIndex):
        # Ref: m2cdate.m:51-52 — extract year/month/day directly
        year = mldate.year.values
        month = mldate.month.values
        day = mldate.day.values
        crspdate = 10000 * year + 100 * month + day
        return crspdate.astype(np.int64)

    if isinstance(mldate, pd.Timestamp):
        # Single pandas Timestamp — extract components directly
        crspdate_val = 10000 * mldate.year + 100 * mldate.month + mldate.day
        return np.array([crspdate_val], dtype=np.int64)

    if isinstance(mldate, datetime.datetime):
        # Single Python datetime — extract components directly
        crspdate_val = 10000 * mldate.year + 100 * mldate.month + mldate.day
        return np.array([crspdate_val], dtype=np.int64)

    # ------------------------------------------------------------------
    # Input Validation (Ref: m2cdate.m:35-45)
    # ------------------------------------------------------------------

    # Ref: m2cdate.m:35-37 — MATLAB: if any(ischar(mldate)) error(...)
    # Check for string input before array conversion to give a clear message.
    if isinstance(mldate, str):
        raise ValueError('MLDATE must be numeric')

    # If already a numpy.ndarray, use directly; otherwise convert via np.asarray.
    # Using isinstance with np.ndarray for efficient type dispatch.
    if isinstance(mldate, np.ndarray):
        mldate_arr = mldate
    else:
        # Convert to numpy array for uniform processing
        # Ref: numpy.asarray used per external_imports schema
        mldate_arr = np.asarray(mldate)

    # Check if the converted array has a string/character dtype
    # Ref: m2cdate.m:35-37 — ischar check
    if mldate_arr.dtype.kind in ('U', 'S', 'O'):
        # dtype.kind 'U' = Unicode string, 'S' = byte string, 'O' = object
        # For object arrays, check if contents are strings
        if mldate_arr.dtype.kind == 'O':
            # Check if any element is a string
            flat = mldate_arr.ravel()
            if len(flat) > 0 and isinstance(flat[0], str):
                raise ValueError('MLDATE must be numeric')
            # Also check if elements are datetime-like objects
            if len(flat) > 0 and isinstance(flat[0], datetime.datetime):
                # Convert array of datetime objects
                dt_index = pd.DatetimeIndex(flat)
                year = dt_index.year.values
                month = dt_index.month.values
                day = dt_index.day.values
                crspdate = 10000 * year + 100 * month + day
                return crspdate.astype(np.int64)
        else:
            raise ValueError('MLDATE must be numeric')

    # Validate that input is numeric (integer or floating-point).
    # Ref: m2cdate.m:35-37 — comprehensive numeric type check.
    # Use np.issubdtype with np.integer, np.floating, and np.number to
    # distinguish integer datenums (e.g., 728960) from float datenums
    # (e.g., 728960.5 for noon), which affects datetime precision.
    is_integer_type = np.issubdtype(mldate_arr.dtype, np.integer)
    is_floating_type = np.issubdtype(mldate_arr.dtype, np.floating)
    is_numeric = np.issubdtype(mldate_arr.dtype, np.number)

    if not is_numeric:
        # Also accept numpy datetime64
        if np.issubdtype(mldate_arr.dtype, np.datetime64):
            # Convert numpy datetime64 to pandas DatetimeIndex for extraction
            dt_index = pd.DatetimeIndex(mldate_arr.ravel())
            year = dt_index.year.values
            month = dt_index.month.values
            day = dt_index.day.values
            crspdate = 10000 * year + 100 * month + day
            return crspdate.astype(np.int64)
        raise ValueError('MLDATE must be numeric')

    # Integer datenums (is_integer_type=True) need conversion to float64
    # for fractional day arithmetic. Float datenums (is_floating_type=True)
    # already have the needed precision for sub-day components.
    # Both cases are handled by the .astype(np.float64) below.

    # Ref: m2cdate.m:39-41 — MATLAB: if ndims(mldate)~=2 && min(size(mldate))~=1
    # In MATLAB, scalars are 1x1 (ndims==2, min(size)==1), vectors are Tx1 or 1xT.
    # In Python, we check x.ndim > 1 for multi-dimensional arrays.
    # Scalars (ndim==0) and 1-D arrays (ndim==1) are allowed.
    if mldate_arr.ndim > 1:
        raise ValueError('MLDATE must be a 1-D array or scalar')

    # Track whether input was scalar for output shape preservation
    is_scalar = mldate_arr.ndim == 0

    # Ensure we work with at least 1-D for uniform processing
    mldate_flat = mldate_arr.ravel().astype(np.float64)

    # ------------------------------------------------------------------
    # Core Conversion Logic (Ref: m2cdate.m:51-52)
    # ------------------------------------------------------------------
    # Ref: m2cdate.m:51 — v = datevec(mldate)
    # MATLAB datevec decomposes a datenum into [year, month, day, hour, min, sec].
    # We replicate this by converting datenum to a datetime via the Unix epoch offset.
    #
    # MATLAB datenum for Unix epoch: datenum(1970,1,1) = 719529
    # Therefore: days_since_unix_epoch = mldate - 719529
    days_since_unix_epoch = mldate_flat - _MATLAB_DATENUM_UNIX_EPOCH

    # Use pandas for vectorized datenum-to-datetime conversion.
    # pd.to_datetime(x, unit='D') interprets x as fractional days since Unix epoch.
    dt_series = pd.to_datetime(days_since_unix_epoch, unit='D')

    # Ref: m2cdate.m:52 — crspdate = 10000*v(:,1) + 100*v(:,2) + v(:,3)
    # Extract year, month, day components and compute CRSP date format.
    year = dt_series.year
    month = dt_series.month
    day = dt_series.day

    crspdate = (10000 * year + 100 * month + day).values.astype(np.int64)

    # Preserve scalar shape: if input was a scalar, return a 0-d array
    if is_scalar:
        return np.int64(crspdate[0])

    return crspdate
