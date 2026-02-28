"""
Convert CRSP dates (YYYYMMDD integer format) to pandas Timestamps / numpy datetime64.

This module provides the ``c2mdate`` function, which converts dates from the
CRSP (Center for Research in Security Prices) YYYYMMDD integer convention into
Python datetime representations suitable for time-series analysis.

Migrated from: utility/c2mdate.m (Version 4.0, Kevin Sheppard)

See Also
--------
mfe_toolbox.utility.m2cdate : Inverse conversion (MATLAB datenum → CRSP date).
mfe_toolbox.utility.x2mdate : Excel serial date → MATLAB date conversion.
"""

import numpy as np
import pandas as pd


def c2mdate(crspdate):
    """
    Convert CRSP dates (YYYYMMDD integers) to pandas Timestamps.

    Provides a simple method to convert between CRSP dates provided by WRDS
    (Wharton Research Data Services) and Python datetime objects.  The CRSP
    date format encodes a calendar date as an 8-digit integer ``YYYYMMDD``.

    Parameters
    ----------
    crspdate : int, float, numpy.ndarray, or array_like
        Scalar or 1-D array of CRSP dates in YYYYMMDD format.  Values must
        be numeric and >= 18000000 (i.e., dates from 1800-01-01 onward).

    Returns
    -------
    mldate : numpy.ndarray
        1-D array of ``numpy.datetime64`` values (backed by pandas
        Timestamps) with the same length as the input.  For scalar input
        a 1-element array is returned, preserving consistent return shape.
        The datetime resolution depends on the installed pandas version
        (``datetime64[us]`` for pandas ≥ 2.0, ``datetime64[ns]`` earlier).

    Raises
    ------
    ValueError
        If *crspdate* contains non-numeric data, is not representable as a
        1-D vector, or contains values < 18000000 (invalid YYYYMMDD format).

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.utility.c2mdate import c2mdate
    >>> crspdate = np.array([19951028, 20090706, 20120401])
    >>> mldate = c2mdate(crspdate)
    >>> mldate  # doctest: +SKIP
    array(['1995-10-28', '2009-07-06', '2012-04-01'], dtype='datetime64[us]')

    Scalar input is also accepted:

    >>> c2mdate(20090706)  # doctest: +SKIP
    array(['2009-07-06'], dtype='datetime64[us]')

    Notes
    -----
    The original MATLAB implementation returns MATLAB serial date numbers
    (``datenum``).  In the Python migration, dates are represented as
    ``numpy.datetime64[ns]`` values (equivalent to ``pandas.Timestamp``),
    which are the standard datetime representation in the scientific Python
    ecosystem.

    References
    ----------
    .. [1] Kevin Sheppard, "MFE Toolbox", utility/c2mdate.m, Revision 2,
       Date: 10/17/2009.
    """

    # ------------------------------------------------------------------
    # Input Validation
    # Ref: c2mdate.m:34-48 — replaces MATLAB ischar / ndims / nargin checks
    # ------------------------------------------------------------------

    # Convert to numpy array, preserving dtype for numeric checking
    crspdate_arr = np.asarray(crspdate)

    # Ref: c2mdate.m:34-36 — MATLAB: if any(ischar(crspdate)) error(...)
    if not np.issubdtype(crspdate_arr.dtype, np.number):
        raise ValueError("CRSPDATE must be numeric")

    # Ensure at least 1-D for uniform downstream processing
    crspdate_arr = np.atleast_1d(crspdate_arr)

    # Ref: c2mdate.m:38-40 — MATLAB: if ndims(crspdate)~=2 && min(size)~=1
    # In Python, we require a 1-D array (vector) after atleast_1d promotion.
    if crspdate_arr.ndim > 1:
        # Attempt to squeeze singleton dimensions (e.g., (T,1) or (1,T))
        crspdate_arr = crspdate_arr.squeeze()
        if crspdate_arr.ndim > 1:
            raise ValueError(
                "CRSPDATE must be a scalar or 1-D vector"
            )
        # Re-apply atleast_1d in case squeeze reduced to 0-d
        crspdate_arr = np.atleast_1d(crspdate_arr)

    # Cast to integer-safe type for integer division.
    # Floating-point YYYYMMDD values (e.g., 20090706.0) are common from
    # data loaders; convert to int64 after validation.
    if np.issubdtype(crspdate_arr.dtype, np.floating):
        crspdate_arr = crspdate_arr.astype(np.int64)

    # Ref: c2mdate.m:46-48 — MATLAB: if any(crspdate < 18000000) error(...)
    if np.any(crspdate_arr < 18000000):
        raise ValueError(
            "CRSPDATE appears to be incorrectly formatted. "
            "Only supported format is YYYYMMDD"
        )

    # ------------------------------------------------------------------
    # Date Component Extraction
    # Ref: c2mdate.m:53-55 — integer arithmetic to split YYYYMMDD
    # ------------------------------------------------------------------

    # Ref: c2mdate.m:53 — yr = floor(crspdate/10000)
    yr = crspdate_arr // 10000

    # Ref: c2mdate.m:54 — mo = floor(mod(crspdate,10000)/100)
    mo = (crspdate_arr % 10000) // 100

    # Ref: c2mdate.m:55 — dd = mod(crspdate,100)
    dd = crspdate_arr % 100

    # ------------------------------------------------------------------
    # Date Construction
    # Ref: c2mdate.m:57 — mldate = datenum(yr, mo, dd)
    # Python equivalent: build a DataFrame of components and use
    # pd.to_datetime for vectorised Timestamp construction.
    # ------------------------------------------------------------------

    # pandas.to_datetime accepts a DataFrame with 'year', 'month', 'day'
    # columns — this is the most efficient vectorised path.
    date_components = pd.DataFrame({
        "year": yr,
        "month": mo,
        "day": dd,
    })

    datetime_index = pd.to_datetime(date_components)

    # Return as a numpy array of datetime64[ns] for consistency with the
    # AAP specification ("numpy.ndarray" return type with datetime content).
    mldate = datetime_index.values  # numpy.ndarray dtype='datetime64[ns]'

    return mldate
