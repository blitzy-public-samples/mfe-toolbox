"""
Excel serial date to Python datetime conversion.

Migrated from utility/x2mdate.m (80 lines) — Author: Kevin Sheppard
Copyright: Kevin Sheppard, kevin.sheppard@economics.ox.ac.uk

Provides the :func:`x2mdate` function that converts Excel serial date numbers
into pandas Timestamps.  This is a reverse-engineered clone of the MATLAB
Financial Toolbox function ``x2mdate`` and should behave equivalently.

See Also
--------
mfe_toolbox.utility.c2mdate : C-epoch-to-MATLAB date conversion.
mfe_toolbox.utility.m2cdate : MATLAB-to-C-epoch date conversion.
"""

import numpy as np
import pandas as pd


def x2mdate(xlsdate, date_type=0):
    """
    Convert Excel serial dates to pandas Timestamps.

    Translates Excel serial date numbers into pandas Timestamp objects using
    one of two base-date conventions.  This is the Python equivalent of the
    MATLAB Financial Toolbox ``x2mdate`` function.

    Parameters
    ----------
    xlsdate : int, float, or numpy.ndarray
        Scalar or array of Excel serial date numbers.  Must be numeric.
    date_type : int or numpy.ndarray, optional
        Specifies the Excel base-date convention:

        * 0 (default) — Windows Excel 1900 date system, base date Dec-30-1899.
        * 1 — Mac Excel 1904 date system, base date Jan-1-1904.

        Can be a scalar (applied uniformly to all elements of *xlsdate*) or
        an array conformable to *xlsdate* for per-element base-date selection.

    Returns
    -------
    mldate : numpy.ndarray
        Array of ``pandas.Timestamp`` objects (dtype ``object``) with the same
        shape as *xlsdate*.  For scalar input the result is a 0-d numpy array.

    Raises
    ------
    ValueError
        If *xlsdate* is not numeric, if *date_type* contains values other than
        0 or 1, or if *date_type* is a non-scalar array that is not conformable
        to *xlsdate*.

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.utility.x2mdate import x2mdate
    >>> result = x2mdate(np.array([35000, 40000, 41000]))
    >>> # Ref: x2mdate.m:18-23 — Expected dates:
    >>> #   35000 → 1995-10-28
    >>> #   40000 → 2009-07-06
    >>> #   41000 → 2012-04-01

    Notes
    -----
    Ref: utility/x2mdate.m — This is a reverse-engineered clone of the MATLAB
    Financial Toolbox ``x2mdate`` function.  Base date offsets:

    * Type 0: Excel serial 0 → Dec-30-1899 (Windows Excel 1900 epoch).
      Note that Excel has a known leap-year bug treating 1900 as a leap year;
      this function mirrors that behaviour for compatibility.
    * Type 1: Excel serial 0 → Jan-1-1904 (Mac Excel 1904 epoch).

    The MATLAB original uses ``datenum('30-Dec-1899')`` and
    ``datenum('1-Jan-1904')`` as additive offsets to produce MATLAB serial date
    numbers.  In Python we use ``pd.Timestamp`` base dates and
    ``pd.to_timedelta`` to produce pandas Timestamps directly.
    """
    # ------------------------------------------------------------------
    # Input Validation — Ref: x2mdate.m:40-67
    # ------------------------------------------------------------------

    # Ref: x2mdate.m:40-42 — Default type=0 handled via Python default arg.

    # Ref: x2mdate.m:44-46 — Validate xlsdate is numeric.
    # Strings and other non-numeric types must be rejected explicitly before
    # the np.asarray call, which would raise an opaque error.
    if isinstance(xlsdate, str):
        raise ValueError('XLSDATE must be numeric')
    if isinstance(xlsdate, np.ndarray) and not np.issubdtype(
        xlsdate.dtype, np.number
    ):
        raise ValueError('XLSDATE must be numeric')

    try:
        xlsdate_arr = np.asarray(xlsdate, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError('XLSDATE must be numeric') from exc

    # Ref: x2mdate.m:48-50 — Dimensionality check.
    # Note: The original MATLAB code has a latent bug (evaluates
    # ndims('xlsdate') — the string literal — instead of ndims(xlsdate)),
    # making the check a no-op.  We implement the *intended* constraint:
    # xlsdate should be at most 2-D.
    if xlsdate_arr.ndim > 2:
        raise ValueError(
            'XLSDATE must be a scalar, 1-D array, or 2-D matrix'
        )

    # ------------------------------------------------------------------
    # Type parameter validation — Ref: x2mdate.m:52-67
    # ------------------------------------------------------------------
    if isinstance(date_type, str):
        raise ValueError('TYPE must be either 0 or 1.')

    try:
        date_type_arr = np.asarray(date_type, dtype=np.int32)
    except (ValueError, TypeError) as exc:
        raise ValueError('TYPE must be either 0 or 1.') from exc

    # Ref: x2mdate.m:56-57 — TYPE must be a scalar or vector (1-D).
    if date_type_arr.ndim > 2 or (
        date_type_arr.ndim == 2 and min(date_type_arr.shape) > 1
    ):
        raise ValueError(
            'TYPE must be either a scalar or a vector conformable to xlsdate'
        )

    # Ref: x2mdate.m:59-61 — A non-scalar type must match xlsdate length.
    if date_type_arr.size > 1:
        if date_type_arr.size != xlsdate_arr.size:
            raise ValueError(
                'TYPE must be either a scalar or a vector conformable '
                'to xlsdate'
            )

    # Ref: x2mdate.m:64-66 — Values must be exclusively 0 or 1.
    if not np.all(np.isin(date_type_arr, [0, 1])):
        raise ValueError('TYPE must be either 0 or 1.')

    # ------------------------------------------------------------------
    # Type expansion — Ref: x2mdate.m:72-75
    # ------------------------------------------------------------------
    # Ref: x2mdate.m:72-73 — If type is scalar, broadcast to xlsdate shape.
    if np.isscalar(date_type) or date_type_arr.size == 1:
        date_type_arr = np.full_like(
            xlsdate_arr, date_type_arr.flat[0], dtype=np.int32
        )
    else:
        # Ensure date_type_arr is reshaped to match xlsdate_arr for
        # element-wise operations.
        date_type_arr = date_type_arr.reshape(xlsdate_arr.shape)

    # Ref: x2mdate.m:75 — type = logical(type);
    type_bool = date_type_arr.astype(bool)

    # ------------------------------------------------------------------
    # Date conversion — Ref: x2mdate.m:77-79
    # ------------------------------------------------------------------
    # Ref: x2mdate.m:78 — datenum('30-Dec-1899')
    base_date_0 = pd.Timestamp('1899-12-30')
    # Ref: x2mdate.m:79 — datenum('1-Jan-1904')
    base_date_1 = pd.Timestamp('1904-01-01')

    # Flatten arrays for uniform element-wise processing.
    flat_xls = xlsdate_arr.ravel()
    flat_type = type_bool.ravel()
    n = flat_xls.shape[0]

    # Ref: x2mdate.m:77 — mldate = zeros(size(xlsdate));
    # Allocate output as an object array to hold pd.Timestamp instances.
    result_flat = np.empty(n, dtype=object)

    # Ref: x2mdate.m:78 — Type 0: Windows Excel 1900 base-date conversion.
    # mldate(~type) = xlsdate(~type) + datenum('30-Dec-1899');
    mask_0 = ~flat_type
    if np.any(mask_0):
        indices_0 = np.where(mask_0)[0]
        # Vectorized timedelta computation via pandas
        timedeltas_0 = pd.to_timedelta(flat_xls[indices_0], unit='D')
        dates_0 = base_date_0 + timedeltas_0
        # Scatter computed timestamps back into the result array.
        for j, i in enumerate(indices_0):
            result_flat[i] = dates_0[j]

    # Ref: x2mdate.m:79 — Type 1: Mac Excel 1904 base-date conversion.
    # mldate(type) = xlsdate(type) + datenum('1-Jan-1904');
    mask_1 = flat_type
    if np.any(mask_1):
        indices_1 = np.where(mask_1)[0]
        timedeltas_1 = pd.to_timedelta(flat_xls[indices_1], unit='D')
        dates_1 = base_date_1 + timedeltas_1
        for j, i in enumerate(indices_1):
            result_flat[i] = dates_1[j]

    # Reshape output to match the original input shape and return.
    # Ref: x2mdate.m — MATLAB output has same size as xlsdate.
    mldate = result_flat.reshape(xlsdate_arr.shape)
    return mldate
