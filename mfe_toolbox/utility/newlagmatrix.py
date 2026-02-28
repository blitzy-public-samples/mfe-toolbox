"""
Lag matrix construction for time series regression.

Migrated from utility/newlagmatrix.m — Author: Kevin Sheppard
(Revision 5, Date: 12/1/2005)

Constructs a matrix of lags (X) and a trimmed dependent variable (Y) for use
in autoregressions and other time series models.

Notes
-----
Original MATLAB name 'lagmatrix' conflicted with a MATLAB built-in file,
hence the function was renamed to 'newlagmatrix'.
"""

import numpy as np

__all__ = ['newlagmatrix']


def newlagmatrix(y, lags, include_constant=1):
    """
    Construct lag matrix for time series regression.

    Builds a regressor matrix containing lagged values of the input series,
    optionally prepending a constant column and/or the contemporaneous value.
    The lag ordering matches the MATLAB original: lag 1 (most recent) first,
    then lag 2, up to the specified number of lags.

    Parameters
    ----------
    y : numpy.ndarray
        T x 1 or T x K data vector/matrix.  If 1-D, it is reshaped to a
        column vector ``(T, 1)``.
    lags : int
        Number of lags to include.  Must be a non-negative integer.
    include_constant : int, optional
        Controls regressor composition:

        - 0 : no constant column
        - 1 : (default) prepend a column of ones before lag columns
        - 2 : prepend a column of ones, then contemporaneous *y*, then lag
          columns

    Returns
    -------
    x : numpy.ndarray
        ``(T - lags) x M`` regressor matrix, where *M* depends on
        *include_constant* and *K*:

        - ``include_constant=0``: ``M = lags * K``
        - ``include_constant=1``: ``M = 1 + lags * K``
        - ``include_constant=2``: ``M = 1 + K + lags * K``

        When ``lags=0`` and ``include_constant=0``, returns a ``(T, 0)``
        empty array.
    y_trimmed : numpy.ndarray
        ``(T - lags) x K`` trimmed dependent variable matrix.

    Raises
    ------
    ValueError
        If *y* is not a 1-D or 2-D array, *lags* is not a non-negative
        integer, *include_constant* is not in ``{0, 1, 2}``, or the number
        of observations is not greater than the number of lags.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> x, y_trimmed = newlagmatrix(y, 2, include_constant=1)
    >>> y_trimmed
    array([[3.],
           [4.],
           [5.]])
    >>> x
    array([[1., 2., 1.],
           [1., 3., 2.],
           [1., 4., 3.]])

    Notes
    -----
    Migrated from ``utility/newlagmatrix.m`` (Kevin Sheppard, Revision 5,
    Date: 12/1/2005).

    Key differences from MATLAB original:

    - Default *include_constant* changed from 0 to 1
      (Ref: newlagmatrix.m:31)
    - *include_constant=2* option added (constant + contemporaneous + lags)
    - ``T x K`` matrix input supported
      (Ref: newlagmatrix.m:39-40 enforced vector-only input)
    - Return order is ``(x, y_trimmed)`` vs MATLAB ``[y, x]``
    - MATLAB ``repmat``/``reshape`` trick replaced with explicit array
      slicing for clarity
    """
    # ------------------------------------------------------------------
    # Input validation
    # Ref: newlagmatrix.m:38 — [T,K]=size(x); ensure array is 2-D
    # ------------------------------------------------------------------
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        # Ref: newlagmatrix.m:43-45 — MATLAB transposes row vectors;
        # Python reshapes 1-D arrays to column vectors (T, 1)
        y = y.reshape(-1, 1)
    if y.ndim != 2:
        raise ValueError('y must be a 1-D or 2-D array.')

    T, K = y.shape  # Ref: newlagmatrix.m:38 — [T,K]=size(x)

    # Ref: newlagmatrix.m:51-53 — floor(nlags)~=nlags || any(nlags<0)
    # Validate lags is a non-negative integer (or integer-valued float)
    try:
        lags_int = int(lags)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('lags must be a non-negative integer.')
    if lags_int != lags or lags_int < 0:
        raise ValueError('lags must be a non-negative integer.')
    lags = lags_int

    # Ref: newlagmatrix.m:47-49 — c must be 0 or 1
    # Extended to support include_constant=2 per Python specification
    if include_constant not in (0, 1, 2):
        raise ValueError('include_constant must be 0, 1, or 2.')

    # Ensure sufficient observations for the requested number of lags
    if lags > 0 and T <= lags:
        raise ValueError(
            f'Number of lags ({lags}) must be less than the number '
            f'of observations ({T}).'
        )

    # ------------------------------------------------------------------
    # Build lag matrix and trim dependent variable
    # ------------------------------------------------------------------
    if lags > 0:
        # Ref: newlagmatrix.m:60 — implicit T_eff = T - lags
        T_eff = T - lags

        # Ref: newlagmatrix.m:59-66 — Lag column construction
        # MATLAB uses a repmat/reshape trick to build the lag matrix in
        # one operation.  The Python equivalent uses explicit array slicing
        # for clarity while producing identical numerical results.
        x_lags = []
        for i in range(1, lags + 1):
            # Ref: newlagmatrix.m:63 — MATLAB 1-indexed: y(lags-i+1:T-i,:)
            # Python 0-indexed equivalent: y[lags-i : T-i, :]
            # Lag ordering: i=1 is most recent (t-1), i=lags is oldest
            x_lags.append(y[lags - i: T - i, :])

        # Ref: newlagmatrix.m:66 — x = lagmatrix(:, 2:nlags)
        # Concatenate all lag columns horizontally
        x = np.hstack(x_lags)

        # Ref: newlagmatrix.m:65 — y = lagmatrix(:, 1) — contemporaneous
        y_trimmed = y[lags:, :]

        # Ref: newlagmatrix.m:67-69 — optionally prepend constant /
        # contemporaneous columns
        if include_constant == 1:
            # Ref: newlagmatrix.m:68 — x = [ones(size(x,1),1) x]
            # Prepend column of ones before lag columns
            x = np.column_stack([np.ones((T_eff, 1)), x])
        elif include_constant == 2:
            # Python spec extension: [constant, contemporaneous, lags]
            # Not in MATLAB original — adds contemporaneous y between
            # the constant column and the lag columns
            x = np.column_stack([np.ones((T_eff, 1)), y_trimmed, x])
    else:
        # Ref: newlagmatrix.m:70-78 — nlags == 0 branch
        y_trimmed = y.copy()

        if include_constant == 1:
            # Ref: newlagmatrix.m:72-73 — x = ones(T, 1)
            x = np.ones((T, 1))
        elif include_constant == 2:
            # Python spec extension: constant + contemporaneous (no lags)
            x = np.column_stack([np.ones((T, 1)), y_trimmed])
        else:
            # Ref: newlagmatrix.m:75-76 — x = [] (empty matrix)
            # Python equivalent: (T, 0) shaped array preserving row count
            x = np.zeros((T, 0))

    return x, y_trimmed
