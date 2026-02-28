"""
Standardization of data columns to mean zero and unit variance.

Migrated from utility/standardize.m (Version 4.0, Kevin Sheppard).
Standardizes each column of a T x K matrix so that each column has mean 0
and variance 1, with an optional flag to skip the demeaning step.

Notes
-----
MATLAB ``std(x)`` uses N-1 normalization (ddof=1) by default, so
``numpy.std(x, axis=0, ddof=1)`` is used throughout to ensure numerical
parity (±1e-6) with the MATLAB reference implementation.
"""

import numpy as np


def standardize(x: np.ndarray, demean: bool = True) -> np.ndarray:
    """
    Standardize data to mean zero and unit variance per column.

    For each column *j* of the input matrix, the standardized output is::

        st[:, j] = (x[:, j] - mean(x[:, j])) / std(x[:, j])

    When *demean* is ``False``, the mean subtraction step is skipped and
    each column is only divided by its sample standard deviation.

    Parameters
    ----------
    x : numpy.ndarray
        T x K data matrix.  A 1-D array is automatically promoted to a
        column vector (T x 1).
    demean : bool, optional
        Whether to subtract the column mean before dividing by the standard
        deviation.  Default is ``True``.

    Returns
    -------
    st : numpy.ndarray
        T x K standardized data matrix (same shape as the input).

    Raises
    ------
    ValueError
        If *x* is not a numeric numpy array, or if *demean* is not a scalar
        boolean / logical value.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
    >>> st = standardize(data)
    >>> np.allclose(st.mean(axis=0), 0, atol=1e-14)
    True
    >>> np.allclose(st.std(axis=0, ddof=1), 1, atol=1e-14)
    True
    """

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    # Ensure x is a numpy ndarray.
    if not isinstance(x, np.ndarray):
        try:
            x = np.asarray(x, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "X must be a numeric numpy ndarray or convertible to one."
            ) from exc

    # Handle 1-D input gracefully: treat as a single column vector.
    # Ref: standardize.m:38 — [T, K] = size(x) requires 2-D.
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    elif x.ndim != 2:
        raise ValueError(
            "X must be a 2-D matrix (or 1-D vector). "
            f"Received array with {x.ndim} dimensions."
        )

    # Validate the *demean* parameter.
    # Ref: standardize.m:34-36 — MATLAB checks
    #   if ndims(demean)~=2 || max(size(demean))~=1
    #       error('DEMEAN must be a logical scalar.')
    if not isinstance(demean, (bool, np.bool_)):
        # Accept Python int 0/1 for convenience but reject anything else.
        if isinstance(demean, (int, np.integer)) and demean in (0, 1):
            demean = bool(demean)
        else:
            raise ValueError("DEMEAN must be a logical scalar.")

    # ------------------------------------------------------------------
    # Compute column-wise statistics
    # ------------------------------------------------------------------
    # Ref: standardize.m:38 — [T, K] = size(x);
    # T and K are available via x.shape but not explicitly needed below
    # thanks to NumPy broadcasting.

    # Ref: standardize.m:39 — mu = mean(x);
    # MATLAB mean(x) computes column-wise mean (axis=0 equivalent).
    mu = np.mean(x, axis=0)

    # Ref: standardize.m:40 — stdev = std(x);
    # CRITICAL: MATLAB std uses N-1 normalization (ddof=1) by default.
    stdev = np.std(x, axis=0, ddof=1)

    # ------------------------------------------------------------------
    # Standardization
    # ------------------------------------------------------------------
    # NumPy broadcasting replaces MATLAB repmat(mu, T, 1) / repmat(stdev, T, 1).
    if demean:
        # Ref: standardize.m:42 — st = (x - repmat(mu,T,1)) ./ repmat(stdev,T,1);
        st = (x - mu) / stdev
    else:
        # Ref: standardize.m:44 — st = x ./ repmat(stdev,T,1);
        st = x / stdev

    return st
