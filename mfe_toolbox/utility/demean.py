"""
Demeaning utility for column-wise mean subtraction.

Migrated from utility/demean.m (MFE Toolbox, Version 4.0).
Original author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
Revision: 3, Date: 1/1/2010

This module provides a single function, ``demean``, which subtracts the
column-wise mean from each column of a T×K data matrix, producing a
mean-zero matrix of the same shape.

See Also
--------
mfe_toolbox.utility.standardize : Standardization (demean + unit variance).
"""

import numpy as np


def demean(y: np.ndarray) -> np.ndarray:
    """
    Subtract column means from a data matrix.

    Each column of the input matrix has its mean subtracted so that the
    resulting matrix has column means of zero (within floating-point
    precision).

    Parameters
    ----------
    y : array_like
        T x K data matrix.  If *y* is 1-D it is treated as a single column
        (T,) vector and the scalar mean is subtracted element-wise.

    Returns
    -------
    y_demeaned : numpy.ndarray
        T x K demeaned data matrix where each column has mean ≈ 0.
        The output shape is identical to the input shape.

    Notes
    -----
    The MATLAB implementation (demean.m) computes::

        mu = mean(y, 1);          % column-wise mean
        x  = bsxfun(@minus, y, mu);  % broadcast subtraction

    In NumPy, ``bsxfun(@minus, …)`` is replaced by native broadcasting,
    making the subtraction ``y - mu`` sufficient.

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.utility.demean import demean
    >>> y = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    >>> demean(y)
    array([[-2., -2.],
           [ 0.,  0.],
           [ 2.,  2.]])
    """
    # Convert input to numpy array to accept lists, tuples, etc.
    # Ref: ensures compatibility with array_like inputs (demean.m accepts any numeric matrix)
    y = np.asarray(y, dtype=np.float64)

    # Compute column-wise mean
    # Ref: demean.m:21 — MATLAB mean(y, 1) maps to np.mean(y, axis=0)
    # For 1-D arrays, axis=0 computes the scalar mean of all elements,
    # which is the correct behavior (single "column").
    mu = np.mean(y, axis=0)

    # Broadcast subtract the column means from each row
    # Ref: demean.m:22 — MATLAB bsxfun(@minus, y, mu) maps to y - mu
    # NumPy broadcasting handles this automatically without bsxfun/repmat.
    y_demeaned = y - mu

    return y_demeaned
