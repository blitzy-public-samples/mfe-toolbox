"""
Convert covariance matrices to correlation matrices with standard deviations.

Migrated from utility/cov2corr.m — Author: Kevin Sheppard
Revision: 1, Date: 10/23/2012

This module provides the :func:`cov2corr` function which decomposes a
covariance matrix (or a time-varying stack of covariance matrices) into
a vector (or matrix) of standard deviations and a correlation matrix
(or stack of correlation matrices).
"""

import numpy as np


def cov2corr(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert covariance matrices to correlation matrices.

    Decomposes a K × K covariance matrix (or K × K × T array of
    time-varying covariance matrices) into standard deviations and
    a correlation matrix (or array of correlation matrices).

    Parameters
    ----------
    cov : numpy.ndarray
        K × K covariance matrix **or** K × K × T array of covariance
        matrices (e.g. from a DCC model).

    Returns
    -------
    sigma : numpy.ndarray
        K-element vector of standard deviations (2-D input) **or**
        T × K matrix of standard deviations (3-D input).
    correl : numpy.ndarray
        K × K correlation matrix (2-D input) **or**
        K × K × T array of correlation matrices (3-D input).

    Raises
    ------
    ValueError
        If *cov* is not a 2-D or 3-D numpy array, or if the leading
        dimensions are not square.

    Notes
    -----
    The correlation matrix is computed as:

    .. math::
        R = D^{-1} \\Sigma D^{-1}

    where *D* = diag(σ₁, …, σ_K) and σ_i = √(Σ_{ii}).

    Examples
    --------
    Single covariance matrix (DCC output for one time period):

    >>> import numpy as np
    >>> C = np.array([[4.0, 2.0], [2.0, 9.0]])
    >>> sigma, correl = cov2corr(C)
    >>> sigma  # doctest: +NORMALIZE_WHITESPACE
    array([2., 3.])
    >>> correl  # doctest: +NORMALIZE_WHITESPACE
    array([[1.        , 0.33333333],
           [0.33333333, 1.        ]])

    See Also
    --------
    mfe_toolbox.multivariate.dcc : DCC model producing time-varying Ht.
    """

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    cov = np.asarray(cov, dtype=float)

    if cov.ndim not in (2, 3):
        raise ValueError(
            "COV must be a 2-D (K x K) or 3-D (K x K x T) numpy array. "
            f"Received array with {cov.ndim} dimensions."
        )

    # For both 2-D and 3-D, the first two dimensions must be square.
    if cov.shape[0] != cov.shape[1]:
        raise ValueError(
            "The first two dimensions of COV must be equal (K x K). "
            f"Received shape {cov.shape}."
        )

    # ------------------------------------------------------------------
    # 2-D case: single covariance matrix  (Ref: cov2corr.m:28-30)
    # ------------------------------------------------------------------
    if cov.ndim == 2:
        # Ref: cov2corr.m:29 — sigma = sqrt(diag(cov))
        sigma = np.sqrt(np.diag(cov))

        # Ref: cov2corr.m:30 — correl = cov ./ (sigma * sigma')
        # MATLAB (sigma * sigma') produces the outer product because sigma
        # is a column vector; the Python equivalent is np.outer(sigma, sigma).
        correl = cov / np.outer(sigma, sigma)

        return sigma, correl

    # ------------------------------------------------------------------
    # 3-D case: time-varying covariance matrices  (Ref: cov2corr.m:31-39)
    # ------------------------------------------------------------------
    # Ref: cov2corr.m:32-33 — T = size(cov, 3); K = size(cov, 2)
    K = cov.shape[0]
    T = cov.shape[2]

    # Pre-allocate output arrays — Ref: cov2corr.m:34-35
    sigma = np.zeros((T, K))
    correl = np.zeros((K, K, T))

    # Ref: cov2corr.m:36-38 — MATLAB loop for t=1:T (1-indexed)
    # Python uses 0-indexed range(T).
    for t in range(T):
        # Ref: cov2corr.m:37 — sigma(t,:) = sqrt(diag(cov(:,:,t)))'
        # np.diag extracts the diagonal; .T in MATLAB transposes the column
        # to a row for assignment into sigma(t,:). In Python the 1-D array
        # from np.diag is naturally compatible with row assignment.
        sigma[t, :] = np.sqrt(np.diag(cov[:, :, t]))

        # Ref: cov2corr.m:38 — correl(:,:,t) = cov(:,:,t) ./ (sigma(t,:)' * sigma(t,:))
        # In MATLAB, sigma(t,:)' * sigma(t,:) is the outer product
        # (column × row → K×K). Python equivalent: np.outer.
        correl[:, :, t] = cov[:, :, t] / np.outer(sigma[t, :], sigma[t, :])

    return sigma, correl
