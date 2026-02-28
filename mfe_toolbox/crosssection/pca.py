"""
Principal Component Analysis with 3 normalization modes.

Migrated from crosssection/pca.m (120 lines) — MFE Toolbox Version 4.0.
Provides PCA using the outer-product, covariance, or correlation matrix
of the input data for eigendecomposition.

Author: Kevin Sheppard
Original MATLAB Revision: 3.01, Date: 2/18/2014
Python migration: faithful 1:1 translation with numpy.linalg.eigh
"""

import numpy as np


def pca(data: np.ndarray, type: str = 'outer'):
    """
    Principal Component Analysis with outer-product, covariance, or correlation matrix.

    Performs eigendecomposition of the selected matrix representation of the
    input data and returns component weights, principal components, eigenvalues,
    explained variance fractions, and cumulative R-squared values.

    When using the 'outer' version of PCA:
        data = princomp @ weights

    When using the 'cov' version of PCA:
        e = princomp @ weights
    where e[:, i] = data[:, i] - mean(data[:, i])

    When using the 'corr' version of PCA:
        e = princomp @ weights
    where e[:, i] = (data[:, i] - mean(data[:, i])) / std(data[:, i])

    Parameters
    ----------
    data : numpy.ndarray
        T x K data matrix where T is the number of observations and K is
        the number of variables.
    type : str, optional
        Determines the matrix used for eigendecomposition. One of:
        - 'outer' (default): uncentered outer-product matrix (data' * data / T)
        - 'cov': sample covariance matrix of demeaned data
        - 'corr': sample correlation matrix of standardized data
        Case-insensitive matching is used.

    Returns
    -------
    weights : numpy.ndarray
        K x K matrix of component weights. The i-th row corresponds to the
        i-th principal component (ordered by decreasing explained variance).
    princomp : numpy.ndarray
        T x K matrix of principal components. The i-th column corresponds to
        the i-th principal component (ordered by decreasing explained variance).
    eigenvals : numpy.ndarray
        K-element array of eigenvalues sorted in descending order.
    explvar : numpy.ndarray
        K-element array of fraction of variance explained by each component.
    cumR2 : numpy.ndarray
        K-element array of cumulative R-squared from including components
        1, 2, ..., i.

    Raises
    ------
    ValueError
        If data is not a 2-D matrix, if type is not recognized, or if a
        zero-variance column is present in 'corr' mode.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> data = np.random.randn(100, 5)
    >>> w, pc, ev, expl, cr2 = pca(data)
    >>> w.shape
    (5, 5)
    >>> pc.shape
    (100, 5)
    """
    # =========================================================================
    # Input Validation — Ref: pca.m lines 40-68
    # =========================================================================

    # Ensure data is a numpy array for consistent behavior
    data = np.asarray(data, dtype=np.float64)

    # Get data dimensions — Ref: pca.m:44  T=size(data,1)
    T = data.shape[0]

    # 2-D check — Ref: pca.m:46-47  if ndims(data)~=2
    if data.ndim != 2:
        raise ValueError('DATA must be a T by K matrix')

    # Type parsing with case-insensitive matching — Ref: pca.m:50-68
    # Handle the type parameter; use internal name to avoid shadowing issues
    # Ref: pca.m:54  lower(type)
    type_lower = type.lower() if isinstance(type, str) else ''

    if type_lower == '' or type_lower == 'outer':
        # Ref: pca.m:55-58  case '' → pca_type=1; case 'outer' → pca_type=1
        pca_type = 1
    elif type_lower == 'cov':
        # Ref: pca.m:59  case 'cov' → pca_type=2
        pca_type = 2
    elif type_lower == 'corr':
        # Ref: pca.m:61  case 'corr' → pca_type=3
        pca_type = 3
    else:
        # Ref: pca.m:63-64
        raise ValueError("TYPE must be either 'cov' or 'corr'.")

    # =========================================================================
    # Matrix Construction — Ref: pca.m lines 76-95
    # =========================================================================

    # Make a working copy to avoid modifying the caller's array
    # This is important because cov and corr modes demean/standardize data
    data = data.copy()

    if pca_type == 1:
        # Outer product mode: uncentered second moment matrix
        # Data is NOT demeaned in this mode
        # Ref: pca.m:79-80  inputmat=data'*data/T
        inputmat = data.T @ data / T

    elif pca_type == 2:
        # Covariance mode: demean, then compute sample covariance
        # Ref: pca.m:83  data = data-repmat(mean(data),T,1)
        # Python broadcasting handles the repmat automatically
        data = data - np.mean(data, axis=0)
        # Ref: pca.m:84  inputmat = cov(data)
        # MATLAB cov(data) normalizes by (T-1); np.cov with bias=False also uses (T-1)
        inputmat = np.cov(data, rowvar=False, bias=False)

    else:  # pca_type == 3
        # Correlation mode: demean, standardize, then compute covariance
        # of standardized data (which equals the correlation matrix)
        # Ref: pca.m:87  data = data-repmat(mean(data),T,1)
        data = data - np.mean(data, axis=0)

        # Ref: pca.m:89  stdevs = std(data)
        # MATLAB std(data) uses N-1 normalization (ddof=1)
        stdevs = np.std(data, axis=0, ddof=1)

        # Ref: pca.m:90-91  if any(stdevs==0) → error(...)
        if np.any(stdevs == 0):
            raise ValueError(
                'One or more of the colums of DATA has no variation, '
                'and the correlation-based method is not applicable.'
            )

        # Ref: pca.m:93  data = data./repmat(stdevs,T,1)
        # Python broadcasting handles the repmat automatically
        data = data / stdevs

        # Ref: pca.m:94  inputmat=cov(data)
        # Covariance of standardized data equals the correlation matrix
        inputmat = np.cov(data, rowvar=False, bias=False)

    # =========================================================================
    # Eigendecomposition and Sorting — Ref: pca.m lines 98-120
    # =========================================================================

    # Ref: pca.m:99  [eigenvects,eigenvals]=eig(inputmat)
    # AAP mandates numpy.linalg.eigh for symmetric eigenvalue decomposition.
    # eigh is numerically stable and efficient for symmetric/Hermitian matrices.
    # eigh returns eigenvalues in ASCENDING order (unlike MATLAB eig which is unsorted).
    eigenvals_raw, eigenvects_raw = np.linalg.eigh(inputmat)

    # Ref: pca.m:101  [eigenvals, order] = sort(diag(eigenvals))
    # eigh already returns ascending, but argsort for robustness
    order = np.argsort(eigenvals_raw)
    eigenvals_sorted = eigenvals_raw[order]
    eigenvects_sorted = eigenvects_raw[:, order]

    # Ref: pca.m:105  eigenvals=rot90(eigenvals,2)
    # rot90 of a column vector twice = reversal → descending order
    eigenvals = eigenvals_sorted[::-1].copy()

    # Ref: pca.m:107  weights=eigenvects
    # At this point eigenvectors are still in ascending column order
    # (matching MATLAB after sort but before flip)
    weights = eigenvects_sorted

    # Ref: pca.m:109  princomp=data*eigenvects
    # Principal components computed with ascending-order eigenvectors
    princomp = data @ eigenvects_sorted

    # Ref: pca.m:111  explvar=eigenvals/sum(eigenvals)
    # Explained variance uses descending eigenvalues
    total_eigenvalue_sum = np.sum(eigenvals)
    explvar = eigenvals / total_eigenvalue_sum

    # Ref: pca.m:114  cumR2=cumsum(explvar)
    cumR2 = np.cumsum(explvar)

    # Ref: pca.m:116  weights=weights'
    # Transpose so rows correspond to components (ascending order)
    weights = weights.T

    # Ref: pca.m:119  weights=flipud(weights)
    # Flip rows so most important component is first (descending)
    weights = np.flipud(weights).copy()

    # Ref: pca.m:120  princomp=fliplr(princomp)
    # Flip columns so most important component is first (descending)
    princomp = np.fliplr(princomp).copy()

    return weights, princomp, eigenvals, explvar, cumR2
