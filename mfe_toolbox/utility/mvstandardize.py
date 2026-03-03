"""
Multivariate standardization to produce mean-zero, unit-variance, uncorrelated data.

Migrated from utility/mvstandardize.m (MFE Toolbox, Version 4.0).
Original author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
Revision: 1, Date: 9/1/2005

This module provides the ``mvstandardize`` function which transforms a
T × K data matrix so that each column has zero mean, unit variance, and
columns are uncorrelated (i.e., the sample covariance of the standardized
data approximates the identity matrix).

See Also
--------
mfe_toolbox.utility.demean : Column-wise mean subtraction.
mfe_toolbox.utility.standardize : Univariate standardization.
"""

import numpy as np
import scipy.linalg

from mfe_toolbox.utility.demean import demean


def mvstandardize(x, sigma=None, demean_flag=True):
    """
    Multivariate standardization: mean-zero, unit-variance, uncorrelated columns.

    Transforms the input data matrix so that the standardized output has
    zero column means (if demeaning is enabled), unit column variances, and
    uncorrelated columns.  The transformation is achieved by multiplying the
    (optionally demeaned) data by the inverse matrix square root of the
    covariance matrix (sigma^{-1/2}).

    Parameters
    ----------
    x : numpy.ndarray
        T by K data matrix where T is the number of observations and K is
        the number of variables.
    sigma : numpy.ndarray or None, optional
        Either a K × K covariance matrix (constant over time) or a K × K × T
        array of time-varying covariance matrices.  If ``None`` (default), the
        sample covariance of *x* is used.
    demean_flag : bool, optional
        Whether to demean the data before standardization.  Default is
        ``True``.

    Returns
    -------
    st : numpy.ndarray
        T by K standardized data matrix.

    Raises
    ------
    ValueError
        If *demean_flag* is not a scalar boolean.
        If *x* is not a 2-D matrix.
        If *sigma* dimensions are incompatible with *x*.
        If *sigma* (or any slice of a 3-D sigma) is not positive definite.

    Notes
    -----
    The MATLAB implementation computes ``sigma^(-0.5)`` using MATLAB's matrix
    power operator.  The Python implementation uses ``scipy.linalg.sqrtm`` to
    compute the matrix square root, then ``np.linalg.inv`` to invert it.

    For the time-varying (3-D sigma) case, the inverse matrix square root is
    computed independently for each observation.

    Ref: mvstandardize.m — Author: Kevin Sheppard, Revision 1, 9/1/2005

    See Also
    --------
    mfe_toolbox.utility.demean : Column-wise demeaning.
    mfe_toolbox.utility.standardize : Univariate standardization.

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.utility.mvstandardize import mvstandardize
    >>> rng = np.random.default_rng(42)
    >>> x = rng.standard_normal((100, 3))
    >>> st = mvstandardize(x)
    >>> st.shape
    (100, 3)
    """
    # ------------------------------------------------------------------
    # Convert input to float64 numpy array for safety
    # ------------------------------------------------------------------
    x = np.asarray(x, dtype=np.float64)

    # ------------------------------------------------------------------
    # Validate demean_flag — Ref: mvstandardize.m:37-39
    # MATLAB checks: ndims(demean)~=2 || max(size(demean))~=1
    # In Python, ensure it is a scalar boolean (bool or np.bool_).
    # Note: validation order rearranged from MATLAB source so that x
    # dimensionality is checked *before* np.cov, which would fail on
    # 3-D+ input.
    # ------------------------------------------------------------------
    if not isinstance(demean_flag, (bool, np.bool_)):
        raise ValueError('DEMEAN must be a logical scalar.')

    # ------------------------------------------------------------------
    # Validate x is 2-D — Ref: mvstandardize.m:41-43
    # MATLAB: if ndims(x)>2 → error
    # ------------------------------------------------------------------
    if x.ndim != 2:
        raise ValueError('X must be a T by K matrix with T>K')

    # ------------------------------------------------------------------
    # Default sigma: sample covariance — Ref: mvstandardize.m:29
    # MATLAB cov(x) with default normalisation (N-1) maps to
    # np.cov(x, rowvar=False) which also uses N-1 denominator.
    # Placed after x validation to ensure x is 2-D before np.cov call.
    # ------------------------------------------------------------------
    if sigma is None:
        sigma = np.cov(x, rowvar=False)
    else:
        sigma = np.asarray(sigma, dtype=np.float64)

    T, K = x.shape

    # ------------------------------------------------------------------
    # Validate sigma — Ref: mvstandardize.m:44-59
    # Handle both 2-D (constant) and 3-D (time-varying) covariance.
    # ------------------------------------------------------------------
    if sigma.ndim == 2:
        # 2-D sigma: must be K × K and positive definite
        # Ref: mvstandardize.m:45 — check size and eigenvalues
        if sigma.shape[0] != K or sigma.shape[1] != K:
            raise ValueError('SIGMA must be a K by K positive definite matrix.')
        # Ref: mvstandardize.m:45 — min(eig(sigma))<=0 checks PD
        # (original MATLAB has parenthesisation typo; intended semantics is
        # positive definiteness via eigenvalue check)
        eigenvalues = np.linalg.eigvalsh(sigma)
        if np.min(eigenvalues) <= 0:
            raise ValueError('SIGMA must be a K by K positive definite matrix.')
    elif sigma.ndim == 3:
        # 3-D sigma: must be K × K × T
        # Ref: mvstandardize.m:49-51
        if sigma.shape[0] != K or sigma.shape[1] != K or sigma.shape[2] != T:
            raise ValueError(
                'If SIGMA is a 3-D matrix, it must be K by K by T.'
            )
        # Ref: mvstandardize.m:52-56 — each slice must be positive definite
        for i in range(T):
            eigenvalues_i = np.linalg.eigvalsh(sigma[:, :, i])
            if np.min(eigenvalues_i) <= 0:
                raise ValueError(
                    'If SIGMA is K by K by T, each covariance must be '
                    'positive definite'
                )
    else:
        # Ref: mvstandardize.m:57-59
        raise ValueError(
            'SIGMA must be either a K by K matrix, or a K by K by T matrix.'
        )

    # ------------------------------------------------------------------
    # Demeaning — Ref: mvstandardize.m:61-62
    # MATLAB: if demean, x = demean(x); end
    # Note: MATLAB source has a naming conflict between the parameter
    # 'demean' and the function demean().  Python avoids this by using
    # the parameter name 'demean_flag'.
    # ------------------------------------------------------------------
    if demean_flag:
        # Ref: mvstandardize.m:62 calls demean(x) to subtract column means
        x = demean(x)

    # ------------------------------------------------------------------
    # Standardization — Ref: mvstandardize.m:65-72
    # MATLAB: st = x * sigma^(-0.5)
    # sigma^(-0.5) is the inverse of the matrix square root.
    # Python: scipy.linalg.sqrtm for matrix square root, then
    # np.linalg.inv for inversion.
    # ------------------------------------------------------------------
    if sigma.ndim == 2:
        # Constant covariance path
        # Ref: mvstandardize.m:66 — st = x * sigma^(-0.5)
        sigma_sqrt = scipy.linalg.sqrtm(sigma)
        # For positive definite matrices sqrtm returns a real result,
        # but take real part to handle negligible imaginary artifacts
        # from floating-point arithmetic.
        sigma_sqrt = np.real(sigma_sqrt)
        sigma_sqrt_inv = np.linalg.inv(sigma_sqrt)
        st = x @ sigma_sqrt_inv
    else:
        # Time-varying covariance path: per-observation transformation
        # Ref: mvstandardize.m:68-71 — MATLAB loop for i=1:T (1-indexed)
        # Python: for i in range(T) (0-indexed)
        st = np.zeros((T, K))
        for i in range(T):
            # Ref: mvstandardize.m:70 — st(i,:) = x(i,:) * sigma(:,:,i)^(-0.5)
            sigma_i_sqrt = scipy.linalg.sqrtm(sigma[:, :, i])
            # Take real part to discard negligible imaginary artifacts
            sigma_i_sqrt = np.real(sigma_i_sqrt)
            sigma_i_sqrt_inv = np.linalg.inv(sigma_i_sqrt)
            st[i, :] = x[i, :] @ sigma_i_sqrt_inv

    return st
