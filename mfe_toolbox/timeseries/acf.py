"""
Theoretical Autocorrelation Function for ARMA(p,q) Processes

Computes the theoretical autocorrelations and long-run variance of an ARMA(p,q)
process using Yule-Walker equations.

Migrated from timeseries/acf.m by Kevin Sheppard.
Original MATLAB: Revision 3, Date 1/1/2007.

Usage
-----
>>> import numpy as np
>>> from mfe_toolbox.timeseries.acf import acf
>>> autocorr, sigma2_y = acf(np.array([0.5]), np.array([]), 10)
>>> # autocorr contains 11 autocorrelations for AR(1) with phi=0.5

Notes
-----
The ARMA model is parameterized as::

    y(t) = phi(1)*y(t-1) + phi(2)*y(t-2) + ... + phi(p)*y(t-p)
         + e(t) + theta(1)*e(t-1) + theta(2)*e(t-2) + ... + theta(q)*e(t-q)

To compute autocorrelations for an ARMA that does not include all lags 1 to P,
insert 0 for any excluded lag. For example, if the model was
    y(t) = phi(2)*y(t-2) + e(t)
then phi = [0, phi2].

See Also
--------
mfe_toolbox.timeseries.inverse_ar_roots : Inverted AR polynomial roots
mfe_toolbox.timeseries.sacf : Sample autocorrelation function

Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
"""

import numpy as np
from scipy.linalg import toeplitz

from mfe_toolbox.timeseries.inverse_ar_roots import inverse_ar_roots


def acf(phi, theta, N, sigma2_e=1.0):
    """
    Compute theoretical autocorrelations and long-run variance of an ARMA(p,q) process.

    Uses Yule-Walker equations to compute the theoretical autocovariance function
    of a stationary ARMA(p,q) process, then normalizes to obtain autocorrelations.

    Parameters
    ----------
    phi : array_like
        Autoregressive parameters, in the order t-1, t-2, .... Can be a scalar,
        1-D array, or empty array for pure MA processes.
    theta : array_like
        Moving average parameters, in the order t-1, t-2, .... Can be a scalar,
        1-D array, or empty array for pure AR processes.
    N : int
        Number of autocorrelations to compute. Must be a non-negative integer.
        The returned autocorrelation vector will have N+1 elements (including lag 0).
    sigma2_e : float, optional
        Variance of errors. Must be positive. Default is 1.0.

    Returns
    -------
    autocorr : numpy.ndarray
        (N+1,) array of autocorrelations. autocorr[0] is always 1.0 (lag 0).
        To recover autocovariances, use ``autocov = autocorr * sigma2_y``.
    sigma2_y : float
        Long-run variance (gamma_0) of the ARMA process with innovation variance
        sigma2_e.

    Raises
    ------
    ValueError
        If N is not a non-negative scalar integer.
        If sigma2_e is not a positive scalar.
        If phi or theta is not a vector (1-D or column vector).
        If the AR parameters in phi do not correspond to a stationary process.

    Notes
    -----
    The ARMA model is parameterized as::

        y(t) = phi[0]*y(t-1) + phi[1]*y(t-2) + ... + phi[p-1]*y(t-p)
             + e(t) + theta[0]*e(t-1) + theta[1]*e(t-2) + ... + theta[q-1]*e(t-q)

    The algorithm constructs a system of linear equations based on the autocovariance
    structure of the ARMA process::

        phi_transformed * gamma = theta_transformed * delta

    where ``delta[i] = E[y_t * e(t-i)]`` and the error variance is ``sigma2_e``.

    For lags beyond ``max(p, q)``, the Yule-Walker recursion is used::

        gamma(k) = phi[0]*gamma(k-1) + phi[1]*gamma(k-2) + ... + phi[p-1]*gamma(k-p)

    References
    ----------
    Hamilton, J.D. (1994). Time Series Analysis. Princeton University Press.

    Examples
    --------
    AR(1) with phi=0.5:

    >>> import numpy as np
    >>> from mfe_toolbox.timeseries.acf import acf
    >>> autocorr, sigma2_y = acf(np.array([0.5]), np.array([]), 5)
    >>> # autocorr ≈ [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]
    """
    # ========================================================================
    # Input Validation
    # Ref: acf.m:30-58
    # ========================================================================

    # Ref: acf.m:34-36 — Validate N is a non-negative scalar integer
    # MATLAB: if length(N)>1 || any(N<0) || N~=floor(N)
    if not isinstance(N, (int, np.integer)) or N < 0:
        raise ValueError('N must be a non-negative scalar integer.')

    # Ref: acf.m:38-42 — Validate sigma2_e is a positive scalar
    # MATLAB: if length(sigma2_e)>1 || any(sigma2_e<=0)
    if np.asarray(sigma2_e).size != 1:
        raise ValueError('sigma2_e must be a positive scalar')
    sigma2_e = float(sigma2_e)
    if sigma2_e <= 0:
        raise ValueError('sigma2_e must be a positive scalar')

    # Ref: acf.m:44-50 — Convert phi to 1D numpy array
    # MATLAB ensures column vector; Python ensures 1D via flatten
    phi = np.asarray(phi, dtype=np.float64)
    if phi.ndim > 1 and min(phi.shape) > 1:
        # Ref: acf.m:46 — error('phi must be a vector')
        raise ValueError('phi must be a vector')
    phi = phi.flatten()

    # Ref: acf.m:52-58 — Convert theta to 1D numpy array
    # MATLAB ensures column vector; Python ensures 1D via flatten
    theta = np.asarray(theta, dtype=np.float64)
    if theta.ndim > 1 and min(theta.shape) > 1:
        # Ref: acf.m:54 — error('theta must be a vector')
        raise ValueError('theta must be a vector')
    theta = theta.flatten()

    # Ref: acf.m:44, 52 — Compute AR and MA orders
    p = len(phi)
    q = len(theta)

    # ========================================================================
    # Stationarity Check
    # Ref: acf.m:60-67
    # ========================================================================

    if p > 0:
        # Ref: acf.m:63 — [rho, stationary] = inverse_ar_roots(phi)
        _rho, stationary = inverse_ar_roots(phi)
        if not stationary:
            # Ref: acf.m:65
            raise ValueError(
                'Autoregressive roots (phi) do not correspond to a stationary process'
            )

    # ========================================================================
    # Core ARMA ACF Computation
    # Ref: acf.m:69-126
    # ========================================================================

    # Ref: acf.m:70 — Save the original values before augmentation
    phi_original = phi.copy()

    # Ref: acf.m:76 — phi = [1; -phi; zeros(q-p, 1)]
    # Since the model form is Phi(L)y(t) = Theta(L)e(t), where
    # Phi(L) = 1 - phi1*L - phi2*L^2 - ..., we reverse the sign of AR params
    # CRITICAL: Note the sign reversal of AR parameters
    phi_aug = np.concatenate([
        np.array([1.0]),
        -phi,
        np.zeros(max(0, q - p))
    ])

    # Ref: acf.m:77 — theta = [1; theta; zeros(p-q, 1)]
    theta_aug = np.concatenate([
        np.array([1.0]),
        theta,
        np.zeros(max(0, p - q))
    ])

    # Ref: acf.m:79 — m is the dimension of the system of linear equations
    # m = max(p, q) + 1
    m = max(p, q) + 1

    # ====================================================================
    # Step 5b: Construct phi_transformed matrix
    # Ref: acf.m:81-93
    # These are all functions of the autoregressive parameters.
    # The autocovariances are found by solving:
    #   phi_transformed * gamma = theta_transformed * delta
    # where delta(i) = E[y_t * e(t-i)], assuming unit error variance.
    # ====================================================================

    # Ref: acf.m:87 — phi_transformed = zeros(m, m)
    phi_transformed = np.zeros((m, m))

    # Ref: acf.m:88 — T = toeplitz(1:m)
    # Creates a symmetric Toeplitz matrix with [1, 2, ..., m] as first row/column
    # MATLAB uses 1-indexed values; these are used as column indices later
    T = toeplitz(np.arange(1, m + 1))

    # Ref: acf.m:89-93 — Fill phi_transformed using Toeplitz index structure
    for i in range(m):
        for j in range(m):
            # Ref: acf.m:91 — MATLAB 1-based: phi_transformed(i, T(i,j)) += phi(j)
            # Python 0-based: column index is T[i,j]-1 (convert MATLAB 1-indexed to 0-indexed)
            col_idx = T[i, j] - 1
            phi_transformed[i, col_idx] += phi_aug[j]

    # ====================================================================
    # Step 5c: Compute delta vector
    # Ref: acf.m:95-98
    # delta(i) = E[y_t * e(t-i)] can be found by solving:
    #   tril(toeplitz(phi_aug)) * delta = theta_aug
    # ====================================================================

    # Ref: acf.m:98 — delta = tril(toeplitz(phi))^(-1) * theta
    # Using np.linalg.solve instead of matrix inverse for numerical stability
    phi_toeplitz_lower = np.tril(toeplitz(phi_aug))
    delta = np.linalg.solve(phi_toeplitz_lower, theta_aug)

    # ====================================================================
    # Step 5d: Construct theta_transformed matrix
    # Ref: acf.m:100-107
    # The form of theta_transformed is:
    # [theta(0) theta(1)    theta(2)   ...         theta(q-1)  theta(q)]
    # [theta(1) theta(2)    ...        theta(q-1)  theta(q)    0       ]
    # [theta(2) theta(3)    ...        theta(q)    0           0       ]
    # [...      ...         ...        ...         ...         ...     ]
    # [theta(q) 0           0          ...         ...         0       ]
    # ====================================================================

    # Ref: acf.m:107 — theta_transformed = flipud(tril(toeplitz(flipud(theta))))
    theta_transformed = np.flipud(np.tril(toeplitz(np.flipud(theta_aug))))

    # ====================================================================
    # Step 5e: Compute first m autocovariances
    # Ref: acf.m:109-110
    # ====================================================================

    # Ref: acf.m:110 — autocov = phi_transformed^(-1) * (theta_transformed * delta)
    # Using np.linalg.solve instead of matrix inverse for numerical stability
    autocov = np.linalg.solve(phi_transformed, theta_transformed @ delta)

    # ====================================================================
    # Step 5f: Extend or truncate autocovariances
    # Ref: acf.m:112-124
    # ====================================================================

    # Ref: acf.m:114 — If fewer than m autocovariances needed, truncate
    if N + 1 < m:
        autocov = autocov[:N + 1]
    # Ref: acf.m:116-123 — If more than m needed, extend using Yule-Walker recursion
    elif N + 1 > m:
        # Ref: acf.m:117 — autocov = [autocov; zeros(N-m+1, 1)]
        autocov = np.concatenate([autocov, np.zeros(N - m + 1)])
        # Ref: acf.m:119 — Yule-Walker recursion only applies when p > 0
        if p > 0:
            for i in range(m, N + 1):
                # Ref: acf.m:121 — MATLAB: autocov(i+1) = phi_original' * autocov(i:-1:(i-p+1))
                # Python 0-indexed: autocov[i] corresponds to MATLAB autocov(i+1)
                # MATLAB autocov(i:-1:(i-p+1)) selects p elements from index i
                #   down to i-p+1 (1-based inclusive)
                # Python: autocov[i-1:i-1-p:-1] selects p elements from index i-1
                #   down to i-p (0-based inclusive)
                autocov[i] = phi_original @ autocov[i - 1:i - 1 - p:-1]

    # ====================================================================
    # Compute Final Outputs
    # Ref: acf.m:125-126
    # ====================================================================

    # Ref: acf.m:125 — sigma2_y = sigma2_e * autocov(1) (MATLAB 1-indexed = Python index 0)
    sigma2_y = float(sigma2_e * autocov[0])

    # Ref: acf.m:126 — autocorr = autocov ./ autocov(1) (normalize by lag-0 autocovariance)
    autocorr = autocov / autocov[0]

    return autocorr, sigma2_y
