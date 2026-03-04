"""
Theoretical Partial Autocorrelation Function for ARMA(p,q) Processes

Computes the theoretical partial autocorrelations of an ARMA(p,q) process using
a partitioned matrix inverse approach (Levinson-Durbin style recursion on the
Yule-Walker Toeplitz system).

Migrated from timeseries/pacf.m by Kevin Sheppard.
Original MATLAB: Revision 3, Date 1/1/2007.

Usage
-----
>>> import numpy as np
>>> from mfe_toolbox.timeseries.pacf import pacf
>>> pautocorr = pacf(np.array([0.5]), np.array([]), 5)
>>> # pautocorr contains 6 partial autocorrelations for AR(1) with phi=0.5

Notes
-----
The ARMA model is parameterized as::

    y(t) = phi(1)*y(t-1) + phi(2)*y(t-2) + ... + phi(p)*y(t-p)
         + e(t) + theta(1)*e(t-1) + theta(2)*e(t-2) + ... + theta(q)*e(t-q)

To compute partial autocorrelations for an ARMA that does not include all lags
1 to P, insert 0 for any excluded lag. For example, if the model was
    y(t) = phi(2)*y(t-2) + e(t)
then phi = [0, phi2].

See Also
--------
mfe_toolbox.timeseries.acf : Theoretical autocorrelation function
mfe_toolbox.timeseries.spacf : Sample partial autocorrelation function

Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
"""

import numpy as np
from scipy.linalg import toeplitz

from mfe_toolbox.timeseries.acf import acf


def pacf(phi, theta, n):
    """
    Compute theoretical partial autocorrelations of an ARMA(p,q) process.

    Uses a partitioned matrix inverse approach (Levinson-Durbin style recursion)
    applied to the Toeplitz autocorrelation matrix to derive partial
    autocorrelations from the theoretical autocorrelation function.

    Parameters
    ----------
    phi : array_like
        Autoregressive parameters, in the order t-1, t-2, .... Can be a scalar,
        1-D array, or empty array for pure MA processes.
    theta : array_like
        Moving average parameters, in the order t-1, t-2, .... Can be a scalar,
        1-D array, or empty array for pure AR processes.
    n : int
        Number of partial autocorrelations to compute. Must be a positive
        integer. The returned array will have n+1 elements (including lag 0 = 1).

    Returns
    -------
    pautocorr : numpy.ndarray
        (n+1,) array of partial autocorrelations. pautocorr[0] is always 1.0
        (lag 0). pautocorr[k] is the partial autocorrelation at lag k for
        k = 1, ..., n.

    Raises
    ------
    ValueError
        If n is not a positive integer scalar.
        If phi is not a vector (1-D or convertible to 1-D).
        If theta is not a vector (1-D or convertible to 1-D).

    Notes
    -----
    The partial autocorrelation at lag k is the coefficient of y(t-k) in the
    optimal linear prediction of y(t) given y(t-1), ..., y(t-k). For a pure
    AR(p) process, the PACF is exactly zero for lags k > p.

    The algorithm:

    1. Compute theoretical autocorrelations via :func:`acf`.
    2. Set PACF(1) = rho(1) (first autocorrelation).
    3. For k >= 2, build the Toeplitz system X'X incrementally using
       the Schur complement (partitioned matrix inverse), and solve
       X'X * beta = X'y to obtain beta(k) = PACF(k).

    References
    ----------
    Hamilton, J.D. (1994). Time Series Analysis. Princeton University Press,
    Chapter 3.

    Examples
    --------
    AR(1) with phi=0.5 — PACF cuts off after lag 1:

    >>> import numpy as np
    >>> from mfe_toolbox.timeseries.pacf import pacf
    >>> result = pacf(np.array([0.5]), np.array([]), 5)
    >>> # result ≈ [1.0, 0.5, 0.0, 0.0, 0.0, 0.0]

    MA(1) with theta=0.5 — PACF decays exponentially:

    >>> result = pacf(np.array([]), np.array([0.5]), 5)
    >>> # result[0] = 1.0, result[1] = 0.4, result[2] ≈ -0.1905, ...
    """
    # ========================================================================
    # Input Validation
    # Ref: pacf.m:31-50
    # ========================================================================

    # Ref: pacf.m:34-36 — Validate n is a positive integer scalar
    # MATLAB: if ~isscalar(N) && N>0 && floor(N)==N  (intent: reject non-positive or non-integer)
    # Python: explicit scalar + positive + integer check
    if np.isscalar(n):
        if not isinstance(n, (int, np.integer)):
            # Allow float values that are exactly integer (e.g. 5.0)
            if isinstance(n, float) and n == int(n) and n > 0:
                n = int(n)
            else:
                raise ValueError('N must be a positive integer')
        elif n <= 0:
            raise ValueError('N must be a positive integer')
    else:
        raise ValueError('N must be a positive integer')

    # Ref: pacf.m:37-43 — Validate phi is a vector and ensure 1-D
    phi = np.asarray(phi, dtype=np.float64)
    if phi.ndim > 1 and min(phi.shape) > 1:
        # Ref: pacf.m:39 — error('phi must be a vector')
        raise ValueError('phi must be a vector')
    phi = phi.ravel()

    # Ref: pacf.m:44-50 — Validate theta is a vector and ensure 1-D
    theta = np.asarray(theta, dtype=np.float64)
    if theta.ndim > 1 and min(theta.shape) > 1:
        # Ref: pacf.m:46 — error('theta must be a vector')
        raise ValueError('theta must be a vector')
    theta = theta.ravel()

    # ========================================================================
    # Step 1: Get theoretical autocorrelations
    # Ref: pacf.m:56-57
    # ========================================================================

    # Ref: pacf.m:56 — ac = acf(phi, theta, N+1)
    # Python acf returns (autocorr, sigma2_y) tuple; we only need autocorrelations
    autocorr_full, _ = acf(phi, theta, n + 1)

    # Ref: pacf.m:57 — ac = ac(2:N+1) → Python: ac = autocorr_full[1:n+1]
    # MATLAB 1-based indexing: ac(2:N+1) selects N elements (lags 1 through N)
    # Python 0-based indexing: autocorr_full[1:n+1] selects n elements (lags 1 through n)
    ac = autocorr_full[1:n + 1]

    # ========================================================================
    # Step 2: Initialize PACF output
    # Ref: pacf.m:60-63
    # ========================================================================

    # Ref: pacf.m:60 — pac = zeros(N, 1) → Python: 1-D array of length n
    pac = np.zeros(n)

    # Ref: pacf.m:63 — pac(1) = ac(1)
    # First partial autocorrelation equals the first autocorrelation
    # MATLAB 1-based: pac(1) = ac(1) → Python 0-based: pac[0] = ac[0]
    pac[0] = ac[0]

    # ========================================================================
    # Step 3: Second PACF via regression (n >= 2)
    # Ref: pacf.m:65-71
    # ========================================================================

    if n >= 2:
        # Ref: pacf.m:67 — XpX = toeplitz([1 ac(1)])
        # Constructs 2×2 Toeplitz autocorrelation matrix: [[1, rho(1)], [rho(1), 1]]
        XpX = toeplitz([1.0, ac[0]])

        # Ref: pacf.m:68 — XpXinv = XpX^(-1) → Python: np.linalg.inv
        XpXinv = np.linalg.inv(XpX)

        # Ref: pacf.m:69-70 — Xpy = ac(1:2); temp = XpXinv * Xpy
        # MATLAB 1-based: ac(1:2) = first 2 autocorrelations
        # Python 0-based: ac[0:2] = first 2 autocorrelations
        Xpy = ac[0:2]
        temp = XpXinv @ Xpy

        # Ref: pacf.m:71 — pac(2) = temp(2) → Python: pac[1] = temp[1]
        pac[1] = temp[1]

        # ====================================================================
        # Step 4: Remaining PACF via partitioned inverse recursion
        # Ref: pacf.m:73-86
        # ====================================================================

        # Ref: pacf.m:73 — for i=3:N → Python: for i in range(2, n)
        # MATLAB i=3 corresponds to Python i=2 (0-based loop variable)
        for i in range(2, n):
            # Ref: pacf.m:74 — Ainv = XpXinv (from previous iteration)
            Ainv = XpXinv

            # Ref: pacf.m:75 — B = ac(i-1:-1:1)
            # MATLAB: reversed autocorrelations from index (i-1) down to 1
            # At MATLAB i=3: B = ac(2:-1:1) = [ac(2), ac(1)] (2 elements)
            # Python i=2 (= MATLAB i-1): ac[i-1::-1] gives correct reversal
            # At Python i=2: ac[1::-1] = [ac[1], ac[0]] (2 elements) ✓
            B = ac[i - 1::-1].reshape(-1, 1)  # Column vector (k, 1)

            # Ref: pacf.m:76 — C = B' (transpose: row vector)
            C = B.T  # Row vector (1, k)

            # Ref: pacf.m:77 — D = 1 (Toeplitz diagonal is always 1)
            D = 1.0

            # Ref: pacf.m:78 — Schur complement inversion formula
            # SDinv = Ainv + Ainv*B*(D-C*Ainv*B)^(-1)*C*Ainv
            # Compute intermediate products for clarity
            AinvB = Ainv @ B               # (k, 1)
            CAinv = C @ Ainv               # (1, k)
            schur_complement = D - (C @ AinvB).item()  # scalar
            schur_inv = 1.0 / schur_complement
            SDinv = Ainv + schur_inv * (AinvB @ CAinv)  # (k, k)

            # Ref: pacf.m:80-81 — Build new XpXinv using partitioned inverse formula
            # XpXinv = [SDinv, -SDinv*B*D^(-1);
            #           -D^(-1)*C*SDinv, D^(-1)+D^(-1)*C*SDinv*B*D^(-1)]
            # Since D = 1, D^(-1) = 1, the formula simplifies:
            SDinvB = SDinv @ B              # (k, 1)
            CSDinv = C @ SDinv              # (1, k)
            CSDinvB = (CSDinv @ B).item()   # scalar: C @ SDinv @ B

            top = np.hstack([SDinv, -SDinvB])           # (k, k+1)
            bottom_left = -CSDinv                        # (1, k)
            bottom_right = np.array([[1.0 + CSDinvB]])   # (1, 1)
            bottom = np.hstack([bottom_left, bottom_right])  # (1, k+1)
            XpXinv = np.vstack([top, bottom])            # (k+1, k+1)

            # Ref: pacf.m:82 — Symmetrize to reduce numerical drift
            XpXinv = (XpXinv + XpXinv.T) / 2.0

            # Ref: pacf.m:83-84 — Solve regression: temp = XpXinv * ac(1:i)
            # MATLAB at i: ac(1:i) selects first i autocorrelations
            # Python at i (= MATLAB i-1): need first (i+1) autocorrelations
            # ac[0:i+1] gives (i+1) elements → matches matrix dimension (i+1)×(i+1)
            Xpy = ac[0:i + 1]
            temp = XpXinv @ Xpy

            # Ref: pacf.m:85 — pac(i) = temp(i) → Python: pac[i] = temp[i]
            # The last regression coefficient is the partial autocorrelation
            pac[i] = temp[i]

    # ========================================================================
    # Step 5: Construct final output
    # Ref: pacf.m:89-90
    # ========================================================================

    # Ref: pacf.m:89 — pautocorr = [1; pac] → prepend 1.0 for lag-0
    pautocorr = np.concatenate([np.array([1.0]), pac])

    # Ref: pacf.m:90 — pautocorr(abs(pautocorr)<100*eps) = 0
    # Zero out values smaller than 100 * machine epsilon to clean numerical noise
    eps_threshold = 100.0 * np.finfo(float).eps
    pautocorr[np.abs(pautocorr) < eps_threshold] = 0.0

    return pautocorr
