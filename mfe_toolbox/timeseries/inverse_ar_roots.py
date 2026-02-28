"""
Inverse AR Roots — Computes the inverted roots of the characteristic equation
of an AR(P) process.

Migrated from timeseries/inverse_ar_roots.m by Kevin Sheppard.
Original MATLAB: Revision 3, Date 1/1/2007.

The process is stationary if min(abs(rho)) > 1, meaning all roots of the
characteristic polynomial lie outside the unit circle.

Usage
-----
>>> import numpy as np
>>> from mfe_toolbox.timeseries.inverse_ar_roots import inverse_ar_roots
>>> rho, stationary = inverse_ar_roots(np.array([0.5]))
>>> # rho contains the inverted roots; stationary is True if process is stationary

See Also
--------
mfe_toolbox.timeseries.armaroots : ARMA root analysis
mfe_toolbox.timeseries.acf : Theoretical autocorrelation function
"""

import numpy as np


def inverse_ar_roots(phi: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Compute the inverted roots of the characteristic equation of an AR(P) process.

    Given AR parameters phi(1), phi(2), ..., phi(P) from the model:
        y(t) = phi(1)*y(t-1) + phi(2)*y(t-2) + ... + phi(P)*y(t-P) + e(t)

    this function computes the roots of the characteristic polynomial:
        1 - phi(1)*z - phi(2)*z^2 - ... - phi(P)*z^P = 0

    The process is stationary if and only if all roots lie strictly outside the
    unit circle, i.e. min(|rho|) > 1.

    Parameters
    ----------
    phi : numpy.ndarray
        A 1-D array or column vector containing the AR parameters in the order
        y(t) = phi[0]*y(t-1) + phi[1]*y(t-2) + ... + phi[P-1]*y(t-P) + e(t).
        Accepts 1-D arrays of shape (P,) or 2-D column vectors of shape (P, 1).

    Returns
    -------
    rho : numpy.ndarray
        A 1-D complex array of length P containing the inverted roots of the
        characteristic equation corresponding to the AR model input.
    stationary : bool
        True if the AR process is stationary (min(|rho|) > 1), False otherwise.

    Raises
    ------
    ValueError
        If ``phi`` is not a 1-D array or column vector (i.e. if it is a matrix
        with more than one column and more than one row such that
        min(shape) > 1).

    Notes
    -----
    - Ref: inverse_ar_roots.m:29 — MATLAB allows column vectors and transposes
      them to row vectors internally. This Python implementation accepts both
      1-D arrays and 2-D column vectors, squeezing them to 1-D before computation.
    - The characteristic polynomial coefficients are constructed as
      ``[-phi[P-1], -phi[P-2], ..., -phi[0], 1]`` in descending power order,
      matching MATLAB's ``roots([-fliplr(phi) 1])`` (Ref: inverse_ar_roots.m:34).
    - Stationarity is determined by checking whether the minimum absolute value
      of all roots exceeds 1 (Ref: inverse_ar_roots.m:35).

    Examples
    --------
    AR(1) with phi=0.5 (stationary):

    >>> rho, stationary = inverse_ar_roots(np.array([0.5]))
    >>> # rho ≈ array([2.0]), stationary = True

    AR(1) with phi=1.0 (unit root, non-stationary):

    >>> rho, stationary = inverse_ar_roots(np.array([1.0]))
    >>> # rho ≈ array([1.0]), stationary = False
    """
    # Convert input to numpy array if not already
    phi = np.asarray(phi, dtype=np.float64)

    # Ref: inverse_ar_roots.m:27 — p = length(phi)
    # Ref: inverse_ar_roots.m:29-32 — Input validation and orientation
    # MATLAB checks: if size(phi,1) >= size(phi,2) && min(size(phi)) == 1
    #   then transpose to row; else error.
    # Python equivalent: accept 1-D arrays and 2-D column vectors (P,1).
    if phi.ndim == 0:
        # Scalar input — treat as AR(1) with single parameter
        phi = phi.reshape(1)
    elif phi.ndim == 1:
        # Already 1-D — no transformation needed
        pass
    elif phi.ndim == 2:
        # 2-D input: must be a column vector (P, 1) or row vector (1, P)
        if min(phi.shape) == 1:
            # Ref: inverse_ar_roots.m:29-30 — squeeze column/row vector to 1-D
            phi = np.squeeze(phi)
        else:
            # Ref: inverse_ar_roots.m:32 — MATLAB: error('Phi should be a column vector.')
            raise ValueError('Phi should be a column vector.')
    else:
        # ndim > 2 — not a valid input
        raise ValueError('Phi should be a column vector.')

    # Ensure phi is non-empty
    if phi.size == 0:
        # Edge case: empty phi means no AR parameters — return empty arrays
        return np.array([], dtype=np.complex128), True

    # Ref: inverse_ar_roots.m:34 — rho = roots([-fliplr(phi) 1])
    # MATLAB roots() and np.roots() both expect polynomial coefficients
    # in descending power order: [a_n, a_{n-1}, ..., a_1, a_0]
    #
    # The characteristic polynomial is:
    #   1 - phi[0]*z - phi[1]*z^2 - ... - phi[P-1]*z^P = 0
    #
    # In descending power order:
    #   -phi[P-1]*z^P - phi[P-2]*z^(P-1) - ... - phi[0]*z + 1
    #
    # Coefficients: [-phi[P-1], -phi[P-2], ..., -phi[0], 1]
    # Which is: np.append(-phi[::-1], 1.0)
    #
    # Ref: inverse_ar_roots.m:34 — MATLAB: roots([-fliplr(phi) 1])
    # -fliplr(phi) reverses and negates; appending 1 gives constant term
    coefficients = np.append(-phi[::-1], 1.0)

    # Compute roots of the characteristic polynomial
    rho = np.roots(coefficients)

    # Ref: inverse_ar_roots.m:35 — stationary = min(abs(rho)) > 1
    # Process is stationary if ALL roots lie strictly outside the unit circle
    stationary = bool(np.min(np.abs(rho)) > 1)

    return rho, stationary
