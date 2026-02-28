"""
ARMA Root Analysis — Compute roots of the AR characteristic equation.

Migrated from timeseries/armaroots.m (Kevin Sheppard, Revision 3, 1/1/2007).

Computes the roots of the characteristic equation of an ARMAX(P,Q) model
as parameterized by armaxfilter. Centralizes AR polynomial coefficient
scattering, parameter-length enforcement, and complex-root modulus reporting.

Examples
--------
Compute the AR roots of an ARMA(2,2):

>>> import numpy as np
>>> phi = np.array([1.3, -0.35])
>>> theta = np.array([0.4, 0.3])
>>> parameters = np.array([1.0, 1.3, -0.35, 0.4, 0.3])
>>> arroots, absarroots = armaroots(parameters, 1, np.array([1, 2]), np.array([1, 2]))

Compute the AR roots of an irregular AR(3) (lags 1 and 3 only):

>>> phi = np.array([1.3, -0.35])
>>> parameters = np.array([1.0, 1.3, -0.35])
>>> arroots, absarroots = armaroots(parameters, 1, np.array([1, 3]), np.array([]))

See Also
--------
armaxfilter : ARMAX model estimation driver.
"""

import numpy as np


def armaroots(parameters, constant, p, q, x=None):
    """
    Compute the roots of the characteristic equation of an ARMAX(P,Q) model.

    Parameters
    ----------
    parameters : numpy.ndarray
        A 1-D array of length ``constant + len(p) + len(q) + x_cols`` containing
        the model parameters, typically an output from ``armaxfilter``.
    constant : int
        Scalar variable: 1 to include a constant term, 0 to exclude.
    p : array_like
        Non-negative integer array representing the AR lag orders to include
        in the model.  For example, ``[1, 2]`` for ARMA(2,q) or ``[1, 3]``
        for irregular AR orders.  An empty array or ``[0]`` indicates no AR
        component.
    q : array_like
        Non-negative integer array representing the MA lag orders to include
        in the model.  An empty array or ``[0]`` indicates no MA component.
    x : numpy.ndarray or None, optional
        A ``(T, K)`` matrix of exogenous variables used solely for
        parameter-length validation.  Default is ``None`` (no exogenous
        variables, equivalent to 0 columns).

    Returns
    -------
    arroots : numpy.ndarray
        A 1-D array of length ``max(p)`` containing the roots of the AR
        characteristic polynomial.  Empty array if there are no AR terms.
    absarroots : numpy.ndarray
        Absolute values (complex moduli) of the AR roots.  Same length as
        ``arroots``.  Empty array if there are no AR terms.

    Raises
    ------
    ValueError
        If inputs are invalid: ``p`` or ``q`` contain non-integers, negative
        values, or duplicate entries; ``constant`` is not 0 or 1; or
        ``parameters`` has an incompatible length.

    Notes
    -----
    This function mirrors the MATLAB ``armaroots.m`` implementation exactly,
    including the correction of a bug on MATLAB source line 67 where
    ``unique(q)`` was incorrectly used instead of ``unique(p)`` for the
    duplicate-lag check on the ``p`` vector.

    The AR characteristic polynomial is constructed as:

    .. math::

        1 - \\phi_1 z - \\phi_2 z^2 - \\ldots - \\phi_{\\max(p)} z^{\\max(p)}

    where coefficients at lag positions not in ``p`` are zero (scattered
    coefficient pattern for irregular AR orders).
    """
    # ------------------------------------------------------------------
    # Handle default for X
    # Ref: armaroots.m:46-48 — MATLAB nargin==4 sets X=[]
    # ------------------------------------------------------------------
    if x is None:
        x = np.zeros((0, 0))
    else:
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        if x.ndim == 1:
            x = x.reshape(-1, 1)

    # ------------------------------------------------------------------
    # Validate and normalize P
    # Ref: armaroots.m:52-70
    # ------------------------------------------------------------------
    p = np.atleast_1d(np.asarray(p, dtype=np.float64)).flatten()

    # Ref: armaroots.m:55-57 — treat empty or scalar zero as "no AR"
    if p.size == 0:
        p = np.array([0.0])
    if p.size == 1 and p[0] == 0:
        p = np.array([], dtype=np.float64)

    if p.size > 0:
        # Ref: armaroots.m:61-63 — must be non-negative integers
        if np.any(p < 0) or np.any(np.floor(p) != p):
            raise ValueError('P must contain non-negative integers only')

        # Ref: armaroots.m:67 — BUG FIX: MATLAB source checks unique(q)
        # instead of unique(p).  Python implementation corrects this to
        # check unique(p) as intended by the error message.
        if len(np.unique(p)) != len(p):
            raise ValueError('P must contain at most one of each lag')

    lp = len(p)

    # ------------------------------------------------------------------
    # Validate and normalize Q
    # Ref: armaroots.m:74-92
    # ------------------------------------------------------------------
    q = np.atleast_1d(np.asarray(q, dtype=np.float64)).flatten()

    # Ref: armaroots.m:77-79 — treat empty or scalar zero as "no MA"
    if q.size == 0:
        q = np.array([0.0])
    if q.size == 1 and q[0] == 0:
        q = np.array([], dtype=np.float64)

    if q.size > 0:
        # Ref: armaroots.m:83-85 — must be non-negative integers
        if np.any(q < 0) or np.any(np.floor(q) != q):
            raise ValueError('Q must contain non-negative integers only')

        # Ref: armaroots.m:89-91 — check for duplicate MA lags
        if len(np.unique(q)) != len(q):
            raise ValueError('Q must contain at most one of each lag')

    lq = len(q)

    # ------------------------------------------------------------------
    # Validate CONSTANT
    # Ref: armaroots.m:96-98
    # ------------------------------------------------------------------
    if constant not in (0, 1):
        raise ValueError('CONSTANT must be 0 or 1')

    # ------------------------------------------------------------------
    # Validate PARAMETERS
    # Ref: armaroots.m:102-110
    # ------------------------------------------------------------------
    parameters = np.atleast_1d(np.asarray(parameters, dtype=np.float64)).flatten()

    # Ref: armaroots.m:105-106 — must be a 1-D vector
    # (MATLAB checks for column vector; Python checks for 1-D)

    # Determine number of exogenous columns for length check
    x_cols = x.shape[1] if x.ndim == 2 and x.shape[0] > 0 else 0

    # Ref: armaroots.m:108-109 — parameter length must be compatible
    expected_length = lp + lq + x_cols + int(constant)
    if len(parameters) != expected_length:
        raise ValueError(
            'PARAMETERS must have length compatible with P, Q, CONSTANT and X'
        )

    # ------------------------------------------------------------------
    # Core root computation
    # Ref: armaroots.m:116-129
    # ------------------------------------------------------------------
    if lp == 0:
        # Ref: armaroots.m:117-118 — no AR component, return empty arrays
        arroots = np.array([], dtype=np.complex128)
        absarroots = np.array([], dtype=np.float64)
    else:
        # Ref: armaroots.m:120-124 — extract AR parameters with constant offset
        if constant:
            # Ref: armaroots.m:121 — MATLAB parameters(2:lp+1) is 1-based;
            # Python parameters[1:lp+1] is 0-based equivalent
            arparameters = parameters[1:lp + 1]
        else:
            # Ref: armaroots.m:123 — MATLAB parameters(1:lp) is 1-based;
            # Python parameters[0:lp] is 0-based equivalent
            arparameters = parameters[0:lp]

        # Ref: armaroots.m:125-126 — scatter AR coefficients into full-length
        # vector for irregular lag patterns.  MATLAB uses auto-extending
        # assignment: zeros(1,lp) then assigns to indices p (which may exceed
        # lp).  Python must allocate max(p) positions explicitly.
        max_p = int(np.max(p))
        formatted_arparameters = np.zeros(max_p)

        # Ref: armaroots.m:126 — MATLAB 1-based p indices map directly to
        # positions; Python uses 0-based indexing so subtract 1
        p_indices = (p.astype(int)) - 1  # 1-based → 0-based
        formatted_arparameters[p_indices] = arparameters

        # Ref: armaroots.m:127 — roots([1 -formatted_arparameters])
        # Construct characteristic polynomial coefficients:
        # [1, -phi_1, -phi_2, ..., -phi_max(p)]
        poly_coeffs = np.concatenate(([1.0], -formatted_arparameters))
        arroots = np.roots(poly_coeffs)

        # Ref: armaroots.m:128 — abs(arroots) gives complex modulus
        absarroots = np.abs(arroots)

    return arroots, absarroots
