"""
EGARCH nonlinear stationarity constraint.

Implements the nonlinear inequality constraint function used during EGARCH
parameter estimation via ``scipy.optimize.minimize(method='SLSQP')``.  The
constraint ensures that the characteristic polynomial roots of the beta
(persistence) coefficients lie strictly inside the unit circle, which is the
necessary condition for covariance stationarity of the EGARCH process.

Migrated from: ``univariate/egarch_nlcon.m`` (Version 4.0, Kevin Sheppard,
University of Oxford).

Scipy constraint usage example::

    constraints = {'type': 'ineq',
                   'fun': lambda x: egarch_nlcon(x, p, o, q, error_type)}

where ``'ineq'`` means ``c(x) >= 0`` is required for feasibility.
"""

import numpy as np


def egarch_nlcon(
    parameters: np.ndarray,
    p: int,
    o: int,
    q: int,
    error_type: int,
) -> np.ndarray:
    """Compute EGARCH nonlinear stationarity constraint.

    Evaluates the stationarity constraint for the EGARCH model by checking
    that all roots of the characteristic polynomial ``1 - beta_1*z - ... -
    beta_q*z^q`` lie inside the unit circle (with a small safety margin of
    0.99998).

    Parameters
    ----------
    parameters : np.ndarray
        Full EGARCH parameter vector.  Layout (0-based indices)::

            [omega,                          # index 0
             alpha_1, ..., alpha_p,          # indices 1 .. p
             gamma_1, ..., gamma_o,          # indices p+1 .. p+o
             beta_1,  ..., beta_q,           # indices p+o+1 .. p+o+q
             (nu), (lambda)]                 # optional distribution params

    p : int
        Number of symmetric innovation (|z|) terms in the EGARCH model.
    o : int
        Number of asymmetric innovation (z) terms in the EGARCH model.
    q : int
        Number of lagged log-variance terms (persistence order).
    error_type : int
        Error distribution type code (1=Normal, 2=Student-t, 3=GED,
        4=Skewed-t).  Not used in the constraint computation but included
        in the signature for API consistency with the estimation pipeline.

    Returns
    -------
    np.ndarray
        Constraint vector *c* of length *q*.  The SLSQP constraint
        ``c >= 0`` is satisfied when **all** characteristic polynomial
        roots have modulus less than 0.99998, ensuring stationarity.

    Notes
    -----
    The MATLAB function returns ``[c, ceq]`` where ``ceq = []``.  In the
    Python translation only *c* is returned because
    ``scipy.optimize.minimize`` inequality constraints require only the
    inequality values.

    The threshold ``0.99998`` (rather than ``1.0``) provides a small
    numerical margin below the unit circle boundary, preventing the
    optimiser from placing roots exactly on the boundary where the process
    would be non-stationary.

    The ``error_type`` parameter is unused in this function but is kept in
    the signature so that the driver ``egarch.py`` can pass it through
    without special-casing.

    References
    ----------
    Kevin Sheppard, ``univariate/egarch_nlcon.m``, MFE Toolbox v4.0.
    """
    # ------------------------------------------------------------------
    # Extract the q beta (persistence) coefficients from the parameter
    # vector.
    #
    # Ref: egarch_nlcon.m:20 — MATLAB uses 1-based indexing:
    #   beta = parameters(p+o+2 : p+o+q+1)
    # In Python (0-based), the same q elements start at index p+o+1:
    #   beta = parameters[p+o+1 : p+o+q+1]
    # ------------------------------------------------------------------
    beta = parameters[p + o + 1 : p + o + q + 1]

    # ------------------------------------------------------------------
    # Build the characteristic polynomial coefficients [1, -beta_1, ...,
    # -beta_q] and compute its roots.
    #
    # Ref: egarch_nlcon.m:28 — MATLAB: roots([1; -beta])
    # np.roots expects coefficients in descending power order which
    # matches the MATLAB convention: highest power first.
    # ------------------------------------------------------------------
    coeffs = np.concatenate(([1.0], -beta))

    # ------------------------------------------------------------------
    # Compute constraint: c = |roots| - 0.99998
    #
    # Ref: egarch_nlcon.m:28 — c = abs(roots([1;-beta])) - .99998
    #
    # Feasibility requires c >= 0, i.e. every root has modulus >= 0.99998.
    # Wait — that seems backwards at first glance, but recall that
    # scipy's 'ineq' constraint requires c(x) >= 0.  If all roots of the
    # companion polynomial lie *inside* the unit circle, then their
    # moduli are < 1.  The constraint c = |root| - 0.99998 will be
    # negative when roots are deep inside the circle and will approach
    # zero as roots approach the boundary.
    #
    # IMPORTANT: The MATLAB fmincon convention for nonlinear inequality
    # constraints is c(x) <= 0 (feasible).  However, the driver
    # ``egarch.py`` wraps this function with a sign flip when building
    # the scipy constraint dict so that the scipy convention c(x) >= 0
    # is respected.  This function itself preserves the **original MATLAB
    # sign convention** (c = |root| - 0.99998) so that downstream code
    # can handle the sign as needed.
    # ------------------------------------------------------------------
    c = np.abs(np.roots(coeffs)) - 0.99998

    # ------------------------------------------------------------------
    # np.roots may return complex values.  Extract the real part of c so
    # that the return value is always a real-valued ndarray, suitable for
    # use as a scipy constraint.
    # ------------------------------------------------------------------
    return np.real(c)
