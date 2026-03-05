"""
Provide starting values for APARCH(P,O,Q) parameter estimation.

Performs a grid search to find decent starting values for APARCH estimation
when no user-supplied values are given.  Uses TARCH-based heuristics: fits
a symmetric TARCH(P,0,Q) model with Normal errors, then repackages the
omega/alpha/beta parameters with zero gamma terms and optional delta=1.0
as APARCH starting values.

If starting values are user-supplied (non-empty), reformats them depending
on the error distribution type to separate model parameters from distribution
shape parameters (nu, lambda).

Migrated from: univariate/aparch_starting_values.m — MFE Toolbox Version 4.0
Original author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
Revision: 3    Date: 9/1/2005

Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
"""

import numpy as np

from mfe_toolbox.univariate.tarch import tarch

__all__ = ['aparch_starting_values']


def aparch_starting_values(
    startingvals: np.ndarray | None,
    epsilon: np.ndarray,
    p: int,
    o: int,
    q: int,
    error_type: int,
    delta_is_estimated: bool = True,
) -> tuple[np.ndarray, float | None, float | None]:
    """Compute starting values for APARCH(P,O,Q) estimation.

    When *startingvals* is ``None`` (or an empty array), a TARCH(P,0,Q)
    model is fitted with Normal errors via :func:`tarch` and the resulting
    omega, alpha, beta estimates are repackaged with zero gamma coefficients
    and an initial delta of 1.0 to seed the APARCH optimiser.

    When *startingvals* is a non-empty array, the distribution shape
    parameters ``nu`` and ``lambda`` are extracted from the tail of the
    vector and the model-parameter portion is returned.

    Parameters
    ----------
    startingvals : np.ndarray or None
        A vector of user-supplied starting values, or ``None`` to perform
        a grid search via TARCH.  When provided, the layout is::

            [omega, alpha(1)…alpha(p), gamma(1)…gamma(o),
             beta(1)…beta(q), [delta,] [nu, [lambda]]]

    epsilon : np.ndarray
        T-element vector of mean-zero residuals (the data series).
    p : int
        Lag order for ARCH (symmetric innovation) terms.  Must be ≥ 1.
    o : int
        Lag order for asymmetric terms.
    q : int
        Lag order for GARCH (lagged conditional variance) terms.
    error_type : int
        Error distribution identifier:

        * 1 — Normal
        * 2 — Student's t
        * 3 — GED (Generalised Error Distribution)
        * 4 — Skewed t
    delta_is_estimated : bool, optional
        Whether the power parameter δ is included in the parameter vector
        (default ``True``).  When ``True``, a delta starting value of 1.0
        is appended during grid search, and the parsing of user-supplied
        values accounts for the extra element.

    Returns
    -------
    startingvals : np.ndarray
        Model-parameter starting values of shape
        ``(1 + p + o + q [+ 1 if delta_is_estimated],)``.
    nu : float or None
        Distribution kurtosis parameter (degrees-of-freedom for Student's t
        and Skewed t, shape for GED).  ``None`` if not applicable
        (Normal errors).
    lambda_param : float or None
        Distribution asymmetry parameter (Skewed t only).  ``None`` if not
        applicable.

    Notes
    -----
    * The grid search path calls ``tarch(epsilon, p, 0, q)`` with Normal
      errors (``error_type='NORMAL'``), which internally runs
      ``scipy.optimize.minimize(method='L-BFGS-B')``.
    * MATLAB variable ``lambda`` is renamed to ``lambda_param`` to avoid
      the Python keyword conflict.

    See Also
    --------
    mfe_toolbox.univariate.tarch.tarch : TARCH/GJR-GARCH estimation driver.
    mfe_toolbox.univariate.aparch.aparch : APARCH estimation driver.

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.univariate.aparch_starting_values import aparch_starting_values
    >>> rng = np.random.default_rng(0)
    >>> eps = rng.standard_normal(500)
    >>> sv, nu, lam = aparch_starting_values(None, eps, 1, 1, 1, 1)
    >>> sv.shape  # omega + alpha + gamma + beta + delta
    (5,)
    """
    # ------------------------------------------------------------------
    # Determine whether we need a grid search or user supplied values
    # Ref: aparch_starting_values.m:39 — isempty(startingvals)
    # ------------------------------------------------------------------
    _is_empty = (
        startingvals is None
        or (isinstance(startingvals, np.ndarray) and startingvals.size == 0)
    )

    if _is_empty:
        # ==============================================================
        #  Grid-search path: fit TARCH(p,0,q) as a heuristic seed
        #  Ref: aparch_starting_values.m:39-68
        # ==============================================================

        # Ref: aparch_starting_values.m:40-46 — optimset for fminunc
        # Translated to scipy.optimize.minimize L-BFGS-B options.
        # Ref: aparch_starting_values.m:46 — MaxFunEvals = 200*(2+p+o+q)
        sv_options: dict = {
            'ftol': 1e-3,       # Ref: aparch_starting_values.m:41 — TolFun 1e-3
            'gtol': 1e-4,       # Ref: aparch_starting_values.m:42 — TolX 1e-4
            'disp': False,      # Ref: aparch_starting_values.m:43 — Display off
            'maxfun': 200 * (2 + p + o + q),
        }

        # Ref: aparch_starting_values.m:48
        #   [parameters, LLs] = tarch(data, p, 0, q, [], 1, [], options);
        # MATLAB passes [] for tarch_type (defaults to type 2 = squared/GARCH)
        # and 1 for error_type (Normal).  In Python, 'NORMAL' is the default.
        tarch_result = tarch(
            epsilon, p, 0, q,
            error_type='NORMAL',
            options=sv_options,
        )
        # tarch() returns 7-tuple: (parameters, LL, ht, VCVrobust, VCV, scores, diagnostics)
        parameters: np.ndarray = np.asarray(tarch_result[0], dtype=np.float64).ravel()

        # ----------------------------------------------------------
        # Build APARCH parameter vector from TARCH estimates.
        # TARCH(p,0,q) returns [omega, alpha_1…alpha_p, beta_1…beta_q]
        # We insert zero gamma coefficients and optionally delta=1.
        # ----------------------------------------------------------
        # Ref: aparch_starting_values.m:49-53
        #   parameters(1:p+1)       → omega + alphas   (MATLAB 1-indexed)
        #   zeros(o,1)              → gamma terms       (all zeros)
        #   parameters(p+2:p+q+1)  → betas             (MATLAB 1-indexed)
        #   1                       → delta starting    (only if deltaIsEstimated)
        #
        # Python 0-indexed equivalents:
        #   parameters[0:p+1]       → omega + alphas
        #   np.zeros(o)             → gamma terms
        #   parameters[p+1:p+q+1]  → betas
        parts: list[np.ndarray] = [
            parameters[0:p + 1],       # Ref: aparch_starting_values.m:50 — parameters(1:p+1)
            np.zeros(o),               # Ref: aparch_starting_values.m:50 — zeros(o,1)
            parameters[p + 1:p + q + 1],  # Ref: aparch_starting_values.m:50 — parameters(p+2:p+q+1)
        ]
        if delta_is_estimated:
            # Ref: aparch_starting_values.m:50 — append delta=1 when estimated
            parts.append(np.array([1.0]))

        output_parameters: np.ndarray = np.concatenate(parts)

        # ----------------------------------------------------------
        # Distribution parameter defaults
        # Ref: aparch_starting_values.m:56-68
        # ----------------------------------------------------------
        nu: float | None = None
        lambda_param: float | None = None

        # Ref: aparch_starting_values.m:61 — if errorType==2, nu=8
        if error_type == 2:
            nu = 8.0
        # Ref: aparch_starting_values.m:63 — elseif errorType==3, nu=1.9
        elif error_type == 3:
            nu = 1.9
        # Ref: aparch_starting_values.m:65-67 — elseif errorType==4, nu=8, lambda=-0.1
        elif error_type == 4:
            nu = 8.0
            lambda_param = -0.1

        # Ref: aparch_starting_values.m:59 — startingvals = parameters
        return output_parameters.copy(), nu, lambda_param

    else:
        # ==============================================================
        #  User-supplied starting values — parse distribution parameters
        #  Ref: aparch_starting_values.m:69-82
        # ==============================================================
        startingvals = np.asarray(startingvals, dtype=np.float64).ravel()

        nu = None           # Ref: aparch_starting_values.m:71 — nu = []
        lambda_param = None  # Ref: aparch_starting_values.m:72 — lambda = []

        # Offset for delta when estimated: int(True)=1, int(False)=0
        delta_offset: int = int(delta_is_estimated)

        # ----------------------------------------------------------
        # Parse nu for Student's t, GED, or Skewed t
        # Ref: aparch_starting_values.m:73-75
        #   if errorType==2 || errorType==3 || errorType==4
        #       nu = startingvals(p+o+q+2+deltaIsEstimated);
        # MATLAB 1-indexed → Python 0-indexed: subtract 1 from MATLAB index
        #   MATLAB index: p+o+q+2+deltaIsEstimated
        #   Python index: p+o+q+1+delta_offset
        # ----------------------------------------------------------
        if error_type in (2, 3, 4):
            # Ref: aparch_starting_values.m:74
            nu = float(startingvals[p + o + q + 1 + delta_offset])

        # ----------------------------------------------------------
        # Parse lambda for Skewed t
        # Ref: aparch_starting_values.m:76-78
        #   if errorType==4
        #       lambda = startingvals(p+o+q+3+deltaIsEstimated);
        # Python 0-indexed: p+o+q+2+delta_offset
        # ----------------------------------------------------------
        if error_type == 4:
            # Ref: aparch_starting_values.m:77
            lambda_param = float(startingvals[p + o + q + 2 + delta_offset])

        # ----------------------------------------------------------
        # Truncate to model parameters only (remove nu/lambda tail)
        # Ref: aparch_starting_values.m:79
        #   startingvals = startingvals(1:p+o+q+1+deltaIsEstimated);
        # MATLAB 1-indexed → Python 0-indexed:
        #   startingvals[0 : p+o+q+1+delta_offset]
        # ----------------------------------------------------------
        startingvals = startingvals[0:p + o + q + 1 + delta_offset].copy()

        return startingvals, nu, lambda_param
