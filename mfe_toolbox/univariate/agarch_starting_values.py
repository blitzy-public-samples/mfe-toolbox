"""
Starting value computation for AGARCH/NAGARCH(P,Q) parameter estimation.

Provides starting values for AGARCH (Asymmetric GARCH) and NAGARCH
(Nonlinear Asymmetric GARCH) model estimation.  When no user-supplied
starting values are provided, a standard TARCH(P,0,Q) model is fit to
obtain initial omega, alpha, and beta parameters, with gamma initialized
to zero.

Migrated from: univariate/agarch_starting_values.m
Original Author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
Revision: 1    Date: 7/12/2009

Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
"""

import numpy as np

from mfe_toolbox.univariate.tarch import tarch

__all__ = ['agarch_starting_values']


def agarch_starting_values(
    startingvals: np.ndarray | None,
    epsilon: np.ndarray,
    p: int,
    q: int,
    model_type: int,
    error_type: int,
) -> tuple[np.ndarray, float | None, float | None]:
    """
    Compute or parse starting values for AGARCH/NAGARCH estimation.

    If ``startingvals`` is ``None`` or empty, estimates a TARCH(p, 0, q) model
    with Normal errors to obtain initial parameters and inserts gamma = 0.
    If ``startingvals`` is provided, extracts distribution parameters (nu
    and/or lambda) and truncates the vector to the GARCH parameter portion.

    Parameters
    ----------
    startingvals : np.ndarray or None
        User-supplied starting values, or ``None`` / empty array to trigger
        automatic initialization via TARCH estimation.  When provided, the
        layout is:

        ``[omega, alpha(1)…alpha(p), gamma, beta(1)…beta(q),
        [nu, [lambda]]]``

        where the trailing distribution parameters depend on ``error_type``.
    epsilon : np.ndarray
        T-by-1 column vector of mean-zero residuals.
    p : int
        Positive scalar integer — number of symmetric innovation lags
        (ARCH terms).
    q : int
        Non-negative scalar integer — number of lagged conditional variance
        terms (GARCH terms).
    model_type : int
        Variance process type:

        * 1 — AGARCH  (Asymmetric GARCH, Engle 1990)
        * 2 — NAGARCH (Nonlinear Asymmetric GARCH, Engle & Ng 1993)
    error_type : int
        Error distribution code:

        * 1 — Normal
        * 2 — Student's t
        * 3 — GED (Generalized Error Distribution)
        * 4 — Skewed t

    Returns
    -------
    startingvals : np.ndarray
        Starting parameter vector of length ``p + q + 2``
        (omega, alpha(1)…alpha(p), gamma, beta(1)…beta(q)).
    nu : float or None
        Distribution shape parameter.  ``None`` if ``error_type == 1``.
    lambda_ : float or None
        Distribution asymmetry parameter.  ``None`` unless
        ``error_type == 4`` (Skewed t).

    Notes
    -----
    * MATLAB ``isempty(startingvals)`` maps to
      ``startingvals is None or len(startingvals) == 0``.
    * MATLAB ``[]`` returns for ``nu`` / ``lambda`` map to Python ``None``.
    * ``lambda_`` is used to avoid conflict with the Python keyword ``lambda``.
    * For NAGARCH (model_type == 2), if the sum of alpha and beta from
      the initial TARCH fit exceeds 0.97, a scaling factor is applied
      to reduce persistence.  The scaling formula is reproduced exactly
      from the MATLAB source — see Ref: agarch_starting_values.m:59-63.

    See Also
    --------
    mfe_toolbox.univariate.agarch : AGARCH/NAGARCH estimation driver.
    mfe_toolbox.univariate.tarch  : TARCH estimation used for initialization.

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.univariate.agarch_starting_values import agarch_starting_values
    >>> rng = np.random.default_rng(42)
    >>> eps = rng.standard_normal(500)
    >>> sv, nu, lam = agarch_starting_values(None, eps, 1, 1, 1, 1)
    >>> sv.shape[0]  # omega + alpha + gamma + beta = 4
    4
    """
    # ------------------------------------------------------------------
    # Case 1: No user-supplied starting values — estimate from data
    # Ref: agarch_starting_values.m:40 — if isempty(startingvals)
    # ------------------------------------------------------------------
    if startingvals is None or (
        isinstance(startingvals, np.ndarray) and startingvals.size == 0
    ):
        nu: float | None = None
        lambda_: float | None = None

        # Ref: agarch_starting_values.m:44-50 — optimset('fminunc') defaults
        # MATLAB TolFun=1e-5  → ftol=1e-5
        # MATLAB TolX=1e-5    → gtol=1e-5
        # MATLAB Display='off' → disp=False
        # MATLAB MaxFunEvals=1200 → maxfun=1200
        options: dict = {
            'ftol': 1e-5,
            'gtol': 1e-5,
            'disp': False,
            'maxfun': 1200,
        }

        # Ref: agarch_starting_values.m:52
        # MATLAB: startingvals = tarch(epsilon, p, 0, q, 'NORMAL', [], [], options)
        # NOTE: MATLAB tarch.m signature has error_type as 5th arg, tarch_type
        # as 6th, but Python tarch.py swaps the order (tarch_type=5th,
        # error_type=6th).  Using keyword arguments to avoid positional
        # confusion.  MATLAB passes [] for tarch_type (defaults to 2='GARCH')
        # and 'NORMAL' for error_type.
        # The tarch function returns a 7-tuple; we only need the first element
        # (parameter estimates).  Ref: agarch_starting_values.m:52
        tarch_result = tarch(
            epsilon, p, 0, q,
            error_type='NORMAL',
            options=options,
        )
        sv = np.asarray(tarch_result[0], dtype=np.float64).ravel()

        # Ref: agarch_starting_values.m:54 — omega = startingvals(1)
        # MATLAB 1-based index 1 → Python 0-based index 0
        omega = sv[0]

        # Ref: agarch_starting_values.m:55 — alpha = startingvals(2:p+1)
        # MATLAB 1-based (2:p+1) → Python 0-based [1:p+1]
        alpha = sv[1:p + 1].copy()

        # Ref: agarch_starting_values.m:56 — beta = startingvals(p+2:p+q+1)
        # MATLAB 1-based (p+2:p+q+1) → Python 0-based [p+1:p+q+1]
        beta = sv[p + 1:p + q + 1].copy()

        # Ref: agarch_starting_values.m:58-64 — NAGARCH persistence scaling
        # For NAGARCH (model_type == 2), if the sum of alpha and beta
        # coefficients exceeds 0.97, apply a multiplicative scaling factor.
        # The formula `scale = 0.97 * (sum(alpha) + sum(beta))` is
        # reproduced EXACTLY from the MATLAB source — no behavior
        # improvements applied.  Ref: agarch_starting_values.m:59-63
        if model_type == 2:
            persistence = float(np.sum(alpha) + np.sum(beta))
            if persistence > 0.97:
                # Ref: agarch_starting_values.m:60 — scale = .97*(sum(alpha)+sum(beta))
                scale = 0.97 * persistence
                # Ref: agarch_starting_values.m:61-62
                alpha = alpha * scale
                beta = beta * scale

        # Ref: agarch_starting_values.m:66 — startingvals = [omega;alpha;0;beta]
        # Insert gamma = 0 between alpha and beta
        startingvals = np.concatenate([
            np.array([omega]),
            alpha,
            np.array([0.0]),
            beta,
        ])

        # Ref: agarch_starting_values.m:68-75 — distribution starting values
        # Use generic defaults for nu and lambda_ when needed
        if error_type == 2:
            # Ref: agarch_starting_values.m:69 — STUDENTST: nu=8
            nu = 8.0
        elif error_type == 3:
            # Ref: agarch_starting_values.m:71 — GED: nu=1.9
            nu = 1.9
        elif error_type == 4:
            # Ref: agarch_starting_values.m:73-74 — SKEWT: nu=8, lambda=-0.1
            nu = 8.0
            lambda_ = -0.1

        return startingvals, nu, lambda_

    # ------------------------------------------------------------------
    # Case 2: User-supplied starting values — parse distribution params
    # Ref: agarch_starting_values.m:76-87
    # ------------------------------------------------------------------
    startingvals = np.asarray(startingvals, dtype=np.float64).ravel()

    nu_out: float | None = None
    lambda_out: float | None = None

    # Ref: agarch_starting_values.m:80-82
    # MATLAB: nu = startingvals(p+q+3)  → Python 0-based: startingvals[p+q+2]
    if error_type in (2, 3, 4):
        nu_out = float(startingvals[p + q + 2])

    # Ref: agarch_starting_values.m:83-85
    # MATLAB: lambda = startingvals(p+q+4)  → Python 0-based: startingvals[p+q+3]
    if error_type == 4:
        lambda_out = float(startingvals[p + q + 3])

    # Ref: agarch_starting_values.m:86
    # MATLAB: startingvals = startingvals(1:p+q+2)
    # Python 0-based: startingvals[:p+q+2]
    # This gives p+q+2 elements: omega, alpha(1..p), gamma, beta(1..q)
    startingvals = startingvals[:p + q + 2]

    return startingvals, nu_out, lambda_out
