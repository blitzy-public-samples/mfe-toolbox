"""
APARCH(P,O,Q) parameter validation.

Validates and normalizes all input parameters for APARCH (Asymmetric Power ARCH)
model estimation.  Ensures that inputs conform to expected types, ranges, and
constraints before passing them to the numerical optimizer.

The APARCH model extends standard GARCH by introducing an asymmetric power
parameter (delta) and leverage coefficients (gamma), requiring additional
validation beyond standard GARCH parameter checks.

Migrated from: univariate/aparch_parameter_check.m
Original Author: Kevin Sheppard
Original Revision: 3, Date: 9/1/2005

Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
"""

import numpy as np


def aparch_parameter_check(
    data,
    p: int,
    o: int,
    q: int,
    error_type=None,
    user_delta=None,
    startingvals=None,
    options=None,
) -> tuple:
    """
    Validate and normalize input parameters for APARCH estimation.

    Performs comprehensive input validation including type checks, range checks,
    stationarity constraints, delta power bounds, and distribution parameter
    bounds.  Returns sanitized copies of every parameter ready for downstream
    optimisation.

    Parameters
    ----------
    data : array_like
        T-by-1 vector of return observations.  Must be a column vector with
        more than one element.
    p : int
        Number of symmetric innovation (ARCH) lags.  Must be a positive
        scalar >= 1.
    o : int
        Number of asymmetric innovation lags.  Must be a non-negative
        scalar >= 0.  Must satisfy ``o <= p``.
    q : int
        Number of lagged conditional variance (GARCH) lags.  Must be a
        non-negative scalar >= 0.
    error_type : {None, 'NORMAL', 'STUDENTST', 'GED', 'SKEWT'}, optional
        Innovation distribution type.  Strings are mapped to integer codes:
        NORMAL -> 1, STUDENTST -> 2, GED -> 3, SKEWT -> 4.  Integer codes
        in {1, 2, 3, 4} are also accepted.  Defaults to ``'NORMAL'`` (1).
    user_delta : float or None, optional
        User-specified power parameter delta for the APARCH model.  If
        provided, must be a real scalar between 0.3 and 4.0.  When provided,
        delta is fixed during estimation rather than being optimized.
        If ``None`` (default), delta is treated as a free parameter and
        included in the optimization vector.
    startingvals : array_like or None, optional
        Starting parameter vector for the optimizer.  Required length depends
        on the error distribution and whether ``user_delta`` is provided:

        Let ``nud = 1`` if user_delta is None (delta estimated), 0 otherwise.

        * NORMAL      : ``p + o + q + 1 + nud``
        * STUDENTST   : ``p + o + q + 2 + nud``
        * GED         : ``p + o + q + 2 + nud``
        * SKEWT       : ``p + o + q + 3 + nud``

        Parameter layout (0-indexed):

        * ``[0]``                     : omega (> 0)
        * ``[1 : p+1]``              : alpha_1 … alpha_p (> 0)
        * ``[p+1 : p+o+1]``          : gamma_1 … gamma_o (in [-1, 1])
        * ``[p+o+1 : p+o+q+1]``      : beta_1 … beta_q
        * ``[p+o+q+1]`` (if nud=1)   : delta (in [0.3, 4])
        * next                        : nu  (> 2.1 for STUDENTST/SKEWT, > 1.05 for GED)
        * next (SKEWT only)           : lambda (in (-0.9, 0.9))

        Stationarity requires ``sum(alpha) + sum(beta) < 1``.

        If ``None`` or empty, the estimator will choose starting values
        automatically.
    options : dict or None, optional
        Optimizer options dictionary compatible with
        ``scipy.optimize.minimize``.  If ``None``, sensible defaults are
        constructed.

    Returns
    -------
    tuple
        ``(p, o, q, error_type, user_delta, no_user_delta, startingvals, options)``
        with all values validated, converted to canonical types, and ready for
        estimation.

        * ``no_user_delta`` is a ``bool``: ``True`` when delta is a free
          parameter (user_delta not provided), ``False`` when delta is
          fixed by the user.

    Raises
    ------
    ValueError
        If any input parameter fails validation.

    Notes
    -----
    Ref: aparch_parameter_check.m — MATLAB-to-Python migration.

    * MATLAB ``error()`` calls are replaced with ``raise ValueError()``.
    * MATLAB ``nargin`` checks are replaced with Python ``None`` defaults.
    * MATLAB ``optimset`` is replaced with a ``dict`` for
      ``scipy.optimize.minimize`` options.
    * MATLAB 1-based indexing is converted to Python 0-based indexing
      throughout.

    See Also
    --------
    mfe_toolbox.univariate.aparch : Main APARCH estimation driver.
    """
    # -------------------------------------------------------------------
    # data validation
    # Ref: aparch_parameter_check.m:26-30
    # MATLAB: size(data,2) > 1 || length(data)==1  →  column-vector check
    # -------------------------------------------------------------------
    if data is None:
        raise ValueError('data is empty.')

    data = np.asarray(data, dtype=np.float64)

    # Ref: aparch_parameter_check.m:26 — reject scalar (0-d array)
    if data.ndim == 0:
        raise ValueError('data series must be a column vector.')

    # Ref: aparch_parameter_check.m:26 — reject multi-column matrix
    if data.ndim >= 2 and data.shape[1] > 1:
        raise ValueError('data series must be a column vector.')

    # Ref: aparch_parameter_check.m:26 — length(data)==1 means single element
    if data.size == 1:
        raise ValueError('data series must be a column vector.')

    # Ref: aparch_parameter_check.m:28-29 — isempty(data)
    if data.size == 0:
        raise ValueError('data is empty.')

    # -------------------------------------------------------------------
    # q validation
    # Ref: aparch_parameter_check.m:35-37
    # MATLAB: (length(q)>1) || any(q<0) || isempty(q)
    # -------------------------------------------------------------------
    if q is None:
        raise ValueError('q must be a non-negative scalar.')
    if not np.isscalar(q):
        raise ValueError('q must be a non-negative scalar.')
    q = int(q)
    if q < 0:
        raise ValueError('q must be a non-negative scalar.')

    # -------------------------------------------------------------------
    # o validation
    # Ref: aparch_parameter_check.m:42-44
    # MATLAB: (length(o)>1) || any(o<0) || isempty(o)
    # -------------------------------------------------------------------
    if o is None:
        raise ValueError('o must be a non-negative scalar.')
    if not np.isscalar(o):
        raise ValueError('o must be a non-negative scalar.')
    o = int(o)
    if o < 0:
        raise ValueError('o must be a non-negative scalar.')

    # -------------------------------------------------------------------
    # p validation
    # Ref: aparch_parameter_check.m:49-51
    # MATLAB: (length(p)>1) || any(p<1) || isempty(p)
    # -------------------------------------------------------------------
    if p is None:
        raise ValueError('p must be positive scalar.')
    if not np.isscalar(p):
        raise ValueError('p must be positive scalar.')
    p = int(p)
    if p < 1:
        raise ValueError('p must be positive scalar.')

    # -------------------------------------------------------------------
    # p and o restriction
    # Ref: aparch_parameter_check.m:56-58
    # APARCH requires o <= p (asymmetric order cannot exceed ARCH order)
    # -------------------------------------------------------------------
    if o > p:
        raise ValueError('O must be less than or equal to P')

    # -------------------------------------------------------------------
    # error_type validation
    # Ref: aparch_parameter_check.m:63-81
    # MATLAB nargin<5 / isempty → Python None default
    # Maps string labels to integer codes: NORMAL=1 STUDENTST=2 GED=3 SKEWT=4
    # -------------------------------------------------------------------
    if error_type is None:
        error_type = 1  # Ref: aparch_parameter_check.m:64 — default 'NORMAL'

    if isinstance(error_type, str):
        _error_map = {
            'NORMAL': 1,
            'STUDENTST': 2,
            'GED': 3,
            'SKEWT': 4,
        }
        if error_type not in _error_map:
            raise ValueError(
                "error_type must be a string and one of: "
                "'NORMAL', 'STUDENTST', 'GED' or 'SKEWT'."
            )
        error_type = _error_map[error_type]
    elif isinstance(error_type, (int, float, np.integer, np.floating)):
        error_type = int(error_type)
        if error_type not in (1, 2, 3, 4):
            raise ValueError(
                "error_type must be a string and one of: "
                "'NORMAL', 'STUDENTST', 'GED' or 'SKEWT'."
            )
    else:
        raise ValueError(
            "error_type must be a string and one of: "
            "'NORMAL', 'STUDENTST', 'GED' or 'SKEWT'."
        )

    # -------------------------------------------------------------------
    # user_delta validation
    # Ref: aparch_parameter_check.m:86-94
    # MATLAB: nargin>5 && ~isempty(userDelta) → Python user_delta not None
    # When user_delta is provided and valid, delta is fixed during estimation
    # (no_user_delta=False).  Otherwise delta is a free parameter
    # (no_user_delta=True).
    # -------------------------------------------------------------------
    if user_delta is not None:
        # Handle edge case where caller passes an empty array-like
        # (equivalent to MATLAB isempty(userDelta) == true)
        _ud_arr = np.asarray(user_delta)
        if _ud_arr.size == 0:
            user_delta = None
            no_user_delta = True
        else:
            # Ref: aparch_parameter_check.m:87 — scalar, real, bounded [0.3, 4]
            if not np.isscalar(user_delta):
                raise ValueError(
                    'USERDELTA must be a scalar between 0.3 and 4'
                )
            if not np.isreal(user_delta):
                raise ValueError(
                    'USERDELTA must be a scalar between 0.3 and 4'
                )
            user_delta = float(user_delta)
            if user_delta > 4.0 or user_delta < 0.3:
                raise ValueError(
                    'USERDELTA must be a scalar between 0.3 and 4'
                )
            no_user_delta = False
    else:
        # Ref: aparch_parameter_check.m:91-93
        user_delta = None
        no_user_delta = True

    # Integer flag matching MATLAB's boolean-as-int arithmetic in index
    # expressions.  In MATLAB, noUserDelta is logical (true/false), and
    # arithmetic on it yields 1 or 0 — e.g., startingvals(p+o+q+2+noUserDelta).
    # Ref: aparch_parameter_check.m:104,108,111,115,118,122,125,128
    nud = int(no_user_delta)

    # -------------------------------------------------------------------
    # starting values validation
    # Ref: aparch_parameter_check.m:99-149
    # MATLAB nargin>6 / isempty → Python None default
    # Parameter vector layout (0-indexed):
    #   [omega, alpha_1..alpha_p, gamma_1..gamma_o, beta_1..beta_q,
    #    (delta if no_user_delta), (nu if T/GED/SkewT), (lambda if SkewT)]
    # -------------------------------------------------------------------
    if startingvals is not None:
        startingvals = np.asarray(startingvals, dtype=np.float64).ravel()

        # Ref: aparch_parameter_check.m:100 — if ~isempty(startingvals)
        if startingvals.size == 0:
            # Empty array treated as "not provided" — matches MATLAB []
            startingvals = None
        else:
            # ----- Length validation per error distribution -----
            if error_type == 1:
                # Ref: aparch_parameter_check.m:103-106 — NORMAL
                # MATLAB: length(startingvals)~=(p+o+q+1+noUserDelta)
                expected_len = p + o + q + 1 + nud
                if len(startingvals) != expected_len:
                    raise ValueError(
                        'startingvals must be a column vector with '
                        'p+o+q+1 elements'
                    )
            elif error_type == 2:
                # Ref: aparch_parameter_check.m:107-113 — STUDENTST
                # MATLAB: length(startingvals)~=(p+o+q+2+noUserDelta)
                expected_len = p + o + q + 2 + nud
                if len(startingvals) != expected_len:
                    raise ValueError(
                        'startingvals must be a column vector with '
                        'p+o+q+2 elements'
                    )
                # Ref: aparch_parameter_check.m:111
                # MATLAB 1-based: startingvals(p+o+q+2+noUserDelta)
                # Python 0-based: startingvals[p+o+q+1+nud]
                if startingvals[p + o + q + 1 + nud] < 2.1:
                    raise ValueError(
                        'Nu must be greater than 2.1 when using '
                        'Students-T errors'
                    )
            elif error_type == 3:
                # Ref: aparch_parameter_check.m:114-120 — GED
                expected_len = p + o + q + 2 + nud
                if len(startingvals) != expected_len:
                    raise ValueError(
                        'startingvals must be a column vector with '
                        'p+o+q+2 elements'
                    )
                # Ref: aparch_parameter_check.m:118 — Nu > 1.05
                # (MATLAB error message says "> 1" but threshold is 1.05)
                if startingvals[p + o + q + 1 + nud] < 1.05:
                    raise ValueError(
                        'Nu must be greater than 1 when using GED errors'
                    )
            elif error_type == 4:
                # Ref: aparch_parameter_check.m:121-131 — SKEWT
                expected_len = p + o + q + 3 + nud
                if len(startingvals) != expected_len:
                    raise ValueError(
                        'startingvals must be a column vector with '
                        'p+o+q+3 elements'
                    )
                # Ref: aparch_parameter_check.m:125 — Nu > 2.1
                # MATLAB 1-based: startingvals(p+o+q+2+noUserDelta)
                # Python 0-based: startingvals[p+o+q+1+nud]
                if startingvals[p + o + q + 1 + nud] < 2.1:
                    raise ValueError(
                        'Nu must be greater than 2.1 when using '
                        'Skew T errors'
                    )
                # Ref: aparch_parameter_check.m:128 — Lambda in (-0.9, 0.9)
                # MATLAB 1-based: startingvals(p+o+q+3+noUserDelta)
                # Python 0-based: startingvals[p+o+q+2+nud]
                lam = startingvals[p + o + q + 2 + nud]
                if lam < -0.9 or lam > 0.9:
                    raise ValueError(
                        'Lambda must be between -.9 and .9 when using '
                        'Skew T errors'
                    )

            # ----- Positivity of omega and alpha -----
            # Ref: aparch_parameter_check.m:132
            # MATLAB startingvals(1:p+1) → Python startingvals[0:p+1]
            # Checks that omega and all alpha coefficients are strictly positive
            if np.any(startingvals[0:p + 1] <= 0):
                raise ValueError(
                    'All startingvals for omega and alpha must be '
                    'strictly greater than zero'
                )

            # ----- Gamma bounds check -----
            # Ref: aparch_parameter_check.m:135-137
            # MATLAB startingvals(p+2:p+o+1) → Python startingvals[p+1:p+o+1]
            # Leverage coefficients must lie in [-1, 1]
            gamma_vals = startingvals[p + 1:p + o + 1]
            if np.any(gamma_vals > 1) or np.any(gamma_vals < -1):
                raise ValueError(
                    'All starting values for gamma must be between -1 and 1'
                )

            # ----- Stationarity constraint -----
            # Ref: aparch_parameter_check.m:138-139
            # MATLAB: sum(startingvals(2:p+1)) + sum(startingvals(p+o+2:p+o+q+1)) >= 1
            # Python: sum(startingvals[1:p+1]) + sum(startingvals[p+o+1:p+o+q+1]) >= 1
            # Note: The MATLAB error message mentions "0.5*tarch" which is
            # inherited from the TARCH template, but the APARCH check only
            # sums alpha and beta (no gamma factor in this constraint).
            alpha_sum = np.sum(startingvals[1:p + 1])
            beta_sum = np.sum(startingvals[p + o + 1:p + o + q + 1])
            if (alpha_sum + beta_sum) >= 1.0:
                raise ValueError(
                    'The sum of the arch, garch and 0.5*tarch '
                    'coefficients must be less than 1'
                )

            # ----- Delta bounds when estimated (no_user_delta=True) -----
            # Ref: aparch_parameter_check.m:141-145
            # MATLAB startingvals(p+o+q+2) → Python startingvals[p+o+q+1]
            # Delta is at the position right after beta when it is a free
            # parameter.
            if no_user_delta:
                delta_val = startingvals[p + o + q + 1]
                if delta_val > 4.0 or delta_val <= 0.3:
                    raise ValueError('delta must be between .3 and 4')

    # -------------------------------------------------------------------
    # options validation and defaults
    # Ref: aparch_parameter_check.m:154-168
    # MATLAB optimset → Python dict for scipy.optimize.minimize
    # -------------------------------------------------------------------
    if options is not None:
        # Ref: aparch_parameter_check.m:155-158 — validate structure
        if not isinstance(options, dict):
            raise ValueError(
                'options is not a valid minimization option structure'
            )
    else:
        # Ref: aparch_parameter_check.m:161-168 — construct defaults
        # MATLAB optimset('fminunc') equivalents for scipy.optimize.minimize:
        #   TolFun  1e-5  → ftol 1e-5
        #   TolX    1e-5  → gtol 1e-5 (gradient tolerance serves similar role)
        #   Display 'iter' → disp True
        #   MaxFunEvals 200*(2+p+q) → maxiter 200*(2+p+q)
        #   Diagnostics 'on' and LargeScale 'off' have no direct scipy equiv.
        options = {
            'ftol': 1e-5,
            'gtol': 1e-5,
            'disp': True,
            'maxiter': 200 * (2 + p + q),
        }

    # -------------------------------------------------------------------
    # Defensive final empty guards
    # Ref: aparch_parameter_check.m:173-178
    # In the original MATLAB these lines set q=0 / o=0 when isempty;
    # they are unreachable because earlier validation errors on empty q/o.
    # Preserved for faithful translation parity.
    # -------------------------------------------------------------------
    if q is None:  # pragma: no cover — unreachable after validation above
        q = 0
    if o is None:  # pragma: no cover — unreachable after validation above
        o = 0

    return p, o, q, error_type, user_delta, no_user_delta, startingvals, options
