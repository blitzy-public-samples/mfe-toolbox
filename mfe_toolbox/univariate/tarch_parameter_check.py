"""
TARCH/GJR-GARCH parameter validation.

Validates and normalizes all input parameters for TARCH(P,O,Q) model estimation.
Ensures that inputs conform to expected types, ranges, and constraints before
passing them to the numerical optimizer.

Migrated from: univariate/tarch_parameter_check.m
Original Author: Kevin Sheppard
Original Revision: 3, Date: 9/1/2005

Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
"""

import numpy as np


def tarch_parameter_check(
    data,
    p: int,
    o: int,
    q: int,
    error_type=None,
    tarch_type=None,
    startingvals=None,
    options=None,
) -> tuple:
    """
    Validate and normalize input parameters for TARCH/GJR-GARCH estimation.

    Performs comprehensive input validation including type checks, range checks,
    stationarity constraints, and distribution parameter bounds.  Returns
    sanitized copies of every parameter ready for downstream optimisation.

    Parameters
    ----------
    data : array_like
        T-by-1 vector of return observations.  Must be a column vector with
        more than one element and no NaN values.
    p : int
        Number of symmetric innovation (ARCH) lags.  Must be a positive
        scalar >= 1.
    o : int
        Number of asymmetric innovation (TARCH) lags.  Must be a non-negative
        scalar >= 0.
    q : int
        Number of lagged conditional variance (GARCH) lags.  Must be a
        non-negative scalar >= 0.
    error_type : {None, 'NORMAL', 'STUDENTST', 'GED', 'SKEWT'}, optional
        Innovation distribution type.  Strings are mapped to integer codes:
        NORMAL -> 1, STUDENTST -> 2, GED -> 3, SKEWT -> 4.  Integer codes
        in {1, 2, 3, 4} are also accepted.  Defaults to 'NORMAL' (1).
    tarch_type : {None, 1, 2}, optional
        TARCH model type.  1 = absolute-value specification, 2 = squared
        specification.  Defaults to 2.
    startingvals : array_like or None, optional
        Starting parameter vector for the optimizer.  Required length depends
        on the error distribution:

        * NORMAL  : ``p + o + q + 1``
        * STUDENTST / GED : ``p + o + q + 2``
        * SKEWT   : ``p + o + q + 3``

        Additional constraints on distribution shape parameters are enforced.
        If ``None`` or empty, the estimator will choose starting values
        automatically.
    options : dict or None, optional
        Optimizer options dictionary compatible with
        ``scipy.optimize.minimize``.  If ``None``, sensible defaults are
        constructed.

    Returns
    -------
    tuple
        ``(p, o, q, error_type, tarch_type, startingvals, options)`` with all
        values validated, converted to canonical types, and ready for
        estimation.

    Raises
    ------
    ValueError
        If any input parameter fails validation.

    Notes
    -----
    Ref: tarch_parameter_check.m — MATLAB-to-Python migration.

    * MATLAB ``error()`` calls are replaced with ``raise ValueError()``.
    * MATLAB ``nargin`` checks are replaced with Python ``None`` defaults.
    * MATLAB ``optimset`` is replaced with a ``dict`` for
      ``scipy.optimize.minimize`` options.
    * MATLAB 1-based indexing is converted to Python 0-based indexing
      throughout.

    See Also
    --------
    mfe_toolbox.univariate.tarch : Main TARCH estimation driver.
    """
    # -------------------------------------------------------------------
    # data validation
    # Ref: tarch_parameter_check.m:26-30
    # MATLAB: size(data,2) > 1 || length(data)==1  →  column-vector check
    # -------------------------------------------------------------------
    if data is None:
        raise ValueError('data is empty.')

    data = np.asarray(data, dtype=np.float64)

    # Ref: tarch_parameter_check.m:26 — reject multi-column or scalar data
    if data.ndim == 0:
        # np.asarray wraps a scalar as 0-d array
        raise ValueError('data series must be a column vector.')
    if data.ndim >= 2 and data.shape[1] > 1:
        raise ValueError('data series must be a column vector.')
    # Ref: tarch_parameter_check.m:26 — length(data)==1 means scalar
    if data.size == 1:
        raise ValueError('data series must be a column vector.')

    # Ref: tarch_parameter_check.m:28-29 — isempty(data)
    if data.size == 0:
        raise ValueError('data is empty.')

    # Additional NaN guard — prevents silent corruption in downstream
    # log-likelihood evaluation.  Uses numpy.isnan + numpy.any as required
    # by the module schema.
    if np.any(np.isnan(data)):
        raise ValueError('data contains NaN values.')

    # -------------------------------------------------------------------
    # q validation
    # Ref: tarch_parameter_check.m:35-37
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
    # Ref: tarch_parameter_check.m:42-44
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
    # Ref: tarch_parameter_check.m:49-51
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
    # error_type validation
    # Ref: tarch_parameter_check.m:56-74
    # MATLAB nargin<5 / isempty → Python None default
    # Maps string labels to integer codes: NORMAL=1 STUDENTST=2 GED=3 SKEWT=4
    # -------------------------------------------------------------------
    if error_type is None:
        error_type = 1  # Ref: tarch_parameter_check.m:57 — default 'NORMAL'

    if isinstance(error_type, str):
        # Ref: tarch_parameter_check.m:60 — isempty check already handled above
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
    # tarch_type validation
    # Ref: tarch_parameter_check.m:79-92
    # MATLAB nargin>5 / isempty → Python None default
    # -------------------------------------------------------------------
    if tarch_type is None:
        tarch_type = 2  # Ref: tarch_parameter_check.m:91 — default is 2
    else:
        # Ref: tarch_parameter_check.m:80 — isempty check
        # In Python, explicit None is already handled; accept 0 as "empty-like"
        if not np.isscalar(tarch_type):
            raise ValueError('tarch_type must be a scalar')
        tarch_type = int(tarch_type)
        # Ref: tarch_parameter_check.m:87 — must be 1 or 2
        if tarch_type not in (1, 2):
            raise ValueError('tarch_type must be either 1 or 2')

    # -------------------------------------------------------------------
    # starting values validation
    # Ref: tarch_parameter_check.m:97-138
    # MATLAB nargin>6 / isempty → Python None default
    # Parameter vector layout:
    #   [omega, alpha_1..alpha_p, gamma_1..gamma_o, beta_1..beta_q, (nu), (lambda)]
    # -------------------------------------------------------------------
    if startingvals is not None:
        startingvals = np.asarray(startingvals, dtype=np.float64).ravel()

        # Ref: tarch_parameter_check.m:98 — if ~isempty(startingvals)
        if startingvals.size == 0:
            # Empty array treated as "not provided" — matches MATLAB []
            startingvals = None
        else:
            # ----- Length validation per error distribution -----
            if error_type == 1:
                # Ref: tarch_parameter_check.m:100-103 — NORMAL
                expected_len = p + o + q + 1
                if len(startingvals) != expected_len:
                    raise ValueError(
                        'startingvals must be a column vector with '
                        'p+o+q+1 elements'
                    )
            elif error_type == 2:
                # Ref: tarch_parameter_check.m:104-110 — STUDENTST
                expected_len = p + o + q + 2
                if len(startingvals) != expected_len:
                    raise ValueError(
                        'startingvals must be a column vector with '
                        'p+o+q+2 elements'
                    )
                # Ref: tarch_parameter_check.m:108 — MATLAB index p+o+q+2
                # Python 0-indexed: p+o+q+1
                if startingvals[p + o + q + 1] < 2.1:
                    raise ValueError(
                        'Nu must be greater than 2.1 when using '
                        'Students-T errors'
                    )
            elif error_type == 3:
                # Ref: tarch_parameter_check.m:111-117 — GED
                expected_len = p + o + q + 2
                if len(startingvals) != expected_len:
                    raise ValueError(
                        'startingvals must be a column vector with '
                        'p+o+q+2 elements'
                    )
                # Ref: tarch_parameter_check.m:115 — Nu > 1.05
                # (MATLAB error message says "> 1" but threshold is 1.05)
                if startingvals[p + o + q + 1] < 1.05:
                    raise ValueError(
                        'Nu must be greater than 1 when using GED errors'
                    )
            elif error_type == 4:
                # Ref: tarch_parameter_check.m:118-127 — SKEWT
                expected_len = p + o + q + 3
                if len(startingvals) != expected_len:
                    raise ValueError(
                        'startingvals must be a column vector with '
                        'p+o+q+3 elements'
                    )
                # Ref: tarch_parameter_check.m:122 — Nu > 2.1
                if startingvals[p + o + q + 1] < 2.1:
                    raise ValueError(
                        'Nu must be greater than 2.1 when using '
                        'Skew T errors'
                    )
                # Ref: tarch_parameter_check.m:125 — Lambda in (-0.9, 0.9)
                lam = startingvals[p + o + q + 2]
                if lam < -0.9 or lam > 0.9:
                    raise ValueError(
                        'Lambda must be between -.9 and .9 when using '
                        'Skew T errors'
                    )

            # ----- Positivity of omega and alpha -----
            # Ref: tarch_parameter_check.m:129
            # MATLAB startingvals(1:p+1) → Python startingvals[0:p+1]
            if np.any(startingvals[0:p + 1] <= 0):
                raise ValueError(
                    'All startingvals for omega and alpha must be '
                    'strictly greater than zero'
                )

            # ----- Stationarity constraint -----
            # Ref: tarch_parameter_check.m:132-134
            # sum(alpha) + 0.5*sum(gamma) + sum(beta) < 1
            # MATLAB startingvals(2:p+1)      → Python [1 : p+1]       (alpha)
            # MATLAB startingvals(p+2:p+o+1)  → Python [p+1 : p+1+o]   (gamma)
            # MATLAB startingvals(p+o+2:p+o+q+1) → Python [p+1+o : p+1+o+q] (beta)
            alpha_sum = np.sum(startingvals[1:p + 1])
            gamma_sum = np.sum(startingvals[p + 1:p + 1 + o])
            beta_sum = np.sum(startingvals[p + 1 + o:p + 1 + o + q])
            if (alpha_sum + 0.5 * gamma_sum + beta_sum) >= 1.0:
                raise ValueError(
                    'The sum of the arch, garch and 0.5*tarch '
                    'coefficients must be less than 1'
                )

    # -------------------------------------------------------------------
    # options validation and defaults
    # Ref: tarch_parameter_check.m:143-158
    # MATLAB optimset → Python dict for scipy.optimize.minimize
    # -------------------------------------------------------------------
    if options is not None:
        # Ref: tarch_parameter_check.m:144-148 — validate structure
        if not isinstance(options, dict):
            raise ValueError(
                'options is not a valid minimization option structure'
            )
    else:
        # Ref: tarch_parameter_check.m:151-157 — construct defaults
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
    # Ref: tarch_parameter_check.m:163-168
    # In the original MATLAB these lines set q=0 / o=0 when isempty;
    # they are unreachable because earlier validation errors on empty q/o.
    # Preserved for faithful translation parity.
    # -------------------------------------------------------------------
    if q is None:  # pragma: no cover — unreachable after validation above
        q = 0
    if o is None:  # pragma: no cover — unreachable after validation above
        o = 0

    return p, o, q, error_type, tarch_type, startingvals, options
