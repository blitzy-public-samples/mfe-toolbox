"""
FIGARCH(Q,D,P) input parameter validation.

Validates and normalizes input parameters for FIGARCH (Fractionally Integrated
GARCH) estimation. FIGARCH uniquely restricts p, q to {0, 1}, unlike standard
GARCH models that allow arbitrary lag orders.

Migrated from: univariate/figarch_parameter_check.m
Original author: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 1    Date: 7/12/2009
"""

import numpy as np


def figarch_parameter_check(
    data,
    p=0,
    q=0,
    error_type='NORMAL',
    trunc_lag=1000,
    startingvals=None,
    options=None,
):
    """
    Validate input parameters for FIGARCH estimation.

    FIGARCH(Q,D,P) input validation. Ensures that all input parameters are
    conformable to what is expected by the FIGARCH estimation routines.
    Parameters are validated, defaults applied where necessary, and all values
    are returned in a normalized tuple for downstream consumption.

    Parameters
    ----------
    data : array_like
        T-element array of mean-zero residuals (epsilon). Must be a 1-D
        numeric vector with more than one observation. Corresponds to the
        MATLAB ``epsilon`` parameter.
    p : int, optional
        ARCH lag order. Must be 0 or 1 (FIGARCH restriction — this constraint
        is unique to FIGARCH among the GARCH model family). Default is 0.
    q : int, optional
        GARCH lag order. Must be 0 or 1 (FIGARCH restriction). Default is 0.
    error_type : {str, int}, optional
        Error distribution type. Accepted string values (case-insensitive):
        ``'NORMAL'``, ``'STUDENTST'``, ``'GED'``, ``'SKEWT'``.
        Accepted integer values: 1 (Normal), 2 (Student's t), 3 (GED),
        4 (Skewed t). Default is ``'NORMAL'``.
    trunc_lag : int, optional
        Truncation lag length for the FIGARCH infinite-order weight
        computation. Must be a positive integer >= 10. Default is 1000.
    startingvals : array_like or None, optional
        Starting parameter values for optimization. The required length
        depends on the combination of ``p``, ``q``, and ``error_type``:

        * Normal (error_type=1): length ``2 + p + q``
        * Student's t (error_type=2): length ``3 + p + q``
        * GED (error_type=3): length ``3 + p + q``
        * Skewed t (error_type=4): length ``4 + p + q``

        Parameter layout: ``[omega, (phi if p=1), d, (beta if q=1), ...]``.
        If ``None``, starting values are computed automatically by the
        estimation driver. Default is ``None``.
    options : dict or None, optional
        Optimizer options dictionary for ``scipy.optimize.minimize``.
        If ``None``, sensible defaults are created matching the original
        MATLAB ``optimset('fminunc')`` configuration. Default is ``None``.

    Returns
    -------
    tuple
        ``(p, q, error_type, trunc_lag, startingvals, options)`` — a 6-element
        tuple with all validated and normalized parameter values. ``p`` and
        ``q`` are Python ``int``; ``error_type`` is an ``int`` in {1,2,3,4};
        ``trunc_lag`` is a positive ``int``; ``startingvals`` is a 1-D
        ``numpy.ndarray`` (or ``None``); ``options`` is a ``dict``.

    Raises
    ------
    ValueError
        If any parameter fails its validation check. Error messages are
        preserved from the original MATLAB implementation for consistency.

    See Also
    --------
    figarch : FIGARCH estimation driver.
    figarch_likelihood : FIGARCH log-likelihood computation.
    figarch_starting_values : FIGARCH starting value computation.
    figarch_transform : FIGARCH parameter transformation to unconstrained space.
    figarch_itransform : FIGARCH inverse parameter transformation.
    figarch_weights : FIGARCH truncation weight computation.

    Notes
    -----
    Ref: figarch_parameter_check.m — FIGARCH restricts p and q to {0, 1},
    unlike standard GARCH models which allow arbitrary lag orders. This is
    because the fractional differencing operator ``d`` already captures
    long-memory dynamics, so higher-order ARCH/GARCH terms are unnecessary.

    The parameter ordering within ``startingvals`` is:

    * Index 0: omega (intercept, must be > 0)
    * Index 1 (if p=1): phi (ARCH coefficient, must satisfy 0 < phi < (1-d)/2)
    * Index 1+p: d (fractional integration parameter, must satisfy 0 < d < 1)
    * Index 2+p (if q=1): beta (GARCH coefficient, additional constraints apply)
    * Remaining indices: distribution shape parameters (nu, lambda)
    """
    # =========================================================================
    # Validate data (epsilon)
    # Ref: figarch_parameter_check.m:28-32
    # =========================================================================
    if data is None:
        raise ValueError('epsilon is empty.')

    data = np.asarray(data, dtype=np.float64)

    if data.size == 0:
        raise ValueError('epsilon is empty.')

    # Ref: figarch_parameter_check.m:28 — size(epsilon,2) > 1 rejects row
    # vectors and matrices; length(epsilon)==1 rejects scalars.
    # In Python, a 2-D input with multiple columns is invalid.
    if data.ndim >= 2 and data.shape[1] > 1:
        raise ValueError('epsilon series must be a column vector.')

    # Squeeze to 1-D for consistent downstream handling
    data = np.squeeze(data)

    # Ref: figarch_parameter_check.m:28 — MATLAB length(epsilon)==1
    # A scalar (ndim==0 after squeeze) or single-element vector is rejected.
    if data.ndim == 0 or data.size == 1:
        raise ValueError('epsilon series must be a column vector.')

    # Final dimensionality guard
    if data.ndim != 1:
        raise ValueError('epsilon series must be a column vector.')

    # Additional NaN guard for production robustness
    if np.any(np.isnan(data)):
        raise ValueError('epsilon contains NaN values.')

    # =========================================================================
    # Validate p
    # Ref: figarch_parameter_check.m:37-42
    # =========================================================================
    if p is None:
        p = 0

    if not np.isscalar(p):
        raise ValueError('P must be either 0 or 1.')

    p = int(p)

    if p not in (0, 1):
        raise ValueError('P must be either 0 or 1.')

    # =========================================================================
    # Validate q
    # Ref: figarch_parameter_check.m:47-52
    # =========================================================================
    if q is None:
        q = 0

    if not np.isscalar(q):
        raise ValueError('Q must be either 0 or 1.')

    q = int(q)

    if q not in (0, 1):
        raise ValueError('Q must be either 0 or 1.')

    # =========================================================================
    # Validate error_type
    # Ref: figarch_parameter_check.m:57-75
    # Supports both string labels and integer codes for flexibility.
    # =========================================================================
    if error_type is None:
        error_type = 'NORMAL'

    if isinstance(error_type, str):
        _error_type_map = {
            'NORMAL': 1,
            'STUDENTST': 2,
            'GED': 3,
            'SKEWT': 4,
        }
        error_type_upper = error_type.upper().strip()
        if error_type_upper not in _error_type_map:
            raise ValueError(
                "errorType must be a string and one of: "
                "'NORMAL', 'STUDENTST', 'GED' or 'SKEWT'."
            )
        error_type = _error_type_map[error_type_upper]
    elif isinstance(error_type, (int, float, np.integer, np.floating)):
        error_type = int(error_type)
        if error_type not in (1, 2, 3, 4):
            raise ValueError(
                "errorType must be a string and one of: "
                "'NORMAL', 'STUDENTST', 'GED' or 'SKEWT'."
            )
    else:
        raise ValueError(
            "errorType must be a string and one of: "
            "'NORMAL', 'STUDENTST', 'GED' or 'SKEWT'."
        )

    # =========================================================================
    # Validate trunc_lag
    # Ref: figarch_parameter_check.m:80-85
    # =========================================================================
    if trunc_lag is None:
        trunc_lag = 1000

    if not np.isscalar(trunc_lag):
        raise ValueError('TRUNCLAG must be a positive integer larger than 10.')

    # Ref: figarch_parameter_check.m:83 — floor(truncLag)~=truncLag ensures
    # integer value; Python conversion handles this after float comparison.
    trunc_lag_float = float(trunc_lag)
    if trunc_lag_float != int(trunc_lag_float):
        raise ValueError('TRUNCLAG must be a positive integer larger than 10.')

    trunc_lag = int(trunc_lag_float)

    # Ref: figarch_parameter_check.m:83 — truncLag < 10 (note: 10 is valid)
    if trunc_lag < 10:
        raise ValueError('TRUNCLAG must be a positive integer larger than 10.')

    # =========================================================================
    # Validate startingvals
    # Ref: figarch_parameter_check.m:92-155
    # =========================================================================
    if startingvals is not None:
        startingvals = np.asarray(startingvals, dtype=np.float64)

        if startingvals.size == 0:
            startingvals = None
        else:
            # Ref: figarch_parameter_check.m:93-95 — transpose row to column.
            # In Python, squeeze to 1-D for uniform handling.
            startingvals = np.squeeze(startingvals)
            if startingvals.ndim == 0:
                startingvals = startingvals.reshape(1)
            if startingvals.ndim != 1:
                # Multi-dimensional after squeeze means it was a matrix
                raise ValueError(
                    'startingvals must be a column vector with 2+p+q elements'
                )

            # ---------------------------------------------------------------
            # Extract and validate individual parameters
            # Ref: figarch_parameter_check.m:97-120
            # Parameter layout (0-indexed):
            #   [0]: omega
            #   [1] (if p=1): phi
            #   [1+p]: d
            #   [2+p] (if q=1): beta
            #   Remaining: distribution shape parameters
            # ---------------------------------------------------------------
            _min_base = 2 + p + q
            if len(startingvals) < _min_base:
                raise ValueError(
                    'startingvals must be a column vector with '
                    f'{_min_base} elements'
                )

            # Ref: figarch_parameter_check.m:104 — omega = startingvals(1)
            # MATLAB 1-indexed → Python 0-indexed
            omega = startingvals[0]

            # Ref: figarch_parameter_check.m:97 — d = startingvals(2+p)
            # MATLAB 1-indexed → Python: index 1+p
            d_val = startingvals[1 + p]

            if p:
                # Ref: figarch_parameter_check.m:99 — phi = startingvals(2)
                # MATLAB 1-indexed → Python: index 1
                phi = startingvals[1]

            if q:
                # Ref: figarch_parameter_check.m:102 — beta = startingvals(3+p)
                # MATLAB 1-indexed → Python: index 2+p
                beta = startingvals[2 + p]

            # Ref: figarch_parameter_check.m:105-107
            if omega <= 0:
                raise ValueError(
                    'Omega must be a positive scalar in STARTINGVALS.'
                )

            # Ref: figarch_parameter_check.m:108-109
            if d_val <= 0 or d_val >= 1:
                raise ValueError(
                    'd must be strictly between 0 and 1 in STARTINGVALS.'
                )

            # Ref: figarch_parameter_check.m:111-113
            if p and (phi >= ((1.0 - d_val) / 2.0) or phi <= 0):
                raise ValueError(
                    'phi must satisfy 0<phi<(1-d)/2 with strict inequalities '
                    'in STARTINGVALS.'
                )

            # Ref: figarch_parameter_check.m:114-120
            # Note: MATLAB operator precedence: && > || , matching Python
            # and > or. The logic preserves original MATLAB evaluation order,
            # including the short-circuit on p in the first branch.
            if q:
                if p and (phi + d_val - beta) <= 0 or beta <= 0:
                    raise ValueError(
                        'beta must satisfy 0<beta<phi + d with strict '
                        'inequalities in STARTINGVALS.'
                    )
                elif (d_val - beta) <= 0 or beta <= 0:
                    raise ValueError(
                        'beta must satisfy 0<beta< d with strict '
                        'inequalities in STARTINGVALS.'
                    )

            # ---------------------------------------------------------------
            # Validate startingvals length per error distribution type
            # Ref: figarch_parameter_check.m:122-151
            # ---------------------------------------------------------------
            n_sv = len(startingvals)

            if error_type == 1:
                # Normal distribution
                # Ref: figarch_parameter_check.m:124-126
                if n_sv != (p + q + 2):
                    raise ValueError(
                        'startingvals must be a column vector with '
                        '2+p+q elements'
                    )

            elif error_type == 2:
                # Student's t distribution
                # Ref: figarch_parameter_check.m:127-133
                if n_sv != (p + q + 3):
                    raise ValueError(
                        'startingvals must be a column vector with '
                        '2+p+q+1 elements'
                    )
                # Ref: figarch_parameter_check.m:131 — MATLAB index p+q+3
                # Python 0-indexed: p+q+2
                if startingvals[p + q + 2] < 2.1:
                    raise ValueError(
                        'Nu must be greater than 2.1 when using '
                        'Students-T errors'
                    )

            elif error_type == 3:
                # GED (Generalized Error Distribution)
                # Ref: figarch_parameter_check.m:134-140
                if n_sv != (p + q + 3):
                    raise ValueError(
                        'startingvals must be a column vector with '
                        'p+q+3 elements'
                    )
                # Ref: figarch_parameter_check.m:138 — MATLAB index p+q+3
                # Python 0-indexed: p+q+2
                if startingvals[p + q + 2] < 1.05:
                    raise ValueError(
                        'Nu must be greater than 1 when using GED errors'
                    )

            elif error_type == 4:
                # Skewed t distribution
                # Ref: figarch_parameter_check.m:141-151
                if n_sv != (p + q + 4):
                    raise ValueError(
                        'startingvals must be a column vector with '
                        'p+q+4 elements'
                    )
                # Ref: figarch_parameter_check.m:145 — MATLAB index p+q+3
                # Python 0-indexed: p+q+2
                if startingvals[p + q + 2] < 2.1:
                    raise ValueError(
                        'Nu must be greater than 2.1 when using '
                        'Skew T errors'
                    )
                # Ref: figarch_parameter_check.m:148 — MATLAB index p+q+4
                # Python 0-indexed: p+q+3
                if (
                    startingvals[p + q + 3] < -0.9
                    or startingvals[p + q + 3] > 0.9
                ):
                    raise ValueError(
                        'Lambda must be between -.9 and .9 when using '
                        'Skew T errors'
                    )

    # =========================================================================
    # Validate options
    # Ref: figarch_parameter_check.m:160-175
    # =========================================================================
    if options is not None:
        # Ref: figarch_parameter_check.m:161-165 — MATLAB validates via
        # optimset(options); Python validates that options is a dict.
        if not isinstance(options, dict):
            raise ValueError(
                'OPTIONS is not a valid minimization option structure'
            )
    else:
        # Ref: figarch_parameter_check.m:168-174 — Create default options
        # matching MATLAB optimset('fminunc') with specified overrides.
        # Translated to scipy.optimize.minimize compatible options dict.
        options = {
            'ftol': 1e-5,       # Ref: figarch_parameter_check.m:169 — TolFun
            'gtol': 1e-5,       # Ref: figarch_parameter_check.m:170 — TolX
            'disp': True,       # Ref: figarch_parameter_check.m:171 — Display='iter'
            'maxiter': 400 * (2 + p + q),   # Ref: figarch_parameter_check.m:174
            'maxfun': 400 * (2 + p + q),    # MaxFunEvals for L-BFGS-B compat
        }

    return (p, q, error_type, trunc_lag, startingvals, options)
