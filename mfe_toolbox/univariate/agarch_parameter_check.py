"""
AGARCH/NAGARCH parameter validation module.

Migrated from univariate/agarch_parameter_check.m (5792 bytes, 178 lines).
Validates all input parameters for AGARCH(P,Q) and NAGARCH(P,Q) estimation,
converting string-type arguments to integer codes and applying default values
where appropriate.  MATLAB ``error()`` calls are replaced with Python
``raise ValueError()``.

Author: Kevin Sheppard (original MATLAB, University of Oxford)
Python migration: Blitzy Platform
"""

import numpy as np


def agarch_parameter_check(data, p, q, error_type=None, model_type=None,
                           startingvals=None, options=None):
    """Validate and normalise input parameters for AGARCH/NAGARCH estimation.

    Translates the validation logic from ``agarch_parameter_check.m``,
    replacing every MATLAB ``error()`` with a Python ``raise ValueError()``.

    Parameters
    ----------
    data : array_like
        A 1-D array of return data (column vector in MATLAB).  Must be
        numeric with at least two observations.
    p : int
        Lag order for symmetric innovation (ARCH) terms.  Must be a
        positive integer >= 1.
        Ref: agarch_parameter_check.m:37 — ``any(p<1)``
    q : int
        Lag order for lagged conditional variance (GARCH) terms.  Must be
        a non-negative integer >= 0.
        Ref: agarch_parameter_check.m:33 — ``any(q<0)`` (allows q=0)
    error_type : {None, str, int}, optional
        Error distribution type.  Accepts a string (``'NORMAL'``,
        ``'STUDENTST'``, ``'GED'``, ``'SKEWT'``) or an integer code
        (1, 2, 3, 4).  Defaults to ``'NORMAL'`` (1) when *None*.
    model_type : {None, str, int}, optional
        Model specification.  Accepts ``'AGARCH'`` / ``'NAGARCH'`` or
        integer 1 / 2.  Defaults to ``'AGARCH'`` (1) when *None*.
    startingvals : array_like or None, optional
        Starting parameter values for the optimiser.  When *None* or
        empty the optimiser will generate starting values internally.
        When provided the vector must satisfy length, positivity,
        stationarity and gamma-bound constraints.
    options : dict or None, optional
        Optimiser options dict (replaces MATLAB ``optimset`` struct).
        When *None* a default set of options is returned.

    Returns
    -------
    tuple
        ``(p, q, error_type, model_type, startingvals, options)`` — all
        validated and normalised.  ``error_type`` and ``model_type`` are
        returned as integers; ``startingvals`` is a 1-D ``numpy.ndarray``
        (empty if not supplied).

    Raises
    ------
    ValueError
        If any parameter fails validation.

    Notes
    -----
    - MATLAB 1-based indexing → Python 0-based; every non-obvious
      adjustment is documented with an inline ``# Ref:`` comment.
    - MATLAB ``nargin`` checks are replaced by *None* default arguments.
    - MATLAB ``isempty`` → ``is None`` or ``size == 0``.
    - ``numpy.quantile`` replaces MATLAB ``quantile`` for gamma bounds.

    References
    ----------
    Source: ``univariate/agarch_parameter_check.m`` (MFE Toolbox v4.0)
    """
    # ------------------------------------------------------------------
    # 1.  Validate data (epsilon)
    # Ref: agarch_parameter_check.m:26-31
    #   size(epsilon,2)>1 || length(epsilon)==1  →  rejects non-column
    #   isempty(epsilon)                         →  rejects empty
    # ------------------------------------------------------------------
    data = np.asarray(data, dtype=np.float64)

    if data.ndim > 2:
        raise ValueError('DATA must be a 1-D array (column vector).')

    if data.ndim == 2:
        # Ref: agarch_parameter_check.m:26 — size(epsilon,2)>1
        if data.shape[1] > 1:
            raise ValueError('DATA must be a column vector (1-D array).')
        # Acceptable (T,1) shape — squeeze to (T,)
        data = data.squeeze()

    # After possible squeeze, data is 0-D or 1-D
    if data.ndim == 0:
        # Scalar input
        raise ValueError(
            'DATA must be a column vector with more than 1 observation.'
        )

    if data.size == 0:
        # Ref: agarch_parameter_check.m:29
        raise ValueError('DATA is empty.')

    if data.size == 1:
        # Ref: agarch_parameter_check.m:26 — length(epsilon)==1
        raise ValueError(
            'DATA must be a column vector with more than 1 observation.'
        )

    # ------------------------------------------------------------------
    # 2.  Validate q
    # Ref: agarch_parameter_check.m:33-35
    #   (length(q)>1) || any(q<0) || isempty(q)
    # ------------------------------------------------------------------
    try:
        q = int(q)
    except (TypeError, ValueError) as exc:
        raise ValueError('Q must be a non-negative scalar.') from exc
    if q < 0:
        raise ValueError('Q must be a non-negative scalar.')

    # ------------------------------------------------------------------
    # 3.  Validate p
    # Ref: agarch_parameter_check.m:37-39
    #   (length(p)>1) || any(p<1) || isempty(p)
    # ------------------------------------------------------------------
    try:
        p = int(p)
    except (TypeError, ValueError) as exc:
        raise ValueError('P must be a positive scalar.') from exc
    if p < 1:
        raise ValueError('P must be a positive scalar.')

    # ------------------------------------------------------------------
    # 4.  Validate and map model_type
    # Ref: agarch_parameter_check.m:41-52
    #   Default 'AGARCH' (1); switch upper(model_type)
    # ------------------------------------------------------------------
    if model_type is None or (isinstance(model_type, str)
                              and model_type.strip() == ''):
        # Ref: agarch_parameter_check.m:42 — nargin<4 || isempty
        model_type = 1
    elif isinstance(model_type, str):
        _mt = model_type.upper().strip()
        if _mt == 'AGARCH':
            model_type = 1
        elif _mt == 'NAGARCH':
            model_type = 2
        else:
            raise ValueError(
                "MODEL_TYPE must be 'AGARCH' or 'NAGARCH'."
            )
    elif isinstance(model_type, (int, float, np.integer, np.floating)):
        model_type = int(model_type)
        if model_type not in (1, 2):
            raise ValueError(
                'MODEL_TYPE must be 1 (AGARCH) or 2 (NAGARCH).'
            )
    else:
        raise ValueError(
            "MODEL_TYPE must be 'AGARCH', 'NAGARCH', 1, or 2."
        )

    # ------------------------------------------------------------------
    # 5.  Validate and map error_type
    # Ref: agarch_parameter_check.m:54-70
    #   Default 'NORMAL' (1); switch upper(error_type)
    # ------------------------------------------------------------------
    if error_type is None or (isinstance(error_type, str)
                              and error_type.strip() == ''):
        # Ref: agarch_parameter_check.m:55 — nargin<5 || isempty
        error_type = 1
    elif isinstance(error_type, str):
        _et = error_type.upper().strip()
        if _et == 'NORMAL':
            error_type = 1
        elif _et == 'STUDENTST':
            error_type = 2
        elif _et == 'GED':
            error_type = 3
        elif _et == 'SKEWT':
            error_type = 4
        else:
            raise ValueError(
                "ERROR_TYPE must be 'NORMAL', 'STUDENTST', 'GED', or "
                "'SKEWT'."
            )
    elif isinstance(error_type, (int, float, np.integer, np.floating)):
        error_type = int(error_type)
        if error_type not in (1, 2, 3, 4):
            raise ValueError(
                'ERROR_TYPE must be 1, 2, 3, or 4.'
            )
    else:
        raise ValueError(
            "ERROR_TYPE must be 'NORMAL', 'STUDENTST', 'GED', 'SKEWT', "
            "or an integer 1-4."
        )

    # ------------------------------------------------------------------
    # 6.  Validate startingvals
    # Ref: agarch_parameter_check.m:72-170
    # ------------------------------------------------------------------
    if startingvals is None:
        # Ref: agarch_parameter_check.m:73 — nargin<6 || isempty
        startingvals = np.array([], dtype=np.float64)
    else:
        startingvals = np.asarray(startingvals, dtype=np.float64)
        # Squeeze (T,1) column vectors to (T,)
        if startingvals.ndim == 2 and startingvals.shape[1] == 1:
            startingvals = startingvals.squeeze()
        elif startingvals.ndim >= 2 and startingvals.shape[-1] > 1:
            # Ref: agarch_parameter_check.m:75 — size(startingvals,2)>1
            raise ValueError('STARTINGVALS must be a column vector.')

    if startingvals.size > 0:
        # -- length validation (depends on error_type) --
        if error_type == 1:
            # NORMAL: length == p + q + 2
            # Ref: agarch_parameter_check.m:80
            expected_len = p + q + 2
            if startingvals.size != expected_len:
                raise ValueError(
                    f'STARTINGVALS must have {expected_len} elements for '
                    f'NORMAL errors (p+q+2 = {p}+{q}+2).'
                )
        elif error_type == 2:
            # STUDENTST: length == p + q + 3
            # Ref: agarch_parameter_check.m:85
            expected_len = p + q + 3
            if startingvals.size != expected_len:
                raise ValueError(
                    f'STARTINGVALS must have {expected_len} elements for '
                    f'STUDENTST errors (p+q+3 = {p}+{q}+3).'
                )
            # Ref: agarch_parameter_check.m:89
            # MATLAB: startingvals(p+q+3) → Python: startingvals[p+q+2]
            nu_val = startingvals[p + q + 2]
            if nu_val <= 2.1:
                raise ValueError(
                    "When ERROR_TYPE is Student's T, NU must be > 2.1."
                )
        elif error_type == 3:
            # GED: length == p + q + 3
            # Ref: agarch_parameter_check.m:93
            expected_len = p + q + 3
            if startingvals.size != expected_len:
                raise ValueError(
                    f'STARTINGVALS must have {expected_len} elements for '
                    f'GED errors (p+q+3 = {p}+{q}+3).'
                )
            # Ref: agarch_parameter_check.m:97
            # MATLAB: startingvals(p+q+3) → Python: startingvals[p+q+2]
            nu_val = startingvals[p + q + 2]
            if nu_val <= 1.05:
                raise ValueError(
                    'When ERROR_TYPE is GED, NU must be > 1.05.'
                )
        elif error_type == 4:
            # SKEWT: length == p + q + 4
            # Ref: agarch_parameter_check.m:101
            expected_len = p + q + 4
            if startingvals.size != expected_len:
                raise ValueError(
                    f'STARTINGVALS must have {expected_len} elements for '
                    f'SKEWT errors (p+q+4 = {p}+{q}+4).'
                )
            # Ref: agarch_parameter_check.m:105
            # MATLAB: startingvals(p+q+3) → Python: startingvals[p+q+2]
            nu_val = startingvals[p + q + 2]
            if nu_val <= 2.1:
                raise ValueError(
                    "When ERROR_TYPE is Skewed T, NU must be > 2.1."
                )
            # Ref: agarch_parameter_check.m:109
            # MATLAB: startingvals(p+q+4) → Python: startingvals[p+q+3]
            lam_val = startingvals[p + q + 3]
            if abs(lam_val) >= 0.9:
                raise ValueError(
                    'When ERROR_TYPE is Skewed T, LAMBDA must be between '
                    '-0.9 and 0.9.'
                )

        # -- Positivity: omega and alpha(1:p) must be > 0 --
        # Ref: agarch_parameter_check.m:115-118
        # MATLAB: startingvals(1:p+1) → Python: startingvals[0:p+1]
        omega_alpha = startingvals[0:p + 1]
        if np.any(omega_alpha <= 0):
            raise ValueError(
                'The starting values for OMEGA and ALPHA must be strictly '
                'positive.'
            )

        # -- Positivity: beta(1:q) must be > 0 --
        # Ref: agarch_parameter_check.m:120-123
        # MATLAB: startingvals(p+3:p+q+2) → Python: startingvals[p+2:p+q+2]
        if q > 0:
            betas = startingvals[p + 2:p + q + 2]
            if np.any(betas <= 0):
                raise ValueError(
                    'The starting values for BETA must be strictly positive.'
                )

        # -- Stationarity: sum(alpha) + sum(beta) < 1 --
        # Ref: agarch_parameter_check.m:125-128
        # MATLAB: sum(startingvals(2:p+1)) + sum(startingvals(p+3:p+q+2))
        # Python: sum(startingvals[1:p+1]) + sum(startingvals[p+2:p+q+2])
        alpha_sum = np.sum(startingvals[1:p + 1])
        beta_sum = (np.sum(startingvals[p + 2:p + q + 2])
                    if q > 0 else 0.0)
        if alpha_sum + beta_sum >= 1.0:
            raise ValueError(
                'The sum of ALPHA and BETA starting values must be less '
                'than 1 for stationarity.'
            )

        # -- Gamma bounds --
        # Ref: agarch_parameter_check.m:130-140
        # MATLAB: startingvals(p+2) → Python: startingvals[p+1]
        gamma_val = startingvals[p + 1]
        if model_type == 1:
            # AGARCH: gamma ∈ (quantile(epsilon, 0.01), quantile(epsilon, 0.99))
            # Ref: agarch_parameter_check.m:132-135
            lower_q = float(np.quantile(data, 0.01))
            upper_q = float(np.quantile(data, 0.99))
            if gamma_val >= upper_q or gamma_val <= lower_q:
                raise ValueError(
                    'For AGARCH, GAMMA must be between the 1st and 99th '
                    f'percentiles of the data ({lower_q:.6f}, '
                    f'{upper_q:.6f}).'
                )
        else:
            # NAGARCH: |gamma| <= 4
            # Ref: agarch_parameter_check.m:137-139
            if abs(gamma_val) > 4.0:
                raise ValueError(
                    'For NAGARCH, GAMMA must be between -4 and 4.'
                )

    # ------------------------------------------------------------------
    # 7.  Validate / create default options
    # Ref: agarch_parameter_check.m:143-165
    # ------------------------------------------------------------------
    if options is None:
        # Ref: agarch_parameter_check.m:144-152
        # Build default scipy-compatible options dict replacing MATLAB
        # optimset('fminunc') with Display='iter', TolFun=1e-5, TolX=1e-5,
        # MaxFunEvals=400*(2+p+q), Diagnostics='off', LargeScale='off'.
        options = {
            'maxiter': 400 * (2 + p + q),
            'disp': True,
            'ftol': 1e-5,
            'gtol': 1e-5,
        }
    else:
        # Ref: agarch_parameter_check.m:154-159 — try optimset(options)
        if not isinstance(options, dict):
            raise ValueError(
                'OPTIONS must be a dict of optimiser options '
                '(replaces MATLAB optimset structure).'
            )

    # Return validated values with the AAP-specified ordering
    # (error_type and model_type swapped relative to MATLAB return order)
    return (p, q, error_type, model_type, startingvals, options)
