"""
IGARCH(p,q) parameter validation module.

Migrated from univariate/igarch_parameter_check.m (MFE Toolbox v4.0)

Validates all input parameters for IGARCH estimation.  IGARCH models impose
a unit-root constraint (sum of all ARCH and GARCH coefficients equals 1),
which reduces the number of free beta parameters from q to q-1.

Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 1    Date: 7/12/2009
"""

import numpy as np


def igarch_parameter_check(data, p, q, error_type=None, igarch_type=None,
                           constant=None, startingvals=None, options=None):
    """
    Validate input parameters for IGARCH(p,q) estimation.

    Ensures that all input parameters are conformable to what is expected
    by the IGARCH estimation routine.  All parameters are validated and
    defaults are applied for any omitted optional arguments.

    Parameters
    ----------
    data : array_like
        T-by-1 array of mean-zero residuals (epsilon).  Must be a column
        vector with more than one observation.
    p : int
        Positive integer representing the number of symmetric innovation
        (ARCH) lags.  Must be >= 1.
    q : int
        Positive integer representing the number of lagged variance
        (GARCH) terms.  Must be >= 1.
    error_type : {None, 'NORMAL', 'STUDENTST', 'GED', 'SKEWT'}, optional
        Error distribution type.  If None or empty, defaults to 'NORMAL'.
        Mapped to integer codes: NORMAL=1, STUDENTST=2, GED=3, SKEWT=4.
    igarch_type : {None, 0, 1}, optional
        Model type specification:
        - 0 : GARCH-type (variance targeting)
        - 1 : AVGARCH-type (standard deviation targeting)
        Defaults to 1 if None.
        Ref: igarch_parameter_check.m:74 — MATLAB default igarchType=2 maps
        to Python igarch_type=1.
    constant : {None, 0, 1}, optional
        Whether to include the intercept (omega) in the model.
        - 0 : No intercept
        - 1 : Include intercept
        Defaults to 1 if None.
    startingvals : array_like or None, optional
        Starting values for optimization.  If provided, must be a 1-D array
        (or column vector) with length depending on error_type:

        - NORMAL   : constant + p + q - 1
        - STUDENTST: constant + p + q     (extra nu parameter)
        - GED      : constant + p + q     (extra nu parameter)
        - SKEWT    : constant + p + q + 1 (extra nu and lambda parameters)

        The q-1 free beta count stems from the IGARCH unit-root constraint.
    options : dict or None, optional
        Optimizer options dict for ``scipy.optimize.minimize``.  If None,
        default options are created that mirror the MATLAB fminunc defaults.

    Returns
    -------
    tuple
        ``(p, q, error_type, igarch_type, constant, startingvals, options)``
        with validated and possibly defaulted values.  ``error_type`` is
        returned as an integer code (1–4).

    Raises
    ------
    ValueError
        If any parameter fails validation.

    Notes
    -----
    IGARCH has a unit-root constraint:  sum(alpha) + sum(beta) = 1.
    Only q-1 beta parameters are freely estimated; the remaining beta
    is implicitly determined as  1 - sum(alpha) - sum(free_betas).

    Ref: igarch_parameter_check.m — MFE Toolbox v4.0 (Kevin Sheppard)
    """
    # ------------------------------------------------------------------
    # Validate data (epsilon)
    # Ref: igarch_parameter_check.m:26-30
    # MATLAB: if size(epsilon,2)>1 || length(epsilon)==1 -> error
    #         elseif isempty(epsilon) -> error
    # ------------------------------------------------------------------
    data = np.asarray(data, dtype=np.float64)

    # Ref: igarch_parameter_check.m:26 — size(epsilon,2)>1 checks for matrix input
    if data.ndim >= 2 and data.shape[1] > 1:
        raise ValueError('EPSILON series must be a column vector.')

    # Ref: igarch_parameter_check.m:26 — length(epsilon)==1 catches scalars
    # Ref: igarch_parameter_check.m:28 — isempty(epsilon) catches empty arrays
    if data.size == 0:
        raise ValueError('EPSILON is empty.')

    if np.isscalar(data) or data.size == 1:
        raise ValueError('EPSILON series must be a column vector.')

    # ------------------------------------------------------------------
    # Validate q — must be a positive scalar integer >= 1
    # Ref: igarch_parameter_check.m:35-37
    # ------------------------------------------------------------------
    if q is None or (hasattr(q, '__len__') and len(q) == 0):
        raise ValueError('Q must be a positive scalar.')
    if not np.isscalar(q):
        raise ValueError('Q must be a positive scalar.')
    q = int(q)
    if q < 1:
        raise ValueError('Q must be a positive scalar.')

    # ------------------------------------------------------------------
    # Validate p — must be a positive scalar integer >= 1
    # Ref: igarch_parameter_check.m:43-45
    # ------------------------------------------------------------------
    if p is None or (hasattr(p, '__len__') and len(p) == 0):
        raise ValueError('P must be positive scalar.')
    if not np.isscalar(p):
        raise ValueError('P must be positive scalar.')
    p = int(p)
    if p < 1:
        raise ValueError('P must be positive scalar.')

    # ------------------------------------------------------------------
    # Validate error_type — string mapped to integer code
    # Ref: igarch_parameter_check.m:50-68
    # MATLAB defaults to 'NORMAL' when nargin<4 or isempty(errorType)
    # ------------------------------------------------------------------
    if error_type is None or (isinstance(error_type, str) and error_type.strip() == ''):
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
                "error_type must be a string and one of: "
                "'NORMAL', 'STUDENTST', 'GED' or 'SKEWT'."
            )
        error_type = _error_type_map[error_type_upper]
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

    # ------------------------------------------------------------------
    # Validate igarch_type — must be 0 or 1
    # Ref: igarch_parameter_check.m:73-83
    # MATLAB uses {1, 2} with default 2; Python API uses {0, 1}
    # Ref: igarch_parameter_check.m:74 — MATLAB default igarchType=2
    #   maps to Python igarch_type=1
    # ------------------------------------------------------------------
    if igarch_type is None:
        igarch_type = 1

    if not np.isscalar(igarch_type):
        raise ValueError('IGARCHTYPE must be a scalar')

    igarch_type = int(igarch_type)
    if igarch_type not in (0, 1):
        raise ValueError('IGARCHTYPE must be either 0 or 1')

    # ------------------------------------------------------------------
    # Validate constant — must be 0 or 1
    # Ref: igarch_parameter_check.m:89-95
    # ------------------------------------------------------------------
    if constant is None:
        constant = 1

    # Ref: igarch_parameter_check.m:92 — constant = double(constant)
    constant = int(constant)
    if constant not in (0, 1):
        raise ValueError('CONSTANT must be either 0 or 1.')

    # ------------------------------------------------------------------
    # Validate starting values
    # Ref: igarch_parameter_check.m:100-145
    #
    # IGARCH unit-root constraint means q-1 free betas, so:
    #   base_params = constant + p + (q - 1)
    # Distribution parameters add: NORMAL=0, STUDENTST=1, GED=1, SKEWT=2
    # ------------------------------------------------------------------
    if startingvals is not None:
        startingvals = np.asarray(startingvals, dtype=np.float64)

        # Determine if input is a valid column vector shape before flattening
        # Ref: igarch_parameter_check.m:103 — size(startingvals,2)~=1
        _is_column_or_1d = (
            startingvals.ndim == 1
            or (startingvals.ndim == 2 and startingvals.shape[1] == 1)
        )
        startingvals = startingvals.ravel()

        if error_type == 1:
            # NORMAL — no extra distribution parameters
            # Ref: igarch_parameter_check.m:102-105
            expected_len = p + q - 1 + constant
            if len(startingvals) != expected_len or not _is_column_or_1d:
                raise ValueError(
                    'startingvals must be a column vector with '
                    'p+q+CONSTANT-1 elements'
                )
        elif error_type == 2:
            # STUDENTST — one extra parameter: nu (degrees of freedom)
            # Ref: igarch_parameter_check.m:106-111
            expected_len = p + q + constant
            if len(startingvals) != expected_len or not _is_column_or_1d:
                raise ValueError(
                    'startingvals must be a column vector with '
                    'p+q+CONSTANT elements'
                )
            # Ref: igarch_parameter_check.m:110 — nu must exceed 2.1
            # Index in MATLAB (1-based): p+q+constant; Python (0-based): p+q+constant-1
            nu_index = p + q + constant - 1
            if startingvals[nu_index] < 2.1:
                raise ValueError(
                    'Nu must be greater than 2.1 when using Students-T errors'
                )
        elif error_type == 3:
            # GED — one extra parameter: nu (shape)
            # Ref: igarch_parameter_check.m:113-119
            expected_len = p + q + constant
            if len(startingvals) != expected_len or not _is_column_or_1d:
                raise ValueError(
                    'startingvals must be a column vector with '
                    'p+q+CONSTANT elements'
                )
            # Ref: igarch_parameter_check.m:117 — nu must exceed 1.05
            nu_index = p + q + constant - 1
            if startingvals[nu_index] < 1.05:
                raise ValueError(
                    'Nu must be greater than 1 when using GED errors'
                )
        elif error_type == 4:
            # SKEWT — two extra parameters: nu and lambda
            # Ref: igarch_parameter_check.m:120-129
            # Note: MATLAB source has a bug using uppercase CONSTANT; Python
            # uses the correct variable name 'constant'.
            expected_len = p + q + constant + 1
            if len(startingvals) != expected_len or not _is_column_or_1d:
                raise ValueError(
                    'startingvals must be a column vector with '
                    'p+q+CONSTANT+1 elements'
                )
            # Ref: igarch_parameter_check.m:124 — nu > 2.1 for Skew-T
            nu_index = p + q + constant - 1
            if startingvals[nu_index] < 2.1:
                raise ValueError(
                    'Nu must be greater than 2.1 when using Skew T errors'
                )
            # Ref: igarch_parameter_check.m:127-128 — lambda in (-0.9, 0.9)
            lambda_index = p + q + constant
            if startingvals[lambda_index] < -0.9 or startingvals[lambda_index] > 0.9:
                raise ValueError(
                    'Lambda must be between -.9 and .9 when using Skew T errors'
                )

        # Ref: igarch_parameter_check.m:132-133 — all omega/alpha/beta must be > 0
        n_model_params = p + constant + q - 1
        if np.any(startingvals[:n_model_params] <= 0):
            raise ValueError(
                'All startingvals for omega, alpha and beta must be '
                'strictly greater than zero'
            )

        # Ref: igarch_parameter_check.m:135-142 — sum(alpha)+sum(beta) < 1
        # Extract ARCH (alpha) coefficients
        # Ref: igarch_parameter_check.m:135 — MATLAB 1-indexed constant+1:p+constant
        #   → Python 0-indexed constant:constant+p
        alphas = startingvals[constant:constant + p]

        # Extract free GARCH (beta) coefficients — only q-1 are free
        # Ref: igarch_parameter_check.m:136 — MATLAB 1-indexed constant+p+1:q+p+constant-1
        #   → Python 0-indexed constant+p:constant+p+q-1
        betas = startingvals[constant + p:constant + p + q - 1]

        # Ref: igarch_parameter_check.m:137-139 — empty betas treated as 0
        if betas.size == 0:
            betas_sum = 0.0
        else:
            betas_sum = np.sum(betas)

        # The implied last beta = 1 - sum(alphas) - sum(free_betas) must be > 0
        # so sum(alphas) + sum(free_betas) must be < 1
        if (np.sum(alphas) + betas_sum) >= 1.0:
            raise ValueError(
                'The sum of the arch and garch coefficients must be less than 1'
            )
    else:
        # Ref: igarch_parameter_check.m:144 — startingvals=[]
        startingvals = np.array([], dtype=np.float64)

    # ------------------------------------------------------------------
    # Validate / create optimizer options
    # Ref: igarch_parameter_check.m:150-165
    # Replace MATLAB optimset('fminunc') with scipy.optimize.minimize options
    # ------------------------------------------------------------------
    if options is not None:
        # Ref: igarch_parameter_check.m:151-154 — validate that options is usable
        if not isinstance(options, dict):
            raise ValueError(
                'options is not a valid minimization option structure'
            )
    else:
        # Ref: igarch_parameter_check.m:158-165 — default fminunc options
        # Translated to scipy.optimize.minimize compatible options dict:
        #   TolFun=1e-5  → ftol
        #   TolX=1e-5    → gtol (gradient tolerance, analogous for L-BFGS-B)
        #   Display=iter  → disp=True
        #   LargeScale=off → (not applicable; L-BFGS-B/SLSQP are used)
        #   MaxFunEvals=200*(2+p+q) → maxiter
        options = {
            'ftol': 1e-5,
            'gtol': 1e-5,
            'disp': True,
            'maxiter': 200 * (2 + p + q),
        }

    return p, q, error_type, igarch_type, constant, startingvals, options
