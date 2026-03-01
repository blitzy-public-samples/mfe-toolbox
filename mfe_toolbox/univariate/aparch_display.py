"""
APARCH model estimation result display.

Produces formatted text output of APARCH(P,O,Q) estimation results including
parameter estimates, standard errors, t-statistics, p-values, log-likelihood,
AIC, and BIC.

Migrated from: univariate/aparch_display.m (264 lines, 7505 bytes)
Original author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
"""

import warnings

import numpy as np
from scipy.stats import norm


def aparch_display(
    parameters: np.ndarray,
    ll: float,
    vcv: np.ndarray,
    epsilon: np.ndarray,
    p: int,
    o: int,
    q: int,
    error_type: str = 'NORMAL',
    user_delta: float | None = None,
) -> tuple[str, float, float]:
    """Display APARCH estimation results with formatted parameter table.

    Prints and returns a formatted summary of APARCH(P,O,Q) estimation results,
    including parameter names (omega, alpha_i, gamma_j, beta_k, delta, nu, lambda),
    standard errors, t-statistics, p-values, log-likelihood, AIC, and BIC.

    Parameters
    ----------
    parameters : np.ndarray
        A 1D array of estimated parameters with structure:
        [omega, alpha(1)...alpha(p), gamma(1)...gamma(o),
         beta(1)...beta(q), delta, [nu, [lambda]]]
        where delta is included only if ``user_delta`` is None (estimated),
        nu is included for STUDENTST/GED/SKEWT, and lambda for SKEWT.
    ll : float
        The log-likelihood value at the optimum.
    vcv : np.ndarray
        Variance-covariance matrix (inverse Hessian) of parameter estimates.
        Must be square with dimension matching ``len(parameters)``.
    epsilon : np.ndarray
        Mean-zero return data (1D array). Used only for ``T = len(epsilon)``
        in AIC/BIC computation. Replaces MATLAB's ``data`` argument.
        Ref: aparch_display.m:4 — MATLAB uses 'DATA' as 4th arg.
    p : int
        Positive integer (>= 1) representing the number of symmetric
        innovations.
    o : int
        Non-negative integer (>= 0) representing the number of asymmetric
        innovations (0 for symmetric processes).
    q : int
        Non-negative integer (>= 0) representing the number of lagged
        conditional variances (0 for pure ARCH).
    error_type : str, optional
        Error distribution used in estimation. One of:
        ``'NORMAL'`` (default), ``'STUDENTST'``, ``'GED'``, ``'SKEWT'``.
    user_delta : float or None, optional
        User-provided fixed delta value. If None (default), delta was estimated
        and appears in the parameters vector. If a float, delta was fixed during
        estimation and is not in the parameters vector.

    Returns
    -------
    tuple[str, float, float]
        A tuple of ``(text, aic, bic)`` where:

        - **text** (*str*) — The complete formatted display string.
        - **aic** (*float*) — Akaike Information Criterion.
        - **bic** (*float*) — Bayesian (Schwarz) Information Criterion.

    Raises
    ------
    ValueError
        If any input validation fails (invalid parameters, incompatible
        dimensions, unsupported error_type, etc.).

    Warns
    -----
    UserWarning
        If the VCV matrix is not positive definite.

    See Also
    --------
    aparch : APARCH model estimation driver.

    Notes
    -----
    Ref: aparch_display.m — MATLAB MFE Toolbox by Kevin Sheppard.
    Parameter names use 1-based labeling for display (matching MATLAB
    convention), while internal array indexing is 0-based (Python convention).
    The MATLAB source has a typo on line 260 (``finaLine`` instead of
    ``finalLine``); this is corrected in the Python version.
    """
    # ===================================================================
    # Input Validation
    # Ref: aparch_display.m:46-127
    # ===================================================================

    # --- Validate user_delta ---
    # Ref: aparch_display.m:65-69
    if user_delta is not None:
        if not (np.isscalar(user_delta) and np.isreal(user_delta)):
            raise ValueError('USERDELTA must be a scalar.')
        user_delta = float(user_delta)

    # --- Validate and normalize error_type ---
    # Ref: aparch_display.m:70-72 — empty/None defaults to NORMAL
    if error_type is None or (isinstance(error_type, str) and error_type.strip() == ''):
        error_type = 'NORMAL'

    # Ref: aparch_display.m:105-106 — non-string triggers error via switch default
    if not isinstance(error_type, str):
        raise ValueError('ERRORTYPE is not one of the supported distributions')

    # Ref: aparch_display.m:108-123 — map error distribution to code and extra
    # parameter count.  NORMAL→0 extra, STUDENTST→1 (nu), GED→1 (nu), SKEWT→2
    # (nu + lambda).
    _error_type_map: dict[str, tuple[int, int]] = {
        'NORMAL': (1, 0),
        'STUDENTST': (2, 1),
        'GED': (3, 1),
        'SKEWT': (4, 2),
    }
    error_type_upper = error_type.upper()
    if error_type_upper not in _error_type_map:
        raise ValueError('ERRORTYPE is not one of the supported distributions')
    error_code, extra_p = _error_type_map[error_type_upper]

    # --- Validate parameters ---
    # Ref: aparch_display.m:74-76 — must be real column vector
    parameters = np.asarray(parameters)
    if np.iscomplexobj(parameters):
        raise ValueError('PARAMETERS must be a column vector.')
    parameters = np.asarray(parameters, dtype=np.float64).ravel()

    # --- Validate ll ---
    # Ref: aparch_display.m:78-80 — must be a real scalar
    if not np.isscalar(ll) or not np.isreal(ll):
        raise ValueError('LL must be a scalar.')
    ll = float(ll)

    # --- Validate vcv ---
    # Ref: aparch_display.m:82-84 — must be square, same size as parameters
    vcv = np.asarray(vcv, dtype=np.float64)
    if vcv.ndim != 2 or vcv.shape[0] != vcv.shape[1] or vcv.shape[0] != len(parameters):
        raise ValueError(
            'VCV must be a square positive definite matrix compatible with PARAMETERS.'
        )

    # Check VCV positive definiteness via eigenvalues of symmetric matrix
    # Ref: aparch_display.m:85-87 — MATLAB uses eig(); Python uses eigvalsh()
    # for real symmetric matrices (numerically more stable).
    eig_min = np.min(np.linalg.eigvalsh(vcv))
    if eig_min <= 0:
        warnings.warn(
            'VCV is not positive definite.',
            stacklevel=2,
        )

    # --- Validate epsilon (data) ---
    # Ref: aparch_display.m:89-91 — must be real column vector
    epsilon = np.asarray(epsilon)
    if np.iscomplexobj(epsilon):
        raise ValueError('DATA must be a T by 1 column vector.')
    epsilon = np.asarray(epsilon, dtype=np.float64).ravel()

    # --- Validate p ---
    # Ref: aparch_display.m:93-95 — positive integer, floor(p)==p
    try:
        p_int = int(p)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('P must be a positive scalar integer')
    if p_int != p or p_int < 1:
        raise ValueError('P must be a positive scalar integer')
    p = p_int

    # --- Validate o ---
    # Ref: aparch_display.m:97-99 — non-negative integer
    try:
        o_int = int(o)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('O must be a non-negative scalar integer')
    if o_int != o or o_int < 0:
        raise ValueError('O must be a non-negative scalar integer')
    o = o_int

    # --- Validate q ---
    # Ref: aparch_display.m:101-103 — non-negative integer
    try:
        q_int = int(q)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('Q must be a non-negative scalar integer')
    if q_int != q or q_int < 0:
        raise ValueError('Q must be a non-negative scalar integer')
    q = q_int

    # --- Validate parameter vector length ---
    # Ref: aparch_display.m:125-127
    # When user_delta is None, delta is estimated → adds 1 to expected length
    delta_count = 0 if user_delta is not None else 1
    expected_len = 1 + p + o + q + delta_count + extra_p
    if len(parameters) != expected_len:
        raise ValueError(
            'Size of PARAMETERS is not compatible with input P, O, Q and USERDELTA'
        )

    # ===================================================================
    # Compute Statistics
    # Ref: aparch_display.m:131-143
    # ===================================================================

    num_params = len(parameters)
    # Ref: aparch_display.m:141 — T is the length of the data vector
    T = len(epsilon)

    # Ref: aparch_display.m:131
    model_name = f'APARCH({p},{o},{q})'

    # Standard errors from VCV diagonal
    # Ref: aparch_display.m:134
    stderr = np.sqrt(np.diag(vcv))

    # T-statistics
    # Ref: aparch_display.m:136
    tstats = parameters / stderr

    # Two-sided p-values from the standard normal distribution
    # Ref: aparch_display.m:138 — MATLAB uses normcdf from duplication/;
    # replaced with scipy.stats.norm.cdf per AAP §0.5.2
    pvals = 2.0 - 2.0 * norm.cdf(np.abs(tstats))

    # Akaike Information Criterion
    # Ref: aparch_display.m:142
    aic = float(-ll / T + 2.0 * num_params / T)

    # Bayesian (Schwarz) Information Criterion
    # Ref: aparch_display.m:143
    bic = float(-ll / T + np.log(T) * num_params / T)

    # ===================================================================
    # Build Parameter Names
    # Ref: aparch_display.m:207-233
    # ===================================================================

    variable_names: list[str] = []

    # Ref: aparch_display.m:208 — omega is always the first parameter
    variable_names.append('omega')

    # Ref: aparch_display.m:210-213 — alpha parameters with 1-based labels
    # (display uses MATLAB 1-based indexing convention)
    for i in range(1, p + 1):
        variable_names.append(f'alpha({i})')

    # Ref: aparch_display.m:214-217 — gamma (asymmetric) parameters
    for i in range(1, o + 1):
        variable_names.append(f'gamma({i})')

    # Ref: aparch_display.m:218-221 — beta (GARCH lag) parameters
    for i in range(1, q + 1):
        variable_names.append(f'beta({i})')

    # Ref: aparch_display.m:222-225 — delta appears only if it was estimated
    if user_delta is None:
        variable_names.append('delta')

    # Ref: aparch_display.m:227-230 — nu for non-Normal distributions
    if error_code > 1:
        variable_names.append('nu')

    # Ref: aparch_display.m:231-233 — lambda only for Skewed-T
    if error_code == 4:
        variable_names.append('lambda')

    # ===================================================================
    # Format Display Output
    # Ref: aparch_display.m:148-262
    # ===================================================================

    # --- Header block ---
    # Ref: aparch_display.m:148-159
    separator = '-' * 50
    header_lines: list[str] = [
        ' ',
        ' ',
        separator,
        model_name,
        separator,
        ' ',
        f'Loglikelihood: {ll:.2f}',
        f'AIC: {aic:.4f}',
        f'BIC: {bic:.4f}',
        ' ',
    ]

    # --- Format numerical columns ---
    # Ref: aparch_display.m:133-139 — num2str with '%4.4f' format
    param_strs = [f'{v:.4f}' for v in parameters]
    stderr_strs = [f'{v:.4f}' for v in stderr]
    tstat_strs = [f'{v:.4f}' for v in tstats]
    pval_strs = [f'{v:.4f}' for v in pvals]

    # Column labels matching MATLAB exactly
    # Ref: aparch_display.m:192 — leading spaces are aesthetic indentation
    col_labels = [' Parameters', '   Std. Err.', '     T-stat', '      P-val']
    col_data = [param_strs, stderr_strs, tstat_strs, pval_strs]

    # Right-align each column so label and data widths match
    # Ref: aparch_display.m:194-204 — MATLAB pads from the left to maxcols
    aligned_labels: list[str] = []
    aligned_cols: list[list[str]] = []
    for label, data_col in zip(col_labels, col_data):
        max_width = max(len(label), max(len(s) for s in data_col))
        aligned_labels.append(label.rjust(max_width))
        aligned_cols.append([s.rjust(max_width) for s in data_col])

    # Right-align variable names to the longest name
    # Ref: aparch_display.m:236-240
    max_name_len = max(len(name) for name in variable_names)
    aligned_names = [name.rjust(max_name_len) for name in variable_names]

    # --- Build parameter table ---
    # Ref: aparch_display.m:242-246
    # Header row: blank name-column placeholder + column labels
    # Ref: aparch_display.m:243 — strvcat(' ', variable_names) puts a space
    # row above the names; in Python the header row uses spaces for the name slot.
    table_header = ' ' * max_name_len
    for label in aligned_labels:
        table_header += '  ' + label

    # Data rows: right-aligned variable name + formatted values
    table_rows: list[str] = [table_header]
    for i in range(num_params):
        row = aligned_names[i]
        for col in aligned_cols:
            row += '  ' + col[i]
        table_rows.append(row)

    # --- Combine all sections ---
    # Ref: aparch_display.m:249-253 — strvcat(text2, outmat)
    all_lines = header_lines + table_rows

    # Optional user_delta annotation
    # Ref: aparch_display.m:257-262
    # Note: MATLAB source line 260 has typo 'finaLine' instead of 'finalLine';
    # corrected in this Python version.
    if user_delta is not None:
        all_lines.append(' ')
        # Ref: aparch_display.m:259 — num2str(userDelta) uses compact format.
        # Python ':g' format drops trailing zeros, matching MATLAB num2str.
        all_lines.append(f'Model estimated with delta = {user_delta:g}')

    # Build complete text string
    text = '\n'.join(all_lines)

    # Print the formatted output to stdout
    # Ref: aparch_display.m:162-164 (header), 255 (table), 258-260 (delta)
    print(text)

    return (text, aic, bic)
