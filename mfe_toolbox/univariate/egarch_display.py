"""
EGARCH estimation results display.

Migrated from univariate/egarch_display.m — displays EGARCH(P,O,Q) estimation
results with log-variance parameter names (omega, alpha, gamma, beta), standard
errors, t-statistics, p-values, log-likelihood, AIC, and BIC in a formatted
text table.

Copyright: Kevin Sheppard, University of Oxford
Python translation preserves MATLAB output format and numerical parity ±1e-6.
"""

import warnings

import numpy as np
from scipy.stats import norm


def egarch_display(
    parameters: np.ndarray,
    ll: float,
    vcv: np.ndarray,
    epsilon: np.ndarray,
    p: int,
    o: int,
    q: int,
    error_type: str = 'NORMAL',
) -> tuple[str, float, float]:
    """Display EGARCH estimation results as a formatted text table.

    Produces a formatted display of EGARCH(P,O,Q) estimation results including
    parameter estimates, standard errors, t-statistics, p-values, the
    log-likelihood value, AIC, and BIC.

    Parameters
    ----------
    parameters : np.ndarray
        1-D array of estimated EGARCH parameters in the order:
        [omega, alpha(1)...alpha(p), gamma(1)...gamma(o), beta(1)...beta(q),
         nu (if STUDENTST/GED/SKEWT), lambda (if SKEWT)].
    ll : float
        Log-likelihood value at the estimated parameters.
    vcv : np.ndarray
        Variance-covariance matrix of the parameter estimates.  Must be a
        square matrix of dimension len(parameters) × len(parameters).
    epsilon : np.ndarray
        1-D array of the data (residuals / innovations) used in estimation.
        Used only to determine the sample size T for AIC/BIC computation.
    p : int
        Number of symmetric innovation (|z|) lags.  Must be >= 1.
    o : int
        Number of asymmetric innovation (z) lags.  Must be >= 0.
    q : int
        Number of GARCH (log-variance) lags.  Must be >= 0.
    error_type : str, optional
        Distribution assumption for the innovations.  One of
        ``'NORMAL'``, ``'STUDENTST'``, ``'GED'``, or ``'SKEWT'``.
        Default is ``'NORMAL'``.

    Returns
    -------
    text : str
        The complete formatted estimation result string.
    aic : float
        Akaike Information Criterion (per observation).
    bic : float
        Bayesian Information Criterion (per observation).

    Raises
    ------
    ValueError
        If any input fails validation (e.g. parameters is not a 1-D real
        array, ll is not a scalar, vcv is not square/compatible, epsilon is
        not 1-D, p/o/q are invalid, error_type is unrecognised, or the
        parameter vector length does not match the model specification).
    """

    # ------------------------------------------------------------------
    # Phase 1 — Input validation
    # Ref: egarch_display.m:74-129
    # ------------------------------------------------------------------

    # Validate parameters — must be a 1-D real numpy array
    # Ref: egarch_display.m:76-77
    parameters = np.asarray(parameters, dtype=np.float64).ravel()
    if not np.all(np.isreal(parameters)):
        raise ValueError('PARAMETERS must be a column vector.')

    # Validate ll — must be a real scalar
    # Ref: egarch_display.m:80-81
    if not np.isscalar(ll):
        raise ValueError('LL must be a scalar.')
    ll = float(ll)

    # Validate vcv — must be square and compatible with parameters
    # Ref: egarch_display.m:84-85
    vcv = np.asarray(vcv, dtype=np.float64)
    num_params = len(parameters)
    if vcv.ndim != 2 or vcv.shape[0] != vcv.shape[1] or vcv.shape[0] != num_params:
        raise ValueError(
            'VCV must be a square positive definite matrix compatible with PARAMETERS.'
        )

    # Check VCV positive-definiteness via eigenvalues
    # Ref: egarch_display.m:85-87 (MATLAB uses eig; Python uses eigvalsh for symmetric)
    eigvals = np.linalg.eigvalsh(vcv)
    if np.any(eigvals <= 0):
        warnings.warn(
            'VCV is not positive definite. Standard errors may be unreliable.',
            UserWarning,
            stacklevel=2,
        )

    # Validate epsilon — must be a 1-D real array
    # Ref: egarch_display.m:88-89
    epsilon = np.asarray(epsilon, dtype=np.float64).ravel()
    if epsilon.ndim != 1 or len(epsilon) == 0:
        raise ValueError('DATA must be a T by 1 column vector.')

    # Validate p — positive integer >= 1
    # Ref: egarch_display.m:92-93
    if not isinstance(p, (int, np.integer)) or p < 1:
        raise ValueError('P must be a positive scalar integer.')

    # Validate o — non-negative integer >= 0
    # Ref: egarch_display.m:96-97
    if not isinstance(o, (int, np.integer)) or o < 0:
        raise ValueError('O must be a non-negative scalar integer.')

    # Validate q — non-negative integer >= 0
    # Ref: egarch_display.m:100-101
    if not isinstance(q, (int, np.integer)) or q < 0:
        raise ValueError('Q must be a non-negative scalar integer.')

    # Map error_type string to numeric code and determine extra parameters
    # Ref: egarch_display.m:107-122
    # NOTE: The MATLAB source has a bug (lines 58-61) that unconditionally
    # resets errorType='NORMAL' when nargin==8.  The Python version fixes this
    # by correctly honouring the user's error_type argument, enabling the
    # distribution parameters (nu, lambda) to be named and displayed.
    error_type_upper = error_type.upper().strip()
    _error_map = {
        'NORMAL': 0,
        'STUDENTST': 1,
        'GED': 1,
        'SKEWT': 2,
    }
    if error_type_upper not in _error_map:
        raise ValueError(
            'ERRORTYPE is not one of the supported distributions '
            "(NORMAL, STUDENTST, GED, SKEWT)."
        )
    extra_p = _error_map[error_type_upper]

    # Validate parameter vector length matches model specification
    # Ref: egarch_display.m:127-129
    expected_len = 1 + p + o + q + extra_p
    if num_params != expected_len:
        raise ValueError(
            f'The length of PARAMETERS ({num_params}) is not compatible with '
            f'the model specification (expected {expected_len}).'
        )

    # ------------------------------------------------------------------
    # Phase 2 — Compute statistics
    # Ref: egarch_display.m:134-158
    # ------------------------------------------------------------------

    # Standard errors from VCV diagonal
    # Ref: egarch_display.m:149
    stderr = np.sqrt(np.diag(vcv))

    # T-statistics
    # Ref: egarch_display.m:151
    tstats = parameters / stderr

    # Two-sided p-values via normal CDF
    # Ref: egarch_display.m:153 — replaces MATLAB normcdf from duplication/
    pvals = 2.0 - 2.0 * norm.cdf(np.abs(tstats))

    # Sample size
    # Ref: egarch_display.m:156
    T = len(epsilon)

    # AIC and BIC (per-observation)
    # Ref: egarch_display.m:157-158
    aic = -ll / T + 2.0 * num_params / T
    bic = -ll / T + np.log(T) * num_params / T

    # ------------------------------------------------------------------
    # Phase 3 — Build parameter names
    # Ref: egarch_display.m:222-243
    # ------------------------------------------------------------------

    names: list[str] = ['omega']

    # alpha(i) — symmetric innovation terms (|z| lags)
    for i in range(1, p + 1):
        names.append(f'alpha({i})')

    # gamma(j) — asymmetric innovation terms (z lags)
    for j in range(1, o + 1):
        names.append(f'gamma({j})')

    # beta(k) — GARCH log-variance lags
    for k in range(1, q + 1):
        names.append(f'beta({k})')

    # Distribution parameters (correctly named, fixing MATLAB bug)
    if error_type_upper in ('STUDENTST', 'GED', 'SKEWT'):
        names.append('nu')
    if error_type_upper == 'SKEWT':
        names.append('lambda')

    # ------------------------------------------------------------------
    # Phase 4 — Format numbers
    # Ref: egarch_display.m:148-154, 183-205
    # ------------------------------------------------------------------

    # Format each numeric value to 4 decimal places (matching %4.4f)
    fmt_params = [f'{v:.4f}' for v in parameters]
    fmt_stderr = [f'{v:.4f}' for v in stderr]
    fmt_tstats = [f'{v:.4f}' for v in tstats]
    fmt_pvals = [f'{v:.4f}' for v in pvals]

    # Determine column widths
    # Ref: egarch_display.m:183-205 — MATLAB pads with leading blanks
    max_name_len = max(len(n) for n in names)

    # Column header labels (right-aligned in MATLAB)
    col_headers = ['Parameters', 'Std. Err.', 'T-stat', 'P-val']

    # Determine width per numeric column — at least as wide as the header
    col_widths: list[int] = []
    for hdr, fmt_col in zip(
        col_headers,
        [fmt_params, fmt_stderr, fmt_tstats, fmt_pvals],
    ):
        w = max(len(hdr), *(len(s) for s in fmt_col))
        col_widths.append(w)

    # ------------------------------------------------------------------
    # Phase 5 — Build display text
    # Ref: egarch_display.m:164-265
    # ------------------------------------------------------------------

    lines: list[str] = []
    sep = '-' * 50

    # Model name
    # Ref: egarch_display.m:131
    model_name = f'EGARCH({p},{o},{q})'

    # Header block
    # Ref: egarch_display.m:164-174
    lines.append(sep)
    lines.append(model_name)
    lines.append(sep)
    lines.append(f'Log Likelihood: {ll:.2f}')
    lines.append(f'AIC: {aic:.4f}')
    lines.append(f'BIC: {bic:.4f}')
    lines.append(sep)

    # Column header row
    # Ref: egarch_display.m:207
    header_parts = [' ' * max_name_len]
    for hdr, w in zip(col_headers, col_widths):
        header_parts.append(hdr.rjust(w))
    header_line = '  '.join(header_parts)
    lines.append(header_line)
    lines.append(sep)

    # Parameter rows — right-aligned values
    # Ref: egarch_display.m:246-265
    for idx in range(num_params):
        name_str = names[idx].rjust(max_name_len)
        val_str = fmt_params[idx].rjust(col_widths[0])
        se_str = fmt_stderr[idx].rjust(col_widths[1])
        ts_str = fmt_tstats[idx].rjust(col_widths[2])
        pv_str = fmt_pvals[idx].rjust(col_widths[3])
        row = f'{name_str}  {val_str}  {se_str}  {ts_str}  {pv_str}'
        lines.append(row)

    lines.append(sep)

    # Combine into a single text string
    text = '\n'.join(lines)

    # Print the formatted output to stdout
    # Ref: egarch_display.m:177-179, 265
    print(text)

    return text, aic, bic
