"""
AGARCH/NAGARCH estimation results display.

Migrated from univariate/agarch_display.m — displays parameter estimates,
standard errors, t-statistics, p-values, log-likelihood, AIC, and BIC
for AGARCH(P,Q) or NAGARCH(P,Q) models.

Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 1    Date: 7/12/2005
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def agarch_display(
    parameters: np.ndarray,
    ll: float,
    vcv: np.ndarray,
    epsilon: np.ndarray,
    p: int,
    q: int,
    model_type: str = 'AGARCH',
    error_type: str = 'NORMAL',
) -> tuple[str, float, float]:
    """
    Display parameters, t-stats, p-values, log-likelihood and AIC/BIC
    from estimates of an AGARCH(P,Q) or NAGARCH(P,Q) model produced
    using :func:`agarch`.

    Parameters
    ----------
    parameters : np.ndarray
        A ``(2 + p + q + extraP)`` element vector of parameters with ordering
        ``[omega, alpha(1)...alpha(p), gamma, beta(1)...beta(q), [nu], [lambda]]``.
        The number of extra distribution parameters ``extraP`` depends on
        ``error_type``: 0 for NORMAL, 1 for STUDENTST/GED, 2 for SKEWT.
    ll : float
        The log-likelihood value at the optimum.
    vcv : np.ndarray
        Variance-covariance matrix (robust sandwich or inverse Hessian).
        Must be square, positive definite, and of dimension matching
        ``len(parameters)``.
    epsilon : np.ndarray
        Column of mean-zero data used to fit the model (T observations).
    p : int
        Positive integer representing the number of symmetric innovation
        terms (ARCH order).
    q : int
        Non-negative integer representing the number of lagged conditional
        variance terms (GARCH order; use 0 for pure ARCH).
    model_type : str, optional
        The type of variance process, either:

        - ``'AGARCH'``  — Asymmetric GARCH, Engle (1990) **[default]**
        - ``'NAGARCH'`` — Nonlinear Asymmetric GARCH, Engle & Ng (1993)
    error_type : str, optional
        The error distribution used in estimation, valid types are:

        - ``'NORMAL'``    — Gaussian innovations **[default]**
        - ``'STUDENTST'`` — Student-t distributed errors
        - ``'GED'``       — Generalized Error Distribution
        - ``'SKEWT'``     — Skewed Student-t distribution

    Returns
    -------
    tuple[str, float, float]
        A 3-tuple ``(text, AIC, BIC)`` where:

        - **text** (*str*) — Formatted multi-line string containing the
          parameter table, log-likelihood, and information criteria.
        - **AIC** (*float*) — Per-observation Akaike Information Criterion:
          ``-ll/T + 2*k/T``.
        - **BIC** (*float*) — Per-observation Bayesian Information Criterion:
          ``-ll/T + log(T)*k/T``.

    Raises
    ------
    ValueError
        If any input parameter fails validation checks (wrong shape,
        incompatible sizes, unsupported model/error type, etc.).

    See Also
    --------
    agarch : Main AGARCH/NAGARCH estimation driver.

    Notes
    -----
    The AIC and BIC are computed on a per-observation basis to match the
    original MATLAB implementation exactly.

    .. code-block:: text

        AIC = -ll / T + 2 * k / T
        BIC = -ll / T + log(T) * k / T

    where ``T`` is the sample size (``len(epsilon)``) and ``k`` is the
    total number of parameters (``len(parameters)``).
    """
    # ==================================================================
    # PARAMETER CHECKING
    # Ref: agarch_display.m:42-127
    # ==================================================================

    # --- parameters: must be a real vector ---
    # Ref: agarch_display.m:60-62 — MATLAB checks isreal and column shape
    parameters = np.asarray(parameters, dtype=np.float64).flatten()
    if not np.all(np.isreal(parameters)):
        raise ValueError('PARAMETERS must be a column vector.')

    # --- ll: must be a real scalar ---
    # Ref: agarch_display.m:64-66
    if not np.isscalar(ll) or not np.isreal(ll):
        raise ValueError('LL must be a scalar.')
    ll = float(ll)

    # --- vcv: must be square, positive-definite, compatible with parameters ---
    # Ref: agarch_display.m:68-70
    vcv = np.asarray(vcv, dtype=np.float64)
    if vcv.ndim != 2 or vcv.shape[0] != vcv.shape[1]:
        raise ValueError(
            'VCV must be a square positive definite matrix compatible '
            'with PARAMETERS.'
        )
    # Ref: agarch_display.m:68 — min(eig(vcv)) > 0 check
    eig_vals = np.linalg.eigvalsh(vcv)
    if np.min(eig_vals) <= 0:
        raise ValueError(
            'VCV must be a square positive definite matrix compatible '
            'with PARAMETERS.'
        )
    if vcv.shape[0] != len(parameters):
        raise ValueError(
            'VCV must be a square positive definite matrix compatible '
            'with PARAMETERS.'
        )

    # --- epsilon: must be a real vector ---
    # Ref: agarch_display.m:72-74
    epsilon = np.asarray(epsilon, dtype=np.float64).flatten()
    if not np.all(np.isreal(epsilon)):
        raise ValueError('EPSILON must be a T by 1 column vector.')

    # --- p: must be a positive scalar integer ---
    # Ref: agarch_display.m:76-78
    if not np.isscalar(p) or p < 1 or int(p) != p:
        raise ValueError('P must be a positive scalar integer')
    p = int(p)

    # --- q: must be a non-negative scalar integer ---
    # Ref: agarch_display.m:80-82
    if not np.isscalar(q) or q < 0 or int(q) != q:
        raise ValueError('Q must be a non-negative scalar integer')
    q = int(q)

    # --- error_type: distribution selection and extra parameter count ---
    # Ref: agarch_display.m:84-106
    if not isinstance(error_type, str) or not error_type:
        error_type = 'NORMAL'
    error_type = error_type.upper()

    # Map error_type to number of additional distribution parameters
    _error_type_extra = {
        'NORMAL': 0,     # No extra parameters
        'STUDENTST': 1,  # nu (degrees of freedom)
        'GED': 1,        # nu (shape parameter)
        'SKEWT': 2,      # nu (df) and lambda (skewness)
    }
    if error_type not in _error_type_extra:
        raise ValueError(
            'ERRORTYPE is not one of the supported distributions'
        )
    extra_p = _error_type_extra[error_type]

    # --- model_type: must be 'AGARCH' or 'NAGARCH' ---
    # Ref: agarch_display.m:107-123
    if not isinstance(model_type, str) or not model_type:
        model_type = 'AGARCH'
    model_type = model_type.upper()

    if model_type not in ('AGARCH', 'NAGARCH'):
        raise ValueError(
            "MODELTYPE must be either 'AGARCH' or 'NAGARCH'."
        )

    # --- parameter count consistency check ---
    # Ref: agarch_display.m:125-127
    # Total: omega(1) + alpha(p) + gamma(1) + beta(q) + extra distribution
    expected_count = 2 + p + q + extra_p
    if len(parameters) != expected_count:
        raise ValueError(
            'Size of PARAMETERS is not compatible with input P and Q'
        )

    # ==================================================================
    # STATISTICAL COMPUTATIONS
    # Ref: agarch_display.m:132-148
    # ==================================================================

    # Model name string
    # Ref: agarch_display.m:132-136
    model_name = f'{model_type}({p},{q})'

    # Sample size and total parameter count
    T = len(epsilon)
    k = len(parameters)

    # Standard errors from VCV diagonal
    # Ref: agarch_display.m:139 — stderr = sqrt(diag(vcv))
    stderr = np.sqrt(np.diag(vcv))

    # t-statistics
    # Ref: agarch_display.m:141 — tstats = parameters ./ stderr
    tstats = parameters / stderr

    # Two-sided p-values using normal CDF
    # Ref: agarch_display.m:143 — pvals = 2 - 2*normcdf(abs(tstats))
    # Replaces MATLAB duplication/normcdf.m with scipy.stats.norm.cdf
    # per AAP Section 0.5.2 duplication elimination rules
    pvals = 2.0 - 2.0 * norm.cdf(np.abs(tstats))

    # Per-observation information criteria (matches MATLAB exactly)
    # Ref: agarch_display.m:147 — AIC = -ll/T + 2*length(parameters)/T
    # Ref: agarch_display.m:148 — BIC = -ll/T + log(T)*length(parameters)/T
    aic = -ll / T + 2.0 * k / T
    bic = -ll / T + np.log(T) * k / T

    # ==================================================================
    # FORMAT HEADER TEXT
    # Ref: agarch_display.m:153-169
    # ==================================================================
    separator = '-' * 50
    header_lines = [
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

    # Print header lines to stdout
    # Ref: agarch_display.m:167-169
    for line in header_lines:
        print(line)

    # ==================================================================
    # FORMAT PARAMETER TABLE
    # Ref: agarch_display.m:172-245
    # ==================================================================

    # Format each numeric column as right-aligned strings with 4 decimals
    # Ref: agarch_display.m:138-144 — num2str(x,'%4.4f')
    # Ref: agarch_display.m:172-195 — right-alignment loop
    param_strs, param_w = _format_column(parameters)
    stderr_strs, stderr_w = _format_column(stderr)
    tstat_strs, tstat_w = _format_column(tstats)
    pval_strs, pval_w = _format_column(pvals)

    # Column labels — matching MATLAB exactly
    # Ref: agarch_display.m:197
    raw_labels = [' Parameters', '   Std. Err.', '     T-stat', '      P-val']
    col_data = [param_strs, stderr_strs, tstat_strs, pval_strs]
    col_widths = [param_w, stderr_w, tstat_w, pval_w]

    # Align labels and data columns to the same width
    # Ref: agarch_display.m:199-209
    aligned_labels = []
    aligned_cols = []
    for i in range(4):
        max_w = max(len(raw_labels[i]), col_widths[i])
        aligned_labels.append(raw_labels[i].rjust(max_w))
        aligned_cols.append([s.rjust(max_w) for s in col_data[i]])

    # Build variable names
    # Ref: agarch_display.m:211-230
    variable_names = _build_variable_names(p, q, extra_p)

    # Right-align variable names
    # Ref: agarch_display.m:226-229
    max_name_len = max(len(name) for name in variable_names)
    aligned_names = [name.rjust(max_name_len) for name in variable_names]

    # Assemble the table rows
    # Ref: agarch_display.m:232-236
    col_sep = '  '

    # Table header: blank name + column labels
    # Ref: agarch_display.m:233 — strvcat(' ', variable_names) puts blank first
    table_header = ' ' * max_name_len + col_sep + col_sep.join(aligned_labels)

    # Data rows: name + values
    table_rows = [table_header]
    for i in range(k):
        row_values = [aligned_cols[j][i] for j in range(4)]
        row = aligned_names[i] + col_sep + col_sep.join(row_values)
        table_rows.append(row)

    # Print parameter table to stdout
    # Ref: agarch_display.m:245 — disp(outmat)
    for line in table_rows:
        print(line)

    # ==================================================================
    # ASSEMBLE COMPLETE TEXT OUTPUT
    # Ref: agarch_display.m:239-243 — text = strvcat(text2, outmat)
    # ==================================================================
    all_lines = header_lines + table_rows
    text = '\n'.join(all_lines)

    return text, aic, bic


def _format_column(values: np.ndarray) -> tuple[list[str], int]:
    """
    Format a numpy array of numeric values to fixed 4-decimal-place strings
    and right-align them to a common width.

    Replicates MATLAB's ``num2str(x, '%4.4f')`` followed by the
    right-alignment loop in agarch_display.m:172-195.

    Parameters
    ----------
    values : np.ndarray
        1-D array of numeric values to format.

    Returns
    -------
    tuple[list[str], int]
        A tuple ``(formatted_strings, max_width)`` where each string is
        right-justified to ``max_width`` characters.
    """
    formatted = [f'{v:.4f}' for v in values]
    max_width = max(len(s) for s in formatted)
    aligned = [s.rjust(max_width) for s in formatted]
    return aligned, max_width


def _build_variable_names(p: int, q: int, extra_p: int) -> list[str]:
    """
    Construct the list of parameter variable names for the display table.

    Ref: agarch_display.m:211-230 — MATLAB builds names for omega,
    alpha(1..p), gamma, beta(1..q). For extra distribution parameters,
    MATLAB leaves them blank (empty cells become rows of spaces).
    Python provides explicit names 'nu' and 'lambda' for clarity.

    Parameters
    ----------
    p : int
        Number of alpha (ARCH) parameters.
    q : int
        Number of beta (GARCH) parameters.
    extra_p : int
        Number of extra distribution parameters (0, 1, or 2).

    Returns
    -------
    list[str]
        List of parameter names in canonical order.
    """
    names: list[str] = []

    # Constant term
    names.append('omega')

    # Ref: agarch_display.m:215-218 — alpha indices are 1-based in MATLAB
    for i in range(1, p + 1):
        names.append(f'alpha({i})')

    # Asymmetry / nonlinearity term
    names.append('gamma')

    # Ref: agarch_display.m:221-224 — beta indices are 1-based in MATLAB
    for i in range(1, q + 1):
        names.append(f'beta({i})')

    # Distribution parameters — MATLAB (agarch_display.m:212) allocates
    # cell(length(parameters),1) but only fills the first 2+p+q entries,
    # leaving extra cells as empty []. Python adds explicit labels.
    if extra_p >= 1:
        names.append('nu')
    if extra_p >= 2:
        names.append('lambda')

    return names
