"""
TARCH/GJR-GARCH estimation result display.

Migrated from univariate/tarch_display.m (MFE Toolbox, Version 4.0).
Displays TARCH/GJR-GARCH estimation results with asymmetry parameter names,
formatted as an aligned text table with standard errors, t-statistics,
and p-values.  Computes and displays AIC/BIC.

Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 3    Date: 9/1/2005
"""

import numpy as np
from scipy.stats import norm


def tarch_display(
    parameters: np.ndarray,
    ll: float,
    vcv: np.ndarray,
    epsilon: np.ndarray,
    p: int,
    o: int,
    q: int,
    tarch_type: str = "GARCH",
    error_type: str = "NORMAL",
) -> tuple[str, float, float]:
    """Display TARCH/GJR-GARCH estimation results.

    Display parameters, t-stats, p-values, log-likelihood and AIC/BIC from
    estimates of a TARCH(P,O,Q) produced using ``tarch``.

    Parameters
    ----------
    parameters : np.ndarray
        A 1-D array of model parameters with structure:
        ``[omega, alpha(1)...alpha(p), gamma(1)...gamma(o),
        beta(1)...beta(q), [nu, lambda]]``.
    ll : float
        The log-likelihood at the optimum.
    vcv : np.ndarray
        Variance-covariance matrix (inverse Hessian) of parameter estimates.
        Must be square, with positive diagonal, and compatible in size with
        *parameters*.
    epsilon : np.ndarray
        A 1-D array of mean-zero data (T observations).  Named ``data`` in
        the original MATLAB source.
    p : int
        Positive integer representing the number of symmetric innovations.
    o : int
        Non-negative integer representing the number of asymmetric
        innovations (0 for symmetric processes).
    q : int
        Non-negative integer representing the number of lags of conditional
        variance (0 for ARCH).
    tarch_type : str, optional
        The type of variance process:

        * ``'TARCH'`` – Model evolves in absolute values (tarchType = 1 in
          MATLAB).
        * ``'GARCH'`` – Model evolves in squares (tarchType = 2 in MATLAB)
          **[DEFAULT]**.
    error_type : str, optional
        The error distribution used:

        * ``'NORMAL'``    – Gaussian innovations **[DEFAULT]**
        * ``'STUDENTST'`` – Student-t distributed errors
        * ``'GED'``       – Generalised Error Distribution
        * ``'SKEWT'``     – Skewed-t distribution

    Returns
    -------
    text : str
        Formatted string with model estimation results (header + parameter
        table).
    AIC : float
        Akaike Information Criterion computed from the log-likelihood.
    BIC : float
        Schwarz / Bayesian Information Criterion computed from the
        log-likelihood.

    Raises
    ------
    ValueError
        If any input fails the validation checks described in
        ``tarch_display.m`` lines 76–129.

    See Also
    --------
    tarch : TARCH/GJR-GARCH model estimation.

    Notes
    -----
    Migrated from ``tarch_display.m``.  Per the Agent Action Plan,
    ``normcdf`` from the ``duplication/`` directory is replaced by
    ``scipy.stats.norm.cdf`` — no custom reimplementation is permitted.

    The argument order of *tarch_type* and *error_type* is swapped relative
    to the MATLAB original (where ``errorType`` preceded ``tarchType``).
    Both are now keyword-only-style with string values instead of the
    numeric codes used in MATLAB.
    """

    # ================================================================
    # PARAMETER CHECKING
    # Ref: tarch_display.m:43-132
    # ================================================================

    # Handle None / empty defaults
    # Ref: tarch_display.m:69-74 — MATLAB replaces empty args with defaults
    if tarch_type is None or tarch_type == "":
        tarch_type = "GARCH"
    if error_type is None or error_type == "":
        error_type = "NORMAL"

    # --- parameters: must be 1-D real array (column-vector equivalent) ---
    # Ref: tarch_display.m:76-77
    parameters = np.asarray(parameters)
    if np.iscomplexobj(parameters):
        raise ValueError("PARAMETERS must be a column vector.")
    # Ref: tarch_display.m:76 — size(parameters,2)~=1 check; in Python we
    # accept any shape that can be flattened to 1-D.
    if parameters.ndim == 2 and parameters.shape[1] != 1:
        raise ValueError("PARAMETERS must be a column vector.")
    parameters = parameters.ravel().astype(np.float64)

    # --- ll: must be a real scalar ---
    # Ref: tarch_display.m:80-81
    _ll = np.asarray(ll)
    if _ll.size != 1 or np.iscomplexobj(_ll):
        raise ValueError("LL must be a scalar.")
    ll = float(_ll.ravel()[0])

    # --- vcv: square, positive diagonal, compatible with parameters ---
    # Ref: tarch_display.m:84-85
    vcv = np.asarray(vcv, dtype=np.float64)
    if (
        vcv.ndim != 2
        or vcv.shape[0] != vcv.shape[1]
        or np.any(np.diag(vcv) <= 0)
        or vcv.shape[0] != len(parameters)
    ):
        raise ValueError(
            "VCV must be a square positive definite matrix "
            "compatible with PARAMETERS."
        )

    # --- epsilon (data): must be 1-D real array ---
    # Ref: tarch_display.m:88-89
    epsilon = np.asarray(epsilon)
    if np.iscomplexobj(epsilon):
        raise ValueError("DATA must be a T by 1 column vector.")
    if epsilon.ndim == 2 and epsilon.shape[1] != 1:
        raise ValueError("DATA must be a T by 1 column vector.")
    epsilon = epsilon.ravel().astype(np.float64)

    # --- p: positive scalar integer ---
    # Ref: tarch_display.m:92-93
    # Note: MATLAB error message deliberately preserves the original typo
    # "postitive" from tarch_display.m:93.
    if not isinstance(p, (int, np.integer)) or p < 1:
        raise ValueError("P must be a postitive scalar integer")

    # --- o: non-negative scalar integer ---
    # Ref: tarch_display.m:96-97
    if not isinstance(o, (int, np.integer)) or o < 0:
        raise ValueError("O must be a non-negative scalar integer")

    # --- q: non-negative scalar integer ---
    # Ref: tarch_display.m:100-101
    if not isinstance(q, (int, np.integer)) or q < 0:
        raise ValueError("Q must be a non-negative scalar integer")

    # --- error_type mapping ---
    # Ref: tarch_display.m:107-122
    error_type_upper = (
        error_type.upper() if isinstance(error_type, str) else ""
    )
    if error_type_upper == "NORMAL":
        error_type_code = 1
        extra_p = 0
    elif error_type_upper == "STUDENTST":
        error_type_code = 2
        extra_p = 1
    elif error_type_upper == "GED":
        error_type_code = 3
        extra_p = 1
    elif error_type_upper == "SKEWT":
        error_type_code = 4
        extra_p = 2
    else:
        raise ValueError(
            "ERRORTYPE is not one of the supported distributions"
        )

    # --- tarch_type mapping: 'TARCH' → 1, 'GARCH' → 2 ---
    # Ref: tarch_display.m:124-126 — MATLAB checks ismember(tarchType,[1 2])
    tarch_type_upper = (
        tarch_type.upper() if isinstance(tarch_type, str) else ""
    )
    if tarch_type_upper == "TARCH":
        tarch_type_code = 1
    elif tarch_type_upper == "GARCH":
        tarch_type_code = 2
    else:
        raise ValueError("TARCHTYPE must be either 1 or 2")

    # --- Verify parameter count ---
    # Ref: tarch_display.m:127-129
    expected_count = 1 + p + o + q + extra_p
    if len(parameters) != expected_count:
        raise ValueError(
            "Size of PARAMETERS is not compatible with input P, O, Q"
        )

    # ================================================================
    # MODEL NAME CONSTRUCTION
    # Ref: tarch_display.m:136-146
    # ================================================================
    if tarch_type_code == 1:
        model_name = f"TARCH({p},{o},{q})"
    else:
        if o > 0:
            model_name = f"GJR-GARCH({p},{o},{q})"
        elif q > 0:
            model_name = f"GARCH({p},{q})"
        else:
            model_name = f"ARCH({p})"

    # ================================================================
    # STATISTICAL COMPUTATION
    # Ref: tarch_display.m:148-158
    # ================================================================

    # Standard errors from the diagonal of the VCV matrix
    # Ref: tarch_display.m:149
    stderr = np.sqrt(np.diag(vcv))

    # t-statistics
    # Ref: tarch_display.m:151
    tstats = parameters / stderr

    # Two-sided p-values using the normal CDF
    # Ref: tarch_display.m:153 — MATLAB uses normcdf from duplication/
    # Per AAP duplication/ elimination rule: replaced with scipy.stats.norm.cdf
    pvals = 2.0 - 2.0 * norm.cdf(np.abs(tstats))

    # Sample size and information criteria
    # Ref: tarch_display.m:156-158
    T = len(epsilon)
    AIC = float(-ll / T + 2.0 * len(parameters) / T)
    BIC = float(-ll / T + np.log(T) * len(parameters) / T)

    # ================================================================
    # FORMAT OUTPUT — HEADER SECTION
    # Ref: tarch_display.m:163-174
    # ================================================================
    header_lines: list[str] = []
    header_lines.append(" ")                             # text{1}
    header_lines.append(" ")                             # text{2}
    header_lines.append("-" * 50)                        # text{3}
    header_lines.append(model_name)                      # text{4}
    header_lines.append("-" * 50)                        # text{5}
    header_lines.append(" ")                             # text{6}
    # Ref: tarch_display.m:171 — sprintf('%1.2f',ll)
    header_lines.append(f"Loglikelihood: {ll:.2f}")      # text{7}
    # Ref: tarch_display.m:172-173 — sprintf('%1.4f',AIC/BIC)
    header_lines.append(f"AIC: {AIC:.4f}")               # text{8}
    header_lines.append(f"BIC: {BIC:.4f}")               # text{9}
    header_lines.append(" ")                             # text{10}

    # Print header section
    # Ref: tarch_display.m:177-179
    for line in header_lines:
        print(line)

    # ================================================================
    # FORMAT PARAMETER TABLE
    # Ref: tarch_display.m:148-154, 182-256, 265
    # ================================================================

    K = len(parameters)

    # Format all numeric values with 4 decimal places
    # Ref: tarch_display.m:148 — num2str(parameters,'%4.4f')
    param_strs = [f"{parameters[i]:.4f}" for i in range(K)]
    stderr_strs = [f"{stderr[i]:.4f}" for i in range(K)]
    tstat_strs = [f"{tstats[i]:.4f}" for i in range(K)]
    pval_strs = [f"{pvals[i]:.4f}" for i in range(K)]

    # Right-align each column to its maximum width
    # Ref: tarch_display.m:182-205 — right-alignment loop
    max_param_w = max(len(s) for s in param_strs)
    max_stderr_w = max(len(s) for s in stderr_strs)
    max_tstat_w = max(len(s) for s in tstat_strs)
    max_pval_w = max(len(s) for s in pval_strs)

    param_strs = [s.rjust(max_param_w) for s in param_strs]
    stderr_strs = [s.rjust(max_stderr_w) for s in stderr_strs]
    tstat_strs = [s.rjust(max_tstat_w) for s in tstat_strs]
    pval_strs = [s.rjust(max_pval_w) for s in pval_strs]

    # Column labels
    # Ref: tarch_display.m:207
    labels = [" Parameters", "   Std. Err.", "     T-stat", "      P-val"]
    cols = [param_strs, stderr_strs, tstat_strs, pval_strs]

    # Ensure each label and its data column share the same width
    # Ref: tarch_display.m:209-219
    for i in range(4):
        label_w = len(labels[i])
        data_w = max(len(cols[i][j]) for j in range(K))
        max_w = max(label_w, data_w)
        labels[i] = labels[i].rjust(max_w)
        cols[i] = [s.rjust(max_w) for s in cols[i]]

    # ================================================================
    # BUILD VARIABLE NAMES
    # Ref: tarch_display.m:222-250
    # ================================================================
    variable_names: list[str] = []
    # Ref: tarch_display.m:223
    variable_names.append("omega")
    # Ref: tarch_display.m:225-228 — 1-indexed: alpha(1), alpha(2), ...
    for i in range(1, p + 1):
        variable_names.append(f"alpha({i})")
    # Ref: tarch_display.m:229-232 — gamma for asymmetric/threshold terms
    for i in range(1, o + 1):
        variable_names.append(f"gamma({i})")
    # Ref: tarch_display.m:233-236
    for i in range(1, q + 1):
        variable_names.append(f"beta({i})")
    # Ref: tarch_display.m:237-239 — distribution shape parameter
    if error_type_code > 1:
        variable_names.append("nu")
    # Ref: tarch_display.m:241-243 — skewness parameter
    if error_type_code == 4:
        variable_names.append("lambda")

    # Right-align variable names to the longest name
    # Ref: tarch_display.m:246-250
    max_var_name_len = max(len(name) for name in variable_names)
    variable_names = [name.rjust(max_var_name_len) for name in variable_names]

    # ================================================================
    # ASSEMBLE PARAMETER TABLE (outmat equivalent)
    # Ref: tarch_display.m:253-256
    # ================================================================

    # Header row: blank (variable-name width) + "  " + label columns
    # Ref: tarch_display.m:253 — strvcat(' ', variable_names)  header is ' '
    table_header = " " * max_var_name_len
    for i in range(4):
        table_header += "  " + labels[i]

    # Data rows: variable name + "  " + value columns
    table_rows: list[str] = [table_header]
    for j in range(K):
        row = variable_names[j]
        for i in range(4):
            row += "  " + cols[i][j]
        table_rows.append(row)

    # Print the parameter table
    # Ref: tarch_display.m:265 — disp(outmat)
    outmat_text = "\n".join(table_rows)
    print(outmat_text)

    # ================================================================
    # BUILD COMPLETE RETURN TEXT
    # Ref: tarch_display.m:259-263
    # text = strvcat(text2, outmat) in MATLAB — all header lines then table
    # ================================================================
    all_lines = header_lines + table_rows
    text = "\n".join(all_lines)

    return text, AIC, BIC
