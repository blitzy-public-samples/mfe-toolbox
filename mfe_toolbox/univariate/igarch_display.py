"""
Display IGARCH estimation results with parameter table, implied beta, and information criteria.

Migrated from univariate/igarch_display.m (MFE Toolbox, Version 4.0).

This module provides a formatted display of IGARCH/IAVARCH model estimation
results including parameter estimates, standard errors, t-statistics, p-values,
the implied final beta (unit-root constraint), and AIC/BIC information criteria.

The implied final beta is computed as ``1 - sum(alpha_i) - sum(beta_j)`` for
``j = 1, ..., q-1``, which enforces the IGARCH unit-root constraint. This
implied parameter is shown in the display but marked with dashes for standard
errors, t-statistics, and p-values since it is not freely estimated.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def igarch_display(
    parameters: np.ndarray,
    ll: float,
    vcv: np.ndarray,
    epsilon: np.ndarray,
    p: int,
    q: int,
    igarch_type: str = "GARCH",
    error_type: str = "NORMAL",
    constant: int = 1,
) -> tuple[str, float, float]:
    """
    Display IGARCH estimation results with parameter statistics and information criteria.

    Formats and prints a results table showing parameter estimates, standard errors,
    t-statistics, and p-values from IGARCH model estimation.  The implied final beta
    parameter (imposed by the IGARCH unit-root constraint) is displayed with dashes
    for its standard error, t-statistic, and p-value.

    Parameters
    ----------
    parameters : np.ndarray
        A 1-D array of length ``constant + p + q - 1 + extra_p`` containing the
        estimated parameters in the order:
        ``[omega, alpha(1), ..., alpha(p), beta(1), ..., beta(q-1), [nu, [lambda]]]``.
        ``omega`` is present only when ``constant == 1``.
    ll : float
        The log-likelihood value at the optimum.
    vcv : np.ndarray
        Square, positive-definite variance-covariance matrix of the parameter
        estimates (typically the inverse Hessian), with dimensions matching
        ``len(parameters)``.
    epsilon : np.ndarray
        A 1-D array of mean-zero residuals of length *T* used solely to
        determine the sample size for AIC/BIC normalisation.
    p : int
        Positive integer giving the number of symmetric innovation (ARCH)
        terms.
    q : int
        Non-negative integer giving the number of lagged conditional variance
        (GARCH) terms.  Use ``q = 0`` for a pure ARCH model.
    igarch_type : str, optional
        The variance process type:
        ``'GARCH'`` — model evolves in squares (default).
        ``'AVGARCH'`` — model evolves in absolute values.
    error_type : str, optional
        The error distribution assumed during estimation:
        ``'NORMAL'``    — Gaussian innovations (default, 0 extra parameters).
        ``'STUDENTST'`` — Student-t (1 extra parameter: *nu*).
        ``'GED'``       — Generalised Error Distribution (1 extra: *nu*).
        ``'SKEWT'``     — Skewed-t (2 extra: *nu* and *lambda*).
    constant : int, optional
        1 (default) to include a constant term (omega) in the model, 0 to
        exclude it.

    Returns
    -------
    text : str
        Multi-line string containing the full formatted results table.
    aic : float
        Akaike Information Criterion computed as ``-ll/T + 2*n/T``.
    bic : float
        Bayesian (Schwarz) Information Criterion computed as
        ``-ll/T + log(T)*n/T``.

    Raises
    ------
    ValueError
        If any input fails validation (non-real parameters, non-positive-definite
        VCV, incompatible dimensions, invalid ``igarch_type`` or ``error_type``,
        etc.).

    Notes
    -----
    * Ref: igarch_display.m — MATLAB code uses 1-indexed arrays; Python uses
      0-indexed.  All index adjustments documented inline.
    * The implied beta value is **not** an estimated parameter; it is derived
      from the IGARCH unit-root constraint and therefore has no standard error.
    * AIC and BIC are normalised by the sample size *T* (the length of
      ``epsilon``).
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    # Ensure parameters is a 1-D numpy array of real values.
    parameters = np.asarray(parameters, dtype=np.float64).ravel()
    if not np.all(np.isreal(parameters)):
        raise ValueError("PARAMETERS must be a real-valued array.")

    # ll must be a real scalar.
    if not np.isscalar(ll) or not np.isreal(ll):
        raise ValueError("LL must be a real scalar.")
    ll = float(ll)

    # vcv must be a square, positive-definite matrix compatible with parameters.
    vcv = np.asarray(vcv, dtype=np.float64)
    if vcv.ndim != 2 or vcv.shape[0] != vcv.shape[1]:
        raise ValueError(
            "VCV must be a square matrix compatible with PARAMETERS."
        )
    if vcv.shape[0] != len(parameters):
        raise ValueError(
            "VCV must be a square matrix compatible with PARAMETERS."
        )
    # Ref: igarch_display.m:80 — check positive definiteness via eigenvalues.
    min_eig = np.min(np.linalg.eigvalsh(vcv))
    if min_eig <= 0:
        raise ValueError(
            "VCV must be a square positive definite matrix compatible with PARAMETERS."
        )

    # epsilon (data) must be a real-valued 1-D array.
    epsilon = np.asarray(epsilon, dtype=np.float64).ravel()
    if not np.all(np.isreal(epsilon)):
        raise ValueError("EPSILON must be a real-valued array.")

    # p must be a positive integer.
    if not np.isscalar(p) or int(p) != p or p < 1:
        raise ValueError("P must be a positive scalar integer.")
    p = int(p)

    # q must be a non-negative integer.
    if not np.isscalar(q) or int(q) != q or q < 0:
        raise ValueError("Q must be a non-negative scalar integer.")
    q = int(q)

    # ------------------------------------------------------------------
    # Map error_type string → internal code and extra parameter count
    # Ref: igarch_display.m:99-113
    # ------------------------------------------------------------------
    error_type = error_type.upper() if isinstance(error_type, str) else ""
    _error_map: dict[str, tuple[int, int]] = {
        "NORMAL": (1, 0),
        "STUDENTST": (2, 1),
        "GED": (3, 1),
        "SKEWT": (4, 2),
    }
    if error_type not in _error_map:
        raise ValueError(
            "ERROR_TYPE must be one of 'NORMAL', 'STUDENTST', 'GED', or 'SKEWT'."
        )
    error_type_code, extra_p = _error_map[error_type]

    # ------------------------------------------------------------------
    # Map igarch_type string → internal code
    # Ref: igarch_display.m:116-117 — accepts 1 or 2; Python API uses strings.
    # 'AVGARCH' → 1 (absolute values), 'GARCH' → 2 (squares)
    # ------------------------------------------------------------------
    igarch_type = igarch_type.upper() if isinstance(igarch_type, str) else ""
    _type_map: dict[str, int] = {"AVGARCH": 1, "GARCH": 2}
    if igarch_type not in _type_map:
        raise ValueError("IGARCH_TYPE must be either 'GARCH' or 'AVGARCH'.")
    igarch_type_code = _type_map[igarch_type]

    # ------------------------------------------------------------------
    # Validate constant parameter
    # ------------------------------------------------------------------
    if constant not in (0, 1):
        raise ValueError("CONSTANT must be 0 or 1.")

    # ------------------------------------------------------------------
    # Verify parameter vector length
    # Ref: igarch_display.m:119-121
    # ------------------------------------------------------------------
    expected_len = constant + p + q - 1 + extra_p
    if len(parameters) != expected_len:
        raise ValueError(
            f"Size of PARAMETERS ({len(parameters)}) is not compatible with "
            f"P={p}, Q={q}, constant={constant}, error_type='{error_type}' "
            f"(expected {expected_len})."
        )

    # ------------------------------------------------------------------
    # Model name
    # Ref: igarch_display.m:128-132
    # ------------------------------------------------------------------
    if igarch_type_code == 1:
        model_name = f"IAVARCH({p},{q})"
    else:
        model_name = f"IGARCH({p},{q})"

    # ------------------------------------------------------------------
    # Compute standard errors, t-statistics, and p-values
    # Ref: igarch_display.m:134-140
    # ------------------------------------------------------------------
    stderr = np.sqrt(np.diag(vcv))
    tstats = parameters / stderr
    pvals = 2.0 - 2.0 * norm.cdf(np.abs(tstats))

    # ------------------------------------------------------------------
    # Compute the implied final beta (unit-root constraint)
    # Ref: igarch_display.m:142-160
    # finalBetaPos = constant + p + q  (MATLAB 1-indexed display position)
    # archParameters = parameters(constant+1 : constant+p+q-1) (MATLAB)
    #                = parameters[constant : constant+p+q-1]   (Python)
    # ------------------------------------------------------------------
    arch_parameters = parameters[constant: constant + p + q - 1]
    final_beta: float = 1.0 - float(np.sum(arch_parameters))

    # ------------------------------------------------------------------
    # Build display arrays — insert implied beta row
    # Ref: igarch_display.m:149-160
    # In the display, the implied beta occupies position (constant + p + q - 1)
    # in 0-indexed terms (i.e. right after the last free beta or last alpha
    # if q == 1).
    # ------------------------------------------------------------------
    n = len(parameters)
    # Ref: igarch_display.m:143 — finalBetaPos is constant+p+q in MATLAB
    # which is index (constant+p+q-1) in 0-based Python.
    insert_pos = constant + p + q - 1  # 0-based index for np.insert

    param_vals = list(parameters)
    stderr_vals = list(stderr)
    tstat_vals = list(tstats)
    pval_vals = list(pvals)

    # Insert implied beta into display lists.
    param_vals.insert(insert_pos, final_beta)
    stderr_vals.insert(insert_pos, None)  # type: ignore[arg-type]
    tstat_vals.insert(insert_pos, None)  # type: ignore[arg-type]
    pval_vals.insert(insert_pos, None)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # AIC / BIC
    # Ref: igarch_display.m:166-168
    # ------------------------------------------------------------------
    t_obs = len(epsilon)  # T = length(data)
    aic: float = -ll / t_obs + 2.0 * n / t_obs
    bic: float = float(-ll / t_obs + np.log(t_obs) * n / t_obs)

    # ------------------------------------------------------------------
    # Variable names
    # Ref: igarch_display.m:232-249
    # When constant==1, include omega; otherwise start directly with alpha.
    # The loop in MATLAB always writes omega at index 1 regardless of
    # constant — this is a MATLAB-side limitation; we correctly condition.
    # All q betas are listed (including the implied one).
    # ------------------------------------------------------------------
    variable_names: list[str] = []
    if constant == 1:
        variable_names.append("omega")
    for i in range(1, p + 1):
        variable_names.append(f"alpha({i})")
    for i in range(1, q + 1):
        variable_names.append(f"beta({i})")
    if error_type_code > 1:
        variable_names.append("nu")
    if error_type_code == 4:
        variable_names.append("lambda")

    # ------------------------------------------------------------------
    # Format numeric columns — right-aligned, 4 decimal places
    # Ref: igarch_display.m:134, 192-214
    # ------------------------------------------------------------------
    num_rows = len(variable_names)

    def _fmt_val(val: float | None, width: int) -> str:
        """Format a numeric value right-aligned, or dashes for implied params."""
        if val is None:
            return "-" * width
        return f"{val:.4f}".rjust(width)

    # First pass: determine column widths
    param_strs_raw = [f"{v:.4f}" for v in param_vals]
    stderr_strs_raw = [
        f"{v:.4f}" if v is not None else "" for v in stderr_vals
    ]
    tstat_strs_raw = [
        f"{v:.4f}" if v is not None else "" for v in tstat_vals
    ]
    pval_strs_raw = [
        f"{v:.4f}" if v is not None else "" for v in pval_vals
    ]

    labels = [" Parameters", "   Std. Err.", "     T-stat", "      P-val"]
    raw_cols = [param_strs_raw, stderr_strs_raw, tstat_strs_raw, pval_strs_raw]

    col_widths: list[int] = []
    for label, col in zip(labels, raw_cols):
        max_data_w = max((len(s) for s in col if s), default=0)
        col_widths.append(max(len(label), max_data_w))

    # Second pass: right-align everything
    formatted_labels = [lab.rjust(w) for lab, w in zip(labels, col_widths)]
    formatted_cols: list[list[str]] = []
    for col_raw, vals, w in zip(raw_cols, [param_vals, stderr_vals, tstat_vals, pval_vals], col_widths):
        formatted_cols.append([_fmt_val(v, w) for v in vals])

    # Right-pad / right-align variable names
    # Ref: igarch_display.m:252-256
    max_var_len = max(len(vn) for vn in variable_names)
    aligned_var_names = [vn.rjust(max_var_len) for vn in variable_names]

    # ------------------------------------------------------------------
    # Build header text
    # Ref: igarch_display.m:173-189
    # ------------------------------------------------------------------
    header_lines: list[str] = [
        " ",
        " ",
        "-" * 50,
        model_name,
        "-" * 50,
        " ",
        f"Loglikelihood: {ll:.2f}",
        f"AIC: {aic:.4f}",
        f"BIC: {bic:.4f}",
        " ",
    ]

    # ------------------------------------------------------------------
    # Build parameter table
    # Ref: igarch_display.m:259-262
    # Column header row, then one row per parameter.
    # ------------------------------------------------------------------
    sep = "  "  # two-space separator between columns

    # Column header row — variable-name column is blank, then label columns.
    table_header = " " * max_var_len
    for fl in formatted_labels:
        table_header += sep + fl

    table_rows: list[str] = [table_header]
    for row_idx in range(num_rows):
        row_str = aligned_var_names[row_idx]
        for col_idx in range(4):
            row_str += sep + formatted_cols[col_idx][row_idx]
        table_rows.append(row_str)

    # ------------------------------------------------------------------
    # Combine header and table into final text
    # Ref: igarch_display.m:265-269
    # ------------------------------------------------------------------
    all_lines = header_lines + table_rows
    text = "\n".join(all_lines)

    # ------------------------------------------------------------------
    # Print the output
    # Ref: igarch_display.m:187-189, 271
    # ------------------------------------------------------------------
    # Print header section
    for line in header_lines:
        print(line)
    # Print parameter table (outmat in MATLAB)
    for line in table_rows:
        print(line)

    return text, aic, bic
