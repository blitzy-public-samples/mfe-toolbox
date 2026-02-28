"""
EGARCH(P,O,Q) parameter validation.

Validates and normalizes all input parameters for EGARCH model estimation.
Ensures conformability of data, model orders, error distribution type,
starting values, and optimizer options before the estimation pipeline begins.

Migrated from: univariate/egarch_parameter_check.m (MFE Toolbox, Kevin Sheppard)
"""

import numpy as np


def egarch_parameter_check(
    data,
    p: int,
    o: int,
    q: int,
    error_type=None,
    startingvals=None,
    options=None,
) -> tuple:
    """
    Validate input parameters for EGARCH(P,O,Q) estimation.

    Checks that all inputs conform to the requirements of the EGARCH model
    estimation pipeline.  Invalid inputs raise ``ValueError`` with descriptive
    messages matching the original MATLAB error strings.

    Parameters
    ----------
    data : array_like
        T-element time series of (mean-zero) returns.  Must be a 1-D column
        vector with at least two observations.
    p : int
        Number of symmetric innovation (|z|) lags.  Non-negative integer;
        at least one of ``p`` or ``o`` must be strictly positive.
    o : int
        Number of asymmetric (leverage / z) lags.  Non-negative integer.
    q : int
        Number of GARCH (log-variance persistence) lags.  Non-negative integer.
    error_type : {None, str, int}, optional
        Error distribution type.  Accepted strings (case-insensitive):
        ``'NORMAL'`` (1), ``'STUDENTST'`` (2), ``'GED'`` (3), ``'SKEWT'`` (4).
        Integer codes 1–4 are also accepted.  Defaults to ``'NORMAL'`` (1)
        when ``None``.
    startingvals : array_like or None, optional
        Starting parameter vector for the optimizer.  When provided, its
        length must match the expected count:

        * NORMAL  (1): ``p + o + q + 1``
        * STUDENTST (2): ``p + o + q + 2``  (last element is nu > 2.1)
        * GED     (3): ``p + o + q + 2``  (last element is nu > 1.05)
        * SKEWT   (4): ``p + o + q + 3``  (penultimate is nu > 1.05)

        Additionally, the sum of beta parameters (persistence) must be < 1.
        Defaults to an empty array when ``None``.
    options : dict or None, optional
        Optimizer options dictionary compatible with
        ``scipy.optimize.minimize``.  Defaults to a standard set of SLSQP
        options when ``None``.

    Returns
    -------
    tuple
        ``(p, o, q, error_type, startingvals, options)`` — validated and
        potentially defaulted versions of the inputs.  ``error_type`` is
        always returned as an integer code (1–4).

    Raises
    ------
    ValueError
        If any input fails validation.
    """

    # ------------------------------------------------------------------
    # 1. DATA validation
    # Ref: egarch_parameter_check.m:26-30
    # ------------------------------------------------------------------
    data = np.asarray(data, dtype=np.float64)

    # Detect multi-column (matrix) or 0-dimensional (scalar) inputs
    if data.ndim > 1 and data.shape[1] > 1:
        # Ref: egarch_parameter_check.m:26 — size(data,2) > 1
        raise ValueError("data series must be a column vector.")
    if data.ndim == 0 or data.size == 1:
        # Ref: egarch_parameter_check.m:26 — length(data)==1 (scalar)
        raise ValueError("data series must be a column vector.")
    if data.size == 0:
        # Ref: egarch_parameter_check.m:28 — isempty(data)
        raise ValueError("data is empty.")

    # Check for NaN values in data
    if np.any(np.isnan(data)):
        raise ValueError("data contains NaN values.")

    # Flatten to guaranteed 1-D
    data = data.ravel()

    # ------------------------------------------------------------------
    # 2. Q validation
    # Ref: egarch_parameter_check.m:35-40
    # ------------------------------------------------------------------
    if q is None:
        # Ref: egarch_parameter_check.m:38-40 — default to 0 when empty
        q = 0
    else:
        if not np.isscalar(q):
            raise ValueError(
                "q must be a non-negative scalar."
            )
        q_val = int(q)
        if q_val < 0:
            # Ref: egarch_parameter_check.m:35-37 — any(q<0)
            raise ValueError(
                "q must be a non-negative scalar."
            )
        q = q_val

    # ------------------------------------------------------------------
    # 3. O validation
    # Ref: egarch_parameter_check.m:45-50
    # ------------------------------------------------------------------
    if o is None:
        # Ref: egarch_parameter_check.m:48-50 — default to 0 when empty
        o = 0
    else:
        if not np.isscalar(o):
            raise ValueError(
                "o must be a non-negative scalar."
            )
        o_val = int(o)
        if o_val < 0:
            # Ref: egarch_parameter_check.m:45-47 — any(o<0)
            raise ValueError(
                "o must be a non-negative scalar."
            )
        o = o_val

    # ------------------------------------------------------------------
    # 4. P validation
    # Ref: egarch_parameter_check.m:55-61
    # ------------------------------------------------------------------
    if p is None:
        raise ValueError("p must be a positive scalar.")
    if not np.isscalar(p):
        raise ValueError("p must be a positive scalar.")
    p_val = int(p)
    if p_val < 0:
        # Ref: egarch_parameter_check.m:55-57 — any(p<0)
        raise ValueError("p must be a positive scalar.")
    p = p_val

    # Ref: egarch_parameter_check.m:59-61 — at least one of p or o non-zero
    if p == 0 and o == 0:
        raise ValueError("One of p or o must be non-zero.")

    # ------------------------------------------------------------------
    # 5. ERROR_TYPE validation and mapping
    # Ref: egarch_parameter_check.m:65-83
    # ------------------------------------------------------------------
    if error_type is None:
        # Ref: egarch_parameter_check.m:67-69 — default to NORMAL
        error_type = 1
    elif isinstance(error_type, str):
        error_type_upper = error_type.upper().strip()
        _error_type_map = {
            "NORMAL": 1,
            "STUDENTST": 2,
            "GED": 3,
            "SKEWT": 4,
        }
        if error_type_upper not in _error_type_map:
            # Ref: egarch_parameter_check.m:81-83
            raise ValueError(
                "error_type must be a string and one of: "
                "'NORMAL', 'STUDENTST', 'GED' or 'SKEWT'."
            )
        error_type = _error_type_map[error_type_upper]
    elif isinstance(error_type, (int, float, np.integer, np.floating)):
        error_type = int(error_type)
        if error_type not in (1, 2, 3, 4):
            raise ValueError(
                "error_type must be one of: "
                "1 (NORMAL), 2 (STUDENTST), 3 (GED) or 4 (SKEWT)."
            )
    else:
        raise ValueError(
            "error_type must be a string and one of: "
            "'NORMAL', 'STUDENTST', 'GED' or 'SKEWT'."
        )

    # ------------------------------------------------------------------
    # 6. STARTINGVALS validation
    # Ref: egarch_parameter_check.m:88-123
    # ------------------------------------------------------------------
    if startingvals is not None:
        startingvals = np.asarray(startingvals, dtype=np.float64).ravel()

        if startingvals.size > 0:
            # Ref: egarch_parameter_check.m:90-116 — length checks per dist
            if error_type == 1:
                # NORMAL: p + o + q + 1 parameters
                # Ref: egarch_parameter_check.m:90-93
                expected_len = p + o + q + 1
                if startingvals.size != expected_len:
                    raise ValueError(
                        "startingvals must have p+o+q+1 elements "
                        "for NORMAL errors."
                    )
            elif error_type == 2:
                # STUDENTST: p + o + q + 2 (extra nu parameter)
                # Ref: egarch_parameter_check.m:94-100
                expected_len = p + o + q + 2
                if startingvals.size != expected_len:
                    raise ValueError(
                        "startingvals must have p+o+q+2 elements "
                        "for Student's-T errors."
                    )
                # Ref: egarch_parameter_check.m:99 — MATLAB 1-indexed
                # startingvals(p+o+q+2); Python 0-indexed: [p+o+q+1]
                if startingvals[p + o + q + 1] < 2.1:
                    raise ValueError(
                        "Nu must be greater than 2.1 when using "
                        "Students-T errors."
                    )
            elif error_type == 3:
                # GED: p + o + q + 2 (extra nu parameter)
                # Ref: egarch_parameter_check.m:101-107
                expected_len = p + o + q + 2
                if startingvals.size != expected_len:
                    raise ValueError(
                        "startingvals must have p+o+q+2 elements "
                        "for GED errors."
                    )
                # Ref: egarch_parameter_check.m:106 — MATLAB 1-indexed
                if startingvals[p + o + q + 1] < 1.05:
                    raise ValueError(
                        "Nu must be greater than 1 when using GED errors."
                    )
            elif error_type == 4:
                # SKEWT: p + o + q + 3 (extra nu + lambda)
                # Ref: egarch_parameter_check.m:108-115
                # Note: MATLAB comment says "GED" on line 109 but this is
                # actually the SKEWT branch (error_type==4) — original typo.
                expected_len = p + o + q + 3
                if startingvals.size != expected_len:
                    raise ValueError(
                        "startingvals must have p+o+q+3 elements "
                        "for Skewed-T errors."
                    )
                # Ref: egarch_parameter_check.m:113 — checks nu param
                # MATLAB 1-indexed: startingvals(p+o+q+2)
                # Python 0-indexed: startingvals[p+o+q+1]
                if startingvals[p + o + q + 1] < 1.05:
                    raise ValueError(
                        "Nu must be greater than 1 when using GED errors."
                    )

            # Ref: egarch_parameter_check.m:117-119
            # Beta persistence constraint: sum of betas must be < 1
            # MATLAB 1-indexed: startingvals(p+o+2 : p+o+q+1)
            # Python 0-indexed: startingvals[p+o+1 : p+o+q+1]
            beta_sum = np.sum(startingvals[p + o + 1 : p + o + q + 1])
            if beta_sum >= 1.0:
                raise ValueError(
                    "The sum of the betas must be less than 1."
                )
        else:
            # Empty array provided — treat as no starting values
            startingvals = np.array([], dtype=np.float64)
    else:
        # Ref: egarch_parameter_check.m:121-122 — nargin<6 → empty
        startingvals = np.array([], dtype=np.float64)

    # ------------------------------------------------------------------
    # 7. OPTIONS validation
    # Ref: egarch_parameter_check.m:125-150
    # ------------------------------------------------------------------
    if options is not None:
        if not isinstance(options, dict):
            # Ref: egarch_parameter_check.m:134
            raise ValueError(
                "options is not a valid minimization option structure."
            )
    else:
        # Ref: egarch_parameter_check.m:136-144
        # MATLAB defaults: optimset('fmincon') with TolFun=1e-5, TolX=1e-5,
        # Display='iter', MaxFunEvals=200*(2+p+q), MaxSQPIter=500,
        # Algorithm='active-set'.
        # Python equivalent for scipy.optimize.minimize(method='SLSQP'):
        options = {
            "ftol": 1e-5,
            "disp": True,
            "maxiter": 200 * (2 + p + q),
        }

    return p, o, q, error_type, startingvals, options
