"""
AIC/SBIC Information Criteria for ARMA model selection.

Computes the Akaike Information Criterion (AIC) and Schwartz/Bayesian
Information Criterion (SBIC) for an ARMA(P,Q) model as parameterized
in armaxfilter.

Migrated from MATLAB MFE Toolbox ``aicsbic.m`` (Revision 3, 10/19/2009)
by Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk).
"""

import numpy as np


def aicsbic(
    errors: np.ndarray,
    constant: int,
    p: np.ndarray | int | list,
    q: np.ndarray | int | list | None = None,
    x: np.ndarray | None = None,
) -> tuple[float, float]:
    """
    Compute the Akaike and Schwartz/Bayes Information Criteria for an
    ARMA(P,Q) as parameterized in armaxfilter.

    This is a helper for :func:`armaxfilter` and uses the same inputs
    ``constant``, ``p``, ``q`` and ``x``.  ``errors`` should be the
    residual vector returned from a call to :func:`armaxfilter` with
    the same parameter values.

    Parameters
    ----------
    errors : array_like
        A T-element vector of residuals from the ARMA regression.
    constant : {0, 1}
        1 to include a constant term in the parameter count, 0 to exclude.
    p : array_like
        Non-negative integer scalar or vector of AR lag orders included in
        the model.  Pass ``0`` or ``[]`` for no AR terms.
    q : array_like, optional
        Non-negative integer scalar or vector of MA lag orders included in
        the model.  Default is ``None`` (no MA terms).
    x : array_like, optional
        A T × K matrix of exogenous regressors.  Default is ``None``
        (no exogenous variables).

    Returns
    -------
    aic : float
        The Akaike Information Criterion,
        ``AIC = log(σ²) + 2·K / T``.
    sbic : float
        The Schwartz/Bayesian Information Criterion,
        ``SBIC = log(σ²) + K·log(T) / T``.

    Raises
    ------
    ValueError
        If any input fails validation (see Notes).

    Notes
    -----
    The variance estimate is ``σ² = errors'·errors / T`` where
    ``T = len(errors)``.  The total parameter count is

    ``K = constant + len(unique(p)) + len(unique(q)) + ncols(x)``

    AIC  = log(σ²) + 2·K / T
    SBIC = log(σ²) + K·log(T) / T

    See Also
    --------
    armaxfilter : ARMAX filter estimation driver.
    aichqcsbic : AIC/HQC/SBIC (adds Hannan-Quinn criterion).
    heterogeneousar : Heterogeneous autoregressive model.

    Examples
    --------
    Compute AIC and SBIC for an AR(1) model:

    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> errors = rng.standard_normal(200)
    >>> aic, sbic = aicsbic(errors, 1, np.array([1]))
    """
    # ------------------------------------------------------------------
    # Handle default arguments
    # Ref: aicsbic.m:43-48 — nargin-based default assignment in MATLAB.
    # In Python we use keyword defaults; None → empty array equivalent.
    # ------------------------------------------------------------------
    if q is None:
        q_raw: np.ndarray = np.array([], dtype=np.float64)
    else:
        q_raw = np.asarray(q, dtype=np.float64)

    # ------------------------------------------------------------------
    # Validate ERRORS
    # Ref: aicsbic.m:52-56
    # MATLAB: size(errors,2) > 1 || length(errors)==1 → reject
    # ------------------------------------------------------------------
    errors_arr: np.ndarray = np.asarray(errors, dtype=np.float64)
    if errors_arr.ndim == 0:
        # Scalar input — analogous to MATLAB length(errors)==1
        raise ValueError('ERRORS series must be a column vector.')
    elif errors_arr.ndim == 1:
        pass  # 1-D is acceptable; size check below
    elif errors_arr.ndim == 2:
        # Ref: aicsbic.m:52 — size(errors,2) > 1 rejects row vectors / matrices
        if errors_arr.shape[1] > 1:
            raise ValueError('ERRORS series must be a column vector.')
        # (T, 1) column vector → flatten to (T,)
        errors_arr = errors_arr.ravel()
    else:
        raise ValueError('ERRORS series must be a column vector.')

    if errors_arr.size == 0:
        # Ref: aicsbic.m:54-55
        raise ValueError('ERRORS is empty.')
    if errors_arr.size == 1:
        # Ref: aicsbic.m:52 — MATLAB length(errors)==1 rejects single-element
        raise ValueError('ERRORS series must be a column vector.')

    t_obs: int = errors_arr.shape[0]

    # ------------------------------------------------------------------
    # Validate P  (AR lag orders)
    # Ref: aicsbic.m:61-82
    # ------------------------------------------------------------------
    p_arr: np.ndarray = np.asarray(p, dtype=np.float64)

    # Ref: aicsbic.m:61-63 — transpose row → column; generalised to ravel
    if p_arr.ndim == 0:
        p_arr = p_arr.reshape(1)
    elif p_arr.ndim == 2:
        # Ref: aicsbic.m:67-68 — min(size(p))~=1 rejects true matrices
        if p_arr.shape[0] > 1 and p_arr.shape[1] > 1:
            raise ValueError('P must be a column vector of included lags')
        p_arr = p_arr.ravel()
    elif p_arr.ndim > 2:
        raise ValueError('P must be a column vector of included lags')

    # Ref: aicsbic.m:64-66 — empty p defaults to scalar zero
    if p_arr.size == 0:
        p_arr = np.array([0.0])

    # Ref: aicsbic.m:70-71 — non-negative integer check
    if np.any(p_arr < 0) or np.any(np.floor(p_arr) != p_arr):
        raise ValueError('P must contain non-negative integers only')

    p_int: np.ndarray = p_arr.astype(np.int64)
    max_p: int = int(p_int.max())

    # Ref: aicsbic.m:73-74 — max(p) >= T - max(p) ⇒ max(p) >= T/2
    if max_p >= (t_obs - max_p):
        raise ValueError('Too many lags in the AR.  max(P)<T/2')

    # Ref: aicsbic.m:77-78 — scalar zero means "no AR terms"
    if p_int.size == 1 and p_int[0] == 0:
        p_int = np.array([], dtype=np.int64)

    # Ref: aicsbic.m:80 — Original MATLAB checks ``unique(q)`` here, which
    # is a copy-paste bug (the error message reads "P …").  We correct
    # the intent and check p for duplicate lag orders instead.
    if np.unique(p_int).size != p_int.size:
        raise ValueError('P must contain at most one of each lag')

    # ------------------------------------------------------------------
    # Validate Q  (MA lag orders)
    # Ref: aicsbic.m:86-107
    # ------------------------------------------------------------------
    q_arr: np.ndarray = q_raw.copy()

    # Ref: aicsbic.m:86-88 — transpose row → column
    if q_arr.ndim == 0:
        q_arr = q_arr.reshape(1)
    elif q_arr.ndim == 2:
        # Ref: aicsbic.m:92-93 — min(size(q))~=1 rejects true matrices
        if q_arr.shape[0] > 1 and q_arr.shape[1] > 1:
            raise ValueError('Q must be a column vector of included lags')
        q_arr = q_arr.ravel()
    elif q_arr.ndim > 2:
        raise ValueError('Q must be a column vector of included lags')

    # Ref: aicsbic.m:89-90 — empty q defaults to scalar zero
    if q_arr.size == 0:
        q_arr = np.array([0.0])

    # Ref: aicsbic.m:95-96 — non-negative integer check
    if np.any(q_arr < 0) or np.any(np.floor(q_arr) != q_arr):
        raise ValueError('Q must contain non-negative integers only')

    q_int: np.ndarray = q_arr.astype(np.int64)
    max_q: int = int(q_int.max())

    # Ref: aicsbic.m:98-99 — max(q) >= T
    if max_q >= t_obs:
        raise ValueError('Too many lags in the AR.  max(Q)<T')

    # Ref: aicsbic.m:101-103 — scalar zero means "no MA terms"
    if q_int.size == 1 and q_int[0] == 0:
        q_int = np.array([], dtype=np.int64)

    # Ref: aicsbic.m:104-106 — Q uniqueness check
    if np.unique(q_int).size != q_int.size:
        raise ValueError('Q must contain at most one of each lag')

    # ------------------------------------------------------------------
    # Validate CONSTANT
    # Ref: aicsbic.m:111-113
    # ------------------------------------------------------------------
    if constant not in (0, 1):
        raise ValueError('CONSTANT must be 0 or 1')

    # ------------------------------------------------------------------
    # Determine number of exogenous regressors
    # Ref: aicsbic.m:122 — size(X, 2)
    # ------------------------------------------------------------------
    if x is None:
        num_exog: int = 0
    else:
        x_arr: np.ndarray = np.asarray(x, dtype=np.float64)
        if x_arr.ndim == 1:
            # Treat a 1-D array as a single exogenous variable
            num_exog = 1 if x_arr.size > 0 else 0
        elif x_arr.ndim == 2:
            num_exog = x_arr.shape[1] if x_arr.size > 0 else 0
        else:
            num_exog = 0

    # ------------------------------------------------------------------
    # AIC / SBIC computation
    # Ref: aicsbic.m:117-125
    # ------------------------------------------------------------------

    # Ref: aicsbic.m:118 — seregression = sqrt(errors' * errors / T)
    # Standard error of regression (RMSE)
    se_regression: float = float(np.sqrt(np.dot(errors_arr, errors_arr) / t_obs))

    # Ref: aicsbic.m:120-122 — parameter count
    # lp = length(unique(p)); lq = length(unique(q));
    lp: int = len(np.unique(p_int))
    lq: int = len(np.unique(q_int))
    k: int = int(constant) + lp + lq + num_exog

    # Ref: aicsbic.m:124 — AIC = log(seregression^2) + 2*K/T
    log_variance: float = float(np.log(se_regression ** 2))
    aic: float = log_variance + 2.0 * k / t_obs

    # Ref: aicsbic.m:125 — SBIC = log(seregression^2) + log(T)*K/T
    sbic: float = log_variance + float(np.log(t_obs)) * k / t_obs

    return aic, sbic
