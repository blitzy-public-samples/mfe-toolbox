"""
Akaike (AIC), Hannan-Quinn (HQC), and Schwarz/Bayes (SBIC) Information Criteria
for ARMA(P,Q) models as parameterized in armaxfilter.

This module provides the :func:`aichqcsbic` function that computes three standard
model-selection criteria from ARMA residuals and model specification parameters.

Migrated from MATLAB source: timeseries/aichqcsbic.m
    Revision: 3    Date: 10/19/2009
    Original author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np


def aichqcsbic(
    errors: np.ndarray,
    constant: int,
    p: np.ndarray,
    q: np.ndarray,
    X: Union[int, np.ndarray, None] = None,
) -> Tuple[float, float, float]:
    """
    Compute Akaike (AIC), Hannan-Quinn (HQC), and Schwarz/Bayes (SBIC)
    Information Criteria for an ARMA(P,Q) model.

    This function is a helper for ``armaxfilter`` and uses the same input
    conventions for ``constant``, ``p``, ``q``, and ``X``.  ``errors`` should
    be the residual vector returned from a call to ``armaxfilter`` with the
    corresponding parameter values.

    Parameters
    ----------
    errors : np.ndarray
        A T-element 1-D array (or T×1 column vector) of residuals from the
        ARMA regression.  Must contain more than one observation.
    constant : int
        Scalar indicator: 1 to include a constant in the parameter count,
        0 to exclude.
    p : np.ndarray
        Non-negative integer array of AR lag orders included in the model.
        Pass ``np.array([])`` or ``np.array([0])`` for no AR terms.
    q : np.ndarray
        Non-negative integer array of MA lag orders included in the model.
        Pass ``np.array([])`` or ``np.array([0])`` for no MA terms.
    X : int, np.ndarray, or None, optional
        A T-by-K matrix of exogenous regressors whose column count is added
        to the parameter count *K*.  ``None`` (default) or ``0`` indicates
        no exogenous variables.

    Returns
    -------
    tuple of (float, float, float)
        ``(aic, hqc, sbic)`` — the three information criteria computed as:

        .. math::

            \\sigma^2 &= \\frac{1}{T} \\sum_{t=1}^{T} e_t^2

            \\text{AIC}  &= \\ln(\\sigma^2) + \\frac{2K}{T}

            \\text{HQC}  &= \\ln(\\sigma^2) + \\frac{2K \\ln(\\ln T)}{T}

            \\text{SBIC} &= \\ln(\\sigma^2) + \\frac{K \\ln T}{T}

        where *K* = ``constant`` + number of unique AR lags + number of
        unique MA lags + number of columns in ``X``.

    Raises
    ------
    ValueError
        If any input fails the validation checks described below.

    Notes
    -----
    The parameter count *K* uses ``len(np.unique(p))`` and
    ``len(np.unique(q))`` so that only *distinct* lag indices contribute.
    Scalar-zero or empty ``p`` / ``q`` inputs contribute zero to *K*.

    Examples
    --------
    >>> import numpy as np
    >>> errors = np.random.default_rng(42).standard_normal(200)
    >>> aic, hqc, sbic = aichqcsbic(errors, 1, np.array([1, 2]), np.array([1]))

    See Also
    --------
    mfe_toolbox.timeseries.armaxfilter : ARMAX estimation filter.
    mfe_toolbox.timeseries.heterogeneousar : Heterogeneous AR model.
    mfe_toolbox.timeseries.aicsbic : Simplified AIC/SBIC variant.
    """

    # ==================================================================
    # Input validation — mirrors aichqcsbic.m lines 38-115
    # ==================================================================

    # --- errors -----------------------------------------------------------
    # Ref: aichqcsbic.m:53-57 — must be a column vector, not scalar, not empty
    errors = np.asarray(errors, dtype=np.float64)

    if errors.ndim >= 2 and errors.shape[1] > 1:
        # Ref: aichqcsbic.m:53 — size(errors,2) > 1 → not a column vector
        raise ValueError("ERRORS series must be a column vector.")

    # Flatten (T, 1) → (T,) for downstream dot-product
    if errors.ndim >= 2:
        errors = errors.ravel()

    if errors.size == 0:
        # Ref: aichqcsbic.m:55-56
        raise ValueError("ERRORS is empty.")

    if errors.ndim == 0 or errors.size == 1:
        # Ref: aichqcsbic.m:53 — length(errors)==1
        raise ValueError("ERRORS series must be a column vector.")

    T: int = errors.shape[0]

    # --- p (AR lags) ------------------------------------------------------
    # Ref: aichqcsbic.m:62-82
    p = np.asarray(p, dtype=np.float64)

    # Reject true 2-D matrices (min(size(p))~=1 check)
    # Ref: aichqcsbic.m:68-69
    if p.ndim > 1 and min(p.shape) > 1:
        raise ValueError("P must be a column vector of included lags")

    p = p.ravel()

    # Ref: aichqcsbic.m:65-67 — empty → temporary 0
    if p.size == 0:
        p = np.array([0.0])

    # Ref: aichqcsbic.m:71-73 — non-negative integers
    if np.any(p < 0) or np.any(np.floor(p) != p):
        raise ValueError("P must contain non-negative integers only")

    # Ref: aichqcsbic.m:74-76 — max(P) >= T - max(P) → too many lags
    if np.max(p) >= (T - np.max(p)):
        raise ValueError("Too many lags in the AR.  max(P)<T/2")

    # Ref: aichqcsbic.m:77-79 — scalar 0 → empty (no AR terms)
    if p.size == 1 and p[0] == 0:
        p = np.array([], dtype=np.float64)

    # Ref: aichqcsbic.m:80-82 — duplicate lags not allowed
    if np.unique(p).size != p.size:
        raise ValueError("P must contain at most one of each lag")

    # --- q (MA lags) ------------------------------------------------------
    # Ref: aichqcsbic.m:86-106  (same pattern as p)
    q = np.asarray(q, dtype=np.float64)

    if q.ndim > 1 and min(q.shape) > 1:
        raise ValueError("Q must be a column vector of included lags")

    q = q.ravel()

    # Ref: aichqcsbic.m:89-91 — empty → temporary 0
    if q.size == 0:
        q = np.array([0.0])

    # Ref: aichqcsbic.m:95-97 — non-negative integers
    if np.any(q < 0) or np.any(np.floor(q) != q):
        raise ValueError("Q must contain non-negative integers only")

    # Ref: aichqcsbic.m:98-100 — max(Q) >= T
    if np.max(q) >= T:
        raise ValueError("Too many lags in the AR.  max(Q)<T")

    # Ref: aichqcsbic.m:101-103 — scalar 0 → empty (no MA terms)
    if q.size == 1 and q[0] == 0:
        q = np.array([], dtype=np.float64)

    # Ref: aichqcsbic.m:104-106 — duplicate lags not allowed
    if np.unique(q).size != q.size:
        raise ValueError("Q must contain at most one of each lag")

    # --- constant ---------------------------------------------------------
    # Ref: aichqcsbic.m:110-112
    if constant not in (0, 1):
        raise ValueError("CONSTANT must be 0 or 1")

    # --- X (exogenous regressors) -----------------------------------------
    # Ref: aichqcsbic.m:121 — only size(X,2) is used
    if X is None:
        ncols_x: int = 0
    elif isinstance(X, (int, float)):
        # Allow callers to pass 0 or an integer for "no exogenous variables"
        ncols_x = 0
    elif isinstance(X, np.ndarray):
        if X.size == 0:
            ncols_x = 0
        elif X.ndim == 1:
            # Treat a 1-D array as a single column of regressors
            ncols_x = 1
        else:
            ncols_x = X.shape[1]
    else:
        ncols_x = 0

    # ==================================================================
    # Compute information criteria — mirrors aichqcsbic.m lines 116-125
    # ==================================================================

    # Ref: aichqcsbic.m:117 — σ² = errors' * errors / T
    sigma_sq: float = float(np.dot(errors, errors) / T)

    # Ref: aichqcsbic.m:119-121
    lp: int = len(np.unique(p)) if p.size > 0 else 0
    lq: int = len(np.unique(q)) if q.size > 0 else 0
    K: int = int(constant) + lp + lq + ncols_x

    # Ref: aichqcsbic.m:123 — AIC = log(σ²) + 2K / T
    log_sigma_sq: float = float(np.log(sigma_sq))
    aic: float = log_sigma_sq + 2.0 * K / T

    # Ref: aichqcsbic.m:124 — HQC = log(σ²) + 2K * log(log(T)) / T
    hqc: float = log_sigma_sq + 2.0 * K * float(np.log(np.log(T))) / T

    # Ref: aichqcsbic.m:125 — SBIC = log(σ²) + K * log(T) / T
    sbic: float = log_sigma_sq + K * float(np.log(T)) / T

    return (aic, hqc, sbic)
