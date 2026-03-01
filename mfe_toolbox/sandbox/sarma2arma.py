"""
SARMA-to-ARMA polynomial conversion.

Migrated from sandbox/sarma2arma.m (MFE Toolbox Version 4.0).

This module converts a SARMA (Seasonal ARMA) parameter vector into its
equivalent non-seasonal ARMA representation by expanding seasonal lag
polynomials via convolution.  The main function ``sarma2arma`` partitions
the input parameter vector into AR and MA sections, delegates each to the
private helper ``_convert_sar_to_ar``, and returns the combined non-seasonal
parameter vector together with the corresponding lag indices.

Functions
---------
sarma2arma
    Partition a SARMA parameter vector and expand seasonal components to
    produce a canonical ARMA representation.
_convert_sar_to_ar
    Build a lag polynomial from non-seasonal and seasonal AR (or MA)
    parameters, convolve the seasonal kernels, and extract the nonzero
    coefficients and their lag positions.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Helper (private) — converts seasonal AR/MA parameters to non-seasonal form
# ---------------------------------------------------------------------------

def _convert_sar_to_ar(
    parameters: np.ndarray,
    p: np.ndarray,
    sp: np.ndarray,
    s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert seasonal AR (or MA) parameters to non-seasonal lag form.

    Builds a lag polynomial ``(1 - a_1 B^{p_1} - a_2 B^{p_2} - ...)`` from
    the non-seasonal lag indices in *p*, then iteratively convolves with each
    seasonal factor ``(1 - Φ_1 B^S - Φ_2 B^{2S} - ...)``.  The resulting
    polynomial's nonzero coefficients (with the constant term removed) are
    returned along with their lag positions.

    Parameters
    ----------
    parameters : np.ndarray
        1-D array of concatenated non-seasonal + seasonal coefficients.
        The first ``len(p)`` elements correspond to non-seasonal lags; the
        remaining elements are consumed in order by each seasonal block
        whose order is given by the corresponding entry of *sp*.
    p : np.ndarray
        1-D integer array of non-seasonal lag indices (e.g. ``[1, 2, 3]``
        for an AR(3) process).  May be empty (length 0) if there are no
        non-seasonal lags.
    sp : np.ndarray
        1-D integer array of seasonal AR (or MA) orders, one per seasonal
        component.  ``sp[i]`` is the number of seasonal lag terms for the
        *i*-th seasonal factor.
    s : np.ndarray
        1-D integer array of seasonal period lengths (e.g. ``12`` for
        monthly data with annual seasonality).  ``s[i]`` is the period
        associated with the *i*-th seasonal factor.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        parameters_out : np.ndarray
            1-D array of the extracted nonzero (non-seasonal-equivalent)
            coefficients from the expanded polynomial.
        lags : np.ndarray
            1-D integer array of the corresponding lag positions (1-based
            lag values matching the MATLAB convention: lag 1 means B^1,
            lag 2 means B^2, etc.).

    Notes
    -----
    Ref: sandbox/sarma2arma.m lines 14–38 (``convertSARtoAR`` sub-function).
    """

    # Ref: sarma2arma.m:15 — parameter read position
    offset: int = 0

    # ------------------------------------------------------------------
    # Build the initial non-seasonal lag polynomial
    # Ref: sarma2arma.m:16-24
    # ------------------------------------------------------------------
    if p is None or len(p) == 0:
        # Ref: sarma2arma.m:16-17 — if isempty(p), poly = 1
        poly = np.array([1.0])
    else:
        p_int = np.asarray(p, dtype=int)
        # Ref: sarma2arma.m:19 — zeros(1, max(p)+1)
        lag_poly = np.zeros(int(np.max(p_int)) + 1)
        # Ref: sarma2arma.m:20 — lagPoly(1) = 1 (MATLAB 1-indexed → Python 0-indexed)
        lag_poly[0] = 1.0
        # Ref: sarma2arma.m:21 — lagPoly(p+1) = -parameters(offset+1:offset+length(p))
        # In MATLAB p contains lag values [1,2,...]; lagPoly(p+1) places at
        # MATLAB 1-indexed positions p+1 which map to Python 0-indexed
        # positions p.  So lag_poly[p_int] addresses the correct lag slots.
        lag_poly[p_int] = -parameters[offset:offset + len(p_int)]
        # Ref: sarma2arma.m:22
        offset += len(p_int)
        poly = lag_poly.copy()

    # ------------------------------------------------------------------
    # Iteratively expand seasonal factors via convolution
    # Ref: sarma2arma.m:25-35
    # ------------------------------------------------------------------
    for i in range(len(s)):
        # Ref: sarma2arma.m:26 — p = Sp(i)  (seasonal order)
        sp_i: int = int(sp[i])
        # Ref: sarma2arma.m:27 — sLag = S(i)  (seasonal period)
        s_lag: int = int(s[i])

        if sp_i > 0:
            # Ref: sarma2arma.m:29 — zeros(1, 1+sLag*p)
            lag_poly = np.zeros(1 + s_lag * sp_i)
            # Ref: sarma2arma.m:30 — lagPoly(1) = 1
            lag_poly[0] = 1.0

            # Ref: sarma2arma.m:31 — lagPoly(sLag+1:sLag:(sLag*p+1))
            # MATLAB 1-indexed: positions sLag+1, 2*sLag+1, ..., sp_i*sLag+1
            # Python 0-indexed: positions sLag, 2*sLag, ..., sp_i*sLag
            seasonal_indices = np.arange(s_lag, s_lag * sp_i + 1, s_lag)
            lag_poly[seasonal_indices] = -parameters[offset:offset + sp_i]

            # Ref: sarma2arma.m:32
            offset += sp_i

            # Ref: sarma2arma.m:33 — poly = conv(poly, lagPoly)
            poly = np.convolve(poly, lag_poly)

    # ------------------------------------------------------------------
    # Extract nonzero coefficients and lag positions
    # Ref: sarma2arma.m:36-38
    # ------------------------------------------------------------------
    # Ref: sarma2arma.m:36 — poly(2:length(poly))  removes constant term
    # (MATLAB index 1).  In Python: poly[1:] removes index 0.
    poly = poly[1:]

    # Ref: sarma2arma.m:37 — lags = find(poly~=0)
    # MATLAB find() returns 1-indexed positions in the truncated polynomial
    # whose positional index *is* the lag value (since position 1 = lag 1).
    # In Python np.nonzero returns 0-indexed positions, so we add 1.
    nonzero_idx = np.nonzero(poly)[0]
    lags = nonzero_idx + 1  # 0-indexed position → 1-based lag value

    # Ref: sarma2arma.m:38 — parameters = -poly(lags)
    # Negate coefficients (undo the "-" convention used when building the
    # lag polynomial).  Use 0-indexed *nonzero_idx* to access the truncated
    # array, not the 1-based *lags*.
    parameters_out = -poly[nonzero_idx]

    return parameters_out, lags


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def sarma2arma(
    parameters: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    seasonal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partition a SARMA parameter vector and expand to ARMA form.

    Given a parameter vector ordered as
    ``[a_1, ..., a_{len(p)}, Φ_{1,1}, ..., Φ_{K,Sp_K},
      b_1, ..., b_{len(q)}, Θ_{1,1}, ..., Θ_{K,Sq_K}]``
    and a *seasonal* specification matrix, this function expands the
    seasonal AR and MA components into their non-seasonal equivalents
    by polynomial convolution.

    Parameters
    ----------
    parameters : np.ndarray
        1-D array of concatenated AR + seasonal-AR + MA + seasonal-MA
        coefficients.  Length must equal
        ``len(p) + sum(Sp) + len(q) + sum(Sq)``
        where ``Sp``, ``Sq`` are derived from *seasonal*.
    p : np.ndarray
        1-D integer array of non-seasonal AR lag indices
        (e.g. ``[1, 2]`` for an AR(2) process).  May be empty.
    q : np.ndarray
        1-D integer array of non-seasonal MA lag indices
        (e.g. ``[1]`` for an MA(1) process).  May be empty.
    seasonal : np.ndarray
        2-D array of shape ``(K, 4)`` where *K* is the number of
        seasonal components.  Column layout (MATLAB convention, mapped
        to 0-indexed Python columns):

        * Column 0 (MATLAB col 1): seasonal AR orders ``Sp``
        * Column 1 (MATLAB col 2): *unused* — reserved
        * Column 2 (MATLAB col 3): seasonal MA orders ``Sq``
        * Column 3 (MATLAB col 4): seasonal period lengths ``S``

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        parameters_out : np.ndarray
            1-D array of concatenated non-seasonal-equivalent AR and MA
            coefficients produced by polynomial expansion.
        ar_p : np.ndarray
            1-D integer array of AR lag positions (1-based lag values).
        ma_q : np.ndarray
            1-D integer array of MA lag positions (1-based lag values).

    Raises
    ------
    ValueError
        If *parameters* is too short for the specified lag structure, or
        if *seasonal* does not have at least 4 columns.
    IndexError
        If *seasonal* column indexing fails due to incorrect shape.

    Notes
    -----
    Ref: sandbox/sarma2arma.m lines 1–12.

    The MATLAB source uses a 1-indexed ``seasonal`` matrix with columns
    ``(:,1)``, ``(:,3)``, ``(:,4)``.  In Python these become columns
    ``[:, 0]``, ``[:, 2]``, ``[:, 3]`` (0-indexed).

    Examples
    --------
    >>> import numpy as np
    >>> params = np.array([0.5, -0.3, 0.8, 0.4, -0.2, 0.6])
    >>> p = np.array([1, 2])
    >>> q = np.array([1])
    >>> seasonal = np.array([[1, 0, 1, 12]])
    >>> out, ar_lags, ma_lags = sarma2arma(params, p, q, seasonal)
    """

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    seasonal = np.asarray(seasonal, dtype=float)
    if seasonal.ndim == 1:
        # Allow a single seasonal row passed as a 1-D array
        seasonal = seasonal.reshape(1, -1)
    if seasonal.shape[1] < 4:
        raise ValueError(
            "seasonal must have at least 4 columns; "
            f"got shape {seasonal.shape}"
        )

    parameters = np.asarray(parameters, dtype=float).ravel()
    p_arr = np.asarray(p, dtype=float).ravel() if p is not None else np.array([], dtype=float)
    q_arr = np.asarray(q, dtype=float).ravel() if q is not None else np.array([], dtype=float)

    # ------------------------------------------------------------------
    # Extract seasonal specification columns
    # Ref: sarma2arma.m:4-6 — MATLAB columns 1,3,4 → Python columns 0,2,3
    # ------------------------------------------------------------------
    # Ref: sarma2arma.m:4 — Sp = seasonal(:,1)  (seasonal AR orders)
    sp = seasonal[:, 0]
    # Ref: sarma2arma.m:5 — Sq = seasonal(:,3)  (seasonal MA orders)
    sq = seasonal[:, 2]
    # Ref: sarma2arma.m:6 — S  = seasonal(:,4)  (seasonal period lengths)
    s_vec = seasonal[:, 3]

    # ------------------------------------------------------------------
    # Partition the parameter vector into AR and MA sections
    # ------------------------------------------------------------------
    # Ref: sarma2arma.m:8 — arParamters = parameters(1:length(p)+sum(Sp))
    # MATLAB 1-indexed slice 1:N → Python 0-indexed slice :N
    p_len: int = len(p_arr)
    sp_total: int = int(np.sum(sp))
    ar_end: int = p_len + sp_total
    ar_params = parameters[:ar_end]

    # Ref: sarma2arma.m:9 — [arParameters, arP] = convertSARtoAR(...)
    ar_parameters, ar_p = _convert_sar_to_ar(ar_params, p_arr, sp, s_vec)

    # Ref: sarma2arma.m:10 — maParamters = parameters(length(p)+sum(Sp)+1:
    #     length(p)+sum(Sp)+length(q)+sum(Sq))
    # MATLAB 1-indexed range start → Python 0-indexed start = ar_end
    q_len: int = len(q_arr)
    sq_total: int = int(np.sum(sq))
    ma_start: int = ar_end
    ma_end: int = ar_end + q_len + sq_total
    ma_params = parameters[ma_start:ma_end]

    # Ref: sarma2arma.m:11 — [maParameters, maQ] = convertSARtoAR(...)
    ma_parameters, ma_q = _convert_sar_to_ar(ma_params, q_arr, sq, s_vec)

    # Ref: sarma2arma.m:12 — parameters = [arParameters maParameters]'
    # In MATLAB the transpose (') makes a column vector; in Python we keep
    # a 1-D array as the canonical representation.
    parameters_out = np.concatenate([ar_parameters, ma_parameters])

    return parameters_out, ar_p, ma_q
