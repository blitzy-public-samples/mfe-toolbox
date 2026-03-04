"""
DCC inference objective function for robust VCV computation.

This module provides the :func:`dcc_inference_objective` function, which
computes the "as if" objective used in the third stage of the DCC/ADCC
estimation procedure.  It is called by the DCC driver during inference
(via ``gradient_2sided`` and ``hessian_2sided_nrows``) to evaluate the
correlation intercept parameters.  This function is NOT used (and cannot
be used) to estimate model parameters directly.

The function reconstructs per-series conditional variances using the
concatenated GARCH parameter vector and univariate estimation results,
standardises the covariance data, and then evaluates a sum-of-squared-
errors objective for the correlation intercept (and optionally for the
asymmetric intercept in ADCC models).

Migrated from ``multivariate/dcc_inference_objective.m`` (80 lines) in
the MFE Toolbox (Version 4.0, Kevin Sheppard, University of Oxford).

Called by:
    - ``dcc.m`` during 3-stage inference via ``gradient_2sided`` and
      ``hessian_2sided_nrows``

Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 1    Date: 4/13/2012

Python migration: Blitzy Platform — MATLAB-to-Python 3.12 migration
"""

import numpy as np

from mfe_toolbox.multivariate.dcc_reconstruct_variance import dcc_reconstruct_variance
from mfe_toolbox.utility.corr_ivech import corr_ivech
from mfe_toolbox.utility.ivech import ivech


def dcc_inference_objective(
    parameters: np.ndarray,
    data: np.ndarray,
    data_asym: np.ndarray,
    m: int,
    l: int,
    n: int,
    univariate: list,
) -> tuple[float, np.ndarray]:
    """Compute the DCC/ADCC "as if" inference objective.

    Evaluates a sum-of-squared-errors objective over the correlation
    intercept parameters for robust VCV computation in the three-stage
    DCC estimation procedure.  The function:

    1. Parses GARCH parameters and intercept matrices from a combined
       parameter vector.
    2. Reconstructs per-series conditional variances via
       :func:`dcc_reconstruct_variance`.
    3. Standardises the K×K×T covariance data by volatilities.
    4. Computes the correlation intercept objective (upper-triangle
       off-diagonal pairs).
    5. Optionally computes the asymmetric intercept objective for ADCC
       models (when ``l > 0``).

    Parameters
    ----------
    parameters : np.ndarray
        1-D combined parameter vector laid out as::

            [garch_params, corr_vech(R), vech(N)]

        where ``garch_params`` has length ``sum(p_i + o_i + q_i + 1)``
        over all *K* series, ``corr_vech(R)`` has ``K*(K-1)/2``
        off-diagonal correlation elements, and ``vech(N)`` (present only
        when ``l > 0``) has ``K*(K+1)/2`` symmetric-matrix elements.
    data : np.ndarray
        K × K × T array of covariance estimators (e.g. realized
        covariance) or outer products of residuals.
    data_asym : np.ndarray
        K × K × T array of asymmetric covariance estimators.  Required
        when ``l > 0``; may be an array of zeros otherwise.
    m : int
        Order of symmetric innovations in the DCC model.  Unused in this
        function but retained for API parity with the MATLAB interface.
    l : int
        Order of asymmetric innovations in the ADCC model.  When ``l > 0``
        the asymmetric intercept matrix *N* is extracted from *parameters*
        and an additional asymmetric objective term is computed.
    n : int
        Order of lagged correlation in the DCC model.  Unused in this
        function but retained for API parity.
    univariate : list[dict]
        List of *K* dictionaries, one per series, each containing the
        first-stage univariate TARCH estimation results with at minimum
        the keys ``'p'``, ``'o'``, ``'q'`` (GARCH orders) plus any
        additional fields required by :func:`dcc_reconstruct_variance`
        (``'m'``, ``'T'``, ``'fdata'``, ``'fIdata'``, ``'back_cast'``,
        ``'tarch_type'``).

    Returns
    -------
    obj : float
        Scalar total "as if" objective (sum of per-observation objectives).
    objs : np.ndarray
        1-D array of length *T* containing per-observation objectives.

    See Also
    --------
    mfe_toolbox.multivariate.dcc : DCC model driver.
    mfe_toolbox.multivariate.ccc_mvgarch : CCC-MVGARCH model driver.
    mfe_toolbox.multivariate.dcc_reconstruct_variance :
        Reconstructs per-series conditional variances.
    mfe_toolbox.utility.corr_ivech : Inverse correlation half-vectorization.
    mfe_toolbox.utility.ivech : Inverse half-vectorization.

    Notes
    -----
    This is a faithful translation of ``dcc_inference_objective.m``.
    Every index change, struct-to-dict conversion, and outer-product
    translation is documented with inline comments referencing the
    original MATLAB source line numbers.

    The ``m`` and ``n`` parameters are accepted but not used in this
    function body, matching the original MATLAB source where they appear
    in the function signature but are suppressed via ``%#ok<INUSL>``.
    """
    # ------------------------------------------------------------------
    # Task 2.1: Extract dimension info
    # Ref: dcc_inference_objective.m:32 — [k,~,T] = size(data);
    # MATLAB size(data) on a K×K×T 3-D array returns [K, K, T].
    # ------------------------------------------------------------------
    k = data.shape[0]   # number of series
    T = data.shape[2]   # number of time observations

    # ------------------------------------------------------------------
    # Task 2.2: Parse GARCH parameters from the combined vector
    # Ref: dcc_inference_objective.m:34-39
    # Each series contributes p+o+q+1 parameters (omega, ARCH, threshold,
    # GARCH coefficients).
    # CRITICAL: MATLAB cell indexing univariate{i} → Python list[i]
    # CRITICAL: MATLAB struct field u.p → Python dict key u['p']
    # CRITICAL: MATLAB 1-indexed for i=1:k → Python for i in range(k)
    # ------------------------------------------------------------------
    count = 0
    for i in range(k):
        u = univariate[i]
        count += u['p'] + u['o'] + u['q'] + 1
    garch_parameters = parameters[:count]
    offset = count

    # ------------------------------------------------------------------
    # Task 2.3: Extract correlation intercept R
    # Ref: dcc_inference_objective.m:41-44
    # R is a K×K symmetric correlation matrix with ones on diagonal,
    # reconstructed from K*(K-1)/2 off-diagonal elements.
    # CRITICAL: MATLAB parameters(offset + (1:count)) is 1-indexed
    #   → Python parameters[offset:offset + count] is 0-indexed slice
    # ------------------------------------------------------------------
    count = k * (k - 1) // 2
    R = corr_ivech(parameters[offset:offset + count])
    offset += count

    # ------------------------------------------------------------------
    # Task 2.4: Conditionally extract asymmetric intercept N
    # Ref: dcc_inference_objective.m:45-48
    # Only present for ADCC models (l > 0).  N is a K×K symmetric matrix
    # reconstructed from K*(K+1)/2 elements.
    # ------------------------------------------------------------------
    N = None  # initialise for clarity
    if l > 0:
        count = k * (k + 1) // 2
        N = ivech(parameters[offset:offset + count])
        # offset is not updated further as N is the last block

    # ------------------------------------------------------------------
    # Task 2.5: Reconstruct per-series conditional variances
    # Ref: dcc_inference_objective.m:50
    # H is a T×K matrix of conditional variances.
    # ------------------------------------------------------------------
    H = dcc_reconstruct_variance(garch_parameters, univariate)

    # ------------------------------------------------------------------
    # Task 2.6: Standardise data by volatilities
    # Ref: dcc_inference_objective.m:51-57
    # For each time observation t, compute the K-vector of volatilities
    # h = sqrt(H(t,:)), form the K×K outer product hh = h'*h, and
    # element-wise divide the covariance slices by hh.
    # CRITICAL: MATLAB h'*h is an outer product (column × row) for a
    #   1×K row vector h.  In Python: np.outer(h, h).
    # CRITICAL: MATLAB 1-indexed loop for t=1:T → Python for t in range(T)
    # ------------------------------------------------------------------
    std_data = np.zeros((k, k, T))
    std_data_asym = np.zeros((k, k, T))
    for t in range(T):
        h = np.sqrt(H[t, :])
        # Ref: dcc_inference_objective.m:55 — h'*h outer product
        hh = np.outer(h, h)
        std_data[:, :, t] = data[:, :, t] / hh
        std_data_asym[:, :, t] = data_asym[:, :, t] / hh

    # ------------------------------------------------------------------
    # Task 2.7: Compute correlation intercept objective
    # Ref: dcc_inference_objective.m:59-67
    # scales is a K-vector of the diagonal of mean(stdData, 3) — i.e.
    # the time-averaged standardised variance for each series.
    # The objective loops over upper-triangle off-diagonal pairs (j, i)
    # with j < i, computing scaled squared deviations from R.
    # CRITICAL: MATLAB diag(mean(X,3)) → np.diag(np.mean(X, axis=2))
    # CRITICAL: MATLAB squeeze(stdData(i,j,:)) → std_data[i, j, :] in
    #   Python (slicing a 3-D array along the last axis yields 1-D).
    # CRITICAL: MATLAB for j=1:k-1, for i=j+1:k (1-indexed) →
    #   Python for j in range(k-1), for i in range(j+1, k) (0-indexed)
    # ------------------------------------------------------------------
    scales = np.diag(np.mean(std_data, axis=2))
    objs = np.zeros(T)
    for j in range(k - 1):        # Columns  (Ref: m:61 — j=1:k-1)
        for i in range(j + 1, k):  # Rows     (Ref: m:62 — i=j+1:k)
            scale = np.sqrt(scales[i] * scales[j])
            # Ref: dcc_inference_objective.m:64 — squeeze(stdData(i,j,:))
            errors = std_data[i, j, :] / scale - R[i, j]
            objs = objs + 0.5 * (errors ** 2)

    # ------------------------------------------------------------------
    # Task 2.8: Conditionally compute asymmetric objective
    # Ref: dcc_inference_objective.m:69-76
    # Only for ADCC models (l > 0).  Loops over lower-triangle-plus-
    # diagonal pairs (j, i) with j <= i, computing squared deviations
    # from the asymmetric intercept matrix N.
    # CRITICAL: MATLAB for j=1:k, for i=j:k (1-indexed) →
    #   Python for j in range(k), for i in range(j, k) (0-indexed)
    # ------------------------------------------------------------------
    if l > 0:
        for j in range(k):           # Columns  (Ref: m:70 — j=1:k)
            for i in range(j, k):     # Rows     (Ref: m:71 — i=j:k)
                # Ref: dcc_inference_objective.m:72
                errors = std_data_asym[i, j, :] - N[i, j]
                objs = objs + 0.5 * (errors ** 2)

    # ------------------------------------------------------------------
    # Task 2.9: Compute total objective
    # Ref: dcc_inference_objective.m:78 — obj = sum(objs);
    # ------------------------------------------------------------------
    obj = float(np.sum(objs))

    return obj, objs
