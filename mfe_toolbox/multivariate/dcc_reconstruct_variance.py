"""
Reconstruct per-series conditional variances from TARCH parameters.

This module provides the :func:`dcc_reconstruct_variance` helper that
replays the TARCH(P,O,Q) variance recursion for each of K series using
the stored univariate-stage estimation results.  It is called during the
DCC/CCC second-stage likelihood evaluation and inference to rebuild the
conditional variance matrix H without re-estimating the first-stage
univariate models.

Migrated from ``multivariate/dcc_reconstruct_variance.m`` (35 lines) in
the MFE Toolbox (Version 4.0, Kevin Sheppard, University of Oxford).

Called by:
    - ``dcc_likelihood.m`` (line 90)
    - ``dcc_inference_objective.m`` (line 50)

Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 1    Date: 17/4/2012

Python migration: Blitzy Platform — MATLAB-to-Python 3.12 migration
"""

import numpy as np

from mfe_toolbox.univariate.tarch_core import tarch_core


def dcc_reconstruct_variance(
    garch_parameters: np.ndarray,
    univariate: list[dict],
) -> np.ndarray:
    """Reconstruct per-series conditional variances from TARCH parameters.

    Computes the conditional variance according to TARCH models for use in
    DCC, CCC-MVGARCH, and related multivariate volatility estimators.  For
    each of the *K* series the function extracts the relevant slice of
    the concatenated GARCH parameter vector, calls :func:`tarch_core` to
    replay the variance recursion, and stores the resulting conditional
    variances in the corresponding column of the output matrix *H*.

    Parameters
    ----------
    garch_parameters : np.ndarray
        1-D array of concatenated univariate GARCH parameters for all *K*
        series, laid out as ``[vol1, vol2, ..., volK]`` where each block
        ``volI`` has length ``1 + p_i + o_i + q_i`` (omega plus ARCH,
        threshold, and GARCH coefficients for series *i*).  The total
        length must equal ``sum(1 + p_i + o_i + q_i)`` over all *K*
        series.
    univariate : list[dict]
        A list of *K* dictionaries, one per series, each containing the
        fields produced by the first-stage univariate TARCH estimation:

        - ``'p'`` (int) — number of symmetric ARCH lags
        - ``'o'`` (int) — number of asymmetric / threshold lags
        - ``'q'`` (int) — number of GARCH lags
        - ``'m'`` (int) — number of back-cast / presample periods
        - ``'T'`` (int) — total length of ``fdata`` / ``fIdata``
          (including presample)
        - ``'fdata'`` (np.ndarray) — transformed data (squared residuals
          or absolute residuals, depending on ``tarch_type``)
        - ``'fIdata'`` (np.ndarray) — asymmetric / threshold data
        - ``'back_cast'`` (float) — back-cast value for initialisation
        - ``'tarch_type'`` (int) — variance-process type (0 = variance,
          1 = absolute-value / std-dev)

    Returns
    -------
    H : np.ndarray
        2-D array of shape ``(T, K)`` containing the reconstructed
        conditional variances, where ``T = univariate[0]['T'] -
        univariate[0]['m']`` (the effective sample length after removing
        presample observations) and ``K = len(univariate)``.

    Raises
    ------
    ValueError
        If *univariate* is an empty list (no series to process).

    See Also
    --------
    mfe_toolbox.univariate.tarch_core.tarch_core :
        Numba JIT-accelerated TARCH variance recursion.
    mfe_toolbox.multivariate.dcc : DCC model driver.
    mfe_toolbox.multivariate.ccc_mvgarch : CCC-MVGARCH model driver.

    Notes
    -----
    This is a faithful translation of ``dcc_reconstruct_variance.m``.
    Every index change and struct-to-dict conversion is documented with
    inline comments referencing the original MATLAB source line numbers.

    The ``univariate`` list-of-dicts structure mirrors the MATLAB cell
    array of structs produced by the first-stage estimation in
    ``dcc_fit_variance.m``.
    """
    # --- Input validation ------------------------------------------------
    # Ref: dcc_reconstruct_variance.m:23 — MATLAB would error on line 24
    # when accessing univariate{1} if the cell is empty.  We raise an
    # explicit ValueError for clarity.
    if not univariate:
        raise ValueError(
            "univariate must be a non-empty list of per-series "
            "estimation dictionaries."
        )

    # --- Determine dimensions --------------------------------------------
    # Ref: dcc_reconstruct_variance.m:23 — k = length(univariate)
    k = len(univariate)

    # Ref: dcc_reconstruct_variance.m:24 — T = univariate{1}.T - univariate{1}.m
    # CRITICAL: MATLAB cell index {1} → Python list index [0] (1→0 indexed)
    # CRITICAL: MATLAB struct field u.T → Python dict key u['T']
    T = univariate[0]['T'] - univariate[0]['m']

    # --- Allocate output -------------------------------------------------
    # Ref: dcc_reconstruct_variance.m:25 — H = zeros(T, k)
    H = np.zeros((T, k))

    # --- Reconstruct variance for each series ----------------------------
    # Ref: dcc_reconstruct_variance.m:27-35
    offset = 0
    for i in range(k):
        # Ref: dcc_reconstruct_variance.m:29 — u = univariate{i}
        # MATLAB 1-based loop for i=1:k → Python 0-based for i in range(k)
        u = univariate[i]

        # Ref: dcc_reconstruct_variance.m:30 — count = u.p + u.o + u.q + 1
        count = u['p'] + u['o'] + u['q'] + 1

        # Ref: dcc_reconstruct_variance.m:31 — volParameters = garchParameters(offset + (1:count))
        # MATLAB 1-indexed offset+(1:count) → Python 0-indexed slice offset:offset+count
        vol_parameters = garch_parameters[offset:offset + count]

        # Ref: dcc_reconstruct_variance.m:32 — offset = offset + count
        offset += count

        # Ref: dcc_reconstruct_variance.m:33 — tarch_core(u.fdata, u.fIdata,
        #   volParameters, u.back_cast, u.p, u.o, u.q, u.m, u.T, u.tarch_type)
        ht = tarch_core(
            u['fdata'],
            u['fIdata'],
            vol_parameters,
            u['back_cast'],
            u['p'],
            u['o'],
            u['q'],
            u['m'],
            u['T'],
            u['tarch_type'],
        )

        # Ref: dcc_reconstruct_variance.m:34 — H(:,i) = ht(u.m+1:u.T)
        # MATLAB 1-indexed inclusive ht(u.m+1:u.T) maps to
        # Python 0-indexed ht[u['m']:u['T']] (same T-m elements)
        H[:, i] = ht[u['m']:u['T']]

    return H
