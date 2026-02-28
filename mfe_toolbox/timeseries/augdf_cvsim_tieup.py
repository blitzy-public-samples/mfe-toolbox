"""
ADF Critical Value Simulation Aggregation.

Aggregates two halves of Monte Carlo simulation results into final
critical-value tables for Augmented Dickey-Fuller (ADF) unit root tests.

This module converts the MATLAB script ``timeseries/augdf_cvsim_tieup.m``
into a reusable Python function. The original MATLAB code is a script
(not a function) that repeats an identical processing pattern for three
deterministic cases (no deterministic terms, constant only, constant + trend).
The Python version abstracts this into a single callable that operates on
one pair of simulation halves at a time.

Notes
-----
Translated from ``timeseries/augdf_cvsim_tieup.m`` (MFE Toolbox v4.0,
Kevin Sheppard, University of Oxford).

References
----------
.. [1] Dickey, D. A. and Fuller, W. A. (1979). Distribution of the
   Estimators for Autoregressive Time Series With a Unit Root.
   *Journal of the American Statistical Association*, 74, 427-431.
"""

import numpy as np


def augdf_cvsim_tieup(tstats_half1: np.ndarray,
                      tstats_half2: np.ndarray,
                      B: int) -> tuple:
    """
    Aggregate two halves of ADF critical-value Monte Carlo simulation results.

    Combines two halves of simulated t-statistics, sorts them column-wise,
    and extracts critical values at 103 predefined quantile levels for
    Augmented Dickey-Fuller unit root testing.

    This function implements the core logic of the MATLAB script
    ``timeseries/augdf_cvsim_tieup.m``, which processes Monte Carlo
    simulation outputs in two halves (from ``augdfcv1.mat`` and
    ``augdfcv2.mat``) and produces critical-value lookup tables.

    Parameters
    ----------
    tstats_half1 : numpy.ndarray
        First half of simulated t-statistics. Shape ``(B, num_T)`` where
        *B* is the number of Monte Carlo replications per half and *num_T*
        is the number of sample sizes tested. May also be a 1-D array of
        shape ``(B,)`` when only one sample size is present.
    tstats_half2 : numpy.ndarray
        Second half of simulated t-statistics. Must have the same shape as
        *tstats_half1*.
    B : int
        Number of bootstrap/Monte Carlo replications per half. The total
        number of replications after concatenation is ``2 * B``.

    Returns
    -------
    cv_table : numpy.ndarray
        Critical value table of shape ``(num_quantiles, num_T)`` where
        *num_quantiles* is 103 (the number of quantile levels). Each column
        corresponds to one sample size and each row to one quantile level.
        If the inputs were 1-D, *num_T* equals 1.
    cvs : numpy.ndarray
        1-D array of length 103 containing the quantile levels used:
        ``[0.001, 0.005, 0.01, 0.02, ..., 0.99, 0.995, 0.999]``.

    Raises
    ------
    ValueError
        If *tstats_half1* or *tstats_half2* are not numpy arrays, if their
        shapes do not match, or if *B* is not a positive integer.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> half1 = rng.standard_normal((1000, 5))
    >>> half2 = rng.standard_normal((1000, 5))
    >>> cv_table, cvs = augdf_cvsim_tieup(half1, half2, 1000)
    >>> cv_table.shape
    (103, 5)
    >>> len(cvs)
    103

    Notes
    -----
    The MATLAB script repeats this pattern three times for ADF cases 1, 2,
    and 4. In Python, callers invoke this function once per case, passing
    the appropriate t-statistic halves.

    The quantile extraction uses MATLAB-compatible indexing:
    ``floor(Cvs * 2 * B)`` produces 1-based row indices in MATLAB. The
    Python version subtracts 1 to obtain 0-based indices and clips to
    prevent out-of-bounds access.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not isinstance(tstats_half1, np.ndarray):
        raise ValueError('tstats_half1 and tstats_half2 must be numpy arrays')
    if not isinstance(tstats_half2, np.ndarray):
        raise ValueError('tstats_half1 and tstats_half2 must be numpy arrays')
    if not isinstance(B, (int, np.integer)) or B <= 0:
        raise ValueError('B must be a positive integer')

    # Promote 1-D inputs to 2-D column vectors for uniform processing
    # Ref: augdf_cvsim_tieup.m — script always operates on 2-D matrices
    input_was_1d = False
    if tstats_half1.ndim == 1:
        tstats_half1 = tstats_half1.reshape(-1, 1)
        input_was_1d = True
    if tstats_half2.ndim == 1:
        tstats_half2 = tstats_half2.reshape(-1, 1)

    if tstats_half1.shape != tstats_half2.shape:
        raise ValueError(
            'tstats_half1 and tstats_half2 must have the same shape'
        )

    # ------------------------------------------------------------------
    # Step 1: Vertical concatenation of the two simulation halves
    # Ref: augdf_cvsim_tieup.m:8 — tstats_case1 = [temp; tstats_case1];
    # ------------------------------------------------------------------
    combined = np.concatenate([tstats_half1, tstats_half2], axis=0)

    # ------------------------------------------------------------------
    # Step 2: Column-wise sort (ascending)
    # Ref: augdf_cvsim_tieup.m:10 — tstats_case1 = sort(tstats_case1);
    # MATLAB sort() default is column-wise ascending; np.sort axis=0 matches.
    # ------------------------------------------------------------------
    combined = np.sort(combined, axis=0)

    # ------------------------------------------------------------------
    # Step 3: Construct quantile levels
    # Ref: augdf_cvsim_tieup.m:11 — Cvs = [.001 .005 .01:.01:.99 .995 .999];
    # MATLAB .01:.01:.99 generates 99 values: 0.01, 0.02, ..., 0.99
    # np.arange(0.01, 1.0, 0.01) produces the identical 99-element sequence.
    # Total quantile levels: 2 + 99 + 2 = 103
    # ------------------------------------------------------------------
    cvs = np.concatenate([
        np.array([0.001, 0.005]),
        np.arange(0.01, 1.0, 0.01),   # 0.01, 0.02, ..., 0.99
        np.array([0.995, 0.999])
    ])

    # ------------------------------------------------------------------
    # Step 4: Extract critical values at quantile positions
    # Ref: augdf_cvsim_tieup.m:12 —
    #     augdf_case1_cv = tstats_case1(floor(Cvs*(2*B)), :);
    #
    # MATLAB uses 1-based indexing, so floor(Cvs * 2*B) gives valid
    # 1-based row indices.  Python uses 0-based indexing, so we subtract 1.
    # np.clip prevents any edge-case out-of-bounds access.
    # ------------------------------------------------------------------
    total_reps = 2 * B
    indices = np.floor(cvs * total_reps).astype(int)

    # Ref: augdf_cvsim_tieup.m:12 — MATLAB 1-indexed; Python 0-indexed
    # MATLAB array(k, :) == Python array[k - 1, :]
    indices = np.clip(indices - 1, 0, combined.shape[0] - 1)

    cv_table = combined[indices, :]

    return cv_table, cvs
