"""
Bootstrap Data Snooping (BSDS) — Hansen-White Superior Predictive Ability Test.

Computes White's Reality Check p-value and Hansen's consistent and lower
p-values for testing whether any model in a set of alternatives has smaller
expected loss than a benchmark.  Supports both studentized and standard
(non-studentized) test statistics and allows either a stationary bootstrap or
a circular block bootstrap for dependence-robust resampling.

The null hypothesis is that the benchmark's average loss is no larger than the
minimum average loss across all comparison models.  The alternative is that at
least one model has a strictly smaller average loss.

If the quantities of interest are *goods* (e.g. returns) rather than *bads*
(e.g. losses), negate both ``bench`` and ``models`` before calling.

References
----------
White, H. (2000), "A Reality Check for Data Snooping", *Econometrica*,
68(5), 1097-1126.

Hansen, P.R. (2005), "A Test for Superior Predictive Ability", *Journal of
Business & Economic Statistics*, 23(4), 365-380.

See Also
--------
mcs : Model Confidence Set procedure.

Notes
-----
Migrated from ``bootstrap/bsds.m``.
Original Author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
Revision: 3    Date: 4/1/2007
"""

import numpy as np

from mfe_toolbox.bootstrap.block_bootstrap import block_bootstrap
from mfe_toolbox.bootstrap.stationary_bootstrap import stationary_bootstrap


def bsds(bench, models, B, w, type_='STUDENTIZED', boot='STATIONARY'):
    """Calculate White's and Hansen's p-values for out-performance.

    Implements the Bootstrap Data Snooping procedure of White (2000) and the
    Superior Predictive Ability (SPA) test of Hansen (2005).  The test
    evaluates whether any model in a collection has statistically significant
    lower loss than a benchmark, correcting for the multiplicity implicit in
    comparing many models simultaneously.

    Three p-values are returned, corresponding to different treatments of
    nuisance parameters associated with models whose expected loss differential
    may be zero:

    * **Consistent** (*c*) — Hansen's SPA p-value that re-centres using a
      data-driven truncation threshold based on the ``log log T`` rule.
    * **Upper** (*u*) — White's original Reality Check p-value that re-centres
      at the sample mean (least favourable to the benchmark).
    * **Lower** (*l*) — Hansen's lower p-value that sets the re-centring for
      models that are *worse* than the benchmark to zero (most favourable to
      the benchmark).

    Parameters
    ----------
    bench : numpy.ndarray
        T-element vector of losses from the benchmark model.  Must be 1-D or
        a column vector (T × 1).
    models : numpy.ndarray
        T × K matrix of losses from each of K comparison models.  The number
        of rows must equal the length of ``bench``.
    B : int
        Number of bootstrap replications.  Must be a positive integer.
    w : int
        Desired block length for the bootstrap.  Must be a positive integer.
    type_ : str, optional
        ``'STANDARD'`` or ``'STUDENTIZED'`` (default).  Studentized residuals
        generally lead to better power, particularly when the loss functions
        are heteroskedastic.
    boot : str, optional
        ``'STATIONARY'`` (default) or ``'BLOCK'``.  Selects the Politis-Romano
        stationary bootstrap or the circular block bootstrap for generating
        resampling indices.

    Returns
    -------
    c : float
        Consistent p-value (Hansen's SPA).
    u : float
        Upper p-value (White's Reality Check).
    l : float
        Lower p-value (Hansen).

    Raises
    ------
    ValueError
        If any input fails validation:
        - ``bench`` is not a column vector or has fewer than 2 observations.
        - ``models`` has a different number of rows than ``bench``.
        - ``B`` or ``w`` is not a positive scalar integer.
        - ``boot`` is not ``'STATIONARY'`` or ``'BLOCK'``.

    Examples
    --------
    Standard Reality Check with 1000 replications and block length 12:

    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> bench = rng.standard_normal(1000) ** 2
    >>> models = rng.standard_normal((1000, 100)) ** 2
    >>> c, u, l = bsds(bench, models, 1000, 12)

    Hansen's SPA on *goods* (negate both bench and models):

    >>> bench_good = 0.01 + rng.standard_normal(1000)
    >>> models_good = rng.standard_normal((1000, 100))
    >>> spa_pval, rc_pval, low_pval = bsds(-bench_good, -models_good, 1000, 12)

    Notes
    -----
    The implementation follows bsds.m exactly.  Key translation notes:

    * MATLAB's 1-indexed subscripts are replaced by 0-indexed NumPy fancy
      indexing throughout.
    * ``repmat(bench, 1, k)`` is replaced by NumPy broadcasting via
      ``bench[:, np.newaxis]``.
    * Bartlett kernel weights and the inner-product variance estimator match
      the MATLAB loop structure line-for-line.
    """
    # =========================================================================
    # Phase 1 — Input Validation
    # Ref: bsds.m:54-98
    # =========================================================================

    # Handle default / empty type_ — Ref: bsds.m:60-68
    if type_ is None or type_ == '':
        type_ = 'STUDENTIZED'

    # Ref: bsds.m:69-73 — determine studentization flag
    is_studentized = type_.upper() == 'STUDENTIZED'

    # Handle default / empty boot — Ref: bsds.m:60-64
    if boot is None or boot == '':
        boot = 'STATIONARY'

    # ------------------------------------------------------------------
    # Validate bench
    # Ref: bsds.m:75-81
    # ------------------------------------------------------------------
    bench = np.asarray(bench, dtype=np.float64)
    if bench.ndim == 2:
        # Ref: bsds.m:76-77 — if kb>1 error
        if bench.shape[1] > 1:
            raise ValueError('BENCH must be a column vector')
        bench = bench.ravel()
    elif bench.ndim > 2:
        raise ValueError('BENCH must be a column vector')
    else:
        bench = np.atleast_1d(bench).ravel()

    tb = len(bench)
    # Ref: bsds.m:79-80 — at least 2 observations
    if tb < 2:
        raise ValueError('BENCH must have at least 2 observations.')

    # ------------------------------------------------------------------
    # Validate models
    # Ref: bsds.m:82-85
    # ------------------------------------------------------------------
    models = np.asarray(models, dtype=np.float64)
    models = np.atleast_2d(models)
    t, k = models.shape
    if t != tb:
        raise ValueError(
            'BENCH and MODELS must have the same number of observations.'
        )

    # ------------------------------------------------------------------
    # Validate B — Ref: bsds.m:86-88
    # ------------------------------------------------------------------
    if not np.isscalar(B) or int(B) != B or B < 1:
        raise ValueError('B must be a positive scalar integer')
    B = int(B)

    # ------------------------------------------------------------------
    # Validate w — Ref: bsds.m:89-91
    # ------------------------------------------------------------------
    if not np.isscalar(w) or int(w) != w or w < 1:
        raise ValueError('W must be a positive scalar integer')
    w = int(w)

    # ------------------------------------------------------------------
    # Validate boot — Ref: bsds.m:92-95
    # ------------------------------------------------------------------
    boot = boot.upper()
    if boot not in ('STATIONARY', 'BLOCK'):
        raise ValueError("BOOT must be either 'STATIONARY' or 'BLOCK'.")

    # =========================================================================
    # Phase 2 — Bootstrap Index Generation
    # Ref: bsds.m:99-103
    # =========================================================================
    # Pass 0-indexed integer sequence as data; use the returned *indices*
    # matrix (T×B, dtype int64, 0-based) for fancy-indexing later.
    # Ref: bsds.m:100 — block_bootstrap((1:t)', B, w)  [MATLAB 1-indexed]
    indices_data = np.arange(t, dtype=np.float64).reshape(-1, 1)
    if boot == 'BLOCK':
        _, bsdata_indices = block_bootstrap(indices_data, B, w)
    else:
        _, bsdata_indices = stationary_bootstrap(indices_data, B, w)

    # =========================================================================
    # Phase 3 — Compute Loss Differentials
    # Ref: bsds.m:106 — diffs = models - repmat(bench, 1, k)
    # =========================================================================
    # Broadcasting: bench is (T,), models is (T, K)
    diffs = models - bench[:, np.newaxis]

    # =========================================================================
    # Phase 4 — Bartlett Kernel Long-Run Variance Estimation
    # Ref: bsds.m:111-122
    # =========================================================================

    # Ref: bsds.m:111 — q = 1/w
    q = 1.0 / w

    # Ref: bsds.m:112-113 — Bartlett-type kernel weights for lags 1..t-1
    # MATLAB: i = 1:t-1;
    #         kappa = ((t-i)./t).*(1-q).^i + i./t.*(1-q).^(t-i);
    i_vals = np.arange(1, t, dtype=np.float64)
    kappa = ((t - i_vals) / t) * (1.0 - q) ** i_vals + \
            (i_vals / t) * (1.0 - q) ** (t - i_vals)

    # Ref: bsds.m:115-122 — long-run variance for each of k models
    vars_ = np.zeros(k)
    for i in range(k):
        # Ref: bsds.m:117 — demean the loss differentials for model i
        workdata = diffs[:, i] - np.mean(diffs[:, i])

        # Ref: bsds.m:118 — lag-0 autocovariance
        vars_[i] = workdata @ workdata / t

        # Ref: bsds.m:119-121 — weighted cross-product terms for lags 1..t-1
        # MATLAB j=1:t-1 (1-indexed); Python j=0:t-2 (0-indexed)
        # MATLAB: kappa(j) * workdata(1:t-j)' * workdata(j+1:t) / t
        # Python: kappa[j] maps to lag j+1;
        #         workdata[:t-1-j] maps to MATLAB workdata(1:t-(j+1))
        #         workdata[j+1:]   maps to MATLAB workdata((j+1)+1:t)
        for j in range(t - 1):
            # Ref: bsds.m:120 — cross-product at lag j+1
            vars_[i] += 2.0 * kappa[j] * (
                workdata[:t - 1 - j] @ workdata[j + 1:]
            ) / t

    # =========================================================================
    # Phase 5 — Recentering Vectors
    # Ref: bsds.m:125-144
    # =========================================================================

    # Ref: bsds.m:128 — log(log(t)) truncation rule
    # (Aold at bsds.m:125-126 is computed but not used in MATLAB — omitted)
    Anew = np.sqrt((vars_ / t) * 2.0 * np.log(np.log(t)))

    # Ref: bsds.m:133 — consistent re-centring: zero out models whose sample
    # mean exceeds the data-driven threshold Anew
    mean_diffs = np.mean(diffs, axis=0)  # shape (k,)
    gc = mean_diffs * (mean_diffs < Anew)

    # Ref: bsds.m:140 — lower re-centring: cap at zero
    gl = np.minimum(0.0, mean_diffs)

    # Ref: bsds.m:144 — upper re-centring: use raw sample mean
    gu = mean_diffs.copy()

    # =========================================================================
    # Phase 6 — Bootstrap Test Statistics
    # Ref: bsds.m:147-163
    # =========================================================================

    # Ref: bsds.m:147-149 — pre-allocate B×K matrices
    perfc = np.zeros((B, k))
    perfl = np.zeros((B, k))
    perfu = np.zeros((B, k))

    # Ref: bsds.m:150-154 — studentize if requested
    if is_studentized:
        std_dev = np.sqrt(vars_)
    else:
        std_dev = np.ones(k)

    # Ref: bsds.m:156-163 — build bootstrap distribution for each model
    for i in range(k):
        workdata = diffs[:, i]
        # Ref: bsds.m:159 — mean(workdata(bsdata)) where bsdata is T×B
        # bsdata_indices is T×B of 0-indexed ints;
        # workdata[bsdata_indices] yields T×B; column means → B-vector
        mworkdata = np.mean(workdata[bsdata_indices], axis=0)

        # Ref: bsds.m:160-162 — re-centred, possibly studentized statistics
        perfc[:, i] = (mworkdata - gc[i]) / std_dev[i]
        perfl[:, i] = (mworkdata - gl[i]) / std_dev[i]
        perfu[:, i] = (mworkdata - gu[i]) / std_dev[i]

    # =========================================================================
    # Phase 7 — P-Value Computation
    # Ref: bsds.m:164-176
    # =========================================================================

    # Ref: bsds.m:165 — observed test statistic (minimum across models)
    stat = np.min(mean_diffs / std_dev)

    # Ref: bsds.m:167-172 — row-wise minima, capped at 0
    perfc = np.minimum(np.min(perfc, axis=1), 0.0)
    perfl = np.minimum(np.min(perfl, axis=1), 0.0)
    perfu = np.minimum(np.min(perfu, axis=1), 0.0)

    # Ref: bsds.m:174-176 — empirical p-values
    c = float(np.mean(perfc < stat))
    l = float(np.mean(perfl < stat))
    u = float(np.mean(perfu < stat))

    return c, u, l
