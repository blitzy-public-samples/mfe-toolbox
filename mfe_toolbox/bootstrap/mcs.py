"""
Model Confidence Set (MCS) of Hansen, Lunde and Nason.

Implements the Model Confidence Set procedure for comparing and selecting among
multiple competing models based on their out-of-sample loss functions. The MCS
identifies a set of models that contains the best model with a given level of
confidence, using both the Range (R) and Semi-Quadratic (SQ) test statistics
and sequential elimination.

The procedure operates on quantities that should be "bads" (e.g., losses). If
the quantities of interest are "goods" (e.g., returns), negate them before
calling this function.

References
----------
Hansen, P.R., Lunde, A., and Nason, J.M. (2011), "The Model Confidence Set",
Econometrica, 79(2), 453-497.

Notes
-----
Migrated from bootstrap/mcs.m
Original Author: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 3    Date: 4/1/2007

See Also
--------
bsds : Bootstrap Data Snooping / Superior Predictive Ability test.
block_bootstrap : Circular block bootstrap for dependent series.
stationary_bootstrap : Stationary bootstrap with geometric block lengths.
"""

import numpy as np

from mfe_toolbox.bootstrap.block_bootstrap import block_bootstrap
from mfe_toolbox.bootstrap.stationary_bootstrap import stationary_bootstrap


def mcs(losses, alpha, B, w, boot='STATIONARY'):
    """Compute the Model Confidence Set of Hansen, Lunde and Nason.

    Performs sequential model elimination using bootstrap-based test statistics
    to identify the set of models that contains the best model with probability
    at least ``1 - alpha``. Both the Range (R) and Semi-Quadratic (SQ) test
    statistics are computed.

    Parameters
    ----------
    losses : array_like
        T-by-K matrix of losses where T is the number of observations and K is
        the number of competing models. Each column represents the loss series
        for one model.
    alpha : float
        Significance level for the MCS, strictly between 0 and 1. Models are
        included in the MCS if their sequential p-value exceeds ``alpha``.
    B : int
        Number of bootstrap replications. Must be a positive integer.
    w : int
        Block length for the bootstrap. Must be a positive integer. For the
        stationary bootstrap, ``w`` is the expected (average) block length.
    boot : str, optional
        Bootstrap method. ``'STATIONARY'`` (default) uses the stationary
        bootstrap with geometric block lengths. ``'BLOCK'`` uses the circular
        block bootstrap with fixed block length.

    Returns
    -------
    includedR : ndarray
        1-D array of 0-based model indices included in the MCS using the
        Range (R) test statistic.
    pvalsR : ndarray
        1-D array of length K containing the sequential p-values for the R
        method. Values are monotonized (non-decreasing) to ensure coherent
        interpretation.
    excludedR : ndarray
        1-D array of 0-based model indices excluded from the MCS using the
        R method, ordered by elimination sequence (first excluded = worst).
    includedSQ : ndarray
        1-D array of 0-based model indices included in the MCS using the
        Semi-Quadratic (SQ) test statistic.
    pvalsSQ : ndarray
        1-D array of length K containing the sequential p-values for the SQ
        method, monotonized (non-decreasing).
    excludedSQ : ndarray
        1-D array of 0-based model indices excluded from the MCS using the
        SQ method, ordered by elimination sequence.

    Raises
    ------
    ValueError
        If ``losses`` has fewer than 2 observations, ``alpha`` is not in
        (0, 1), ``B`` is not a positive integer, ``w`` is not a positive
        integer, or ``boot`` is not ``'STATIONARY'`` or ``'BLOCK'``.

    Notes
    -----
    The MCS procedure works as follows:

    1. Compute pairwise mean loss differentials ``d_ij = E[L_i - L_j]`` and
       their bootstrap standard deviations.
    2. Compute standardized test statistics under two formulations:

       - **Range (R):** ``T_R = max_{i,j} |t_{ij}|`` — maximum absolute
         standardized pairwise differential.
       - **Semi-Quadratic (SQ):** ``T_SQ = (1/2) sum_{i,j} t_{ij}^2`` —
         sum of squared standardized differentials.

    3. Sequentially eliminate the model with the largest standardized average
       excess loss, computing bootstrap p-values at each step.
    4. Monotonize p-values (enforce non-decreasing sequence) so that the MCS
       at level ``alpha`` is nested within the MCS at any level ``alpha' > alpha``.
    5. Include all models whose sequential p-value exceeds ``alpha``.

    All returned model indices are 0-based (Python convention), unlike the
    original MATLAB implementation which uses 1-based indices.

    Examples
    --------
    MCS with 5% size, 1000 bootstrap replications, average block length 12:

    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> losses = rng.standard_normal((500, 5)) + np.linspace(0.1, 0.5, 5)
    >>> inclR, pvR, exclR, inclSQ, pvSQ, exclSQ = mcs(losses, 0.05, 200, 12)

    MCS on "goods" (negate to convert to losses):

    >>> gains = rng.standard_normal((500, 5)) + np.linspace(0.1, 0.5, 5)
    >>> inclR, pvR, exclR, inclSQ, pvSQ, exclSQ = mcs(-gains, 0.05, 200, 12)

    MCS with circular block bootstrap:

    >>> inclR, pvR, exclR, inclSQ, pvSQ, exclSQ = mcs(
    ...     losses, 0.05, 200, 12, boot='BLOCK'
    ... )
    """
    # ===========================================================================
    # Input Checking
    # Ref: mcs.m:43-69 — input validation block
    # ===========================================================================

    # Ensure losses is a 2-D numpy array of float64
    losses = np.asarray(losses, dtype=np.float64)
    if losses.ndim == 1:
        # Ref: mcs.m expects T-by-K matrix; a 1-D vector is T-by-1
        losses = losses.reshape(-1, 1)

    # Ref: mcs.m:53-55 — t=size(losses,1); if t<2 error
    t = losses.shape[0]
    if t < 2:
        raise ValueError('LOSSES must have at least 2 observations.')

    # Ref: mcs.m:57-58 — alpha must be a scalar between 0 and 1
    if not np.isscalar(alpha) or alpha >= 1.0 or alpha <= 0.0:
        raise ValueError('ALPHA must be a scalar between 0 and 1')

    # Ref: mcs.m:60-61 — B must be a positive scalar integer
    if not np.isscalar(B) or B < 1 or int(B) != B:
        raise ValueError('B must be a positive scalar integer')
    B = int(B)

    # Ref: mcs.m:63-64 — w must be a positive scalar integer
    if not np.isscalar(w) or w < 1 or int(w) != w:
        raise ValueError('W must be a positive scalar integer')
    w = int(w)

    # Ref: mcs.m:66-68 — boot must be 'STATIONARY' or 'BLOCK'
    boot = str(boot).upper()
    if boot not in ('STATIONARY', 'BLOCK'):
        raise ValueError("BOOT must be either 'STATIONARY' or 'BLOCK'.")

    # ===========================================================================
    # 1. Compute bootstrap indices for the entire procedure
    # Ref: mcs.m:74-79 — generate T×B matrix of bootstrap resampling indices
    # ===========================================================================
    # Pass np.arange(t) as data so that bsdata values ARE the 0-based indices.
    # We use the second return value (indices) which is int64 and 0-based.
    index_data = np.arange(t, dtype=np.float64).reshape(-1, 1)
    if boot == 'BLOCK':
        # Ref: mcs.m:76 — [bsdata]=block_bootstrap((1:t)',B,w);
        _, bsdata = block_bootstrap(index_data, B, w)
    else:
        # Ref: mcs.m:78 — [bsdata]=stationary_bootstrap((1:t)',B,w);
        _, bsdata = stationary_bootstrap(index_data, B, w)
    # bsdata: (T, B) int64 array of 0-based indices into losses

    # ===========================================================================
    # 2. Compute pairwise mean loss differentials dij-bar
    # Ref: mcs.m:82-87 — M0=size(losses,2); dijbar computation
    # ===========================================================================
    M0 = losses.shape[1]  # number of models

    # Ref: mcs.m:84-87 — dijbar(j,:) = mean(losses - repmat(losses(:,j),1,M0))
    # dijbar[j, i] = mean_t(loss_i(t) - loss_j(t)) = mean(losses[:,i]) - mean(losses[:,j])
    loss_means = np.mean(losses, axis=0)  # (M0,)
    # Ref: mcs.m:85-87 — vectorized: dijbar[j, i] = loss_means[i] - loss_means[j]
    dijbar = loss_means[np.newaxis, :] - loss_means[:, np.newaxis]  # (M0, M0)

    # ===========================================================================
    # 3. Compute bootstrap dij-bar-star and variance
    # Ref: mcs.m:91-101
    # ===========================================================================
    # Ref: mcs.m:91 — dijbarstar=zeros(M0,M0,B)
    dijbarstar = np.zeros((M0, M0, B), dtype=np.float64)

    # Ref: mcs.m:92-98 — for each bootstrap b, compute mean losses and differentials
    for b in range(B):
        # Ref: mcs.m:93 — meanworkdata=mean(losses(bsdata(:,b),:));
        meanworkdata = np.mean(losses[bsdata[:, b], :], axis=0)  # (M0,)
        # Ref: mcs.m:95-96 — dijbarstar(j,:,b) = meanworkdata - meanworkdata(j)
        # Vectorized: dijbarstar[:, :, b][j, i] = meanworkdata[i] - meanworkdata[j]
        dijbarstar[:, :, b] = (
            meanworkdata[np.newaxis, :] - meanworkdata[:, np.newaxis]
        )

    # Ref: mcs.m:100 — vardijbar = mean((dijbarstar - repmat(dijbar,[1 1 B])).^2, 3)
    # Compute variance of the bootstrap differentials (across bootstrap dimension)
    vardijbar = np.mean(
        (dijbarstar - dijbar[:, :, np.newaxis]) ** 2, axis=2
    )  # (M0, M0)

    # Ref: mcs.m:101 — vardijbar = vardijbar + diag(ones(M0,1))
    # Add identity to diagonal to prevent division by zero (diagonal is self-comparison)
    vardijbar = vardijbar + np.eye(M0)

    # ===========================================================================
    # 4. Compute standardized bootstrap and data statistics
    # Ref: mcs.m:107-108 — z0 and zdata0 for use in elimination loops
    # ===========================================================================
    # Ref: mcs.m:107 — z0 = (dijbarstar - dijbar) ./ sqrt(vardijbar)
    sqrt_var = np.sqrt(vardijbar)  # (M0, M0)
    z0 = (dijbarstar - dijbar[:, :, np.newaxis]) / sqrt_var[:, :, np.newaxis]
    # z0: (M0, M0, B)

    # Ref: mcs.m:108 — zdata0 = dijbar ./ sqrt(vardijbar)
    zdata0 = dijbar / sqrt_var  # (M0, M0)

    # ===========================================================================
    # 5. Range (R) method — sequential elimination
    # Ref: mcs.m:110-147
    # ===========================================================================
    # Ref: mcs.m:110-111 — initialize exclusion order and p-values
    excludedR = np.zeros(M0, dtype=np.intp)
    pvalsR = np.ones(M0, dtype=np.float64)

    # Ref: mcs.m:112-132 — sequential elimination loop (M0-1 iterations)
    for i in range(M0 - 1):
        # Ref: mcs.m:113 — included=setdiff(1:M0,excludedR)
        # Get indices of models not yet excluded (0-based)
        included = np.setdiff1d(np.arange(M0), excludedR[:i])
        m = len(included)

        # Ref: mcs.m:115 — z=z0(included,included,:)
        # Extract sub-array for included models; np.ix_ indexes first 2 dims
        z = z0[np.ix_(included, included)]  # (m, m, B)

        # Ref: mcs.m:117 — empdistTR=squeeze(max(max(abs(z),[],1),[],2))
        # Maximum absolute standardized differential across all model pairs per bootstrap
        empdistTR = np.max(np.abs(z), axis=(0, 1))  # (B,)

        # Ref: mcs.m:118-119 — zdata=zdata0(included,included); TR=max(max(zdata))
        zdata = zdata0[np.ix_(included, included)]  # (m, m)
        TR = np.max(zdata)

        # Ref: mcs.m:120 — pvalsR(i)=mean(empdistTR>TR)
        pvalsR[i] = np.mean(empdistTR > TR)

        # Ref: mcs.m:124 — dibar=mean(dijbar(included,included),1)*(m/(m-1))
        # Bias-corrected average excess loss for each included model
        sub_dijbar = dijbar[np.ix_(included, included)]  # (m, m)
        dibar = np.mean(sub_dijbar, axis=0) * (m / (m - 1))  # (m,)

        # Ref: mcs.m:126 — dibstar=squeeze(mean(dijbarstar(included,included,:),1))*(m/(m-1))
        sub_dijbarstar = dijbarstar[np.ix_(included, included)]  # (m, m, B)
        dibstar = np.mean(sub_dijbarstar, axis=0) * (m / (m - 1))  # (m, B)

        # Ref: mcs.m:127 — vardi=mean((dibstar'-repmat(dibar,B,1)).^2)
        # Variance of the bias-corrected average excess loss
        # dibstar.T is (B, m); dibar is (m,) — broadcasting gives (B, m)
        vardi = np.mean((dibstar.T - dibar) ** 2, axis=0)  # (m,)

        # Ref: mcs.m:128 — t=dibar./sqrt(vardi)
        # t-statistic for each model (avoid MATLAB variable name shadow with 't')
        t_stat = dibar / np.sqrt(vardi)  # (m,)

        # Ref: mcs.m:130-131 — [temp,modeltoremove] = max(t); excludedR(i)=included(modeltoremove)
        # Remove the model with the largest standardized excess loss
        modeltoremove = np.argmax(t_stat)
        excludedR[i] = included[modeltoremove]

    # Ref: mcs.m:134-141 — Monotonize p-values (running maximum)
    # The MCS p-value is the maximum of all p-values up to that point,
    # ensuring a nested MCS structure.
    pvalsR = np.maximum.accumulate(pvalsR)

    # Ref: mcs.m:143 — excludedR(end)=setdiff(1:M0,excludedR)
    # Add the final remaining model (last to be "eliminated" = best model)
    excludedR[M0 - 1] = np.setdiff1d(np.arange(M0), excludedR[:M0 - 1])[0]

    # Ref: mcs.m:145-147 — split into included and excluded based on alpha threshold
    # pl = find(pvalsR >= alpha, 1, 'first')  — MATLAB 1-based
    pl_indices = np.where(pvalsR >= alpha)[0]
    if len(pl_indices) > 0:
        pl = pl_indices[0]
    else:
        # All p-values below alpha — no models included
        pl = M0

    # Ref: mcs.m:146 — includedR=excludedR(pl:M0)
    # Models from position pl onward are included (they survived elimination)
    includedR = excludedR[pl:].copy()

    # Ref: mcs.m:147 — excludedR=excludedR(1:pl-1)
    excludedR = excludedR[:pl].copy()

    # ===========================================================================
    # 6. Semi-Quadratic (SQ) method — sequential elimination
    # Ref: mcs.m:150-190 — identical structure to R method with SQ statistic
    # ===========================================================================
    # Ref: mcs.m:150-151
    excludedSQ = np.zeros(M0, dtype=np.intp)
    pvalsSQ = np.ones(M0, dtype=np.float64)

    # Ref: mcs.m:152-175 — sequential elimination loop
    for i in range(M0 - 1):
        # Ref: mcs.m:153 — included=setdiff(1:M0,excludedSQ)
        included = np.setdiff1d(np.arange(M0), excludedSQ[:i])
        m = len(included)

        # Ref: mcs.m:155 — z=z0(included,included,:)
        z = z0[np.ix_(included, included)]  # (m, m, B)

        # Ref: mcs.m:158 — empdistTSQ=squeeze(sum(sum(z.^2))/2)
        # Sum of squared standardized differentials, divided by 2 to avoid double-counting
        empdistTSQ = np.sum(z ** 2, axis=(0, 1)) / 2.0  # (B,)

        # Ref: mcs.m:160-161 — zdata=zdata0(included,included); TSQ=sum(sum(zdata.^2))/2
        zdata = zdata0[np.ix_(included, included)]  # (m, m)
        TSQ = np.sum(zdata ** 2) / 2.0

        # Ref: mcs.m:163 — pvalsSQ(i)=mean(empdistTSQ>TSQ)
        pvalsSQ[i] = np.mean(empdistTSQ > TSQ)

        # Ref: mcs.m:167 — dibar=mean(dijbar(included,included),1)*(m/(m-1))
        sub_dijbar = dijbar[np.ix_(included, included)]  # (m, m)
        dibar = np.mean(sub_dijbar, axis=0) * (m / (m - 1))  # (m,)

        # Ref: mcs.m:169 — dibstar=squeeze(mean(dijbarstar(included,included,:),1))*(m/(m-1))
        sub_dijbarstar = dijbarstar[np.ix_(included, included)]  # (m, m, B)
        dibstar = np.mean(sub_dijbarstar, axis=0) * (m / (m - 1))  # (m, B)

        # Ref: mcs.m:170 — vardi=mean((dibstar'-repmat(dibar,B,1)).^2)
        vardi = np.mean((dibstar.T - dibar) ** 2, axis=0)  # (m,)

        # Ref: mcs.m:171 — t=dibar./sqrt(vardi)
        t_stat = dibar / np.sqrt(vardi)  # (m,)

        # Ref: mcs.m:173-174 — remove model with largest t-statistic
        modeltoremove = np.argmax(t_stat)
        excludedSQ[i] = included[modeltoremove]

    # Ref: mcs.m:177-183 — Monotonize p-values (running maximum)
    pvalsSQ = np.maximum.accumulate(pvalsSQ)

    # Ref: mcs.m:186 — excludedSQ(end)=setdiff(1:M0,excludedSQ)
    excludedSQ[M0 - 1] = np.setdiff1d(np.arange(M0), excludedSQ[:M0 - 1])[0]

    # Ref: mcs.m:188-190 — split into included and excluded
    pl_indices = np.where(pvalsSQ >= alpha)[0]
    if len(pl_indices) > 0:
        pl = pl_indices[0]
    else:
        pl = M0

    # Ref: mcs.m:189 — includedSQ=excludedSQ(pl:M0)
    includedSQ = excludedSQ[pl:].copy()

    # Ref: mcs.m:190 — excludedSQ=excludedSQ(1:pl-1)
    excludedSQ = excludedSQ[:pl].copy()

    return (includedR, pvalsR, excludedR, includedSQ, pvalsSQ, excludedSQ)
