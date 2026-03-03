"""
VAR Impulse Response Functions.

Migrated from ``timeseries/impulseresponse.m`` — Author: Kevin Sheppard
(Revision 3.0, Date: 1/1/2007)

Computes impulse responses for a VAR(P) or irregular VAR(P) and standard
errors under a variety of assumptions on the covariance of the errors:

1. Conditionally Homoskedastic and Uncorrelated
2. Conditionally Homoskedastic but Correlated
3. Heteroskedastic but Conditionally Uncorrelated
4. Heteroskedastic and Correlated

The covariance identification method is determined by the ``sqrttype``
parameter, which supports unit shocks, diagonal scaling, Cholesky
decomposition, spectral decomposition (matrix square root), and the
Generalized Impulse Response of Pesaran and Shin (1998).

Notes
-----
The implementation uses the VMA(∞) recursive representation of the VAR
to compute impulse responses, then applies the delta method (via the G
matrix) to obtain asymptotic standard errors. The G matrices are
constructed from Kronecker products of the scaled impulse response
matrices and used with the parameter variance-covariance matrix from
:func:`~mfe_toolbox.timeseries.vectorar.vectorar`.

References
----------
Sheppard, K. (2009). MFE Toolbox Version 4.0.
Lütkepohl, H. (2005). *New Introduction to Multiple Time Series
    Analysis*. Springer.
Pesaran, M.H. and Shin, Y. (1998). Generalized impulse response
    analysis in linear multivariate models. *Economics Letters*,
    58(1), 17–29.
"""

import numpy as np
from scipy import linalg as scipy_linalg
import matplotlib.pyplot as plt

from mfe_toolbox.timeseries.vectorar import vectorar

__all__ = ['impulseresponse']


def impulseresponse(
    y: np.ndarray,
    constant: int,
    lags: np.ndarray,
    leads: int,
    sqrttype=None,
    graph=None,
    het: int = None,
    uncorr: int = None,
) -> tuple:
    """
    Compute impulse responses for a VAR(P) or irregular VAR(P).

    Estimates a VAR model, computes h-step impulse response functions
    using the VMA(∞) representation, and provides asymptotic standard
    errors via the delta method.

    Parameters
    ----------
    y : np.ndarray
        A ``(T, K)`` matrix of multivariate data, where ``T`` is the
        number of observations and ``K`` is the number of variables.
    constant : int
        Include a constant in the VAR:

        - ``1`` — include intercept
        - ``0`` — no intercept
    lags : np.ndarray or array_like
        Non-negative integer vector representing the VAR orders to
        include in the model. Can be non-contiguous (e.g., ``[1, 3]``
        for an irregular VAR). All elements must be positive integers.
    leads : int
        Number of leads (horizons) to compute the impulse response
        function. Must be a positive integer.
    sqrttype : int or np.ndarray, optional
        Determines the covariance decomposition used for shock
        identification. If scalar, must be one of:

        - ``0`` — Unit (unscaled) shocks; covariance is identity
        - ``1`` — **[DEFAULT]** Scaled but uncorrelated shocks
          (diagonal std dev)
        - ``2`` — Scaled and correlated shocks, Cholesky decomposition
        - ``3`` — Scaled and correlated shocks, spectral decomposition
          (matrix square root)
        - ``4`` — Generalized impulse response of Pesaran and Shin

        If a ``(K, K)`` positive definite matrix, it is used directly
        as the covariance square root.
    graph : int or bool, optional
        Whether to produce a plot of the IRF with ±1.96 std dev bands.
        Default is ``True`` (``1``).
    het : int, optional
        Covariance estimator type for the underlying VAR:

        - ``0`` — Homoskedastic
        - ``1`` — Heteroskedastic **[default]**
    uncorr : int, optional
        Assumed error covariance structure for the underlying VAR:

        - ``0`` — Correlated errors **[default]**
        - ``1`` — Uncorrelated errors

    Returns
    -------
    tuple
        A 3-element tuple:

        - **impulses** (*np.ndarray*) — ``(K, K, leads+1)`` array where
          ``impulses[i, j, h]`` is the response of variable ``i`` to
          shock ``j`` at horizon ``h`` (``h=0`` is the contemporaneous
          response).
        - **impulsesstd** (*np.ndarray*) — ``(K, K, leads+1)`` array of
          asymptotic standard errors for the impulse responses.
        - **hfig** (*matplotlib.figure.Figure or None*) — Figure handle
          if ``graph=True``, otherwise ``None``.

    Raises
    ------
    ValueError
        If any input fails validation.

    Examples
    --------
    Compute IRF for 12 leads from a VAR(1) with a constant:

    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> y = rng.standard_normal((200, 3))
    >>> impulses, impulsesstd, hfig = impulseresponse(
    ...     y, 1, np.array([1]), 12, graph=0)

    Compute IRF from an irregular VAR with lags 1 and 3:

    >>> impulses, impulsesstd, hfig = impulseresponse(
    ...     y, 1, np.array([1, 3]), 12, graph=0)

    See Also
    --------
    vectorar : VAR estimation.
    mfe_toolbox.timeseries.vectorarvcv : VAR VCV estimation.
    mfe_toolbox.timeseries.grangercause : Granger causality tests.
    """
    # ==================================================================
    # Input Validation
    # Ref: impulseresponse.m:75-178
    # ==================================================================

    # ------------------------------------------------------------------
    # Validate y
    # Ref: impulseresponse.m:98-99
    # ------------------------------------------------------------------
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError('Y must be T by K')

    # Ref: impulseresponse.m:102 — Size of the cross-section
    K = y.shape[1]

    # ------------------------------------------------------------------
    # Validate constant
    # Ref: impulseresponse.m:104-109
    # ------------------------------------------------------------------
    if not np.isscalar(constant):
        raise ValueError('CONSTANT must be either 0 or 1')
    constant = int(constant)
    if constant not in (0, 1):
        raise ValueError('CONSTANT must be either 0 or 1')

    # ------------------------------------------------------------------
    # Validate lags
    # Ref: impulseresponse.m:111-126
    # ------------------------------------------------------------------
    lags = np.asarray(lags, dtype=np.float64).ravel()
    if lags.ndim != 1:
        raise ValueError(
            'LAGS must be a vector of positive integers containing '
            'lags to include'
        )
    # Ref: impulseresponse.m:117 — all elements positive
    if not np.all(lags > 0):
        raise ValueError(
            'LAGS must be a vector of positive integers containing '
            'lags to include'
        )
    # Ref: impulseresponse.m:120 — all elements are integers
    if not np.all(np.floor(lags) == lags):
        raise ValueError(
            'LAGS must be a vector of positive integers containing '
            'lags to include'
        )
    # Ref: impulseresponse.m:123 — unique elements
    if len(lags) != len(np.unique(lags)):
        raise ValueError('LAGS must be a vector of unique elements')
    # Ref: impulseresponse.m:126 — sort
    lags = np.sort(lags).astype(np.int64)

    # ------------------------------------------------------------------
    # Validate leads
    # Ref: impulseresponse.m:128-130
    # ------------------------------------------------------------------
    if not np.isscalar(leads) or leads < 1 or int(leads) != leads:
        raise ValueError('LEADS must be a positive scalar.')
    leads = int(leads)

    # ------------------------------------------------------------------
    # Validate sqrttype
    # Ref: impulseresponse.m:132-151
    # ------------------------------------------------------------------
    if sqrttype is None:
        sqrttype = 1
    user_cov_sqrt_provided = False
    if np.isscalar(sqrttype):
        sqrttype = int(sqrttype)
        if sqrttype not in (0, 1, 2, 3, 4):
            raise ValueError(
                'SQRTTYPE must be either a scalar (0,1,2,3,4) or a '
                'positive definite K by K matrix'
            )
    else:
        sqrttype = np.asarray(sqrttype, dtype=np.float64)
        # Ref: impulseresponse.m:141-142
        if sqrttype.ndim != 2:
            raise ValueError(
                'SQRTTYPE must be either a scalar (0,1,2,3) or a '
                'positive definite K by K matrix'
            )
        # Ref: impulseresponse.m:144-145
        if sqrttype.shape[0] != K or sqrttype.shape[0] != sqrttype.shape[1]:
            raise ValueError(
                'SQRTTYPE must be either a scalar (0,1,2,3) or a '
                'positive definite K by K matrix'
            )
        # Ref: impulseresponse.m:147 — min(eig(sqrttype)) > 0
        if np.min(np.linalg.eigvals(sqrttype).real) <= 0:
            raise ValueError(
                'SQRTTYPE must be either a scalar (0,1,2,3) or a '
                'positive definite K by K matrix'
            )
        user_cov_sqrt_provided = True

    # ------------------------------------------------------------------
    # Validate graph
    # Ref: impulseresponse.m:153-158
    # ------------------------------------------------------------------
    if graph is None:
        graph = True
    if np.isscalar(graph):
        graph_val = int(graph)
        if graph_val not in (0, 1):
            raise ValueError('GRAPH must be a scalar, either 1 or 0.')
        graph = bool(graph_val)
    else:
        raise ValueError('GRAPH must be a scalar, either 1 or 0.')

    # ------------------------------------------------------------------
    # Validate het
    # Ref: impulseresponse.m:160-168
    # ------------------------------------------------------------------
    if het is None:
        het = 1
    if not np.isscalar(het):
        raise ValueError('HET must be either 0 or 1')
    het = int(het)
    if het not in (0, 1):
        raise ValueError('HET must be either 0 or 1')

    # ------------------------------------------------------------------
    # Validate uncorr
    # Ref: impulseresponse.m:170-178
    # ------------------------------------------------------------------
    if uncorr is None:
        uncorr = 0
    if not np.isscalar(uncorr):
        raise ValueError('UNCORR must be either 0 or 1')
    uncorr = int(uncorr)
    if uncorr not in (0, 1):
        raise ValueError('UNCORR must be either 0 or 1')

    # ==================================================================
    # VAR Estimation
    # Ref: impulseresponse.m:184
    # ==================================================================
    (parameters, _stderr, _tstat, _pval, _const_est, _conststd,
     _r2, _errors, s2, _paramvec, vcv) = vectorar(
        y, constant, lags, het, uncorr
    )

    # ==================================================================
    # Lag structure handling
    # Ref: impulseresponse.m:187-194
    # ==================================================================
    maxlag = int(np.max(lags))

    # Ref: impulseresponse.m:189 — Find lags not included (irregular VAR)
    nolags = np.setdiff1d(np.arange(1, maxlag + 1), lags)

    # Ref: impulseresponse.m:192-194 — Fill missing parameter matrices
    # with zeros (K x K) so all lags from 1 to maxlag have a matrix.
    for lag in nolags:
        # Ref: impulseresponse.m:193 — parameters{nolags(i)} = zeros(K)
        parameters[lag - 1] = np.zeros((K, K))

    # ==================================================================
    # Covariance Square Root (sig12)
    # Ref: impulseresponse.m:196-220
    # ==================================================================
    if not user_cov_sqrt_provided:
        if sqrttype == 0:
            # Ref: impulseresponse.m:199 — Unit shocks
            sig12 = np.eye(K)
        elif sqrttype == 1:
            # Ref: impulseresponse.m:201 — Scaled but uncorrelated
            sig12 = np.diag(np.sqrt(np.diag(s2)))
        elif sqrttype == 2:
            # Ref: impulseresponse.m:203 — Cholesky decomposition
            # MATLAB chol(A) returns UPPER triangular R; chol(A)' is lower.
            # numpy cholesky returns LOWER triangular directly.
            sig12 = np.linalg.cholesky(s2)
        elif sqrttype == 3:
            # Ref: impulseresponse.m:205 — s2^(0.5) matrix square root
            # via spectral decomposition
            sig12 = np.real(scipy_linalg.sqrtm(s2))
        else:
            # Ref: impulseresponse.m:207-215 — sqrttype == 4
            # Generalized impulse response (Pesaran and Shin, 1998):
            # K reorderings of the covariance matrix where each variable
            # is ordered first when computing its shock column.
            sig12 = np.zeros((K, K))
            for i in range(K):
                # Ref: impulseresponse.m:209 — order = [i setdiff(1:K,i)]
                # Python 0-based: put variable i first, then the rest
                order = np.concatenate(
                    [[i], np.setdiff1d(np.arange(K), i)]
                ).astype(np.intp)
                # Ref: impulseresponse.m:210 — Reorder covariance matrix
                s2_temp = s2[np.ix_(order, order)]
                # Ref: impulseresponse.m:211 — sTemp = chol(s2Temp)'
                # MATLAB chol gives upper; ' gives lower. numpy gives lower.
                s_temp = np.linalg.cholesky(s2_temp)
                # Ref: impulseresponse.m:212 — Compute inverse permutation
                re_order = np.argsort(order)
                # Ref: impulseresponse.m:213 — Reorder back
                s_temp = s_temp[np.ix_(re_order, re_order)]
                # Ref: impulseresponse.m:214 — Take column i
                sig12[:, i] = s_temp[:, i]
    else:
        # Ref: impulseresponse.m:219 — User-provided matrix
        sig12 = sqrttype

    # ==================================================================
    # Xi Tensor: VMA(∞) representation via recursion
    # Ref: impulseresponse.m:222-237
    # ==================================================================
    # Ref: impulseresponse.m:223 — Initialize K x K x (leads+1) tensor
    Xi = np.zeros((K, K, leads + 1))

    for i in range(K):
        # Ref: impulseresponse.m:226 — Per-variable impulse accumulator
        this_impulse = np.zeros((leads + 1, K))
        # Ref: impulseresponse.m:227-229 — Unit shock to variable i
        # Ref: impulseresponse.m:228 — ei = zeros(K,1); ei(i)=1
        ei = np.zeros(K)
        ei[i] = 1.0
        # Ref: impulseresponse.m:230 — this_impulse(1,:) = shock'
        this_impulse[0, :] = ei

        # Ref: impulseresponse.m:231-235 — Recursive VMA computation
        for t in range(1, leads + 1):
            for j in range(1, min(t, maxlag) + 1):
                # Ref: impulseresponse.m:233
                # MATLAB: this_impulse(t+1,:) += (parameters{j}*this_impulse(t+1-j,:)')'
                # In Python 0-based: parameters[j-1] @ this_impulse[t-j, :]
                this_impulse[t, :] += (
                    parameters[j - 1] @ this_impulse[t - j, :]
                )

        # Ref: impulseresponse.m:236
        # MATLAB: Xi(:,i,:) = reshape(this_impulse', K, 1, leads+1)
        # this_impulse is (leads+1, K); .T is (K, leads+1)
        Xi[:, i, :] = this_impulse.T

    # ==================================================================
    # Compute Scaled Impulses: impulses = Xi * sig12 * ei for each shock
    # Ref: impulseresponse.m:248-256
    # ==================================================================
    impulses = np.zeros((K, K, leads + 1))
    for j_h in range(leads + 1):
        # Ref: impulseresponse.m:251-255 — Loop over shocks
        # Simplified: impulses(:,:,j) = Xi(:,:,j) * sig12
        # because Xi(:,:,j)*sig12*ei for each ei just selects columns
        impulses[:, :, j_h] = Xi[:, :, j_h] @ sig12

    # ==================================================================
    # G Matrix Construction (delta method gradient)
    # Ref: impulseresponse.m:259-276
    #
    # NOTE: The original MATLAB uses K*maxlag+1 for the RHS width,
    # which assumes a constant column is always present. To ensure
    # dimensional consistency when constant=0, we use K*maxlag+constant.
    # This matches the TFparameters mask dimension and the expanded vcv.
    # Ref: impulseresponse.m:244,264,280
    # ==================================================================
    n_cols_full = K * maxlag + constant
    nP_full = K * n_cols_full

    # Ref: impulseresponse.m:259 — Initialize G tensor
    G = np.zeros((K * K, nP_full, leads))

    for s in range(1, leads + 1):
        for i in range(1, s + 1):
            # Ref: impulseresponse.m:263 — LHS = impulses(:,:,i-1+1)
            # MATLAB 1-based impulses(:,:,i) → Python 0-based impulses[:,:,i-1]
            # The "+1" in the MATLAB source accounts for the 0-lag impulse
            # being stored at position 1 (1-based).
            LHS = impulses[:, :, i - 1]

            # Ref: impulseresponse.m:264 — RHS matrix
            RHS = np.zeros((K, n_cols_full))

            for j_lag in range(1, maxlag + 1):
                # Ref: impulseresponse.m:267 — Condition s-i+1 >= 1
                # (always true since i <= s, but kept for fidelity)
                if s - i + 1 >= 1:
                    # Ref: impulseresponse.m:268
                    # MATLAB: matrixToAdd = impulses(:,:,s-i+1)'
                    # MATLAB 1-based index s-i+1 → Python 0-based s-i
                    matrix_to_add = impulses[:, :, s - i].T
                else:
                    matrix_to_add = np.zeros((K, K))

                # Ref: impulseresponse.m:272
                # MATLAB: RHS(:,(j-1)*K+2:j*K+1) = matrixToAdd
                # Using constant-aware offset for dimensional consistency
                col_start = (j_lag - 1) * K + constant
                col_end = j_lag * K + constant
                RHS[:, col_start:col_end] = matrix_to_add

            # Ref: impulseresponse.m:274 — Accumulate Kronecker product
            # MATLAB 1-based G(:,:,s) → Python 0-based G[:,:,s-1]
            G[:, :, s - 1] += np.kron(LHS, RHS)

    # ==================================================================
    # TF Parameter Mask and VCV Expansion
    # Ref: impulseresponse.m:278-301
    # ==================================================================

    # Ref: impulseresponse.m:280 — Boolean mask of actual parameters
    # Shape: K × (K*maxlag + constant)
    TF = np.zeros((K, n_cols_full), dtype=bool)
    if constant:
        # Ref: impulseresponse.m:282 — Constant column is TRUE
        TF[:, 0] = True
    for i_lag in range(1, maxlag + 1):
        if i_lag in lags:
            # Ref: impulseresponse.m:286
            # MATLAB: TFparameters(:,(i-1)*K+constant+1:i*K+constant) = true
            col_start = (i_lag - 1) * K + constant
            col_end = i_lag * K + constant
            TF[:, col_start:col_end] = True

    # Ref: impulseresponse.m:289-291 — Vectorize column-major
    # MATLAB: TFparameters = TFparameters'; TFparameters = TFparameters(:)
    # In Python: transpose then ravel gives column-major (Fortran) ordering
    TF_vec = TF.T.ravel()
    pl = np.where(TF_vec)[0]

    # Ref: impulseresponse.m:296-301 — Expand vcv into full-dimension vcv2
    # Places the dense vcv values at positions corresponding to actual
    # (estimated) parameters in the expanded parameter space.
    vcv2 = np.zeros((nP_full, nP_full))
    for i_idx in range(len(pl)):
        # Ref: impulseresponse.m:298-299
        # MATLAB: temp = vcv(index,:); vcv2(pl(i),pl) = temp
        vcv2[pl[i_idx], pl] = vcv[i_idx, :]

    # ==================================================================
    # vecXivcv: Covariance of vectorized impulse responses (delta method)
    # Ref: impulseresponse.m:305-308
    # ==================================================================
    vecXivcv = np.zeros((K * K, K * K, leads + 1))
    for i_lead in range(G.shape[2]):
        # Ref: impulseresponse.m:307
        # MATLAB: vecXivcv(:,:,i+1) = G(:,:,i)*vcv2*G(:,:,i)'
        vecXivcv[:, :, i_lead + 1] = (
            G[:, :, i_lead] @ vcv2 @ G[:, :, i_lead].T
        )

    # ==================================================================
    # Impulse Response Standard Errors
    # Ref: impulseresponse.m:322-331
    # ==================================================================
    impulsesstd = np.zeros((K, K, leads + 1))
    for j_h in range(leads + 1):
        # Ref: impulseresponse.m:327 — stdErr = sqrt(diag(vecXivcv(:,:,j)))
        std_err = np.sqrt(np.diag(vecXivcv[:, :, j_h]))
        # Ref: impulseresponse.m:328 — stdErr = reshape(stdErr,K,K)'
        # MATLAB reshape fills column-major; Python order='F' replicates this
        std_err = std_err.reshape((K, K), order='F').T
        impulsesstd[:, :, j_h] = std_err

    # ==================================================================
    # Optional Plotting
    # Ref: impulseresponse.m:334-376
    # ==================================================================
    hfig = None
    if graph:
        # Ref: impulseresponse.m:335-336 — Axis limit accumulators
        LB_mat = np.zeros((K, K))
        UB_mat = np.zeros((K, K))
        # Ref: impulseresponse.m:337-338 — Create figure (800x600 pixels)
        hfig = plt.figure(figsize=(8, 6))
        x_vals = np.arange(leads + 1)

        for i_row in range(K):
            for j_col in range(K):
                # Ref: impulseresponse.m:342 — subplot(K,K,(i-1)*K+j)
                # Python 1-based subplot index
                ax = plt.subplot(K, K, i_row * K + j_col + 1)

                # Ref: impulseresponse.m:343 — Extract IRF and std
                irf = np.squeeze(impulses[i_row, j_col, :])
                irf_std = np.squeeze(impulsesstd[i_row, j_col, :])

                # Ref: impulseresponse.m:343 — Plot IRF, +/-1.96 bands, zero
                ax.plot(
                    x_vals, irf,
                    linewidth=2, color=(0.5, 0.5, 1.0)
                )
                # Ref: impulseresponse.m:344-345 — Upper confidence band
                ax.plot(
                    x_vals, irf + 1.96 * irf_std,
                    linewidth=2, color=(0.25, 0.25, 0.5), linestyle=':'
                )
                # Ref: impulseresponse.m:346 — Lower confidence band
                ax.plot(
                    x_vals, irf - 1.96 * irf_std,
                    linewidth=2, color=(0.25, 0.25, 0.5), linestyle=':'
                )
                # Ref: impulseresponse.m:347 — Zero reference line
                ax.plot(
                    [0, leads], [0, 0],
                    linewidth=1, color=(0, 0, 0), linestyle='-'
                )

                # Ref: impulseresponse.m:349 — Title on first row only
                if i_row == 0:
                    # Ref: impulseresponse.m:349 — 1-based shock label
                    plt.title('e_{}'.format(j_col + 1))
                # Ref: impulseresponse.m:352 — Y-label on first column only
                if j_col == 0:
                    # Ref: impulseresponse.m:352 — 1-based variable label
                    plt.ylabel('y_{}'.format(i_row + 1))

                # Ref: impulseresponse.m:354-361 — Axis tight with 5% padding
                ax.axis('tight')
                ax_limits = list(ax.axis())
                spread = ax_limits[3] - ax_limits[2]
                ax_limits[3] += 0.05 * spread
                ax_limits[2] -= 0.05 * spread
                ax.axis(ax_limits)
                LB_mat[i_row, j_col] = ax_limits[2]
                UB_mat[i_row, j_col] = ax_limits[3]

        # Ref: impulseresponse.m:363-364 — Consistent y-limits per row
        UB_row = np.max(UB_mat, axis=1)
        LB_row = np.min(LB_mat, axis=1)

        # Ref: impulseresponse.m:365-372 — Apply consistent row limits
        for i_row in range(K):
            for j_col in range(K):
                ax = hfig.axes[i_row * K + j_col]
                ax_limits = list(ax.axis())
                ax_limits[2] = LB_row[i_row]
                ax_limits[3] = UB_row[i_row]
                ax.axis(ax_limits)

    return impulses, impulsesstd, hfig
