"""
Bootstrap Confidence Bands for VAR Impulse Response Functions.

Migrated from ``timeseries/impulseresponse_bootstrap.m`` — Author: Kevin
Sheppard (Revision 3.0, Date: 1/1/2007)

Computes bootstrap confidence intervals for impulse responses of a
VAR(P) or irregular VAR(P) model. Three bootstrap schemes are supported:

1. **IID** — Independent and identically distributed resampling of VAR
   residuals (default).
2. **STATIONARY** — Politis-Romano stationary bootstrap with geometric
   block lengths, suitable for weakly dependent residuals.
3. **BLOCK** — Fixed-block-length circular block bootstrap following
   Künsch (1989).

The covariance identification method for shock scaling is determined by
the ``sqrttype`` parameter, which supports unit shocks, diagonal scaling,
Cholesky decomposition, spectral decomposition (matrix square root), and
the Generalized Impulse Response of Pesaran and Shin (1998).

Notes
-----
The bootstrap procedure:

1. Estimate the VAR model from the original data.
2. Compute the point-estimate impulse responses using the VMA(∞)
   recursive representation.
3. For each of *B* bootstrap replications:
   a. Resample VAR residuals using the selected bootstrap scheme.
   b. Construct synthetic VAR history from resampled residuals and
      the original estimated parameters.
   c. Re-estimate the VAR on the synthetic data.
   d. Compute impulse responses for the bootstrap replicate using the
      **original** covariance square root (sig12) for shock scaling.
4. Compute pointwise 2.5% and 97.5% quantile-based confidence intervals
   across bootstrap replicates.

References
----------
Sheppard, K. (2009). MFE Toolbox Version 4.0.
Lütkepohl, H. (2005). *New Introduction to Multiple Time Series
    Analysis*. Springer.
Politis, D.N. and Romano, J.P. (1994). The Stationary Bootstrap.
    *Journal of the American Statistical Association*, 89, 1303-1313.
Pesaran, M.H. and Shin, Y. (1998). Generalized impulse response
    analysis in linear multivariate models. *Economics Letters*,
    58(1), 17-29.

See Also
--------
vectorar : VAR estimation.
impulseresponse : VAR impulse response functions with asymptotic SE.
stationary_bootstrap : Politis-Romano stationary bootstrap.
block_bootstrap : Circular block bootstrap.
"""

import numpy as np
import warnings
import matplotlib.pyplot as plt

from mfe_toolbox.timeseries.vectorar import vectorar
from mfe_toolbox.timeseries.impulseresponse import impulseresponse
from mfe_toolbox.bootstrap.stationary_bootstrap import stationary_bootstrap
from mfe_toolbox.bootstrap.block_bootstrap import block_bootstrap

__all__ = ['impulseresponse_bootstrap']


def impulseresponse_bootstrap(
    y: np.ndarray,
    constant: int,
    lags: np.ndarray,
    leads: int,
    sqrttype=None,
    graph=None,
    bootstrap=None,
    B=None,
    w=None,
) -> tuple:
    """
    Compute bootstrap confidence bands for VAR impulse responses.

    Estimates a VAR model, computes h-step impulse response functions,
    and provides bootstrap confidence intervals via residual resampling.

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
        Whether to produce a plot of the IRF with bootstrap confidence
        bands. Default is ``True`` (``1``).
    bootstrap : str, optional
        Bootstrap scheme to use:

        - ``'IID'`` — **[DEFAULT]** IID resampling of residuals
        - ``'STATIONARY'`` — Stationary bootstrap with geometric blocks
        - ``'BLOCK'`` — Fixed block-length circular bootstrap
    B : int, optional
        Number of bootstrap replications. Default is ``1000``.
        Must be at least ``10``.
    w : int, optional
        Window/block length for STATIONARY or BLOCK bootstraps.
        Ignored for IID. Default is ``1``.

    Returns
    -------
    tuple
        A 4-element tuple:

        - **impulses** (*np.ndarray*) — ``(K, K, leads+1)`` array where
          ``impulses[i, j, h]`` is the response of variable ``i`` to
          shock ``j`` at horizon ``h`` (``h=0`` is contemporaneous).
        - **impulse_lower_ci** (*np.ndarray*) — ``(K, K, leads+1)``
          array of 2.5th percentile bootstrap lower bounds.
        - **impulse_upper_ci** (*np.ndarray*) — ``(K, K, leads+1)``
          array of 97.5th percentile bootstrap upper bounds.
        - **hfig** (*matplotlib.figure.Figure or None*) — Figure handle
          if ``graph=True``, otherwise ``None``.

    Raises
    ------
    ValueError
        If any input fails validation.

    Examples
    --------
    Compute bootstrap IRF for 12 leads from a VAR(1) with 500 reps:

    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> y = rng.standard_normal((200, 2))
    >>> imp, lo, hi, fig = impulseresponse_bootstrap(
    ...     y, 1, np.array([1]), 12, graph=0, B=100)
    >>> imp.shape
    (2, 2, 13)

    See Also
    --------
    vectorar : VAR estimation.
    impulseresponse : VAR impulse response with asymptotic standard errors.
    """
    # ==================================================================
    # Input Validation
    # Ref: impulseresponse_bootstrap.m:76-186
    # ==================================================================

    # ------------------------------------------------------------------
    # Validate y
    # Ref: impulseresponse_bootstrap.m:103-106
    # ------------------------------------------------------------------
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 2:
        # Ref: impulseresponse_bootstrap.m:105 — error('Y must be T by K')
        raise ValueError('Y must be T by K')

    T = y.shape[0]
    K = y.shape[1]

    # ------------------------------------------------------------------
    # Validate constant
    # Ref: impulseresponse_bootstrap.m:109-115
    # ------------------------------------------------------------------
    if not np.isscalar(constant):
        raise ValueError('CONSTANT must be either 0 or 1')
    constant = int(constant)
    if constant not in (0, 1):
        raise ValueError('CONSTANT must be either 0 or 1')

    # ------------------------------------------------------------------
    # Validate lags
    # Ref: impulseresponse_bootstrap.m:117-132
    # ------------------------------------------------------------------
    lags = np.asarray(lags, dtype=np.float64).ravel()
    if lags.ndim != 1:
        raise ValueError(
            'LAGS must be a vector of positive integers containing '
            'lags to include'
        )
    # Ref: impulseresponse_bootstrap.m:123 — all lags > 0
    if not np.all(lags > 0):
        raise ValueError(
            'LAGS must be a vector of positive integers containing '
            'lags to include'
        )
    # Ref: impulseresponse_bootstrap.m:126 — integer check
    if not np.all(np.floor(lags) == lags):
        raise ValueError(
            'LAGS must be a vector of positive integers containing '
            'lags to include'
        )
    # Ref: impulseresponse_bootstrap.m:129 — unique check
    if len(lags) != len(np.unique(lags)):
        raise ValueError('LAGS must be a vector of unique elements')
    # Ref: impulseresponse_bootstrap.m:132 — sort
    lags = np.sort(lags).astype(np.int64)

    # ------------------------------------------------------------------
    # Validate leads
    # Ref: impulseresponse_bootstrap.m:134-136
    # ------------------------------------------------------------------
    if not np.isscalar(leads) or leads < 1 or int(leads) != leads:
        raise ValueError('LEADS must be a positive scalar.')
    leads = int(leads)

    # ------------------------------------------------------------------
    # Validate sqrttype
    # Ref: impulseresponse_bootstrap.m:138-157
    # ------------------------------------------------------------------
    if sqrttype is None:
        sqrttype = 1
    user_cov_sqrt_provided = False
    if np.isscalar(sqrttype):
        sqrttype_val = int(sqrttype)
        if sqrttype_val not in (0, 1, 2, 3, 4):
            raise ValueError(
                'SQRTTYPE must be either a scalar (0,1,2,3,4) or a '
                'positive definite K by K matrix'
            )
    else:
        sqrttype = np.asarray(sqrttype, dtype=np.float64)
        sqrttype_val = None  # Flag: matrix mode
        # Ref: impulseresponse_bootstrap.m:147-148
        if sqrttype.ndim != 2:
            raise ValueError(
                'SQRTTYPE must be either a scalar (0,1,2,3) or a '
                'positive definite K by K matrix'
            )
        # Ref: impulseresponse_bootstrap.m:150-151
        if sqrttype.shape[0] != K or sqrttype.shape[0] != sqrttype.shape[1]:
            raise ValueError(
                'SQRTTYPE must be either a scalar (0,1,2,3) or a '
                'positive definite K by K matrix'
            )
        # Ref: impulseresponse_bootstrap.m:153 — min(eig(sqrttype)) > 0
        if np.min(np.linalg.eigvals(sqrttype).real) <= 0:
            raise ValueError(
                'SQRTTYPE must be either a scalar (0,1,2,3) or a '
                'positive definite K by K matrix'
            )
        user_cov_sqrt_provided = True

    # ------------------------------------------------------------------
    # Validate graph
    # Ref: impulseresponse_bootstrap.m:159-164
    # ------------------------------------------------------------------
    if graph is None:
        graph = True
    if np.isscalar(graph):
        graph_int = int(graph)
        if graph_int not in (0, 1):
            raise ValueError(
                'GRAPH must be a scalar, either 1 or 0 (true or false).'
            )
        graph = bool(graph_int)
    else:
        raise ValueError(
            'GRAPH must be a scalar, either 1 or 0 (true or false).'
        )

    # ------------------------------------------------------------------
    # Validate bootstrap
    # Ref: impulseresponse_bootstrap.m:166-172
    # ------------------------------------------------------------------
    if bootstrap is None:
        bootstrap = 'iid'
    bootstrap = str(bootstrap).lower()
    if bootstrap not in ('iid', 'stationary', 'block'):
        raise ValueError(
            "BOOTSTRAP must be either 'STATIONARY', 'BLOCK' or 'IID'"
        )

    # ------------------------------------------------------------------
    # Validate B
    # Ref: impulseresponse_bootstrap.m:174-179
    # ------------------------------------------------------------------
    if B is None:
        B = 1000
    B_int = int(B)
    if B_int < 10 or int(B) != B:
        raise ValueError(
            'B must be an integer greater than or equal to 10. The '
            'recommended value is 1000, and B should not be smaller '
            'than 100.'
        )

    # ------------------------------------------------------------------
    # Validate w
    # Ref: impulseresponse_bootstrap.m:181-186
    # ------------------------------------------------------------------
    if w is None:
        w = 1
    w_int = int(w)
    if w_int < 1 or int(w) != w:
        raise ValueError('W must be a positive integer.')

    # ==================================================================
    # VAR Estimation
    # Ref: impulseresponse_bootstrap.m:192
    # ==================================================================
    (parameters, _stderr, _tstat, _pval, const_est, _conststd,
     _r2, errors, s2, _paramvec, _vcv) = vectorar(y, constant, lags)

    # ------------------------------------------------------------------
    # Lag structure handling
    # Ref: impulseresponse_bootstrap.m:195-202
    # ------------------------------------------------------------------
    maxlag = int(np.max(lags))

    # Ref: impulseresponse_bootstrap.m:197 — Find lags not included
    # MATLAB: nolags = setdiff(1:maxlag, lags)
    # Python 0-based: lags from 1 to maxlag inclusive
    nolags = np.setdiff1d(np.arange(1, maxlag + 1), lags)

    # Ref: impulseresponse_bootstrap.m:200-202 — Fill missing parameter
    # matrices with zeros (K x K) for irregular VAR support
    for lag in nolags:
        # Ref: impulseresponse_bootstrap.m:201 — parameters{nolags(i)}=zeros(K)
        parameters[lag - 1] = np.zeros((K, K))

    # ==================================================================
    # Covariance Square Root (sig12)
    # Ref: impulseresponse_bootstrap.m:204-228
    # Used for scaling impulse responses throughout the bootstrap
    # ==================================================================
    if not user_cov_sqrt_provided:
        if sqrttype_val == 0:
            # Ref: impulseresponse_bootstrap.m:207 — Unit shocks
            sig12 = np.eye(K)
        elif sqrttype_val == 1:
            # Ref: impulseresponse_bootstrap.m:209 — Scaled but uncorrelated
            sig12 = np.diag(np.sqrt(np.diag(s2)))
        elif sqrttype_val == 2:
            # Ref: impulseresponse_bootstrap.m:211 — Cholesky decomposition
            # MATLAB chol(s2) returns UPPER triangular; chol(s2)' is lower.
            # numpy cholesky returns LOWER triangular directly.
            sig12 = np.linalg.cholesky(s2)
        elif sqrttype_val == 3:
            # Ref: impulseresponse_bootstrap.m:213 — s2^(0.5) spectral
            from scipy.linalg import sqrtm
            sig12 = np.real(sqrtm(s2))
        else:
            # Ref: impulseresponse_bootstrap.m:215-224 — sqrttype == 4
            # Generalized impulse response (Pesaran and Shin, 1998):
            # K reorderings where each variable is first for its shock
            sig12 = np.zeros((K, K))
            for i in range(K):
                # Ref: impulseresponse_bootstrap.m:217 — reorder
                order = np.concatenate(
                    [[i], np.setdiff1d(np.arange(K), i)]
                ).astype(np.intp)
                s2_temp = s2[np.ix_(order, order)]
                # Ref: impulseresponse_bootstrap.m:219 — chol(s2Temp)'
                s_temp = np.linalg.cholesky(s2_temp)
                # Ref: impulseresponse_bootstrap.m:220 — reorder back
                re_order = np.argsort(order)
                s_temp = s_temp[np.ix_(re_order, re_order)]
                sig12[:, i] = s_temp[:, i]
    else:
        # Ref: impulseresponse_bootstrap.m:227 — User-provided matrix
        sig12 = sqrttype

    # ==================================================================
    # Xi Tensor: VMA(∞) representation via recursion
    # Ref: impulseresponse_bootstrap.m:231-245
    # ==================================================================
    # Ref: impulseresponse_bootstrap.m:231 — Initialize K x K x (leads+1)
    Xi = np.zeros((K, K, leads + 1))

    for i in range(K):
        # Ref: impulseresponse_bootstrap.m:233 — Per-variable impulse
        this_impulse = np.zeros((leads + 1, K))
        # Ref: impulseresponse_bootstrap.m:234-236 — Unit shock ei
        ei = np.zeros(K)
        ei[i] = 1.0
        # Ref: impulseresponse_bootstrap.m:238 — Initial impulse
        # MATLAB 1-based: this_impulse(1,:) = shock'
        # Python 0-based: this_impulse[0, :] = ei
        this_impulse[0, :] = ei

        # Ref: impulseresponse_bootstrap.m:239-243 — Recursive VMA
        for t in range(1, leads + 1):
            for j in range(1, min(t, maxlag) + 1):
                # Ref: impulseresponse_bootstrap.m:241
                # MATLAB: this_impulse(t+1,:) += (parameters{j}*this_impulse(t+1-j,:)')'
                # Python 0-based: parameters[j-1] @ this_impulse[t-j, :]
                this_impulse[t, :] += (
                    parameters[j - 1] @ this_impulse[t - j, :]
                )

        # Ref: impulseresponse_bootstrap.m:244
        # MATLAB: Xi(:,i,:) = reshape(this_impulse', K, 1, leads+1)
        Xi[:, i, :] = this_impulse.T

    # ==================================================================
    # Bootstrap Residual Resampling
    # Ref: impulseresponse_bootstrap.m:248-311
    # ==================================================================
    T_eff = T - maxlag  # Effective sample size (number of errors)
    rng = np.random.default_rng()

    # Ref: impulseresponse_bootstrap.m:250-257 — Generate bootstrap indices
    # MATLAB passes (1:T-maxlag)' as data to bootstrap functions.
    # Python: pass np.arange(T_eff) and use returned values as 0-based indices.
    if bootstrap == 'stationary':
        # Ref: impulseresponse_bootstrap.m:252
        data_idx = np.arange(T_eff, dtype=np.float64).reshape(-1, 1)
        bs_data, _ = stationary_bootstrap(data_idx, B_int, w_int)
        boot_indices = bs_data.astype(np.int64)  # (T_eff, B), 0-based
    elif bootstrap == 'block':
        # Ref: impulseresponse_bootstrap.m:254
        data_idx = np.arange(T_eff, dtype=np.float64).reshape(-1, 1)
        bs_data, _ = block_bootstrap(data_idx, B_int, w_int)
        boot_indices = bs_data.astype(np.int64)  # (T_eff, B), 0-based
    else:
        # Ref: impulseresponse_bootstrap.m:256
        # MATLAB: indices = ceil(rand(T-maxlag,B)*(T-maxlag))  — 1-based
        # Python: 0-based integers in [0, T_eff)
        boot_indices = rng.integers(0, T_eff, size=(T_eff, B_int))

    # Ref: impulseresponse_bootstrap.m:259 — Prepend warm-up indices
    # MATLAB: indices = [indices(T-3*maxlag+1:T-maxlag,:); indices]
    # This prepends 2*maxlag rows from the tail of boot_indices
    n_warmup = min(2 * maxlag, T_eff)
    prefix = boot_indices[T_eff - n_warmup:, :]
    boot_indices = np.vstack([prefix, boot_indices])
    # Shape: (T_eff + n_warmup, B_int)

    # Ref: impulseresponse_bootstrap.m:261 — Random initial values
    # MATLAB: initialValues = ceil(rand(B,1)*(T-(maxlag-1)))+(maxlag-1)
    # Generates values from maxlag to T (1-based), i.e., maxlag-1 to T-1 (0-based)
    initial_values = rng.integers(maxlag - 1, T, size=B_int)

    # ------------------------------------------------------------------
    # Bootstrap loop
    # Ref: impulseresponse_bootstrap.m:262-311
    # ------------------------------------------------------------------
    final_bootstrap_impulses = np.zeros((K, K, leads + 1, B_int))

    for b_idx in range(B_int):
        try:
            # Ref: impulseresponse_bootstrap.m:265 — Resample errors
            # MATLAB: bootstrapErrors = errors(indices(:,i),:)
            bootstrap_errors = errors[boot_indices[:, b_idx], :]

            # Ref: impulseresponse_bootstrap.m:266-276 — Synthetic history
            temp_y = np.zeros((T + maxlag, K))

            # Ref: impulseresponse_bootstrap.m:267
            # ORIGINAL MATLAB uses initialValues(B) for ALL replicates (bug).
            # Comment "% FIX ME  Index problem" at line 264 confirms this.
            # Python uses initialValues[b_idx] (the intended per-replicate value).
            init_val = initial_values[b_idx]  # 0-based index
            # Ref: impulseresponse_bootstrap.m:267 — y(initVal+(-maxlag+1:0),:)
            # Python 0-based: y[init_val - maxlag + 1 : init_val + 1, :]
            temp_y[:maxlag, :] = y[init_val - maxlag + 1:init_val + 1, :]

            # Ref: impulseresponse_bootstrap.m:268-276 — Simulate VAR
            for t in range(maxlag, T + maxlag):
                # Ref: impulseresponse_bootstrap.m:269-271
                if constant:
                    temp_y[t, :] = const_est.copy()
                # else temp_y[t, :] remains zero from initialization

                # Ref: impulseresponse_bootstrap.m:272-274 — AR component
                for j_lag in range(1, maxlag + 1):
                    temp_y[t, :] += (
                        parameters[j_lag - 1] @ temp_y[t - j_lag, :]
                    )

                # Ref: impulseresponse_bootstrap.m:275 — Add resampled error
                # MATLAB: tempY(t,:) += bootstrapErrors(t-maxlag,:)
                # Python 0-based: index t - maxlag
                temp_y[t, :] += bootstrap_errors[t - maxlag, :]

            # Ref: impulseresponse_bootstrap.m:277 — Trim initial values
            # MATLAB: tempY = tempY(maxlag+1:T+maxlag,:)
            temp_y = temp_y[maxlag:, :]  # Shape: (T, K)

            # Ref: impulseresponse_bootstrap.m:278 — Re-estimate VAR
            boot_params_tuple = vectorar(temp_y, constant, lags)
            boot_parameters = boot_params_tuple[0]  # list of KxK matrices

            # Ref: impulseresponse_bootstrap.m:280-282 — Fill missing lags
            for lag in nolags:
                boot_parameters[lag - 1] = np.zeros((K, K))

            # Ref: impulseresponse_bootstrap.m:284-298 — Bootstrap Xi
            boot_xi = np.zeros((K, K, leads + 1))
            for ii in range(K):
                this_imp = np.zeros((leads + 1, K))
                ei_b = np.zeros(K)
                ei_b[ii] = 1.0
                this_imp[0, :] = ei_b
                for t in range(1, leads + 1):
                    for j_lag in range(1, min(t, maxlag) + 1):
                        # Ref: impulseresponse_bootstrap.m:294
                        this_imp[t, :] += (
                            boot_parameters[j_lag - 1]
                            @ this_imp[t - j_lag, :]
                        )
                boot_xi[:, ii, :] = this_imp.T

            # Ref: impulseresponse_bootstrap.m:301-309 — Scale by sig12
            # Impulses = Xi * sig12 * ei for each shock i, but this
            # simplifies to Xi * sig12 since we iterate over all ei.
            boot_impulses = np.zeros((K, K, leads + 1))
            for j_h in range(leads + 1):
                # Ref: impulseresponse_bootstrap.m:307
                # bootstrapImpulses(:,ii,j)=bootstrapXi(:,:,j)*sig12*ei
                # Vectorized: boot_impulses[:,:,j] = boot_xi[:,:,j] @ sig12
                boot_impulses[:, :, j_h] = boot_xi[:, :, j_h] @ sig12

            # Ref: impulseresponse_bootstrap.m:310
            final_bootstrap_impulses[:, :, :, b_idx] = boot_impulses

        except Exception as exc:
            # Handle bootstrap replicates that fail (e.g., singular matrix
            # during Cholesky or near-singular VAR re-estimation)
            warnings.warn(
                f'Bootstrap replicate {b_idx + 1} failed: {exc}. '
                f'Using NaN for this replicate.'
            )
            final_bootstrap_impulses[:, :, :, b_idx] = np.nan

    # ==================================================================
    # Bootstrap Confidence Intervals
    # Ref: impulseresponse_bootstrap.m:312-313
    # ==================================================================
    # MATLAB: impulseLowerCI = quantile(finalBootstrapImpulses, .025, 4)
    # Python: np.percentile with axis=3 (the B dimension)
    impulse_lower_ci = np.percentile(
        final_bootstrap_impulses, 2.5, axis=3
    )
    impulse_upper_ci = np.percentile(
        final_bootstrap_impulses, 97.5, axis=3
    )

    # ==================================================================
    # Compute Scaled Original Impulses
    # Ref: impulseresponse_bootstrap.m:325-334
    # Delegate to impulseresponse() for the point-estimate IRF.
    # This produces identical results to the inline VMA + sig12 scaling.
    # ==================================================================
    impulses_from_ir, _, _ = impulseresponse(
        y, constant, lags, leads, sqrttype, graph=False
    )
    impulses = impulses_from_ir

    # ==================================================================
    # Optional Plotting
    # Ref: impulseresponse_bootstrap.m:337-379
    # ==================================================================
    hfig = None
    if graph:
        # Ref: impulseresponse_bootstrap.m:338-339 — Axis limit accumulators
        LB_mat = np.zeros((K, K))
        UB_mat = np.zeros((K, K))

        # Ref: impulseresponse_bootstrap.m:340-341 — Create figure
        hfig = plt.figure(figsize=(8, 6))
        x_vals = np.arange(leads + 1)

        for i_row in range(K):
            for j_col in range(K):
                # Ref: impulseresponse_bootstrap.m:345
                # MATLAB: subplot(K,K,(i-1)*K+j)  — 1-based
                # Python: 1-based subplot index
                ax = plt.subplot(K, K, i_row * K + j_col + 1)

                # Extract IRF and CI data for this (i_row, j_col) pair
                irf = np.squeeze(impulses[i_row, j_col, :])
                lower = np.squeeze(impulse_lower_ci[i_row, j_col, :])
                upper = np.squeeze(impulse_upper_ci[i_row, j_col, :])

                # Ref: impulseresponse_bootstrap.m:346-350 — Plot lines
                # h(1): IRF line (blue-ish, solid, thick)
                ax.plot(
                    x_vals, irf,
                    linewidth=2, color=(0.5, 0.5, 1.0)
                )
                # h(2): Lower CI (darker blue, dotted)
                ax.plot(
                    x_vals, lower,
                    linewidth=2, color=(0.25, 0.25, 0.5), linestyle=':'
                )
                # h(3): Upper CI (darker blue, dotted)
                ax.plot(
                    x_vals, upper,
                    linewidth=2, color=(0.25, 0.25, 0.5), linestyle=':'
                )
                # h(4): Zero reference line (black, solid, thin)
                ax.plot(
                    [0, leads], [0, 0],
                    linewidth=1, color=(0, 0, 0), linestyle='-'
                )

                # Shade the confidence band for visual clarity
                ax.fill_between(
                    x_vals, lower, upper,
                    alpha=0.15, color=(0.25, 0.25, 0.5)
                )

                # Ref: impulseresponse_bootstrap.m:351-353 — Titles/labels
                if i_row == 0:
                    # Ref: impulseresponse_bootstrap.m:352
                    plt.title('e_{}'.format(j_col + 1))
                if j_col == 0:
                    # Ref: impulseresponse_bootstrap.m:354-355
                    plt.ylabel('y_{}'.format(i_row + 1))

                # Ref: impulseresponse_bootstrap.m:357-363 — Axis tight + pad
                ax.axis('tight')
                ax_limits = list(ax.axis())
                spread = ax_limits[3] - ax_limits[2]
                ax_limits[3] += 0.05 * spread
                ax_limits[2] -= 0.05 * spread
                ax.axis(ax_limits)
                LB_mat[i_row, j_col] = ax_limits[2]
                UB_mat[i_row, j_col] = ax_limits[3]

        # Ref: impulseresponse_bootstrap.m:366-367 — Consistent row limits
        UB_row = np.max(UB_mat, axis=1)
        LB_row = np.min(LB_mat, axis=1)

        # Ref: impulseresponse_bootstrap.m:368-376 — Apply row limits
        for i_row in range(K):
            for j_col in range(K):
                ax = hfig.axes[i_row * K + j_col]
                ax_limits = list(ax.axis())
                ax_limits[2] = LB_row[i_row]
                ax_limits[3] = UB_row[i_row]
                ax.axis(ax_limits)

        plt.tight_layout()

    return impulses, impulse_lower_ci, impulse_upper_ci, hfig
