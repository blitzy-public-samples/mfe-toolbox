"""
ARMAX(P,Q) estimation driver.

Migrated from: timeseries/armaxfilter.m (Kevin Sheppard, Revision 4, 10/19/2009)

This module provides the :func:`armaxfilter` function, the main estimation
driver for ARMAX(P,Q) models with optional exogenous regressors and
heteroscedastic errors (GLS).

The ARMAX model is::

    y(t) = const + Σ_i φ_i · y(t − p_i) + Σ_j β_j · x(t, j)
                 + Σ_l θ_l · e(t − q_l) + e(t)

When no MA terms are present (Q is empty), the model reduces to an AR(X)
regression estimated via OLS (``numpy.linalg.lstsq``).  When MA terms are
present, parameters are estimated via nonlinear least squares
(``scipy.optimize.least_squares``, replacing MATLAB ``lsqnonlin``).

Public API
----------
armaxfilter(y, constant, p, q, x, starting_vals, options, hold_back, sigma2)
    -> (parameters, LL, errors, se_regression, diagnostics, vcv_robust, vcv,
        likelihoods, scores)

See Also
--------
mfe_toolbox.timeseries.armaxerrors : ARMAX residual computation.
mfe_toolbox.timeseries.armaxfilter_likelihood : ARMAX Gaussian log-likelihood.
mfe_toolbox.timeseries.armaxfilter_simulate : ARMAX simulation.
"""

import warnings

import numpy as np
from scipy.optimize import least_squares

from mfe_toolbox.timeseries.armaxerrors import armaxerrors
from mfe_toolbox.timeseries.armaxfilter_likelihood import armaxfilter_likelihood
from mfe_toolbox.timeseries.convert_ma_roots import convert_ma_roots
from mfe_toolbox.timeseries.armaroots import armaroots
from mfe_toolbox.timeseries.aichqcsbic import aichqcsbic
from mfe_toolbox.utility.robustvcv import robustvcv
from mfe_toolbox.utility.newlagmatrix import newlagmatrix
from mfe_toolbox.utility.hessian_2sided import hessian_2sided
from mfe_toolbox.utility.gradient_2sided import gradient_2sided


def armaxfilter(y, constant, p, q, x=None, starting_vals=None, options=None,
                hold_back=None, sigma2=None):
    """
    Estimate an ARMAX(P,Q) model via OLS or nonlinear least squares.

    This is the main driver for ARMAX estimation.  When only AR (and
    optionally exogenous) terms are present, the model is estimated via OLS.
    When MA terms are present, ``scipy.optimize.least_squares`` is used
    (replacing MATLAB ``lsqnonlin``).

    Parameters
    ----------
    y : array_like
        T × 1 column vector of observations.
    constant : int
        Scalar indicator: ``1`` to include a constant term, ``0`` to exclude.
    p : array_like
        Non-negative integer vector of AR lag orders to include.
        Pass ``0`` or an empty array for no AR terms.
    q : array_like
        Non-negative integer vector of MA lag orders to include.
        Pass ``0`` or an empty array for no MA terms.
    x : array_like or None, optional
        T × K matrix of exogenous regressors.  These align directly with
        ``y``; if they are time series, they must be pre-shifted (lagged).
        ``None`` (default) means no exogenous regressors.
    starting_vals : array_like or None, optional
        1-D vector of starting values with length
        ``constant + len(p) + len(q) + K``.  Layout is
        ``[constant?, AR(1) … AR(nP), X(1) … X(K), MA(1) … MA(nQ)]``.
        ``None`` (default) triggers automatic starting value computation.
    options : dict or None, optional
        Dictionary of options passed to ``scipy.optimize.least_squares``.
        If ``None``, default options are used.
    hold_back : int or None, optional
        Number of observations to withhold at the start of the sample.
        Useful for producing comparable likelihoods across models with
        different lag lengths.  ``None`` (default) uses ``max(p)``.
    sigma2 : array_like or None, optional
        T × 1 vector of conditional variances for GLS estimation.
        ``None`` (default) uses ``ones(T)``.

    Returns
    -------
    parameters : numpy.ndarray
        Estimated parameter vector (same layout as *starting_vals*).
    LL : float
        Maximized log-likelihood value.
    errors : numpy.ndarray
        T × 1 vector of residuals from the estimated model.
    se_regression : float
        Standard error of the regression.
    diagnostics : dict
        Diagnostic information containing keys:
        ``'P'``, ``'Q'``, ``'C'``, ``'nX'``, ``'AIC'``, ``'HQC'``,
        ``'SBIC'``, ``'ADJT'``, ``'T'``, ``'ARROOTS'``, ``'ABSARROOTS'``,
        ``'holdBack'``.
    vcv_robust : numpy.ndarray
        Robust (sandwich) parameter variance-covariance matrix.
    vcv : numpy.ndarray
        Non-robust (inverse Hessian) variance-covariance matrix.
    likelihoods : numpy.ndarray
        T × 1 vector of per-observation log-likelihoods.
    scores : numpy.ndarray
        T × K matrix of numerical scores (per-observation, per-parameter).

    Raises
    ------
    ValueError
        If any input fails validation (see Notes).

    Notes
    -----
    * Parameter ordering and return shapes are identical to the original
      MATLAB function.
    * All MATLAB ``error()`` calls are translated to ``raise ValueError()``.
    * All MATLAB ``warning()`` calls are translated to ``warnings.warn()``.
    * Optimizer migration: ``lsqnonlin`` → ``scipy.optimize.least_squares``.
    * OLS fallback: when ``max(q) == 0``, uses ``numpy.linalg.lstsq``.
    * Numerical parity target: ``numpy.testing.assert_allclose(atol=1e-6,
      rtol=1e-4)`` against MATLAB reference.

    Examples
    --------
    Fit an ARMA(1,1):

    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> y = rng.standard_normal(500)
    >>> params, ll, *_ = armaxfilter(y, 1, np.array([1]), np.array([1]))

    Fit an AR(3) with no constant:

    >>> params, ll, *_ = armaxfilter(y, 0, np.array([1, 2, 3]), np.array([]))
    """

    # ==================================================================
    # Phase 1: Input Validation
    # Ref: armaxfilter.m lines 87-250
    # ==================================================================

    # ------------------------------------------------------------------
    # y validation
    # Ref: armaxfilter.m:132-136
    # ------------------------------------------------------------------
    y = np.asarray(y, dtype=np.float64).squeeze()
    if y.ndim == 0 or y.size <= 1:
        raise ValueError('y is empty.')
    if y.ndim != 1:
        raise ValueError('y series must be a column vector.')
    T_orig = y.shape[0]

    # ------------------------------------------------------------------
    # x validation
    # Ref: armaxfilter.m:128
    # ------------------------------------------------------------------
    if x is None:
        x = np.zeros((T_orig, 0), dtype=np.float64)
    else:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError('x must be a 2-D array.')
        if x.shape[0] != T_orig:
            raise ValueError(
                f'x must have the same number of rows as y. '
                f'x has {x.shape[0]} rows, y has {T_orig}.'
            )
    K = x.shape[1]

    # ------------------------------------------------------------------
    # p validation
    # Ref: armaxfilter.m:141-163
    # ------------------------------------------------------------------
    p = np.asarray(p, dtype=np.float64).ravel()
    if p.size == 0:
        # Ref: armaxfilter.m:144-146 — treat empty as 0
        p = np.array([0.0])
    if np.any(p < 0) or np.any(np.floor(p) != p):
        raise ValueError('P must contain non-negative integers only')
    maxp = int(np.max(p))
    if maxp > 0 and maxp >= (T_orig - maxp):
        raise ValueError('Too many lags in the AR.  max(P)<T/2')
    # Ref: armaxfilter.m:157-159 — if scalar 0, make empty
    if p.size == 1 and p[0] == 0:
        p = np.array([], dtype=np.float64)
    if len(np.unique(p)) != len(p):
        raise ValueError('P must contain at most one of each lag')
    nP = len(p)

    # ------------------------------------------------------------------
    # q validation
    # Ref: armaxfilter.m:167-189
    # ------------------------------------------------------------------
    q = np.asarray(q, dtype=np.float64).ravel()
    if q.size == 0:
        q = np.array([0.0])
    if np.any(q < 0) or np.any(np.floor(q) != q):
        raise ValueError('Q must contain non-negative integers only')
    maxq = int(np.max(q))
    if maxq >= T_orig:
        raise ValueError('Too many lags in the AR.  max(Q)<T')
    # Ref: armaxfilter.m:183-185 — if scalar 0, make empty
    if q.size == 1 and q[0] == 0:
        q = np.array([], dtype=np.float64)
    if len(np.unique(q)) != len(q):
        raise ValueError('Q must contain at most one of each lag')
    nQ = len(q)

    # ------------------------------------------------------------------
    # constant validation
    # Ref: armaxfilter.m:193-198
    # ------------------------------------------------------------------
    if not constant and nP == 0 and nQ == 0:
        raise ValueError('At least one of CONSTANT, P or Q must be nonnegative')
    if constant not in (0, 1):
        raise ValueError('CONSTANT must be 0 or 1')

    # ------------------------------------------------------------------
    # starting_vals validation
    # Ref: armaxfilter.m:203-215
    # ------------------------------------------------------------------
    if starting_vals is not None:
        starting_vals = np.asarray(starting_vals, dtype=np.float64).ravel()
        expected_len = nP + nQ + K + constant
        if len(starting_vals) != expected_len:
            raise ValueError(
                'STARTINGVALS must be a column vector with '
                'Constant+length(p)+length(q)+K elements'
            )
    # ------------------------------------------------------------------
    # options defaults
    # Ref: armaxfilter.m:220-232
    # ------------------------------------------------------------------
    if options is None and maxq > 0:
        # Ref: armaxfilter.m:229-231 — default options for lsqnonlin/least_squares
        options = {
            'max_nfev': 1000 * (maxp + maxq + constant + K) ** 2,
        }
    elif options is None:
        options = {}

    # ------------------------------------------------------------------
    # sigma2 validation
    # Ref: armaxfilter.m:237-246
    # ------------------------------------------------------------------
    if sigma2 is not None:
        sigma2 = np.asarray(sigma2, dtype=np.float64).ravel()
        if sigma2.shape[0] != T_orig or np.any(sigma2 < 0):
            raise ValueError(
                'SIGMA2 must be a T by 1 vector containing strictly '
                'positive numbers.'
            )
    else:
        sigma2 = np.ones(T_orig, dtype=np.float64)

    # ==================================================================
    # Phase 2: Setup
    # Ref: armaxfilter.m:252-275
    # ==================================================================
    T50 = max(int(np.ceil(np.log(T_orig))), 2)

    # Ref: armaxfilter.m:257 — zerosToPad = max(maxq - maxp, 0)
    zeros_to_pad = max(maxq - maxp, 0)
    m = max(maxp, maxq)

    # Ref: armaxfilter.m:259-265 — holdBack logic
    if hold_back is not None:
        hold_back = int(hold_back)
        if hold_back > maxp:
            m = m + (hold_back - maxp)
    else:
        hold_back = maxp

    # Ref: armaxfilter.m:266 — yAugmentedForMA
    y_aug = np.concatenate([np.zeros(zeros_to_pad), y])

    # Ref: armaxfilter.m:267-271 — xAugmentedForMA
    if K > 0:
        x_aug = np.vstack([np.zeros((zeros_to_pad, K)), x])
    else:
        x_aug = np.zeros(len(y_aug), dtype=np.float64)

    # Ref: armaxfilter.m:272 — sigmaAugmentedForMA = [ones(zerosToPad,1);sqrt(sigma2)]
    sigma_aug = np.concatenate([np.ones(zeros_to_pad), np.sqrt(sigma2)])

    # Ref: armaxfilter.m:273 — tempSigma = ones(size(yAugmentedForMA))
    temp_sigma = np.ones(len(y_aug), dtype=np.float64)

    # Ref: armaxfilter.m:274 — T updated to augmented length
    T = len(y_aug)

    # ==================================================================
    # Phase 3: Starting Values
    # Ref: armaxfilter.m:277-395
    # ==================================================================
    starting_vals_problem = False

    # Ref: armaxfilter.m:278-296 — validate user-provided starting values
    if starting_vals is not None and maxq > 0:
        # Ref: armaxfilter.m:279 — yTilde = y(holdBack+1:length(y))
        # MATLAB 1-indexed: holdBack+1 maps to Python 0-indexed: hold_back
        y_tilde = y[hold_back:]
        if constant:
            y_tilde = y_tilde - np.mean(y_tilde)
        worst_case_sse = np.dot(y_tilde, y_tilde)

        # Ref: armaxfilter.m:285-286 — check for invertible MA
        ma_start_idx = constant + nP + K
        ma_end_idx = constant + nP + K + nQ
        ma_parameters = starting_vals[ma_start_idx:ma_end_idx].copy()
        new_ma_parameters = convert_ma_roots(ma_parameters, q)
        if np.any(new_ma_parameters != ma_parameters):
            starting_vals = starting_vals.copy()
            starting_vals[ma_start_idx:ma_end_idx] = new_ma_parameters

        # Ref: armaxfilter.m:290 — compute SSE with starting values
        e = armaxerrors(starting_vals, p, q, constant, y_aug, x_aug, m,
                        temp_sigma)
        sse_starting = np.dot(e, e)

        # Ref: armaxfilter.m:292-295 — check if starting vals are worse than naive
        if np.isnan(sse_starting) or (worst_case_sse < sse_starting):
            starting_vals_problem = True
            warnings.warn(
                'User provided STARTINGVALS produce a higher SSE then a '
                'naive model and so are being ignored.',
                stacklevel=2
            )

    # ------------------------------------------------------------------
    # Estimation path selection
    # Ref: armaxfilter.m:302-395
    # ------------------------------------------------------------------
    if maxq == 0:
        # =============================================================
        # Pure AR/ARX: OLS solution
        # Ref: armaxfilter.m:302-313
        # =============================================================
        # Ref: armaxfilter.m:304 — [Y,regressors]=newlagmatrix(y,maxp,constant)
        Y, regressors = newlagmatrix(y, maxp, constant)

        if constant:
            # Ref: armaxfilter.m:306 — regressors = regressors(:,[1 p'+1])
            # MATLAB 1-indexed: column 1 is constant, p'+1 are AR lag columns
            # Python: column 0 is constant from newlagmatrix, p (1-based) maps
            # to column indices p (since newlagmatrix puts const at col 0)
            p_int = p.astype(int)
            cols = np.concatenate([[0], p_int])
            regressors = regressors[:, cols]
        else:
            # Ref: armaxfilter.m:308 — regressors = regressors(:,p')
            # When no constant, newlagmatrix returns lag columns starting at col 0
            # p is 1-based lag index; col for lag i is i-1
            p_int = p.astype(int)
            # Ref: armaxfilter.m:308 — MATLAB p' used as 1-based column indices
            # In MATLAB without constant, columns are lag 1, lag 2, ..., lag maxp
            # Column index for lag l is l-1 in Python (0-based)
            cols = p_int - 1
            regressors = regressors[:, cols]

        if K > 0:
            # Ref: armaxfilter.m:311 — regressors = [regressors x(maxp+1:T,:)]
            # MATLAB 1-indexed: maxp+1 maps to Python 0-indexed: maxp
            regressors = np.hstack([regressors, x[maxp:T_orig, :]])

        # Ref: armaxfilter.m:313 — parameters = regressors\Y
        Y_flat = Y.ravel()
        parameters, _, _, _ = np.linalg.lstsq(regressors, Y_flat, rcond=None)

    elif maxq > 0 and (starting_vals is None or starting_vals_problem):
        # =============================================================
        # MA present, need to compute starting values iteratively
        # Ref: armaxfilter.m:314-395
        # =============================================================

        # Ref: armaxfilter.m:316 — high-order AR approximation
        high_order = max(maxp, T50)
        Y_high, regressors_high = newlagmatrix(y, high_order, constant)
        if K > 0:
            # Ref: armaxfilter.m:318 — append x(max(maxp,T50)+1:size(x,1),:)
            regressors_high = np.hstack([
                regressors_high, x[high_order:T_orig, :]
            ])
        Y_high_flat = Y_high.ravel()
        b = np.linalg.lstsq(regressors_high, Y_high_flat, rcond=None)[0]
        e = Y_high_flat - regressors_high @ b

        # Ref: armaxfilter.m:324 — AR lag matrix for iterative starting values
        Y, regressors = newlagmatrix(y, maxp, constant)
        if constant:
            # Ref: armaxfilter.m:326
            p_int = p.astype(int)
            cols = np.concatenate([[0], p_int])
            regressors = regressors[:, cols]
        else:
            # Ref: armaxfilter.m:328
            p_int = p.astype(int)
            cols = p_int - 1
            regressors = regressors[:, cols]

        if K > 0:
            # Ref: armaxfilter.m:331
            regressors = np.hstack([regressors, x[maxp:T_orig, :]])

        starting_val_regressors = regressors.copy()
        Y_flat = Y.ravel()

        # Ref: armaxfilter.m:335-344 — initial OLS starting values
        if regressors.shape[1] > 0:
            b_init = np.linalg.lstsq(regressors, Y_flat, rcond=None)[0]
            starting_vals_0 = np.concatenate([b_init, np.zeros(nQ)])
            e2 = armaxerrors(starting_vals_0, p, q, constant, y_aug, x_aug,
                             m, temp_sigma)
            sse_0 = np.dot(e2, e2)
        else:
            e2 = Y_flat.copy()
            sse_0 = np.dot(e2, e2)
            starting_vals_0 = np.zeros(nQ)

        # Ref: armaxfilter.m:349-350
        count = 1
        max_iter = 20
        iterative_starting_vals = np.zeros((max_iter, constant + nP + K + nQ))
        iterative_sse = np.full(max_iter, sse_0 + 1.0)
        last_starting_vals = None

        # Ref: armaxfilter.m:352-379 — iterative starting value loop
        while count <= max_iter:
            # Ref: armaxfilter.m:353-355 — convergence check
            if count >= 3 and last_starting_vals is not None:
                if np.max(np.abs(starting_vals - last_starting_vals)) < 0.01:
                    break

            # Ref: armaxfilter.m:357-358 — pad e to match y length
            if len(e) < T_orig:
                e = np.concatenate([np.zeros(T_orig - len(e)), e])

            # Ref: armaxfilter.m:361-362 — pad if maxq > maxp
            if maxq > maxp:
                e = np.concatenate([np.zeros(maxq - maxp), e])

            # Ref: armaxfilter.m:365 — build error lag matrix
            _, elags = newlagmatrix(e, max(maxq, maxp), 0)

            # Ref: armaxfilter.m:367 — select correct columns: q' (1-based)
            # Python: q is 1-based lag index → column index q_int - 1
            q_int = q.astype(int)
            elags = elags[:, q_int - 1]

            # Ref: armaxfilter.m:368
            combined_regressors = np.hstack([starting_val_regressors, elags])

            # Ref: armaxfilter.m:370-371
            if count > 1:
                last_starting_vals = starting_vals.copy()

            # Ref: armaxfilter.m:373 — OLS
            starting_vals = np.linalg.lstsq(
                combined_regressors, Y_flat, rcond=None
            )[0]

            # Ref: armaxfilter.m:374
            e2 = armaxerrors(starting_vals, p, q, constant, y_aug, x_aug,
                             m, temp_sigma)
            # Ref: armaxfilter.m:375-376 — MATLAB 1-indexed count
            iterative_starting_vals[count - 1, :] = starting_vals
            iterative_sse[count - 1] = np.dot(e2, e2)

            # Ref: armaxfilter.m:377 — update error estimate
            e = Y_flat - combined_regressors @ starting_vals
            count += 1

        # Ref: armaxfilter.m:380-384 — select best starting values
        min_idx = int(np.argmin(iterative_sse))
        min_sse = iterative_sse[min_idx]
        starting_vals = iterative_starting_vals[min_idx, :].copy()
        if sse_0 < min_sse:
            starting_vals = starting_vals_0.copy()

        # Ref: armaxfilter.m:385-394 — check MA invertibility of starting vals
        ma_params_full = np.zeros(maxq)
        q_int = q.astype(int)
        # Ref: armaxfilter.m:386 — MATLAB 1-indexed: MAparameters(q) = ...
        # Python 0-indexed: ma_params_full[q_int - 1]
        ma_start = constant + nP + K
        ma_end = constant + nP + K + nQ
        ma_params_full[q_int - 1] = starting_vals[ma_start:ma_end]

        poly_coeffs = np.concatenate([[1.0], ma_params_full])
        roots_val = np.roots(poly_coeffs)
        if len(roots_val) > 0 and np.max(np.abs(roots_val)) > 1.0:
            # Ref: armaxfilter.m:389-393 — try to invert
            ma_params_sel = starting_vals[ma_start:ma_end].copy()
            new_ma_params = convert_ma_roots(ma_params_sel, q)
            if np.any(new_ma_params != ma_params_sel):
                starting_vals = starting_vals.copy()
                starting_vals[ma_start:ma_end] = new_ma_params

    # ==================================================================
    # Phase 4: Nonlinear Optimization (when MA terms present)
    # Ref: armaxfilter.m:398-417
    # ==================================================================
    if maxq > 0:
        # Ref: armaxfilter.m:400 — lsqnonlin replacement with least_squares
        # MATLAB: lsqnonlin('armaxerrors', startingVals, [], [], options, ...)
        # Python: least_squares(fun, x0, args=..., **options)

        def _objective(params):
            """Wrapper for least_squares: returns residual vector."""
            return armaxerrors(params, p, q, constant, y_aug, x_aug, m,
                              sigma_aug)

        ls_options = {}
        if 'max_nfev' in options:
            ls_options['max_nfev'] = options['max_nfev']
        # Ref: armaxfilter.m:231 — Display='iter' not applicable to scipy
        ls_options['verbose'] = options.get('verbose', 0)

        result = least_squares(
            _objective,
            starting_vals,
            method='trf',
            **ls_options
        )
        parameters = result.x.copy()

        # Ref: armaxfilter.m:402-417 — check MA invertibility post-optimization
        ma_params_full = np.zeros(maxq)
        q_int = q.astype(int)
        ma_start = constant + nP + K
        ma_end = constant + nP + K + nQ
        ma_params_full[q_int - 1] = parameters[ma_start:ma_end]

        poly_coeffs = np.concatenate([[1.0], ma_params_full])
        roots_val = np.roots(poly_coeffs)
        if len(roots_val) > 0 and np.max(np.abs(roots_val)) > 1.0:
            # Ref: armaxfilter.m:408-409 — try to convert
            ma_params_sel = parameters[ma_start:ma_end].copy()
            new_ma_params = convert_ma_roots(ma_params_sel, q)

            if np.any(new_ma_params != ma_params_sel):
                parameters[ma_start:ma_end] = new_ma_params
            else:
                # Ref: armaxfilter.m:414 — warning when not invertible
                warnings.warn(
                    'MA parameters are not invertible, and it is not possible '
                    'to invert them with in the selected irregular MA '
                    'specification.',
                    stacklevel=2
                )

            # Ref: armaxfilter.m:416 — re-optimize with converted parameters
            result = least_squares(
                _objective,
                parameters,
                method='trf',
                **ls_options
            )
            parameters = result.x.copy()

    # ==================================================================
    # Phase 5: Post-Estimation
    # Ref: armaxfilter.m:426-429
    # ==================================================================
    # Ref: armaxfilter.m:426 — compute final likelihood
    neg_ll, neg_likelihoods, errors_full = armaxfilter_likelihood(
        parameters, p, q, constant, y_aug, x_aug, m, sigma_aug
    )

    # Ref: armaxfilter.m:427-428 — negate to get positive log-likelihood
    likelihoods = -neg_likelihoods
    LL = float(-neg_ll)

    # Ref: armaxfilter.m:429 — SE regression
    n_params = len(parameters)
    se_regression = float(np.sqrt(
        np.dot(errors_full, errors_full) / (len(errors_full) - n_params)
    ))

    # ==================================================================
    # Phase 6: Diagnostics
    # Ref: armaxfilter.m:431-447
    # ==================================================================
    diagnostics = {}
    diagnostics['P'] = p.copy() if len(p) > 0 else np.array([], dtype=np.float64)
    diagnostics['Q'] = q.copy() if len(q) > 0 else np.array([], dtype=np.float64)
    # Ref: armaxfilter.m:434 — adjT = length(y) - holdBack
    diagnostics['adjT'] = T_orig - hold_back
    diagnostics['T'] = T_orig
    diagnostics['K'] = K

    # Ref: armaxfilter.m:437 — information criteria
    aic, hqc, sbic = aichqcsbic(errors_full, constant, p, q, x)
    diagnostics['AIC'] = aic
    diagnostics['HQC'] = hqc
    diagnostics['SBIC'] = sbic
    diagnostics['C'] = constant
    diagnostics['nX'] = K

    # Ref: armaxfilter.m:443 — AR roots
    ar_roots, abs_ar_roots = armaroots(parameters, constant, p, q, x)
    diagnostics['ARROOTS'] = ar_roots
    diagnostics['ABSARROOTS'] = abs_ar_roots
    diagnostics['holdBack'] = hold_back

    # ==================================================================
    # Phase 7: Variance-Covariance Estimation
    # Ref: armaxfilter.m:449-480
    # ==================================================================
    if maxq == 0:
        # Ref: armaxfilter.m:451-475 — analytical VCV for pure AR/ARX
        _, lags = newlagmatrix(y, maxp, 0)
        # Ref: armaxfilter.m:453 — lags = lags(:,p)
        # MATLAB 1-indexed p used as column selector
        p_int = p.astype(int)
        lags = lags[:, p_int - 1]

        tau = lags.shape[0]
        if constant:
            # Ref: armaxfilter.m:456
            lags = np.hstack([np.ones((tau, 1)), lags])
        if K > 0:
            # Ref: armaxfilter.m:459 — X = [lags x(T-tau+1:T,:)]
            # MATLAB 1-indexed: T-tau+1 maps to Python: T_orig - tau
            X_mat = np.hstack([lags, x[T_orig - tau:T_orig, :]])
        else:
            X_mat = lags

        # Ref: armaxfilter.m:465-466 — trim errors to post-maxp
        T_e = len(errors_full)
        e_trimmed = errors_full[maxp:T_e]
        T_eff = len(e_trimmed)

        # Ref: armaxfilter.m:468 — XpXi = (X'*X/T)\eye(size(X,2))
        n_cols = X_mat.shape[1]
        XpX_over_T = (X_mat.T @ X_mat) / T_eff
        XpXi = np.linalg.inv(XpX_over_T)

        # Ref: armaxfilter.m:469-473 — sandwich middle term
        XeeX = np.zeros((n_cols, n_cols))
        for t in range(T_eff):
            # Ref: armaxfilter.m:471 — MATLAB 1-indexed loop t=1:T
            x_t = X_mat[t, :].reshape(-1, 1)
            XeeX += e_trimmed[t] ** 2 * (x_t @ x_t.T)
        XeeX /= T_eff

        # Ref: armaxfilter.m:474 — VCVrobust = XpXi*XeeX*XpXi/T
        vcv_robust = (XpXi @ XeeX @ XpXi) / T_eff

        # Ref: armaxfilter.m:475 — VCV = SEregression^2 * XpXi / T
        vcv = (se_regression ** 2) * XpXi / T_eff
    else:
        # Ref: armaxfilter.m:477-478 — robust VCV via robustvcv for MA models
        # robustvcv expects fun(theta, *args) -> (scalar, array)
        # armaxfilter_likelihood returns 3 values (llf, likelihoods, errors);
        # we must wrap it so robustvcv only sees the first 2 (MATLAB nargout=2).
        def _ll_for_robustvcv(params, *fn_args):
            """Wrapper returning (scalar_ll, per_obs_ll) for robustvcv."""
            ll_s, ll_v, _ = armaxfilter_likelihood(params, *fn_args)
            return ll_s, ll_v

        vcv_robust, _, B_mat, scores_mat, _, _ = robustvcv(
            _ll_for_robustvcv, parameters, 0,
            p, q, constant, y_aug, x_aug, m, sigma_aug
        )

        # Ref: armaxfilter.m:478 — VCV = B^(-1)/(T-m)
        # B here is the A matrix (Hessian/T) from robustvcv
        # Actually in the MATLAB: VCV = B^(-1)/(T-m) where B is the 2nd return
        # from robustvcv. robustvcv returns (VCV, A, B, scores, hess, gs)
        # In MATLAB: [VCVrobust,~,B,scores]=robustvcv(...)
        # The "B" in MATLAB code is the 3rd return value = B (score covariance)
        # But MATLAB line 478 says VCV=B^(-1)/(T-m), which uses the Hessian A
        # Actually re-reading: MATLAB returns [VCVrobust,A,B,scores] from robustvcv
        # Line 477: [VCVrobust,~,B,scores] = robustvcv(...) — skips A, takes B
        # Line 478: VCV = B^(-1)/(T-m)
        # This uses B = the score covariance matrix as the VCV base
        # Wait, that doesn't make sense. Let me re-read the MATLAB.
        #
        # Ref: robustvcv.m returns [VCV, A, B, scores]
        # MATLAB line 477: [VCVrobust, ~, B, scores] = robustvcv(...)
        #   VCVrobust = 1st return (sandwich VCV)
        #   ~ = 2nd return (A = Hessian/T, skipped)
        #   B = 3rd return (score covariance)
        #   scores = 4th return
        # MATLAB line 478: VCV = B^(-1)/(T-m)
        #   This inverts the score covariance matrix — but that seems wrong
        #   for a standard Hessian-based VCV. Let's check what B actually is.
        #   In robustvcv.m:72, A = hess/T, and hess is the full Hessian.
        #   In robustvcv.m:77, B = cov(scores) when nw=0.
        #   So VCV = cov(scores)^(-1)/(T-m).
        #   Actually that IS the Hessian-based VCV if we think about it differently:
        #   The MATLAB source has VCV=B^(-1)/(T-m) which uses B=cov(scores),
        #   but a more natural interpretation is VCV = A^(-1)/T = Hessian^(-1)/T^2
        #   Let me just faithfully translate it.
        vcv = np.linalg.inv(B_mat) / (T - m)
        scores_mat = scores_mat  # already T x K

    # Assemble scores output
    if maxq == 0:
        # For AR models, compute numerical scores via gradient_2sided
        def _ll_scalar(params):
            ll_val, _, _ = armaxfilter_likelihood(
                params, p, q, constant, y_aug, x_aug, m, sigma_aug
            )
            return float(ll_val)

        def _ll_individual(params):
            ll_val, ll_vec, _ = armaxfilter_likelihood(
                params, p, q, constant, y_aug, x_aug, m, sigma_aug
            )
            return float(ll_val), ll_vec

        _, scores_mat = gradient_2sided(
            _ll_individual, parameters, compute_scores=True
        )

    return (parameters, LL, errors_full, se_regression, diagnostics,
            vcv_robust, vcv, likelihoods, scores_mat)
