"""
Scalar DCC(m,n) and ADCC(m,l,n) multivariate volatility model estimation.

Estimates a scalar Dynamic Conditional Correlation model with TARCH(p,o,q) or
GJR-GARCH(p,o,q) univariate conditional variances.  Two estimation strategies
are supported:

- **3-stage** (default):
    1. Fit per-series TARCH/GJR-GARCH via :func:`dcc_fit_variance`.
    2. Estimate correlation intercept *R* from standardised residuals.
    3. Estimate DCC dynamics (alpha, gamma, beta) via constrained
       optimisation.

- **2-stage**:
    1. Fit per-series TARCH/GJR-GARCH via :func:`dcc_fit_variance`.
    2. Jointly estimate correlation intercept *R* and dynamics.

Three composite-likelihood options are available:

- ``'None'`` — standard QMLE.
- ``'Diagonal'`` — adjacent-pair composite likelihood.
- ``'Full'`` — all-pairs composite likelihood.

The DCC Q(t) dynamics follow:

.. math::

    Q_t = \\text{intercept} + \\sum_{i=1}^{m} a_i \\, z_{t-i} z_{t-i}'
          + \\sum_{i=1}^{l} g_i \\, n_{t-i} n_{t-i}'
          + \\sum_{i=1}^{n} b_i \\, Q_{t-i}

with conditional correlation:

.. math::

    R_t = \\text{diag}(Q_t)^{-1/2} \\, Q_t \\, \\text{diag}(Q_t)^{-1/2}

and conditional covariance:

.. math::

    H_t = \\Sigma_t \\, R_t \\, \\Sigma_t

Migrated from ``multivariate/dcc.m`` (473 lines) — MFE Toolbox Version 4.0.

Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 1    Date: 4/13/2012

Python migration: Blitzy Platform — MATLAB-to-Python 3.12 migration.

See Also
--------
mfe_toolbox.multivariate.ccc_mvgarch : CCC-MVGARCH model.
mfe_toolbox.multivariate.bekk : BEKK model.
mfe_toolbox.multivariate.rarch : RARCH model.
mfe_toolbox.multivariate.scalar_vt_vech : Scalar VT-VECH model.
mfe_toolbox.univariate.tarch : Underlying per-series volatility model.
"""

import warnings

import numpy as np
from scipy.optimize import minimize

from mfe_toolbox.multivariate.dcc_fit_variance import dcc_fit_variance
from mfe_toolbox.multivariate.dcc_likelihood import dcc_likelihood
from mfe_toolbox.multivariate.dcc_inference_objective import dcc_inference_objective
from mfe_toolbox.utility.cov2corr import cov2corr
from mfe_toolbox.utility.r2z import r2z
from mfe_toolbox.utility.z2r import z2r
from mfe_toolbox.utility.corr_vech import corr_vech
from mfe_toolbox.utility.vech import vech
from mfe_toolbox.utility.hessian_2sided_nrows import hessian_2sided_nrows
from mfe_toolbox.utility.gradient_2sided import gradient_2sided
from mfe_toolbox.utility.covnw import covnw

__all__ = ['dcc']


def dcc(
    data: np.ndarray,
    data_asym: np.ndarray | None = None,
    m: int = 1,
    l: int = 0,
    n: int = 1,
    p: np.ndarray | int | None = None,
    o: np.ndarray | int | None = None,
    q: np.ndarray | int | None = None,
    gjr_type: np.ndarray | int | None = None,
    method: str | None = None,
    composite: str | None = None,
    starting_vals: np.ndarray | None = None,
    options: dict | None = None,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Estimate scalar DCC(m,n) or ADCC(m,l,n) multivariate volatility model.

    Parameters
    ----------
    data : numpy.ndarray
        T × K matrix of zero-mean residuals **or** K × K × T array of
        covariance estimators (e.g. realised covariance).
    data_asym : numpy.ndarray or None, optional
        K × K × T array of asymmetric covariance estimators.  Only required
        when *data* is 3-dimensional and ``l > 0`` or any ``o > 0``.
    m : int, optional
        Order of symmetric innovations in the DCC model.  Default 1.
    l : int, optional
        Order of asymmetric innovations in the ADCC model.  Default 0.
    n : int, optional
        Order of lagged correlation in the DCC model.  Default 1.
    p : int or array_like or None, optional
        Positive integer (or K-element vector) — number of symmetric
        innovation lags per series for the univariate TARCH models.
        Default 1.
    o : int or array_like or None, optional
        Non-negative integer (or K-element vector) — number of asymmetric
        innovation lags per series.  Default 0.
    q : int or array_like or None, optional
        Non-negative integer (or K-element vector) — number of lagged
        conditional variance terms per series.  Default 1.
    gjr_type : int or array_like or None, optional
        Model type code per series: 1 (TARCH/AVGARCH) or 2 (GJR-GARCH).
        Default 2.
    method : str or None, optional
        Estimation method: ``'3-stage'`` (default) or ``'2-stage'``.
    composite : str or None, optional
        Composite likelihood mode: ``'None'`` (default), ``'Diagonal'``
        or ``'Full'``.
    starting_vals : numpy.ndarray or None, optional
        Starting values for all model parameters.  If ``None``, a grid
        search is performed.
    options : dict or None, optional
        Options dict passed to ``scipy.optimize.minimize``.

    Returns
    -------
    parameters : numpy.ndarray
        Estimated parameter vector.  Layout depends on *method*:

        - 3-stage: ``[VOL(1)...VOL(K), corr_vech(R), vech(N), alpha, gamma, beta]``
        - 2-stage: ``[VOL(1)...VOL(K), corr_vech(R), alpha, gamma, beta]``

        where ``VOL(j)`` is a ``(1 + p_j + o_j + q_j)``-element vector.
    ll : float
        Maximised log-likelihood value (positive).
    Ht : numpy.ndarray
        K × K × T array of conditional covariance matrices.
    VCV : numpy.ndarray
        ``numParams × numParams`` robust sandwich variance-covariance matrix
        (A⁻¹ B A⁻¹ / T).
    scores : numpy.ndarray
        T × numParams matrix of individual observation scores.
    diagnostics : dict
        Dictionary containing estimation diagnostics including ``'stage'``,
        ``'R'`` (unconditional correlation), ``'N'`` (asymmetric intercept),
        ``'univariate'`` (per-series TARCH results), and ``'g_scale'``.

    Raises
    ------
    ValueError
        If input dimensions or parameter values are invalid.

    Notes
    -----
    The dynamics of the correlations in a DCC model are:

    **3-stage:**

    .. math::

        Q_t = R (1 - \\sum a - \\sum b) - (\\sum g) N
              + \\sum_{i=1}^{m} a_i \\, e_{t-i} e_{t-i}'
              + \\sum_{i=1}^{l} g_i \\, v_{t-i} v_{t-i}'
              + \\sum_{i=1}^{n} b_i \\, Q_{t-i}

    **2-stage:**

    .. math::

        Q_t = R \\cdot \\text{scale} + \\text{dynamic terms}

    where ``v(t,:) = e(t,:) .* (e(t,:) < 0)`` and
    ``scale = sqrt(1 - sum(a) - sum(b) - gScale*sum(g))``.

    Examples
    --------
    >>> import numpy as np
    >>> # DCC(1,1)
    >>> # parameters = dcc(data, None, 1, 0, 1)
    >>> # ADCC(1,1,1)
    >>> # parameters = dcc(data, None, 1, 1, 1)
    >>> # ADCC(1,1,1), 2-stage
    >>> # parameters = dcc(data, None, 1, 1, 1, method='2-stage')

    References
    ----------
    .. [1] Engle, R. (2002). "Dynamic Conditional Correlation: A Simple
       Class of Multivariate Generalized Autoregressive Conditional
       Heteroskedasticity Models." *Journal of Business & Economic
       Statistics*, 20(3), 339-350.
    .. [2] Cappiello, L., Engle, R. F., and Sheppard, K. (2006).
       "Asymmetric Dynamics in the Correlations of Global Equity and Bond
       Returns." *Journal of Financial Econometrics*, 4(4), 537-572.
    """
    # ==================================================================
    # Input Argument Checking — Ref: dcc.m:77-286
    # ==================================================================
    data = np.asarray(data, dtype=np.float64)

    # ------------------------------------------------------------------
    # Handle 2-D data (T×K returns) vs 3-D data (K×K×T covariances)
    # Ref: dcc.m:131-149
    # ------------------------------------------------------------------
    if data.ndim == 2:
        # Ref: dcc.m:132-140 — T×K matrix of zero-mean residuals
        T, k = data.shape
        data2d = data.copy()
        # Ref: dcc.m:134 — eta = data.*(data<0); negative part of returns
        eta = data2d * (data2d < 0).astype(np.float64)
        # Ref: dcc.m:135-140 — construct K×K×T outer product arrays
        data_3d = np.zeros((k, k, T), dtype=np.float64)
        if data_asym is None:
            data_asym_3d = np.zeros((k, k, T), dtype=np.float64)
        else:
            data_asym_3d = np.asarray(data_asym, dtype=np.float64)
        for t in range(T):
            # Ref: dcc.m:138 — data(:,:,t) = data2d(t,:)' * data2d(t,:)
            data_3d[:, :, t] = np.outer(data2d[t, :], data2d[t, :])
            # Ref: dcc.m:139 — dataAsym(:,:,t) = eta(t,:)' * eta(t,:)
            if data_asym is None:
                data_asym_3d[:, :, t] = np.outer(eta[t, :], eta[t, :])
    elif data.ndim == 3:
        # Ref: dcc.m:141-146 — K×K×T array of covariance estimators
        k = data.shape[0]
        T = data.shape[2]
        data_3d = data.copy()
        if data_asym is None:
            raise ValueError(
                'DATAASYM is required when DATA is a 3-dimensional array.'
            )
        data_asym_3d = np.asarray(data_asym, dtype=np.float64)
        # Ref: dcc.m:143-146 — reconstruct pseudo-returns from diagonal elements
        # Note: MATLAB original has a bug (zeros(k,T) instead of zeros(T,k));
        # corrected here to produce T×k matrix as required by dcc_fit_variance.
        data2d = np.zeros((T, k), dtype=np.float64)
        for i in range(k):
            # Ref: dcc.m:145 — signed sqrt of diagonal variance
            diag_var = data_3d[i, i, :]     # shape (T,)
            diag_asym = data_asym_3d[i, i, :]  # shape (T,)
            sign_vec = 1.0 - 2.0 * (diag_asym == 0).astype(np.float64)
            data2d[:, i] = sign_vec * np.sqrt(np.abs(diag_var))
    else:
        # Ref: dcc.m:148
        raise ValueError(
            'DATA must be either a T x K matrix or a K x K x T '
            '3-dimensional array.'
        )

    # ------------------------------------------------------------------
    # Validate DATAASYM (if 3-D was provided externally)
    # Ref: dcc.m:151-159
    # ------------------------------------------------------------------
    if data_asym is not None and data.ndim == 2:
        data_asym_check = np.asarray(data_asym, dtype=np.float64)
        if data_asym_check.ndim != 3:
            raise ValueError('DATAASYM must be a K x K x T array.')
        k1, k2, T2 = data_asym_check.shape
        if k1 != k or k2 != k or T2 != T:
            raise ValueError('DATAASYM must be a K x K x T array.')
        data_asym_3d = data_asym_check

    # ------------------------------------------------------------------
    # Validate DCC orders: m, l, n
    # Ref: dcc.m:161-175
    # ------------------------------------------------------------------
    if not isinstance(m, (int, np.integer)) or m < 1:
        raise ValueError('M must be a positive integer.')
    if l is None:
        l = 0
    if not isinstance(l, (int, np.integer)) or l < 0:
        raise ValueError('L must be a non-negative integer.')
    if n is None:
        n = 0
    if not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError('N must be a non-negative integer.')
    # Cast to Python int for safety
    m = int(m)
    l = int(l)
    n = int(n)

    # ------------------------------------------------------------------
    # Validate and expand GARCH orders: p, o, q
    # Ref: dcc.m:177-212
    # ------------------------------------------------------------------
    if p is None:
        p = np.ones(k, dtype=np.int64)
    else:
        p = np.atleast_1d(np.asarray(p, dtype=np.int64)).ravel()
        if p.size == 1:
            # Ref: dcc.m:181 — scalar → K-vector
            p = np.full(k, int(p[0]), dtype=np.int64)
    if o is None:
        o = np.zeros(k, dtype=np.int64)
    else:
        o = np.atleast_1d(np.asarray(o, dtype=np.int64)).ravel()
        if o.size == 1:
            # Ref: dcc.m:187 — scalar → K-vector
            o = np.full(k, int(o[0]), dtype=np.int64)
    if q is None:
        q = np.ones(k, dtype=np.int64)
    else:
        q = np.atleast_1d(np.asarray(q, dtype=np.int64)).ravel()
        if q.size == 1:
            # Ref: dcc.m:193 — scalar → K-vector
            q = np.full(k, int(q[0]), dtype=np.int64)

    # Ref: dcc.m:195-203 — validate p, o, q element-wise
    if np.any(p != np.floor(p)) or np.any(p < 1) or p.size != k:
        raise ValueError(
            'All elements in P must be positive, and P must be either '
            'scalar or K by 1.'
        )
    if np.any(o != np.floor(o)) or np.any(o < 0) or o.size != k:
        raise ValueError(
            'All elements in O must be non-negative, and O must be either '
            'scalar or K by 1.'
        )
    if np.any(q != np.floor(q)) or np.any(q < 0) or q.size != k:
        raise ValueError(
            'All elements in Q must be non-negative, and Q must be either '
            'scalar or K by 1.'
        )

    # ------------------------------------------------------------------
    # Validate and expand gjr_type
    # Ref: dcc.m:204-212
    # ------------------------------------------------------------------
    if gjr_type is None:
        gjr_type = np.full(k, 2, dtype=np.int64)
    else:
        gjr_type = np.atleast_1d(np.asarray(gjr_type, dtype=np.int64)).ravel()
        if gjr_type.size == 1:
            # Ref: dcc.m:208 — scalar → K-vector
            gjr_type = np.full(k, int(gjr_type[0]), dtype=np.int64)
    if not np.all(np.isin(gjr_type, [1, 2])) or gjr_type.size != k:
        raise ValueError(
            'GJRTYPE must be in {1, 2} and must be either scalar or K by 1.'
        )

    # ------------------------------------------------------------------
    # Validate METHOD
    # Ref: dcc.m:214-225
    # ------------------------------------------------------------------
    if method is None:
        method = '3-stage'
    method = method.lower()
    if method not in ('3-stage', '2-stage'):
        raise ValueError("METHOD must be either '3-stage' or '2-stage'.")
    stage = 3 if method == '3-stage' else 2

    # ------------------------------------------------------------------
    # Validate COMPOSITE
    # Ref: dcc.m:228-245
    # ------------------------------------------------------------------
    if composite is None:
        composite = 'none'
    composite = composite.lower()
    if composite not in ('none', 'diagonal', 'full'):
        raise ValueError(
            "COMPOSITE must be one of 'None', 'Diagonal' or 'Full'."
        )
    if composite == 'none':
        composite_int = 0
    elif composite == 'diagonal':
        composite_int = 1
    else:
        composite_int = 2
    # Ref: dcc.m:242-244 — upgrade diagonal to full in 2-stage mode
    if stage == 2 and composite_int == 1:
        warnings.warn(
            "When METHOD is '2-stage', COMPOSITE must be either 'None' or "
            "'Full'. Upgrading COMPOSITE to 'Full'.",
            stacklevel=2,
        )
        composite_int = 2

    # ------------------------------------------------------------------
    # Validate STARTING_VALS and split into TARCH + DCC components
    # Ref: dcc.m:247-273
    # ------------------------------------------------------------------
    tarch_starting_vals = None
    dcc_starting_vals = None
    if starting_vals is not None:
        starting_vals = np.atleast_1d(
            np.asarray(starting_vals, dtype=np.float64)
        ).ravel()
        # Ref: dcc.m:248-254 — expected parameter count
        count = int(k + np.sum(p) + np.sum(o) + np.sum(q))
        count += m + l + n
        if stage == 2 or l == 0:
            # Ref: dcc.m:251 — K(K-1)/2 correlation intercept elements
            count += k * (k - 1) // 2
        elif stage == 3:
            # Ref: dcc.m:253 — K(K+1)/2 + K(K-1)/2 for N + R intercepts
            count += k * (k + 1) // 2 + k * (k - 1) // 2
        if starting_vals.size != count:
            raise ValueError(
                'STARTING_VALS does not contain the correct number of '
                'parameters.'
            )
        # Ref: dcc.m:258-269 — split starting values
        tarch_count = int(k + np.sum(p) + np.sum(o) + np.sum(q))
        tarch_starting_vals = starting_vals[:tarch_count]
        offset_sv = tarch_count
        if stage == 2 or l == 0:
            intercept_count = k * (k - 1) // 2
        elif stage == 3:
            intercept_count = k * (k + 1) // 2 + k * (k - 1) // 2
        else:
            intercept_count = 0
        offset_sv += intercept_count
        dcc_count = m + l + n
        dcc_starting_vals = starting_vals[offset_sv:offset_sv + dcc_count]

    # ------------------------------------------------------------------
    # Validate / set default OPTIONS
    # Ref: dcc.m:276-286
    # ------------------------------------------------------------------
    if options is not None:
        if not isinstance(options, dict):
            raise ValueError(
                'OPTIONS does not appear to be a valid options dictionary.'
            )
        scipy_options = options.copy()
    else:
        # Ref: dcc.m:277-281 — default fmincon options → scipy equivalents
        scipy_options = {
            'maxiter': 1000,
            'ftol': 1e-9,
            'disp': False,
        }

    # ==================================================================
    # Stage 1 — Univariate Volatility Models
    # Ref: dcc.m:287-298
    # ==================================================================
    H_var, univariate = dcc_fit_variance(
        data2d, p, o, q, gjr_type, tarch_starting_vals
    )

    # Ref: dcc.m:291-298 — standardise 3-D data by conditional volatilities
    std_data = data_3d.copy()
    std_data_asym = data_asym_3d.copy()
    for t in range(T):
        # Ref: dcc.m:294 — h = sqrt(H(t,:))
        h = np.sqrt(H_var[t, :])
        # Ref: dcc.m:295 — hh = h'*h (K×K outer product)
        hh = np.outer(h, h)
        # Ref: dcc.m:296-297 — element-wise division by hh
        std_data[:, :, t] = std_data[:, :, t] / hh
        std_data_asym[:, :, t] = std_data_asym[:, :, t] / hh

    # ==================================================================
    # Back-casts
    # Ref: dcc.m:300-309
    # ==================================================================
    # Ref: dcc.m:302 — exponentially decaying weights w = .06*.94.^(0:sqrt(T))
    n_weights = int(np.floor(np.sqrt(T))) + 1
    w = 0.06 * (0.94 ** np.arange(n_weights))
    w = w / np.sum(w)
    # Ref: dcc.m:304-309 — weighted average of initial observations
    back_cast = np.zeros((k, k), dtype=np.float64)
    back_cast_asym = np.zeros((k, k), dtype=np.float64)
    for i in range(len(w)):
        back_cast += w[i] * std_data[:, :, i]
        back_cast_asym += w[i] * std_data_asym[:, :, i]

    # ==================================================================
    # Correlation Intercept R and Asymmetric Intercept N
    # Ref: dcc.m:312-317
    # ==================================================================
    # Ref: dcc.m:313 — R = mean(stdData,3); (mean across 3rd dimension)
    R = np.mean(std_data, axis=2)
    # Ref: dcc.m:314-315 — normalise R to a proper correlation matrix
    r_diag = np.sqrt(np.diag(R))
    R = R / np.outer(r_diag, r_diag)
    # Ref: dcc.m:316 — N = mean(stdDataAsym,3)
    N = np.mean(std_data_asym, axis=2)

    # Ref: dcc.m:317 — scale = max(eig(R^{-0.5} * N * R^{-0.5}))
    # Compute R^{-1/2} via eigendecomposition of R (symmetric PD)
    eigvals_R, eigvecs_R = np.linalg.eigh(R)
    # Clamp eigenvalues to avoid sqrt of negative due to numerical noise
    eigvals_R = np.maximum(eigvals_R, 1e-14)
    R_neg_half = eigvecs_R @ np.diag(1.0 / np.sqrt(eigvals_R)) @ eigvecs_R.T
    scale_matrix = R_neg_half @ N @ R_neg_half
    # Ref: dcc.m:317 — max(eig(...)); use eig (not eigh) per schema requirement
    scale_eigvals = np.linalg.eig(scale_matrix)[0].real
    scale = float(np.max(scale_eigvals))

    # ==================================================================
    # Starting Values — Grid Search
    # Ref: dcc.m:319-354
    # ==================================================================
    if dcc_starting_vals is not None:
        sv_opt = dcc_starting_vals.copy()
    else:
        # Ref: dcc.m:324-335 — grid search over alpha, gamma (if asymmetric),
        # and theta (total persistence)
        a_grid = np.array([0.01, 0.03, 0.05, 0.1])
        theta_grid = np.array([0.99, 0.97, 0.95])
        if l > 0:
            # Ref: dcc.m:327-330 — ADCC grid with gamma
            g_grid = np.array([0.01, 0.03, 0.05])
            aa, gg, tt = np.meshgrid(a_grid, g_grid, theta_grid, indexing='ij')
            candidates = np.unique(
                np.column_stack([
                    aa.ravel(),
                    gg.ravel(),
                    tt.ravel() - aa.ravel() - scale * gg.ravel()
                ]),
                axis=0,
            )
            is_asym = 1
        else:
            # Ref: dcc.m:332-334 — symmetric DCC grid
            aa, tt = np.meshgrid(a_grid, theta_grid, indexing='ij')
            candidates = np.unique(
                np.column_stack([aa.ravel(), tt.ravel() - aa.ravel()]),
                axis=0,
            )
            is_asym = 0

        # Ref: dcc.m:336-345 — evaluate likelihood at each candidate
        min_ll = np.inf
        sv_opt = candidates[0, :].copy()
        is_joint_gs = False
        is_inference_gs = False
        for i in range(candidates.shape[0]):
            try:
                ll_val = dcc_likelihood(
                    candidates[i, :],
                    std_data, std_data_asym,
                    1, is_asym, 1,   # m=1, l=is_asym, n=1 for grid search
                    R, N,
                    back_cast, back_cast_asym,
                    3, composite_int,
                    is_joint_gs, is_inference_gs,
                    None, None,      # g_scale and univariate not needed
                )[0]
            except Exception:
                ll_val = np.inf
            if ll_val < min_ll:
                sv_opt = candidates[i, :].copy()
                min_ll = ll_val

        # Ref: dcc.m:346-354 — expand best grid values to full m/l/n orders
        a_sv = sv_opt[0]
        if l > 0:
            g_sv = sv_opt[1]
            b_sv = sv_opt[2]
            # Ref: dcc.m:350 — distribute equally across orders
            sv_opt = np.concatenate([
                np.full(m, a_sv / m),
                np.full(l, g_sv / l),
                np.full(n, b_sv / n),
            ])
        else:
            b_sv = sv_opt[1]
            # Ref: dcc.m:353
            sv_opt = np.concatenate([
                np.full(m, a_sv / m),
                np.full(n, b_sv / n),
            ])

    # ==================================================================
    # Bounds and Constraints for DCC Dynamics Optimisation
    # Ref: dcc.m:356-364
    # ==================================================================
    n_dcc_params = len(sv_opt)
    # Ref: dcc.m:356-357 — LB=0, UB=1 for all DCC dynamics parameters
    bounds_list = [(1e-8, 0.9999)] * n_dcc_params

    # Ref: dcc.m:358-360 — linear inequality constraint A*x <= b
    # where A(m+1:m+l) = scale, rest are ones
    A_constr = np.ones(n_dcc_params, dtype=np.float64)
    A_constr[m:m + l] = scale  # weight asymmetric terms by eigenvalue scale
    b_constr = 0.99998

    # Ref: dcc.m:362-364 — check starting values satisfy constraint
    if np.dot(sv_opt, A_constr) >= b_constr:
        # Scale back starting values to satisfy the constraint
        sv_opt = sv_opt * (0.9 * b_constr / np.dot(sv_opt, A_constr))

    # scipy SLSQP inequality constraint: c(x) >= 0  ⟺  b - A·x >= 0
    stationarity_constraint = {
        'type': 'ineq',
        'fun': lambda x: b_constr - np.dot(A_constr, x),
    }

    # ==================================================================
    # Stage 3 — DCC Dynamics Estimation
    # Ref: dcc.m:366-368
    # ==================================================================
    result = minimize(
        lambda x, *args: dcc_likelihood(x, *args)[0],
        sv_opt,
        args=(
            std_data, std_data_asym, m, l, n,
            R, N, back_cast, back_cast_asym,
            3, composite_int,
            False, False,      # is_joint=False, is_inference=False
            None, None,        # g_scale, univariate not needed for stage 3
        ),
        method='SLSQP',
        bounds=bounds_list,
        constraints=stationarity_constraint,
        options=scipy_options,
    )
    dcc_params = result.x

    # Ref: dcc.m:370-373 — extract dynamics parameters and compute g_scale
    g_scale = np.diag(N)  # Ref: dcc.m:370 — gScale = diag(N)
    alpha_params = dcc_params[:m]
    gamma_params = dcc_params[m:m + l]
    beta_params = dcc_params[m + l:m + l + n]

    # ==================================================================
    # Stage 2 — Joint Estimation of R and Dynamics (if method='2-stage')
    # Ref: dcc.m:375-390
    # ==================================================================
    if stage == 2:
        # Ref: dcc.m:376 — intercept = R*(1-sum(a)-sum(b)) - N*sum(g)
        intercept = (
            R * (1.0 - np.sum(alpha_params) - np.sum(beta_params))
            - N * np.sum(gamma_params)
        )
        # Ref: dcc.m:377 — [~, rescaledIntercept] = cov2corr(intercept)
        _, rescaled_intercept = cov2corr(intercept)
        # Ref: dcc.m:378 — z = r2z(rescaledIntercept)
        z = r2z(rescaled_intercept)

        # Ref: dcc.m:379 — joint starting values: [z', dynamics]
        sv_joint = np.concatenate([z, dcc_params])
        n_z = k * (k - 1) // 2
        n_dynamics = len(dcc_params)

        # Ref: dcc.m:381-384 — joint bounds and constraints
        bounds_joint = (
            [(-np.inf, np.inf)] * n_z
            + [(1e-8, 0.9999)] * n_dynamics
        )
        A_constr_joint = np.concatenate([
            np.zeros(n_z), A_constr
        ])
        b_constr_joint = 0.99998

        stationarity_constraint_joint = {
            'type': 'ineq',
            'fun': lambda x: b_constr_joint - np.dot(A_constr_joint, x),
        }

        # Ref: dcc.m:385 — joint optimisation with stage=2
        result_joint = minimize(
            lambda x, *args: dcc_likelihood(x, *args)[0],
            sv_joint,
            args=(
                std_data, std_data_asym, m, l, n,
                R, N, back_cast, back_cast_asym,
                2, composite_int,
                False, False,  # is_joint=False, is_inference=False
                g_scale, None,  # g_scale needed for stage 2; univariate not
            ),
            method='SLSQP',
            bounds=bounds_joint,
            constraints=stationarity_constraint_joint,
            options=scipy_options,
        )
        joint_params = result_joint.x

        # Ref: dcc.m:386-389 — extract z and dynamics from joint result
        z_opt = joint_params[:n_z]
        R = z2r(z_opt)
        dcc_params = joint_params[n_z:]
        # Ref: dcc.m:389 — parameters = [corr_vech(R)', dynamics]
        dcc_params = np.concatenate([corr_vech(R), dcc_params])

    # ==================================================================
    # Construct Full Parameter Vector
    # Ref: dcc.m:392-407
    # ==================================================================
    # Ref: dcc.m:395-398 — concatenate per-series GARCH parameters
    garch_parameters = np.concatenate(
        [univariate[i]['parameters'].ravel() for i in range(k)]
    )

    if stage == 3:
        # Ref: dcc.m:399-404 — 3-stage: [GARCH, corr_vech(R), vech(N), dynamics]
        if l > 0:
            # Ref: dcc.m:401 — vech returns 2-D column; ravel for concatenation
            parameters = np.concatenate([
                garch_parameters, corr_vech(R), vech(N).ravel(), dcc_params
            ])
        else:
            # Ref: dcc.m:403
            parameters = np.concatenate([
                garch_parameters, corr_vech(R), dcc_params
            ])
    elif stage == 2:
        # Ref: dcc.m:406 — 2-stage: [GARCH, (already contains corr_vech(R))]
        parameters = np.concatenate([garch_parameters, dcc_params])

    # ==================================================================
    # Final Joint Likelihood Evaluation
    # Ref: dcc.m:409-417
    # ==================================================================
    is_joint_final = True
    is_inference_final = True
    # Ref: dcc.m:411 — uses original (non-standardised) 3-D data
    ll_neg, _, Rt = dcc_likelihood(
        parameters, data_3d, data_asym_3d, m, l, n,
        None, None,  # R and N extracted from parameters when is_joint=True
        back_cast, back_cast_asym,
        stage, composite_int,
        is_joint_final, is_inference_final,
        g_scale, univariate,
    )
    # Ref: dcc.m:412 — negate to get positive log-likelihood
    ll = -ll_neg

    # Ref: dcc.m:413-417 — construct K×K×T conditional covariance Ht
    Ht = np.zeros((k, k, T), dtype=np.float64)
    for t in range(T):
        # Ref: dcc.m:415 — h = sqrt(H(t,:))
        h = np.sqrt(H_var[t, :])
        # Ref: dcc.m:416 — Ht(:,:,t) = Rt(:,:,t) .* (h'*h)
        Ht[:, :, t] = Rt[:, :, t] * np.outer(h, h)

    # ==================================================================
    # Inference — Robust Sandwich VCV: A^{-1} B A^{-1}' / T
    # Ref: dcc.m:418-471
    # ==================================================================
    v = len(parameters)
    A_info = np.zeros((v, v), dtype=np.float64)
    scores_mat = np.zeros((T, v), dtype=np.float64)
    offset = 0

    # Ref: dcc.m:430-437 — populate GARCH blocks from univariate results
    for i in range(k):
        u = univariate[i]
        count = 1 + int(u['p']) + int(u['o']) + int(u['q'])
        ind = slice(offset, offset + count)
        A_info[ind, ind] = u['A']
        scores_mat[:, ind] = u['scores']
        offset += count

    # ------------------------------------------------------------------
    # Wrappers for hessian / gradient that return correct shapes
    # dcc_likelihood returns (ll, lls, Rt); hessian needs scalar,
    # gradient with scores needs (scalar, T-array)
    # dcc_inference_objective returns (obj, lls); hessian needs scalar
    # ------------------------------------------------------------------
    def _ll_scalar(params, *args):
        """Wrapper returning only the scalar log-likelihood."""
        return dcc_likelihood(params, *args)[0]

    def _ll_scores(params, *args):
        """Wrapper returning (scalar, T-array) for gradient_2sided."""
        result_ll = dcc_likelihood(params, *args)
        return result_ll[0], result_ll[1]

    def _inf_scalar(params, *args):
        """Wrapper returning only the scalar objective for dcc_inference_objective."""
        return dcc_inference_objective(params, *args)[0]

    if stage == 2:
        # Ref: dcc.m:440-449 — 2-stage inference
        count = k * (k - 1) // 2 + m + l + n

        # Ref: dcc.m:443 — Hessian of dcc_likelihood w.r.t. last 'count' params
        hess = hessian_2sided_nrows(
            _ll_scalar, parameters, count,
            data_3d, data_asym_3d, m, l, n,
            None, None,
            back_cast, back_cast_asym,
            stage, composite_int,
            is_joint_final, is_inference_final,
            g_scale, univariate,
        )
        # Ref: dcc.m:444 — A(offset+(1:count), :) = H/T
        A_info[offset:offset + count, :] = hess / T

        # Ref: dcc.m:445 — gradient with per-observation scores
        _, s = gradient_2sided(
            _ll_scores, parameters,
            data_3d, data_asym_3d, m, l, n,
            None, None,
            back_cast, back_cast_asym,
            stage, composite_int,
            is_joint_final, is_inference_final,
            g_scale, univariate,
            compute_scores=True,
        )
        # Ref: dcc.m:446 — only take columns for the DCC block
        scores_mat[:, offset:offset + count] = s[:, offset:offset + count]

        # Ref: dcc.m:447 — B = cov(scores); sample covariance of all scores
        B_cov = np.cov(scores_mat, rowvar=False)
        # Ref: dcc.m:448 — Ainv = A \ eye(v)
        Ainv = np.linalg.solve(A_info, np.eye(v))
        # Ref: dcc.m:449 — VCV = Ainv * B * Ainv' / T
        VCV = Ainv @ B_cov @ Ainv.T / T

    elif stage == 3:
        # Ref: dcc.m:450-470 — 3-stage inference

        # Block 1: dcc_inference_objective (correlation intercept R, and N if l>0)
        # Ref: dcc.m:452-454 — count for correlation intercept block
        count_intercept = k * (k - 1) // 2
        if l > 0:
            count_intercept += k * (k + 1) // 2

        # Ref: dcc.m:456 — subset of parameters up to intercept block
        temp_params = parameters[:offset + count_intercept]

        # Ref: dcc.m:457 — gradient of dcc_inference_objective
        _, s_inf = gradient_2sided(
            dcc_inference_objective, temp_params,
            data_3d, data_asym_3d, m, l, n, univariate,
            compute_scores=True,
        )
        # Ref: dcc.m:458 — insert intercept scores
        scores_mat[:, offset:offset + count_intercept] = (
            s_inf[:, offset:offset + count_intercept]
        )

        # Ref: dcc.m:459 — Hessian of dcc_inference_objective
        hess_inf = hessian_2sided_nrows(
            _inf_scalar, temp_params, count_intercept,
            data_3d, data_asym_3d, m, l, n, univariate,
        )
        # Ref: dcc.m:460 — A(offset+(1:count), 1:(count+offset)) = H/T
        A_info[
            offset:offset + count_intercept,
            :offset + count_intercept
        ] = hess_inf / T
        offset += count_intercept

        # Block 2: dcc_likelihood (DCC dynamics a, g, b)
        # Ref: dcc.m:462 — count = m + l + n
        count_dynamics = m + l + n

        # Ref: dcc.m:464 — Hessian of dcc_likelihood for dynamics block
        hess_ll = hessian_2sided_nrows(
            _ll_scalar, parameters, count_dynamics,
            data_3d, data_asym_3d, m, l, n,
            None, None,
            back_cast, back_cast_asym,
            stage, composite_int,
            is_joint_final, is_inference_final,
            g_scale, univariate,
        )
        # Ref: dcc.m:465 — A(offset+(1:count), :) = H/T
        A_info[offset:offset + count_dynamics, :] = hess_ll / T

        # Ref: dcc.m:466 — gradient of dcc_likelihood for dynamics scores
        _, s_ll = gradient_2sided(
            _ll_scores, parameters,
            data_3d, data_asym_3d, m, l, n,
            None, None,
            back_cast, back_cast_asym,
            stage, composite_int,
            is_joint_final, is_inference_final,
            g_scale, univariate,
            compute_scores=True,
        )
        # Ref: dcc.m:467 — dynamics block scores
        scores_mat[:, offset:offset + count_dynamics] = (
            s_ll[:, offset:offset + count_dynamics]
        )

        # Ref: dcc.m:468 — B = covnw(scores); HAC covariance
        B_cov = covnw(scores_mat)
        # Ref: dcc.m:469 — Ainv = A \ eye(v)
        Ainv = np.linalg.solve(A_info, np.eye(v))
        # Ref: dcc.m:470 — VCV = Ainv * B * Ainv' / T
        VCV = Ainv @ B_cov @ Ainv.T / T

    # ==================================================================
    # Diagnostics
    # Ref: dcc.m:473 — diagnostics = []; (empty in MATLAB; dict in Python)
    # ==================================================================
    diagnostics = {
        'stage': stage,
        'R': R,
        'N': N,
        'univariate': univariate,
        'g_scale': g_scale,
        'scale': scale,
        'back_cast': back_cast,
        'back_cast_asym': back_cast_asym,
    }

    return parameters, ll, Ht, VCV, scores_mat, diagnostics
