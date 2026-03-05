"""
Estimation of scalar RCC(m,n) multivariate volatility model with TARCH/GJR-GARCH
conditional variances.

The Rotated Conditional Correlation (RCC) model applies RARCH-style dynamics to
the *correlation* space rather than the covariance space. Three model types are
supported:

    * **Scalar** — common alpha/beta dynamics across all asset pairs
    * **CP** (Common Persistence) — per-asset diagonal A matrices with shared
      persistence parameter theta
    * **Diagonal** — full diagonal A and B matrices per asset

Two estimation methods are available:

    * **3-stage** (default) — Stage 1: per-series TARCH/GJR; Stage 2: correlation
      intercept R; Stage 3: dynamics parameters
    * **2-stage** — Stage 1: per-series TARCH/GJR; Stage 2: joint R + dynamics

Migrated from ``multivariate/rcc.m`` (459 lines) in the MFE Toolbox
(Version 4.0, Kevin Sheppard, University of Oxford).

Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 1    Date: 4/13/2012

Python migration: Blitzy Platform — MATLAB-to-Python 3.12 migration.

See Also
--------
mfe_toolbox.multivariate.dcc : DCC/ADCC estimation driver.
mfe_toolbox.multivariate.ccc_mvgarch : CCC-MVGARCH driver.
mfe_toolbox.multivariate.bekk : BEKK model driver.
mfe_toolbox.multivariate.rarch : RARCH model driver.
mfe_toolbox.multivariate.scalar_vt_vech : Scalar VT-VECH driver.
mfe_toolbox.multivariate.matrix_garch : Matrix GARCH driver.
mfe_toolbox.univariate.tarch : TARCH/GJR-GARCH estimation.
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.linalg
from scipy.optimize import minimize

from mfe_toolbox.multivariate.dcc_fit_variance import dcc_fit_variance
from mfe_toolbox.multivariate.rcc_likelihood import rcc_likelihood
from mfe_toolbox.multivariate.rcc_constraint import rcc_constraint
from mfe_toolbox.multivariate.dcc_inference_objective import dcc_inference_objective
from mfe_toolbox.utility.r2z import r2z
from mfe_toolbox.utility.z2r import z2r
from mfe_toolbox.utility.corr_vech import corr_vech
from mfe_toolbox.utility.hessian_2sided_nrows import hessian_2sided_nrows
from mfe_toolbox.utility.gradient_2sided import gradient_2sided
from mfe_toolbox.utility.covnw import covnw

__all__ = ['rcc']


def rcc(
    data: np.ndarray,
    data_asym: np.ndarray | None = None,
    m: int = 1,
    n: int = 1,
    p: int | np.ndarray | None = None,
    o: int | np.ndarray | None = None,
    q: int | np.ndarray | None = None,
    gjr_type: int | np.ndarray | None = None,
    type_model: str | None = None,
    method: str | None = None,
    composite: str | None = None,
    starting_vals: np.ndarray | None = None,
    options: dict | None = None,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Estimate a scalar RCC(m,n) multivariate volatility model.

    Parameters
    ----------
    data : numpy.ndarray
        T × K matrix of zero-mean residuals, **or** a K × K × T array of
        covariance estimators (e.g. realized covariance).
    data_asym : numpy.ndarray or None, optional
        K × K × T array of asymmetric covariance estimators. Only required
        when *data* is 3-dimensional and the univariate models use asymmetric
        terms (``o > 0``).
    m : int, optional
        Order of symmetric innovations in the RCC model (default 1).
    n : int, optional
        Order of lagged correlation in the RCC model (default 1).
    p : int or array_like or None, optional
        Number of symmetric innovation lags in the per-series GARCH model.
        Scalar (applied to all series) or K-vector. Default 1.
    o : int or array_like or None, optional
        Number of asymmetric innovation lags. Scalar or K-vector. Default 0.
    q : int or array_like or None, optional
        Number of conditional variance lags. Scalar or K-vector. Default 1.
    gjr_type : int or array_like or None, optional
        Model type code per series: 1 = TARCH/AVGARCH, 2 = GJR-GARCH
        (default 2). Scalar or K-vector.
    type_model : str or None, optional
        One of ``'Scalar'`` (default), ``'CP'`` (Common Persistence), or
        ``'Diagonal'``.
    method : str or None, optional
        Estimation method: ``'3-stage'`` (default) or ``'2-stage'``.
    composite : str or None, optional
        Composite likelihood type: ``'None'`` (default), ``'Diagonal'``,
        or ``'Full'``.
    starting_vals : numpy.ndarray or None, optional
        Full starting-value vector. Layout depends on *type_model*; see Notes.
    options : dict or None, optional
        Options dict passed to ``scipy.optimize.minimize`` (SLSQP).

    Returns
    -------
    parameters : numpy.ndarray
        Estimated parameter vector (column). Layout:

        * Scalar: ``[VOL(1)…VOL(K), corr_vech(R)', alpha, beta]``
        * CP: ``[VOL(1)…VOL(K), corr_vech(R)', diag(A1)…diag(Am), theta]``
        * Diagonal: ``[VOL(1)…VOL(K), corr_vech(R)', diag(A1)…diag(Am),
          diag(B1)…diag(Bn)]``

    ll : float
        Log-likelihood at the optimum (positive value).
    Ht : numpy.ndarray
        K × K × T array of conditional covariance matrices.
    VCV : numpy.ndarray
        ``V × V`` robust parameter covariance (sandwich estimator
        A⁻¹ B A⁻¹' / T).
    scores : numpy.ndarray
        T × V matrix of per-observation scores.
    diagnostics : dict
        Diagnostic information (currently empty, reserved for future use).

    Notes
    -----
    The function replicates the MATLAB ``rcc.m`` estimation pipeline:

    1. Fit per-series TARCH/GJR-GARCH via :func:`dcc_fit_variance`.
    2. Standardise covariance data by fitted volatilities.
    3. Estimate unconditional correlation R.
    4. Compute exponentially-weighted back-cast for recursion initialisation.
    5. Grid-search starting values for scalar dynamics.
    6. Optimise dynamics via ``scipy.optimize.minimize(method='SLSQP')``
       with :func:`rcc_constraint` stationarity constraints.
    7. Optionally refine with CP or Diagonal parameterisation.
    8. For 2-stage, jointly optimise R and dynamics.
    9. Reconstruct conditional covariance matrices Ht.
    10. Compute robust sandwich VCV via block numerical Hessians and
        gradients.

    References
    ----------
    Noureldin, Shephard, Sheppard (2012). "Multivariate Rotated ARCH
    models." *Journal of Econometrics*.

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.multivariate.rcc import rcc
    >>> rng = np.random.default_rng(42)
    >>> data = rng.standard_normal((500, 3)) * 0.01
    >>> params, ll, Ht, VCV, scores, diag = rcc(data, m=1, n=1)
    """
    # ==================================================================
    # Input Argument Checking — Ref: rcc.m:68-290
    # ==================================================================

    data = np.asarray(data, dtype=np.float64)

    # ------------------------------------------------------------------
    # Data dimensionality handling — Ref: rcc.m:129-147
    # ------------------------------------------------------------------
    if data.ndim == 2:
        # Ref: rcc.m:130-138 — 2D input: T×K residuals
        T, k = data.shape
        data2d = data.copy()
        # Ref: rcc.m:132 — eta = data.*(data<0)
        eta = data * (data < 0)
        # Ref: rcc.m:133-138 — construct K×K×T outer products
        data_3d = np.zeros((k, k, T), dtype=np.float64)
        data_asym_3d = np.zeros((k, k, T), dtype=np.float64)
        for t in range(T):
            # Ref: rcc.m:136 — data(:,:,t) = data2d(t,:)' * data2d(t,:)
            data_3d[:, :, t] = np.outer(data2d[t, :], data2d[t, :])
            # Ref: rcc.m:137 — dataAsym(:,:,t) = eta(t,:)' * eta(t,:)
            data_asym_3d[:, :, t] = np.outer(eta[t, :], eta[t, :])
        data = data_3d
        data_asym = data_asym_3d

    elif data.ndim == 3:
        # Ref: rcc.m:139-144 — 3D input: K×K×T covariance estimators
        k = data.shape[0]
        T = data.shape[2]
        if data_asym is None:
            data_asym = np.zeros_like(data)
        data_asym = np.asarray(data_asym, dtype=np.float64)
        # Ref: rcc.m:141-144 — extract 2D pseudo-returns from diagonal
        # NOTE: MATLAB source has a bug (uses uppercase K); corrected to k.
        data2d = np.zeros((T, k), dtype=np.float64)
        for i in range(k):
            # Ref: rcc.m:143 — sign from asymmetric data, magnitude from variance
            diag_var = data[i, i, :].copy()
            diag_asym = data_asym[i, i, :].copy()
            sign_vals = 1.0 - 2.0 * (diag_asym == 0).astype(np.float64)
            data2d[:, i] = sign_vals * np.sqrt(np.maximum(diag_var, 0.0))
    else:
        raise ValueError(
            'DATA must be either a T x K matrix or a K x K x T '
            '3-dimensional array.'
        )

    # ------------------------------------------------------------------
    # Validate dataAsym — Ref: rcc.m:149-157
    # ------------------------------------------------------------------
    if data_asym is not None:
        data_asym = np.asarray(data_asym, dtype=np.float64)
        if data_asym.ndim != 3:
            raise ValueError('DATA_ASYM must be a K x K x T array.')
        k1, k2, T2 = data_asym.shape
        if k1 != k or k2 != k or T2 != T:
            raise ValueError(
                f'DATA_ASYM must be a {k} x {k} x {T} array, '
                f'got {k1} x {k2} x {T2}.'
            )

    # ------------------------------------------------------------------
    # Validate m and n — Ref: rcc.m:159-167
    # ------------------------------------------------------------------
    if not isinstance(m, (int, np.integer)) or m < 1:
        raise ValueError('M must be a positive integer.')
    m = int(m)

    if n is None:
        n = 0
    if not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError('N must be a non-negative integer.')
    n = int(n)

    # ------------------------------------------------------------------
    # Expand and validate p, o, q, gjr_type — Ref: rcc.m:169-204
    # ------------------------------------------------------------------
    if p is None:
        p = 1
    p_arr = np.atleast_1d(np.asarray(p, dtype=np.int64)).ravel()
    if p_arr.size == 1:
        p_arr = np.full(k, p_arr[0], dtype=np.int64)
    if np.any(p_arr != np.floor(p_arr)) or np.any(p_arr < 1) or p_arr.size != k:
        raise ValueError(
            'All elements in P must be positive integers, and P must be '
            'either scalar or K-element.'
        )

    if o is None:
        o = 0
    o_arr = np.atleast_1d(np.asarray(o, dtype=np.int64)).ravel()
    if o_arr.size == 1:
        o_arr = np.full(k, o_arr[0], dtype=np.int64)
    if np.any(o_arr != np.floor(o_arr)) or np.any(o_arr < 0) or o_arr.size != k:
        raise ValueError(
            'All elements in O must be non-negative integers, and O must be '
            'either scalar or K-element.'
        )

    if q is None:
        q = 1
    q_arr = np.atleast_1d(np.asarray(q, dtype=np.int64)).ravel()
    if q_arr.size == 1:
        q_arr = np.full(k, q_arr[0], dtype=np.int64)
    if np.any(q_arr != np.floor(q_arr)) or np.any(q_arr < 0) or q_arr.size != k:
        raise ValueError(
            'All elements in Q must be non-negative integers, and Q must be '
            'either scalar or K-element.'
        )

    if gjr_type is None:
        gjr_type = 2
    gjr_arr = np.atleast_1d(np.asarray(gjr_type, dtype=np.int64)).ravel()
    if gjr_arr.size == 1:
        gjr_arr = np.full(k, gjr_arr[0], dtype=np.int64)
    if not np.all(np.isin(gjr_arr, [1, 2])) or gjr_arr.size != k:
        raise ValueError(
            'GJRTYPE must be in {1, 2} and must be either scalar or K-element.'
        )

    # ------------------------------------------------------------------
    # Parse type_model — Ref: rcc.m:206-220
    # ------------------------------------------------------------------
    if type_model is None:
        type_model = 'scalar'
    type_str = str(type_model).lower().strip()
    if type_str not in ('scalar', 'cp', 'diagonal'):
        raise ValueError(
            "TYPE must be one of 'Scalar', 'CP', or 'Diagonal'."
        )
    type_code: int = {'scalar': 1, 'cp': 2, 'diagonal': 3}[type_str]

    # Ref: rcc.m:222-225 — CP forces n=1
    if type_code == 2 and n != 1:
        warnings.warn(
            "When using 'CP', N must always equal 1. Setting N to 1.",
            stacklevel=2,
        )
        n = 1

    # ------------------------------------------------------------------
    # Parse method — Ref: rcc.m:227-238
    # ------------------------------------------------------------------
    if method is None:
        method = '3-stage'
    method_str = str(method).lower().strip()
    if method_str not in ('3-stage', '2-stage'):
        raise ValueError("METHOD must be either '3-stage' or '2-stage'.")
    stage: int = 3 if method_str == '3-stage' else 2

    # ------------------------------------------------------------------
    # Parse composite — Ref: rcc.m:240-257
    # ------------------------------------------------------------------
    if composite is None:
        composite = 'none'
    composite_str = str(composite).lower().strip()
    if composite_str not in ('none', 'diagonal', 'full'):
        raise ValueError(
            "COMPOSITE must be one of 'None', 'Diagonal', or 'Full'."
        )
    composite_code: int = {'none': 0, 'diagonal': 1, 'full': 2}[composite_str]

    # Ref: rcc.m:254-257 — 2-stage with diagonal composite forces full
    if stage == 2 and composite_code == 1:
        warnings.warn(
            "When METHOD is '2-stage', COMPOSITE must be either 'None' or "
            "'Full'. Forcing COMPOSITE to 'Full'.",
            stacklevel=2,
        )
        composite_code = 2

    # ------------------------------------------------------------------
    # Parse starting values — Ref: rcc.m:259-278
    # ------------------------------------------------------------------
    tarch_starting_vals: np.ndarray | None = None
    rcc_starting_vals: np.ndarray | None = None

    if starting_vals is not None:
        starting_vals = np.asarray(starting_vals, dtype=np.float64).ravel()
        # Ref: rcc.m:260 — count total expected parameters
        garch_count = int(k + np.sum(p_arr) + np.sum(o_arr) + np.sum(q_arr))
        if type_code == 1:
            dyn_count = m + n
        elif type_code == 2:
            dyn_count = m * k + n
        else:
            dyn_count = (m + n) * k
        corr_count = k * (k - 1) // 2
        total_count = garch_count + corr_count + dyn_count

        if len(starting_vals) != total_count:
            raise ValueError(
                f'STARTING_VALS does not contain the correct number of '
                f'parameters. Expected {total_count}, got {len(starting_vals)}.'
            )
        # Ref: rcc.m:272-274 — split into TARCH and RCC portions
        tarch_starting_vals = starting_vals[:garch_count]
        offset_sv = garch_count + corr_count
        rcc_starting_vals = starting_vals[offset_sv:]

    # ------------------------------------------------------------------
    # Validate / build scipy options — Ref: rcc.m:280-290
    # ------------------------------------------------------------------
    if options is None:
        # Ref: rcc.m:281-285 — default MATLAB options with iter display
        scipy_options: dict = {
            'disp': False,
            'maxiter': 1000,
            'ftol': 1e-9,
        }
    else:
        if not isinstance(options, dict):
            raise ValueError(
                'OPTIONS must be a dict of scipy.optimize.minimize options.'
            )
        scipy_options = dict(options)

    # ==================================================================
    # Stage 1: Univariate volatility models — Ref: rcc.m:291-300
    # ==================================================================
    # Ref: rcc.m:294
    H, univariate = dcc_fit_variance(
        data2d, p_arr, o_arr, q_arr, gjr_arr, tarch_starting_vals
    )

    # Ref: rcc.m:295-300 — standardise 3D data by conditional volatilities
    std_data = data.copy()
    for t in range(T):
        # Ref: rcc.m:297 — h = sqrt(H(t,:))
        h = np.sqrt(H[t, :])
        # Ref: rcc.m:298 — hh = h'*h (outer product)
        hh = np.outer(h, h)
        # Ref: rcc.m:299 — stdData(:,:,t) = stdData(:,:,t) ./ hh
        std_data[:, :, t] = std_data[:, :, t] / hh

    # ==================================================================
    # Unconditional correlation estimation — Ref: rcc.m:301-307
    # ==================================================================
    # Ref: rcc.m:304 — R = mean(stdData, 3)
    R = np.mean(std_data, axis=2)

    # Ref: rcc.m:305 — Rm12 = R^(-0.5) (matrix inverse square root)
    R_sqrt = scipy.linalg.sqrtm(R)
    if np.iscomplexobj(R_sqrt):
        R_sqrt = R_sqrt.real
    R_sqrt = np.asarray(R_sqrt, dtype=np.float64)
    Rm12 = np.linalg.inv(R_sqrt)

    # Ref: rcc.m:306-307 — normalise R to correlation matrix
    r_scale = np.sqrt(np.diag(R))
    R = R / np.outer(r_scale, r_scale)

    # ==================================================================
    # Back cast — Ref: rcc.m:309-314
    # ==================================================================
    # Ref: rcc.m:309 — exponentially decaying weights
    n_weights = int(np.sqrt(T)) + 1
    w = 0.06 * 0.94 ** np.arange(n_weights)
    w = w / np.sum(w)

    # Ref: rcc.m:311-314
    back_cast = np.zeros((k, k), dtype=np.float64)
    for i in range(len(w)):
        # Ref: rcc.m:313 — Rm12 * stdData(:,:,i) * Rm12
        back_cast += w[i] * (Rm12 @ std_data[:, :, i] @ Rm12)

    # ==================================================================
    # Starting Values — Ref: rcc.m:316-341
    # ==================================================================
    if rcc_starting_vals is not None:
        # Ref: rcc.m:319 — use provided starting values
        sv = rcc_starting_vals.copy()
    else:
        # Ref: rcc.m:321-337 — grid search for starting values
        a_vals = np.array([0.01, 0.03, 0.05, 0.1])
        theta_vals = np.array([0.99, 0.97, 0.95])
        # Ref: rcc.m:323 — [a, theta] = ndgrid(a, theta)
        a_grid, theta_grid = np.meshgrid(a_vals, theta_vals, indexing='ij')
        # Ref: rcc.m:324 — parameters = sqrt(unique([a(:) theta(:)-a(:)], 'rows'))
        pairs = np.column_stack([
            a_grid.ravel(order='F'),
            (theta_grid - a_grid).ravel(order='F'),
        ])
        unique_pairs = np.unique(pairs, axis=0)
        # Filter out non-positive values before sqrt (safety)
        unique_pairs = unique_pairs[np.all(unique_pairs >= 0, axis=1)]
        candidates = np.sqrt(unique_pairs)

        # Ref: rcc.m:325-334 — evaluate each candidate
        min_ll = np.inf
        is_joint_sv = False
        is_inference_sv = False
        sv = candidates[0, :] if len(candidates) > 0 else np.array([0.2, 0.9])

        for i in range(len(candidates)):
            try:
                ll_val, _, _ = rcc_likelihood(
                    candidates[i, :], std_data, 1, 1, R, back_cast,
                    3, 1, composite_code, is_joint_sv, is_inference_sv,
                    r_scale, univariate,
                )
                if ll_val < min_ll:
                    sv = candidates[i, :].copy()
                    min_ll = ll_val
            except Exception:
                # Ref: rcc.m:329 — skip invalid candidates silently
                continue

        # Ref: rcc.m:335-337 — expand to m+n parameters
        a_best = sv[0] ** 2
        b_best = sv[1] ** 2
        a_parts = np.sqrt(a_best * np.ones(m) / m)
        if n > 0:
            b_parts = np.sqrt(b_best * np.ones(n) / n)
            sv = np.concatenate([a_parts, b_parts])
        else:
            sv = a_parts

    # Ref: rcc.m:339-341 — ensure row vector
    sv = sv.ravel()

    # Ref: rcc.m:342-343 — bounds
    UB = np.ones_like(sv)
    LB = -UB

    # ==================================================================
    # Optimisation: Scalar model — Ref: rcc.m:348-350
    # ==================================================================
    is_joint_opt = False
    is_inference_opt = False

    # Ref: rcc.m:350 — fmincon → scipy.optimize.minimize(SLSQP)
    # Scalar model first (type_code=1), stage=3
    opt_result = minimize(
        fun=lambda params: rcc_likelihood(
            params, std_data, m, n, R, back_cast, 3, 1, composite_code,
            is_joint_opt, is_inference_opt, r_scale, univariate,
        )[0],
        x0=sv,
        method='SLSQP',
        bounds=list(zip(LB, UB)),
        constraints=[{
            'type': 'ineq',
            'fun': lambda params, _m=m, _n=n, _k=k: rcc_constraint(
                params, _m, _n, _k, 1, stage=3
            )[0],
        }],
        options=scipy_options,
    )
    opt_params = opt_result.x.copy()

    # ==================================================================
    # Expand starting values for CP/Diagonal — Ref: rcc.m:351-367
    # ==================================================================
    if type_code == 2:
        # Ref: rcc.m:352-354 — CP: replicate scalar A across assets
        A_mat = np.outer(np.ones(k), opt_params[:m])  # k × m
        theta = np.sqrt(np.sum(opt_params ** 2))
        sv = np.concatenate([A_mat.ravel(order='F'), [theta]])
        UB = np.ones_like(sv)
        LB = -UB

    elif type_code == 3:
        # Ref: rcc.m:358-363 — Diagonal: replicate scalar A and B
        A_mat = np.outer(np.ones(k), opt_params[:m])  # k × m
        B_mat = np.outer(np.ones(k), opt_params[m:m + n])  # k × n
        sv = np.concatenate([A_mat.ravel(order='F'), B_mat.ravel(order='F')])
        UB = np.ones_like(sv)
        LB = -UB

    if type_code > 1:
        # Ref: rcc.m:366 — re-optimise with expanded parameterisation
        opt_result = minimize(
            fun=lambda params: rcc_likelihood(
                params, std_data, m, n, R, back_cast, 3, type_code,
                composite_code, is_joint_opt, is_inference_opt, r_scale,
                univariate,
            )[0],
            x0=sv,
            method='SLSQP',
            bounds=list(zip(LB, UB)),
            constraints=[{
                'type': 'ineq',
                'fun': lambda params, _m=m, _n=n, _k=k, _tc=type_code: (
                    rcc_constraint(params, _m, _n, _k, _tc, stage=3)[0]
                ),
            }],
            options=scipy_options,
        )
        opt_params = opt_result.x.copy()

    # ==================================================================
    # 2-stage estimation (joint R + dynamics) — Ref: rcc.m:369-381
    # ==================================================================
    if stage == 2:
        # Ref: rcc.m:370 — Fisher z-transform of R
        z = r2z(R)
        # Ref: rcc.m:371 — prepend z to dynamics parameters
        sv = np.concatenate([z.ravel(), opt_params])

        # Ref: rcc.m:373-374 — bounds for joint optimisation
        n_corr = k * (k - 1) // 2
        LB = np.concatenate([
            -np.inf * np.ones(n_corr),
            -np.ones(len(opt_params)),
        ])
        UB = np.concatenate([
            np.inf * np.ones(n_corr),
            np.ones(len(opt_params)),
        ])

        # Ref: rcc.m:375 — joint optimisation
        opt_result = minimize(
            fun=lambda params: rcc_likelihood(
                params, std_data, m, n, R, back_cast, 2, type_code,
                composite_code, is_joint_opt, is_inference_opt, r_scale,
                univariate,
            )[0],
            x0=sv,
            method='SLSQP',
            bounds=list(zip(LB, UB)),
            constraints=[{
                'type': 'ineq',
                'fun': lambda params, _m=m, _n=n, _k=k, _tc=type_code: (
                    rcc_constraint(params, _m, _n, _k, _tc, stage=2)[0]
                ),
            }],
            options=scipy_options,
        )
        opt_params = opt_result.x.copy()

        # Ref: rcc.m:377-380 — extract R from optimised z-parameters
        z_opt = opt_params[:n_corr]
        R = z2r(z_opt)
        dynamics_params = opt_params[n_corr:]
        # Ref: rcc.m:380 — reassemble as [corr_vech(R)' dynamics]
        opt_params = np.concatenate([corr_vech(R).ravel(), dynamics_params])

    # ==================================================================
    # Assemble full parameter vector — Ref: rcc.m:383-394
    # ==================================================================
    # Ref: rcc.m:386-389 — collect GARCH parameters from univariate results
    garch_parameters = np.concatenate([
        univariate[i]['parameters'].ravel()
        for i in range(k)
    ])

    if stage == 3:
        # Ref: rcc.m:391 — [garchParams corr_vech(R)' dynamics]
        full_params = np.concatenate([
            garch_parameters,
            corr_vech(R).ravel(),
            opt_params,
        ])
    elif stage == 2:
        # Ref: rcc.m:393 — [garchParams corr_vech(R)' dynamics]
        # opt_params already contains [corr_vech(R) dynamics] from above
        full_params = np.concatenate([garch_parameters, opt_params])
    else:
        full_params = np.concatenate([garch_parameters, opt_params])

    # ==================================================================
    # Final likelihood and Ht — Ref: rcc.m:396-404
    # ==================================================================
    is_joint_final = True
    is_inference_final = True

    # Ref: rcc.m:398 — final joint likelihood evaluation
    ll_neg, _, Rt = rcc_likelihood(
        full_params, data, m, n, R, back_cast, 2, type_code,
        composite_code, is_joint_final, is_inference_final,
        r_scale, univariate,
    )
    # Ref: rcc.m:399 — negate to get actual log-likelihood
    ll = -ll_neg

    # Ref: rcc.m:400-404 — reconstruct conditional covariance Ht
    Ht = np.zeros((k, k, T), dtype=np.float64)
    for t in range(T):
        # Ref: rcc.m:402 — h = sqrt(H(t,:))
        h = np.sqrt(H[t, :])
        # Ref: rcc.m:403 — Ht(:,:,t) = Rt(:,:,t) .* (h'*h)
        Ht[:, :, t] = Rt[:, :, t] * np.outer(h, h)

    # ==================================================================
    # Inference — Ref: rcc.m:406-457
    # ==================================================================
    # Ref: rcc.m:409 — transpose to column vector
    parameters_col = full_params.copy()

    v = len(parameters_col)
    A_mat_inf = np.zeros((v, v), dtype=np.float64)
    scores_mat = np.zeros((T, v), dtype=np.float64)
    offset_inf = 0

    # Ref: rcc.m:419-426 — block 1: univariate GARCH A-matrix and scores
    for i in range(k):
        u = univariate[i]
        count_i = 1 + int(u['p']) + int(u['o']) + int(u['q'])
        idx_start = offset_inf
        idx_end = offset_inf + count_i
        # Ref: rcc.m:423 — A(ind, ind) = u.A
        u_A = np.asarray(u['A'], dtype=np.float64)
        A_mat_inf[idx_start:idx_end, idx_start:idx_end] = u_A
        # Ref: rcc.m:425 — scores(:, ind) = u.scores
        u_scores = np.asarray(u['scores'], dtype=np.float64)
        scores_mat[:, idx_start:idx_end] = u_scores[:T, :]
        offset_inf += count_i

    # ------------------------------------------------------------------
    # Wrapper functions for numerical differentiation
    # hessian_2sided_nrows expects f(x, *args) → scalar
    # gradient_2sided with compute_scores expects f(x, *args) → (scalar, T-array)
    # ------------------------------------------------------------------
    def _rcc_ll_scalar(params, *args):
        """Wrapper returning only the scalar likelihood."""
        result = rcc_likelihood(params, *args)
        return float(result[0])

    def _rcc_ll_scores(params, *args):
        """Wrapper returning (scalar, per-obs array) for gradient scores."""
        result = rcc_likelihood(params, *args)
        return float(result[0]), result[1]

    def _dcc_inf_scalar(params, *args):
        """Wrapper returning only the scalar objective."""
        result = dcc_inference_objective(params, *args)
        return float(result[0])

    def _dcc_inf_scores(params, *args):
        """Wrapper returning (scalar, per-obs array) for gradient scores."""
        result = dcc_inference_objective(params, *args)
        return float(result[0]), result[1]

    # ------------------------------------------------------------------
    # Block inference — Ref: rcc.m:429-457
    # ------------------------------------------------------------------
    rcc_ll_args = (
        data, m, n, R, back_cast, stage, type_code, composite_code,
        is_joint_final, is_inference_final, r_scale, univariate,
    )

    if stage == 2:
        # Ref: rcc.m:431-438 — 2-stage inference
        count_2s = k * (k - 1) // 2 + m + n

        # Ref: rcc.m:432 — partial Hessian (last count rows)
        H_hess = hessian_2sided_nrows(
            _rcc_ll_scalar, parameters_col, count_2s, *rcc_ll_args,
        )
        # Ref: rcc.m:433 — A(offset+(1:count), :) = H/T
        A_mat_inf[offset_inf:offset_inf + count_2s, :] = H_hess / T

        # Ref: rcc.m:434 — gradient with individual scores
        _, s_grad = gradient_2sided(
            _rcc_ll_scores, parameters_col, *rcc_ll_args,
            compute_scores=True,
        )
        # Ref: rcc.m:435 — scores(:, offset+(1:count)) = s(:, offset+(1:count))
        scores_mat[:, offset_inf:offset_inf + count_2s] = (
            s_grad[:T, offset_inf:offset_inf + count_2s]
        )

        # Ref: rcc.m:436 — B = cov(scores)
        B_cov = np.cov(scores_mat[:T, :], rowvar=False)

        # Ref: rcc.m:437-438 — sandwich VCV
        Ainv = np.linalg.solve(A_mat_inf, np.eye(v))
        VCV = Ainv @ B_cov @ Ainv.T / T

    elif stage == 3:
        # Ref: rcc.m:440-456 — 3-stage inference

        # --- Block 2: dcc_inference_objective (correlation intercept) ---
        count_corr = k * (k - 1) // 2
        temp_params = parameters_col[:offset_inf + count_corr]

        dcc_inf_args = (data, data_asym, m, 0, n, univariate)

        # Ref: rcc.m:443 — gradient with individual scores
        _, s_dcc = gradient_2sided(
            _dcc_inf_scores, temp_params, *dcc_inf_args,
            compute_scores=True,
        )
        # Ref: rcc.m:444
        scores_mat[:, offset_inf:offset_inf + count_corr] = (
            s_dcc[:T, offset_inf:offset_inf + count_corr]
        )

        # Ref: rcc.m:445 — partial Hessian
        H_dcc = hessian_2sided_nrows(
            _dcc_inf_scalar, temp_params, count_corr, *dcc_inf_args,
        )
        # Ref: rcc.m:446
        A_mat_inf[
            offset_inf:offset_inf + count_corr,
            :offset_inf + count_corr,
        ] = H_dcc / T
        offset_inf += count_corr

        # --- Block 3: rcc_likelihood (dynamics) ---
        # Ref: rcc.m:449 — count = m + n (matches MATLAB source for all types)
        count_dyn = m + n

        # Ref: rcc.m:450 — partial Hessian for dynamics block
        H_rcc = hessian_2sided_nrows(
            _rcc_ll_scalar, parameters_col, count_dyn, *rcc_ll_args,
        )
        # Ref: rcc.m:451
        A_mat_inf[offset_inf:offset_inf + count_dyn, :] = H_rcc / T

        # Ref: rcc.m:452 — gradient with scores
        _, s_rcc = gradient_2sided(
            _rcc_ll_scores, parameters_col, *rcc_ll_args,
            compute_scores=True,
        )
        # Ref: rcc.m:453
        scores_mat[:, offset_inf:offset_inf + count_dyn] = (
            s_rcc[:T, offset_inf:offset_inf + count_dyn]
        )

        # Ref: rcc.m:454 — Newey-West HAC covariance of scores
        B_cov = covnw(scores_mat[:T, :])

        # Ref: rcc.m:455-456 — sandwich VCV
        Ainv = np.linalg.solve(A_mat_inf, np.eye(v))
        VCV = Ainv @ B_cov @ Ainv.T / T
    else:
        # Fallback (should not occur given valid stage values)
        VCV = np.zeros((v, v), dtype=np.float64)

    # ==================================================================
    # Return — Ref: rcc.m:459
    # ==================================================================
    # Ref: rcc.m:459 — diagnostics = [] (empty in MATLAB)
    diagnostics: dict = {}

    return parameters_col, ll, Ht, VCV, scores_mat, diagnostics
