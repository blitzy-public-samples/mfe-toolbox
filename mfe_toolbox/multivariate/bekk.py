"""
BEKK(p, o, q) Multivariate Volatility Model Estimation Driver.

Estimates symmetric and asymmetric BEKK(p, o, q) multivariate GARCH models
using constrained optimisation.  The conditional covariance dynamics follow:

    H(t) = C * C'
           + sum_{j=1}^{p} A_j' * OP(t-j)     * A_j
           + sum_{j=1}^{o} G_j' * OPA(t-j)     * G_j
           + sum_{j=1}^{q} B_j' * H(t-j)       * B_j

where C is the lower-triangular Cholesky factor of the intercept,
OP(t) = r_t * r_t'  is the symmetric outer product,
OPA(t) = eta_t * eta_t'  is the asymmetric outer product with
eta_t = r_t .* (r_t < 0), and A, G, B are the coefficient matrices
parameterised as Scalar, Diagonal, or Full.

Three parameterisation types are supported:
    * ``'Scalar'``   (type 1) — A_j = a_j * I(K),  G_j = g_j * I(K),  B_j = b_j * I(K).
    * ``'Diagonal'`` (type 2) — A_j = diag(a_j),  G_j = diag(g_j),  B_j = diag(b_j).
    * ``'Full'``     (type 3) — A_j, G_j, B_j are unrestricted K × K matrices.

Migrated from: multivariate/bekk.m (209 lines)
Copyright: Kevin Sheppard, kevin.sheppard@economics.ox.ac.uk
Revision: 1    Date: 3/27/2012

References
----------
Engle, R. F. and Kroner, K. F. (1995). Multivariate Simultaneous
Generalized ARCH.  *Econometric Theory*, 11(1), 122–150.

See Also
--------
bekk_simulate : BEKK simulation.
bekk_parameter_transform : Parameter vector ↔ matrix conversion.
bekk_likelihood : BEKK negative log-likelihood.
bekk_constraint : Stationarity constraint.
rarch : Rotated ARCH model.
"""

import warnings

import numpy as np
from scipy.optimize import minimize

from mfe_toolbox.multivariate.bekk_likelihood import bekk_likelihood
from mfe_toolbox.multivariate.bekk_constraint import bekk_constraint
from mfe_toolbox.multivariate.bekk_parameter_transform import bekk_parameter_transform
from mfe_toolbox.multivariate.scalar_vt_vech import scalar_vt_vech
from mfe_toolbox.utility.chol2vec import chol2vec
from mfe_toolbox.utility.robustvcv import robustvcv


# ---------------------------------------------------------------------------
# Internal helper wrappers
# ---------------------------------------------------------------------------

def _bekk_ll_wrapper(parameters, data, data_asym, p, o, q,
                     back_cast, back_cast_asym, type_code):
    """Return ``(scalar_ll, per_period_lls)`` from :func:`bekk_likelihood`.

    :func:`robustvcv` expects ``fun(theta, *args) → (float, 1-D array)``
    but :func:`bekk_likelihood` returns ``(ll, lls, Ht)`` — the third
    output (conditional covariance tensor) must be stripped.

    Ref: bekk.m:207–209 — MATLAB ``nargout`` auto-discards extra outputs;
    Python requires an explicit wrapper.
    """
    ll, lls, _ = bekk_likelihood(
        parameters, data, data_asym, p, o, q,
        back_cast, back_cast_asym, type_code,
    )
    return ll, lls


def _bekk_obj_for_minimize(parameters, data, data_asym, p, o, q,
                           back_cast, back_cast_asym, type_code):
    """Objective adapter for ``scipy.optimize.minimize``.

    Returns only the scalar negative log-likelihood (first element of the
    :func:`bekk_likelihood` output tuple).

    Ref: bekk.m:200 — MATLAB ``fmincon(@bekk_likelihood, ...)`` captures
    only the scalar output by default.
    """
    ll, _, _ = bekk_likelihood(
        parameters, data, data_asym, p, o, q,
        back_cast, back_cast_asym, type_code,
    )
    return float(ll)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def bekk(
    data: np.ndarray,
    data_asym: np.ndarray | None = None,
    p: int = 1,
    o: int = 0,
    q: int = 1,
    type_model: str | int = 'Scalar',
    starting_vals: np.ndarray | None = None,
    options: dict | None = None,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate a symmetric or asymmetric BEKK(p, o, q) multivariate volatility model.

    Parameters
    ----------
    data : numpy.ndarray
        Either a T × K matrix of zero-mean residuals **or** a K × K × T
        3-D array of covariance estimators (e.g. realised covariance).
        When 2-D the function internally computes outer products
        ``r_t @ r_t.T`` and the asymmetric counterpart.
    data_asym : numpy.ndarray or None, optional
        K × K × T array of asymmetric covariance estimators.  Required when
        ``data`` is 3-D and ``o > 0``.  When ``data`` is 2-D or ``o == 0``
        this parameter is ignored and may be ``None``.
    p : int, optional
        Positive integer — number of symmetric innovation lags (ARCH order).
        Default is 1.
    o : int, optional
        Non-negative integer — number of asymmetric (leverage) innovation
        lags.  Default is 0.
    q : int, optional
        Non-negative integer — number of conditional covariance lags (GARCH
        order).  Default is 1.
    type_model : {``'Scalar'``, ``'Diagonal'``, ``'Full'``} or {1, 2, 3}, optional
        Parameterisation type.  String values are case-insensitive.
        Default is ``'Scalar'``.
    starting_vals : numpy.ndarray or None, optional
        Flat 1-D starting value vector.  If ``None`` (default), starting
        values are computed automatically via :func:`scalar_vt_vech`.
        The expected length depends on ``type_model``:

        * Scalar:   ``k*(k+1)/2 + (p + o + q)``
        * Diagonal: ``k*(k+1)/2 + (p + o + q) * k``
        * Full:     ``k*(k+1)/2 + (p + o + q) * k * k``

    options : dict or None, optional
        Options dictionary passed to ``scipy.optimize.minimize`` (SLSQP).
        Keys such as ``'maxiter'``, ``'ftol'``, ``'disp'`` are forwarded
        directly.  If ``None``, sensible defaults are used.

    Returns
    -------
    parameters : numpy.ndarray
        Estimated parameter vector.  The layout depends on ``type_model``:

        * **Scalar**:
          ``[CC', a(1)…a(p), g(1)…g(o), b(1)…b(q)]`` — all scalars.
        * **Diagonal**:
          ``[CC', diag(A_1)'…diag(A_p)', diag(G_1)'…diag(G_o)',
          diag(B_1)'…diag(B_q)']``.
        * **Full**:
          ``[CC', vec(A_1)…vec(A_p), vec(G_1)…vec(G_o),
          vec(B_1)…vec(B_q)]``.

        where ``CC' = chol2vec(chol(C)')`` and ``vec(M) = M(:)`` in
        column-major (Fortran) order.
    ll : float
        Positive log-likelihood at the optimum (negated from the
        minimiser's objective).
    Ht : numpy.ndarray
        K × K × T array of conditional covariance matrices.
    VCV : numpy.ndarray
        ``numParams × numParams`` robust (sandwich) parameter
        variance-covariance matrix ``A^{-1} B A^{-1} / T``.
    scores : numpy.ndarray
        T × ``numParams`` matrix of per-observation numerical scores.

    Raises
    ------
    ValueError
        If input dimensions are inconsistent, ``p < 1``, ``o < 0``,
        ``q < 0``, ``type_model`` is unrecognised, or ``starting_vals``
        has the wrong length.

    Notes
    -----
    The estimation replaces MATLAB's ``fmincon`` with
    ``scipy.optimize.minimize(method='SLSQP')`` using translated box bounds
    and a nonlinear stationarity constraint provided by
    :func:`bekk_constraint`.

    Use :func:`bekk_parameter_transform` to convert the flat parameter
    vector back to structured matrices ``(C, A, G, B)``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> data = rng.standard_normal((500, 2))
    >>> params, ll, Ht, VCV, scores = bekk(data, p=1, o=0, q=1)
    >>> params.shape[0] > 0
    True
    """

    # ==================================================================
    # Section 1 — Input Argument Checking
    # Ref: bekk.m:55–144
    # ==================================================================

    # --- Coerce data to float64 array ---
    data = np.asarray(data, dtype=np.float64)

    # --- Determine dimensions T and k ---
    # Ref: bekk.m:58–61
    if data.ndim == 3:
        # Ref: bekk.m:60 — [k,~,T] = size(data)
        k = data.shape[0]
        T = data.shape[2]
    elif data.ndim == 2:
        # Ref: bekk.m:58 — [T,k] = size(data)
        T, k = data.shape
    else:
        raise ValueError(
            'DATA must be either a T by K matrix or a K by K by T array.'
        )

    # Ref: bekk.m:82–84 — T must exceed K
    if T <= k:
        raise ValueError(
            'DATA must be either a T by K matrix or a K by K by T array, '
            'and T must be larger than K.'
        )

    # Ref: bekk.m:85–89 — Validate data_asym when DATA is 3-D and o > 0
    if data.ndim == 3 and o > 0:
        if data_asym is None:
            raise ValueError(
                'DATAASYM must be provided when O>0 and DATA is a 3D array.'
            )
        data_asym = np.asarray(data_asym, dtype=np.float64)
        if data_asym.ndim != 3 or data_asym.shape != data.shape:
            raise ValueError(
                'DATAASYM must be provided when O>0 and DATA is a 3D array.'
            )

    # --- Validate p, o, q ---
    # Ref: bekk.m:91–105
    if not isinstance(p, (int, np.integer)) or p < 1:
        raise ValueError('P must be a positive scalar.')
    # Ref: bekk.m:94–96
    if o is None:
        o = 0
    if not isinstance(o, (int, np.integer)) or o < 0:
        raise ValueError('O must be a non-negative scalar.')
    # Ref: bekk.m:100–102
    if q is None:
        q = 0
    if not isinstance(q, (int, np.integer)) or q < 0:
        raise ValueError('Q must be a non-negative scalar.')

    # --- Convert type string → integer code ---
    # Ref: bekk.m:107–115
    if isinstance(type_model, str):
        type_lower = type_model.strip().lower()
        if type_lower == 'scalar':
            type_code = 1
        elif type_lower == 'diagonal':
            type_code = 2
        elif type_lower == 'full':
            type_code = 3
        else:
            raise ValueError("TYPE must be 'Scalar', 'Diagonal' or 'Full'.")
    elif isinstance(type_model, (int, np.integer)):
        type_code = int(type_model)
        if type_code not in (1, 2, 3):
            raise ValueError("TYPE must be 'Scalar', 'Diagonal' or 'Full'.")
    else:
        raise ValueError("TYPE must be 'Scalar', 'Diagonal' or 'Full'.")

    # --- Validate starting_vals length ---
    # Ref: bekk.m:117–131
    k2 = k * (k + 1) // 2  # Ref: bekk.m:117 — k2 = k*(k+1)/2
    if starting_vals is not None:
        starting_vals = np.asarray(starting_vals, dtype=np.float64).ravel()
        if type_code == 1:
            count = p + o + q
        elif type_code == 2:
            count = (p + o + q) * k
        else:  # type_code == 3
            count = (p + o + q) * k * k
        count = count + k2
        if len(starting_vals) != count:
            raise ValueError(
                'STARTINGVALS does not have the expected number of elements.'
            )

    # --- Default SLSQP options ---
    # Ref: bekk.m:133–144 — replaces optimset('fmincon') defaults
    if options is None:
        options = {
            'maxiter': 1000,
            'disp': False,
            'ftol': 1e-10,
        }

    # ==================================================================
    # Section 2 — Data Transformation
    # Ref: bekk.m:148–156
    # ==================================================================
    if data.ndim == 2:
        # Ref: bekk.m:149–155 — compute K×K×T outer-product arrays
        temp = np.zeros((k, k, T))
        data_asym_3d = np.zeros((k, k, T))
        for i in range(T):
            # Ref: bekk.m:151 — data(i,:)'*data(i,:)  (MATLAB 1-indexed)
            row = data[i, :]
            temp[:, :, i] = np.outer(row, row)
            # Ref: bekk.m:152–153 — asymmetric: negative returns only
            eta = row * (row < 0.0)
            data_asym_3d[:, :, i] = np.outer(eta, eta)
        data = temp
        data_asym = data_asym_3d
    else:
        # Data is already 3-D; ensure data_asym is populated
        if data_asym is None:
            data_asym = np.zeros((k, k, T))
        else:
            data_asym = np.asarray(data_asym, dtype=np.float64)

    # ==================================================================
    # Section 3 — Starting Values
    # Ref: bekk.m:160–181
    # ==================================================================
    if starting_vals is None:
        # Ref: bekk.m:161–163 — use scalar_vt_vech options with no display
        starting_options = {
            'maxiter': 500,
            'disp': False,
        }
        # Ref: bekk.m:164 — scalar_vt_vech returns
        #   (parameters, ll, ht, intercept, VCV, scores, diagnostics)
        sv_params, _, _, intercept, _, _, _ = scalar_vt_vech(
            data, data_asym, p, o, q, options=starting_options,
        )
        # Ref: bekk.m:165–166 — Cholesky of intercept → half-vec
        # MATLAB: C = chol2vec(chol(C)');
        # chol(C) in MATLAB returns upper-triangular; chol(C)' is lower-tri.
        # np.linalg.cholesky returns lower-triangular directly.
        C_chol = np.linalg.cholesky(intercept)
        C_vec = chol2vec(C_chol)

        # Ref: bekk.m:167–174 — determine per-parameter shape
        if type_code == 1:
            shape = np.float64(1.0)
        elif type_code == 2:
            shape = np.ones(k)
        else:  # type_code == 3
            shape = np.eye(k)

        # Ref: bekk.m:175–180 — build starting value vector for A/G/B
        sv_parts = []
        for i in range(p + o + q):
            # Ref: bekk.m:177 — temp = sqrt(startingVals(i))
            # sv_params is 0-indexed; MATLAB is 1-indexed
            temp_scalar = np.sqrt(np.abs(sv_params[i]))
            # Ref: bekk.m:178 — temp = temp * shape
            temp_scaled = temp_scalar * shape
            # Ref: bekk.m:179 — sv = [sv; temp(:)]
            # MATLAB temp(:) stacks column-major (order='F')
            sv_parts.append(np.atleast_1d(temp_scaled).ravel(order='F'))

        # Ref: bekk.m:181 — startingVals = [C; sv]
        starting_vals = np.concatenate(
            [C_vec.ravel(), np.concatenate(sv_parts)]
        )

    # ==================================================================
    # Section 4 — Bounds Construction
    # Ref: bekk.m:183–185
    # ==================================================================
    # Ref: bekk.m:183 — UB = .99998 * ones(size(startingVals))
    UB = 0.99998 * np.ones(len(starting_vals))
    # Ref: bekk.m:184 — UB(1:k2) = inf  (intercept params are unbounded)
    UB[:k2] = np.inf
    # Ref: bekk.m:185 — LB = -UB
    LB = -UB
    bounds = list(zip(LB.tolist(), UB.tolist()))

    # ==================================================================
    # Section 5 — Backcast Computation
    # Ref: bekk.m:186–193
    # ==================================================================
    # Ref: bekk.m:186 — m = ceil(sqrt(T))
    m = int(np.ceil(np.sqrt(T)))
    # Ref: bekk.m:187 — w = .06 * .94.^(0:(m-1))
    w = 0.06 * (0.94 ** np.arange(m))
    # Ref: bekk.m:188 — w = reshape(w/sum(w), [1 1 m])
    w = w / np.sum(w)
    w_3d = w.reshape(1, 1, m)
    # Ref: bekk.m:189 — backCast = sum(bsxfun(@times, w, data(:,:,1:m)), 3)
    back_cast = np.sum(w_3d * data[:, :, :m], axis=2)
    # Ref: bekk.m:190 — backCastAsym = backCast (default copy)
    back_cast_asym = back_cast.copy()
    # Ref: bekk.m:191–193 — override if o > 0
    if o > 0:
        back_cast_asym = np.sum(w_3d * data_asym[:, :, :m], axis=2)

    # ==================================================================
    # Section 6 — Constrained Optimisation
    # Ref: bekk.m:196–203
    # ==================================================================

    # Ref: bekk.m:200–201 — nonlinear stationarity constraint
    # bekk_constraint.py already returns values in scipy convention
    # (c >= 0 is feasible), so no additional sign flip is needed.
    # Ref: bekk_constraint.py:173 — "Negate for scipy convention"
    constraint = {
        'type': 'ineq',
        'fun': lambda x: bekk_constraint(
            x, data, data_asym, p, o, q,
            back_cast, back_cast_asym, type_code,
        )[0],
    }

    # Ref: bekk.m:199 — warning('off')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # Ref: bekk.m:200 — fmincon(@bekk_likelihood, startingVals, ...)
        result = minimize(
            _bekk_obj_for_minimize,
            starting_vals,
            args=(data, data_asym, p, o, q,
                  back_cast, back_cast_asym, type_code),
            method='SLSQP',
            bounds=bounds,
            constraints=constraint,
            options=options,
        )
    # Ref: bekk.m:201 — warning('on')

    parameters = np.asarray(result.x, dtype=np.float64)

    # Issue convergence warning if the optimizer did not converge
    if not result.success:
        warnings.warn(
            f"BEKK optimisation did not converge.  "
            f"Status: {result.status}.  Message: {result.message}",
            RuntimeWarning,
            stacklevel=2,
        )

    # ==================================================================
    # Section 7 — Final Likelihood Evaluation
    # Ref: bekk.m:202–203
    # ==================================================================
    # Ref: bekk.m:202 — [ll,~,Ht] = bekk_likelihood(parameters, ...)
    ll_neg, _, Ht = bekk_likelihood(
        parameters, data, data_asym, p, o, q,
        back_cast, back_cast_asym, type_code,
    )
    # Ref: bekk.m:203 — ll = -ll  (convert negative LL to positive LL)
    ll = float(-ll_neg)

    # ==================================================================
    # Section 8 — Robust Inference
    # Ref: bekk.m:207–209
    # ==================================================================
    # Ref: bekk.m:207 — if nargout>=4
    # In Python we always compute VCV and scores since the function
    # signature promises all five outputs.
    #
    # Ref: bekk.m:208 — [VCV,~,~,scores] = robustvcv(
    #          @bekk_likelihood, parameters, 0, data, ...)
    # robustvcv expects fun(theta, *args) → (scalar, 1-D array).
    # _bekk_ll_wrapper strips the third (Ht) output from bekk_likelihood.
    VCV, _, _, scores, _, _ = robustvcv(
        _bekk_ll_wrapper, parameters, 0,
        data, data_asym, p, o, q,
        back_cast, back_cast_asym, type_code,
    )

    return parameters, ll, Ht, VCV, scores
