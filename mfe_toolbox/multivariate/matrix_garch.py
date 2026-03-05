"""
Symmetric and asymmetric MATRIX multivariate GARCH estimation.

Estimates K-dimensional Matrix GARCH(P,O,Q) models where conditional
covariance dynamics are specified with full K x K parameter matrices using
Hadamard (element-wise) products:

    H(t) = CC' + A(1)A(1)' .* r_{t-1}r_{t-1}' + ... + A(P)A(P)' .* r_{t-P}r_{t-P}'
               + G(1)G(1)' .* n_{t-1}n_{t-1}' + ... + G(O)G(O)' .* n_{t-O}n_{t-O}'
               + B(1)B(1)' .* H(t-1) + ... + B(Q)B(Q)' .* H(t-Q)

where ``n_t = r_t .* (r_t < 0)`` captures leverage effects.  When using
realized measures, ``RM_{t-1}`` replaces ``r_{t-1}r_{t-1}'``, and the
asymmetric realized measure replaces ``n_{t-1}n_{t-1}'``.

Each parameter matrix M is parameterized as ``LL'`` where L is a K x K lower
triangular Cholesky factor packed via :func:`~mfe_toolbox.utility.chol2vec.chol2vec`
/ :func:`~mfe_toolbox.utility.vec2chol.vec2chol`, ensuring positive
semi-definiteness.

Notes
-----
Migrated from ``multivariate/matrix_garch.m`` (MATLAB MFE Toolbox v4.0,
Revision 3, 10/28/2009).

Author: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk

Key migration changes from MATLAB to Python:

- ``fminunc`` replaced by ``scipy.optimize.minimize(method='L-BFGS-B')``
  for unconstrained optimization.
- ``optimset`` replaced by a Python dict passed to ``minimize(options=...)``.
- MATLAB 1-based indexing converted to Python 0-based indexing throughout.
- ``chol(M)'`` (MATLAB upper-triangular transposed to lower) replaced by
  ``numpy.linalg.cholesky(M)`` (directly returns lower-triangular).
- ``.*`` (Hadamard product) maps to ``*`` (numpy element-wise multiply).
- Diagnostics loop corrected from original ``1+p+o+1`` to ``1+p+o+q``
  (apparent typo in matrix_garch.m line 243; ``q`` mistyped as ``1``).

References
----------
Kevin Sheppard, MFE Toolbox Version 4.0 (2009).
    See also: MATRIX_GARCH_LIKELIHOOD, SCALAR_VT_VECH
"""

import warnings

import numpy as np
from scipy.optimize import minimize

from mfe_toolbox.multivariate.matrix_garch_likelihood import matrix_garch_likelihood
from mfe_toolbox.multivariate.scalar_vt_vech import scalar_vt_vech
from mfe_toolbox.utility.chol2vec import chol2vec
from mfe_toolbox.utility.vec2chol import vec2chol
from mfe_toolbox.utility.robustvcv import robustvcv


def _obj_wrapper(params, data, data_asym, p, o, q, back_cast, back_cast_asym):
    """Return scalar negative log-likelihood for ``scipy.optimize.minimize``.

    Wraps :func:`matrix_garch_likelihood` to return only the scalar
    objective value, discarding per-observation log-likelihoods and
    conditional covariance matrices.

    Ref: matrix_garch.m:206 — MATLAB ``fminunc`` captures only fval
    (first output); ``scipy.optimize.minimize`` expects a scalar return.
    """
    ll, _, _ = matrix_garch_likelihood(
        params, data, data_asym, p, o, q, back_cast, back_cast_asym
    )
    return float(ll)


def _ll_for_robustvcv(params, data, data_asym, p, o, q, back_cast, back_cast_asym):
    """Return ``(scalar_ll, per_obs_lls)`` tuple for :func:`robustvcv`.

    Wraps :func:`matrix_garch_likelihood` to return the 2-tuple expected
    by :func:`~mfe_toolbox.utility.robustvcv.robustvcv`, discarding the
    conditional covariance array.

    Ref: matrix_garch.m:230 — MATLAB ``robustvcv`` captures two outputs
    via ``nargout=2``; Python requires an explicit wrapper.
    """
    ll, lls, _ = matrix_garch_likelihood(
        params, data, data_asym, p, o, q, back_cast, back_cast_asym
    )
    return float(ll), lls


def matrix_garch(
    data: np.ndarray,
    data_asym: np.ndarray | None = None,
    p: int = 1,
    o: int = 0,
    q: int = 0,
    starting_vals: np.ndarray | None = None,
    options: dict | None = None,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Estimate symmetric and asymmetric MATRIX multivariate GARCH models.

    The conditional variance, H(t), of a MATRIX GARCH is modeled as::

        H(t) = CC' + A(1)A(1)' .* r_{t-1}'*r_{t-1} + ...
                   + A(P)A(P)' .* r_{t-P}'*r_{t-P}
                   + G(1)G(1)' .* n_{t-1}'*n_{t-1} + ...
                   + G(O)G(O)' .* n_{t-O}'*n_{t-O}
                   + B(1)B(1)' .* H(t-1) + ... + B(Q)B(Q)' .* H(t-Q)

    where ``n_t = r_t .* (r_t < 0)`` and ``.*`` is the Hadamard product.
    If using realized measures, ``RM_{t-1}`` replaces ``r_{t-1}'*r_{t-1}``,
    and the asymmetric version replaces ``n_{t-1}'*n_{t-1}``.

    Parameters
    ----------
    data : numpy.ndarray
        Either a T x K matrix of zero-mean residuals **or** a K x K x T
        3-D array of covariance estimators (e.g. realized covariance
        matrices).  When 2-D, the function internally computes outer
        products ``r_t @ r_t.T`` and the asymmetric counterpart.
    data_asym : numpy.ndarray or None, optional
        K x K x T array of asymmetric covariance estimators (e.g.
        realized covariance scaled by sign indicators).  Must be ``None``
        when ``data`` is 2-D (asymmetric data is auto-computed).
        Required to be non-empty when ``data`` is 3-D and ``o > 0``.
    p : int, optional
        Positive integer — number of lagged symmetric innovation terms
        (ARCH order).  Default is 1.
    o : int, optional
        Non-negative integer — number of asymmetric (leverage) lags.
        Default is 0.
    q : int, optional
        Non-negative integer — number of lagged conditional covariance
        terms (GARCH order).  Default is 0.
    starting_vals : numpy.ndarray or None, optional
        Initial parameter vector.  If ``None``, starting values are
        computed automatically from a scalar VT-VECH model via
        :func:`~mfe_toolbox.multivariate.scalar_vt_vech.scalar_vt_vech`.
    options : dict or None, optional
        Options dictionary passed to ``scipy.optimize.minimize``.  Keys
        such as ``'maxiter'``, ``'maxfun'``, ``'disp'`` are forwarded to
        the L-BFGS-B solver.  If ``None``, sensible defaults are used
        with ``maxfun = 1000 * K(K+1)/2 * (1+P+O+Q)``.

    Returns
    -------
    parameters : numpy.ndarray, shape (K(K+1)/2 * (1+P+O+Q),)
        Estimated parameter vector with layout::

            [vech(C)' vech(A(1))' ... vech(A(P))' vech(G(1))' ...
             vech(G(O))' vech(B(1))' ... vech(B(Q))']'

        where each vech block contains ``K(K+1)/2`` elements (lower
        triangular Cholesky factor packed via ``chol2vec``).
    ll : float
        Positive log-likelihood at the optimum.
    ht : numpy.ndarray, shape (K, K, T)
        3-D array of K x K conditional covariance matrices.
    VCV : numpy.ndarray, shape (numParams, numParams)
        Robust sandwich variance-covariance matrix
        (White / Bollerslev-Wooldridge).
    scores : numpy.ndarray, shape (T, numParams)
        Individual score contributions per observation and parameter.
    diagnostics : dict
        Optimization diagnostics with keys:

        - ``'EXITFLAG'``: Optimizer exit status (0 = converged).
        - ``'ITERATIONS'``: Number of iterations performed.
        - ``'FUNCCOUNT'``: Number of function evaluations.
        - ``'MESSAGE'``: Optimizer termination message.
        - ``'C'``: K x K intercept matrix ``CC'``.
        - ``'A'``: K x K x P array of ARCH matrices ``A(i)A(i)'``.
        - ``'G'``: K x K x O asymmetric matrices (``None`` if O=0).
        - ``'B'``: K x K x Q GARCH matrices (``None`` if Q=0).

    Raises
    ------
    ValueError
        If input dimensions are inconsistent, parameters violate
        constraints, or model specification is invalid.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> data = rng.standard_normal((500, 2))
    >>> params, ll, ht, vcv, scores, diag = matrix_garch(
    ...     data, p=1, o=0, q=1
    ... )

    See Also
    --------
    mfe_toolbox.multivariate.matrix_garch_likelihood :
        Log-likelihood evaluator.
    mfe_toolbox.multivariate.scalar_vt_vech :
        Scalar VT-VECH model used for starting value computation.
    mfe_toolbox.utility.chol2vec :
        Lower triangular to half-vec conversion.
    mfe_toolbox.utility.vec2chol :
        Half-vec to lower triangular reconstruction.
    mfe_toolbox.utility.robustvcv :
        Robust sandwich VCV estimator.
    """
    # ==================================================================
    # Section 1: Input Argument Checking
    # Ref: matrix_garch.m:55-167
    # ==================================================================

    data = np.asarray(data, dtype=np.float64)

    # --- Validate and preprocess data dimensions ---
    # Ref: matrix_garch.m:76-102
    if data.ndim == 2:
        # Ref: matrix_garch.m:77-87 — data is T x K; compute outer products
        t, k = data.shape
        if data_asym is not None:
            # Ref: matrix_garch.m:79
            raise ValueError(
                'If DATA is a T by K matrix, DATAASYM must be empty (None).'
            )
        # Ref: matrix_garch.m:81-87 — compute K x K x T outer products and
        # asymmetric outer products for negative returns
        temp = np.zeros((k, k, t))
        data_asym_3d = np.zeros((k, k, t))
        for i in range(t):
            # Ref: matrix_garch.m:84 — data(i,:)'*data(i,:)
            row = data[i, :]
            temp[:, :, i] = np.outer(row, row)
            # Ref: matrix_garch.m:85 — (data(i,:).*(data(i,:)<0))'*(...)
            neg_row = row * (row < 0)
            data_asym_3d[:, :, i] = np.outer(neg_row, neg_row)
        data = temp
        data_asym = data_asym_3d

    elif data.ndim == 3:
        # Ref: matrix_garch.m:88-102 — data is K x K x T 3-D array
        k, m, t = data.shape
        if m != k:
            # Ref: matrix_garch.m:91
            raise ValueError(
                'DATA must be K by K by T if a 3D array.'
            )
        if data_asym is not None:
            data_asym = np.asarray(data_asym, dtype=np.float64)
            # Ref: matrix_garch.m:94-95
            if data_asym.ndim != 3:
                raise ValueError(
                    'DATAASYM must be a 3D array with the same '
                    'dimensions as DATA'
                )
            # Ref: matrix_garch.m:97-99
            k2_a, m2, t2 = data_asym.shape
            if k2_a != k or m2 != m or t2 != t:
                raise ValueError(
                    'DATAASYM must be a 3D array with the same '
                    'dimensions as DATA'
                )
    else:
        raise ValueError(
            'DATA must be a T by K matrix or a K by K by T 3D array.'
        )

    # Ref: matrix_garch.m:103 — k2 = k*(k+1)/2
    k2 = k * (k + 1) // 2

    # Ref: matrix_garch.m:104-106 — T > K > 1 validation
    if min(t, k) < 2 or t < k:
        raise ValueError(
            'DATA must be a T by K matrix or a K by K by T 3D array, '
            'T>K>1'
        )

    # --- Validate p ---
    # Ref: matrix_garch.m:109-111
    if not isinstance(p, (int, np.integer)) or p < 1:
        raise ValueError('P must be a positive scalar')

    # --- Validate o ---
    # Ref: matrix_garch.m:112-120
    if o is None:
        o = 0
    if not isinstance(o, (int, np.integer)) or o < 0:
        raise ValueError('O must be a non-negative scalar')
    if o > 0 and data_asym is None:
        # Ref: matrix_garch.m:118-119
        raise ValueError('DATAASYM must be non-empty if O>0.')

    # --- Validate q ---
    # Ref: matrix_garch.m:122-127
    if q is None:
        q = 0
    if not isinstance(q, (int, np.integer)) or q < 0:
        raise ValueError('Q must be a non-negative scalar')

    # --- Validate starting values (minimal, as in MATLAB source) ---
    # Ref: matrix_garch.m:130-150
    # Note: The original MATLAB code has an incomplete validation for
    # user-provided starting values (documented as FIXME in the source).
    # This migration preserves the same minimal validation behavior.
    if starting_vals is not None:
        starting_vals = np.asarray(starting_vals, dtype=np.float64).ravel()
        # Ref: matrix_garch.m:134 — minimal length check
        if len(starting_vals) < (p + o + q):
            raise ValueError(
                'STARTINGVALS should be a P+O+Q by 1 vector'
            )
        # Ref: matrix_garch.m:140-149 — constraint validation on first
        # p+o+q elements interpreted as scalar parameters
        kappa = 2
        A_sv = starting_vals[:p]
        G_sv = starting_vals[p:p + o]
        B_sv = starting_vals[p + o:p + o + q]
        if (np.sum(A_sv) + np.sum(G_sv) / kappa
                + np.sum(B_sv)) >= 0.999998:
            raise ValueError(
                'Weighted sum of STARTINGVALUES must be less than 1. '
                'See Comments.'
            )
        if np.any(A_sv < 0) or np.any(B_sv < 0) or np.any(G_sv < 0):
            raise ValueError(
                'STARTINGVALS must all be nonnegative.'
            )

    # --- Setup optimizer options ---
    # Ref: matrix_garch.m:153-164
    # Replace optimset('fminunc') with scipy-compatible dict.
    if options is None:
        # Ref: matrix_garch.m:154-158
        # Display='iter' → 'disp': False (Pythonic default: silent)
        # MaxFunEvals = 1000*k2*(1+p+o+q)
        options = {
            'disp': False,
            'maxfun': 1000 * k2 * (1 + p + o + q),
            'maxiter': 500,
        }
    else:
        # Ref: matrix_garch.m:160-164 — validate options structure
        if not isinstance(options, dict):
            raise ValueError('OPTIONS is not a valid options dictionary')

    # ==================================================================
    # Section 2: Backcast Computation
    # Ref: matrix_garch.m:170-180
    # Exponentially weighted average of the first ``tau`` observations
    # for initializing pre-sample covariance values.
    # ==================================================================
    back_cast = np.zeros((k, k))
    back_cast_asym = np.zeros((k, k))
    # Ref: matrix_garch.m:172 — tau = max(ceil(sqrt(t)), k)
    tau = int(max(np.ceil(np.sqrt(t)), k))
    # Ref: matrix_garch.m:173 — weights = .06 * .94.^(0:tau)
    weights = 0.06 * (0.94 ** np.arange(tau + 1))
    weights = weights / np.sum(weights)
    # Ref: matrix_garch.m:175-180 — exponentially weighted initialization
    for i in range(tau):
        # Ref: matrix_garch.m:176 — MATLAB 1-indexed; Python 0-indexed
        back_cast = back_cast + weights[i] * data[:, :, i]
        if o > 0:
            # Ref: matrix_garch.m:178
            back_cast_asym = (
                back_cast_asym + weights[i] * data_asym[:, :, i]
            )

    # ==================================================================
    # Section 3: Starting Values from Scalar VT-VECH
    # Ref: matrix_garch.m:183-201
    # When no starting values are provided, estimate a simpler scalar
    # VT-VECH model first.  Its parameter matrices are decomposed via
    # Cholesky factorization and vectorized to form the initial parameter
    # vector for Matrix GARCH optimization.
    # ==================================================================
    if starting_vals is None:
        # Ref: matrix_garch.m:184-189 — relaxed options for starting values
        # TolX=1e-4, TolFun=1e-4 mapped to ftol/gtol for L-BFGS-B
        starting_options = {
            'ftol': 1e-4,
            'gtol': 1e-4,
            'maxiter': 200,
            'maxfun': 400,
        }

        # Ref: matrix_garch.m:190
        # scalar_vt_vech(data, dataAsym, p, o, q, [], [], startingOptions)
        sv_params, _, _, _, _, _, sv_diagnostics = scalar_vt_vech(
            data, data_asym, p, o, q, None, None, starting_options
        )

        # Ref: matrix_garch.m:191 — CpC = diagnostics.intercept
        CpC = sv_diagnostics['intercept']

        # Ref: matrix_garch.m:193 — startingvals = zeros(k2*(1+p+o+q), 1)
        starting_vals = np.zeros(k2 * (1 + p + o + q))

        # Ref: matrix_garch.m:194 — startingvals(1:k2) = chol2vec(chol(CpC)')
        # MATLAB chol() returns upper triangular; chol()' is lower tri.
        # numpy.linalg.cholesky() returns lower triangular directly.
        starting_vals[:k2] = chol2vec(np.linalg.cholesky(CpC))

        # Ref: matrix_garch.m:195-200 — fill A, G, B starting value blocks
        index = k2
        for i in range(p + o + q):
            # Ref: matrix_garch.m:197 — MATLAB 1-indexed; Python 0-indexed
            # matrixParameters = scalarVechStartingvals(i) *
            #   (.02*eye(k) + .98*ones(k))
            matrix_parameters = float(sv_params[i]) * (
                0.02 * np.eye(k) + 0.98 * np.ones((k, k))
            )
            # Ref: matrix_garch.m:198
            # chol2vec(chol(matrixParameters)')
            starting_vals[index:index + k2] = chol2vec(
                np.linalg.cholesky(matrix_parameters)
            )
            index += k2

    # ==================================================================
    # Section 4: Main Optimization
    # Ref: matrix_garch.m:204-216
    # Replace fminunc with scipy.optimize.minimize(method='L-BFGS-B')
    # ==================================================================

    # Ref: matrix_garch.m:204 — warning('off','MATLAB:illConditionedMatrix')
    # Suppress numerical warnings during optimization.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)

        # Ref: matrix_garch.m:205 — Evaluate initial log-likelihood for
        # the robustification convergence check
        ll0 = _obj_wrapper(
            starting_vals, data, data_asym, p, o, q,
            back_cast, back_cast_asym
        )

        # Ref: matrix_garch.m:206 — Main optimization
        # [parameters,ll,exitflag,output] =
        #   fminunc('matrix_garch_likelihood', startingvals, options, ...)
        result = minimize(
            _obj_wrapper,
            starting_vals,
            args=(data, data_asym, p, o, q, back_cast, back_cast_asym),
            method='L-BFGS-B',
            options=options,
        )

        # ==============================================================
        # Section 5: Estimation Robustification
        # Ref: matrix_garch.m:211-216
        # If the optimizer did not converge but the function value
        # improved over the starting point, retry with increased budget.
        # ==============================================================
        # Ref: matrix_garch.m:211 — if exitflag<=0 && ll<ll0
        if not result.success and result.fun < ll0:
            # Ref: matrix_garch.m:213-214
            retry_options = {
                'maxfun': 4 * 100 * (p + q),
                'maxiter': 2 * 100 * (p + q),
                'disp': False,
            }
            # Ref: matrix_garch.m:215
            result = minimize(
                _obj_wrapper,
                result.x,
                args=(
                    data, data_asym, p, o, q,
                    back_cast, back_cast_asym
                ),
                method='L-BFGS-B',
                options=retry_options,
            )

    # Ref: matrix_garch.m:217 — warning('on','MATLAB:illConditionedMatrix')
    # (Context manager restores warnings automatically.)

    parameters = result.x.copy()

    # ==================================================================
    # Section 6: Final Likelihood Evaluation
    # Ref: matrix_garch.m:222-224
    # ==================================================================
    # Ref: matrix_garch.m:222 — [ll,lls,ht] = matrix_garch_likelihood(...)
    ll_neg, lls, ht = matrix_garch_likelihood(
        parameters, data, data_asym, p, o, q,
        back_cast, back_cast_asym
    )
    # Ref: matrix_garch.m:223 — ll = -ll (negate for positive log-lik)
    ll = -ll_neg

    # ==================================================================
    # Section 7: Robust Sandwich VCV Computation
    # Ref: matrix_garch.m:230
    # [VCV,A,B,scores] = robustvcv('matrix_garch_likelihood',
    #                               parameters, 0, ...)
    # ==================================================================
    num_params = len(parameters)
    try:
        VCV, A_mat, B_mat, scores, _, _ = robustvcv(
            _ll_for_robustvcv, parameters, 0,
            data, data_asym, p, o, q, back_cast, back_cast_asym
        )
    except np.linalg.LinAlgError:
        # Ref: matrix_garch.m — In MATLAB, inv() on a singular Hessian
        # produces Inf/NaN; replicate with NaN-filled arrays rather than
        # crashing, consistent with MATLAB's behavior of returning Inf.
        warnings.warn(
            'Hessian is singular; robust VCV could not be computed. '
            'VCV and scores are filled with NaN.',
            RuntimeWarning,
            stacklevel=2,
        )
        VCV = np.full((num_params, num_params), np.nan)
        scores = np.full((t, num_params), np.nan)

    # ==================================================================
    # Section 8: Diagnostics
    # Ref: matrix_garch.m:236-260
    # Reconstruct full K x K positive semi-definite parameter matrices
    # from the optimized Cholesky-factor parameter vector.
    # ==================================================================
    diagnostics = {}
    # Ref: matrix_garch.m:237-240
    diagnostics['EXITFLAG'] = result.status
    diagnostics['ITERATIONS'] = result.nit
    diagnostics['FUNCCOUNT'] = result.nfev
    diagnostics['MESSAGE'] = result.message

    # Ref: matrix_garch.m:241-247 — Unpack parameter vector into matrices
    parameter_matrices = np.zeros((k, k, 1 + p + o + q))
    index = 0
    # Ref: matrix_garch.m:243 — Original MATLAB has ``for i=1:(1+p+o+1)``
    # which is an apparent typo for ``1+p+o+q`` (variable ``q`` mistyped
    # as literal ``1``).  Corrected here: iterate over all 1+p+o+q blocks.
    for i in range(1 + p + o + q):
        # Ref: matrix_garch.m:244 — temp = vec2chol(parameters(...))
        temp = vec2chol(parameters[index:index + k2])
        # Ref: matrix_garch.m:245 — parameterMatrices(:,:,i) = temp*temp'
        parameter_matrices[:, :, i] = temp @ temp.T
        index += k2

    # Ref: matrix_garch.m:248 — diagnostics.C (intercept matrix CC')
    diagnostics['C'] = parameter_matrices[:, :, 0].copy()

    # Ref: matrix_garch.m:249 — diagnostics.A = parameterMatrices(:,:,2:p+1)
    # Python 0-based: indices 1 to p inclusive → slice [1:p+1]
    diagnostics['A'] = parameter_matrices[:, :, 1:p + 1].copy()

    # Ref: matrix_garch.m:250-254 — diagnostics.G (asymmetric matrices)
    if o > 0:
        # Ref: matrix_garch.m:251 — parameterMatrices(:,:,p+2:p+o+1)
        # Python 0-based: indices p+1 to p+o inclusive → [p+1:p+o+1]
        diagnostics['G'] = parameter_matrices[
            :, :, p + 1:p + o + 1
        ].copy()
    else:
        diagnostics['G'] = None

    # Ref: matrix_garch.m:256-260 — diagnostics.B (GARCH matrices)
    if q > 0:
        # Ref: matrix_garch.m:257 — parameterMatrices(:,:,p+o+2:p+o+q+1)
        # Python 0-based: indices p+o+1 to p+o+q → [p+o+1:p+o+q+1]
        diagnostics['B'] = parameter_matrices[
            :, :, p + o + 1:p + o + q + 1
        ].copy()
    else:
        diagnostics['B'] = None

    return parameters, ll, ht, VCV, scores, diagnostics
