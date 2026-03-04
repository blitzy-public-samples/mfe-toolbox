"""
IGARCH(P,Q) parameter estimation driver with unit-root constraint.

Estimates Integrated GARCH (IGARCH) models where the sum of all ARCH and
GARCH coefficients equals 1.  Supports four error distributions (Normal,
Student-t, GED, Skewed-t) and two variance process formulations (standard
IGARCH modelling variance in squares, and IAVGARCH modelling in absolute
values).

The IGARCH unit-root constraint ``sum(alpha) + sum(beta) = 1`` is enforced
by estimating only ``q-1`` free betas; the last beta is computed as
``1 - sum(alpha) - sum(beta_1:q-1)`` inside ``igarch_core``.

Migrated from: univariate/igarch.m — MFE Toolbox Version 4.0 (12223 bytes)
Original author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
Revision: 3    Date: 9/1/2005

Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk

See Also
--------
igarch_likelihood : IGARCH negative log-likelihood evaluation.
igarch_core : Numba JIT conditional variance recursion.
igarch_parameter_check : Input validation and default assignment.
igarch_starting_values : Grid search for starting parameter values.
igarch_transform : Constrained → unconstrained parameter mapping.
igarch_itransform : Unconstrained → constrained parameter mapping.
"""

import warnings

import numpy as np
from scipy.optimize import minimize

from mfe_toolbox.univariate.igarch_parameter_check import igarch_parameter_check
from mfe_toolbox.univariate.igarch_starting_values import igarch_starting_values
from mfe_toolbox.univariate.igarch_transform import igarch_transform
from mfe_toolbox.univariate.igarch_itransform import igarch_itransform
from mfe_toolbox.univariate.igarch_likelihood import igarch_likelihood
from mfe_toolbox.utility.robustvcv import robustvcv


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _to_1d(value):
    """Convert *value* to a flat 1-D ``np.ndarray``, or ``np.empty(0)``."""
    if value is None:
        return np.empty(0, dtype=np.float64)
    arr = np.atleast_1d(np.asarray(value, dtype=np.float64)).ravel()
    return arr


def _concat_parts(*arrays):
    """Concatenate arrays, silently dropping empty ones."""
    parts = [_to_1d(a) for a in arrays]
    parts = [p for p in parts if p.size > 0]
    if parts:
        return np.concatenate(parts)
    return np.array([], dtype=np.float64)


def _build_scipy_options(options_dict):
    """Extract SLSQP-compatible options from the validated options dict.

    The parameter-check module produces a dict that may contain keys
    intended for L-BFGS-B (``gtol``).  SLSQP only recognises ``ftol``,
    ``maxiter``, ``disp``, and ``eps``.  Unknown keys are silently
    discarded to avoid ``scipy`` warnings.
    """
    _allowed = {'ftol', 'maxiter', 'disp', 'eps'}
    return {k: v for k, v in options_dict.items() if k in _allowed}


def _compute_ordered_candidates(
    epsilon, p, q, igarch_type_int, constant_int, back_cast, T
):
    """Replicate the MATLAB grid-search to produce *ordered* alternative
    starting-value candidates for the robustness loop.

    The MATLAB ``igarch_starting_values`` returns ``orderedParameters``
    (all grid candidates sorted by likelihood) as a 5th output.  The
    Python port only returns the best candidate, so we recompute the
    full sorted grid here.

    Ref: igarch_starting_values.m:57 — a=[.05 .1 .2]
    """
    # Ref: igarch_starting_values.m:57
    alpha_grid = np.array([0.05, 0.1, 0.2])
    n_candidates = len(alpha_grid)
    n_params = constant_int + p + max(q - 1, 0)

    candidates = np.zeros((n_candidates, n_params))
    lls = np.zeros(n_candidates)

    # Ref: igarch_starting_values.m:67-73 — compute adj_factor and base backcast
    if igarch_type_int == 1:
        # AVGARCH (absolute-value formulation)
        adj_factor = np.sqrt(2.0 / np.pi)
        bc_val = float(np.mean(np.abs(epsilon)))
    else:
        # GARCH (squared formulation)
        adj_factor = 1.0
        bc_val = float(np.var(epsilon, ddof=1))

    for idx, alpha_total in enumerate(alpha_grid):
        # Ref: igarch_starting_values.m:81
        temp_alpha = alpha_total * np.ones(p) / p
        # Ref: igarch_starting_values.m:83
        beta_total = 1.0 - np.sum(temp_alpha)

        # Ref: igarch_starting_values.m:85-89
        if constant_int:
            omega = np.array([bc_val * 0.01 * adj_factor])
        else:
            omega = np.array([], dtype=np.float64)

        # Ref: igarch_starting_values.m:92-96
        if q > 1:
            free_betas = beta_total * np.ones(q - 1) / q
        else:
            free_betas = np.array([], dtype=np.float64)

        candidate = np.concatenate([omega, temp_alpha, free_betas])
        candidates[idx, :] = candidate

        # Evaluate under Normal distribution (error_type=1) for ranking.
        # Ref: igarch_starting_values.m:100
        ll_val, _, _ = igarch_likelihood(
            candidate, epsilon, p, q, igarch_type_int, 1,
            back_cast, T, constant_int, False,
        )
        lls[idx] = ll_val

    # Ref: igarch_starting_values.m:105 — sort ascending (lowest neg-LL first)
    sorted_indices = np.argsort(lls)
    return candidates[sorted_indices]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def igarch(
    epsilon: np.ndarray,
    p: int,
    q: int,
    igarch_type: str = 'GARCH',
    error_type: str = 'NORMAL',
    startingvals: np.ndarray | None = None,
    options: dict | None = None,
    constant: bool = True,
) -> tuple[
    np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict
]:
    """Estimate an IGARCH(P,Q) model via maximum likelihood.

    IGARCH models impose the unit-root constraint that the sum of all ARCH
    and GARCH coefficients equals 1.  Estimation is performed using
    ``scipy.optimize.minimize`` with ``method='SLSQP'``, with the
    constraint enforced implicitly by estimating only ``q-1`` free betas
    (the last beta is a residual computed inside ``igarch_core``).

    Parameters
    ----------
    epsilon : np.ndarray
        T-element column vector (or 1-D array) of mean-zero residuals.
    p : int
        Positive integer — number of ARCH (alpha) lags.  Must be >= 1.
    q : int
        Positive integer — number of GARCH (beta) lags.  Must be >= 1.
    igarch_type : {'GARCH', 'AVGARCH'}, optional
        Variance process formulation:

        * ``'GARCH'`` — Model evolves in squares (default).
          Maps internally to ``igarch_type=0``.
        * ``'AVGARCH'`` — Model evolves in absolute values.
          Maps internally to ``igarch_type=1``.

        Ref: igarch.m:21-23 — MATLAB ``IGARCHTYPE`` 2 (squares, default)
        maps to Python ``'GARCH'``; MATLAB 1 (abs) maps to ``'AVGARCH'``.
    error_type : {'NORMAL', 'STUDENTST', 'GED', 'SKEWT'}, optional
        Innovation distribution.  Default is ``'NORMAL'``.
    startingvals : np.ndarray or None, optional
        Starting values for the optimizer.  When ``None`` (default), a
        grid search selects initial values automatically.  If provided,
        must be a 1-D vector of length:

        * NORMAL   : ``constant + p + q - 1``
        * STUDENTST: ``constant + p + q``   (extra nu)
        * GED      : ``constant + p + q``   (extra nu)
        * SKEWT    : ``constant + p + q + 1`` (extra nu + lambda)

    options : dict or None, optional
        Options dict forwarded to ``scipy.optimize.minimize``.  When
        ``None``, sensible defaults mirroring the MATLAB ``optimset``
        are created by ``igarch_parameter_check``.
    constant : bool, optional
        Whether to include an intercept (omega) in the variance equation.
        Default is ``True`` (include).  Ref: igarch.m:24-25.

    Returns
    -------
    parameters : np.ndarray
        Estimated parameter vector
        ``[omega, alpha_1…alpha_p, beta_1…beta_{q-1}, nu?, lambda?]``.
        The final beta is **not** stored; it equals
        ``1 - sum(alpha) - sum(beta_1:q-1)``.
    LL : float
        Maximised log-likelihood (positive).
    ht : np.ndarray
        Estimated conditional variances, length ``len(epsilon)``.
    VCVrobust : np.ndarray
        Robust (sandwich) variance-covariance matrix.
    VCV : np.ndarray
        Non-robust variance-covariance matrix (inverse Hessian / T).
    scores : np.ndarray
        Matrix of numerical scores (T × k).
    diagnostics : dict
        Optimisation diagnostics with keys ``'EXITFLAG'``,
        ``'ITERATIONS'``, ``'FUNCCOUNT'``, ``'MESSAGE'``.

    Raises
    ------
    ValueError
        If any input parameter is invalid (delegated to
        ``igarch_parameter_check``).

    Notes
    -----
    The conditional variance of an IGARCH(P,Q) model is:

    .. math::

        g(h_t) = \\omega + \\sum_{i=1}^{p} \\alpha_i f(\\varepsilon_{t-i})
                 + \\sum_{j=1}^{q} \\beta_j g(h_{t-j})

    where :math:`f(x) = x^2,\\; g(x) = x` for ``'GARCH'`` and
    :math:`f(x) = |x|,\\; g(x) = \\sqrt{x}` for ``'AVGARCH'``.

    The constraints (generally wrong as noted in the original MATLAB):

    1. :math:`\\omega > 0` (if constant)
    2. :math:`\\alpha_i \\ge 0`
    3. :math:`\\beta_j \\ge 0`
    4. :math:`\\sum \\alpha_i + \\sum \\beta_j = 1` (unit-root)
    5. :math:`\\nu > 2` for Student-t; :math:`\\nu > 1` for GED
    6. :math:`-0.99 < \\lambda < 0.99` for Skewed-t

    References
    ----------
    Engle, R.F. and Bollerslev, T. (1986). "Modelling the Persistence of
    Conditional Variances". Econometric Reviews, 5, 1-50.

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.univariate.igarch import igarch
    >>> rng = np.random.default_rng(42)
    >>> eps = rng.standard_normal(500)
    >>> params, ll, ht, vcvr, vcv, sc, diag = igarch(eps, 1, 1)
    >>> params.shape[0] >= 2  # at least omega + alpha_1
    True

    See Also
    --------
    igarch_likelihood : Negative log-likelihood evaluation.
    igarch_core : Numba JIT variance recursion.
    """
    # ==================================================================
    # 0. Map string arguments to integer codes
    # Ref: igarch.m:21-23 — MATLAB igarchType {1,2} mapped to Python {1,0}
    #   'GARCH'   → 0 (squares; matches MATLAB default igarchType=2)
    #   'AVGARCH'  → 1 (absolute values; matches MATLAB igarchType=1)
    # ==================================================================
    _igarch_type_map = {'GARCH': 0, 'AVGARCH': 1}
    if igarch_type is None:
        igarch_type_int: int | None = None
    elif isinstance(igarch_type, str):
        _key = igarch_type.upper().strip()
        if _key not in _igarch_type_map:
            raise ValueError(
                "igarch_type must be 'GARCH' or 'AVGARCH', "
                f"got '{igarch_type}'"
            )
        igarch_type_int = _igarch_type_map[_key]
    else:
        igarch_type_int = int(igarch_type)

    constant_int = int(bool(constant))

    # ==================================================================
    # 1. Input checking — delegates to igarch_parameter_check
    # Ref: igarch.m:90-105 — MATLAB nargin switch replaced by Python
    #   keyword arguments with defaults.
    # ==================================================================
    (
        p, q, error_type_int, igarch_type_int, constant_int,
        sv_checked, opt_dict,
    ) = igarch_parameter_check(
        epsilon, p, q, error_type, igarch_type_int,
        constant_int, startingvals, options,
    )

    # Ensure epsilon is a proper 1-D float64 array for all subsequent use
    epsilon = np.asarray(epsilon, dtype=np.float64).ravel()

    # ==================================================================
    # 2. Initial setup
    # Ref: igarch.m:112 — m = max([p q])
    # ==================================================================
    m: int = max(p, q)

    # ==================================================================
    # 3. Back-cast computation
    # Ref: igarch.m:116-140
    # Exponentially decaying weights on the first sqrt(T) observations.
    # ==================================================================
    n_obs: int = len(epsilon)
    # Ref: igarch.m:121 — backCastLength = max(floor(length(epsilon)^(1/2)),1)
    back_cast_length: int = max(int(np.floor(np.sqrt(n_obs))), 1)
    # Ref: igarch.m:122 — backCastWeights = .05*(.9.^(0:backCastLength))
    back_cast_weights = 0.05 * (0.9 ** np.arange(back_cast_length + 1))
    # Ref: igarch.m:123 — normalise weights
    back_cast_weights = back_cast_weights / np.sum(back_cast_weights)

    if igarch_type_int == 1:
        # Ref: igarch.m:117-127 — AVGARCH (absolute values)
        back_cast = float(
            back_cast_weights @ np.abs(epsilon[: back_cast_length + 1])
        )
        if back_cast == 0:
            # Ref: igarch.m:125-126
            back_cast = float(np.mean(np.abs(epsilon)))
    else:
        # Ref: igarch.m:128-139 — GARCH (squares)
        back_cast = float(
            back_cast_weights @ (epsilon[: back_cast_length + 1] ** 2)
        )
        if back_cast == 0:
            # Ref: igarch.m:138-139 — cov(epsilon) for a vector is var(…, ddof=1)
            back_cast = float(np.var(epsilon, ddof=1))

    # Ref: igarch.m:142 — T = size(fepsilon,1) = len(epsilon) + m
    T: int = n_obs + m

    # ==================================================================
    # 4. Starting values
    # Ref: igarch.m:149-164
    # ==================================================================
    # Ref: igarch.m:149-153 — flag whether user supplied starting values
    if sv_checked is not None and sv_checked.size > 0:
        starting_flag: int = 1
    else:
        starting_flag = 0

    # Ref: igarch.m:155 — grid search (or parse user values)
    sv_out, nu, lambda_param = igarch_starting_values(
        sv_checked, epsilon, p, q, igarch_type_int, error_type_int,
        constant_int,
    )

    # Ref: igarch.m:157 — startingvals = [startingvals; nu; lambda]
    sv_full = _concat_parts(
        sv_out,
        np.array([nu]) if nu is not None else None,
        np.array([lambda_param]) if lambda_param is not None else None,
    )

    # Ref: igarch.m:159 — transform starting vals to unconstrained space
    garch_trans, nu_trans, lambda_trans = igarch_transform(
        sv_full, p, q, error_type_int, constant_int,
    )
    # Ref: igarch.m:161 — reassemble transformed vector
    sv_transformed = _concat_parts(garch_trans, nu_trans, lambda_trans)

    # ==================================================================
    # 5. Define the optimisation objective
    # Ref: igarch.m:169 — LL0 = igarch_likelihood(…, true)
    # The likelihood function returns the *negative* log-likelihood when
    # estim_flag=True (parameters in unconstrained space).
    # ==================================================================
    def _objective(params):
        """Negative log-likelihood for scipy minimiser."""
        ll_val, _, _ = igarch_likelihood(
            params, epsilon, p, q, igarch_type_int, error_type_int,
            back_cast, T, constant_int, True,
        )
        return ll_val

    # Ref: igarch.m:169 — initial LL for convergence checking
    LL0: float = _objective(sv_transformed)

    # Build SLSQP-compatible options
    scipy_opts = _build_scipy_options(opt_dict)

    # ==================================================================
    # 6. First optimisation attempt
    # Ref: igarch.m:171
    # ==================================================================
    result = minimize(
        _objective, sv_transformed, method='SLSQP', options=scipy_opts,
    )
    parameters = result.x.copy()
    LL: float = float(result.fun)
    exitflag: int = 1 if result.success else 0
    output = result

    # ==================================================================
    # 7. Robustness check 1 — retry with doubled iterations
    # Ref: igarch.m:179-198
    # If not converged but the LL improved on the starting value,
    # retry with more iterations (analogous to MATLAB steepest descent).
    # ==================================================================
    if exitflag <= 0 and LL < LL0:
        # Ref: igarch.m:186-194 — double maxiter/maxfunevals
        retry_maxiter = scipy_opts.get('maxiter', 200 * len(parameters)) * 2
        retry_opts = {**scipy_opts, 'maxiter': retry_maxiter}

        # Ref: igarch.m:197
        result = minimize(
            _objective, parameters, method='SLSQP', options=retry_opts,
        )
        parameters = result.x.copy()
        LL = float(result.fun)
        exitflag = 1 if result.success else 0
        output = result

    # ==================================================================
    # 8. Robustness check 2 — alternative starting values
    # Ref: igarch.m:202-251
    # Only triggered when the user did NOT supply starting values and
    # convergence still has not been achieved.
    #
    # The MATLAB ``igarch_starting_values`` returns ``orderedParameters``
    # (all grid candidates sorted by likelihood).  The Python port
    # returns only the best candidate, so we replicate the grid here
    # via ``_compute_ordered_candidates``.
    # ==================================================================
    if starting_flag == 0 and exitflag <= 0:
        ordered_params = _compute_ordered_candidates(
            epsilon, p, q, igarch_type_int, constant_int, back_cast, T,
        )

        # Ref: igarch.m:206-209
        robust_parameters: list[np.ndarray] = [parameters.copy()]
        robust_ll: list[float] = [LL]

        # Ref: igarch.m:211 — index starts at 2 (MATLAB 1-indexed);
        # Python 0-indexed: best candidate was index 0, try 1, 2, …
        alt_index: int = 1

        while exitflag <= 0:
            # Ref: igarch.m:243-249 — break when candidates exhausted
            if alt_index >= len(ordered_params):
                warnings.warn(
                    'Convergence not achieved.  Use results with caution',
                    stacklevel=2,
                )
                best_idx = int(np.argmin(robust_ll))
                LL = robust_ll[best_idx]
                parameters = robust_parameters[best_idx].copy()
                break

            # Ref: igarch.m:217 — assemble alternative starting values
            alt_sv = ordered_params[alt_index]
            alt_full = _concat_parts(
                alt_sv,
                np.array([nu]) if nu is not None else None,
                np.array([lambda_param]) if lambda_param is not None else None,
            )

            # Ref: igarch.m:219 — transform to unconstrained space
            alt_garch_trans, alt_nu_trans, alt_lambda_trans = igarch_transform(
                alt_full, p, q, error_type_int, constant_int,
            )
            alt_transformed = _concat_parts(
                alt_garch_trans, alt_nu_trans, alt_lambda_trans,
            )

            # Ref: igarch.m:223 — evaluate initial LL for this candidate
            LL0_alt: float = _objective(alt_transformed)

            # Ref: igarch.m:224-226 — optimise from alternative start
            result = minimize(
                _objective, alt_transformed, method='SLSQP',
                options=scipy_opts,
            )
            parameters = result.x.copy()
            LL = float(result.fun)
            exitflag = 1 if result.success else 0
            output = result

            # Ref: igarch.m:227-236 — retry with more iterations
            if exitflag <= 0 and LL < LL0_alt:
                retry_maxiter = (
                    scipy_opts.get('maxiter', 200 * len(parameters)) * 2
                )
                retry_opts = {**scipy_opts, 'maxiter': retry_maxiter}
                result = minimize(
                    _objective, parameters, method='SLSQP',
                    options=retry_opts,
                )
                parameters = result.x.copy()
                LL = float(result.fun)
                exitflag = 1 if result.success else 0
                output = result

            # Ref: igarch.m:238-239 — store results for this candidate
            robust_parameters.append(parameters.copy())
            robust_ll.append(LL)
            # Ref: igarch.m:241
            alt_index += 1

    # ==================================================================
    # 9. Inverse-transform parameters back to constrained space
    # Ref: igarch.m:254
    # ==================================================================
    params_constrained, nu_final, lambda_final = igarch_itransform(
        parameters, p, q, error_type_int, constant_int,
    )
    # Ref: igarch.m:255 — parameters=[parameters; nu; lambda]
    parameters_out = _concat_parts(
        params_constrained, nu_final, lambda_final,
    )

    # ==================================================================
    # 10. Compute final log-likelihood and conditional variances
    # Ref: igarch.m:257-260
    # ==================================================================
    LL_neg, _lls, ht = igarch_likelihood(
        parameters_out, epsilon, p, q, igarch_type_int, error_type_int,
        back_cast, T, constant_int, False,
    )
    # Ref: igarch.m:259 — LL = -LL (convert negative to positive LL)
    LL_out: float = -LL_neg

    # ==================================================================
    # 11. Robust variance-covariance matrix
    # Ref: igarch.m:263-266
    # ==================================================================
    def _ll_for_vcv(params, *args):
        """Wrapper returning (scalar_LL, individual_lls) for robustvcv."""
        ll_scalar, ll_individual, _ = igarch_likelihood(params, *args)
        return ll_scalar, ll_individual

    # Ref: igarch.m:264 — nw=0 (no Newey-West)
    # Ref: igarch.m:265
    VCVrobust, _A, _B, scores, hess, _gs = robustvcv(
        _ll_for_vcv, parameters_out, 0,
        epsilon, p, q, igarch_type_int, error_type_int,
        back_cast, T, constant_int, False,
    )

    # Ref: igarch.m:266 — VCV = hess^(-1) / (T-m)
    # hess from robustvcv is actually A = numerical_hessian / T.
    VCV = np.linalg.inv(hess) / (T - m)

    # ==================================================================
    # 12. Diagnostics
    # Ref: igarch.m:270-273
    # ==================================================================
    diagnostics: dict = {
        'EXITFLAG': exitflag,
        'ITERATIONS': getattr(output, 'nit', 0),
        'FUNCCOUNT': getattr(output, 'nfev', 0),
        'MESSAGE': getattr(output, 'message', ''),
    }

    return parameters_out, LL_out, ht, VCVrobust, VCV, scores, diagnostics
