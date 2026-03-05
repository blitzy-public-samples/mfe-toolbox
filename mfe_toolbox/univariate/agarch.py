"""
AGARCH(P,Q) and NAGARCH(P,Q) estimation driver.

Orchestrates the full estimation pipeline for Asymmetric GARCH (AGARCH,
Engle 1990) and Nonlinear Asymmetric GARCH (NAGARCH, Engle & Ng 1993)
models with four distributional assumptions: Normal, Student's t,
Generalized Error Distribution, and Hansen's skewed t.

Pipeline stages:
  1. Parameter validation  (agarch_parameter_check)
  2. Starting values       (agarch_starting_values)
  3. Parameter transform    (agarch_transform)
  4. Optimization           (scipy.optimize.minimize, method='L-BFGS-B')
  5. Inverse transform      (agarch_itransform)
  6. Inference              (robustvcv, hessian_2sided, gradient_2sided)
  7. Display                (agarch_display)

Migrated from: univariate/agarch.m (8794 bytes, ~220 lines)
Original author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
Revision: 1    Date: 7/12/2009

Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk

The conditional variance, h(t), of an AGARCH(P,Q) process is:

    h(t) = omega
           + alpha(1)*(r_{t-1} - gamma)^2 + ... + alpha(p)*(r_{t-p} - gamma)^2
           + beta(1)*h(t-1) + ... + beta(q)*h(t-q)

The conditional variance, h(t), of a NAGARCH(P,Q) process is:

    h(t) = omega
           + alpha(1)*(r_{t-1} - gamma*sqrt(h(t-1)))^2 + ...
           + alpha(p)*(r_{t-p} - gamma*sqrt(h(t-p)))^2
           + beta(1)*h(t-1) + ... + beta(q)*h(t-q)

See Also
--------
agarch_likelihood : Negative log-likelihood computation.
agarch_core : Numba JIT conditional variance recursion.
agarch_parameter_check : Input validation.
agarch_transform / agarch_itransform : Parameter space mappings.
"""

import numpy as np
from scipy.optimize import minimize

from mfe_toolbox.univariate.agarch_parameter_check import agarch_parameter_check
from mfe_toolbox.univariate.agarch_starting_values import agarch_starting_values
from mfe_toolbox.univariate.agarch_transform import agarch_transform
from mfe_toolbox.univariate.agarch_itransform import agarch_itransform
from mfe_toolbox.univariate.agarch_likelihood import agarch_likelihood
from mfe_toolbox.univariate.agarch_core import agarch_core
from mfe_toolbox.univariate.agarch_display import agarch_display
from mfe_toolbox.utility.robustvcv import robustvcv
from mfe_toolbox.utility.hessian_2sided import hessian_2sided
from mfe_toolbox.utility.gradient_2sided import gradient_2sided

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
# Integer code → string name mappings for display
_MODEL_TYPE_NAMES = {1: 'AGARCH', 2: 'NAGARCH'}
_ERROR_TYPE_NAMES = {1: 'NORMAL', 2: 'STUDENTST', 3: 'GED', 4: 'SKEWT'}


def agarch(
    epsilon: np.ndarray,
    p: int,
    q: int,
    model_type: str = 'AGARCH',
    error_type: str = 'NORMAL',
    startingvals: np.ndarray | None = None,
    options: dict | None = None,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Estimate an AGARCH(P,Q) or NAGARCH(P,Q) model.

    Orchestrates the complete estimation pipeline: input validation,
    starting value selection, parameter transformation, constrained
    optimisation via ``scipy.optimize.minimize(method='L-BFGS-B')``,
    inverse transformation, robust inference, and result display.

    Parameters
    ----------
    epsilon : np.ndarray
        A 1-D array (column vector) of mean-zero return data.
    p : int
        Positive integer — number of symmetric innovation (ARCH) lags.
    q : int
        Non-negative integer — number of lagged conditional variance
        (GARCH) terms.  Use ``q=0`` for a pure ARCH specification.
    model_type : str, optional
        Variance process type:

        - ``'AGARCH'``  — Asymmetric GARCH, Engle (1990) **[default]**
        - ``'NAGARCH'`` — Nonlinear Asymmetric GARCH, Engle & Ng (1993)
    error_type : str, optional
        Innovation distribution:

        - ``'NORMAL'``    — Gaussian innovations **[default]**
        - ``'STUDENTST'`` — Student's t distributed errors
        - ``'GED'``       — Generalized Error Distribution
        - ``'SKEWT'``     — Hansen's skewed Student's t
    startingvals : np.ndarray or None, optional
        Starting parameter vector of length ``(2+p+q)`` plus 1 for
        ``STUDENTST``/``GED`` (nu) or plus 2 for ``SKEWT`` (nu, lambda).
        Layout: ``[omega, alpha(1)...alpha(p), gamma, beta(1)...beta(q),
        [nu, [lambda]]]``.  When ``None``, starting values are computed
        automatically via a TARCH grid search.
    options : dict or None, optional
        Optimiser options dict passed to ``scipy.optimize.minimize``.
        When ``None``, defaults are:
        ``{'maxiter': 400*(2+p+q), 'disp': True, 'ftol': 1e-5,
        'gtol': 1e-5}``.

    Returns
    -------
    parameters : np.ndarray
        Estimated parameter vector: ``[omega, alpha(1)...alpha(p),
        gamma, beta(1)...beta(q), [nu, [lambda]]]``.
    LL : float
        Maximised log-likelihood value (positive).
    ht : np.ndarray
        Conditional variance series of length ``len(epsilon)``.
    VCVrobust : np.ndarray
        Robust (sandwich) variance-covariance matrix (K × K).
    VCV : np.ndarray
        Non-robust VCV ``inv(Hessian) / (T - m)`` (K × K).
    scores : np.ndarray
        Numerical score matrix (T × K), per-observation per-parameter.
    diagnostics : dict
        Optimisation diagnostics with keys ``'EXITFLAG'``,
        ``'ITERATIONS'``, ``'FUNCCOUNT'``, ``'MESSAGE'``.

    Raises
    ------
    ValueError
        If any input parameter fails validation (see
        :func:`agarch_parameter_check`).

    Notes
    -----
    - Replaces MATLAB ``fminunc`` / ``optimset`` with
      ``scipy.optimize.minimize(method='L-BFGS-B')`` and a Python
      options dict.
    - All array indexing is 0-based.  Non-obvious MATLAB→Python index
      adjustments are annotated with ``# Ref: agarch.m:<line>`` comments.
    - Numerical parity with MATLAB outputs to ±1e-6.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> eps = rng.standard_normal(1000) * 0.01
    >>> params, ll, ht, Vr, V, sc, diag = agarch(eps, 1, 1)
    """
    # ==================================================================
    # 1. Input Checking
    # Ref: agarch.m:82-95 — switch nargin / agarch_parameter_check
    # ==================================================================
    # Ensure epsilon is a 1-D float64 array for all subsequent operations.
    epsilon = np.asarray(epsilon, dtype=np.float64)
    if epsilon.ndim == 2 and epsilon.shape[1] == 1:
        epsilon = epsilon.ravel()
    elif epsilon.ndim != 1:
        raise ValueError('EPSILON must be a 1-D array (column vector).')

    # agarch_parameter_check in Python takes (data, p, q, error_type,
    # model_type, ...) — note model_type and error_type are swapped
    # relative to the MATLAB positional order.  Use keyword arguments
    # to avoid confusion.
    # Ref: agarch.m:84-92 — nargin-dependent call routing
    p, q, error_type_code, model_type_code, startingvals, options = (
        agarch_parameter_check(
            data=epsilon,
            p=p,
            q=q,
            error_type=error_type,
            model_type=model_type,
            startingvals=startingvals,
            options=options,
        )
    )

    # ==================================================================
    # 2. Initial Setup
    # Ref: agarch.m:102 — m = max([p q])
    # ==================================================================
    m = max(p, q)

    # ==================================================================
    # 3. Back-cast Computation
    # Ref: agarch.m:106-112 — EWMA-style exponentially decaying weights
    # ==================================================================
    T_orig = len(epsilon)
    # Ref: agarch.m:106 — back_cast_length = max(floor(length(epsilon)^(1/2)),1)
    back_cast_length = max(int(np.floor(T_orig ** 0.5)), 1)
    # Ref: agarch.m:107 — back_cast_weights = .05*(.9.^(0:back_cast_length))
    w = 0.05 * (0.9 ** np.arange(0, back_cast_length + 1))
    # Ref: agarch.m:108 — normalise weights
    w = w / np.sum(w)
    # Ref: agarch.m:109 — weighted sum of first (back_cast_length+1)
    #   squared residuals.  MATLAB: epsilon(1:back_cast_length+1) (1-indexed)
    #   Python: epsilon[0:back_cast_length+1] (0-indexed)
    back_cast = float(w @ (epsilon[0:back_cast_length + 1] ** 2))
    # Ref: agarch.m:110-112 — fallback if backcast is zero
    if back_cast == 0:
        # MATLAB cov(epsilon) for a vector returns scalar variance (ddof=1)
        back_cast = float(np.cov(epsilon))

    # ==================================================================
    # 4. Data Augmentation
    # Ref: agarch.m:113 — epsilon_augmented = [sqrt(back_cast)*ones(m,1); epsilon]
    # ==================================================================
    epsilon_augmented = np.concatenate(
        [np.sqrt(back_cast) * np.ones(m), epsilon]
    )
    # Ref: agarch.m:115 — T = size(epsilon_augmented,1)
    T = len(epsilon_augmented)

    # ==================================================================
    # 5. Starting Values
    # Ref: agarch.m:123-131
    # ==================================================================
    startingvals, nu, lambda_ = agarch_starting_values(
        startingvals, epsilon, p, q, model_type_code, error_type_code
    )
    # Ref: agarch.m:125 — startingvals = [startingvals; nu; lambda]
    # In MATLAB, appending [] is a no-op; in Python, we only append non-None.
    startingvals_full = _append_dist_params(startingvals, nu, lambda_)

    # Ref: agarch.m:127 — transform_bounds = quantile(epsilon,[.01 .99])
    transform_bounds = np.quantile(epsilon, [0.01, 0.99])

    # Ref: agarch.m:129 — agarch_transform
    garch_trans, nu_trans, lambda_trans = agarch_transform(
        startingvals_full, p, q, model_type_code, error_type_code,
        transform_bounds,
    )
    # Ref: agarch.m:131 — reassemble transformed vector
    startingvals_transformed = _append_dist_params(
        garch_trans, nu_trans, lambda_trans
    )

    # ==================================================================
    # 6. Optimisation
    # Ref: agarch.m:139-141
    # ==================================================================
    # Ref: agarch.m:139 — LL0 baseline for convergence check
    LL0 = agarch_likelihood(
        startingvals_transformed, epsilon_augmented, p, q,
        model_type_code, error_type_code, transform_bounds,
        back_cast, T, True,  # estim_flag=True → unconstrained space
    )[0]

    # Ref: agarch.m:141 — fminunc('agarch_likelihood', ...)
    # scipy.optimize.minimize requires a scalar-returning objective.
    opt_args = (
        epsilon_augmented, p, q, model_type_code, error_type_code,
        transform_bounds, back_cast, T, True,
    )
    result = minimize(
        fun=_scalar_objective,
        x0=startingvals_transformed,
        args=opt_args,
        method='L-BFGS-B',
        options=options,
    )

    # ==================================================================
    # 7. Estimation Robustness — retry on non-convergence with improvement
    # Ref: agarch.m:149-168
    # ==================================================================
    # Ref: agarch.m:149 — if exitflag<=0 && LL<LL0
    if not result.success and result.fun < LL0:
        num_params = len(result.x)
        retry_options = dict(options)
        # Ref: agarch.m:156-164 — increase iteration/evaluation budgets
        retry_options['maxiter'] = 2 * max(
            retry_options.get('maxiter', 100 * num_params),
            100 * num_params,
        )
        retry_options['maxfun'] = 4 * max(
            retry_options.get('maxfun', 100 * num_params),
            100 * num_params,
        )
        # Ref: agarch.m:167 — re-optimise from current best
        result = minimize(
            fun=_scalar_objective,
            x0=result.x,
            args=opt_args,
            method='L-BFGS-B',
            options=retry_options,
        )

    # ==================================================================
    # 8. Inverse Transform — map back to constrained parameter space
    # Ref: agarch.m:171-172
    # ==================================================================
    garch_params, nu_final, lambda_final = agarch_itransform(
        result.x, p, q, model_type_code, error_type_code, transform_bounds
    )
    # Ref: agarch.m:172 — parameters = [parameters; nu; lambda]
    parameters = _append_dist_params(garch_params, nu_final, lambda_final)

    # ==================================================================
    # 9. Final Log-Likelihood Evaluation
    # Ref: agarch.m:174-177 — compute LL, likelihoods, ht without estim_flag
    # ==================================================================
    LL_neg, _likelihoods, ht = agarch_likelihood(
        parameters, epsilon_augmented, p, q, model_type_code,
        error_type_code, transform_bounds, back_cast, T, False,
    )
    # Ref: agarch.m:176 — LL = -LL  (un-negate to get actual log-likelihood)
    LL = -LL_neg

    # ==================================================================
    # 10. Robust Inference — sandwich VCV, standard errors
    # Ref: agarch.m:181-183
    # ==================================================================
    nw = 0  # Ref: agarch.m:181 — no Newey-West lags on scores
    # robustvcv expects fun(theta, *args) → (scalar, 1-D array).
    # agarch_likelihood returns a 3-tuple; wrap to return first two elements.
    vcv_args = (
        epsilon_augmented, p, q, model_type_code, error_type_code,
        transform_bounds, back_cast, T,
    )
    VCVrobust, A, B, scores, hess, _gross_scores = robustvcv(
        _ll_for_vcv, parameters, nw, *vcv_args
    )
    # Ref: agarch.m:183 — VCV = hess^(-1) / (T - m)
    #   hess is the A matrix = H / T from robustvcv.
    VCV = np.linalg.inv(hess) / (T - m)

    # ==================================================================
    # 11. Display Results
    # Ref: AAP requirement — call agarch_display
    # Note: The original MATLAB agarch.m does not call display from the
    #   driver; it is a standalone utility.  We call it here per AAP but
    #   guard against edge-case failures (e.g., near-singular VCV on
    #   short data or overparameterized specifications) to ensure the
    #   function always returns results even if display fails.
    # ==================================================================
    model_type_str = _MODEL_TYPE_NAMES[model_type_code]
    error_type_str = _ERROR_TYPE_NAMES[error_type_code]
    try:
        agarch_display(
            parameters, LL, VCVrobust, epsilon, p, q,
            model_type_str, error_type_str,
        )
    except (ValueError, np.linalg.LinAlgError):
        # Display is informational only; estimation results are still valid.
        pass

    # ==================================================================
    # 12. Diagnostics
    # Ref: agarch.m:187-191
    # ==================================================================
    diagnostics: dict = {
        'EXITFLAG': 1 if result.success else 0,
        'ITERATIONS': getattr(result, 'nit', 0),
        'FUNCCOUNT': getattr(result, 'nfev', 0),
        'MESSAGE': getattr(result, 'message', ''),
    }

    return parameters, LL, ht, VCVrobust, VCV, scores, diagnostics


# ======================================================================
# Private Helpers
# ======================================================================

def _scalar_objective(
    params: np.ndarray,
    *args,
) -> float:
    """Scalar wrapper around agarch_likelihood for scipy.optimize.minimize.

    ``scipy.optimize.minimize`` requires the objective to return a scalar.
    ``agarch_likelihood`` returns ``(LL, LLS, ht)``; this helper extracts
    the first element (negated total log-likelihood).

    Parameters
    ----------
    params : np.ndarray
        Current parameter vector (in unconstrained space when
        estim_flag=True is in *args*).
    *args : tuple
        Forwarded to :func:`agarch_likelihood`.

    Returns
    -------
    float
        Negated total log-likelihood (scalar to minimise).
    """
    return agarch_likelihood(params, *args)[0]


def _ll_for_vcv(
    params: np.ndarray,
    *args,
) -> tuple[float, np.ndarray]:
    """Two-element wrapper around agarch_likelihood for robustvcv.

    ``robustvcv`` requires ``fun(theta, *args) → (scalar, 1-D array)``.
    ``agarch_likelihood`` returns ``(LL, LLS, ht)``; this helper returns
    ``(LL, LLS)`` and discards ``ht``.

    Parameters
    ----------
    params : np.ndarray
        Parameter vector (constrained space — estim_flag defaults to False).
    *args : tuple
        Forwarded to :func:`agarch_likelihood`.

    Returns
    -------
    tuple[float, np.ndarray]
        ``(LL, LLS)`` where LL is the negated total log-likelihood and
        LLS is the 1-D array of negated per-observation log-likelihoods.
    """
    LL, LLS, _ht = agarch_likelihood(params, *args)
    return LL, LLS


def _append_dist_params(
    garch_params: np.ndarray,
    nu: float | None,
    lambda_: float | None,
) -> np.ndarray:
    """Concatenate GARCH parameters with distribution shape parameters.

    In MATLAB, ``[params; nu; lambda]`` silently ignores empty (``[]``)
    values.  This helper replicates that behavior by only appending
    non-None values.

    Parameters
    ----------
    garch_params : np.ndarray
        Core GARCH parameter vector of length ``2 + p + q``.
    nu : float or None
        Distribution degrees-of-freedom / shape parameter.
    lambda_ : float or None
        Distribution skewness parameter (Skewed-t only).

    Returns
    -------
    np.ndarray
        Concatenated 1-D parameter vector.
    """
    parts = [np.asarray(garch_params, dtype=np.float64).ravel()]
    if nu is not None:
        parts.append(np.array([float(nu)], dtype=np.float64))
    if lambda_ is not None:
        parts.append(np.array([float(lambda_)], dtype=np.float64))
    if len(parts) == 1:
        return parts[0]
    return np.concatenate(parts)
