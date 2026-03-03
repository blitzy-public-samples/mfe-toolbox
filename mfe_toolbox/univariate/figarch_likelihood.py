"""
Log-likelihood computation for FIGARCH(Q,D,P) estimation using ARCH(infinity)
truncated weight representation.

Computes the FIGARCH log-likelihood by constructing the ARCH(infinity)
representation with truncated weights. The conditional variance h(t) is
computed as a weighted sum of past squared residuals, where the weights are
derived from the fractional differencing parameter d, the AR parameter phi,
and the MA parameter beta via :func:`figarch_weights`.

This module is called both during optimization (with ``estim_flag=True`` for
automatic inverse parameter transform) and for final likelihood evaluation
(with ``estim_flag=False`` for already-constrained parameters).

Migrated from: ``univariate/figarch_likelihood.m`` (MFE Toolbox Version 4.0)

See Also
--------
mfe_toolbox.univariate.figarch : FIGARCH model driver.
mfe_toolbox.univariate.figarch_weights : ARCH(infinity) weight computation.
mfe_toolbox.univariate.figarch_itransform : Inverse parameter transformation.
mfe_toolbox.univariate.figarch_parameter_check : Parameter validation.
mfe_toolbox.univariate.figarch_starting_values : FIGARCH initialization.
mfe_toolbox.univariate.figarch_transform : FIGARCH parameter transformation.

Notes
-----
Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 1    Date: 7/12/2009
"""

import numpy as np

from mfe_toolbox.univariate.figarch_weights import figarch_weights
from mfe_toolbox.univariate.figarch_itransform import figarch_itransform
from mfe_toolbox.distributions.normloglik import normloglik
from mfe_toolbox.distributions.stdtloglik import stdtloglik
from mfe_toolbox.distributions.gedloglik import gedloglik
from mfe_toolbox.distributions.skewtloglik import skewtloglik


def figarch_likelihood(
    parameters: np.ndarray,
    epsilon: np.ndarray,
    p: int,
    q: int,
    error_type: int,
    truncLag: int,
    back_cast: float,
    T: int,
    estim_flag: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Compute the negative log-likelihood for a FIGARCH(Q,D,P) model.

    Uses the ARCH(infinity) truncated weight representation to compute
    conditional variances, then evaluates the log-likelihood under the
    specified error distribution. Returns negated values suitable for
    minimization-based optimizers (scipy.optimize.minimize).

    Parameters
    ----------
    parameters : np.ndarray
        Parameter vector. Layout depends on ``(p, q, error_type)``:

        - Core parameters: ``[omega, (phi if p=1), d, (beta if q=1)]``
        - Appended distribution params: ``[nu]`` for Student-t (2) or GED (3),
          ``[nu, lambda]`` for Skewed-t (4)

        When ``estim_flag`` is True, parameters are in unconstrained
        (transformed) space and will be inverse-transformed internally.
    epsilon : np.ndarray
        T-element (or longer) array of mean-zero residuals.
    p : int
        0 or 1 indicating whether the autoregressive (phi) term is present
        in the FIGARCH model.
    q : int
        0 or 1 indicating whether the moving-average (beta) term is present
        in the FIGARCH model.
    error_type : int
        Error distribution type:

        - 1 — Normal
        - 2 — Student's t
        - 3 — Generalized Error Distribution (GED)
        - 4 — Hansen's Skewed Student's t
    truncLag : int
        Truncation lag length for the ARCH(infinity) representation.
        Controls the number of past squared residuals used in the
        conditional variance computation.
    back_cast : float
        Backcast value used for pre-sample squared residuals. Typically
        the unconditional variance of the residuals.
    T : int
        Number of observations to use from ``epsilon``.
    estim_flag : bool, optional
        If True, inverse-transform parameters from unconstrained to
        constrained space before evaluation. Default is False.

    Returns
    -------
    LL : float
        Negative log-likelihood (i.e., ``-1 * actual LL``). Suitable for
        minimization-based optimizers.
    lls : np.ndarray
        T-element array of per-observation negative log-likelihoods.
    ht : np.ndarray
        T-element array of conditional variances.

    Raises
    ------
    ValueError
        If ``error_type`` is not in {1, 2, 3, 4}.

    Notes
    -----
    The conditional variance is computed as:

    .. math::

        h(t) = \\omega + \\sum_{i=1}^{\\mathrm{truncLag}} \\lambda(i)
               \\cdot \\varepsilon^{2}(t-i)

    where :math:`\\lambda(i)` are the ARCH(infinity) truncation weights from
    :func:`figarch_weights`, and pre-sample squared residuals
    :math:`\\varepsilon^{2}(t-i)` for :math:`t-i < 1` are replaced by
    ``back_cast``.

    The function internally constructs the augmented squared residual vector
    ``epsilon2 = [back_cast * ones(truncLag); epsilon[:T]**2]`` as described
    in the MATLAB source (Ref: figarch_likelihood.m docstring).

    References
    ----------
    .. [1] Baillie, R. T., Bollerslev, T., & Mikkelsen, H. O. (1996).
       Fractionally integrated generalized autoregressive conditional
       heteroskedasticity. *Journal of Econometrics*, 74(1), 3-30.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> eps = rng.standard_normal(200)
    >>> params = np.array([0.1, 0.3])  # [omega, d] for FIGARCH(0,d,0)
    >>> LL, lls, ht = figarch_likelihood(params, eps, 0, 0, 1, 100, 1.0, 200)
    >>> isinstance(LL, float)
    True
    >>> ht.shape
    (200,)
    """
    # ------------------------------------------------------------------
    # Input conversion
    # Ref: figarch_likelihood.m:1 — parameters, epsilon are column vectors
    # ------------------------------------------------------------------
    parameters = np.asarray(parameters, dtype=np.float64).ravel()
    epsilon = np.asarray(epsilon, dtype=np.float64).ravel()
    back_cast = np.float64(back_cast)

    # ------------------------------------------------------------------
    # Parameter extraction with optional inverse transform
    # Ref: figarch_likelihood.m:39-50
    # ------------------------------------------------------------------
    nu = None
    lam = None

    if estim_flag:
        # Ref: figarch_likelihood.m:40 — [parameters,nu,lambda] = figarch_itransform(...)
        # Inverse-transform from unconstrained optimizer space to constrained
        # FIGARCH parameter space. Returns core params and distribution params.
        parameters, nu, lam = figarch_itransform(parameters, p, q, error_type)
    else:
        # Ref: figarch_likelihood.m:42-49 — strip distribution parameters
        # from the end of the parameter vector
        if error_type == 2 or error_type == 3:
            # Ref: figarch_likelihood.m:43-44 — nu = parameters(end);
            #   parameters = parameters(1:end-1)
            nu = float(parameters[-1])
            parameters = parameters[:-1]
        elif error_type == 4:
            # Ref: figarch_likelihood.m:45-48 — nu = parameters(end-1);
            #   lambda = parameters(end); parameters = parameters(1:end-2)
            nu = float(parameters[-2])
            lam = float(parameters[-1])
            parameters = parameters[:-2]

    # ------------------------------------------------------------------
    # Extract core model parameters
    # Ref: figarch_likelihood.m:52-53
    # ------------------------------------------------------------------
    # Ref: figarch_likelihood.m:52 — omega = parameters(1) (MATLAB 1-based)
    omega = float(parameters[0])

    # Ref: figarch_likelihood.m:53 — figarchWeightParameters = parameters(2:2+p+q)
    # MATLAB 1-based parameters(2:2+p+q) → Python 0-based parameters[1:2+p+q]
    # This yields a (1+p+q)-element vector: [phi (if p), d, beta (if q)]
    figarch_weight_params = parameters[1:2 + p + q]

    # ------------------------------------------------------------------
    # Compute ARCH(infinity) truncation weights
    # Ref: figarch_likelihood.m:55 — archWeights = figarch_weights(...)
    # ------------------------------------------------------------------
    arch_weights = figarch_weights(figarch_weight_params, p, q, truncLag)

    # ------------------------------------------------------------------
    # Construct augmented squared residuals: [backcast; epsilon^2]
    # Ref: figarch_likelihood.m docstring —
    #   epsilon2 = [zeros(TRUNCLAG,1)+BACKCAST; EPSILON.^2]
    # Using np.empty for efficient pre-allocation, then fill segments.
    # ------------------------------------------------------------------
    epsilon2 = np.empty(truncLag + T, dtype=np.float64)
    epsilon2[:truncLag] = back_cast
    epsilon2[truncLag:truncLag + T] = epsilon[:T] ** 2

    # ------------------------------------------------------------------
    # Conditional variance computation via ARCH(infinity) representation
    # Ref: figarch_likelihood.m:56-61
    #   tau = truncLag+1:truncLag+T;  (MATLAB 1-based)
    #   ht = zeros(size(epsilon2));
    #   for t = tau
    #       ht(t) = omega + archWeights' * epsilon2(t-1:-1:t-truncLag);
    #   end
    #   ht = ht(tau);
    #
    # The MATLAB loop computes the dot product of archWeights with a
    # reversed window of truncLag squared residuals ending at t-1.
    #
    # Optimization: pre-reverse archWeights once so that the inner
    # product uses a forward slice epsilon2[i:i+truncLag] in each
    # iteration, avoiding per-iteration reversal overhead.
    #
    # Index mapping verification:
    #   MATLAB t=truncLag+1 (1-based) → Python i=0
    #     MATLAB: archWeights'*epsilon2(truncLag:-1:1) → 1st weight * backcast(truncLag) + ...
    #     Python: rev_weights * epsilon2[0:truncLag] → same elements (all backcast)  ✓
    #   MATLAB t=truncLag+T (1-based) → Python i=T-1
    #     MATLAB: archWeights'*epsilon2(truncLag+T-1:-1:T) → last T observations
    #     Python: rev_weights * epsilon2[T-1:T-1+truncLag] → same elements  ✓
    # ------------------------------------------------------------------
    rev_weights = arch_weights[::-1].copy()

    ht = np.zeros(T, dtype=np.float64)
    for i in range(T):
        # Ref: figarch_likelihood.m:59 — ht(t) = omega + archWeights' * epsilon2(t-1:-1:t-truncLag)
        # With pre-reversed weights: np.sum(rev_weights * epsilon2[i:i+truncLag])
        ht[i] = omega + np.sum(rev_weights * epsilon2[i:i + truncLag])

    # Numerical safeguard: clamp very small or negative conditional variances
    # to prevent NaN from np.log(negative) inside likelihood functions.
    # MATLAB would produce complex log values; Python np.log produces NaN.
    # This does not alter behavior for valid parameter configurations.
    ht = np.maximum(ht, 1e-20)

    # ------------------------------------------------------------------
    # Distribution-specific log-likelihood computation
    # Ref: figarch_likelihood.m:64-81 — switch on errorType
    # All log-likelihood functions return (LL, lls) as positive values;
    # we negate both for minimization-based optimization.
    # ------------------------------------------------------------------
    if error_type == 1:
        # Ref: figarch_likelihood.m:66 — [LL, LLS] = normloglik(epsilon, 0, ht)
        # normloglik requires column-vector inputs of shape (T, 1)
        eps_col = epsilon[:T].reshape(-1, 1)
        ht_col = ht.reshape(-1, 1)
        LL, lls = normloglik(eps_col, np.float64(0.0), ht_col)
        # Ref: figarch_likelihood.m:67-68 — LLS = -LLS; LL = -LL
        lls = -lls.ravel()
        LL = -LL
    elif error_type == 2:
        # Ref: figarch_likelihood.m:70 — [LL, LLS] = stdtloglik(epsilon, 0, ht, nu)
        LL, lls = stdtloglik(epsilon[:T], 0.0, ht, nu)
        # Ref: figarch_likelihood.m:71-72 — LLS = -LLS; LL = -LL
        lls = -lls
        LL = -LL
    elif error_type == 3:
        # Ref: figarch_likelihood.m:73 — [LL, LLS] = gedloglik(epsilon, 0, ht, nu)
        LL, lls = gedloglik(epsilon[:T], 0.0, ht, nu)
        # Ref: figarch_likelihood.m:74-75 — LLS = -LLS; LL = -LL
        lls = -lls
        LL = -LL
    elif error_type == 4:
        # Ref: figarch_likelihood.m:78 — [LL, LLS] = skewtloglik(epsilon,0,ht,nu,lambda)
        LL, lls = skewtloglik(epsilon[:T], 0.0, ht, nu, lam)
        # Ref: figarch_likelihood.m:79-80 — LLS = -LLS; LL = -LL
        lls = -lls
        LL = -LL
    else:
        raise ValueError(
            f"Invalid error_type: {error_type}. Must be 1 (Normal), "
            f"2 (Student-t), 3 (GED), or 4 (Skewed-t)."
        )

    return LL, lls, ht
