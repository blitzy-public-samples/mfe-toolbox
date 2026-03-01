"""
Log-likelihood computation for FIGARCH(Q,D,P) estimation.

Computes the FIGARCH log-likelihood using the ARCH(infinity) truncated weight
representation.  The conditional variance h(t) is computed as a weighted sum
of past squared residuals, where the weights are derived from the fractional
differencing parameter d, the AR parameter phi, and the MA parameter beta via
:func:`figarch_weights`.

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
    p: int,
    q: int,
    epsilon: np.ndarray,
    epsilon2: np.ndarray,
    trunc_lag: int,
    error_type: int,
    estim_flag: bool = False,
) -> tuple:
    """
    Compute negative log-likelihood for FIGARCH(Q,D,P) model.

    Evaluates the FIGARCH log-likelihood using the ARCH(infinity) truncated
    weight representation.  When called during optimization (``estim_flag=True``),
    the parameters are first inverse-transformed from unconstrained space.

    Parameters
    ----------
    parameters : np.ndarray
        Parameter vector.  Layout depends on (p, q, error_type):

        - Core: ``[omega, (phi if p=1), d, (beta if q=1)]``
        - Appended: ``[nu]`` for Student-t/GED, ``[nu, lambda]`` for Skewed-t

        When ``estim_flag=True``, parameters are in unconstrained (transformed)
        space and will be inverse-transformed internally.
    p : int
        0 or 1 indicating whether the autoregressive (phi) term is present.
    q : int
        0 or 1 indicating whether the moving-average (beta) term is present.
    epsilon : np.ndarray
        T-element array of mean-zero residuals.
    epsilon2 : np.ndarray
        (trunc_lag + T)-element array of squared residuals augmented with
        backcast values: ``[backcast * ones(trunc_lag); epsilon**2]``.
    trunc_lag : int
        Truncation lag length for the ARCH(infinity) representation.
    error_type : int
        Error distribution type: 1=Normal, 2=Student-t, 3=GED, 4=Skewed-t.
    estim_flag : bool, optional
        If ``True``, inverse-transform parameters before evaluation.
        Default is ``False``.

    Returns
    -------
    tuple
        - ``LL`` (float): Negative log-likelihood (i.e., -1 * actual LL).
        - ``lls`` (np.ndarray): T-element array of per-observation negative
          log-likelihoods.
        - ``ht`` (np.ndarray): T-element array of conditional variances.

    Notes
    -----
    Ref: figarch_likelihood.m:39-50 — Parameter extraction with optional
    inverse transform when ``estim_flag`` is true.

    Ref: figarch_likelihood.m:52-61 — ARCH(infinity) weight computation and
    conditional variance recursion using backcast-augmented squared residuals.

    Ref: figarch_likelihood.m:64-81 — Distribution-specific log-likelihood
    computation via normloglik, stdtloglik, gedloglik, or skewtloglik.
    """
    parameters = np.asarray(parameters, dtype=np.float64).ravel()

    # ----------------------------------------------------------------
    # Parameter extraction with optional inverse transform
    # Ref: figarch_likelihood.m:39-50
    # ----------------------------------------------------------------
    nu = None
    lam = None

    if estim_flag:
        # Ref: figarch_likelihood.m:40 — inverse transform from unconstrained
        params_core, nu, lam = figarch_itransform(parameters, p, q, error_type)
        # Reconstruct full parameter vector in constrained space
        parts = [params_core]
        if nu is not None:
            parts.append(np.array([nu]))
        if lam is not None:
            parts.append(np.array([lam]))
        parameters = np.concatenate(parts)
    else:
        # Ref: figarch_likelihood.m:42-50 — extract nu/lambda from tail
        if error_type == 2 or error_type == 3:
            # Ref: figarch_likelihood.m:43-44 — nu = parameters(end)
            nu = parameters[-1]
            parameters = parameters[:-1]
        elif error_type == 4:
            # Ref: figarch_likelihood.m:45-47 — nu = parameters(end-1), lambda = parameters(end)
            nu = parameters[-2]
            lam = parameters[-1]
            parameters = parameters[:-2]

    # ----------------------------------------------------------------
    # Extract core parameters and compute ARCH(infinity) weights
    # Ref: figarch_likelihood.m:52-55
    # ----------------------------------------------------------------
    omega = parameters[0]
    # Ref: figarch_likelihood.m:53 — figarchWeightParameters = parameters(2:2+p+q)
    # MATLAB 1-indexed: parameters(2:2+p+q) → Python 0-indexed: parameters[1:1+p+q+1]
    # This extracts the phi/d/beta subvector (excluding omega)
    figarch_weight_params = parameters[1:2 + p + q]

    T = len(epsilon)
    # Ref: figarch_likelihood.m:55 — archWeights = figarch_weights(figarchWeightParameters,p,q,truncLag)
    arch_weights = figarch_weights(figarch_weight_params, p, q, trunc_lag)

    # ----------------------------------------------------------------
    # Conditional variance computation via ARCH(infinity) representation
    # Ref: figarch_likelihood.m:56-61
    #   tau = truncLag+1:truncLag+T;
    #   ht = zeros(size(epsilon2));
    #   for t = tau
    #       ht(t) = omega + archWeights' * epsilon2(t-1:-1:t-truncLag);
    #   end
    #   ht = ht(tau);
    #
    # MATLAB indexing: tau is 1-based [truncLag+1, ..., truncLag+T]
    # Python indexing: tau is 0-based [truncLag, ..., truncLag+T-1]
    # epsilon2[t-1:-1:t-truncLag] in MATLAB = reversed slice of
    # trunc_lag elements ending at t-1.
    # In Python 0-based: epsilon2[t-trunc_lag:t] reversed → dot with arch_weights
    # ----------------------------------------------------------------
    ht_full = np.zeros(len(epsilon2))

    for t in range(trunc_lag, trunc_lag + T):
        # Ref: figarch_likelihood.m:59 — ht(t) = omega + archWeights' * epsilon2(t-1:-1:t-truncLag)
        # MATLAB: epsilon2(t-1:-1:t-truncLag) is a reversed slice of length truncLag
        # Python 0-based: epsilon2[t-truncLag:t] gives elements [t-truncLag, ..., t-1]
        # archWeights[0] corresponds to lag 1, so we need reversed order:
        # epsilon2[t-1], epsilon2[t-2], ..., epsilon2[t-truncLag]
        # Ref: figarch_likelihood.m:59 — MATLAB's t-1:-1:t-truncLag is length truncLag
        # Python: epsilon2[t-truncLag:t][::-1] reverses the forward slice correctly.
        # Note: cannot use epsilon2[t-1:t-truncLag-1:-1] because when
        # t-truncLag-1 evaluates to -1 Python interprets it as the last
        # element rather than "before index 0", producing an empty slice.
        reversed_eps2 = epsilon2[t - trunc_lag:t][::-1] if trunc_lag > 0 else np.array([])
        ht_full[t] = omega + np.dot(arch_weights, reversed_eps2)

    # Ref: figarch_likelihood.m:61 — ht = ht(tau)
    ht = ht_full[trunc_lag:trunc_lag + T]

    # Ensure conditional variances are positive for numerical stability
    ht = np.maximum(ht, 1e-20)

    # ----------------------------------------------------------------
    # Distribution-specific log-likelihood
    # Ref: figarch_likelihood.m:64-81
    # ----------------------------------------------------------------
    if error_type == 1:
        # Ref: figarch_likelihood.m:66 — [LL, LLS] = normloglik(epsilon, 0, ht)
        # normloglik requires (T, 1) column vector inputs; reshape and flatten.
        eps_col = epsilon.reshape(-1, 1)
        ht_col = ht.reshape(-1, 1)
        LL, lls = normloglik(eps_col, 0.0, ht_col)
        lls = -lls.ravel()
        LL = -LL
    elif error_type == 2:
        # Ref: figarch_likelihood.m:70 — [LL, LLS] = stdtloglik(epsilon, 0, ht, nu)
        LL, lls = stdtloglik(epsilon, 0.0, ht, nu)
        lls = -lls
        LL = -LL
    elif error_type == 3:
        # Ref: figarch_likelihood.m:74 — [LL, LLS] = gedloglik(epsilon, 0, ht, nu)
        LL, lls = gedloglik(epsilon, 0.0, ht, nu)
        lls = -lls
        LL = -LL
    elif error_type == 4:
        # Ref: figarch_likelihood.m:78 — [LL, LLS] = skewtloglik(epsilon, 0, ht, nu, lambda)
        LL, lls = skewtloglik(epsilon, 0.0, ht, nu, lam)
        lls = -lls
        LL = -LL
    else:
        raise ValueError(f'Unknown error_type: {error_type}')

    return LL, lls, ht
