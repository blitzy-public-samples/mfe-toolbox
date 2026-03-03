"""
Log-likelihood computation for the APARCH(P,O,Q) model.

Computes the negative log-likelihood, per-observation negative
log-likelihoods, and conditional variance series for the Asymmetric Power
ARCH (APARCH) model of Ding, Granger and Engle (1993), supporting Normal,
Student's *t*, GED, and Hansen's Skewed *t* error distributions.

The function serves as the objective for ``scipy.optimize.minimize`` during
APARCH estimation: when ``estim_flag`` is True the unconstrained optimizer
parameters are transformed to the constrained parameter space via
``aparch_itransform`` before evaluation.  When ``estim_flag`` is False the
parameters are expected in their constrained (model-native) representation.

Migrated from: ``univariate/aparch_likelihood.m`` (MFE Toolbox v4.0, 110 lines)
Original Author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
Revision: 3    Date: 9/1/2005

References
----------
Ding, Z., Granger, C.W.J. and Engle, R.F. (1993),
"A Long Memory Property of Stock Market Returns and a New Model",
*Journal of Empirical Finance*, 1, 83-106.
"""

import numpy as np

from mfe_toolbox.univariate.aparch_core import aparch_core
from mfe_toolbox.univariate.aparch_itransform import aparch_itransform
from mfe_toolbox.distributions.normloglik import normloglik
from mfe_toolbox.distributions.stdtloglik import stdtloglik
from mfe_toolbox.distributions.gedloglik import gedloglik
from mfe_toolbox.distributions.skewtloglik import skewtloglik


def aparch_likelihood(
    parameters: np.ndarray,
    data_aug: np.ndarray,
    abs_data_aug: np.ndarray,
    p: int,
    o: int,
    q: int,
    error_type: int,
    T: int,
    delta_is_estimated: int,
    user_delta: float | None = None,
    estim_flag: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Log-likelihood for APARCH(P,O,Q) estimation.

    Parameters
    ----------
    parameters : np.ndarray
        1-D parameter vector.  Layout depends on ``estim_flag``:

        * During optimisation (``estim_flag=True``): unconstrained
          parameters as produced by the optimiser.
        * Otherwise: constrained model parameters
          ``[omega, alpha(1:p), gamma(1:o), beta(1:q), delta?, nu?, lambda?]``
          where ``delta`` is present only when ``delta_is_estimated == 1``,
          and ``nu``/``lambda`` are present for non-Normal distributions.
    data_aug : np.ndarray
        1-D float64 array of length ``T`` containing mean-zero residuals
        augmented with ``m = max(p, o, q)`` leading zeros.
    abs_data_aug : np.ndarray
        Absolute value of ``data_aug``.  Preserved in the function signature
        for API parity with the MATLAB source; the Python ``aparch_core``
        computes absolute values internally.
    p : int
        Positive integer — number of symmetric innovation (ARCH) lags.
    o : int
        Non-negative integer — number of asymmetric innovation lags.
    q : int
        Non-negative integer — number of conditional variance (GARCH) lags.
    error_type : int
        Innovation distribution type:

        * 1 — Normal
        * 2 — Standardized Student's *t*
        * 3 — Generalized Error Distribution (GED)
        * 4 — Hansen's Skewed Student's *t*
    T : int
        Total length of ``data_aug`` (including prepended zeros).
    delta_is_estimated : int
        ``1`` if the power parameter delta is jointly estimated, ``0`` if a
        user-supplied delta is provided via ``user_delta``.
    user_delta : float or None, optional
        Fixed power parameter value when ``delta_is_estimated == 0``.
        Ignored when ``delta_is_estimated == 1``.
    estim_flag : bool, optional
        If ``True`` the parameters are transformed from the unconstrained
        optimiser space to the constrained model space via
        ``aparch_itransform``.  Default is ``False``.
        Ref: aparch_likelihood.m:23-26 — MATLAB ``nargin==11``

    Returns
    -------
    LL : float
        Negative total log-likelihood (for minimisation).
    LLS : np.ndarray
        1-D array of per-observation negative log-likelihoods.
    ht : np.ndarray
        1-D array of conditional variances for observations ``m+1`` to ``T``.

    Notes
    -----
    The return values ``LL`` and ``LLS`` are **negated** so that the function
    can be used directly as an objective for ``scipy.optimize.minimize``
    (which performs minimisation).  This matches the MATLAB convention where
    the return is "minus 1 times the log-likelihood".

    See Also
    --------
    aparch : Main APARCH estimation driver.
    aparch_core : APARCH conditional variance recursion (Numba JIT).
    aparch_itransform : Inverse parameter transformation.
    """
    # ------------------------------------------------------------------
    # Initialise distribution shape parameters
    # ------------------------------------------------------------------
    nu: float | None = None
    lambda_param: float | None = None

    # Ensure parameters is a writable 1-D float64 copy
    parameters = np.asarray(parameters, dtype=np.float64).flatten().copy()

    # ==================================================================
    # Step 1: Parameter Transformation
    # Ref: aparch_likelihood.m:39-60
    # ==================================================================
    if estim_flag:
        # Transform from unconstrained (optimizer) space to constrained
        # Ref: aparch_likelihood.m:42
        parameters, nu, lambda_param = aparch_itransform(
            parameters, p, o, q, error_type, delta_is_estimated
        )
        # If delta is NOT estimated, append the user-supplied value
        # Ref: aparch_likelihood.m:43-45
        if not delta_is_estimated:
            parameters = np.append(parameters, user_delta)
    else:
        # Parameters are already in constrained space; parse distribution
        # shape parameters from the tail of the vector.
        # Ref: aparch_likelihood.m:47-59
        if error_type == 2 or error_type == 3:
            # Student's t or GED — extract nu
            # Ref: aparch_likelihood.m:50 — MATLAB 1-indexed
            #   parameters(p+o+q+2+deltaIsEstimated)
            #   → Python 0-indexed: parameters[p+o+q+1+delta_is_estimated]
            nu = float(parameters[p + o + q + 1 + delta_is_estimated])
            # Truncate to core model parameters (omega, alpha, gamma, beta, delta)
            # Ref: aparch_likelihood.m:51 — parameters=parameters(1:2+p+o+q)
            parameters = parameters[: 2 + p + o + q].copy()
        elif error_type == 4:
            # Hansen's Skewed t — extract nu and lambda
            # Ref: aparch_likelihood.m:53-54
            nu = float(parameters[p + o + q + 1 + delta_is_estimated])
            lambda_param = float(
                parameters[p + o + q + 2 + delta_is_estimated]
            )
            # Truncate to core model parameters
            # Ref: aparch_likelihood.m:55 — parameters=parameters(1:2+p+o+q)
            parameters = parameters[: 2 + p + o + q].copy()
        # For error_type == 1 (Normal): no distribution parameters to parse

        # Append user-supplied delta if not jointly estimated
        # Ref: aparch_likelihood.m:57-59
        if not delta_is_estimated:
            parameters = np.append(parameters, user_delta)

    # ==================================================================
    # Step 2: Back-cast Computation
    # Ref: aparch_likelihood.m:62-73
    # ==================================================================
    # Maximum lag order
    # Ref: aparch_likelihood.m:62 — m = max([p o q])
    m: int = max(p, o, q)

    # Extract delta from the constrained parameter vector
    # Ref: aparch_likelihood.m:64 — MATLAB parameters(1+p+o+q+1)
    #   → Python 0-indexed: parameters[p+o+q+1]
    delta: float = float(parameters[p + o + q + 1])

    # Slice the effective data (after the augmented zero prefix)
    # Ref: aparch_likelihood.m:66 — MATLAB data_aug(m+1:T) → Python [m:T]
    data = data_aug[m:T]

    # Back-cast length = floor(sqrt(len(data))), at least 1
    # Ref: aparch_likelihood.m:67
    back_cast_length: int = max(int(np.floor(len(data) ** 0.5)), 1)

    # Exponentially decaying weights: 0.05 * 0.9^k for k = 0, 1, ..., back_cast_length
    # Ref: aparch_likelihood.m:68
    back_cast_weights = 0.05 * (0.9 ** np.arange(back_cast_length + 1))

    # Normalise weights to sum to 1
    # Ref: aparch_likelihood.m:69
    back_cast_weights = back_cast_weights / np.sum(back_cast_weights)

    # Weighted average of |data|^delta
    # Ref: aparch_likelihood.m:70
    # MATLAB * for row × column = dot product → Python @ operator
    back_cast: float = float(
        back_cast_weights @ (np.abs(data[: back_cast_length + 1]) ** delta)
    )

    # Fallback when back-cast is zero (degenerate data)
    # Ref: aparch_likelihood.m:71-73
    if back_cast == 0.0:
        back_cast = float(np.mean(np.abs(data_aug[m:T]) ** delta))

    # ==================================================================
    # Step 3: Variance Bounds (preserved for MATLAB faithfulness)
    # Ref: aparch_likelihood.m:76-78
    # Note: The Python aparch_core does not accept LB/UB parameters;
    # these are computed here to maintain source-code correspondence.
    # ==================================================================
    # Lower bound — MATLAB cov(x) on a vector returns scalar variance
    # Ref: aparch_likelihood.m:77 — cov(data_aug(m+1:T))/100000
    LB: float = float(np.var(data_aug[m:T], ddof=1) / 100000.0)  # noqa: F841

    # Upper bound
    # Ref: aparch_likelihood.m:78 — 100*max(data_aug.^2)
    UB: float = float(100.0 * np.max(data_aug ** 2))  # noqa: F841

    # ==================================================================
    # Step 4: Core APARCH Variance Recursion
    # Ref: aparch_likelihood.m:82
    # MATLAB: ht = aparch_core(data_aug, abs_data_aug, parameters, p, o, q,
    #                          m, T, back_cast, LB, UB)
    # Python aparch_core has a simplified signature: abs values computed
    # internally and LB/UB bounds omitted (see aparch_core.py notes).
    # ==================================================================
    ht: np.ndarray = aparch_core(
        data_aug, parameters, back_cast, p, o, q, m, T
    )

    # ==================================================================
    # Step 5: Log-Likelihood Computation
    # Ref: aparch_likelihood.m:86-109
    # ==================================================================
    # Slice to the relevant observations (drop the augmented prefix)
    # Ref: aparch_likelihood.m:86 — MATLAB t=(m+1):T → Python slice [m:T]
    ht = ht[m:T]
    data = data_aug[m:T]

    # Dispatch to the appropriate distribution log-likelihood
    # All log-likelihoods are NEGATED for minimisation (scipy.optimize.minimize)
    # Ref: aparch_likelihood.m:90-109
    if error_type == 1:
        # NORMAL distribution
        # Ref: aparch_likelihood.m:92 — normloglik expects (T,1) column vectors
        LL, LLS = normloglik(data.reshape(-1, 1), 0, ht.reshape(-1, 1))
        # Flatten LLS from (T,1) to 1-D and negate
        # Ref: aparch_likelihood.m:93-94
        LLS = -LLS.flatten()
        LL = -LL
    elif error_type == 2:
        # STUDENTST distribution
        # Ref: aparch_likelihood.m:99
        LL, LLS = stdtloglik(data, 0, ht, nu)
        # Ref: aparch_likelihood.m:100-101
        LLS = -LLS
        LL = -LL
    elif error_type == 3:
        # GED distribution
        # Ref: aparch_likelihood.m:103
        LL, LLS = gedloglik(data, 0, ht, nu)
        # Ref: aparch_likelihood.m:104-105
        LLS = -LLS
        LL = -LL
    elif error_type == 4:
        # SKEWT distribution — uses lambda_param (Python reserved word avoidance)
        # Ref: aparch_likelihood.m:107
        LL, LLS = skewtloglik(data, 0, ht, nu, lambda_param)
        # Ref: aparch_likelihood.m:108-109
        LLS = -LLS
        LL = -LL
    else:
        raise ValueError(
            f"Invalid error_type: {error_type}. Must be 1, 2, 3, or 4."
        )

    # ==================================================================
    # Return negated log-likelihood, per-obs negated LLs, and variance
    # Ref: aparch_likelihood.m:1 — [LL, LLS, ht]
    # ==================================================================
    return float(LL), np.asarray(LLS, dtype=np.float64), np.asarray(ht, dtype=np.float64)
