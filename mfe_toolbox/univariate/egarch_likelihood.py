"""
EGARCH(P,O,Q) log-likelihood computation for scipy.optimize.minimize.

Computes the negative log-likelihood, per-observation negative
log-likelihoods, and conditional variance series for an EGARCH(P,O,Q)
model.  Supports four innovation distributions: Normal, Student's t,
Generalized Error Distribution (GED), and Hansen's Skewed t.

EGARCH models the *log* of conditional variance, which naturally
enforces positivity of variance without explicit parameter constraints.
The ``upper`` bound caps the exponentiated variance to prevent
numerical overflow during the recursion.

During estimation (``estim_flag=True``), unconstrained parameters
produced by the optimizer are mapped to the constrained space via
:func:`egarch_itransform` before the variance recursion.  The returned
log-likelihood is **negated** so that ``scipy.optimize.minimize`` (a
minimizer) effectively maximizes the actual log-likelihood.

Migrated from: ``univariate/egarch_likelihood.m`` — MFE Toolbox Version 4.0
Original author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
Revision: 3    Date: 9/1/2005

Python migration: Blitzy Platform — MATLAB-to-Python 3.12 migration.
"""

import numpy as np

from mfe_toolbox.univariate.egarch_core import egarch_core
from mfe_toolbox.univariate.egarch_itransform import egarch_itransform
from mfe_toolbox.distributions.normloglik import normloglik
from mfe_toolbox.distributions.stdtloglik import stdtloglik
from mfe_toolbox.distributions.gedloglik import gedloglik
from mfe_toolbox.distributions.skewtloglik import skewtloglik

__all__ = ['egarch_likelihood']


def egarch_likelihood(
    parameters: np.ndarray,
    epsilon_aug: np.ndarray,
    p: int,
    o: int,
    q: int,
    error_type: int,
    back_cast: float,
    T: int,
    upper: float,
    estim_flag: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute the negative log-likelihood for an EGARCH(P, O, Q) model.

    This function is the objective function passed to
    ``scipy.optimize.minimize`` during EGARCH parameter estimation.
    It evaluates the EGARCH log-variance recursion via :func:`egarch_core`
    and then computes the distributional log-likelihood.

    Parameters
    ----------
    parameters : np.ndarray
        1-D parameter vector.  Layout depends on context:

        * **During estimation** (``estim_flag=True``): unconstrained
          parameters produced by the optimizer.  The vector is passed
          through :func:`egarch_itransform` to map to the constrained
          space before evaluation.
        * **After estimation** (``estim_flag=False``): constrained
          parameters::

              [omega, alpha_1, …, alpha_p,
               gamma_1, …, gamma_o,
               beta_1, …, beta_q,
               nu?,  lambda?]

          Distribution shape parameters ``nu`` and ``lambda`` are
          appended when ``error_type`` requires them (see below).

    epsilon_aug : np.ndarray
        1-D array of length *T* containing mean-zero residuals
        (epsilon), prepended with *m* = ``max(p, o, q)`` zero-padding
        values for the backcast initialization.
        Ref: egarch.m:105 — ``data_augmented = [zeros(m,1); data]``.
    p : int
        Positive integer, number of symmetric innovation lags (ARCH
        terms).  Must be >= 1.
    o : int
        Non-negative integer, number of asymmetric innovation lags.
        Use 0 for a symmetric EGARCH process.
    q : int
        Non-negative integer, number of conditional variance lags
        (GARCH terms).  Use 0 for a pure ARCH specification.
    error_type : int
        Innovation distribution type:

        * 1 — Normal (Gaussian)
        * 2 — Standardized Student's t
        * 3 — Generalized Error Distribution (GED)
        * 4 — Hansen's Skewed Student's t
    back_cast : float
        Scalar used to initialise the first *m* entries of the
        log-variance recursion.  Typically ``log(unconditional variance)``
        for EGARCH.
        Ref: egarch.m:98-103 — back_cast = log(weighted avg of squared data).
    T : int
        Total length of ``epsilon_aug`` (including zero-padding).
    upper : float
        Upper bound for conditional variance.  Values exceeding this
        bound are capped to prevent variance explosion during the
        exponentiation of the log-variance.  Typically
        ``10000 * max(epsilon_aug ** 2)`` as computed by the driver.
        Ref: egarch_likelihood.m:61 — ``upper=10000*max(data.^2)``.
    estim_flag : bool, optional
        If ``True``, parameters are in unconstrained optimizer space and
        are mapped to constrained space via :func:`egarch_itransform`
        before evaluation.  Defaults to ``False``.

    Returns
    -------
    LL : float
        **Negative** of the total log-likelihood (i.e. minus one times
        the log-likelihood).  Suitable for direct minimization.
    LLS : np.ndarray
        1-D array of length ``T - m`` containing the **negative**
        per-observation log-likelihoods.
    ht : np.ndarray
        1-D array of length ``T - m`` containing the conditional
        variances for the relevant observations (backcast-free).

    Notes
    -----
    * The sign convention follows the original MATLAB function:
      ``LL = -sum(log f(data | parameters))``.  This allows
      ``scipy.optimize.minimize`` to be used directly.
    * The return value ``ht`` is the *sliced* conditional variance
      series starting from observation ``m + 1`` (0-based index ``m``),
      matching the MATLAB convention ``ht(t)`` with ``t = (m+1):T``.
    * EGARCH uses log-variance parameterization, so omega, alpha, gamma,
      and beta are relatively unconstrained.  Only distribution shape
      parameters require transformation.
    * Numerical parity with the MATLAB implementation is maintained to
      ±1e-6 for all supported error distributions.

    See Also
    --------
    egarch : Main EGARCH estimation driver.
    egarch_core : Numba JIT log-variance recursion.
    egarch_itransform : Inverse parameter transformation.

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.univariate.egarch_likelihood import egarch_likelihood
    >>> rng = np.random.default_rng(42)
    >>> T_obs = 500
    >>> epsilon = rng.standard_normal(T_obs) * 0.01
    >>> p, o, q = 1, 1, 1
    >>> m = max(p, o, q)
    >>> epsilon_aug = np.concatenate([np.zeros(m), epsilon])
    >>> T_total = len(epsilon_aug)
    >>> upper = 10000.0 * float(np.max(epsilon_aug ** 2))
    >>> bc = float(np.log(np.var(epsilon)))
    >>> params = np.array([-0.1, 0.05, -0.03, 0.95])
    >>> LL, LLS, ht = egarch_likelihood(params, epsilon_aug, p, o, q,
    ...                                  1, bc, T_total, upper)
    >>> LL > 0  # negative of negative log-likelihood → positive for Normal
    True
    """
    # ------------------------------------------------------------------
    # Ensure input arrays are contiguous float64 for Numba / NumPy
    # ------------------------------------------------------------------
    parameters = np.asarray(parameters, dtype=np.float64).ravel()
    epsilon_aug = np.asarray(epsilon_aug, dtype=np.float64).ravel()

    # ------------------------------------------------------------------
    # Parameter transformation / parsing
    # Ref: egarch_likelihood.m:38-46
    # ------------------------------------------------------------------

    # Initialise distribution parameters to NaN (unused for Normal)
    nu: float = np.nan
    lam: float = np.nan  # 'lam' avoids shadowing Python's 'lambda' keyword

    if estim_flag:
        # Ref: egarch_likelihood.m:38-39 — If for estimation, transform
        # the parameters from unconstrained to constrained space
        parameters = egarch_itransform(parameters, p, o, q, error_type)

    # ------------------------------------------------------------------
    # Extract distribution shape parameters from the parameter vector
    # Ref: egarch_likelihood.m:48-56
    # ------------------------------------------------------------------
    if error_type == 2 or error_type == 3:
        # Ref: egarch_likelihood.m:49-51 — Separate nu from remaining
        # MATLAB 1-based index p+o+q+2 → Python 0-based index p+o+q+1
        nu = float(parameters[p + o + q + 1])
        # Ref: egarch_likelihood.m:51 — parameters=parameters(1:1+p+o+q)
        parameters = parameters[: 1 + p + o + q].copy()
    elif error_type == 4:
        # Ref: egarch_likelihood.m:53 — lambda=parameters(p+o+q+3)
        # MATLAB 1-based p+o+q+3 → Python 0-based p+o+q+2
        lam = float(parameters[p + o + q + 2])
        # Ref: egarch_likelihood.m:54 — nu=parameters(p+o+q+2)
        nu = float(parameters[p + o + q + 1])
        # Ref: egarch_likelihood.m:55 — parameters=parameters(1:1+p+o+q)
        parameters = parameters[: 1 + p + o + q].copy()
    # For error_type == 1 (Normal): no distribution parameters to extract

    # ------------------------------------------------------------------
    # Backcast length
    # Ref: egarch_likelihood.m:59 — m = max([p o q])
    # ------------------------------------------------------------------
    m: int = max(p, o, q)

    # ------------------------------------------------------------------
    # Compute conditional variances via EGARCH log-variance recursion
    # Ref: egarch_likelihood.m:63 —
    #   ht=egarch_core(data,parameters,back_cast,upper,p,o,q,m,T)
    # The upper bound prevents exp(log_variance) from overflowing.
    # egarch_core returns actual variances (exp of log-variance), not
    # log-variances.
    # ------------------------------------------------------------------
    ht = egarch_core(epsilon_aug, parameters, back_cast, upper, p, o, q, m, T)

    # ------------------------------------------------------------------
    # Select relevant observations (skip backcast padding)
    # Ref: egarch_likelihood.m:66-69 — t = (m+1):T [MATLAB 1-based]
    # Python 0-based equivalent: indices m through T-1 → slice [m:T]
    # ------------------------------------------------------------------
    ht_relevant = ht[m:T]
    data_relevant = epsilon_aug[m:T]

    # ------------------------------------------------------------------
    # Distribution log-likelihood computation
    # Each distribution function returns (LL, lls) where LL = sum(lls)
    # and both are the POSITIVE log-likelihood.  We negate to produce
    # the objective for minimization.
    # Ref: egarch_likelihood.m:71-88
    # ------------------------------------------------------------------
    if error_type == 1:
        # Normal (Gaussian) distribution
        # Ref: egarch_likelihood.m:73 — [LL, LLS] = normloglik(data, 0, ht)
        # normloglik requires x as shape (T, 1) and sigma2 as shape (T, 1)
        LL, LLS = normloglik(
            data_relevant.reshape(-1, 1),
            np.float64(0.0),
            ht_relevant.reshape(-1, 1),
        )
        # Ref: egarch_likelihood.m:74-75 — negate for minimization
        LLS = -LLS.ravel()
        LL = -float(LL)

    elif error_type == 2:
        # Standardized Student's t distribution
        # Ref: egarch_likelihood.m:77 — [LL, LLS] = stdtloglik(data, 0, ht, nu)
        LL, LLS = stdtloglik(data_relevant, np.float64(0.0), ht_relevant, nu)
        # Ref: egarch_likelihood.m:78-79
        LLS = -np.asarray(LLS, dtype=np.float64).ravel()
        LL = -float(LL)

    elif error_type == 3:
        # Generalized Error Distribution (GED)
        # Ref: egarch_likelihood.m:81 — [LL, LLS] = gedloglik(data, 0, ht, nu)
        LL, LLS = gedloglik(data_relevant, np.float64(0.0), ht_relevant, nu)
        # Ref: egarch_likelihood.m:82-83
        LLS = -np.asarray(LLS, dtype=np.float64).ravel()
        LL = -float(LL)

    elif error_type == 4:
        # Hansen's Skewed Student's t distribution
        # Ref: egarch_likelihood.m:85 — [LL, LLS] = skewtloglik(data, 0, ht, nu, lambda)
        LL, LLS = skewtloglik(data_relevant, np.float64(0.0), ht_relevant, nu, lam)
        # Ref: egarch_likelihood.m:86-87
        LLS = -np.asarray(LLS, dtype=np.float64).ravel()
        LL = -float(LL)

    else:
        raise ValueError(
            f"error_type must be 1, 2, 3, or 4 — received {error_type}"
        )

    return float(LL), LLS, ht_relevant
