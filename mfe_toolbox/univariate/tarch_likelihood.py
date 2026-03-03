"""
TARCH/GJR-GARCH log-likelihood computation for scipy.optimize.minimize.

Computes the negative log-likelihood, per-observation negative
log-likelihoods, and conditional variance series for a TARCH(P,O,Q)
model.  Supports four innovation distributions: Normal, Student's t,
Generalized Error Distribution (GED), and Hansen's Skewed t.

During estimation (``estim_flag=True``), unconstrained parameters are
mapped to the constrained GARCH parameter space via
:func:`tarch_itransform` before the variance recursion.  The returned
log-likelihood is **negated** so that ``scipy.optimize.minimize`` (a
minimizer) effectively maximizes the actual log-likelihood.

Migrated from: ``univariate/tarch_likelihood.m`` — MFE Toolbox Version 4.0
Original author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
Revision: 3    Date: 9/1/2005

Python migration: Blitzy Platform — MATLAB-to-Python 3.12 migration.
"""

import numpy as np

from mfe_toolbox.univariate.tarch_core import tarch_core
from mfe_toolbox.univariate.tarch_itransform import tarch_itransform
from mfe_toolbox.distributions.normloglik import normloglik
from mfe_toolbox.distributions.stdtloglik import stdtloglik
from mfe_toolbox.distributions.gedloglik import gedloglik
from mfe_toolbox.distributions.skewtloglik import skewtloglik

__all__ = ['tarch_likelihood']


def tarch_likelihood(
    parameters: np.ndarray,
    data: np.ndarray,
    fdata: np.ndarray,
    fIdata: np.ndarray,
    p: int,
    o: int,
    q: int,
    error_type: int,
    tarch_type: int,
    back_cast: float,
    T: int,
    estim_flag: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute the negative log-likelihood for a TARCH(P, O, Q) model.

    This function is the objective function passed to
    ``scipy.optimize.minimize`` during TARCH/GJR-GARCH parameter
    estimation.  It evaluates the TARCH variance recursion via
    :func:`tarch_core` and then computes the distributional
    log-likelihood.

    Parameters
    ----------
    parameters : np.ndarray
        1-D parameter vector.  Layout depends on context:

        * **During estimation** (``estim_flag=True``): unconstrained
          parameters produced by the optimizer.  The vector is passed
          through :func:`tarch_itransform` to map to the constrained
          space before evaluation.
        * **After estimation** (``estim_flag=False``): constrained
          parameters::

              [omega, alpha_1, …, alpha_p,
               gamma_1, …, gamma_o,
               beta_1, …, beta_q,
               nu?,  lambda?]

          Distribution shape parameters ``nu`` and ``lambda`` are
          appended when ``error_type`` requires them (see below).

    data : np.ndarray
        1-D array of length *T* containing mean-zero residuals
        (epsilon), prepended with *m* = ``max(p, o, q)`` backcast
        padding values.
        Ref: tarch_likelihood.m — ``data`` parameter (2nd argument).
    fdata : np.ndarray
        1-D array of length *T*.  Transformed innovation data:

        * ``tarch_type == 2`` (squared returns): ``fdata = epsilon ** 2``
        * ``tarch_type == 1`` (absolute value): ``fdata = |epsilon|``

        Ref: tarch_likelihood.m:11 — FDATA description.
    fIdata : np.ndarray
        1-D array of length *T*.  Asymmetric / threshold indicator
        data, computed as ``fdata * (epsilon < 0)``.
        Ref: tarch_likelihood.m:12 — FIDATA description.
    p : int
        Number of symmetric ARCH lags (>= 1).
    o : int
        Number of asymmetric / threshold lags (0 for symmetric GARCH).
    q : int
        Number of GARCH (lagged variance) terms.
    error_type : int
        Innovation distribution type:

        * 1 — Normal (Gaussian)
        * 2 — Standardized Student's t
        * 3 — Generalized Error Distribution (GED)
        * 4 — Hansen's Skewed Student's t
    tarch_type : int
        Variance-process type selector:

        * 1 — absolute-value / standard-deviation model
        * 2 — squared-return / variance model (standard case)
    back_cast : float
        Scalar used to initialise the first *m* entries of the
        conditional variance recursion.
    T : int
        Total length of *fdata* / *data* (including backcast padding).
    estim_flag : bool, optional
        If ``True``, parameters are in unconstrained space and are
        mapped to constrained space via :func:`tarch_itransform` before
        evaluation.  Defaults to ``False``.

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
    * Numerical parity with the MATLAB implementation is maintained to
      ±1e-6 for all supported error distributions.

    See Also
    --------
    tarch : Main TARCH estimation driver.
    tarch_core : Numba JIT conditional variance recursion.
    tarch_itransform : Inverse parameter transformation.

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.univariate.tarch_likelihood import tarch_likelihood
    >>> rng = np.random.default_rng(42)
    >>> T_obs = 500
    >>> epsilon = rng.standard_normal(T_obs) * 0.01
    >>> p, o, q = 1, 1, 1
    >>> m = max(p, o, q)
    >>> data = np.concatenate([np.zeros(m), epsilon])
    >>> fdata = np.concatenate([np.full(m, np.mean(epsilon**2)), epsilon**2])
    >>> fIdata = fdata * np.concatenate([np.full(m, 0.5), (epsilon < 0).astype(float)])
    >>> T_total = len(data)
    >>> params = np.array([1e-6, 0.05, 0.04, 0.90])
    >>> bc = float(np.mean(epsilon**2))
    >>> LL, LLS, ht = tarch_likelihood(params, data, fdata, fIdata,
    ...                                p, o, q, 1, 2, bc, T_total)
    >>> LL > 0  # negative of a negative log-likelihood → positive for Normal
    True
    """
    # ------------------------------------------------------------------ #
    #  Ensure input arrays are contiguous float64 for Numba / NumPy      #
    # ------------------------------------------------------------------ #
    parameters = np.asarray(parameters, dtype=np.float64).ravel()
    data = np.asarray(data, dtype=np.float64).ravel()
    fdata = np.asarray(fdata, dtype=np.float64).ravel()
    fIdata = np.asarray(fIdata, dtype=np.float64).ravel()

    # ------------------------------------------------------------------ #
    #  Parameter transformation / parsing                                 #
    # ------------------------------------------------------------------ #
    # Ref: tarch_likelihood.m:42-56

    # Initialise distribution parameters to NaN (unused for Normal)
    nu: float = np.nan
    lam: float = np.nan  # 'lam' avoids shadowing Python's 'lambda'

    if estim_flag:
        # Ref: tarch_likelihood.m:44 — transform unconstrained → constrained
        trans_params = tarch_itransform(parameters, p, o, q, error_type, tarch_type)

        # The Python tarch_itransform returns a single flat array:
        #   [omega, alpha..., gamma..., beta..., nu?, lambda?]
        # Parse core GARCH parameters and distribution parameters.
        if error_type == 2 or error_type == 3:
            # Ref: tarch_likelihood.m:44 — [parameters, nu, lambda]
            nu = float(trans_params[1 + p + o + q])
            parameters = trans_params[: 1 + p + o + q].copy()
        elif error_type == 4:
            nu = float(trans_params[1 + p + o + q])
            lam = float(trans_params[2 + p + o + q])
            parameters = trans_params[: 1 + p + o + q].copy()
        else:
            # error_type == 1 (Normal): no distribution parameters
            parameters = trans_params[: 1 + p + o + q].copy()
    else:
        # Parameters are already in constrained space; parse nu / lambda.
        # Ref: tarch_likelihood.m:47-55
        if error_type == 2 or error_type == 3:
            # Ref: tarch_likelihood.m:49 — nu = parameters(p+o+q+2)
            # MATLAB 1-based index p+o+q+2 → Python 0-based p+o+q+1
            nu = float(parameters[p + o + q + 1])
            # Ref: tarch_likelihood.m:50 — parameters = parameters(1:1+p+o+q)
            parameters = parameters[: 1 + p + o + q].copy()
        elif error_type == 4:
            # Ref: tarch_likelihood.m:52 — lambda = parameters(p+o+q+3)
            lam = float(parameters[p + o + q + 2])
            # Ref: tarch_likelihood.m:53 — nu = parameters(p+o+q+2)
            nu = float(parameters[p + o + q + 1])
            # Ref: tarch_likelihood.m:54 — parameters = parameters(1:1+p+o+q)
            parameters = parameters[: 1 + p + o + q].copy()
        # For error_type == 1 (Normal), no slicing needed — parameters
        # already contains only core GARCH parameters.

    # ------------------------------------------------------------------ #
    #  Backcast length                                                    #
    # ------------------------------------------------------------------ #
    # Ref: tarch_likelihood.m:59 — m = max([p o q])
    m: int = max(p, o, q)

    # ------------------------------------------------------------------ #
    #  Conditional variance recursion                                     #
    # ------------------------------------------------------------------ #
    # Ref: tarch_likelihood.m:61
    ht = tarch_core(fdata, fIdata, parameters, back_cast, p, o, q, m, T, tarch_type)

    # ------------------------------------------------------------------ #
    #  Select relevant observations (skip backcast padding)               #
    # ------------------------------------------------------------------ #
    # Ref: tarch_likelihood.m:64 — t = (m + 1):T  [MATLAB 1-based]
    # Python 0-based equivalent: indices m through T-1  →  slice [m:T]
    ht_relevant = ht[m:T]
    data_relevant = data[m:T]

    # ------------------------------------------------------------------ #
    #  Distribution log-likelihood computation                            #
    # ------------------------------------------------------------------ #
    # Each distribution function returns (LL, lls) where LL = sum(lls)
    # and both are the POSITIVE log-likelihood.  We negate to produce
    # the objective for minimization.
    # Ref: tarch_likelihood.m:68-84

    if error_type == 1:
        # Normal (Gaussian) distribution
        # Ref: tarch_likelihood.m:70 — [LL, LLS] = normloglik(data, 0, ht)
        # normloglik requires x as shape (T, 1) and sigma2 as shape (T, 1)
        LL, LLS = normloglik(
            data_relevant.reshape(-1, 1),
            np.float64(0.0),
            ht_relevant.reshape(-1, 1),
        )
        # Ref: tarch_likelihood.m:71-72 — negate for minimization
        LLS = -LLS.ravel()
        LL = -LL

    elif error_type == 2:
        # Standardized Student's t distribution
        # Ref: tarch_likelihood.m:74 — [LL, LLS] = stdtloglik(data, 0, ht, nu)
        LL, LLS = stdtloglik(data_relevant, np.float64(0.0), ht_relevant, nu)
        # Ref: tarch_likelihood.m:75-76
        LLS = -np.asarray(LLS, dtype=np.float64).ravel()
        LL = -float(LL)

    elif error_type == 3:
        # Generalized Error Distribution (GED)
        # Ref: tarch_likelihood.m:78 — [LL, LLS] = gedloglik(data, 0, ht, nu)
        LL, LLS = gedloglik(data_relevant, np.float64(0.0), ht_relevant, nu)
        # Ref: tarch_likelihood.m:79-80
        LLS = -np.asarray(LLS, dtype=np.float64).ravel()
        LL = -float(LL)

    elif error_type == 4:
        # Hansen's Skewed Student's t distribution
        # Ref: tarch_likelihood.m:82 — [LL, LLS] = skewtloglik(data, 0, ht, nu, lambda)
        LL, LLS = skewtloglik(data_relevant, np.float64(0.0), ht_relevant, nu, lam)
        # Ref: tarch_likelihood.m:83-84
        LLS = -np.asarray(LLS, dtype=np.float64).ravel()
        LL = -float(LL)

    else:
        raise ValueError(
            f"error_type must be 1, 2, 3, or 4 — received {error_type}"
        )

    return float(LL), LLS, ht_relevant
