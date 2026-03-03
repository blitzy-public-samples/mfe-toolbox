"""
Log-likelihood for AGARCH(P,Q) and NAGARCH(P,Q) estimation.

Computes the NEGATED log-likelihood for use with ``scipy.optimize.minimize``
(which performs minimization). Returns the negated total log-likelihood,
the negated per-observation log-likelihoods, and the conditional variance
series.

Migrated from: ``univariate/agarch_likelihood.m`` (85 lines)
Original author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
Revision: 1    Date: 7/12/2009

The function supports four distributional assumptions for the innovation
process:

- **Normal** (``error_type=1``): Gaussian innovations via :func:`normloglik`
- **Student's t** (``error_type=2``): Standardized Student's t via :func:`stdtloglik`
- **GED** (``error_type=3``): Generalized Error Distribution via :func:`gedloglik`
- **Skewed t** (``error_type=4``): Hansen's skewed t via :func:`skewtloglik`

See Also
--------
agarch : AGARCH/NAGARCH driver (model estimation entry point).
agarch_core : Numba JIT-accelerated conditional variance recursion.
agarch_itransform : Inverse parameter transformation (unconstrained → constrained).
"""

import numpy as np

from mfe_toolbox.univariate.agarch_core import agarch_core
from mfe_toolbox.univariate.agarch_itransform import agarch_itransform
from mfe_toolbox.distributions.normloglik import normloglik
from mfe_toolbox.distributions.stdtloglik import stdtloglik
from mfe_toolbox.distributions.gedloglik import gedloglik
from mfe_toolbox.distributions.skewtloglik import skewtloglik


def agarch_likelihood(
    parameters: np.ndarray,
    epsilon_aug: np.ndarray,
    p: int,
    q: int,
    model_type: int,
    error_type: int,
    transform_bounds: np.ndarray,
    back_cast: float,
    T: int,
    estim_flag: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Compute the NEGATED log-likelihood for AGARCH(P,Q) / NAGARCH(P,Q).

    This function is the objective supplied to ``scipy.optimize.minimize``
    during AGARCH/NAGARCH estimation.  Because ``minimize`` performs
    *minimization*, the log-likelihood is negated so that maximizing the
    likelihood is equivalent to minimizing the return value.

    Parameters
    ----------
    parameters : np.ndarray
        1-D parameter vector.  Layout depends on ``error_type``:

        - Base (length ``2 + p + q``):
          ``[omega, alpha_1, ..., alpha_p, gamma, beta_1, ..., beta_q]``
        - ``error_type == 2`` or ``3``: base + ``[nu]`` (shape param)
        - ``error_type == 4``: base + ``[nu, lambda]`` (shape + asymmetry)

        When ``estim_flag`` is True the parameters are in unconstrained
        space and will be mapped to the constrained space via
        :func:`agarch_itransform`.
    epsilon_aug : np.ndarray
        1-D array of length ``T`` containing mean-zero return data,
        augmented with ``max(p, q)`` back-cast values prepended at the
        front.
    p : int
        Positive integer — number of symmetric innovation (ARCH) terms.
    q : int
        Non-negative integer — number of lagged variance (GARCH) terms.
    model_type : int
        Variance model selection:

        - ``1``: AGARCH  — shock term ``(r_t - gamma)^2``
        - ``2``: NAGARCH — shock term ``(r_t - gamma * sqrt(h_t))^2``
    error_type : int
        Innovation distribution assumption:

        - ``1``: Normal
        - ``2``: Standardized Student's t
        - ``3``: Generalized Error Distribution (GED)
        - ``4``: Hansen's skewed Student's t
    transform_bounds : np.ndarray
        2-element array with the 0.01 and 0.99 quantiles of the data,
        used by :func:`agarch_itransform` for the gamma parameter
        transformation when ``estim_flag`` is True.
    back_cast : float
        Initialisation value for the conditional variance recursion
        (typically the unconditional sample variance).
    T : int
        Total length of ``epsilon_aug`` (including back-cast values).
    estim_flag : bool, optional
        If True, ``parameters`` are in unconstrained optimiser space and
        will be transformed via :func:`agarch_itransform` before use.
        Default is False.

    Returns
    -------
    LL : float
        **Negated** total log-likelihood (scalar).  Minimising this value
        is equivalent to maximising the log-likelihood.
    LLS : np.ndarray
        1-D array of length ``T - max(p, q)`` containing the **negated**
        per-observation log-likelihoods.
    ht : np.ndarray
        1-D array of length ``T - max(p, q)`` containing the conditional
        variances for the effective (non-backcast) observations.

    Raises
    ------
    ValueError
        If ``error_type`` is not in ``{1, 2, 3, 4}``.

    Notes
    -----
    **Index convention (MATLAB → Python):**

    The MATLAB source uses 1-based indexing::

        t = (m+1):T;          % MATLAB — selects observations m+1 .. T
        ht = ht(t);

    In Python (0-based), the equivalent slice is ``m:T``::

        ht = ht[m:T]          # Python — selects indices m .. T-1

    Ref: agarch_likelihood.m:64 — ``t = (m + 1):T;``

    **Parameter extraction (MATLAB → Python):**

    MATLAB ``parameters(p+q+3)`` → Python ``parameters[p+q+2]`` (0-based).
    Ref: agarch_likelihood.m:50-55.

    Examples
    --------
    >>> import numpy as np
    >>> params = np.array([0.01, 0.05, 0.1, 0.85])
    >>> data = np.random.default_rng(42).standard_normal(500)
    >>> LL, LLS, ht = agarch_likelihood(
    ...     params, data, 1, 1, 1, 1,
    ...     np.array([-2.33, 2.33]), np.var(data), 500
    ... )
    """
    # ------------------------------------------------------------------
    # Ensure inputs are proper numpy arrays
    # ------------------------------------------------------------------
    parameters = np.asarray(parameters, dtype=np.float64).ravel()
    epsilon_aug = np.asarray(epsilon_aug, dtype=np.float64).ravel()

    # ------------------------------------------------------------------
    # Parameter transformation / extraction
    # Ref: agarch_likelihood.m:43-57
    # ------------------------------------------------------------------
    nu: float = 0.0       # initialised; only used for error_type >= 2
    lambda_: float = 0.0  # initialised; only used for error_type == 4

    if estim_flag:
        # Ref: agarch_likelihood.m:43-45 — transform from unconstrained space
        parameters, nu, lambda_ = agarch_itransform(
            parameters, p, q, model_type, error_type, transform_bounds
        )
    else:
        # Ref: agarch_likelihood.m:47-56 — parse distribution parameters
        if error_type == 2 or error_type == 3:
            # Ref: agarch_likelihood.m:50 — MATLAB: nu = parameters(p+q+3)
            # Python 0-based: parameters[p+q+2]
            nu = parameters[p + q + 2]
            # Ref: agarch_likelihood.m:51 — MATLAB: parameters = parameters(1:2+p+q)
            parameters = parameters[:2 + p + q]
        elif error_type == 4:
            # Ref: agarch_likelihood.m:53 — MATLAB: lambda = parameters(p+q+4)
            # Python 0-based: parameters[p+q+3]
            lambda_ = parameters[p + q + 3]
            # Ref: agarch_likelihood.m:54 — MATLAB: nu = parameters(p+q+3)
            # Python 0-based: parameters[p+q+2]
            nu = parameters[p + q + 2]
            # Ref: agarch_likelihood.m:55 — MATLAB: parameters = parameters(1:2+p+q)
            parameters = parameters[:2 + p + q]

    # ------------------------------------------------------------------
    # Backcast length
    # Ref: agarch_likelihood.m:59 — m = max([p q])
    # ------------------------------------------------------------------
    m: int = max(p, q)

    # ------------------------------------------------------------------
    # Conditional variance recursion
    # Ref: agarch_likelihood.m:61 — ht = agarch_core(...)
    # ------------------------------------------------------------------
    ht = agarch_core(epsilon_aug, parameters, back_cast, p, q, m, T, model_type)

    # ------------------------------------------------------------------
    # Extract effective (non-backcast) observations
    # Ref: agarch_likelihood.m:64-66
    # MATLAB: t = (m+1):T → Python 0-based: m:T
    # ------------------------------------------------------------------
    ht = ht[m:T]
    epsilon = epsilon_aug[m:T]

    # ------------------------------------------------------------------
    # Log-likelihood computation by distribution type
    # Ref: agarch_likelihood.m:68-85
    # All distribution functions return (LL, lls) where LL = sum(lls).
    # We negate both for scipy.optimize.minimize compatibility.
    # ------------------------------------------------------------------
    if error_type == 1:
        # Normal distribution
        # Ref: agarch_likelihood.m:70 — normloglik(epsilon, 0, ht)
        # normloglik requires x as column vector (T, 1); sigma2 can be 1-D
        epsilon_col = epsilon.reshape(-1, 1)
        LL, lls = normloglik(epsilon_col, 0, ht)
        # normloglik returns lls as (n, 1); flatten for consistency
        LLS = -lls.ravel()
        LL = -LL
    elif error_type == 2:
        # Standardized Student's t distribution
        # Ref: agarch_likelihood.m:74 — stdtloglik(epsilon, 0, ht, nu)
        LL, lls = stdtloglik(epsilon, 0, ht, nu)
        LLS = -lls
        LL = -LL
    elif error_type == 3:
        # Generalized Error Distribution (GED)
        # Ref: agarch_likelihood.m:78 — gedloglik(epsilon, 0, ht, nu)
        LL, lls = gedloglik(epsilon, 0, ht, nu)
        LLS = -lls
        LL = -LL
    elif error_type == 4:
        # Hansen's Skewed Student's t distribution
        # Ref: agarch_likelihood.m:82 — skewtloglik(epsilon, 0, ht, nu, lambda)
        LL, lls = skewtloglik(epsilon, 0, ht, nu, lambda_)
        LLS = -lls
        LL = -LL
    else:
        raise ValueError(
            f"error_type must be 1 (Normal), 2 (Student's t), 3 (GED), or "
            f"4 (Skewed t), got {error_type}"
        )

    return LL, LLS, ht
