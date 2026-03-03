"""
IGARCH(P,Q) log-likelihood computation.

Computes the negative log-likelihood for Integrated GARCH (IGARCH) models
supporting four error distributions: Normal, Student's t, GED, and
Hansen's Skewed t.  The IGARCH unit-root constraint is enforced internally
by ``igarch_core``, where the last GARCH coefficient is computed as
``1 - sum(alpha) - sum(beta_free)``.

When called during optimisation (``estim_flag=True``), the parameter
vector is first mapped from unconstrained to constrained space via
``igarch_itransform``.

Migrated from: univariate/igarch_likelihood.m — MFE Toolbox Version 4.0
Original author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
Revision: 1, Date: 7/12/2009

References
----------
Engle, R.F. and Bollerslev, T. (1986), "Modelling the Persistence of
Conditional Variances", Econometric Reviews, 5, 1-50.

See Also
--------
igarch : IGARCH estimation driver.
igarch_core : Numba JIT conditional variance recursion.
igarch_itransform : Inverse parameter transformation.
igarch_transform : Forward parameter transformation.
"""

import numpy as np

from mfe_toolbox.univariate.igarch_core import igarch_core
from mfe_toolbox.univariate.igarch_itransform import igarch_itransform
from mfe_toolbox.distributions.normloglik import normloglik
from mfe_toolbox.distributions.stdtloglik import stdtloglik
from mfe_toolbox.distributions.gedloglik import gedloglik
from mfe_toolbox.distributions.skewtloglik import skewtloglik


def igarch_likelihood(
    parameters: np.ndarray,
    epsilon: np.ndarray,
    p: int,
    q: int,
    igarch_type: int,
    error_type: int,
    back_cast: float,
    T: int,
    constant: int,
    estim_flag: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute the negative log-likelihood for an IGARCH(P,Q) model.

    This function evaluates the IGARCH log-likelihood for a given parameter
    vector and data series.  It computes the transformed residuals
    (``fepsilon``) internally from ``epsilon`` based on ``igarch_type``,
    augments them with back-cast initial values, runs the IGARCH variance
    recursion, and then evaluates the distribution-specific log-likelihood.

    Parameters
    ----------
    parameters : np.ndarray
        Parameter vector.  Layout depends on context:

        * During estimation (``estim_flag=True``): unconstrained parameters
          that will be transformed via ``igarch_itransform``.
        * Outside estimation: constrained parameters ordered as
          ``[omega, alpha_1…alpha_p, beta_1…beta_{q-1}, nu, lambda]``
          where ``nu`` and ``lambda`` are present only for the
          corresponding ``error_type``.

        The last GARCH coefficient (``beta_q``) is **not** stored; it is
        computed internally to satisfy the IGARCH unit-root constraint.
    epsilon : np.ndarray
        Column vector (or 1-D array) of mean-zero residuals of length *N*.
        Ref: igarch_likelihood.m:11 — ``EPSILON — A column of mean zero data``
    p : int
        Positive integer — number of ARCH (alpha) lags.
    q : int
        Non-negative integer — number of GARCH (beta) lags (0 for ARCH).
    igarch_type : int
        Variance process formulation:

        * 1 — absolute-value recursion (IAVARCH):
          ``fepsilon = |epsilon|``
        * 2 — squared recursion (standard IGARCH):
          ``fepsilon = epsilon ** 2``
    error_type : int
        Innovation distribution:

        * 1 — Normal (Gaussian)
        * 2 — Standardised Student's t
        * 3 — Generalised Error Distribution (GED)
        * 4 — Hansen's Skewed t
    back_cast : float
        Scalar back-cast value used to initialise both the pre-sample
        transformed residuals and the first ``m`` conditional variances.
    T : int
        Total length of the augmented series (``len(epsilon) + m`` where
        ``m = max(p, q)``).  Passed for API compatibility with the
        MATLAB calling convention; the function verifies or recomputes
        internally.
    constant : int
        * 1 — include an intercept (omega) in the variance equation.
        * 0 — no intercept.
    estim_flag : bool, optional
        If ``True``, ``parameters`` are in unconstrained space and will
        be transformed via ``igarch_itransform`` before evaluation.
        Default is ``False``.

    Returns
    -------
    LL : float
        Negative of the total log-likelihood (i.e. the objective to
        **minimise** during estimation).
    lls : np.ndarray
        1-D array of per-observation negative log-likelihoods, length
        equal to ``len(epsilon)``.
    ht : np.ndarray
        1-D array of conditional variances for the data observations
        (back-cast initialisation period excluded), length equal to
        ``len(epsilon)``.

    Raises
    ------
    ValueError
        If ``error_type`` is not in ``{1, 2, 3, 4}``.

    Notes
    -----
    The MATLAB function ``igarch_likelihood.m`` receives a pre-computed
    ``fepsilon`` (augmented with ``m`` back-cast values) from the driver
    ``igarch.m``.  In this Python migration the signature is simplified:
    ``fepsilon`` is computed from ``epsilon`` internally, and the back-cast
    augmentation is performed here using ``back_cast``.

    Ref: igarch_likelihood.m:1-88

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.univariate.igarch_likelihood import igarch_likelihood
    >>> rng = np.random.default_rng(42)
    >>> eps = rng.standard_normal(200)
    >>> params = np.array([0.01, 0.1])  # omega, alpha; beta implied by unit-root
    >>> LL, lls, ht = igarch_likelihood(params, eps, p=1, q=1,
    ...     igarch_type=2, error_type=1, back_cast=1.0,
    ...     T=201, constant=1, estim_flag=False)
    >>> isinstance(LL, float)
    True
    """
    # ------------------------------------------------------------------
    # Step 0: Ensure inputs are proper numpy arrays.
    # ------------------------------------------------------------------
    parameters = np.asarray(parameters, dtype=np.float64).flatten()
    epsilon_arr = np.asarray(epsilon, dtype=np.float64).flatten()

    # ------------------------------------------------------------------
    # Step 1: Parameter transformation / parsing.
    # Ref: igarch_likelihood.m:46-60
    # ------------------------------------------------------------------
    # Initialise distribution parameters as empty (unused for Normal).
    nu: float | np.ndarray = np.empty(0)
    lambda_param: float | np.ndarray = np.empty(0)

    if estim_flag:
        # Ref: igarch_likelihood.m:48 — transform from unconstrained space
        parameters, nu, lambda_param = igarch_itransform(
            parameters, p, q, error_type, constant
        )
    else:
        # Ref: igarch_likelihood.m:51-59 — parse nu / lambda from tail
        if error_type == 2 or error_type == 3:
            # Ref: igarch_likelihood.m:53 — nu = parameters(p+q+constant)
            # MATLAB 1-indexed → Python 0-indexed
            nu = float(parameters[p + q + constant - 1])
            # Ref: igarch_likelihood.m:54 — parameters = parameters(1:constant+p+q-1)
            parameters = parameters[: constant + p + q - 1]
        elif error_type == 4:
            # Ref: igarch_likelihood.m:56 — lambda = parameters(p+q+constant+1)
            lambda_param = float(parameters[p + q + constant])
            # Ref: igarch_likelihood.m:57 — nu = parameters(p+q+constant)
            nu = float(parameters[p + q + constant - 1])
            # Ref: igarch_likelihood.m:58 — parameters = parameters(1:constant+p+q-1)
            parameters = parameters[: constant + p + q - 1]
        # error_type == 1: no distribution parameters to extract

    # ------------------------------------------------------------------
    # Step 2: Compute transformed residuals (fepsilon) from epsilon.
    # In MATLAB, fepsilon is pre-computed by the driver igarch.m and
    # passed as a separate argument.  Here we compute it internally.
    # Ref: igarch.m:117-118 (type 1) and igarch.m:130 (type 2)
    # ------------------------------------------------------------------
    if igarch_type == 1:
        # Ref: igarch.m:118 — fepsilon = abs(epsilon)
        fepsilon = np.abs(epsilon_arr)
    else:
        # Ref: igarch.m:130 — fepsilon = epsilon.^2 (igarch_type == 2)
        fepsilon = epsilon_arr ** 2

    # ------------------------------------------------------------------
    # Step 3: Augment fepsilon with m initial values for pre-sample lags.
    # In MATLAB, igarch.m prepends m values (mean(abs(epsilon)) for
    # type 1, var(epsilon) for type 2).  Here we use back_cast as the
    # initialisation value — it is the driver's best estimate of the
    # unconditional variance (or its transform) and is numerically close.
    # Ref: igarch.m:118 (type 1), igarch.m:130 (type 2)
    # ------------------------------------------------------------------
    m = max(p, q)  # Ref: igarch_likelihood.m:63 — m = max([p q])

    fepsilon_aug = np.concatenate(
        [np.full(m, back_cast, dtype=np.float64), fepsilon]
    )
    T_total = len(fepsilon_aug)  # = len(epsilon_arr) + m

    # ------------------------------------------------------------------
    # Step 4: Compute conditional variances via the IGARCH recursion.
    # Ref: igarch_likelihood.m:65 — ht = igarch_core(fepsilon, …)
    # igarch_core enforces the unit-root constraint internally.
    # ------------------------------------------------------------------
    ht_full = igarch_core(
        fepsilon_aug,
        parameters,
        back_cast,
        p,
        q,
        m,
        T_total,
        igarch_type,
        constant,
    )

    # ------------------------------------------------------------------
    # Step 5: Select observations after the back-cast initialisation.
    # Ref: igarch_likelihood.m:67-68
    #   MATLAB: t = (m+1):T; ht = ht(t);   (1-indexed, inclusive)
    #   Python: ht = ht_full[m:]             (0-indexed)
    # Resulting ht has the same length as epsilon_arr.
    # ------------------------------------------------------------------
    ht = ht_full[m:]

    # ------------------------------------------------------------------
    # Step 6: Evaluate the distribution-specific log-likelihood.
    # Each branch calls the corresponding distribution's loglik function,
    # then negates both the total and per-observation values because the
    # IGARCH driver minimises the *negative* log-likelihood.
    # Ref: igarch_likelihood.m:70-87
    # ------------------------------------------------------------------
    if error_type == 1:
        # Ref: igarch_likelihood.m:72 — [LL, LLS] = normloglik(epsilon,0,ht)
        # normloglik requires x as a (T,1) column vector.
        epsilon_col = epsilon_arr.reshape(-1, 1)
        ht_col = ht.reshape(-1, 1)
        LL, lls_raw = normloglik(epsilon_col, 0, ht_col)
        # Ref: igarch_likelihood.m:73-74 — LLS = -LLS; LL = -LL;
        lls = -np.asarray(lls_raw, dtype=np.float64).flatten()
        LL = -float(LL)
    elif error_type == 2:
        # Ref: igarch_likelihood.m:76 — [LL, LLS] = stdtloglik(epsilon,0,ht,nu)
        LL, lls_raw = stdtloglik(epsilon_arr, 0, ht, nu)
        # Ref: igarch_likelihood.m:77-78
        lls = -np.asarray(lls_raw, dtype=np.float64).flatten()
        LL = -float(LL)
    elif error_type == 3:
        # Ref: igarch_likelihood.m:80 — [LL, LLS] = gedloglik(epsilon,0,ht,nu)
        LL, lls_raw = gedloglik(epsilon_arr, 0, ht, nu)
        # Ref: igarch_likelihood.m:81-82
        lls = -np.asarray(lls_raw, dtype=np.float64).flatten()
        LL = -float(LL)
    elif error_type == 4:
        # Ref: igarch_likelihood.m:84 — [LL, LLS] = skewtloglik(epsilon,0,ht,nu,lambda)
        LL, lls_raw = skewtloglik(epsilon_arr, 0, ht, nu, lambda_param)
        # Ref: igarch_likelihood.m:85-86
        lls = -np.asarray(lls_raw, dtype=np.float64).flatten()
        LL = -float(LL)
    else:
        raise ValueError(
            f"Invalid error_type: {error_type}. Must be 1, 2, 3, or 4."
        )

    return LL, lls, ht
