"""
Robust sandwich variance-covariance matrix estimator.

Computes the robust (White or Newey-West) sandwich VCV matrix using two-sided
numerical derivatives for score computation and Hessian estimation.  This is a
general-purpose inference utility consumed by multiple GARCH model drivers
across the ``mfe_toolbox`` package.

Migrated from: utility/robustvcv.m (82 lines)
Original author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
Revision: 1  Date: 9/1/2005

Migrated to Python as part of the MFE Toolbox MATLAB-to-Python 3.12 migration.
"""

import numpy as np

from mfe_toolbox.utility.hessian_2sided import hessian_2sided
from mfe_toolbox.utility.covnw import covnw


def robustvcv(
    fun: callable,
    theta: np.ndarray,
    nw: int,
    *args,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a robust variance-covariance matrix numerically, including
    Newey-West style score covariance using 2-sided derivatives.

    This function simplifies calculating sandwich covariance estimators for
    (Q)MLE estimation.  The "A" matrix is estimated via a two-sided
    numerical Hessian (divided by T), and the "B" matrix is estimated as
    either the sample covariance of the individual scores (White, when
    ``nw == 0``) or the Newey-West HAC covariance (when ``nw > 0``).

    Parameters
    ----------
    fun : callable
        Log-likelihood function with signature ``fun(theta, *args)``
        returning a 2-tuple ``(ll_scalar, ll_individual)`` where

        * ``ll_scalar`` (float) — sum of the log-likelihoods, and
        * ``ll_individual`` (array_like of length T) — per-observation
          log-likelihoods.
    theta : numpy.ndarray
        Parameter vector at the optimum, typically from a numerical
        optimizer (e.g. ``scipy.optimize.minimize``).
    nw : int
        Number of Newey-West lags for HAC score covariance.  Set to ``0``
        for White's heteroskedasticity-consistent estimator.
    *args : tuple
        Additional positional arguments forwarded to *fun*.

    Returns
    -------
    VCV : numpy.ndarray
        K × K robust (sandwich) variance-covariance matrix.
    A : numpy.ndarray
        K × K "A" matrix, equal to the numerical Hessian divided by T.
    B : numpy.ndarray
        K × K "B" matrix (score covariance — White or Newey-West).
    scores : numpy.ndarray
        T × K matrix of numerical scores (per-observation, per-parameter).
    hess : numpy.ndarray
        K × K estimated Hessian (equal to *A*; both are Hessian / T,
        matching the MATLAB convention where ``hess = A`` is assigned
        before return).
    gross_scores : numpy.ndarray
        Length-K array of numerical scores of the aggregate log-likelihood.

    Raises
    ------
    ValueError
        If *theta* is empty or *nw* is not a non-negative integer.
    numpy.linalg.LinAlgError
        If the A matrix is singular and cannot be inverted.

    Notes
    -----
    The robust (sandwich) VCV estimator is:

    .. math::

        \\text{VCV} = \\frac{1}{T}\\, A^{-1}\\, B\\, A^{-1}

    where

    * :math:`A = H / T`, with *H* the numerical Hessian of the total
      log-likelihood evaluated at ``theta``, and
    * :math:`B = \\operatorname{Cov}(\\text{scores})` for White's
      estimator (``nw == 0``), or
      :math:`B = \\operatorname{covnw}(\\text{scores}, \\text{nw})`
      for the Newey-West HAC estimator (``nw > 0``).

    Scores are computed via two-sided central finite differences with step
    sizes :math:`h_i = \\max(|\\theta_i| \\cdot \\varepsilon^{1/3},\\;
    10^{-8})`, where :math:`\\varepsilon` is machine epsilon.

    References
    ----------
    White, H. (1994). *Estimation, Inference and Specification Analysis*.
    Cambridge University Press.

    See Also
    --------
    mfe_toolbox.utility.hessian_2sided : Two-sided numerical Hessian.
    mfe_toolbox.utility.covnw : Newey-West HAC covariance estimator.

    Examples
    --------
    >>> import numpy as np
    >>> def neg_normal_ll(theta, data):
    ...     resid = data - theta[0]
    ...     lls = -0.5 * np.log(2.0 * np.pi) - 0.5 * resid ** 2
    ...     return float(np.sum(lls)), lls
    >>> rng = np.random.default_rng(42)
    >>> data = rng.standard_normal(200)
    >>> VCV, A, B, scores, hess, gs = robustvcv(
    ...     neg_normal_ll, np.array([0.0]), 0, data
    ... )
    """
    # ------------------------------------------------------------------
    # Input Argument Checking  (Ref: robustvcv.m:33-41)
    # ------------------------------------------------------------------
    # Ref: robustvcv.m:36-38 — Ensure theta is a 1-D array
    # In MATLAB: if size(theta,1)<size(theta,2)  theta=theta'; end
    theta = np.atleast_1d(np.asarray(theta, dtype=float)).ravel()

    # Ref: robustvcv.m:44 — k=length(theta)
    k: int = len(theta)
    if k == 0:
        raise ValueError("theta must be a non-empty parameter vector.")

    # Validate nw: must be a non-negative integer
    # Accept float values that are exact integers (e.g. 0.0) for MATLAB compat
    try:
        nw = int(nw)
    except (TypeError, ValueError) as exc:
        raise ValueError("nw must be a non-negative integer.") from exc
    if nw < 0:
        raise ValueError("nw must be a non-negative integer.")

    # ------------------------------------------------------------------
    # Step Size Computation  (Ref: robustvcv.m:45-46)
    # ------------------------------------------------------------------
    # Ref: robustvcv.m:45 — h=max(abs(theta*eps^(1/3)),1e-8)
    # eps^(1/3) is optimal for central finite differences
    eps_cbrt: float = np.finfo(float).eps ** (1.0 / 3.0)
    h: np.ndarray = np.maximum(np.abs(theta) * eps_cbrt, 1e-8)

    # Ref: robustvcv.m:46 — h=diag(h)  (perturbation matrix: k × k diagonal)
    h_diag: np.ndarray = np.diag(h)

    # ------------------------------------------------------------------
    # Base Likelihood Evaluation  (Ref: robustvcv.m:48-50)
    # ------------------------------------------------------------------
    # Ref: robustvcv.m:48 — [~,like]=feval(fun,theta,varargin{:})
    _, like = fun(theta, *args)
    like = np.asarray(like, dtype=float).ravel()
    # Ref: robustvcv.m:50 — t=length(like)
    t: int = len(like)

    # ------------------------------------------------------------------
    # Forward / Backward Perturbed Likelihood Evaluations
    # (Ref: robustvcv.m:52-61)
    # ------------------------------------------------------------------
    # Ref: robustvcv.m:52-55 — initialise storage
    LLFp: np.ndarray = np.zeros(k)
    LLFm: np.ndarray = np.zeros(k)
    likep: np.ndarray = np.zeros((t, k))
    likem: np.ndarray = np.zeros((t, k))

    # Ref: robustvcv.m:56-61 — Two-sided finite difference scores; 0-indexed
    for i in range(k):
        # Ref: robustvcv.m:57 — thetaph=theta+h(:,i)
        thetaph: np.ndarray = theta + h_diag[:, i]
        ll_val, ll_vec = fun(thetaph, *args)
        LLFp[i] = ll_val
        likep[:, i] = np.asarray(ll_vec, dtype=float).ravel()

        # Ref: robustvcv.m:59 — thetamh=theta-h(:,i)
        thetamh: np.ndarray = theta - h_diag[:, i]
        ll_val, ll_vec = fun(thetamh, *args)
        LLFm[i] = ll_val
        likem[:, i] = np.asarray(ll_vec, dtype=float).ravel()

    # ------------------------------------------------------------------
    # Numerical Scores Computation  (Ref: robustvcv.m:63-69)
    # ------------------------------------------------------------------
    scores: np.ndarray = np.zeros((t, k))
    gross_scores: np.ndarray = np.zeros(k)

    # Ref: robustvcv.m:65 — h=diag(h) extracts diagonal back to vector
    # In Python h is already a 1-D vector; h_diag was the matrix form.
    for i in range(k):
        # Ref: robustvcv.m:67 — scores(:,i)=(likep(:,i)-likem(:,i))./(2*h(i))
        scores[:, i] = (likep[:, i] - likem[:, i]) / (2.0 * h[i])
        # Ref: robustvcv.m:68 — gross_scores(i)=(LLFp(i)-LLFm(i))./(2*h(i))
        gross_scores[i] = (LLFp[i] - LLFm[i]) / (2.0 * h[i])

    # ------------------------------------------------------------------
    # Hessian and VCV Computation  (Ref: robustvcv.m:71-82)
    # ------------------------------------------------------------------
    # Ref: robustvcv.m:71 — hess=hessian_2sided(fun,theta,varargin{:})
    # CRITICAL: hessian_2sided expects f(x,*args)->scalar.  In MATLAB the
    # nargout=1 mechanism silently captures only the scalar output from fun.
    # In Python we must explicitly wrap fun to extract the scalar.
    def _scalar_fun(params: np.ndarray, *fn_args) -> float:
        """Wrapper that returns only the scalar log-likelihood."""
        ll_scalar, _ = fun(params, *fn_args)
        return float(ll_scalar)

    hess_raw: np.ndarray = hessian_2sided(_scalar_fun, theta, *args)

    # Ref: robustvcv.m:72 — A=hess/t
    A: np.ndarray = hess_raw / t
    # Ref: robustvcv.m:73 — hess=A  (overwrite hess with A; both returned)
    hess: np.ndarray = A

    # Ref: robustvcv.m:74 — Ainv=A^(-1)
    Ainv: np.ndarray = np.linalg.inv(A)

    # Ref: robustvcv.m:75-82 — White sandwich estimator or Newey-West variant
    if nw == 0:
        # Ref: robustvcv.m:77 — B=cov(scores)
        # MATLAB cov() uses 1/(T-1) normalization; np.cov(rowvar=False) matches
        B: np.ndarray = np.cov(scores, rowvar=False)
        # Ensure B is always 2-D (np.cov can return 0-D for single-variable)
        B = np.atleast_2d(B)
    else:
        # Ref: robustvcv.m:80 — B=covnw(scores,nw)
        B = covnw(scores, nw)

    # Ref: robustvcv.m:78,81 — VCV=(Ainv*B*Ainv)/t
    VCV: np.ndarray = (Ainv @ B @ Ainv) / t

    return VCV, A, B, scores, hess, gross_scores
