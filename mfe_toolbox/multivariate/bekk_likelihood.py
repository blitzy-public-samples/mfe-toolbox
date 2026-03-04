"""
BEKK(p,o,q) Multivariate Volatility Model — Negative Log-Likelihood

Computes the negative log-likelihood for the BEKK(p,o,q) multivariate GARCH
model, suitable as an objective function for scipy.optimize.minimize.

The BEKK model specifies conditional covariance dynamics as::

    H(t) = C
           + sum_{j=1}^{p} A_j' * data(:,:,t-j) * A_j
           + sum_{j=1}^{o} G_j' * data_asym(:,:,t-j) * G_j
           + sum_{j=1}^{q} B_j' * H(:,:,t-j) * B_j

where C = L @ L.T is the intercept matrix (guaranteed PSD via Cholesky
parameterisation), A contains p symmetric innovation matrices, G contains
o asymmetric innovation matrices, and B contains q smoothing matrices.

Migrated from: multivariate/bekk_likelihood.m (72 lines)
Copyright: Kevin Sheppard, kevin.sheppard@economics.ox.ac.uk

References
----------
Engle, R.F. and Kroner, K.F. (1995). Multivariate Simultaneous Generalized
ARCH. Econometric Theory, 11(1), 122-150.
"""

import numpy as np

from mfe_toolbox.multivariate.bekk_parameter_transform import bekk_parameter_transform


def bekk_likelihood(parameters, data, data_asym, p, o, q,
                    back_cast, back_cast_asym, type_model):
    """
    Compute the negative log-likelihood for a BEKK(p,o,q) model.

    Parameters
    ----------
    parameters : numpy.ndarray
        Flat 1-D parameter vector containing:

        * First k*(k+1)/2 elements: vech of lower-triangular Cholesky
          factor of the intercept matrix C.
        * Remaining elements: vectorised A, G, B matrices whose format
          depends on *type_model* (Scalar → 1 element each, Diagonal → k
          elements each, Full → k*k elements each).

    data : numpy.ndarray
        K × K × T array of outer products ``r_t @ r_t.T`` for each
        observation *t*.
    data_asym : numpy.ndarray
        K × K × T array of asymmetric outer products
        ``eta_t @ eta_t.T`` where ``eta_t = r_t * (r_t < 0)``.
        Used only when *o* > 0.
    p : int
        Number of symmetric innovation lags (ARCH order).  Must be >= 1.
    o : int
        Number of asymmetric innovation lags.  0 for symmetric BEKK.
    q : int
        Number of conditional covariance lags (GARCH order).
    back_cast : numpy.ndarray
        K × K positive semi-definite matrix used for pre-sample values
        of both ``data`` and ``H(t)`` when lag indices fall before the
        sample start.
    back_cast_asym : numpy.ndarray
        K × K positive semi-definite matrix used for pre-sample values
        of ``data_asym`` when lag indices fall before the sample start.
    type_model : int
        Model type:  1 = Scalar,  2 = Diagonal,  3 = Full BEKK.

    Returns
    -------
    ll : float
        Total negative log-likelihood (for minimisation).  Set to 1e7 if
        numerical issues (NaN, Inf, or non-PD covariance) are encountered.
    lls : numpy.ndarray
        T-element 1-D array of per-period negative log-likelihood
        contributions.
    Ht : numpy.ndarray
        K × K × T array of conditional covariance matrices.

    Notes
    -----
    Numerical stability is ensured by:

    * ``np.linalg.slogdet`` for the log-determinant (avoids overflow).
    * ``np.linalg.solve`` instead of explicit matrix inversion for the
      quadratic form ``trace(H_t^{-1} data_t)``.
    * A 1e7 penalty returned whenever H_t is not positive definite.

    The per-period contribution is

    .. math::

        \\ell_t = \\tfrac{1}{2}\\bigl(k \\ln 2\\pi
                  + \\ln|H_t|
                  + \\operatorname{tr}(H_t^{-1} \\, r_t r_t')\\bigr)

    Ref: bekk_likelihood.m — MFE Toolbox by Kevin Sheppard
    """
    # ------------------------------------------------------------------
    # 1.  Extract dimensions
    # Ref: bekk_likelihood.m:33 — [k,~,T]=size(data);
    # ------------------------------------------------------------------
    k = data.shape[0]
    T = data.shape[2]

    # ------------------------------------------------------------------
    # 2.  Unpack flat parameter vector → structured BEKK matrices
    # Ref: bekk_likelihood.m:37 —
    #      [C,A,G,B]=bekk_parameter_transform(parameters,p,o,q,k,type)
    # ------------------------------------------------------------------
    C, A, G, B = bekk_parameter_transform(parameters, p, o, q, k, type_model)

    # ------------------------------------------------------------------
    # 3.  Allocate output arrays and likelihood constant
    # Ref: bekk_likelihood.m:39-41
    # ------------------------------------------------------------------
    Ht = np.zeros((k, k, T))
    lls = np.zeros(T)
    log_lik_const = k * np.log(2.0 * np.pi)

    # Track whether the recursion encountered a non-PD matrix or other
    # numerical problem.  In MATLAB the equivalent check at the end is:
    #   if isnan(ll) || isinf(ll) || ~isreal(ll)  →  ll = 1e7
    # Ref: bekk_likelihood.m:70-72
    ll_invalid = False

    # ------------------------------------------------------------------
    # 4.  Main BEKK recursion  (MATLAB for i=1:T → Python for i in range(T))
    # Ref: bekk_likelihood.m:43-67
    # ------------------------------------------------------------------
    for i in range(T):
        # 4a.  Start with intercept  C = L @ L.T  (PSD by construction)
        # Ref: bekk_likelihood.m:44 — Ht(:,:,i)=C;
        Ht[:, :, i] = C.copy()

        # ---------------------------------------------------------------
        # 4b.  Symmetric ARCH terms   A_j' * OP_{t-j} * A_j
        # Ref: bekk_likelihood.m:45-51
        # MATLAB j=1..p  →  Python j=0..p-1 (lag number is j+1)
        # Backcast condition: MATLAB (i-j)<=0  →  Python i-j-1 < 0
        # ---------------------------------------------------------------
        for j in range(p):
            if i - j - 1 < 0:
                # Ref: bekk_likelihood.m:47 — pre-sample
                Ht[:, :, i] += A[:, :, j].T @ back_cast @ A[:, :, j]
            else:
                # Ref: bekk_likelihood.m:49 — in-sample
                Ht[:, :, i] += (
                    A[:, :, j].T @ data[:, :, i - j - 1] @ A[:, :, j]
                )

        # ---------------------------------------------------------------
        # 4c.  Asymmetric terms   G_j' * OPA_{t-j} * G_j   (only if o>0)
        # Ref: bekk_likelihood.m:52-58
        # ---------------------------------------------------------------
        for j in range(o):
            if i - j - 1 < 0:
                # Ref: bekk_likelihood.m:54 — pre-sample asymmetric
                Ht[:, :, i] += (
                    G[:, :, j].T @ back_cast_asym @ G[:, :, j]
                )
            else:
                # Ref: bekk_likelihood.m:56 — in-sample asymmetric
                Ht[:, :, i] += (
                    G[:, :, j].T @ data_asym[:, :, i - j - 1] @ G[:, :, j]
                )

        # ---------------------------------------------------------------
        # 4d.  GARCH terms   B_j' * H_{t-j} * B_j
        # Ref: bekk_likelihood.m:59-65
        # ---------------------------------------------------------------
        for j in range(q):
            if i - j - 1 < 0:
                # Ref: bekk_likelihood.m:61 — pre-sample (use backcast)
                Ht[:, :, i] += B[:, :, j].T @ back_cast @ B[:, :, j]
            else:
                # Ref: bekk_likelihood.m:63 — in-sample (use lagged H)
                Ht[:, :, i] += (
                    B[:, :, j].T @ Ht[:, :, i - j - 1] @ B[:, :, j]
                )

        # ---------------------------------------------------------------
        # 4e.  Per-period negative log-likelihood
        # Ref: bekk_likelihood.m:66
        #   MATLAB: lls(i) = 0.5*(logLikConst
        #               + log(det(Ht(:,:,i)))
        #               + sum(diag( Ht(:,:,i)\data(:,:,i) )))
        # ---------------------------------------------------------------

        # Fast early-exit: diagonal elements of Ht must be positive
        # (necessary condition for PD). np.diag extracts them cheaply.
        if np.any(np.diag(Ht[:, :, i]) <= 0):
            ll_invalid = True
            lls[i] = 0.5 * (log_lik_const + 1e6)
            continue

        # Numerical stability: use slogdet instead of log(det(...))
        sign, logabsdet = np.linalg.slogdet(Ht[:, :, i])

        if sign <= 0:
            # Ht is not positive definite.  In MATLAB, log(det(<0)) gives
            # a complex value caught by ~isreal(ll) at the end.
            ll_invalid = True
            lls[i] = 0.5 * (log_lik_const + 1e6)
            continue

        # Compute  trace( H_t^{-1} data_t )  via solve.
        # MATLAB: sum(diag( Ht(:,:,i) \ data(:,:,i) ))
        # np.linalg.solve(A, B) returns  A^{-1} B  without forming inv(A).
        try:
            Ht_inv_data = np.linalg.solve(Ht[:, :, i], data[:, :, i])
        except np.linalg.LinAlgError:
            # Singular matrix — cannot solve the linear system
            ll_invalid = True
            lls[i] = 0.5 * (log_lik_const + 1e6)
            continue

        # Ref: bekk_likelihood.m:66 — sum(diag(...)) = trace(...)
        # np.trace sums diagonal elements; equivalent to np.sum(np.diag(...)).
        trace_val = np.trace(Ht_inv_data)

        lls[i] = 0.5 * (log_lik_const + logabsdet + trace_val)

    # ------------------------------------------------------------------
    # 5.  Aggregate total negative log-likelihood
    # Ref: bekk_likelihood.m:68 — ll = sum(lls);
    # ------------------------------------------------------------------
    ll = float(np.sum(lls))

    # ------------------------------------------------------------------
    # 6.  Safety check — identical to MATLAB guard
    # Ref: bekk_likelihood.m:70-72
    #   if isnan(ll) || isinf(ll) || ~isreal(ll)  →  ll = 1e7
    # In Python, slogdet never produces complex results, so the
    # ~isreal branch is replaced by the ll_invalid flag set whenever
    # sign <= 0 (non-PD matrix).
    # ------------------------------------------------------------------
    if ll_invalid or np.isnan(ll) or np.isinf(ll):
        ll = 1e7

    return ll, lls, Ht
