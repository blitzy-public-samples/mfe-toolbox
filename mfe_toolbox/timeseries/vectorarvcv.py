"""
VAR (Vector Autoregression) parameter variance-covariance matrix estimation.

Computes the variance-covariance matrix for parameters estimated using vectorar
under one of four error structure assumptions:

1. Conditionally Homoskedastic and Uncorrelated
2. Conditionally Homoskedastic but Correlated
3. Heteroskedastic but Conditionally Uncorrelated
4. Heteroskedastic and Correlated

This module is a direct migration of ``timeseries/vectorarvcv.m`` from the
MFE Toolbox (Kevin Sheppard, University of Oxford).

Notes
-----
The four VCV estimation modes correspond to common assumptions in VAR
inference:

- **Homoskedastic + Uncorrelated** uses a classical SUR-type VCV with
  diagonal error covariance (only own-equation variance on the diagonal).
- **Homoskedastic + Correlated** uses a GLS-type VCV with full cross-equation
  error covariance via a Kronecker product structure.
- **Heteroskedastic + Uncorrelated** applies White-type robust sandwich
  estimation independently per equation (block-diagonal meat).
- **Heteroskedastic + Correlated** applies the full robust sandwich estimator
  across all equations simultaneously.

References
----------
Sheppard, K. (2009). MFE Toolbox Version 4.0.
Hamilton, J.D. (1994). Time Series Analysis. Princeton University Press.
Lütkepohl, H. (2005). New Introduction to Multiple Time Series Analysis.
    Springer.
"""

import numpy as np


def vectorarvcv(
    X: np.ndarray,
    errors: np.ndarray,
    het: int = 1,
    uncorr: int = 0,
) -> np.ndarray:
    """
    Estimate the variance-covariance matrix for VAR parameters.

    Computes the VCV matrix for parameters estimated using vectorar under one
    of four error structure assumptions determined by the ``het`` and ``uncorr``
    flags.

    Parameters
    ----------
    X : np.ndarray
        A ``(T, Np)`` matrix of regressors for each dependent variable, where
        ``T`` is the number of observations and ``Np`` is the number of
        parameters per equation (number of lags times K plus constant if
        included).
    errors : np.ndarray
        A ``(T, K)`` matrix of residuals, where ``K`` is the number of
        equations in the VAR system. Each column corresponds to the residuals
        from one equation.
    het : int, optional
        Scalar integer indicating the type of covariance estimator:

        - ``0`` — Homoskedastic (assumes constant conditional variance)
        - ``1`` — Heteroskedastic (robust to conditional heteroskedasticity)
          **[default]**
    uncorr : int, optional
        Scalar integer indicating the assumed structure of the error covariance
        matrix:

        - ``0`` — Correlated errors (allows cross-equation error correlation)
          **[default]**
        - ``1`` — Uncorrelated errors (assumes diagonal error covariance)

    Returns
    -------
    np.ndarray
        A ``(K * Np, K * Np)`` variance-covariance matrix of the estimated
        VAR parameters. The parameter ordering follows the Kronecker structure:
        parameters for equation 1 first, then equation 2, etc.

    Raises
    ------
    ValueError
        If ``X`` is not a 2-dimensional array.
    ValueError
        If ``errors`` is not a 2-dimensional array.
    ValueError
        If the number of rows of ``X`` does not match the number of rows
        of ``errors``.
    ValueError
        If ``het`` is not 0 or 1.
    ValueError
        If ``uncorr`` is not 0 or 1.

    Notes
    -----
    This is a helper function for :func:`vectorar` and
    :func:`grangercause`.

    The four VCV computation modes are:

    **Case 1: Heteroskedastic + Correlated** (``het=1, uncorr=0``)
        Full White-type robust sandwich estimator:

        .. math::

            VCV = A^{-1} B A^{-1} / T

        where :math:`A^{-1} = I_K \\otimes (X'X/T)^{-1}` and
        :math:`B = S'S/T` with :math:`S` being the score matrix.

    **Case 2: Heteroskedastic + Uncorrelated** (``het=1, uncorr=1``)
        Block-diagonal robust sandwich estimator, where the meat matrix
        :math:`B` is block-diagonal with blocks for each equation.

    **Case 3: Homoskedastic + Correlated** (``het=0, uncorr=0``)
        Classical GLS-type VCV using the Kronecker product:

        .. math::

            VCV = (\\hat{\\Sigma} \\otimes (X'X/T)^{-1}) / T

    **Case 4: Homoskedastic + Uncorrelated** (``het=0, uncorr=1``)
        Diagonal covariance Kronecker product:

        .. math::

            VCV = (\\text{diag}(\\text{diag}(\\hat{\\Sigma})) \\otimes
            (X'X/T)^{-1}) / T

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> T, Np, K = 100, 3, 2
    >>> X = rng.standard_normal((T, Np))
    >>> errors = rng.standard_normal((T, K))
    >>> vcv = vectorarvcv(X, errors, het=1, uncorr=0)
    >>> vcv.shape
    (6, 6)

    See Also
    --------
    vectorar : VAR estimation driver that calls this function.
    grangercause : Granger causality test using VAR parameter inference.

    References
    ----------
    .. [1] Sheppard, K. (2009). MFE Toolbox Version 4.0, vectorarvcv.m.
    .. [2] White, H. (1980). "A Heteroskedasticity-Consistent Covariance
       Matrix Estimator and a Direct Test for Heteroskedasticity."
       Econometrica, 48(4), 817-838.
    """
    # ----------------------------------------------------------------
    # Input validation
    # Ref: vectorarvcv.m — MATLAB does not validate inputs explicitly;
    # Python version adds robust validation for production use.
    # ----------------------------------------------------------------
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError(
            "X must be a 2-dimensional numpy.ndarray with shape (T, Np). "
            f"Received ndim={getattr(X, 'ndim', 'N/A')}."
        )
    if not isinstance(errors, np.ndarray) or errors.ndim != 2:
        raise ValueError(
            "errors must be a 2-dimensional numpy.ndarray with shape (T, K). "
            f"Received ndim={getattr(errors, 'ndim', 'N/A')}."
        )
    if X.shape[0] != errors.shape[0]:
        raise ValueError(
            "The number of rows of X and errors must match (both equal T). "
            f"X has {X.shape[0]} rows, errors has {errors.shape[0]} rows."
        )
    if het not in (0, 1):
        raise ValueError(
            "het must be 0 (homoskedastic) or 1 (heteroskedastic). "
            f"Received het={het}."
        )
    if uncorr not in (0, 1):
        raise ValueError(
            "uncorr must be 0 (correlated) or 1 (uncorrelated). "
            f"Received uncorr={uncorr}."
        )

    # ----------------------------------------------------------------
    # Common computations shared across all four VCV modes
    # Ref: vectorarvcv.m:36-43
    # ----------------------------------------------------------------
    # Number of observations
    # Ref: vectorarvcv.m:36 — T = size(X,1);
    T: int = X.shape[0]

    # Sample error covariance matrix (K x K)
    # Ref: vectorarvcv.m:37 — s2=errors'*errors/T;
    s2: np.ndarray = errors.T @ errors / T

    # Number of parameters per equation
    # Ref: vectorarvcv.m:38 — Np = size(X,2);
    Np: int = X.shape[1]

    # Inverse of the scaled cross-product matrix (Np x Np)
    # Ref: vectorarvcv.m:39 — XpXi = ((X'*X)/T)^(-1);
    # Using np.linalg.inv for matrix inversion as specified in AAP.
    XpXi: np.ndarray = np.linalg.inv((X.T @ X) / T)

    # Number of equations in the VAR system
    # Ref: vectorarvcv.m:40 — K=size(errors,2);
    K: int = errors.shape[1]

    # ----------------------------------------------------------------
    # Heteroskedastic modes require the score matrix S
    # Only computed when het == 1 to avoid unnecessary computation.
    # Ref: vectorarvcv.m:41-43
    # ----------------------------------------------------------------
    if het == 1:
        # Tile regressors K times horizontally: T x (Np*K)
        # Ref: vectorarvcv.m:41 — X2=repmat(X,1,K);
        X2: np.ndarray = np.tile(X, (1, K))

        # Tile errors and reshape to align with X2 columns.
        # MATLAB: e2 = reshape(repmat(errors, Np, 1), T, K*Np)
        # repmat(errors, Np, 1) stacks errors Np times vertically → (T*Np, K)
        # reshape to (T, K*Np) in column-major (Fortran) order produces:
        #   columns 0..Np-1    = errors[:,0] repeated Np times
        #   columns Np..2Np-1  = errors[:,1] repeated Np times
        #   etc.
        # Ref: vectorarvcv.m:42 — e2=reshape(repmat(errors,Np,1),T,K*Np);
        e2: np.ndarray = np.reshape(
            np.tile(errors, (Np, 1)), (T, K * Np), order='F'
        )

        # Element-wise score matrix: S_{t,j} = X_{t,j mod Np} * e_{t,j//Np}
        # Ref: vectorarvcv.m:43 — s=X2.*e2;
        s: np.ndarray = X2 * e2

    # ----------------------------------------------------------------
    # VCV computation based on the four modes
    # ----------------------------------------------------------------
    vcv: np.ndarray

    if het == 1 and uncorr == 0:
        # ------------------------------------------------------------------
        # Case 1: Heteroskedastic + Correlated — Full White sandwich
        # Ref: vectorarvcv.m:45-49
        # ------------------------------------------------------------------
        # Bread matrix: Kronecker of I_K and (X'X/T)^{-1}
        # Ref: vectorarvcv.m:47 — Ainv = kron(eye(K),XpXi);
        Ainv: np.ndarray = np.kron(np.eye(K), XpXi)

        # Meat matrix: full score outer product
        # Ref: vectorarvcv.m:48 — B=(s'*s)/T;
        B: np.ndarray = (s.T @ s) / T

        # Sandwich: Ainv * B * Ainv / T
        # Ref: vectorarvcv.m:49 — VCV=Ainv*B*Ainv/T;
        vcv = Ainv @ B @ Ainv / T

    elif het == 1 and uncorr == 1:
        # ------------------------------------------------------------------
        # Case 2: Heteroskedastic + Uncorrelated — Block-diagonal sandwich
        # Ref: vectorarvcv.m:50-58
        # ------------------------------------------------------------------
        # Bread matrix (same structure as Case 1)
        # Ref: vectorarvcv.m:51 — Ainv = kron(eye(K),XpXi);
        Ainv = np.kron(np.eye(K), XpXi)

        # Block-diagonal meat: only own-equation blocks are nonzero
        # Ref: vectorarvcv.m:52 — B=zeros(Np*K);
        B = np.zeros((Np * K, Np * K))

        # Ref: vectorarvcv.m:53-57 — loop over equations
        for i in range(K):
            # Ref: vectorarvcv.m:54 — sel=(i-1)*Np+1:i*Np;
            # Python 0-indexed: sel = i*Np : (i+1)*Np
            sel: slice = slice(i * Np, (i + 1) * Np)

            # Extract score columns for equation i
            # Ref: vectorarvcv.m:55 — temp = s(:,sel);
            temp: np.ndarray = s[:, sel]

            # Fill diagonal block with own-equation outer product
            # Ref: vectorarvcv.m:56 — B(sel,sel)=temp'*temp/T;
            B[sel, sel] = temp.T @ temp / T

        # Sandwich: Ainv * B * Ainv / T
        # Ref: vectorarvcv.m:58 — VCV=Ainv*B*Ainv/T;
        vcv = Ainv @ B @ Ainv / T

    elif het == 0 and uncorr == 0:
        # ------------------------------------------------------------------
        # Case 3: Homoskedastic + Correlated — Kronecker VCV
        # This is the classical GLS-type covariance with full cross-equation
        # error correlation.
        # Ref: vectorarvcv.m:59-61
        # ------------------------------------------------------------------
        # VCV = kron(sigma_hat, (X'X/T)^{-1}) / T
        # Ref: vectorarvcv.m:61 — VCV=kron(s2,XpXi)/T;
        vcv = np.kron(s2, XpXi) / T

    elif het == 0 and uncorr == 1:
        # ------------------------------------------------------------------
        # Case 4: Homoskedastic + Uncorrelated — Diagonal Kronecker VCV
        # Uses only the diagonal of the error covariance matrix (own-equation
        # variances), discarding cross-equation correlations.
        # Ref: vectorarvcv.m:62-64
        # ------------------------------------------------------------------
        # diag(diag(s2)) extracts diagonal elements and forms a diagonal matrix
        # Ref: vectorarvcv.m:64 — VCV=kron(diag(diag(s2)),XpXi)/T;
        vcv = np.kron(np.diag(np.diag(s2)), XpXi) / T

    else:
        # This branch should never be reached due to input validation above,
        # but is included as a defensive safeguard.
        raise ValueError(
            f"Invalid combination of het={het} and uncorr={uncorr}. "
            "het must be 0 or 1, uncorr must be 0 or 1."
        )

    return vcv
