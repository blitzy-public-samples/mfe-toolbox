"""
Multivariate normal distribution log-likelihood computation.

Migrated from distributions/mvnormloglik.m (MFE Toolbox, Version 4.0).
Computes the log-likelihood for T observations of a K-dimensional multivariate
normal distribution with optional mean and covariance parameters.

Two computation paths are provided:
  1. Constant covariance (K x K sigma): vectorised via Cholesky decomposition.
  2. Time-varying covariance (K x K x T sigma): per-observation loop.

References
----------
[1] Cassella and Berger (1990) 'Statistical Inference'

See Also
--------
normloglik : Univariate normal log-likelihood.

Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 1    Date: 2/1/2007
"""

import numpy

__all__ = ['mvnormloglik']


def mvnormloglik(x, mu=None, sigma=None):
    """
    Compute the log-likelihood for the multivariate normal distribution.

    Parameters
    ----------
    x : array_like
        T x K matrix of observations, where T is the number of observations
        and K is the dimension of the multivariate normal.
    mu : array_like or None, optional
        Mean vector or matrix. Can be:

        - ``None`` (default) : zero mean assumed (no demeaning applied).
        - Shape ``(K,)`` or ``(K, 1)`` vector : broadcast to T x K via
          ``numpy.tile`` and subtracted from *x*.
        - Shape ``(T, K)`` array : subtracted from *x* directly.

        Ref: mvnormloglik.m:40-47, 50-57.
    sigma : array_like or None, optional
        Covariance matrix. Can be:

        - ``None`` (default) : K x K identity matrix used.
        - Shape ``(K, K)`` : constant covariance — must be symmetric and
          positive definite.
        - Shape ``(K, K, T)`` : time-varying covariance — each K x K slice
          must be symmetric and positive definite.

        Ref: mvnormloglik.m:58-78.

    Returns
    -------
    LL : float
        Total log-likelihood value (sum of individual log-likelihoods).
    lls : numpy.ndarray
        1-D array of shape ``(T,)`` containing per-observation
        log-likelihood values.

    Raises
    ------
    ValueError
        If *x* is not 2-D, if *mu* has incompatible shape, if *sigma* is
        not positive definite or symmetric, or if *sigma* has incompatible
        dimensions.

    Notes
    -----
    The multivariate normal log-likelihood for observation *t* is:

    .. math::

        \\ell_t = -\\tfrac{1}{2}\\bigl(K\\ln(2\\pi)
                  + \\ln|\\Sigma_t|
                  + \\mathbf{x}_t^\\top \\Sigma_t^{-1} \\mathbf{x}_t\\bigr)

    MATLAB-to-Python translation decisions:

    * ``det(sigma)`` replaced by ``numpy.linalg.slogdet`` for numerical
      stability (avoids overflow/underflow for large *K*).
      Ref: mvnormloglik.m:89.
    * ``sigma^(-0.5)`` replaced by Cholesky decomposition and forward
      substitution: ``L = cholesky(sigma); stdx = solve(L, x.T).T``.
      Ref: mvnormloglik.m:88.
    * ``x(t,:)*inv(s)*x(t,:)'`` replaced by
      ``x[t, :] @ numpy.linalg.solve(s, x[t, :])`` to avoid explicit
      matrix inversion.  Ref: mvnormloglik.m:98.
    * ``repmat(mu', T, 1)`` replaced by ``numpy.tile(mu.flatten(), (T, 1))``.
      Ref: mvnormloglik.m:45, 55.
    * ``eig(sigma)`` replaced by ``numpy.linalg.eigvalsh(sigma)`` for the
      positive-definiteness check (``eigvalsh`` exploits symmetry).
      Ref: mvnormloglik.m:61.
    * MATLAB 3-D sigma indexing ``sigma(:,:,t)`` maps to
      ``sigma[:, :, t]`` with 0-based *t*.
      Ref: mvnormloglik.m:97.

    References
    ----------
    .. [1] Cassella and Berger (1990) 'Statistical Inference'
    """
    # ------------------------------------------------------------------
    # Convert x to float64 — Ref: mvnormloglik.m:27 [T,K]=size(x)
    # ------------------------------------------------------------------
    x = numpy.asarray(x, dtype=numpy.float64)

    # Validate x is 2-D — Ref: mvnormloglik.m:32-34
    if x.ndim != 2:
        raise ValueError('X must be a T by K matrix')

    T, K = x.shape  # Ref: mvnormloglik.m:27

    # ------------------------------------------------------------------
    # Handle mu parameter — Ref: mvnormloglik.m:36-57
    # ------------------------------------------------------------------
    if mu is not None:
        mu = numpy.asarray(mu, dtype=numpy.float64)

        # Normalise scalar / column-vector shapes so downstream logic is
        # uniform.  A 0-D scalar becomes (1,); a (K,1) column becomes (K,).
        if mu.ndim == 0:
            mu = mu.reshape(1)
        if mu.ndim == 2 and mu.shape[1] == 1:
            # Ref: mvnormloglik.m:41 — all(size(mu)==[K 1])
            mu = mu.flatten()

        if mu.ndim == 1:
            # K-vector: broadcast to T x K
            # Ref: mvnormloglik.m:44-46 — mu=repmat(mu',T,1)
            if mu.shape[0] != K:
                raise ValueError(
                    'MU must be either a K by 1 vector or the same size as X'
                )
            x = x - numpy.tile(mu, (T, 1))
        elif mu.ndim == 2:
            # T x K matrix — subtract directly
            # Ref: mvnormloglik.m:47, 57 — x=x-mu
            if mu.shape[0] != T or mu.shape[1] != K:
                raise ValueError(
                    'MU must be either a K by 1 vector or the same size as X'
                )
            x = x - mu
        else:
            raise ValueError(
                'MU must be either a K by 1 vector or the same size as X'
            )

    # ------------------------------------------------------------------
    # Handle sigma parameter — Ref: mvnormloglik.m:36-81
    # ------------------------------------------------------------------
    sigma_is_2d = True  # default path

    if sigma is None:
        # Ref: mvnormloglik.m:37-38 — sigma = eye(K)
        sigma = numpy.eye(K, dtype=numpy.float64)
        sigma_is_2d = True
    else:
        sigma = numpy.asarray(sigma, dtype=numpy.float64)

        if sigma.ndim == 2:
            # ---- Constant covariance (K x K) ----
            # Ref: mvnormloglik.m:59-64
            if sigma.shape[0] != K or sigma.shape[1] != K:
                raise ValueError(
                    'SIGMA must be either K by K and positive definite, '
                    'or K by K by T and contain only positive definite '
                    'matrices.'
                )
            # Symmetry check — Ref: mvnormloglik.m:61 any(any(sigma~=sigma'))
            if not numpy.allclose(sigma, sigma.T):
                raise ValueError(
                    'SIGMA must be either K by K and positive definite, '
                    'or K by K by T and contain only positive definite '
                    'matrices.'
                )
            # Positive-definiteness check
            # Ref: mvnormloglik.m:61 — min(eig(sigma))<=0
            # Using eigvalsh (exploits symmetry, returns sorted eigenvalues)
            eig_vals = numpy.linalg.eigvalsh(sigma)
            if eig_vals.min() <= 0:
                raise ValueError(
                    'SIGMA must be either K by K and positive definite, '
                    'or K by K by T and contain only positive definite '
                    'matrices.'
                )
            sigma_is_2d = True

        elif sigma.ndim == 3:
            # ---- Time-varying covariance (K x K x T) ----
            # Ref: mvnormloglik.m:65-75
            if (sigma.shape[0] != K
                    or sigma.shape[1] != K
                    or sigma.shape[2] != T):
                raise ValueError(
                    'SIGMA must be either K by K and positive definite, '
                    'or K by K by T and contain only positive definite '
                    'matrices.'
                )
            # Validate each K x K slice is symmetric and PD
            # Ref: mvnormloglik.m:69-74
            for t_idx in range(T):
                s = sigma[:, :, t_idx]
                # Ref: mvnormloglik.m:71 — any(any(s~=s'))
                if not numpy.allclose(s, s.T):
                    raise ValueError(
                        'SIGMA must be either K by K and positive definite, '
                        'or K by K by T and contain only positive definite '
                        'matrices.'
                    )
                # Ref: mvnormloglik.m:71 — min(eig(s)<=0)
                # NOTE: MATLAB source has a bug here: min(eig(s)<=0) checks
                # if ALL eigenvalues are <=0 (boolean comparison inside min).
                # The intended check is min(eig(s))<=0.  Python implements
                # the correct intended behaviour.
                eig_vals = numpy.linalg.eigvalsh(s)
                if eig_vals.min() <= 0:
                    raise ValueError(
                        'SIGMA must be either K by K and positive definite, '
                        'or K by K by T and contain only positive definite '
                        'matrices.'
                    )
            sigma_is_2d = False

        else:
            # Ref: mvnormloglik.m:76-78
            raise ValueError(
                'SIGMA must be either K by K and positive definite, '
                'or K by K by T and contain only positive definite '
                'matrices.'
            )

    # ------------------------------------------------------------------
    # Compute log-likelihood — Ref: mvnormloglik.m:86-101
    # ------------------------------------------------------------------
    if sigma_is_2d:
        # ---- Constant covariance path ----
        # Ref: mvnormloglik.m:87-93

        # Cholesky decomposition for numerically stable whitening.
        # Replaces MATLAB sigma^(-0.5) matrix power.
        # L is lower triangular such that L @ L.T == sigma.
        # Ref: mvnormloglik.m:88 — stdx = x * sigma^(-0.5)
        L = numpy.linalg.cholesky(sigma)

        # Solve L @ Z = x.T  →  Z = L^{-1} @ x.T  →  stdx = Z.T
        # Each row of stdx satisfies:  stdx[t,:] @ stdx[t,:] ==
        #   x[t,:] @ sigma^{-1} @ x[t,:]  (the Mahalanobis distance).
        stdx = numpy.linalg.solve(L, x.T).T  # shape (T, K)

        # Log-determinant via slogdet for numerical stability.
        # Replaces MATLAB log(det(sigma)).
        # Ref: mvnormloglik.m:89 — T*log(det(sigma))
        sign, logdet = numpy.linalg.slogdet(sigma)

        # Total log-likelihood
        # Ref: mvnormloglik.m:89
        # LL = -0.5*(T*K*log(2*pi) + T*log(det(sigma)) + sum(sum(stdx.^2)))
        LL = -0.5 * (
            T * K * numpy.log(2.0 * numpy.pi)
            + T * logdet
            + numpy.sum(stdx ** 2)
        )

        # Per-observation log-likelihoods
        # Ref: mvnormloglik.m:92
        # lls = -0.5*(K*log(2*pi) + log(det(sigma)) + sum(stdx.^2, 2))
        lls = -0.5 * (
            K * numpy.log(2.0 * numpy.pi)
            + logdet
            + numpy.sum(stdx ** 2, axis=1)
        )

    else:
        # ---- Time-varying covariance path ----
        # Ref: mvnormloglik.m:94-101

        # Ref: mvnormloglik.m:95 — lls=zeros(T,1)
        # Python: 1-D array of length T (MATLAB column vector)
        lls = numpy.zeros(T, dtype=numpy.float64)

        for t in range(T):
            # Ref: mvnormloglik.m:97 — s=sigma(:,:,t)
            # Python 0-indexed; MATLAB was 1-indexed (t=1:T)
            s = sigma[:, :, t]

            # Log-determinant — Ref: mvnormloglik.m:98 log(det(s))
            # Using slogdet for numerical stability.
            sign, logdet = numpy.linalg.slogdet(s)

            # Quadratic form  x_t' @ Sigma_t^{-1} @ x_t
            # Ref: mvnormloglik.m:98 — x(t,:)*inv(s)*x(t,:)'
            # Using solve instead of explicit inv for numerical stability.
            quad_form = x[t, :] @ numpy.linalg.solve(s, x[t, :])

            # Ref: mvnormloglik.m:98
            lls[t] = -0.5 * (
                K * numpy.log(2.0 * numpy.pi) + logdet + quad_form
            )

        # Ref: mvnormloglik.m:100 — LL=sum(lls)
        LL = numpy.sum(lls)

    return float(LL), lls
