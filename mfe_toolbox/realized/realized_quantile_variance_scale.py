"""
Quantile Realized Variance scaling factor computation via Monte Carlo simulation.

Computes the scale factors, Global Minimum Variance Portfolio (GMVP) weights,
and non-scaled covariance matrix needed for the Quantile Realized Variance (QRV)
estimator. Uses Monte Carlo integration to estimate the expected values of
order statistics from standard normal samples.

Migrated from: realized/realized_quantile_variance_scale.m
Original author: Kevin Sheppard
Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 1    Date: 5/1/2008
"""

import numpy as np


def realized_quantile_variance_scale(
    samplesperbin: int,
    quantiles: np.ndarray,
    simulations: int = 10_000_000,
    symmetric: bool = False,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute quantile RV scale factors, GMVP weights, and covariance via Monte Carlo.

    Uses Monte Carlo integration to compute the expected values of squared order
    statistics from standard normal samples, the covariance of these order statistics,
    and the Global Minimum Variance Portfolio (GMVP) weights for optimal combination.

    Parameters
    ----------
    samplesperbin : int
        Number of returns to use in each bin when computing the quantiles. Must be
        a positive integer >= 2.
    quantiles : array_like
        k-element vector of quantile values. When ``symmetric=True``, must satisfy
        ``0 < quantiles <= 1``. When ``symmetric=False``, must satisfy
        ``0.5 < quantiles <= 1``. The product ``quantiles * samplesperbin`` must
        produce integer values.
    simulations : int, optional
        Number of Monte Carlo simulations. Should usually be at least 1,000,000.
        Default is 10,000,000.
    symmetric : bool, optional
        If True, uses the symmetric estimator where squared normal variates are
        sorted. If False (default), uses both upper and lower quantiles of the
        sorted normal variates.
    seed : int, optional
        Seed for the random number generator for reproducibility. Default is 0.
        The original MATLAB code uses ``randn('state', datenum('MAR-26-1974'))``
        which corresponds to datenum value 720741.

    Returns
    -------
    scales : numpy.ndarray
        Shape ``(k,)`` vector of scale factors for standardized quantile-based
        estimates of variance.
    weights : numpy.ndarray
        Shape ``(k,)`` vector of GMVP weights producing the optimal (lowest
        variance) combination.
    covar : numpy.ndarray
        Shape ``(k, k)`` non-scaled covariance matrix of the quantile realized
        variance. Forms the actual (scaled) covariance when multiplied by the
        integrated quarticity.

    Raises
    ------
    ValueError
        If ``samplesperbin`` is not a positive integer >= 2.
        If ``quantiles`` is empty or contains values outside valid range.
        If ``simulations`` is not a positive integer.
        If computed indices are out of bounds for the given ``samplesperbin``.

    Notes
    -----
    The function uses chunked processing (blocks of ``floor(2**13 / samplesperbin)``
    rows) to manage memory for large simulation counts.

    The GMVP weights are computed via the standard formula:

    .. math::

        w = \\Sigma^{-1} \\mathbf{1} / (\\mathbf{1}^T \\Sigma^{-1} \\mathbf{1})

    where :math:`\\Sigma` is the scaled covariance matrix.

    See Also
    --------
    realized_quantile_variance : QRV estimator that uses these scale factors.
    realized_quantile_weight_simulation : Two-phase MC simulation for precomputing weights.

    Examples
    --------
    >>> import numpy as np
    >>> q = np.array([0.85, 0.90, 0.95])
    >>> scales, weights, covar = realized_quantile_variance_scale(20, q, simulations=100000)
    >>> print(scales.shape, weights.shape, covar.shape)
    (3,) (3,) (3, 3)
    """
    # -------------------------------------------------------------------------
    # Input validation
    # -------------------------------------------------------------------------
    quantiles = np.asarray(quantiles, dtype=np.float64).ravel()

    if not isinstance(samplesperbin, (int, np.integer)) or samplesperbin < 2:
        raise ValueError(
            "SAMPLESPERBIN must be a positive integer >= 2. "
            f"Received: {samplesperbin}"
        )

    if quantiles.size == 0:
        raise ValueError("QUANTILES must be a non-empty array.")

    if symmetric:
        if np.any(quantiles <= 0.0) or np.any(quantiles > 1.0):
            raise ValueError(
                "When symmetric=True, QUANTILES must satisfy 0 < quantiles <= 1."
            )
    else:
        if np.any(quantiles <= 0.5) or np.any(quantiles > 1.0):
            raise ValueError(
                "When symmetric=False, QUANTILES must satisfy 0.5 < quantiles <= 1."
            )

    if not isinstance(simulations, (int, np.integer)) or simulations < 1:
        raise ValueError(
            "SIMULATIONS must be a positive integer. "
            f"Received: {simulations}"
        )

    # -------------------------------------------------------------------------
    # RNG initialization for reproducibility
    # Ref: realized_quantile_variance_scale.m:51-52 — MATLAB saves/restores
    # randn('state') and seeds with datenum('MAR-26-1974'). In Python we use
    # numpy.random.default_rng(seed) for deterministic results.
    # -------------------------------------------------------------------------
    rng = np.random.default_rng(seed)

    k = len(quantiles)

    # -------------------------------------------------------------------------
    # Compute quantile indices
    # Ref: realized_quantile_variance_scale.m:56-61 — MATLAB 1-indexed → Python 0-indexed
    # -------------------------------------------------------------------------
    if symmetric:
        # Ref: realized_quantile_variance_scale.m:57
        # MATLAB: indices = round(samplesperbin*quantiles)  (1-based)
        # Python: subtract 1 for 0-based indexing
        indices = np.round(samplesperbin * quantiles).astype(np.intp) - 1
        # Validate indices are within bounds
        if np.any(indices < 0) or np.any(indices >= samplesperbin):
            raise ValueError(
                "Computed symmetric indices are out of bounds. Check that "
                "quantiles * samplesperbin produces valid integer indices."
            )
    else:
        # Ref: realized_quantile_variance_scale.m:59-60
        # MATLAB: indicesHigh = round(samplesperbin*quantiles)         (1-based)
        # MATLAB: indicesLow  = round(samplesperbin*(1-quantiles)+1)   (1-based)
        # Python: subtract 1 for 0-based indexing
        indices_high = np.round(samplesperbin * quantiles).astype(np.intp) - 1
        indices_low = np.round(
            samplesperbin * (1.0 - quantiles) + 1.0
        ).astype(np.intp) - 1
        # Validate indices are within bounds
        if np.any(indices_high < 0) or np.any(indices_high >= samplesperbin):
            raise ValueError(
                "Computed high indices are out of bounds. Check that "
                "quantiles * samplesperbin produces valid integer indices."
            )
        if np.any(indices_low < 0) or np.any(indices_low >= samplesperbin):
            raise ValueError(
                "Computed low indices are out of bounds. Check that "
                "(1 - quantiles) * samplesperbin + 1 produces valid integer indices."
            )

    # -------------------------------------------------------------------------
    # Chunked processing setup
    # Ref: realized_quantile_variance_scale.m:67-68
    # -------------------------------------------------------------------------
    # Ref: realized_quantile_variance_scale.m:67 — rows = min(floor(2^13/samplesperbin), simulations)
    rows = min(int(np.floor(2**13 / samplesperbin)), simulations)
    if rows < 1:
        rows = 1  # Ensure at least one row per chunk

    # Ref: realized_quantile_variance_scale.m:68 — iter = ceil(simulations/rows)
    iter_count = int(np.ceil(simulations / rows))
    remaining = simulations

    # Ref: realized_quantile_variance_scale.m:71 — scales = zeros(1, k)  → 1D vector
    scales = np.zeros(k, dtype=np.float64)
    # Ref: realized_quantile_variance_scale.m:72 — opMatrix = zeros(k)   → k×k matrix
    # Note: MATLAB zeros(k) creates k×k matrix, NOT a length-k vector
    op_matrix = np.zeros((k, k), dtype=np.float64)

    # -------------------------------------------------------------------------
    # Monte Carlo integration loop
    # Ref: realized_quantile_variance_scale.m:74-88
    # -------------------------------------------------------------------------
    for _i in range(iter_count):
        # Ref: realized_quantile_variance_scale.m:75-76
        num_this_iter = min(remaining, rows)
        remaining = remaining - rows

        # Ref: realized_quantile_variance_scale.m:78
        # MATLAB: x = randn(numThisIter, samplesperbin)
        x = rng.standard_normal((num_this_iter, samplesperbin))

        if symmetric:
            # Ref: realized_quantile_variance_scale.m:80-81
            # MATLAB: x = sort(x.^2, 2)   → sort squared values along dim 2 (columns)
            # Python: np.sort(x**2, axis=1) → sort along axis 1 (columns)
            x = np.sort(x**2, axis=1)
            x2 = x[:, indices]
        else:
            # Ref: realized_quantile_variance_scale.m:83-84
            # MATLAB: x = sort(x, 2)   → sort along dim 2
            # Python: np.sort(x, axis=1) → sort along axis 1
            x = np.sort(x, axis=1)
            x2 = x[:, indices_high] ** 2 + x[:, indices_low] ** 2

        # Ref: realized_quantile_variance_scale.m:86
        # MATLAB: scales = scales + sum(x2)/simulations
        # Note: MATLAB sum(x2) sums along columns (dim 1) → np.sum(x2, axis=0)
        scales += np.sum(x2, axis=0) / simulations

        # Ref: realized_quantile_variance_scale.m:87
        # MATLAB: opMatrix = opMatrix + x2'*x2/simulations
        # x2 is (num_this_iter, k), x2' is (k, num_this_iter)
        # x2' * x2 → (k, k) outer product sum
        op_matrix += (x2.T @ x2) / simulations

    # -------------------------------------------------------------------------
    # Compute the covariance matrix
    # Ref: realized_quantile_variance_scale.m:91
    # MATLAB: covar = opMatrix - scales'*scales
    # scales is (1, k) row vector in MATLAB; scales' is (k, 1)
    # scales' * scales = outer product → (k, k)
    # In Python: np.outer(scales, scales) produces same (k, k) result
    # -------------------------------------------------------------------------
    covar = op_matrix - np.outer(scales, scales)

    # Ref: realized_quantile_variance_scale.m:92
    # MATLAB: covar = samplesperbin * diag(1./scales) * covar * diag(1./scales)
    inv_scale_diag = np.diag(1.0 / scales)
    covar = samplesperbin * (inv_scale_diag @ covar @ inv_scale_diag)

    # -------------------------------------------------------------------------
    # Compute GMVP weights
    # Ref: realized_quantile_variance_scale.m:94
    # MATLAB: weights = covar^(-1) * ones(k,1) / (ones(1,k) * covar^(-1) * ones(k,1))
    # This is the standard GMVP formula: w = Σ^{-1} 1 / (1' Σ^{-1} 1)
    # -------------------------------------------------------------------------
    covar_inv = np.linalg.inv(covar)
    ones_vec = np.ones((k, 1), dtype=np.float64)

    # Numerator: covar_inv @ ones_vec → shape (k, 1)
    numerator = covar_inv @ ones_vec
    # Denominator: ones_vec.T @ covar_inv @ ones_vec → scalar (1, 1) matrix
    denominator = ones_vec.T @ covar_inv @ ones_vec

    weights = numerator / denominator
    # Flatten to 1D array
    weights = weights.ravel()

    return scales, weights, covar
