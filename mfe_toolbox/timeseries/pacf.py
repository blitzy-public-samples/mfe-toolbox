"""
Sample Partial Autocorrelation Function via Levinson-Durbin / Yule-Walker.

Computes sample partial autocorrelations from observed time series data
using the Levinson-Durbin recursion on biased (MLE) Yule-Walker
autocovariance estimates with denominator ``T``.

The core algorithm implements the partitioned matrix inverse (Schur
complement) approach from the original MATLAB MFE Toolbox
``timeseries/pacf.m`` by Kevin Sheppard, adapted to operate on sample
autocovariances rather than theoretical ARMA autocovariances.

The biased autocovariance estimator uses denominator ``T`` (not ``T-k``),
matching MATLAB's default convention and the ``statsmodels``
``method='ywm'`` convention.

See Also
--------
mfe_toolbox.timeseries.spacf : Sample PACF via OLS regression.
mfe_toolbox.timeseries.sacf : Sample autocorrelation function.

References
----------
.. [1] Kevin Sheppard, "MFE Toolbox", ``timeseries/pacf.m``, Revision 3,
       2007.

Notes
-----
Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 4    Date: 2024
"""

import numpy as np
from scipy.linalg import toeplitz


def pacf(y: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute sample partial autocorrelations via Levinson-Durbin recursion.

    Estimates partial autocorrelation coefficients from observed time series
    data using the Yule-Walker method with biased (MLE) autocovariance
    estimates (denominator ``T``).  The Levinson-Durbin recursion is
    implemented via partitioned matrix inverse (Schur complement), preserving
    the numerical approach of the original MATLAB ``pacf.m``.

    Parameters
    ----------
    y : array_like
        1-D array of time series observations with ``T`` elements.  Arrays
        with shape ``(T, 1)`` or ``(1, T)`` are automatically raveled to
        1-D.  The sample mean is subtracted internally before computation.
    lags : int
        Number of partial autocorrelations to compute.  Must be a positive
        integer strictly less than ``len(y) / 2``.

    Returns
    -------
    pacf_vals : numpy.ndarray
        Shape ``(lags + 1,)`` array of partial autocorrelations.
        ``pacf_vals[0]`` is always ``1.0`` (lag-0 identity).
        ``pacf_vals[k]`` is the partial autocorrelation at lag ``k`` for
        ``k = 1, ..., lags``.
    bounds : numpy.ndarray
        Shape ``(lags + 1,)`` array of asymptotic confidence bounds
        computed as ``1.96 / sqrt(T)``, where ``T = len(y)``.  Under the
        null hypothesis of white noise, approximately 95% of sample PACF
        values should fall within ``[-bounds[k], +bounds[k]]``.

    Raises
    ------
    ValueError
        If ``y`` is not a 1-D array (or cannot be raveled to 1-D).
    ValueError
        If ``y`` is empty or contains non-numeric values.
    ValueError
        If ``y`` contains NaN or Inf values.
    ValueError
        If ``y`` has zero variance (constant series after mean removal).
    ValueError
        If ``lags`` is not a positive integer.
    ValueError
        If ``lags >= len(y) / 2`` (safety threshold).
    ValueError
        If the autocorrelation matrix becomes singular during
        Levinson-Durbin recursion (degenerate data).

    Notes
    -----
    The algorithm proceeds as follows:

    1. Input ``y`` is mean-demeaned: ``y = y - mean(y)``.
    2. Biased (MLE) autocovariances are computed with denominator ``T``:
       ``gamma[k] = (1/T) * sum(y[t] * y[t-k])`` for ``t = k, ..., T-1``.
    3. Autocovariances are normalized to autocorrelations:
       ``rho[k] = gamma[k] / gamma[0]``.
    4. The Levinson-Durbin recursion via partitioned matrix inverse (Schur
       complement) is applied to extract partial autocorrelations from
       successive Yule-Walker systems.  (Ref: pacf.m lines 56-90)
    5. The output vector is prepended with ``1.0`` at index 0 and values
       below ``100 * eps`` (machine epsilon) are zeroed.

    The biased denominator ``T`` matches MATLAB's default autocovariance
    convention and the ``statsmodels`` ``method='ywm'`` Yule-Walker MLE
    approach, ensuring numerical parity with MATLAB outputs.

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.timeseries.pacf import pacf
    >>> rng = np.random.default_rng(42)
    >>> y = rng.standard_normal(500)
    >>> pacf_vals, bounds = pacf(y, 20)
    >>> pacf_vals[0]  # lag-0 identity
    1.0
    >>> pacf_vals.shape
    (21,)
    >>> bounds.shape
    (21,)
    """
    # ------------------------------------------------------------------
    # Input Validation
    # ------------------------------------------------------------------
    # Validate y: convert to float64 numpy array
    try:
        y = np.asarray(y, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "y must be a numeric array convertible to float64."
        ) from exc

    # Allow (T, 1) or (1, T) shaped arrays; reject truly 2-D+
    if y.ndim > 1:
        if min(y.shape) > 1:
            raise ValueError(
                "y must be a 1-D array or a column/row vector.  "
                f"Got shape {y.shape}."
            )
        y = y.ravel()

    if y.ndim == 0 or len(y) == 0:
        raise ValueError("y must be a non-empty 1-D array.")

    # Reject NaN/Inf values — these produce undefined autocovariances
    if not np.all(np.isfinite(y)):
        raise ValueError("y must not contain NaN or Inf values.")

    # Validate lags: must be a positive integer scalar
    if isinstance(lags, (float, np.floating)):
        # Allow float that is exactly integer (e.g. 5.0)
        if lags != int(lags) or np.isnan(lags) or np.isinf(lags):
            raise ValueError("lags must be a positive integer.")
        lags = int(lags)
    if not isinstance(lags, (int, np.integer)):
        raise ValueError("lags must be a positive integer.")
    lags = int(lags)
    if lags <= 0:
        raise ValueError("lags must be a positive integer.")

    # Safety threshold: lags must be strictly less than T/2
    # Ref: User-mandated validation matching MFE convention
    T = len(y)
    if lags >= T / 2:
        raise ValueError(
            f"lags must be less than len(y) / 2.  "
            f"Got lags={lags} with len(y)={T} (len(y)/2={T / 2})."
        )

    # ------------------------------------------------------------------
    # Biased (MLE) Autocovariance Computation
    # ------------------------------------------------------------------
    # Mean demeaning — subtract sample mean before autocovariance
    # computation, matching MATLAB convention.
    y = y - np.mean(y)

    # Biased autocovariance with denominator T (not T-k),
    # matching MATLAB convention and statsmodels method='ywm'.
    # gamma[k] = (1/T) * sum_{t=k}^{T-1} y[t] * y[t-k]
    gamma = np.empty(lags + 1, dtype=np.float64)
    for k in range(lags + 1):
        gamma[k] = (1.0 / T) * np.sum(y[k:] * y[: T - k])

    # Guard against zero-variance input (constant series after demeaning).
    # gamma[0] == 0 means y is constant; PACF is undefined.
    if gamma[0] == 0.0:
        raise ValueError(
            "y has zero variance after mean removal (constant series); "
            "PACF is undefined."
        )

    # Normalize to autocorrelations: rho[k] = gamma[k] / gamma[0]
    # ac has exactly `lags` elements (lags 1 through lags)
    ac = gamma[1:] / gamma[0]

    # ------------------------------------------------------------------
    # Levinson-Durbin Recursion via Partitioned Matrix Inverse
    # ------------------------------------------------------------------
    # This section is preserved from the MATLAB pacf.m (lines 56-90)
    # with only variable name changes (n -> lags).

    # Initialize PACF output array.  Ref: pacf.m:60
    pac = np.zeros(lags, dtype=np.float64)

    # First partial autocorrelation equals first autocorrelation.
    # Ref: pacf.m:63
    pac[0] = ac[0]

    # Partitioned inverse recursion for lags >= 2.  Ref: pacf.m:65-87
    if lags >= 2:
        # For lag 2, build and solve the 2x2 Toeplitz system.
        # Ref: pacf.m:67-71
        XpX = toeplitz([1.0, ac[0]])
        XpXinv = np.linalg.inv(XpX)
        Xpy = ac[0:2]
        temp = XpXinv @ Xpy
        pac[1] = temp[1]

        # For lags 3..N, use the partitioned matrix inverse (Schur
        # complement) to update XpXinv incrementally.
        # Ref: pacf.m:73-87
        for i in range(2, lags):
            Ainv = XpXinv
            # Reversed autocorrelations as column vector.
            # Ref: pacf.m:75
            B = ac[i - 1 :: -1].reshape(-1, 1)
            # Row vector (transpose).  Ref: pacf.m:76
            C = B.T
            # Scalar diagonal block.  Ref: pacf.m:77
            D = 1.0

            # Precompute matrix-vector products for Schur complement.
            AinvB = Ainv @ B
            CAinv = C @ Ainv

            # Schur complement scalar inverse.  Ref: pacf.m:78
            schur_scalar = D - (C @ AinvB).item()
            if abs(schur_scalar) < 100.0 * np.finfo(float).eps:
                raise ValueError(
                    "Degenerate autocorrelation matrix encountered at "
                    f"lag {i + 1}; Schur complement is singular."
                )
            schur_inv = 1.0 / schur_scalar

            # Updated inverse of the augmented Toeplitz system via
            # partitioned matrix inverse formula.
            # SDinv = Ainv + Ainv*B*(D - C*Ainv*B)^{-1}*C*Ainv
            # Ref: pacf.m:78
            SDinv = Ainv + schur_inv * (AinvB @ CAinv)

            # Build the full (i+1) x (i+1) inverse from partitioned
            # blocks.  Ref: pacf.m:79-80
            Dinv = 1.0 / D
            top_right = -SDinv @ B * Dinv
            bottom_left = -Dinv * (C @ SDinv)
            bottom_right = Dinv + Dinv * (C @ SDinv @ B) * Dinv

            # Assemble the new XpXinv.  Ref: pacf.m:79-80
            XpXinv = np.empty((i + 1, i + 1), dtype=np.float64)
            XpXinv[:i, :i] = SDinv
            XpXinv[:i, i:] = top_right
            XpXinv[i:, :i] = bottom_left
            XpXinv[i, i] = bottom_right.item()

            # Symmetrize to control numerical drift.  Ref: pacf.m:82
            XpXinv = (XpXinv + XpXinv.T) / 2.0

            # Solve the augmented Yule-Walker system.  Ref: pacf.m:84
            Xpy = ac[0 : i + 1]
            temp = XpXinv @ Xpy

            # Extract the i-th partial autocorrelation.  Ref: pacf.m:85
            pac[i] = temp[i]

    # ------------------------------------------------------------------
    # Output Construction
    # ------------------------------------------------------------------
    # Prepend 1.0 for lag-0 identity.  Ref: pacf.m:89
    pautocorr = np.concatenate([np.array([1.0]), pac])

    # Zero values below numerical threshold.  Ref: pacf.m:90
    pautocorr[np.abs(pautocorr) < 100.0 * np.finfo(float).eps] = 0.0

    # Asymptotic confidence bounds: +/- 1.96 / sqrt(T)
    # Under the null hypothesis of white noise (Bartlett approximation).
    bounds = np.ones(lags + 1, dtype=np.float64) * (1.96 / np.sqrt(T))

    return (pautocorr, bounds)
