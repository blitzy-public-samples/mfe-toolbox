"""
Composite normal log-likelihood (bivariate pairwise) for K-dimensional arrays.

Computes the negative composite normal log-likelihood using bivariate pairwise
evaluation over a K-dimensional covariance matrix. This is a helper function
for various multivariate GARCH models including DCC, BEKK, RARCH, and
scalar_vt_vech.

This module merges the functionality of the original MATLAB
``composite_likelihood.m`` fallback and the C MEX accelerated
``composite_likelihood.c`` kernel into a single Numba JIT-compiled Python
implementation, providing comparable performance to the C version without
requiring a C compilation toolchain.

Notes
-----
Migrated from:
  - ``distributions/composite_likelihood.m`` (MATLAB fallback, 51 lines)
  - ``mex_source/composite_likelihood.c`` (C MEX kernel, 133 lines)

The MATLAB version used 1-based indices. This Python version uses **0-based**
indices throughout. All callers must provide 0-based index pairs.

The hardcoded constant ``likConst = 3.67575413281869`` equals ``2 * log(2 * pi)``.

References
----------
Kevin Sheppard, MFE Toolbox Version 4.0 (2009).
    See also: DCC, SCALAR_VT_VECH, BEKK, RARCH

Examples
--------
>>> import numpy as np
>>> from mfe_toolbox.distributions.composite_likelihood import composite_likelihood
>>> S = np.array([[1.0, 0.5], [0.5, 1.0]])
>>> data = np.array([0.3, -0.2])
>>> indices = np.array([[0, 1]], dtype=np.int64)
>>> ll = composite_likelihood(S, data, indices)
"""

import math

import numba
import numpy


# ---------------------------------------------------------------------------
# Numba JIT core: square (K x K) data path
# Ref: composite_likelihood.m:25-37, composite_likelihood.c:60-74
# ---------------------------------------------------------------------------
@numba.jit(nopython=True, cache=True)
def _composite_likelihood_core_square(
    S: numpy.ndarray,
    data: numpy.ndarray,
    indices: numpy.ndarray,
    q: int,
) -> float:
    """Compute composite likelihood for square (K x K) data matrix.

    This inner-loop kernel replaces the C MEX ``composite_likelihood_core``
    function for the ``m == n`` (square data) branch.

    Parameters
    ----------
    S : numpy.ndarray, shape (K, K), dtype float64
        Covariance matrix.
    data : numpy.ndarray, shape (K, K), dtype float64
        Square data matrix, typically an outer product of returns.
    indices : numpy.ndarray, shape (Q, 2), dtype int64
        0-based index pairs for pairwise likelihood evaluation.
    q : int
        Number of index pairs (``indices.shape[0]``).

    Returns
    -------
    float
        Negative composite normal log-likelihood value.

    Notes
    -----
    Ref: composite_likelihood.m:26-37 — MATLAB uses 1-based indexing;
    this function uses 0-based indexing.

    Ref: composite_likelihood.c:60-74 — C MEX column-major access pattern
    ``S[i*m+i]`` is replaced by NumPy 2D indexing ``S[i, i]``.
    """
    # Hardcoded constant: 2 * log(2 * pi) — Ref: composite_likelihood.m:23
    likConst = 3.67575413281869
    ll = 0.0
    scale = float(q)

    for k in range(q):
        # Ref: composite_likelihood.m:27-28 — MATLAB 1-based indices(k,1)
        # Python 0-based indices[k, 0]
        i = indices[k, 0]
        j = indices[k, 1]

        # Extract bivariate covariance sub-block from S
        # Ref: composite_likelihood.m:29-31
        s11 = S[i, i]
        s12 = S[i, j]
        s22 = S[j, j]

        # 2x2 determinant of the bivariate covariance sub-matrix
        # Ref: composite_likelihood.m:32
        det_val = s11 * s22 - s12 * s12

        # Extract bivariate data sub-block from square data matrix
        # Ref: composite_likelihood.m:33-35
        x11 = data[i, i]
        x12 = data[i, j]
        x22 = data[j, j]

        # Accumulate the bivariate normal negative log-likelihood
        # Ref: composite_likelihood.m:36 — formula:
        # 0.5 * (likConst + log(det) + (s22*x11 - 2*s12*x12 + s11*x22)/det) / q
        # Using math.log for scalar log in Numba nopython mode
        ll += 0.5 * (
            likConst
            + math.log(det_val)
            + (s22 * x11 - 2.0 * s12 * x12 + s11 * x22) / det_val
        ) / scale

    return ll


# ---------------------------------------------------------------------------
# Numba JIT core: vector (K,) data path
# Ref: composite_likelihood.m:38-50, composite_likelihood.c:76-91
# ---------------------------------------------------------------------------
@numba.jit(nopython=True, cache=True)
def _composite_likelihood_core_vector(
    S: numpy.ndarray,
    data: numpy.ndarray,
    indices: numpy.ndarray,
    q: int,
) -> float:
    """Compute composite likelihood for vector (K,) data.

    This inner-loop kernel replaces the C MEX ``composite_likelihood_core``
    function for the ``m != n`` (vector data) branch.  When data is a 1-D
    vector of returns, the outer-product entries are computed on the fly as
    ``data[i] * data[j]``.

    Parameters
    ----------
    S : numpy.ndarray, shape (K, K), dtype float64
        Covariance matrix.
    data : numpy.ndarray, shape (K,), dtype float64
        1-D data vector (e.g., returns).
    indices : numpy.ndarray, shape (Q, 2), dtype int64
        0-based index pairs for pairwise likelihood evaluation.
    q : int
        Number of index pairs (``indices.shape[0]``).

    Returns
    -------
    float
        Negative composite normal log-likelihood value.

    Notes
    -----
    Ref: composite_likelihood.m:39-50 — MATLAB uses 1-based indexing;
    this function uses 0-based indexing.

    Ref: composite_likelihood.c:76-91 — C MEX vector path with
    ``X[i]*X[i]``, ``X[i]*X[j]``, ``X[j]*X[j]`` product computation.
    """
    # Hardcoded constant: 2 * log(2 * pi) — Ref: composite_likelihood.m:23
    likConst = 3.67575413281869
    ll = 0.0
    scale = float(q)

    for k in range(q):
        # Ref: composite_likelihood.m:40-41 — MATLAB 1-based to Python 0-based
        i = indices[k, 0]
        j = indices[k, 1]

        # Extract bivariate covariance sub-block from S
        # Ref: composite_likelihood.m:42-44
        s11 = S[i, i]
        s12 = S[i, j]
        s22 = S[j, j]

        # 2x2 determinant of the bivariate covariance sub-matrix
        # Ref: composite_likelihood.m:45
        det_val = s11 * s22 - s12 * s12

        # Compute bivariate outer-product entries on the fly from vector data
        # Ref: composite_likelihood.m:46-48
        x11 = data[i] * data[i]
        x12 = data[i] * data[j]
        x22 = data[j] * data[j]

        # Accumulate the bivariate normal negative log-likelihood
        # Ref: composite_likelihood.m:49 — same formula as square branch
        # Using math.log for scalar log in Numba nopython mode
        ll += 0.5 * (
            likConst
            + math.log(det_val)
            + (s22 * x11 - 2.0 * s12 * x12 + s11 * x22) / det_val
        ) / scale

    return ll


# ---------------------------------------------------------------------------
# Public wrapper function
# ---------------------------------------------------------------------------
def composite_likelihood(
    S: numpy.ndarray,
    data: numpy.ndarray,
    indices: numpy.ndarray,
) -> float:
    """Compute the negative composite normal log-likelihood (bivariate pairwise).

    Evaluates the composite likelihood by summing bivariate normal
    log-likelihood contributions over all index pairs specified in
    ``indices``.  This is a helper function for multivariate GARCH models
    (DCC, BEKK, RARCH, scalar_vt_vech).

    The computation dispatches to one of two Numba JIT-compiled inner-loop
    kernels depending on whether ``data`` is a square matrix or a 1-D vector:

    - **Square path** (``data`` is K × K): extracts sub-blocks directly from
      the data matrix.  Typically used when ``data`` is an outer product of
      returns, i.e. ``data = r @ r.T``.
    - **Vector path** (``data`` is 1-D of length K): computes outer-product
      entries on the fly as ``data[i] * data[j]``.

    Parameters
    ----------
    S : array_like, shape (K, K)
        Covariance matrix.  Must be symmetric positive semi-definite (not
        explicitly checked for performance).
    data : array_like, shape (K, K) or (K,)
        Either a K × K matrix (e.g., outer product of returns) or a 1-D
        vector of length K (e.g., returns).  If a 2-D non-square array is
        provided with one dimension equal to 1, it is squeezed to a 1-D
        vector automatically.
    indices : array_like, shape (Q, 2)
        Q × 2 array of **0-based** index pairs specifying which bivariate
        pairs to include in the composite likelihood.  Each row ``[i, j]``
        references elements in ``S`` and ``data``.

        .. note::
           The original MATLAB version used **1-based** indices.  All Python
           callers must supply **0-based** indices.

    Returns
    -------
    float
        The negative composite normal log-likelihood value, averaged over
        the ``Q`` index pairs.

    Raises
    ------
    ValueError
        If ``S`` is not a 2-D square matrix.
    ValueError
        If ``indices`` is not a 2-D array with exactly 2 columns.
    ValueError
        If ``data`` cannot be interpreted as a square matrix or 1-D vector.

    Notes
    -----
    The composite likelihood formula for each bivariate pair ``(i, j)`` is:

    .. math::
        \\ell_k = 0.5 \\left( c + \\log(|\\Sigma_{ij}|) +
        \\frac{\\sigma_{jj} x_{ii} - 2 \\sigma_{ij} x_{ij} +
        \\sigma_{ii} x_{jj}}{|\\Sigma_{ij}|} \\right)

    where :math:`c = 2 \\log(2\\pi) \\approx 3.67575413281869` and
    :math:`|\\Sigma_{ij}| = \\sigma_{ii} \\sigma_{jj} - \\sigma_{ij}^2`.

    The total composite likelihood is the average over all ``Q`` pairs:
    :math:`\\ell = \\frac{1}{Q} \\sum_{k=1}^{Q} \\ell_k`.

    This replaces both the MATLAB ``.m`` fallback and the C MEX
    ``composite_likelihood.c`` kernel.

    Migrated from:
    - ``distributions/composite_likelihood.m`` (lines 1-51)
    - ``mex_source/composite_likelihood.c`` (lines 1-133)

    See Also
    --------
    mfe_toolbox.multivariate.dcc : DCC model driver.
    mfe_toolbox.multivariate.bekk : BEKK model driver.
    mfe_toolbox.multivariate.rarch : RARCH model driver.
    mfe_toolbox.multivariate.scalar_vt_vech : Scalar VT-VECH model driver.

    Examples
    --------
    Square data (outer product):

    >>> import numpy as np
    >>> S = np.array([[1.0, 0.5], [0.5, 1.0]])
    >>> data = np.array([[0.09, -0.06], [-0.06, 0.04]])  # outer product
    >>> indices = np.array([[0, 1]], dtype=np.int64)
    >>> ll = composite_likelihood(S, data, indices)

    Vector data (returns):

    >>> data_vec = np.array([0.3, -0.2])
    >>> ll_vec = composite_likelihood(S, data_vec, indices)
    """
    # --- Input coercion -------------------------------------------------
    # Ref: composite_likelihood.c:108-110 — MEX gateway extracts raw pointers;
    # the Python wrapper ensures proper dtype and contiguity.
    S = numpy.asarray(S, dtype=numpy.float64)
    data = numpy.asarray(data, dtype=numpy.float64)
    indices = numpy.asarray(indices, dtype=numpy.int64)

    # --- Input validation ------------------------------------------------
    # Validate S is a 2-D square matrix
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError(
            f"S must be a 2-D square matrix, got shape {S.shape}."
        )

    # Validate indices is 2-D with 2 columns
    # Ref: composite_likelihood.m:21 — q = size(indices, 1)
    if indices.ndim != 2 or indices.shape[1] != 2:
        raise ValueError(
            f"indices must be a 2-D array with 2 columns, got shape {indices.shape}."
        )

    q = indices.shape[0]
    if q == 0:
        return 0.0

    # --- Dispatch based on data shape ------------------------------------
    # Ref: composite_likelihood.m:22 — [m, n] = size(data)
    # Ref: composite_likelihood.m:25 — if m == n (square branch)
    # Ref: composite_likelihood.c:117-123 — MEX gateway swaps m,n if n > m

    if data.ndim == 2 and data.shape[0] == data.shape[1]:
        # Square data path — data is K x K (e.g. outer product of returns)
        # Ref: composite_likelihood.m:25-37
        return float(
            _composite_likelihood_core_square(S, data, indices, q)
        )
    elif data.ndim == 1:
        # Vector data path — data is 1-D of length K
        # Ref: composite_likelihood.m:38-50
        return float(
            _composite_likelihood_core_vector(S, data, indices, q)
        )
    elif data.ndim == 2:
        # Handle non-square 2-D arrays: squeeze if one dimension is 1
        # Ref: composite_likelihood.c:117-123 — MEX swaps m,n if n > m,
        # effectively treating row vectors and column vectors as 1-D.
        m, n = data.shape
        if m == 1 or n == 1:
            # Squeeze to 1-D vector
            data_flat = data.ravel()
            return float(
                _composite_likelihood_core_vector(S, data_flat, indices, q)
            )
        else:
            raise ValueError(
                f"data must be a square matrix (K x K) or a 1-D vector (K,), "
                f"got non-square 2-D array with shape {data.shape}."
            )
    else:
        raise ValueError(
            f"data must be 1-D or 2-D, got {data.ndim}-D array."
        )
