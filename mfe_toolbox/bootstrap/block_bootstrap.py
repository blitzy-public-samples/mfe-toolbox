"""
Circular block bootstrap for bootstrapping stationary, dependent time series.

Implements a fixed-block-length circular block bootstrap following the methodology
described in Künsch (1989) and Politis & Romano (1994). This is suitable for
constructing bootstrap confidence intervals and performing inference on weakly
dependent time series where observations exhibit serial correlation.

The circular variant wraps indices that exceed T back to the beginning of the
series, ensuring each observation has equal probability of inclusion.

Notes
-----
To generate bootstrap index sequences for other uses, such as bootstrapping
vector processes, set ``data`` to ``np.arange(T).reshape(-1, 1)``.

References
----------
Künsch, H.R. (1989). The Jackknife and the Bootstrap for General Stationary
Observations. *Annals of Statistics*, 17(3), 1217-1241.

Politis, D.N. & Romano, J.P. (1994). The Stationary Bootstrap. *Journal of the
American Statistical Association*, 89(428), 1303-1313.

See Also
--------
stationary_bootstrap : Stationary bootstrap with geometrically distributed
    block lengths.

Author
------
Original MATLAB: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
Python migration: Blitzy AI

Revision
--------
MATLAB Revision 2, 12/31/2001 → Python 3.12 migration
"""

import numpy as np


def block_bootstrap(
    data: np.ndarray, B: int, w: int
) -> tuple[np.ndarray, np.ndarray]:
    """Circular block bootstrap for dependent time series.

    Generates ``B`` bootstrap replications of the input ``data`` using a
    circular block bootstrap with fixed block length ``w``. Blocks of length
    ``w`` are drawn with random starting positions uniformly distributed over
    ``{0, 1, ..., T-1}`` (0-indexed). Indices that exceed ``T-1`` wrap
    circularly to the beginning of the series.

    Parameters
    ----------
    data : np.ndarray
        T-by-1 column vector (or 1-D array of length T) of data to be
        bootstrapped. Must contain at least 2 observations.
    B : int
        Number of bootstrap replications. Must be a positive integer.
    w : int
        Block length. Must be a positive integer satisfying ``1 <= w <= T``.

    Returns
    -------
    bsdata : np.ndarray
        T-by-B matrix of bootstrapped data, where each column is one
        bootstrap replication constructed from contiguous (circularly wrapped)
        blocks of the original data.
    indices : np.ndarray
        T-by-B matrix of 0-based index positions into ``data`` such that
        ``bsdata == data.ravel()[indices]`` element-wise.

    Raises
    ------
    ValueError
        If ``data`` is not a column vector, has fewer than 2 observations,
        ``w`` is not a positive integer no greater than T, or ``B`` is not
        a positive integer.

    Examples
    --------
    >>> import numpy as np
    >>> rng_state = np.random.default_rng(42)
    >>> data = np.arange(1, 11, dtype=float).reshape(-1, 1)
    >>> bsdata, indices = block_bootstrap(data, 3, 4)
    >>> bsdata.shape
    (10, 3)
    >>> indices.shape
    (10, 3)
    >>> np.all(bsdata == data.ravel()[indices])
    True

    Notes
    -----
    The algorithm proceeds as follows:

    1. Compute the number of blocks needed per replication:
       ``s = ceil(T / w)``.
    2. Draw ``s * B`` random starting indices uniformly from ``{0, ..., T-1}``,
       arranged as an ``s``-by-``B`` matrix ``Bs``.
    3. For each block ``i`` (``i = 0, ..., s-1``), fill rows
       ``[i*w, (i+1)*w)`` of the index matrix with
       ``Bs[i, :] + [0, 1, ..., w-1]``.
    4. Truncate the index matrix to the first ``T`` rows.
    5. Apply circular wrap-around: any index ``>= T`` is mapped to
       ``index - T`` (guaranteed to land in ``[0, T-1]`` since ``w <= T``).
    6. Construct ``bsdata`` by indexing into the flattened data vector.

    This is a **circular** block bootstrap: the data series is conceptually
    wrapped into a circle so that blocks starting near the end of the series
    continue from the beginning, ensuring all observations have equal marginal
    inclusion probability.
    """
    # ------------------------------------------------------------------
    # Input validation
    # Ref: block_bootstrap.m:30-46 — validate nargin, dimensions, types
    # ------------------------------------------------------------------

    # Ensure data is a numpy array
    data = np.asarray(data, dtype=np.float64)

    # Handle 1-D input by reshaping to column vector
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # Ref: block_bootstrap.m:34-36 — [t,k]=size(data); if k>1 error
    t, k = data.shape
    if k > 1:
        raise ValueError("DATA must be a column vector")

    # Ref: block_bootstrap.m:38-39 — if t<2 error
    if t < 2:
        raise ValueError("DATA must have at least 2 observations.")

    # Ref: block_bootstrap.m:41-42 — w must be positive scalar integer <= t
    if not np.isscalar(w) or int(w) != w or w < 1 or w > t:
        raise ValueError(
            "W must be a positive scalar integer smaller than T"
        )

    # Ref: block_bootstrap.m:44-45 — B must be positive scalar integer
    if not np.isscalar(B) or int(B) != B or B < 1:
        raise ValueError("B must be a positive scalar integer")

    # Cast to Python int for safe use as sizes/indices
    w = int(w)
    B = int(B)

    # ------------------------------------------------------------------
    # Block bootstrap algorithm
    # Ref: block_bootstrap.m:51-65
    # ------------------------------------------------------------------

    # Number of blocks needed per bootstrap sample
    # Ref: block_bootstrap.m:52 — s=ceil(t/w)
    s = int(np.ceil(t / w))

    # Generate random starting positions for each block (0-indexed)
    # Ref: block_bootstrap.m:54 — Bs=floor(rand(s,B)*t)+1  (MATLAB 1-indexed)
    # Python: integers in [0, t) — equivalent to MATLAB's [1, t] shifted by -1
    rng = np.random.default_rng()
    Bs = rng.integers(0, t, size=(s, B))  # shape (s, B), values in [0, t-1]

    # Pre-allocate index matrix (may be larger than T if T is not
    # divisible by w; we truncate later)
    # Ref: block_bootstrap.m:55 — indices=zeros(s*w,B)
    indices = np.zeros((s * w, B), dtype=np.int64)

    # Adder vector: [0, 1, 2, ..., w-1] tiled across B columns
    # Ref: block_bootstrap.m:58 — adder=repmat((0:w-1)',1,B)
    adder = np.tile(np.arange(w, dtype=np.int64).reshape(-1, 1), (1, B))

    # Fill blocks into the index matrix
    # Ref: block_bootstrap.m:59-62 — MATLAB loop i=1:w:t with 1-indexed
    # Python: iterate over block index, compute row range from block count
    for i in range(s):
        row_start = i * w
        row_end = row_start + w
        # Ref: block_bootstrap.m:60 — repmat(Bs(index,:),w,1)+adder
        indices[row_start:row_end, :] = (
            np.tile(Bs[i, :], (w, 1)) + adder
        )

    # Truncate to exactly T rows (discard padding from last partial block)
    # Ref: block_bootstrap.m:63 — indices=indices(1:t,:)
    indices = indices[:t, :]

    # Circular wrap-around: map indices >= T back into [0, T-1]
    # Ref: block_bootstrap.m:64 — indices(indices>t) = indices(indices>t)-t
    # MATLAB uses >t because indices are 1-based; Python uses >=t for 0-based
    # Since w <= t, subtracting t once is guaranteed to produce a valid index
    mask = indices >= t
    indices[mask] = indices[mask] - t

    # Construct bootstrapped data by fancy-indexing into the flattened data
    # Ref: block_bootstrap.m:65 — bsdata=data(indices)
    # data is (T, 1); flatten to 1-D for correct advanced indexing
    bsdata = data.ravel()[indices]

    return bsdata, indices
