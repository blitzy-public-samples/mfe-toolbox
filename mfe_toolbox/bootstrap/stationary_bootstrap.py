"""
Stationary Bootstrap for Dependent Time Series.

Implements the Politis-Romano stationary bootstrap for bootstrapping stationary,
dependent series with geometric block-length distribution. At each time step,
a Bernoulli draw determines whether to start a new block at a uniformly random
position or to continue from the previous position, yielding geometrically
distributed block lengths with expected length w.

References
----------
Politis, D.N. and Romano, J.P. (1994), "The Stationary Bootstrap",
Journal of the American Statistical Association, 89, 1303-1313.

Notes
-----
Migrated from bootstrap/stationary_bootstrap.m
Original Author: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 2    Date: 12/31/2001
"""

import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def _stationary_bootstrap_core(indices, select, t, B):
    """
    Numba-accelerated inner loop for stationary bootstrap index generation.

    Processes rows 1 through t-1 for each bootstrap column. Where the Bernoulli
    indicator ``select[i, j]`` is False (no new block), the index continues from
    the previous row: ``indices[i, j] = indices[i-1, j] + 1``. Positions where
    ``select`` is True already hold their new random starting index.

    After the continuation loop, circular wrapping maps any index >= t back into
    the valid 0-indexed range [0, t-1] by subtracting t. A single subtraction
    suffices because the maximum possible index before wrapping is (t-1) + (t-1)
    = 2t-2, which after subtracting t gives t-2 < t.

    Parameters
    ----------
    indices : ndarray of int64, shape (t, B)
        Pre-allocated index matrix. Row 0 and all positions where
        ``select`` is True are already filled with random starting
        indices in [0, t-1].
    select : ndarray of bool, shape (t, B)
        Boolean matrix where True indicates a new block start at that
        position.
    t : int
        Number of time periods (rows in data).
    B : int
        Number of bootstrap replications (columns).

    Returns
    -------
    indices : ndarray of int64, shape (t, B)
        Completed index matrix with circular wrapping applied. Every
        entry is in [0, t-1].
    """
    # Ref: stationary_bootstrap.m:61-65 — row-by-row continuation
    # MATLAB: for i=2:t
    #             indices(i,~select(i,:))=indices(i-1,~select(i,:))+1;
    #         end
    # Python uses 0-indexed rows; loop from 1 to t-1.
    for j in range(B):
        for i in range(1, t):
            if not select[i, j]:
                indices[i, j] = indices[i - 1, j] + 1

    # Ref: stationary_bootstrap.m:66 — circular wrap-around
    # MATLAB: indices(indices>t) = indices(indices>t)-t;   (1-indexed)
    # Python: indices >= t wraps to [0, t-1]               (0-indexed)
    for j in range(B):
        for i in range(t):
            if indices[i, j] >= t:
                indices[i, j] = indices[i, j] - t

    return indices


def stationary_bootstrap(data, B, w):
    """
    Stationary bootstrap for dependent time series.

    Implements the Politis-Romano stationary bootstrap with geometric
    block-length distribution for resampling stationary, dependent
    time series. At each time step after the first, a Bernoulli(p)
    draw (where p = 1/w) determines whether to jump to a new uniformly
    random position or to continue sequentially from the previous
    position with circular wrap-around.

    Parameters
    ----------
    data : ndarray, shape (T,) or (T, 1)
        Column vector of data to be bootstrapped.
    B : int
        Number of bootstrap replications. Must be a positive integer.
    w : float
        Average block length. The probability of starting a new block
        at each step is p = 1/w. Block lengths follow a geometric
        distribution with expected length w. Must be a positive scalar.

    Returns
    -------
    bsdata : ndarray, shape (T, B)
        Bootstrapped data matrix where each column is one bootstrap
        replication constructed from the original data using the
        generated indices.
    indices : ndarray, shape (T, B)
        0-indexed positions into the flattened data array such that
        ``bsdata == data.ravel()[indices]``.

    Raises
    ------
    ValueError
        If ``data`` is not a column vector, has fewer than 2 observations,
        ``w`` is not a positive scalar, or ``B`` is not a positive integer.

    Notes
    -----
    To generate bootstrap index sequences for other uses (e.g.,
    bootstrapping vector processes), set ``data`` to
    ``np.arange(T).reshape(-1, 1).astype(float)``.

    The stationary bootstrap differs from the block bootstrap in that
    block lengths are random (geometric distribution) rather than fixed,
    ensuring the resampled series is strictly stationary.

    The algorithm proceeds as follows:

    1. Draw initial positions uniformly from {0, 1, ..., T-1} for each
       of the B replications.
    2. For each subsequent row i = 1, ..., T-1 and each column j:

       - With probability p = 1/w, jump to a new uniformly random
         position (new block start).
       - With probability 1 - p, continue from the previous position
         (``indices[i, j] = indices[i-1, j] + 1``), wrapping circularly
         if the index reaches T.

    3. Construct ``bsdata`` by indexing the original data at the
       generated positions.

    See Also
    --------
    block_bootstrap : Fixed block-length circular bootstrap.

    References
    ----------
    Politis, D.N. and Romano, J.P. (1994), "The Stationary Bootstrap",
    Journal of the American Statistical Association, 89, 1303-1313.

    Examples
    --------
    >>> import numpy as np
    >>> rng_state = np.random.default_rng(42)
    >>> data = np.arange(1, 11, dtype=float).reshape(-1, 1)
    >>> bsdata, indices = stationary_bootstrap(data, 100, 5.0)
    >>> bsdata.shape
    (10, 100)
    >>> indices.shape
    (10, 100)
    >>> np.all((indices >= 0) & (indices < 10))
    True
    """
    # =========================================================================
    # Input Checking
    # Ref: stationary_bootstrap.m:28-50
    # =========================================================================
    data = np.asarray(data, dtype=np.float64)

    # Handle 1-D input by converting to column vector
    # Ref: stationary_bootstrap.m:35 — [t,k]=size(data);
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    t, k = data.shape

    # Ref: stationary_bootstrap.m:36-38 — data must be column vector
    if k > 1:
        raise ValueError('DATA must be a column vector')

    # Ref: stationary_bootstrap.m:39-41 — must have at least 2 observations
    if t < 2:
        raise ValueError('DATA must have at least 2 observations.')

    # Ref: stationary_bootstrap.m:42-44 — w must be positive scalar
    if not np.isscalar(w) or w <= 0:
        raise ValueError('W must be a positive scalar.')

    # Ref: stationary_bootstrap.m:45-47 — B must be positive scalar integer
    if not np.isscalar(B) or B < 1 or np.floor(B) != B:
        raise ValueError('B must be a positive scalar integer')

    B = int(B)

    # =========================================================================
    # Bootstrap Index Generation
    # Ref: stationary_bootstrap.m:52-66
    # =========================================================================

    # Ref: stationary_bootstrap.m:53 — define probability of new block
    p = 1.0 / w

    # Random number generator following AAP convention
    rng = np.random.default_rng()

    # Ref: stationary_bootstrap.m:55 — pre-allocate index matrix
    indices = np.zeros((t, B), dtype=np.int64)

    # Ref: stationary_bootstrap.m:57 — initial random positions
    # MATLAB: indices(1,:) = ceil(t*rand(1,B));  — generates 1..t (1-indexed)
    # Python: rng.integers(0, t, size=B)         — generates 0..t-1 (0-indexed)
    indices[0, :] = rng.integers(0, t, size=B)

    # Ref: stationary_bootstrap.m:59 — Bernoulli draws for new block start
    # Each entry is True with probability p = 1/w
    select = rng.random((t, B)) < p

    # Ref: stationary_bootstrap.m:60 — assign new random positions where selected
    # MATLAB: indices(select) = ceil(rand(1, sum(sum(select))) * t);
    # Python: 0-indexed random integers for all True positions
    n_selected = int(np.sum(select))
    if n_selected > 0:
        indices[select] = rng.integers(0, t, size=n_selected)

    # Ref: stationary_bootstrap.m:61-66 — inner loop with circular wrapping
    # Accelerated via Numba JIT (replaces potential MEX acceleration path)
    indices = _stationary_bootstrap_core(indices, select, t, B)

    # =========================================================================
    # Construct Bootstrapped Data
    # Ref: stationary_bootstrap.m:68 — bsdata = data(indices);
    # =========================================================================
    # Flatten data to 1-D for integer indexing; result shape is (T, B)
    data_flat = data.ravel()
    bsdata = data_flat[indices]

    return bsdata, indices
