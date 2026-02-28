"""
Baxter-King band-pass filter for trend/cycle/noise decomposition.

Migrated from timeseries/bkfilter.m — implements the Baxter-King (1999) approximate
band-pass filter using symmetric finite-order moving average representations to
decompose time series data into trend, cyclic, and noise components.

The filter constructs ideal band-pass weights via sinc functions at specified
high-frequency and low-frequency cutoff periods, normalizes them to be mean-preserving,
and applies symmetric convolution windows to the data.  The first and last K
observations receive boundary treatment (trend = Y, cyclic = 0, noise = 0).

Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 1    Date: 19/10/2009
"""

import numpy as np


def bkfilter(y, p, q, k=12):
    """
    Baxter-King filtering of multiple time series.

    Applies the Baxter-King (1999) approximate band-pass filter to decompose
    time series data into trend, cyclic, and noise components using symmetric
    finite-order moving average representations.

    Parameters
    ----------
    y : array_like
        A T by K matrix of data to be filtered.  T is the number of time
        observations and K is the number of series.  A 1-D array of length
        T is treated as a single column (T, 1).
    p : int or float
        Number of periods to use in the higher-frequency filter (e.g. 6 for
        quarterly data).  Must be at least 1.
    q : int or float
        Number of periods to use in the lower-frequency filter (e.g. 32 for
        quarterly data).  Must satisfy Q >= P.  Q can be ``numpy.inf``, in
        which case the low-pass filter becomes a (2K+1)-point uniform moving
        average.
    k : int, optional
        Number of points to use in the finite approximation bandpass filter.
        Default is 12.  The filter sets the first and last K observations to
        boundary values (trend = Y, cyclic = 0, noise = 0).

    Returns
    -------
    trend : numpy.ndarray
        T by K matrix containing the filtered trend component.  The first
        and last K rows equal Y.
    cyclic : numpy.ndarray
        T by K matrix containing the filtered cyclic component.  The first
        and last K rows are 0.
    noise : numpy.ndarray
        T by K matrix containing the filtered noise component.  The first
        and last K rows are 0.

    Notes
    -----
    The decomposition satisfies ``Y = TREND + CYCLIC + NOISE`` for interior
    points, where the trend is produced by the low-pass filter (cutoff Q)
    and the cyclic component is the difference between the high-pass filter
    (cutoff P) and the low-pass filter.  The noise component is the residual
    after removing the high-pass filtered output from Y.

    Ideal band-pass filter weights are computed as:

    .. math::

        b_j = \\frac{\\sin(2\\pi j / p) - \\sin(2\\pi j / q)}{\\pi j}
        \\quad \\text{for } j = 1, \\ldots, K

    with center weight :math:`b_0 = 2/p - 2/q`.  Weights are then adjusted
    by an additive constant :math:`\\theta = (1 - \\sum b_j) / (2K + 1)` to
    ensure the filter is mean-preserving.

    Recommended values:

    * Quarterly data: P = 6, Q = 32 or Q = 40
    * Monthly data: P = 18, Q = 96 or Q = 120

    Setting Q = P produces a single bandpass filter and the cyclic component
    will be identically zero.

    When ``q = numpy.inf``, the angular frequency ``fq = 0`` and all
    low-frequency weights become zero before normalization.  After
    mean-preserving adjustment the low-pass filter reduces to a uniform
    ``1 / (2K + 1)`` moving average, which is the expected limiting
    behaviour.

    References
    ----------
    Baxter, M. and King, R.G. (1999), "Measuring Business Cycles:
    Approximate Band-Pass Filters for Economic Time Series", *Review of
    Economics and Statistics*, 81(4), 575-593.

    See Also
    --------
    hp_filter : Hodrick-Prescott filter.
    beveridgenelson : Beveridge-Nelson trend-cycle decomposition.

    Examples
    --------
    Standard BK filter with periods of 6 and 32 for quarterly data:

    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> y = rng.standard_normal((100, 1))
    >>> trend, cyclic, noise = bkfilter(y, 6, 32)
    >>> trend.shape
    (100, 1)

    BK filter for low-pass filtering only (cyclic will be zero):

    >>> trend, cyclic, noise = bkfilter(y, 40, 40)

    BK filter using a 2-sided 20-point approximation:

    >>> trend, cyclic, noise = bkfilter(y, 6, 32, 20)

    BK filter with ``q = inf`` (pure moving-average low-pass):

    >>> trend, cyclic, noise = bkfilter(y, 6, np.inf, 12)
    """
    # ------------------------------------------------------------------
    # Input Checking
    # Ref: bkfilter.m:48-66 — nargin handling replaced by default k=12
    # ------------------------------------------------------------------

    # Convert input to numpy array and ensure it is 2-D.
    # Ref: bkfilter.m:55 — ndims(y) > 2 check
    y_arr = np.asarray(y, dtype=float)
    if y_arr.ndim == 1:
        # Ref: bkfilter.m — MATLAB column vectors are (T, 1); replicate
        # this convention for 1-D Python inputs.
        y_arr = np.atleast_2d(y_arr).T  # shape (T, 1)
    elif y_arr.ndim > 2:
        raise ValueError('Y must be a T by K matrix with T>=4')

    T, n = y_arr.shape

    # Ref: bkfilter.m:55 — size(y,1) < k/2 guard
    if T < k / 2:
        raise ValueError('Y must be a T by K matrix with T>=4')

    # Ref: bkfilter.m:58-59 — P validation
    if not np.isscalar(p) or p <= 0 or p < 1:
        raise ValueError('P must be a positive scalar.')

    # Ref: bkfilter.m:61-62 — Q validation; allows q = inf since
    # inf >= p for any finite positive p.
    if not np.isscalar(q) or q <= 0 or q < p:
        raise ValueError('Q must be a positive scalar with Q>=P.')

    # Validate k is a positive integer
    if not np.isscalar(k) or k < 1:
        raise ValueError('K must be a positive integer.')

    k = int(k)

    # ------------------------------------------------------------------
    # Core BK Filter Computation
    # Ref: bkfilter.m:70-91
    # ------------------------------------------------------------------

    # Compute angular frequencies.
    # Ref: bkfilter.m:72-73 — fp = (2*pi)/p; fq = (2*pi)/q
    # When q = inf, fq evaluates to 0.0, which is correct.
    fp = (2.0 * np.pi) / p
    fq = (2.0 * np.pi) / q

    # Index array for j = 1, ..., k used in sinc-based weight computation.
    # Ref: bkfilter.m:77 — (1:k) in MATLAB maps to np.arange(1, k+1)
    j = np.arange(1, k + 1, dtype=float)

    # ------------------------------------------------------------------
    # Construct high-frequency (period P) filter weights  bp
    # Ref: bkfilter.m:75-81
    # Ideal weights: center = fp/pi, tails = sin(j*fp)/(j*pi)
    # ------------------------------------------------------------------
    bp = np.zeros(2 * k + 1)
    # Ref: bkfilter.m:76 — bp(k+1) = fp/pi  (1-indexed → 0-indexed: bp[k])
    bp[k] = fp / np.pi
    # Ref: bkfilter.m:77 — weights = (sin((1:k)*fp)./((1:k).*pi))'
    weights_p = np.sin(j * fp) / (j * np.pi)
    # Ref: bkfilter.m:78 — bp(1:k) = flipud(weights)  — left symmetric wing
    bp[:k] = weights_p[::-1]
    # Ref: bkfilter.m:79 — bp(k+2:2*k+1) = weights  — right symmetric wing
    bp[k + 1:2 * k + 1] = weights_p
    # Mean-preserving normalisation: adjust weights so they sum to 1.
    # Ref: bkfilter.m:80-81
    thetap = (1.0 - np.sum(bp)) / (2 * k + 1)
    bp = bp + thetap

    # ------------------------------------------------------------------
    # Construct low-frequency (period Q) filter weights  bq
    # Ref: bkfilter.m:83-89
    # When q = inf, fq = 0: all sinc weights are 0, and after
    # normalisation bq = 1/(2K+1) everywhere (uniform moving average).
    # ------------------------------------------------------------------
    bq = np.zeros(2 * k + 1)
    # Ref: bkfilter.m:84 — bq(k+1) = fq/pi
    bq[k] = fq / np.pi
    # Ref: bkfilter.m:85
    weights_q = np.sin(j * fq) / (j * np.pi)
    # Ref: bkfilter.m:86 — left symmetric wing
    bq[:k] = weights_q[::-1]
    # Ref: bkfilter.m:87 — right symmetric wing
    bq[k + 1:2 * k + 1] = weights_q
    # Mean-preserving normalisation
    # Ref: bkfilter.m:88-89
    thetaq = (1.0 - np.sum(bq)) / (2 * k + 1)
    bq = bq + thetaq

    # ------------------------------------------------------------------
    # Band-pass (cyclic) filter = difference of high and low filters
    # Ref: bkfilter.m:91 — b = bp - bq
    # ------------------------------------------------------------------
    b = bp - bq

    # ------------------------------------------------------------------
    # Apply filters via symmetric convolution
    # Ref: bkfilter.m:93-104
    # ------------------------------------------------------------------

    # Initialise outputs.
    # Ref: bkfilter.m:93-95
    # Boundary rule: first and last K points have trend = Y,
    # cyclic = 0, noise = 0.
    trend = y_arr.copy()
    cyclic = np.zeros((T, n))
    noise = np.zeros((T, n))

    # Apply filters to interior points (excluding first and last K).
    # Ref: bkfilter.m:97 — for t = k+1 : T-k  (1-indexed)
    #   → Python 0-indexed: range(k, T - k)
    for t in range(k, T - k):
        # Extract the symmetric (2K+1)-point window centred at t.
        # Ref: bkfilter.m:99 — y(t-k:t+k,:) (1-indexed, inclusive)
        #   → y_arr[t-k : t+k+1, :] (0-indexed, exclusive end)
        window = y_arr[t - k:t + k + 1, :]

        # Trend is the low-pass filtered component: bq' * window
        # Ref: bkfilter.m:99 — trend(t,:) = bq' * y(t-k:t+k,:)
        trend[t, :] = bq @ window

        # Cyclic is the band-pass filtered component: (bp - bq)' * window
        # Ref: bkfilter.m:101 — cyclic(t,:) = b' * y(t-k:t+k,:)
        cyclic[t, :] = b @ window

        # Noise is the residual: y(t) - bp' * window
        # Ref: bkfilter.m:103 — noise(t,:) = y(t,:) - bp' * y(t-k:t+k,:)
        noise[t, :] = y_arr[t, :] - bp @ window

    return trend, cyclic, noise
