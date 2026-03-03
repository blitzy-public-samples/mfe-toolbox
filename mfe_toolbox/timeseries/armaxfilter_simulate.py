"""
ARMAX(P,Q) simulation with normal errors.

Migrated from timeseries/armaxfilter_simulate.m — Author: Kevin Sheppard
(Revision 3, Date: 9/1/2005)

Simulates AR, MA, ARMA, and ARMAX data with a burn-in period to eliminate
initial condition effects.

Notes
-----
The ARMAX(P,Q) model simulated is::

    y(t) = const + arp(1)*y(t-1) + ... + arp(P)*y(t-P)
                 + ma(1)*e(t-1)  + ... + ma(Q)*e(t-Q)
                 + xp(1)*x(t,1)  + ... + xp(K)*x(t,K)
                 + e(t)

See Also
--------
armaxfilter : ARMAX estimation.
heterogeneousar : Heterogeneous AR (HAR) model.
"""

import numpy as np

from mfe_toolbox.utility.newlagmatrix import newlagmatrix

__all__ = ['armaxfilter_simulate']


def armaxfilter_simulate(T, const, ar=0, ARparams=None, ma=0, MAparams=None,
                         X=None, Xparams=None):
    """
    ARMAX(P,Q) simulation with normal errors.

    Simulates AR, MA, ARMA, and ARMAX data with a configurable burn-in
    period of 2000 observations to eliminate initial condition effects.
    The burn-in is discarded before returning results.

    Parameters
    ----------
    T : int or numpy.ndarray
        Length of data series to simulate. If a 1-D array with more than
        one element is provided, it is interpreted as user-supplied random
        innovations (errors), and T is inferred as the length of that
        array.
    const : float
        Value of the constant in the model. Set to 0 to omit.
    ar : int, optional
        Order of the AR component. To include only selected lags (e.g.,
        t-1 and t-3), set to 3 and zero out the unused coefficients in
        ``ARparams``. Default is 0 (no AR).
    ARparams : array_like or None, optional
        ``ar``-length vector of AR parameters. Required when ``ar > 0``.
        Default is None.
    ma : int, optional
        Order of the MA component. To include only selected lags of the
        error (e.g., t-1 and t-3), set to 3 and zero out the unused
        coefficients in ``MAparams``. Default is 0 (no MA).
    MAparams : array_like or None, optional
        ``ma``-length vector of MA parameters. Required when ``ma > 0``.
        Default is None.
    X : numpy.ndarray or None, optional
        T × K matrix of exogenous variables. Default is None.
    Xparams : array_like or None, optional
        K-length vector of parameters on the exogenous variables. Required
        when ``X`` is provided. Default is None.

    Returns
    -------
    y : numpy.ndarray
        T-length 1-D vector of simulated data.
    errors : numpy.ndarray
        T-length 1-D vector of innovations used in the simulation.

    Raises
    ------
    ValueError
        If the number of AR/MA/X parameters is incorrect, if ``ar`` or
        ``ma`` are negative, if ``const`` is not a scalar, or if
        user-supplied errors are not a 1-D vector.

    Examples
    --------
    Simulate an AR(1) with a constant:

    >>> y, e = armaxfilter_simulate(500, 0.5, 1, [0.9])

    Simulate an AR(1) without a constant:

    >>> y, e = armaxfilter_simulate(500, 0, 1, [0.9])

    Simulate an ARMA(1,1) with a constant:

    >>> y, e = armaxfilter_simulate(500, 0.5, 1, [0.95], 1, [-0.5])

    Simulate a MA(1) with a constant:

    >>> y, e = armaxfilter_simulate(500, 0.5, 0, None, 1, [-0.5])

    Simulate a seasonal MA(4) with a constant:

    >>> y, e = armaxfilter_simulate(500, 0.5, 0, None, 4, [0.6, 0, 0, 0.2])

    Notes
    -----
    Migrated from ``timeseries/armaxfilter_simulate.m`` (Kevin Sheppard,
    Revision 3, Date: 9/1/2005).

    When user-supplied errors are provided (``T`` is an array), a
    bootstrap sample of 2000 observations drawn with replacement from
    those errors is prepended as the burn-in period.

    MATLAB 1-based indexing is translated to Python 0-based indexing
    throughout the AR/MA recursion loops.
    """
    # ------------------------------------------------------------------
    # Input validation
    # Ref: armaxfilter_simulate.m:56-122 — Input Checking block
    # ------------------------------------------------------------------

    # --- Normalize ar and ma to non-negative integers ---
    # Ref: armaxfilter_simulate.m:59-67 — nargin-based defaults set ar=0, ma=0
    if ar is None:
        ar = 0
    else:
        ar_arr = np.asarray(ar)
        # Ref: armaxfilter_simulate.m:61 — ar=0 when omitted (empty)
        ar = int(ar_arr.flat[0]) if ar_arr.size > 0 else 0

    if ma is None:
        ma = 0
    else:
        ma_arr = np.asarray(ma)
        # Ref: armaxfilter_simulate.m:62 — ma=0 when omitted (empty)
        ma = int(ma_arr.flat[0]) if ma_arr.size > 0 else 0

    # --- Normalize parameter arrays to 1-D float64 ---
    # Ref: armaxfilter_simulate.m:93-95 — MATLAB transposes column to row;
    # Python .ravel() achieves the same flattening.
    if ARparams is None:
        ARparams = np.array([], dtype=np.float64)
    else:
        ARparams = np.asarray(ARparams, dtype=np.float64).ravel()

    # Ref: armaxfilter_simulate.m:101-103 — same transpose for MAparams
    if MAparams is None:
        MAparams = np.array([], dtype=np.float64)
    else:
        MAparams = np.asarray(MAparams, dtype=np.float64).ravel()

    if Xparams is None:
        Xparams = np.array([], dtype=np.float64)
    else:
        Xparams = np.asarray(Xparams, dtype=np.float64).ravel()

    # --- Handle T as scalar length or user-supplied errors ---
    # Ref: armaxfilter_simulate.m:81-87
    user_errors = False
    T_input = np.asarray(T, dtype=np.float64)

    if T_input.size > 1:
        # User supplied errors — T is a vector of innovations
        # Ref: armaxfilter_simulate.m:84-86 — min(size(e))>1 check
        if T_input.ndim > 1 and min(T_input.shape) > 1:
            raise ValueError(
                'If using user supplied errors, these must be a column vector '
                '(e.g. T=numpy.random.default_rng().standard_normal(100))')
        e_user = T_input.ravel()
        T_len = len(e_user)
        user_errors = True
    else:
        T_len = int(T_input.flat[0])

    # --- Validate AR parameters ---
    # Ref: armaxfilter_simulate.m:89-91 — length(ARparams)<ar
    if ar > 0 and len(ARparams) < ar:
        raise ValueError('Incorrect number of AR parameters')

    # --- Validate MA parameters ---
    # Ref: armaxfilter_simulate.m:97-99 — length(MAparams)<ma
    if ma > 0 and len(MAparams) < ma:
        raise ValueError('Incorrect number of MA parameters')

    # --- Validate exogenous variables ---
    # Ref: armaxfilter_simulate.m:105-111
    has_exog = X is not None
    if has_exog:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        # Ref: armaxfilter_simulate.m:105-107 — length(Xparams)<size(X,2)
        if len(Xparams) < X.shape[1]:
            raise ValueError(
                'Incorrect number of X parameters. Xparams should be K by 1.')
        # Ref: armaxfilter_simulate.m:109-111 — length(X)~=T
        if X.shape[0] != T_len:
            raise ValueError('X should be length T')
        # Ref: armaxfilter_simulate.m:128-130 — ensure Xparams is row-oriented
        Xparams = Xparams[:X.shape[1]]

    # --- Validate ar and ma are non-negative scalars ---
    # Ref: armaxfilter_simulate.m:113-115
    if ar < 0 or ma < 0:
        raise ValueError('MA and AR must all be non negative scalars.')

    # --- Validate const is scalar ---
    # Ref: armaxfilter_simulate.m:117-119 — isscalar(const)
    if not np.isscalar(const):
        raise ValueError('CONST must be a scalar')
    const = float(const)

    # ------------------------------------------------------------------
    # Simulation setup
    # ------------------------------------------------------------------

    # Ref: armaxfilter_simulate.m:134 — B=2000 burn-in period
    B = 2000
    # Ref: AAP Section 0.7.3 — randn → rng.standard_normal via default_rng()
    rng = np.random.default_rng()

    # --- Generate or extend error vector ---
    # Ref: armaxfilter_simulate.m:136-140
    if user_errors:
        # Ref: armaxfilter_simulate.m:137 — e=[e(ceil(rand(B,1)*T));e]
        # MATLAB ceil(rand(B,1)*T) draws 1-indexed bootstrap indices {1..T}
        # Python 0-indexed: subtract 1 to get {0..T-1}
        bootstrap_idx = np.ceil(rng.random(B) * T_len).astype(int) - 1
        # Guard against the (negligible probability) case rand returns 0.0
        bootstrap_idx = np.clip(bootstrap_idx, 0, T_len - 1)
        e = np.concatenate([e_user[bootstrap_idx], e_user])
    else:
        # Ref: armaxfilter_simulate.m:139 — e=randn(T+B,1)
        e = rng.standard_normal(T_len + B)

    # --- Set up exogenous contribution ---
    # Ref: armaxfilter_simulate.m:124-131, 143-145
    total_len = len(e)  # T_len + B

    if has_exog:
        # Ref: armaxfilter_simulate.m:143 — meanX=mean(X)
        mean_X = np.mean(X, axis=0)  # (K,) column means
        # Ref: armaxfilter_simulate.m:144 — repmat(meanX,length(e)-length(X),1)*Xparams'
        # Replicate the mean exogenous effect over the burn-in period
        n_burnin_rows = total_len - X.shape[0]
        burn_part = np.tile(mean_X, (n_burnin_rows, 1)) @ Xparams  # (n_burnin_rows,)
        # Ref: armaxfilter_simulate.m:145 — X*Xparams'
        actual_part = X @ Xparams  # (T_len,)
        exog = np.concatenate([burn_part, actual_part]) + const
        # Store scalar mean effect for starting value computation
        mean_xb = float(mean_X @ Xparams)
    else:
        # Ref: armaxfilter_simulate.m:124-126 — X=0, Xparams=0
        # No exogenous variables: exog is just the constant
        exog = np.full(total_len, const)
        mean_xb = 0.0

    # ------------------------------------------------------------------
    # MA component
    # Ref: armaxfilter_simulate.m:147-154
    # ------------------------------------------------------------------
    if ma > 0:
        # Ref: armaxfilter_simulate.m:148 — e=[zeros(ma,1); e]
        # Prepend ma zeros so lagged errors start at zero
        e = np.concatenate([np.zeros(ma), e])
        # Ref: armaxfilter_simulate.m:149 — [e,elag]=newlagmatrix(e,ma,0)
        # Splits extended error into contemporaneous (trimmed) and lagged columns
        e_trimmed, elag = newlagmatrix(e, ma, 0)
        # newlagmatrix returns 2-D arrays; flatten contemporaneous back to 1-D
        e = e_trimmed.ravel()
        # Ref: armaxfilter_simulate.m:150 — exog=exog+elag*MAparams'+e
        # elag is (total_len × ma), MAparams[:ma] is (ma,)
        exog = exog + elag @ MAparams[:ma] + e
    else:
        # Ref: armaxfilter_simulate.m:153 — exog=exog+e
        exog = exog + e

    # ------------------------------------------------------------------
    # AR recursion
    # Ref: armaxfilter_simulate.m:160-174
    # ------------------------------------------------------------------

    # Ref: armaxfilter_simulate.m:160-164 — compute stationary starting value y0
    ar_sum = float(np.sum(ARparams[:ar])) if ar > 0 else 0.0
    if np.abs(ar_sum) < 1.0:
        # Ref: armaxfilter_simulate.m:161 — y0=(const+meanX*Xparams')/(1-sum(ARparams))
        # Unconditional mean of a stationary ARMAX process
        y0 = (const + mean_xb) / (1.0 - ar_sum)
    else:
        # Ref: armaxfilter_simulate.m:163 — explosive or unit root process
        y0 = 0.0

    # Ref: armaxfilter_simulate.m:166 — y=repmat(y0,length(e),1)
    # Initialize all y values to the starting value y0
    n_total = len(exog)
    y = np.full(n_total, y0)

    if ar > 0:
        ar_coefs = ARparams[:ar]
        # Ref: armaxfilter_simulate.m:169-171 — for i=ar+1:length(e) (1-indexed)
        # Python 0-indexed: loop from index ar to n_total-1
        for i in range(ar, n_total):
            # Ref: armaxfilter_simulate.m:170 — y(i)=exog(i)+ARparams*y(i-1:-1:i-ar)
            # MATLAB y(i-1:-1:i-ar) reversed = [y(i-1), y(i-2), ..., y(i-ar)]
            # Python y[i-ar:i][::-1] = [y[i-1], y[i-2], ..., y[i-ar]]
            y[i] = exog[i] + np.sum(ar_coefs * y[i - ar:i][::-1])
    else:
        # Ref: armaxfilter_simulate.m:173 — y=exog when no AR
        y = exog.copy()

    # ------------------------------------------------------------------
    # Trim burn-in and return
    # Ref: armaxfilter_simulate.m:177-178
    # MATLAB: y(B+1:T+B) → Python 0-indexed: y[B:T_len+B]
    # ------------------------------------------------------------------
    trim_indices = np.arange(B, T_len + B)
    y_out = y[trim_indices]
    errors_out = e[trim_indices]

    return y_out, errors_out
