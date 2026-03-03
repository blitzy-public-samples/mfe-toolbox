"""
ARMA Forecasting — h-step ahead predictions from estimated ARMA models.

Migrated from:
  - timeseries/arma_forecaster.m  (Kevin Sheppard, Revision 3, 1/1/2007)

This module implements the ``arma_forecaster`` function which produces h-step
ahead forecasts from an ARMA(P,Q) model using estimated parameters and the
innovation history obtained via :func:`~mfe_toolbox.timeseries.armaxerrors.armaxerrors`.

The forecasting procedure:

1. Recovers the residual (innovation) series from the fitted model via
   ``armaxerrors``.
2. Constructs h-step ahead forecasts by forward recursion:
   - AR component: weighted sum of lagged observed/forecasted values
   - MA component: weighted sum of lagged residuals (future errors replaced
     by their unconditional expectation of 0 beyond the error horizon)
3. Aligns actual future values y(t+h) to position t for comparison.
4. Computes forecast errors and theoretical forecast standard deviation
   under homoskedasticity.

Public API
----------
arma_forecaster(y, parameters, constant, p, q, r, h, seregression, hold_back)
    -> (yhattph, ytph, forerr, ystd)

See Also
--------
mfe_toolbox.timeseries.armaxfilter : ARMAX model estimation driver.
mfe_toolbox.timeseries.heterogeneousar : Heterogeneous AR (HAR) model.
"""

import numpy as np

from mfe_toolbox.timeseries.armaxerrors import armaxerrors


def arma_forecaster(y, parameters, constant, p, q, r, h,
                    seregression=None, hold_back=None):
    """
    Produce h-step ahead forecasts from an ARMA(P,Q) model.

    Starting at observation *r* and ending at the end of the sample, this
    function produces h-step ahead forecasts, shifts the observed series so
    that y(t+h) aligns with the forecast at position t, computes forecast
    errors, and the theoretical forecast standard deviation (assuming
    homoskedasticity).

    Parameters
    ----------
    y : array_like
        1-D observed series (column vector in MATLAB).
    parameters : array_like
        Estimated parameter vector laid out as
        ``[constant?, AR_1 … AR_np, MA_1 … MA_nq]`` where ``constant?``
        is present only when *constant* == 1.
    constant : {0, 1}
        1 to include a constant term, 0 to exclude.
    p : array_like
        Non-negative integer vector of AR lag orders included in the model.
        May be empty or ``[0]`` to indicate no AR component.
    q : array_like
        Non-negative integer vector of MA lag orders included in the model.
        May be empty or ``[0]`` to indicate no MA component.
    r : int
        Length of the estimation (regression) sample.  The first *r*
        observations are used for estimation; the remainder for prediction
        so that ``r + P_forecast = T``.
    h : int
        Forecast horizon (number of steps ahead).
    seregression : float, optional
        Standard error of the regression, used to compute confidence
        intervals.  Default is 1.0.
    hold_back : int or None, optional
        Additional hold-back beyond ``max(p)`` for burn-in.  Default is
        ``None`` (no additional hold-back).

    Returns
    -------
    yhattph : numpy.ndarray, shape (T,)
        h-step ahead forecasts.  Element at position *t* is the time-*t*
        forecast of y(t+h).  The first *r* − 1 elements are ``NaN``;
        the next *T* − *r* − *h* are pseudo in-sample forecasts; the
        final *h* are genuine out-of-sample forecasts.
    ytph : numpy.ndarray, shape (T,)
        Actual data at time t+h shifted to position t.  The first *r* − 1
        elements and the last *h* elements are ``NaN``.
    forerr : numpy.ndarray, shape (T,)
        Forecast errors: ``ytph − yhattph``.
    ystd : float
        Theoretical standard deviation of the h-step ahead forecast under
        the assumption of homoskedastic innovations.

    Raises
    ------
    ValueError
        On invalid inputs (non-column y, incompatible parameter length,
        non-positive r or h, etc.).

    Notes
    -----
    * Values not relevant for the forecasting exercise are returned as
      ``NaN``.
    * The forecast standard deviation ``ystd`` is computed from the
      MA(∞) representation's impulse response coefficients and assumes
      constant innovation variance equal to ``seregression ** 2``.
    * Ref: arma_forecaster.m — the original MATLAB implementation.

    See Also
    --------
    mfe_toolbox.timeseries.armaxfilter : ARMAX model estimation.
    mfe_toolbox.timeseries.heterogeneousar : Heterogeneous AR model.
    """

    # ==================================================================
    # Input Validation  (Ref: arma_forecaster.m:54-158)
    # ==================================================================

    # --- y validation (Ref: arma_forecaster.m:63-68) ---
    y = np.asarray(y, dtype=np.float64).ravel()
    # Ref: arma_forecaster.m:63 — scalar or row-vector y is invalid
    if y.size == 1:
        raise ValueError('y series must be a column vector.')
    # Ref: arma_forecaster.m:66
    if y.size == 0:
        raise ValueError('y is empty.')
    T = len(y)

    # --- p validation (Ref: arma_forecaster.m:72-91) ---
    p = np.asarray(p, dtype=np.float64).ravel()
    # Ref: arma_forecaster.m:75-77
    if p.size == 0:
        p = np.array([0.0])
    # Ref: arma_forecaster.m:81-83
    if np.any(p < 0) or np.any(np.floor(p) != p):
        raise ValueError('P must contain non-negative integers only')
    # Ref: arma_forecaster.m:84-86
    if np.max(p) > 0 and np.max(p) >= (T - np.max(p)):
        raise ValueError('Too many lags in the AR.  max(P)<T/2')

    # Ref: arma_forecaster.m:88 — NOTE: the original MATLAB source checks
    # unique(q) here, not unique(p).  The error message says 'P' but tests
    # the q vector.  This is a copy-paste artifact in the original code.
    # We replicate the exact MATLAB behaviour for numerical parity.
    q_raw = np.asarray(q, dtype=np.float64).ravel()
    if q_raw.size > 0 and len(np.unique(q_raw)) != len(q_raw):
        raise ValueError('P must contain at most one of each lag')
    # Ref: arma_forecaster.m:91 — count non-zero AR lags
    np_count = int(np.sum(p != 0))

    # --- q validation (Ref: arma_forecaster.m:96-112) ---
    q = q_raw.copy()
    # Ref: arma_forecaster.m:99-101
    if q.size == 0:
        q = np.array([0.0])
    # Ref: arma_forecaster.m:105-107
    if np.any(q < 0) or np.any(np.floor(q) != q):
        raise ValueError('Q must contain non-negative integers only')
    # Ref: arma_forecaster.m:108-110
    if len(np.unique(q)) != len(q):
        raise ValueError('Q must contain at most one of each lag')
    # Ref: arma_forecaster.m:111
    nq_count = int(np.sum(q != 0))

    # --- constant validation (Ref: arma_forecaster.m:116-121) ---
    # Ref: arma_forecaster.m:116-118
    if (not constant
            and (p.size == 0 or np.max(p) == 0)
            and (q.size == 0 or np.max(q) == 0)):
        raise ValueError(
            'At least one of CONSTANT, P or Q must be nonnegative')
    # Ref: arma_forecaster.m:119-121
    if constant not in (0, 1):
        raise ValueError('CONSTANT must be 0 or 1')
    constant = int(constant)

    # --- parameters validation (Ref: arma_forecaster.m:125-133) ---
    parameters = np.asarray(parameters, dtype=np.float64).ravel()
    expected_len = constant + np_count + nq_count
    if parameters.size == 0 or len(parameters) != expected_len:
        raise ValueError(
            'length of PARAMETERS input must compatible with the '
            'constant, AR and MA lags')

    # --- r validation (Ref: arma_forecaster.m:137-139) ---
    r = int(r)
    if r <= 0 or r > T or r != int(r):
        raise ValueError(
            'R must be a postive scalar less that or equal to T')

    # --- h validation (Ref: arma_forecaster.m:143-145) ---
    h = int(h)
    if h <= 0:
        raise ValueError('H must be a postive scalar')

    # --- seregression (Ref: arma_forecaster.m:149-155) ---
    if seregression is not None:
        seregression = float(seregression)
        if seregression <= 0:
            raise ValueError('SEREGRESSION must be a postive number.')
    else:
        seregression = 1.0

    # --- hold_back default (Ref: arma_forecaster.m:57-59) ---
    if hold_back is not None:
        hold_back = int(hold_back)

    # ==================================================================
    # Parameter augmentation for pure-AR or pure-MA cases
    # (Ref: arma_forecaster.m:163-175)
    #
    # If the AR or MA component is absent (all zeros or empty), a single
    # dummy lag with coefficient 0 is inserted.  This simplifies the
    # forward-recursion loop by ensuring both AR and MA vectors are always
    # non-empty.
    # ==================================================================

    # Ref: arma_forecaster.m:163-170 — insert dummy AR(1)=0 if no AR
    if p.size == 0 or np.all(p == 0):
        p = np.array([1.0])
        if constant == 1:
            # Ref: arma_forecaster.m:166
            # [const; 0; remaining_params]
            parameters = np.concatenate([
                parameters[0:1],
                np.array([0.0]),
                parameters[1:]
            ])
        else:
            # Ref: arma_forecaster.m:168 — original MATLAB edge-case logic
            parameters = np.concatenate([
                parameters[0:1],
                np.array([0.0]),
                parameters[0:]
            ])

    # Ref: arma_forecaster.m:172-175 — append dummy MA(1)=0 if no MA
    if q.size == 0 or np.all(q == 0):
        q = np.array([1.0])
        parameters = np.concatenate([parameters, np.array([0.0])])

    # Ref: arma_forecaster.m:177-178 — recompute max lags after augmentation
    maxp = int(np.max(p))
    maxq = int(np.max(q))

    # ==================================================================
    # Step 1: Produce the fit series and get estimated residuals
    # (Ref: arma_forecaster.m:181-209)
    # ==================================================================

    yorig = y.copy()  # Ref: arma_forecaster.m:181

    # Ref: arma_forecaster.m:185-186
    zeros_to_pad = max(maxq - maxp, 0)
    m_for_errors = max(maxp, maxq)

    # Ref: arma_forecaster.m:187-191 — adjust burn-in for holdBack
    if hold_back is not None:
        if hold_back > maxp:
            m_for_errors = m_for_errors + (hold_back - maxp)

    # Ref: arma_forecaster.m:193 — prepend zeros to align AR/MA lags
    y_aug = np.concatenate([np.zeros(zeros_to_pad), y])

    # Ref: arma_forecaster.m:195-196 — redefine counts after augmentation
    np_aug = len(p)
    nq_aug = len(q)

    # Ref: arma_forecaster.m:198 — compute errors using armaxerrors
    # MATLAB passes [] for x (no exogenous); we pass 1-D zeros (k → 0).
    # MATLAB passes ones(size(y)) for sigma (no GLS weighting).
    errors = armaxerrors(
        parameters, p, q, constant, y_aug,
        np.zeros(len(y_aug)),   # x placeholder (no exogenous regressors)
        m_for_errors,
        np.ones(len(y_aug))     # sigma = 1 (homoskedastic)
    )

    # --- Extract parameter sub-vectors ---
    # Ref: arma_forecaster.m:199-209
    if constant:
        constantp = parameters[0]
    else:
        constantp = 0.0

    # Ref: arma_forecaster.m:204-206 — AR parameters
    ar_params = parameters[constant:constant + np_aug]

    # Ref: arma_forecaster.m:208-209 — MA parameters
    ma_params = parameters[constant + np_aug:constant + np_aug + nq_aug]

    # ==================================================================
    # Step 2: h-step ahead forecast via forward recursion
    # (Ref: arma_forecaster.m:212-231)
    # ==================================================================

    yhattph = np.full(T, np.nan)

    # Ref: arma_forecaster.m:214 — reset m to maxq for forecast recursion
    m = maxq

    # Integer lag arrays for indexing inside the recursion
    p_int = p.astype(np.intp)
    q_int = q.astype(np.intp)

    # Ref: arma_forecaster.m:215-231
    # MATLAB: for t = r : T  (1-based inclusive loop)
    # We keep t_m as the 1-based MATLAB-equivalent variable and convert
    # to 0-based indices explicitly where needed.
    for t_m in range(r, T + 1):
        # Ref: arma_forecaster.m:216-217
        ytemp = np.zeros(T + m + h)
        etemp = np.zeros(T + m + h)

        # Ref: arma_forecaster.m:218-219
        # MATLAB: ytemp(m+1 : m+t) = y(1+zerosToPad : t+zerosToPad)
        # Python 0-based: ytemp[m : m+t_m] = y_aug[zeros_to_pad : t_m+zeros_to_pad]
        ytemp[m:m + t_m] = y_aug[zeros_to_pad:t_m + zeros_to_pad]
        etemp[m:m + t_m] = errors[zeros_to_pad:t_m + zeros_to_pad]

        # Ref: arma_forecaster.m:220-227 — inner h-step recursion
        for i in range(1, h + 1):
            # Ref: arma_forecaster.m:226
            # MATLAB 1-based index into ytemp: t_m + m + i
            # Python 0-based index: t_m + m + i - 1
            idx = t_m + m + i - 1

            # AR indices: MATLAB ytemp(t+m-p+i) → Python ytemp[t_m+m-p+i-1]
            ar_idx = t_m + m - p_int + i - 1
            # MA indices: MATLAB etemp(t+m-q+i) → Python etemp[t_m+m-q+i-1]
            ma_idx = t_m + m - q_int + i - 1

            ytemp[idx] = (constantp
                          + np.dot(ar_params, ytemp[ar_idx])
                          + np.dot(ma_params, etemp[ma_idx]))

        # Ref: arma_forecaster.m:230
        # MATLAB: yhattph(t) = ytemp(t + m + h)
        # Python 0-based: yhattph[t_m - 1] = ytemp[t_m + m + h - 1]
        yhattph[t_m - 1] = ytemp[t_m + m + h - 1]

    # ==================================================================
    # Step 3: Shift the y's to t+h alignment
    # (Ref: arma_forecaster.m:233)
    #
    # MATLAB: ytph = [NaN*ones(r-1,1); yorig(r+h:T); NaN*ones(h,1)]
    # First r-1 positions are NaN, then T-r-h+1 actual future values,
    # then final h positions are NaN (no data to compare forecasts).
    # ==================================================================

    ytph = np.full(T, np.nan)
    # Ref: arma_forecaster.m:233 — yorig(r+h : length(yorig))
    # Python 0-based: yorig[r+h-1 : T]
    src_start = r + h - 1  # 0-based index into yorig
    if src_start < T:
        n_vals = T - src_start
        ytph[r - 1:r - 1 + n_vals] = yorig[src_start:T]

    # ==================================================================
    # Step 4: Compute the forecast errors
    # (Ref: arma_forecaster.m:237)
    # ==================================================================

    # Ref: arma_forecaster.m:237 — code computes ytph - yhattph
    # (Note: MATLAB docstring incorrectly states YHATTPH-YTPH)
    forerr = ytph - yhattph

    # ==================================================================
    # Step 5: Compute the theoretical forecast standard deviation
    # (Ref: arma_forecaster.m:240-254)
    #
    # Uses the MA(∞) impulse response representation to compute the
    # variance of the h-step ahead forecast error, assuming constant
    # innovation variance σ² = seregression².
    # ==================================================================

    # Ref: arma_forecaster.m:240-241 — build full AR coefficient vector
    newar = np.zeros(max(h, maxp))
    # MATLAB 1-based: newar(p) = arparameters
    # Python 0-based: newar[p_int - 1] = ar_params
    newar[p_int - 1] = ar_params

    # Ref: arma_forecaster.m:242-243 — build full MA coefficient vector
    newma = np.zeros(max(h, maxq))
    newma[q_int - 1] = ma_params

    # Ref: arma_forecaster.m:245-246
    new_vec = np.concatenate([[1.0], newma])
    old = np.zeros(h)

    # Ref: arma_forecaster.m:248-253 — impulse response recursion
    # Computes the MA(∞) coefficients ψ_0, ψ_1, …, ψ_{h-1} from the
    # ARMA representation.  These are stored in reversed order in ``old``.
    for i in range(1, h + 1):
        # Ref: arma_forecaster.m:249 — MATLAB old(h-i+1) = new(i)
        # Python 0-based: old[h - i] = new_vec[i - 1]
        old[h - i] = new_vec[i - 1]
        for j in range(1, i):
            # Ref: arma_forecaster.m:251
            # MATLAB: old(h-i+1) = old(h-i+1) + newar(j)*old(h-i+1+j)
            # Python: old[h-i] += newar[j-1] * old[h-i+j]
            old[h - i] += newar[j - 1] * old[h - i + j]

    # Ref: arma_forecaster.m:254
    ystd = float(np.sqrt(np.sum(old ** 2)) * seregression)

    return yhattph, ytph, forerr, ystd
