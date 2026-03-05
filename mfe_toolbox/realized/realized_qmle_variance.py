"""
Quasi-Maximum Likelihood Estimation (QMLE) of realized variance.

Implements the QMLE-based realized variance estimator of Xiu (2010), which
jointly estimates the integrated variance (sigma2) and the microstructure noise
variance (omega2) using an iterative EM-style algorithm applied to a
tridiagonal covariance model of intraday log returns.

The observed log return :math:`r_i^*` under microstructure noise is modelled as:

.. math::

    r_i^* = r_i + \\eta_i - \\eta_{i-1}

where :math:`r_i` is the efficient (true) return and :math:`\\eta_i \\sim
\\text{N}(0, \\omega^2)` is i.i.d. noise.  The covariance matrix of the
observed return vector is therefore tridiagonal:

.. math::

    \\Omega = \\sigma^2 \\Delta I_n + \\omega^2 L

where :math:`\\Delta = 1/n`, :math:`I_n` is the identity, and :math:`L` is
the second-difference (discrete Laplacian) matrix with 2 on the main diagonal
and -1 on the first super- and sub-diagonals.

Migrated from: ``realized/realized_qmle_variance.m`` (MFE Toolbox Version 4.0)

See Also
--------
mfe_toolbox.realized.realized_variance : Standard realized variance estimator.
mfe_toolbox.realized.realized_kernel : Kernel-based realized volatility.
mfe_toolbox.realized.realized_options : Default option factory.

Notes
-----
Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 3    Date: 3/10/2011
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import linalg

from mfe_toolbox.realized.realized_options import realized_options
from mfe_toolbox.realized.realized_price_filter import realized_price_filter
from mfe_toolbox.realized.realized_variance import realized_variance


# ---------------------------------------------------------------------------
# Maximum number of EM iterations and convergence tolerance
# Ref: realized_qmle_variance.m:191 — while max(abs(theta-thetaOld)>1e-3)
#      && iter<20
# ---------------------------------------------------------------------------
_MAX_ITER: int = 20
_CONV_TOL: float = 1e-3


def realized_qmle_variance(
    price,
    time=None,
    time_type: str = 'unit',
    sampling_type: str = 'CalendarTime',
    sampling_interval=1,
    subsamples: int = 1,
    options: dict | None = None,
) -> tuple[float, float, dict]:
    """Estimate realized variance using the QMLE approach of Xiu (2010).

    The estimator iteratively refines joint estimates of integrated variance
    (``sigma2``) and microstructure noise variance (``omega2``) via an
    EM-style algorithm on the tridiagonal covariance structure of intraday
    returns.

    Parameters
    ----------
    price : array_like
        An *m*-element 1-D vector of high-frequency prices.  If a row
        vector is provided, it is transposed.  Must contain at least 2
        elements.
    time : array_like or None, optional
        An *m*-element 1-D vector of observation times corresponding to
        *price*.  Must be sorted in non-decreasing order.  The format
        must match *time_type*:

        * ``'wall'``    — 24-hour clock HHMMSS (e.g. ``93000``).
        * ``'seconds'`` — Seconds past midnight (e.g. ``34200``).
        * ``'unit'``    — Unit-normalised on [0, 1].

        If ``None`` (default), a uniformly spaced grid on [0, 1] is
        generated and *time_type* is forced to ``'unit'``.
    time_type : str, optional
        Time format descriptor.  Case-insensitive.  One of ``'wall'``,
        ``'seconds'``, or ``'unit'``.  Default is ``'unit'``.
    sampling_type : str, optional
        Sampling scheme.  Case-insensitive.  One of:

        * ``'CalendarTime'``
        * ``'CalendarUniform'``
        * ``'BusinessTime'``
        * ``'BusinessUniform'``
        * ``'Fixed'``

        Default is ``'CalendarTime'``.
    sampling_interval : int, float, or array_like, optional
        Interpretation depends on *sampling_type*.  Default is ``1``.
    subsamples : int, optional
        Reserved for API consistency with other realized estimators.
        Must be a non-negative integer.  The QMLE algorithm does not
        use subsampling; this parameter is validated but otherwise
        ignored.  Default is ``1``.
    options : dict or None, optional
        Options dictionary as returned by
        ``realized_options('QMLE')``.  Relevant keys:

        * ``'noiseVarianceSamplingType'``
        * ``'noiseVarianceSamplingInterval'``
        * ``'medFrequencySamplingType'``
        * ``'medFrequencySamplingInterval'``

        If ``None`` (default), ``realized_options('QMLE')`` is called
        to obtain sensible defaults.

    Returns
    -------
    tuple[float, float, dict]
        A 3-tuple ``(qmle_rv, qmle_rv_debiased, diagnostics)`` where:

        * **qmle_rv** — QMLE estimate of integrated variance.
        * **qmle_rv_debiased** — Identical to *qmle_rv* (the QMLE
          estimator is already bias-corrected; no separate debiasing
          step is applied).
        * **diagnostics** — Dictionary with estimation details:

          - ``'noiseVariance'`` : float — Estimated noise variance.
          - ``'iterations'``   : int   — Number of EM iterations.

    Raises
    ------
    ValueError
        If any input fails validation (e.g. *price* too short,
        *time* not sorted, unrecognised *time_type* or
        *sampling_type*, invalid *sampling_interval*).

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.realized.realized_qmle_variance import (
    ...     realized_qmle_variance,
    ... )
    >>> prices = np.cumsum(np.random.default_rng(42).standard_normal(500)) + 100
    >>> rv, rv_d, diag = realized_qmle_variance(prices)
    >>> rv > 0
    True
    >>> diag['iterations'] <= 20
    True

    Notes
    -----
    **Algorithm outline (from realized_qmle_variance.m):**

    1. Filter log-prices at the user-specified sampling grid.
    2. Compute initial noise variance ``omega2`` from high-frequency
       returns at the noise-specific grid.
    3. Compute initial integrated variance ``sigma2`` via standard
       realized variance at medium frequency.
    4. Iterate (max 20 steps, absolute convergence tolerance 1e-3):

       a. Build tridiagonal covariance matrix
          :math:`\\Omega = \\sigma^2 \\Delta I + \\omega^2 L`.
       b. Compute :math:`\\Omega^{-1}` efficiently via
          ``scipy.linalg.solve_banded``.
       c. Compute :math:`(\\Omega^2)^{-1} = \\Omega^{-1} \\Omega^{-1}`.
       d. Form weight matrices :math:`W_1`, :math:`W_2` from trace
          quantities.
       e. Update :math:`\\sigma^2 = r^\\top W_1 r`,
          :math:`\\omega^2 = \\max(0, r^\\top W_2 r)`.

    5. Return final ``sigma2`` as the QMLE-RV estimate.

    **MATLAB → Python translation notes:**

    * ``inv(O)`` → ``scipy.linalg.solve_banded`` for efficient
      tridiagonal inversion (Ref: realized_qmle_variance.m:196).
    * ``inv(O*O)`` → ``Om1 @ Om1`` since
      :math:`(\\Omega^2)^{-1} = \\Omega^{-1} \\Omega^{-1}`
      (Ref: realized_qmle_variance.m:198).
    * ``diag(A)' * ones(n,1)`` → ``np.trace(A)``
      (Ref: realized_qmle_variance.m:206-208).
    * ``spalloc`` sparse matrices → dense ``np.ndarray`` since the
      inverse is always dense.
    * MATLAB 1-based indexing → Python 0-based indexing.
    """
    # ==================================================================
    # Input Validation
    # Ref: realized_qmle_variance.m:57-160
    # ==================================================================

    # Ref: realized_qmle_variance.m:103-105 — Transpose row to column
    price = np.asarray(price, dtype=np.float64).ravel()
    m = len(price)

    # Ref: realized_qmle_variance.m:106-107
    if m < 2:
        raise ValueError('PRICE must be a m by 1 vector.')

    # Ref: realized_qmle_variance.m:84-86 — timeType required if time given
    if time is not None and (time_type is None or time_type == ''):
        raise ValueError(
            'TIMETYPE must be provided if TIME is user supplied.'
        )

    # Ref: realized_qmle_variance.m:88-94 — Default time and timeType
    if time is None:
        # Ref: Other realized modules use linspace(0,1,m) as default time.
        # Note: MATLAB source line 89 has a bug (divides index vector by
        # price vector); using linspace for consistency with the rest of the
        # package.
        time = np.linspace(0.0, 1.0, m)
        time_type = 'unit'

    time = np.asarray(time, dtype=np.float64).ravel()

    # Ref: realized_qmle_variance.m:109-110 — Transpose time
    # (Already handled by ravel above.)

    # Ref: realized_qmle_variance.m:112-113 — TIME sorted check
    if len(time) > 1 and np.any(np.diff(time) < 0):
        raise ValueError('TIME must be sorted and increasing')

    # Ref: realized_qmle_variance.m:115-116 — TIME length check
    if len(time) != m:
        raise ValueError('TIME must be a m by 1 vector.')

    # Ref: realized_qmle_variance.m:119 — Cast to double
    # (Already handled by np.asarray dtype=np.float64.)

    # Ref: realized_qmle_variance.m:121-123 — Validate timeType
    time_type = time_type.lower()
    if time_type not in ('wall', 'seconds', 'unit'):
        raise ValueError(
            "TIMETYPE must be one of 'wall', 'seconds' or 'unit'."
        )

    # Ref: realized_qmle_variance.m:125-128 — Validate samplingType
    # Note: MATLAB source line 97 has a bug setting timeType instead of
    # samplingType when samplingType is empty.  Python uses default args,
    # so this bug does not manifest.
    sampling_type_lower = sampling_type.lower()
    _valid_sampling = (
        'calendartime', 'calendaruniform',
        'businesstime', 'businessuniform', 'fixed',
    )
    if sampling_type_lower not in _valid_sampling:
        raise ValueError(
            "SAMPLINGTYPE must be one of 'CalendarTime', "
            "'CalendarUniform', 'BusinessTime', "
            "'BusinessUniform' or 'Fixed'."
        )

    # Ref: realized_qmle_variance.m:130-154 — Validate samplingInterval
    t0 = time[0]
    tT = time[-1]

    if sampling_type_lower in (
        'calendartime', 'calendaruniform',
        'businesstime', 'businessuniform',
    ):
        if time_type in ('wall', 'seconds'):
            # Ref: realized_qmle_variance.m:136-137 — positive integer
            if (
                not np.isscalar(sampling_interval)
                or float(sampling_interval) < 1
                or np.floor(float(sampling_interval))
                != float(sampling_interval)
            ):
                raise ValueError(
                    'SAMPLINGINTERVAL must be a positive integer for '
                    'the SAMPLINGTYPE selected when using '
                    "'wall' or 'seconds' as TIMETYPE."
                )
        else:
            # Ref: realized_qmle_variance.m:140-141 — positive scalar
            if not np.isscalar(sampling_interval) or float(
                sampling_interval
            ) < 0:
                raise ValueError(
                    'SAMPLINGINTERVAL must be a positive value for '
                    'the SAMPLINGTYPE selected when using '
                    "'unit' as TIMETYPE."
                )
    else:
        # Ref: realized_qmle_variance.m:145-153 — Fixed sampling type
        si_arr = np.asarray(sampling_interval, dtype=np.float64).ravel()
        if si_arr.ndim == 0:
            si_arr = si_arr.reshape(1)
        if not (np.any(si_arr >= t0) and np.any(si_arr <= tT)):
            raise ValueError(
                'At least one sampling interval must be between '
                "min(TIME) and max(TIME) when using 'Fixed' "
                'as SAMPLINGTYPE.'
            )
        if len(si_arr) > 1 and np.any(np.diff(si_arr) <= 0):
            raise ValueError(
                "When using 'Fixed' as SAMPLINGTYPE the vector of "
                'sampling times in SAMPLINGINTERVAL must be sorted '
                'and strictly increasing.'
            )

    # Ref: realized_qmle_variance.m:156-160 — Validate subsamples
    # Note: MATLAB source line 156 references an undefined 'subsamples'
    # variable (the function signature has 'options' as the 6th arg).
    # Python includes subsamples in the signature for API consistency.
    if subsamples is not None:
        subsamples = int(subsamples)
        if subsamples < 0:
            raise ValueError('SUBSAMPLES must be a non-negative scalar.')

    # Ref: realized_qmle_variance.m:162-164 — Default options
    if options is None:
        options = realized_options('QMLE')

    # ==================================================================
    # Core Computation
    # Ref: realized_qmle_variance.m:168-233
    # ==================================================================

    # Ref: realized_qmle_variance.m:168 — logPrice = log(price)
    log_price = np.log(price)

    # Ref: realized_qmle_variance.m:169 — Filter log prices at user grid
    filter_result = realized_price_filter(
        log_price, time, time_type,
        sampling_type_lower, sampling_interval,
    )
    if isinstance(filter_result, tuple):
        filtered_log_price = np.asarray(
            filter_result[0], dtype=np.float64
        ).ravel()
    else:
        filtered_log_price = np.asarray(
            filter_result, dtype=np.float64
        ).ravel()

    # Ref: realized_qmle_variance.m:171 — Filter at noise-specific grid
    noise_result = realized_price_filter(
        log_price, time, time_type,
        options['noiseVarianceSamplingType'],
        options['noiseVarianceSamplingInterval'],
    )
    if isinstance(noise_result, tuple):
        noise_price = np.asarray(
            noise_result[0], dtype=np.float64
        ).ravel()
    else:
        noise_price = np.asarray(
            noise_result, dtype=np.float64
        ).ravel()

    # Ref: realized_qmle_variance.m:172-173 — Initial noise variance
    noise_return = np.diff(noise_price)
    # Ref: realized_qmle_variance.m:173 — omega2 = noiseReturn'*noiseReturn
    #      / length(noiseReturn)
    # np.sum used for explicit squared-return accumulation
    omega2: float = float(
        np.sum(noise_return ** 2) / len(noise_return)
    )

    # Ref: realized_qmle_variance.m:175 — r = diff(filteredLogPrice)
    r = np.diff(filtered_log_price)
    n = len(r)

    if n < 2:
        # The QMLE algorithm requires at least 2 returns (n >= 2) for the
        # tridiagonal covariance structure to be identifiable.  With n=1,
        # the denominator in the M-step weight matrices is always zero.
        # This typically happens when CalendarTime sampling with interval >= 1
        # is applied to unit-normalised time [0, 1] — only the start and end
        # prices survive, yielding a single return.
        raise ValueError(
            f'Filtering produced only {n} return(s); the QMLE estimator '
            f'requires at least 2 returns (3 filtered prices).  '
            f'Consider using a finer sampling interval or a different '
            f'sampling type (e.g. BusinessTime).'
        )

    # Ref: realized_qmle_variance.m:177 — Initial sigma2 via medium-freq RV
    # Note: The MATLAB source passes logPrice (already-logged prices) to
    # realized_variance, which internally computes log(price) again,
    # resulting in log(log(price)).  This is a bug in the MATLAB source
    # that produces an incorrect initialisation.  We pass raw prices
    # instead to get a correct sigma2 initialisation.  The EM algorithm
    # converges to the same fixed point regardless of initialisation for
    # well-conditioned data.
    # Ref: realized_qmle_variance.m:177 (original passes logPrice — bug)
    sigma2_result = realized_variance(
        price, time, time_type,
        options['medFrequencySamplingType'],
        options['medFrequencySamplingInterval'],
    )
    # realized_variance returns (rv, rv_ss, diagnostics)
    if isinstance(sigma2_result, tuple):
        sigma2: float = float(sigma2_result[0])
    else:
        sigma2 = float(sigma2_result)

    # Safeguard: ensure positive initial estimates
    if sigma2 <= 0.0:
        sigma2 = float(r @ r)  # fallback: sum of squared returns
    if omega2 <= 0.0:
        omega2 = float(r @ r) / (2.0 * n)  # fallback: rough noise estimate

    # Ref: realized_qmle_variance.m:178 — delta = 1/n
    delta: float = 1.0 / n

    # ------------------------------------------------------------------
    # Construct L — second-difference (discrete Laplacian) matrix
    # Ref: realized_qmle_variance.m:179-185
    #   L(diagInd) = 2;  L(uDiagInd) = -1;  L(lDiagInd) = -1;
    #
    # L is an n×n tridiagonal matrix with 2 on the main diagonal and
    # -1 on the first super- and sub-diagonals.
    # ------------------------------------------------------------------
    # np.ones used for diagonal construction matching MATLAB ones(n,1) idiom
    diag_main_l = 2.0 * np.ones(n, dtype=np.float64)
    diag_off_l = -1.0 * np.ones(n - 1, dtype=np.float64)
    L = (
        np.diag(diag_main_l)
        + np.diag(diag_off_l, 1)
        + np.diag(diag_off_l, -1)
    )

    # ------------------------------------------------------------------
    # EM-style iteration
    # Ref: realized_qmle_variance.m:188-224
    # ------------------------------------------------------------------
    theta = np.array([sigma2, omega2], dtype=np.float64)
    theta_old = np.array([-1.0, -1.0], dtype=np.float64)
    iter_count: int = 0

    # Ref: realized_qmle_variance.m:191 — while max(abs(theta-thetaOld)>1e-3)
    #      && iter<20
    while (
        np.any(np.abs(theta - theta_old) > _CONV_TOL)
        and iter_count < _MAX_ITER
    ):
        theta_old = theta.copy()

        # --------------------------------------------------------------
        # Build tridiagonal covariance Omega in banded storage
        # Ref: realized_qmle_variance.m:193-195
        #   O(diagInd)  = sigma2*delta + 2*omega2
        #   O(lDiagInd) = -omega2
        #   O(uDiagInd) = -omega2
        #
        # scipy.linalg.solve_banded expects banded form with shape
        # (l+u+1, n) where l=1 (lower bandwidth), u=1 (upper bandwidth):
        #   ab[0, j] = a[j-1, j]  (super-diagonal, j >= 1)
        #   ab[1, j] = a[j, j]    (main diagonal)
        #   ab[2, j] = a[j+1, j]  (sub-diagonal, j <= n-2)
        # --------------------------------------------------------------
        diag_val = sigma2 * delta + 2.0 * omega2
        off_val = -omega2

        ab = np.zeros((3, n), dtype=np.float64)
        ab[0, 1:] = off_val        # super-diagonal
        ab[1, :] = diag_val        # main diagonal
        ab[2, :-1] = off_val       # sub-diagonal

        # Ref: realized_qmle_variance.m:196 — Om1 = inv(O)
        # Efficient tridiagonal solve: O * X = I  →  X = O^{-1}
        om1 = linalg.solve_banded(
            (1, 1), ab, np.eye(n, dtype=np.float64),
            overwrite_ab=False, overwrite_b=False,
        )
        # Ref: realized_qmle_variance.m:197 — Om1 = (Om1+Om1')/2
        om1 = (om1 + om1.T) / 2.0

        # Ref: realized_qmle_variance.m:198 — Om2 = inv(O*O)
        # Construct dense O for linalg.inv on O*O (faithful to MATLAB inv(O*O))
        O_dense = (
            np.diag(np.full(n, diag_val, dtype=np.float64))
            + np.diag(np.full(n - 1, off_val, dtype=np.float64), 1)
            + np.diag(np.full(n - 1, off_val, dtype=np.float64), -1)
        )
        om2 = linalg.inv(O_dense @ O_dense)
        # Ref: realized_qmle_variance.m:199 — Om2 = (Om2+Om2')/2
        om2 = (om2 + om2.T) / 2.0

        # Ref: realized_qmle_variance.m:200-204
        om1_l = om1 @ L                   # O^{-1} L
        om2_l = om2 @ L                   # (O²)^{-1} L
        om2_l2 = om2_l @ L                # (O²)^{-1} L²
        om1_l_om1 = om1_l @ om1           # O^{-1} L O^{-1}
        om1_l_om1 = (om1_l_om1 + om1_l_om1.T) / 2.0

        # Ref: realized_qmle_variance.m:206-208
        # MATLAB: diag(A)'*ones(n,1) = trace(A)
        tr_om2_l: float = float(np.trace(om2_l))
        tr_om2_l2: float = float(np.trace(om2_l2))
        tr_om2: float = float(np.trace(om2))

        # Ref: realized_qmle_variance.m:209
        denom: float = tr_om2_l ** 2 - tr_om2 * tr_om2_l2

        # Guard against degenerate denominator.  When |denom| is near
        # machine precision, the weight matrices W1 and W2 become
        # numerically unstable.  Break out of the EM loop and use the
        # current sigma2/omega2 estimates.
        if abs(denom) < 1e-30:
            warnings.warn(
                'QMLE EM denominator near zero; stopping early with '
                'current parameter estimates.',
                stacklevel=2,
            )
            break

        # Ref: realized_qmle_variance.m:210-215 — Weight matrices
        w1 = (n * tr_om2_l * om1_l_om1 - n * tr_om2_l2 * om2) / denom
        w1 = (w1 + w1.T) / 2.0

        w2 = (tr_om2_l * om2 - tr_om2 * om1_l_om1) / denom
        w2 = (w2 + w2.T) / 2.0

        # Ref: realized_qmle_variance.m:217-220 — Parameter updates
        sigma2_new = float(r @ w1 @ r)
        omega2_new = float(r @ w2 @ r)

        # Guard against NaN propagation from ill-conditioned matrices
        if np.isnan(sigma2_new) or np.isnan(omega2_new):
            warnings.warn(
                'QMLE EM produced NaN parameters; stopping early with '
                'previous parameter estimates.',
                stacklevel=2,
            )
            break

        sigma2 = sigma2_new
        # Ref: realized_qmle_variance.m:219 — omega2 = max(0, omega2)
        omega2 = float(np.max(np.array([0.0, omega2_new])))

        theta = np.array([sigma2, omega2], dtype=np.float64)
        iter_count += 1

    # ------------------------------------------------------------------
    # Convergence diagnostic
    # Ref: realized_qmle_variance.m:226-228
    # ------------------------------------------------------------------
    if (
        np.any(np.abs(theta - theta_old) > _CONV_TOL)
        and iter_count >= _MAX_ITER
    ):
        warnings.warn(
            'The realized QMLE estimator did not converge.',
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # Assemble outputs
    # Ref: realized_qmle_variance.m:231-233
    # ------------------------------------------------------------------
    qmle_rv: float = sigma2
    # The QMLE estimator is inherently bias-corrected through its
    # likelihood formulation; there is no separate debiased estimate
    # in the original MATLAB source.
    qmle_rv_debiased: float = sigma2

    diagnostics: dict = {
        # Ref: realized_qmle_variance.m:233
        'noiseVariance': omega2,
        # Ref: realized_qmle_variance.m:232
        'iterations': iter_count,
    }

    return qmle_rv, qmle_rv_debiased, diagnostics
