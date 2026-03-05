"""
DCC variance fitting via per-series TARCH estimation.

Fits TARCH/GJR-GARCH models to each column (series) of a multivariate
data matrix to produce conditional variance estimates used by the DCC
(Dynamic Conditional Correlation) and related multivariate GARCH
estimators.

For each of the *K* series the function calls
:func:`mfe_toolbox.univariate.tarch.tarch` with series-specific lag
orders (*p*, *o*, *q*) and model type, then aggregates the per-series
conditional variance vectors into a single T × K matrix and collects
the full estimation output into a list of dictionaries for later use
by the DCC likelihood and inference routines.

Migrated from: ``multivariate/dcc_fit_variance.m`` — MFE Toolbox Version 4.0
Original author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
Revision: 1    Date: 17/4/2012

Python migration: Blitzy Platform — MATLAB-to-Python 3.12 migration.

Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
"""

import numpy as np

from mfe_toolbox.univariate.tarch import tarch

__all__ = ['dcc_fit_variance']


def dcc_fit_variance(
    data: np.ndarray,
    p: np.ndarray,
    o: np.ndarray,
    q: np.ndarray,
    gjr_type: np.ndarray,
    starting_vals: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Fit per-series TARCH models for DCC and related estimators.

    Iterates over the *K* columns of *data*, fitting a TARCH(p_i, o_i, q_i)
    model to each series.  Returns the stacked conditional variances and a
    list of per-series estimation diagnostics required by the DCC driver.

    Parameters
    ----------
    data : np.ndarray
        T × K matrix of mean-zero residuals.  Each column is a separate
        time series.
    p : array_like
        K-element array of positive integers — number of symmetric
        innovation lags (ARCH terms) per series.
    o : array_like
        K-element array of non-negative integers — number of asymmetric
        innovation lags (threshold terms) per series.  Use 0 for
        symmetric processes.
    q : array_like
        K-element array of non-negative integers — number of lagged
        conditional variance terms (GARCH terms) per series.
    gjr_type : array_like
        K-element array of model-type codes per series:

        * 1 — model evolves in absolute values (AVGARCH / TARCH)
        * 2 — model evolves in squares (GJR-GARCH)  [DEFAULT in MATLAB]
    starting_vals : np.ndarray or None, optional
        Concatenated starting-value vector across all *K* series.  Length
        must equal ``sum(1 + p[i] + o[i] + q[i])`` for *i* = 0, …, K−1.
        If ``None``, each per-series TARCH call performs its own grid
        search for starting values.

    Returns
    -------
    H : np.ndarray
        T × K matrix of conditional variances, where ``H[:, i]`` holds
        the fitted conditional variance series for the *i*-th asset.
    univariate : list of dict
        Length-*K* list of dictionaries, one per series.  Each dictionary
        contains the following keys (matching the MATLAB ``univariate{i}``
        struct fields from ``dcc_fit_variance.m``):

        * ``'p'`` — symmetric lag order (int)
        * ``'o'`` — asymmetric lag order (int)
        * ``'q'`` — GARCH lag order (int)
        * ``'fdata'`` — augmented f(epsilon) array (np.ndarray)
        * ``'fIdata'`` — augmented f(epsilon)·I(epsilon<0) array (np.ndarray)
        * ``'back_cast'`` — backcast value for variance initialisation (float)
        * ``'m'`` — max(p, o, q) (int)
        * ``'T'`` — augmented data length (int)
        * ``'tarch_type'`` — model type code (int)
        * ``'parameters'`` — estimated parameter vector (np.ndarray)
        * ``'ht'`` — conditional variance series, length T (np.ndarray)
        * ``'A'`` — information matrix from robust VCV (np.ndarray)
        * ``'scores'`` — per-observation score matrix (np.ndarray)

    Raises
    ------
    ValueError
        If *data* is not 2-D, or if the lengths of *p*, *o*, *q*, and
        *gjr_type* do not match the number of columns in *data*, or if
        *starting_vals* has an incorrect total length.

    Notes
    -----
    * The function delegates all optimizer logic to
      :func:`mfe_toolbox.univariate.tarch.tarch`.  The optimizer
      selection (``scipy.optimize.minimize`` with ``method='L-BFGS-B'``)
      is handled inside ``tarch``.
    * In the original MATLAB code the positional argument order for
      ``tarch`` is ``(epsilon, p, o, q, error_type, tarch_type, …)``.
      The Python ``tarch`` function swaps the order to
      ``(epsilon, p, o, q, tarch_type, error_type, …)``.  This function
      uses keyword arguments to avoid ambiguity.

    See Also
    --------
    mfe_toolbox.univariate.tarch.tarch : TARCH/GJR-GARCH estimation driver.
    mfe_toolbox.multivariate.dcc.dcc : DCC/ADCC estimation driver.
    mfe_toolbox.multivariate.ccc_mvgarch.ccc_mvgarch : CCC-MVGARCH driver.

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.multivariate.dcc_fit_variance import dcc_fit_variance
    >>> rng = np.random.default_rng(42)
    >>> data = rng.standard_normal((500, 3)) * 0.01
    >>> p = np.array([1, 1, 1])
    >>> o = np.array([1, 1, 1])
    >>> q = np.array([1, 1, 1])
    >>> gjr = np.array([2, 2, 2])
    >>> H, uv = dcc_fit_variance(data, p, o, q, gjr)
    >>> H.shape
    (500, 3)
    >>> len(uv)
    3
    """
    # ------------------------------------------------------------------ #
    #  Input coercion and validation                                      #
    # ------------------------------------------------------------------ #
    # Ref: dcc_fit_variance.m:36 — [T,k] = size(data)
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        # Single series: reshape to column vector
        data = data.reshape(-1, 1)
    if data.ndim != 2:
        raise ValueError(
            "data must be a 2-D array with shape (T, K), "
            f"received ndim={data.ndim}"
        )

    # Coerce lag-order and model-type vectors to 1-D integer arrays
    p_arr = np.asarray(p, dtype=np.int64).ravel()
    o_arr = np.asarray(o, dtype=np.int64).ravel()
    q_arr = np.asarray(q, dtype=np.int64).ravel()
    gjr_arr = np.asarray(gjr_type, dtype=np.int64).ravel()

    T_obs, k = data.shape  # Ref: dcc_fit_variance.m:36

    # Validate that all specification vectors have length K
    for name, arr in [('p', p_arr), ('o', o_arr), ('q', q_arr),
                      ('gjr_type', gjr_arr)]:
        if arr.shape[0] != k:
            raise ValueError(
                f"Length of '{name}' ({arr.shape[0]}) must equal the number "
                f"of series K={k}"
            )

    # ------------------------------------------------------------------ #
    #  Starting-value handling                                            #
    #  Ref: dcc_fit_variance.m:32-34 — ensure column orientation          #
    # ------------------------------------------------------------------ #
    if starting_vals is not None:
        starting_vals = np.asarray(starting_vals, dtype=np.float64).ravel()
        # Ref: dcc_fit_variance.m:32-34 — transpose if row vector
        # In Python, .ravel() already produces a 1-D array.

        # Validate total length: sum of (1 + p[i] + o[i] + q[i]) for all i
        expected_len = int(np.sum(1 + p_arr + o_arr + q_arr))
        if starting_vals.shape[0] != expected_len:
            raise ValueError(
                f"starting_vals length ({starting_vals.shape[0]}) must equal "
                f"sum(1 + p[i] + o[i] + q[i]) = {expected_len}"
            )

    # ------------------------------------------------------------------ #
    #  Initialise output containers                                       #
    #  Ref: dcc_fit_variance.m:37-38                                      #
    # ------------------------------------------------------------------ #
    H = np.zeros((T_obs, k), dtype=np.float64)  # Ref: dcc_fit_variance.m:37
    univariate: list[dict | None] = [None] * k  # Ref: dcc_fit_variance.m:38

    # ------------------------------------------------------------------ #
    #  Optimizer options for tarch                                         #
    #  Ref: dcc_fit_variance.m:39-41                                      #
    #  MATLAB: univariteOptions = optimset('fminunc');                     #
    #          univariteOptions.Display = 'none';                          #
    #          univariteOptions.LargeScale = 'off';                        #
    #  Python: pass options dict — actual L-BFGS-B selection handled by   #
    #          tarch() internally.                                         #
    # ------------------------------------------------------------------ #
    options: dict = {'disp': False}

    # ------------------------------------------------------------------ #
    #  Starting-value offset tracker                                      #
    #  Ref: dcc_fit_variance.m:42 — offset = 0                            #
    # ------------------------------------------------------------------ #
    offset: int = 0

    # ------------------------------------------------------------------ #
    #  Per-series TARCH fitting loop                                       #
    #  Ref: dcc_fit_variance.m:44-68 — for i=1:k (MATLAB 1-based)        #
    #  Python: for i in range(k) (0-based)                                #
    # ------------------------------------------------------------------ #
    for i in range(k):
        # ---- Starting values for this series ---- #
        # Ref: dcc_fit_variance.m:45-51
        if starting_vals is not None:
            # Ref: dcc_fit_variance.m:46 — count = 1+p(i)+o(i)+q(i)
            count = int(1 + p_arr[i] + o_arr[i] + q_arr[i])
            # Ref: dcc_fit_variance.m:47 — volStartingVals = startingVals(offset + (1:count))
            # MATLAB 1-based → Python 0-based slicing
            vol_starting_vals = starting_vals[offset:offset + count]
            # Ref: dcc_fit_variance.m:48 — offset = offset + count
            offset += count
        else:
            # Ref: dcc_fit_variance.m:50 — volStartingVals = []
            vol_starting_vals = None

        # ---- Call TARCH estimation ---- #
        # Ref: dcc_fit_variance.m:52
        # MATLAB: tarch(data(:,i), p(i), o(i), q(i), [], gjrType(i),
        #               volStartingVals, univariteOptions)
        # MATLAB arg order: epsilon, p, o, q, error_type, tarch_type, startingvals, options
        # Python arg order: epsilon, p, o, q, tarch_type, error_type, startingvals, options
        # Use keyword arguments to avoid positional ambiguity.
        parameters, ll, ht, vcv_robust, vcv, scores, diagnostics = tarch(
            data[:, i],
            int(p_arr[i]),
            int(o_arr[i]),
            int(q_arr[i]),
            tarch_type=int(gjr_arr[i]),  # Ref: dcc_fit_variance.m:52 — gjrType(i)
            startingvals=vol_starting_vals,  # Ref: dcc_fit_variance.m:52 — volStartingVals
            options=options,  # Ref: dcc_fit_variance.m:52 — univariteOptions
        )
        # Note: error_type is omitted → defaults to 'NORMAL' (Ref: dcc_fit_variance.m:52 — [])

        # ---- Build per-series diagnostics dict ---- #
        # Ref: dcc_fit_variance.m:54-66 — univariate{i}.fieldName = value
        # MATLAB struct → Python dict, preserving exact key names
        univariate[i] = {
            'p': int(p_arr[i]),                        # Ref: dcc_fit_variance.m:54
            'o': int(o_arr[i]),                        # Ref: dcc_fit_variance.m:55
            'q': int(q_arr[i]),                        # Ref: dcc_fit_variance.m:56
            'fdata': diagnostics['fdata'],             # Ref: dcc_fit_variance.m:57
            'fIdata': diagnostics['fIdata'],           # Ref: dcc_fit_variance.m:58
            'back_cast': diagnostics['back_cast'],     # Ref: dcc_fit_variance.m:59
            'm': diagnostics['m'],                     # Ref: dcc_fit_variance.m:60
            'T': diagnostics['T'],                     # Ref: dcc_fit_variance.m:61
            'tarch_type': int(gjr_arr[i]),             # Ref: dcc_fit_variance.m:62
            'parameters': parameters,                  # Ref: dcc_fit_variance.m:63
            'ht': ht,                                  # Ref: dcc_fit_variance.m:64
            'A': diagnostics['A'],                     # Ref: dcc_fit_variance.m:65
            'scores': scores,                          # Ref: dcc_fit_variance.m:66
        }

        # ---- Store conditional variance in output matrix ---- #
        # Ref: dcc_fit_variance.m:67 — H(:,i) = ht
        H[:, i] = ht

    return H, univariate  # type: ignore[return-value]
