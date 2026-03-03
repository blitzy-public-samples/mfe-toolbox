"""
O-GARCH (Orthogonal GARCH) targeted univariate log-likelihood.

This module implements the per-factor log-likelihood function used during
Orthogonal GARCH estimation.  For each principal component (factor), a
standard GARCH(P,Q) model is fitted with the identifying constraint that
the intercept equals ``1 - sum(alpha_i) - sum(beta_j)``, ensuring the
unconditional variance of each factor is unity.

The function is consumed by :func:`mfe_toolbox.multivariate.o_mvgarch` which
iterates over all retained principal components, fitting each factor's
conditional variance independently before reconstructing the full
multivariate covariance matrix via :math:`H_t = W F_t W' + \\Omega`.

Migrated from ``multivariate/ogarch_likelihood.m`` (Kevin Sheppard,
MFE Toolbox v4.0, Revision 1, 4/15/2012).

Notes
-----
* Numerical parity target: ±1e-6 against MATLAB reference outputs.
* The MATLAB ``tarch_core_simple.m`` accepts raw data and transforms it
  internally, whereas the Python :func:`tarch_core_simple` (Numba JIT)
  expects pre-transformed ``fdata`` and ``fIdata`` arrays.  This module
  handles the required data transformation before calling the core
  recursion.

See Also
--------
mfe_toolbox.multivariate.o_mvgarch : O-GARCH driver (consumes this function).
mfe_toolbox.univariate.tarch_core_simple : Simplified TARCH variance recursion.
"""

import numpy as np

from mfe_toolbox.univariate.tarch_core_simple import tarch_core_simple


def ogarch_likelihood(
    parameters: np.ndarray,
    data: np.ndarray,
    p: int,
    q: int,
    gjr_type: int,
    back_cast: float,
) -> tuple[float, np.ndarray]:
    """Compute the targeted univariate normal log-likelihood for one factor
    in an O-GARCH model.

    The O-GARCH identifying assumption constrains the intercept (omega) to be
    ``1 - sum(parameters)`` so that the unconditional variance of each
    standardised principal component is unity.  Negative parameter values
    (including the implied intercept) are clamped to zero before the variance
    recursion.

    Parameters
    ----------
    parameters : np.ndarray
        ``(p + q,)`` vector of GARCH parameters laid out as
        ``[alpha_1 … alpha_p, beta_1 … beta_q]``.  The intercept (omega)
        is **not** included — it is inferred as ``1 - sum(parameters)``.
    data : np.ndarray
        ``(T,)`` array of zero-mean residuals (a single principal component /
        factor).
    p : int
        Positive integer — number of symmetric innovation lags (ARCH order).
    q : int
        Non-negative integer — number of conditional-variance lags
        (GARCH order).
    gjr_type : int
        Variance model type passed through to :func:`tarch_core_simple`:

        * 1 — AVGARCH: model evolves in absolute values.
        * 2 — Standard GARCH / ARCH: model evolves in squared values.
    back_cast : float
        Back-cast value used to initialise the variance recursion for
        pre-sample lags.

    Returns
    -------
    ll : float
        Total (summed) negative log-likelihood evaluated at *parameters*.
    lls : np.ndarray
        ``(T,)`` array of per-observation negative log-likelihoods.

    Notes
    -----
    The O-GARCH identifying assumption is:

    .. math::

        \\omega = 1 - \\sum_{i=1}^{p} \\alpha_i - \\sum_{j=1}^{q} \\beta_j

    This ensures the unconditional variance of the factor equals one, which
    is required because the factors are standardised principal components.

    Negative implied parameters are clamped to zero (Ref: ogarch_likelihood.m:26)
    to maintain positive semi-definiteness of the conditional variance.

    The per-observation negative log-likelihood under Gaussianity is:

    .. math::

        \\ell_t = \\frac{1}{2}\\left(\\log(2\\pi) + \\log(h_t) +
                  \\frac{r_t^2}{h_t}\\right)

    References
    ----------
    * Kevin Sheppard, ``multivariate/ogarch_likelihood.m``, MFE Toolbox v4.0.
    * Ref lines: ogarch_likelihood.m:25–29.

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.multivariate.ogarch_likelihood import ogarch_likelihood
    >>> rng = np.random.default_rng(42)
    >>> data = rng.standard_normal(500)
    >>> params = np.array([0.05, 0.90])  # alpha=0.05, beta=0.90, omega=0.05
    >>> ll, lls = ogarch_likelihood(params, data, p=1, q=1, gjr_type=2, back_cast=1.0)
    >>> ll > 0
    True
    >>> lls.shape
    (500,)
    """
    # Ensure data is a contiguous 1-D float64 array for Numba compatibility.
    data = np.asarray(data, dtype=np.float64).ravel()
    parameters = np.asarray(parameters, dtype=np.float64).ravel()

    # ------------------------------------------------------------------
    # Step 1: Construct the full parameter vector with the inferred intercept.
    # Ref: ogarch_likelihood.m:25 — volParameters = [1-sum(parameters) parameters]
    # The O-GARCH identifying assumption: omega = 1 - sum(alpha_i + beta_j).
    # ------------------------------------------------------------------
    intercept = 1.0 - np.sum(parameters)
    vol_parameters = np.concatenate((np.array([intercept]), parameters))

    # ------------------------------------------------------------------
    # Step 2: Clamp negative parameters to zero.
    # Ref: ogarch_likelihood.m:26 — volParameters(volParameters<0) = 0
    # This prevents negative contributions to the conditional variance,
    # ensuring h(t) remains non-negative throughout the recursion.
    # ------------------------------------------------------------------
    vol_parameters = np.maximum(vol_parameters, 0.0)

    # ------------------------------------------------------------------
    # Step 3: Pre-transform data for the Numba JIT tarch_core_simple.
    # The MATLAB tarch_core_simple.m performs this transformation internally
    # (lines 45-50), but the Python Numba version expects pre-transformed
    # fdata and fIdata arrays for nopython mode compatibility.
    # Ref: tarch_core_simple.m:45-50
    # ------------------------------------------------------------------
    if gjr_type == 1:
        # AVGARCH: model evolves in absolute values
        fdata = np.abs(data)
    else:
        # Standard GARCH/ARCH: model evolves in squares
        fdata = data ** 2

    # Asymmetric indicator-weighted data: fdata * I(data < 0)
    # Ref: tarch_core_simple.m:50 — fIdata = fdata.*(data<0)
    # For O-GARCH, o=0 so fIdata is not actually used in the recursion,
    # but the function signature requires it.
    fIdata = fdata * (data < 0).astype(np.float64)

    T = data.shape[0]
    o = 0  # O-GARCH uses no asymmetric terms — Ref: ogarch_likelihood.m:27
    m = max(p, o, q)  # Maximum lag order for interface compatibility

    # ------------------------------------------------------------------
    # Step 4: Compute conditional variances via the TARCH recursion.
    # Ref: ogarch_likelihood.m:27 —
    #   v = tarch_core_simple(data, volParameters, backCast, 0, p, 0, q, gjrType)
    # The Python Numba version has a different signature that requires
    # pre-transformed data (fdata, fIdata) and explicit m, T parameters.
    # ------------------------------------------------------------------
    v = tarch_core_simple(
        fdata, fIdata, vol_parameters, back_cast, p, o, q, m, T, gjr_type
    )

    # ------------------------------------------------------------------
    # Step 5: Compute per-observation negative log-likelihood.
    # Ref: ogarch_likelihood.m:28 —
    #   lls = 0.5 * (log(2*pi) + log(v) + data.^2./v)
    # Standard univariate normal negative log-likelihood (up to constant).
    # ------------------------------------------------------------------
    lls = 0.5 * (np.log(2.0 * np.pi) + np.log(v) + data ** 2 / v)

    # ------------------------------------------------------------------
    # Step 6: Aggregate total negative log-likelihood.
    # Ref: ogarch_likelihood.m:29 — ll = sum(lls)
    # ------------------------------------------------------------------
    ll = float(np.sum(lls))

    return ll, lls
