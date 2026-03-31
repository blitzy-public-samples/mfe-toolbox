"""
MFE Toolbox — Time Series Subpackage.

This subpackage provides a comprehensive suite of time series analysis tools
migrated from the MATLAB MFE Toolbox (Version 4.0). It includes:

- **ARMAX estimation**: Full ARMAX(p,q) model fitting with maximum likelihood,
  residual computation, simulation, and forecasting (armaxfilter, armaxerrors,
  armaxfilter_core, armaxfilter_likelihood, armaxfilter_simulate,
  arma_forecaster).
- **VAR models**: Vector autoregression estimation with heteroskedastic-robust
  and cross-equation covariance options (vectorar, vectorarvcv).
- **HAR model**: Heterogeneous autoregression for realized volatility modeling
  (heterogeneousar).
- **ADF unit root tests**: Augmented Dickey-Fuller tests with automatic lag
  selection and asymptotic critical values (augdf, augdfautolag, augdfcv,
  augdf_cvsim_tieup).
- **Filters**: Hodrick-Prescott trend-cycle decomposition and Baxter-King
  band-pass filter (hp_filter, bkfilter).
- **Impulse response**: VAR impulse response functions with bootstrap
  confidence bands (impulseresponse, impulseresponse_bootstrap).
- **Granger causality**: F-test based Granger causality testing (grangercause).
- **ACF / PACF**: Theoretical and sample autocorrelation and partial
  autocorrelation functions (acf, pacf, sacf, spacf).
- **Information criteria**: AIC, BIC, Hannan-Quinn, and Schwarz criterion
  computation (aicsbic, aichqcsbic).
- **Spectral analysis**: Frequency response from filter weight sequences
  (weights_to_frequency_response).
- **ARMA diagnostics**: Root analysis, MA root conversion, inverse AR roots
  (armaroots, convert_ma_roots, inverse_ar_roots).
- **OLS with HAC**: Newey-West heteroskedasticity and autocorrelation
  consistent OLS regression (olsnw).
- **Beveridge-Nelson decomposition**: Permanent/transitory decomposition
  of non-stationary time series (beveridgenelson).
- **Residual diagnostics**: Time series residual plotting utilities
  (tsresidualplot).

Usage
-----
Import specific functions from their respective modules::

    from mfe_toolbox.timeseries.acf import acf
    from mfe_toolbox.timeseries.armaxfilter import armaxfilter
    from mfe_toolbox.timeseries.vectorar import vectorar

This subpackage does **not** eagerly import all modules at init time to keep
package loading lightweight. Use explicit submodule imports as shown above.

Notes
-----
All functions preserve the parameter ordering and return array shapes of the
original MATLAB MFE Toolbox implementations. Numerical parity with MATLAB
outputs is maintained to within ±1e-6 (absolute tolerance).
"""

__all__ = [
    "acf",
    "aichqcsbic",
    "aicsbic",
    "arma_forecaster",
    "armaroots",
    "armaxerrors",
    "armaxfilter",
    "armaxfilter_core",
    "armaxfilter_likelihood",
    "armaxfilter_simulate",
    "augdf",
    "augdf_cvsim_tieup",
    "augdfautolag",
    "augdfcv",
    "beveridgenelson",
    "bkfilter",
    "convert_ma_roots",
    "grangercause",
    "heterogeneousar",
    "hp_filter",
    "impulseresponse",
    "impulseresponse_bootstrap",
    "inverse_ar_roots",
    "olsnw",
    "pacf",
    "sacf",
    "spacf",
    "tsresidualplot",
    "vectorar",
    "vectorarvcv",
    "weights_to_frequency_response",
]
