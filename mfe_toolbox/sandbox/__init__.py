"""
Sandbox subpackage — prototype and experimental modules.

Contains experimental SARIMA/heavy-tail modeling utilities migrated from
the MATLAB MFE Toolbox sandbox/ directory. Some modules (sarima) remain
as stubs from the original MATLAB source.

Modules
-------
heavy_test : HEAVY model test/demonstration script
sarima : SARIMA entry point stub (not yet implemented)
sarimax_errors : SARIMAX residual computation
sarimax_likelihood : SARIMAX log-likelihood
sarma2arma : SARMA-to-ARMA polynomial conversion
sdiff : Seasonal differencing operator
"""

__all__ = [
    "heavy_test",
    "sarima",
    "sarimax_errors",
    "sarimax_likelihood",
    "sarma2arma",
    "sdiff",
]
