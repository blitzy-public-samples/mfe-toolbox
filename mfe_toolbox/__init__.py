"""
MFE Toolbox - Financial Econometrics for Python.

A comprehensive library for financial econometrics, providing tools for:
- Univariate GARCH modeling (AGARCH, APARCH, EGARCH, FIGARCH, HEAVY, IGARCH, TARCH)
- Multivariate GARCH modeling (BEKK, CCC, DCC, GO-GARCH, Matrix, O-GARCH, RARCH, RCC, RiskMetrics, VT-VECH)
- Time series analysis (ARMAX, VAR, HAR, ADF, HP/BK filters, impulse response, Granger causality)
- Realized volatility estimation (19+ estimators with microstructure noise handling)
- Statistical distributions (GED, standardized-t, skewed-t, composite likelihood)
- Bootstrap methods (block bootstrap, stationary bootstrap, SPA/BSDS, MCS)
- Diagnostic tests (Berkowitz, Jarque-Bera, KS, Ljung-Box, LM)
- Cross-sectional analysis (OLS with robust SEs, PCA)
- Utility functions (HAC estimators, numerical derivatives, matrix parameterizations)

Migrated from MATLAB MFE Toolbox Version 4.0 (28-Oct-2009) by Kevin Sheppard.
"""

# Ref: Contents.m — "% Version 4.0 28-Oct-2009" → semantic version "4.0.0"
__version__ = "4.0.0"

__author__ = "Kevin Sheppard"

# Subpackage names exposed at the top level.
# Users import specific functions via e.g. `from mfe_toolbox.univariate.tarch import tarch`.
# No heavy imports are performed here to keep import time minimal.
__all__ = [
    "univariate",
    "multivariate",
    "timeseries",
    "realized",
    "distributions",
    "utility",
    "bootstrap",
    "tests",
    "crosssection",
    "sandbox",
    "gui",
]
