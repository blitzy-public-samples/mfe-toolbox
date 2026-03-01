"""MFE Toolbox pytest test suite.

Provides comprehensive parity tests for the MATLAB-to-Python migration,
verifying numerical equivalence (±1e-6 atol, ±1e-4 rtol) against MATLAB
reference fixtures for all migrated modules.

Test subdirectories:
- test_univariate/ — GARCH model family tests
- test_multivariate/ — Multivariate GARCH model tests
- test_timeseries/ — Time series analysis tests
- test_realized/ — Realized volatility estimator tests
- test_distributions/ — Distribution function tests
- test_utility/ — Utility function tests
- test_bootstrap/ — Bootstrap method tests
- test_crosssection/ — Cross-sectional analysis tests
- test_sandbox/ — Sandbox module tests
- test_tests/ — Diagnostic test parity tests
- test_gui/ — PyQt6 GUI integration tests
"""
