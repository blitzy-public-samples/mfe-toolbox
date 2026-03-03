"""Pytest tests for the sandbox (experimental) subpackage of mfe_toolbox.

Contains test modules verifying correct migration of prototype/experimental
MATLAB functions from the sandbox/ directory, including SARIMA stubs,
SARIMAX estimation components, seasonal differencing, and HEAVY model tests.

Modules:
    test_heavy_test: Tests for HEAVY model experimental test script
    test_sarima: Tests for SARIMA stub (verifies NotImplementedError)
    test_sarma2arma: Tests for SARMA-to-ARMA conversion
    test_sdiff: Tests for seasonal differencing
    test_sarimax_errors: Tests for SARIMAX residual computation
    test_sarimax_likelihood: Tests for SARIMAX log-likelihood
"""
