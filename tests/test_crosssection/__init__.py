"""Pytest test package for mfe_toolbox.crosssection module.

Contains parity and unit tests for cross-sectional analysis functions:
- test_ols.py: OLS regression with White heteroskedasticity-robust standard errors
- test_pca.py: Principal Component Analysis with outer-product, covariance, and correlation modes

Tests verify numerical parity (±1e-6 atol, ±1e-4 rtol) against MATLAB reference
fixtures generated from crosssection/ols.m and crosssection/pca.m.
"""
