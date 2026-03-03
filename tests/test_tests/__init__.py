"""Parity tests for diagnostic statistical tests (mfe_toolbox.tests subpackage).

Contains pytest test modules verifying numerical parity (±1e-6 atol, ±1e-4 rtol)
against MATLAB reference fixtures for the 5 migrated diagnostic test functions:

- test_berkowitz.py — Berkowitz density forecast test (TS/CS modes)
- test_jarquebera.py — Jarque-Bera normality test
- test_kolmogorov.py — Kolmogorov-Smirnov distributional test
- test_ljungbox.py — Ljung-Box serial correlation test
- test_lmtest1.py — LM serial correlation test (heteroskedasticity-robust)

Fixture data is loaded from tests/fixtures/tests/ (.npy and .csv files).
"""
