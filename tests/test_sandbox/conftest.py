"""Shared pytest fixtures for the ``test_sandbox`` subpackage.

This module provides convenience aliases and domain-specific fixtures used by
all test modules in ``tests/test_sandbox/``.  It builds on top of the
session-scoped fixtures defined in the root ``tests/conftest.py``, including:

* ``sandbox_fixture_dir`` — Path to ``tests/fixtures/sandbox/``
* ``rng`` — Seeded ``numpy.random.Generator`` (seed=42)
* ``ATOL`` / ``RTOL`` — Numerical tolerance constants (1e-6 / 1e-4)
* ``assert_allclose`` — Wrapper around ``numpy.testing.assert_allclose``

Sandbox-specific fixtures cover the six migrated modules:

* ``sarima`` — stub that raises ``NotImplementedError``
* ``sarma2arma`` — SARMA-to-ARMA polynomial conversion
* ``sdiff`` — seasonal differencing operator
* ``sarimax_errors`` — SARIMAX residual computation
* ``sarimax_likelihood`` — SARIMAX log-likelihood
* ``heavy_test`` — HEAVY model demonstration script
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Convenience aliases for root conftest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sb_fixture_dir(sandbox_fixture_dir):
    """Short alias for the sandbox fixture directory."""
    return sandbox_fixture_dir


@pytest.fixture
def sb_rng(rng):
    """Short alias for the seeded random number generator."""
    return rng


# ---------------------------------------------------------------------------
# SARMA / SARIMAX parameter fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sarma_params():
    """Representative SARMA parameter vector and specification.

    Returns a dict with keys:
        parameters : np.ndarray — combined AR + MA + seasonal parameter vector
        p : np.ndarray — AR lag indices
        q : np.ndarray — MA lag indices
        seasonal : np.ndarray — seasonal spec array [Sp, ?, Sq, S] per season
    """
    return {
        "parameters": np.array([0.5, -0.3, 0.4, 0.2, -0.1, 0.3]),
        "p": np.array([1, 2]),
        "q": np.array([1]),
        "seasonal": np.array([[1, 0, 1, 12]]),
    }


@pytest.fixture
def sarimax_data(rng):
    """Simulated SARIMAX estimation data.

    Returns a dict with keys:
        y : np.ndarray — (T,) dependent variable
        x : np.ndarray — (T, m) exogenous variable matrix (includes constant)
        sigma : np.ndarray — (T,) conditional standard deviations
        T : int — sample size
        m : int — number of exogenous columns
    """
    T = 200
    m = 2  # constant + one exogenous regressor
    x = np.column_stack([np.ones(T), rng.standard_normal(T)])
    y = 0.5 * x[:, 0] + 0.3 * x[:, 1] + rng.standard_normal(T) * 0.5
    sigma = np.full(T, 0.5)
    return {"y": y, "x": x, "sigma": sigma, "T": T, "m": m}


# ---------------------------------------------------------------------------
# Seasonal differencing fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sdiff_spec():
    """Representative seasonal differencing specification.

    Returns a dict with keys:
        x : np.ndarray — input time series
        d : np.ndarray — differencing orders
        S : np.ndarray — seasonal periods
    """
    rng_local = np.random.default_rng(42)
    return {
        "x": rng_local.standard_normal(200),
        "d": np.array([1, 1]),
        "S": np.array([1, 12]),
    }


# ---------------------------------------------------------------------------
# HEAVY model test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def heavy_test_params():
    """Default HEAVY model parameters matching sandbox/heavy_test.m.

    Returns a dict with keys:
        parameters : np.ndarray — [0.3, 0.05, 0.2, 0.4, 0.7, 0.55]
        p : np.ndarray — transition matrix [[0, 1], [0, 1]]
        q : np.ndarray — covariance identity matrix (2×2)
        m : np.ndarray — means [1, 390]
        T : int — sample size (1000)
    """
    return {
        "parameters": np.array([0.3, 0.05, 0.2, 0.4, 0.7, 0.55]),
        "p": np.array([[0, 1], [0, 1]]),
        "q": np.eye(2),
        "m": np.array([1.0, 390.0]),
        "T": 1000,
    }
