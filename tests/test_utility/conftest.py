"""Shared pytest fixtures for utility function tests.

This local conftest.py provides shared fixtures specific to the
``tests/test_utility/`` directory.  Every test file in this directory
automatically inherits these fixtures through pytest's conftest discovery
mechanism.

Relationship to ``tests/conftest.py``
--------------------------------------
The root ``tests/conftest.py`` provides session-scoped fixtures that this
module depends on via pytest's fixture injection:

- ``rng`` — ``numpy.random.default_rng(42)`` seeded generator
- ``utility_fixture_dir`` — path to ``tests/fixtures/utility/``
- ``univariate_data`` — T=1000 mean-zero simulated returns
- ``multivariate_data`` — T=1000, K=3 mean-zero simulated returns
- ``regression_data`` — T=200 regression ``(y, X)`` pair
- ``ATOL`` / ``RTOL`` — numerical tolerance constants

This local conftest adds utility-specific shared fixtures that would
otherwise be duplicated across the 29+ test files in this directory:

- ``util_fixture_dir`` — convenience alias for ``utility_fixture_dir``
- ``util_rng`` — convenience alias for ``rng``
- ``sample_pd_matrix`` — 4×4 symmetric positive-definite matrix
- ``sample_corr_matrix`` — 4×4 valid correlation matrix
- ``sample_cholesky`` — lower-triangular Cholesky factor of ``sample_pd_matrix``
- ``sample_vector_data`` — T=200 univariate array for data-processing tests
- ``sample_matrix_data`` — T=200, K=3 multivariate array for standardization tests
- ``ols_data`` — OLS regression data for HAC/robust VCV tests
- ``known_quadratic`` — quadratic test function with analytical gradient/Hessian
- ``date_reference_pairs`` — MATLAB datenum ↔ Python datetime mapping
- ``lag_spec`` — lag specification dict for ``newlagmatrix`` tests
- ``assert_symmetric`` — helper for matrix symmetry checks
- ``assert_positive_definite`` — helper for positive-definiteness checks
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Convenience Aliases
# ---------------------------------------------------------------------------

@pytest.fixture
def util_fixture_dir(utility_fixture_dir: Any):
    """Convenience alias for the root conftest ``utility_fixture_dir`` fixture.

    Returns the ``pathlib.Path`` pointing to ``tests/fixtures/utility/``.

    Parameters
    ----------
    utility_fixture_dir : Path
        Injected from root ``tests/conftest.py`` (session-scoped).

    Returns
    -------
    Path
        Directory containing utility fixture ``.npy`` and ``.csv`` files.
    """
    return utility_fixture_dir


@pytest.fixture
def util_rng(rng: np.random.Generator) -> np.random.Generator:
    """Convenience alias for the root conftest ``rng`` fixture.

    Returns the ``numpy.random.default_rng(42)`` seeded generator.

    Parameters
    ----------
    rng : np.random.Generator
        Injected from root ``tests/conftest.py`` (session-scoped, seed=42).

    Returns
    -------
    np.random.Generator
        Seeded random number generator.
    """
    return rng


# ---------------------------------------------------------------------------
# Matrix Parameterization / Covariance Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_pd_matrix() -> np.ndarray:
    """A deterministic 4×4 symmetric positive-definite (SPD) matrix.

    Constructed as ``A @ A.T + diag`` to guarantee positive-definiteness.
    Used by tests for ``vech``, ``ivech``, ``chol2vec``, ``vec2chol``,
    ``cov2corr``, ``corr_vech``, ``corr_ivech``, and ``robustvcv``.

    The matrix is intentionally small (4×4) so that hand-verification of
    ``vech``/``ivech`` outputs (10 elements for a 4×4 matrix) is feasible.

    Returns
    -------
    np.ndarray
        Shape ``(4, 4)`` symmetric positive-definite matrix.
    """
    # Ref: MATLAB convention — generate SPD matrix as A*A' + eps*I
    A = np.array([
        [2.0, 0.5, 0.3, 0.1],
        [0.5, 3.0, 0.7, 0.2],
        [0.3, 0.7, 1.5, 0.4],
        [0.1, 0.2, 0.4, 2.5],
    ])
    # Ensure perfect symmetry and positive-definiteness
    spd = A @ A.T + 0.1 * np.eye(4)
    return spd


@pytest.fixture
def sample_corr_matrix() -> np.ndarray:
    """A deterministic 4×4 valid correlation matrix.

    Constructed from ``sample_pd_matrix`` by normalizing to unit diagonal,
    guaranteeing all correlation matrix properties: symmetric, unit diagonal,
    positive-definite, and all elements in ``[-1, 1]``.

    Used by tests for ``corr_vech``, ``corr_ivech``, ``r2phi``, ``phi2r``,
    ``r2z``, ``z2r``, and ``cov2corr``.

    Returns
    -------
    np.ndarray
        Shape ``(4, 4)`` valid correlation matrix with ones on the diagonal.
    """
    # Build SPD matrix identically to sample_pd_matrix
    A = np.array([
        [2.0, 0.5, 0.3, 0.1],
        [0.5, 3.0, 0.7, 0.2],
        [0.3, 0.7, 1.5, 0.4],
        [0.1, 0.2, 0.4, 2.5],
    ])
    spd = A @ A.T + 0.1 * np.eye(4)
    # Normalize to correlation matrix: R[i,j] = C[i,j] / sqrt(C[i,i] * C[j,j])
    d = np.sqrt(np.diag(spd))
    corr = spd / np.outer(d, d)
    # Force exact unit diagonal (avoid floating-point drift)
    np.fill_diagonal(corr, 1.0)
    return corr


@pytest.fixture
def sample_cholesky(sample_pd_matrix: np.ndarray) -> np.ndarray:
    """Lower-triangular Cholesky factor of ``sample_pd_matrix``.

    ``numpy.linalg.cholesky`` returns the lower-triangular factor ``L`` such
    that ``L @ L.T == sample_pd_matrix``.  Note that MATLAB's ``chol``
    returns the upper-triangular factor, so Python translations must
    transpose appropriately.

    Used by tests for ``chol2vec`` and ``vec2chol``.

    Parameters
    ----------
    sample_pd_matrix : np.ndarray
        Injected 4×4 SPD matrix from this conftest.

    Returns
    -------
    np.ndarray
        Shape ``(4, 4)`` lower-triangular matrix.
    """
    return np.linalg.cholesky(sample_pd_matrix)


# ---------------------------------------------------------------------------
# Data-Processing Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_vector_data(rng: np.random.Generator) -> np.ndarray:
    """T=200 univariate data array for data-processing utility tests.

    Used by tests for ``demean``, ``standardize``, ``newlagmatrix``,
    and ``pltdens``.

    Parameters
    ----------
    rng : np.random.Generator
        Injected from root ``tests/conftest.py`` (session-scoped, seed=42).

    Returns
    -------
    np.ndarray
        Shape ``(200,)`` array with non-zero mean and unit-order variance.
    """
    # Add an offset so that demeaning produces a meaningful change
    return rng.standard_normal(200) + 1.5


@pytest.fixture
def sample_matrix_data(rng: np.random.Generator) -> np.ndarray:
    """T=200, K=3 multivariate data array for standardization tests.

    Used by tests for ``mvstandardize``, ``standardize``, and ``demean``
    (multivariate overloads).  Columns have distinct means and variances
    so that standardization produces non-trivial transformations.

    Parameters
    ----------
    rng : np.random.Generator
        Injected from root ``tests/conftest.py`` (session-scoped, seed=42).

    Returns
    -------
    np.ndarray
        Shape ``(200, 3)`` array with column-specific means and scales.
    """
    T, K = 200, 3
    raw = rng.standard_normal((T, K))
    # Apply distinct column means [1.0, -0.5, 2.0] and scales [0.5, 2.0, 1.5]
    means = np.array([1.0, -0.5, 2.0])
    scales = np.array([0.5, 2.0, 1.5])
    return raw * scales + means


# ---------------------------------------------------------------------------
# HAC / Robust VCV Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ols_data(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """OLS regression data for HAC and robust VCV tests.

    Provides a complete regression setup with regressors, true coefficients,
    response, OLS residuals, OLS coefficient estimates, and the
    ``(X'X)^{-1}`` bread matrix.  This avoids duplicating OLS computation
    logic across tests for ``covnw``, ``covvar``, and ``robustvcv``.

    Parameters
    ----------
    rng : np.random.Generator
        Injected from root ``tests/conftest.py`` (session-scoped, seed=42).

    Returns
    -------
    dict[str, np.ndarray]
        Keys:
        - ``"X"`` — shape ``(200, 3)`` regressor matrix
        - ``"y"`` — shape ``(200,)`` response vector
        - ``"beta_true"`` — shape ``(3,)`` true coefficients ``[0.5, -1.0, 0.3]``
        - ``"beta_hat"`` — shape ``(3,)`` OLS estimates ``(X'X)^{-1} X'y``
        - ``"residuals"`` — shape ``(200,)`` OLS residuals ``y - X @ beta_hat``
        - ``"XtX_inv"`` — shape ``(3, 3)`` bread matrix ``(X'X)^{-1}``
        - ``"T"`` — int, number of observations (200)
        - ``"K"`` — int, number of regressors (3)
    """
    T, K = 200, 3
    X = rng.standard_normal((T, K))
    beta_true = np.array([0.5, -1.0, 0.3])
    y = X @ beta_true + rng.standard_normal(T)
    # OLS estimation
    XtX_inv = np.linalg.inv(X.T @ X)
    beta_hat = XtX_inv @ (X.T @ y)
    residuals = y - X @ beta_hat
    return {
        "X": X,
        "y": y,
        "beta_true": beta_true,
        "beta_hat": beta_hat,
        "residuals": residuals,
        "XtX_inv": XtX_inv,
        "T": T,
        "K": K,
    }


# ---------------------------------------------------------------------------
# Numerical Derivatives Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def known_quadratic() -> dict[str, Any]:
    """Quadratic test function with known analytical gradient and Hessian.

    Provides a function ``f(x) = 0.5 * x' H x + g0' x + c`` together with
    the analytical gradient ``grad(x) = H x + g0`` and the constant Hessian
    ``H``.  This enables exact validation of ``gradient_2sided`` and
    ``hessian_2sided``/``hessian_2sided_nrows`` against closed-form results.

    The Hessian ``H`` is a 3×3 symmetric positive-definite matrix so that
    the quadratic is strictly convex and gradient/Hessian are well-defined
    everywhere.

    Returns
    -------
    dict[str, Any]
        Keys:
        - ``"func"`` — callable ``f(x) -> float`` implementing the quadratic
        - ``"grad"`` — callable ``grad(x) -> np.ndarray`` returning the
          analytical gradient at ``x``
        - ``"hess"`` — np.ndarray shape ``(3, 3)`` — the constant Hessian
        - ``"x0"`` — np.ndarray shape ``(3,)`` — a test evaluation point
        - ``"f_at_x0"`` — float — ``f(x0)``
        - ``"grad_at_x0"`` — np.ndarray shape ``(3,)`` — ``grad(x0)``
    """
    # H must be SPD for a proper convex quadratic
    H = np.array([
        [4.0, 1.0, 0.5],
        [1.0, 3.0, 0.7],
        [0.5, 0.7, 2.0],
    ])
    g0 = np.array([1.0, -0.5, 0.2])
    c = 3.0
    x0 = np.array([1.0, -1.0, 0.5])

    def func(x: np.ndarray) -> float:
        """Quadratic function: f(x) = 0.5 * x' H x + g0' x + c."""
        return float(0.5 * x @ H @ x + g0 @ x + c)

    def grad(x: np.ndarray) -> np.ndarray:
        """Analytical gradient: grad(x) = H x + g0."""
        return H @ x + g0

    f_at_x0 = func(x0)
    grad_at_x0 = grad(x0)

    return {
        "func": func,
        "grad": grad,
        "hess": H,
        "x0": x0,
        "f_at_x0": f_at_x0,
        "grad_at_x0": grad_at_x0,
    }


# ---------------------------------------------------------------------------
# Date Conversion Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def date_reference_pairs() -> list[dict[str, Any]]:
    """Reference MATLAB datenum ↔ Python datetime pairs for date conversion.

    Provides known-correct mappings between MATLAB serial date numbers and
    their equivalent Python datetime representations.  MATLAB's ``datenum``
    defines day 1 as January 1, 0000 AD, while Python's ``datetime`` epoch
    starts at ``datetime(1, 1, 1)`` (year 1).

    These reference pairs enable testing of ``c2mdate``, ``m2cdate``, and
    ``x2mdate`` against known values without requiring MATLAB or Octave.

    Returns
    -------
    list[dict[str, Any]]
        Each dict has keys:
        - ``"matlab_datenum"`` — float, MATLAB serial date number
        - ``"year"`` — int
        - ``"month"`` — int
        - ``"day"`` — int
        - ``"description"`` — str, human-readable label for the date
    """
    return [
        {
            "matlab_datenum": 1.0,
            "year": 0,
            "month": 1,
            "day": 1,
            "description": "MATLAB epoch: Jan 1, 0000",
        },
        {
            "matlab_datenum": 719529.0,
            "year": 1970,
            "month": 1,
            "day": 1,
            "description": "Unix epoch: Jan 1, 1970",
        },
        {
            "matlab_datenum": 730486.0,
            "year": 2000,
            "month": 1,
            "day": 1,
            "description": "Y2K: Jan 1, 2000",
        },
        {
            "matlab_datenum": 733773.0,
            "year": 2009,
            "month": 1,
            "day": 1,
            "description": "Jan 1, 2009 (near MFE Toolbox release)",
        },
    ]


# ---------------------------------------------------------------------------
# Lag Matrix Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def lag_spec(rng: np.random.Generator) -> dict[str, Any]:
    """Lag matrix specification for ``newlagmatrix`` tests.

    Provides a time series vector, a set of lag values, and the expected
    dimensions of the output lag matrix.  This avoids duplicating lag
    construction logic in individual test files.

    Parameters
    ----------
    rng : np.random.Generator
        Injected from root ``tests/conftest.py`` (session-scoped, seed=42).

    Returns
    -------
    dict[str, Any]
        Keys:
        - ``"y"`` — np.ndarray shape ``(100,)`` — input time series
        - ``"lags"`` — list[int] — lag orders ``[1, 2, 5]``
        - ``"max_lag"`` — int — maximum lag (5)
        - ``"expected_rows"`` — int — output rows (``100 - 5 = 95``)
        - ``"expected_cols"`` — int — output columns (3, one per lag)
    """
    y = rng.standard_normal(100)
    lags = [1, 2, 5]
    max_lag = max(lags)
    return {
        "y": y,
        "lags": lags,
        "max_lag": max_lag,
        "expected_rows": len(y) - max_lag,
        "expected_cols": len(lags),
    }


# ---------------------------------------------------------------------------
# Assertion Helper Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def assert_symmetric():
    """Helper fixture to check a matrix is symmetric within tolerance.

    Returns a callable ``_check(M, atol=1e-10)`` that validates:
    1. ``M`` is a 2-D square matrix.
    2. ``M`` is equal to its transpose within absolute tolerance ``atol``.

    Used across tests for ``vech``/``ivech`` round-trips, ``cov2corr``
    output validation, ``covnw`` output validation, and any other utility
    producing symmetric matrices.

    Returns
    -------
    callable
        A function ``_check(M: np.ndarray, atol: float = 1e-10) -> None``
        that raises ``AssertionError`` if ``M`` is not symmetric.
    """
    def _check(M: np.ndarray, atol: float = 1e-10) -> None:
        assert M.ndim == 2 and M.shape[0] == M.shape[1], (
            f"Matrix must be square 2-D, got shape {M.shape}"
        )
        np.testing.assert_allclose(
            M, M.T, atol=atol,
            err_msg="Matrix is not symmetric: M != M.T",
        )

    return _check


@pytest.fixture
def assert_positive_definite():
    """Helper fixture to check a matrix is positive definite.

    Returns a callable ``_check(M, atol=1e-10)`` that validates:
    1. ``M`` is a 2-D square matrix.
    2. All eigenvalues of ``M`` exceed ``-atol``.

    Uses ``numpy.linalg.eigvalsh`` which is numerically stable for
    symmetric/Hermitian matrices.

    Used across tests for ``cov2corr``, ``covnw``, ``robustvcv``,
    ``vec2chol`` round-trip validation, and other utilities producing
    covariance or correlation matrices.

    Returns
    -------
    callable
        A function ``_check(M: np.ndarray, atol: float = 1e-10) -> None``
        that raises ``AssertionError`` if ``M`` is not positive definite.
    """
    def _check(M: np.ndarray, atol: float = 1e-10) -> None:
        assert M.ndim == 2 and M.shape[0] == M.shape[1], (
            f"Matrix must be square 2-D, got shape {M.shape}"
        )
        eigs = np.linalg.eigvalsh(M)
        assert np.all(eigs > -atol), (
            f"Matrix is not positive definite; "
            f"min eigenvalue = {eigs.min():.2e}"
        )

    return _check
