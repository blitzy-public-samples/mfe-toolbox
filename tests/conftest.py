"""Shared pytest fixtures and configuration for MFE Toolbox test suite.

Provides:
- Numerical tolerance constants (ATOL=1e-6, RTOL=1e-4) for MATLAB parity testing
- Fixture file loading helpers (.npy/.csv from tests/fixtures/)
- MFE_FIXTURE_DIR environment variable support for CI override
- Common test data generators (random seeds, standardized return series)
- Parametrize helpers for common test patterns across all subpackages

Per AAP Section 0.5.1: This is the foundational test infrastructure file.
Per AAP Section 0.7.1: All migrated functions MUST pass
    numpy.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-4)
    against MATLAB-generated fixtures.
"""

import os
from pathlib import Path
from typing import Any

import numpy as np
import numpy.testing as npt
import pytest


# ---------------------------------------------------------------------------
# Numerical Parity Tolerance Constants
# ---------------------------------------------------------------------------
# Per AAP Section 0.7.1: numpy.testing.assert_allclose(actual, expected,
#     atol=1e-6, rtol=1e-4) is the standard for MATLAB reference comparison.
# These are HARD requirements — every migrated function must satisfy them.
ATOL: float = 1e-6
RTOL: float = 1e-4


# ---------------------------------------------------------------------------
# Fixture Directory Resolution
# ---------------------------------------------------------------------------
# Default fixture directory sits alongside this conftest.py as tests/fixtures/
_DEFAULT_FIXTURE_DIR: Path = Path(__file__).parent / "fixtures"


def get_fixture_dir() -> Path:
    """Return the fixture directory path, respecting MFE_FIXTURE_DIR env var.

    Per AAP Section 0.7.2: Optional ``MFE_FIXTURE_DIR`` environment variable
    override for fixture path during CI.  When the environment variable is set,
    its value is used as the fixture root; otherwise the default
    ``tests/fixtures/`` directory (relative to this file) is returned.

    Returns
    -------
    Path
        Resolved fixture directory path.
    """
    env_dir: str | None = os.environ.get("MFE_FIXTURE_DIR")
    if env_dir is not None and env_dir != "":
        return Path(env_dir)
    return _DEFAULT_FIXTURE_DIR


# ---------------------------------------------------------------------------
# Fixture Loading Helpers
# ---------------------------------------------------------------------------

def load_fixture_npy(fixture_dir: Path, name: str) -> np.ndarray:
    """Load a ``.npy`` fixture file from a subpackage fixture directory.

    If the fixture file does not exist the test is gracefully skipped via
    ``pytest.skip`` so that the test suite degrades cleanly when fixtures
    have not yet been generated.

    Parameters
    ----------
    fixture_dir : Path
        Path to the subpackage fixture directory
        (e.g. ``tests/fixtures/univariate``).
    name : str
        Base name of the fixture file **without** the ``.npy`` extension.

    Returns
    -------
    np.ndarray
        Array loaded from the fixture file.

    Raises
    ------
    pytest.skip
        When the fixture file is not found on disk.
    """
    path: Path = fixture_dir / f"{name}.npy"
    if not path.exists():
        pytest.skip(f"Fixture file not found: {path}")
    return np.load(path, allow_pickle=True)


def load_fixture_csv(fixture_dir: Path, name: str) -> np.ndarray:
    """Load a ``.csv`` fixture file as a numpy array.

    Comma-delimited values are read with ``numpy.loadtxt``.  If the fixture
    file does not exist the test is gracefully skipped.

    Parameters
    ----------
    fixture_dir : Path
        Path to the subpackage fixture directory
        (e.g. ``tests/fixtures/distributions``).
    name : str
        Base name of the fixture file **without** the ``.csv`` extension.

    Returns
    -------
    np.ndarray
        Array loaded from the CSV file.

    Raises
    ------
    pytest.skip
        When the fixture file is not found on disk.
    """
    path: Path = fixture_dir / f"{name}.csv"
    if not path.exists():
        pytest.skip(f"Fixture file not found: {path}")
    return np.loadtxt(path, delimiter=",")


# ---------------------------------------------------------------------------
# Numerical Parity Assertion Helper
# ---------------------------------------------------------------------------

def assert_allclose(
    actual: np.ndarray,
    expected: np.ndarray,
    atol: float = ATOL,
    rtol: float = RTOL,
    err_msg: str = "",
) -> None:
    """Assert numerical parity between actual and expected arrays.

    Thin wrapper around ``numpy.testing.assert_allclose`` with MFE Toolbox
    default tolerances (``atol=1e-6``, ``rtol=1e-4``) per AAP Section 0.7.1.

    Parameters
    ----------
    actual : np.ndarray
        Computed Python result.
    expected : np.ndarray
        MATLAB reference fixture value.
    atol : float, optional
        Absolute tolerance (default: ``1e-6``).
    rtol : float, optional
        Relative tolerance (default: ``1e-4``).
    err_msg : str, optional
        Optional error message prefix for diagnostics.
    """
    npt.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=err_msg)


# ---------------------------------------------------------------------------
# Pytest Configuration Hooks
# ---------------------------------------------------------------------------

def pytest_configure(config: Any) -> None:
    """Register custom markers for the MFE Toolbox test suite.

    Markers registered:

    * **slow** — marks tests as slow (deselect with ``-m "not slow"``).
    * **gui** — marks GUI tests requiring PyQt6 (deselect with
      ``-m "not gui"``).
    * **parity** — marks MATLAB parity tests requiring fixture files.
    * **requires_fixtures** — marks tests that require generated fixture
      files from ``scripts/generate_fixtures.m`` +
      ``scripts/convert_fixtures.py``.
    """
    config.addinivalue_line(
        "markers",
        'slow: marks tests as slow (deselect with \'-m "not slow"\')',
    )
    config.addinivalue_line(
        "markers",
        'gui: marks GUI tests requiring PyQt6 (deselect with \'-m "not gui"\')',
    )
    config.addinivalue_line(
        "markers",
        "parity: marks MATLAB parity tests requiring fixture files",
    )
    config.addinivalue_line(
        "markers",
        "requires_fixtures: marks tests that require generated fixture files",
    )


# ---------------------------------------------------------------------------
# Session-Scoped Pytest Fixtures — Fixture Directories
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    """Return the base fixture directory path.

    Delegates to :func:`get_fixture_dir` which honours the
    ``MFE_FIXTURE_DIR`` environment variable for CI overrides.
    """
    return get_fixture_dir()


@pytest.fixture(scope="session")
def univariate_fixture_dir(fixture_dir: Path) -> Path:
    """Return the ``univariate`` fixture subdirectory."""
    return fixture_dir / "univariate"


@pytest.fixture(scope="session")
def multivariate_fixture_dir(fixture_dir: Path) -> Path:
    """Return the ``multivariate`` fixture subdirectory."""
    return fixture_dir / "multivariate"


@pytest.fixture(scope="session")
def timeseries_fixture_dir(fixture_dir: Path) -> Path:
    """Return the ``timeseries`` fixture subdirectory."""
    return fixture_dir / "timeseries"


@pytest.fixture(scope="session")
def realized_fixture_dir(fixture_dir: Path) -> Path:
    """Return the ``realized`` fixture subdirectory."""
    return fixture_dir / "realized"


@pytest.fixture(scope="session")
def distributions_fixture_dir(fixture_dir: Path) -> Path:
    """Return the ``distributions`` fixture subdirectory."""
    return fixture_dir / "distributions"


@pytest.fixture(scope="session")
def utility_fixture_dir(fixture_dir: Path) -> Path:
    """Return the ``utility`` fixture subdirectory."""
    return fixture_dir / "utility"


@pytest.fixture(scope="session")
def bootstrap_fixture_dir(fixture_dir: Path) -> Path:
    """Return the ``bootstrap`` fixture subdirectory."""
    return fixture_dir / "bootstrap"


@pytest.fixture(scope="session")
def crosssection_fixture_dir(fixture_dir: Path) -> Path:
    """Return the ``crosssection`` fixture subdirectory."""
    return fixture_dir / "crosssection"


@pytest.fixture(scope="session")
def sandbox_fixture_dir(fixture_dir: Path) -> Path:
    """Return the ``sandbox`` fixture subdirectory."""
    return fixture_dir / "sandbox"


@pytest.fixture(scope="session")
def tests_fixture_dir(fixture_dir: Path) -> Path:
    """Return the ``tests`` (diagnostic tests) fixture subdirectory."""
    return fixture_dir / "tests"


# ---------------------------------------------------------------------------
# Session-Scoped Pytest Fixtures — Common Test Data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Return a seeded random number generator for reproducible tests.

    Uses seed **42** to match fixture generation
    (``scripts/generate_fixtures.m`` uses ``rng(42, 'twister')``).

    Returns
    -------
    np.random.Generator
        A NumPy random generator seeded with 42.
    """
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def univariate_data(rng: np.random.Generator) -> np.ndarray:
    """Generate a standard T=1000 mean-zero return series for univariate tests.

    Matches the ``epsilon`` variable from ``generate_fixtures.m``::

        T = 1000;
        epsilon = randn(T, 1);
        epsilon = epsilon - mean(epsilon);

    Returns
    -------
    np.ndarray
        Shape ``(1000,)`` mean-zero random series.
    """
    T: int = 1000
    data: np.ndarray = rng.standard_normal(T)
    data = data - data.mean()
    return data


@pytest.fixture(scope="session")
def multivariate_data(rng: np.random.Generator) -> np.ndarray:
    """Generate T=1000, K=3 multivariate return data.

    Matches ``data_mv`` from ``generate_fixtures.m``::

        data_mv = randn(T, K);
        data_mv = data_mv - repmat(mean(data_mv), T, 1);

    Returns
    -------
    np.ndarray
        Shape ``(1000, 3)`` mean-zero multivariate series.
    """
    T: int = 1000
    K: int = 3
    data: np.ndarray = rng.standard_normal((T, K))
    data = data - data.mean(axis=0)
    return data


@pytest.fixture(scope="session")
def regression_data(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Generate T=200 regression data ``(y, X)`` for cross-sectional tests.

    Matches ``generate_fixtures.m``::

        X_reg = randn(T_reg, 3);
        beta_true = [0.5; -1.0; 0.3];
        y_reg = X_reg * beta_true + randn(T_reg, 1);

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(y, X)`` where ``y`` has shape ``(200,)`` and ``X`` has shape
        ``(200, 3)``.
    """
    T: int = 200
    X: np.ndarray = rng.standard_normal((T, 3))
    beta_true: np.ndarray = np.array([0.5, -1.0, 0.3])
    y: np.ndarray = X @ beta_true + rng.standard_normal(T)
    return y, X
