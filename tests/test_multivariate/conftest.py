"""Shared pytest fixtures for multivariate GARCH model tests.

This local conftest.py provides shared fixtures specific to the
``tests/test_multivariate/`` directory.  Every test file in this directory
automatically inherits these fixtures through pytest's conftest discovery
mechanism.

Relationship to ``tests/conftest.py``
--------------------------------------
The root ``tests/conftest.py`` provides session-scoped fixtures that this
module depends on via pytest's fixture injection:

- ``multivariate_data`` — T=1000, K=3 zero-mean simulated returns
- ``rng`` — ``numpy.random.default_rng(42)`` seeded generator
- ``multivariate_fixture_dir`` — path to ``tests/fixtures/multivariate/``
- ``ATOL`` / ``RTOL`` — numerical tolerance constants

This local conftest adds multivariate-GARCH-specific shared fixtures that
would otherwise be duplicated across the 10+ test files in this directory:

- ``mv_data`` — convenience alias for ``multivariate_data``
- ``mv_data_asym`` — asymmetric (negative return indicator) data
- ``backcast_cov`` — EWMA backcast covariance matrix
- ``backcast_asym`` — EWMA asymmetric backcast matrix
- ``sample_cov`` — sample covariance matrix
- ``sample_corr`` — sample correlation matrix
- ``K`` / ``T`` — dimension and length constants
- ``assert_ht_positive_definite`` — helper for PD checks on Ht arrays
- ``assert_correlation_matrix`` — helper for correlation matrix validation
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Convenience Data Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mv_data(multivariate_data: np.ndarray) -> np.ndarray:
    """Convenience alias for the root conftest ``multivariate_data`` fixture.

    Returns T=1000, K=3 zero-mean simulated multivariate return data generated
    by ``numpy.random.default_rng(42).standard_normal((1000, 3))`` and demeaned
    along axis 0.

    Parameters
    ----------
    multivariate_data : np.ndarray
        Injected from root ``tests/conftest.py`` (session-scoped, T=1000, K=3).

    Returns
    -------
    np.ndarray
        Shape ``(1000, 3)`` mean-zero multivariate return series.
    """
    return multivariate_data


@pytest.fixture
def mv_data_asym(multivariate_data: np.ndarray) -> np.ndarray:
    """Asymmetric (negative return indicator) data for GJR/TARCH models.

    Computes ``data_asym[t, k] = data[t, k] * (data[t, k] < 0)`` which retains
    only negative return values and zeros out positive returns.  This follows the
    MATLAB convention ``data .* (data < 0)`` used across BEKK, DCC, RCC, CCC,
    and Scalar VT-VECH multivariate models when the asymmetric order ``o > 0``.

    Parameters
    ----------
    multivariate_data : np.ndarray
        Injected from root ``tests/conftest.py`` (session-scoped, T=1000, K=3).

    Returns
    -------
    np.ndarray
        Shape ``(1000, 3)`` array containing only negative return values (positive
        entries replaced with zero).
    """
    # Ref: bekk.m, dcc.m, scalar_vt_vech.m — MATLAB uses data .* (data < 0)
    return multivariate_data * (multivariate_data < 0)


# ---------------------------------------------------------------------------
# Backcast Covariance Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def backcast_cov(multivariate_data: np.ndarray) -> np.ndarray:
    """Exponentially weighted sample covariance backcast for multivariate models.

    Follows the MATLAB MFE Toolbox convention for EWMA backcasting used across
    BEKK, DCC, RCC, Scalar VT-VECH, and RARCH models for variance initialization.

    The computation uses an exponential weighting scheme with decay parameter
    ``lambda = 0.94`` (``tau = 1 - lambda = 0.06``) over the first
    ``ceil(sqrt(T))`` observations:

        weights[i] = tau * lambda^i,   i = 0, 1, ..., m-1
        weights = weights / sum(weights)
        backcast = sum_i  weights[i] * data[i] @ data[i].T

    Parameters
    ----------
    multivariate_data : np.ndarray
        Injected from root ``tests/conftest.py`` (session-scoped, T=1000, K=3).

    Returns
    -------
    np.ndarray
        Shape ``(K, K)`` = ``(3, 3)`` positive-definite backcast covariance
        matrix.
    """
    T, K = multivariate_data.shape
    # Ref: bekk.m, dcc.m — MATLAB uses ceil(sqrt(T)) initial observations
    m = int(np.ceil(np.sqrt(T)))
    # EWMA decay parameters matching MATLAB convention
    tau = 0.06
    lam = 0.94
    # Compute exponential weights
    n_obs = min(m, T)
    weights = tau * lam ** np.arange(n_obs)
    weights = weights / weights.sum()
    # Weighted sum of outer products
    backcast = np.zeros((K, K))
    for i in range(n_obs):
        backcast += weights[i] * np.outer(multivariate_data[i], multivariate_data[i])
    return backcast


@pytest.fixture
def backcast_asym(multivariate_data: np.ndarray) -> np.ndarray:
    """Exponentially weighted asymmetric backcast for models with leverage.

    Follows the same EWMA convention as :func:`backcast_cov` but operates on
    the negative-return outer products.  Used by BEKK, DCC, RCC, CCC, and
    Scalar VT-VECH models with asymmetric order ``o > 0``.

    The negative-return data is computed as ``data_neg = data * (data < 0)``
    following the MATLAB convention ``data .* (data < 0)``, and then the
    exponentially weighted outer product sum is formed identically to
    ``backcast_cov``.

    Parameters
    ----------
    multivariate_data : np.ndarray
        Injected from root ``tests/conftest.py`` (session-scoped, T=1000, K=3).

    Returns
    -------
    np.ndarray
        Shape ``(K, K)`` = ``(3, 3)`` asymmetric backcast covariance matrix
        (positive semi-definite).
    """
    # Ref: dcc.m, scalar_vt_vech.m — MATLAB uses data .* (data < 0) for asymmetric
    data_neg = multivariate_data * (multivariate_data < 0)
    T, K = data_neg.shape
    m = int(np.ceil(np.sqrt(T)))
    tau = 0.06
    lam = 0.94
    n_obs = min(m, T)
    weights = tau * lam ** np.arange(n_obs)
    weights = weights / weights.sum()
    backcast = np.zeros((K, K))
    for i in range(n_obs):
        backcast += weights[i] * np.outer(data_neg[i], data_neg[i])
    return backcast


# ---------------------------------------------------------------------------
# Sample Statistics Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_cov(multivariate_data: np.ndarray) -> np.ndarray:
    """Sample covariance matrix of multivariate data.

    Computes the unbiased sample covariance using ``np.cov(data, rowvar=False)``
    which divides by ``T-1`` (Bessel correction), consistent with MATLAB's
    ``cov(data)`` default behaviour.

    Used for variance targeting (Scalar VT-VECH), initial values (BEKK, RARCH),
    and validation checks across the multivariate test suite.

    Parameters
    ----------
    multivariate_data : np.ndarray
        Injected from root ``tests/conftest.py`` (session-scoped, T=1000, K=3).

    Returns
    -------
    np.ndarray
        Shape ``(K, K)`` = ``(3, 3)`` sample covariance matrix (symmetric,
        positive-definite for non-degenerate data).
    """
    return np.cov(multivariate_data, rowvar=False)


@pytest.fixture
def sample_corr(multivariate_data: np.ndarray) -> np.ndarray:
    """Sample correlation matrix of multivariate data.

    Computes the Pearson sample correlation matrix from the sample covariance
    by normalizing each entry by the product of the marginal standard
    deviations: ``R[i,j] = Cov[i,j] / (std[i] * std[j])``.

    Used for DCC/CCC/RCC correlation targeting and validation.

    Parameters
    ----------
    multivariate_data : np.ndarray
        Injected from root ``tests/conftest.py`` (session-scoped, T=1000, K=3).

    Returns
    -------
    np.ndarray
        Shape ``(K, K)`` = ``(3, 3)`` sample correlation matrix with diagonal
        elements equal to 1.0.
    """
    cov = np.cov(multivariate_data, rowvar=False)
    std = np.sqrt(np.diag(cov))
    return cov / np.outer(std, std)


# ---------------------------------------------------------------------------
# Dimension Constants
# ---------------------------------------------------------------------------

@pytest.fixture
def K() -> int:
    """Dimension of multivariate data (K=3 throughout test suite).

    This constant matches the ``K=3`` column count used by the root conftest's
    ``multivariate_data`` fixture and is consistent across all multivariate
    GARCH model tests.

    Returns
    -------
    int
        Always returns 3.
    """
    return 3


@pytest.fixture
def T() -> int:
    """Number of observations in test data (T=1000).

    This constant matches the ``T=1000`` row count used by the root conftest's
    ``multivariate_data`` fixture and is consistent across all multivariate
    GARCH model tests.

    Returns
    -------
    int
        Always returns 1000.
    """
    return 1000


# ---------------------------------------------------------------------------
# Assertion Helper Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def assert_ht_positive_definite():
    """Helper fixture to check all K×K slices of Ht are positive definite.

    Returns a callable ``_check(Ht)`` that validates positive-definiteness of
    conditional covariance matrices.  Supports both 3-D arrays (K×K×T stacked
    covariance matrices) and single 2-D matrices (K×K).

    The check uses ``numpy.linalg.eigvalsh`` which is numerically stable for
    symmetric/Hermitian matrices and verifies that every eigenvalue exceeds
    ``-1e-10`` (a small tolerance to account for floating-point rounding in
    near-singular cases).

    Returns
    -------
    callable
        A function ``_check(Ht: np.ndarray) -> None`` that raises
        ``AssertionError`` if any slice of ``Ht`` is not positive definite.

    Examples
    --------
    >>> assert_ht_positive_definite(Ht)  # Ht shape (3, 3, 1000)
    >>> assert_ht_positive_definite(single_cov)  # shape (3, 3)
    """
    def _check(Ht: np.ndarray) -> None:
        if Ht.ndim == 3:
            # Ht is K×K×T — validate each time slice
            K_dim, K2, T_dim = Ht.shape
            assert K_dim == K2, (
                f"Ht must be K×K×T with equal first two dimensions, got {Ht.shape}"
            )
            for t in range(T_dim):
                eigs = np.linalg.eigvalsh(Ht[:, :, t])
                assert np.all(eigs > -1e-10), (
                    f"Ht[:,:,{t}] is not positive definite; "
                    f"min eigenvalue = {eigs.min():.2e}"
                )
        elif Ht.ndim == 2:
            # Ht is a single K×K matrix
            eigs = np.linalg.eigvalsh(Ht)
            assert np.all(eigs > -1e-10), (
                f"Matrix is not positive definite; "
                f"min eigenvalue = {eigs.min():.2e}"
            )
        else:
            raise ValueError(
                f"Ht must be 2-D (K×K) or 3-D (K×K×T), got ndim={Ht.ndim}"
            )

    return _check


@pytest.fixture
def assert_correlation_matrix():
    """Helper fixture to check a matrix is a valid correlation matrix.

    Returns a callable ``_check(R, atol=1e-10)`` that validates the four
    defining properties of a correlation matrix:

    1. **Square**: R must be a 2-D square matrix.
    2. **Symmetric**: ``R == R.T`` within tolerance.
    3. **Unit diagonal**: ``diag(R) == 1`` within tolerance.
    4. **Positive definite**: All eigenvalues > ``-atol``.
    5. **Bounded**: All elements in ``[-1 - atol, 1 + atol]``.

    The ``atol`` parameter controls the tolerance for all checks and defaults
    to ``1e-10``, which is tighter than the parity tolerance (``1e-6``) because
    correlation matrix properties are structural invariants, not numerical
    approximation outcomes.

    Returns
    -------
    callable
        A function ``_check(R: np.ndarray, atol: float = 1e-10) -> None``
        that raises ``AssertionError`` if ``R`` is not a valid correlation
        matrix.

    Examples
    --------
    >>> assert_correlation_matrix(R)  # R shape (3, 3)
    >>> assert_correlation_matrix(R, atol=1e-6)  # relaxed tolerance
    """
    def _check(R: np.ndarray, atol: float = 1e-10) -> None:
        # Property 1: Square
        assert R.ndim == 2 and R.shape[0] == R.shape[1], (
            f"R must be a square 2-D matrix, got shape {R.shape}"
        )
        n = R.shape[0]

        # Property 2: Symmetric
        np.testing.assert_allclose(
            R, R.T, atol=atol,
            err_msg="Correlation matrix R is not symmetric"
        )

        # Property 3: Unit diagonal
        np.testing.assert_allclose(
            np.diag(R), np.ones(n), atol=atol,
            err_msg="Correlation matrix R does not have unit diagonal"
        )

        # Property 4: Positive definite
        eigs = np.linalg.eigvalsh(R)
        assert np.all(eigs > -atol), (
            f"Correlation matrix R is not positive definite; "
            f"min eigenvalue = {eigs.min():.2e}"
        )

        # Property 5: Bounded in [-1, 1]
        assert np.all(R >= -1.0 - atol) and np.all(R <= 1.0 + atol), (
            f"Correlation matrix R has elements outside [-1, 1]; "
            f"range = [{R.min():.6f}, {R.max():.6f}]"
        )

    return _check
