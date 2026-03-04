"""
Pytest tests for mfe_toolbox.utility.mvstandardize — multivariate standardization.

Tests the ``mvstandardize`` function which transforms a T × K data matrix so
that each column has zero mean, unit variance, and columns are uncorrelated
(sample covariance of standardized data ≈ identity matrix).

Source: utility/mvstandardize.m (MFE Toolbox, Version 4.0)
Author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)

Parity tolerances: atol=1e-6, rtol=1e-4 per AAP Section 0.7.1.
"""

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.utility.mvstandardize import mvstandardize

# Import tolerance constants from conftest (auto-discovered by pytest).
# These are accessed via the conftest module namespace or directly.
# ATOL and RTOL are used explicitly in some comparisons for clarity.
from tests.conftest import ATOL, RTOL, load_fixture_npy, assert_allclose


# ---------------------------------------------------------------------------
# Test 1: Basic whitening — T×K data → whitened output
# ---------------------------------------------------------------------------
def test_mvstandardize_basic(multivariate_data: np.ndarray) -> None:
    """Verify mvstandardize returns a valid whitened array for T×K data.

    Uses the session-scoped multivariate_data fixture (T=1000, K=3).
    Checks that the result is a numpy ndarray with the same shape as input,
    finite values, and that the output covariance approximates the identity.
    """
    result = mvstandardize(multivariate_data)

    # Result must be a numpy ndarray
    assert isinstance(result, np.ndarray), "Result must be np.ndarray"

    # Shape must be preserved: T×K
    assert result.shape == multivariate_data.shape, (
        f"Expected shape {multivariate_data.shape}, got {result.shape}"
    )

    # All values must be finite (no NaN or inf)
    assert np.all(np.isfinite(result)), "Result contains non-finite values"

    # Output covariance should approximate identity
    K = multivariate_data.shape[1]
    result_cov = np.cov(result, rowvar=False)
    npt.assert_allclose(
        result_cov, np.eye(K), atol=0.1, rtol=0.1,
        err_msg="Whitened output covariance should approximate identity"
    )


# ---------------------------------------------------------------------------
# Test 2: Output covariance is identity
# ---------------------------------------------------------------------------
def test_mvstandardize_output_covariance_identity(
    multivariate_data: np.ndarray,
) -> None:
    """Verify cov(result) ≈ eye(K) for large T.

    With T=1000 and K=3, the sample covariance of the standardized data
    should be very close to the identity matrix.  Uses a slightly relaxed
    tolerance (atol=1e-2) because sample covariance converges at rate 1/√T.
    """
    result = mvstandardize(multivariate_data)
    K = multivariate_data.shape[1]

    result_cov = np.cov(result, rowvar=False)
    npt.assert_allclose(
        result_cov, np.eye(K), atol=1e-2, rtol=1e-2,
        err_msg="Sample covariance of whitened data should be near identity"
    )

    # Diagonal elements should be very close to 1
    npt.assert_allclose(
        np.diag(result_cov), np.ones(K), atol=1e-2, rtol=1e-2,
        err_msg="Diagonal of covariance should be near 1"
    )


# ---------------------------------------------------------------------------
# Test 3: Output has zero mean
# ---------------------------------------------------------------------------
def test_mvstandardize_output_zero_mean(
    multivariate_data: np.ndarray,
) -> None:
    """Verify column means ≈ 0 when demean_flag=True (default).

    The multivariate_data fixture is already demeaned, but mvstandardize
    applies demeaning by default.  After whitening, means should be
    numerically zero.
    """
    result = mvstandardize(multivariate_data)

    column_means = np.mean(result, axis=0)
    npt.assert_allclose(
        column_means, np.zeros(multivariate_data.shape[1]),
        atol=ATOL, rtol=RTOL,
        err_msg="Column means should be zero when demeaning is enabled"
    )


# ---------------------------------------------------------------------------
# Test 4: Custom sigma
# ---------------------------------------------------------------------------
def test_mvstandardize_custom_sigma(
    multivariate_data: np.ndarray,
) -> None:
    """Verify mvstandardize accepts and uses an explicit covariance matrix.

    Passes a known positive-definite sigma and confirms the function
    produces output different from the default (sample covariance) path.
    Also verifies the output is valid (finite, correct shape).
    """
    K = multivariate_data.shape[1]

    # Construct a known PD sigma (diagonal with different variances)
    custom_sigma = np.diag([2.0, 0.5, 1.5])

    result_custom = mvstandardize(multivariate_data, sigma=custom_sigma)
    result_default = mvstandardize(multivariate_data)

    # Shape must be preserved
    assert result_custom.shape == multivariate_data.shape

    # All values must be finite
    assert np.all(np.isfinite(result_custom))

    # Results with custom sigma should differ from default
    assert not np.allclose(result_custom, result_default), (
        "Custom sigma result should differ from default sigma result"
    )


# ---------------------------------------------------------------------------
# Test 5: No demeaning mode
# ---------------------------------------------------------------------------
def test_mvstandardize_no_demean(rng: np.random.Generator) -> None:
    """Verify demean_flag=False skips mean subtraction, only whitens.

    Generate data with non-zero mean, standardize with demean_flag=False,
    and confirm that column means of the output are NOT zero (unlike the
    demeaned case).
    """
    T, K = 500, 3
    # Generate data with non-zero mean
    x = rng.standard_normal((T, K)) + np.array([5.0, -3.0, 2.0])
    sigma = np.cov(x, rowvar=False)

    # With demeaning
    result_demean = mvstandardize(x, sigma=sigma, demean_flag=True)
    # Without demeaning
    result_no_demean = mvstandardize(x, sigma=sigma, demean_flag=False)

    # Demeaned result should have near-zero means
    npt.assert_allclose(
        np.mean(result_demean, axis=0), np.zeros(K), atol=1e-2,
        err_msg="Demeaned result should have zero means"
    )

    # Non-demeaned result should NOT have zero means (data has large means)
    no_demean_means = np.mean(result_no_demean, axis=0)
    assert np.max(np.abs(no_demean_means)) > 1.0, (
        "Non-demeaned result should retain non-zero column means"
    )

    # Results should differ
    assert not np.allclose(result_demean, result_no_demean), (
        "Demeaned and non-demeaned results should differ"
    )


# ---------------------------------------------------------------------------
# Test 6: 3-D time-varying sigma
# ---------------------------------------------------------------------------
def test_mvstandardize_3d_sigma(rng: np.random.Generator) -> None:
    """Verify mvstandardize handles K×K×T time-varying covariance.

    Constructs a 3-D sigma array where each slice is a positive definite
    matrix, and verifies the function produces valid output.
    """
    T, K = 50, 3
    x = rng.standard_normal((T, K))

    # Construct K×K×T array of PD matrices
    # Each slice is a scaled identity + small perturbation (ensuring PD)
    sigma_3d = np.zeros((K, K, T))
    for t in range(T):
        # Generate a random PD matrix via A @ A.T + I
        A = rng.standard_normal((K, K)) * 0.3
        sigma_3d[:, :, t] = A @ A.T + np.eye(K) * 2.0

    result = mvstandardize(x, sigma=sigma_3d, demean_flag=True)

    # Shape must be T×K
    assert result.shape == (T, K), (
        f"Expected shape ({T}, {K}), got {result.shape}"
    )

    # All values must be finite
    assert np.all(np.isfinite(result)), "3D sigma result contains non-finite values"


# ---------------------------------------------------------------------------
# Test 7: 2-D vs 3-D sigma equivalence
# ---------------------------------------------------------------------------
def test_mvstandardize_2d_vs_3d(rng: np.random.Generator) -> None:
    """Verify 2-D constant sigma and 3-D tiled sigma produce same result.

    When a 3-D sigma has identical slices (constant over time), the output
    should match the 2-D sigma case.
    """
    T, K = 100, 3
    x = rng.standard_normal((T, K))

    # Compute sample covariance as the constant sigma
    sigma_2d = np.cov(x, rowvar=False)

    # Tile into 3-D: K×K×T with identical slices
    sigma_3d = np.tile(sigma_2d[:, :, np.newaxis], (1, 1, T))
    assert sigma_3d.shape == (K, K, T), (
        f"3D sigma shape mismatch: {sigma_3d.shape}"
    )

    result_2d = mvstandardize(x, sigma=sigma_2d, demean_flag=True)
    result_3d = mvstandardize(x, sigma=sigma_3d, demean_flag=True)

    # Results should be numerically identical (same sigma at every time step)
    npt.assert_allclose(
        result_2d, result_3d, atol=ATOL, rtol=RTOL,
        err_msg="2-D and tiled 3-D sigma should produce identical output"
    )


# ---------------------------------------------------------------------------
# Test 8: Output shape preservation
# ---------------------------------------------------------------------------
def test_mvstandardize_output_shape(rng: np.random.Generator) -> None:
    """Verify T×K input produces T×K output for various shapes.

    Tests multiple (T, K) combinations to confirm shape is always preserved.
    """
    test_shapes = [(50, 2), (200, 5), (10, 3), (1000, 1)]

    for T, K in test_shapes:
        x = rng.standard_normal((T, K))

        # For K=1, np.cov returns a scalar; provide explicit sigma
        # Ref: numpy behavior — np.cov of single column returns 0-d array
        if K == 1:
            sigma = np.atleast_2d(np.cov(x, rowvar=False))
        else:
            sigma = None

        result = mvstandardize(x, sigma=sigma)

        assert result.shape == (T, K), (
            f"Shape mismatch for T={T}, K={K}: "
            f"expected ({T}, {K}), got {result.shape}"
        )
        assert isinstance(result, np.ndarray), (
            f"Result for T={T}, K={K} must be np.ndarray"
        )


# ---------------------------------------------------------------------------
# Test 9: Non-PSD sigma raises error
# ---------------------------------------------------------------------------
def test_mvstandardize_non_psd_sigma_raises(rng: np.random.Generator) -> None:
    """Verify non-positive-definite sigma raises ValueError.

    Constructs a matrix with a negative eigenvalue and passes it as sigma.
    The Cholesky-based or eigenvalue-based validation should reject it.
    Ref: mvstandardize.m:45 — min(eig(sigma))<=0 check
    """
    T, K = 100, 3
    x = rng.standard_normal((T, K))

    # Construct a non-PSD matrix: start with identity, add rank-1 negative
    # perturbation to make one eigenvalue negative
    bad_sigma = np.eye(K)
    bad_sigma[0, 0] = -1.0  # Force negative eigenvalue

    # Verify it's actually non-PSD
    eigenvalues = np.linalg.eigvalsh(bad_sigma)
    assert np.min(eigenvalues) <= 0, "Test matrix should be non-PSD"

    with pytest.raises(ValueError, match="positive definite"):
        mvstandardize(x, sigma=bad_sigma)


# ---------------------------------------------------------------------------
# Test 10: Single column (K=1)
# ---------------------------------------------------------------------------
def test_mvstandardize_single_column(rng: np.random.Generator) -> None:
    """Verify K=1 case produces scalar-like standardization.

    For a single column, mvstandardize should divide by the standard
    deviation (square root of variance), producing unit-variance output.
    An explicit 1×1 sigma is provided because np.cov returns a 0-d array
    for single-column input.
    """
    T = 200
    x = rng.standard_normal((T, 1))

    # Provide explicit 1×1 sigma to handle np.cov's 0-d return for K=1
    # Ref: mvstandardize.m:29 — MATLAB cov() returns 1×1 matrix for single col
    sigma_1x1 = np.atleast_2d(np.cov(x, rowvar=False))
    assert sigma_1x1.shape == (1, 1), "Sigma should be (1, 1) for K=1"

    result = mvstandardize(x, sigma=sigma_1x1, demean_flag=True)

    # Shape preserved
    assert result.shape == (T, 1), (
        f"Expected shape ({T}, 1), got {result.shape}"
    )

    # Output should have approximately zero mean
    npt.assert_allclose(
        np.mean(result, axis=0), np.zeros(1), atol=ATOL, rtol=RTOL,
        err_msg="Single column output mean should be near zero"
    )

    # Output variance should be approximately 1
    output_var = np.var(result, ddof=1)
    npt.assert_allclose(
        output_var, 1.0, atol=0.1, rtol=0.1,
        err_msg="Single column output variance should be near 1"
    )


# ---------------------------------------------------------------------------
# Test 11: MATLAB fixture parity
# ---------------------------------------------------------------------------
@pytest.mark.parity
def test_mvstandardize_fixture_parity(utility_fixture_dir) -> None:
    """Compare Python mvstandardize against MATLAB-generated reference fixtures.

    Loads the pre-computed MATLAB reference data from
    tests/fixtures/utility/mvstandardize.npy and verifies numerical parity
    for default (demean=True), no-demean (demean=False), and identity-sigma
    cases.

    Per AAP Section 0.7.1: atol=1e-6, rtol=1e-4.
    """
    # Load fixture — will pytest.skip if file not found
    fixture = load_fixture_npy(utility_fixture_dir, "mvstandardize")

    # The fixture is a 0-d object array wrapping a dict
    if fixture.dtype == object and fixture.ndim == 0:
        data = fixture.item()
    elif fixture.dtype == object:
        data = fixture.flat[0] if fixture.size > 0 else fixture.item()
    else:
        pytest.skip("Unexpected fixture format for mvstandardize")

    # Extract fixture fields
    input_X = data["input_X"]
    sigma_sample_cov = data["sigma_sample_cov"]
    expected_default = data["stddata_default_demean_true"]
    expected_no_demean = data["stddata_demean_false"]
    expected_sigma_identity = data["stddata_sigma_identity"]
    sigma_identity = data["sigma_identity"]
    expected_column_means = data["stddata_column_means"]
    expected_covariance = data["stddata_covariance"]

    # --- Test 11a: Default (demean=True, sample covariance) ---
    result_default = mvstandardize(input_X, sigma=sigma_sample_cov, demean_flag=True)
    assert_allclose(
        result_default, expected_default,
        err_msg="Default mvstandardize (demean=True) fixture parity failed"
    )

    # --- Test 11b: No demeaning (demean=False, sample covariance) ---
    result_no_demean = mvstandardize(
        input_X, sigma=sigma_sample_cov, demean_flag=False
    )
    assert_allclose(
        result_no_demean, expected_no_demean,
        err_msg="mvstandardize (demean=False) fixture parity failed"
    )

    # --- Test 11c: Identity sigma ---
    result_identity = mvstandardize(
        input_X, sigma=sigma_identity, demean_flag=True
    )
    assert_allclose(
        result_identity, expected_sigma_identity,
        err_msg="mvstandardize (sigma=identity) fixture parity failed"
    )

    # --- Test 11d: Output column means parity ---
    actual_column_means = np.mean(result_default, axis=0)
    assert_allclose(
        actual_column_means, expected_column_means,
        err_msg="Output column means fixture parity failed"
    )

    # --- Test 11e: Output covariance parity ---
    actual_covariance = np.cov(result_default, rowvar=False)
    assert_allclose(
        actual_covariance, expected_covariance,
        err_msg="Output covariance fixture parity failed"
    )
