"""
Pytest tests for mfe_toolbox.timeseries.vectorarvcv — VAR parameter VCV estimation.

Tests the vectorarvcv function which computes parameter variance-covariance matrices
for Vector Autoregression (VAR) models under 4 error structure assumptions:
    - (het=0, uncorr=0): Homoskedastic, correlated (standard SUR/OLS)
    - (het=0, uncorr=1): Homoskedastic, uncorrelated (equation-by-equation OLS)
    - (het=1, uncorr=0): Heteroskedastic, correlated (White-type sandwich)
    - (het=1, uncorr=1): Heteroskedastic, uncorrelated (block-diagonal sandwich)

Reference: timeseries/vectorarvcv.m — Kevin Sheppard, MFE Toolbox v4.0
"""
import os

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.timeseries.vectorarvcv import vectorarvcv

# Numerical parity constants per AAP Section 0.7.1
ATOL = 1e-6
RTOL = 1e-4


@pytest.fixture
def vcv_inputs(rng):
    """Generate synthetic VAR regression data for testing vectorarvcv.

    Creates T=200 observations with K=2 equations and Np=K*P+1=7 regressors
    (P=3 lags per variable plus a constant column), using the session-scoped
    reproducible RNG (seed=42) from conftest.py.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        X : np.ndarray of shape (200, 7) — regressor matrix
        errors : np.ndarray of shape (200, 2) — residual matrix
    """
    T = 200
    K = 2
    P = 3
    Np = K * P + 1  # 7 regressors: 3 lags x 2 variables + 1 constant
    X = rng.standard_normal((T, Np))
    errors = rng.standard_normal((T, K))
    return X, errors


# =========================================================================
# Phase 3 — Output Structure Tests (5 tests)
# =========================================================================


def test_vectorarvcv_returns_matrix(vcv_inputs):
    """vectorarvcv must return a 2-dimensional numpy ndarray for all modes."""
    X, errors = vcv_inputs
    for het in (0, 1):
        for uncorr in (0, 1):
            vcv = vectorarvcv(X, errors, het=het, uncorr=uncorr)
            assert isinstance(vcv, np.ndarray), (
                f"Result must be an ndarray for het={het}, uncorr={uncorr}, "
                f"got {type(vcv).__name__}"
            )
            assert vcv.ndim == 2, (
                f"Result must be 2-dimensional for het={het}, uncorr={uncorr}, "
                f"got ndim={vcv.ndim}"
            )


def test_vectorarvcv_shape(vcv_inputs):
    """VCV shape must be (K*Np, K*Np) where K=errors.shape[1], Np=X.shape[1].

    For the fixture with K=2, Np=7 the expected shape is (14, 14).
    """
    X, errors = vcv_inputs
    K = errors.shape[1]
    Np = X.shape[1]
    expected_size = K * Np
    for het in (0, 1):
        for uncorr in (0, 1):
            vcv = vectorarvcv(X, errors, het=het, uncorr=uncorr)
            assert vcv.shape == (expected_size, expected_size), (
                f"Shape mismatch for het={het}, uncorr={uncorr}: "
                f"{vcv.shape} != {(expected_size, expected_size)}"
            )


def test_vectorarvcv_symmetric(vcv_inputs):
    """VCV matrix must be symmetric (VCV == VCV.T) for all 4 modes."""
    X, errors = vcv_inputs
    for het in (0, 1):
        for uncorr in (0, 1):
            vcv = vectorarvcv(X, errors, het=het, uncorr=uncorr)
            npt.assert_allclose(
                vcv, vcv.T, atol=ATOL, rtol=RTOL,
                err_msg=f"VCV not symmetric for het={het}, uncorr={uncorr}"
            )


def test_vectorarvcv_psd(vcv_inputs):
    """VCV matrix must be positive semi-definite (all eigenvalues >= 0).

    Uses np.linalg.eigvalsh for symmetric matrix eigenvalue computation.
    A small negative tolerance (-ATOL) is allowed for floating-point noise.
    """
    X, errors = vcv_inputs
    for het in (0, 1):
        for uncorr in (0, 1):
            vcv = vectorarvcv(X, errors, het=het, uncorr=uncorr)
            eigenvalues = np.linalg.eigvalsh(vcv)
            assert np.all(eigenvalues >= -ATOL), (
                f"Negative eigenvalue for het={het}, uncorr={uncorr}: "
                f"min eigenvalue = {eigenvalues.min():.2e}"
            )


def test_vectorarvcv_diagonal_positive(vcv_inputs):
    """All diagonal elements (parameter variances) must be strictly positive."""
    X, errors = vcv_inputs
    for het in (0, 1):
        for uncorr in (0, 1):
            vcv = vectorarvcv(X, errors, het=het, uncorr=uncorr)
            diag_vals = np.diag(vcv)
            assert np.all(diag_vals > 0), (
                f"Non-positive diagonal for het={het}, uncorr={uncorr}: "
                f"min diagonal = {diag_vals.min():.2e}"
            )


# =========================================================================
# Phase 4 — Four Mode Tests with Manual Computation (4 tests)
# =========================================================================


def test_vectorarvcv_het0_uncorr0(vcv_inputs):
    """Homoskedastic, correlated: VCV = kron(s2, XpXi) / T.

    This is the standard SUR-OLS VCV under homoskedasticity and cross-equation
    correlation (Zellner SUR result when all equations share X).
    Ref: vectorarvcv.m lines 59-61 — VCV = kron(s2, XpXi) / T
    """
    X, errors = vcv_inputs
    T = X.shape[0]

    # Manual computation of common quantities
    # Ref: vectorarvcv.m:37 — s2 = errors' * errors / T
    s2 = errors.T @ errors / T
    # Ref: vectorarvcv.m:39 — XpXi = ((X'*X)/T)^(-1)
    XpXi = np.linalg.inv((X.T @ X) / T)

    # Ref: vectorarvcv.m:61 — VCV = kron(s2, XpXi) / T
    expected = np.kron(s2, XpXi) / T

    vcv = vectorarvcv(X, errors, het=0, uncorr=0)
    npt.assert_allclose(
        vcv, expected, atol=ATOL, rtol=RTOL,
        err_msg="het=0, uncorr=0 does not match manual kron(s2, XpXi)/T"
    )


def test_vectorarvcv_het0_uncorr1(vcv_inputs):
    """Homoskedastic, uncorrelated: VCV = kron(diag(diag(s2)), XpXi) / T.

    Treats equations as independent; zeroes out cross-equation error covariance,
    retaining only own-equation variances on the diagonal of s2.
    Ref: vectorarvcv.m lines 62-64 — VCV = kron(diag(diag(s2)), XpXi) / T
    """
    X, errors = vcv_inputs
    T = X.shape[0]

    s2 = errors.T @ errors / T
    XpXi = np.linalg.inv((X.T @ X) / T)

    # Ref: vectorarvcv.m:64 — VCV = kron(diag(diag(s2)), XpXi) / T
    # np.diag(np.diag(s2)) extracts diagonal then re-creates diagonal matrix
    expected = np.kron(np.diag(np.diag(s2)), XpXi) / T

    vcv = vectorarvcv(X, errors, het=0, uncorr=1)
    npt.assert_allclose(
        vcv, expected, atol=ATOL, rtol=RTOL,
        err_msg="het=0, uncorr=1 does not match manual kron(diag(diag(s2)), XpXi)/T"
    )


def test_vectorarvcv_het1_uncorr0(vcv_inputs):
    """Heteroskedastic, correlated: White-type sandwich VCV.

    Computes the full heteroskedasticity-robust VCV using the sandwich formula:
        Ainv = kron(I_K, XpXi)
        B = (s' * s) / T   where s = X2 .* e2
        VCV = Ainv * B * Ainv / T

    Ref: vectorarvcv.m lines 41-49
    """
    X, errors = vcv_inputs
    T = X.shape[0]
    Np = X.shape[1]
    K = errors.shape[1]

    XpXi = np.linalg.inv((X.T @ X) / T)

    # Build score matrix exactly as the MATLAB/Python implementation
    # Ref: vectorarvcv.m:41 — X2 = repmat(X, 1, K)
    X2 = np.tile(X, (1, K))
    # Ref: vectorarvcv.m:42 — e2 = reshape(repmat(errors, Np, 1), T, K*Np)
    # np.tile(errors, (Np, 1)) → (T*Np, K); reshape (T, K*Np) with column-major
    e2 = np.reshape(np.tile(errors, (Np, 1)), (T, K * Np), order='F')
    # Ref: vectorarvcv.m:43 — s = X2 .* e2
    s = X2 * e2

    # Ref: vectorarvcv.m:47 — Ainv = kron(eye(K), XpXi)
    Ainv = np.kron(np.eye(K), XpXi)
    # Ref: vectorarvcv.m:48 — B = (s' * s) / T
    B = (s.T @ s) / T
    # Ref: vectorarvcv.m:49 — VCV = Ainv * B * Ainv / T
    expected = Ainv @ B @ Ainv / T

    vcv = vectorarvcv(X, errors, het=1, uncorr=0)
    npt.assert_allclose(
        vcv, expected, atol=ATOL, rtol=RTOL,
        err_msg="het=1, uncorr=0 does not match manual White sandwich computation"
    )


def test_vectorarvcv_het1_uncorr1(vcv_inputs):
    """Heteroskedastic, uncorrelated: Block-diagonal sandwich VCV.

    Same sandwich structure as het=1, uncorr=0, but the meat matrix B is
    block-diagonal — zeroing out cross-equation score covariances so that
    each equation's VCV block is independent.

    Ref: vectorarvcv.m lines 50-58
    """
    X, errors = vcv_inputs
    T = X.shape[0]
    Np = X.shape[1]
    K = errors.shape[1]

    XpXi = np.linalg.inv((X.T @ X) / T)

    # Build score matrix
    # Ref: vectorarvcv.m:41-43
    X2 = np.tile(X, (1, K))
    e2 = np.reshape(np.tile(errors, (Np, 1)), (T, K * Np), order='F')
    s = X2 * e2

    # Block-diagonal meat: only own-equation blocks are nonzero
    # Ref: vectorarvcv.m:52 — B = zeros(Np*K)
    B = np.zeros((K * Np, K * Np))
    # Ref: vectorarvcv.m:53-57 — loop over K equations
    for i in range(K):
        # Ref: vectorarvcv.m:54-56 — sel, temp, B(sel,sel)
        si = s[:, i * Np:(i + 1) * Np]
        B[i * Np:(i + 1) * Np, i * Np:(i + 1) * Np] = (si.T @ si) / T

    # Ref: vectorarvcv.m:51 — Ainv = kron(eye(K), XpXi)
    Ainv = np.kron(np.eye(K), XpXi)
    # Ref: vectorarvcv.m:58 — VCV = Ainv * B * Ainv / T
    expected = Ainv @ B @ Ainv / T

    vcv = vectorarvcv(X, errors, het=1, uncorr=1)
    npt.assert_allclose(
        vcv, expected, atol=ATOL, rtol=RTOL,
        err_msg="het=1, uncorr=1 does not match manual block-diagonal sandwich"
    )


# =========================================================================
# Phase 5 — Property Tests (3 tests)
# =========================================================================


def test_vectorarvcv_modes_differ(vcv_inputs):
    """All 4 VCV modes must produce materially different results.

    With generic random data, all four error structure assumptions lead to
    distinct VCV matrices. This confirms the function actually branches
    correctly among the 4 modes.
    """
    X, errors = vcv_inputs
    results = {}
    for het in (0, 1):
        for uncorr in (0, 1):
            results[(het, uncorr)] = vectorarvcv(X, errors, het=het, uncorr=uncorr)

    # Compare each pair — none should be numerically identical
    keys = list(results.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert not np.allclose(
                results[keys[i]], results[keys[j]], atol=ATOL, rtol=RTOL
            ), (
                f"Modes {keys[i]} and {keys[j]} produced identical VCV matrices; "
                "expected all 4 modes to differ"
            )


def test_vectorarvcv_uncorrelated_block_diagonal(vcv_inputs):
    """VCV with uncorr=1 must have block-diagonal structure.

    Off-diagonal blocks (cross-equation covariances) must be exactly zero
    for both homoskedastic and heteroskedastic uncorrelated modes.
    This verifies that setting uncorr=1 properly eliminates cross-equation
    parameter covariance.
    """
    X, errors = vcv_inputs
    Np = X.shape[1]
    K = errors.shape[1]

    for het in (0, 1):
        vcv = vectorarvcv(X, errors, het=het, uncorr=1)
        # Check that off-diagonal blocks are all zero
        for i in range(K):
            for j in range(K):
                if i != j:
                    block = vcv[i * Np:(i + 1) * Np, j * Np:(j + 1) * Np]
                    npt.assert_allclose(
                        block,
                        np.zeros((Np, Np)),
                        atol=ATOL,
                        err_msg=(
                            f"Off-diagonal block ({i},{j}) is non-zero "
                            f"for het={het}, uncorr=1"
                        ),
                    )


def test_vectorarvcv_homoskedastic_consistency(vcv_inputs):
    """Under i.i.d. normal data, het=1 VCV approximates het=0 VCV.

    With normally distributed errors (no heteroskedasticity), the White-type
    robust VCV (het=1) should be broadly consistent with the classical
    homoskedastic VCV (het=0). Specifically, the diagonal elements (parameter
    variances) should be of the same order of magnitude, with ratios
    remaining within a reasonable range [0.3, 3.0].
    """
    X, errors = vcv_inputs

    # Test both correlated and uncorrelated variants
    for uncorr in (0, 1):
        vcv_homo = vectorarvcv(X, errors, het=0, uncorr=uncorr)
        vcv_hetero = vectorarvcv(X, errors, het=1, uncorr=uncorr)

        diag_homo = np.diag(vcv_homo)
        diag_hetero = np.diag(vcv_hetero)

        # Under normality, White and OLS standard errors should be within
        # a reasonable factor of each other (same magnitude but not exact)
        ratio = diag_hetero / diag_homo
        assert np.all(ratio > 0.3), (
            f"Hetero diagonal too small vs homo for uncorr={uncorr}: "
            f"min ratio = {ratio.min():.4f}"
        )
        assert np.all(ratio < 3.0), (
            f"Hetero diagonal too large vs homo for uncorr={uncorr}: "
            f"max ratio = {ratio.max():.4f}"
        )


# =========================================================================
# Phase 6 — Parity Test (1 test)
# =========================================================================


def test_vectorarvcv_parity(fixture_dir):
    """Parity test against MATLAB-generated reference fixture.

    Loads the vectorarvcv.npy fixture containing regressor matrix X, error
    matrix errors, and reference VCV matrices for all 4 modes. Compares
    Python output to the MATLAB reference at ±1e-6 absolute tolerance
    per AAP Section 0.7.1.

    The fixture file structure is a pickled dict with keys:
        'X'             — (T, Np) regressor matrix
        'errors'        — (T, K)  residual matrix
        'het1_uncorr0'  — dict with 'VCV' key for mode (het=1, uncorr=0)
        'het0_uncorr0'  — dict with 'VCV' key for mode (het=0, uncorr=0)
        'het1_uncorr1'  — dict with 'VCV' key for mode (het=1, uncorr=1)
        'het0_uncorr1'  — dict with 'VCV' key for mode (het=0, uncorr=1)
    """
    fixture_path = os.path.join(str(fixture_dir), 'timeseries', 'vectorarvcv.npy')
    if not os.path.exists(fixture_path):
        pytest.skip(f"Fixture file not found: {fixture_path}")

    data = np.load(fixture_path, allow_pickle=True).item()

    X = data['X']
    errors = data['errors']

    # Map of (het, uncorr) tuples to fixture dictionary keys
    mode_keys = {
        (1, 0): 'het1_uncorr0',
        (0, 0): 'het0_uncorr0',
        (1, 1): 'het1_uncorr1',
        (0, 1): 'het0_uncorr1',
    }

    for (het, uncorr), key in mode_keys.items():
        if key not in data:
            pytest.skip(f"Fixture missing mode key '{key}'")
        expected_vcv = data[key]['VCV']
        actual_vcv = vectorarvcv(X, errors, het=het, uncorr=uncorr)
        npt.assert_allclose(
            actual_vcv, expected_vcv, atol=ATOL, rtol=RTOL,
            err_msg=(
                f"Parity failure for mode {key} (het={het}, uncorr={uncorr}): "
                f"Python output does not match MATLAB reference within tolerance"
            ),
        )
