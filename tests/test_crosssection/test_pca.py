"""
Comprehensive pytest test module for ``mfe_toolbox.crosssection.pca``.

Tests principal component analysis with 3 normalization modes:
  - ``'outer'`` (uncentered outer-product)
  - ``'cov'`` (covariance of demeaned data)
  - ``'corr'`` (correlation / covariance of standardized data)

Covers unit tests, integration tests, property tests, input validation
tests, and numerical parity tests against MATLAB reference outputs.

Source Reference:
  - MATLAB: ``crosssection/pca.m`` (120 lines, Kevin Sheppard, Rev 3.01)
  - Python: ``mfe_toolbox/crosssection/pca.py``

AAP Section 0.5.1: ``tests/test_crosssection/*.py — CREATE``
AAP Section 0.7.1: Numerical parity contract ±1e-6 atol, ±1e-4 rtol
"""

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.crosssection.pca import pca
from tests.conftest import ATOL, RTOL, load_fixture_npy, assert_allclose


# ---------------------------------------------------------------------------
# Eigenvector sign-alignment helper
# ---------------------------------------------------------------------------

def align_eigenvector_signs(
    actual: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """Align eigenvector signs between *actual* and *reference* (column-wise).

    Eigenvectors are unique only up to sign.  For each column *i* of
    *actual*, if its dot-product with the corresponding column of
    *reference* is negative the column is negated so that sign conventions
    match for numerical comparison.

    Parameters
    ----------
    actual : np.ndarray
        M × N array whose columns are eigenvectors to align.
    reference : np.ndarray
        M × N reference eigenvector array.

    Returns
    -------
    np.ndarray
        Copy of *actual* with columns sign-aligned to *reference*.
    """
    result = actual.copy()
    for i in range(result.shape[1]):
        # Ref: pca.m eigenvector sign ambiguity — eig() / eigh() output
        # differs in sign convention between MATLAB and NumPy.
        if np.dot(result[:, i], reference[:, i]) < 0:
            result[:, i] *= -1
    return result


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pca_data() -> np.ndarray:
    """T=200, K=4 data with clear principal component structure (seed=42).

    Introduces correlation between columns so that eigenvalues are well
    separated, making the PCA decomposition more testable.
    """
    rng = np.random.default_rng(42)
    T, K = 200, 4
    # Base independent random data
    z = rng.standard_normal((T, K))
    # Add correlation structure for distinct eigenvalues
    z[:, 1] += 0.5 * z[:, 0]
    z[:, 2] += 0.3 * z[:, 0] + 0.2 * z[:, 1]
    z[:, 3] += 0.1 * z[:, 0]
    return z


@pytest.fixture
def simple_pca_data() -> np.ndarray:
    """T=100, K=2 correlated data (seed=99).

    Simple 2-variable dataset with high correlation (~0.7) for basic
    property verification and edge-case testing.
    """
    rng = np.random.default_rng(99)
    T = 100
    z1 = rng.standard_normal(T)
    z2 = 0.7 * z1 + 0.3 * rng.standard_normal(T)
    return np.column_stack([z1, z2])


# ---------------------------------------------------------------------------
# TestPCA — comprehensive test class
# ---------------------------------------------------------------------------

class TestPCA:
    """Tests for ``mfe_toolbox.crosssection.pca.pca``."""

    # -----------------------------------------------------------------------
    # 1.2  Basic outer-mode tests
    # -----------------------------------------------------------------------

    def test_pca_outer_default(self, pca_data: np.ndarray) -> None:
        """Default pca() call returns 5 outputs with correct shapes."""
        T, K = pca_data.shape
        weights, princomp, eigenvals, explvar, cumR2 = pca(pca_data)

        assert weights.shape == (K, K), "weights should be K×K"
        assert princomp.shape == (T, K), "princomp should be T×K"
        assert eigenvals.shape == (K,), "eigenvals should be K-vector"
        assert explvar.shape == (K,), "explvar should be K-vector"
        assert cumR2.shape == (K,), "cumR2 should be K-vector"

    def test_pca_outer_explicit(self, pca_data: np.ndarray) -> None:
        """Explicit type='outer' produces identical results to default call."""
        res_default = pca(pca_data)
        res_explicit = pca(pca_data, type='outer')
        for idx, (d, e) in enumerate(zip(res_default, res_explicit)):
            assert np.array_equal(d, e), (
                f"Output index {idx} differs between default and 'outer'"
            )
            npt.assert_array_equal(
                d, e, err_msg=f"Output index {idx} differs between default and 'outer'"
            )

    def test_pca_outer_empty_string(self, pca_data: np.ndarray) -> None:
        """Empty string type='' falls back to outer mode. Ref: pca.m:55-56."""
        res_default = pca(pca_data)
        res_empty = pca(pca_data, type='')
        for idx, (d, e) in enumerate(zip(res_default, res_empty)):
            npt.assert_array_equal(
                d, e, err_msg=f"Output index {idx} differs between default and ''"
            )

    def test_pca_outer_no_demeaning(self, pca_data: np.ndarray) -> None:
        """Outer mode uses data.T @ data / T — NO demeaning. Ref: pca.m:79-80.

        Verifies that eigenvalues match the eigendecomposition of the
        uncentered outer-product matrix.
        """
        T, K = pca_data.shape
        _, _, eigenvals, _, _ = pca(pca_data)

        # Manual computation of the outer-product matrix
        inputmat = pca_data.T @ pca_data / T
        manual_eigenvals = np.sort(np.linalg.eigh(inputmat)[0])[::-1]
        npt.assert_allclose(eigenvals, manual_eigenvals, atol=ATOL, rtol=RTOL)

    def test_pca_outer_reconstruction(self, pca_data: np.ndarray) -> None:
        """Outer mode: data ≈ princomp @ weights. Ref: pca.m:23."""
        weights, princomp, _, _, _ = pca(pca_data)
        reconstructed = princomp @ weights
        npt.assert_allclose(reconstructed, pca_data, atol=ATOL, rtol=RTOL)

    # -----------------------------------------------------------------------
    # 1.3  Covariance mode tests
    # -----------------------------------------------------------------------

    def test_pca_cov_mode(self, pca_data: np.ndarray) -> None:
        """Cov mode returns 5 outputs with correct shapes."""
        T, K = pca_data.shape
        weights, princomp, eigenvals, explvar, cumR2 = pca(
            pca_data, type='cov'
        )
        assert weights.shape == (K, K)
        assert princomp.shape == (T, K)
        assert eigenvals.shape == (K,)
        assert explvar.shape == (K,)
        assert cumR2.shape == (K,)

    def test_pca_cov_demeaned_reconstruction(self, pca_data: np.ndarray) -> None:
        """Cov mode: demeaned_data ≈ princomp @ weights. Ref: pca.m:26-28."""
        weights, princomp, _, _, _ = pca(pca_data, type='cov')
        demeaned = pca_data - np.mean(pca_data, axis=0)
        reconstructed = princomp @ weights
        npt.assert_allclose(reconstructed, demeaned, atol=ATOL, rtol=RTOL)

    def test_pca_cov_eigenvalues_match_variance(
        self, pca_data: np.ndarray
    ) -> None:
        """Cov mode eigenvalues match eigh(np.cov(demeaned)) eigenvalues."""
        _, _, eigenvals, _, _ = pca(pca_data, type='cov')
        # Manually compute covariance matrix
        demeaned = pca_data - np.mean(pca_data, axis=0)
        cov_mat = np.cov(demeaned, rowvar=False, bias=False)
        # Eigendecompose and sort descending
        ref_eigenvals = np.sort(np.linalg.eigh(cov_mat)[0])[::-1]
        npt.assert_allclose(eigenvals, ref_eigenvals, atol=ATOL, rtol=RTOL)

    # -----------------------------------------------------------------------
    # 1.4  Correlation mode tests
    # -----------------------------------------------------------------------

    def test_pca_corr_mode(self, pca_data: np.ndarray) -> None:
        """Corr mode returns 5 outputs with correct shapes."""
        T, K = pca_data.shape
        weights, princomp, eigenvals, explvar, cumR2 = pca(
            pca_data, type='corr'
        )
        assert weights.shape == (K, K)
        assert princomp.shape == (T, K)
        assert eigenvals.shape == (K,)
        assert explvar.shape == (K,)
        assert cumR2.shape == (K,)

    def test_pca_corr_standardized_reconstruction(
        self, pca_data: np.ndarray
    ) -> None:
        """Corr mode: standardized_data ≈ princomp @ weights. Ref: pca.m:31-33."""
        weights, princomp, _, _, _ = pca(pca_data, type='corr')
        # Manual standardization: demean then divide by std(ddof=1)
        # Ref: pca.m:87-93
        demeaned = pca_data - np.mean(pca_data, axis=0)
        stdevs = np.std(demeaned, axis=0, ddof=1)
        standardized = demeaned / stdevs
        reconstructed = princomp @ weights
        npt.assert_allclose(reconstructed, standardized, atol=ATOL, rtol=RTOL)

    def test_pca_corr_zero_variance_raises(self) -> None:
        """Corr mode raises ValueError for zero-variance column. Ref: pca.m:90-91."""
        rng = np.random.default_rng(12)
        data = rng.standard_normal((100, 3))
        # Set one column to a constant — zero variance after demeaning
        data[:, 1] = 5.0
        with pytest.raises(ValueError, match="no variation"):
            pca(data, type='corr')

    # -----------------------------------------------------------------------
    # 1.5  Eigenvalue / component ordering
    # -----------------------------------------------------------------------

    def test_pca_eigenvalues_descending(self, pca_data: np.ndarray) -> None:
        """Eigenvalues are sorted in descending order. Ref: pca.m:105."""
        _, _, eigenvals, _, _ = pca(pca_data)
        diffs = np.diff(eigenvals)
        # Every successive difference should be ≤ 0 (descending)
        assert np.all(diffs <= ATOL), (
            f"Eigenvalues not descending: diffs = {diffs}"
        )

    def test_pca_eigenvalues_nonnegative(self, pca_data: np.ndarray) -> None:
        """All eigenvalues are non-negative for a valid PCA decomposition."""
        _, _, eigenvals, _, _ = pca(pca_data)
        # Allow tiny negative noise from numerical round-off
        assert np.all(eigenvals >= -ATOL), (
            f"Negative eigenvalue(s): {eigenvals}"
        )

    def test_pca_explvar_sums_to_one(self, pca_data: np.ndarray) -> None:
        """Explained variance fractions sum to 1.0. Ref: pca.m:111."""
        _, _, _, explvar, _ = pca(pca_data)
        npt.assert_allclose(np.sum(explvar), 1.0, atol=ATOL)

    def test_pca_cumr2_monotone(self, pca_data: np.ndarray) -> None:
        """Cumulative R² is monotonically non-decreasing, final ≈ 1.0. Ref: pca.m:114."""
        _, _, _, _, cumR2 = pca(pca_data)
        diffs = np.diff(cumR2)
        assert np.all(diffs >= -ATOL), (
            f"cumR2 not monotone: diffs = {diffs}"
        )
        npt.assert_allclose(cumR2[-1], 1.0, atol=ATOL)

    def test_pca_cumr2_from_explvar(self, pca_data: np.ndarray) -> None:
        """cumR2 equals cumsum(explvar). Ref: pca.m:114."""
        _, _, _, explvar, cumR2 = pca(pca_data)
        npt.assert_allclose(cumR2, np.cumsum(explvar), atol=ATOL, rtol=RTOL)

    def test_pca_first_component_most_variance(
        self, pca_data: np.ndarray
    ) -> None:
        """First component explains the most variance (explvar descending)."""
        _, _, _, explvar, _ = pca(pca_data)
        diffs = np.diff(explvar)
        # Every successive difference should be ≤ 0 (descending)
        assert np.all(diffs <= ATOL), (
            f"explvar not descending: {explvar}"
        )

    # -----------------------------------------------------------------------
    # 1.6  Component properties
    # -----------------------------------------------------------------------

    def test_pca_weights_orthogonal(self, pca_data: np.ndarray) -> None:
        """Weight matrix rows are orthonormal: W @ W.T ≈ I."""
        weights, _, _, _, _ = pca(pca_data)
        K = weights.shape[0]
        product = weights @ weights.T
        npt.assert_allclose(product, np.eye(K), atol=ATOL, rtol=RTOL)

    def test_pca_princomp_columns_uncorrelated(
        self, multivariate_data: np.ndarray
    ) -> None:
        """Principal component columns are uncorrelated (off-diagonal ≈ 0).

        Uses cov mode on the conftest ``multivariate_data`` fixture
        (T=1000, K=3, mean-zero) so that principal components are mean-zero
        and the correlation matrix is well-defined.  The larger sample size
        gives a tighter bound on off-diagonal correlations.
        """
        _, princomp, _, _, _ = pca(multivariate_data, type='cov')
        K = princomp.shape[1]
        corr_matrix = np.corrcoef(princomp, rowvar=False)
        # Off-diagonal elements should be near zero
        off_diag = corr_matrix - np.eye(K)
        npt.assert_allclose(
            off_diag, np.zeros((K, K)), atol=1e-4,
            err_msg="Principal components should be uncorrelated"
        )

    # -----------------------------------------------------------------------
    # 1.7  Input validation
    # -----------------------------------------------------------------------

    def test_pca_not_2d_raises(self) -> None:
        """1-D input raises ValueError. Ref: pca.m:46-47."""
        with pytest.raises(ValueError, match="T by K"):
            pca(np.array([1.0, 2.0, 3.0]))

    def test_pca_3d_raises(self) -> None:
        """3-D input raises ValueError. Ref: pca.m:46-47."""
        with pytest.raises(ValueError, match="T by K"):
            pca(np.ones((5, 3, 2)))

    def test_pca_invalid_type_raises(self) -> None:
        """Invalid type string raises ValueError. Ref: pca.m:63-64."""
        data = np.random.default_rng(0).standard_normal((50, 3))
        with pytest.raises(ValueError, match="TYPE"):
            pca(data, type='invalid')

    def test_pca_case_insensitive_type(self, pca_data: np.ndarray) -> None:
        """Type parameter is case-insensitive. Ref: pca.m:54 lower(type)."""
        # All case variants should produce identical results
        res_lower = pca(pca_data, type='cov')
        res_upper = pca(pca_data, type='COV')
        res_mixed = pca(pca_data, type='Cov')
        for idx, (lo, up, mx) in enumerate(
            zip(res_lower, res_upper, res_mixed)
        ):
            npt.assert_array_equal(
                lo, up, err_msg=f"'cov' vs 'COV' at output {idx}"
            )
            npt.assert_array_equal(
                lo, mx, err_msg=f"'cov' vs 'Cov' at output {idx}"
            )

        # Repeat for corr
        res_corr = pca(pca_data, type='corr')
        res_CORR = pca(pca_data, type='CORR')
        res_Corr = pca(pca_data, type='Corr')
        for idx, (a, b, c) in enumerate(
            zip(res_corr, res_CORR, res_Corr)
        ):
            npt.assert_array_equal(a, b, err_msg=f"'corr' vs 'CORR' at {idx}")
            npt.assert_array_equal(a, c, err_msg=f"'corr' vs 'Corr' at {idx}")

    # -----------------------------------------------------------------------
    # 1.8  Parity tests (MATLAB fixture-based)
    # -----------------------------------------------------------------------

    @pytest.mark.parity
    def test_pca_outer_parity(self, crosssection_fixture_dir) -> None:
        """MATLAB parity: outer mode — all 5 outputs match reference.

        Loads the composite ``pca.npy`` fixture containing MATLAB reference
        data and expected outputs for outer, cov, and corr modes.
        Sign ambiguity in eigenvectors is handled by column-wise alignment.
        """
        fixture_arr = load_fixture_npy(crosssection_fixture_dir, "pca")
        fix = fixture_arr.item() if fixture_arr.ndim == 0 else fixture_arr

        data = fix['data']
        weights, princomp, eigenvals, explvar, cumR2 = pca(data, type='outer')

        # Eigenvalues, explvar, cumR2 are sign-independent
        assert_allclose(eigenvals, fix['outer_eigenvals'], err_msg="outer eigenvals")
        assert_allclose(explvar, fix['outer_explvar'], err_msg="outer explvar")
        assert_allclose(cumR2, fix['outer_cumR2'], err_msg="outer cumR2")

        # Eigenvectors need sign alignment.
        # princomp columns and weight rows share the same sign ambiguity.
        # Determine sign convention from princomp (more elements → robust).
        ref_pc = fix['outer_princomp']
        ref_w = fix['outer_weights']
        K = princomp.shape[1]
        signs = np.ones(K)
        for i in range(K):
            if np.dot(princomp[:, i], ref_pc[:, i]) < 0:
                signs[i] = -1.0
        aligned_pc = princomp * signs[np.newaxis, :]
        aligned_w = weights * signs[:, np.newaxis]

        assert_allclose(aligned_pc, ref_pc, err_msg="outer princomp")
        assert_allclose(aligned_w, ref_w, err_msg="outer weights")

    @pytest.mark.parity
    def test_pca_cov_parity(self, crosssection_fixture_dir) -> None:
        """MATLAB parity: cov mode — all 5 outputs match reference."""
        fixture_arr = load_fixture_npy(crosssection_fixture_dir, "pca")
        fix = fixture_arr.item() if fixture_arr.ndim == 0 else fixture_arr

        data = fix['data']
        weights, princomp, eigenvals, explvar, cumR2 = pca(data, type='cov')

        assert_allclose(eigenvals, fix['cov_eigenvals'], err_msg="cov eigenvals")
        assert_allclose(explvar, fix['cov_explvar'], err_msg="cov explvar")
        assert_allclose(cumR2, fix['cov_cumR2'], err_msg="cov cumR2")

        # Sign alignment
        ref_pc = fix['cov_princomp']
        ref_w = fix['cov_weights']
        K = princomp.shape[1]
        signs = np.ones(K)
        for i in range(K):
            if np.dot(princomp[:, i], ref_pc[:, i]) < 0:
                signs[i] = -1.0
        aligned_pc = princomp * signs[np.newaxis, :]
        aligned_w = weights * signs[:, np.newaxis]

        assert_allclose(aligned_pc, ref_pc, err_msg="cov princomp")
        assert_allclose(aligned_w, ref_w, err_msg="cov weights")

    @pytest.mark.parity
    def test_pca_corr_parity(self, crosssection_fixture_dir) -> None:
        """MATLAB parity: corr mode — all 5 outputs match reference."""
        fixture_arr = load_fixture_npy(crosssection_fixture_dir, "pca")
        fix = fixture_arr.item() if fixture_arr.ndim == 0 else fixture_arr

        data = fix['data']
        weights, princomp, eigenvals, explvar, cumR2 = pca(data, type='corr')

        assert_allclose(eigenvals, fix['corr_eigenvals'], err_msg="corr eigenvals")
        assert_allclose(explvar, fix['corr_explvar'], err_msg="corr explvar")
        assert_allclose(cumR2, fix['corr_cumR2'], err_msg="corr cumR2")

        # Sign alignment
        ref_pc = fix['corr_princomp']
        ref_w = fix['corr_weights']
        K = princomp.shape[1]
        signs = np.ones(K)
        for i in range(K):
            if np.dot(princomp[:, i], ref_pc[:, i]) < 0:
                signs[i] = -1.0
        aligned_pc = princomp * signs[np.newaxis, :]
        aligned_w = weights * signs[:, np.newaxis]

        assert_allclose(aligned_pc, ref_pc, err_msg="corr princomp")
        assert_allclose(aligned_w, ref_w, err_msg="corr weights")

    # -----------------------------------------------------------------------
    # 1.9  Parametrized mode tests
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("mode", ['outer', 'cov', 'corr'])
    def test_pca_all_modes_parametrized(
        self, pca_data: np.ndarray, mode: str
    ) -> None:
        """Common PCA properties hold across all three modes.

        Verifies: output shapes, eigenvalue ordering, explvar sums to 1,
        cumR2 monotone and final ≈ 1, weights orthogonal.
        """
        T, K = pca_data.shape
        weights, princomp, eigenvals, explvar, cumR2 = pca(
            pca_data, type=mode
        )

        # Shape checks
        assert weights.shape == (K, K), f"[{mode}] weights shape"
        assert princomp.shape == (T, K), f"[{mode}] princomp shape"
        assert eigenvals.shape == (K,), f"[{mode}] eigenvals shape"
        assert explvar.shape == (K,), f"[{mode}] explvar shape"
        assert cumR2.shape == (K,), f"[{mode}] cumR2 shape"

        # Eigenvalues descending and non-negative
        assert np.all(np.diff(eigenvals) <= ATOL), (
            f"[{mode}] eigenvalues not descending"
        )
        assert np.all(eigenvals >= -ATOL), (
            f"[{mode}] negative eigenvalue"
        )

        # explvar sums to 1
        npt.assert_allclose(
            np.sum(explvar), 1.0, atol=ATOL,
            err_msg=f"[{mode}] explvar does not sum to 1"
        )

        # cumR2 monotone non-decreasing, final ≈ 1
        assert np.all(np.diff(cumR2) >= -ATOL), (
            f"[{mode}] cumR2 not monotone"
        )
        npt.assert_allclose(
            cumR2[-1], 1.0, atol=ATOL,
            err_msg=f"[{mode}] cumR2[-1] != 1"
        )

        # Weights orthogonal: W @ W.T ≈ I
        npt.assert_allclose(
            weights @ weights.T, np.eye(K), atol=ATOL, rtol=RTOL,
            err_msg=f"[{mode}] weights not orthogonal"
        )

    # -----------------------------------------------------------------------
    # 1.10  Edge cases
    # -----------------------------------------------------------------------

    def test_pca_single_column(self) -> None:
        """PCA on T×1 data produces valid 1-component results."""
        rng = np.random.default_rng(77)
        data = rng.standard_normal((50, 1))
        weights, princomp, eigenvals, explvar, cumR2 = pca(data)

        assert weights.shape == (1, 1)
        assert princomp.shape == (50, 1)
        assert eigenvals.shape == (1,)
        assert explvar.shape == (1,)
        assert cumR2.shape == (1,)

        # Single component must explain 100% of variance
        npt.assert_allclose(explvar[0], 1.0, atol=ATOL)
        npt.assert_allclose(cumR2[0], 1.0, atol=ATOL)

    def test_pca_square_data(self) -> None:
        """PCA works when T = K (square data matrix)."""
        rng = np.random.default_rng(55)
        N = 10
        data = rng.standard_normal((N, N))
        weights, princomp, eigenvals, explvar, cumR2 = pca(data)

        assert weights.shape == (N, N)
        assert princomp.shape == (N, N)
        assert eigenvals.shape == (N,)
        npt.assert_allclose(np.sum(explvar), 1.0, atol=ATOL)
        # Reconstruction identity for outer mode
        npt.assert_allclose(princomp @ weights, data, atol=ATOL, rtol=RTOL)

    def test_pca_data_not_modified(self, pca_data: np.ndarray) -> None:
        """Input data array is not modified by any PCA mode.

        The function makes an internal copy (pca.py:118) to avoid mutating
        the caller's array.
        """
        for mode in ['outer', 'cov', 'corr']:
            data_copy = pca_data.copy()
            pca(data_copy, type=mode)
            npt.assert_array_equal(
                data_copy, pca_data,
                err_msg=f"Data modified by mode='{mode}'"
            )

    def test_pca_return_types(self, pca_data: np.ndarray) -> None:
        """All 5 PCA outputs are numpy.ndarray instances."""
        outputs = pca(pca_data)
        for idx, out in enumerate(outputs):
            assert isinstance(out, np.ndarray), (
                f"Output {idx} is {type(out).__name__}, expected np.ndarray"
            )
