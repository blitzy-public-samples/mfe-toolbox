"""
Comprehensive pytest test module for mfe_toolbox.crosssection.ols.

Tests OLS regression with both homoskedastic (standard) and White
heteroskedasticity-robust standard errors.  Covers unit tests, integration
tests, and numerical parity tests against MATLAB reference outputs generated
by Octave via ``scripts/generate_fixtures.m``.

Source reference: crosssection/ols.m (127 lines, Author: Kevin Sheppard,
                  Revision 3, Date: 9/1/2005)
AAP Sections: 0.5.1 (parametrized parity tests for crosssection)
              0.7.1 (numerical parity contract ±1e-6 atol, ±1e-4 rtol)

OLS function specification (8-tuple return):
    b, tstat, s2, vcv, vcvwhite, R2, Rbar, yhat = ols(y, x, c=1)
"""

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.crosssection.ols import ols
from tests.conftest import ATOL, RTOL, load_fixture_npy, assert_allclose


# ---------------------------------------------------------------------------
# Local Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def regression_data_with_constant(regression_data):
    """Extend regression_data with a known-coefficient model for precise testing.

    Uses the regression_data fixture from conftest.py (T=200, 3 regressors,
    y = X @ [0.5, -1.0, 0.3] + noise, seed=42).
    """
    y, X = regression_data
    return y, X


@pytest.fixture
def simple_regression():
    """Simple 2-variable regression with known solution for exact verification.

    Creates a low-noise (sigma=0.1) regression so that estimated coefficients
    should be close to the true values [1.0, -0.5].  Uses a deterministic
    seed for reproducibility.
    """
    rng = np.random.default_rng(123)
    N = 100
    x = rng.standard_normal((N, 2))
    beta_true = np.array([1.0, -0.5])
    y = x @ beta_true + 0.1 * rng.standard_normal(N)
    return y, x, beta_true


# ---------------------------------------------------------------------------
# TestOLS — Comprehensive OLS Test Suite
# ---------------------------------------------------------------------------

class TestOLS:
    """Comprehensive test suite for the OLS regression function.

    Covers:
    - Basic functionality with / without constant
    - Manual verification of all 8 return values
    - Input validation error paths
    - MATLAB fixture-based numerical parity
    - Edge cases (single regressor, perfect fit, return types)
    """

    # ===================================================================
    # 1.2  Basic Functionality Tests
    # ===================================================================

    def test_ols_with_constant(self, regression_data):
        """Call ols(y, X, c=1) using regression_data fixture (T=200, K=3).

        Verifies shape, sign, and structural properties of all 8 outputs
        when a constant is included.  Ref: ols.m overall structure.
        """
        y, X = regression_data
        b, tstat, s2, vcv, vcvwhite, R2, Rbar, yhat = ols(y, X, c=1)

        N, K_orig = X.shape
        K = K_orig + 1  # constant prepended — Ref: ols.m:89,93

        # b has shape (K, 1) — 3 regressors + 1 constant
        assert b.shape[0] == K, f"Expected {K} coefficients, got {b.shape[0]}"

        # tstat has same leading dimension as b
        assert tstat.shape[0] == K, "tstat dimension mismatch"

        # s2 is a positive scalar (float) — Ref: ols.m:107
        assert isinstance(s2, (float, np.floating)), "s2 must be a scalar"
        assert s2 > 0, "s2 must be positive"

        # vcv is (K, K) with positive diagonal — Ref: ols.m:109
        assert vcv.shape == (K, K), f"vcv shape mismatch: {vcv.shape}"
        assert np.all(np.diag(vcv) > 0), "vcv diagonal must be positive"

        # vcvwhite is (K, K) with positive diagonal — Ref: ols.m:115
        assert vcvwhite.shape == (K, K), f"vcvwhite shape mismatch"
        assert np.all(np.diag(vcvwhite) > 0), "vcvwhite diagonal must be positive"

        # R² in [0, 1] — Ref: ols.m:122
        assert 0.0 <= R2 <= 1.0, f"R2 out of range: {R2}"

        # Rbar ≤ R2 when K > 1 — Ref: ols.m:123
        assert Rbar <= R2 + 1e-15, "Adjusted R² should be ≤ R²"

        # yhat has correct length — Ref: ols.m:103
        assert yhat.shape[0] == N, f"yhat length mismatch: {yhat.shape[0]}"

        # Orthogonality: residuals sum ≈ 0 when constant included
        # (OLS first-order condition when constant is in the design matrix)
        epsilon = y.reshape(-1) - yhat.flatten()
        npt.assert_allclose(
            np.sum(epsilon), 0.0, atol=1e-10,
            err_msg="Residuals must sum to zero with a constant"
        )

    def test_ols_without_constant(self, regression_data):
        """Call ols(y, X, c=0) using regression_data — no constant.

        Verifies correct dimensions and that R² uses the uncentered formula.
        Ref: ols.m:124-125 for uncentered R².
        """
        y, X = regression_data
        b, tstat, s2, vcv, vcvwhite, R2, Rbar, yhat = ols(y, X, c=0)

        N, K = X.shape

        # b has shape (K,) or (K,1) — no constant
        assert b.shape[0] == K, f"Expected {K} coefficients without constant"
        assert tstat.shape[0] == K
        assert isinstance(s2, (float, np.floating)) and s2 > 0
        assert vcv.shape == (K, K)
        assert vcvwhite.shape == (K, K)
        assert yhat.shape[0] == N

        # R² uses uncentered formula: 1 - (e'e)/(y'y)
        # Ref: ols.m:124-125
        epsilon = y.reshape(-1) - yhat.flatten()
        sse = np.dot(epsilon, epsilon)
        sst_uncentered = np.dot(y.reshape(-1), y.reshape(-1))
        R2_manual = 1.0 - sse / sst_uncentered
        npt.assert_allclose(R2, R2_manual, atol=ATOL, rtol=RTOL,
                            err_msg="R² must use uncentered formula when c=0")

    def test_ols_default_constant(self, regression_data):
        """Call ols(y, X) without specifying c — verify default c=1.

        Ref: ols.m:68-69 — MATLAB ``if nargin==2, c=1; end``
        """
        y, X = regression_data
        result_default = ols(y, X)
        result_explicit = ols(y, X, c=1)

        # All 8 outputs must be bit-identical
        npt.assert_array_equal(result_default[0], result_explicit[0],
                               err_msg="b mismatch")
        npt.assert_array_equal(result_default[1], result_explicit[1],
                               err_msg="tstat mismatch")
        assert result_default[2] == result_explicit[2], "s2 mismatch"
        npt.assert_array_equal(result_default[3], result_explicit[3],
                               err_msg="vcv mismatch")
        npt.assert_array_equal(result_default[4], result_explicit[4],
                               err_msg="vcvwhite mismatch")
        assert result_default[5] == result_explicit[5], "R2 mismatch"
        assert result_default[6] == result_explicit[6], "Rbar mismatch"
        npt.assert_array_equal(result_default[7], result_explicit[7],
                               err_msg="yhat mismatch")

    def test_ols_coefficient_recovery(self, simple_regression):
        """Sanity check: estimated b close to true beta with low noise.

        Using simple_regression fixture (N=100, sigma=0.1), coefficients
        should be recovered within a reasonable tolerance given noise.
        """
        y, x, beta_true = simple_regression
        # c=0 since the DGP has no intercept
        b, *_ = ols(y, x, c=0)
        # With sigma = 0.1, coefficients should be within ~0.2 of truth
        npt.assert_allclose(b.flatten(), beta_true, atol=0.2,
                            err_msg="Coefficient recovery failed")

    # ===================================================================
    # 1.3  Return Value Property Tests
    # ===================================================================

    def test_ols_fitted_values(self, regression_data):
        """Verify yhat = X_aug @ b where X_aug includes constant column.

        Ref: ols.m:103 — ``yhat = x*b``
        """
        y, X = regression_data
        b, tstat, s2, vcv, vcvwhite, R2, Rbar, yhat = ols(y, X, c=1)

        N = X.shape[0]
        # Reconstruct augmented X with constant prepended (Ref: ols.m:89)
        X_aug = np.column_stack([np.ones(N), X])
        yhat_manual = X_aug @ b.flatten()

        npt.assert_allclose(yhat.flatten(), yhat_manual, atol=ATOL, rtol=RTOL,
                            err_msg="Fitted values do not match X_aug @ b")

    def test_ols_residual_variance(self, regression_data):
        """Verify s2 = epsilon' @ epsilon / (N - K).

        Ref: ols.m:107 — ``s2 = epsilon'*epsilon/(N-K)``
        """
        y, X = regression_data
        b, tstat, s2, vcv, vcvwhite, R2, Rbar, yhat = ols(y, X, c=1)

        N = X.shape[0]
        K = b.shape[0]  # includes constant
        # Compute residuals manually
        epsilon = y.reshape(-1, 1) - yhat
        s2_manual = float((epsilon.T @ epsilon).item()) / (N - K)

        npt.assert_allclose(s2, s2_manual, atol=ATOL, rtol=RTOL,
                            err_msg="Residual variance s2 formula mismatch")

    def test_ols_vcv_formula(self, regression_data):
        """Verify vcv = s2 * inv(X'X).

        Ref: ols.m:109 — ``vcv = s2*(x'*x)^(-1)``
        """
        y, X = regression_data
        b, tstat, s2, vcv, vcvwhite, R2, Rbar, yhat = ols(y, X, c=1)

        N = X.shape[0]
        # Reconstruct augmented X (Ref: ols.m:89)
        X_aug = np.column_stack([np.ones(N), X])
        vcv_manual = s2 * np.linalg.inv(X_aug.T @ X_aug)

        npt.assert_allclose(vcv, vcv_manual, atol=ATOL, rtol=RTOL,
                            err_msg="Homoskedastic VCV formula mismatch")

    def test_ols_white_vcv_formula(self, regression_data):
        """Verify White sandwich VCV = (X'X)^{-1} X'ee'X (X'X)^{-1}.

        Ref: ols.m:111-115 — scores, XeeX, XpXi, vcvwhite formulas.
        """
        y, X = regression_data
        b, tstat, s2, vcv, vcvwhite, R2, Rbar, yhat = ols(y, X, c=1)

        N = X.shape[0]
        X_aug = np.column_stack([np.ones(N), X])
        epsilon = y.reshape(-1, 1) - yhat  # (N, 1)

        # Ref: ols.m:111 — scores = x .* repmat(epsilon, 1, K)
        # Python broadcasting: (N, K) * (N, 1) → (N, K)
        scores = X_aug * epsilon
        # Ref: ols.m:112 — XeeX = scores' * scores
        XeeX = scores.T @ scores
        # Ref: ols.m:114 — XpXi = (x'*x)^(-1)
        XpXi = np.linalg.inv(X_aug.T @ X_aug)
        # Ref: ols.m:115 — vcvwhite = XpXi * XeeX * XpXi
        vcvwhite_manual = XpXi @ XeeX @ XpXi

        npt.assert_allclose(vcvwhite, vcvwhite_manual, atol=ATOL, rtol=RTOL,
                            err_msg="White sandwich VCV formula mismatch")

    def test_ols_tstat_from_white(self, regression_data):
        """Verify t-statistics use White robust SEs, not standard SEs.

        Ref: ols.m:117 — ``tstat = b ./ sqrt(diag(vcvwhite))``
        """
        y, X = regression_data
        b, tstat, s2, vcv, vcvwhite, R2, Rbar, yhat = ols(y, X, c=1)

        # t-stats use White robust SEs (NOT homoskedastic SEs from vcv)
        tstat_manual = b.flatten() / np.sqrt(np.diag(vcvwhite))

        npt.assert_allclose(tstat.flatten(), tstat_manual, atol=ATOL, rtol=RTOL,
                            err_msg="t-stats must use White robust SEs")

        # Also verify that standard t-stats (from vcv) would differ
        tstat_standard = b.flatten() / np.sqrt(np.diag(vcv))
        # They should NOT be identical (unless heteroskedasticity is zero)
        # This is a structural check, not a numerical tolerance check
        assert not np.allclose(tstat.flatten(), tstat_standard, atol=1e-12), \
            "White t-stats should differ from homoskedastic t-stats"

    def test_ols_r_squared_centered(self, regression_data):
        """Verify centered R² formula when c=1.

        Ref: ols.m:120-122
            ytilde = y - mean(y);
            R2 = 1 - (epsilon'*epsilon) / (ytilde'*ytilde);
        """
        y, X = regression_data
        b, tstat, s2, vcv, vcvwhite, R2, Rbar, yhat = ols(y, X, c=1)

        epsilon = y.reshape(-1) - yhat.flatten()
        ytilde = y.reshape(-1) - np.mean(y)
        sse = np.dot(epsilon, epsilon)
        sst_centered = np.dot(ytilde, ytilde)
        R2_manual = 1.0 - sse / sst_centered

        npt.assert_allclose(R2, R2_manual, atol=ATOL, rtol=RTOL,
                            err_msg="Centered R² formula mismatch (c=1)")

    def test_ols_r_squared_uncentered(self, regression_data):
        """Verify uncentered R² formula when c=0.

        Ref: ols.m:124-125
            R2 = 1 - (epsilon'*epsilon) / (y'*y);
        """
        y, X = regression_data
        b, tstat, s2, vcv, vcvwhite, R2, Rbar, yhat = ols(y, X, c=0)

        epsilon = y.reshape(-1) - yhat.flatten()
        sse = np.dot(epsilon, epsilon)
        sst_uncentered = np.dot(y.reshape(-1), y.reshape(-1))
        R2_manual = 1.0 - sse / sst_uncentered

        npt.assert_allclose(R2, R2_manual, atol=ATOL, rtol=RTOL,
                            err_msg="Uncentered R² formula mismatch (c=0)")

    def test_ols_adjusted_r_squared(self, regression_data):
        """Verify adjusted R² for both centered (c=1) and uncentered (c=0).

        Ref: ols.m:123 — Rbar = 1 - (e'e)/(sst) * (N-1)/(N-K)
        Ref: ols.m:126 — same formula with uncentered SST
        """
        y, X = regression_data
        N = X.shape[0]

        # --- With constant (centered) — Ref: ols.m:123 ---
        b1, _, _, _, _, R2_1, Rbar_1, yhat1 = ols(y, X, c=1)
        K1 = b1.shape[0]  # includes constant
        epsilon1 = y.reshape(-1) - yhat1.flatten()
        ytilde = y.reshape(-1) - np.mean(y)
        sse1 = np.dot(epsilon1, epsilon1)
        sst1 = np.dot(ytilde, ytilde)
        Rbar_1_manual = 1.0 - (sse1 / sst1) * (N - 1) / (N - K1)
        npt.assert_allclose(Rbar_1, Rbar_1_manual, atol=ATOL, rtol=RTOL,
                            err_msg="Centered adjusted R² mismatch (c=1)")

        # --- Without constant (uncentered) — Ref: ols.m:126 ---
        b0, _, _, _, _, R2_0, Rbar_0, yhat0 = ols(y, X, c=0)
        K0 = b0.shape[0]
        epsilon0 = y.reshape(-1) - yhat0.flatten()
        sse0 = np.dot(epsilon0, epsilon0)
        sst0 = np.dot(y.reshape(-1), y.reshape(-1))
        Rbar_0_manual = 1.0 - (sse0 / sst0) * (N - 1) / (N - K0)
        npt.assert_allclose(Rbar_0, Rbar_0_manual, atol=ATOL, rtol=RTOL,
                            err_msg="Uncentered adjusted R² mismatch (c=0)")

    # ===================================================================
    # 1.4  Input Validation Tests
    # ===================================================================

    def test_ols_y_auto_transpose(self):
        """Pass y as a row vector (1, N); verify auto-transpose produces same result.

        Ref: ols.m:48-49 — ``if size(y,1) < size(y,2), y = y'; end``
        """
        rng = np.random.default_rng(99)
        x = rng.standard_normal((50, 2))
        y_col = rng.standard_normal(50)
        y_row = y_col.reshape(1, -1)  # (1, 50) row vector

        b_col, t_col, s2_col, vcv_col, vcvw_col, R2_col, Rb_col, yh_col = ols(y_col, x, c=1)
        b_row, t_row, s2_row, vcv_row, vcvw_row, R2_row, Rb_row, yh_row = ols(y_row, x, c=1)

        # All 8 outputs should be identical
        npt.assert_allclose(b_col, b_row, atol=ATOL, rtol=RTOL,
                            err_msg="Auto-transpose should produce same b")
        npt.assert_allclose(t_col, t_row, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(s2_col, s2_row, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(vcv_col, vcv_row, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(vcvw_col, vcvw_row, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(R2_col, R2_row, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(Rb_col, Rb_row, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(yh_col.flatten(), yh_row.flatten(), atol=ATOL, rtol=RTOL)

    def test_ols_y_not_column_raises(self):
        """Pass y as (N, 2) matrix; should raise ValueError.

        Ref: ols.m:51-52 — ``error('Y must be a column vector')``
        """
        rng = np.random.default_rng(100)
        x = rng.standard_normal((50, 2))
        y_multi = rng.standard_normal((50, 2))  # multi-column y
        with pytest.raises(ValueError, match="Y must be a column vector"):
            ols(y_multi, x, c=1)

    def test_ols_x_row_mismatch_raises(self):
        """Pass x with different row count than y; should raise ValueError.

        Ref: ols.m:57-58 — ``error('X must have the same number of rows as Y.')``
        """
        rng = np.random.default_rng(101)
        y = rng.standard_normal(50)
        x = rng.standard_normal((30, 2))  # 30 rows ≠ 50
        with pytest.raises(ValueError, match="X must have the same number of rows as Y"):
            ols(y, x, c=1)

    def test_ols_x_too_many_cols_raises(self):
        """Pass x with more columns than rows; should raise ValueError.

        Ref: ols.m:60-61 — ``error('The number of columns of X must be
        grater than or equal to T')``
        Note: the MATLAB source contains the typo "grater"; Python preserves it.
        """
        rng = np.random.default_rng(102)
        y = rng.standard_normal(5)
        x = rng.standard_normal((5, 10))  # 10 columns > 5 rows
        with pytest.raises(ValueError, match="grater"):
            ols(y, x, c=0)

    def test_ols_x_rank_deficient_raises(self):
        """Pass x with linearly dependent columns; should raise ValueError.

        Ref: ols.m:63-64 — ``error('X is rank deficient')``
        """
        y = np.random.default_rng(103).standard_normal(50)
        # Two identical columns → rank 1, but 2 columns
        x = np.column_stack([np.ones(50), np.ones(50)])
        with pytest.raises(ValueError, match="X is rank deficient"):
            ols(y, x, c=0)

    def test_ols_c_invalid_raises(self):
        """Pass c=2 or c=-1; should raise ValueError.

        Ref: ols.m:71-72 — ``error('C must be either 0 or 1.')``
        """
        rng = np.random.default_rng(104)
        y = rng.standard_normal(50)
        x = rng.standard_normal((50, 2))
        with pytest.raises(ValueError, match="C must be either 0 or 1"):
            ols(y, x, c=2)
        with pytest.raises(ValueError, match="C must be either 0 or 1"):
            ols(y, x, c=-1)

    def test_ols_no_model_raises(self):
        """Pass empty x with c=0; should raise ValueError.

        Ref: ols.m:75-77 — ``error('The model must include a constant
        or at least one X')``
        """
        y = np.random.default_rng(105).standard_normal(50)
        x = np.empty((50, 0))
        with pytest.raises(ValueError, match="must include a constant"):
            ols(y, x, c=0)

    def test_ols_x_contains_constant_raises(self):
        """Pass x with a constant column and c=1; should raise ValueError.

        Ref: ols.m:90-92 — ``error('X appears to contains a constant column.
        Use C to add a constant.')``
        Note: the MATLAB source contains the typo "contains" (should be
        "contain"); Python preserves it for API fidelity.
        """
        rng = np.random.default_rng(106)
        y = rng.standard_normal(50)
        # x already includes a column of ones
        x_with_const = np.column_stack([np.ones(50), rng.standard_normal((50, 2))])
        with pytest.raises(ValueError, match="X appears to contains a constant column"):
            ols(y, x_with_const, c=1)

    # ===================================================================
    # 1.5  Parity Tests (Fixture-Based)
    # ===================================================================

    @pytest.mark.parity
    def test_ols_parity_with_constant(self, crosssection_fixture_dir):
        """Compare all 8 OLS outputs against MATLAB/Octave fixtures (c=1).

        Loads fixture input data (ols_X_reg, ols_y_reg) generated by Octave
        and compares Python OLS outputs against each of the 8 fixture output
        arrays individually per AAP Section 0.7.1.

        Tolerance: atol=1e-6, rtol=1e-4.
        """
        # Load MATLAB-generated fixture inputs
        X_fix = load_fixture_npy(crosssection_fixture_dir, "ols_X_reg")
        y_fix = load_fixture_npy(crosssection_fixture_dir, "ols_y_reg")

        # Call Python OLS (c=1 default, matching MATLAB default)
        b, tstat, s2, vcv, vcvwhite, R2, Rbar, yhat = ols(y_fix, X_fix, c=1)

        # --- Compare all 8 outputs individually ---

        # 1. b (coefficients) — Python returns (K,1), fixture is (K,)
        b_exp = load_fixture_npy(crosssection_fixture_dir, "ols_ols_b")
        assert_allclose(b.flatten(), b_exp, err_msg="b (coefficients)")

        # 2. tstat — Ref: ols.m:117
        tstat_exp = load_fixture_npy(crosssection_fixture_dir, "ols_ols_tstat")
        assert_allclose(tstat.flatten(), tstat_exp, err_msg="tstat")

        # 3. s2 (residual variance) — Ref: ols.m:107
        s2_exp = load_fixture_npy(crosssection_fixture_dir, "ols_ols_s2")
        assert_allclose(np.array(s2), np.atleast_1d(s2_exp).flatten(),
                        err_msg="s2 (residual variance)")

        # 4. vcv (homoskedastic VCV) — Ref: ols.m:109
        vcv_exp = load_fixture_npy(crosssection_fixture_dir, "ols_ols_vcv")
        assert_allclose(vcv, vcv_exp, err_msg="vcv (homoskedastic VCV)")

        # 5. vcvwhite (White robust VCV) — Ref: ols.m:115
        vcvw_exp = load_fixture_npy(crosssection_fixture_dir, "ols_ols_vcvwhite")
        assert_allclose(vcvwhite, vcvw_exp, err_msg="vcvwhite (White robust VCV)")

        # 6. R2 — Ref: ols.m:122
        R2_exp = load_fixture_npy(crosssection_fixture_dir, "ols_ols_R2")
        assert_allclose(np.array(R2), np.atleast_1d(R2_exp).flatten(),
                        err_msg="R2")

        # 7. Rbar (adjusted R²) — Ref: ols.m:123
        Rbar_exp = load_fixture_npy(crosssection_fixture_dir, "ols_ols_Rbar")
        assert_allclose(np.array(Rbar), np.atleast_1d(Rbar_exp).flatten(),
                        err_msg="Rbar (adjusted R2)")

        # 8. yhat (fitted values) — Ref: ols.m:103
        yhat_exp = load_fixture_npy(crosssection_fixture_dir, "ols_ols_yhat")
        assert_allclose(yhat.flatten(), yhat_exp, err_msg="yhat (fitted values)")

    @pytest.mark.parity
    def test_ols_parity_without_constant(self, crosssection_fixture_dir):
        """Compare OLS outputs for c=0 against manual verification.

        Loads fixture input data and runs OLS without constant.  Since
        MATLAB fixtures were generated with c=1 default, we verify
        the c=0 path by manual formula checks using the fixture input data.
        """
        # Load MATLAB-generated fixture inputs
        X_fix = load_fixture_npy(crosssection_fixture_dir, "ols_X_reg")
        y_fix = load_fixture_npy(crosssection_fixture_dir, "ols_y_reg")

        # Call Python OLS with c=0
        b, tstat, s2, vcv, vcvwhite, R2, Rbar, yhat = ols(y_fix, X_fix, c=0)

        N, K = X_fix.shape
        assert b.shape[0] == K, "c=0 should have K coefficients (no constant)"

        # Manually verify all formulas for the c=0 path
        # Ref: ols.m:97 — b = x\y
        b_manual = np.linalg.lstsq(X_fix, y_fix.reshape(-1, 1), rcond=None)[0]
        npt.assert_allclose(b.flatten(), b_manual.flatten(), atol=ATOL, rtol=RTOL,
                            err_msg="b mismatch for c=0")

        # Ref: ols.m:103-107
        yhat_manual = (X_fix @ b_manual).flatten()
        epsilon = y_fix.reshape(-1) - yhat_manual
        s2_manual = np.dot(epsilon, epsilon) / (N - K)
        npt.assert_allclose(s2, s2_manual, atol=ATOL, rtol=RTOL,
                            err_msg="s2 mismatch for c=0")

        # Ref: ols.m:124-125 — uncentered R²
        sse = np.dot(epsilon, epsilon)
        sst_uncentered = np.dot(y_fix.reshape(-1), y_fix.reshape(-1))
        R2_manual = 1.0 - sse / sst_uncentered
        npt.assert_allclose(R2, R2_manual, atol=ATOL, rtol=RTOL,
                            err_msg="Uncentered R² mismatch for c=0")

        # Ref: ols.m:126 — uncentered adjusted R²
        Rbar_manual = 1.0 - (sse / sst_uncentered) * (N - 1) / (N - K)
        npt.assert_allclose(Rbar, Rbar_manual, atol=ATOL, rtol=RTOL,
                            err_msg="Uncentered Rbar mismatch for c=0")

    # ===================================================================
    # 1.6  Edge Cases
    # ===================================================================

    def test_ols_single_regressor(self):
        """Single column X (K=1) with c=1: 2 total parameters.

        Verifies OLS handles the minimum viable regression gracefully.
        """
        rng = np.random.default_rng(107)
        N = 80
        x = rng.standard_normal((N, 1))
        y = 2.0 * x.flatten() + 0.5 + 0.1 * rng.standard_normal(N)
        b, tstat, s2, vcv, vcvwhite, R2, Rbar, yhat = ols(y, x, c=1)

        # Constant + 1 regressor = 2 parameters
        assert b.shape[0] == 2, "Should have 2 parameters (const + 1 regressor)"
        assert vcv.shape == (2, 2)
        assert vcvwhite.shape == (2, 2)
        assert 0.0 <= R2 <= 1.0, f"R² out of range: {R2}"
        # With such a strong signal, R² should be high
        assert R2 > 0.9, f"R² should be high for strong linear signal: {R2}"

    def test_ols_perfect_fit(self):
        """Create y = X @ b exactly (no noise).  Verify R2 ≈ 1.0, s2 ≈ 0.

        With zero residuals, the regression achieves a perfect fit.
        """
        rng = np.random.default_rng(108)
        N = 50
        x = rng.standard_normal((N, 2))
        beta = np.array([3.0, -2.0])
        y = x @ beta  # perfect fit, zero noise

        b, tstat, s2, vcv, vcvwhite, R2, Rbar, yhat = ols(y, x, c=0)

        # Coefficients should be recovered exactly
        npt.assert_allclose(b.flatten(), beta, atol=1e-10,
                            err_msg="Perfect fit should recover exact beta")
        # R² should be 1.0 (no unexplained variance)
        npt.assert_allclose(R2, 1.0, atol=1e-10,
                            err_msg="R² should be 1.0 for perfect fit")
        # Residual variance should be 0 (or very close to machine precision)
        npt.assert_allclose(s2, 0.0, atol=1e-10,
                            err_msg="s2 should be ~0 for perfect fit")

    def test_ols_return_types(self, regression_data):
        """Verify all return types: ndarray for vectors/matrices, float for scalars.

        Per AAP Section 0.1.1: All return types must be numpy.ndarray or
        float as contextually appropriate.
        """
        y, X = regression_data
        b, tstat, s2, vcv, vcvwhite, R2, Rbar, yhat = ols(y, X, c=1)

        # Vector / matrix outputs: numpy.ndarray
        assert isinstance(b, np.ndarray), f"b type: {type(b)}"
        assert isinstance(tstat, np.ndarray), f"tstat type: {type(tstat)}"
        assert isinstance(vcv, np.ndarray), f"vcv type: {type(vcv)}"
        assert isinstance(vcvwhite, np.ndarray), f"vcvwhite type: {type(vcvwhite)}"
        assert isinstance(yhat, np.ndarray), f"yhat type: {type(yhat)}"

        # Scalar outputs: float (or numpy floating)
        assert isinstance(s2, (float, np.floating)), f"s2 type: {type(s2)}"
        assert isinstance(R2, (float, np.floating)), f"R2 type: {type(R2)}"
        assert isinstance(Rbar, (float, np.floating)), f"Rbar type: {type(Rbar)}"
