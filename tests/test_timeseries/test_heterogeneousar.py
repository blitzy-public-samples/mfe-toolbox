"""
Pytest tests for ``mfe_toolbox.timeseries.heterogeneousar`` — Heterogeneous
Autoregression (HAR) model estimator.

The HAR model (Corsi 2009) regresses a dependent variable on averages of its
own lagged values over different horizons (e.g. daily, weekly, monthly).  This
test module validates:

* Output structure (6-tuple, shapes, types, diagnostics keys)
* Statistical/functional correctness (vector vs matrix p, STANDARD vs MODIFIED)
* Variance-covariance matrix properties (symmetry, positive semi-definiteness)
* Edge cases (single lag, zero-padded errors, input validation)
* Numerical parity against MATLAB/Octave reference fixtures (±1e-6 / ±1e-4)

Per AAP Section 0.7.1: every migrated function MUST pass
``numpy.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-4)``
against MATLAB-generated fixtures.

Source MATLAB Reference: ``timeseries/heterogeneousar.m`` (Kevin Sheppard,
Revision 1, 7/13/2009).
"""

from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.timeseries.heterogeneousar import heterogeneousar

# ---------------------------------------------------------------------------
# Numerical Parity Tolerance Constants (per AAP Section 0.7.1)
# ---------------------------------------------------------------------------
ATOL: float = 1e-6
RTOL: float = 1e-4

# ---------------------------------------------------------------------------
# Fixture File Discovery
# ---------------------------------------------------------------------------
FIXTURE_DIR: Path = Path(__file__).parent.parent / "fixtures" / "timeseries"
FIXTURE_FILE: Path = FIXTURE_DIR / "heterogeneousar.npy"
HAS_FIXTURES: bool = FIXTURE_FILE.exists()


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def har_data(rng):
    """Simulate HAR-like data for functional/property tests.

    Generates a T=1000 time series with known HAR structure:
        y(t) = 0.2*y(t-1) + 0.3*mean(y(t-5:t)) + 0.4*mean(y(t-22:t)) + e(t)

    Uses the shared ``rng`` fixture (session-scoped, seed=42) from
    ``tests/conftest.py`` for reproducible random number generation.
    """
    T = 1000
    y = np.zeros(T)
    e = rng.standard_normal(T)
    for t in range(22, T):
        # Ref: AAP agent_prompt — HAR-like DGP
        y[t] = (0.2 * y[t - 1]
                + 0.3 * np.mean(y[t - 5:t])
                + 0.4 * np.mean(y[t - 22:t])
                + e[t])
    return y


@pytest.fixture(scope="session")
def fixture_data():
    """Load MATLAB/Octave reference fixture data for parity tests.

    Returns the full fixture dictionary if the fixture file exists,
    otherwise returns ``None``.  Individual parity tests are gated by
    ``@pytest.mark.skipif(not HAS_FIXTURES, ...)``.
    """
    if HAS_FIXTURES:
        return np.load(FIXTURE_FILE, allow_pickle=True).item()
    return None


# ===========================================================================
# Phase 2 — Output Structure Tests
# ===========================================================================

class TestOutputStructure:
    """Tests verifying the shape, type, and structure of heterogeneousar
    return values."""

    def test_returns_six_tuple(self, har_data):
        """heterogeneousar returns a 6-tuple:
        (parameters, errors, SEregression, diagnostics, VCVrobust, VCV)."""
        result = heterogeneousar(har_data, 1, np.array([1, 5, 22]))
        assert isinstance(result, tuple), "Return value must be a tuple"
        assert len(result) == 6, "Must return exactly 6 elements"

    def test_parameters_shape_with_constant(self, har_data):
        """With constant=1 and p=[1,5,22] → parameters shape is (4,)."""
        params, *_ = heterogeneousar(har_data, 1, np.array([1, 5, 22]))
        assert isinstance(params, np.ndarray)
        assert params.shape == (4,), (
            "Expected (4,) = 1 constant + 3 HAR coefficients"
        )

    def test_parameters_shape_without_constant(self, har_data):
        """With constant=0 and p=[1,5,22] → parameters shape is (3,)."""
        params, *_ = heterogeneousar(har_data, 0, np.array([1, 5, 22]))
        assert isinstance(params, np.ndarray)
        assert params.shape == (3,), (
            "Expected (3,) = 3 HAR coefficients, no constant"
        )

    def test_errors_shape(self, har_data):
        """errors.shape == (T,) where T = len(y)."""
        T = len(har_data)
        _, errors, *_ = heterogeneousar(har_data, 1, np.array([1, 5, 22]))
        assert isinstance(errors, np.ndarray)
        assert errors.shape == (T,), (
            f"Expected errors shape ({T},), got {errors.shape}"
        )

    def test_se_regression_positive(self, har_data):
        """SEregression must be a positive scalar."""
        _, _, se, *_ = heterogeneousar(har_data, 1, np.array([1, 5, 22]))
        assert isinstance(se, float), "SEregression must be a float"
        assert se > 0, "SEregression must be positive"

    def test_diagnostics_keys(self, har_data):
        """Diagnostics dict contains all required keys from the Python
        implementation (matching MATLAB struct fields)."""
        _, _, _, diag, *_ = heterogeneousar(
            har_data, 1, np.array([1, 5, 22])
        )
        assert isinstance(diag, dict), "diagnostics must be a dict"
        # Keys matching the Python implementation of heterogeneousar.py
        required_keys = {
            'P', 'C', 'T', 'adjT', 'spec',
            'AIC', 'SBIC', 'ARparameterization',
            'arroots', 'absarroots',
        }
        missing = required_keys - set(diag.keys())
        assert not missing, f"Missing diagnostics keys: {missing}"

    def test_vcv_shapes(self, har_data):
        """VCVrobust and VCV must be square matrices of dimension numX."""
        p = np.array([1, 5, 22])
        num_params = 1 + len(p)  # constant + 3 HAR coefficients = 4
        _, _, _, _, vcv_robust, vcv = heterogeneousar(har_data, 1, p)
        assert vcv_robust.shape == (num_params, num_params)
        assert vcv.shape == (num_params, num_params)


# ===========================================================================
# Phase 3 — Statistical / Functional Tests
# ===========================================================================

class TestStatisticalFunctional:
    """Tests verifying statistical and functional correctness of the HAR
    estimator across different parameterizations."""

    def test_standard_har_column_vector(self, har_data):
        """Standard HAR with column-vector p=[1;5;22] produces correct
        parameter count: 3 HAR + 1 constant = 4."""
        # Ref: heterogeneousar.m:15-16 — column vector format
        p_col = np.array([[1], [5], [22]])
        params, *_ = heterogeneousar(har_data, 1, p_col)
        assert params.shape == (4,), (
            "Column-vector p should yield 4 parameters (const + 3 HAR)"
        )

    def test_matrix_notation_equivalence(self, har_data):
        """Vector p=[1,5,22] and matrix p=[[1,1],[1,5],[1,22]] produce
        identical parameters.

        Ref: heterogeneousar.m:17-19 — matrix format is equivalent when
        start indices are all 1.
        """
        p_vec = np.array([1, 5, 22])
        p_mat = np.array([[1, 1], [1, 5], [1, 22]])
        params_vec, *_ = heterogeneousar(har_data, 1, p_vec)
        params_mat, *_ = heterogeneousar(har_data, 1, p_mat)
        npt.assert_allclose(
            params_vec, params_mat, atol=ATOL, rtol=RTOL,
            err_msg="Vector and matrix p notations must yield identical params"
        )

    def test_no_constant(self, har_data):
        """constant=0 excludes the intercept: parameter count = len(p)."""
        p = np.array([1, 5, 22])
        params, *_ = heterogeneousar(har_data, 0, p)
        assert params.shape == (3,), (
            "Without constant, parameter count should equal len(p)=3"
        )

    def test_skipping_lags_matrix(self, har_data):
        """Matrix p=[[1,1],[5,5],[1,22]] allows skipping intermediate lags.
        Produces 4 parameters with constant (const + lag1 + lag5 + avg1:22).

        Ref: heterogeneousar.m:19-21 — matrix format allows non-contiguous
        averaging windows.
        """
        p_skip = np.array([[1, 1], [5, 5], [1, 22]])
        params, *_ = heterogeneousar(har_data, 1, p_skip)
        assert params.shape == (4,), (
            "Skip-lag matrix p should yield 4 parameters (const + 3 terms)"
        )

    def test_modified_spec_valid_output(self, har_data):
        """MODIFIED spec produces valid parameters with correct shape.
        The non-overlapping reparameterization transforms the lag indicator
        matrix to non-overlapping windows via row-echelon elimination.

        Ref: heterogeneousar.m:152-202 — MODIFIED parameterization.
        """
        params_std, errors_std, se_std, diag_std, _, _ = heterogeneousar(
            har_data, 1, np.array([1, 5, 22]), spec='STANDARD'
        )
        params_mod, errors_mod, se_mod, diag_mod, _, _ = heterogeneousar(
            har_data, 1, np.array([1, 5, 22]), spec='MODIFIED'
        )
        # Same number of parameters regardless of spec
        assert params_mod.shape == params_std.shape
        # Modified produces valid SE
        assert se_mod > 0
        # Errors have same length
        assert errors_mod.shape == errors_std.shape
        # Spec is recorded in diagnostics
        assert diag_mod['spec'] == 'MODIFIED'

    def test_diagnostics_adjt_consistency(self, har_data):
        """diagnostics['adjT'] = T - max(max(p)) (effective sample after
        lag removal).

        Ref: heterogeneousar.m:234 — diagnostics.adjT = T (length after
        newlagmatrix removes maxP rows).
        """
        T = len(har_data)
        max_p = 22
        _, _, _, diag, _, _ = heterogeneousar(
            har_data, 1, np.array([1, 5, 22])
        )
        assert diag['T'] == T, "diagnostics['T'] must equal len(y)"
        assert diag['adjT'] == T - max_p, (
            f"diagnostics['adjT'] must equal T - maxP = {T} - {max_p}"
        )


# ===========================================================================
# Phase 4 — VCV Property Tests
# ===========================================================================

class TestVCVProperties:
    """Tests verifying mathematical properties of the variance-covariance
    matrices returned by the HAR estimator."""

    def test_vcv_symmetric(self, har_data):
        """Both VCV and VCVrobust must be symmetric matrices."""
        _, _, _, _, vcv_robust, vcv = heterogeneousar(
            har_data, 1, np.array([1, 5, 22])
        )
        npt.assert_allclose(
            vcv, vcv.T, atol=ATOL,
            err_msg="VCV must be symmetric"
        )
        npt.assert_allclose(
            vcv_robust, vcv_robust.T, atol=ATOL,
            err_msg="VCVrobust must be symmetric"
        )

    def test_vcv_positive_semidefinite(self, har_data):
        """VCV and VCVrobust must be positive semi-definite (all eigenvalues
        >= -ATOL).

        Uses ``np.linalg.eigvalsh`` for symmetric matrices.
        """
        _, _, _, _, vcv_robust, vcv = heterogeneousar(
            har_data, 1, np.array([1, 5, 22])
        )
        eigvals_vcv = np.linalg.eigvalsh(vcv)
        eigvals_robust = np.linalg.eigvalsh(vcv_robust)
        assert np.all(eigvals_vcv >= -ATOL), (
            f"VCV has negative eigenvalue: {eigvals_vcv.min()}"
        )
        assert np.all(eigvals_robust >= -ATOL), (
            f"VCVrobust has negative eigenvalue: {eigvals_robust.min()}"
        )

    def test_nw_effect_on_robust_vcv(self, har_data):
        """nw=0 (White) vs nw=5 (Newey-West): VCVrobust matrices must
        differ because the long-run variance estimator changes.

        Ref: heterogeneousar.m:222 — B = covnw(s, nw, 0); nw changes B.
        """
        _, _, _, _, vcv_robust_0, _ = heterogeneousar(
            har_data, 1, np.array([1, 5, 22]), nw=0
        )
        _, _, _, _, vcv_robust_5, _ = heterogeneousar(
            har_data, 1, np.array([1, 5, 22]), nw=5
        )
        assert not np.allclose(vcv_robust_0, vcv_robust_5), (
            "VCVrobust with nw=0 and nw=5 must differ"
        )


# ===========================================================================
# Phase 5 — Edge Case Tests
# ===========================================================================

class TestEdgeCases:
    """Tests verifying correct behavior at boundary conditions and with
    minimal inputs."""

    def test_single_lag(self, har_data):
        """p=[1] (single lag): AR(1) equivalent. Returns 2 params with
        constant (intercept + 1 lag coefficient).

        Ref: heterogeneousar.m handles single-lag vector p=[1] by converting
        to matrix form [[1,1]].
        """
        params, _, se, diag, vcv_r, vcv = heterogeneousar(
            har_data, 1, np.array([1])
        )
        assert params.shape == (2,), (
            "Single lag with constant should give 2 parameters"
        )
        assert se > 0
        assert vcv.shape == (2, 2)
        assert vcv_r.shape == (2, 2)

    def test_errors_first_maxp_zero(self, har_data):
        """For p=[1,5,22], first max(max(p))=22 elements of errors are zero;
        the remaining elements are non-zero residuals.

        Ref: heterogeneousar.m:226 — errors = [zeros(length(y)-T,1); errors]
        """
        max_p = 22
        _, errors, *_ = heterogeneousar(
            har_data, 1, np.array([1, 5, 22])
        )
        # First maxP entries must be exactly zero (padding)
        npt.assert_array_equal(
            errors[:max_p], np.zeros(max_p),
            err_msg="First max(P) error elements must be zero"
        )
        # Remaining entries should contain non-zero residuals
        assert not np.all(errors[max_p:] == 0.0), (
            "Post-padding errors should not all be zero"
        )

    def test_large_lag(self, har_data):
        """p=[1,5,50] with T=1000: still produces valid results with
        larger maxP, reducing effective sample size accordingly."""
        max_p = 50
        T = len(har_data)
        params, errors, se, diag, _, _ = heterogeneousar(
            har_data, 1, np.array([1, 5, 50])
        )
        assert params.shape == (4,)
        assert errors.shape == (T,)
        assert se > 0
        assert diag['adjT'] == T - max_p

    def test_two_lag_p(self, har_data):
        """p=[1,5] (two lags only): produces 3 params with constant."""
        params, *_ = heterogeneousar(har_data, 1, np.array([1, 5]))
        assert params.shape == (3,), (
            "Two lags with constant should give 3 parameters"
        )


# ===========================================================================
# Phase 5b — Input Validation Tests
# ===========================================================================

class TestInputValidation:
    """Tests verifying that invalid inputs raise appropriate exceptions.

    Per AAP Section 0.7.1: If MATLAB errors on invalid input, the Python
    equivalent must raise an equivalent exception.
    """

    def test_invalid_constant_value(self):
        """constant=2 raises ValueError.
        Ref: heterogeneousar.m:123-124 — ismember(constant,[0 1])"""
        y = np.random.default_rng(99).standard_normal(100)
        with pytest.raises(ValueError, match="CONSTANT must be 0 or 1"):
            heterogeneousar(y, 2, np.array([1, 5, 22]))

    def test_invalid_p_negative(self):
        """Negative values in p raise ValueError.
        Ref: heterogeneousar.m:112-113 — min(min(p))<1"""
        y = np.random.default_rng(99).standard_normal(100)
        with pytest.raises(ValueError, match="positive integers"):
            heterogeneousar(y, 1, np.array([-1, 5, 22]))

    def test_invalid_p_non_integer(self):
        """Non-integer values in p raise ValueError.
        Ref: heterogeneousar.m:112 — any(any(p~=floor(p)))"""
        y = np.random.default_rng(99).standard_normal(100)
        with pytest.raises(ValueError, match="positive integers"):
            heterogeneousar(y, 1, np.array([1.5, 5, 22]))

    def test_invalid_nw_negative(self):
        """Negative nw raises ValueError.
        Ref: heterogeneousar.m:133 — nw<0"""
        y = np.random.default_rng(99).standard_normal(100)
        with pytest.raises(ValueError, match="non-negative integer"):
            heterogeneousar(y, 1, np.array([1, 5, 22]), nw=-1)

    def test_invalid_spec_string(self):
        """Invalid spec string raises ValueError.
        Ref: heterogeneousar.m:144 — ~ismember(spec,{'STANDARD','MODIFIED'})"""
        y = np.random.default_rng(99).standard_normal(100)
        with pytest.raises(ValueError, match="SPEC must be"):
            heterogeneousar(y, 1, np.array([1, 5, 22]), spec='INVALID')

    def test_y_must_be_column(self):
        """Multi-column y raises ValueError.
        Ref: heterogeneousar.m:91 — size(y,2)>1"""
        y_wide = np.random.default_rng(99).standard_normal((100, 2))
        with pytest.raises(ValueError, match="column vector"):
            heterogeneousar(y_wide, 1, np.array([1, 5, 22]))


# ===========================================================================
# Phase 6 — MATLAB Parity Tests
# ===========================================================================

class TestMATLABParity:
    """Numerical parity tests against MATLAB/Octave-generated reference
    fixtures.

    Per AAP Section 0.7.1: All assertions use
    ``npt.assert_allclose(actual, expected, atol=1e-6, rtol=1e-4)``.

    Fixture structure (heterogeneousar.npy):
    - 'input_y': (1000,) test input data generated in Octave
    - 'standard_har_1_5_22': Standard HAR(1,5,22) reference outputs
    - 'modified_har_1_5_22': Modified HAR(1,5,22) reference outputs
    - 'har_nw5_1_5_22': HAR(1,5,22) with NW(5) reference outputs
    - 'matrix_p_har_1_5_22': Matrix-format p HAR reference outputs
    """

    @pytest.mark.skipif(
        not HAS_FIXTURES,
        reason="Fixture file not found: heterogeneousar.npy"
    )
    def test_parity_standard_har(self, fixture_data):
        """Standard HAR(1,5,22) parity: parameters, errors, SE, VCVs."""
        ref = fixture_data['standard_har_1_5_22']
        y = fixture_data['input_y']

        params, errors, se, diag, vcv_robust, vcv = heterogeneousar(
            y, ref['constant'], ref['p'],
            nw=ref['nw'], spec=ref['spec']
        )

        npt.assert_allclose(
            params, ref['parameters'], atol=ATOL, rtol=RTOL,
            err_msg="Standard HAR parameters mismatch"
        )
        npt.assert_allclose(
            errors, ref['errors'], atol=ATOL, rtol=RTOL,
            err_msg="Standard HAR errors mismatch"
        )
        npt.assert_allclose(
            se, ref['SEregression'], atol=ATOL, rtol=RTOL,
            err_msg="Standard HAR SEregression mismatch"
        )
        npt.assert_allclose(
            vcv_robust, ref['VCVrobust'], atol=ATOL, rtol=RTOL,
            err_msg="Standard HAR VCVrobust mismatch"
        )
        npt.assert_allclose(
            vcv, ref['VCV'], atol=ATOL, rtol=RTOL,
            err_msg="Standard HAR VCV mismatch"
        )
        # Verify diagnostics scalars
        assert diag['T'] == ref['T']
        assert diag['adjT'] == ref['adjT']
        assert diag['C'] == ref['C']
        npt.assert_allclose(
            diag['AIC'], ref['AIC'], atol=ATOL, rtol=RTOL,
            err_msg="Standard HAR AIC mismatch"
        )
        npt.assert_allclose(
            diag['SBIC'], ref['SBIC'], atol=ATOL, rtol=RTOL,
            err_msg="Standard HAR SBIC mismatch"
        )

    @pytest.mark.skipif(
        not HAS_FIXTURES,
        reason="Fixture file not found: heterogeneousar.npy"
    )
    def test_parity_modified_har(self, fixture_data):
        """Modified HAR(1,5,22) parity: non-overlapping reparameterization."""
        ref = fixture_data['modified_har_1_5_22']
        y = fixture_data['input_y']

        params, errors, se, diag, vcv_robust, vcv = heterogeneousar(
            y, ref['constant'], ref['p'],
            nw=ref['nw'], spec=ref['spec']
        )

        npt.assert_allclose(
            params, ref['parameters'], atol=ATOL, rtol=RTOL,
            err_msg="Modified HAR parameters mismatch"
        )
        npt.assert_allclose(
            errors, ref['errors'], atol=ATOL, rtol=RTOL,
            err_msg="Modified HAR errors mismatch"
        )
        npt.assert_allclose(
            se, ref['SEregression'], atol=ATOL, rtol=RTOL,
            err_msg="Modified HAR SEregression mismatch"
        )
        npt.assert_allclose(
            vcv_robust, ref['VCVrobust'], atol=ATOL, rtol=RTOL,
            err_msg="Modified HAR VCVrobust mismatch"
        )
        npt.assert_allclose(
            vcv, ref['VCV'], atol=ATOL, rtol=RTOL,
            err_msg="Modified HAR VCV mismatch"
        )
        npt.assert_allclose(
            diag['AIC'], ref['AIC'], atol=ATOL, rtol=RTOL,
            err_msg="Modified HAR AIC mismatch"
        )
        npt.assert_allclose(
            diag['SBIC'], ref['SBIC'], atol=ATOL, rtol=RTOL,
            err_msg="Modified HAR SBIC mismatch"
        )

    @pytest.mark.skipif(
        not HAS_FIXTURES,
        reason="Fixture file not found: heterogeneousar.npy"
    )
    def test_parity_nw5(self, fixture_data):
        """HAR(1,5,22) with Newey-West nw=5 parity: VCVrobust must
        differ from nw=0 and match fixture exactly."""
        ref = fixture_data['har_nw5_1_5_22']
        y = fixture_data['input_y']

        params, errors, se, diag, vcv_robust, vcv = heterogeneousar(
            y, ref['constant'], ref['p'],
            nw=ref['nw'], spec=ref['spec']
        )

        # Parameters and non-robust VCV should be identical to standard HAR
        # (NW only affects VCVrobust)
        npt.assert_allclose(
            params, ref['parameters'], atol=ATOL, rtol=RTOL,
            err_msg="NW5 HAR parameters mismatch"
        )
        npt.assert_allclose(
            errors, ref['errors'], atol=ATOL, rtol=RTOL,
            err_msg="NW5 HAR errors mismatch"
        )
        npt.assert_allclose(
            se, ref['SEregression'], atol=ATOL, rtol=RTOL,
            err_msg="NW5 HAR SEregression mismatch"
        )
        npt.assert_allclose(
            vcv_robust, ref['VCVrobust'], atol=ATOL, rtol=RTOL,
            err_msg="NW5 HAR VCVrobust mismatch"
        )
        npt.assert_allclose(
            vcv, ref['VCV'], atol=ATOL, rtol=RTOL,
            err_msg="NW5 HAR VCV mismatch"
        )

    @pytest.mark.skipif(
        not HAS_FIXTURES,
        reason="Fixture file not found: heterogeneousar.npy"
    )
    def test_parity_matrix_p_format(self, fixture_data):
        """Matrix-format p=[[1,1],[1,5],[1,22]] parity test.

        Ref: heterogeneousar.m:17-19 — Matrix format should produce
        identical results to vector format p=[1,5,22].
        """
        ref = fixture_data['matrix_p_har_1_5_22']
        y = fixture_data['input_y']

        params, errors, se, diag, vcv_robust, vcv = heterogeneousar(
            y, ref['constant'], ref['p'],
            nw=ref['nw'], spec=ref['spec']
        )

        npt.assert_allclose(
            params, ref['parameters'], atol=ATOL, rtol=RTOL,
            err_msg="Matrix-p HAR parameters mismatch"
        )
        npt.assert_allclose(
            errors, ref['errors'], atol=ATOL, rtol=RTOL,
            err_msg="Matrix-p HAR errors mismatch"
        )
        npt.assert_allclose(
            se, ref['SEregression'], atol=ATOL, rtol=RTOL,
            err_msg="Matrix-p HAR SEregression mismatch"
        )
        npt.assert_allclose(
            vcv_robust, ref['VCVrobust'], atol=ATOL, rtol=RTOL,
            err_msg="Matrix-p HAR VCVrobust mismatch"
        )
        npt.assert_allclose(
            vcv, ref['VCV'], atol=ATOL, rtol=RTOL,
            err_msg="Matrix-p HAR VCV mismatch"
        )

    @pytest.mark.skipif(
        not HAS_FIXTURES,
        reason="Fixture file not found: heterogeneousar.npy"
    )
    def test_parity_standard_vs_matrix_identical(self, fixture_data):
        """Vector and matrix p formats produce identical MATLAB fixture
        outputs, confirming format equivalence.

        Ref: heterogeneousar.m:105 — p=[ones(size(p)) p] converts vector
        to matrix.
        """
        ref_vec = fixture_data['standard_har_1_5_22']
        ref_mat = fixture_data['matrix_p_har_1_5_22']

        npt.assert_allclose(
            ref_vec['parameters'], ref_mat['parameters'],
            atol=ATOL, rtol=RTOL,
            err_msg="MATLAB vector and matrix p fixtures must be identical"
        )
        npt.assert_allclose(
            ref_vec['errors'], ref_mat['errors'],
            atol=ATOL, rtol=RTOL,
            err_msg="MATLAB vector and matrix p errors must be identical"
        )
