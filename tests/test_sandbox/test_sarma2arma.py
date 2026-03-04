"""Pytest parity and unit tests for SARMA-to-ARMA polynomial conversion.

Tests the ``mfe_toolbox.sandbox.sarma2arma.sarma2arma`` function migrated from
MATLAB ``sandbox/sarma2arma.m``.  Verifies numerical parity (±1e-6 atol,
±1e-4 rtol) against MATLAB/Octave reference fixtures stored as ``.npy`` files
in ``tests/fixtures/sandbox/``.

The SARMA-to-ARMA conversion expands seasonal lag polynomials via convolution::

    (1 - φ₁B - φ₂B² - ...)(1 - Φ₁B^S - Φ₂B^{2S} - ...)

producing an equivalent non-seasonal ARMA representation.

Test Coverage:
    - Import verification
    - Pure AR / pure MA passthrough (no seasonal component)
    - AR(1)+SAR(1), MA(1)+SMA(1) seasonal expansion at S=12
    - Full ARMA(1,1)+SARMA(1,1)_12
    - Higher-order AR(2)+SAR(1) at S=4
    - Polynomial convolution correctness
    - Multiple seasonal period groups
    - Empty seasonal passthrough
    - Input validation / error handling
    - MATLAB fixture parity (6 reference configurations)
    - Return type, shape, sorting, and positivity invariants

Ref: AAP Section 0.7.1 — ATOL=1e-6, RTOL=1e-4 for all parity comparisons.
Ref: AAP Section 0.7.3 — MATLAB conv → np.convolve; find → np.nonzero.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
from pathlib import Path

from mfe_toolbox.sandbox.sarma2arma import sarma2arma

# ---------------------------------------------------------------------------
# Tolerance constants imported from conftest (also defined locally for clarity)
# ---------------------------------------------------------------------------
# Per AAP Section 0.7.1 — hard numerical parity thresholds
from tests.conftest import ATOL, RTOL, load_fixture_npy


# ============================================================================
# Phase 1: Unit Tests — No Fixture Dependency
# ============================================================================


class TestSarma2armaImports:
    """Verify that the module and function are importable."""

    def test_sarma2arma_imports(self) -> None:
        """sarma2arma should be importable from mfe_toolbox.sandbox.sarma2arma."""
        from mfe_toolbox.sandbox.sarma2arma import sarma2arma as fn
        assert callable(fn), "sarma2arma must be callable"

    def test_convert_sar_to_ar_imports(self) -> None:
        """Private helper _convert_sar_to_ar should be importable for testing."""
        from mfe_toolbox.sandbox.sarma2arma import _convert_sar_to_ar
        assert callable(_convert_sar_to_ar), "_convert_sar_to_ar must be callable"


class TestSarma2armaPureAR:
    """Pure AR models without seasonal components."""

    def test_sarma2arma_pure_ar1_no_seasonal(self) -> None:
        """AR(1) without seasonal: parameters, arP, maQ unchanged.

        parameters = [0.5] (phi_1=0.5)
        p = [1], q = [], seasonal has zero seasonal orders.
        Expected: parameters=[0.5], arP=[1], maQ=[].
        """
        params = np.array([0.5])
        p = np.array([1.0])
        q = np.array([], dtype=float)
        # seasonal with zero seasonal AR/MA orders (no expansion)
        seasonal = np.array([[0.0, 0.0, 0.0, 12.0]])

        out_params, ar_p, ma_q = sarma2arma(params, p, q, seasonal)

        npt.assert_allclose(out_params, np.array([0.5]), atol=ATOL, rtol=RTOL)
        npt.assert_array_equal(ar_p, np.array([1]))
        assert len(ma_q) == 0, "maQ should be empty for pure AR"

    def test_sarma2arma_pure_ar2_no_seasonal(self) -> None:
        """AR(2) without seasonal: parameters and arP unchanged.

        parameters = [0.5, 0.3], p = [1, 2], q = [].
        """
        params = np.array([0.5, 0.3])
        p = np.array([1.0, 2.0])
        q = np.array([], dtype=float)
        seasonal = np.array([[0.0, 0.0, 0.0, 12.0]])

        out_params, ar_p, ma_q = sarma2arma(params, p, q, seasonal)

        npt.assert_allclose(out_params, np.array([0.5, 0.3]), atol=ATOL, rtol=RTOL)
        npt.assert_array_equal(ar_p, np.array([1, 2]))
        assert len(ma_q) == 0, "maQ should be empty for pure AR"


class TestSarma2armaPureMA:
    """Pure MA models without seasonal components."""

    def test_sarma2arma_pure_ma1_no_seasonal(self) -> None:
        """MA(1) without seasonal: parameters, arP empty, maQ=[1].

        parameters = [0.3] (theta_1=0.3)
        p = [], q = [1], seasonal has zero seasonal orders.
        """
        params = np.array([0.3])
        p = np.array([], dtype=float)
        q = np.array([1.0])
        seasonal = np.array([[0.0, 0.0, 0.0, 12.0]])

        out_params, ar_p, ma_q = sarma2arma(params, p, q, seasonal)

        npt.assert_allclose(out_params, np.array([0.3]), atol=ATOL, rtol=RTOL)
        assert len(ar_p) == 0, "arP should be empty for pure MA"
        npt.assert_array_equal(ma_q, np.array([1]))


class TestSarma2armaSeasonalExpansion:
    """Seasonal AR/MA expansion via polynomial convolution."""

    def test_sarma2arma_ar1_sar1_s12(self) -> None:
        """AR(1) + SAR(1) at S=12: expand (1-0.5B)(1-0.3B^12).

        Ref: sarma2arma.m — convolution of non-seasonal and seasonal lag polys.
        Input: parameters=[0.5, 0.3], p=[1], q=[], seasonal=[[1,0,0,12]]
        Expected arP=[1, 12, 13] with coefficients [0.5, 0.3, -0.15].
        """
        params = np.array([0.5, 0.3])
        p = np.array([1.0])
        q = np.array([], dtype=float)
        seasonal = np.array([[1.0, 0.0, 0.0, 12.0]])

        out_params, ar_p, ma_q = sarma2arma(params, p, q, seasonal)

        # Expanded AR parameters from convolution
        expected_params = np.array([0.5, 0.3, -0.15])
        expected_arP = np.array([1, 12, 13])

        npt.assert_allclose(out_params, expected_params, atol=ATOL, rtol=RTOL)
        npt.assert_array_equal(ar_p, expected_arP)
        assert len(ma_q) == 0, "maQ should be empty (no MA terms)"

    def test_sarma2arma_ma1_sma1_s12(self) -> None:
        """MA(1) + SMA(1) at S=12: expand (1-0.4B)(1-0.2B^12).

        Input: parameters=[0.4, 0.2], p=[], q=[1], seasonal=[[0,0,1,12]]
        Expected maQ=[1, 12, 13] with coefficients [0.4, 0.2, -0.08].
        """
        params = np.array([0.4, 0.2])
        p = np.array([], dtype=float)
        q = np.array([1.0])
        seasonal = np.array([[0.0, 0.0, 1.0, 12.0]])

        out_params, ar_p, ma_q = sarma2arma(params, p, q, seasonal)

        expected_params = np.array([0.4, 0.2, -0.08])
        expected_maQ = np.array([1, 12, 13])

        npt.assert_allclose(out_params, expected_params, atol=ATOL, rtol=RTOL)
        assert len(ar_p) == 0, "arP should be empty (no AR terms)"
        npt.assert_array_equal(ma_q, expected_maQ)

    def test_sarma2arma_arma11_sarma11_s12(self) -> None:
        """ARMA(1,1) + SARMA(1,1)_12: both AR and MA seasonal expansion.

        Input: parameters=[0.5, 0.3, 0.4, 0.2], p=[1], q=[1],
               seasonal=[[1,0,1,12]]
        Expected: AR expanded to lags [1,12,13], MA expanded to lags [1,12,13].
        """
        params = np.array([0.5, 0.3, 0.4, 0.2])
        p = np.array([1.0])
        q = np.array([1.0])
        seasonal = np.array([[1.0, 0.0, 1.0, 12.0]])

        out_params, ar_p, ma_q = sarma2arma(params, p, q, seasonal)

        # AR side: (1-0.5B)(1-0.3B^12) → coeffs [0.5, 0.3, -0.15]
        # MA side: (1-0.4B)(1-0.2B^12) → coeffs [0.4, 0.2, -0.08]
        expected_params = np.array([0.5, 0.3, -0.15, 0.4, 0.2, -0.08])
        expected_arP = np.array([1, 12, 13])
        expected_maQ = np.array([1, 12, 13])

        npt.assert_allclose(out_params, expected_params, atol=ATOL, rtol=RTOL)
        npt.assert_array_equal(ar_p, expected_arP)
        npt.assert_array_equal(ma_q, expected_maQ)

    def test_sarma2arma_ar2_sar1_s4(self) -> None:
        """AR(2) + SAR(1) at S=4: convolution of (1-0.3B+0.1B^2)(1-0.5B^4).

        Ref: sarma2arma.m — tests higher-order non-seasonal with seasonal.
        Input: parameters=[0.3, -0.1, 0.5], p=[1,2], q=[], seasonal=[[1,0,0,4]]

        Polynomial convolution:
          AR poly: [1, -0.3, 0.1]  (at lags 0, 1, 2)
          SAR poly: [1, 0, 0, 0, -0.5]  (at lags 0, 1, 2, 3, 4)
          Combined: [1, -0.3, 0.1, 0, -0.5, 0.15, -0.05]

        After removing lag 0 and extracting nonzero:
          lags = [1, 2, 4, 5, 6], coeffs = [0.3, -0.1, 0.5, -0.15, 0.05]
        """
        params = np.array([0.3, -0.1, 0.5])
        p = np.array([1.0, 2.0])
        q = np.array([], dtype=float)
        seasonal = np.array([[1.0, 0.0, 0.0, 4.0]])

        out_params, ar_p, ma_q = sarma2arma(params, p, q, seasonal)

        expected_params = np.array([0.3, -0.1, 0.5, -0.15, 0.05])
        expected_arP = np.array([1, 2, 4, 5, 6])

        npt.assert_allclose(out_params, expected_params, atol=ATOL, rtol=RTOL)
        npt.assert_array_equal(ar_p, expected_arP)
        assert len(ma_q) == 0, "maQ should be empty (no MA terms)"


class TestSarma2armaConvolution:
    """Verify polynomial convolution correctness independently."""

    def test_sarma2arma_convolution_correctness(self) -> None:
        """Verify expanded polynomial via independent np.convolve reference.

        For AR(1) with phi=0.5 and SAR(1) with Phi=0.3 at S=4:
          AR poly: [1, -0.5]
          SAR poly: [1, 0, 0, 0, -0.3]
          conv result: [1, -0.5, 0, 0, -0.3, 0.15]

        After removing constant and extracting nonzero lags:
          lags [1, 4, 5], coefficients [0.5, 0.3, -0.15]
        """
        params = np.array([0.5, 0.3])
        p = np.array([1.0])
        q = np.array([], dtype=float)
        seasonal = np.array([[1.0, 0.0, 0.0, 4.0]])

        out_params, ar_p, ma_q = sarma2arma(params, p, q, seasonal)

        # Independently verify via np.convolve
        ar_poly = np.array([1.0, -0.5])
        sar_poly = np.array([1.0, 0.0, 0.0, 0.0, -0.3])
        combined = np.convolve(ar_poly, sar_poly)

        # Remove constant term (index 0) and find nonzero
        truncated = combined[1:]
        ref_nonzero_idx = np.nonzero(truncated)[0]
        ref_lags = ref_nonzero_idx + 1  # Convert to 1-based
        ref_coeffs = -truncated[ref_nonzero_idx]  # Negate convention

        npt.assert_allclose(out_params, ref_coeffs, atol=ATOL, rtol=RTOL)
        npt.assert_array_equal(ar_p, ref_lags)

    def test_sarma2arma_convolution_independent_verification(self) -> None:
        """Cross-check expanded params match manual polynomial multiplication.

        (1 - 0.5*B)(1 - 0.3*B^12) = 1 - 0.5*B - 0.3*B^12 + 0.15*B^13

        Coefficients at each lag (with sign convention):
          lag 1: 0.5, lag 12: 0.3, lag 13: -0.15
        """
        params = np.array([0.5, 0.3])
        p = np.array([1.0])
        q = np.array([], dtype=float)
        seasonal = np.array([[1.0, 0.0, 0.0, 12.0]])

        out_params, ar_p, _ = sarma2arma(params, p, q, seasonal)

        # Manual computation: phi_1 = 0.5, Phi_1 = 0.3
        # Expanded: coeff at lag 1 = phi_1 = 0.5
        #           coeff at lag 12 = Phi_1 = 0.3
        #           coeff at lag 13 = -(phi_1 * Phi_1) = -(0.5 * 0.3) = -0.15
        expected_coeffs = np.array([0.5, 0.3, -0.15])
        expected_lags = np.array([1, 12, 13])

        npt.assert_allclose(out_params, expected_coeffs, atol=ATOL, rtol=RTOL)
        npt.assert_array_equal(ar_p, expected_lags)


class TestSarma2armaParameterCount:
    """Verify output parameter count matches expanded lag structure."""

    def test_sarma2arma_parameter_count(self) -> None:
        """Output parameter count must equal len(arP) + len(maQ).

        For ARMA(1,1)+SARMA(1,1)_12: input 4 params → output 6 params.
        """
        params = np.array([0.5, 0.3, 0.4, 0.2])
        p = np.array([1.0])
        q = np.array([1.0])
        seasonal = np.array([[1.0, 0.0, 1.0, 12.0]])

        out_params, ar_p, ma_q = sarma2arma(params, p, q, seasonal)

        expected_count = len(ar_p) + len(ma_q)
        assert len(out_params) == expected_count, (
            f"Output param count {len(out_params)} != "
            f"len(arP) + len(maQ) = {expected_count}"
        )

    def test_sarma2arma_parameter_count_pure_ar(self) -> None:
        """Pure AR: output param count = len(arP)."""
        params = np.array([0.5])
        p = np.array([1.0])
        q = np.array([], dtype=float)
        seasonal = np.array([[0.0, 0.0, 0.0, 12.0]])

        out_params, ar_p, ma_q = sarma2arma(params, p, q, seasonal)

        assert len(out_params) == len(ar_p) + len(ma_q)
        assert len(ma_q) == 0

    def test_sarma2arma_parameter_count_higher_order(self) -> None:
        """Higher-order SAR(2)_12: 1 AR + 2 SAR → 5 expanded AR params."""
        params = np.array([0.5, 0.3, 0.1])
        p = np.array([1.0])
        q = np.array([], dtype=float)
        seasonal = np.array([[2.0, 0.0, 0.0, 12.0]])

        out_params, ar_p, ma_q = sarma2arma(params, p, q, seasonal)

        # (1-0.5B)(1-0.3B^12-0.1B^24) → 5 nonzero lags
        assert len(out_params) == 5
        assert len(ar_p) == 5
        assert len(ma_q) == 0


class TestSarma2armaEmptySeasonal:
    """Empty or zero-order seasonal array (pure ARMA passthrough)."""

    def test_sarma2arma_empty_seasonal(self) -> None:
        """Seasonal with zero AR/MA orders: pure ARMA passthrough.

        seasonal = [[0, 0, 0, 12]] → no expansion applied.
        """
        params = np.array([0.5, 0.3, 0.4])
        p = np.array([1.0, 2.0])
        q = np.array([1.0])
        seasonal = np.array([[0.0, 0.0, 0.0, 12.0]])

        out_params, ar_p, ma_q = sarma2arma(params, p, q, seasonal)

        npt.assert_allclose(out_params, np.array([0.5, 0.3, 0.4]),
                            atol=ATOL, rtol=RTOL)
        npt.assert_array_equal(ar_p, np.array([1, 2]))
        npt.assert_array_equal(ma_q, np.array([1]))

    def test_sarma2arma_zero_rows_seasonal(self) -> None:
        """Seasonal with 0 rows (empty 2D array): pure ARMA passthrough.

        This is an edge case where no seasonal component is specified at all.
        """
        params = np.array([0.5, 0.3])
        p = np.array([1.0])
        q = np.array([1.0])
        seasonal = np.empty((0, 4))

        out_params, ar_p, ma_q = sarma2arma(params, p, q, seasonal)

        npt.assert_allclose(out_params, np.array([0.5, 0.3]),
                            atol=ATOL, rtol=RTOL)
        npt.assert_array_equal(ar_p, np.array([1]))
        npt.assert_array_equal(ma_q, np.array([1]))


class TestSarma2armaMultipleSeasonalGroups:
    """Multiple seasonal period specifications."""

    def test_sarma2arma_multiple_seasonal_groups(self) -> None:
        """Two seasonal groups: quarterly (S=4) + monthly (S=12).

        Input: parameters=[0.5, 0.3, 0.2, 0.4, 0.1, 0.15], p=[1], q=[1]
               seasonal=[[1,0,1,4],[1,0,1,12]]

        The iterative convolution expands both seasonal components:
          AR: (1-0.5B)(1-0.3B^4)(1-0.2B^12)
          MA: (1-0.4B)(1-0.1B^4)(1-0.15B^12)
        """
        params = np.array([0.5, 0.3, 0.2, 0.4, 0.1, 0.15])
        p = np.array([1.0])
        q = np.array([1.0])
        seasonal = np.array([[1.0, 0.0, 1.0, 4.0],
                             [1.0, 0.0, 1.0, 12.0]])

        out_params, ar_p, ma_q = sarma2arma(params, p, q, seasonal)

        # Expected from MATLAB fixture (TC4)
        expected_ar_params = np.array([0.5, 0.3, -0.15, 0.2, -0.1, -0.06, 0.03])
        expected_ma_params = np.array([0.4, 0.1, -0.04, 0.15, -0.06, -0.015, 0.006])
        expected_params = np.concatenate([expected_ar_params, expected_ma_params])
        expected_arP = np.array([1, 4, 5, 12, 13, 16, 17])
        expected_maQ = np.array([1, 4, 5, 12, 13, 16, 17])

        npt.assert_allclose(out_params, expected_params, atol=ATOL, rtol=RTOL)
        npt.assert_array_equal(ar_p, expected_arP)
        npt.assert_array_equal(ma_q, expected_maQ)


class TestSarma2armaHigherOrder:
    """Higher-order seasonal components."""

    def test_sarma2arma_higher_order_sar2_s12(self) -> None:
        """SAR(2) at S=12: quadratic seasonal polynomial.

        Input: parameters=[0.5, 0.3, 0.1], p=[1], q=[], seasonal=[[2,0,0,12]]
        (1-0.5B)(1-0.3B^12-0.1B^24) produces 5 expanded lags.
        """
        params = np.array([0.5, 0.3, 0.1])
        p = np.array([1.0])
        q = np.array([], dtype=float)
        seasonal = np.array([[2.0, 0.0, 0.0, 12.0]])

        out_params, ar_p, ma_q = sarma2arma(params, p, q, seasonal)

        # Convolution: [1,-0.5] * [1, 0,...,0, -0.3, 0,...,0, -0.1]
        # = [1, -0.5, 0,..., -0.3, 0.15, 0,..., -0.1, 0.05]
        expected_params = np.array([0.5, 0.3, -0.15, 0.1, -0.05])
        expected_arP = np.array([1, 12, 13, 24, 25])

        npt.assert_allclose(out_params, expected_params, atol=ATOL, rtol=RTOL)
        npt.assert_array_equal(ar_p, expected_arP)
        assert len(ma_q) == 0


# ============================================================================
# Phase 2: Input Validation Tests
# ============================================================================


class TestSarma2armaInputValidation:
    """Test error handling for invalid inputs."""

    def test_sarma2arma_invalid_seasonal_shape(self) -> None:
        """seasonal with fewer than 4 columns should raise ValueError.

        Ref: sarma2arma.py line 246-250 — explicit shape validation.
        """
        params = np.array([0.5, 0.3])
        p = np.array([1.0])
        q = np.array([], dtype=float)
        # seasonal with only 3 columns (missing S column)
        seasonal_bad = np.array([[1.0, 0.0, 0.0]])

        with pytest.raises(ValueError, match="at least 4 columns"):
            sarma2arma(params, p, q, seasonal_bad)

    def test_sarma2arma_invalid_seasonal_2cols(self) -> None:
        """seasonal with only 2 columns should raise ValueError."""
        params = np.array([0.5])
        p = np.array([1.0])
        q = np.array([], dtype=float)
        seasonal_bad = np.array([[1.0, 12.0]])

        with pytest.raises(ValueError, match="at least 4 columns"):
            sarma2arma(params, p, q, seasonal_bad)

    def test_sarma2arma_wrong_param_count_raises(self) -> None:
        """Parameter vector too short for specified lag structure should error.

        With p=[1,2], seasonal SAR(1)_12: need at least 3 params for AR side,
        but providing only 1 should cause an indexing error.
        """
        params = np.array([0.5])  # Too short: need p(2) + SAR(1) = 3 AR params
        p = np.array([1.0, 2.0])
        q = np.array([], dtype=float)
        seasonal = np.array([[1.0, 0.0, 0.0, 12.0]])

        # The code should raise some error (ValueError or IndexError)
        # when parameters cannot be partitioned correctly
        with pytest.raises((ValueError, IndexError)):
            sarma2arma(params, p, q, seasonal)

    def test_sarma2arma_1d_seasonal_reshaped(self) -> None:
        """A 1-D seasonal array should be auto-reshaped to (1, N).

        Ref: sarma2arma.py line 243-245 — if seasonal.ndim == 1, reshape.
        """
        params = np.array([0.5, 0.3])
        p = np.array([1.0])
        q = np.array([], dtype=float)
        # 1-D array with 4 elements (should be reshaped to (1, 4))
        seasonal_1d = np.array([1.0, 0.0, 0.0, 12.0])

        out_params, ar_p, ma_q = sarma2arma(params, p, q, seasonal_1d)

        expected_params = np.array([0.5, 0.3, -0.15])
        expected_arP = np.array([1, 12, 13])

        npt.assert_allclose(out_params, expected_params, atol=ATOL, rtol=RTOL)
        npt.assert_array_equal(ar_p, expected_arP)


# ============================================================================
# Phase 3: Parity Tests Against MATLAB Fixtures
# ============================================================================


def _load_sarma2arma_fixture(sandbox_fixture_dir: Path):
    """Load and parse the sarma2arma fixture file.

    Returns a list of test case dicts from the .npy fixture, or None if the
    fixture file cannot be found (triggering pytest.skip in the caller).
    """
    data = load_fixture_npy(sandbox_fixture_dir, "sarma2arma")
    fixture_dict = data.item()
    return fixture_dict["test_cases"]


@pytest.mark.parity
class TestSarma2armaParity:
    """MATLAB fixture-based parity tests."""

    def test_sarma2arma_parity_basic(self, sandbox_fixture_dir: Path) -> None:
        """TC1: Simple seasonal AR(1) S=12 — compare against MATLAB output.

        Loads sarma2arma.npy fixture, extracts test case 1 inputs/outputs,
        and compares Python sarma2arma output to MATLAB reference.
        """
        test_cases = _load_sarma2arma_fixture(sandbox_fixture_dir)
        tc = test_cases[0]  # TC1: simple_seasonal_ar1_s12

        assert tc["description"] == "simple_seasonal_ar1_s12"

        out_params, ar_p, ma_q = sarma2arma(
            tc["input_parameters"].copy(),
            tc["p_input"].copy(),
            tc["q_input"].copy(),
            tc["seasonal_input"].copy(),
        )

        # Parameter values: numerical parity
        npt.assert_allclose(out_params, tc["output_parameters"],
                            atol=ATOL, rtol=RTOL,
                            err_msg="TC1 output parameters mismatch")
        # Lag indices: exact integer match
        npt.assert_array_equal(ar_p, tc["arP_output"],
                               err_msg="TC1 arP mismatch")
        npt.assert_array_equal(ma_q, tc["maQ_output"],
                               err_msg="TC1 maQ mismatch")

    def test_sarma2arma_parity_seasonal_ma(self, sandbox_fixture_dir: Path) -> None:
        """TC2: Simple seasonal MA(1) S=12 — parity check."""
        test_cases = _load_sarma2arma_fixture(sandbox_fixture_dir)
        tc = test_cases[1]  # TC2: simple_seasonal_ma1_s12

        assert tc["description"] == "simple_seasonal_ma1_s12"

        out_params, ar_p, ma_q = sarma2arma(
            tc["input_parameters"].copy(),
            tc["p_input"].copy(),
            tc["q_input"].copy(),
            tc["seasonal_input"].copy(),
        )

        npt.assert_allclose(out_params, tc["output_parameters"],
                            atol=ATOL, rtol=RTOL,
                            err_msg="TC2 output parameters mismatch")
        npt.assert_array_equal(ar_p, tc["arP_output"],
                               err_msg="TC2 arP mismatch")
        npt.assert_array_equal(ma_q, tc["maQ_output"],
                               err_msg="TC2 maQ mismatch")

    def test_sarma2arma_parity_sarima_full(self, sandbox_fixture_dir: Path) -> None:
        """TC3: SARIMA(1,0,1)(1,0,1)_12 — full AR+MA seasonal parity."""
        test_cases = _load_sarma2arma_fixture(sandbox_fixture_dir)
        tc = test_cases[2]  # TC3: sarima_101_101_12

        assert tc["description"] == "sarima_101_101_12"

        out_params, ar_p, ma_q = sarma2arma(
            tc["input_parameters"].copy(),
            tc["p_input"].copy(),
            tc["q_input"].copy(),
            tc["seasonal_input"].copy(),
        )

        npt.assert_allclose(out_params, tc["output_parameters"],
                            atol=ATOL, rtol=RTOL,
                            err_msg="TC3 output parameters mismatch")
        npt.assert_array_equal(ar_p, tc["arP_output"],
                               err_msg="TC3 arP mismatch")
        npt.assert_array_equal(ma_q, tc["maQ_output"],
                               err_msg="TC3 maQ mismatch")

    def test_sarma2arma_parity_no_seasonal(self, sandbox_fixture_dir: Path) -> None:
        """TC5: No seasonal — passthrough parity check."""
        test_cases = _load_sarma2arma_fixture(sandbox_fixture_dir)
        tc = test_cases[4]  # TC5: no_seasonal_passthrough

        assert tc["description"] == "no_seasonal_passthrough"

        out_params, ar_p, ma_q = sarma2arma(
            tc["input_parameters"].copy(),
            tc["p_input"].copy(),
            tc["q_input"].copy(),
            tc["seasonal_input"].copy(),
        )

        npt.assert_allclose(out_params, tc["output_parameters"],
                            atol=ATOL, rtol=RTOL,
                            err_msg="TC5 output parameters mismatch")
        npt.assert_array_equal(ar_p, tc["arP_output"],
                               err_msg="TC5 arP mismatch")
        npt.assert_array_equal(ma_q, tc["maQ_output"],
                               err_msg="TC5 maQ mismatch")

    def test_sarma2arma_parity_higher_order(self, sandbox_fixture_dir: Path) -> None:
        """TC6: Higher-order SAR(2)_12 — parity check."""
        test_cases = _load_sarma2arma_fixture(sandbox_fixture_dir)
        tc = test_cases[5]  # TC6: higher_order_seasonal_ar2_s12

        assert tc["description"] == "higher_order_seasonal_ar2_s12"

        out_params, ar_p, ma_q = sarma2arma(
            tc["input_parameters"].copy(),
            tc["p_input"].copy(),
            tc["q_input"].copy(),
            tc["seasonal_input"].copy(),
        )

        npt.assert_allclose(out_params, tc["output_parameters"],
                            atol=ATOL, rtol=RTOL,
                            err_msg="TC6 output parameters mismatch")
        npt.assert_array_equal(ar_p, tc["arP_output"],
                               err_msg="TC6 arP mismatch")
        npt.assert_array_equal(ma_q, tc["maQ_output"],
                               err_msg="TC6 maQ mismatch")

    @pytest.mark.parametrize("tc_index", [0, 1, 2, 3, 4, 5],
                             ids=["TC1_ar1_sar1_s12",
                                  "TC2_ma1_sma1_s12",
                                  "TC3_sarima_101_101_12",
                                  "TC4_multiple_seasonal",
                                  "TC5_no_seasonal",
                                  "TC6_sar2_s12"])
    def test_sarma2arma_parity_multiple_configs(
        self, sandbox_fixture_dir: Path, tc_index: int
    ) -> None:
        """Parametrized parity test across all 6 MATLAB fixture configurations.

        For each test case:
          - Load fixture input and expected output
          - Call Python sarma2arma
          - Assert numerical parity for parameters (atol=1e-6, rtol=1e-4)
          - Assert exact match for integer lag indices (arP, maQ)
        """
        test_cases = _load_sarma2arma_fixture(sandbox_fixture_dir)
        tc = test_cases[tc_index]

        out_params, ar_p, ma_q = sarma2arma(
            tc["input_parameters"].copy(),
            tc["p_input"].copy(),
            tc["q_input"].copy(),
            tc["seasonal_input"].copy(),
        )

        npt.assert_allclose(
            out_params, tc["output_parameters"],
            atol=ATOL, rtol=RTOL,
            err_msg=f"TC{tc_index + 1} ({tc['description']}) "
                    f"output parameters mismatch",
        )
        npt.assert_array_equal(
            ar_p, tc["arP_output"],
            err_msg=f"TC{tc_index + 1} ({tc['description']}) arP mismatch",
        )
        npt.assert_array_equal(
            ma_q, tc["maQ_output"],
            err_msg=f"TC{tc_index + 1} ({tc['description']}) maQ mismatch",
        )


# ============================================================================
# Phase 4: Return Type and Shape Tests
# ============================================================================


class TestSarma2armaReturnTypes:
    """Verify return types and shapes."""

    def test_sarma2arma_return_types(self) -> None:
        """All three return values must be numpy ndarrays."""
        params = np.array([0.5, 0.3, 0.4, 0.2])
        p = np.array([1.0])
        q = np.array([1.0])
        seasonal = np.array([[1.0, 0.0, 1.0, 12.0]])

        out_params, ar_p, ma_q = sarma2arma(params, p, q, seasonal)

        assert isinstance(out_params, np.ndarray), (
            f"parameters should be ndarray, got {type(out_params)}"
        )
        assert isinstance(ar_p, np.ndarray), (
            f"arP should be ndarray, got {type(ar_p)}"
        )
        assert isinstance(ma_q, np.ndarray), (
            f"maQ should be ndarray, got {type(ma_q)}"
        )

    def test_sarma2arma_return_1d(self) -> None:
        """All returns must be 1-D arrays."""
        params = np.array([0.5, 0.3, 0.4, 0.2])
        p = np.array([1.0])
        q = np.array([1.0])
        seasonal = np.array([[1.0, 0.0, 1.0, 12.0]])

        out_params, ar_p, ma_q = sarma2arma(params, p, q, seasonal)

        assert out_params.ndim == 1, (
            f"parameters should be 1-D, got ndim={out_params.ndim}"
        )
        assert ar_p.ndim == 1, f"arP should be 1-D, got ndim={ar_p.ndim}"
        assert ma_q.ndim == 1, f"maQ should be 1-D, got ndim={ma_q.ndim}"

    def test_sarma2arma_return_1d_empty(self) -> None:
        """Empty return arrays should also be 1-D."""
        params = np.array([0.5])
        p = np.array([1.0])
        q = np.array([], dtype=float)
        seasonal = np.array([[0.0, 0.0, 0.0, 12.0]])

        out_params, ar_p, ma_q = sarma2arma(params, p, q, seasonal)

        assert ma_q.ndim == 1, f"Empty maQ should be 1-D, got ndim={ma_q.ndim}"

    def test_sarma2arma_return_types_empty_ar(self) -> None:
        """Return types valid even with empty AR component."""
        params = np.array([0.3])
        p = np.array([], dtype=float)
        q = np.array([1.0])
        seasonal = np.array([[0.0, 0.0, 0.0, 12.0]])

        out_params, ar_p, ma_q = sarma2arma(params, p, q, seasonal)

        assert isinstance(ar_p, np.ndarray)
        assert ar_p.ndim == 1
        assert len(ar_p) == 0


class TestSarma2armaLagInvariants:
    """Verify lag index invariants: sorted ascending, all positive."""

    def test_sarma2arma_arP_sorted(self) -> None:
        """arP lag indices must be sorted in ascending order."""
        params = np.array([0.3, -0.1, 0.5])
        p = np.array([1.0, 2.0])
        q = np.array([], dtype=float)
        seasonal = np.array([[1.0, 0.0, 0.0, 4.0]])

        _, ar_p, _ = sarma2arma(params, p, q, seasonal)

        # ar_p should be [1, 2, 4, 5, 6] — verify sorted
        assert len(ar_p) > 0, "arP should be non-empty for this case"
        for i in range(len(ar_p) - 1):
            assert ar_p[i] < ar_p[i + 1], (
                f"arP not sorted at position {i}: {ar_p[i]} >= {ar_p[i + 1]}"
            )

    def test_sarma2arma_maQ_sorted(self) -> None:
        """maQ lag indices must be sorted in ascending order."""
        params = np.array([0.5, 0.3, 0.4, 0.2])
        p = np.array([1.0])
        q = np.array([1.0])
        seasonal = np.array([[1.0, 0.0, 1.0, 12.0]])

        _, _, ma_q = sarma2arma(params, p, q, seasonal)

        # ma_q should be [1, 12, 13] — verify sorted
        assert len(ma_q) > 0, "maQ should be non-empty for this case"
        for i in range(len(ma_q) - 1):
            assert ma_q[i] < ma_q[i + 1], (
                f"maQ not sorted at position {i}: {ma_q[i]} >= {ma_q[i + 1]}"
            )

    def test_sarma2arma_lags_positive(self) -> None:
        """All lag indices must be strictly positive (> 0).

        Lag values represent powers of the backshift operator B,
        so they must be at least 1.
        """
        params = np.array([0.5, 0.3, 0.2, 0.4, 0.1, 0.15])
        p = np.array([1.0])
        q = np.array([1.0])
        seasonal = np.array([[1.0, 0.0, 1.0, 4.0],
                             [1.0, 0.0, 1.0, 12.0]])

        _, ar_p, ma_q = sarma2arma(params, p, q, seasonal)

        if len(ar_p) > 0:
            assert np.all(ar_p > 0), (
                f"All arP lags must be > 0, got min={ar_p.min()}"
            )
        if len(ma_q) > 0:
            assert np.all(ma_q > 0), (
                f"All maQ lags must be > 0, got min={ma_q.min()}"
            )

    def test_sarma2arma_lags_integer_valued(self) -> None:
        """Lag indices should be integer-valued (even if stored as float)."""
        params = np.array([0.5, 0.3])
        p = np.array([1.0])
        q = np.array([], dtype=float)
        seasonal = np.array([[1.0, 0.0, 0.0, 12.0]])

        _, ar_p, _ = sarma2arma(params, p, q, seasonal)

        # Each lag should be a whole number
        for lag in ar_p:
            assert float(lag) == int(lag), (
                f"Lag value {lag} is not integer-valued"
            )

    def test_sarma2arma_arP_sorted_multiple_seasonal(self) -> None:
        """arP sorted for multiple seasonal groups (complex case)."""
        params = np.array([0.5, 0.3, 0.2, 0.4, 0.1, 0.15])
        p = np.array([1.0])
        q = np.array([1.0])
        seasonal = np.array([[1.0, 0.0, 1.0, 4.0],
                             [1.0, 0.0, 1.0, 12.0]])

        _, ar_p, ma_q = sarma2arma(params, p, q, seasonal)

        # Verify both arP and maQ are sorted
        if len(ar_p) > 1:
            assert np.all(np.diff(ar_p) > 0), "arP should be strictly ascending"
        if len(ma_q) > 1:
            assert np.all(np.diff(ma_q) > 0), "maQ should be strictly ascending"


# ============================================================================
# Phase 5: Edge Case and Robustness Tests
# ============================================================================


class TestSarma2armaEdgeCases:
    """Additional edge cases and robustness checks."""

    def test_sarma2arma_tuple_return(self) -> None:
        """Return value should be a tuple of length 3."""
        params = np.array([0.5])
        p = np.array([1.0])
        q = np.array([], dtype=float)
        seasonal = np.array([[0.0, 0.0, 0.0, 12.0]])

        result = sarma2arma(params, p, q, seasonal)

        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 3, f"Expected 3-tuple, got {len(result)}-tuple"

    def test_sarma2arma_parameters_not_mutated(self) -> None:
        """Input parameter array should not be mutated by sarma2arma.

        This ensures the function makes copies rather than modifying in-place.
        """
        params_orig = np.array([0.5, 0.3])
        params = params_orig.copy()
        p = np.array([1.0])
        q = np.array([], dtype=float)
        seasonal = np.array([[1.0, 0.0, 0.0, 12.0]])

        sarma2arma(params, p, q, seasonal)

        npt.assert_array_equal(
            params, params_orig,
            err_msg="Input parameters array was mutated"
        )

    def test_sarma2arma_small_coefficients(self) -> None:
        """Very small seasonal coefficients should still produce correct lags.

        Even tiny nonzero coefficients should appear in the expanded lags.
        """
        params = np.array([0.5, 1e-10])
        p = np.array([1.0])
        q = np.array([], dtype=float)
        seasonal = np.array([[1.0, 0.0, 0.0, 12.0]])

        out_params, ar_p, _ = sarma2arma(params, p, q, seasonal)

        # With Phi=1e-10, cross-term phi*Phi=5e-11 is tiny but nonzero
        # Lags should still be [1, 12, 13]
        expected_arP = np.array([1, 12, 13])
        npt.assert_array_equal(ar_p, expected_arP)

    def test_sarma2arma_large_seasonal_period(self) -> None:
        """Large seasonal period (S=52, weekly) should work correctly."""
        params = np.array([0.5, 0.3])
        p = np.array([1.0])
        q = np.array([], dtype=float)
        seasonal = np.array([[1.0, 0.0, 0.0, 52.0]])

        out_params, ar_p, _ = sarma2arma(params, p, q, seasonal)

        expected_arP = np.array([1, 52, 53])
        expected_params = np.array([0.5, 0.3, -(0.5 * 0.3)])

        npt.assert_array_equal(ar_p, expected_arP)
        npt.assert_allclose(out_params, expected_params, atol=ATOL, rtol=RTOL)
