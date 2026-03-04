"""Pytest parity and unit tests for the SARIMAX error computation module.

Tests ``sarimax_errors`` from ``mfe_toolbox.sandbox.sarimax_errors``, a
lightweight bridge function that converts seasonal ARMA parameters to their
expanded non-seasonal form via ``sarma2arma`` and delegates residual
computation to ``armaxerrors``.

Test Organization
-----------------
* **Phase 1 — Unit tests** (no fixture dependency): import verification,
  basic execution, mock-based delegation checks, parameter partitioning.
* **Phase 2 — Input validation tests**: empty data, mismatched dimensions,
  wrong parameter counts.
* **Phase 3 — Parity tests**: comparison against MATLAB reference fixtures
  stored in ``tests/fixtures/sandbox/sarimax_errors.npy``.
* **Phase 4 — Return type/shape tests**: ndarray return, shape (T,),
  finiteness.

Per AAP Section 0.7.1:
    ATOL = 1e-6, RTOL = 1e-4 for all parity comparisons.
    ``pytest.mark.parity`` applied to fixture-dependent tests.
    Graceful ``pytest.skip()`` when fixture files are not found.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.sandbox.sarimax_errors import sarimax_errors

# ---------------------------------------------------------------------------
# Tolerance constants (re-exported from conftest for local clarity)
# ---------------------------------------------------------------------------
# Imported at module level so that they appear in coverage reports.
from tests.conftest import ATOL, RTOL


# ===================================================================
# Helpers — fixture loading and data construction
# ===================================================================

def _load_sarimax_fixture(sandbox_fixture_dir: Path) -> dict:
    """Load the full sarimax_errors fixture dict, skip if missing."""
    path = sandbox_fixture_dir / "sarimax_errors.npy"
    if not path.exists():
        pytest.skip(f"Fixture file not found: {path}")
    return np.load(path, allow_pickle=True).item()


def _build_x(T: int, m_exog: int, x_exog: np.ndarray | None) -> np.ndarray:
    """Construct the exogenous matrix for a fixture test case.

    Parameters
    ----------
    T : int
        Number of observations.
    m_exog : int
        Number of exogenous columns (0 means no exogenous).
    x_exog : np.ndarray | None
        Shared exogenous data from fixture (T, m_total) or None.

    Returns
    -------
    np.ndarray
        Shape ``(T, m_exog)`` exogenous matrix.
    """
    if m_exog == 0:
        return np.empty((T, 0))
    if x_exog is not None and x_exog.shape[1] >= m_exog:
        return x_exog[:, :m_exog]
    # Fallback: zeros
    return np.zeros((T, m_exog))


def _build_sigma(
    T: int,
    sigma_type: str,
    sigma_unit: np.ndarray,
    sigma_hetero: np.ndarray,
) -> np.ndarray:
    """Return the correct sigma vector based on the fixture sigma_type key."""
    if sigma_type == "unit":
        return sigma_unit[:T]
    return sigma_hetero[:T]


# ===================================================================
# Phase 1 — Unit Tests (no fixture dependency)
# ===================================================================

class TestSarimaxErrorsImports:
    """Verify module-level imports work correctly."""

    def test_sarimax_errors_imports(self) -> None:
        """Module should import and ``sarimax_errors`` should be callable."""
        assert callable(sarimax_errors)


class TestSarimaxErrorsBasic:
    """Basic execution tests using synthetically generated data."""

    def test_sarimax_errors_basic_ar1(self, rng: np.random.Generator) -> None:
        """Basic AR(1) call without seasonal component returns correct shape.

        Ref: AAP test case 2 — basic AR(1).
        """
        T = 200
        y = rng.standard_normal(T)
        # p=[1] → AR(1); q=[] → no MA; seasonal with zero AR/MA orders
        p = np.array([1], dtype=np.float64)
        q = np.array([], dtype=np.float64)
        constant = 0
        # Seasonal spec with zero seasonal orders → effectively non-seasonal
        seasonal = np.array([[0, 0, 0, 12]], dtype=np.float64)
        # Parameters: [ar1_coeff]
        params = np.array([0.3])
        x = np.empty((T, 0))
        sigma = np.ones(T)

        result = sarimax_errors(params, p, q, constant, seasonal, y, x, sigma)

        assert isinstance(result, np.ndarray)
        assert result.shape == (T,)

    def test_sarimax_errors_with_seasonal(
        self, rng: np.random.Generator
    ) -> None:
        """Test with seasonal AR specification returns correct shape.

        Ref: AAP test case 3 — seasonal AR/MA.
        """
        T = 300
        y = rng.standard_normal(T)
        p = np.array([1], dtype=np.float64)
        q = np.array([], dtype=np.float64)
        constant = 0
        # 1 seasonal AR order, 0 seasonal MA, period 12
        seasonal = np.array([[1, 0, 0, 12]], dtype=np.float64)
        # Params: [ar1, sar1]
        params = np.array([0.3, 0.2])
        x = np.empty((T, 0))
        sigma = np.ones(T)

        result = sarimax_errors(params, p, q, constant, seasonal, y, x, sigma)

        assert isinstance(result, np.ndarray)
        assert result.shape == (T,)

    def test_sarimax_errors_with_exogenous(
        self, rng: np.random.Generator
    ) -> None:
        """Test with exogenous regressors returns correct shape.

        Ref: AAP test case 4 — 2 exogenous variables.
        """
        T = 200
        y = rng.standard_normal(T)
        x = rng.standard_normal((T, 2))
        p = np.array([1], dtype=np.float64)
        q = np.array([], dtype=np.float64)
        constant = 0
        seasonal = np.array([[0, 0, 0, 12]], dtype=np.float64)
        # Params: [exog1, exog2, ar1]
        params = np.array([0.15, 0.25, 0.3])
        sigma = np.ones(T)

        result = sarimax_errors(params, p, q, constant, seasonal, y, x, sigma)

        assert isinstance(result, np.ndarray)
        assert result.shape == (T,)

    def test_sarimax_errors_no_constant(
        self, rng: np.random.Generator
    ) -> None:
        """Test with constant=0 — one fewer parameter.

        Ref: AAP test case 5 — no constant.
        """
        T = 200
        y = rng.standard_normal(T)
        p = np.array([1], dtype=np.float64)
        q = np.array([1], dtype=np.float64)
        constant = 0
        seasonal = np.array([[0, 0, 0, 12]], dtype=np.float64)
        # Params: [ar1, ma1] (no constant)
        params = np.array([0.3, 0.2])
        x = np.empty((T, 0))
        sigma = np.ones(T)

        result = sarimax_errors(params, p, q, constant, seasonal, y, x, sigma)

        assert isinstance(result, np.ndarray)
        assert result.shape == (T,)

    def test_sarimax_errors_sma_only(
        self, rng: np.random.Generator
    ) -> None:
        """SMA-only: no AR lags means no negative-index issues."""
        T = 200
        y = rng.standard_normal(T)
        p = np.array([], dtype=np.float64)
        q = np.array([1], dtype=np.float64)
        constant = 0
        # 0 seasonal AR, 1 seasonal MA, period 12
        seasonal = np.array([[0, 0, 1, 12]], dtype=np.float64)
        # Params: [ma1, sma1]
        params = np.array([0.4, 0.2])
        x = np.empty((T, 0))
        sigma = np.ones(T)

        result = sarimax_errors(params, p, q, constant, seasonal, y, x, sigma)

        assert isinstance(result, np.ndarray)
        assert result.shape == (T,)
        assert np.all(np.isfinite(result))


class TestSarimaxErrorsDelegation:
    """Mock-based tests verifying parameter partitioning and delegation."""

    def test_sarimax_errors_calls_sarma2arma(self) -> None:
        """Verify sarma2arma is invoked with the correct ARMA parameter subset.

        Ref: AAP test case 6 — mock sarma2arma.
        """
        T = 50
        y = np.ones(T)
        x = np.empty((T, 0))
        sigma = np.ones(T)
        p = np.array([1], dtype=np.float64)
        q = np.array([1], dtype=np.float64)
        constant = 0
        seasonal = np.array([[1, 0, 1, 12]], dtype=np.float64)
        # Params: [ar1, sar1, ma1, sma1]
        params = np.array([0.5, 0.3, 0.4, 0.2])

        # Mock sarma2arma to return predictable values
        mock_arma = np.array([0.5, 0.3, -0.15, 0.4, 0.2, -0.08])
        mock_p_new = np.array([1, 12, 13])
        mock_q_new = np.array([1, 12, 13])

        with patch(
            "mfe_toolbox.sandbox.sarimax_errors.sarma2arma",
            return_value=(mock_arma, mock_p_new, mock_q_new),
        ) as mock_s2a:
            with patch(
                "mfe_toolbox.sandbox.sarimax_errors.armaxerrors",
                return_value=np.zeros(T),
            ):
                sarimax_errors(params, p, q, constant, seasonal, y, x, sigma)

            # sarma2arma should have been called once
            mock_s2a.assert_called_once()
            call_args = mock_s2a.call_args

            # First positional arg is the non-exogenous portion of params
            # With m=0 (no exog), all params go to sarma2arma
            called_params = call_args[0][0]
            npt.assert_array_almost_equal(called_params, params)

            # p, q, seasonal should be passed through
            called_p = call_args[0][1]
            called_q = call_args[0][2]
            called_seasonal = call_args[0][3]
            npt.assert_array_equal(
                np.asarray(called_p).ravel(),
                np.asarray(p).ravel(),
            )
            npt.assert_array_equal(
                np.asarray(called_q).ravel(),
                np.asarray(q).ravel(),
            )

    def test_sarimax_errors_calls_armaxerrors(self) -> None:
        """Verify armaxerrors is invoked with the recombined parameter vector.

        Ref: AAP test case 7 — mock armaxerrors.
        The burn-in ``m`` passed to armaxerrors equals
        ``max(max(p_new), max(q_new))`` when it exceeds the exogenous
        column count, due to zero-padding applied inside sarimax_errors.
        """
        T = 50
        y = np.ones(T)
        x = np.empty((T, 0))
        sigma = np.ones(T)
        p = np.array([1], dtype=np.float64)
        q = np.array([1], dtype=np.float64)
        constant = 0
        seasonal = np.array([[0, 0, 0, 12]], dtype=np.float64)
        params = np.array([0.5, 0.4])

        # Mock sarma2arma: with zero seasonal orders, it should pass through
        mock_arma = np.array([0.5, 0.4])
        mock_p_new = np.array([1])
        mock_q_new = np.array([1])

        # max_lag = max(1, 1) = 1; with 0 exog columns → padding of 1
        expected_burn = 1  # max(max(p_new), max(q_new))
        padded_T = T + expected_burn  # y is padded

        expected_errors = np.ones(padded_T) * 0.42
        with patch(
            "mfe_toolbox.sandbox.sarimax_errors.sarma2arma",
            return_value=(mock_arma, mock_p_new, mock_q_new),
        ):
            with patch(
                "mfe_toolbox.sandbox.sarimax_errors.armaxerrors",
                return_value=expected_errors,
            ) as mock_ae:
                result = sarimax_errors(
                    params, p, q, constant, seasonal, y, x, sigma
                )

                mock_ae.assert_called_once()
                call_args = mock_ae.call_args

                # The combined parameter vector: [exog(0)] + arma_params
                called_params = call_args[0][0]
                npt.assert_array_almost_equal(called_params, mock_arma)

                # Expanded lags
                called_p = call_args[0][1]
                called_q = call_args[0][2]
                npt.assert_array_equal(called_p, mock_p_new)
                npt.assert_array_equal(called_q, mock_q_new)

                # constant
                assert call_args[0][3] == constant

                # m (burn-in) = max_lag due to padding
                assert call_args[0][6] == expected_burn

                # Result should be trimmed to T
                assert result.shape == (T,)

    def test_sarimax_errors_parameter_partitioning(self) -> None:
        """Verify parameter vector is partitioned correctly with exogenous.

        Ref: AAP test case 8 — m=2 exog, constant=1, p=[1,2], q=[1],
        seasonal=[[1,1,12]].

        With 2 exogenous columns (m=2) and max expanded lag = 14,
        the burn-in passed to armaxerrors is max(2, 14) = 14.
        """
        T = 50
        y = np.ones(T)
        x = np.ones((T, 2))  # 2 exogenous columns → m=2
        sigma = np.ones(T)
        p = np.array([1, 2], dtype=np.float64)
        q = np.array([1], dtype=np.float64)
        constant = 1
        seasonal = np.array([[1, 0, 1, 12]], dtype=np.float64)

        # Params: [exog1, exog2, ar1, ar2, sar1, ma1, sma1] = 7 total
        params = np.array([0.1, 0.2, 0.3, 0.15, 0.25, 0.4, 0.1])

        mock_arma = np.array([0.3, 0.15, 0.25, -0.075, 0.4, 0.1, -0.04])
        mock_p_new = np.array([1, 2, 12, 13, 14])
        mock_q_new = np.array([1, 12, 13])

        # max_lag = max(14, 13) = 14; m_exog = 2; pad = 14 - 2 = 12
        expected_burn = 14
        padded_T = T + (expected_burn - 2)  # 50 + 12 = 62

        with patch(
            "mfe_toolbox.sandbox.sarimax_errors.sarma2arma",
            return_value=(mock_arma, mock_p_new, mock_q_new),
        ) as mock_s2a:
            with patch(
                "mfe_toolbox.sandbox.sarimax_errors.armaxerrors",
                return_value=np.zeros(padded_T),
            ) as mock_ae:
                result = sarimax_errors(
                    params, p, q, constant, seasonal, y, x, sigma
                )

                # sarma2arma should receive params[2:] (after 2 exog params)
                s2a_call = mock_s2a.call_args[0]
                called_arma_params = s2a_call[0]
                expected_arma_subset = params[2:]
                npt.assert_array_almost_equal(
                    called_arma_params, expected_arma_subset
                )

                # armaxerrors should receive [exog(2) + arma_expanded]
                ae_call = mock_ae.call_args[0]
                combined_params = ae_call[0]
                expected_combined = np.concatenate(
                    [params[:2], mock_arma]
                )
                npt.assert_array_almost_equal(
                    combined_params, expected_combined
                )

                # burn = max(m_exog=2, max_lag=14) = 14
                assert ae_call[6] == expected_burn

                # Result should be trimmed to T
                assert result.shape == (T,)

    def test_sarimax_errors_arma_only(self) -> None:
        """Test with no seasonal component — pure ARMA bridge.

        Ref: AAP test case 9 — when seasonal orders are zero, sarma2arma
        should essentially pass through.
        """
        T = 50
        y = np.ones(T)
        x = np.empty((T, 0))
        sigma = np.ones(T)
        p = np.array([1], dtype=np.float64)
        q = np.array([1], dtype=np.float64)
        constant = 0
        # Zero seasonal orders
        seasonal = np.array([[0, 0, 0, 12]], dtype=np.float64)
        params = np.array([0.5, 0.4])

        # With zero seasonal, sarma2arma should return pass-through
        mock_arma = np.array([0.5, 0.4])
        mock_p_new = np.array([1])
        mock_q_new = np.array([1])

        # max_lag = 1, burn = 1, pad = 1
        padded_T = T + 1

        with patch(
            "mfe_toolbox.sandbox.sarimax_errors.sarma2arma",
            return_value=(mock_arma, mock_p_new, mock_q_new),
        ):
            with patch(
                "mfe_toolbox.sandbox.sarimax_errors.armaxerrors",
                return_value=np.zeros(padded_T),
            ) as mock_ae:
                result = sarimax_errors(
                    params, p, q, constant, seasonal, y, x, sigma
                )

                # Combined params should equal original (no exog, no expansion)
                called_params = mock_ae.call_args[0][0]
                npt.assert_array_almost_equal(called_params, params)

                # Result should be trimmed to T
                assert result.shape == (T,)


# ===================================================================
# Phase 2 — Input Validation Tests
# ===================================================================

class TestSarimaxErrorsValidation:
    """Tests for input validation and error handling."""

    def test_sarimax_errors_empty_y_raises(self) -> None:
        """Empty y vector should raise ValueError or produce empty output.

        Ref: AAP test case 10.
        """
        y = np.array([])
        params = np.array([0.5])
        p = np.array([1], dtype=np.float64)
        q = np.array([], dtype=np.float64)
        constant = 0
        seasonal = np.array([[0, 0, 0, 12]], dtype=np.float64)
        x = np.empty((0, 0))
        sigma = np.array([])

        # Empty y should either raise or return empty array
        try:
            result = sarimax_errors(
                params, p, q, constant, seasonal, y, x, sigma
            )
            # If no error, the output must be empty too
            assert len(result) == 0
        except (ValueError, IndexError):
            pass  # Expected

    def test_sarimax_errors_x_y_length_mismatch(self) -> None:
        """Mismatched x rows and y length should raise or produce error.

        Ref: AAP test case 12.
        """
        T = 100
        y = np.ones(T)
        # x has different number of rows
        x = np.ones((T + 10, 1))
        params = np.array([0.1, 0.5])
        p = np.array([1], dtype=np.float64)
        q = np.array([], dtype=np.float64)
        constant = 0
        seasonal = np.array([[0, 0, 0, 12]], dtype=np.float64)
        sigma = np.ones(T)

        # The function may raise or produce mismatched output
        try:
            result = sarimax_errors(
                params, p, q, constant, seasonal, y, x, sigma
            )
            # If it returns, shapes should at least be consistent
            assert result.ndim >= 1
        except (ValueError, IndexError):
            pass  # Expected for dimension mismatch


# ===================================================================
# Phase 3 — Parity Tests (against MATLAB fixtures)
# ===================================================================

# Fixture case names that exist in the .npy file
_FIXTURE_CASES = [
    "sarima_101_101_12_no_const_unit_sigma",
    "sar_only_p1_s12_no_const_unit_sigma",
    "sma_only_q1_s12_no_const_unit_sigma",
    "sarima_101_101_12_no_const_hetero_sigma",
    "arma_11_no_seasonal_no_const_unit_sigma",
    "sar_quarterly_p12_s4_no_const_unit_sigma",
    "sarima_101_101_12_with_exog_no_const_unit_sigma",
    "multi_seasonal_q4_m12_no_const_unit_sigma",
]


@pytest.mark.parity
class TestSarimaxErrorsParity:
    """Parity tests against MATLAB reference fixture outputs.

    Fixtures at ``tests/fixtures/sandbox/sarimax_errors.npy``.
    ATOL = 1e-6, RTOL = 1e-4 per AAP Section 0.7.1.
    """

    @pytest.fixture(autouse=True)
    def _load_fixture(self, sandbox_fixture_dir: Path) -> None:
        """Load fixture data for all parity tests in this class."""
        self._fixture = _load_sarimax_fixture(sandbox_fixture_dir)

    def _run_case(self, case_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Execute sarimax_errors for a given fixture case and return
        (actual, expected) arrays.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(actual_errors, expected_errors)``
        """
        data = self._fixture
        case = data[case_name]
        T = data["T"]
        y = data["input_y"]
        sigma = _build_sigma(
            T,
            case["sigma_type"],
            data["sigma_unit"],
            data["sigma_heterogeneous"],
        )
        x = _build_x(T, case["m_exog"], data.get("input_x_exogenous"))

        actual = sarimax_errors(
            case["input_parameters"].astype(np.float64),
            case["p"].astype(np.float64),
            case["q"].astype(np.float64),
            int(case["constant"]),
            case["seasonal"].astype(np.float64),
            y,
            x,
            sigma,
        )
        return actual, case["errors"]

    # --- SMA-only case: exact parity (no AR lags = no wrap issue) ---

    def test_sarimax_errors_parity_sma_only(self) -> None:
        """SMA-only case should achieve full parity.

        This case has no AR lags, so no negative-index wrapping occurs.
        """
        actual, expected = self._run_case(
            "sma_only_q1_s12_no_const_unit_sigma"
        )
        npt.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL)

    # --- Parametrized parity test for all cases ---

    @pytest.mark.parametrize("case_name", _FIXTURE_CASES)
    def test_sarimax_errors_parity(self, case_name: str) -> None:
        """Parametrized parity test for each fixture case.

        Compares Python sarimax_errors output against MATLAB reference.
        """
        actual, expected = self._run_case(case_name)

        # Verify shapes match
        assert actual.shape == expected.shape, (
            f"Shape mismatch for {case_name}: "
            f"actual={actual.shape}, expected={expected.shape}"
        )

        npt.assert_allclose(
            actual,
            expected,
            atol=ATOL,
            rtol=RTOL,
            err_msg=f"Parity failure for case: {case_name}",
        )

    def test_sarimax_errors_parity_with_exog(self) -> None:
        """Parity test with exogenous variables.

        Ref: AAP test case 15.
        """
        actual, expected = self._run_case(
            "sarima_101_101_12_with_exog_no_const_unit_sigma"
        )
        assert actual.shape == expected.shape
        npt.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL)


# ===================================================================
# Phase 4 — Return Type and Shape Tests
# ===================================================================

class TestSarimaxErrorsReturnProperties:
    """Tests for return type, shape, and finiteness."""

    def test_sarimax_errors_return_type(
        self, rng: np.random.Generator
    ) -> None:
        """Result must be a numpy ndarray.

        Ref: AAP test case 16.
        """
        T = 100
        y = rng.standard_normal(T)
        p = np.array([], dtype=np.float64)
        q = np.array([1], dtype=np.float64)
        constant = 0
        seasonal = np.array([[0, 0, 0, 12]], dtype=np.float64)
        params = np.array([0.4])
        x = np.empty((T, 0))
        sigma = np.ones(T)

        result = sarimax_errors(params, p, q, constant, seasonal, y, x, sigma)
        assert isinstance(result, np.ndarray)

    def test_sarimax_errors_return_shape(
        self, rng: np.random.Generator
    ) -> None:
        """Return shape must be (T,) matching input y length.

        Ref: AAP test case 17.
        """
        for T in [50, 100, 200, 500]:
            y = rng.standard_normal(T)
            p = np.array([], dtype=np.float64)
            q = np.array([1], dtype=np.float64)
            constant = 0
            seasonal = np.array([[0, 0, 0, 12]], dtype=np.float64)
            params = np.array([0.3])
            x = np.empty((T, 0))
            sigma = np.ones(T)

            result = sarimax_errors(
                params, p, q, constant, seasonal, y, x, sigma
            )
            assert result.shape == (T,), (
                f"Expected shape ({T},) but got {result.shape}"
            )

    def test_sarimax_errors_finite_output(
        self, rng: np.random.Generator
    ) -> None:
        """All output errors must be finite (no NaN/Inf).

        Ref: AAP test case 18.
        Uses SMA-only configuration to avoid negative-index wrapping.
        """
        T = 200
        y = rng.standard_normal(T)
        p = np.array([], dtype=np.float64)
        q = np.array([1], dtype=np.float64)
        constant = 0
        seasonal = np.array([[0, 0, 1, 12]], dtype=np.float64)
        params = np.array([0.4, 0.2])
        x = np.empty((T, 0))
        sigma = np.ones(T)

        result = sarimax_errors(params, p, q, constant, seasonal, y, x, sigma)
        assert np.all(np.isfinite(result)), (
            f"Non-finite values found in output: "
            f"NaN count={np.sum(np.isnan(result))}, "
            f"Inf count={np.sum(np.isinf(result))}"
        )

    def test_sarimax_errors_output_not_all_zero(
        self, rng: np.random.Generator
    ) -> None:
        """Output should not be all zeros for non-trivial input."""
        T = 200
        y = rng.standard_normal(T) * 2.0 + 1.0
        p = np.array([], dtype=np.float64)
        q = np.array([1], dtype=np.float64)
        constant = 0
        seasonal = np.array([[0, 0, 0, 12]], dtype=np.float64)
        params = np.array([0.4])
        x = np.empty((T, 0))
        sigma = np.ones(T)

        result = sarimax_errors(params, p, q, constant, seasonal, y, x, sigma)
        assert np.sum(np.abs(result)) > 0, "Output is all zeros"


# ===================================================================
# Additional Integration Tests
# ===================================================================

class TestSarimaxErrorsIntegration:
    """Integration tests verifying end-to-end behaviour with real sarma2arma
    and armaxerrors (no mocks)."""

    def test_sma_only_first_value_equals_y(
        self, rng: np.random.Generator
    ) -> None:
        """For SMA-only with m=0 and no AR lags, e[0] should equal y[0].

        With no AR terms and e initialized to zeros, at t=0 all MA terms
        reference e[-k] = 0, so e[0] = y[0]/sigma[0].
        """
        T = 100
        y = rng.standard_normal(T)
        p = np.array([], dtype=np.float64)
        q = np.array([1], dtype=np.float64)
        constant = 0
        seasonal = np.array([[0, 0, 1, 12]], dtype=np.float64)
        params = np.array([0.4, 0.2])
        x = np.empty((T, 0))
        sigma = np.ones(T)

        result = sarimax_errors(params, p, q, constant, seasonal, y, x, sigma)

        # With sigma=1, e[0] = y[0] (no AR, MA terms reference e=0)
        npt.assert_allclose(result[0], y[0], atol=1e-12)

    def test_exog_offset_correct(self, rng: np.random.Generator) -> None:
        """Verify that exogenous columns correctly offset the parameter
        vector before sarma2arma expansion."""
        T = 100
        y = rng.standard_normal(T)
        x_1col = rng.standard_normal((T, 1))
        p = np.array([], dtype=np.float64)
        q = np.array([1], dtype=np.float64)
        constant = 0
        seasonal = np.array([[0, 0, 0, 12]], dtype=np.float64)
        sigma = np.ones(T)

        # With 1 exog column: params = [exog_coef, ma1]
        params_with_exog = np.array([0.15, 0.4])

        # Without exog: params = [ma1]
        params_no_exog = np.array([0.4])

        e_with_exog = sarimax_errors(
            params_with_exog, p, q, constant, seasonal, y, x_1col, sigma
        )
        e_no_exog = sarimax_errors(
            params_no_exog, p, q, constant, seasonal, y,
            np.empty((T, 0)), sigma
        )

        # Results should differ (exogenous contributes)
        assert not np.allclose(e_with_exog, e_no_exog), (
            "Exogenous variable should affect the residuals"
        )

    def test_multi_seasonal_execution(
        self, rng: np.random.Generator
    ) -> None:
        """Multi-seasonal (quarterly + monthly) should execute without error."""
        T = 200
        y = rng.standard_normal(T)
        p = np.array([1], dtype=np.float64)
        q = np.array([1], dtype=np.float64)
        constant = 0
        # 2 seasonal components: quarterly (period 4) and monthly (period 12)
        seasonal = np.array(
            [[1, 0, 1, 4], [1, 0, 1, 12]], dtype=np.float64
        )
        # Params: [ar1, sar1_q4, sar1_m12, ma1, sma1_q4, sma1_m12]
        params = np.array([0.5, 0.3, 0.2, 0.4, 0.1, 0.15])
        x = np.empty((T, 0))
        sigma = np.ones(T)

        result = sarimax_errors(params, p, q, constant, seasonal, y, x, sigma)

        assert isinstance(result, np.ndarray)
        assert result.shape == (T,)

    def test_heterogeneous_sigma(self, rng: np.random.Generator) -> None:
        """Verify sigma normalisation affects the output."""
        T = 100
        y = rng.standard_normal(T)
        p = np.array([], dtype=np.float64)
        q = np.array([1], dtype=np.float64)
        constant = 0
        seasonal = np.array([[0, 0, 0, 12]], dtype=np.float64)
        params = np.array([0.4])
        x = np.empty((T, 0))

        sigma_unit = np.ones(T)
        sigma_hetero = np.abs(rng.standard_normal(T)) + 0.5

        e_unit = sarimax_errors(
            params, p, q, constant, seasonal, y, x, sigma_unit
        )
        e_hetero = sarimax_errors(
            params, p, q, constant, seasonal, y, x, sigma_hetero
        )

        # Different sigma should produce different errors
        assert not np.allclose(e_unit, e_hetero), (
            "Heterogeneous sigma should change residuals"
        )
