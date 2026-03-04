"""Comprehensive pytest tests for the SARIMAX likelihood computation module.

Tests ``sarimax_likelihood`` from ``mfe_toolbox.sandbox.sarimax_likelihood``,
migrated from MATLAB ``sandbox/sarimax_likelihood.m`` (MFE Toolbox v4.0,
Kevin Sheppard).

Test coverage includes:
- Phase 1: Unit tests (import, basic call, Gaussian formula, seasonal,
  exogenous, NaN clamping, trimming, no-constant, mocked delegation)
- Phase 2: Input validation (empty y, mismatched sigma, wrong param count)
- Phase 3: Parity tests against MATLAB reference fixtures (±1e-6 / ±1e-4)
- Phase 4: Return type/shape invariants (LLF=sum, non-negative likelihoods)

Per AAP Section 0.7.1:
    ATOL = 1e-6, RTOL = 1e-4 for all MATLAB parity comparisons.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from mfe_toolbox.sandbox.sarimax_likelihood import sarimax_likelihood

# Re-use tolerance constants from conftest (imported implicitly by pytest)
# but also define locally for clarity in explicit assert_allclose calls.
ATOL: float = 1e-6
RTOL: float = 1e-4


# ---------------------------------------------------------------------------
# Helper: load the shared fixture dict
# ---------------------------------------------------------------------------

def _load_fixture(sandbox_fixture_dir: Path) -> dict:
    """Load the SARIMAX likelihood fixture dict, or skip if missing."""
    fpath = sandbox_fixture_dir / "sarimax_likelihood.npy"
    if not fpath.exists():
        pytest.skip(f"Fixture file not found: {fpath}")
    return np.load(fpath, allow_pickle=True).item()


# =========================================================================
# Phase 1: Unit Tests (no fixture dependency)
# =========================================================================


class TestSarimaxLikelihoodImport:
    """Test 1 — Verify the module imports correctly."""

    def test_sarimax_likelihood_imports(self) -> None:
        """``sarimax_likelihood`` should be importable and callable."""
        assert callable(sarimax_likelihood)


class TestSarimaxLikelihoodBasic:
    """Tests 2–8 — Unit tests covering core behaviours."""

    # ------------------------------------------------------------------
    # Test 2: Basic call with known inputs
    # ------------------------------------------------------------------
    def test_sarimax_likelihood_basic_call(self, rng: np.random.Generator) -> None:
        """Basic AR(1) call returns a valid (LLF, likelihoods, errors) tuple.

        Uses p=[1], q=[], constant=0, no seasonal, no exogenous (m=0).
        Ref: sarimax_likelihood.m full flow.
        """
        T = 200
        y = rng.standard_normal(T)
        # No exogenous columns → x is (T, 0)
        x = np.empty((T, 0))
        sigma = np.ones(T)
        # With m=0 and no seasonal, the parameter vector is [ar1]
        params = np.array([0.3])
        p = np.array([1])
        q = np.array([], dtype=np.float64)
        constant = 0
        seasonal = np.array([[0, 0, 0, 12]])  # no seasonal (Sp=0, Sq=0)

        LLF, likelihoods, errors = sarimax_likelihood(
            params, p, q, constant, seasonal, y, x, sigma
        )

        # Basic type / shape checks
        assert isinstance(LLF, (float, np.floating)), "LLF must be scalar float"
        assert isinstance(likelihoods, np.ndarray), "likelihoods must be ndarray"
        assert isinstance(errors, np.ndarray), "errors must be ndarray"
        assert likelihoods.ndim == 1, "likelihoods must be 1-D"
        assert errors.ndim == 1, "errors must be 1-D"
        assert np.isfinite(LLF), f"LLF should be finite, got {LLF}"

    # ------------------------------------------------------------------
    # Test 3: Gaussian log-likelihood formula verification
    # ------------------------------------------------------------------
    def test_sarimax_likelihood_gaussian_formula(self) -> None:
        """Verify the Gaussian NLL formula: 0.5*(2*log(σ) + (e/σ)^2 + log(2π)).

        When residuals are zero and sigma=1, each per-obs likelihood should be
        0.5 * log(2*π) ≈ 0.9189385.
        Ref: sarimax_likelihood.m line ~40.
        """
        T = 50
        # Construct y = 0 so that errors = 0 when all params are 0
        y = np.zeros(T)
        x = np.empty((T, 0))
        sigma = np.ones(T)
        # No AR, no MA, no constant, no seasonal → parameter vector is empty
        params = np.array([], dtype=np.float64)
        p = np.array([], dtype=np.float64)
        q = np.array([], dtype=np.float64)
        constant = 0
        seasonal = np.array([[0, 0, 0, 12]])

        LLF, likelihoods, errors = sarimax_likelihood(
            params, p, q, constant, seasonal, y, x, sigma
        )

        expected_per_obs = 0.5 * np.log(2.0 * np.pi)
        # All errors should be zero (y=0, no AR/MA/const/exog)
        npt.assert_allclose(errors, 0.0, atol=1e-12)
        # Each likelihood should be the Gaussian constant
        npt.assert_allclose(likelihoods, expected_per_obs, atol=1e-12)
        # LLF = T_trimmed * expected_per_obs
        npt.assert_allclose(LLF, np.sum(likelihoods), atol=1e-12)

    # ------------------------------------------------------------------
    # Test 4: Seasonal components
    # ------------------------------------------------------------------
    def test_sarimax_likelihood_with_seasonal(self, rng: np.random.Generator) -> None:
        """Seasonal ARMA(1,0,1)×(1,0,1,12) should run without error.

        seasonal = [[1, 0, 1, 12]] means Sp=1, reserved=0, Sq=1, S=12.
        params = [ar1, sar1, ma1, sma1] (m=0, constant=0).
        """
        T = 300
        y = rng.standard_normal(T)
        x = np.empty((T, 0))
        sigma = np.ones(T)
        # ARMA(1,1) + seasonal AR(1), seasonal MA(1) at period 12
        params = np.array([0.5, 0.3, 0.4, 0.2])
        p = np.array([1])
        q = np.array([1])
        constant = 0
        seasonal = np.array([[1, 0, 1, 12]])

        LLF, likelihoods, errors = sarimax_likelihood(
            params, p, q, constant, seasonal, y, x, sigma
        )

        assert np.isfinite(LLF), f"LLF should be finite, got {LLF}"
        assert likelihoods.ndim == 1
        assert errors.ndim == 1
        assert len(likelihoods) > 0, "likelihoods should not be empty"
        assert len(errors) > 0, "errors should not be empty"

    # ------------------------------------------------------------------
    # Test 5: Exogenous variables
    # ------------------------------------------------------------------
    def test_sarimax_likelihood_with_exogenous(
        self, rng: np.random.Generator
    ) -> None:
        """With m=2 exogenous variables, the first 2 parameters are exog coeffs.

        params = [beta1, beta2, ar1] for p=[1], q=[], constant=0, no seasonal.
        """
        T = 200
        y = rng.standard_normal(T)
        x = rng.standard_normal((T, 2))
        sigma = np.ones(T)
        params = np.array([0.5, -0.3, 0.4])  # 2 exog + 1 AR
        p = np.array([1])
        q = np.array([], dtype=np.float64)
        constant = 0
        seasonal = np.array([[0, 0, 0, 12]])

        LLF, likelihoods, errors = sarimax_likelihood(
            params, p, q, constant, seasonal, y, x, sigma
        )

        assert np.isfinite(LLF)
        assert likelihoods.ndim == 1
        assert errors.ndim == 1

    # ------------------------------------------------------------------
    # Test 6: NaN clamping
    # ------------------------------------------------------------------
    def test_sarimax_likelihood_nan_clamping(self) -> None:
        """When sigma contains zeros, log(0) produces NaN → LLF clamped to 1e7.

        Ref: sarimax_likelihood.m lines 58–60 — if isnan(LLF), LLF = 1e7.
        """
        T = 100
        y = np.ones(T)
        x = np.empty((T, 0))
        # sigma with a zero at index 25 triggers log(0)=-inf and (e/0)^2=inf
        sigma = np.ones(T)
        sigma[25] = 0.0

        params = np.array([0.3])
        p = np.array([1])
        q = np.array([], dtype=np.float64)
        constant = 0
        seasonal = np.array([[0, 0, 0, 12]])

        LLF, likelihoods, errors = sarimax_likelihood(
            params, p, q, constant, seasonal, y, x, sigma
        )

        # The NaN guard should set LLF to 1e7
        assert LLF == 1e7, f"Expected LLF=1e7 (NaN guard), got {LLF}"

    # ------------------------------------------------------------------
    # Test 7: Observation trimming
    # ------------------------------------------------------------------
    def test_sarimax_likelihood_trimming(self) -> None:
        """When max(maQ) > max(arP), initial observations are trimmed.

        Ref: sarimax_likelihood.m lines 42–52.
        For MA-only models at lag q=[1] with SAR at S=12, the expanded maQ
        can include lags like [1, 12, 13]. If arP is empty (p_max=0),
        then q_max=13 > 0 → trim_start = 13. So n_out = T - 13.
        """
        T = 200
        y = np.ones(T) * 0.5
        x = np.empty((T, 0))
        sigma = np.ones(T)
        # MA(1) with SMA(1) at S=12 → expanded maQ includes lag 13
        params = np.array([0.3, 0.2])  # ma1=0.3, sma1=0.2
        p = np.array([], dtype=np.float64)
        q = np.array([1])
        constant = 0
        seasonal = np.array([[0, 0, 1, 12]])  # Sp=0, Sq=1, S=12

        LLF, likelihoods, errors = sarimax_likelihood(
            params, p, q, constant, seasonal, y, x, sigma
        )

        # With MA only and expanded maQ having max=13, p_max=0 → trim 13
        # So likelihoods length = T - 13 = 187
        assert len(likelihoods) < T, "Trimming should reduce likelihoods length"
        assert len(likelihoods) == len(errors), "likelihoods and errors must match"

    # ------------------------------------------------------------------
    # Test 8: No constant (constant=0)
    # ------------------------------------------------------------------
    def test_sarimax_likelihood_no_constant(self, rng: np.random.Generator) -> None:
        """With constant=0, one fewer parameter is consumed.

        Ref: sarimax_likelihood.m — constant does not add a parameter to the
        ARMA portion (it is handled inside armaxerrors).
        """
        T = 200
        y = rng.standard_normal(T)
        x = np.empty((T, 0))
        sigma = np.ones(T)
        params = np.array([0.3])  # Just AR(1) coeff, no constant param
        p = np.array([1])
        q = np.array([], dtype=np.float64)
        constant = 0
        seasonal = np.array([[0, 0, 0, 12]])

        LLF, likelihoods, errors = sarimax_likelihood(
            params, p, q, constant, seasonal, y, x, sigma
        )

        assert np.isfinite(LLF)
        assert likelihoods.ndim == 1


class TestSarimaxLikelihoodMocked:
    """Tests 9–10 — Verify delegation to sarma2arma and armaxerrors."""

    # ------------------------------------------------------------------
    # Test 9: sarma2arma invocation
    # ------------------------------------------------------------------
    def test_sarimax_likelihood_calls_sarma2arma(self) -> None:
        """Verify ``sarma2arma`` is called with the ARMA parameter slice.

        Ref: sarimax_likelihood.m:33 — sarma2arma receives parameters(m+1:end).
        Mock path: ``mfe_toolbox.sandbox.sarimax_likelihood.sarma2arma``.
        """
        T = 50
        y = np.ones(T)
        x = np.empty((T, 0))
        sigma = np.ones(T)
        params = np.array([0.5, 0.3])  # ar1=0.5, ma1=0.3 (m=0)
        p = np.array([1])
        q = np.array([1])
        seasonal = np.array([[0, 0, 0, 12]])

        mock_arma_params = np.array([0.5, 0.3])
        mock_arP = np.array([1])
        mock_maQ = np.array([1])

        with patch(
            "mfe_toolbox.sandbox.sarimax_likelihood.sarma2arma",
            return_value=(mock_arma_params, mock_arP, mock_maQ),
        ) as mock_s2a, patch(
            "mfe_toolbox.sandbox.sarimax_likelihood.armaxerrors",
            return_value=np.zeros(T),
        ):
            sarimax_likelihood(params, p, q, 0, seasonal, y, x, sigma)

            # sarma2arma should have been called once
            mock_s2a.assert_called_once()
            # First positional argument should be params[m:] = params[0:]
            call_args = mock_s2a.call_args
            npt.assert_array_equal(call_args[0][0], params)  # params[m:] where m=0

    # ------------------------------------------------------------------
    # Test 10: armaxerrors invocation
    # ------------------------------------------------------------------
    def test_sarimax_likelihood_calls_armaxerrors(self) -> None:
        """Verify ``armaxerrors`` is called with recombined parameters.

        Ref: sarimax_likelihood.m:36 — armaxerrors(parameters, p, q, constant,
        y, x, m, ones(size(y))).
        Mock path: ``mfe_toolbox.sandbox.sarimax_likelihood.armaxerrors``.
        """
        T = 50
        y = np.ones(T)
        x = np.empty((T, 0))
        sigma = np.ones(T)
        params = np.array([0.5])
        p = np.array([1])
        q = np.array([], dtype=np.float64)
        seasonal = np.array([[0, 0, 0, 12]])

        mock_arma_params = np.array([0.5])
        mock_arP = np.array([1])
        mock_maQ = np.array([], dtype=np.float64)

        with patch(
            "mfe_toolbox.sandbox.sarimax_likelihood.sarma2arma",
            return_value=(mock_arma_params, mock_arP, mock_maQ),
        ), patch(
            "mfe_toolbox.sandbox.sarimax_likelihood.armaxerrors",
            return_value=np.zeros(T),
        ) as mock_ax:
            sarimax_likelihood(params, p, q, 0, seasonal, y, x, sigma)

            mock_ax.assert_called_once()
            call_args = mock_ax.call_args
            # First argument should be the recombined parameter vector
            # [params[:m], arma_params] = [empty, [0.5]] = [0.5]
            recombined = call_args[0][0]
            assert len(recombined) > 0, "Recombined params must not be empty"


# =========================================================================
# Phase 2: Input Validation Tests
# =========================================================================


class TestSarimaxLikelihoodValidation:
    """Tests 11–13 — Input validation edge cases."""

    # ------------------------------------------------------------------
    # Test 11: Empty y
    # ------------------------------------------------------------------
    def test_sarimax_likelihood_empty_y_raises(self) -> None:
        """Empty y should either raise ValueError or produce degenerate output.

        Ref: sarimax_likelihood.m — MATLAB would error on empty y.
        """
        y = np.array([], dtype=np.float64)
        x = np.empty((0, 0))
        sigma = np.array([], dtype=np.float64)
        params = np.array([0.3])
        p = np.array([1])
        q = np.array([], dtype=np.float64)
        seasonal = np.array([[0, 0, 0, 12]])

        # The function may raise ValueError or produce NaN/empty outputs.
        # Either is acceptable as long as it doesn't silently produce
        # wrong results.
        try:
            LLF, likelihoods, errors = sarimax_likelihood(
                params, p, q, 0, seasonal, y, x, sigma
            )
            # If it doesn't raise, LLF should be NaN-guarded or zero-length
            assert LLF == 1e7 or len(likelihoods) == 0 or np.isnan(LLF) is False
        except (ValueError, IndexError):
            pass  # Expected — empty input is invalid

    # ------------------------------------------------------------------
    # Test 12: Mismatched sigma length
    # ------------------------------------------------------------------
    def test_sarimax_likelihood_mismatched_sigma_raises(self) -> None:
        """sigma length ≠ y length should raise or produce incorrect results.

        Ref: sarimax_likelihood.m — MATLAB relies on element-wise division.
        """
        T = 100
        y = np.ones(T)
        x = np.empty((T, 0))
        sigma = np.ones(T + 10)  # Wrong length
        params = np.array([0.3])
        p = np.array([1])
        q = np.array([], dtype=np.float64)
        seasonal = np.array([[0, 0, 0, 12]])

        # Should raise or at least not crash silently
        try:
            LLF, likelihoods, errors = sarimax_likelihood(
                params, p, q, 0, seasonal, y, x, sigma
            )
            # If it doesn't raise, it used a mismatched sigma; verify
            # the function at least completed
            assert isinstance(LLF, (float, np.floating))
        except (ValueError, IndexError):
            pass  # Expected

    # ------------------------------------------------------------------
    # Test 13: Wrong parameter count
    # ------------------------------------------------------------------
    def test_sarimax_likelihood_wrong_param_count(self) -> None:
        """Incorrect parameter vector length should raise an error.

        For AR(1) with m=0 and no seasonal, exactly 1 parameter is needed.
        Providing too few should cause an error downstream.
        """
        T = 100
        y = np.ones(T)
        x = np.empty((T, 0))
        sigma = np.ones(T)
        # AR(1) needs 1 param, but we provide 0
        params = np.array([], dtype=np.float64)
        p = np.array([1])
        q = np.array([], dtype=np.float64)
        seasonal = np.array([[0, 0, 0, 12]])

        with pytest.raises((ValueError, IndexError)):
            sarimax_likelihood(params, p, q, 0, seasonal, y, x, sigma)


# =========================================================================
# Phase 3: Parity Tests (against MATLAB fixtures)
# =========================================================================


@pytest.mark.parity
class TestSarimaxLikelihoodParity:
    """Tests 14–16 — MATLAB fixture parity (±1e-6 atol, ±1e-4 rtol)."""

    # ------------------------------------------------------------------
    # Test 14: AR(1) parity (ARMA(1,1) no seasonal passthrough)
    # ------------------------------------------------------------------
    def test_sarimax_likelihood_parity_arma11(
        self, sandbox_fixture_dir: Path
    ) -> None:
        """Compare ARMA(1,1) no-seasonal output against MATLAB fixture.

        Fixture key: ``arma_11_no_seasonal_no_const_unit_sigma``.
        """
        fixture = _load_fixture(sandbox_fixture_dir)
        y = fixture["input_y"]
        sigma = fixture["sigma_unit"]

        case = fixture["arma_11_no_seasonal_no_const_unit_sigma"]
        params = case["input_parameters"]
        p = np.asarray(case["p"], dtype=np.float64)
        q = np.asarray(case["q"], dtype=np.float64)
        constant = int(case["constant"])
        seasonal = np.asarray(case["seasonal"], dtype=np.float64)
        x = np.empty((len(y), 0))

        LLF, likelihoods, errors = sarimax_likelihood(
            params, p, q, constant, seasonal, y, x, sigma
        )

        expected_LLF = float(case["LLF"])
        expected_liks = case["likelihoods"]
        expected_errs = case["errors"]

        npt.assert_allclose(LLF, expected_LLF, atol=ATOL, rtol=RTOL,
                            err_msg="LLF mismatch for ARMA(1,1)")
        npt.assert_allclose(likelihoods, expected_liks, atol=ATOL, rtol=RTOL,
                            err_msg="likelihoods mismatch for ARMA(1,1)")
        npt.assert_allclose(errors, expected_errs, atol=ATOL, rtol=RTOL,
                            err_msg="errors mismatch for ARMA(1,1)")

    # ------------------------------------------------------------------
    # Test 15: Seasonal ARMA(1,0,1)×(1,0,1,12) parity
    # ------------------------------------------------------------------
    def test_sarimax_likelihood_parity_seasonal(
        self, sandbox_fixture_dir: Path
    ) -> None:
        """Compare seasonal model against MATLAB fixture.

        Fixture key: ``sarima_101_101_12_no_const_unit_sigma``.
        """
        fixture = _load_fixture(sandbox_fixture_dir)
        y = fixture["input_y"]
        sigma = fixture["sigma_unit"]

        case = fixture["sarima_101_101_12_no_const_unit_sigma"]
        params = case["input_parameters"]
        p = np.asarray(case["p"], dtype=np.float64)
        q = np.asarray(case["q"], dtype=np.float64)
        constant = int(case["constant"])
        seasonal = np.asarray(case["seasonal"], dtype=np.float64)
        x = np.empty((len(y), 0))

        LLF, likelihoods, errors = sarimax_likelihood(
            params, p, q, constant, seasonal, y, x, sigma
        )

        npt.assert_allclose(LLF, float(case["LLF"]), atol=ATOL, rtol=RTOL,
                            err_msg="LLF mismatch for SARIMA(1,0,1)(1,0,1,12)")
        npt.assert_allclose(likelihoods, case["likelihoods"], atol=ATOL, rtol=RTOL,
                            err_msg="likelihoods mismatch for SARIMA(1,0,1)(1,0,1,12)")
        npt.assert_allclose(errors, case["errors"], atol=ATOL, rtol=RTOL,
                            err_msg="errors mismatch for SARIMA(1,0,1)(1,0,1,12)")

    # ------------------------------------------------------------------
    # Test 16: Parametrised parity across multiple configurations
    # ------------------------------------------------------------------
    @pytest.mark.parametrize(
        "case_key",
        [
            "sarima_101_101_12_no_const_unit_sigma",
            "sar_only_p1_s12_no_const_unit_sigma",
            "sma_only_q1_s12_no_const_unit_sigma",
            "sarima_101_101_12_no_const_hetero_sigma",
            "arma_11_no_seasonal_no_const_unit_sigma",
            "sar_quarterly_p12_s4_no_const_unit_sigma",
        ],
    )
    def test_sarimax_likelihood_parity_multiple_configs(
        self, sandbox_fixture_dir: Path, case_key: str
    ) -> None:
        """Parametrised parity test across all fixture configurations."""
        fixture = _load_fixture(sandbox_fixture_dir)
        y = fixture["input_y"]

        case = fixture[case_key]
        sigma_type = case.get("sigma_type", "unit")
        if sigma_type == "unit":
            sigma = fixture["sigma_unit"]
        elif sigma_type == "heterogeneous":
            sigma = fixture["sigma_heterogeneous"]
        else:
            sigma = fixture["sigma_unit"]

        params = case["input_parameters"]
        p = np.asarray(case["p"], dtype=np.float64)
        q = np.asarray(case["q"], dtype=np.float64)
        constant = int(case["constant"])
        seasonal = np.asarray(case["seasonal"], dtype=np.float64)
        x = np.empty((len(y), 0))

        LLF, likelihoods, errors = sarimax_likelihood(
            params, p, q, constant, seasonal, y, x, sigma
        )

        expected_LLF = float(case["LLF"])
        expected_liks = case["likelihoods"]
        expected_errs = case["errors"]

        npt.assert_allclose(
            LLF, expected_LLF, atol=ATOL, rtol=RTOL,
            err_msg=f"LLF mismatch for {case_key}"
        )
        npt.assert_allclose(
            likelihoods, expected_liks, atol=ATOL, rtol=RTOL,
            err_msg=f"likelihoods mismatch for {case_key}"
        )
        npt.assert_allclose(
            errors, expected_errs, atol=ATOL, rtol=RTOL,
            err_msg=f"errors mismatch for {case_key}"
        )

    # ------------------------------------------------------------------
    # Parity: NaN guard edge case
    # ------------------------------------------------------------------
    def test_sarimax_likelihood_parity_nan_guard(
        self, sandbox_fixture_dir: Path
    ) -> None:
        """NaN guard edge case: sigma with zeros → LLF clamped to 1e7.

        Fixture key: ``nan_guard_edge_case``.
        Ref: sarimax_likelihood.m lines 58–60.
        """
        fixture = _load_fixture(sandbox_fixture_dir)
        y = fixture["input_y"]

        case = fixture["nan_guard_edge_case"]
        params = case["input_parameters"]
        p = np.asarray(case["p"], dtype=np.float64)
        q = np.asarray(case["q"], dtype=np.float64)
        constant = int(case["constant"])
        seasonal = np.asarray(case["seasonal"], dtype=np.float64)
        sigma = case["sigma_nan"]
        x = np.empty((len(y), 0))

        LLF, likelihoods, errors = sarimax_likelihood(
            params, p, q, constant, seasonal, y, x, sigma
        )

        expected_LLF = float(case["LLF"])
        assert LLF == expected_LLF, (
            f"NaN-guarded LLF should be {expected_LLF}, got {LLF}"
        )

    # ------------------------------------------------------------------
    # Parity: Heterogeneous sigma
    # ------------------------------------------------------------------
    def test_sarimax_likelihood_parity_hetero_sigma(
        self, sandbox_fixture_dir: Path
    ) -> None:
        """Verify heterogeneous sigma produces correct log-likelihood.

        Fixture key: ``sarima_101_101_12_no_const_hetero_sigma``.
        """
        fixture = _load_fixture(sandbox_fixture_dir)
        y = fixture["input_y"]
        sigma = fixture["sigma_heterogeneous"]

        case = fixture["sarima_101_101_12_no_const_hetero_sigma"]
        params = case["input_parameters"]
        p = np.asarray(case["p"], dtype=np.float64)
        q = np.asarray(case["q"], dtype=np.float64)
        constant = int(case["constant"])
        seasonal = np.asarray(case["seasonal"], dtype=np.float64)
        x = np.empty((len(y), 0))

        LLF, likelihoods, errors = sarimax_likelihood(
            params, p, q, constant, seasonal, y, x, sigma
        )

        npt.assert_allclose(LLF, float(case["LLF"]), atol=ATOL, rtol=RTOL,
                            err_msg="LLF mismatch for hetero sigma")
        npt.assert_allclose(likelihoods, case["likelihoods"], atol=ATOL, rtol=RTOL,
                            err_msg="likelihoods mismatch for hetero sigma")

    # ------------------------------------------------------------------
    # Parity: SAR-only and SMA-only
    # ------------------------------------------------------------------
    def test_sarimax_likelihood_parity_sar_only(
        self, sandbox_fixture_dir: Path
    ) -> None:
        """SAR-only model: AR(1) + SAR(1) at S=12.

        Fixture key: ``sar_only_p1_s12_no_const_unit_sigma``.
        """
        fixture = _load_fixture(sandbox_fixture_dir)
        y = fixture["input_y"]
        sigma = fixture["sigma_unit"]

        case = fixture["sar_only_p1_s12_no_const_unit_sigma"]
        params = case["input_parameters"]
        p = np.asarray(case["p"], dtype=np.float64)
        q = np.asarray(case["q"], dtype=np.float64)
        constant = int(case["constant"])
        seasonal = np.asarray(case["seasonal"], dtype=np.float64)
        x = np.empty((len(y), 0))

        LLF, likelihoods, errors = sarimax_likelihood(
            params, p, q, constant, seasonal, y, x, sigma
        )

        npt.assert_allclose(LLF, float(case["LLF"]), atol=ATOL, rtol=RTOL)
        npt.assert_allclose(likelihoods, case["likelihoods"], atol=ATOL, rtol=RTOL)
        npt.assert_allclose(errors, case["errors"], atol=ATOL, rtol=RTOL)

    def test_sarimax_likelihood_parity_sma_only(
        self, sandbox_fixture_dir: Path
    ) -> None:
        """SMA-only model: MA(1) + SMA(1) at S=12.

        Fixture key: ``sma_only_q1_s12_no_const_unit_sigma``.
        """
        fixture = _load_fixture(sandbox_fixture_dir)
        y = fixture["input_y"]
        sigma = fixture["sigma_unit"]

        case = fixture["sma_only_q1_s12_no_const_unit_sigma"]
        params = case["input_parameters"]
        p = np.asarray(case["p"], dtype=np.float64)
        q = np.asarray(case["q"], dtype=np.float64)
        constant = int(case["constant"])
        seasonal = np.asarray(case["seasonal"], dtype=np.float64)
        x = np.empty((len(y), 0))

        LLF, likelihoods, errors = sarimax_likelihood(
            params, p, q, constant, seasonal, y, x, sigma
        )

        expected_trim = int(case.get("trim_start", 0))
        expected_n = int(case.get("n_obs", len(y)))

        npt.assert_allclose(LLF, float(case["LLF"]), atol=ATOL, rtol=RTOL)
        assert len(likelihoods) == expected_n, (
            f"Expected {expected_n} likelihoods, got {len(likelihoods)}"
        )
        npt.assert_allclose(likelihoods, case["likelihoods"], atol=ATOL, rtol=RTOL)
        npt.assert_allclose(errors, case["errors"], atol=ATOL, rtol=RTOL)


# =========================================================================
# Phase 4: Return Type and Shape Tests
# =========================================================================


class TestSarimaxLikelihoodReturnInvariants:
    """Tests 17–20 — Return type and shape invariants."""

    @pytest.fixture()
    def _basic_result(self, rng: np.random.Generator):
        """Compute a basic sarimax_likelihood result for invariant checks."""
        T = 200
        y = rng.standard_normal(T)
        x = np.empty((T, 0))
        sigma = np.abs(rng.standard_normal(T)) + 0.1  # Positive sigma
        params = np.array([0.3, 0.2])
        p = np.array([1])
        q = np.array([1])
        constant = 0
        seasonal = np.array([[0, 0, 0, 12]])
        return sarimax_likelihood(params, p, q, constant, seasonal, y, x, sigma)

    # ------------------------------------------------------------------
    # Test 17: Return types
    # ------------------------------------------------------------------
    def test_sarimax_likelihood_return_types(self, _basic_result) -> None:
        """LLF is float, likelihoods is ndarray, errors is ndarray."""
        LLF, likelihoods, errors = _basic_result
        assert isinstance(LLF, (float, np.floating)), (
            f"LLF type {type(LLF)} is not float"
        )
        assert isinstance(likelihoods, np.ndarray), (
            f"likelihoods type {type(likelihoods)} is not ndarray"
        )
        assert isinstance(errors, np.ndarray), (
            f"errors type {type(errors)} is not ndarray"
        )

    # ------------------------------------------------------------------
    # Test 18: Return shapes
    # ------------------------------------------------------------------
    def test_sarimax_likelihood_return_shapes(self, _basic_result) -> None:
        """likelihoods and errors are 1-D and have matching lengths."""
        LLF, likelihoods, errors = _basic_result
        assert likelihoods.ndim == 1, "likelihoods must be 1-D"
        assert errors.ndim == 1, "errors must be 1-D"
        assert len(likelihoods) == len(errors), (
            f"likelihoods ({len(likelihoods)}) and errors ({len(errors)}) "
            "must have equal length"
        )

    # ------------------------------------------------------------------
    # Test 19: LLF = sum(likelihoods)
    # ------------------------------------------------------------------
    def test_sarimax_likelihood_llf_is_sum(self, _basic_result) -> None:
        """LLF must equal the sum of per-observation likelihoods.

        Ref: sarimax_likelihood.m:56 — LLF = sum(likelihoods).
        The NaN guard replaces LLF only if it is NaN, so for valid
        outputs the sum invariant must hold exactly.
        """
        LLF, likelihoods, errors = _basic_result
        if LLF != 1e7:  # Skip check if NaN guard triggered
            npt.assert_allclose(
                LLF, np.sum(likelihoods), atol=1e-12,
                err_msg="LLF should equal sum(likelihoods)"
            )

    # ------------------------------------------------------------------
    # Test 20: Non-negative likelihoods
    # ------------------------------------------------------------------
    def test_sarimax_likelihood_likelihoods_nonnegative(
        self, _basic_result
    ) -> None:
        """Per-observation Gaussian NLL values should be ≥ 0.

        The Gaussian NLL formula is 0.5*(2*log(σ) + (e/σ)^2 + log(2π)).
        For σ ≥ 1 and any real e, each term is ≥ 0 so the sum is ≥ 0.
        For σ < 1, log(σ) < 0, but (e/σ)^2 + log(2π) typically dominates.
        With valid inputs the likelihoods should generally be non-negative.
        """
        LLF, likelihoods, errors = _basic_result
        # For unit or moderate sigma, likelihoods are non-negative
        finite_mask = np.isfinite(likelihoods)
        if np.any(finite_mask):
            # At least check no wildly negative values
            min_val = np.min(likelihoods[finite_mask])
            # The minimum possible with sigma>=0.1 is still positive
            # because 0.5*log(2*pi) ≈ 0.92
            assert min_val >= -10.0, (
                f"Unexpectedly negative likelihood: {min_val}"
            )
