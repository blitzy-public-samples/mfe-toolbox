"""
Pytest parity and unit tests for ``mfe_toolbox.timeseries.aicsbic``.

Tests the simpler two-output variant computing Akaike (AIC) and
Schwartz/Bayes (SBIC) Information Criteria against:
  1. Manually computed known values derived from the MATLAB formula
     (``timeseries/aicsbic.m`` lines 117-125).
  2. Cross-check consistency with ``aichqcsbic`` (three-output variant).
  3. MATLAB/Octave reference fixtures stored under
     ``tests/fixtures/timeseries/aicsbic.npy``.

MATLAB source reference: ``timeseries/aicsbic.m``
  Signature: ``[aic, sbic] = aicsbic(errors, constant, p, q, X)``
  Returns: AIC = ln(σ²) + 2K/T,  SBIC = ln(σ²) + K·ln(T)/T
  K = constant + length(unique(p)) + length(unique(q)) + cols(X)

Per AAP Section 0.7.1 all numerical comparisons use
``numpy.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-4)``.

Migrated from MATLAB source: timeseries/aicsbic.m
    Revision: 3    Date: 10/19/2009
    Original author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.timeseries.aicsbic import aicsbic
from mfe_toolbox.timeseries.aichqcsbic import aichqcsbic

# ---------------------------------------------------------------------------
# Module-level constants — AAP §0.7.1
# ---------------------------------------------------------------------------
ATOL: float = 1e-6
RTOL: float = 1e-4


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng() -> np.random.Generator:
    """Return a seeded random generator for reproducible test data."""
    return np.random.default_rng(42)


@pytest.fixture
def residuals(rng: np.random.Generator) -> np.ndarray:
    """T=200 standard-normal residual vector."""
    return rng.standard_normal(200)


@pytest.fixture
def large_residuals(rng: np.random.Generator) -> np.ndarray:
    """T=1000 standard-normal residual vector for penalty-behaviour tests.

    With T=1000 > e² ≈ 7.389, the SBIC penalty per parameter
    (ln(T)/T ≈ 0.006908) exceeds the AIC penalty (2/T = 0.002),
    ensuring SBIC ≥ AIC when K > 0.
    """
    return rng.standard_normal(1000)


# ---------------------------------------------------------------------------
# Helper — fixture directory resolution
# ---------------------------------------------------------------------------

def _fixture_dir() -> Path:
    """Return the timeseries fixture directory, honouring MFE_FIXTURE_DIR."""
    env_dir: str | None = os.environ.get("MFE_FIXTURE_DIR")
    if env_dir is not None and env_dir != "":
        return Path(env_dir) / "timeseries"
    return Path(__file__).resolve().parent.parent / "fixtures" / "timeseries"


# ---------------------------------------------------------------------------
# Test 1: Return structure — aicsbic must return exactly 2 values
# ---------------------------------------------------------------------------

class TestAicsbicReturnStructure:
    """Verify the function returns exactly two values (aic, sbic)."""

    def test_aicsbic_returns_two_values(self, residuals: np.ndarray) -> None:
        """aicsbic must return a 2-tuple (aic, sbic)."""
        result = aicsbic(
            residuals,
            1,
            np.array([1]),
            np.array([1]),
            None,
        )
        assert isinstance(result, tuple), (
            f"Expected tuple return, got {type(result)}"
        )
        assert len(result) == 2, (
            f"Expected 2 return values (aic, sbic), got {len(result)}"
        )

    def test_aicsbic_returns_two_values_minimal(self, residuals: np.ndarray) -> None:
        """aicsbic with minimal args (no q, no X) still returns 2-tuple."""
        result = aicsbic(residuals, 0, np.array([1]))
        assert len(result) == 2, (
            f"Expected 2 return values for minimal call, got {len(result)}"
        )


# ---------------------------------------------------------------------------
# Test 2: Output types — both outputs must be float scalars
# ---------------------------------------------------------------------------

class TestAicsbicOutputTypes:
    """Verify both outputs are Python/numpy float scalars."""

    def test_aicsbic_output_types(self, residuals: np.ndarray) -> None:
        """Each returned criterion must be a float scalar, not an array."""
        aic, sbic = aicsbic(
            residuals,
            1,
            np.array([1]),
            np.array([1]),
            None,
        )
        for name, val in [("aic", aic), ("sbic", sbic)]:
            assert isinstance(val, (float, np.floating)), (
                f"{name} should be a float scalar, got {type(val)}"
            )

    def test_aicsbic_output_not_array(self, residuals: np.ndarray) -> None:
        """Returned values must not be numpy arrays."""
        aic, sbic = aicsbic(residuals, 1, np.array([1, 2]), np.array([1]))
        for name, val in [("aic", aic), ("sbic", sbic)]:
            assert not isinstance(val, np.ndarray), (
                f"{name} should not be a numpy array"
            )


# ---------------------------------------------------------------------------
# Test 3: Known values (manual computation from MATLAB formula)
# ---------------------------------------------------------------------------

class TestAicsbicKnownValues:
    """Verify AIC/SBIC against manually computed expected values.

    MATLAB formula (aicsbic.m lines 117-125):
        T = length(errors);
        seregression = sqrt(errors' * errors / T);
        K = constant + length(unique(p)) + length(unique(q)) + size(X,2);
        AIC  = log(seregression^2) + 2*K / T;
        SBIC = log(seregression^2) + log(T)*K / T;

    Since seregression^2 = errors'*errors/T = σ², these simplify to:
        AIC  = log(σ²) + 2*K/T
        SBIC = log(σ²) + K*log(T)/T
    """

    def test_aicsbic_known_values_constant_errors(self) -> None:
        """T=100, errors=0.5, constant=1, p=[1], q=[] ⇒ K=2."""
        T = 100
        errors = np.ones(T) * 0.5
        constant = 1
        p = np.array([1])
        # q defaults to None (no MA terms)

        # Manual expected values
        # Ref: aicsbic.m:118 — seregression = sqrt(errors'*errors/T) = sqrt(0.25) = 0.5
        # Ref: aicsbic.m:124 — log(seregression^2) = log(0.25)
        sigma_sq = np.dot(errors, errors) / T  # 0.25
        # K = constant(1) + lp(1) + lq(0) + ncols_X(0) = 2
        K = constant + 1 + 0 + 0
        expected_aic = np.log(sigma_sq) + 2.0 * K / T
        expected_sbic = np.log(sigma_sq) + np.log(T) * K / T

        actual_aic, actual_sbic = aicsbic(errors, constant, p)

        npt.assert_allclose(actual_aic, expected_aic, atol=ATOL, rtol=RTOL,
                            err_msg="AIC mismatch for T=100 constant-error test")
        npt.assert_allclose(actual_sbic, expected_sbic, atol=ATOL, rtol=RTOL,
                            err_msg="SBIC mismatch for T=100 constant-error test")

    def test_aicsbic_known_values_larger_k(self) -> None:
        """T=200, errors=1.0, constant=1, p=[1,3], q=[2] ⇒ K=4."""
        T = 200
        errors = np.ones(T) * 1.0
        constant = 1
        p = np.array([1, 3])
        q = np.array([2])

        sigma_sq = np.dot(errors, errors) / T  # 1.0
        K = constant + 2 + 1 + 0  # 1 + 2 + 1 + 0 = 4
        expected_aic = np.log(sigma_sq) + 2.0 * K / T
        expected_sbic = np.log(sigma_sq) + np.log(T) * K / T

        aic, sbic = aicsbic(errors, constant, p, q)

        npt.assert_allclose(aic, expected_aic, atol=ATOL, rtol=RTOL,
                            err_msg="AIC mismatch for K=4 known-value test")
        npt.assert_allclose(sbic, expected_sbic, atol=ATOL, rtol=RTOL,
                            err_msg="SBIC mismatch for K=4 known-value test")

    def test_aicsbic_known_values_random_errors(self) -> None:
        """T=500 random errors, constant=1, p=[1], q=[1] ⇒ K=3."""
        rng = np.random.default_rng(123)
        T = 500
        errors = rng.standard_normal(T)
        constant = 1
        p = np.array([1])
        q = np.array([1])

        sigma_sq = np.dot(errors, errors) / T
        K = constant + 1 + 1 + 0  # 1 + 1 + 1 + 0 = 3
        expected_aic = np.log(sigma_sq) + 2.0 * K / T
        expected_sbic = np.log(sigma_sq) + np.log(T) * K / T

        aic, sbic = aicsbic(errors, constant, p, q)

        npt.assert_allclose(aic, expected_aic, atol=ATOL, rtol=RTOL,
                            err_msg="AIC mismatch for T=500 random-error test")
        npt.assert_allclose(sbic, expected_sbic, atol=ATOL, rtol=RTOL,
                            err_msg="SBIC mismatch for T=500 random-error test")


# ---------------------------------------------------------------------------
# Test 4: Cross-check with aichqcsbic
# ---------------------------------------------------------------------------

class TestAicsbicConsistentWithAichqcsbic:
    """AIC and SBIC from aicsbic must match those from aichqcsbic exactly."""

    def test_aicsbic_consistent_with_aichqcsbic(self) -> None:
        """Cross-check with aichqcsbic for ARMA(2,1) with constant=1."""
        errors = np.random.default_rng(42).standard_normal(200)
        aic1, sbic1 = aicsbic(errors, 1, np.array([1, 2]), np.array([1]), None)
        aic2, _, sbic2 = aichqcsbic(errors, 1, np.array([1, 2]), np.array([1]), 0)
        npt.assert_allclose(aic1, aic2, atol=ATOL, rtol=RTOL,
                            err_msg="AIC from aicsbic vs aichqcsbic mismatch")
        npt.assert_allclose(sbic1, sbic2, atol=ATOL, rtol=RTOL,
                            err_msg="SBIC from aicsbic vs aichqcsbic mismatch")

    def test_aicsbic_consistent_with_aichqcsbic_empty_q(self) -> None:
        """Cross-check for AR(1) only (q empty) with constant=1."""
        errors = np.random.default_rng(99).standard_normal(300)
        aic1, sbic1 = aicsbic(errors, 1, np.array([1]), np.array([]), None)
        aic2, _, sbic2 = aichqcsbic(errors, 1, np.array([1]), np.array([]), 0)
        npt.assert_allclose(aic1, aic2, atol=ATOL, rtol=RTOL,
                            err_msg="AIC mismatch for AR(1) cross-check")
        npt.assert_allclose(sbic1, sbic2, atol=ATOL, rtol=RTOL,
                            err_msg="SBIC mismatch for AR(1) cross-check")

    def test_aicsbic_consistent_with_aichqcsbic_no_constant(self) -> None:
        """Cross-check for MA(1) with constant=0."""
        errors = np.random.default_rng(7).standard_normal(150)
        aic1, sbic1 = aicsbic(errors, 0, np.array([]), np.array([1]), None)
        aic2, _, sbic2 = aichqcsbic(errors, 0, np.array([]), np.array([1]), 0)
        npt.assert_allclose(aic1, aic2, atol=ATOL, rtol=RTOL,
                            err_msg="AIC mismatch for MA(1) no-const cross-check")
        npt.assert_allclose(sbic1, sbic2, atol=ATOL, rtol=RTOL,
                            err_msg="SBIC mismatch for MA(1) no-const cross-check")

    def test_aicsbic_consistent_with_aichqcsbic_with_exog(self) -> None:
        """Cross-check when exogenous matrix X is supplied."""
        rng = np.random.default_rng(55)
        T = 200
        errors = rng.standard_normal(T)
        X_mat = rng.standard_normal((T, 2))
        aic1, sbic1 = aicsbic(errors, 1, np.array([1]), np.array([1]), X_mat)
        aic2, _, sbic2 = aichqcsbic(errors, 1, np.array([1]), np.array([1]), X_mat)
        npt.assert_allclose(aic1, aic2, atol=ATOL, rtol=RTOL,
                            err_msg="AIC mismatch with exogenous X cross-check")
        npt.assert_allclose(sbic1, sbic2, atol=ATOL, rtol=RTOL,
                            err_msg="SBIC mismatch with exogenous X cross-check")


# ---------------------------------------------------------------------------
# Test 5: SBIC penalizes more than AIC for T > e²
# ---------------------------------------------------------------------------

class TestAicsbicSbicPenalizesMore:
    """For T > e² ≈ 7.389, the SBIC penalty per-parameter exceeds AIC.

    AIC penalty coefficient = 2/T
    SBIC penalty coefficient = ln(T)/T

    When T > e² ≈ 7.389 we have ln(T) > 2, so SBIC penalty > AIC penalty.
    Since both share the same log-variance term, SBIC ≥ AIC when K > 0.
    """

    def test_aicsbic_sbic_penalizes_more_t200(self, residuals: np.ndarray) -> None:
        """With T=200, SBIC should be ≥ AIC (both criteria share log(σ²))."""
        aic, sbic = aicsbic(
            residuals,
            1,
            np.array([1, 2]),
            np.array([1]),
            None,
        )
        assert sbic >= aic, (
            f"SBIC ({sbic:.10f}) should be ≥ AIC ({aic:.10f}) for T=200, K=4"
        )

    def test_aicsbic_sbic_penalizes_more_t1000(
        self, large_residuals: np.ndarray
    ) -> None:
        """With T=1000, SBIC penalty is even more dominant relative to AIC."""
        aic, sbic = aicsbic(
            large_residuals,
            1,
            np.array([1, 2, 3]),
            np.array([1, 2]),
            None,
        )
        assert sbic >= aic, (
            f"SBIC ({sbic:.10f}) should be ≥ AIC ({aic:.10f}) for T=1000, K=6"
        )

    def test_aicsbic_sbic_equals_aic_when_k_zero(
        self, residuals: np.ndarray
    ) -> None:
        """With K=0 (constant=0, no p, no q, no X), AIC = SBIC = log(σ²)."""
        aic, sbic = aicsbic(residuals, 0, np.array([]), np.array([]), None)
        npt.assert_allclose(aic, sbic, atol=ATOL, rtol=RTOL,
                            err_msg="AIC and SBIC should be equal when K=0")

    def test_aicsbic_sbic_penalty_minus_aic_penalty(
        self, residuals: np.ndarray
    ) -> None:
        """Verify SBIC - AIC = K * (ln(T) - 2) / T for a given K."""
        T = len(residuals)
        p = np.array([1, 2])
        q = np.array([1])
        constant = 1
        # K = 1 + 2 + 1 = 4
        K = 4

        aic, sbic = aicsbic(residuals, constant, p, q, None)

        # SBIC - AIC = K * (ln(T) - 2) / T
        expected_diff = K * (np.log(T) - 2.0) / T
        npt.assert_allclose(sbic - aic, expected_diff, atol=ATOL, rtol=RTOL,
                            err_msg="SBIC-AIC penalty difference mismatch")


# ---------------------------------------------------------------------------
# Test 6: Empty lags — p=[], q=[] ⇒ K = constant + 0 + 0 + cols(X)
# ---------------------------------------------------------------------------

class TestAicsbicEmptyLags:
    """Verify correct behaviour when both AR and MA lag vectors are empty."""

    def test_aicsbic_empty_lags_with_constant(self, residuals: np.ndarray) -> None:
        """p=[] and q=[] with constant=1 ⇒ K=1."""
        T = len(residuals)
        constant = 1
        p = np.array([])
        q = np.array([])

        aic, sbic = aicsbic(residuals, constant, p, q)

        # Manual computation: K = 1 (constant only)
        sigma_sq = float(np.dot(residuals, residuals) / T)
        log_s2 = np.log(sigma_sq)
        K = 1
        expected_aic = log_s2 + 2.0 * K / T
        expected_sbic = log_s2 + np.log(T) * K / T

        npt.assert_allclose(aic, expected_aic, atol=ATOL, rtol=RTOL,
                            err_msg="AIC with empty p and q, constant=1")
        npt.assert_allclose(sbic, expected_sbic, atol=ATOL, rtol=RTOL,
                            err_msg="SBIC with empty p and q, constant=1")

    def test_aicsbic_empty_lags_no_constant(self, residuals: np.ndarray) -> None:
        """p=[] and q=[] with constant=0 ⇒ K=0, penalty vanishes."""
        T = len(residuals)
        aic, sbic = aicsbic(residuals, 0, np.array([]), np.array([]))

        # With K=0, both criteria equal log(σ²)
        sigma_sq = float(np.dot(residuals, residuals) / T)
        expected = np.log(sigma_sq)

        npt.assert_allclose(aic, expected, atol=ATOL, rtol=RTOL,
                            err_msg="AIC should equal log(σ²) when K=0")
        npt.assert_allclose(sbic, expected, atol=ATOL, rtol=RTOL,
                            err_msg="SBIC should equal log(σ²) when K=0")

    def test_aicsbic_scalar_zero_lags(self, residuals: np.ndarray) -> None:
        """p=0 and q=0 (scalar zeros) should behave like empty lags."""
        aic_empty, sbic_empty = aicsbic(
            residuals, 1, np.array([]), np.array([])
        )
        aic_zero, sbic_zero = aicsbic(residuals, 1, 0, 0)

        npt.assert_allclose(aic_zero, aic_empty, atol=ATOL, rtol=RTOL,
                            err_msg="AIC: scalar-zero p/q vs empty p/q mismatch")
        npt.assert_allclose(sbic_zero, sbic_empty, atol=ATOL, rtol=RTOL,
                            err_msg="SBIC: scalar-zero p/q vs empty p/q mismatch")


# ---------------------------------------------------------------------------
# Test 7: Constant effect — constant=1 adds exactly 1 to K
# ---------------------------------------------------------------------------

class TestAicsbicWithConstant:
    """Verify that constant=1 increases K by exactly 1 relative to constant=0."""

    def test_aicsbic_with_constant(self, residuals: np.ndarray) -> None:
        """IC(constant=1) - IC(constant=0) = penalty_factor * 1/T."""
        T = len(residuals)
        p = np.array([1])
        q = np.array([1])

        aic_c1, sbic_c1 = aicsbic(residuals, 1, p, q)
        aic_c0, sbic_c0 = aicsbic(residuals, 0, p, q)

        # Expected penalty differences when K changes by 1
        # Ref: aicsbic.m:124 — AIC penalty = 2*K/T  → diff = 2/T
        expected_aic_diff = 2.0 / T
        # Ref: aicsbic.m:125 — SBIC penalty = K*log(T)/T → diff = log(T)/T
        expected_sbic_diff = np.log(T) / T

        npt.assert_allclose(
            aic_c1 - aic_c0, expected_aic_diff, atol=ATOL, rtol=RTOL,
            err_msg="AIC constant=1 vs constant=0 penalty difference"
        )
        npt.assert_allclose(
            sbic_c1 - sbic_c0, expected_sbic_diff, atol=ATOL, rtol=RTOL,
            err_msg="SBIC constant=1 vs constant=0 penalty difference"
        )

    def test_aicsbic_constant_adds_exactly_one(
        self, large_residuals: np.ndarray
    ) -> None:
        """Constant effect is consistent across different sample sizes."""
        T = len(large_residuals)
        p = np.array([1, 2])
        q = np.array([1])

        aic_c1, sbic_c1 = aicsbic(large_residuals, 1, p, q)
        aic_c0, sbic_c0 = aicsbic(large_residuals, 0, p, q)

        expected_aic_diff = 2.0 / T
        expected_sbic_diff = np.log(T) / T

        npt.assert_allclose(
            aic_c1 - aic_c0, expected_aic_diff, atol=ATOL, rtol=RTOL,
            err_msg="AIC constant effect for T=1000"
        )
        npt.assert_allclose(
            sbic_c1 - sbic_c0, expected_sbic_diff, atol=ATOL, rtol=RTOL,
            err_msg="SBIC constant effect for T=1000"
        )


# ---------------------------------------------------------------------------
# Test 8: MATLAB fixture parity
# ---------------------------------------------------------------------------

class TestAicsbicParity:
    """Numerical parity tests against MATLAB/Octave reference fixtures.

    The fixture file ``tests/fixtures/timeseries/aicsbic.npy`` contains
    a dictionary with scenario keys, each holding:
      - errors: the residual vector
      - constant: 0 or 1
      - p: AR lag vector
      - q: MA lag vector
      - X: exogenous matrix (may be empty)
      - T: sample size
      - aic: MATLAB-computed AIC
      - sbic: MATLAB-computed SBIC
    """

    @pytest.mark.parity
    def test_aicsbic_parity_scenario_1(self) -> None:
        """ARMA(1,1): constant=1, p=[1], q=[1], T=1000."""
        fdir = _fixture_dir()
        fpath = fdir / "aicsbic.npy"
        if not fpath.exists():
            pytest.skip(f"Fixture file not found: {fpath}")

        data = np.load(fpath, allow_pickle=True).item()
        scenario = data["scenario_1"]

        errors = scenario["errors"]
        constant = int(scenario["constant"])
        p_val = scenario["p"]
        q_val = scenario["q"]
        x_val = scenario.get("X", None)

        # Handle empty X from fixture (may be empty ndarray)
        if isinstance(x_val, np.ndarray) and x_val.size == 0:
            x_val = None

        aic, sbic = aicsbic(errors, constant, p_val, q_val, x_val)

        expected_aic = float(scenario["aic"])
        expected_sbic = float(scenario["sbic"])

        npt.assert_allclose(aic, expected_aic, atol=ATOL, rtol=RTOL,
                            err_msg=f"AIC parity: {scenario['description']}")
        npt.assert_allclose(sbic, expected_sbic, atol=ATOL, rtol=RTOL,
                            err_msg=f"SBIC parity: {scenario['description']}")

    @pytest.mark.parity
    def test_aicsbic_parity_scenario_2(self) -> None:
        """AR(2): constant=1, p=[1,2], q=[], T=1000."""
        fdir = _fixture_dir()
        fpath = fdir / "aicsbic.npy"
        if not fpath.exists():
            pytest.skip(f"Fixture file not found: {fpath}")

        data = np.load(fpath, allow_pickle=True).item()
        scenario = data["scenario_2"]

        errors = scenario["errors"]
        constant = int(scenario["constant"])
        p_val = scenario["p"]
        q_val = scenario["q"]
        x_val = scenario.get("X", None)
        if isinstance(x_val, np.ndarray) and x_val.size == 0:
            x_val = None

        aic, sbic = aicsbic(errors, constant, p_val, q_val, x_val)

        expected_aic = float(scenario["aic"])
        expected_sbic = float(scenario["sbic"])

        npt.assert_allclose(aic, expected_aic, atol=ATOL, rtol=RTOL,
                            err_msg=f"AIC parity: {scenario['description']}")
        npt.assert_allclose(sbic, expected_sbic, atol=ATOL, rtol=RTOL,
                            err_msg=f"SBIC parity: {scenario['description']}")

    @pytest.mark.parity
    def test_aicsbic_parity_scenario_3(self) -> None:
        """ARMA(2,1)+exog: constant=1, p=[1,2], q=[1], X=Tx1, T=1000."""
        fdir = _fixture_dir()
        fpath = fdir / "aicsbic.npy"
        if not fpath.exists():
            pytest.skip(f"Fixture file not found: {fpath}")

        data = np.load(fpath, allow_pickle=True).item()
        scenario = data["scenario_3"]

        errors = scenario["errors"]
        constant = int(scenario["constant"])
        p_val = scenario["p"]
        q_val = scenario["q"]
        x_val = scenario.get("X", None)
        # For scenario_3, X is a (1000, 1) matrix — pass as-is
        if isinstance(x_val, np.ndarray) and x_val.size == 0:
            x_val = None

        aic, sbic = aicsbic(errors, constant, p_val, q_val, x_val)

        expected_aic = float(scenario["aic"])
        expected_sbic = float(scenario["sbic"])

        npt.assert_allclose(aic, expected_aic, atol=ATOL, rtol=RTOL,
                            err_msg=f"AIC parity: {scenario['description']}")
        npt.assert_allclose(sbic, expected_sbic, atol=ATOL, rtol=RTOL,
                            err_msg=f"SBIC parity: {scenario['description']}")

    @pytest.mark.parity
    def test_aicsbic_parity_scenario_4(self) -> None:
        """MA(1) no constant: constant=0, p=[], q=[1], T=1000."""
        fdir = _fixture_dir()
        fpath = fdir / "aicsbic.npy"
        if not fpath.exists():
            pytest.skip(f"Fixture file not found: {fpath}")

        data = np.load(fpath, allow_pickle=True).item()
        scenario = data["scenario_4"]

        errors = scenario["errors"]
        constant = int(scenario["constant"])
        p_val = scenario["p"]
        q_val = scenario["q"]
        x_val = scenario.get("X", None)
        if isinstance(x_val, np.ndarray) and x_val.size == 0:
            x_val = None

        aic, sbic = aicsbic(errors, constant, p_val, q_val, x_val)

        expected_aic = float(scenario["aic"])
        expected_sbic = float(scenario["sbic"])

        npt.assert_allclose(aic, expected_aic, atol=ATOL, rtol=RTOL,
                            err_msg=f"AIC parity: {scenario['description']}")
        npt.assert_allclose(sbic, expected_sbic, atol=ATOL, rtol=RTOL,
                            err_msg=f"SBIC parity: {scenario['description']}")

    @pytest.mark.parity
    def test_aicsbic_parity_all_scenarios(self) -> None:
        """Parametrized sweep over all fixture scenarios."""
        fdir = _fixture_dir()
        fpath = fdir / "aicsbic.npy"
        if not fpath.exists():
            pytest.skip(f"Fixture file not found: {fpath}")

        data = np.load(fpath, allow_pickle=True).item()

        for scenario_key, scenario in data.items():
            errors = scenario["errors"]
            constant = int(scenario["constant"])
            p_val = scenario["p"]
            q_val = scenario["q"]
            x_val = scenario.get("X", None)
            if isinstance(x_val, np.ndarray) and x_val.size == 0:
                x_val = None

            aic, sbic = aicsbic(errors, constant, p_val, q_val, x_val)

            expected_aic = float(scenario["aic"])
            expected_sbic = float(scenario["sbic"])

            npt.assert_allclose(
                aic, expected_aic, atol=ATOL, rtol=RTOL,
                err_msg=f"AIC parity failed for {scenario_key}: "
                        f"{scenario['description']}",
            )
            npt.assert_allclose(
                sbic, expected_sbic, atol=ATOL, rtol=RTOL,
                err_msg=f"SBIC parity failed for {scenario_key}: "
                        f"{scenario['description']}",
            )


# ---------------------------------------------------------------------------
# Test 9: Exogenous variable column counting
# ---------------------------------------------------------------------------

class TestAicsbicWithExogenous:
    """Verify that exogenous regressor columns correctly increase K."""

    def test_aicsbic_exogenous_adds_columns(self, residuals: np.ndarray) -> None:
        """Passing X as a T×3 matrix adds 3 to K relative to X=None."""
        T = len(residuals)
        p = np.array([1])
        q = np.array([1])

        aic_no_x, sbic_no_x = aicsbic(residuals, 1, p, q)

        rng_local = np.random.default_rng(99)
        X_mat = rng_local.standard_normal((T, 3))
        aic_x3, sbic_x3 = aicsbic(residuals, 1, p, q, X_mat)

        delta_K = 3
        expected_aic_diff = 2.0 * delta_K / T
        expected_sbic_diff = np.log(T) * delta_K / T

        npt.assert_allclose(
            aic_x3 - aic_no_x, expected_aic_diff, atol=ATOL, rtol=RTOL,
            err_msg="AIC difference for X=Tx3 vs X=None"
        )
        npt.assert_allclose(
            sbic_x3 - sbic_no_x, expected_sbic_diff, atol=ATOL, rtol=RTOL,
            err_msg="SBIC difference for X=Tx3 vs X=None"
        )

    def test_aicsbic_single_exogenous_column(self, residuals: np.ndarray) -> None:
        """A single exogenous column (T,1) adds 1 to K."""
        T = len(residuals)
        p = np.array([1])
        q = np.array([])
        rng_local = np.random.default_rng(88)
        X_col = rng_local.standard_normal((T, 1))

        aic_no_x, sbic_no_x = aicsbic(residuals, 1, p, q)
        aic_x1, sbic_x1 = aicsbic(residuals, 1, p, q, X_col)

        expected_aic_diff = 2.0 / T
        expected_sbic_diff = np.log(T) / T

        npt.assert_allclose(
            aic_x1 - aic_no_x, expected_aic_diff, atol=ATOL, rtol=RTOL,
            err_msg="AIC difference for X=Tx1 vs X=None"
        )
        npt.assert_allclose(
            sbic_x1 - sbic_no_x, expected_sbic_diff, atol=ATOL, rtol=RTOL,
            err_msg="SBIC difference for X=Tx1 vs X=None"
        )


# ---------------------------------------------------------------------------
# Test 10: Input validation (error paths)
# ---------------------------------------------------------------------------

class TestAicsbicInputValidation:
    """Verify that invalid inputs raise ValueError as the MATLAB source does."""

    def test_aicsbic_empty_errors_raises(self) -> None:
        """Empty error vector must raise ValueError.

        Ref: aicsbic.m:54-55 — ERRORS is empty.
        """
        with pytest.raises(ValueError, match="ERRORS"):
            aicsbic(np.array([]), 1, np.array([1]))

    def test_aicsbic_scalar_error_raises(self) -> None:
        """Scalar error input must raise ValueError.

        Ref: aicsbic.m:52 — length(errors)==1 rejects single-element.
        """
        with pytest.raises(ValueError, match="ERRORS"):
            aicsbic(np.array([1.0]), 1, np.array([1]))

    def test_aicsbic_row_vector_errors_raises(self) -> None:
        """Row vector (2D with cols > 1) must raise ValueError.

        Ref: aicsbic.m:52 — size(errors,2) > 1 rejects row vectors.
        """
        errors = np.ones((1, 100))
        with pytest.raises(ValueError, match="ERRORS"):
            aicsbic(errors, 1, np.array([1]))

    def test_aicsbic_negative_p_raises(self) -> None:
        """Negative AR lag indices must raise ValueError.

        Ref: aicsbic.m:70-71 — P must contain non-negative integers only.
        """
        errors = np.random.default_rng(0).standard_normal(100)
        with pytest.raises(ValueError, match="P must contain non-negative"):
            aicsbic(errors, 1, np.array([-1]))

    def test_aicsbic_noninteger_p_raises(self) -> None:
        """Non-integer AR lag indices must raise ValueError.

        Ref: aicsbic.m:70-71 — floor(p) ~= p check.
        """
        errors = np.random.default_rng(0).standard_normal(100)
        with pytest.raises(ValueError, match="P must contain non-negative"):
            aicsbic(errors, 1, np.array([1.5]))

    def test_aicsbic_negative_q_raises(self) -> None:
        """Negative MA lag indices must raise ValueError.

        Ref: aicsbic.m:95-96 — Q must contain non-negative integers only.
        """
        errors = np.random.default_rng(0).standard_normal(100)
        with pytest.raises(ValueError, match="Q must contain non-negative"):
            aicsbic(errors, 1, np.array([1]), np.array([-1]))

    def test_aicsbic_invalid_constant_raises(self) -> None:
        """Constant not in {0, 1} must raise ValueError.

        Ref: aicsbic.m:111-113 — CONSTANT must be 0 or 1.
        """
        errors = np.random.default_rng(0).standard_normal(100)
        with pytest.raises(ValueError, match="CONSTANT must be 0 or 1"):
            aicsbic(errors, 2, np.array([1]))

    def test_aicsbic_too_many_ar_lags_raises(self) -> None:
        """max(P) >= T/2 must raise ValueError.

        Ref: aicsbic.m:73-74 — Too many lags in the AR.
        """
        T = 20
        errors = np.random.default_rng(0).standard_normal(T)
        with pytest.raises(ValueError, match="Too many lags"):
            aicsbic(errors, 1, np.array([11]))  # 11 >= 20/2 = 10


# ---------------------------------------------------------------------------
# Test 11: Additional edge cases and formula verification
# ---------------------------------------------------------------------------

class TestAicsbicEdgeCases:
    """Additional edge case tests for robustness."""

    def test_aicsbic_column_vector_input(self) -> None:
        """A (T, 1) column vector should be accepted and handled correctly.

        Ref: aicsbic.m:52 — size(errors,2) > 1 check; (T,1) passes.
        """
        rng = np.random.default_rng(42)
        T = 100
        errors_1d = rng.standard_normal(T)
        errors_2d = errors_1d.reshape(-1, 1)

        aic_1d, sbic_1d = aicsbic(errors_1d, 1, np.array([1]))
        aic_2d, sbic_2d = aicsbic(errors_2d, 1, np.array([1]))

        npt.assert_allclose(aic_1d, aic_2d, atol=ATOL, rtol=RTOL,
                            err_msg="AIC should match for 1D vs column vector")
        npt.assert_allclose(sbic_1d, sbic_2d, atol=ATOL, rtol=RTOL,
                            err_msg="SBIC should match for 1D vs column vector")

    def test_aicsbic_list_inputs(self) -> None:
        """p and q as Python lists should work (converted via np.asarray)."""
        rng = np.random.default_rng(42)
        errors = rng.standard_normal(200)

        aic_list, sbic_list = aicsbic(errors, 1, [1, 2], [1])
        aic_arr, sbic_arr = aicsbic(
            errors, 1, np.array([1, 2]), np.array([1])
        )

        npt.assert_allclose(aic_list, aic_arr, atol=ATOL, rtol=RTOL,
                            err_msg="AIC should match for list vs array p/q")
        npt.assert_allclose(sbic_list, sbic_arr, atol=ATOL, rtol=RTOL,
                            err_msg="SBIC should match for list vs array p/q")

    def test_aicsbic_aic_formula_decomposition(self) -> None:
        """Decompose AIC = log(σ²) + 2K/T and verify each term."""
        rng = np.random.default_rng(42)
        T = 300
        errors = rng.standard_normal(T)
        constant = 1
        p = np.array([1, 3])
        q = np.array([2])

        aic, sbic = aicsbic(errors, constant, p, q)

        sigma_sq = np.dot(errors, errors) / T
        log_s2 = np.log(sigma_sq)
        K = constant + 2 + 1  # 1 + 2 + 1 = 4

        # Verify AIC = log(σ²) + 2K/T
        aic_penalty = 2.0 * K / T
        npt.assert_allclose(aic, log_s2 + aic_penalty, atol=ATOL, rtol=RTOL,
                            err_msg="AIC formula decomposition")

        # Verify SBIC = log(σ²) + K*ln(T)/T
        sbic_penalty = K * np.log(T) / T
        npt.assert_allclose(sbic, log_s2 + sbic_penalty, atol=ATOL, rtol=RTOL,
                            err_msg="SBIC formula decomposition")

    def test_aicsbic_monotone_in_k(self, residuals: np.ndarray) -> None:
        """More parameters (higher K) should increase both AIC and SBIC
        (since the variance term is identical for the same errors)."""
        # K=1: constant only
        aic1, sbic1 = aicsbic(residuals, 1, np.array([]), np.array([]))
        # K=3: constant + AR(1) + MA(1)
        aic3, sbic3 = aicsbic(residuals, 1, np.array([1]), np.array([1]))
        # K=5: constant + AR(1,2) + MA(1,2)
        aic5, sbic5 = aicsbic(residuals, 1, np.array([1, 2]), np.array([1, 2]))

        assert aic1 < aic3 < aic5, (
            f"AIC should increase with K: AIC(K=1)={aic1:.8f}, "
            f"AIC(K=3)={aic3:.8f}, AIC(K=5)={aic5:.8f}"
        )
        assert sbic1 < sbic3 < sbic5, (
            f"SBIC should increase with K: SBIC(K=1)={sbic1:.8f}, "
            f"SBIC(K=3)={sbic3:.8f}, SBIC(K=5)={sbic5:.8f}"
        )
