"""
Pytest parity and unit tests for ``mfe_toolbox.timeseries.aichqcsbic``.

Tests the Akaike (AIC), Hannan-Quinn (HQC), and Schwarz/Bayes (SBIC)
information criteria computation against:
  1. Manually computed known values derived from the MATLAB formula
     (``timeseries/aichqcsbic.m`` lines 116-125).
  2. MATLAB/Octave reference fixtures stored under
     ``tests/fixtures/timeseries/aichqcsbic.npy``.

Per AAP Section 0.7.1 all numerical comparisons use
``numpy.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-4)``.

Migrated from MATLAB source: timeseries/aichqcsbic.m
    Revision: 3    Date: 10/19/2009
    Original author: Kevin Sheppard (kevin.sheppard@economics.ox.ac.uk)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

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
    """T=1000 standard-normal residual vector for penalty-behaviour tests."""
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
# Test 1: Return structure
# ---------------------------------------------------------------------------

class TestAichqcsbicReturnStructure:
    """Verify the function returns exactly three values."""

    def test_aichqcsbic_returns_three_values(self, residuals: np.ndarray) -> None:
        """aichqcsbic must return a 3-tuple (aic, hqc, sbic)."""
        result = aichqcsbic(
            residuals,
            1,
            np.array([1]),
            np.array([1]),
            0,
        )
        assert len(result) == 3, (
            f"Expected 3 return values (aic, hqc, sbic), got {len(result)}"
        )


# ---------------------------------------------------------------------------
# Test 2: Output types
# ---------------------------------------------------------------------------

class TestAichqcsbicOutputTypes:
    """Verify all three outputs are Python/numpy float scalars."""

    def test_aichqcsbic_output_types(self, residuals: np.ndarray) -> None:
        """Each returned criterion must be a float scalar, not an array."""
        aic, hqc, sbic = aichqcsbic(
            residuals,
            1,
            np.array([1]),
            np.array([1]),
            0,
        )
        for name, val in [("aic", aic), ("hqc", hqc), ("sbic", sbic)]:
            assert isinstance(val, (float, np.floating)), (
                f"{name} should be a float scalar, got {type(val)}"
            )


# ---------------------------------------------------------------------------
# Test 3: Known values (manual computation from MATLAB formula)
# ---------------------------------------------------------------------------

class TestAichqcsbicKnownValues:
    """Verify criteria against manually computed expected values.

    MATLAB formula (aichqcsbic.m lines 116-125):
        T = length(errors);
        sigma_sq = errors' * errors / T;
        K = constant + length(unique(p)) + length(unique(q)) + size(X, 2);
        AIC  = log(sigma_sq) + 2*K / T;
        HQC  = log(sigma_sq) + 2*K * log(log(T)) / T;
        SBIC = log(sigma_sq) + log(T) * K / T;
    """

    def test_aichqcsbic_known_values(self) -> None:
        """T=100, errors=0.5, constant=1, p=[1], q=[], X=0."""
        T = 100
        errors = np.ones(T) * 0.5
        constant = 1
        p = np.array([1])
        q = np.array([])  # empty — contributes 0 to K

        # Manual expected values
        # Ref: aichqcsbic.m:117 — sigma_sq = errors'*errors/T
        sigma_sq = np.dot(errors, errors) / T  # 0.25
        # Ref: aichqcsbic.m:119-121 — K = constant + lp + lq + ncols_X
        K = constant + len(np.unique(p)) + 0 + 0  # 1 + 1 + 0 + 0 = 2
        expected_aic = np.log(sigma_sq) + 2.0 * K / T
        expected_hqc = np.log(sigma_sq) + 2.0 * K * np.log(np.log(T)) / T
        expected_sbic = np.log(sigma_sq) + np.log(T) * K / T

        actual_aic, actual_hqc, actual_sbic = aichqcsbic(
            errors, constant, p, q, 0
        )

        npt.assert_allclose(actual_aic, expected_aic, atol=ATOL, rtol=RTOL,
                            err_msg="AIC mismatch for known-value test")
        npt.assert_allclose(actual_hqc, expected_hqc, atol=ATOL, rtol=RTOL,
                            err_msg="HQC mismatch for known-value test")
        npt.assert_allclose(actual_sbic, expected_sbic, atol=ATOL, rtol=RTOL,
                            err_msg="SBIC mismatch for known-value test")

    def test_aichqcsbic_known_values_larger_k(self) -> None:
        """T=200, errors=1.0, constant=1, p=[1,3], q=[2], X=0  ⇒  K=4."""
        T = 200
        errors = np.ones(T) * 1.0
        constant = 1
        p = np.array([1, 3])
        q = np.array([2])

        sigma_sq = np.dot(errors, errors) / T  # 1.0
        K = constant + 2 + 1 + 0  # 1 + 2 + 1 + 0 = 4
        expected_aic = np.log(sigma_sq) + 2.0 * K / T
        expected_hqc = np.log(sigma_sq) + 2.0 * K * np.log(np.log(T)) / T
        expected_sbic = np.log(sigma_sq) + np.log(T) * K / T

        aic, hqc, sbic = aichqcsbic(errors, constant, p, q, 0)

        npt.assert_allclose(aic, expected_aic, atol=ATOL, rtol=RTOL,
                            err_msg="AIC mismatch for K=4 known-value test")
        npt.assert_allclose(hqc, expected_hqc, atol=ATOL, rtol=RTOL,
                            err_msg="HQC mismatch for K=4 known-value test")
        npt.assert_allclose(sbic, expected_sbic, atol=ATOL, rtol=RTOL,
                            err_msg="SBIC mismatch for K=4 known-value test")


# ---------------------------------------------------------------------------
# Test 4: Penalty ordering  AIC ≤ HQC ≤ SBIC  for T ≥ 16
# ---------------------------------------------------------------------------

class TestAichqcsbicOrdering:
    """For moderate T the penalty coefficients satisfy 2/T ≤ 2*ln(ln(T))/T ≤ ln(T)/T."""

    def test_aichqcsbic_ordering(self, residuals: np.ndarray) -> None:
        """AIC ≤ HQC ≤ SBIC with T=200, p=[1,2], q=[1]."""
        aic, hqc, sbic = aichqcsbic(
            residuals,
            1,
            np.array([1, 2]),
            np.array([1]),
            0,
        )
        assert aic <= hqc, (
            f"AIC ({aic:.8f}) should be ≤ HQC ({hqc:.8f}) for T=200"
        )
        assert hqc <= sbic, (
            f"HQC ({hqc:.8f}) should be ≤ SBIC ({sbic:.8f}) for T=200"
        )

    def test_aichqcsbic_ordering_large_T(self, large_residuals: np.ndarray) -> None:
        """AIC ≤ HQC ≤ SBIC with T=1000, p=[1,2,3], q=[1,2]."""
        aic, hqc, sbic = aichqcsbic(
            large_residuals,
            1,
            np.array([1, 2, 3]),
            np.array([1, 2]),
            0,
        )
        assert aic <= hqc, (
            f"AIC ({aic:.8f}) should be ≤ HQC ({hqc:.8f}) for T=1000"
        )
        assert hqc <= sbic, (
            f"HQC ({hqc:.8f}) should be ≤ SBIC ({sbic:.8f}) for T=1000"
        )


# ---------------------------------------------------------------------------
# Test 5: Constant=0 vs Constant=1 penalty difference
# ---------------------------------------------------------------------------

class TestAichqcsbicConstantEffect:
    """Verify that constant=1 increases K by exactly 1 relative to constant=0."""

    def test_aichqcsbic_constant_zero(self, residuals: np.ndarray) -> None:
        """IC(constant=1) - IC(constant=0) = penalty_factor * 1/T."""
        T = len(residuals)
        p = np.array([1])
        q = np.array([1])

        aic_c1, hqc_c1, sbic_c1 = aichqcsbic(residuals, 1, p, q, 0)
        aic_c0, hqc_c0, sbic_c0 = aichqcsbic(residuals, 0, p, q, 0)

        # Expected penalty differences when K changes by 1
        # Ref: aichqcsbic.m:123 — AIC penalty = 2*K/T  → diff = 2/T
        expected_aic_diff = 2.0 / T
        # Ref: aichqcsbic.m:124 — HQC penalty = 2*K*log(log(T))/T → diff = 2*log(log(T))/T
        expected_hqc_diff = 2.0 * np.log(np.log(T)) / T
        # Ref: aichqcsbic.m:125 — SBIC penalty = K*log(T)/T → diff = log(T)/T
        expected_sbic_diff = np.log(T) / T

        npt.assert_allclose(
            aic_c1 - aic_c0, expected_aic_diff, atol=ATOL, rtol=RTOL,
            err_msg="AIC constant=1 vs constant=0 penalty difference"
        )
        npt.assert_allclose(
            hqc_c1 - hqc_c0, expected_hqc_diff, atol=ATOL, rtol=RTOL,
            err_msg="HQC constant=1 vs constant=0 penalty difference"
        )
        npt.assert_allclose(
            sbic_c1 - sbic_c0, expected_sbic_diff, atol=ATOL, rtol=RTOL,
            err_msg="SBIC constant=1 vs constant=0 penalty difference"
        )


# ---------------------------------------------------------------------------
# Test 6: Empty p and q
# ---------------------------------------------------------------------------

class TestAichqcsbicEmptyPQ:
    """Verify correct behaviour when both AR and MA lag vectors are empty."""

    def test_aichqcsbic_empty_p_q(self, residuals: np.ndarray) -> None:
        """p=[] and q=[] should contribute 0 to K.  K = constant + 0 + 0 + 0."""
        T = len(residuals)
        constant = 1
        p = np.array([])
        q = np.array([])

        aic, hqc, sbic = aichqcsbic(residuals, constant, p, q, 0)

        # Manual expected values: K=1 (constant only)
        sigma_sq = float(np.dot(residuals, residuals) / T)
        log_s2 = np.log(sigma_sq)
        K = 1
        expected_aic = log_s2 + 2.0 * K / T
        expected_hqc = log_s2 + 2.0 * K * np.log(np.log(T)) / T
        expected_sbic = log_s2 + np.log(T) * K / T

        npt.assert_allclose(aic, expected_aic, atol=ATOL, rtol=RTOL,
                            err_msg="AIC with empty p and q")
        npt.assert_allclose(hqc, expected_hqc, atol=ATOL, rtol=RTOL,
                            err_msg="HQC with empty p and q")
        npt.assert_allclose(sbic, expected_sbic, atol=ATOL, rtol=RTOL,
                            err_msg="SBIC with empty p and q")

    def test_aichqcsbic_empty_p_q_no_constant(self, residuals: np.ndarray) -> None:
        """p=[] and q=[] with constant=0 ⇒ K=0, penalty terms vanish."""
        T = len(residuals)
        aic, hqc, sbic = aichqcsbic(residuals, 0, np.array([]), np.array([]), 0)

        # With K=0 all penalties are zero; all three criteria equal log(sigma_sq)
        sigma_sq = float(np.dot(residuals, residuals) / T)
        expected = np.log(sigma_sq)

        npt.assert_allclose(aic, expected, atol=ATOL, rtol=RTOL,
                            err_msg="AIC should equal log(sigma_sq) when K=0")
        npt.assert_allclose(hqc, expected, atol=ATOL, rtol=RTOL,
                            err_msg="HQC should equal log(sigma_sq) when K=0")
        npt.assert_allclose(sbic, expected, atol=ATOL, rtol=RTOL,
                            err_msg="SBIC should equal log(sigma_sq) when K=0")


# ---------------------------------------------------------------------------
# Test 7: Exogenous variable count (scalar X)
# ---------------------------------------------------------------------------

class TestAichqcsbicWithExogenous:
    """Verify that exogenous regressor columns correctly increase K."""

    def test_aichqcsbic_with_exogenous_scalar(self, residuals: np.ndarray) -> None:
        """Passing X as a T×3 matrix should add 3 to K relative to X=0."""
        T = len(residuals)
        p = np.array([1])
        q = np.array([1])

        aic_no_x, hqc_no_x, sbic_no_x = aichqcsbic(residuals, 1, p, q, 0)

        # Create an exogenous matrix with 3 columns
        rng = np.random.default_rng(99)
        X_mat = rng.standard_normal((T, 3))
        aic_x3, hqc_x3, sbic_x3 = aichqcsbic(residuals, 1, p, q, X_mat)

        # Difference in K = 3 ⇒ expected penalty differences
        delta_K = 3
        expected_aic_diff = 2.0 * delta_K / T
        expected_hqc_diff = 2.0 * delta_K * np.log(np.log(T)) / T
        expected_sbic_diff = np.log(T) * delta_K / T

        npt.assert_allclose(
            aic_x3 - aic_no_x, expected_aic_diff, atol=ATOL, rtol=RTOL,
            err_msg="AIC difference for X=Tx3 vs X=0"
        )
        npt.assert_allclose(
            hqc_x3 - hqc_no_x, expected_hqc_diff, atol=ATOL, rtol=RTOL,
            err_msg="HQC difference for X=Tx3 vs X=0"
        )
        npt.assert_allclose(
            sbic_x3 - sbic_no_x, expected_sbic_diff, atol=ATOL, rtol=RTOL,
            err_msg="SBIC difference for X=Tx3 vs X=0"
        )


# ---------------------------------------------------------------------------
# Test 8: X as a matrix (column counting via size(X,2))
# ---------------------------------------------------------------------------

class TestAichqcsbicXAsMatrix:
    """Verify K counts the columns of a matrix X via X.shape[1]."""

    def test_aichqcsbic_X_as_matrix(self, residuals: np.ndarray) -> None:
        """T×3 matrix X should contribute 3 to K."""
        T = len(residuals)
        rng = np.random.default_rng(77)
        X_mat = rng.standard_normal((T, 3))
        constant = 1
        p = np.array([1, 2])
        q = np.array([1])

        aic, hqc, sbic = aichqcsbic(residuals, constant, p, q, X_mat)

        # K = constant(1) + len(unique(p))(2) + len(unique(q))(1) + ncols(X)(3) = 7
        K = 7
        sigma_sq = float(np.dot(residuals, residuals) / T)
        log_s2 = np.log(sigma_sq)
        expected_aic = log_s2 + 2.0 * K / T
        expected_hqc = log_s2 + 2.0 * K * np.log(np.log(T)) / T
        expected_sbic = log_s2 + np.log(T) * K / T

        npt.assert_allclose(aic, expected_aic, atol=ATOL, rtol=RTOL,
                            err_msg="AIC with T×3 exogenous matrix")
        npt.assert_allclose(hqc, expected_hqc, atol=ATOL, rtol=RTOL,
                            err_msg="HQC with T×3 exogenous matrix")
        npt.assert_allclose(sbic, expected_sbic, atol=ATOL, rtol=RTOL,
                            err_msg="SBIC with T×3 exogenous matrix")

    def test_aichqcsbic_X_empty_array(self, residuals: np.ndarray) -> None:
        """A (T, 0) shaped array (empty columns) should contribute 0 to K."""
        T = len(residuals)
        X_empty = np.empty((T, 0))
        constant = 1
        p = np.array([1])
        q = np.array([])

        aic, hqc, sbic = aichqcsbic(residuals, constant, p, q, X_empty)

        # K = 1 + 1 + 0 + 0 = 2
        K = 2
        sigma_sq = float(np.dot(residuals, residuals) / T)
        log_s2 = np.log(sigma_sq)
        expected_aic = log_s2 + 2.0 * K / T
        expected_hqc = log_s2 + 2.0 * K * np.log(np.log(T)) / T
        expected_sbic = log_s2 + np.log(T) * K / T

        npt.assert_allclose(aic, expected_aic, atol=ATOL, rtol=RTOL,
                            err_msg="AIC with (T, 0) empty exogenous matrix")
        npt.assert_allclose(hqc, expected_hqc, atol=ATOL, rtol=RTOL,
                            err_msg="HQC with (T, 0) empty exogenous matrix")
        npt.assert_allclose(sbic, expected_sbic, atol=ATOL, rtol=RTOL,
                            err_msg="SBIC with (T, 0) empty exogenous matrix")


# ---------------------------------------------------------------------------
# Test 9: MATLAB/Octave fixture-based parity
# ---------------------------------------------------------------------------

class TestAichqcsbicParity:
    """Validate against MATLAB/Octave reference fixtures (tests/fixtures/timeseries/aichqcsbic.npy)."""

    @pytest.mark.parity
    def test_aichqcsbic_parity(self) -> None:
        """Load all scenarios from the fixture file and compare aic, hqc, sbic."""
        fixture_path = _fixture_dir() / "aichqcsbic.npy"
        if not fixture_path.exists():
            pytest.skip(f"Fixture file not found: {fixture_path}")

        # The fixture is a numpy object array wrapping a dict of scenario dicts
        data = np.load(fixture_path, allow_pickle=True).item()

        for scenario_key in sorted(data.keys()):
            scenario = data[scenario_key]
            errors = np.asarray(scenario["errors"], dtype=np.float64).ravel()
            constant = int(scenario["constant"])
            p = np.asarray(scenario["p"], dtype=np.int64)
            q = np.asarray(scenario["q"], dtype=np.int64)
            X_fixture = scenario["X"]

            # Determine the X argument:
            # Fixture stores X as (T, 0) for no exogenous or (T, k) for k regressors
            if isinstance(X_fixture, np.ndarray) and X_fixture.ndim == 2:
                if X_fixture.shape[1] == 0:
                    X_arg: int | np.ndarray | None = None
                else:
                    X_arg = X_fixture
            else:
                X_arg = None

            expected_aic = float(scenario["aic"])
            expected_hqc = float(scenario["hqc"])
            expected_sbic = float(scenario["sbic"])

            actual_aic, actual_hqc, actual_sbic = aichqcsbic(
                errors, constant, p, q, X_arg
            )

            desc = scenario.get("description", scenario_key)

            npt.assert_allclose(
                actual_aic, expected_aic, atol=ATOL, rtol=RTOL,
                err_msg=f"AIC parity failure for {desc}",
            )
            npt.assert_allclose(
                actual_hqc, expected_hqc, atol=ATOL, rtol=RTOL,
                err_msg=f"HQC parity failure for {desc}",
            )
            npt.assert_allclose(
                actual_sbic, expected_sbic, atol=ATOL, rtol=RTOL,
                err_msg=f"SBIC parity failure for {desc}",
            )

    @pytest.mark.parity
    @pytest.mark.parametrize(
        "scenario_key",
        ["scenario_1", "scenario_2", "scenario_3", "scenario_4"],
    )
    def test_aichqcsbic_parity_parametrized(self, scenario_key: str) -> None:
        """Parametrized per-scenario parity test for clearer failure output."""
        fixture_path = _fixture_dir() / "aichqcsbic.npy"
        if not fixture_path.exists():
            pytest.skip(f"Fixture file not found: {fixture_path}")

        data = np.load(fixture_path, allow_pickle=True).item()
        if scenario_key not in data:
            pytest.skip(f"Scenario '{scenario_key}' not found in fixture")

        scenario = data[scenario_key]
        errors = np.asarray(scenario["errors"], dtype=np.float64).ravel()
        constant = int(scenario["constant"])
        p = np.asarray(scenario["p"], dtype=np.int64)
        q = np.asarray(scenario["q"], dtype=np.int64)
        X_fixture = scenario["X"]

        if isinstance(X_fixture, np.ndarray) and X_fixture.ndim == 2 and X_fixture.shape[1] == 0:
            X_arg: int | np.ndarray | None = None
        elif isinstance(X_fixture, np.ndarray) and X_fixture.ndim == 2:
            X_arg = X_fixture
        else:
            X_arg = None

        actual_aic, actual_hqc, actual_sbic = aichqcsbic(
            errors, constant, p, q, X_arg
        )

        npt.assert_allclose(
            actual_aic, float(scenario["aic"]), atol=ATOL, rtol=RTOL,
            err_msg=f"AIC parity for {scenario_key}",
        )
        npt.assert_allclose(
            actual_hqc, float(scenario["hqc"]), atol=ATOL, rtol=RTOL,
            err_msg=f"HQC parity for {scenario_key}",
        )
        npt.assert_allclose(
            actual_sbic, float(scenario["sbic"]), atol=ATOL, rtol=RTOL,
            err_msg=f"SBIC parity for {scenario_key}",
        )


# ---------------------------------------------------------------------------
# Additional tests — duplicate lags, X=None default, input validation
# ---------------------------------------------------------------------------

class TestAichqcsbicDuplicateLagHandling:
    """Verify that duplicate lag values in p/q trigger a ValueError.

    Ref: aichqcsbic.m:80-82 — length(unique(p)) ~= length(p) → error
    """

    def test_duplicate_p_raises(self, residuals: np.ndarray) -> None:
        """Passing p with duplicate entries should raise ValueError."""
        with pytest.raises(ValueError, match="P must contain at most one of each lag"):
            aichqcsbic(residuals, 1, np.array([1, 1]), np.array([1]), 0)

    def test_duplicate_q_raises(self, residuals: np.ndarray) -> None:
        """Passing q with duplicate entries should raise ValueError."""
        with pytest.raises(ValueError, match="Q must contain at most one of each lag"):
            aichqcsbic(residuals, 1, np.array([1]), np.array([2, 2]), 0)


class TestAichqcsbicInputValidation:
    """Verify correct ValueError raising for invalid inputs."""

    def test_constant_invalid_raises(self, residuals: np.ndarray) -> None:
        """constant must be 0 or 1; other values should raise."""
        with pytest.raises(ValueError, match="CONSTANT must be 0 or 1"):
            aichqcsbic(residuals, 2, np.array([1]), np.array([1]), 0)

    def test_empty_errors_raises(self) -> None:
        """Empty errors vector should raise ValueError."""
        with pytest.raises(ValueError, match="ERRORS is empty"):
            aichqcsbic(np.array([]), 1, np.array([1]), np.array([1]), 0)

    def test_scalar_errors_raises(self) -> None:
        """A single-element errors vector should raise ValueError."""
        with pytest.raises(ValueError, match="ERRORS series must be a column vector"):
            aichqcsbic(np.array([1.0]), 1, np.array([]), np.array([]), 0)

    def test_negative_p_raises(self, residuals: np.ndarray) -> None:
        """Negative values in p should raise ValueError."""
        with pytest.raises(ValueError, match="P must contain non-negative integers only"):
            aichqcsbic(residuals, 1, np.array([-1]), np.array([1]), 0)

    def test_negative_q_raises(self, residuals: np.ndarray) -> None:
        """Negative values in q should raise ValueError."""
        with pytest.raises(ValueError, match="Q must contain non-negative integers only"):
            aichqcsbic(residuals, 1, np.array([1]), np.array([-1]), 0)


class TestAichqcsbicXDefault:
    """Verify that X=None (default) behaves identically to X=0."""

    def test_X_none_equals_X_zero(self, residuals: np.ndarray) -> None:
        """aichqcsbic(errors, c, p, q) should equal aichqcsbic(errors, c, p, q, 0)."""
        p = np.array([1])
        q = np.array([1])
        result_none = aichqcsbic(residuals, 1, p, q)
        result_zero = aichqcsbic(residuals, 1, p, q, 0)

        npt.assert_allclose(result_none[0], result_zero[0], atol=ATOL, rtol=RTOL,
                            err_msg="AIC: X=None vs X=0")
        npt.assert_allclose(result_none[1], result_zero[1], atol=ATOL, rtol=RTOL,
                            err_msg="HQC: X=None vs X=0")
        npt.assert_allclose(result_none[2], result_zero[2], atol=ATOL, rtol=RTOL,
                            err_msg="SBIC: X=None vs X=0")
