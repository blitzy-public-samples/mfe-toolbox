"""
Pytest tests for mfe_toolbox.timeseries.armaroots.

Tests the computation of roots and absolute roots of the AR characteristic
polynomial from ARMAX parameter vectors.

Source MATLAB reference: timeseries/armaroots.m (Kevin Sheppard, Rev 3, 2007)
Algorithm:
    formatted_arparameters = zeros(1, max(p))
    formatted_arparameters(p) = arparameters
    arroots = roots([1, -formatted_arparameters])
    absarroots = abs(arroots)

Per AAP Section 0.7.1: all numerical assertions use
    numpy.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-4)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.timeseries.armaroots import armaroots

# ---------------------------------------------------------------------------
# Numerical Parity Constants — per AAP rule: ±1e-6 absolute tolerance
# ---------------------------------------------------------------------------
ATOL: float = 1e-6
RTOL: float = 1e-4

# ---------------------------------------------------------------------------
# Fixture directory resolution
# ---------------------------------------------------------------------------
_FIXTURE_DIR: Path = Path(__file__).resolve().parent.parent / "fixtures" / "timeseries"


def _fixture_path(name: str) -> Path:
    """Return the path to a timeseries fixture file (no extension)."""
    env_dir = os.environ.get("MFE_FIXTURE_DIR")
    if env_dir:
        return Path(env_dir) / "timeseries" / name
    return _FIXTURE_DIR / name


def _has_fixture(name: str) -> bool:
    """Check if a given .npy fixture file exists."""
    return _fixture_path(name).with_suffix(".npy").exists()


# ---------------------------------------------------------------------------
# Helper: sort roots for stable comparison
# ---------------------------------------------------------------------------
def _sort_roots(roots: np.ndarray) -> np.ndarray:
    """Sort complex roots by (abs, real, imag) for deterministic comparison.

    MATLAB and NumPy may return polynomial roots in different orders.
    Sorting by absolute value first, then real and imaginary parts, ensures
    stable comparison across platforms.
    """
    if roots.size == 0:
        return roots
    # Sort by absolute value, then by real part, then by imaginary part
    sort_keys = np.lexsort((roots.imag, roots.real, np.abs(roots)))
    return roots[sort_keys]


# ---------------------------------------------------------------------------
# Test: Return type and shape contract
# ---------------------------------------------------------------------------
class TestArmarootsReturnContract:
    """Verify the function returns the correct types and tuple structure."""

    def test_armaroots_returns_two_outputs(self) -> None:
        """armaroots must return a 2-tuple (arroots, absarroots)."""
        params = np.array([0.0, 0.8])  # [constant=0, phi=0.8]
        result = armaroots(params, constant=1, p=np.array([1]), q=np.array([]))
        assert isinstance(result, tuple), "armaroots must return a tuple"
        assert len(result) == 2, "armaroots must return exactly 2 outputs"

    def test_armaroots_outputs_are_ndarrays(self) -> None:
        """Both arroots and absarroots must be numpy ndarrays."""
        params = np.array([0.0, 0.8])
        arroots_out, absarroots_out = armaroots(
            params, constant=1, p=np.array([1]), q=np.array([])
        )
        assert isinstance(arroots_out, np.ndarray), "arroots must be ndarray"
        assert isinstance(absarroots_out, np.ndarray), "absarroots must be ndarray"

    def test_armaroots_output_shapes_match(self) -> None:
        """arroots and absarroots must have the same shape."""
        params = np.array([1.0, 1.3, -0.35, 0.4, 0.3])
        arroots_out, absarroots_out = armaroots(
            params, constant=1, p=np.array([1, 2]), q=np.array([1, 2])
        )
        assert arroots_out.shape == absarroots_out.shape, (
            "arroots and absarroots must have same shape"
        )


# ---------------------------------------------------------------------------
# Test: Absolute roots are abs of roots
# ---------------------------------------------------------------------------
class TestArmarootsAbsRoots:
    """Verify absarroots == |arroots| identically."""

    def test_armaroots_absroots_are_abs_of_roots(self) -> None:
        """absarroots must be the complex modulus of arroots for real roots."""
        params = np.array([0.0, 0.8])
        arroots_out, absarroots_out = armaroots(
            params, constant=1, p=np.array([1]), q=np.array([])
        )
        npt.assert_allclose(absarroots_out, np.abs(arroots_out), atol=ATOL, rtol=RTOL)

    def test_armaroots_absroots_complex_case(self) -> None:
        """absarroots must be the complex modulus of arroots for complex roots."""
        # AR(2) with phi1=0.5, phi2=-0.5 produces complex conjugate roots
        params = np.array([1.0, 0.5, -0.5])
        arroots_out, absarroots_out = armaroots(
            params, constant=1, p=np.array([1, 2]), q=np.array([])
        )
        npt.assert_allclose(absarroots_out, np.abs(arroots_out), atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# Test: AR(1) single root
# ---------------------------------------------------------------------------
class TestArmarootsAR1:
    """Verify AR(1) root computation against analytical results."""

    def test_armaroots_ar1_single_root(self) -> None:
        """AR(1) with phi=0.8: companion polynomial z - 0.8, root = 0.8.

        Ref: armaroots.m computes roots([1, -phi]) which is the companion
        polynomial root z = phi, NOT the lag polynomial root z = 1/phi.
        Confirmed by fixture scenario_3: phi=0.7 yields root=0.7.
        """
        params = np.array([0.0, 0.8])  # [constant=0, phi=0.8]
        arroots_out, absarroots_out = armaroots(
            params, constant=1, p=np.array([1]), q=np.array([])
        )
        # Companion polynomial root = phi = 0.8
        npt.assert_allclose(absarroots_out, np.array([0.8]), atol=ATOL, rtol=RTOL)

    def test_armaroots_ar1_negative_phi(self) -> None:
        """AR(1) with phi=-0.5: companion root = -0.5, |root| = 0.5."""
        params = np.array([0.0, -0.5])
        arroots_out, absarroots_out = armaroots(
            params, constant=1, p=np.array([1]), q=np.array([])
        )
        # Companion polynomial root = phi = -0.5
        npt.assert_allclose(arroots_out.real, np.array([-0.5]), atol=ATOL, rtol=RTOL)
        npt.assert_allclose(absarroots_out, np.array([0.5]), atol=ATOL, rtol=RTOL)

    def test_armaroots_ar1_no_constant(self) -> None:
        """AR(1) with no constant: phi extracted from position 0 of parameters."""
        params = np.array([0.9])  # [phi=0.9], no constant
        arroots_out, absarroots_out = armaroots(
            params, constant=0, p=np.array([1]), q=np.array([])
        )
        # Companion polynomial root = phi = 0.9
        npt.assert_allclose(absarroots_out, np.array([0.9]), atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# Test: AR(2) roots
# ---------------------------------------------------------------------------
class TestArmarootsAR2:
    """Verify AR(2) root computation against analytical polynomial roots."""

    def test_armaroots_ar2_roots(self) -> None:
        """AR(2) with phi1=1.3, phi2=-0.35: roots of 1 - 1.3*z + 0.35*z^2.

        Ref: armaroots.m:25 — ARMA(2,2) example: phi = [1.3, -0.35]
        The characteristic polynomial is [1, -1.3, 0.35]
        """
        params = np.array([1.0, 1.3, -0.35, 0.4, 0.3])
        arroots_out, absarroots_out = armaroots(
            params, constant=1, p=np.array([1, 2]), q=np.array([1, 2])
        )
        # Verify against numpy.roots directly
        # Polynomial: [1, -1.3, 0.35] → roots
        expected_roots = np.sort(np.abs(np.roots([1, -1.3, 0.35])))
        actual_sorted = np.sort(absarroots_out)
        npt.assert_allclose(actual_sorted, expected_roots, atol=ATOL, rtol=RTOL)
        # These should be 2 roots for AR(2)
        assert arroots_out.shape[0] == 2, "AR(2) must yield exactly 2 roots"

    def test_armaroots_ar2_near_unit_root(self) -> None:
        """AR(2) near unit root: phi1=0.99, phi2=-0.01.

        Roots of [1, -0.99, 0.01] — one root very close to 1.
        """
        params = np.array([0.1, 0.99, -0.01])
        arroots_out, absarroots_out = armaroots(
            params, constant=1, p=np.array([1, 2]), q=np.array([])
        )
        expected_roots = np.sort(np.abs(np.roots([1, -0.99, 0.01])))
        actual_sorted = np.sort(absarroots_out)
        npt.assert_allclose(actual_sorted, expected_roots, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# Test: Stationarity conditions
# ---------------------------------------------------------------------------
class TestArmarootsStationarity:
    """Test stationarity analysis via AR root magnitudes."""

    def test_armaroots_stationary_process(self) -> None:
        """All |roots| < 1 indicates stationarity for AR(1) phi=0.5.

        Companion root = 0.5, |root| = 0.5 < 1, so process is stationary.
        Note: armaroots returns companion polynomial roots, so stationarity
        is |root| < 1 (roots inside unit circle), NOT |root| > 1.
        """
        params = np.array([0.0, 0.5])
        arroots_out, absarroots_out = armaroots(
            params, constant=1, p=np.array([1]), q=np.array([])
        )
        assert np.all(absarroots_out < 1.0), (
            "All companion roots must be < 1 for a stationary process"
        )

    def test_armaroots_unit_root_process(self) -> None:
        """phi=1.0 gives root exactly on unit circle: companion root = 1.0."""
        params = np.array([0.0, 1.0])
        arroots_out, absarroots_out = armaroots(
            params, constant=1, p=np.array([1]), q=np.array([])
        )
        npt.assert_allclose(absarroots_out, np.array([1.0]), atol=ATOL, rtol=RTOL)

    def test_armaroots_nonstationary_process(self) -> None:
        """phi=1.2 gives |root| > 1: companion root = 1.2.

        Explosive/non-stationary process — companion root outside unit circle.
        """
        params = np.array([0.0, 1.2])
        arroots_out, absarroots_out = armaroots(
            params, constant=1, p=np.array([1]), q=np.array([])
        )
        npt.assert_allclose(absarroots_out, np.array([1.2]), atol=ATOL, rtol=RTOL)
        assert np.all(absarroots_out > 1.0), (
            "Explosive process companion root should be > 1"
        )

    def test_armaroots_ar2_stationary(self) -> None:
        """AR(2) with phi1=0.5, phi2=0.3: verify companion roots match numpy.roots."""
        params = np.array([0.0, 0.5, 0.3])
        arroots_out, absarroots_out = armaroots(
            params, constant=1, p=np.array([1, 2]), q=np.array([])
        )
        # Companion polynomial: [1, -0.5, -0.3]
        expected_roots = np.roots([1, -0.5, -0.3])
        expected_abs = np.sort(np.abs(expected_roots))
        actual_sorted = np.sort(absarroots_out)
        npt.assert_allclose(actual_sorted, expected_abs, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# Test: Empty AR (no AR terms)
# ---------------------------------------------------------------------------
class TestArmarootsEmptyAR:
    """Verify behavior when no AR component is present."""

    def test_armaroots_empty_ar(self) -> None:
        """p=[] → no AR roots, returns empty arrays."""
        # Pure MA(1): constant + 1 MA parameter, no AR parameters
        params = np.array([1.0, 0.5])
        arroots_out, absarroots_out = armaroots(
            params, constant=1, p=np.array([]), q=np.array([1])
        )
        assert arroots_out.size == 0, "No AR roots when p is empty"
        assert absarroots_out.size == 0, "No absolute AR roots when p is empty"

    def test_armaroots_p_zero(self) -> None:
        """p=[0] should be treated the same as p=[] (no AR component)."""
        params = np.array([1.0, 0.5])
        arroots_out, absarroots_out = armaroots(
            params, constant=1, p=np.array([0]), q=np.array([1])
        )
        assert arroots_out.size == 0, "p=[0] should yield no AR roots"
        assert absarroots_out.size == 0, "p=[0] should yield no absolute AR roots"


# ---------------------------------------------------------------------------
# Test: Constant and exogenous regressors
# ---------------------------------------------------------------------------
class TestArmarootsWithConstantAndExog:
    """Verify correct AR parameter extraction when constant and exogenous
    regressors are present."""

    def test_armaroots_with_constant_and_exog(self) -> None:
        """AR roots extracted correctly from the middle of the parameter vector
        when both constant and exogenous regressors are present.

        Parameter vector layout: [constant, AR1, AR2, exog1, exog2, MA1]
        constant=1, p=[1,2], q=[1], X has 2 columns
        """
        # Layout: [const, phi1, phi2, beta1, beta2, theta1]
        params = np.array([0.5, 0.6, -0.3, 1.0, 2.0, 0.4])
        # Create a dummy X with 2 columns (only shape matters for validation)
        X = np.ones((100, 2))
        arroots_out, absarroots_out = armaroots(
            params, constant=1, p=np.array([1, 2]), q=np.array([1]), x=X
        )
        # AR parameters extracted: phi1=0.6, phi2=-0.3
        # Polynomial: [1, -0.6, 0.3]
        expected_roots = np.sort(np.abs(np.roots([1, -0.6, 0.3])))
        actual_sorted = np.sort(absarroots_out)
        npt.assert_allclose(actual_sorted, expected_roots, atol=ATOL, rtol=RTOL)

    def test_armaroots_no_constant_with_exog(self) -> None:
        """AR roots with no constant but with exogenous regressors.

        Parameter vector layout: [AR1, exog1, MA1]
        constant=0, p=[1], q=[1], x has 1 column
        """
        params = np.array([0.7, 1.5, 0.3])
        X_data = np.ones((100, 1))
        arroots_out, absarroots_out = armaroots(
            params, constant=0, p=np.array([1]), q=np.array([1]), x=X_data
        )
        # AR parameter extracted: phi1=0.7 (at position 0)
        # Companion root = phi1 = 0.7
        npt.assert_allclose(absarroots_out, np.array([0.7]), atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# Test: Complex conjugate roots
# ---------------------------------------------------------------------------
class TestArmarootsComplexRoots:
    """Verify complex root computation and magnitude for AR(2) models."""

    def test_armaroots_complex_roots(self) -> None:
        """AR(2) with complex conjugate roots: verify both root values and magnitudes.

        phi1=0.5, phi2=-0.5 yields complex conjugate roots.
        Polynomial: [1, -0.5, 0.5]
        Discriminant: 0.25 - 2.0 = -1.75 < 0 → complex roots
        """
        params = np.array([0.0, 0.5, -0.5])
        arroots_out, absarroots_out = armaroots(
            params, constant=1, p=np.array([1, 2]), q=np.array([])
        )
        # Compute expected roots analytically
        expected_roots = np.roots([1, -0.5, 0.5])
        expected_abs = np.sort(np.abs(expected_roots))
        actual_sorted = np.sort(absarroots_out)
        # Verify magnitudes match
        npt.assert_allclose(actual_sorted, expected_abs, atol=ATOL, rtol=RTOL)
        # Verify roots are actually complex (non-zero imaginary parts)
        assert np.any(np.abs(arroots_out.imag) > 1e-10), (
            "AR(2) with phi1=0.5, phi2=-0.5 should have complex roots"
        )
        # For complex conjugate pair, magnitudes should be equal
        npt.assert_allclose(absarroots_out[0], absarroots_out[1], atol=ATOL, rtol=RTOL)

    def test_armaroots_complex_roots_magnitude(self) -> None:
        """Verify complex modulus computation for known conjugate pair.

        phi1=1.0, phi2=-0.5 → polynomial [1, -1.0, 0.5]
        roots = (1 ± sqrt(1 - 2)) / 1 → complex
        """
        params = np.array([0.0, 1.0, -0.5])
        arroots_out, absarroots_out = armaroots(
            params, constant=1, p=np.array([1, 2]), q=np.array([])
        )
        expected_roots = np.roots([1, -1.0, 0.5])
        expected_abs = np.sort(np.abs(expected_roots))
        actual_sorted = np.sort(absarroots_out)
        npt.assert_allclose(actual_sorted, expected_abs, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# Test: Irregular AR lag patterns
# ---------------------------------------------------------------------------
class TestArmarootsIrregularLags:
    """Verify correct handling of irregular (non-consecutive) AR lag orders."""

    def test_armaroots_irregular_ar_lags_1_3(self) -> None:
        """Irregular AR with lags [1, 3]: zero coefficient at lag 2.

        Ref: armaroots.m:28-29 — Example: phi = [1.3, -0.35], p = [1, 3]
        AR parameters scattered to [1.3, 0, -0.35]
        Polynomial: [1, -1.3, 0, 0.35]
        """
        params = np.array([1.0, 1.3, -0.35])
        arroots_out, absarroots_out = armaroots(
            params, constant=1, p=np.array([1, 3]), q=np.array([])
        )
        # Polynomial: [1, -1.3, 0, 0.35] — 3 roots
        expected_roots = np.roots([1, -1.3, 0, 0.35])
        expected_abs = np.sort(np.abs(expected_roots))
        actual_sorted = np.sort(absarroots_out)
        npt.assert_allclose(actual_sorted, expected_abs, atol=ATOL, rtol=RTOL)
        assert arroots_out.shape[0] == 3, "Irregular AR(1,3) must yield max(p)=3 roots"

    def test_armaroots_irregular_ar_lags_2_4(self) -> None:
        """Irregular AR with lags [2, 4]: zero coefficients at lags 1 and 3.

        AR parameters scattered to [0, phi1, 0, phi2]
        """
        params = np.array([0.5, 0.3, -0.2])  # [const, phi_2, phi_4]
        arroots_out, absarroots_out = armaroots(
            params, constant=1, p=np.array([2, 4]), q=np.array([])
        )
        # Polynomial: [1, 0, -0.3, 0, 0.2]
        expected_roots = np.roots([1, 0, -0.3, 0, 0.2])
        expected_abs = np.sort(np.abs(expected_roots))
        actual_sorted = np.sort(absarroots_out)
        npt.assert_allclose(actual_sorted, expected_abs, atol=ATOL, rtol=RTOL)
        assert arroots_out.shape[0] == 4, "Irregular AR(2,4) must yield max(p)=4 roots"


# ---------------------------------------------------------------------------
# Test: Input validation and error handling
# ---------------------------------------------------------------------------
class TestArmarootsInputValidation:
    """Verify that invalid inputs raise appropriate errors.

    Per AAP Section 0.7.1: If a MATLAB function errors on invalid input,
    the Python equivalent must raise an equivalent exception.
    """

    def test_armaroots_invalid_constant(self) -> None:
        """CONSTANT not in {0, 1} must raise ValueError."""
        params = np.array([0.0, 0.8])
        with pytest.raises(ValueError, match="CONSTANT must be 0 or 1"):
            armaroots(params, constant=2, p=np.array([1]), q=np.array([]))

    def test_armaroots_negative_p(self) -> None:
        """Negative lag in p must raise ValueError."""
        params = np.array([0.0, 0.5])
        with pytest.raises(ValueError, match="non-negative integers"):
            armaroots(params, constant=1, p=np.array([-1]), q=np.array([]))

    def test_armaroots_non_integer_p(self) -> None:
        """Non-integer lag in p must raise ValueError."""
        params = np.array([0.0, 0.5])
        with pytest.raises(ValueError, match="non-negative integers"):
            armaroots(params, constant=1, p=np.array([1.5]), q=np.array([]))

    def test_armaroots_duplicate_p(self) -> None:
        """Duplicate lag in p must raise ValueError."""
        params = np.array([0.0, 0.5, 0.3])
        with pytest.raises(ValueError, match="at most one of each lag"):
            armaroots(params, constant=1, p=np.array([1, 1]), q=np.array([]))

    def test_armaroots_negative_q(self) -> None:
        """Negative lag in q must raise ValueError."""
        params = np.array([0.0, 0.5])
        with pytest.raises(ValueError, match="non-negative integers"):
            armaroots(params, constant=1, p=np.array([]), q=np.array([-1]))

    def test_armaroots_incompatible_parameters_length(self) -> None:
        """Parameter vector with wrong length must raise ValueError."""
        params = np.array([0.0, 0.5, 0.3, 0.7])  # Too many params
        with pytest.raises(ValueError, match="length compatible"):
            armaroots(params, constant=1, p=np.array([1]), q=np.array([]))


# ---------------------------------------------------------------------------
# Test: MATLAB fixture parity
# ---------------------------------------------------------------------------
class TestArmarootsFixtureParity:
    """MATLAB numerical parity tests using generated fixture data.

    Fixtures generated by scripts/generate_fixtures.m and converted via
    scripts/convert_fixtures.py. If fixtures are not available, tests are
    skipped gracefully.
    """

    @pytest.mark.skipif(
        not _has_fixture("armaroots"),
        reason="Fixture file armaroots.npy not found",
    )
    def test_armaroots_parity_scenario_1(self, timeseries_fixture_dir: Path) -> None:
        """Parity: ARMA(2,2) — scenario 1 from MATLAB fixtures."""
        from tests.conftest import load_fixture_npy

        data = load_fixture_npy(timeseries_fixture_dir, "armaroots")
        scenario = data.item()["scenario_1"]

        params = scenario["parameters"]
        constant = int(scenario["constant"])
        p = scenario["p"]
        q = scenario["q"]
        expected_arroots = scenario["arroots"]
        expected_absarroots = scenario["absarroots"]

        arroots_out, absarroots_out = armaroots(params, constant=constant, p=p, q=q)

        # Sort for stable comparison (root ordering not guaranteed)
        npt.assert_allclose(
            np.sort(np.abs(arroots_out)),
            np.sort(np.abs(expected_arroots)),
            atol=ATOL,
            rtol=RTOL,
            err_msg="ARMA(2,2) arroots absolute values mismatch",
        )
        npt.assert_allclose(
            np.sort(absarroots_out),
            np.sort(expected_absarroots),
            atol=ATOL,
            rtol=RTOL,
            err_msg="ARMA(2,2) absarroots mismatch",
        )

    @pytest.mark.skipif(
        not _has_fixture("armaroots"),
        reason="Fixture file armaroots.npy not found",
    )
    def test_armaroots_parity_scenario_2(self, timeseries_fixture_dir: Path) -> None:
        """Parity: Irregular AR(3) with lags [1,3] — scenario 2."""
        from tests.conftest import load_fixture_npy

        data = load_fixture_npy(timeseries_fixture_dir, "armaroots")
        scenario = data.item()["scenario_2"]

        params = scenario["parameters"]
        constant = int(scenario["constant"])
        p = scenario["p"]
        q = scenario["q"]
        expected_arroots = scenario["arroots"]
        expected_absarroots = scenario["absarroots"]

        arroots_out, absarroots_out = armaroots(params, constant=constant, p=p, q=q)

        npt.assert_allclose(
            np.sort(np.abs(arroots_out)),
            np.sort(np.abs(expected_arroots)),
            atol=ATOL,
            rtol=RTOL,
            err_msg="Irregular AR(3) arroots absolute values mismatch",
        )
        npt.assert_allclose(
            np.sort(absarroots_out),
            np.sort(expected_absarroots),
            atol=ATOL,
            rtol=RTOL,
            err_msg="Irregular AR(3) absarroots mismatch",
        )

    @pytest.mark.skipif(
        not _has_fixture("armaroots"),
        reason="Fixture file armaroots.npy not found",
    )
    def test_armaroots_parity_scenario_3(self, timeseries_fixture_dir: Path) -> None:
        """Parity: AR(1) stable — scenario 3."""
        from tests.conftest import load_fixture_npy

        data = load_fixture_npy(timeseries_fixture_dir, "armaroots")
        scenario = data.item()["scenario_3"]

        params = scenario["parameters"]
        constant = int(scenario["constant"])
        p = scenario["p"]
        q = scenario["q"]
        expected_arroots = scenario["arroots"]
        expected_absarroots = scenario["absarroots"]

        arroots_out, absarroots_out = armaroots(params, constant=constant, p=p, q=q)

        npt.assert_allclose(
            np.sort(absarroots_out),
            np.sort(expected_absarroots),
            atol=ATOL,
            rtol=RTOL,
            err_msg="AR(1) stable absarroots mismatch",
        )

    @pytest.mark.skipif(
        not _has_fixture("armaroots"),
        reason="Fixture file armaroots.npy not found",
    )
    def test_armaroots_parity_scenario_4(self, timeseries_fixture_dir: Path) -> None:
        """Parity: AR(2) near unit root — scenario 4."""
        from tests.conftest import load_fixture_npy

        data = load_fixture_npy(timeseries_fixture_dir, "armaroots")
        scenario = data.item()["scenario_4"]

        params = scenario["parameters"]
        constant = int(scenario["constant"])
        p = scenario["p"]
        q = scenario["q"]
        expected_arroots = scenario["arroots"]
        expected_absarroots = scenario["absarroots"]

        arroots_out, absarroots_out = armaroots(params, constant=constant, p=p, q=q)

        npt.assert_allclose(
            np.sort(np.abs(arroots_out)),
            np.sort(np.abs(expected_arroots)),
            atol=ATOL,
            rtol=RTOL,
            err_msg="AR(2) near unit root arroots absolute values mismatch",
        )
        npt.assert_allclose(
            np.sort(absarroots_out),
            np.sort(expected_absarroots),
            atol=ATOL,
            rtol=RTOL,
            err_msg="AR(2) near unit root absarroots mismatch",
        )

    @pytest.mark.skipif(
        not _has_fixture("armaroots"),
        reason="Fixture file armaroots.npy not found",
    )
    def test_armaroots_parity_all_scenarios(self, timeseries_fixture_dir: Path) -> None:
        """Parity: Iterate over ALL fixture scenarios and verify each."""
        from tests.conftest import load_fixture_npy

        data = load_fixture_npy(timeseries_fixture_dir, "armaroots")
        scenarios = data.item()

        for key, scenario in scenarios.items():
            params = scenario["parameters"]
            constant = int(scenario["constant"])
            p = scenario["p"]
            q = scenario["q"]
            expected_absarroots = scenario["absarroots"]

            arroots_out, absarroots_out = armaroots(
                params, constant=constant, p=p, q=q
            )

            npt.assert_allclose(
                np.sort(absarroots_out),
                np.sort(expected_absarroots),
                atol=ATOL,
                rtol=RTOL,
                err_msg=f"Parity failed for {key}: {scenario.get('description', '')}",
            )
