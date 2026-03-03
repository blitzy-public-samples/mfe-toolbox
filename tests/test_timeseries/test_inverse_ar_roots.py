"""
Pytest tests for mfe_toolbox.timeseries.inverse_ar_roots.

Tests the computation of inverted roots of the AR(P) characteristic equation
and the associated stationarity check.

Source MATLAB reference: timeseries/inverse_ar_roots.m (Kevin Sheppard)
Algorithm: rho = roots([-fliplr(phi) 1]); stationary = min(abs(rho)) > 1

Per AAP Section 0.7.1: all numerical assertions use
    numpy.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-4)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.timeseries.inverse_ar_roots import inverse_ar_roots

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
    """Check if a given .npy fixture exists."""
    return _fixture_path(name).with_suffix(".npy").exists()


# ---------------------------------------------------------------------------
# Helper: sort roots for stable comparison
# ---------------------------------------------------------------------------
def _sort_roots(roots: np.ndarray) -> np.ndarray:
    """Sort complex roots by (real part, imag part) for deterministic comparison.

    MATLAB and NumPy may return polynomial roots in different orders.
    Sorting by real part first, then imaginary part, ensures stable comparison
    across platforms.
    """
    # Ref: inverse_ar_roots.m — root ordering is not guaranteed by MATLAB roots()
    idx = np.lexsort((roots.imag, roots.real))
    return roots[idx]


# ===================================================================
# Test 1: Return type and structure
# ===================================================================
class TestInverseArRootsReturnType:
    """Verify return type contract: tuple of (np.ndarray, bool)."""

    def test_inverse_ar_roots_returns_two(self) -> None:
        """inverse_ar_roots must return a tuple of exactly 2 elements."""
        phi = np.array([0.5])
        result = inverse_ar_roots(phi)
        assert isinstance(result, tuple), "Return value must be a tuple"
        assert len(result) == 2, "Tuple must have exactly 2 elements"

    def test_inverse_ar_roots_rho_is_ndarray(self) -> None:
        """First element (rho) must be a numpy ndarray."""
        rho, _ = inverse_ar_roots(np.array([0.5]))
        assert isinstance(rho, np.ndarray), "rho must be a numpy.ndarray"

    def test_inverse_ar_roots_stationary_is_bool(self) -> None:
        """Second element (stationary) must be a Python bool."""
        _, stationary = inverse_ar_roots(np.array([0.5]))
        assert isinstance(stationary, bool), "stationary must be a bool"


# ===================================================================
# Test 2: Root count matches AR order
# ===================================================================
class TestInverseArRootsRhoLength:
    """Verify that the number of roots equals the AR order P."""

    @pytest.mark.parametrize(
        "phi, expected_len",
        [
            (np.array([0.5]), 1),
            (np.array([0.5, -0.2]), 2),
            (np.array([0.5, -0.2, 0.1]), 3),
            (np.array([0.3, -0.1, 0.05, -0.02]), 4),
        ],
        ids=["AR(1)", "AR(2)", "AR(3)", "AR(4)"],
    )
    def test_inverse_ar_roots_rho_length(
        self, phi: np.ndarray, expected_len: int
    ) -> None:
        """len(rho) must equal the number of AR parameters (order P)."""
        rho, _ = inverse_ar_roots(phi)
        assert len(rho) == expected_len, (
            f"Expected {expected_len} roots for AR({expected_len}), got {len(rho)}"
        )


# ===================================================================
# Test 3: AR(1) stationary case
# ===================================================================
class TestInverseArRootsAR1Stationary:
    """AR(1) with |phi| < 1 must be stationary; root = 1/phi."""

    def test_inverse_ar_roots_ar1_stationary(self) -> None:
        """phi = 0.5 → root = 1/0.5 = 2.0, |root| = 2.0 > 1 → stationary."""
        phi = np.array([0.5])
        rho, stationary = inverse_ar_roots(phi)

        assert stationary is True, "AR(1) with phi=0.5 must be stationary"
        # Ref: inverse_ar_roots.m:34 — root of 1 - 0.5*z = 0 is z = 2.0
        npt.assert_allclose(
            np.abs(rho), np.array([2.0]), atol=ATOL, rtol=RTOL,
            err_msg="AR(1) phi=0.5: |root| should be 2.0"
        )

    def test_inverse_ar_roots_ar1_negative_phi_stationary(self) -> None:
        """phi = -0.5 → root = 1/(-0.5) = -2.0, |root| = 2.0 > 1 → stationary."""
        phi = np.array([-0.5])
        rho, stationary = inverse_ar_roots(phi)

        assert stationary is True, "AR(1) with phi=-0.5 must be stationary"
        npt.assert_allclose(
            np.abs(rho), np.array([2.0]), atol=ATOL, rtol=RTOL,
            err_msg="AR(1) phi=-0.5: |root| should be 2.0"
        )


# ===================================================================
# Test 4: AR(1) unit root case
# ===================================================================
class TestInverseArRootsAR1UnitRoot:
    """AR(1) with phi=1.0 has root exactly on the unit circle → not stationary."""

    def test_inverse_ar_roots_ar1_unit_root(self) -> None:
        """phi = 1.0 → root = 1.0, |root| = 1.0, NOT > 1 → stationary = False."""
        phi = np.array([1.0])
        rho, stationary = inverse_ar_roots(phi)

        assert stationary is False, (
            "AR(1) with phi=1.0 must NOT be stationary (root on unit circle)"
        )
        # Ref: inverse_ar_roots.m:35 — stationary = min(abs(rho)) > 1
        # min(|1.0|) = 1.0, which is NOT > 1
        npt.assert_allclose(
            np.abs(rho), np.array([1.0]), atol=ATOL, rtol=RTOL,
            err_msg="AR(1) phi=1.0: |root| should be exactly 1.0"
        )


# ===================================================================
# Test 5: AR(1) explosive case
# ===================================================================
class TestInverseArRootsAR1Explosive:
    """AR(1) with |phi| > 1 is explosive → root inside unit circle → not stationary."""

    def test_inverse_ar_roots_ar1_explosive(self) -> None:
        """phi = 1.5 → root = 1/1.5 ≈ 0.667, |root| < 1 → stationary = False."""
        phi = np.array([1.5])
        rho, stationary = inverse_ar_roots(phi)

        assert stationary is False, (
            "AR(1) with phi=1.5 must NOT be stationary (root inside unit circle)"
        )
        expected_abs = np.array([1.0 / 1.5])
        npt.assert_allclose(
            np.abs(rho), expected_abs, atol=ATOL, rtol=RTOL,
            err_msg="AR(1) phi=1.5: |root| should be 1/1.5 ≈ 0.6667"
        )


# ===================================================================
# Test 6: AR(2) stationary case
# ===================================================================
class TestInverseArRootsAR2Stationary:
    """AR(2) with parameters yielding all roots outside unit circle."""

    def test_inverse_ar_roots_ar2_stationary(self) -> None:
        """phi = [0.5, -0.2]: both roots have |root| > 1 → stationary = True."""
        phi = np.array([0.5, -0.2])
        rho, stationary = inverse_ar_roots(phi)

        assert stationary is True, "AR(2) with phi=[0.5,-0.2] must be stationary"
        assert np.min(np.abs(rho)) > 1.0, (
            "All roots must lie strictly outside the unit circle"
        )
        assert len(rho) == 2, "AR(2) must return exactly 2 roots"


# ===================================================================
# Test 7: AR(2) complex conjugate roots
# ===================================================================
class TestInverseArRootsAR2ComplexRoots:
    """AR(2) producing complex conjugate root pairs."""

    def test_inverse_ar_roots_ar2_complex_roots(self) -> None:
        """phi = [0.6, -0.3]: characteristic polynomial has complex conjugate roots."""
        phi = np.array([0.6, -0.3])
        rho, stationary = inverse_ar_roots(phi)

        # Roots should be complex conjugate pairs
        assert np.any(np.iscomplex(rho)), (
            "AR(2) phi=[0.6,-0.3] should produce complex roots"
        )

        # Complex conjugate pairs have the same absolute value
        abs_rho = np.sort(np.abs(rho))
        npt.assert_allclose(
            abs_rho[0], abs_rho[1], atol=ATOL, rtol=RTOL,
            err_msg="Complex conjugate roots must have identical modulus"
        )

    def test_inverse_ar_roots_ar2_complex_conjugate_symmetry(self) -> None:
        """Complex conjugate roots have equal real parts and opposite imaginary parts."""
        phi = np.array([0.6, -0.3])
        rho, _ = inverse_ar_roots(phi)

        sorted_rho = _sort_roots(rho)
        # Real parts should be equal
        npt.assert_allclose(
            sorted_rho[0].real, sorted_rho[1].real, atol=ATOL, rtol=RTOL,
            err_msg="Complex conjugate roots must have equal real parts"
        )
        # Imaginary parts should be negatives of each other
        npt.assert_allclose(
            sorted_rho[0].imag, -sorted_rho[1].imag, atol=ATOL, rtol=RTOL,
            err_msg="Complex conjugate roots must have opposite imaginary parts"
        )


# ===================================================================
# Test 8: AR(2) unit root / non-stationary
# ===================================================================
class TestInverseArRootsAR2UnitRoot:
    """AR(2) with a root on or inside the unit circle → non-stationary."""

    def test_inverse_ar_roots_ar2_unit_root(self) -> None:
        """phi = [0.5, 0.5]: produces roots at -2.0 and 1.0.

        Root at 1.0 is exactly on the unit circle, so min(|rho|) = 1.0,
        which is NOT > 1 → stationary = False.

        Characteristic polynomial: 1 - 0.5*z - 0.5*z^2 = 0
        Coefficients (descending): [-0.5, -0.5, 1]
        Roots: z = -2 and z = 1
        """
        phi = np.array([0.5, 0.5])
        rho, stationary = inverse_ar_roots(phi)

        assert stationary is False, (
            "AR(2) with phi=[0.5,0.5] must NOT be stationary (root at unit circle)"
        )
        # Verify root moduli
        abs_rho_sorted = np.sort(np.abs(rho))
        # Smallest root modulus should be 1.0 (on unit circle)
        npt.assert_allclose(
            abs_rho_sorted[0], 1.0, atol=ATOL, rtol=RTOL,
            err_msg="One root should be on the unit circle (|root|=1)"
        )


# ===================================================================
# Test 9: Column vector input handling
# ===================================================================
class TestInverseArRootsColumnInput:
    """Verify that 2-D column vector inputs are handled correctly."""

    def test_inverse_ar_roots_column_input(self) -> None:
        """phi as 2-D column vector [[0.5], [-0.2]] should match 1-D [0.5, -0.2].

        Ref: inverse_ar_roots.m:29 — MATLAB transposes column to row vector.
        Python implementation squeezes 2-D column to 1-D.
        """
        phi_1d = np.array([0.5, -0.2])
        phi_2d = np.array([[0.5], [-0.2]])

        rho_1d, stat_1d = inverse_ar_roots(phi_1d)
        rho_2d, stat_2d = inverse_ar_roots(phi_2d)

        # Stationarity should match
        assert stat_1d == stat_2d, (
            "Column vector and 1-D vector must produce same stationarity"
        )

        # Root values should match (sort for stable comparison)
        rho_1d_sorted = _sort_roots(rho_1d)
        rho_2d_sorted = _sort_roots(rho_2d)
        npt.assert_allclose(
            rho_2d_sorted, rho_1d_sorted, atol=ATOL, rtol=RTOL,
            err_msg="Column vector input must produce same roots as 1-D input"
        )

    def test_inverse_ar_roots_row_vector_input(self) -> None:
        """phi as 2-D row vector [[0.5, -0.2]] should also be handled correctly.

        Ref: inverse_ar_roots.m:29-31 — MATLAB allows row vectors by checking
        min(size(phi))==1. Python squeezes any (1,P) or (P,1) shape.
        """
        phi_1d = np.array([0.5, -0.2])
        phi_row = np.array([[0.5, -0.2]])

        rho_1d, stat_1d = inverse_ar_roots(phi_1d)
        rho_row, stat_row = inverse_ar_roots(phi_row)

        assert stat_1d == stat_row
        npt.assert_allclose(
            _sort_roots(rho_row), _sort_roots(rho_1d), atol=ATOL, rtol=RTOL,
            err_msg="Row vector input must produce same roots as 1-D input"
        )


# ===================================================================
# Test 10: Zero coefficient edge case
# ===================================================================
class TestInverseArRootsZeroCoefficient:
    """AR(2) with zero coefficient at lag 1 — AR effect only at lag 2."""

    def test_inverse_ar_roots_zero_coefficient(self) -> None:
        """phi = [0.0, 0.3]: zero at lag 1, positive at lag 2.

        Characteristic polynomial: 1 - 0.0*z - 0.3*z^2 = 0
        Equivalent to: 1 - 0.3*z^2 = 0 → z^2 = 1/0.3 → z = ±√(1/0.3) ≈ ±1.826
        Both roots have |root| > 1 → stationary.
        """
        phi = np.array([0.0, 0.3])
        rho, stationary = inverse_ar_roots(phi)

        assert len(rho) == 2, "AR(2) must return 2 roots even if one coefficient is 0"
        assert stationary is True, (
            "AR(2) phi=[0.0,0.3] must be stationary (both roots ≈ ±1.826)"
        )

        # Roots should be real and symmetric: +sqrt(1/0.3) and -sqrt(1/0.3)
        expected_abs = np.sqrt(1.0 / 0.3)
        npt.assert_allclose(
            np.sort(np.abs(rho)),
            np.array([expected_abs, expected_abs]),
            atol=ATOL, rtol=RTOL,
            err_msg="Both roots should have |root| = sqrt(1/0.3) ≈ 1.826"
        )


# ===================================================================
# Test 11: MATLAB parity tests using fixture data
# ===================================================================
class TestInverseArRootsParity:
    """Compare Python output against MATLAB-generated reference fixtures.

    Fixture file: tests/fixtures/timeseries/inverse_ar_roots.npy
    Contains 5 scenarios with phi, rho (complex), and stationary flag
    generated by MATLAB/Octave.
    """

    @pytest.mark.parity
    @pytest.mark.skipif(
        not _has_fixture("inverse_ar_roots"),
        reason="Fixture file tests/fixtures/timeseries/inverse_ar_roots.npy not found",
    )
    def test_parity_scenario_1_stationary_ar1(self) -> None:
        """Fixture scenario 1: Stationary AR(1) phi=[0.7], |rho|>1."""
        fixture = np.load(
            _fixture_path("inverse_ar_roots").with_suffix(".npy"),
            allow_pickle=True,
        ).item()

        phi = fixture["scenario_1_phi"]
        expected_rho = fixture["scenario_1_rho"]
        expected_stationary = fixture["scenario_1_stationary"]

        rho, stationary = inverse_ar_roots(phi)

        assert stationary == expected_stationary, (
            f"Stationarity mismatch: expected {expected_stationary}, got {stationary}"
        )
        npt.assert_allclose(
            _sort_roots(rho), _sort_roots(expected_rho), atol=ATOL, rtol=RTOL,
            err_msg="Scenario 1 (stationary AR(1) phi=0.7): rho mismatch"
        )

    @pytest.mark.parity
    @pytest.mark.skipif(
        not _has_fixture("inverse_ar_roots"),
        reason="Fixture file tests/fixtures/timeseries/inverse_ar_roots.npy not found",
    )
    def test_parity_scenario_2_non_stationary_ar1(self) -> None:
        """Fixture scenario 2: Non-stationary AR(1) phi=[1.0], unit root."""
        fixture = np.load(
            _fixture_path("inverse_ar_roots").with_suffix(".npy"),
            allow_pickle=True,
        ).item()

        phi = fixture["scenario_2_phi"]
        expected_rho = fixture["scenario_2_rho"]
        expected_stationary = fixture["scenario_2_stationary"]

        rho, stationary = inverse_ar_roots(phi)

        assert stationary == expected_stationary, (
            f"Stationarity mismatch: expected {expected_stationary}, got {stationary}"
        )
        npt.assert_allclose(
            _sort_roots(rho), _sort_roots(expected_rho), atol=ATOL, rtol=RTOL,
            err_msg="Scenario 2 (non-stationary AR(1) phi=1.0): rho mismatch"
        )

    @pytest.mark.parity
    @pytest.mark.skipif(
        not _has_fixture("inverse_ar_roots"),
        reason="Fixture file tests/fixtures/timeseries/inverse_ar_roots.npy not found",
    )
    def test_parity_scenario_3_stationary_ar2(self) -> None:
        """Fixture scenario 3: Stationary AR(2) phi=[0.5, -0.3], complex roots."""
        fixture = np.load(
            _fixture_path("inverse_ar_roots").with_suffix(".npy"),
            allow_pickle=True,
        ).item()

        phi = fixture["scenario_3_phi"]
        expected_rho = fixture["scenario_3_rho"]
        expected_stationary = fixture["scenario_3_stationary"]

        rho, stationary = inverse_ar_roots(phi)

        assert stationary == expected_stationary, (
            f"Stationarity mismatch: expected {expected_stationary}, got {stationary}"
        )
        # Compare sorted roots for deterministic order
        npt.assert_allclose(
            _sort_roots(rho), _sort_roots(expected_rho), atol=ATOL, rtol=RTOL,
            err_msg="Scenario 3 (stationary AR(2) phi=[0.5,-0.3]): rho mismatch"
        )

    @pytest.mark.parity
    @pytest.mark.skipif(
        not _has_fixture("inverse_ar_roots"),
        reason="Fixture file tests/fixtures/timeseries/inverse_ar_roots.npy not found",
    )
    def test_parity_scenario_4_near_unit_root_ar2(self) -> None:
        """Fixture scenario 4: Near unit root AR(2) phi=[1.5, -0.55]."""
        fixture = np.load(
            _fixture_path("inverse_ar_roots").with_suffix(".npy"),
            allow_pickle=True,
        ).item()

        phi = fixture["scenario_4_phi"]
        expected_rho = fixture["scenario_4_rho"]
        expected_stationary = fixture["scenario_4_stationary"]

        rho, stationary = inverse_ar_roots(phi)

        assert stationary == expected_stationary, (
            f"Stationarity mismatch: expected {expected_stationary}, got {stationary}"
        )
        npt.assert_allclose(
            _sort_roots(rho), _sort_roots(expected_rho), atol=ATOL, rtol=RTOL,
            err_msg="Scenario 4 (near unit root AR(2) phi=[1.5,-0.55]): rho mismatch"
        )

    @pytest.mark.parity
    @pytest.mark.skipif(
        not _has_fixture("inverse_ar_roots"),
        reason="Fixture file tests/fixtures/timeseries/inverse_ar_roots.npy not found",
    )
    def test_parity_scenario_5_complex_roots(self) -> None:
        """Fixture scenario 5: Complex roots AR(2) phi=[0.8, -0.6]."""
        fixture = np.load(
            _fixture_path("inverse_ar_roots").with_suffix(".npy"),
            allow_pickle=True,
        ).item()

        phi = fixture["scenario_5_phi"]
        expected_rho = fixture["scenario_5_rho"]
        expected_stationary = fixture["scenario_5_stationary"]

        rho, stationary = inverse_ar_roots(phi)

        assert stationary == expected_stationary, (
            f"Stationarity mismatch: expected {expected_stationary}, got {stationary}"
        )
        npt.assert_allclose(
            _sort_roots(rho), _sort_roots(expected_rho), atol=ATOL, rtol=RTOL,
            err_msg="Scenario 5 (complex roots AR(2) phi=[0.8,-0.6]): rho mismatch"
        )


# ===================================================================
# Additional tests: Error handling and edge cases
# ===================================================================
class TestInverseArRootsErrorHandling:
    """Verify that invalid inputs raise appropriate exceptions.

    Ref: inverse_ar_roots.m:32 — MATLAB: error('Phi should be a column vector.')
    Python: raise ValueError('Phi should be a column vector.')
    """

    def test_inverse_ar_roots_matrix_input_raises(self) -> None:
        """A 2-D matrix (not vector) must raise ValueError."""
        phi_matrix = np.array([[0.5, 0.3], [0.2, 0.1]])
        with pytest.raises(ValueError, match="Phi should be a column vector"):
            inverse_ar_roots(phi_matrix)

    def test_inverse_ar_roots_3d_input_raises(self) -> None:
        """A 3-D array must raise ValueError."""
        phi_3d = np.array([[[0.5]]])
        with pytest.raises(ValueError, match="Phi should be a column vector"):
            inverse_ar_roots(phi_3d)


class TestInverseArRootsEdgeCases:
    """Additional edge cases for robustness."""

    def test_inverse_ar_roots_empty_input(self) -> None:
        """Empty phi should return empty rho and stationary=True (vacuous truth)."""
        rho, stationary = inverse_ar_roots(np.array([]))
        assert len(rho) == 0, "Empty phi must return empty rho"
        assert stationary is True, "Empty phi: vacuously stationary"

    def test_inverse_ar_roots_ar3_stationary(self) -> None:
        """AR(3) with small coefficients → stationary with 3 roots."""
        phi = np.array([0.5, -0.2, 0.1])
        rho, stationary = inverse_ar_roots(phi)

        assert len(rho) == 3, "AR(3) must return 3 roots"
        assert stationary is True, (
            "AR(3) with phi=[0.5,-0.2,0.1] must be stationary"
        )
        assert np.min(np.abs(rho)) > 1.0, "All roots must lie outside unit circle"

    def test_inverse_ar_roots_scalar_input(self) -> None:
        """Scalar input (0-D array) should be treated as AR(1)."""
        rho, stationary = inverse_ar_roots(np.float64(0.5))
        assert len(rho) == 1, "Scalar input treated as AR(1) → 1 root"
        assert stationary is True, "phi=0.5 is stationary"
        npt.assert_allclose(
            np.abs(rho), np.array([2.0]), atol=ATOL, rtol=RTOL,
            err_msg="Scalar phi=0.5: |root| should be 2.0"
        )

    def test_inverse_ar_roots_near_zero_phi(self) -> None:
        """Very small AR(1) coefficient → root with very large modulus."""
        phi = np.array([1e-8])
        rho, stationary = inverse_ar_roots(phi)
        assert stationary is True, "Very small phi is stationary"
        # Root = 1/1e-8 = 1e8
        npt.assert_allclose(
            np.abs(rho), np.array([1e8]), atol=1.0, rtol=RTOL,
            err_msg="Very small phi should give very large root modulus"
        )

    @pytest.mark.parametrize(
        "phi_val, expected_stationary",
        [
            (0.99, True),   # Root = 1/0.99 ≈ 1.0101 > 1
            (1.01, False),  # Root = 1/1.01 ≈ 0.9901 < 1
        ],
        ids=["phi=0.99_stationary", "phi=1.01_explosive"],
    )
    def test_inverse_ar_roots_boundary_stationarity(
        self, phi_val: float, expected_stationary: bool
    ) -> None:
        """Test stationarity near the boundary |phi| = 1."""
        phi = np.array([phi_val])
        _, stationary = inverse_ar_roots(phi)
        assert stationary is expected_stationary, (
            f"phi={phi_val}: expected stationary={expected_stationary}"
        )
