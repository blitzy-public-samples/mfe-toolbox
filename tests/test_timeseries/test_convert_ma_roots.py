"""
Pytest tests for mfe_toolbox.timeseries.convert_ma_roots.

Tests the MA polynomial root invertibility conversion function which inspects
an MA parameterization for invertibility and reflects roots outside the unit
circle back inside via the reciprocal mapping z_new = 1/z.

Source MATLAB reference: timeseries/convert_ma_roots.m
Per AAP Section 0.7.1: All comparisons use
    numpy.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-4)
"""

import os
import warnings

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.timeseries.convert_ma_roots import convert_ma_roots

# ---------------------------------------------------------------------------
# Numerical Parity Tolerance Constants
# Per AAP Section 0.7.1: Hard requirements for MATLAB reference comparison.
# ---------------------------------------------------------------------------
ATOL: float = 1e-6
"""Absolute tolerance for numerical parity assertions."""

RTOL: float = 1e-4
"""Relative tolerance for numerical parity assertions."""


# ---------------------------------------------------------------------------
# Fixture Helpers
# ---------------------------------------------------------------------------
_FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "fixtures", "timeseries"
)

_FIXTURE_NPY = os.path.join(_FIXTURE_DIR, "convert_ma_roots.npy")
_FIXTURE_CSV = os.path.join(_FIXTURE_DIR, "convert_ma_roots.csv")

# Override fixture dir from environment if set (CI support)
# Ref: AAP Section 0.7.2 — MFE_FIXTURE_DIR environment variable
_env_fixture_dir = os.environ.get("MFE_FIXTURE_DIR")
if _env_fixture_dir:
    _FIXTURE_DIR = os.path.join(_env_fixture_dir, "timeseries")
    _FIXTURE_NPY = os.path.join(_FIXTURE_DIR, "convert_ma_roots.npy")
    _FIXTURE_CSV = os.path.join(_FIXTURE_DIR, "convert_ma_roots.csv")

_HAS_FIXTURES = os.path.exists(_FIXTURE_NPY) or os.path.exists(_FIXTURE_CSV)


# ---------------------------------------------------------------------------
# Phase 1: Core Test Functions
# ---------------------------------------------------------------------------


class TestConvertMaRootsAlreadyInvertible:
    """Test MA parameterizations that are already invertible.

    When all roots of the MA polynomial lie outside the unit circle (or
    equivalently all reflected roots lie inside), the returned parameters
    should be unchanged within numerical tolerance.
    """

    def test_convert_ma_roots_already_invertible(self) -> None:
        """MA(1) with theta=0.3 — root is already outside unit circle.

        Ref: convert_ma_roots.m:12-14
        Polynomial [1, 0.3] has root -1/0.3 ≈ -3.333. Since |root|>1,
        it is reflected to 1/(-3.333) ≈ -0.3. poly([-0.3]) = [1, 0.3].
        The coefficient 0.3 is unchanged after the round-trip.
        """
        params = np.array([0.3])
        q = np.array([1])
        result = convert_ma_roots(params, q)

        # Ref: convert_ma_roots.m:14 — After reflection and poly(), same value
        npt.assert_allclose(result, params, atol=ATOL, rtol=RTOL)

    def test_ma2_already_invertible(self) -> None:
        """MA(2) with params=[0.4, 0.3] — roots already satisfy invertibility.

        Ref: convert_ma_roots.m:12-14
        Polynomial [1, 0.4, 0.3] has complex conjugate roots inside unit
        circle, so no change needed.
        """
        params = np.array([0.4, 0.3])
        q = np.array([1, 2])
        result = convert_ma_roots(params, q)

        npt.assert_allclose(result, params, atol=ATOL, rtol=RTOL)


class TestConvertMaRootsInversion:
    """Test MA parameterizations that require root inversion."""

    def test_convert_ma_roots_needs_inversion(self) -> None:
        """MA(1) with theta=2.0 — root at -2.0 reflected to -0.5.

        Ref: convert_ma_roots.m:13 — lambda(abs(lambda)>1) = 1./lambda(...)
        Polynomial [1, 2.0] has root -2.0, |root|=2>1.
        After reflection: root = 1/(-2) = -0.5.
        poly([-0.5]) = [1, 0.5]. Expected output: 0.5.
        """
        params = np.array([2.0])
        q = np.array([1])
        result = convert_ma_roots(params, q)

        expected = np.array([0.5])
        npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)

    def test_convert_ma_roots_needs_inversion_1_5(self) -> None:
        """MA(1) with theta=1.5 — root reflected to 1/1.5 = 0.6667.

        Ref: convert_ma_roots.m:13
        Polynomial [1, 1.5] has root -1.5, |root|=1.5>1.
        After reflection: root = 1/(-1.5) ≈ -0.6667.
        poly([-0.6667]) = [1, 0.6667]. Expected: 2/3.
        """
        params = np.array([1.5])
        q = np.array([1])
        result = convert_ma_roots(params, q)

        expected = np.array([1.0 / 1.5])
        npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)


class TestConvertMaRootsUnitRoot:
    """Test MA with root exactly on the unit circle."""

    def test_convert_ma_roots_unit_root(self) -> None:
        """MA(1) with theta=-1.0 — root at z=1.0 exactly on unit circle.

        Ref: convert_ma_roots.m:13 — Strictly greater than (>1), NOT >=1.
        Polynomial [1, -1] has root 1.0, |root|=1.0 which is NOT > 1.
        Therefore no reflection occurs and the parameters remain unchanged.
        """
        params = np.array([-1.0])
        q = np.array([1])
        result = convert_ma_roots(params, q)

        # Root exactly on unit circle → no reflection → parameters unchanged
        npt.assert_allclose(result, params, atol=ATOL, rtol=RTOL)


class TestConvertMaRootsContiguousLags:
    """Test MA with contiguous (regular/dense) lag structures."""

    def test_convert_ma_roots_contiguous_lags(self) -> None:
        """MA(3) with q=[1,2,3] — regular contiguous lags, root reflection.

        Ref: convert_ma_roots.m:27-28
        Since length(q) == max(q), this is the dense/regular case.
        Input [3.0, 0.5, 0.2] produces large roots that need inversion.
        """
        params = np.array([3.0, 0.5, 0.2])
        q = np.array([1, 2, 3])
        result = convert_ma_roots(params, q)

        # Independently compute expected result
        # Ref: convert_ma_roots.m:10-15
        full_poly = np.concatenate(([1.0], params))
        roots = np.roots(full_poly)
        outside = np.abs(roots) > 1.0
        roots[outside] = 1.0 / roots[outside]
        new_poly = np.real(np.poly(roots))
        new_poly[np.abs(new_poly) < 1e-10] = 0.0
        expected = new_poly[1:]  # Skip leading 1

        npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)
        assert result.shape == params.shape, "Output shape must match input shape"

        # Verify all roots of result polynomial are inside unit circle
        result_roots = np.roots(np.concatenate(([1.0], result)))
        assert np.all(np.abs(result_roots) <= 1.0 + ATOL), (
            "All roots of inverted polynomial must be inside unit circle"
        )


class TestConvertMaRootsNoncontiguousLags:
    """Test MA with non-contiguous (irregular/sparse) lag structures."""

    def test_convert_ma_roots_noncontiguous_no_warning(self) -> None:
        """Irregular MA with q=[1,3] where inversion preserves lag structure.

        Ref: convert_ma_roots.m:17-26
        Input [0.5, 0.3] with q=[1,3]. Full polynomial [1, 0.5, 0, 0.3].
        Roots are already inside unit circle, so after reflection and
        reconstruction the non-zero lag indices remain at [1, 3].
        No warning should be emitted.
        """
        params = np.array([0.5, 0.3])
        q = np.array([1, 3])

        # Should NOT emit a warning
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = convert_ma_roots(params, q)

        npt.assert_allclose(result, params, atol=ATOL, rtol=RTOL)

    def test_convert_ma_roots_noncontiguous_with_warning(self) -> None:
        """Irregular MA with q=[1,3] where inversion changes lag indices.

        Ref: convert_ma_roots.m:21-23
        Input [2.0, 3.0] with q=[1,3]. After root reflection and polynomial
        reconstruction, the non-zero coefficient indices differ from the
        original q=[1,3]. A UserWarning is issued and the original (un-inverted)
        parameters are returned.
        """
        params = np.array([2.0, 3.0])
        q = np.array([1, 3])

        with pytest.warns(UserWarning, match="irregular MA cannot be inverted"):
            result = convert_ma_roots(params, q)

        # Ref: convert_ma_roots.m:23 — parameters = MAparameters(q)
        # Original parameters are returned unchanged when lag indices shift
        npt.assert_allclose(result, params, atol=ATOL, rtol=RTOL)


class TestConvertMaRootsOutputShape:
    """Test output array shape/dimensionality guarantees."""

    def test_convert_ma_roots_output_shape_1d(self) -> None:
        """Output is a 1-D numpy array (column vector equivalent in Python).

        Ref: convert_ma_roots.m:31-33 — MATLAB ensures column vector output.
        In Python/NumPy convention, this corresponds to a 1-D array with
        shape (n,).
        """
        params = np.array([0.3])
        q = np.array([1])
        result = convert_ma_roots(params, q)

        assert isinstance(result, np.ndarray), "Output must be numpy.ndarray"
        assert result.ndim == 1, "Output must be 1-dimensional"
        assert result.shape == (1,), f"Expected shape (1,), got {result.shape}"

    def test_convert_ma_roots_output_shape_multi(self) -> None:
        """Multiple parameter output preserves correct shape.

        Ref: convert_ma_roots.m:31-33
        """
        params = np.array([0.3, 0.2, 0.1])
        q = np.array([1, 2, 3])
        result = convert_ma_roots(params, q)

        assert result.ndim == 1, "Output must be 1-dimensional"
        assert result.shape == (3,), f"Expected shape (3,), got {result.shape}"

    def test_convert_ma_roots_output_length_matches_input(self) -> None:
        """Output length always equals input parameter vector length.

        Ref: convert_ma_roots.m — output indexed by q, so len(output)==len(q).
        """
        for n in [1, 2, 5]:
            params = np.full(n, 0.2)
            q = np.arange(1, n + 1)
            result = convert_ma_roots(params, q)
            assert len(result) == n, (
                f"Output length {len(result)} must equal input length {n}"
            )


class TestConvertMaRootsSmallCoefficients:
    """Test zeroing of near-zero coefficients."""

    def test_convert_ma_roots_small_coefficients_zeroed(self) -> None:
        """Coefficients with absolute value < 1e-10 are zeroed.

        Ref: convert_ma_roots.m:15 — parameters(abs(parameters)<1e-10) = 0
        Using very small parameter (1e-15), root is tiny (~-1e-15, inside
        unit circle), no reflection. After poly() and zeroing, the coefficient
        smaller than 1e-10 becomes exactly 0.
        """
        params = np.array([1e-15])
        q = np.array([1])
        result = convert_ma_roots(params, q)

        # Ref: convert_ma_roots.m:15 — near-zero coefficients set to 0
        expected = np.array([0.0])
        npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)


class TestConvertMaRootsComplexRoots:
    """Test handling of complex conjugate root pairs."""

    def test_convert_ma_roots_complex_roots(self) -> None:
        """MA(2) with complex conjugate roots inside unit circle.

        Ref: convert_ma_roots.m:12-14
        Polynomial [1, 0, 0.9] has roots ±0.949i (complex conjugate pair).
        |root| = sqrt(0.9) ≈ 0.949 < 1, so no reflection occurs.
        Parameters remain [0.0, 0.9] unchanged.
        """
        params = np.array([0.0, 0.9])
        q = np.array([1, 2])
        result = convert_ma_roots(params, q)

        # Complex roots inside unit circle → no change
        npt.assert_allclose(result, params, atol=ATOL, rtol=RTOL)

    def test_convert_ma_roots_complex_roots_outside(self) -> None:
        """MA(2) with complex conjugate roots outside unit circle.

        Ref: convert_ma_roots.m:12-14
        Polynomial [1, 0, 2.0] has roots with |root| = sqrt(2) ≈ 1.414 > 1.
        Both conjugate roots are reflected. After reflection |root| < 1.
        """
        params = np.array([0.0, 2.0])
        q = np.array([1, 2])
        result = convert_ma_roots(params, q)

        # Independently verify
        full_poly = np.concatenate(([1.0], params))
        roots = np.roots(full_poly)
        outside = np.abs(roots) > 1.0
        roots[outside] = 1.0 / roots[outside]
        new_poly = np.real(np.poly(roots))
        new_poly[np.abs(new_poly) < 1e-10] = 0.0
        expected = new_poly[q]

        npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)

        # All roots must now be inside unit circle
        result_roots = np.roots(np.concatenate(([1.0], result)))
        assert np.all(np.abs(result_roots) <= 1.0 + ATOL)


class TestConvertMaRootsMultipleOutside:
    """Test MA with multiple roots outside the unit circle."""

    def test_convert_ma_roots_multiple_roots_outside(self) -> None:
        """MA(2) with params=[5.0, 6.0] — multiple roots needing reflection.

        Ref: convert_ma_roots.m:13 — all roots with |lambda|>1 are reflected.
        Polynomial [1, 5, 6] has roots -2 and -3, both outside unit circle.
        After reflection: roots become -0.5 and -1/3.
        poly([-0.5, -1/3]) = [1, 5/6, 1/6].
        Expected output: [5/6, 1/6] ≈ [0.8333, 0.1667].
        """
        params = np.array([5.0, 6.0])
        q = np.array([1, 2])
        result = convert_ma_roots(params, q)

        # Roots of [1, 5, 6] are -2 and -3
        # Reflected: -0.5 and -1/3
        # poly([-0.5, -1/3]) = [1, 5/6, 1/6]
        expected = np.array([5.0 / 6.0, 1.0 / 6.0])
        npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)

        # Verify all roots of result are inside unit circle
        result_roots = np.roots(np.concatenate(([1.0], result)))
        assert np.all(np.abs(result_roots) <= 1.0 + ATOL)


class TestConvertMaRootsParity:
    """Fixture-based parity comparison against MATLAB reference outputs."""

    @pytest.mark.skipif(
        not _HAS_FIXTURES,
        reason="Fixture files not found in tests/fixtures/timeseries/",
    )
    def test_convert_ma_roots_parity_npy(self) -> None:
        """Compare against MATLAB-generated .npy fixture data.

        Ref: AAP Section 0.7.1 — Every migrated function MUST pass
        numpy.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-4)
        against MATLAB-generated fixtures.
        """
        if not os.path.exists(_FIXTURE_NPY):
            pytest.skip(f"Fixture file not found: {_FIXTURE_NPY}")

        fixture_data = np.load(_FIXTURE_NPY, allow_pickle=True).item()

        for scenario_key, scenario in fixture_data.items():
            params_input = np.asarray(scenario["parameters_input"], dtype=np.float64)
            q = np.asarray(scenario["q"], dtype=np.int64)
            expected_output = np.asarray(
                scenario["expected_output"], dtype=np.float64
            )

            # Some scenarios may trigger warnings (irregular MA)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = convert_ma_roots(params_input, q)

            npt.assert_allclose(
                result,
                expected_output,
                atol=ATOL,
                rtol=RTOL,
                err_msg=(
                    f"Parity failure for {scenario_key}: "
                    f"{scenario.get('description', '')}"
                ),
            )

    @pytest.mark.skipif(
        not _HAS_FIXTURES,
        reason="Fixture files not found in tests/fixtures/timeseries/",
    )
    def test_convert_ma_roots_parity_csv(self) -> None:
        """Compare against MATLAB-generated .csv fixture data.

        The CSV fixture encodes multiple scenarios row-by-row with columns:
        scenario, param_index, q_1, q_2, input_value, output_value.
        """
        if not os.path.exists(_FIXTURE_CSV):
            pytest.skip(f"Fixture file not found: {_FIXTURE_CSV}")

        # Parse CSV fixture — group by scenario
        import csv

        scenarios: dict[str, dict] = {}
        with open(_FIXTURE_CSV, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row["scenario"]
                if name not in scenarios:
                    scenarios[name] = {
                        "inputs": [],
                        "outputs": [],
                        "q_values": set(),
                    }
                scenarios[name]["inputs"].append(float(row["input_value"]))
                scenarios[name]["outputs"].append(float(row["output_value"]))
                # Collect all non-empty q columns
                for qcol in ["q_1", "q_2"]:
                    if row.get(qcol, ""):
                        scenarios[name]["q_values"].add(int(row[qcol]))

        for name, data in scenarios.items():
            params_input = np.array(data["inputs"], dtype=np.float64)
            q = np.array(sorted(data["q_values"]), dtype=np.int64)
            expected_output = np.array(data["outputs"], dtype=np.float64)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = convert_ma_roots(params_input, q)

            npt.assert_allclose(
                result,
                expected_output,
                atol=ATOL,
                rtol=RTOL,
                err_msg=f"CSV parity failure for scenario '{name}'",
            )


# ---------------------------------------------------------------------------
# Phase 2: Edge Cases
# ---------------------------------------------------------------------------


class TestConvertMaRootsEdgeCases:
    """Edge case tests for boundary and special inputs."""

    def test_convert_ma_roots_single_lag(self) -> None:
        """Simplest possible case: single lag q=[1] with small MA coefficient.

        Ref: convert_ma_roots.m:10-14
        MA(1) with theta=0.1. Root = -10, |root|>1, reflected to -0.1.
        poly([-0.1]) = [1, 0.1]. Output unchanged at 0.1.
        """
        params = np.array([0.1])
        q = np.array([1])
        result = convert_ma_roots(params, q)

        npt.assert_allclose(result, np.array([0.1]), atol=ATOL, rtol=RTOL)

    def test_convert_ma_roots_high_order(self) -> None:
        """High-order MA(10) with contiguous q=[1..10], multiple roots to reflect.

        Ref: convert_ma_roots.m:10-28
        A 10th-order MA polynomial with some roots outside the unit circle.
        All roots must be reflected inside after conversion.
        """
        params = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        q = np.arange(1, 11)
        result = convert_ma_roots(params, q)

        # Independently compute expected via the MATLAB algorithm
        # Ref: convert_ma_roots.m:10-15
        full_poly = np.concatenate(([1.0], params))
        roots = np.roots(full_poly)
        outside = np.abs(roots) > 1.0
        roots[outside] = 1.0 / roots[outside]
        new_poly = np.real(np.poly(roots))
        new_poly[np.abs(new_poly) < 1e-10] = 0.0
        expected = new_poly[1:]

        npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)

        # Verify all roots of result are inside the unit circle
        result_roots = np.roots(np.concatenate(([1.0], result)))
        assert np.all(np.abs(result_roots) <= 1.0 + ATOL), (
            "All roots must be inside unit circle after inversion"
        )
        assert result.shape == (10,), f"Expected shape (10,), got {result.shape}"

    def test_convert_ma_roots_negative_params(self) -> None:
        """MA(1) with negative theta=-0.5 — root at z=2, reflected to z=0.5.

        Ref: convert_ma_roots.m:12-14
        Polynomial [1, -0.5] has root 0.5. |root|<1, no reflection.
        """
        params = np.array([-0.5])
        q = np.array([1])
        result = convert_ma_roots(params, q)

        # Root of [1, -0.5] is 0.5, which is inside unit circle
        npt.assert_allclose(result, params, atol=ATOL, rtol=RTOL)

    def test_convert_ma_roots_large_negative_params(self) -> None:
        """MA(1) with theta=-3.0 — root at z=1/3, inside unit circle.

        Ref: convert_ma_roots.m:12-14
        Polynomial [1, -3.0] has root 3.0. |root|=3>1, reflected to 1/3.
        poly([1/3]) = [1, -1/3]. Expected output: -1/3.
        """
        params = np.array([-3.0])
        q = np.array([1])
        result = convert_ma_roots(params, q)

        expected = np.array([-1.0 / 3.0])
        npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)

    def test_convert_ma_roots_ma2_one_root_outside(self) -> None:
        """MA(2) where only one of two roots is outside the unit circle.

        Ref: convert_ma_roots.m:13
        Polynomial [1, 0.5, -2.0] has one root inside and one outside.
        Only the root outside is reflected.
        """
        params = np.array([0.5, -2.0])
        q = np.array([1, 2])
        result = convert_ma_roots(params, q)

        # Verify independently
        full_poly = np.concatenate(([1.0], params))
        roots = np.roots(full_poly)
        outside = np.abs(roots) > 1.0
        roots[outside] = 1.0 / roots[outside]
        new_poly = np.real(np.poly(roots))
        new_poly[np.abs(new_poly) < 1e-10] = 0.0
        expected = new_poly[q]

        npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)

        # All roots of output must be inside unit circle
        result_roots = np.roots(np.concatenate(([1.0], result)))
        assert np.all(np.abs(result_roots) <= 1.0 + ATOL)


class TestConvertMaRootsInputValidation:
    """Test input validation and error handling."""

    def test_convert_ma_roots_empty_params_raises(self) -> None:
        """Empty parameter array should raise ValueError.

        Ref: Python implementation input validation.
        """
        with pytest.raises(ValueError, match="must not be empty"):
            convert_ma_roots(np.array([]), np.array([1]))

    def test_convert_ma_roots_empty_q_raises(self) -> None:
        """Empty q array should raise ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            convert_ma_roots(np.array([0.5]), np.array([], dtype=int))

    def test_convert_ma_roots_nonpositive_q_raises(self) -> None:
        """q with non-positive elements should raise ValueError.

        Ref: convert_ma_roots.m — q values are 1-based lag positions.
        """
        with pytest.raises(ValueError, match="positive integers"):
            convert_ma_roots(np.array([0.5]), np.array([0]))

    def test_convert_ma_roots_length_mismatch_raises(self) -> None:
        """Mismatched lengths of parameters and q should raise ValueError."""
        with pytest.raises(ValueError, match="Length"):
            convert_ma_roots(np.array([0.5, 0.3]), np.array([1]))


class TestConvertMaRootsReturnType:
    """Test return type and data type guarantees."""

    def test_return_type_is_ndarray(self) -> None:
        """Return value must always be a numpy.ndarray."""
        result = convert_ma_roots(np.array([0.5]), np.array([1]))
        assert isinstance(result, np.ndarray)

    def test_return_dtype_is_float(self) -> None:
        """Return value must have floating-point dtype."""
        result = convert_ma_roots(np.array([0.5]), np.array([1]))
        assert np.issubdtype(result.dtype, np.floating)

    def test_return_is_real(self) -> None:
        """Return value must be real (no imaginary component).

        Even when the MA polynomial has complex conjugate roots, the
        coefficients of the polynomial with real input must be real.
        """
        # MA(2) that produces complex roots
        result = convert_ma_roots(np.array([0.0, 0.9]), np.array([1, 2]))
        assert np.all(np.isreal(result)), "Output must be real-valued"


# ---------------------------------------------------------------------------
# Parametrized Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "params, q, expected_description",
    [
        (np.array([0.3]), np.array([1]), "MA(1) invertible"),
        (np.array([2.0]), np.array([1]), "MA(1) non-invertible"),
        (np.array([-1.0]), np.array([1]), "MA(1) unit root"),
        (np.array([0.4, 0.3]), np.array([1, 2]), "MA(2) invertible"),
        (np.array([5.0, 6.0]), np.array([1, 2]), "MA(2) large coefficients"),
    ],
    ids=[
        "MA1-invertible",
        "MA1-non-invertible",
        "MA1-unit-root",
        "MA2-invertible",
        "MA2-large",
    ],
)
def test_convert_ma_roots_parametrized_invertibility(
    params: np.ndarray, q: np.ndarray, expected_description: str
) -> None:
    """Parametrized test: all results must have roots inside unit circle.

    Ref: convert_ma_roots.m — The function enforces invertibility by
    reflecting roots outside the unit circle. After conversion, ALL roots
    of the output polynomial must satisfy |root| <= 1.
    """
    result = convert_ma_roots(params, q)

    # Build output polynomial and verify root magnitudes
    result_poly = np.concatenate(([1.0], np.zeros(int(np.max(q)))))
    result_poly[q] = result
    result_roots = np.roots(result_poly)

    assert np.all(np.abs(result_roots) <= 1.0 + ATOL), (
        f"[{expected_description}] All roots of inverted polynomial must be "
        f"inside unit circle. Max |root| = {np.max(np.abs(result_roots)):.6f}"
    )
