"""Pytest parity and unit tests for ``mfe_toolbox.sandbox.sdiff``.

Tests the seasonal differencing function migrated from MATLAB ``sandbox/sdiff.m``
(MFE Toolbox Version 4.0).  Covers:

* **Unit tests** — Known-value tests verifying the polynomial construction,
  lag/scale extraction, and filter application logic for various ``(d, S)``
  combinations.
* **Input validation tests** — Behaviour on edge-case inputs (empty, too short,
  zero/negative orders, scalar arguments, 2-D arrays).
* **Parity tests** — Numerical comparison against MATLAB-generated reference
  fixtures stored in ``tests/fixtures/sandbox/sdiff.npy`` using the
  project-wide tolerance constants ``ATOL = 1e-6`` and ``RTOL = 1e-4``.
* **Return type / shape tests** — Structural assertions on the three-element
  return tuple ``(y, lags, scales)``.

Per AAP Section 0.7.1:
    numpy.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-4)
    is required for all fixture-based comparisons.

Ref: sandbox/sdiff.m (25 lines).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.sandbox.sdiff import sdiff

# Import tolerance constants from the project-wide conftest.
# These are module-level constants, so we import them directly.
from tests.conftest import ATOL, RTOL

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_full_polynomial(d_arr: np.ndarray, S_arr: np.ndarray) -> np.ndarray:
    """Independently compute the seasonal differencing polynomial.

    Builds the product polynomial ``prod_{i} (1 - B^{S[i]})^{d[i]}`` using
    ``np.convolve``, mirroring the MATLAB implementation but without any
    extraction or filtering step.  Used as the ground-truth reference for
    verifying that ``sdiff`` constructs the correct polynomial.
    """
    poly = np.array([1.0])
    for i in range(len(d_arr)):
        for _j in range(int(d_arr[i])):
            kernel = np.zeros(int(S_arr[i]) + 1)
            kernel[0] = 1.0
            kernel[-1] = -1.0
            poly = np.convolve(poly, kernel)
    return poly


# =========================================================================
# Phase 1: Unit Tests (no fixture dependency)
# =========================================================================


class TestSdiffImports:
    """Verify that the sdiff module is importable."""

    def test_sdiff_imports(self) -> None:
        """Smoke-test: the ``sdiff`` function can be imported."""
        assert callable(sdiff)


class TestSdiffFirstDifference:
    """d=1, S=1 — simple first-order difference operator ``(1 - B)``.

    Polynomial: ``[1, -1]``.
    Nonzero lags after removing constant: ``[1]``, scales ``[-1]``.

    NOTE (Ref: sdiff.m:21): The MATLAB implementation applies the filter
    starting from ``j=2:length(lags)``, which means when there is only ONE
    non-constant lag the filter loop does not execute.  The output ``y`` is
    therefore ``x[ml:]`` (a tail slice of x), NOT the mathematical first
    difference ``x[t] - x[t-1]``.  The Python migration faithfully preserves
    this original MATLAB behaviour.
    """

    def test_sdiff_first_difference(self) -> None:
        x = np.array([1.0, 3.0, 6.0, 10.0, 15.0])
        y, lags, scales = sdiff(x, np.array([1]), np.array([1]))

        # Polynomial (1-B): nonzero positions after removing constant at lag 0
        npt.assert_array_equal(lags, np.array([1]))
        npt.assert_array_equal(scales, np.array([-1.0]))

        # Output y = x[ml:] = x[1:] due to filter loop starting at j=2
        # Ref: sdiff.m:21 — j=2:length(lags) with length(lags)==1 ⇒ no iterations
        expected_y = np.array([3.0, 6.0, 10.0, 15.0])
        npt.assert_allclose(y, expected_y, atol=1e-12)

    def test_sdiff_first_difference_length(self) -> None:
        x = np.array([1.0, 3.0, 6.0, 10.0, 15.0])
        y, lags, _scales = sdiff(x, np.array([1]), np.array([1]))
        assert len(y) == len(x) - int(np.max(lags))


class TestSdiffSeasonalQuarterly:
    """d=1, S=4 — quarterly seasonal difference ``(1 - B^4)``."""

    def test_sdiff_seasonal_quarterly(self) -> None:
        x = np.arange(1.0, 21.0)  # 20 elements
        y, lags, scales = sdiff(x, np.array([1]), np.array([4]))

        # Polynomial [1, 0, 0, 0, -1]: nonzero after removing constant → [4]
        npt.assert_array_equal(lags, np.array([4]))
        npt.assert_array_equal(scales, np.array([-1.0]))

        # y = x[4:] (filter loop does not execute for single lag)
        # Ref: sdiff.m:21 — j=2:1 ⇒ empty loop
        expected_y = np.arange(5.0, 21.0)
        npt.assert_allclose(y, expected_y, atol=1e-12)
        assert len(y) == 16  # 20 - 4


class TestSdiffSecondDifference:
    """d=2, S=1 — second-order difference ``(1 - B)^2 = 1 - 2B + B^2``."""

    def test_sdiff_second_difference(self) -> None:
        x = np.array([1.0, 4.0, 9.0, 16.0, 25.0, 36.0])
        y, lags, scales = sdiff(x, np.array([2]), np.array([1]))

        # Polynomial [1, -2, 1]: after removing constant → lags=[1,2], scales=[-2,1]
        npt.assert_array_equal(lags, np.array([1, 2]))
        npt.assert_allclose(scales, np.array([-2.0, 1.0]), atol=1e-12)

        # Filter: y = x[2:] + scales[1]*x[0:T-2]
        # = [9,16,25,36] + 1*[1,4,9,16] = [10,20,34,52]
        # Ref: sdiff.m:21-23 — loop runs j=2 only, applying scales(2)=1 at lags(2)=2
        expected_y = np.array([10.0, 20.0, 34.0, 52.0])
        npt.assert_allclose(y, expected_y, atol=1e-12)

    def test_sdiff_second_difference_length(self) -> None:
        x = np.array([1.0, 4.0, 9.0, 16.0, 25.0, 36.0])
        y, lags, _s = sdiff(x, np.array([2]), np.array([1]))
        assert len(y) == len(x) - int(np.max(lags))


class TestSdiffSeasonalMonthly:
    """d=1, S=12 — monthly seasonal difference ``(1 - B^12)``."""

    def test_sdiff_seasonal_monthly(self) -> None:
        x = np.arange(1.0, 61.0)  # T=60 (5 years of monthly data)
        y, lags, scales = sdiff(x, np.array([1]), np.array([12]))

        npt.assert_array_equal(lags, np.array([12]))
        npt.assert_array_equal(scales, np.array([-1.0]))

        # y = x[12:] (single-lag case: loop doesn't execute)
        expected_y = np.arange(13.0, 61.0)
        npt.assert_allclose(y, expected_y, atol=1e-12)
        assert len(y) == 48  # 60 - 12


class TestSdiffSecondSeasonal:
    """d=2, S=12 — ``(1 - B^12)^2 = 1 - 2B^12 + B^24``."""

    def test_sdiff_second_seasonal_lags_scales(self) -> None:
        x = np.arange(1.0, 61.0)
        y, lags, scales = sdiff(x, np.array([2]), np.array([12]))

        npt.assert_array_equal(lags, np.array([12, 24]))
        npt.assert_allclose(scales, np.array([-2.0, 1.0]), atol=1e-12)

    def test_sdiff_second_seasonal_length(self) -> None:
        x = np.arange(1.0, 61.0)
        y, lags, _s = sdiff(x, np.array([2]), np.array([12]))
        assert len(y) == len(x) - int(np.max(lags))  # 60 - 24 = 36

    def test_sdiff_second_seasonal_values(self) -> None:
        x = np.arange(1.0, 61.0)
        y, lags, scales = sdiff(x, np.array([2]), np.array([12]))

        # Filter: y = x[24:] + 1*x[0:36]
        # Ref: sdiff.m:21-23 — loop j=2 only, scales(2)=1, lags(2)=24
        T = len(x)
        ml = int(np.max(lags))
        expected_y = x[ml:T] + 1.0 * x[0 : T - 24]
        npt.assert_allclose(y, expected_y, atol=1e-12)


class TestSdiffOutputLength:
    """Verify output length = T - max(lags) for various (d, S, T) combinations."""

    @pytest.mark.parametrize(
        "d, S, T",
        [
            (np.array([1]), np.array([1]), 100),
            (np.array([1]), np.array([4]), 100),
            (np.array([2]), np.array([1]), 100),
            (np.array([1]), np.array([12]), 60),
            (np.array([2]), np.array([12]), 60),
            (np.array([3]), np.array([1]), 50),
            (np.array([1, 1]), np.array([1, 12]), 100),
        ],
    )
    def test_sdiff_output_length(
        self, d: np.ndarray, S: np.ndarray, T: int, rng: np.random.Generator
    ) -> None:
        x = rng.standard_normal(T)
        y, lags, _scales = sdiff(x, d, S)

        if len(lags) > 0:
            expected_len = T - int(np.max(lags))
            assert len(y) == expected_len, (
                f"Expected length {expected_len}, got {len(y)} "
                f"for d={d}, S={S}, T={T}, max_lag={int(np.max(lags))}"
            )
        else:
            assert len(y) == 0


class TestSdiffLagsScalesConsistency:
    """Verify lags and scales match the independently computed polynomial."""

    @pytest.mark.parametrize(
        "d, S, expected_lags, expected_scales",
        [
            # (1-B): nonzero at [0,1], after removing constant → [1], scales=[-1]
            (np.array([1]), np.array([1]), [1], [-1.0]),
            # (1-B)^2 = 1-2B+B^2: after removing constant → [1,2], [-2,1]
            (np.array([2]), np.array([1]), [1, 2], [-2.0, 1.0]),
            # (1-B^4): after removing constant → [4], [-1]
            (np.array([1]), np.array([4]), [4], [-1.0]),
            # (1-B)(1-B^12) = 1 - B - B^12 + B^13: → [1,12,13], [-1,-1,1]
            (np.array([1, 1]), np.array([1, 12]), [1, 12, 13], [-1.0, -1.0, 1.0]),
            # (1-B^12)^2 = 1 - 2B^12 + B^24: → [12,24], [-2,1]
            (np.array([2]), np.array([12]), [12, 24], [-2.0, 1.0]),
        ],
    )
    def test_sdiff_lags_and_scales_consistency(
        self,
        d: np.ndarray,
        S: np.ndarray,
        expected_lags: list[int],
        expected_scales: list[float],
    ) -> None:
        x = np.arange(1.0, 101.0)
        _y, lags, scales = sdiff(x, d, S)

        npt.assert_array_equal(lags, np.array(expected_lags))
        npt.assert_allclose(scales, np.array(expected_scales), atol=1e-12)

    @pytest.mark.parametrize(
        "d, S",
        [
            (np.array([1]), np.array([1])),
            (np.array([2]), np.array([1])),
            (np.array([3]), np.array([1])),
            (np.array([1]), np.array([4])),
            (np.array([2]), np.array([12])),
            (np.array([1, 1]), np.array([1, 12])),
        ],
    )
    def test_sdiff_lags_match_polynomial_nonzeros(
        self, d: np.ndarray, S: np.ndarray
    ) -> None:
        """Verify lags correspond to nonzero polynomial positions (minus constant)."""
        x = np.arange(1.0, 101.0)
        _y, lags, scales = sdiff(x, d, S)

        full_poly = _compute_full_polynomial(d, S)
        # Nonzero positions in full polynomial (0-indexed)
        all_nonzero = np.nonzero(full_poly)[0]
        # Remove constant term at position 0
        expected_lags = all_nonzero[1:]
        expected_scales = full_poly[expected_lags]

        npt.assert_array_equal(lags, expected_lags)
        npt.assert_allclose(scales, expected_scales, atol=1e-12)


class TestSdiffConvolutionProperty:
    """Verify iterative convolution builds the correct polynomial."""

    def test_sdiff_third_order_difference(self) -> None:
        """d=3, S=1: ``(1-B)^3 = 1 - 3B + 3B^2 - B^3``."""
        x = np.arange(1.0, 11.0)
        y, lags, scales = sdiff(x, np.array([3]), np.array([1]))

        # After removing constant → lags=[1,2,3], scales=[-3,3,-1]
        npt.assert_array_equal(lags, np.array([1, 2, 3]))
        npt.assert_allclose(scales, np.array([-3.0, 3.0, -1.0]), atol=1e-12)

        # Verify full polynomial independently
        full_poly = _compute_full_polynomial(np.array([3]), np.array([1]))
        npt.assert_allclose(full_poly, np.array([1.0, -3.0, 3.0, -1.0]), atol=1e-12)

    def test_sdiff_third_order_values(self) -> None:
        """Check computed output for d=3, S=1 on linear sequence."""
        x = np.arange(1.0, 11.0)
        y, lags, scales = sdiff(x, np.array([3]), np.array([1]))

        # Filter: ml=3, T=10
        # y = x[3:] + 3*x[1:8] + (-1)*x[0:7]
        T = len(x)
        ml = int(np.max(lags))
        expected_y = x[ml:T].copy()
        # j=1: scales[1]=3, lags[1]=2
        expected_y = expected_y + 3.0 * x[ml - 2 : T - 2]
        # j=2: scales[2]=-1, lags[2]=3
        expected_y = expected_y + (-1.0) * x[ml - 3 : T - 3]

        npt.assert_allclose(y, expected_y, atol=1e-12)


class TestSdiffRandomData:
    """Test with random data for shape/type validity."""

    @pytest.mark.parametrize(
        "d, S",
        [
            (np.array([1]), np.array([1])),
            (np.array([1]), np.array([4])),
            (np.array([2]), np.array([1])),
            (np.array([1]), np.array([12])),
            (np.array([1, 1]), np.array([1, 12])),
        ],
    )
    def test_sdiff_random_data(
        self, d: np.ndarray, S: np.ndarray, rng: np.random.Generator
    ) -> None:
        x = rng.standard_normal(500)
        y, lags, scales = sdiff(x, d, S)

        # Return types
        assert isinstance(y, np.ndarray)
        assert isinstance(lags, np.ndarray)
        assert isinstance(scales, np.ndarray)

        # Output length
        if len(lags) > 0:
            assert len(y) == len(x) - int(np.max(lags))

        # Finite values
        assert np.all(np.isfinite(y))
        assert np.all(np.isfinite(lags))
        assert np.all(np.isfinite(scales))


# =========================================================================
# Phase 2: Input Validation / Edge Case Tests
# =========================================================================


class TestSdiffEdgeCases:
    """Verify behaviour on edge-case and unusual inputs.

    NOTE: The faithful MATLAB migration does NOT raise ``ValueError`` for
    invalid ``d`` or ``S`` values.  Instead, zero/negative ``d`` produces
    an empty polynomial with no lags, and zero ``S`` similarly returns no
    differencing.  This matches the original MATLAB ``sdiff.m`` behaviour
    where ``for j=1:d(i)`` simply does not execute for non-positive ``d``.
    """

    def test_sdiff_d_zero_returns_empty(self) -> None:
        """d=0 → no convolution, empty lags and empty y."""
        x = np.arange(1.0, 11.0)
        y, lags, scales = sdiff(x, np.array([0]), np.array([1]))
        assert len(lags) == 0
        assert len(scales) == 0
        assert len(y) == 0

    def test_sdiff_d_negative_returns_empty(self) -> None:
        """d<0 → range(negative) is empty, no convolution."""
        x = np.arange(1.0, 11.0)
        y, lags, scales = sdiff(x, np.array([-1]), np.array([1]))
        assert len(lags) == 0
        assert len(scales) == 0
        assert len(y) == 0

    def test_sdiff_S_zero_returns_empty(self) -> None:
        """S=0 → kernel is [−1] (single element), poly has no useful lags."""
        x = np.arange(1.0, 11.0)
        y, lags, scales = sdiff(x, np.array([1]), np.array([0]))
        assert len(lags) == 0
        assert len(scales) == 0
        assert len(y) == 0

    def test_sdiff_S_negative_raises(self) -> None:
        """S<0 → kernel has zero-length array, indexing raises."""
        x = np.arange(1.0, 11.0)
        with pytest.raises((IndexError, ValueError)):
            sdiff(x, np.array([1]), np.array([-1]))

    def test_sdiff_x_too_short_returns_empty(self) -> None:
        """x shorter than max_lag → empty y but valid lags/scales."""
        x = np.array([1.0, 2.0])
        y, lags, scales = sdiff(x, np.array([1]), np.array([4]))
        npt.assert_array_equal(lags, np.array([4]))
        npt.assert_array_equal(scales, np.array([-1.0]))
        assert len(y) == 0

    def test_sdiff_empty_x_returns_empty_y(self) -> None:
        """Empty x → empty y, but polynomial structure is still computed."""
        y, lags, scales = sdiff(np.array([]), np.array([1]), np.array([1]))
        npt.assert_array_equal(lags, np.array([1]))
        npt.assert_array_equal(scales, np.array([-1.0]))
        assert len(y) == 0

    def test_sdiff_none_x_returns_empty_y(self) -> None:
        """None x → handled gracefully, same as empty."""
        y, lags, scales = sdiff(None, np.array([1]), np.array([1]))
        npt.assert_array_equal(lags, np.array([1]))
        npt.assert_array_equal(scales, np.array([-1.0]))
        assert len(y) == 0

    def test_sdiff_scalar_d_and_S(self) -> None:
        """Scalar d and S (not arrays) are accepted via np.atleast_1d."""
        x = np.arange(1.0, 11.0)
        y, lags, scales = sdiff(x, 1, 1)
        npt.assert_array_equal(lags, np.array([1]))
        npt.assert_array_equal(scales, np.array([-1.0]))
        assert len(y) == len(x) - 1

    def test_sdiff_float_d_truncated(self) -> None:
        """Non-integer d is truncated via int() — same as floor for positive."""
        x = np.arange(1.0, 11.0)
        y_float, lags_float, scales_float = sdiff(x, np.array([1.7]), np.array([1]))
        y_int, lags_int, scales_int = sdiff(x, np.array([1]), np.array([1]))
        npt.assert_array_equal(lags_float, lags_int)
        npt.assert_allclose(scales_float, scales_int, atol=1e-12)
        npt.assert_allclose(y_float, y_int, atol=1e-12)

    def test_sdiff_2d_x_handled(self) -> None:
        """2-D input x is processed along the first axis (MATLAB column behaviour)."""
        x_2d = np.arange(20, dtype=float).reshape(10, 2)
        y, lags, scales = sdiff(x_2d, np.array([1]), np.array([1]))
        # Should process without error; lags/scales are independent of x shape
        npt.assert_array_equal(lags, np.array([1]))
        npt.assert_array_equal(scales, np.array([-1.0]))
        # Output y has shape (T-1, 2) since x is (10, 2) and ml=1
        assert y.shape == (9, 2)


# =========================================================================
# Phase 3: Parity Tests (against MATLAB fixtures)
# =========================================================================


@pytest.mark.parity
class TestSdiffParity:
    """Compare Python ``sdiff`` output against MATLAB reference fixtures.

    Fixtures are stored in ``tests/fixtures/sandbox/sdiff.npy`` as an
    object-dtype array containing a dict keyed by test-case name.  Each
    value is a dict with keys:

    * ``input_x`` — input time series (1-D float64 array)
    * ``d_values`` — differencing orders (1-D float64 array)
    * ``S_values`` — seasonal periods (1-D float64 array)
    * ``output_y`` — expected differenced series
    * ``output_lags`` — expected lag indices
    * ``output_scales`` — expected polynomial coefficients
    """

    @staticmethod
    def _load_fixture_case(
        fixture_dir: Path, case_name: str
    ) -> dict[str, np.ndarray]:
        """Load a named test case from the sdiff fixture file."""
        fixture_path = fixture_dir / "sdiff.npy"
        if not fixture_path.exists():
            pytest.skip(f"Fixture file not found: {fixture_path}")
        data = np.load(fixture_path, allow_pickle=True).item()
        if case_name not in data:
            pytest.skip(f"Fixture case '{case_name}' not found in {fixture_path}")
        return data[case_name]

    def test_sdiff_parity_simple_diff(self, sandbox_fixture_dir: Path) -> None:
        """Parity: d=1, S=1 (simple first-order difference)."""
        case = self._load_fixture_case(sandbox_fixture_dir, "simple_diff")
        y, lags, scales = sdiff(
            case["input_x"], case["d_values"], case["S_values"]
        )
        npt.assert_allclose(y, case["output_y"], atol=ATOL, rtol=RTOL)
        npt.assert_allclose(lags, case["output_lags"], atol=ATOL, rtol=RTOL)
        npt.assert_allclose(scales, case["output_scales"], atol=ATOL, rtol=RTOL)

    def test_sdiff_parity_seasonal_diff(self, sandbox_fixture_dir: Path) -> None:
        """Parity: d=1, S=12 (monthly seasonal difference)."""
        case = self._load_fixture_case(sandbox_fixture_dir, "seasonal_diff")
        y, lags, scales = sdiff(
            case["input_x"], case["d_values"], case["S_values"]
        )
        npt.assert_allclose(y, case["output_y"], atol=ATOL, rtol=RTOL)
        npt.assert_allclose(lags, case["output_lags"], atol=ATOL, rtol=RTOL)
        npt.assert_allclose(scales, case["output_scales"], atol=ATOL, rtol=RTOL)

    def test_sdiff_parity_quarterly_seasonal(
        self, sandbox_fixture_dir: Path
    ) -> None:
        """Parity: d=1, S=4 (quarterly seasonal difference)."""
        case = self._load_fixture_case(sandbox_fixture_dir, "quarterly_seasonal")
        y, lags, scales = sdiff(
            case["input_x"], case["d_values"], case["S_values"]
        )
        npt.assert_allclose(y, case["output_y"], atol=ATOL, rtol=RTOL)
        npt.assert_allclose(lags, case["output_lags"], atol=ATOL, rtol=RTOL)
        npt.assert_allclose(scales, case["output_scales"], atol=ATOL, rtol=RTOL)

    def test_sdiff_parity_double_seasonal(
        self, sandbox_fixture_dir: Path
    ) -> None:
        """Parity: d=2, S=12 — second-order seasonal difference."""
        case = self._load_fixture_case(sandbox_fixture_dir, "double_seasonal")
        y, lags, scales = sdiff(
            case["input_x"], case["d_values"], case["S_values"]
        )
        npt.assert_allclose(y, case["output_y"], atol=ATOL, rtol=RTOL)
        npt.assert_allclose(lags, case["output_lags"], atol=ATOL, rtol=RTOL)
        npt.assert_allclose(scales, case["output_scales"], atol=ATOL, rtol=RTOL)

    def test_sdiff_parity_combined_diff(self, sandbox_fixture_dir: Path) -> None:
        """Parity: d=[1,1], S=[1,12] — combined regular + seasonal diff."""
        case = self._load_fixture_case(sandbox_fixture_dir, "combined_diff")
        y, lags, scales = sdiff(
            case["input_x"], case["d_values"], case["S_values"]
        )
        npt.assert_allclose(y, case["output_y"], atol=ATOL, rtol=RTOL)
        npt.assert_allclose(lags, case["output_lags"], atol=ATOL, rtol=RTOL)
        npt.assert_allclose(scales, case["output_scales"], atol=ATOL, rtol=RTOL)

    def test_sdiff_parity_empty_data(self, sandbox_fixture_dir: Path) -> None:
        """Parity: empty input x — should return empty y."""
        case = self._load_fixture_case(sandbox_fixture_dir, "empty_data")
        y, lags, scales = sdiff(
            case["input_x"], case["d_values"], case["S_values"]
        )
        assert len(y) == 0
        npt.assert_allclose(lags, case["output_lags"], atol=ATOL, rtol=RTOL)
        npt.assert_allclose(scales, case["output_scales"], atol=ATOL, rtol=RTOL)

    def test_sdiff_parity_short_data(self, sandbox_fixture_dir: Path) -> None:
        """Parity: x too short for filter — should return empty y."""
        case = self._load_fixture_case(sandbox_fixture_dir, "short_data")
        y, lags, scales = sdiff(
            case["input_x"], case["d_values"], case["S_values"]
        )
        assert len(y) == 0
        npt.assert_allclose(lags, case["output_lags"], atol=ATOL, rtol=RTOL)
        npt.assert_allclose(scales, case["output_scales"], atol=ATOL, rtol=RTOL)

    @pytest.mark.parametrize(
        "case_name",
        [
            "simple_diff",
            "seasonal_diff",
            "quarterly_seasonal",
            "double_seasonal",
            "combined_diff",
        ],
    )
    def test_sdiff_parity_multiple_configs(
        self, sandbox_fixture_dir: Path, case_name: str
    ) -> None:
        """Parametrized parity across all non-edge fixture cases."""
        case = self._load_fixture_case(sandbox_fixture_dir, case_name)
        y, lags, scales = sdiff(
            case["input_x"], case["d_values"], case["S_values"]
        )
        npt.assert_allclose(
            y, case["output_y"], atol=ATOL, rtol=RTOL,
            err_msg=f"y mismatch for fixture case '{case_name}'",
        )
        npt.assert_allclose(
            lags, case["output_lags"], atol=ATOL, rtol=RTOL,
            err_msg=f"lags mismatch for fixture case '{case_name}'",
        )
        npt.assert_allclose(
            scales, case["output_scales"], atol=ATOL, rtol=RTOL,
            err_msg=f"scales mismatch for fixture case '{case_name}'",
        )


# =========================================================================
# Phase 4: Return Type and Shape Tests
# =========================================================================


class TestSdiffReturnTypes:
    """Verify return types, shapes, and structural invariants."""

    def test_sdiff_return_types(self) -> None:
        """y, lags, and scales are all numpy ndarrays."""
        x = np.arange(1.0, 51.0)
        y, lags, scales = sdiff(x, np.array([1]), np.array([1]))
        assert isinstance(y, np.ndarray)
        assert isinstance(lags, np.ndarray)
        assert isinstance(scales, np.ndarray)

    def test_sdiff_return_tuple_length(self) -> None:
        """sdiff returns exactly a 3-element tuple."""
        x = np.arange(1.0, 51.0)
        result = sdiff(x, np.array([1]), np.array([1]))
        assert len(result) == 3

    def test_sdiff_y_1d(self) -> None:
        """Output y is 1-D for 1-D input."""
        x = np.arange(1.0, 51.0)
        y, _lags, _scales = sdiff(x, np.array([2]), np.array([1]))
        assert y.ndim == 1

    def test_sdiff_lags_1d(self) -> None:
        """Lags array is always 1-D."""
        x = np.arange(1.0, 51.0)
        _y, lags, _scales = sdiff(x, np.array([2]), np.array([1]))
        assert lags.ndim == 1

    def test_sdiff_scales_1d(self) -> None:
        """Scales array is always 1-D."""
        x = np.arange(1.0, 51.0)
        _y, _lags, scales = sdiff(x, np.array([2]), np.array([1]))
        assert scales.ndim == 1

    def test_sdiff_lags_scales_same_length(self) -> None:
        """Lags and scales arrays have the same length."""
        for d, S in [
            (np.array([1]), np.array([1])),
            (np.array([2]), np.array([1])),
            (np.array([1]), np.array([4])),
            (np.array([1, 1]), np.array([1, 12])),
        ]:
            x = np.arange(1.0, 101.0)
            _y, lags, scales = sdiff(x, d, S)
            assert len(lags) == len(scales), (
                f"len(lags)={len(lags)} != len(scales)={len(scales)} "
                f"for d={d}, S={S}"
            )


class TestSdiffFiniteValues:
    """Ensure all output values are finite (no NaN/Inf)."""

    @pytest.mark.parametrize(
        "d, S",
        [
            (np.array([1]), np.array([1])),
            (np.array([2]), np.array([1])),
            (np.array([1]), np.array([4])),
            (np.array([1]), np.array([12])),
            (np.array([2]), np.array([12])),
            (np.array([3]), np.array([1])),
            (np.array([1, 1]), np.array([1, 12])),
        ],
    )
    def test_sdiff_y_finite(
        self, d: np.ndarray, S: np.ndarray, rng: np.random.Generator
    ) -> None:
        x = rng.standard_normal(200)
        y, lags, scales = sdiff(x, d, S)
        assert np.all(np.isfinite(y)), "y contains non-finite values"
        assert np.all(np.isfinite(lags)), "lags contains non-finite values"
        assert np.all(np.isfinite(scales)), "scales contains non-finite values"


class TestSdiffScalesSum:
    """Verify that the full polynomial (constant + returned scales) sums to zero.

    For any seasonal differencing operator ``prod(1 - B^{S[i]})^{d[i]}``,
    evaluating the polynomial at ``B = 1`` yields zero.  The returned ``scales``
    array excludes the constant term (coefficient +1 at lag 0).  Therefore:

        sum(scales) + 1 == 0    ⟹    sum(scales) == -1
    """

    @pytest.mark.parametrize(
        "d, S",
        [
            (np.array([1]), np.array([1])),
            (np.array([2]), np.array([1])),
            (np.array([3]), np.array([1])),
            (np.array([1]), np.array([4])),
            (np.array([1]), np.array([12])),
            (np.array([2]), np.array([12])),
            (np.array([1, 1]), np.array([1, 12])),
            (np.array([2, 1]), np.array([4, 12])),
        ],
    )
    def test_sdiff_scales_sum(self, d: np.ndarray, S: np.ndarray) -> None:
        x = np.arange(1.0, 101.0)
        _y, _lags, scales = sdiff(x, d, S)
        # Full polynomial sums to 0; constant term (coeff 1) is excluded
        # from returned scales, so sum(scales) == -1.
        npt.assert_allclose(
            np.sum(scales) + 1.0,
            0.0,
            atol=1e-12,
            err_msg=(
                f"Full polynomial should sum to zero. "
                f"sum(scales)+1 = {np.sum(scales) + 1.0} for d={d}, S={S}"
            ),
        )

    def test_sdiff_scales_sum_matches_polynomial(self) -> None:
        """Cross-verify: sum of independently computed full polynomial is zero."""
        for d, S in [
            (np.array([1]), np.array([1])),
            (np.array([2]), np.array([1])),
            (np.array([1, 1]), np.array([1, 12])),
        ]:
            poly = _compute_full_polynomial(d, S)
            npt.assert_allclose(np.sum(poly), 0.0, atol=1e-12)


# =========================================================================
# Additional Regression / Docstring Tests
# =========================================================================


class TestSdiffDocstringExample:
    """Verify the docstring example in ``sdiff.py``."""

    def test_docstring_example(self) -> None:
        """Reproduce the docstring: sdiff(arange(1,11), [2], [3])."""
        x = np.arange(1.0, 11.0)
        y, lags, scales = sdiff(x, np.array([2]), np.array([3]))

        # Docstring claims:
        #   lags = array([3, 6])
        #   scales = array([-2., 1.])
        npt.assert_array_equal(lags, np.array([3, 6]))
        npt.assert_allclose(scales, np.array([-2.0, 1.0]), atol=1e-12)

        # Computed y: x[6:] + 1*x[0:4] = [7,8,9,10] + [1,2,3,4] = [8,10,12,14]
        npt.assert_allclose(y, np.array([8.0, 10.0, 12.0, 14.0]), atol=1e-12)


class TestSdiffWithConftest:
    """Tests using fixtures provided by the sandbox conftest."""

    def test_sdiff_with_sdiff_spec_fixture(self, sdiff_spec: dict) -> None:
        """Use the ``sdiff_spec`` fixture from ``test_sandbox/conftest.py``."""
        x = sdiff_spec["x"]
        d = sdiff_spec["d"]
        S = sdiff_spec["S"]

        y, lags, scales = sdiff(x, d, S)

        # d=[1,1], S=[1,12] → (1-B)(1-B^12)
        npt.assert_array_equal(lags, np.array([1, 12, 13]))
        npt.assert_allclose(scales, np.array([-1.0, -1.0, 1.0]), atol=1e-12)

        # Verify output length
        assert len(y) == len(x) - int(np.max(lags))

        # All output values should be finite
        assert np.all(np.isfinite(y))
