"""Pytest test suite for mfe_toolbox.utility.demean.

Comprehensive tests for column-wise demeaning of a T×K data matrix.
Covers basic functionality, edge cases, shape preservation, dtype
handling, large matrices, and MATLAB fixture-based numerical parity.

Source Reference
----------------
utility/demean.m (24 lines, Revision 3, 1/1/2010, Kevin Sheppard)

Key MATLAB behaviour verified:
- ``mu = mean(y, 1);``  — column-wise mean
- ``x  = bsxfun(@minus, y, mu);``  — broadcast subtraction
- Output has the same shape as input; each column mean ≈ 0

Numerical parity target: ``numpy.testing.assert_allclose(atol=1e-6, rtol=1e-4)``
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.utility.demean import demean
from tests.conftest import ATOL, RTOL, load_fixture_npy

# ---------------------------------------------------------------------------
# Tolerance constants — per AAP Section 0.7.1
# ---------------------------------------------------------------------------
# Re-exported from conftest for clarity; used in all parity assertions.
_ATOL: float = ATOL  # 1e-6
_RTOL: float = RTOL  # 1e-4


# ---------------------------------------------------------------------------
# Fixture path resolution
# ---------------------------------------------------------------------------
_FIXTURE_DIR: Path = (
    Path(os.environ.get("MFE_FIXTURE_DIR", ""))
    if os.environ.get("MFE_FIXTURE_DIR")
    else Path(__file__).resolve().parent.parent / "fixtures" / "utility"
)


def _load_demean_fixture() -> dict:
    """Load the demean fixture dictionary from ``demean.npy``.

    The fixture is a 0-d object ndarray wrapping a dict with keys like
    ``input_10x3``, ``output_10x3``, ``means_10x3``, etc.  Returns the
    inner dict.  If the file is absent the calling test is gracefully
    skipped via ``pytest.skip``.
    """
    arr = load_fixture_npy(_FIXTURE_DIR, "demean")
    # The fixture is stored as a 0-d object array wrapping a dict
    return arr.item() if arr.shape == () else arr


# ===================================================================
# 1. test_demean_basic
# ===================================================================


def test_demean_basic() -> None:
    """Known matrix → demeaned result.  Each column mean should be ≈ 0.

    Creates a simple 3×2 matrix, demeans it, and verifies that each
    column mean of the result is approximately zero.
    """
    data = np.array([[10.0, 20.0],
                     [20.0, 40.0],
                     [30.0, 60.0]])
    result = demean(data)

    # Column means of demeaned data should be effectively zero
    col_means = np.mean(result, axis=0)
    npt.assert_allclose(col_means, 0.0, atol=1e-12,
                        err_msg="Column means after demeaning are not zero")

    # Verify shape is preserved
    assert result.shape == data.shape


# ===================================================================
# 2. test_demean_column_means_zero
# ===================================================================


def test_demean_column_means_zero() -> None:
    """After demeaning, column means should be approximately zero.

    Uses a random 50×4 matrix to ensure generality beyond the basic test.
    """
    rng = np.random.default_rng(12345)
    data = rng.standard_normal((50, 4)) * 100 + 500  # non-zero mean cols
    result = demean(data)

    col_means = np.mean(result, axis=0)
    npt.assert_allclose(
        col_means, np.zeros(4), atol=1e-12,
        err_msg="Column means must be ≈ 0 after demeaning",
    )


# ===================================================================
# 3. test_demean_single_column
# ===================================================================


def test_demean_single_column() -> None:
    """T×1 vector → demeaned vector.

    A single-column 2-D array should be demeaned by its scalar mean.
    """
    data = np.array([[2.0],
                     [4.0],
                     [6.0],
                     [8.0]])
    # Mean = 5.0; expected = [-3, -1, 1, 3]
    expected = np.array([[-3.0],
                         [-1.0],
                         [1.0],
                         [3.0]])
    result = demean(data)
    npt.assert_allclose(result, expected, atol=1e-12)


# ===================================================================
# 4. test_demean_single_row
# ===================================================================


def test_demean_single_row() -> None:
    """1×K → all zeros (mean of a single value equals itself).

    When there is only one row, the column means equal the data values,
    so demeaning produces a row of zeros.
    """
    data = np.array([[3.0, 7.0, -2.0, 100.0]])
    result = demean(data)

    expected = np.zeros((1, 4))
    npt.assert_allclose(result, expected, atol=1e-14,
                        err_msg="Single-row demean must produce all zeros")


# ===================================================================
# 5. test_demean_already_zero_mean
# ===================================================================


def test_demean_already_zero_mean() -> None:
    """Data with zero column means → unchanged after demeaning.

    Constructs a matrix where each column already has mean zero and
    verifies the output is identical (within floating-point tolerance).
    """
    # Manually constructed zero-mean columns
    data = np.array([[-1.0, -2.0],
                     [0.0,  0.0],
                     [1.0,  2.0]])
    # Column means are already [0, 0]
    assert np.allclose(np.mean(data, axis=0), 0.0, atol=1e-14)

    result = demean(data)
    npt.assert_allclose(result, data, atol=1e-14,
                        err_msg="Zero-mean data should be unchanged")


# ===================================================================
# 6. test_demean_output_shape
# ===================================================================


@pytest.mark.parametrize("shape", [
    (10, 1),
    (1, 10),
    (5, 5),
    (100, 3),
    (3, 100),
    (1, 1),
])
def test_demean_output_shape(shape: tuple[int, int]) -> None:
    """Output shape must match input shape for various T×K dimensions."""
    rng = np.random.default_rng(9999)
    data = rng.standard_normal(shape)
    result = demean(data)
    assert result.shape == shape, (
        f"Expected shape {shape}, got {result.shape}"
    )


# ===================================================================
# 7. test_demean_preserves_dtype
# ===================================================================


def test_demean_preserves_dtype() -> None:
    """Float64 in → float64 out.

    The ``demean`` implementation converts inputs to float64 via
    ``np.asarray(y, dtype=np.float64)``, so the output must always be
    float64 regardless of input dtype.
    """
    # float64 input
    data_f64 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    result_f64 = demean(data_f64)
    assert result_f64.dtype == np.float64, (
        f"Expected float64 output, got {result_f64.dtype}"
    )

    # float32 input — should still produce float64 output
    data_f32 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    result_f32 = demean(data_f32)
    assert result_f32.dtype == np.float64, (
        f"Expected float64 output from float32 input, got {result_f32.dtype}"
    )

    # integer input — should produce float64 output
    data_int = np.array([[1, 2], [3, 4]], dtype=np.int64)
    result_int = demean(data_int)
    assert result_int.dtype == np.float64, (
        f"Expected float64 output from int64 input, got {result_int.dtype}"
    )


# ===================================================================
# 8. test_demean_known_values
# ===================================================================


def test_demean_known_values() -> None:
    """Specific small matrix with hand-computed expected values.

    Input:  [[1, 2], [3, 4]]
    Column means: [2, 3]
    Expected: [[-1, -1], [1, 1]]
    """
    data = np.array([[1.0, 2.0],
                     [3.0, 4.0]])
    expected = np.array([[-1.0, -1.0],
                         [1.0, 1.0]])
    result = demean(data)
    npt.assert_allclose(result, expected, atol=1e-14,
                        err_msg="Known-value demeaning failed")

    # Additional hand-computed case:
    # Input:  [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    # Column means: [4, 5, 6]
    # Expected: [[-3, -3, -3], [0, 0, 0], [3, 3, 3]]
    data2 = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0]])
    expected2 = np.array([[-3.0, -3.0, -3.0],
                          [0.0, 0.0, 0.0],
                          [3.0, 3.0, 3.0]])
    result2 = demean(data2)
    npt.assert_allclose(result2, expected2, atol=1e-14,
                        err_msg="Known-value demeaning (3×3) failed")


# ===================================================================
# 9. test_demean_large_matrix
# ===================================================================


def test_demean_large_matrix() -> None:
    """T=1000, K=5 random data — verify column means are zero and shape is preserved.

    Uses a fixed seed for reproducibility.  Exercises the function on a
    realistically-sized financial data matrix.
    """
    rng = np.random.default_rng(42)
    T, K = 1000, 5
    data = rng.standard_normal((T, K)) * 10 + np.array([50, -20, 0, 100, -50])

    result = demean(data)

    # Shape must be preserved
    assert result.shape == (T, K)

    # Column means must be ≈ 0
    col_means = np.mean(result, axis=0)
    npt.assert_allclose(col_means, np.zeros(K), atol=1e-12,
                        err_msg="Large matrix column means not zero after demeaning")

    # Verify that the demeaned data equals data minus column means
    manual_demeaned = data - np.mean(data, axis=0)
    npt.assert_allclose(result, manual_demeaned, atol=1e-14,
                        err_msg="Demean result differs from manual computation")


# ===================================================================
# 10. test_demean_fixture_parity
# ===================================================================


@pytest.mark.parity
class TestDemeanFixtureParity:
    """MATLAB fixture comparison tests.

    These tests load Octave-generated reference inputs and outputs from
    ``tests/fixtures/utility/demean.npy`` and compare the Python
    ``demean()`` results against the MATLAB/Octave reference values to
    enforce strict numerical parity (±1e-6 absolute, ±1e-4 relative).
    """

    def test_fixture_10x3(self) -> None:
        """Parity test: 10×3 matrix."""
        fixture = _load_demean_fixture()
        inp = fixture["input_10x3"]
        expected_out = fixture["output_10x3"]
        result = demean(inp)
        npt.assert_allclose(result, expected_out, atol=_ATOL, rtol=_RTOL,
                            err_msg="10×3 fixture parity failed")

    def test_fixture_5x1(self) -> None:
        """Parity test: 5×1 single-column vector."""
        fixture = _load_demean_fixture()
        inp = fixture["input_5x1"]
        expected_out = fixture["output_5x1"]
        result = demean(inp)
        npt.assert_allclose(result, expected_out, atol=_ATOL, rtol=_RTOL,
                            err_msg="5×1 fixture parity failed")

    def test_fixture_8x1(self) -> None:
        """Parity test: 8×1 single-column vector."""
        fixture = _load_demean_fixture()
        inp = fixture["input_8x1"]
        expected_out = fixture["output_8x1"]
        result = demean(inp)
        npt.assert_allclose(result, expected_out, atol=_ATOL, rtol=_RTOL,
                            err_msg="8×1 fixture parity failed")

    def test_fixture_5x4(self) -> None:
        """Parity test: 5×4 matrix."""
        fixture = _load_demean_fixture()
        inp = fixture["input_5x4"]
        expected_out = fixture["output_5x4"]
        result = demean(inp)
        npt.assert_allclose(result, expected_out, atol=_ATOL, rtol=_RTOL,
                            err_msg="5×4 fixture parity failed")

    def test_fixture_1000x3(self) -> None:
        """Parity test: 1000×3 large matrix."""
        fixture = _load_demean_fixture()
        inp = fixture["input_1000x3"]
        expected_out = fixture["output_1000x3"]
        result = demean(inp)
        npt.assert_allclose(result, expected_out, atol=_ATOL, rtol=_RTOL,
                            err_msg="1000×3 fixture parity failed")

    def test_fixture_1x5(self) -> None:
        """Parity test: 1×5 single-row vector (should yield all zeros)."""
        fixture = _load_demean_fixture()
        inp = fixture["input_1x5"]
        expected_out = fixture["output_1x5"]
        result = demean(inp)
        npt.assert_allclose(result, expected_out, atol=_ATOL, rtol=_RTOL,
                            err_msg="1×5 fixture parity failed")

    def test_fixture_means_match(self) -> None:
        """Verify that Python-computed column means match MATLAB fixture means.

        The fixture dict stores ``means_*`` arrays representing the column
        means computed by MATLAB's ``mean(y, 1)``.  This test ensures that
        NumPy's ``np.mean(y, axis=0)`` produces identical values.
        """
        fixture = _load_demean_fixture()
        for suffix in ("10x3", "5x1", "8x1", "5x4", "1000x3", "1x5"):
            inp = fixture[f"input_{suffix}"]
            expected_means = fixture[f"means_{suffix}"]
            # Ref: demean.m:21 — MATLAB mean(y, 1) ≡ np.mean(y, axis=0)
            computed_means = np.mean(inp, axis=0)
            npt.assert_allclose(
                computed_means, expected_means, atol=_ATOL, rtol=_RTOL,
                err_msg=f"Column means mismatch for {suffix} fixture",
            )

    def test_fixture_demean_out_npy(self) -> None:
        """Validate the separate ``demean_demean_out.npy`` fixture.

        The standalone file stores the MATLAB/Octave demeaned output from
        the global 1000×3 test data used by the fixture generation script.
        We verify that this output represents properly demeaned data:
        column means ≈ 0, shape is (1000, 3), and it matches its CSV
        counterpart.
        """
        demean_out_path = _FIXTURE_DIR / "demean_demean_out.npy"
        if not demean_out_path.exists():
            pytest.skip(f"Fixture file not found: {demean_out_path}")
        expected_out = np.load(demean_out_path)

        # Verify shape
        assert expected_out.shape == (1000, 3), (
            f"Expected shape (1000, 3), got {expected_out.shape}"
        )

        # Verify column means are ≈ 0 (hallmark of demeaned data)
        col_means = np.mean(expected_out, axis=0)
        npt.assert_allclose(col_means, np.zeros(3), atol=1e-12,
                            err_msg="demean_demean_out.npy column means not zero")

        # If the CSV counterpart exists, verify consistency
        demean_out_csv_path = _FIXTURE_DIR / "demean_demean_out.csv"
        if demean_out_csv_path.exists():
            csv_data = np.loadtxt(demean_out_csv_path, delimiter=",")
            npt.assert_allclose(expected_out, csv_data, atol=1e-14,
                                err_msg="demean_demean_out .npy and .csv disagree")

        # Cross-validate: applying demean to the output should be a no-op
        # since the data is already demeaned
        re_demeaned = demean(expected_out)
        npt.assert_allclose(re_demeaned, expected_out, atol=1e-10,
                            err_msg="Demeaning already-demeaned fixture data changed it")


# ===================================================================
# Additional edge case tests
# ===================================================================


def test_demean_1d_array() -> None:
    """1-D array input — should demean as a single column.

    The ``demean`` implementation supports 1-D inputs by using
    ``np.mean(y, axis=0)`` which computes the scalar mean for 1-D arrays.
    """
    data = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    # Mean = 6.0; expected = [-4, -2, 0, 2, 4]
    expected = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
    result = demean(data)
    npt.assert_allclose(result, expected, atol=1e-14)
    assert result.shape == data.shape


def test_demean_constant_columns() -> None:
    """Columns with constant values should produce zero columns."""
    data = np.array([[5.0, 3.0],
                     [5.0, 3.0],
                     [5.0, 3.0]])
    expected = np.zeros((3, 2))
    result = demean(data)
    npt.assert_allclose(result, expected, atol=1e-14)


def test_demean_list_input() -> None:
    """Python list input — demean should accept list-like inputs.

    The implementation uses ``np.asarray(y, dtype=np.float64)`` which
    handles list inputs gracefully.
    """
    data = [[1.0, 2.0], [3.0, 4.0]]
    result = demean(data)
    expected = np.array([[-1.0, -1.0], [1.0, 1.0]])
    npt.assert_allclose(result, expected, atol=1e-14)
    assert isinstance(result, np.ndarray)
