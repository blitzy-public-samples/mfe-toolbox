"""Pytest test suite for mfe_toolbox.utility.standardize.

Comprehensive tests for column standardization (mean zero, unit variance)
of a T×K matrix.  Covers basic functionality, MATLAB numerical parity
(ddof=1), edge cases, demean modes, and fixture-based regression tests.

Source Reference
----------------
utility/standardize.m (45 lines, Revision 1, 9/1/2005, Kevin Sheppard)

Key MATLAB behaviors verified:
- ``std(x)`` uses ddof=1 (sample std with N-1 denominator)
- ``standardize(x)``  →  ``(x - mean(x)) ./ std(x)``   per column
- ``standardize(x, 0)``  →  ``x ./ std(x)``             per column
- Returns a T×K matrix of the same shape as the input

Numerical parity target: ``numpy.testing.assert_allclose(atol=1e-6, rtol=1e-4)``
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.utility.standardize import standardize

# ---------------------------------------------------------------------------
# Tolerance constants — per AAP Section 0.7.1
# ---------------------------------------------------------------------------
ATOL: float = 1e-6
RTOL: float = 1e-4


# ---------------------------------------------------------------------------
# Fixture path resolution
# ---------------------------------------------------------------------------
_FIXTURE_DIR: Path = (
    Path(os.environ.get("MFE_FIXTURE_DIR", ""))
    if os.environ.get("MFE_FIXTURE_DIR")
    else Path(__file__).resolve().parent.parent / "fixtures" / "utility"
)

_FIXTURE_NPY: Path = _FIXTURE_DIR / "standardize.npy"


# ===================================================================
# Test Suite
# ===================================================================


class TestStandardizeBasic:
    """Basic functionality tests for standardize."""

    def test_standardize_basic(self) -> None:
        """Standardize a known T×K matrix and verify output properties.

        Creates a small 5×3 matrix, standardizes it, and checks that
        the result is a numpy array with column means ≈ 0 and column
        stds (ddof=1) ≈ 1.
        """
        rng = np.random.default_rng(42)
        data = rng.standard_normal((5, 3)) * 10 + 5  # non-zero mean, non-unit var
        result = standardize(data)

        # Output must be a numpy ndarray
        assert isinstance(result, np.ndarray)

        # Column means should be ≈ 0
        col_means = np.mean(result, axis=0)
        npt.assert_allclose(col_means, np.zeros(3), atol=ATOL)

        # Ref: standardize.m:40 — MATLAB std(x) uses ddof=1 (N-1 denominator)
        col_stds = np.std(result, axis=0, ddof=1)
        npt.assert_allclose(col_stds, np.ones(3), atol=ATOL)

    def test_standardize_column_means_zero(self) -> None:
        """After standardization with demean=True, column means must be ≈ 0."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, 4))
        result = standardize(data)

        col_means = np.mean(result, axis=0)
        npt.assert_allclose(col_means, np.zeros(4), atol=ATOL)

    def test_standardize_column_std_one(self) -> None:
        """After standardization, column stds (ddof=1) must be ≈ 1.

        Ref: standardize.m:40 — MATLAB std(x) uses ddof=1 by default.
        """
        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, 4))
        result = standardize(data)

        # Ref: standardize.m:40 — MATLAB std(x) uses ddof=1 (N-1 denominator)
        col_stds = np.std(result, axis=0, ddof=1)
        npt.assert_allclose(col_stds, np.ones(4), atol=ATOL)

    def test_standardize_output_is_ndarray(self) -> None:
        """Return value must be a numpy ndarray."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = standardize(data)
        assert isinstance(result, np.ndarray)


class TestStandardizeKnownValues:
    """Tests with hand-computable expected values."""

    def test_standardize_known_values(self) -> None:
        """Verify standardize against hand-computed expectations.

        Input:  [[1,2], [3,4], [5,6]]
        Mean:   [3, 4]
        Std(ddof=1): [2, 2]
        Expected: [[-1,-1], [0,0], [1,1]]

        Ref: standardize.m:42 — st=(x-repmat(mu,T,1))./repmat(stdev,T,1)
        """
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        expected = np.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])

        result = standardize(data)
        npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)

    def test_standardize_known_values_means(self) -> None:
        """Verify that column means of the known example are correct."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        expected_mean = np.array([3.0, 4.0])

        # Ref: standardize.m:39 — mu = mean(x);
        actual_mean = np.mean(data, axis=0)
        npt.assert_allclose(actual_mean, expected_mean, atol=ATOL)

    def test_standardize_known_values_stds(self) -> None:
        """Verify that column stds (ddof=1) of the known example are correct."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        expected_std = np.array([2.0, 2.0])

        # Ref: standardize.m:40 — stdev = std(x); uses ddof=1
        actual_std = np.std(data, axis=0, ddof=1)
        npt.assert_allclose(actual_std, expected_std, atol=ATOL)

    def test_standardize_known_single_column(self) -> None:
        """Hand-computable single-column example.

        Input: [2, 4, 6]  →  mean=4, std(ddof=1)=2  →  [-1, 0, 1]
        """
        data = np.array([[2.0], [4.0], [6.0]])
        expected = np.array([[-1.0], [0.0], [1.0]])

        result = standardize(data)
        npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)


class TestStandardizeSingleColumn:
    """Tests for single-column (T×1) and 1-D vector inputs."""

    def test_standardize_single_column_2d(self) -> None:
        """Standardize a T×1 column vector (2-D array)."""
        data = np.array([[10.0], [20.0], [30.0], [40.0], [50.0]])
        result = standardize(data)

        # Ref: standardize.m:38 — [T, K] = size(x); preserves shape
        assert result.shape == data.shape

        # Column mean ≈ 0
        npt.assert_allclose(np.mean(result, axis=0), np.zeros(1), atol=ATOL)

        # Column std(ddof=1) ≈ 1
        npt.assert_allclose(np.std(result, axis=0, ddof=1), np.ones(1), atol=ATOL)

    def test_standardize_1d_input(self) -> None:
        """Standardize a 1-D array — should be promoted to column vector.

        Ref: standardize.py:74 — 1-D input reshaped to (-1, 1).
        """
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = standardize(data)

        # 1-D input is promoted to (T, 1)
        assert result.shape == (5, 1)

        # Column mean ≈ 0
        npt.assert_allclose(np.mean(result, axis=0), np.zeros(1), atol=ATOL)

        # Column std(ddof=1) ≈ 1
        npt.assert_allclose(np.std(result, axis=0, ddof=1), np.ones(1), atol=ATOL)


class TestStandardizeNoDemean:
    """Tests for the demean=False mode."""

    def test_standardize_no_demean(self) -> None:
        """With demean=False, only divide by std — do not subtract mean.

        Ref: standardize.m:44 — st = x ./ repmat(stdev,T,1);
        """
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        # Ref: standardize.m:40 — stdev = std(x); uses ddof=1
        sigma = np.std(data, axis=0, ddof=1)
        expected = data / sigma

        result = standardize(data, demean=False)
        npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)

    def test_standardize_no_demean_means_not_zero(self) -> None:
        """With demean=False, column means of result should NOT be zero
        (unless original data was already mean-zero).
        """
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = standardize(data, demean=False)

        col_means = np.mean(result, axis=0)
        # Means are data_mean / std(data), which is NOT zero for this data
        assert not np.allclose(col_means, 0.0, atol=1e-10)

    def test_standardize_no_demean_preserves_scaling(self) -> None:
        """With demean=False, result columns should have std(ddof=1) ≈ 1
        only if the original data was already mean-zero.

        For non-mean-zero data, std of result = std(data/sigma) which
        equals 1 since dividing by std is a linear scaling.
        """
        rng = np.random.default_rng(99)
        data = rng.standard_normal((50, 3))
        result = standardize(data, demean=False)

        # Dividing each column by its std gives std(result, ddof=1) = 1
        result_stds = np.std(result, axis=0, ddof=1)
        npt.assert_allclose(result_stds, np.ones(3), atol=ATOL)

    def test_standardize_demean_int_zero(self) -> None:
        """demean=0 (integer) should behave like demean=False.

        Ref: standardize.py:88 — accepts int 0/1 for convenience.
        """
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        result_false = standardize(data, demean=False)
        result_int0 = standardize(data, demean=0)
        npt.assert_allclose(result_int0, result_false, atol=ATOL)

    def test_standardize_demean_int_one(self) -> None:
        """demean=1 (integer) should behave like demean=True.

        Ref: standardize.py:88 — accepts int 0/1 for convenience.
        """
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        result_true = standardize(data, demean=True)
        result_int1 = standardize(data, demean=1)
        npt.assert_allclose(result_int1, result_true, atol=ATOL)


class TestStandardizeDdof1:
    """CRITICAL: Verify MATLAB parity through ddof=1 usage."""

    def test_standardize_ddof1_parity(self) -> None:
        """CRITICAL TEST for MATLAB parity.

        MATLAB std(x) uses ddof=1 (N-1 denominator), which differs from
        numpy's default ddof=0 (N denominator).  For small T, the difference
        is measurable.

        Ref: standardize.m:40 — stdev = std(x); MATLAB std uses N-1
        """
        # Small T=3 to maximise the ddof gap
        data = np.array([[1.0, 10.0], [4.0, 20.0], [7.0, 30.0]])

        # Compute stds both ways
        expected_ddof0 = np.std(data, axis=0, ddof=0)
        expected_ddof1 = np.std(data, axis=0, ddof=1)

        # Confirm they are different for this small data
        assert not np.allclose(expected_ddof0, expected_ddof1, atol=1e-10), (
            "ddof=0 and ddof=1 must differ for T=3"
        )

        # Standardize and verify it used ddof=1
        result = standardize(data)

        # If ddof=1 was used, column stds of result should be exactly 1
        result_stds_ddof1 = np.std(result, axis=0, ddof=1)
        npt.assert_allclose(result_stds_ddof1, np.ones(2), atol=ATOL)

        # If ddof=0 was used instead, the result stds(ddof=1) would NOT be 1
        # This verifies the implementation is using ddof=1 (MATLAB parity)

    def test_standardize_ddof1_explicit_computation(self) -> None:
        """Verify the standardization formula explicitly with ddof=1.

        Expected: (data - mean) / std(ddof=1)
        """
        data = np.array([[2.0], [4.0], [9.0]])
        mu = np.mean(data, axis=0)          # 5.0
        sigma = np.std(data, axis=0, ddof=1)  # sqrt(13) ≈ 3.60555

        expected = (data - mu) / sigma
        result = standardize(data)

        npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)

    def test_standardize_ddof1_not_ddof0(self) -> None:
        """Verify result is NOT consistent with ddof=0 normalization."""
        data = np.array([[1.0], [3.0], [5.0]])

        # ddof=0: std = sqrt(8/3) ≈ 1.6330
        # ddof=1: std = sqrt(4) = 2.0
        sigma_ddof0 = np.std(data, axis=0, ddof=0)
        sigma_ddof1 = np.std(data, axis=0, ddof=1)

        # Compute what the result would be under each convention
        expected_ddof0 = (data - np.mean(data, axis=0)) / sigma_ddof0
        expected_ddof1 = (data - np.mean(data, axis=0)) / sigma_ddof1

        result = standardize(data)

        # Result must match ddof=1 expectation
        npt.assert_allclose(result, expected_ddof1, atol=ATOL, rtol=RTOL)

        # Result must NOT match ddof=0 expectation
        assert not np.allclose(result, expected_ddof0, atol=1e-10), (
            "standardize must use ddof=1 (MATLAB parity), not ddof=0"
        )


class TestStandardizeOutputShape:
    """Verify output shapes for various input shapes."""

    @pytest.mark.parametrize(
        "shape",
        [(10, 1), (10, 5), (100, 3), (50, 10), (3, 2)],
        ids=["10x1", "10x5", "100x3", "50x10", "3x2"],
    )
    def test_standardize_output_shape(self, shape: tuple[int, int]) -> None:
        """Output shape must match input shape for all valid 2-D inputs."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(shape)
        result = standardize(data)
        assert result.shape == data.shape

    def test_standardize_1d_output_shape(self) -> None:
        """1-D input is promoted to (T, 1) — output shape should match.

        Ref: standardize.py:74 — 1-D input reshaped to (-1, 1).
        """
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = standardize(data)
        assert result.shape == (5, 1)


class TestStandardizeLargeRandom:
    """Tests with large random data for statistical verification."""

    def test_standardize_large_random(self) -> None:
        """Standardize T=1000, K=5 random data — verify statistical properties."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((1000, 5))
        result = standardize(data)

        # Output shape must match
        assert result.shape == data.shape

        # Column means ≈ 0
        col_means = np.mean(result, axis=0)
        npt.assert_allclose(col_means, np.zeros(5), atol=ATOL)

        # Ref: standardize.m:40 — MATLAB std(x) uses ddof=1
        col_stds = np.std(result, axis=0, ddof=1)
        npt.assert_allclose(col_stds, np.ones(5), atol=ATOL)

    def test_standardize_large_random_no_demean(self) -> None:
        """Large random data with demean=False — stds should still be 1."""
        rng = np.random.default_rng(123)
        data = rng.standard_normal((500, 3))
        result = standardize(data, demean=False)

        assert result.shape == data.shape

        # Column stds(ddof=1) must still be ≈ 1
        col_stds = np.std(result, axis=0, ddof=1)
        npt.assert_allclose(col_stds, np.ones(3), atol=ATOL)

    def test_standardize_large_scaled_data(self) -> None:
        """Standardize data with large mean and variance offsets."""
        rng = np.random.default_rng(42)
        # Data with large mean offset and non-unit variance
        data = rng.standard_normal((200, 4)) * 100 + 5000
        result = standardize(data)

        col_means = np.mean(result, axis=0)
        npt.assert_allclose(col_means, np.zeros(4), atol=ATOL)

        col_stds = np.std(result, axis=0, ddof=1)
        npt.assert_allclose(col_stds, np.ones(4), atol=ATOL)


class TestStandardizeFixtureParity:
    """MATLAB/Octave fixture-based numerical parity tests."""

    @pytest.mark.skipif(
        not _FIXTURE_NPY.exists(),
        reason=f"Fixture file not found: {_FIXTURE_NPY}",
    )
    def test_standardize_fixture_parity_demean_true(self) -> None:
        """Verify numerical parity with MATLAB output (demean=True).

        Loads precomputed MATLAB/Octave fixture from
        tests/fixtures/utility/standardize.npy and compares Python output
        using assert_allclose(atol=1e-6, rtol=1e-4).

        Ref: standardize.m:42 — st=(x-repmat(mu,T,1))./repmat(stdev,T,1)
        """
        fixture = np.load(_FIXTURE_NPY, allow_pickle=True).item()
        input_data: np.ndarray = fixture["input_data"]
        expected: np.ndarray = fixture["result_demean_true"]

        result = standardize(input_data, demean=True)

        npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL,
                            err_msg="MATLAB parity failure: standardize(x, demean=True)")

    @pytest.mark.skipif(
        not _FIXTURE_NPY.exists(),
        reason=f"Fixture file not found: {_FIXTURE_NPY}",
    )
    def test_standardize_fixture_parity_demean_false(self) -> None:
        """Verify numerical parity with MATLAB output (demean=False).

        Ref: standardize.m:44 — st = x ./ repmat(stdev,T,1);
        """
        fixture = np.load(_FIXTURE_NPY, allow_pickle=True).item()
        input_data: np.ndarray = fixture["input_data"]
        expected: np.ndarray = fixture["result_demean_false"]

        result = standardize(input_data, demean=False)

        npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL,
                            err_msg="MATLAB parity failure: standardize(x, demean=False)")

    @pytest.mark.skipif(
        not _FIXTURE_NPY.exists(),
        reason=f"Fixture file not found: {_FIXTURE_NPY}",
    )
    def test_standardize_fixture_parity_column_stats(self) -> None:
        """Verify that standardized fixture data has correct column statistics."""
        fixture = np.load(_FIXTURE_NPY, allow_pickle=True).item()
        input_data: np.ndarray = fixture["input_data"]

        result = standardize(input_data, demean=True)

        # After demeaning + scaling, columns must be mean ≈ 0, std(ddof=1) ≈ 1
        col_means = np.mean(result, axis=0)
        npt.assert_allclose(col_means, np.zeros(input_data.shape[1]), atol=ATOL)

        col_stds = np.std(result, axis=0, ddof=1)
        npt.assert_allclose(col_stds, np.ones(input_data.shape[1]), atol=ATOL)


class TestStandardizeEdgeCases:
    """Edge case and error handling tests."""

    def test_standardize_invalid_demean_string(self) -> None:
        """Non-boolean/non-scalar demean must raise ValueError.

        Ref: standardize.m:34-36 — MATLAB checks ndims(demean)~=2 || max(size(demean))~=1
        """
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="DEMEAN must be a logical scalar"):
            standardize(data, demean="yes")

    def test_standardize_invalid_demean_float(self) -> None:
        """Float demean value must raise ValueError."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="DEMEAN must be a logical scalar"):
            standardize(data, demean=2.5)

    def test_standardize_invalid_demean_list(self) -> None:
        """List demean value must raise ValueError.

        Ref: standardize.m:34 — max(size(demean))~=1 rejects non-scalar.
        """
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="DEMEAN must be a logical scalar"):
            standardize(data, demean=[True])

    def test_standardize_invalid_demean_int_out_of_range(self) -> None:
        """Integer demean outside {0,1} must raise ValueError."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="DEMEAN must be a logical scalar"):
            standardize(data, demean=2)

    def test_standardize_3d_array(self) -> None:
        """3-D array input must raise ValueError.

        Ref: standardize.py:76-80 — rejects arrays with ndim != 2.
        """
        data = np.ones((3, 4, 5))
        with pytest.raises(ValueError, match="2-D matrix"):
            standardize(data)

    def test_standardize_single_row(self) -> None:
        """T=1 edge case — std(ddof=1) is 0, produces NaN/Inf.

        With T=1, ddof=1 gives std = 0, so division produces NaN.
        This mirrors MATLAB behaviour: std of a scalar row → 0.
        """
        data = np.array([[1.0, 2.0]])
        result = standardize(data)

        # T=1 → std(ddof=1) = 0 → division by zero → NaN
        assert np.all(np.isnan(result))

    def test_standardize_two_rows(self) -> None:
        """T=2 edge case — minimal valid case for ddof=1."""
        data = np.array([[1.0, 10.0], [3.0, 20.0]])
        result = standardize(data)

        # mean = [2, 15], std(ddof=1) = [sqrt(2), sqrt(50)]
        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0, ddof=1)
        expected = (data - mu) / sigma

        npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)

    def test_standardize_constant_column(self) -> None:
        """Column with zero variance — std(ddof=1)=0 → NaN/Inf.

        When a column is constant, std=0 and dividing by 0 produces
        NaN or Inf.  This mirrors MATLAB behaviour.
        """
        data = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
        result = standardize(data)

        # First column is constant → std=0 → NaN
        assert np.all(np.isnan(result[:, 0]) | np.isinf(result[:, 0]))

        # Second column is valid
        expected_col1 = (data[:, 1] - np.mean(data[:, 1])) / np.std(data[:, 1], ddof=1)
        npt.assert_allclose(result[:, 1], expected_col1, atol=ATOL)

    def test_standardize_list_input(self) -> None:
        """Plain Python list input should be converted to ndarray.

        Ref: standardize.py:64-70 — converts non-ndarray to float array.
        """
        data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        expected = np.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])

        result = standardize(data)
        npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)

    def test_standardize_boolean_demean_types(self) -> None:
        """Both Python bool and numpy.bool_ should be accepted for demean."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        # Python bool True
        result_py_true = standardize(data, demean=True)

        # numpy bool_
        result_np_true = standardize(data, demean=np.bool_(True))

        npt.assert_allclose(result_py_true, result_np_true, atol=ATOL)

    def test_standardize_non_convertible_input(self) -> None:
        """Input that cannot be converted to a float ndarray must raise ValueError.

        Ref: standardize.py:65-70 — catches TypeError/ValueError on asarray.
        """
        with pytest.raises(ValueError, match="numeric numpy ndarray"):
            standardize({"a": 1, "b": 2})


class TestStandardizeReturnStructure:
    """Tests verifying the return structure of standardize.

    Note: The Python implementation of standardize returns a single
    numpy.ndarray (matching MATLAB's single-output signature).
    """

    def test_standardize_returns_ndarray(self) -> None:
        """standardize must return a numpy.ndarray."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = standardize(data)
        assert isinstance(result, np.ndarray)

    def test_standardize_returns_float_dtype(self) -> None:
        """Output dtype must be floating-point."""
        data = np.array([[1, 2], [3, 4], [5, 6]], dtype=int)
        result = standardize(data)
        assert np.issubdtype(result.dtype, np.floating)

    def test_standardize_preserves_column_count(self) -> None:
        """Number of columns must be preserved."""
        for k in [1, 2, 5, 10]:
            rng = np.random.default_rng(42)
            data = rng.standard_normal((20, k))
            result = standardize(data)
            assert result.shape[1] == k

    def test_standardize_preserves_row_count(self) -> None:
        """Number of rows must be preserved."""
        for t in [2, 10, 50, 200]:
            rng = np.random.default_rng(42)
            data = rng.standard_normal((t, 3))
            result = standardize(data)
            assert result.shape[0] == t


class TestStandardizeReproducibility:
    """Tests verifying deterministic, reproducible results."""

    def test_standardize_deterministic(self) -> None:
        """Calling standardize twice on the same data gives identical results."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result1 = standardize(data)
        result2 = standardize(data)
        npt.assert_allclose(result1, result2, atol=0)

    def test_standardize_does_not_modify_input(self) -> None:
        """standardize must not modify the input array in-place."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        data_copy = data.copy()
        _ = standardize(data)
        npt.assert_allclose(data, data_copy, atol=0)
