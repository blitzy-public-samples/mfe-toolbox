"""
Pytest test suite for mfe_toolbox.distributions.iscompatible — shape compatibility checker.

Tests the foundational ``iscompatible`` utility used by nearly all distribution
functions (GED, standardized-t, skewed-t, etc.) to validate parameter shapes
against requested output dimensions and to broadcast scalar parameters to
a common output size.

Covers:
- 1-parameter compatibility (GED use case: v only)
- 2-parameter compatibility (skewed-t use case: v, lambda)
- Size specification variants (tuple, separate args, missing)
- Error handling (too few params, empty params, size mismatch)
- Broadcasting output verification (scalar expansion, shape matching)
- Integration scenarios mimicking real distribution function calls

Source reference: distributions/iscompatible.m (MFE Toolbox v4.0, Kevin Sheppard)
"""

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.distributions.iscompatible import iscompatible

# Import tolerance constants from conftest (auto-discovered by pytest)
from tests.conftest import ATOL, RTOL


# ---------------------------------------------------------------------------
# Phase 1: Basic Compatibility Tests — Single Parameter (narg=1)
# ---------------------------------------------------------------------------


class TestIscompatibleSingleParam:
    """Tests for the single-parameter case (narg=1), typical for GED distribution."""

    def test_scalar_param_with_explicit_size(self):
        """Scalar parameter v=5.0 with explicit size (100, 1) should broadcast.

        Ref: iscompatible.m:56-58 — all scalars → param_size = [1 1].
        Ref: iscompatible.m:132-133 — all scalar params + explicit sizeOut → use sizeOut.
        """
        result = iscompatible(1, 5.0, 100, 1)
        error, errortext, size_out = result[0], result[1], result[2]
        p1 = result[3]

        assert error == 0, f"Expected no error, got error=1: {errortext}"
        assert errortext == ''
        assert size_out == (100, 1)
        assert p1.shape == (100, 1)
        npt.assert_array_equal(p1, np.full((100, 1), 5.0))

    def test_vector_param_matching_size(self):
        """2D column vector param with matching size specification should succeed.

        Ref: iscompatible.m:60-62 — one non-scalar → its size becomes param_size.
        Ref: iscompatible.m:117-119 — param_size == sizeOut → error=0.
        """
        v = np.ones((100, 1))
        result = iscompatible(1, v, 100, 1)
        error, errortext, size_out = result[0], result[1], result[2]
        p1 = result[3]

        assert error == 0
        assert errortext == ''
        assert size_out == (100, 1)
        assert p1.shape == (100, 1)
        npt.assert_array_equal(p1, v)

    def test_1d_vector_param_matching_1d_size(self):
        """1D vector param with matching 1D size should succeed.

        In Python, np.ones(100) has shape (100,), unlike MATLAB's (100, 1).
        """
        v = np.ones(100)
        # Pass size as a single array [100] (1D shape spec)
        result = iscompatible(1, v, np.array([100]))
        error, errortext, size_out = result[0], result[1], result[2]
        p1 = result[3]

        assert error == 0
        assert errortext == ''
        assert size_out == (100,)
        assert p1.shape == (100,)

    def test_scalar_param_no_size(self):
        """Scalar parameter with no explicit size → sizeOut inferred as (1, 1).

        Ref: iscompatible.m:56-58 — all scalars → param_size = [1 1].
        Ref: iscompatible.m:127-130 — no sizeOut provided → use param_size.
        """
        result = iscompatible(1, 5.0)
        error, errortext, size_out = result[0], result[1], result[2]
        p1 = result[3]

        assert error == 0
        assert errortext == ''
        assert size_out == (1, 1)
        assert p1.shape == (1, 1)
        npt.assert_array_equal(p1, np.full((1, 1), 5.0))

    def test_vector_param_no_size(self):
        """Vector parameter with no explicit size → sizeOut inferred from vector shape.

        Ref: iscompatible.m:60-62 — one non-scalar → param_size = size(param).
        Ref: iscompatible.m:127-130 — no sizeOut → use param_size.
        """
        v = np.array([1.0, 2.0, 3.0])
        result = iscompatible(1, v)
        error, errortext, size_out = result[0], result[1], result[2]
        p1 = result[3]

        assert error == 0
        assert errortext == ''
        assert size_out == (3,)
        assert p1.shape == (3,)
        npt.assert_array_equal(p1, v)

    def test_2d_matrix_param_no_size(self):
        """2D matrix parameter with no size spec → sizeOut inferred from matrix shape."""
        v = np.ones((5, 3))
        result = iscompatible(1, v)
        error, errortext, size_out = result[0], result[1], result[2]
        p1 = result[3]

        assert error == 0
        assert errortext == ''
        assert size_out == (5, 3)
        assert p1.shape == (5, 3)


# ---------------------------------------------------------------------------
# Phase 2: Two-Parameter Compatibility Tests (narg=2)
# ---------------------------------------------------------------------------


class TestIscompatibleTwoParams:
    """Tests for the two-parameter case (narg=2), typical for skewed-t (v, lambda)."""

    def test_two_scalars_with_size(self):
        """Two scalar parameters with explicit size should both broadcast.

        Ref: iscompatible.m:56-58 — all scalars → param_size = [1 1].
        Ref: iscompatible.m:132-133 — all scalar + sizeOut → use sizeOut.
        Ref: iscompatible.m:138-141 — repmat each scalar to sizeOut.
        """
        result = iscompatible(2, 5.0, -0.2, 100, 1)
        error, errortext, size_out = result[0], result[1], result[2]
        p1, p2 = result[3], result[4]

        assert error == 0
        assert errortext == ''
        assert size_out == (100, 1)
        assert p1.shape == (100, 1)
        assert p2.shape == (100, 1)
        npt.assert_array_equal(p1, np.full((100, 1), 5.0))
        npt.assert_array_equal(p2, np.full((100, 1), -0.2))

    def test_two_vectors_matching(self):
        """Two vectors of matching shape with explicit matching size.

        Ref: iscompatible.m:64-81 — multiple non-scalar, same size → OK.
        """
        v = 5.0 * np.ones((100, 1))
        lam = -0.2 * np.ones((100, 1))
        result = iscompatible(2, v, lam, 100, 1)
        error, errortext, size_out = result[0], result[1], result[2]
        p1, p2 = result[3], result[4]

        assert error == 0
        assert errortext == ''
        assert size_out == (100, 1)
        npt.assert_array_equal(p1, v)
        npt.assert_array_equal(p2, lam)

    def test_mixed_scalar_vector(self):
        """One scalar, one vector: scalar should broadcast to match vector shape.

        Ref: iscompatible.m:60-62 — one non-scalar → param_size from non-scalar.
        Ref: iscompatible.m:139-140 — scalar → repmat.
        """
        lam_array = np.array([-0.1, -0.2, -0.3])
        result = iscompatible(2, 5.0, lam_array)
        error, errortext, size_out = result[0], result[1], result[2]
        p1, p2 = result[3], result[4]

        assert error == 0
        assert errortext == ''
        assert size_out == (3,)
        assert p1.shape == (3,)
        assert p2.shape == (3,)
        npt.assert_array_equal(p1, np.full(3, 5.0))
        npt.assert_array_equal(p2, lam_array)

    def test_two_vectors_mismatch(self):
        """Two non-scalar parameters with different sizes → error.

        Ref: iscompatible.m:69-77 — different non-scalar sizes → error.
        """
        v = np.ones(100)
        lam = np.ones(50)
        result = iscompatible(2, v, lam)
        error, errortext, size_out = result[0], result[1], result[2]

        assert error == 1
        assert 'size mismatch' in errortext.lower() or 'common size' in errortext.lower()
        assert size_out == ()

    def test_two_scalars_no_size(self):
        """Two scalar parameters, no size specification → sizeOut = (1, 1)."""
        result = iscompatible(2, 5.0, -0.2)
        error, errortext, size_out = result[0], result[1], result[2]
        p1, p2 = result[3], result[4]

        assert error == 0
        assert errortext == ''
        assert size_out == (1, 1)
        assert p1.shape == (1, 1)
        assert p2.shape == (1, 1)

    def test_two_vectors_no_size(self):
        """Two matching vectors, no size → sizeOut inferred from vector shape."""
        v = 5.0 * np.ones(10)
        lam = -0.2 * np.ones(10)
        result = iscompatible(2, v, lam)
        error, errortext, size_out = result[0], result[1], result[2]

        assert error == 0
        assert size_out == (10,)


# ---------------------------------------------------------------------------
# Phase 3: Size Specification Variants
# ---------------------------------------------------------------------------


class TestIscompatibleSizeSpec:
    """Tests for different ways of specifying the requested output size."""

    def test_size_as_array_vector(self):
        """Size specified as a single numpy array vector [100, 1].

        Ref: iscompatible.m:88-91 — single vector sizeOut arg.
        """
        result = iscompatible(1, 5.0, np.array([100, 1]))
        error, errortext, size_out = result[0], result[1], result[2]
        p1 = result[3]

        assert error == 0
        assert size_out == (100, 1)
        assert p1.shape == (100, 1)

    def test_size_as_list(self):
        """Size specified as a Python list [100, 1]."""
        result = iscompatible(1, 5.0, [100, 1])
        error, errortext, size_out = result[0], result[1], result[2]
        p1 = result[3]

        assert error == 0
        assert size_out == (100, 1)
        assert p1.shape == (100, 1)

    def test_size_as_separate_scalar_args(self):
        """Size specified as separate scalar arguments: 100, 1.

        Ref: iscompatible.m:99-107 — multiple scalar args → cell2mat.
        """
        result = iscompatible(1, 5.0, 100, 1)
        error, errortext, size_out = result[0], result[1], result[2]
        p1 = result[3]

        assert error == 0
        assert size_out == (100, 1)
        assert p1.shape == (100, 1)

    def test_size_as_single_scalar(self):
        """Size specified as a single scalar dimension: 50.

        This corresponds to a 1D output size of (50,).
        """
        result = iscompatible(1, 5.0, 50)
        error, errortext, size_out = result[0], result[1], result[2]
        p1 = result[3]

        assert error == 0
        assert size_out == (50,)
        assert p1.shape == (50,)

    def test_size_mismatch_with_param(self):
        """Non-scalar param shape (50, 1) with size (100, 1) → error.

        Ref: iscompatible.m:116-126 — param_size != sizeOut → error.
        """
        v = np.ones((50, 1))
        result = iscompatible(1, v, 100, 1)
        error, errortext, size_out = result[0], result[1], result[2]

        assert error == 1
        assert 'compatible sizes' in errortext.lower() or 'not of compatible' in errortext.lower()
        assert size_out == ()

    def test_size_3d(self):
        """Size specified as 3D: (5, 3, 2) for a scalar parameter."""
        result = iscompatible(1, 5.0, 5, 3, 2)
        error, errortext, size_out = result[0], result[1], result[2]
        p1 = result[3]

        assert error == 0
        assert size_out == (5, 3, 2)
        assert p1.shape == (5, 3, 2)
        npt.assert_array_equal(p1, np.full((5, 3, 2), 5.0))


# ---------------------------------------------------------------------------
# Phase 4: Error Handling Tests
# ---------------------------------------------------------------------------


class TestIscompatibleErrors:
    """Tests for error conditions and error reporting."""

    def test_too_few_params(self):
        """Calling with fewer args than narg → error=1.

        Ref: iscompatible.m:43-48 — length(varargin) < narg → error.
        """
        result = iscompatible(2, 5.0)  # narg=2 but only 1 param provided
        error, errortext, size_out = result[0], result[1], result[2]

        assert error == 1
        assert 'too few' in errortext.lower()
        assert size_out == ()

    def test_no_args_with_positive_narg(self):
        """narg=1 but zero additional args → error."""
        result = iscompatible(1)
        error, errortext, size_out = result[0], result[1], result[2]

        assert error == 1
        assert 'too few' in errortext.lower()

    def test_none_param_triggers_error(self):
        """None parameter should be treated as empty → error.

        Ref: iscompatible.m:43 — cellfun('isempty', varargin) checks all inputs.
        """
        result = iscompatible(1, None)
        error, errortext, size_out = result[0], result[1], result[2]

        assert error == 1
        assert 'nonempty' in errortext.lower() or 'too few' in errortext.lower()

    def test_empty_array_param_triggers_error(self):
        """Empty array as parameter → error.

        Ref: iscompatible.m:43 — cellfun('isempty', ...) catches empty arrays.
        """
        result = iscompatible(1, np.array([]))
        error, errortext, size_out = result[0], result[1], result[2]

        assert error == 1
        assert size_out == ()

    def test_returns_error_flag_as_int(self):
        """Error flag should be an integer (0 or 1)."""
        # Success case
        result_ok = iscompatible(1, 5.0, 10, 1)
        assert isinstance(result_ok[0], (int, np.integer))
        assert result_ok[0] in (0, 1)

        # Error case
        result_err = iscompatible(2, 5.0)
        assert isinstance(result_err[0], (int, np.integer))
        assert result_err[0] in (0, 1)

    def test_error_text_nonempty_on_error(self):
        """When incompatible, errortext must be a non-empty string."""
        result = iscompatible(2, np.ones(10), np.ones(5))
        assert result[0] == 1
        assert isinstance(result[1], str)
        assert len(result[1]) > 0

    def test_error_text_empty_on_success(self):
        """When compatible, errortext must be an empty string."""
        result = iscompatible(1, 5.0, 10, 1)
        assert result[0] == 0
        assert isinstance(result[1], str)
        assert result[1] == ''

    def test_param_size_mismatch_two_nonscalar(self):
        """Two non-scalar params with different shapes → error.

        Ref: iscompatible.m:72-77 — ~isequal(size(...), size(...)) → error.
        """
        v = np.ones((10, 2))
        lam = np.ones((10, 3))
        result = iscompatible(2, v, lam)
        error, errortext, size_out = result[0], result[1], result[2]

        assert error == 1
        assert 'size mismatch' in errortext.lower() or 'common size' in errortext.lower()

    def test_error_returns_correct_tuple_length(self):
        """On error, the return tuple should still have 3 + narg elements."""
        # narg=2, error case
        result = iscompatible(2, np.ones(10), np.ones(5))
        # Should be: (error, errortext, size_out, p1_empty, p2_empty)
        assert len(result) == 5  # 3 + narg(2)

    def test_empty_size_arg_triggers_error(self):
        """Empty array as a size argument → error.

        Ref: iscompatible.m:43 — checks ALL varargin for emptiness.
        """
        result = iscompatible(1, 5.0, np.array([]))
        error = result[0]
        assert error == 1


# ---------------------------------------------------------------------------
# Phase 5: Broadcasting Output Verification
# ---------------------------------------------------------------------------


class TestIscompatibleBroadcasting:
    """Tests verifying that scalar parameters are correctly broadcast."""

    def test_scalar_broadcast_to_2d(self):
        """Scalar parameter is tiled to requested (10, 5) shape.

        Ref: iscompatible.m:139-140 — repmat(scalar, sizeOut).
        """
        result = iscompatible(1, 3.14, 10, 5)
        p1 = result[3]

        assert p1.shape == (10, 5)
        npt.assert_array_equal(p1, np.full((10, 5), 3.14))

    def test_scalar_broadcast_to_1d(self):
        """Scalar parameter broadcast to 1D shape (50,)."""
        result = iscompatible(1, 7.0, np.array([50]))
        p1 = result[3]

        assert p1.shape == (50,)
        npt.assert_array_equal(p1, np.full(50, 7.0))

    def test_output_shapes_all_match_sizeout(self):
        """All broadcast outputs must have shape equal to sizeOut."""
        v_scalar = 5.0
        lam_scalar = -0.2
        result = iscompatible(2, v_scalar, lam_scalar, 20, 3)
        size_out = result[2]
        p1, p2 = result[3], result[4]

        assert p1.shape == size_out
        assert p2.shape == size_out

    def test_nonscalar_param_not_modified(self):
        """Non-scalar parameter values should be preserved in output."""
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = iscompatible(1, v, np.array([5]))
        p1 = result[3]

        npt.assert_array_equal(p1, v)

    def test_no_modify_original_array(self):
        """Input arrays should not be modified in-place by iscompatible."""
        v = np.array([1.0, 2.0, 3.0])
        v_copy = v.copy()
        lam = np.array([0.1, 0.2, 0.3])
        lam_copy = lam.copy()

        _ = iscompatible(2, v, lam)

        npt.assert_array_equal(v, v_copy)
        npt.assert_array_equal(lam, lam_copy)

    def test_mixed_broadcast_two_params(self):
        """With two params where one is scalar and one is vector,
        the scalar should be broadcast and the vector preserved."""
        vec = np.array([10.0, 20.0, 30.0])
        result = iscompatible(2, 99.0, vec)
        error = result[0]
        p1, p2 = result[3], result[4]

        assert error == 0
        assert p1.shape == (3,)
        assert p2.shape == (3,)
        npt.assert_array_equal(p1, np.full(3, 99.0))
        npt.assert_array_equal(p2, vec)

    def test_broadcast_output_is_writable(self):
        """Broadcast output arrays should be writable (not read-only views)."""
        result = iscompatible(1, 5.0, 10)
        p1 = result[3]

        # Should not raise; read-only arrays would throw ValueError
        p1[0] = 999.0
        assert p1[0] == 999.0


# ---------------------------------------------------------------------------
# Phase 6: Integration Scenarios with Distribution Functions
# ---------------------------------------------------------------------------


class TestIscompatibleIntegration:
    """Integration tests mimicking how real distribution functions call iscompatible.

    These tests simulate the actual calling patterns used by:
    - GED functions: iscompatible(1, v, size(x)) — 1 parameter (shape v)
    - Skewed-t functions: iscompatible(2, v, lambda, size(x)) — 2 parameters
    """

    def test_gedcdf_scenario(self):
        """Simulate gedcdf's call pattern: iscompatible(1, v, *x.shape).

        gedcdf.m calls: iscompatible(1, v, size(x))
        where v=1.5 is a scalar shape param and x is (100, 1).
        """
        v = 1.5
        x = np.random.default_rng(42).standard_normal((100, 1))
        # Simulate: iscompatible(1, v, size(x)[0], size(x)[1])
        result = iscompatible(1, v, x.shape[0], x.shape[1])
        error, errortext, size_out = result[0], result[1], result[2]
        p1 = result[3]

        assert error == 0
        assert errortext == ''
        assert size_out == (100, 1)
        assert p1.shape == (100, 1)
        npt.assert_array_equal(p1, np.full((100, 1), 1.5))

    def test_skewtpdf_scenario(self):
        """Simulate skewtpdf's call pattern: iscompatible(2, v, lambda, *x.shape).

        skewtpdf.m calls: iscompatible(2, v, lambda, size(x))
        where v=5, lambda=-0.2, and x is (100, 1).
        """
        v = 5.0
        lam = -0.2
        x = np.random.default_rng(42).standard_normal((100, 1))
        # Simulate: iscompatible(2, v, lambda, size(x)[0], size(x)[1])
        result = iscompatible(2, v, lam, x.shape[0], x.shape[1])
        error, errortext, size_out = result[0], result[1], result[2]
        p1, p2 = result[3], result[4]

        assert error == 0
        assert errortext == ''
        assert size_out == (100, 1)
        assert p1.shape == (100, 1)
        assert p2.shape == (100, 1)
        npt.assert_array_equal(p1, np.full((100, 1), 5.0))
        npt.assert_array_equal(p2, np.full((100, 1), -0.2))

    def test_gedcdf_vector_v_scenario(self):
        """GED with vector v (same size as x) — no broadcasting needed.

        Some usage patterns pass per-observation shape parameters.
        """
        rng = np.random.default_rng(42)
        x = rng.standard_normal((50, 1))
        v = np.full((50, 1), 1.5)
        # iscompatible(1, v, *x.shape)
        result = iscompatible(1, v, x.shape[0], x.shape[1])
        error, errortext, size_out = result[0], result[1], result[2]
        p1 = result[3]

        assert error == 0
        assert size_out == (50, 1)
        npt.assert_array_equal(p1, v)

    def test_skewtpdf_mixed_scenario(self):
        """Skewed-t with scalar v and vector lambda (or vice versa).

        This is an unusual but valid calling pattern.
        """
        lam_array = np.full((30, 1), -0.2)
        result = iscompatible(2, 5.0, lam_array, 30, 1)
        error, errortext, size_out = result[0], result[1], result[2]
        p1, p2 = result[3], result[4]

        assert error == 0
        assert size_out == (30, 1)
        assert p1.shape == (30, 1)
        npt.assert_array_equal(p1, np.full((30, 1), 5.0))
        npt.assert_array_equal(p2, lam_array)

    def test_skewtpdf_vector_mismatch_scenario(self):
        """Skewed-t with incompatible v and lambda vector sizes → error.

        This would indicate a programming error in the calling code.
        """
        v_arr = np.ones((100, 1))
        lam_arr = np.ones((50, 1))
        result = iscompatible(2, v_arr, lam_arr, 100, 1)
        error = result[0]

        assert error == 1


# ---------------------------------------------------------------------------
# Phase 7: Parametrized Tests for Systematic Coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("narg,args,expected_error,expected_size", [
    # Success cases: scalar params
    (1, (5.0, 10), 0, (10,)),
    (1, (5.0, 10, 1), 0, (10, 1)),
    (1, (5.0,), 0, (1, 1)),
    (2, (5.0, -0.2, 10, 1), 0, (10, 1)),
    (2, (5.0, -0.2), 0, (1, 1)),
    # Success cases: array params
    (1, (np.ones(10),), 0, (10,)),
    (1, (np.ones((5, 3)),), 0, (5, 3)),
    # Error cases
    (2, (5.0,), 1, ()),  # too few params
    (1, (np.array([]),), 1, ()),  # empty param
])
def test_iscompatible_parametrized(narg, args, expected_error, expected_size):
    """Parametrized test covering multiple success and error scenarios."""
    result = iscompatible(narg, *args)
    error, errortext, size_out = result[0], result[1], result[2]

    assert error == expected_error
    assert size_out == expected_size
    if expected_error == 0:
        assert errortext == ''
    else:
        assert isinstance(errortext, str)


@pytest.mark.parametrize("shape", [
    (1,),
    (10,),
    (10, 1),
    (5, 3),
    (2, 3, 4),
    (100, 1),
])
def test_iscompatible_scalar_broadcast_shapes(shape):
    """Verify scalar broadcasting produces correct shape for various output sizes."""
    size_args = list(shape)
    result = iscompatible(1, 42.0, *size_args)
    error, errortext, size_out = result[0], result[1], result[2]
    p1 = result[3]

    assert error == 0
    assert size_out == shape
    assert p1.shape == shape
    npt.assert_array_equal(p1, np.full(shape, 42.0))


# ---------------------------------------------------------------------------
# Phase 8: Return Value Structure Tests
# ---------------------------------------------------------------------------


class TestIscompatibleReturnStructure:
    """Tests verifying the structure and types of return values."""

    def test_return_tuple_length_narg1(self):
        """With narg=1, return should have 4 elements: (error, text, size, p1)."""
        result = iscompatible(1, 5.0, 10, 1)
        assert len(result) == 4

    def test_return_tuple_length_narg2(self):
        """With narg=2, return should have 5 elements: (error, text, size, p1, p2)."""
        result = iscompatible(2, 5.0, -0.2, 10, 1)
        assert len(result) == 5

    def test_return_tuple_length_narg3(self):
        """With narg=3, return should have 6 elements."""
        result = iscompatible(3, 1.0, 2.0, 3.0, 10, 1)
        assert len(result) == 6

    def test_error_return_tuple_length_preserved(self):
        """Even on error, return tuple length should be 3 + narg."""
        result = iscompatible(2, np.ones(5), np.ones(10))
        assert len(result) == 5  # 3 + 2

    def test_size_out_is_tuple(self):
        """size_out should always be a tuple of ints on success."""
        result = iscompatible(1, 5.0, 10, 1)
        size_out = result[2]
        assert isinstance(size_out, tuple)
        assert all(isinstance(s, int) for s in size_out)

    def test_broadcast_params_are_ndarray(self):
        """All broadcast parameter outputs should be numpy ndarrays."""
        result = iscompatible(2, 5.0, -0.2, 10, 1)
        p1, p2 = result[3], result[4]
        assert isinstance(p1, np.ndarray)
        assert isinstance(p2, np.ndarray)

    def test_error_flag_is_zero_on_success(self):
        """Successful calls must return error=0."""
        result = iscompatible(1, 5.0, 10, 1)
        assert result[0] == 0

    def test_error_flag_is_one_on_failure(self):
        """Failed calls must return error=1."""
        result = iscompatible(2, np.ones(5), np.ones(10))
        assert result[0] == 1


# ---------------------------------------------------------------------------
# Phase 9: Edge Cases and Numerical Stability
# ---------------------------------------------------------------------------


class TestIscompatibleEdgeCases:
    """Edge case tests for boundary conditions and special values."""

    def test_zero_narg(self):
        """narg=0 with size spec only → should work (no params to check).

        With zero params, param_size defaults to (1,1) and size_out is used.
        """
        result = iscompatible(0, 10, 1)
        error, errortext, size_out = result[0], result[1], result[2]

        assert error == 0
        assert size_out == (10, 1)
        # No broadcast params expected (narg=0)
        assert len(result) == 3

    def test_integer_param(self):
        """Integer parameter should work identically to float."""
        result = iscompatible(1, 5, 10, 1)
        error = result[0]
        p1 = result[3]

        assert error == 0
        assert p1.shape == (10, 1)

    def test_large_size_spec(self):
        """Large size specification (1000, 1) should work without issue."""
        result = iscompatible(1, 3.14, 1000, 1)
        error = result[0]
        p1 = result[3]

        assert error == 0
        assert p1.shape == (1000, 1)

    def test_single_element_array(self):
        """Single-element array param should behave like a scalar.

        Ref: iscompatible.m:54 — length of single-element array is 1.
        """
        v = np.array([5.0])
        result = iscompatible(1, v, 10, 1)
        error = result[0]
        p1 = result[3]

        assert error == 0
        assert p1.shape == (10, 1)
        npt.assert_array_equal(p1, np.full((10, 1), 5.0))

    def test_negative_value_param(self):
        """Negative parameter values are valid (e.g., lambda in skewed-t)."""
        result = iscompatible(1, -0.5, 10, 1)
        error = result[0]
        p1 = result[3]

        assert error == 0
        npt.assert_array_equal(p1, np.full((10, 1), -0.5))

    def test_zero_value_param(self):
        """Zero as parameter value is valid."""
        result = iscompatible(1, 0.0, 10, 1)
        error = result[0]
        p1 = result[3]

        assert error == 0
        npt.assert_array_equal(p1, np.zeros((10, 1)))

    def test_nan_value_param(self):
        """NaN as parameter value should not trigger an error
        (NaN is not empty — it's a valid float)."""
        result = iscompatible(1, float('nan'), 10, 1)
        error = result[0]
        p1 = result[3]

        assert error == 0
        assert p1.shape == (10, 1)
        assert np.all(np.isnan(p1))

    def test_inf_value_param(self):
        """Infinity as parameter value should not trigger an error."""
        result = iscompatible(1, float('inf'), 10, 1)
        error = result[0]
        p1 = result[3]

        assert error == 0
        assert p1.shape == (10, 1)
        assert np.all(np.isinf(p1))

    def test_multiple_same_nonscalar_params(self):
        """Three non-scalar params, all same shape → success."""
        a = np.ones((10, 3))
        b = 2.0 * np.ones((10, 3))
        c = 3.0 * np.ones((10, 3))
        result = iscompatible(3, a, b, c)
        error = result[0]
        size_out = result[2]

        assert error == 0
        assert size_out == (10, 3)

    def test_size_spec_with_size_one_dimensions(self):
        """Size spec with 1-size dimensions: (1, 1)."""
        result = iscompatible(1, 5.0, 1, 1)
        error = result[0]
        size_out = result[2]
        p1 = result[3]

        assert error == 0
        assert size_out == (1, 1)
        assert p1.shape == (1, 1)

    def test_size_as_2d_row_vector(self):
        """Size specified as a 2D row vector np.array([[10, 5]]).

        Ref: iscompatible.m:90 — ndims==2 && length==numel (row vector case).
        Covers line 217-220 of iscompatible.py (2D row/col vector size spec).
        """
        result = iscompatible(1, 5.0, np.array([[10, 5]]))
        error, errortext, size_out = result[0], result[1], result[2]
        p1 = result[3]

        assert error == 0
        assert errortext == ''
        assert size_out == (10, 5)
        assert p1.shape == (10, 5)

    def test_size_as_2d_column_vector(self):
        """Size specified as a 2D column vector np.array([[10], [5]]).

        Ref: iscompatible.m:90 — ndims==2 && length==numel (column vector case).
        Covers line 217-220 of iscompatible.py.
        """
        result = iscompatible(1, 5.0, np.array([[10], [5]]))
        error, errortext, size_out = result[0], result[1], result[2]
        p1 = result[3]

        assert error == 0
        assert errortext == ''
        assert size_out == (10, 5)
        assert p1.shape == (10, 5)

    def test_size_as_2d_matrix_triggers_error(self):
        """Size specified as a 2D matrix (not a vector) → error.

        Ref: iscompatible.m:93-97 — invalid size spec.
        Covers lines 221-226 of iscompatible.py (matrix size spec → error).
        """
        result = iscompatible(1, 5.0, np.array([[10, 5], [3, 4]]))
        error, errortext, size_out = result[0], result[1], result[2]

        assert error == 1
        assert 'cannot be parsed' in errortext.lower()
        assert size_out == ()

    def test_non_scalar_in_separate_size_args_triggers_error(self):
        """When separate size args contain a non-scalar → error.

        Ref: iscompatible.m:99-107 — all size args must be scalar.
        Covers line 234-238 of iscompatible.py (non-scalar in multi-size args).
        """
        # Pass a non-scalar array as one of the separate size args
        result = iscompatible(1, 5.0, 10, np.array([1, 2]))
        error, errortext, size_out = result[0], result[1], result[2]

        assert error == 1
        assert 'cannot be parsed' in errortext.lower()
        assert size_out == ()
