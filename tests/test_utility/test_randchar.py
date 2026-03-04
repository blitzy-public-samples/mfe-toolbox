"""Pytest test suite for mfe_toolbox.utility.randchar — random character string generator.

Tests verify statistical properties of the random output (not exact values),
including string length, character range (A-Z only), return type, distribution
coverage, edge cases (zero length, single character), and error handling
(negative input).

No MATLAB fixture parity is needed for this random function — tests focus
exclusively on output properties.

Source reference: utility/randchar.m (4 lines)
    u = rand(1,n); u = floor(u*26)+65; c = char(u);
"""

import string

import pytest

from mfe_toolbox.utility.randchar import randchar


# ---------------------------------------------------------------------------
# Test 1: test_randchar_length
# ---------------------------------------------------------------------------
def test_randchar_length() -> None:
    """randchar(10) returns a string of exactly length 10."""
    result = randchar(10)
    assert len(result) == 10, f"Expected length 10, got {len(result)}"


# ---------------------------------------------------------------------------
# Test 2: test_randchar_uppercase_only
# ---------------------------------------------------------------------------
def test_randchar_uppercase_only() -> None:
    """All characters produced by randchar are uppercase ASCII A-Z."""
    result = randchar(100)
    for ch in result:
        assert ch in string.ascii_uppercase, (
            f"Character '{ch}' is not in A-Z"
        )


# ---------------------------------------------------------------------------
# Test 3: test_randchar_single_char
# ---------------------------------------------------------------------------
def test_randchar_single_char() -> None:
    """randchar(1) returns a single uppercase character."""
    result = randchar(1)
    assert len(result) == 1, f"Expected length 1, got {len(result)}"
    assert result in string.ascii_uppercase, (
        f"Character '{result}' is not in A-Z"
    )


# ---------------------------------------------------------------------------
# Test 4: test_randchar_different_calls
# ---------------------------------------------------------------------------
def test_randchar_different_calls() -> None:
    """Two successive calls to randchar should (almost certainly) differ.

    With n=50, the probability that two independent random strings of 50
    uppercase characters are identical is (1/26)^50 ≈ 10^{-71}, making
    a false failure astronomically unlikely.
    """
    result_a = randchar(50)
    result_b = randchar(50)
    assert result_a != result_b, (
        "Two independent calls returned identical strings — "
        "extremely unlikely for n=50"
    )


# ---------------------------------------------------------------------------
# Test 5: test_randchar_zero_length
# ---------------------------------------------------------------------------
def test_randchar_zero_length() -> None:
    """randchar(0) returns an empty string."""
    result = randchar(0)
    assert result == "", f"Expected empty string, got '{result}'"
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Test 6: test_randchar_large_n
# ---------------------------------------------------------------------------
def test_randchar_large_n() -> None:
    """randchar(1000) produces a 1000-character string with all chars in A-Z."""
    result = randchar(1000)
    assert len(result) == 1000, f"Expected length 1000, got {len(result)}"
    # Verify every character is uppercase A-Z
    for ch in result:
        assert ch in string.ascii_uppercase, (
            f"Character '{ch}' is not in A-Z"
        )


# ---------------------------------------------------------------------------
# Test 7: test_randchar_returns_string
# ---------------------------------------------------------------------------
def test_randchar_returns_string() -> None:
    """randchar returns a Python str object."""
    result = randchar(5)
    assert isinstance(result, str), (
        f"Expected type str, got {type(result).__name__}"
    )


# ---------------------------------------------------------------------------
# Test 8: test_randchar_char_distribution
# ---------------------------------------------------------------------------
def test_randchar_char_distribution() -> None:
    """Over a large sample, all 26 uppercase letters should appear.

    With n=10_000, the expected count per letter is ~385.  The probability
    of *any* letter being absent follows a union bound with each individual
    miss probability ≈ (25/26)^10_000 ≈ 0, so this test is extremely
    reliable.
    """
    result = randchar(10_000)
    observed = set(result)
    missing = set(string.ascii_uppercase) - observed
    assert len(missing) == 0, (
        f"Letters missing from 10 000-char sample: {sorted(missing)}"
    )


# ---------------------------------------------------------------------------
# Test 9: test_randchar_negative_raises
# ---------------------------------------------------------------------------
def test_randchar_negative_raises() -> None:
    """randchar(-1) must raise ValueError.

    Ref: randchar.py — input validation rejects n < 0 with ValueError,
    matching the spirit of MATLAB's error() on invalid input.
    """
    with pytest.raises(ValueError):
        randchar(-1)
