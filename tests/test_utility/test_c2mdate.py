"""Pytest tests for ``mfe_toolbox.utility.c2mdate`` — CRSP date conversion.

Tests verify correct conversion of CRSP dates (YYYYMMDD integers) to Python
``datetime64`` / ``pandas.Timestamp`` equivalents, including:
- Single scalar and vector inputs
- Known MATLAB datenum reference values
- Roundtrip parity with ``m2cdate``
- Edge cases (Y2K, end-of-month, leap years)
- Error handling (non-numeric, invalid dates)
- Output type validation
- MATLAB fixture numerical parity

Source reference: ``utility/c2mdate.m`` (Version 4.0, Kevin Sheppard)

Per AAP Section 0.7.1: Date conversion tests may use exact equality
rather than atol/rtol for integer dates.
"""

import datetime
import os
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from mfe_toolbox.utility.c2mdate import c2mdate
from mfe_toolbox.utility.m2cdate import m2cdate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# MATLAB datenum for Unix epoch (1970-01-01 00:00:00).
# datenum(1970, 1, 1) == 719529 in MATLAB.
_MATLAB_DATENUM_UNIX_EPOCH: int = 719529


# ---------------------------------------------------------------------------
# Helper: Convert MATLAB datenum(s) to numpy datetime64 for comparison
# ---------------------------------------------------------------------------

def _datenum_to_datetime64(datenum_arr: np.ndarray) -> np.ndarray:
    """Convert MATLAB datenum values to numpy datetime64[us] array.

    Parameters
    ----------
    datenum_arr : np.ndarray
        Array of MATLAB serial date numbers (float).

    Returns
    -------
    np.ndarray
        Array of ``datetime64[us]`` values equivalent to the input datenums.
    """
    days_since_epoch = np.asarray(datenum_arr, dtype=np.float64) - _MATLAB_DATENUM_UNIX_EPOCH
    # Use pandas for reliable vectorized conversion, then extract as numpy
    dt_index = pd.to_datetime(days_since_epoch, unit="D")
    return dt_index.values


# ===========================================================================
# Test 1: test_c2mdate_single_date
# ===========================================================================

def test_c2mdate_single_date():
    """Verify c2mdate converts a single CRSP date (19951028) to Oct 28, 1995.

    Ref: c2mdate.m example — ``crspdate = 19951028`` → ``datestr(mldate)``
    produces ``'28-Oct-1995'``.
    """
    result = c2mdate(19951028)

    # Result should be a numpy array
    assert isinstance(result, np.ndarray), (
        f"Expected np.ndarray, got {type(result)}"
    )
    # Result should contain a datetime64 value
    assert np.issubdtype(result.dtype, np.datetime64), (
        f"Expected datetime64 dtype, got {result.dtype}"
    )

    # Build expected date as numpy datetime64 for comparison
    expected_dt = np.datetime64("1995-10-28")
    # Compare date portion (strip time component via casting)
    actual_date = result[0].astype("datetime64[D]")
    assert actual_date == expected_dt, (
        f"Expected 1995-10-28, got {actual_date}"
    )


# ===========================================================================
# Test 2: test_c2mdate_vector
# ===========================================================================

def test_c2mdate_vector():
    """Verify c2mdate handles a vector of CRSP dates correctly.

    Ref: c2mdate.m:14-19 — example with [19951028, 20090706, 20120401].
    """
    crsp_dates = np.array([19951028, 20090706, 20120401])
    result = c2mdate(crsp_dates)

    # Result should be an ndarray with 3 elements
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,), f"Expected shape (3,), got {result.shape}"
    assert np.issubdtype(result.dtype, np.datetime64)

    # Check each date
    expected_dates = np.array([
        np.datetime64("1995-10-28"),
        np.datetime64("2009-07-06"),
        np.datetime64("2012-04-01"),
    ])
    actual_dates = result.astype("datetime64[D]")
    npt.assert_array_equal(actual_dates, expected_dates)


# ===========================================================================
# Test 3: test_c2mdate_known_dates
# ===========================================================================

@pytest.mark.parametrize(
    "crsp_date, matlab_datenum, expected_date_str",
    [
        # Ref: c2mdate.m:14-19 — example values from source comments
        # datenum(1995,10,28) = 728960 in MATLAB
        (19951028, 728960, "1995-10-28"),
        # datenum(2009,7,6) = 733960 in MATLAB
        (20090706, 733960, "2009-07-06"),
        # datenum(2012,4,1) = 734960 in MATLAB
        (20120401, 734960, "2012-04-01"),
    ],
    ids=["oct_28_1995", "jul_06_2009", "apr_01_2012"],
)
def test_c2mdate_known_dates(crsp_date, matlab_datenum, expected_date_str):
    """Verify c2mdate produces the correct date for known MATLAB datenum values.

    The MATLAB datenum reference values are from the c2mdate.m source comments.
    MATLAB datenum 1 = Jan 1, 0000; datenum(1995,10,28) = 728960.

    The Python c2mdate returns datetime64 values.  We verify that:
    1. The output date matches the expected calendar date.
    2. Converting the MATLAB datenum independently yields the same date.
    """
    result = c2mdate(crsp_date)
    actual_date = result[0].astype("datetime64[D]")
    expected_date = np.datetime64(expected_date_str)

    # Verify Python output matches expected date
    assert actual_date == expected_date, (
        f"CRSP {crsp_date}: expected {expected_date_str}, got {actual_date}"
    )

    # Cross-verify: MATLAB datenum → datetime should produce the same date
    datenum_as_dt = _datenum_to_datetime64(np.array([matlab_datenum]))
    datenum_date = datenum_as_dt[0].astype("datetime64[D]")
    assert datenum_date == expected_date, (
        f"MATLAB datenum {matlab_datenum} should map to {expected_date_str}, "
        f"got {datenum_date}"
    )


# ===========================================================================
# Test 4: test_c2mdate_roundtrip_m2cdate
# ===========================================================================

def test_c2mdate_roundtrip_m2cdate():
    """Verify m2cdate(c2mdate(crspdate)) reproduces original CRSP dates exactly.

    The roundtrip path is:
    1. c2mdate(crsp_ints) → numpy datetime64 array
    2. Wrap in pd.DatetimeIndex for m2cdate input
    3. m2cdate(DatetimeIndex) → CRSP integer array
    4. Assert roundtrip integers match original

    Per AAP Section 0.7.1: date conversion tests may use exact equality.
    """
    crsp_dates = np.array([19951028, 20090706, 20120401, 20000101, 20201231])

    # Forward pass: CRSP → datetime64
    dt_result = c2mdate(crsp_dates)

    # Inverse pass: datetime64 → CRSP integers via m2cdate
    # m2cdate accepts pd.DatetimeIndex
    dt_index = pd.DatetimeIndex(dt_result)
    roundtrip = m2cdate(dt_index)

    # Roundtrip should reproduce original CRSP dates exactly
    npt.assert_array_equal(
        roundtrip, crsp_dates,
        err_msg="Roundtrip m2cdate(c2mdate(crsp)) failed to reproduce originals",
    )


# ===========================================================================
# Test 5: test_c2mdate_year2000
# ===========================================================================

def test_c2mdate_year2000():
    """Verify c2mdate correctly handles the Y2K date (20000101 → Jan 1, 2000).

    This tests potential issues with year-2000 date handling in the
    year/month/day integer arithmetic (yr = crsp // 10000 etc.).
    """
    result = c2mdate(20000101)
    actual_date = result[0].astype("datetime64[D]")
    expected = np.datetime64("2000-01-01")
    assert actual_date == expected, (
        f"Expected 2000-01-01, got {actual_date}"
    )


# ===========================================================================
# Test 6: test_c2mdate_end_of_month
# ===========================================================================

def test_c2mdate_end_of_month():
    """Verify c2mdate handles end-of-month and end-of-year dates correctly.

    Tests Dec 31, 2020 (end of year), Feb 29, 2020 (leap year), and
    Jan 31, 2021 (end of month) to validate day extraction logic.

    Ref: c2mdate.m:55 — ``dd = mod(crspdate, 100)`` for day extraction.
    """
    test_cases = [
        (20201231, "2020-12-31"),  # End of year
        (20200229, "2020-02-29"),  # Leap year Feb 29
        (20210131, "2021-01-31"),  # End of January
    ]
    for crsp_date, expected_str in test_cases:
        result = c2mdate(crsp_date)
        actual_date = result[0].astype("datetime64[D]")
        expected = np.datetime64(expected_str)
        assert actual_date == expected, (
            f"CRSP {crsp_date}: expected {expected_str}, got {actual_date}"
        )


# ===========================================================================
# Test 7: test_c2mdate_non_numeric_raises
# ===========================================================================

def test_c2mdate_non_numeric_raises():
    """Verify c2mdate raises ValueError for non-numeric (string) input.

    Ref: c2mdate.m:34-36 — ``if any(ischar(crspdate)) error('CRSPDATE must
    be numeric')``.  The Python implementation raises ValueError.
    """
    # String scalar
    with pytest.raises(ValueError, match="numeric"):
        c2mdate("19951028")

    # List of strings
    with pytest.raises(ValueError, match="numeric"):
        c2mdate(["19951028", "20090706"])

    # Numpy array of strings
    with pytest.raises(ValueError, match="numeric"):
        c2mdate(np.array(["19951028", "20090706"]))


# ===========================================================================
# Test 8: test_c2mdate_invalid_date_handling
# ===========================================================================

def test_c2mdate_invalid_date_handling():
    """Verify c2mdate handles invalid CRSP dates appropriately.

    Invalid date components (e.g. month=13, day=32) should produce an error
    rather than silently returning garbage dates.  The Python implementation
    delegates to ``pd.to_datetime`` which raises on invalid components.

    Additionally, dates < 18000000 should be rejected per c2mdate.m:46-48.
    """
    # Value below 18000000 threshold
    with pytest.raises(ValueError, match="incorrectly formatted|YYYYMMDD"):
        c2mdate(10000101)

    # Month 13 — pandas.to_datetime will raise
    with pytest.raises((ValueError, KeyError)):
        c2mdate(20201301)

    # Day 32 — pandas.to_datetime will raise
    with pytest.raises((ValueError, KeyError)):
        c2mdate(20200132)


# ===========================================================================
# Test 9: test_c2mdate_output_type
# ===========================================================================

def test_c2mdate_output_type():
    """Verify c2mdate returns the correct output type.

    Per AAP specification, c2mdate returns ``numpy.ndarray`` with
    ``datetime64`` dtype.  Scalar input should return a 1-element array
    (preserving consistent return shape per c2mdate.py docstring).
    """
    # Scalar input
    result_scalar = c2mdate(19951028)
    assert isinstance(result_scalar, np.ndarray), (
        f"Scalar input: expected np.ndarray, got {type(result_scalar)}"
    )
    assert np.issubdtype(result_scalar.dtype, np.datetime64), (
        f"Scalar input: expected datetime64 dtype, got {result_scalar.dtype}"
    )
    # Should be 1-D (1-element array)
    assert result_scalar.ndim == 1, (
        f"Scalar input: expected 1-D, got ndim={result_scalar.ndim}"
    )
    assert len(result_scalar) == 1, (
        f"Scalar input: expected length 1, got {len(result_scalar)}"
    )

    # Vector input
    result_vec = c2mdate(np.array([19951028, 20090706]))
    assert isinstance(result_vec, np.ndarray)
    assert np.issubdtype(result_vec.dtype, np.datetime64)
    assert result_vec.ndim == 1
    assert len(result_vec) == 2

    # Float input (YYYYMMDD as float, common from data loaders)
    result_float = c2mdate(19951028.0)
    assert isinstance(result_float, np.ndarray)
    assert np.issubdtype(result_float.dtype, np.datetime64)

    # The float result should match the integer result
    npt.assert_array_equal(
        result_float.astype("datetime64[D]"),
        result_scalar.astype("datetime64[D]"),
    )


# ===========================================================================
# Test 10: test_c2mdate_fixture_parity
# ===========================================================================

def test_c2mdate_fixture_parity(utility_fixture_dir):
    """Verify c2mdate output matches MATLAB-generated fixture data.

    Loads the fixture file ``tests/fixtures/utility/c2mdate.npy`` which
    contains a dict with keys:
    - ``'input'``: array of CRSP dates (YYYYMMDD floats)
    - ``'expected_output'``: array of MATLAB datenum values

    The Python c2mdate returns ``datetime64`` values, not MATLAB datenums.
    To compare, we convert both to the same date representation:
    - Python output → datetime64[D] (date only)
    - MATLAB datenum → datetime64[D] via Unix epoch offset

    Per AAP Section 0.7.1: exact date equality for date conversion tests.
    """
    fixture_path = utility_fixture_dir / "c2mdate.npy"
    if not fixture_path.exists():
        pytest.skip(f"Fixture file not found: {fixture_path}")

    fixture = np.load(fixture_path, allow_pickle=True).item()
    crsp_input = fixture["input"]
    expected_datenums = fixture["expected_output"]

    # Python c2mdate output: datetime64 array
    result = c2mdate(crsp_input)
    actual_dates = result.astype("datetime64[D]")

    # Convert expected MATLAB datenums to datetime64[D]
    expected_dates = _datenum_to_datetime64(expected_datenums).astype("datetime64[D]")

    # All dates should match exactly
    npt.assert_array_equal(
        actual_dates,
        expected_dates,
        err_msg=(
            "MATLAB fixture parity failure: c2mdate output dates "
            "do not match MATLAB datenum-derived dates"
        ),
    )
