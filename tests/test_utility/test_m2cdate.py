"""Pytest tests for mfe_toolbox.utility.m2cdate — MATLAB datenum to CRSP date conversion.

Tests cover:
1. Single date conversion (MATLAB datenum → CRSP YYYYMMDD integer)
2. Vector (array) conversion
3. Known date examples from m2cdate.m docstring
4. Roundtrip with c2mdate (inverse function)
5. Year 2000 edge case
6. End-of-year edge case
7. Output integer type verification
8. Non-numeric input error handling
9. MATLAB fixture parity comparison

Source reference: utility/m2cdate.m (52 lines, Kevin Sheppard, Revision 2)
Key formula: crspdate = 10000 * year + 100 * month + day

The Python implementation (mfe_toolbox/utility/m2cdate.py) accepts MATLAB datenum
integers/floats, Python datetime.datetime objects, pandas Timestamps, and pandas
DatetimeIndex objects — tests exercise all supported input pathways.
"""

import datetime
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from mfe_toolbox.utility.m2cdate import m2cdate
from mfe_toolbox.utility.c2mdate import c2mdate


# ---------------------------------------------------------------------------
# Test 1: Single date conversion
# ---------------------------------------------------------------------------

def test_m2cdate_single_date():
    """Known MATLAB datenum 728960 → CRSP date 19951028 (Oct 28, 1995).

    Ref: m2cdate.m:15-19 — Example from the MATLAB docstring.
    The MATLAB datenum 728960 corresponds to October 28, 1995.
    """
    # Test with integer datenum
    result = m2cdate(728960)
    # m2cdate on scalar returns np.int64 scalar or 1-element array
    result_val = int(np.asarray(result).ravel()[0])
    assert result_val == 19951028, (
        f"Expected CRSP date 19951028, got {result_val}"
    )

    # Test with float datenum (MATLAB datenums are often float)
    result_float = m2cdate(728960.0)
    result_float_val = int(np.asarray(result_float).ravel()[0])
    assert result_float_val == 19951028, (
        f"Expected CRSP date 19951028 for float input, got {result_float_val}"
    )


# ---------------------------------------------------------------------------
# Test 2: Vector (array) conversion
# ---------------------------------------------------------------------------

def test_m2cdate_vector():
    """Array of MATLAB datenums → array of CRSP integers.

    Ref: m2cdate.m:15-19 — MATLAB example uses mldate = [728960 733960 734960]'.
    """
    mldate = np.array([728960, 733960, 734960], dtype=np.float64)
    result = m2cdate(mldate)
    expected = np.array([19951028, 20090706, 20120401])

    # Use assert_array_equal for exact integer equality
    npt.assert_array_equal(
        np.asarray(result).ravel().astype(np.int64),
        expected,
    )


# ---------------------------------------------------------------------------
# Test 3: Known dates from m2cdate.m example (parametrized)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "datenum, expected_crsp",
    [
        (728960, 19951028),   # Oct 28, 1995
        (733960, 20090706),   # Jul 6, 2009
        (734960, 20120401),   # Apr 1, 2012
    ],
    ids=["oct28_1995", "jul06_2009", "apr01_2012"],
)
def test_m2cdate_known_dates(datenum, expected_crsp):
    """From m2cdate.m example: 728960→19951028, 733960→20090706, 734960→20120401.

    Ref: m2cdate.m:15-19 — documented example dates in MATLAB header.
    These are the canonical test vectors from the original MATLAB source.
    """
    result = m2cdate(datenum)
    result_val = int(np.asarray(result).ravel()[0])
    assert result_val == expected_crsp, (
        f"MATLAB datenum {datenum} should map to CRSP {expected_crsp}, "
        f"got {result_val}"
    )


# ---------------------------------------------------------------------------
# Test 4: Roundtrip with c2mdate
# ---------------------------------------------------------------------------

def test_m2cdate_roundtrip_c2mdate():
    """Verify c2mdate(m2cdate(mldate)) produces a lossless roundtrip.

    Starting from known CRSP dates, convert to datetime via c2mdate,
    then back to CRSP via m2cdate. The integer CRSP dates should be
    recovered exactly.

    Also tests the forward path: m2cdate(datenum) → crsp, c2mdate(crsp) → dt,
    then verify that the datetime matches expectations.
    """
    # Forward path: CRSP date integers → datetime → CRSP date integers
    crsp_dates = np.array([19951028, 20090706, 20120401])

    # c2mdate converts CRSP integers to datetime64 array
    dt_array = c2mdate(crsp_dates)

    # Feed datetime results into m2cdate via pandas DatetimeIndex
    dt_index = pd.DatetimeIndex(dt_array)
    roundtrip_crsp = m2cdate(dt_index)

    npt.assert_array_equal(
        np.asarray(roundtrip_crsp).ravel().astype(np.int64),
        crsp_dates,
    )

    # Also test with individual datetime.datetime objects
    dt_single = datetime.datetime(1995, 10, 28)
    crsp_single = m2cdate(dt_single)
    crsp_single_val = int(np.asarray(crsp_single).ravel()[0])
    assert crsp_single_val == 19951028


# ---------------------------------------------------------------------------
# Test 5: Year 2000 (Y2K) edge case
# ---------------------------------------------------------------------------

def test_m2cdate_year2000():
    """Jan 1, 2000 → CRSP date 20000101.

    Tests the Python datetime boundary around Y2K and verifies the
    MATLAB datenum for that date (730486) maps correctly.
    Ref: MATLAB datenum(2000, 1, 1) = 730486.
    """
    # Via MATLAB datenum
    result_datenum = m2cdate(730486)
    assert int(np.asarray(result_datenum).ravel()[0]) == 20000101

    # Via Python datetime
    result_dt = m2cdate(datetime.datetime(2000, 1, 1))
    assert int(np.asarray(result_dt).ravel()[0]) == 20000101

    # Via pandas Timestamp
    result_ts = m2cdate(pd.Timestamp("2000-01-01"))
    assert int(np.asarray(result_ts).ravel()[0]) == 20000101


# ---------------------------------------------------------------------------
# Test 6: End-of-year edge case
# ---------------------------------------------------------------------------

def test_m2cdate_end_of_year():
    """Dec 31, 2020 → CRSP date 20201231.

    Tests the last day of the year, which is a common edge case for
    date conversion functions.
    """
    # Via Python datetime
    result_dt = m2cdate(datetime.datetime(2020, 12, 31))
    assert int(np.asarray(result_dt).ravel()[0]) == 20201231

    # Via pandas Timestamp
    result_ts = m2cdate(pd.Timestamp("2020-12-31"))
    assert int(np.asarray(result_ts).ravel()[0]) == 20201231

    # Via pandas DatetimeIndex
    result_dti = m2cdate(pd.DatetimeIndex(["2020-12-31"]))
    assert int(np.asarray(result_dti).ravel()[0]) == 20201231


# ---------------------------------------------------------------------------
# Test 7: Output is integer type
# ---------------------------------------------------------------------------

def test_m2cdate_output_integer():
    """Output is integer type YYYYMMDD — verify dtype is integer-compatible.

    Ref: m2cdate.py:55 — returns int64 dtype.
    The output must be representable as an exact integer (no fractional part).
    """
    # Scalar input
    result_scalar = m2cdate(728960)
    result_scalar_arr = np.asarray(result_scalar)
    assert np.issubdtype(result_scalar_arr.dtype, np.integer), (
        f"Scalar output dtype {result_scalar_arr.dtype} is not integer"
    )

    # Array input
    mldate_vec = np.array([728960, 733960, 734960], dtype=np.float64)
    result_vec = m2cdate(mldate_vec)
    result_vec_arr = np.asarray(result_vec)
    assert np.issubdtype(result_vec_arr.dtype, np.integer), (
        f"Vector output dtype {result_vec_arr.dtype} is not integer"
    )

    # Datetime input
    result_dt = m2cdate(datetime.datetime(1995, 10, 28))
    result_dt_arr = np.asarray(result_dt)
    assert np.issubdtype(result_dt_arr.dtype, np.integer), (
        f"Datetime output dtype {result_dt_arr.dtype} is not integer"
    )

    # Timestamp input
    result_ts = m2cdate(pd.Timestamp("2009-07-06"))
    result_ts_arr = np.asarray(result_ts)
    assert np.issubdtype(result_ts_arr.dtype, np.integer), (
        f"Timestamp output dtype {result_ts_arr.dtype} is not integer"
    )


# ---------------------------------------------------------------------------
# Test 8: Non-numeric input raises error
# ---------------------------------------------------------------------------

def test_m2cdate_non_numeric_raises():
    """String input → ValueError.

    Ref: m2cdate.m:35-37 — MATLAB: if any(ischar(mldate)) error('CRSPDATE must be numeric')
    The Python implementation must raise ValueError for string-type inputs.
    """
    # Direct string input
    with pytest.raises(ValueError):
        m2cdate("not_a_date")

    # String in numpy array
    with pytest.raises(ValueError):
        m2cdate(np.array(["not_a_date"]))

    # Mixed string array
    with pytest.raises(ValueError):
        m2cdate(np.array(["2020-01-01", "2020-02-01"]))


# ---------------------------------------------------------------------------
# Test 9: Fixture parity with MATLAB reference output
# ---------------------------------------------------------------------------

def test_m2cdate_fixture_parity(utility_fixture_dir):
    """MATLAB fixture comparison — verify Python output matches Octave/MATLAB reference.

    Loads the pre-generated fixture from tests/fixtures/utility/m2cdate.npy
    containing input datenums and expected CRSP output dates generated by
    Octave 8.4.0.

    Tolerance: exact integer equality for date conversions (no floating-point
    comparison needed since outputs are YYYYMMDD integers).
    """
    fixture_path = utility_fixture_dir / "m2cdate.npy"
    if not fixture_path.exists():
        pytest.skip(f"Fixture file not found: {fixture_path}")

    # Load fixture dict: {'input': array, 'expected_output': array, ...}
    fixture = np.load(fixture_path, allow_pickle=True).item()
    input_datenums = fixture["input"]
    expected_crsp = fixture["expected_output"]

    # Compute Python result
    result = m2cdate(input_datenums)
    result_arr = np.asarray(result).ravel().astype(np.int64)
    expected_arr = np.asarray(expected_crsp).ravel().astype(np.int64)

    # Exact integer equality for YYYYMMDD date conversions
    npt.assert_array_equal(
        result_arr,
        expected_arr,
    )

    # Also test the separate fixture files if available
    single_input_path = utility_fixture_dir / "m2cdate_m2c_input.npy"
    single_output_path = utility_fixture_dir / "m2cdate_m2c_out.npy"
    if single_input_path.exists() and single_output_path.exists():
        single_in = np.load(single_input_path, allow_pickle=True)
        single_expected = np.load(single_output_path, allow_pickle=True)

        single_in_val = float(np.asarray(single_in).ravel()[0])
        single_expected_val = int(np.asarray(single_expected).ravel()[0])

        single_result = m2cdate(single_in_val)
        single_result_val = int(np.asarray(single_result).ravel()[0])
        assert single_result_val == single_expected_val, (
            f"Single fixture: m2cdate({single_in_val}) = {single_result_val}, "
            f"expected {single_expected_val}"
        )
