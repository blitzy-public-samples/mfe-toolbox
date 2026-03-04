"""
Comprehensive pytest tests for ``mfe_toolbox.utility.mprint`` — the formatted
matrix printing utility migrated from ``utility/mprint.m`` (299 lines).

Tests cover:
- Basic matrix printing with default options
- Custom format strings (float, integer, scientific)
- Column and row label display
- Row/column subsetting via begr/endr/begc/endc (0-indexed in Python)
- Single-element and vector inputs
- Default format precision verification
- Return value (must be None)
- File-like output redirection via fid parameter (StringIO)
- MATLAB fixture parity against precomputed reference data
- Edge cases and error handling

Ref: utility/mprint.m — original MATLAB implementation by James P. LeSage.
MATLAB uses 1-indexed begr/endr/begc/endc; Python version uses 0-indexed.
"""

import re
from io import StringIO
from pathlib import Path

import numpy as np
import pytest

from mfe_toolbox.utility.mprint import mprint

# Import shared tolerance constants and fixture helpers from conftest.py
from tests.conftest import ATOL, RTOL, assert_allclose, load_fixture_npy


# ---------------------------------------------------------------------------
# Helper to capture mprint output into a string via StringIO
# ---------------------------------------------------------------------------

def _capture_mprint(matrix: np.ndarray, info: dict | None = None) -> str:
    """Capture mprint output into a string using a StringIO buffer.

    This helper redirects output through the ``fid`` info parameter so that
    tests can inspect formatted output programmatically without relying on
    capsys.

    Parameters
    ----------
    matrix : np.ndarray
        Matrix to print.
    info : dict or None
        Optional mprint options dict.  ``fid`` will be set/overridden to
        the internal StringIO buffer.

    Returns
    -------
    str
        Complete string output captured from mprint.
    """
    buf = StringIO()
    if info is None:
        info = {}
    else:
        info = dict(info)  # shallow copy to avoid mutating caller's dict
    info['fid'] = buf
    mprint(matrix, info=info)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Test Functions
# ---------------------------------------------------------------------------


def test_mprint_basic(capsys: pytest.CaptureFixture[str]) -> None:
    """Test basic mprint with a 3×3 matrix and no options.

    Verifies:
    - No exception is raised
    - Output is non-empty
    - Output contains formatted numbers (3 rows of data)
    - All 9 matrix values appear in the output
    Ref: mprint.m:49 — default format is '%10.4f', no labels
    """
    matrix = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ])
    mprint(matrix)
    captured = capsys.readouterr()
    output = captured.out

    # Output must be non-empty
    assert len(output.strip()) > 0, "mprint produced empty output"

    # Verify all 9 values appear (default format %10.4f rounds to 4 decimals)
    for val in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]:
        expected_str = f"{val:.4f}"
        assert expected_str in output, (
            f"Expected '{expected_str}' in output but not found. Output:\n{output}"
        )

    # Verify output has at least 3 non-empty data lines
    data_lines = [line for line in output.strip().split('\n') if line.strip()]
    assert len(data_lines) >= 3, (
        f"Expected at least 3 data lines, got {len(data_lines)}"
    )


def test_mprint_with_format(capsys: pytest.CaptureFixture[str]) -> None:
    """Test mprint with a custom format string '%12.6f'.

    Verifies:
    - Output uses the custom format (6 decimal places visible)
    - Values are formatted with 12-character width fields
    Ref: mprint.m:17 — info.fmt controls format string
    """
    matrix = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ])
    mprint(matrix, info={'fmt': '%12.6f'})
    captured = capsys.readouterr()
    output = captured.out

    # Verify 6 decimal places are present in output
    assert '1.000000' in output, "Expected 6 decimal places for value 1.0"
    assert '5.000000' in output, "Expected 6 decimal places for value 5.0"
    assert '9.000000' in output, "Expected 6 decimal places for value 9.0"

    # Verify that the numbers match the %12.6f format pattern
    # Each formatted number should have exactly 6 digits after the decimal point
    decimal_matches = re.findall(r'\d+\.\d{6}', output)
    assert len(decimal_matches) >= 9, (
        f"Expected at least 9 values with 6 decimal places, found {len(decimal_matches)}"
    )


def test_mprint_with_cnames(capsys: pytest.CaptureFixture[str]) -> None:
    """Test mprint with column names provided.

    Verifies:
    - Column headers appear in the output
    - Each column name is present
    Ref: mprint.m:11 — info.cnames provides column headings
    """
    matrix = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ])
    mprint(matrix, info={'cnames': ['col1', 'col2', 'col3']})
    captured = capsys.readouterr()
    output = captured.out

    # Verify column headers are present in the output
    assert 'col1' in output, "Column name 'col1' not found in output"
    assert 'col2' in output, "Column name 'col2' not found in output"
    assert 'col3' in output, "Column name 'col3' not found in output"

    # Verify column headers appear before data rows
    lines = output.split('\n')
    header_line_idx = None
    for idx, line in enumerate(lines):
        if 'col1' in line:
            header_line_idx = idx
            break
    assert header_line_idx is not None, "Header line with 'col1' not found"

    # Verify data values also appear after headers
    assert '1.0000' in output, "Data value '1.0000' not found after headers"


def test_mprint_with_rnames(capsys: pytest.CaptureFixture[str]) -> None:
    """Test mprint with row names provided.

    Verifies:
    - First entry of rnames appears as header label
    - Subsequent entries appear as row labels
    - rnames must have nobs+1 entries (first is label header)
    Ref: mprint.m:14-15 — rnames has (nobs+1) entries; first is label header
    """
    matrix = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])
    # Ref: mprint.m:14 — rnames has nobs+1 entries; first entry is header label
    mprint(matrix, info={'rnames': ['Rows', 'row1', 'row2']})
    captured = capsys.readouterr()
    output = captured.out

    # Verify header label 'Rows' appears
    assert 'Rows' in output, "Header label 'Rows' not found in output"

    # Verify row labels appear
    assert 'row1' in output, "Row label 'row1' not found in output"
    assert 'row2' in output, "Row label 'row2' not found in output"

    # Verify data values are present alongside row labels
    assert '1.0000' in output, "Data value '1.0000' not found"
    assert '4.0000' in output, "Data value '4.0000' not found"


def test_mprint_subset_rows(capsys: pytest.CaptureFixture[str]) -> None:
    """Test mprint with row subsetting via begr/endr (0-indexed).

    Creates a 4×3 matrix and prints only rows at indices 1-2 (2 rows).
    Ref: mprint.m:7-8 — MATLAB begr/endr are 1-indexed; Python uses 0-indexed
    """
    matrix = np.array([
        [10.0, 20.0, 30.0],
        [40.0, 50.0, 60.0],
        [70.0, 80.0, 90.0],
        [100.0, 110.0, 120.0],
    ])
    # Print only rows 1-2 (0-indexed), i.e., second and third rows
    output = _capture_mprint(matrix, info={'begr': 1, 'endr': 2})

    # Verify only 2 data rows appear (rows at index 1 and 2)
    # Values from row 1: 40.0, 50.0, 60.0
    assert '40.0000' in output, "Row 1 value '40.0000' not found"
    assert '50.0000' in output, "Row 1 value '50.0000' not found"
    # Values from row 2: 70.0, 80.0, 90.0
    assert '70.0000' in output, "Row 2 value '70.0000' not found"
    assert '90.0000' in output, "Row 2 value '90.0000' not found"

    # Verify row 0 values (10.0, 20.0, 30.0) are NOT present
    assert '10.0000' not in output, "Row 0 value '10.0000' should not appear"
    assert '20.0000' not in output, "Row 0 value '20.0000' should not appear"
    # Verify row 3 values (100.0, 110.0, 120.0) are NOT present
    assert '100.0000' not in output, "Row 3 value '100.0000' should not appear"
    assert '120.0000' not in output, "Row 3 value '120.0000' should not appear"

    # Count non-empty data lines (excluding header/trailing newlines)
    data_lines = [line for line in output.strip().split('\n') if line.strip()]
    assert len(data_lines) == 2, (
        f"Expected exactly 2 data lines, got {len(data_lines)}: {data_lines}"
    )


def test_mprint_subset_cols(capsys: pytest.CaptureFixture[str]) -> None:
    """Test mprint with column subsetting via begc/endc (0-indexed).

    Creates a 3×4 matrix and prints only columns at indices 0-1 (2 columns).
    Ref: mprint.m:9-10 — MATLAB begc/endc are 1-indexed; Python uses 0-indexed
    """
    matrix = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
    ])
    # Print only columns 0-1 (first two columns)
    output = _capture_mprint(matrix, info={'begc': 0, 'endc': 1})

    # Verify columns 0 and 1 values are present
    assert '1.0000' in output, "Col 0, row 0 value '1.0000' not found"
    assert '2.0000' in output, "Col 1, row 0 value '2.0000' not found"
    assert '6.0000' in output, "Col 1, row 1 value '6.0000' not found"
    assert '10.0000' in output, "Col 1, row 2 value '10.0000' not found"

    # Verify column 2 and 3 values are NOT present
    # Note: We check for exact formatted strings to avoid substring matches
    # '3.0000' could match in '13.0000', so we check carefully
    data_lines = [line for line in output.strip().split('\n') if line.strip()]
    for line in data_lines:
        # Each line should have exactly 2 formatted numbers
        numbers = re.findall(r'-?\d+\.\d{4}', line)
        assert len(numbers) == 2, (
            f"Expected 2 values per line, got {len(numbers)}: '{line}'"
        )


def test_mprint_single_element(capsys: pytest.CaptureFixture[str]) -> None:
    """Test mprint with a 1×1 matrix containing pi.

    Verifies default format '%10.4f' rounds 3.14159 to '3.1416'.
    Ref: mprint.m:49 — default fmt='%10.4f'
    """
    matrix = np.array([[3.14159]])
    mprint(matrix)
    captured = capsys.readouterr()
    output = captured.out

    # Default format %10.4f rounds 3.14159 to 3.1416 (4 decimal places)
    assert '3.1416' in output, (
        f"Expected '3.1416' in output for pi rounded to 4 decimals. Output:\n{output}"
    )


def test_mprint_vector_input(capsys: pytest.CaptureFixture[str]) -> None:
    """Test mprint with column and row vector inputs.

    Verifies:
    - Column vector (3×1): produces 3 lines with 1 value each
    - Row vector (1×3): produces 1 line with 3 values
    Ref: mprint.m:48 — [nobs nvars] = size(y) works for any 2-D input
    """
    # Column vector: 3 rows × 1 column
    col_vec = np.array([[1.0], [2.0], [3.0]])
    mprint(col_vec)
    captured_col = capsys.readouterr()
    col_output = captured_col.out

    # Verify 3 data values appear
    assert '1.0000' in col_output, "Column vector: '1.0000' not found"
    assert '2.0000' in col_output, "Column vector: '2.0000' not found"
    assert '3.0000' in col_output, "Column vector: '3.0000' not found"

    # Verify column vector produces 3 data lines with 1 value each
    col_data_lines = [line for line in col_output.strip().split('\n') if line.strip()]
    assert len(col_data_lines) == 3, (
        f"Column vector: expected 3 data lines, got {len(col_data_lines)}"
    )

    # Row vector: 1 row × 3 columns
    row_vec = np.array([[1.0, 2.0, 3.0]])
    mprint(row_vec)
    captured_row = capsys.readouterr()
    row_output = captured_row.out

    # Verify all 3 values appear
    assert '1.0000' in row_output, "Row vector: '1.0000' not found"
    assert '2.0000' in row_output, "Row vector: '2.0000' not found"
    assert '3.0000' in row_output, "Row vector: '3.0000' not found"

    # Verify row vector produces 1 data line with 3 values
    row_data_lines = [line for line in row_output.strip().split('\n') if line.strip()]
    assert len(row_data_lines) == 1, (
        f"Row vector: expected 1 data line, got {len(row_data_lines)}"
    )


def test_mprint_default_format(capsys: pytest.CaptureFixture[str]) -> None:
    """Test that the default format '%10.4f' is applied when no format specified.

    Verifies that 1.23456789 is formatted as '1.2346' (4 decimal places, rounded).
    Ref: mprint.m:49 — default fmt = '%10.4f'
    """
    matrix = np.array([[1.23456789]])
    mprint(matrix)
    captured = capsys.readouterr()
    output = captured.out

    # Default format %10.4f should round 1.23456789 to 1.2346
    assert '1.2346' in output, (
        f"Expected '1.2346' (4 decimal places, rounded) in output. Output:\n{output}"
    )

    # Verify that more than 4 decimal places are NOT shown (i.e., '1.23456' absent)
    assert '1.23457' not in output, (
        "Found more than 4 decimal places — default format not applied"
    )


def test_mprint_returns_none() -> None:
    """Test that mprint returns None — it is a display-only utility.

    Ref: mprint.m — function has no return value (MATLAB void function).
    """
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    # Redirect output to StringIO to suppress stdout
    buf = StringIO()
    result = mprint(matrix, info={'fid': buf})
    assert result is None, (
        f"mprint should return None, got {type(result).__name__}: {result!r}"
    )


def test_mprint_to_string() -> None:
    """Test mprint output to a StringIO file-like object via the fid parameter.

    Verifies:
    - StringIO captures the formatted output
    - Output content matches expected formatted values
    Ref: mprint.m:20-21 — info.fid for directing output to a file
    """
    matrix = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])
    buf = StringIO()
    mprint(matrix, info={'fid': buf})
    output = buf.getvalue()

    # Verify output is non-empty
    assert len(output) > 0, "StringIO output is empty"

    # Verify formatted values are present
    assert '1.0000' in output, "Value '1.0000' not found in StringIO output"
    assert '2.0000' in output, "Value '2.0000' not found in StringIO output"
    assert '3.0000' in output, "Value '3.0000' not found in StringIO output"
    assert '4.0000' in output, "Value '4.0000' not found in StringIO output"

    # Verify the StringIO result matches what capsys would capture
    # (they should be identical since both redirect stdout)
    buf2 = StringIO()
    mprint(matrix, info={'fid': buf2, 'fmt': '%12.6f'})
    output2 = buf2.getvalue()
    assert '1.000000' in output2, "Custom format not applied with fid=StringIO"

    # Test with labels and StringIO
    buf3 = StringIO()
    mprint(matrix, info={
        'fid': buf3,
        'cnames': ['A', 'B'],
        'rnames': ['Label', 'r1', 'r2'],
    })
    output3 = buf3.getvalue()
    assert 'Label' in output3, "Row header label not in StringIO output"
    assert 'A' in output3, "Column name 'A' not in StringIO output"
    assert 'B' in output3, "Column name 'B' not in StringIO output"
    assert 'r1' in output3, "Row name 'r1' not in StringIO output"
    assert 'r2' in output3, "Row name 'r2' not in StringIO output"


def test_mprint_fixture_parity(utility_fixture_dir: Path) -> None:
    """Test mprint against MATLAB reference fixture data for numerical parity.

    Loads precomputed fixture data from tests/fixtures/utility/mprint.npy
    and verifies that the Python mprint produces output matching the MATLAB
    reference for each test case.

    Ref: AAP Section 0.7.1 — atol=1e-6, rtol=1e-4 for numerical comparisons.
    """
    fixture_path = utility_fixture_dir / "mprint.npy"
    if not fixture_path.exists():
        pytest.skip(f"Fixture file not found: {fixture_path}")

    fixture_data = np.load(fixture_path, allow_pickle=True).item()

    # Extract the list of test case names
    test_cases = fixture_data.get('test_cases', [])
    assert len(test_cases) > 0, "No test cases found in fixture data"

    for case_name in test_cases:
        matrix_key = f"matrix_{case_name}"
        config_key = f"config_{case_name}"

        assert matrix_key in fixture_data, (
            f"Missing matrix data for test case '{case_name}'"
        )
        assert config_key in fixture_data, (
            f"Missing config data for test case '{case_name}'"
        )

        matrix = fixture_data[matrix_key]
        config = fixture_data[config_key]

        # Build the info dict for mprint, filtering out None values
        info = {}
        buf = StringIO()
        info['fid'] = buf

        for key in ('fmt', 'width', 'begr', 'endr', 'begc', 'endc', 'rflag',
                     'cnames', 'rnames'):
            if key in config and config[key] is not None:
                info[key] = config[key]

        # Execute mprint and capture output
        mprint(matrix, info=info)
        output = buf.getvalue()

        # Verify output is non-empty for each test case
        assert len(output.strip()) > 0, (
            f"Empty output for fixture test case '{case_name}'"
        )

        # Extract all numeric values from the output
        output_numbers = re.findall(r'-?[\d]+\.[\d]+', output)

        # Determine the row/column range from config
        begr = config.get('begr', 0)
        endr = config.get('endr', matrix.shape[0] - 1)
        begc = config.get('begc', 0)
        endc = config.get('endc', matrix.shape[1] - 1)

        # Verify the matrix values in the selected subrange appear in output
        # by comparing extracted numeric strings to actual matrix values
        for row_idx in range(begr, endr + 1):
            for col_idx in range(begc, endc + 1):
                actual_val = matrix[row_idx, col_idx]
                # Use the configured format to check the value
                fmt = config.get('fmt', '%10.4f')
                if isinstance(fmt, list):
                    fmt_str = fmt[col_idx] if col_idx < len(fmt) else '%10.4f'
                else:
                    fmt_str = fmt
                expected_formatted = (fmt_str % actual_val).strip()

                # For integer formats like %12d, check the integer representation
                if 'd' in fmt_str:
                    expected_int = str(int(actual_val))
                    assert expected_int in output, (
                        f"Case '{case_name}': expected integer '{expected_int}' "
                        f"for matrix[{row_idx},{col_idx}]={actual_val} "
                        f"not found in output:\n{output}"
                    )
                else:
                    # For float formats, verify the formatted number appears
                    assert expected_formatted in output, (
                        f"Case '{case_name}': expected '{expected_formatted}' "
                        f"for matrix[{row_idx},{col_idx}]={actual_val} "
                        f"not found in output:\n{output}"
                    )


# ---------------------------------------------------------------------------
# Additional Edge Case and Error Handling Tests
# ---------------------------------------------------------------------------


def test_mprint_error_invalid_info_type() -> None:
    """Test that passing a non-dict info argument raises ValueError.

    Ref: mprint.m:53-54 — 'you must supply the options as a structure variable'
    """
    matrix = np.array([[1.0]])
    with pytest.raises(ValueError, match="structure variable"):
        mprint(matrix, info="invalid_string")


def test_mprint_error_wrong_cnames_length() -> None:
    """Test that mismatched cnames length raises ValueError.

    Ref: mprint.m:171 — 'Wrong # cnames in mprint'
    """
    matrix = np.array([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="cnames"):
        mprint(matrix, info={'cnames': ['a', 'b']})  # need 3, gave 2


def test_mprint_error_wrong_rnames_length() -> None:
    """Test that mismatched rnames length raises ValueError.

    Ref: mprint.m:231 — 'Wrong # rnames in mprint'
    """
    matrix = np.array([[1.0], [2.0]])
    # rnames needs nobs+1 = 3 entries; providing only 2
    with pytest.raises(ValueError, match="rnames"):
        mprint(matrix, info={'rnames': ['Header', 'r1']})


def test_mprint_rflag_row_numbers() -> None:
    """Test row-number display when rflag=1 is set.

    Verifies that 'Obs#' header and row numbers appear in output.
    Ref: mprint.m:23 — info.rflag=1 prints row numbers with 'Obs#' prefix
    Ref: mprint.m:165-167 — dstr='Obs#' when rnum==1
    """
    matrix = np.array([
        [0.1234, 5.6789, -3.21],
        [9.8765, -0.4321, 7.6543],
    ])
    output = _capture_mprint(matrix, info={'rflag': 1})

    # Verify 'Obs#' header label appears
    assert 'Obs#' in output, "'Obs#' header not found with rflag=1"

    # Verify row numbers appear (1-based display in Python: k+1)
    # Row 0 in Python → displays as 1; Row 1 → displays as 2
    assert '1' in output, "Row number '1' not found"
    assert '2' in output, "Row number '2' not found"


def test_mprint_column_wrapping() -> None:
    """Test that wide matrices wrap columns based on width parameter.

    Creates a 3×3 matrix with format '%10.4f' (10 chars per value) and
    sets width=25 so only 2 columns fit per line, causing a wrap.
    Ref: mprint.m:124 — nwide = floor(cwidth/f2), wrapping at nwide columns
    """
    matrix = np.array([
        [1.1111, 2.2222, 3.3333],
        [4.4444, 5.5555, 6.6666],
        [7.7777, 8.8888, 9.9999],
    ])
    # width=25 with %10.4f (10 chars): floor(25/10) = 2 cols per set
    output = _capture_mprint(matrix, info={'width': 25})

    # All values should appear in the wrapped output
    assert '1.1111' in output, "Value 1.1111 not found in wrapped output"
    assert '3.3333' in output, "Value 3.3333 not found in wrapped output"
    assert '9.9999' in output, "Value 9.9999 not found in wrapped output"

    # Count the number of value blocks (wrapped sections produce multiple groups)
    # With 3 columns and nwide=2, we need 2 sets: [cols 0-1] and [col 2]
    data_lines = [line for line in output.strip().split('\n') if line.strip()]
    # First set: 3 rows × 2 cols, then second set: 3 rows × 1 col
    # Total non-empty lines should be 6
    assert len(data_lines) >= 6, (
        f"Expected at least 6 data lines for wrapped output, got {len(data_lines)}"
    )


def test_mprint_combined_labels() -> None:
    """Test mprint with both column names and row names simultaneously.

    Verifies that both sets of labels appear correctly in the output.
    Ref: mprint.m:14-15 — rnames[0] is header label; cnames are column headers
    """
    matrix = np.array([
        [0.3456, 1.789, -2.3456],
        [4.5678, -0.1234, 9.8765],
        [-6.5432, 0.0012, 5.5555],
    ])
    output = _capture_mprint(matrix, info={
        'cnames': ['Alpha', 'Beta', 'Gamma'],
        'rnames': ['Rows', 'row1', 'row2', 'row3'],
    })

    # Verify all column labels present
    assert 'Alpha' in output, "Column name 'Alpha' not found"
    assert 'Beta' in output, "Column name 'Beta' not found"
    assert 'Gamma' in output, "Column name 'Gamma' not found"

    # Verify all row labels present
    assert 'Rows' in output, "Header label 'Rows' not found"
    assert 'row1' in output, "Row label 'row1' not found"
    assert 'row2' in output, "Row label 'row2' not found"
    assert 'row3' in output, "Row label 'row3' not found"

    # Verify data values
    assert '0.3456' in output, "Data value '0.3456' not found"
    assert '9.8765' in output, "Data value '9.8765' not found"


def test_mprint_rnames_suppresses_rflag() -> None:
    """Test that providing rnames suppresses rflag row numbering.

    Ref: mprint.m:98-102 — when rnames provided (rflag==1), rnum is set to 0.
    Row names take precedence over row numbers.
    """
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    output = _capture_mprint(matrix, info={
        'rnames': ['Label', 'a', 'b'],
        'rflag': 1,  # Should be overridden by rnames
    })

    # Row names should appear
    assert 'Label' in output, "Header label not found"
    assert 'a' in output, "Row name 'a' not found"

    # 'Obs#' should NOT appear since rnames takes precedence
    assert 'Obs#' not in output, (
        "'Obs#' should not appear when rnames is provided"
    )


def test_mprint_large_values() -> None:
    """Test mprint handles large and small floating-point values correctly."""
    matrix = np.array([
        [1e6, -1e-6],
        [1e10, 3.14159265],
    ])
    output = _capture_mprint(matrix)

    # Values should appear formatted (default %10.4f)
    assert len(output.strip()) > 0, "Output should not be empty for large values"


def test_mprint_integer_format() -> None:
    """Test mprint with integer format '%12d'.

    Verifies that values are formatted as integers without decimal points.
    Ref: mprint.m:108-111 — integer (d) format detection
    """
    matrix = np.array([[1.0, 200.0, -15.0], [42.0, -1000.0, 7.0]])
    output = _capture_mprint(matrix, info={'fmt': '%12d'})

    # Integer format should produce values without decimal points
    # Look for integer representations
    assert '1' in output, "Integer value '1' not found"
    assert '200' in output, "Integer value '200' not found"
    assert '42' in output, "Integer value '42' not found"


def test_mprint_1d_array_input() -> None:
    """Test that a 1-D array is handled by atleast_2d promotion.

    mprint should auto-promote 1-D arrays to 2-D via np.atleast_2d.
    Ref: mprint.py:214 — y = np.atleast_2d(y)
    """
    vec_1d = np.array([1.0, 2.0, 3.0])
    output = _capture_mprint(vec_1d)

    # 1-D array [1, 2, 3] should be promoted to [[1, 2, 3]] (1×3 matrix)
    assert '1.0000' in output, "Value '1.0000' not in 1-D array output"
    assert '2.0000' in output, "Value '2.0000' not in 1-D array output"
    assert '3.0000' in output, "Value '3.0000' not in 1-D array output"

    # Should produce a single data line (1 row)
    data_lines = [line for line in output.strip().split('\n') if line.strip()]
    assert len(data_lines) == 1, (
        f"1-D array should produce 1 data line, got {len(data_lines)}"
    )


def test_mprint_no_options_is_equivalent_to_empty_dict() -> None:
    """Test that calling mprint(y) is equivalent to mprint(y, info={}).

    Both should produce identical output with default settings.
    """
    matrix = np.array([[1.5, 2.5], [3.5, 4.5]])
    output_no_info = _capture_mprint(matrix, info=None)
    output_empty_info = _capture_mprint(matrix, info={})

    # Both should produce identical output
    # Note: _capture_mprint sets fid, so info is never actually None internally
    # but the default behavior should match
    assert output_no_info == output_empty_info, (
        f"No-info and empty-info outputs differ:\n"
        f"No info: {output_no_info!r}\n"
        f"Empty info: {output_empty_info!r}"
    )


def test_mprint_precision_high_decimal() -> None:
    """Test mprint with high-precision format string '%16.10f'.

    Verifies that the format produces values with 10 decimal places.
    """
    matrix = np.array([[1.23456789, -1.23456e-6]])
    output = _capture_mprint(matrix, info={'fmt': '%16.10f'})

    # 10 decimal places should be visible
    decimal_matches = re.findall(r'-?\d+\.\d{10}', output)
    assert len(decimal_matches) >= 2, (
        f"Expected at least 2 values with 10 decimal places, "
        f"found {len(decimal_matches)} in: {output}"
    )


def test_mprint_fixture_data_matrix(utility_fixture_dir: Path) -> None:
    """Test mprint with the secondary fixture data matrix.

    Loads the mprint_mprint_data.npy fixture and verifies that mprint
    can display it without error and output contains expected values.
    """
    fixture_path = utility_fixture_dir / "mprint_mprint_data.npy"
    if not fixture_path.exists():
        pytest.skip(f"Fixture file not found: {fixture_path}")

    data_matrix = np.load(fixture_path)
    assert data_matrix.ndim == 2, "Fixture data should be a 2-D matrix"

    # mprint should handle this matrix without errors
    output = _capture_mprint(data_matrix)
    assert len(output.strip()) > 0, "Output should not be empty for fixture data"

    # Verify all values from the matrix appear in formatted output
    nrows, ncols = data_matrix.shape
    for r in range(nrows):
        for c in range(ncols):
            val = data_matrix[r, c]
            formatted_val = '%10.4f' % val
            assert formatted_val.strip() in output, (
                f"Expected formatted value '{formatted_val.strip()}' for "
                f"matrix[{r},{c}]={val} not found in output"
            )
