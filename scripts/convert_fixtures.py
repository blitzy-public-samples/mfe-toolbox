#!/usr/bin/env python3
"""Convert MATLAB .mat fixture files to .npy and .csv formats.

This script is part of the MFE Toolbox MATLAB-to-Python migration fixture pipeline:
    MATLAB/Octave (generate_fixtures.m) -> .mat files -> convert_fixtures.py -> .npy/.csv -> pytest

The script handles two categories of .mat files:
    1. Fixture files produced by scripts/generate_fixtures.m, organized by subpackage
       in the fixtures_mat/ directory tree.
    2. Pre-existing .mat data files from the original MFE Toolbox source repository
       (realized/ES_20090817.mat, realized/realized_quantile_scales.mat,
       realized/realized_range_simulation_results.mat).

Usage:
    python scripts/convert_fixtures.py [--mat-dir MAT_DIR] [--output-dir OUTPUT_DIR]
    python scripts/convert_fixtures.py --data-dir /path/to/mfe-toolbox-source

Environment Variables:
    MFE_FIXTURE_DIR - Override for fixture output path during CI

Examples:
    # Convert fixture .mat files from generate_fixtures.m output
    python scripts/convert_fixtures.py --mat-dir fixtures_mat --output-dir tests/fixtures

    # Also convert pre-existing realized/*.mat data files
    python scripts/convert_fixtures.py --data-dir . --output-dir tests/fixtures

    # Use CI environment variable for output path
    MFE_FIXTURE_DIR=/ci/fixtures python scripts/convert_fixtures.py
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Subpackage directory names matching the Python package structure.
# Ref: buildZipFile.m:17-30 — MATLAB dirList defines the 13 module directories;
# the Python target uses 10 subpackages (duplication/ eliminated, GUI/ -> gui/,
# mex_source/ and dlls/ replaced by Numba).
SUBPACKAGE_DIRS: list[str] = [
    "univariate",
    "multivariate",
    "timeseries",
    "realized",
    "distributions",
    "utility",
    "bootstrap",
    "crosssection",
    "sandbox",
    "tests",
]

# Pre-existing .mat data files from the original MFE Toolbox source repository
# that need to be converted alongside the generated fixture files.
# Ref: AAP Section 0.3.1 — Data fixture file destinations.
REALIZED_DATA_FILES: list[str] = [
    "ES_20090817.mat",
    "realized_quantile_scales.mat",
    "realized_range_simulation_results.mat",
]

# Metadata keys returned by scipy.io.loadmat that are not user variables and
# must be skipped during conversion.
LOADMAT_SKIP_KEYS: frozenset[str] = frozenset({
    "__header__",
    "__version__",
    "__globals__",
})


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _is_numeric_array(arr: np.ndarray) -> bool:
    """Check if a numpy array contains numeric data suitable for CSV export.

    Parameters
    ----------
    arr : np.ndarray
        The array to check.

    Returns
    -------
    bool
        True if the array has a numeric (integer, float, complex) dtype and
        is at most 2-dimensional, making it appropriate for CSV export.
    """
    return (
        isinstance(arr, np.ndarray)
        and arr.dtype.kind in ("i", "u", "f", "c")  # int, uint, float, complex
        and arr.ndim <= 2
    )


def _squeeze_matlab_array(arr: np.ndarray) -> np.ndarray:
    """Squeeze MATLAB-style (N,1) or (1,N) arrays to 1-D where appropriate.

    MATLAB stores all vectors as 2-D column or row vectors.  This function
    squeezes trailing singleton dimensions to produce natural Python 1-D
    arrays, while leaving genuinely multi-dimensional arrays intact.

    Parameters
    ----------
    arr : np.ndarray
        Input array, possibly with singleton dimensions.

    Returns
    -------
    np.ndarray
        Squeezed array.  Scalar (0-d) results are preserved as 0-d arrays.
    """
    if not isinstance(arr, np.ndarray):
        return arr
    squeezed = np.squeeze(arr)
    # Preserve 0-d arrays as scalars; do not further manipulate
    return squeezed


def _save_struct_fields(
    struct_arr: np.ndarray,
    prefix: str,
    output_dir: Path,
    write_csv: bool,
) -> int:
    """Recursively save fields from a MATLAB struct (numpy structured array).

    MATLAB structs loaded by ``scipy.io.loadmat`` become numpy arrays with a
    structured dtype whose field names correspond to the struct field names.
    The struct data is wrapped in a (1,1) shaped array; each field value is
    accessed via ``struct_arr[0, 0][field_name]``.

    Parameters
    ----------
    struct_arr : np.ndarray
        Structured numpy array as returned by loadmat for a MATLAB struct.
    prefix : str
        Name prefix for saved files (e.g. the original variable name).
    output_dir : Path
        Directory where .npy/.csv files are written.
    write_csv : bool
        Whether to also write .csv files for numeric 1-D/2-D arrays.

    Returns
    -------
    int
        Number of individual field arrays saved.
    """
    count = 0
    if struct_arr.dtype.names is None:
        return count

    # Struct arrays from loadmat are (1,1) shaped; access the single element
    try:
        element = struct_arr.flat[0]
    except (IndexError, ValueError):
        return count

    for field_name in struct_arr.dtype.names:
        try:
            field_data = element[field_name]
        except (IndexError, KeyError, ValueError):
            continue

        if not isinstance(field_data, np.ndarray):
            continue

        # Recursive struct handling: if the field itself is a struct
        if field_data.dtype.names is not None:
            count += _save_struct_fields(
                field_data,
                f"{prefix}_{field_name}",
                output_dir,
                write_csv,
            )
            continue

        # Squeeze singleton dimensions from field data
        field_data = _squeeze_matlab_array(field_data)

        # Save as .npy
        npy_path = output_dir / f"{prefix}_{field_name}.npy"
        np.save(str(npy_path), field_data)
        count += 1

        # Optionally save as .csv for numeric arrays
        if write_csv and _is_numeric_array(field_data):
            csv_path = output_dir / f"{prefix}_{field_name}.csv"
            try:
                np.savetxt(str(csv_path), field_data, delimiter=",")
            except (ValueError, TypeError):
                # Some arrays cannot be saved as CSV (e.g. 0-d scalars)
                pass

    return count


def _save_cell_array(
    cell_arr: np.ndarray,
    var_name: str,
    output_dir: Path,
    write_csv: bool,
) -> int:
    """Save a MATLAB cell array (numpy object array) element-by-element.

    MATLAB cell arrays loaded by ``scipy.io.loadmat`` become numpy arrays with
    ``dtype=object``.  Each element is itself a numpy array.  This function
    saves each element separately with an index suffix.

    Parameters
    ----------
    cell_arr : np.ndarray
        Object-dtype numpy array representing a MATLAB cell array.
    var_name : str
        Base variable name for file naming.
    output_dir : Path
        Output directory.
    write_csv : bool
        Whether to also write CSV for numeric elements.

    Returns
    -------
    int
        Number of individual elements saved.
    """
    count = 0
    flat = cell_arr.ravel()
    for idx, element in enumerate(flat):
        if not isinstance(element, np.ndarray):
            continue
        element = _squeeze_matlab_array(element)

        npy_path = output_dir / f"{var_name}_cell{idx}.npy"
        np.save(str(npy_path), element)
        count += 1

        if write_csv and _is_numeric_array(element):
            csv_path = output_dir / f"{var_name}_cell{idx}.csv"
            try:
                np.savetxt(str(csv_path), element, delimiter=",")
            except (ValueError, TypeError):
                pass

    return count


# ---------------------------------------------------------------------------
# Core conversion functions
# ---------------------------------------------------------------------------

def convert_mat_file(
    mat_path: Path,
    output_subdir: Path,
    write_csv: bool = True,
) -> int:
    """Convert a single .mat file to .npy (and optionally .csv) files.

    Each variable inside the .mat file is saved as a separate .npy file named
    ``<mat_stem>_<variable_name>.npy`` under *output_subdir*.  MATLAB metadata
    keys (``__header__``, ``__version__``, ``__globals__``) are skipped.

    For MATLAB structs, each field is saved recursively.  For cell arrays,
    each element is saved with an index suffix.

    Parameters
    ----------
    mat_path : Path
        Path to the input .mat file.
    output_subdir : Path
        Directory where output .npy/.csv files are written.
    write_csv : bool, optional
        If True (default), also save 1-D and 2-D numeric arrays as .csv.

    Returns
    -------
    int
        Total number of variables (or struct fields / cell elements) converted.

    Raises
    ------
    This function does not raise on conversion errors for individual variables;
    it prints warnings and continues.
    """
    output_subdir.mkdir(parents=True, exist_ok=True)
    mat_stem = mat_path.stem
    var_count = 0

    # Attempt to load the .mat file
    try:
        mat_data = sio.loadmat(str(mat_path), squeeze_me=False)
    except NotImplementedError:
        # scipy.io.loadmat cannot read HDF5-based v7.3 .mat files
        print(
            f"  WARNING: '{mat_path.name}' is a v7.3 HDF5 .mat file — "
            "scipy.io.loadmat does not support this format. "
            "Install h5py and use h5py.File() instead. Skipping."
        )
        return 0
    except Exception as exc:
        print(f"  ERROR: Failed to load '{mat_path.name}': {exc}")
        return 0

    # Iterate over all variables, skipping metadata keys
    for var_name, var_data in mat_data.items():
        if var_name in LOADMAT_SKIP_KEYS:
            continue

        if not isinstance(var_data, np.ndarray):
            # Rare edge case: loadmat returned a non-array value
            print(
                f"  WARNING: Variable '{var_name}' in '{mat_path.name}' is "
                f"type {type(var_data).__name__}, not ndarray. Skipping."
            )
            continue

        # Determine the output file name prefix
        file_prefix = f"{mat_stem}_{var_name}"

        try:
            # --- Handle MATLAB structs (structured dtype) ---
            if var_data.dtype.names is not None:
                saved = _save_struct_fields(
                    var_data, file_prefix, output_subdir, write_csv
                )
                var_count += saved
                continue

            # --- Handle MATLAB cell arrays (object dtype) ---
            if var_data.dtype == object:
                saved = _save_cell_array(
                    var_data, file_prefix, output_subdir, write_csv
                )
                var_count += saved
                continue

            # --- Handle regular numeric / logical / char arrays ---
            squeezed = _squeeze_matlab_array(var_data)

            # Warn on all-NaN arrays (might indicate empty/missing data)
            if squeezed.dtype.kind == "f" and squeezed.size > 0:
                if np.all(np.isnan(squeezed)):
                    print(
                        f"  WARNING: Variable '{var_name}' in "
                        f"'{mat_path.name}' contains only NaN values."
                    )

            # Save as .npy
            npy_path = output_subdir / f"{file_prefix}.npy"
            np.save(str(npy_path), squeezed)
            var_count += 1

            # Optionally save as .csv for 1-D/2-D numeric arrays
            if write_csv and _is_numeric_array(squeezed):
                csv_path = output_subdir / f"{file_prefix}.csv"
                try:
                    np.savetxt(str(csv_path), squeezed, delimiter=",")
                except (ValueError, TypeError):
                    # Edge case: 0-d scalar arrays cannot be saved via savetxt
                    pass

        except Exception as exc:
            print(
                f"  ERROR: Failed to convert variable '{var_name}' "
                f"in '{mat_path.name}': {exc}"
            )

    return var_count


def convert_directory(
    mat_dir: Path,
    output_dir: Path,
    write_csv: bool = True,
) -> tuple[int, int, list[str]]:
    """Convert all .mat files in a directory tree to .npy/.csv.

    The function walks *mat_dir*, locating all ``.mat`` files.  Each file's
    relative path within *mat_dir* determines the output subdirectory under
    *output_dir*, preserving the subpackage organization produced by
    ``generate_fixtures.m``.

    Parameters
    ----------
    mat_dir : Path
        Root directory containing .mat fixture files, organized by subpackage
        subdirectories (e.g. ``fixtures_mat/univariate/``, etc.).
    output_dir : Path
        Root output directory (e.g. ``tests/fixtures/``).
    write_csv : bool, optional
        If True (default), also generate .csv files.

    Returns
    -------
    tuple[int, int, list[str]]
        A 3-tuple of (files_processed, total_variables, list_of_errors).
    """
    if not mat_dir.is_dir():
        print(f"WARNING: Fixture .mat directory '{mat_dir}' does not exist. Skipping.")
        return 0, 0, []

    files_processed = 0
    total_vars = 0
    errors: list[str] = []

    # Collect all .mat files recursively
    mat_files = sorted(mat_dir.rglob("*.mat"))
    if not mat_files:
        print(f"  No .mat files found in '{mat_dir}'.")
        return 0, 0, []

    print(f"  Found {len(mat_files)} .mat file(s) in '{mat_dir}'.")

    for mat_path in mat_files:
        # Determine the relative output subdirectory
        try:
            rel_path = mat_path.relative_to(mat_dir)
        except ValueError:
            rel_path = Path(mat_path.name)

        # The parent of the relative path gives us the subpackage directory
        output_subdir = output_dir / rel_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)

        print(f"  Converting: {rel_path} ...", end=" ")

        try:
            n_vars = convert_mat_file(mat_path, output_subdir, write_csv)
            total_vars += n_vars
            files_processed += 1
            print(f"{n_vars} variable(s).")
        except Exception as exc:
            error_msg = f"{rel_path}: {exc}"
            errors.append(error_msg)
            print(f"FAILED: {exc}")

    return files_processed, total_vars, errors


def convert_data_files(
    data_dir: Path,
    output_dir: Path,
    write_csv: bool = True,
) -> tuple[int, int, list[str]]:
    """Convert the 3 pre-existing realized/*.mat data files.

    These files are part of the original MFE Toolbox source repository and
    contain sample high-frequency data and precomputed simulation results:

    - ``realized/ES_20090817.mat`` -> ``tests/fixtures/realized/ES_20090817.npy``
    - ``realized/realized_quantile_scales.mat`` ->
      ``tests/fixtures/realized/realized_quantile_scales.npy``
    - ``realized/realized_range_simulation_results.mat`` ->
      ``tests/fixtures/realized/realized_range_simulation_results.npy``

    Parameters
    ----------
    data_dir : Path
        Path to the root of the original MFE Toolbox source (the directory
        containing the ``realized/`` subdirectory).
    output_dir : Path
        Root output directory for fixtures (e.g. ``tests/fixtures/``).
    write_csv : bool, optional
        If True (default), also generate .csv files.

    Returns
    -------
    tuple[int, int, list[str]]
        A 3-tuple of (files_processed, total_variables, list_of_errors).
    """
    realized_src = data_dir / "realized"
    realized_out = output_dir / "realized"
    realized_out.mkdir(parents=True, exist_ok=True)

    files_processed = 0
    total_vars = 0
    errors: list[str] = []

    for mat_filename in REALIZED_DATA_FILES:
        mat_path = realized_src / mat_filename
        if not mat_path.exists():
            msg = f"Data file not found: {mat_path}"
            print(f"  WARNING: {msg}")
            errors.append(msg)
            continue

        print(f"  Converting data file: realized/{mat_filename} ...", end=" ")

        try:
            n_vars = convert_mat_file(mat_path, realized_out, write_csv)
            total_vars += n_vars
            files_processed += 1
            print(f"{n_vars} variable(s).")
        except Exception as exc:
            error_msg = f"realized/{mat_filename}: {exc}"
            errors.append(error_msg)
            print(f"FAILED: {exc}")

    return files_processed, total_vars, errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the fixture conversion script.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes: mat_dir, output_dir, data_dir,
        csv, no_csv.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Convert MATLAB .mat fixture files to .npy and .csv formats "
            "for the MFE Toolbox pytest parity test suite."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Environment Variables:\n"
            "  MFE_FIXTURE_DIR  Override output directory for CI integration\n"
            "\n"
            "Examples:\n"
            "  python scripts/convert_fixtures.py\n"
            "  python scripts/convert_fixtures.py --mat-dir fixtures_mat\n"
            "  python scripts/convert_fixtures.py --data-dir . --no-csv\n"
        ),
    )
    parser.add_argument(
        "--mat-dir",
        type=str,
        default="fixtures_mat",
        help=(
            "Directory containing .mat fixture files from generate_fixtures.m "
            "(default: fixtures_mat)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory for .npy/.csv files "
            "(default: tests/fixtures/ or MFE_FIXTURE_DIR env var)"
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=(
            "Path to the original MFE Toolbox source for converting "
            "realized/*.mat data files. If not specified, data file "
            "conversion is skipped unless realized/ exists in the "
            "current directory."
        ),
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        default=True,
        help="Generate .csv files alongside .npy (default: True)",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        default=False,
        help="Disable .csv generation (only produce .npy files)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Main entry point for the fixture conversion script.

    Orchestrates the conversion of both generated fixture .mat files and
    pre-existing data .mat files, prints progress, and exits with an
    appropriate status code.
    """
    args = parse_args()

    # Resolve output directory: MFE_FIXTURE_DIR env var > --output-dir > default
    output_dir = Path(
        os.environ.get("MFE_FIXTURE_DIR", args.output_dir or "tests/fixtures")
    )

    # Determine CSV writing preference
    write_csv = args.csv and not args.no_csv

    mat_dir = Path(args.mat_dir)

    # Print banner
    print("=" * 70)
    print("MFE Toolbox Fixture Converter")
    print("  .mat -> .npy" + (" + .csv" if write_csv else ""))
    print("=" * 70)
    print(f"  Fixture .mat directory : {mat_dir}")
    print(f"  Output directory       : {output_dir}")
    if args.data_dir:
        print(f"  Data source directory  : {args.data_dir}")
    print()

    # Ensure output directory and all subpackage subdirectories exist
    output_dir.mkdir(parents=True, exist_ok=True)
    for subpkg in SUBPACKAGE_DIRS:
        (output_dir / subpkg).mkdir(parents=True, exist_ok=True)

    all_errors: list[str] = []
    grand_files = 0
    grand_vars = 0

    # --- Phase 1: Convert generated fixture .mat files ---
    print("Phase 1: Converting generated fixture .mat files...")
    if mat_dir.is_dir():
        f_count, v_count, f_errors = convert_directory(
            mat_dir, output_dir, write_csv
        )
        grand_files += f_count
        grand_vars += v_count
        all_errors.extend(f_errors)
    else:
        print(
            f"  Fixture directory '{mat_dir}' not found — skipping. "
            "Run generate_fixtures.m first to produce .mat fixtures."
        )
    print()

    # --- Phase 2: Convert pre-existing realized/*.mat data files ---
    print("Phase 2: Converting pre-existing realized/*.mat data files...")
    # Determine data directory: explicit --data-dir, or try current directory
    data_dir_path: Path | None = None
    if args.data_dir is not None:
        data_dir_path = Path(args.data_dir)
    else:
        # Auto-detect: check if realized/ exists in current directory
        if (Path(".") / "realized").is_dir():
            # Check if any of the expected .mat files exist
            realized_dir = Path(".") / "realized"
            if any((realized_dir / f).exists() for f in REALIZED_DATA_FILES):
                data_dir_path = Path(".")

    if data_dir_path is not None:
        d_count, d_vars, d_errors = convert_data_files(
            data_dir_path, output_dir, write_csv
        )
        grand_files += d_count
        grand_vars += d_vars
        all_errors.extend(d_errors)
    else:
        print(
            "  No data source directory specified or auto-detected — "
            "skipping data file conversion."
        )
        print(
            "  Use --data-dir to specify the path to the original "
            "MFE Toolbox source."
        )
    print()

    # --- Summary ---
    print("=" * 70)
    print("Conversion Summary")
    print("=" * 70)
    print(f"  Files processed    : {grand_files}")
    print(f"  Variables converted : {grand_vars}")
    print(f"  Output directory   : {output_dir.resolve()}")
    print(f"  CSV generation     : {'enabled' if write_csv else 'disabled'}")

    if all_errors:
        print(f"\n  Errors encountered : {len(all_errors)}")
        print("  " + "-" * 40)
        for err in all_errors:
            print(f"    - {err}")
        print()
        print("Some conversions failed. Check the errors above.")
        sys.exit(1)
    else:
        print("\n  All conversions completed successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
