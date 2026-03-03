"""
Pytest tests for mfe_toolbox.utility.dirod — date-ordered directory listing utility.

Tests cover the Python migration of utility/dirod.m (49 lines) which provides
date-sorted directory listing similar to Windows ``dir /od``, using
``pathlib.Path.glob()`` instead of MATLAB ``dir()``.

This utility is filesystem-dependent. All tests use pytest's ``tmp_path``
fixture plus ``monkeypatch.chdir`` to create controlled test directories,
ensuring deterministic and isolated execution. No MATLAB fixture parity is
needed since behavior is inherently filesystem-dependent.

Author: Kevin Sheppard (original MATLAB, Revision 1, 6/7/2010)
Migration tests for mfe_toolbox Python package.
"""

import os
import pathlib
import runpy
import subprocess
import sys
import time

import pytest

from mfe_toolbox.utility.dirod import dirod, _print_listing, _safe_mtime


# ---------------------------------------------------------------------------
# Helper: create files with specific modification times inside a directory
# ---------------------------------------------------------------------------

def _create_file(directory, name, content="", mtime=None):
    """
    Create a file inside *directory* with optional explicit modification time.

    Parameters
    ----------
    directory : pathlib.Path
        Target directory (typically pytest's ``tmp_path``).
    name : str
        Filename to create.
    content : str
        File content (default empty).
    mtime : float or None
        If provided, set the file's modification time (seconds since epoch)
        via ``os.utime()``.  If *None*, the file keeps the OS-assigned time.

    Returns
    -------
    pathlib.Path
        Full path to the created file.
    """
    fpath = directory / name
    fpath.write_text(content)
    if mtime is not None:
        # Ref: schema external_imports — os.utime() sets specific timestamps
        os.utime(fpath, (mtime, mtime))
    return fpath


# ---------------------------------------------------------------------------
# 1. test_dirod_basic — Create temp directory with files, verify listing
# ---------------------------------------------------------------------------

def test_dirod_basic(tmp_path, monkeypatch):
    """Verify that dirod returns a non-empty listing for a directory with files."""
    _create_file(tmp_path, "alpha.txt", "hello")
    _create_file(tmp_path, "beta.txt", "world")
    _create_file(tmp_path, "gamma.py", "pass")

    # dirod operates on the current working directory (pathlib.Path('.').glob())
    monkeypatch.chdir(tmp_path)

    result = dirod()
    assert isinstance(result, list)
    assert len(result) >= 3  # at least the 3 files we created


# ---------------------------------------------------------------------------
# 2. test_dirod_sorted_by_date — Verify date-sorted order
# ---------------------------------------------------------------------------

def test_dirod_sorted_by_date(tmp_path, monkeypatch):
    """
    Create files with known modification times and verify dirod returns them
    sorted by modification time in ascending order.

    Ref: dirod.m:32 — MATLAB sorts by datenum; Python sorts by st_mtime.
    """
    # Assign distinct, well-separated modification times (seconds since epoch)
    base_time = 1_600_000_000.0  # arbitrary epoch timestamp

    # Create files with explicit mtimes: newest → oldest at creation time
    # but we want to verify ascending sort by mtime, so assign:
    #   file_old.txt  → base_time        (oldest)
    #   file_mid.txt  → base_time + 100  (middle)
    #   file_new.txt  → base_time + 200  (newest)
    _create_file(tmp_path, "file_new.txt", "new", mtime=base_time + 200)
    _create_file(tmp_path, "file_old.txt", "old", mtime=base_time)
    _create_file(tmp_path, "file_mid.txt", "mid", mtime=base_time + 100)

    monkeypatch.chdir(tmp_path)
    result = dirod("*.txt")

    # Extract ordered names from the listing
    names = [entry["name"] for entry in result]

    assert names == ["file_old.txt", "file_mid.txt", "file_new.txt"], (
        f"Expected ascending date order, got: {names}"
    )


# ---------------------------------------------------------------------------
# 3. test_dirod_file_mask — Pattern '*.txt' returns only .txt files
# ---------------------------------------------------------------------------

def test_dirod_file_mask(tmp_path, monkeypatch):
    """Verify that a glob pattern restricts results to matching files only."""
    _create_file(tmp_path, "report.txt")
    _create_file(tmp_path, "data.csv")
    _create_file(tmp_path, "script.py")
    _create_file(tmp_path, "notes.txt")

    monkeypatch.chdir(tmp_path)
    result = dirod("*.txt")

    names = {entry["name"] for entry in result}
    assert names == {"report.txt", "notes.txt"}, (
        f"Expected only .txt files, got: {names}"
    )


# ---------------------------------------------------------------------------
# 4. test_dirod_wildcard_all — Default mask '*' returns all files
# ---------------------------------------------------------------------------

def test_dirod_wildcard_all(tmp_path, monkeypatch):
    """Default mask '*' should return all files and directories."""
    _create_file(tmp_path, "a.txt")
    _create_file(tmp_path, "b.csv")
    _create_file(tmp_path, "c.py")

    monkeypatch.chdir(tmp_path)
    result = dirod()  # default mask is '*'

    names = {entry["name"] for entry in result}
    # All three files must be present (possibly more if OS creates extras)
    assert {"a.txt", "b.csv", "c.py"}.issubset(names), (
        f"Expected all files present, got: {names}"
    )


# ---------------------------------------------------------------------------
# 5. test_dirod_empty_directory — Empty dir returns empty listing
# ---------------------------------------------------------------------------

def test_dirod_empty_directory(tmp_path, monkeypatch):
    """An empty directory should produce an empty listing."""
    monkeypatch.chdir(tmp_path)

    result = dirod()
    assert isinstance(result, list)
    assert len(result) == 0, f"Expected empty list for empty directory, got {len(result)} entries"


# ---------------------------------------------------------------------------
# 6. test_dirod_returns_list — Verify return type is list
# ---------------------------------------------------------------------------

def test_dirod_returns_list(tmp_path, monkeypatch):
    """Verify that dirod always returns a list (even with a single file)."""
    _create_file(tmp_path, "only.dat")

    monkeypatch.chdir(tmp_path)
    result = dirod()

    assert isinstance(result, list), f"Expected list, got {type(result).__name__}"
    # Each element should be a dict
    for entry in result:
        assert isinstance(entry, dict), f"Expected dict entry, got {type(entry).__name__}"


# ---------------------------------------------------------------------------
# 7. test_dirod_contains_names — Each entry has a 'name' field
# ---------------------------------------------------------------------------

def test_dirod_contains_names(tmp_path, monkeypatch):
    """Every entry in the listing must have 'name' and 'date' keys."""
    _create_file(tmp_path, "foo.txt")
    _create_file(tmp_path, "bar.csv")

    monkeypatch.chdir(tmp_path)
    result = dirod()

    for entry in result:
        assert "name" in entry, f"Entry missing 'name' key: {entry}"
        assert "date" in entry, f"Entry missing 'date' key: {entry}"
        # 'name' should be a non-empty string (basename only, not full path)
        assert isinstance(entry["name"], str) and len(entry["name"]) > 0
        # 'date' should be a non-empty string
        assert isinstance(entry["date"], str) and len(entry["date"]) > 0


# ---------------------------------------------------------------------------
# 8. test_dirod_nonexistent_pattern — Pattern matching nothing → empty result
# ---------------------------------------------------------------------------

def test_dirod_nonexistent_pattern(tmp_path, monkeypatch):
    """A glob pattern that matches no files should return an empty list."""
    _create_file(tmp_path, "data.csv")
    _create_file(tmp_path, "report.txt")

    monkeypatch.chdir(tmp_path)
    result = dirod("*.xyz")  # no .xyz files exist

    assert isinstance(result, list)
    assert len(result) == 0, f"Expected empty result for non-matching pattern, got {len(result)}"


# ---------------------------------------------------------------------------
# 9. test_dirod_single_file — Directory with exactly one file
# ---------------------------------------------------------------------------

def test_dirod_single_file(tmp_path, monkeypatch):
    """A directory with a single file should return a one-element listing."""
    _create_file(tmp_path, "lonely.txt", "content")

    monkeypatch.chdir(tmp_path)
    result = dirod()

    assert len(result) == 1, f"Expected 1 entry, got {len(result)}"
    assert result[0]["name"] == "lonely.txt"
    assert "date" in result[0]


# ---------------------------------------------------------------------------
# Additional edge-case tests for robustness
# ---------------------------------------------------------------------------

def test_dirod_invalid_mask_raises(tmp_path, monkeypatch):
    """
    Passing a non-string mask should raise ValueError.

    Ref: dirod.m:28 — MATLAB error('Too many input arguments.') maps to
    Python raise ValueError(...) for invalid argument types.
    """
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError):
        dirod(123)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        dirod(None)  # type: ignore[arg-type]


def test_dirod_date_format(tmp_path, monkeypatch):
    """
    Verify the 'date' field format matches 'DD-Mon-YYYY HH:MM:SS'.

    Ref: dirod.py:114-115 — Python strftime('%d-%b-%Y %H:%M:%S') produces
    the same format as MATLAB's default date display.
    """
    _create_file(tmp_path, "check.txt")

    monkeypatch.chdir(tmp_path)
    result = dirod()

    assert len(result) >= 1
    date_str = result[0]["date"]
    # Basic structural check: should contain day, month-abbreviation, year
    parts = date_str.split()
    assert len(parts) == 2, f"Expected 'date time' format, got: {date_str!r}"
    date_part, time_part = parts
    # date_part should have format DD-Mon-YYYY (11 chars)
    date_tokens = date_part.split("-")
    assert len(date_tokens) == 3, f"Expected DD-Mon-YYYY, got: {date_part!r}"
    # time_part should have format HH:MM:SS
    time_tokens = time_part.split(":")
    assert len(time_tokens) == 3, f"Expected HH:MM:SS, got: {time_part!r}"


def test_dirod_specific_mask(tmp_path, monkeypatch):
    """
    Verify that a specific mask pattern like 'd*.m' works correctly.

    Ref: dirod.m:11 — MATLAB example masks include '*.m', 'd*.m'.
    """
    _create_file(tmp_path, "data.m")
    _create_file(tmp_path, "driver.m")
    _create_file(tmp_path, "utils.m")
    _create_file(tmp_path, "data.txt")

    monkeypatch.chdir(tmp_path)
    result = dirod("d*.m")

    names = {entry["name"] for entry in result}
    assert names == {"data.m", "driver.m"}, (
        f"Expected d*.m to match data.m and driver.m, got: {names}"
    )


# ---------------------------------------------------------------------------
# Coverage: _print_listing helper function (Ref: dirod.m:40-45)
# ---------------------------------------------------------------------------

def test_print_listing_with_entries(tmp_path, monkeypatch, capsys):
    """
    Verify _print_listing produces formatted output to stdout.

    Ref: dirod.m:40-45 — MATLAB nargout==0 branch prints formatted listing.
    """
    _create_file(tmp_path, "alpha.txt")
    _create_file(tmp_path, "beta_longer_name.txt")

    monkeypatch.chdir(tmp_path)
    entries = dirod()
    _print_listing(entries)

    captured = capsys.readouterr()
    assert len(captured.out) > 0
    # Should contain file names
    for entry in entries:
        assert entry["name"] in captured.out


def test_print_listing_empty(capsys):
    """
    Verify _print_listing handles an empty listing gracefully.

    Ref: dirod.py:162-164 — empty dir_data branch.
    """
    _print_listing([])
    captured = capsys.readouterr()
    # Should print at least a space
    assert captured.out.strip() == ""


def test_print_listing_single_entry(capsys):
    """Verify _print_listing handles a single entry listing."""
    entries = [{"name": "test.txt", "date": "01-Jan-2024 12:00:00"}]
    _print_listing(entries)

    captured = capsys.readouterr()
    assert "test.txt" in captured.out
    assert "01-Jan-2024" in captured.out


# ---------------------------------------------------------------------------
# Coverage: _safe_mtime helper function — broken symlink edge case
# ---------------------------------------------------------------------------

def test_safe_mtime_normal_file(tmp_path):
    """Verify _safe_mtime returns a positive mtime for a normal file."""
    fpath = tmp_path / "normal.txt"
    fpath.write_text("content")

    mtime = _safe_mtime(fpath)
    assert mtime > 0.0


def test_safe_mtime_broken_symlink(tmp_path):
    """
    Verify _safe_mtime returns 0.0 for a broken symlink (OSError path).

    Ref: dirod.py:142-147 — except (OSError, PermissionError) → return 0.0
    """
    target = tmp_path / "nonexistent_target.txt"
    link = tmp_path / "broken_link.txt"
    link.symlink_to(target)  # target doesn't exist → broken symlink

    mtime = _safe_mtime(link)
    assert mtime == 0.0, f"Expected 0.0 for broken symlink, got {mtime}"


def test_safe_mtime_nonexistent_path(tmp_path):
    """Verify _safe_mtime returns 0.0 for a path that doesn't exist."""
    fake = tmp_path / "does_not_exist.txt"
    mtime = _safe_mtime(fake)
    assert mtime == 0.0


# ---------------------------------------------------------------------------
# Coverage: __main__ script execution via subprocess
# ---------------------------------------------------------------------------

def test_dirod_main_no_args(tmp_path, monkeypatch, capsys):
    """
    Verify the module can be executed as a script with no arguments.

    Ref: dirod.py:182-195 — __main__ block that prints listing.
    Uses runpy to execute within the same process so coverage is tracked.
    """
    _create_file(tmp_path, "sample.txt")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["dirod"])

    # runpy executes the __main__ block within this process (coverage-tracked)
    runpy.run_module("mfe_toolbox.utility.dirod", run_name="__main__",
                     alter_sys=True)

    captured = capsys.readouterr()
    assert "sample.txt" in captured.out


def test_dirod_main_with_mask(tmp_path, monkeypatch, capsys):
    """
    Verify the module script accepts a glob pattern argument.

    Ref: dirod.py:193 — mask = sys.argv[1] if len(sys.argv) == 2 else '*'
    """
    _create_file(tmp_path, "data.csv")
    _create_file(tmp_path, "data.txt")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["dirod", "*.csv"])

    runpy.run_module("mfe_toolbox.utility.dirod", run_name="__main__",
                     alter_sys=True)

    captured = capsys.readouterr()
    assert "data.csv" in captured.out
    assert "data.txt" not in captured.out


def test_dirod_main_too_many_args(tmp_path, monkeypatch):
    """
    Verify the module script errors with > 1 argument.

    Ref: dirod.m:28 — MATLAB error('Too many input arguments.')
    Ref: dirod.py:187-191 — Python prints error and exits with code 1.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["dirod", "*.txt", "extra"])

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("mfe_toolbox.utility.dirod", run_name="__main__",
                         alter_sys=True)

    assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# Coverage: Invalid glob pattern (Ref: dirod.py:88-94 except ValueError)
# ---------------------------------------------------------------------------

def test_dirod_invalid_glob_pattern(tmp_path, monkeypatch):
    """
    Verify dirod wraps pathlib ValueError for invalid glob patterns.

    Ref: dirod.py:88-94 — pathlib raises ValueError for certain invalid
    patterns; dirod catches and re-raises with a user-friendly message.
    We monkeypatch Path.glob to force this error path.
    """
    monkeypatch.chdir(tmp_path)

    # Monkeypatch pathlib.Path.glob to simulate an invalid pattern error
    def _bad_glob(self, pattern):
        raise ValueError("embedded null character")

    monkeypatch.setattr(pathlib.Path, "glob", _bad_glob)

    with pytest.raises(ValueError, match="Invalid glob pattern"):
        dirod("anything")


# ---------------------------------------------------------------------------
# Coverage: dirod with broken symlinks in listing
# ---------------------------------------------------------------------------

def test_dirod_with_broken_symlink(tmp_path, monkeypatch):
    """
    Verify dirod handles directories containing broken symlinks gracefully.

    Broken symlinks should sort to the beginning (mtime=0.0) and still
    appear in the listing.
    """
    # Create a normal file and a broken symlink
    _create_file(tmp_path, "normal.txt", "content", mtime=1_600_000_100.0)
    broken_link = tmp_path / "broken.txt"
    broken_link.symlink_to(tmp_path / "nonexistent")

    monkeypatch.chdir(tmp_path)
    result = dirod()

    names = [entry["name"] for entry in result]
    # Broken symlink should sort first (mtime=0.0)
    assert "broken.txt" in names
    assert "normal.txt" in names
    if len(names) == 2:
        assert names[0] == "broken.txt", (
            "Broken symlink should sort first (mtime=0.0)"
        )
