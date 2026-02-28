"""
Date-sorted directory listing utility.

Migrated from utility/dirod.m (49 lines) — provides date-sorted directory
listing similar to the Windows command ``dir /od``, using ``pathlib.Path``
instead of MATLAB ``dir()``.

This is a pure filesystem utility with no numerical dependencies. It uses
only Python standard library modules (pathlib, datetime).

Author: Kevin Sheppard (original MATLAB, Revision 1, 6/7/2010)
"""

import pathlib
import datetime


def dirod(filenamemask: str = '*') -> list[dict]:
    """
    Date-sorted directory listing.

    Lists files and directories matching a glob pattern in the current
    working directory, sorted by modification date in ascending order.
    Similar to the Windows command ``dir /od``.

    Parameters
    ----------
    filenamemask : str, optional
        File name mask (glob pattern), e.g. ``'*'``, ``'*.m'``, ``'d*.m'``.
        Default is ``'*'``.

    Returns
    -------
    dir_data : list[dict]
        List of dicts sorted by modification date (ascending), each with keys:

        - ``'name'`` (str): File or directory name (basename only).
        - ``'date'`` (str): Human-readable modification date string in the
          format ``'DD-Mon-YYYY HH:MM:SS'`` (e.g. ``'07-Jun-2010 14:32:01'``).

    Raises
    ------
    ValueError
        If ``filenamemask`` is not a string.

    Notes
    -----
    - Ref: dirod.m:23-29 — MATLAB ``varargin`` with 0 or 1 args is replaced
      by Python default parameter ``filenamemask='*'``.
    - Ref: dirod.m:28 — MATLAB ``error('Too many input arguments.')`` maps to
      Python ``raise ValueError(...)`` for invalid argument types.
    - Ref: dirod.m:31 — MATLAB ``dir(arg)`` is replaced by
      ``pathlib.Path('.').glob(filenamemask)`` per AAP instruction.
    - Ref: dirod.m:32 — MATLAB sorts by ``datenum``; Python sorts by
      ``os.stat_result.st_mtime`` which is equivalent (seconds since epoch).
    - Ref: dirod.m:40-48 — MATLAB ``nargout==0`` determines print vs return.
      In Python, the function always returns data; callers print if desired.
    - Ref: dirod.m:47 — MATLAB returns a struct array with ``.name`` and
      ``.date`` fields; Python returns ``list[dict]`` with ``'name'`` and
      ``'date'`` keys.

    Examples
    --------
    List all files in the current directory sorted by date:

    >>> result = dirod()  # doctest: +SKIP
    >>> for entry in result:
    ...     print(f"{entry['name']:30s} {entry['date']}")

    List only Python files:

    >>> result = dirod('*.py')  # doctest: +SKIP
    """
    # Ref: dirod.m:23-29 — Argument validation
    # MATLAB uses varargin with length check and error() for > 1 arg.
    # Python enforces type via explicit check since the function signature
    # already restricts to a single parameter with a default value.
    if not isinstance(filenamemask, str):
        # Ref: dirod.m:28 — MATLAB error('Too many input arguments.') → ValueError
        raise ValueError(
            'filenamemask must be a string glob pattern '
            '(e.g. \'*\', \'*.m\', \'d*.m\')'
        )

    # Ref: dirod.m:31 — MATLAB dir(arg) → pathlib.Path('.').glob(filenamemask)
    # AAP: "Replace MATLAB dir with pathlib.Path listing"
    # pathlib.Path.glob returns a generator; materialize to list for sorting.
    try:
        entries = list(pathlib.Path('.').glob(filenamemask))
    except ValueError as exc:
        # pathlib raises ValueError for invalid glob patterns (e.g. embedded NUL)
        raise ValueError(
            f'Invalid glob pattern: {filenamemask!r}'
        ) from exc

    # Ref: dirod.m:32 — [~,index]=sort([dirData.datenum]); dirData = dirData(index);
    # MATLAB sorts by datenum (modification time as serial date number).
    # Python equivalent: sort by st_mtime (modification time in seconds since epoch).
    # Ascending order matches MATLAB's default sort behavior.
    sorted_entries = sorted(
        entries,
        key=lambda entry: _safe_mtime(entry)
    )

    # Ref: dirod.m:40-48 — Build output structure
    # MATLAB struct array with .name and .date fields → Python list[dict]
    # with 'name' and 'date' keys.
    dir_data: list[dict] = []
    for entry in sorted_entries:
        # Ref: dirod.m:43 — MATLAB dirData(i).date provides human-readable
        # date string; Python uses datetime.fromtimestamp + strftime.
        mtime = _safe_mtime(entry)
        mod_time = datetime.datetime.fromtimestamp(mtime)
        # Format matches MATLAB's default date display: 'DD-Mon-YYYY HH:MM:SS'
        date_str = mod_time.strftime('%d-%b-%Y %H:%M:%S')
        dir_data.append({
            'name': entry.name,
            'date': date_str,
        })

    return dir_data


def _safe_mtime(path: pathlib.Path) -> float:
    """
    Safely retrieve modification time for a path entry.

    Handles edge cases where stat() might fail (e.g., broken symlinks,
    permission errors) by returning 0.0 so the entry sorts to the
    beginning of the list rather than crashing the entire listing.

    Parameters
    ----------
    path : pathlib.Path
        Filesystem path to query.

    Returns
    -------
    mtime : float
        Modification time in seconds since epoch, or 0.0 on error.
    """
    try:
        return path.stat().st_mtime
    except (OSError, PermissionError):
        # Gracefully handle broken symlinks, permission denied, etc.
        # Sort these entries to the beginning (oldest).
        return 0.0


def _print_listing(dir_data: list[dict]) -> None:
    """
    Print a formatted directory listing to stdout.

    Mirrors the display behavior of the MATLAB ``dirod`` function when
    called with ``nargout==0`` (Ref: dirod.m:40-45).

    Parameters
    ----------
    dir_data : list[dict]
        Directory listing as returned by :func:`dirod`.
    """
    if not dir_data:
        print(' ')
        return

    # Ref: dirod.m:35-38 — Compute max filename length for column alignment
    max_name_len = max(len(entry['name']) for entry in dir_data)

    # Ref: dirod.m:41-45 — Formatted display with aligned columns
    print(' ')
    for entry in dir_data:
        name = entry['name']
        date = entry['date']
        # Ref: dirod.m:43 — MATLAB pads with repmat(' ',1,fileNameLen-length(name)+2)
        padding = ' ' * (max_name_len - len(name) + 2)
        print(f'{name}{padding}{date}')
    print(' ')


# Ref: dirod.m:40 — MATLAB nargout==0 branch prints to stdout.
# When this module is run directly as a script, print the listing.
if __name__ == '__main__':
    import sys

    # Support optional glob pattern from command line:
    #   python -m mfe_toolbox.utility.dirod '*.py'
    if len(sys.argv) > 2:
        # Ref: dirod.m:28 — MATLAB error('Too many input arguments.')
        print('Error: Too many input arguments. 0 or 1 input only.',
              file=sys.stderr)
        sys.exit(1)

    mask = sys.argv[1] if len(sys.argv) == 2 else '*'
    listing = dirod(mask)
    _print_listing(listing)
