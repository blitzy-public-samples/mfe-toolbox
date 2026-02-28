"""
Formatted matrix printer with configurable row/column labels, format strings,
wrapping, and file output.

Migrated from utility/mprint.m (~299 lines) — Originally by James P. LeSage,
Dept of Economics, University of Toledo.

This module provides the :func:`mprint` function, which prints a 2-D numpy
array in formatted tabular form.  Features include per-column format strings,
column and row labels, automatic column wrapping for wide matrices, optional
row numbering, and output redirection to any file-like object.

Notes
-----
MATLAB-to-Python translation rules applied:
- ``fprintf(fid, ...)`` → ``fid.write(...)``
- ``strvcat`` → list of strings
- MATLAB ``struct`` → Python ``dict``
- ``fid = 1`` → ``sys.stdout``
- 1-indexed ``begr``/``endr``/``begc``/``endc`` → 0-indexed
- ``size(y)`` → ``y.shape``
- ``strjust(s, 'right')`` → Python ``%Ns`` format (right-justifies by default)
- ``error('msg')`` → ``raise ValueError('msg')``
"""

import sys

import numpy as np


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------

def _parse_format_string(fmt_str):
    """
    Parse a C-style format string to extract field width, decimal precision,
    and format type (integer vs float).

    Ref: mprint.m:105-122 — MATLAB uses ``strtok`` to parse format components.

    Parameters
    ----------
    fmt_str : str
        A C-style format string such as ``'%10.4f'``, ``'%12d'``,
        ``'%14.6e'``, or ``'%10.4g'``.

    Returns
    -------
    field_width : int
        Total field width extracted from the format string.
    decimal : str
        Decimal precision as a string (e.g. ``'4'``).  Empty string for
        integer (``'d'``) formats.
    is_float : bool
        ``True`` if the format represents a floating-point type
        (``f``, ``e``, ``E``, ``g``, ``G``).
    is_integer : bool
        ``True`` if the format is an integer (``d``) type.
    """
    s = fmt_str.strip()
    # Strip leading '%'
    # Ref: mprint.m:106 — strtok(fmt, '%') removes '%' prefix
    if s.startswith('%'):
        s = s[1:]
    s = s.rstrip()  # Remove any trailing whitespace

    if not s:
        # Degenerate case — return safe defaults
        return 10, '4', True, False

    last_char = s[-1]

    # ------------------------------------------------------------------
    # Integer format  (e.g. '12d' from '%12d')
    # Ref: mprint.m:108-111
    # ------------------------------------------------------------------
    if last_char == 'd':
        body = s[:-1]
        try:
            field_width = int(body) if body else 10
        except ValueError:
            field_width = 10
        return field_width, '', False, True

    # ------------------------------------------------------------------
    # Float format  (e.g. '10.4f' from '%10.4f')
    # Ref: mprint.m:113-122
    # ------------------------------------------------------------------
    if last_char == 'f':
        body = s[:-1]  # e.g. '10.4'
        if '.' in body:
            parts = body.split('.', 1)
            try:
                field_width = int(parts[0]) if parts[0] else 10
            except ValueError:
                field_width = 10
            decimal = parts[1] if parts[1] else '4'
        else:
            try:
                field_width = int(body) if body else 10
            except ValueError:
                field_width = 10
            decimal = '0'
        return field_width, decimal, True, False

    # ------------------------------------------------------------------
    # Scientific / general formats  (e, E, g, G)
    # These are handled analogously to 'f'.
    # ------------------------------------------------------------------
    if last_char in ('e', 'E', 'g', 'G'):
        body = s[:-1]
        if '.' in body:
            parts = body.split('.', 1)
            try:
                field_width = int(parts[0]) if parts[0] else 12
            except ValueError:
                field_width = 12
            decimal = parts[1] if parts[1] else '4'
        else:
            try:
                field_width = int(body) if body else 12
            except ValueError:
                field_width = 12
            decimal = '4'
        return field_width, decimal, True, False

    # ------------------------------------------------------------------
    # Unknown format — attempt to interpret body as a width number
    # ------------------------------------------------------------------
    try:
        field_width = int(s)
    except ValueError:
        field_width = 10
    return field_width, '4', True, False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def mprint(y, info=None):
    """
    Print an (nobs × nvar) matrix in formatted tabular form.

    Faithfully migrated from ``mprint.m`` by James P. LeSage, University of
    Toledo.  All MATLAB 1-indexed defaults have been converted to Python
    0-indexed equivalents.

    Parameters
    ----------
    y : numpy.ndarray
        Matrix to print.  1-D arrays are promoted to 2-D via
        :func:`numpy.atleast_2d`.
    info : dict or None, optional
        Printing options (all keys are optional):

        begr : int
            Beginning row, **0-indexed** (default ``0``).
        endr : int
            Ending row, **0-indexed** (default ``nobs - 1``).
        begc : int
            Beginning column, **0-indexed** (default ``0``).
        endc : int
            Ending column, **0-indexed** (default ``nvar - 1``).
        cnames : list of str
            Column name strings.  Length **must** equal ``nvar``.
        rnames : list of str
            Row name strings.  The first element is the header label;
            subsequent elements label each row.  Length **must** equal
            ``nobs + 1``.
        fmt : str or list of str
            C-style format string, e.g. ``'%10.4f'`` (the default).
            May also be a *list* of format strings with length ``nvar``
            (one format per column).
        fid : file-like
            Writable file object for output (default ``sys.stdout``).
        rflag : int
            ``1`` to display row numbers, ``0`` for none (default ``0``).
            Automatically overridden to ``0`` when *rnames* is provided.
        width : int
            Maximum line width in characters before column-wrapping is
            triggered (default ``80``).

    Raises
    ------
    ValueError
        If *info* is not a ``dict``, or if the length of *cnames* / *rnames*
        does not match the matrix dimensions.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> mprint(y)
        1.0000    2.0000
        3.0000    4.0000
    <BLANKLINE>
    >>> info = {'cnames': ['col1', 'col2'],
    ...         'rnames': ['Rows', 'r1', 'r2'],
    ...         'fmt': '%12.4f'}
    >>> mprint(y, info)
    Rows         col1         col2
      r1       1.0000       2.0000
      r2       3.0000       4.0000
    <BLANKLINE>
    """
    # ------------------------------------------------------------------
    # Ensure *y* is a 2-D numpy array
    # Ref: mprint.m:48 — [nobs nvars] = size(y)
    # ------------------------------------------------------------------
    if not isinstance(y, np.ndarray):
        y = np.asarray(y, dtype=float)
    y = np.atleast_2d(y)
    if y.dtype.kind not in ('f', 'i', 'u', 'c'):
        y = y.astype(float)
    nobs, nvars = y.shape

    # ------------------------------------------------------------------
    # Setup defaults
    # Ref: mprint.m:47-49 — MATLAB defaults (1-indexed) → Python 0-indexed
    # ------------------------------------------------------------------
    fid = sys.stdout     # Ref: mprint.m:47 — MATLAB fid=1 is stdout
    rflag = 0            # Internal: 1 when rnames is provided
    cflag = 0            # Internal: 1 when cnames is provided
    rnum = 0             # Row-number display (from info['rflag'])
    nfmts = 1            # Number of format strings
    cwidth = 80          # Wrap width
    begr = 0             # Ref: mprint.m:49 — MATLAB begr=1 → Python 0
    endr = nobs - 1
    begc = 0             # Ref: mprint.m:49 — MATLAB begc=1 → Python 0
    endc = nvars - 1
    fmt = '%10.4f'       # Default format string
    cnames = None
    rnames = None

    # ------------------------------------------------------------------
    # Parse *info* dict
    # Ref: mprint.m:50-95
    # ------------------------------------------------------------------
    if info is not None:
        if not isinstance(info, dict):
            raise ValueError(
                'mprint: you must supply the options as a structure variable'
            )

        # --- format string(s) ---
        # Ref: mprint.m:59-68
        if 'fmt' in info:
            fmts_val = info['fmt']
            if isinstance(fmts_val, (list, tuple)):
                nfmts = len(fmts_val)
                if nfmts == nvars:
                    fmt = list(fmts_val)
                elif nfmts == 1:
                    fmt = fmts_val[0]
                    nfmts = 1
                else:
                    raise ValueError(
                        'mprint: wrong # of formats in string -- need nvar'
                    )
            else:
                fmt = str(fmts_val)
                nfmts = 1

        # --- file descriptor ---
        # Ref: mprint.m:69-70
        if 'fid' in info:
            fid = info['fid']

        # --- column / row range (0-indexed) ---
        # Ref: mprint.m:71-78
        if 'begc' in info:
            begc = int(info['begc'])
        if 'begr' in info:
            begr = int(info['begr'])
        if 'endc' in info:
            endc = int(info['endc'])
        if 'endr' in info:
            endr = int(info['endr'])

        # --- wrap width ---
        # Ref: mprint.m:79-80
        if 'width' in info:
            cwidth = int(info['width'])

        # --- column names ---
        # Ref: mprint.m:81-83 — MATLAB strvcat → Python list of strings
        if 'cnames' in info:
            cnames = list(info['cnames'])
            cflag = 1

        # --- row names ---
        # Ref: mprint.m:84-86
        if 'rnames' in info:
            rnames = list(info['rnames'])
            rflag = 1

        # --- row-number flag ---
        # Ref: mprint.m:87-88
        if 'rflag' in info:
            rnum = int(info['rflag'])

    # ------------------------------------------------------------------
    # Row-name / row-number conflict resolution
    # Ref: mprint.m:98-102 — if rnames provided, suppress row numbers
    # ------------------------------------------------------------------
    if rflag == 1:
        rnum = 0

    # ------------------------------------------------------------------
    # Parse format string(s) → extract field widths & types
    # Ref: mprint.m:104-159
    # ------------------------------------------------------------------
    if nfmts == 1:
        # --- Single format string ---
        # Ref: mprint.m:105-126
        f2, decimal, fflag, dflag = _parse_format_string(fmt)
        nwide = cwidth // f2 if f2 > 0 else 8
        if nwide < 1:
            nwide = 1
        nvar = endc - begc + 1
        # Ref: mprint.m:126 — nsets = ceil(nvar/nwide)
        nsets = int(np.ceil(nvar / nwide)) if nwide > 0 else 1
    else:
        # --- Multiple format strings (one per column) ---
        # Ref: mprint.m:127-159
        f2v = []
        dflagv = []
        fflagv = []
        decimalv = []
        nwidev = []
        nsetsv = []
        for ii in range(nfmts):
            fw, dec, ff, df = _parse_format_string(fmt[ii])
            f2v.append(fw)
            dflagv.append(df)
            fflagv.append(ff)
            decimalv.append(dec)
            nw = cwidth // fw if fw > 0 else 8
            if nw < 1:
                nw = 1
            nwidev.append(nw)
            nvar_local = endc - begc + 1
            ns = int(np.ceil(nvar_local / nw)) if nw > 0 else 1
            nsetsv.append(ns)
        # Ref: mprint.m:157-158
        nsets = min(nsetsv) if nsetsv else 1
        nwide = max(nwidev) if nwidev else 8

    # ------------------------------------------------------------------
    # Build format strings for headers and data output
    # Ref: mprint.m:161-238
    # ------------------------------------------------------------------
    dstr = ''
    if rnum == 1:
        # Ref: mprint.m:165-167
        dstr = 'Obs#'

    # Initialise placeholders — will be populated below
    sfmt = ''
    ffmt = ''
    sfmtv = []
    ffmtv = []
    rfmt = ''

    if cflag == 1:
        # We have column headings
        # Ref: mprint.m:169-201
        if len(cnames) != nvars:
            raise ValueError('Wrong # cnames in mprint')

        # Maximum column-name width
        # Ref: mprint.m:170 — [vsize nsize] = size(cnames)
        nsize = max(len(name) for name in cnames) if cnames else 0

        if nfmts == 1:
            # Single format — adjust width to max(field_width, name_width)
            # Ref: mprint.m:172-184
            nmax = max(f2, nsize)
            sfmt = '%' + str(nmax) + 's '
            if dflag:
                ffmt = '%' + str(nmax) + 'd '
            elif fflag:
                ffmt = '%' + str(nmax) + '.' + decimal + 'f '
            else:
                ffmt = '%' + str(nmax) + 's '
        else:
            # Multiple formats — per-column format strings
            # Ref: mprint.m:185-201
            sfmtv = []
            ffmtv = []
            for ii in range(nfmts):
                nmax = max(f2v[ii], nsize)
                sfmtv.append('%' + str(nmax) + 's ')
                if dflagv[ii]:
                    ffmtv.append('%' + str(nmax) + 'd ')
                elif fflagv[ii]:
                    ffmtv.append(
                        '%' + str(nmax) + '.' + decimalv[ii] + 'f '
                    )
                else:
                    ffmtv.append('%' + str(nmax) + 's ')

    else:
        # No column headings
        # Ref: mprint.m:202-227
        if nfmts == 1:
            # Ref: mprint.m:203-212 — augment format with trailing space
            nmax = f2
            if dflag:
                ffmt = '%' + str(nmax) + 'd '
            elif fflag:
                ffmt = '%' + str(nmax) + '.' + decimal + 'f '
            else:
                ffmt = fmt + ' '
        else:
            # Ref: mprint.m:213-226 — per-column with trailing space
            ffmtv = []
            for ii in range(nfmts):
                nmax_ii = f2v[ii]
                if dflagv[ii]:
                    ffmtv.append('%' + str(nmax_ii) + 'd ')
                elif fflagv[ii]:
                    ffmtv.append(
                        '%' + str(nmax_ii) + '.' + decimalv[ii] + 'f '
                    )
                else:
                    ffmtv.append(fmt[ii] + ' ')

    if rflag == 1:
        # We have row labels
        # Ref: mprint.m:229-234
        if len(rnames) != nobs + 1:
            raise ValueError('Wrong # rnames in mprint')
        nsize_r = max(len(name) for name in rnames) if rnames else 0
        rfmt = '%' + str(nsize_r) + 's '

    # Fallback: no row labels and no column labels → use raw format
    # Ref: mprint.m:236-238
    if rflag == 0 and cflag == 0 and nfmts == 1:
        ffmt = fmt

    # ------------------------------------------------------------------
    # Print the matrix
    # Ref: mprint.m:241-298
    # ------------------------------------------------------------------
    for j in range(nsets):
        # Column range for this wrapped section
        # Ref: mprint.m:248 — MATLAB: (j-1)*nwide+begc : j*nwide+begc-1
        col_start = j * nwide + begc
        col_end = min((j + 1) * nwide + begc - 1, endc)

        # ----- Column headers -----
        if nfmts == 1:
            # Ref: mprint.m:242-260
            if rnum == 1:
                fid.write('%5s ' % dstr)
            elif rflag == 1:
                fid.write(rfmt % rnames[0])

            if cflag == 1:
                for i in range(col_start, col_end + 1):
                    # Ref: mprint.m:253 — strjust(cnames(i,:),'right')
                    # Python '%Ns' right-justifies by default
                    fid.write(sfmt % cnames[i])
            fid.write('\n')
        else:
            # Multiple formats
            # Ref: mprint.m:261-280
            if rnum == 1:
                fid.write('%5s ' % dstr)
            elif rflag == 1:
                fid.write(rfmt % rnames[0])

            if cflag == 1:
                for i in range(col_start, col_end + 1):
                    # Ref: mprint.m:272 — sfmtv{i}
                    fid.write(sfmtv[i] % cnames[i])
            fid.write('\n')

        # ----- Data rows -----
        # Ref: mprint.m:281-296
        for k in range(begr, endr + 1):
            # Row number or row label
            if rnum == 1:
                # Ref: mprint.m:282 — MATLAB prints k (1-indexed)
                # Python: print k+1 for 1-based display
                fid.write('%5d ' % (k + 1))
            elif rflag == 1:
                # Ref: mprint.m:284 — rnames(k+1,:) (MATLAB 1-indexed)
                # Python: rnames[k+1] since k is 0-indexed
                fid.write(rfmt % rnames[k + 1])

            # Data columns for this wrapped section
            for col in range(col_start, col_end + 1):
                # Ref: mprint.m:288-292
                if nfmts == 1:
                    fid.write(ffmt % y[k, col])
                else:
                    fid.write(ffmtv[col] % y[k, col])

            fid.write('\n')

        # Trailing newline between wrapped sections
        # Ref: mprint.m:297
        fid.write('\n')
