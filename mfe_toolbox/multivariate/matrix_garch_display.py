"""
Matrix GARCH parameter display module.

Displays estimated Matrix GARCH model parameters in a human-readable
H(t) recursion format, showing the intercept CC', symmetric innovation
matrices A, asymmetric matrices G, and lagged covariance matrices B.

Migrated from: multivariate/matrix_garch_display.m
Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 3    Date: 3/10/2011
"""

import numpy as np

from mfe_toolbox.utility.vec2chol import vec2chol


def matrix_garch_display(parameters, p, o, q, k):
    """
    Display Matrix GARCH parameter estimates in a readable H(t) recursion format.

    Unpacks a flat vech-parameterized vector into the symmetric positive-definite
    coefficient matrices C, A(i), G(j), B(l) of a Matrix GARCH model and prints
    the estimated conditional covariance equation::

        H(t) = CC' + A(1)A(1)' .* r(t-1)r(t-1)' + ...
             + G(j)G(j)' .* n(t-j)n(t-j)' + ...
             + B(l)B(l)' .* H(t-l)

    Parameters
    ----------
    parameters : numpy.ndarray
        1-D parameter vector of length ``k*(k+1)/2 * (1 + p + o + q)``, where
        each contiguous block of ``k*(k+1)/2`` elements encodes a lower-triangular
        Cholesky factor via :func:`vec2chol`. The parameter ordering is:
        ``[vech(L_C), vech(L_A1), ..., vech(L_Ap), vech(L_G1), ..., vech(L_Go),
        vech(L_B1), ..., vech(L_Bq)]`` where ``L @ L.T`` produces the full
        symmetric PD matrix for each block.
    p : int
        Number of symmetric innovation lags (must be >= 0).
    o : int
        Number of asymmetric innovation lags (must be >= 0).
    q : int
        Number of lagged conditional covariance terms (must be >= 0).
    k : int
        Number of assets (dimension K of the K x K covariance matrix).

    Returns
    -------
    C : numpy.ndarray
        K x K intercept matrix (CC').
    A : numpy.ndarray
        K x K x P array of symmetric innovation parameter matrices.
        Empty (K x K x 0) when ``p == 0``.
    G : numpy.ndarray
        K x K x O array of asymmetric innovation parameter matrices.
        Empty (K x K x 0) when ``o == 0``.
    B : numpy.ndarray
        K x K x Q array of lagged conditional covariance parameter matrices.
        Empty (K x K x 0) when ``q == 0``.

    Raises
    ------
    ValueError
        If ``p``, ``o``, or ``q`` are not non-negative integers, if ``k`` is
        not a positive integer, or if the length of ``parameters`` does not
        match ``k*(k+1)/2 * (1+p+o+q)``.

    Notes
    -----
    Each block of ``k*(k+1)/2`` parameters is reconstructed via ``vec2chol``
    into a lower-triangular matrix L, and the full symmetric positive-definite
    parameter matrix is formed as ``L @ L.T``.

    The original MATLAB source (``matrix_garch_display.m``, line 57) contains
    a loop-bound typo ``for i=1:(1+p+o+1)`` which should be
    ``for i=1:(1+p+o+q)``. This Python implementation uses the corrected
    bound ``range(1 + p + o + q)``, consistent with the parameter vector
    length validation and subsequent matrix slicing logic.

    References
    ----------
    Ref: multivariate/matrix_garch_display.m — Kevin Sheppard, Revision 3

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.multivariate.matrix_garch_display import matrix_garch_display
    >>> # Minimal example: Matrix GARCH(1,0,1) with k=2
    >>> k_ex = 2
    >>> k2_ex = k_ex * (k_ex + 1) // 2  # 3 elements per block
    >>> np.random.seed(42)
    >>> params = np.abs(np.random.randn(k2_ex * (1 + 1 + 0 + 1)))  # 3 blocks
    >>> C, A, G, B = matrix_garch_display(params, p=1, o=0, q=1, k=2)
    """
    # -----------------------------------------------------------------------
    # Input Validation
    # Ref: matrix_garch_display.m:35-37 — Ensure parameters is a 1-D array
    # -----------------------------------------------------------------------
    parameters = np.asarray(parameters, dtype=np.float64).ravel()

    # Ref: matrix_garch_display.m:38-39 — Validate p, o, q non-negative integers
    p = int(p)
    o = int(o)
    q = int(q)
    if p < 0 or o < 0 or q < 0:
        raise ValueError('P, O, and Q must all be non-negative integers.')

    # Validate k is a positive integer
    k = int(k)
    if k < 1:
        raise ValueError('K must be a positive integer.')

    # Ref: matrix_garch_display.m:42-46 — Validate parameter vector length
    num_matrices = 1 + p + o + q
    k2 = k * (k + 1) // 2  # Number of parameters per Cholesky factor block

    # Ref: matrix_garch_display.m:52 — Verify k is consistent with k2 via
    # k = floor(sqrt(2*k2)). Uses np.sqrt and np.floor per schema requirements.
    k_check = int(np.floor(np.sqrt(2.0 * k2)))
    if k_check != k:
        raise ValueError(
            f'Inconsistent k={k}: floor(sqrt(2*k2))={k_check} does not match.'
        )

    expected_len = k2 * num_matrices
    if len(parameters) != expected_len:
        raise ValueError(
            f'PARAMETERS does not have the expected number of elements: '
            f'K*(K+1)/2*(1+P+O+Q) = {expected_len}, got {len(parameters)}.'
        )

    # -----------------------------------------------------------------------
    # Reconstruct parameter matrices from Cholesky factor blocks
    # Ref: matrix_garch_display.m:55-61
    # NOTE: The MATLAB source line 57 has a typo: loop bound is (1+p+o+1)
    # instead of (1+p+o+q). We use the corrected range(num_matrices).
    # -----------------------------------------------------------------------
    parameter_matrices = np.zeros((k, k, num_matrices))
    index = 0
    for i in range(num_matrices):
        # Ref: matrix_garch_display.m:58 — vec2chol converts a k*(k+1)/2-length
        # vector into a k x k lower-triangular Cholesky factor L
        temp = vec2chol(parameters[index:index + k2])
        # Ref: matrix_garch_display.m:59 — Form symmetric PD matrix as L @ L.T
        parameter_matrices[:, :, i] = temp @ temp.T
        index += k2

    # -----------------------------------------------------------------------
    # Extract C, A, G, B slices from the 3-D parameter_matrices array
    # Ref: matrix_garch_display.m:63-66
    # -----------------------------------------------------------------------
    # Ref: matrix_garch_display.m:63 — C = parameterMatrices(:,:,1)
    # MATLAB 1-indexed → Python 0-indexed
    C = parameter_matrices[:, :, 0]

    # Ref: matrix_garch_display.m:64 — A = parameterMatrices(:,:,2:p+1)
    # MATLAB 2:p+1 inclusive → Python 1:1+p
    A = parameter_matrices[:, :, 1:1 + p]

    # Ref: matrix_garch_display.m:65 — G = parameterMatrices(:,:,2+p:1+p+o)
    # MATLAB (2+p):(1+p+o) inclusive → Python (1+p):(1+p+o)
    G = parameter_matrices[:, :, 1 + p:1 + p + o]

    # Ref: matrix_garch_display.m:66 — B = parameterMatrices(:,:,2+p+o:1+p+o+q)
    # MATLAB (2+p+o):(1+p+o+q) inclusive → Python (1+p+o):(1+p+o+q)
    B = parameter_matrices[:, :, 1 + p + o:1 + p + o + q]

    # -----------------------------------------------------------------------
    # Display formatting
    # Ref: matrix_garch_display.m:68-83
    # -----------------------------------------------------------------------
    # Ref: matrix_garch_display.m:68 — Middle row for "H(t) =" and "+"
    # MATLAB: k1 = ceil(k/2) is a 1-based row index; Python 0-based equivalent
    k1_matlab = int(np.ceil(k / 2))  # 1-based middle row
    mid_row = k1_matlab - 1  # 0-based middle row index

    def _format_matrix_rows(mat):
        """Format a k x k matrix into k row strings with aligned numbers.

        Mimics MATLAB's num2str() which uses approximately 4-5 significant digits
        for display. Uses fixed-point format with 4 decimal places in a 10-character
        wide field for consistent column alignment.

        Parameters
        ----------
        mat : numpy.ndarray
            k x k matrix to format.

        Returns
        -------
        list of str
            k strings, one per row of the matrix.
        """
        rows = []
        for row_idx in range(k):
            vals = []
            for col_idx in range(k):
                val = mat[row_idx, col_idx]
                vals.append(f'{val:10.4f}')
            rows.append(' '.join(vals))
        return rows

    # Build output lines row by row
    output_lines = []
    for row_idx in range(k):
        parts = []

        # Ref: matrix_garch_display.m:69 — "H(t) =" label on middle row
        if row_idx == mid_row:
            parts.append('H(t) =')
        else:
            parts.append('      ')

        # Ref: matrix_garch_display.m:70 — Intercept C matrix with brackets
        c_rows = _format_matrix_rows(C)
        parts.append(f' [{c_rows[row_idx]}]')

        # Ref: matrix_garch_display.m:72-75 — Symmetric A matrices with annotations
        for i in range(p):
            # Ref: matrix_garch_display.m:71 — "+" on middle row, space otherwise
            if row_idx == mid_row:
                parts.append(' +')
            else:
                parts.append('  ')

            a_rows = _format_matrix_rows(A[:, :, i])
            parts.append(f' [{a_rows[row_idx]}]')

            # Ref: matrix_garch_display.m:74 — Annotation ".*r(t-i)r(t-i)'"
            # MATLAB pads non-middle rows with repmat(' ',_,15) spaces
            lag_str = str(i + 1)  # MATLAB 1-based lag number
            annotation = f".*r(t-{lag_str})r(t-{lag_str})'"
            ann_width = len(annotation)
            if row_idx == mid_row:
                parts.append(annotation)
            else:
                parts.append(' ' * ann_width)

        # Ref: matrix_garch_display.m:76-79 — Asymmetric G matrices with annotations
        for i in range(o):
            if row_idx == mid_row:
                parts.append(' +')
            else:
                parts.append('  ')

            g_rows = _format_matrix_rows(G[:, :, i])
            parts.append(f' [{g_rows[row_idx]}]')

            # Ref: matrix_garch_display.m:78 — Annotation ".*n(t-i)n(t-i)'"
            lag_str = str(i + 1)
            annotation = f".*n(t-{lag_str})n(t-{lag_str})'"
            ann_width = len(annotation)
            if row_idx == mid_row:
                parts.append(annotation)
            else:
                parts.append(' ' * ann_width)

        # Ref: matrix_garch_display.m:80-83 — Lagged covariance B matrices
        for i in range(q):
            if row_idx == mid_row:
                parts.append(' +')
            else:
                parts.append('  ')

            b_rows = _format_matrix_rows(B[:, :, i])
            parts.append(f' [{b_rows[row_idx]}]')

            # Ref: matrix_garch_display.m:82 — Annotation ".*H(t-i)"
            lag_str = str(i + 1)
            annotation = f'.*H(t-{lag_str})'
            ann_width = len(annotation)
            if row_idx == mid_row:
                parts.append(annotation)
            else:
                parts.append(' ' * ann_width)

        output_lines.append(''.join(parts))

    # Ref: matrix_garch_display.m:85 — Print the formatted display
    for line in output_lines:
        print(line)

    # Ref: matrix_garch_display.m:86-90 — Print conditional note about RC terms
    if o > 0:
        print("Note: r(t-i)r(t-i)' is the RC, and n(t-i)n(t-i)' is the "
              "asymmetric RC term, if the model used realized measures.")
    else:
        print("Note: r(t-i)r(t-i)' is the RC if the model used "
              "realized measures.")

    return C, A, G, B
