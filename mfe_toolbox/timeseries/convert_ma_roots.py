"""
MA polynomial root conversion for invertibility enforcement.

Migrated from: timeseries/convert_ma_roots.m (MFE Toolbox v4.0)

This module provides the convert_ma_roots function which inspects an MA
(Moving Average) polynomial parameterization for invertibility. If any roots
lie outside the unit circle, the function reflects them inside via the
reciprocal map (1/z), then reconstructs the polynomial. For irregular
(sparse) MA specifications, a warning is issued when inversion would
change the lag index structure.

Functions
---------
convert_ma_roots
    Checks an MA parameterization for invertibility and inverts it if possible.
"""

import numpy as np
import warnings


def convert_ma_roots(parameters, q):
    """
    Check an MA parameterization for invertibility and invert it if possible.

    Inspects the roots of an MA polynomial. If any roots lie outside the unit
    circle (|root| > 1), they are reflected inside via the reciprocal mapping
    root_new = 1/root. The polynomial is then reconstructed from the modified
    roots. For irregular (sparse) MA specifications where the lag indices are
    not contiguous, a warning is issued if inversion would change the required
    lag indices, and the original parameters are returned unchanged.

    Parameters
    ----------
    parameters : array_like
        MA parameter values corresponding to the lag indices specified in ``q``.
        Length must match ``len(q)``.
    q : array_like of int
        1-based lag indices of the MA parameters. For a standard MA(3), this
        would be ``[1, 2, 3]``. For a sparse MA with lags 1 and 4 only, this
        would be ``[1, 4]``. Values must be positive integers.

    Returns
    -------
    numpy.ndarray
        Converted (invertible) MA parameters as a 1-D array. If inversion is
        not possible without changing the lag structure, the original
        parameters are returned unchanged.

    Raises
    ------
    ValueError
        If ``parameters`` is empty, ``q`` is empty, elements of ``q`` are
        non-positive, or the lengths of ``parameters`` and ``q`` do not match.

    Warnings
    --------
    UserWarning
        Issued when an irregular MA polynomial cannot be inverted without
        changing the lag indices. In this case the original parameters are
        returned.

    Notes
    -----
    The invertibility condition for an MA(q) process requires all roots of
    the MA lag polynomial ``1 + theta_1*z + theta_2*z^2 + ... + theta_q*z^q``
    to lie inside the unit circle. Roots outside the unit circle are reflected
    via the mapping ``z_new = 1/z``, which preserves the modulus relationship
    while placing the root inside the unit circle.

    For polynomials with real coefficients, complex roots appear in conjugate
    pairs. The reciprocal reflection preserves this conjugate-pair property,
    ensuring the reconstructed polynomial retains real coefficients.

    References
    ----------
    .. [1] Hamilton, J.D. (1994). *Time Series Analysis*. Princeton University
       Press. Chapter 4: MA invertibility conditions.

    Examples
    --------
    >>> import numpy as np
    >>> from mfe_toolbox.timeseries.convert_ma_roots import convert_ma_roots
    >>> # Standard MA(2) with roots outside unit circle
    >>> params = np.array([0.5, 0.3])
    >>> q = np.array([1, 2])
    >>> result = convert_ma_roots(params, q)
    >>> result.shape
    (2,)
    """
    # -------------------------------------------------------------------------
    # Input validation
    # -------------------------------------------------------------------------
    parameters = np.asarray(parameters, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.int64).ravel()

    if parameters.size == 0:
        raise ValueError("'parameters' must not be empty.")
    if q.size == 0:
        raise ValueError("'q' must not be empty.")
    if np.any(q < 1):
        raise ValueError(
            "All elements of 'q' must be positive integers (1-based lag indices)."
        )
    if parameters.size != q.size:
        raise ValueError(
            f"Length of 'parameters' ({parameters.size}) must match length of "
            f"'q' ({q.size})."
        )

    # -------------------------------------------------------------------------
    # Build full MA parameter vector and extract polynomial roots
    # Ref: convert_ma_roots.m:10-12
    # -------------------------------------------------------------------------
    # MATLAB: MAparameters = zeros(1, max(q));
    max_q = int(np.max(q))
    ma_parameters = np.zeros(max_q, dtype=np.float64)

    # Ref: convert_ma_roots.m:11 — MATLAB 1-indexed; Python 0-indexed
    # Place the provided parameters at their respective lag positions
    ma_parameters[q - 1] = parameters

    # Ref: convert_ma_roots.m:12 — Build characteristic polynomial [1, a1, a2, ..., a_max_q]
    # and find its roots
    poly_coeffs = np.concatenate(([1.0], ma_parameters))
    roots_lambda = np.roots(poly_coeffs)

    # -------------------------------------------------------------------------
    # Reflect roots outside the unit circle
    # Ref: convert_ma_roots.m:13 — lambda(abs(lambda)>1) = 1./lambda(abs(lambda)>1)
    # For root z with |z|>1, the reflected root is 1/z which has |1/z|<1
    # -------------------------------------------------------------------------
    outside_mask = np.abs(roots_lambda) > 1.0
    if np.any(outside_mask):
        roots_lambda[outside_mask] = 1.0 / roots_lambda[outside_mask]

    # -------------------------------------------------------------------------
    # Reconstruct polynomial from (possibly modified) roots
    # Ref: convert_ma_roots.m:14 — parameters = poly(lambda)
    # np.poly returns coefficients in descending power order with leading 1,
    # matching MATLAB's poly() behavior exactly
    # -------------------------------------------------------------------------
    converted_poly = np.real(np.poly(roots_lambda))

    # Ref: convert_ma_roots.m:15 — Zero out near-zero coefficients
    # MATLAB: parameters(abs(parameters)<1e-10) = 0
    converted_poly[np.abs(converted_poly) < 1e-10] = 0.0

    # -------------------------------------------------------------------------
    # Handle sparse vs. dense MA lag structures
    # Ref: convert_ma_roots.m:17-29
    # -------------------------------------------------------------------------
    if len(q) < max_q:
        # Irregular (sparse) MA — check if inversion changed the lag structure
        # Ref: convert_ma_roots.m:19 — newq = find(abs(parameters)>1e-10)
        # np.where returns 0-indexed positions in the polynomial array;
        # position 0 is the constant (leading 1), position k is lag k.
        # This means the 0-indexed positions after removing the constant
        # ARE the 1-based lag indices directly.
        nonzero_indices = np.where(np.abs(converted_poly) > 1e-10)[0]

        # Ref: convert_ma_roots.m:20 — newq = newq(2:length(newq))-1
        # Remove the constant term index, remaining indices are the lag numbers.
        # In MATLAB: subtract 1 from 1-based polynomial indices to get 1-based lags.
        # In Python: 0-based polynomial positions after the constant = lag numbers.
        new_q = nonzero_indices[1:]  # Remove constant term position (index 0)

        # Ref: convert_ma_roots.m:21-26 — Compare old and new lag structures
        if len(q) != len(new_q) or np.any(q != new_q):
            # Ref: convert_ma_roots.m:22-23 — Issue warning, return original params
            lag_str = " ".join(str(int(x)) for x in new_q)
            warnings.warn(
                "The irregular MA cannot be inverted without changing the lag "
                f"indices. The required lag indices are: {lag_str}",
                stacklevel=2,
            )
            # Return the original (non-inverted) parameters
            result = ma_parameters[q - 1].copy()
        else:
            # Ref: convert_ma_roots.m:25 — parameters = parameters(q+1)
            # In MATLAB, q+1 converts 1-based lag indices to 1-based polynomial
            # positions. In Python, q values (1-based lags) directly index the
            # 0-based polynomial array (position k = lag k).
            result = converted_poly[q].copy()
    else:
        # Ref: convert_ma_roots.m:28 — Dense MA: parameters = parameters(q+1)
        # All lags from 1 to max_q are present
        result = converted_poly[q].copy()

    # -------------------------------------------------------------------------
    # Ensure 1-D output (column vector equivalent in Python)
    # Ref: convert_ma_roots.m:31-33 — Transpose to column if row vector
    # In Python/NumPy, 1-D arrays are the standard convention for vectors.
    # -------------------------------------------------------------------------
    return result.ravel()
