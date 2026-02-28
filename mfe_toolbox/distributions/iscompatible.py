"""
Shape compatibility checker for distribution function parameters.

Migrated from distributions/iscompatible.m (MFE Toolbox Version 4.0).
Validates that a set of input parameters have compatible sizes and optionally
broadcasts scalar parameters to a common output size.

This is a helper function used by all distribution random variate generators
and PDF/CDF/INV functions in mfe_toolbox.distributions.

Author: Kevin Sheppard (original MATLAB)
Copyright: Kevin Sheppard, kevin.sheppard@economics.ox.ac.uk
Revision: 3    Date: 9/1/2005

MATLAB-to-Python migration notes:
- varargin → *args with slicing by narg
- cellfun('length', params) → list comprehension with max(shape)
- cellfun('isempty', params) → size == 0 check via np.asarray
- repmat(scalar, sizeOut) → np.tile(scalar, size_out)
- nargout > 3 conditional → always broadcast (Python has no nargout)
- MATLAB 1-indexed → Python 0-indexed (no index changes needed here)
"""

import numpy as np

__all__ = ['iscompatible']


def iscompatible(narg, *args):
    """
    Check whether input parameters are compatible with a requested output size.

    Validates that the first ``narg`` arguments in ``*args`` have compatible
    shapes (all scalar or all matching non-scalar shapes), optionally compares
    against an explicit output size specification, and broadcasts scalar
    parameters to the common output size.

    Parameters
    ----------
    narg : int
        Number of parameters to be checked (K). The first K elements of
        ``*args`` are treated as distribution parameters.
    *args : array_like
        First ``narg`` arguments are parameters (P1, P2, ..., PK).
        Remaining arguments specify the requested output size, either as:

        - A single array/list vector: ``[S1, S2, ..., SN]``
        - A series of scalar ints: ``S1, S2, ..., SN``

        If no size arguments are provided, the output size is inferred from
        the parameter shapes.

    Returns
    -------
    error : int
        1 if there is a problem with the input parameters, 0 otherwise.
    errortext : str
        A message detailing the nature of the problem. Empty string if no
        error.
    size_out : tuple of int
        The matched output size as a tuple. Empty tuple ``()`` if error.
    *params : numpy.ndarray
        Parameters transformed to all have the same dimension so that each
        output parameter has shape equal to ``size_out``. There will be
        ``narg`` additional return values, one per parameter. On error,
        each is an empty ``numpy.ndarray``.

    Notes
    -----
    This is a helper function for distribution PDF, CDF, inverse CDF, and
    random number generators in ``mfe_toolbox.distributions``.

    The function faithfully translates the MATLAB ``iscompatible.m`` logic:

    1. Parse the first ``narg`` args as parameters, remaining as size spec.
    2. Determine a common parameter shape:
       - All scalar → ``(1, 1)`` (MATLAB convention).
       - One non-scalar → its shape.
       - Multiple non-scalars → must all have identical shapes.
    3. Parse optional size specification (single vector or series of scalars).
    4. Compare parameter shape to requested size.
    5. Broadcast scalar parameters to the common output size via
       ``numpy.tile``.

    Ref: iscompatible.m — MFE Toolbox v4.0 by Kevin Sheppard.

    Examples
    --------
    All scalar parameters with explicit size:

    >>> err, errtext, sizeout, p1, p2 = iscompatible(2, 1.0, 2.0, 3, 4)
    >>> err
    0
    >>> sizeout
    (3, 4)
    >>> p1.shape
    (3, 4)

    Mixed scalar and array parameters:

    >>> import numpy as np
    >>> err, errtext, sizeout, p1, p2 = iscompatible(
    ...     2, 5.0, np.array([1.0, 2.0, 3.0]))
    >>> err
    0
    >>> sizeout
    (3,)
    >>> p1.shape
    (3,)
    """
    # Ref: iscompatible.m:39-40 — Initialize default error text
    errortext = ''

    # ------------------------------------------------------------------
    # Helper: construct error returns with narg empty parameter placeholders
    # ------------------------------------------------------------------
    def _error_return(err_text):
        """Return a standardized error tuple with empty placeholder arrays."""
        empty_params = tuple(np.asarray([]) for _ in range(narg))
        return (1, err_text, ()) + empty_params

    # ------------------------------------------------------------------
    # Step 1: Validate minimum argument count and non-emptiness
    # Ref: iscompatible.m:43-48
    #   if length(varargin)<narg || any(cellfun('isempty',varargin))
    # ------------------------------------------------------------------
    if len(args) < narg:
        return _error_return(
            'Too few parameters.  All inputs must be nonempty.'
        )

    # Check ALL args (parameters + size spec) for None or empty arrays
    # Ref: iscompatible.m:43 — cellfun('isempty', varargin) checks all
    for a in args:
        if a is None:
            return _error_return(
                'Too few parameters.  All inputs must be nonempty.'
            )
        arr_check = np.asarray(a)
        if arr_check.size == 0:
            return _error_return(
                'Too few parameters.  All inputs must be nonempty.'
            )

    # ------------------------------------------------------------------
    # Step 2: Parse the first narg arguments as parameters
    # Ref: iscompatible.m:52 — params = varargin(1:narg)
    # ------------------------------------------------------------------
    params = [np.asarray(args[i]) for i in range(narg)]

    # ------------------------------------------------------------------
    # Step 3: Get parameter "lengths" (MATLAB length = max dimension size)
    # Ref: iscompatible.m:54 — param_len = cellfun('length', params)
    # In MATLAB, length(x) returns max(size(x)), minimum 1 for scalars.
    # ------------------------------------------------------------------
    def _matlab_length(arr):
        """Compute MATLAB-equivalent length(): max of all dimension sizes."""
        if arr.ndim == 0:
            return 1
        if arr.size == 0:
            return 0
        return max(arr.shape)

    param_len = [_matlab_length(p) for p in params]

    # ------------------------------------------------------------------
    # Step 4: Determine the common parameter size
    # Ref: iscompatible.m:56-82
    # ------------------------------------------------------------------
    if len(param_len) == 0 or all(pl == 1 for pl in param_len):
        # All parameters are scalars (or no parameters at all)
        # Ref: iscompatible.m:58 — param_size = [1 1]
        param_size = (1, 1)

    elif sum(1 for pl in param_len if pl > 1) == 1:
        # Exactly one parameter is non-scalar — its shape becomes the common
        # size for all parameters.
        # Ref: iscompatible.m:62 — param_size = size(params{param_len>1})
        idx = next(i for i, pl in enumerate(param_len) if pl > 1)
        param_size = tuple(params[idx].shape)

    else:
        # Multiple non-scalar parameters — verify all have the same shape.
        # Ref: iscompatible.m:67-81
        non_scalar_indices = [i for i, pl in enumerate(param_len) if pl > 1]
        non_scalar_params = [params[i] for i in non_scalar_indices]

        # Ref: iscompatible.m:69-77 — Compare consecutive non-scalar shapes
        for i in range(len(non_scalar_params) - 1):
            if non_scalar_params[i].shape != non_scalar_params[i + 1].shape:
                return _error_return(
                    'Parameter size mismatch.  Must be either scalar or '
                    'of a common size.'
                )

        # Ref: iscompatible.m:81 — All non-scalars confirmed same shape
        param_size = tuple(non_scalar_params[0].shape)

    # ------------------------------------------------------------------
    # Step 5: Parse the requested output size from remaining arguments
    # Ref: iscompatible.m:86-111
    # ------------------------------------------------------------------
    if len(args) > narg:
        size_args = list(args[narg:])

        if len(size_args) == 1:
            # Single size argument — must be a valid vector (1D or 2D row/col)
            # Ref: iscompatible.m:88-97
            s = np.asarray(size_args[0])

            if s.ndim == 0:
                # Scalar size specification (e.g., iscompatible(1, v, 5))
                size_out = (int(s),)
            elif s.ndim == 1:
                # 1D array — valid vector of dimension sizes
                size_out = tuple(int(x) for x in s)
            elif s.ndim == 2 and min(s.shape) <= 1:
                # 2D row vector (1×N) or column vector (N×1) — valid
                # Ref: iscompatible.m:90 — ndims==2 && length==numel
                size_out = tuple(int(x) for x in s.ravel())
            else:
                # Matrix or higher-dimensional array — cannot be parsed as size
                return _error_return(
                    'Requested output size cannot be parsed.  Should be a '
                    'vector or series of scalars.'
                )

        else:
            # Multiple size arguments — all must be scalar integers
            # Ref: iscompatible.m:99-107
            size_vals = []
            for s in size_args:
                arr_s = np.asarray(s)
                if arr_s.size != 1:
                    return _error_return(
                        'Requested output size cannot be parsed.  Should be '
                        'a vector or series of scalars.'
                    )
                size_vals.append(int(arr_s.ravel()[0]))
            size_out = tuple(size_vals)

    else:
        # No size specification provided
        # Ref: iscompatible.m:110 — sizeOut = []
        size_out = ()

    # ------------------------------------------------------------------
    # Step 6: Compare parameter size to requested output size
    # Ref: iscompatible.m:116-134
    #
    # Three cases:
    #   1) Non-scalar params AND explicit size → must be equal
    #   2) No explicit size → use param_size
    #   3) All scalar params AND explicit size → use requested size
    # ------------------------------------------------------------------
    # prod(param_size) != 1 means at least one dimension > 1
    param_product = int(np.prod(param_size)) if param_size else 0
    param_is_nonscalar = (param_product != 1)
    has_size_out = (len(size_out) > 0)

    if param_is_nonscalar and has_size_out:
        # Both non-scalar params and explicit sizeOut — must match exactly
        # Ref: iscompatible.m:117-126
        if param_size == size_out:
            error = 0
        else:
            return _error_return(
                'Requested output size and parameters are not of '
                'compatible sizes.'
            )

    elif not has_size_out:
        # No explicit sizeOut requested — infer from parameter shapes
        # Ref: iscompatible.m:128-130
        size_out = param_size
        error = 0

    else:
        # All parameters are scalar AND an explicit sizeOut is provided
        # Ref: iscompatible.m:132-133
        error = 0

    # ------------------------------------------------------------------
    # Step 7: Broadcast scalar parameters to the common output size
    # Ref: iscompatible.m:137-145
    #   if nargout>3
    #       for i=1:narg
    #           if length(varargin{i})==1
    #               varargout{i} = repmat(varargin{i}, sizeOut);
    #           else
    #               varargout{i} = varargin{i};
    #           end
    #       end
    #   end
    #
    # In Python we always compute broadcasted params (no nargout concept).
    # ------------------------------------------------------------------
    broadcasted = []
    for i in range(narg):
        p = params[i]

        # Ref: iscompatible.m:139 — length(varargin{i})==1 checks for scalar
        if p.size == 1:
            # Scalar parameter — replicate to output size using np.tile
            # Ref: iscompatible.m:140 — repmat(varargin{i}, sizeOut)
            # np.tile replicates data, matching MATLAB repmat for scalars
            scalar_val = p.flat[0]
            broadcasted.append(np.tile(scalar_val, size_out))
        else:
            # Non-scalar parameter — already validated shape matches size_out
            # Ref: iscompatible.m:142 — varargout{i} = varargin{i}
            # Use np.broadcast_to as a shape-compatibility assertion, then
            # copy to ensure the output array is writable (broadcast_to
            # returns a read-only view when shapes already match).
            broadcasted.append(np.broadcast_to(p, size_out).copy())

    return (error, errortext, size_out) + tuple(broadcasted)
