"""
Spectral analysis: frequency response computation from filter weights.

Migrated from timeseries/weights_to_frequency_response.m (Kevin Sheppard)

Computes the frequency response on [0, 0.5] (normalized frequency, equivalent
to [0, pi] radians/sample) for a linear filter defined by output weights A
and input weights W in the relationship::

    A' y  =  W' x

The gain at each frequency is |H(f)| where

    H(f) = (W' exp(-j 2 pi f (1:M))) / (A' exp(-j 2 pi f (1:K)))

This is the standard discrete-time Fourier transform ratio used for
characterising the magnitude response of ARMA-type filters.
"""

from __future__ import annotations

import numpy as np


def weights_to_frequency_response(
    a: np.ndarray,
    w: np.ndarray,
    n: int = 100,
) -> np.ndarray:
    """
    Compute the frequency response gain on [0, 0.5] for a filter defined by
    output weights *a* and input weights *w*.

    For each normalised frequency f in ``linspace(0, 0.5, n)`` the gain is

    .. math::

        \\text{fr}(f) = \\left|
            \\frac{\\mathbf{w}^\\top \\exp(-j 2\\pi f \\,[1 \\ldots M]^\\top)}
                 {\\mathbf{a}^\\top \\exp(-j 2\\pi f \\,[1 \\ldots K]^\\top)}
        \\right|

    Parameters
    ----------
    a : array_like
        K-element vector of output (left-hand-side) weights.  For a pure
        moving-average filter set ``a = np.array([1.0])``.
    w : array_like
        M-element vector of input (right-hand-side) weights.
    n : int, optional
        Number of equally-spaced frequency evaluation points in [0, 0.5].
        Default is 100.

    Returns
    -------
    fr : numpy.ndarray
        1-D array of length *n* containing the gain (magnitude of the
        complex frequency response) at each frequency point.

    Raises
    ------
    ValueError
        If *a* or *w* is empty, or if *n* is not a positive integer.

    Notes
    -----
    * The frequency axis uses normalised frequencies in [0, 0.5] (cycles per
      sample).  Multiply by ``2 * numpy.pi`` to convert to radians/sample.
    * Ref: weights_to_frequency_response.m — the MATLAB implementation uses a
      scalar loop over frequencies; this Python port vectorises the computation
      using ``numpy.outer`` for the phase matrix.
    * MATLAB 1-indexed ``(1:M)`` is preserved in the DFT summation indices
      (``numpy.arange(1, M+1)``) to ensure numerical parity.

    Examples
    --------
    Centred MA(3) filter with unit-pass-through output weight:

    >>> import numpy as np
    >>> fr = weights_to_frequency_response(
    ...     np.array([0.0, 1.0, 0.0]),
    ...     np.array([1.0/3, 1.0/3, 1.0/3]),
    ... )
    >>> fr.shape
    (100,)
    >>> float(np.round(fr[0], 6))  # DC gain
    1.0
    """
    # ------------------------------------------------------------------
    # Input validation and normalisation
    # ------------------------------------------------------------------
    # Ref: weights_to_frequency_response.m:34-39 — ensure column vectors
    a = np.asarray(a, dtype=np.float64).ravel()
    w = np.asarray(w, dtype=np.float64).ravel()

    if a.size == 0:
        raise ValueError(
            "Output weight vector 'a' must contain at least one element."
        )
    if w.size == 0:
        raise ValueError(
            "Input weight vector 'w' must contain at least one element."
        )

    # Ref: weights_to_frequency_response.m:29-31 — nargin==2 default n=100
    # Accept float-valued n for MATLAB compatibility (MATLAB has no int type
    # for function arguments) and convert to int.
    if isinstance(n, (float, np.floating)):
        if n != int(n):
            raise ValueError(
                "Number of frequency points 'n' must be an integer value."
            )
        n = int(n)
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError(
            "Number of frequency points 'n' must be a positive integer."
        )

    # ------------------------------------------------------------------
    # Frequency grid
    # ------------------------------------------------------------------
    # Ref: weights_to_frequency_response.m:33 — f = linspace(0, 0.5, n)
    f = np.linspace(0.0, 0.5, n)

    # ------------------------------------------------------------------
    # Filter dimensions
    # ------------------------------------------------------------------
    # Ref: weights_to_frequency_response.m:40-41
    m = w.size  # length of input weight vector
    k = a.size  # length of output weight vector

    # ------------------------------------------------------------------
    # Vectorised frequency response computation
    # ------------------------------------------------------------------
    # Build 1-indexed summation indices matching MATLAB (1:m) and (1:k).
    # Ref: weights_to_frequency_response.m:44 — (1:m)' and (1:k)'
    indices_w = np.arange(1, m + 1, dtype=np.float64)  # shape (m,)
    indices_a = np.arange(1, k + 1, dtype=np.float64)  # shape (k,)

    # Phase matrices: outer product of indices with 2*pi*f
    # Ref: weights_to_frequency_response.m:44 — exp(-1i*(1:m)'*2*pi*f(j))
    phase_w = np.outer(indices_w, 2.0 * np.pi * f)  # shape (m, n)
    phase_a = np.outer(indices_a, 2.0 * np.pi * f)  # shape (k, n)

    # Complex exponentials
    exp_w = np.exp(-1j * phase_w)  # shape (m, n)
    exp_a = np.exp(-1j * phase_a)  # shape (k, n)

    # Dot products along the weight axis for all frequencies simultaneously.
    # MATLAB: w' * exp_w(:,j) gives a scalar per frequency;
    # Python:  w @ exp_w  gives a (n,) vector across all frequencies.
    numerator = w @ exp_w      # (n,) complex
    denominator = a @ exp_a    # (n,) complex

    # ------------------------------------------------------------------
    # Gain: magnitude of the complex frequency response
    # ------------------------------------------------------------------
    # Ref: weights_to_frequency_response.m:46 — fr = abs(fr)
    # Pre-allocate with zeros to match MATLAB initialisation pattern
    # Ref: weights_to_frequency_response.m:42 — fr = zeros(size(f))
    fr = np.zeros(n, dtype=np.float64)
    fr[:] = np.abs(numerator / denominator)

    return fr
