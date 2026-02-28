"""
Random uppercase character string generator.

Migrated from utility/randchar.m — generates a string of random uppercase
ASCII characters (A-Z) using NumPy's modern random number generator.

Author: Kevin Sheppard (original MATLAB)
Migration: MATLAB → Python 3.12
"""

import numpy as np


def randchar(n: int) -> str:
    """
    Generate a string of n random uppercase ASCII characters.

    Produces a string composed of n independent uniformly-distributed random
    characters drawn from the uppercase Latin alphabet (A-Z, ASCII 65-90).

    Parameters
    ----------
    n : int
        Number of characters to generate. Must be a non-negative integer.

    Returns
    -------
    c : str
        String of n random uppercase letters [A-Z].  Returns an empty string
        when ``n == 0``.

    Raises
    ------
    ValueError
        If *n* is negative or not an integer type.

    Examples
    --------
    >>> import numpy as np
    >>> result = randchar(5)
    >>> len(result) == 5
    True
    >>> all('A' <= ch <= 'Z' for ch in result)
    True

    Notes
    -----
    - Uses ``numpy.random.default_rng()`` per project convention (AAP rule).
    - Ref: utility/randchar.m — MATLAB implementation:
        ``u = rand(1,n); u = floor(u*26)+65; c = char(u);``
    - MATLAB ``rand(1,n)`` generates a 1×n row vector of U(0,1) values;
      Python equivalent is ``rng.random(n)`` producing a 1-D array of size n.
    - MATLAB ``floor()`` → ``numpy.floor()`` (identical semantics).
    - MATLAB ``char()`` converts numeric vector to character array →
      Python ``chr()`` converts individual int to character, joined via
      ``''.join()``.
    """
    # ---- Input validation ------------------------------------------------
    # MATLAB error() on invalid input → Python raises equivalent exception.
    if not isinstance(n, (int, np.integer)):
        raise ValueError(
            f"n must be a non-negative integer, got type {type(n).__name__}"
        )
    if n < 0:
        raise ValueError(
            f"n must be a non-negative integer, got {n}"
        )

    # Edge case: n == 0 returns empty string (consistent with MATLAB char([]))
    if n == 0:
        return ""

    # ---- Random generation -----------------------------------------------
    # Ref: randchar.m:3 — MATLAB: u = rand(1, n);
    # AAP rule: use numpy.random.default_rng() instead of numpy.random.rand()
    rng = np.random.default_rng()
    u = rng.random(n)

    # Ref: randchar.m:4 — MATLAB: u = floor(u * 26) + 65;
    # Maps uniform [0, 1) to integer codes [65, 90] (ASCII 'A' through 'Z').
    # np.floor(u * 26) produces values in {0, 1, ..., 25} since u ∈ [0, 1).
    codes = np.floor(u * 26).astype(np.intp) + 65

    # Ref: randchar.m:5 — MATLAB: c = char(u);
    # Convert integer ASCII codes to a Python string of uppercase characters.
    c = "".join(chr(v) for v in codes)

    return c
