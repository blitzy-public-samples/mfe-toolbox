"""
SARIMA model estimation stub.

This module reserves the SARIMA (Seasonal ARIMA) entry point for future
implementation. The original MATLAB function ``sandbox/sarima.m`` was an
empty stub with an 11-argument signature and no body.

Per the migration specification (AAP Section 0.7.2), the Python equivalent
preserves this as a stub that unconditionally raises ``NotImplementedError``.

Notes
-----
The MATLAB source file ``sandbox/sarima.m`` contained only a function
declaration line with no computation or documentation beyond the signature:

.. code-block:: matlab

    function sarima(y,c,p,q,d,seasonal,x,startingVals,options,holdBack,sigma2)

All 11 parameters are preserved in the Python signature using snake_case
naming conventions and ``None`` defaults (mirroring MATLAB's ``nargin``-based
partial argument passing).
"""


def sarima(
    y=None,
    c=None,
    p=None,
    q=None,
    d=None,
    seasonal=None,
    x=None,
    starting_vals=None,
    options=None,
    hold_back=None,
    sigma2=None,
):
    """
    SARIMA model estimation (not yet implemented).

    This function is a placeholder that reserves the SARIMA entry point
    for future implementation.  The original MATLAB function
    (``sandbox/sarima.m``) was an empty stub with no body.

    Parameters
    ----------
    y : array_like, optional
        Dependent variable vector.
    c : int, optional
        Constant indicator (1 to include a constant, 0 otherwise).
    p : array_like, optional
        AR lag indices.
    q : array_like, optional
        MA lag indices.
    d : array_like, optional
        Differencing orders.
    seasonal : array_like, optional
        Seasonal specification array (columns define seasonal AR order,
        seasonal MA order, and seasonal period).
    x : array_like, optional
        Exogenous regressor matrix.
    starting_vals : array_like, optional
        Starting values for the numerical optimizer.
    options : dict, optional
        Optimizer configuration options.
    hold_back : int, optional
        Number of initial observations to hold back for pre-sample
        conditioning.
    sigma2 : array_like, optional
        Conditional variance series.

    Raises
    ------
    NotImplementedError
        Always raised.  This function is a stub reserved for future
        implementation.

    References
    ----------
    Migrated from ``sandbox/sarima.m`` in the MFE Toolbox (Version 4.0,
    Kevin Sheppard, University of Oxford).  The MATLAB source was an empty
    function — no computation was present.
    """
    # Ref: sarima.m:1 — Original MATLAB function had no body at all.
    raise NotImplementedError(
        "sarima is not yet implemented. "
        "The original MATLAB function was an empty stub."
    )
