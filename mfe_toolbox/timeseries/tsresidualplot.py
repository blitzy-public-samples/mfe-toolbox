"""
Time series residual diagnostic plot.

Produces a two-panel figure overlaying observed data with fitted values
(top panel) and residuals (bottom panel), with optional date-axis
formatting when MATLAB-style numeric dates or Python datetime-like
objects are supplied.

Migrated from: timeseries/tsresidualplot.m (Version 4.0, Kevin Sheppard)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure


def tsresidualplot(
    y: np.ndarray,
    errors: np.ndarray,
    dates: np.ndarray | None = None,
) -> tuple[list, Figure]:
    """
    Produce a diagnostic plot of time series data, fitted values, and residuals.

    Creates a two-panel figure:

    * **Top panel** — observed data *y* overlaid with fitted values
      (*y* − *errors*), labelled "Data" and "Fit".
    * **Bottom panel** — residuals (*errors*), labelled "Residual".

    Parameters
    ----------
    y : array_like
        A T-element vector of observed time series data.  Row vectors
        are automatically transposed to column vectors.
    errors : array_like
        A T-element vector of residuals, typically produced by
        :func:`~mfe_toolbox.timeseries.armaxfilter.armaxfilter`.
        Row vectors are automatically transposed.
    dates : array_like or None, optional
        A T-element vector of numeric dates (e.g. MATLAB datenum values
        or ``matplotlib.dates`` floats) or ``datetime``-like objects.
        When provided the x-axis displays dates instead of observation
        indices.  If *None* (default), x-axis uses 1-based observation
        indices ``1, 2, …, T``.

    Returns
    -------
    axes : list of matplotlib.axes.Axes
        A two-element list ``[ax_top, ax_bottom]`` of the subplot axes
        handles.  Callers can further customise axes (e.g.
        ``mdates.DateFormatter``) via these handles.
    fig : matplotlib.figure.Figure
        The figure containing both subplots.

    Raises
    ------
    ValueError
        If *y* or *errors* is not a vector, if their shapes differ,
        or if *dates* is non-numeric or non-vector.

    Notes
    -----
    The MATLAB version supports 0, 1 or 2 output arguments via
    ``nargout``/``varargout``.  The Python version always returns both
    the axes list and the figure; callers may simply ignore the second
    element if not needed.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> y = np.cumsum(rng.standard_normal(200))
    >>> errors = rng.standard_normal(200) * 0.5
    >>> axes, fig = tsresidualplot(y, errors)

    Plot with numeric dates (e.g. MATLAB datenum-style):

    >>> dates = np.arange(733043, 733043 + len(y), dtype=np.float64)
    >>> axes, fig = tsresidualplot(y, errors, dates)

    See Also
    --------
    mfe_toolbox.timeseries.armaxfilter : ARMAX filter driver.
    """

    # ------------------------------------------------------------------
    # Input Validation  (Ref: tsresidualplot.m:48-93)
    # ------------------------------------------------------------------

    # Coerce to float64 numpy arrays
    y = np.asarray(y, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)

    # Ref: tsresidualplot.m:54 — T = length(y)
    T: int = max(y.shape) if y.ndim >= 1 else 1

    # Auto-transpose row vectors to column vectors
    # Ref: tsresidualplot.m:55-57
    if y.ndim == 2 and y.shape[1] > y.shape[0]:
        y = y.T
    # Ref: tsresidualplot.m:58-60
    if errors.ndim == 2 and errors.shape[1] > errors.shape[0]:
        errors = errors.T

    # Validate that y is a column vector (T, ) or (T, 1)
    # Ref: tsresidualplot.m:62-64
    if y.ndim > 2 or (y.ndim == 2 and y.shape[1] != 1):
        raise ValueError("Y must be a column vector of data")

    # Validate that errors is a column vector
    # Ref: tsresidualplot.m:66-68
    # Note: MATLAB source has a bug — error message says "Y" for errors.
    # We use the corrected message per AAP instructions.
    if errors.ndim > 2 or (errors.ndim == 2 and errors.shape[1] != 1):
        raise ValueError("ERRORS must be a column vector of data")

    # Ref: tsresidualplot.m:69-71 — shapes must match
    if y.shape != errors.shape:
        raise ValueError("Y and ERRORS must have the same dimensions")

    # Ref: tsresidualplot.m:54 — recompute T after possible transpose
    T = y.shape[0]

    # Handle optional dates argument
    if dates is None:
        # Ref: tsresidualplot.m:73-75 — default 1-based indices (1:T)'
        dates = np.arange(1, T + 1, dtype=np.float64)
        dateflag: bool = False
    else:
        # Coerce to numpy array
        # Ref: tsresidualplot.m:86-88 — must be numeric
        try:
            dates = np.asarray(dates, dtype=np.float64)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                "DATES must contain numeric MATLAB dates, not strings."
            ) from exc

        # Ref: tsresidualplot.m:77-79 — auto-transpose row vector
        if dates.ndim == 2 and dates.shape[1] > dates.shape[0]:
            dates = dates.T

        # Ref: tsresidualplot.m:80-84 — must be column vector
        if dates.ndim > 2 or (dates.ndim == 2 and dates.shape[1] != 1):
            raise ValueError(
                "DATES must be a column vector with the same dimensions as Y"
            )

        dateflag = True

    # ------------------------------------------------------------------
    # Flatten all arrays to 1-D for matplotlib plotting
    # ------------------------------------------------------------------
    y_flat: np.ndarray = y.ravel()
    errors_flat: np.ndarray = errors.ravel()
    dates_flat: np.ndarray = dates.ravel()

    # Ref: tsresidualplot.m:96 — fitted values = y - errors
    yhat_flat: np.ndarray = y_flat - errors_flat

    # ------------------------------------------------------------------
    # Core Plotting Logic  (Ref: tsresidualplot.m:96-131)
    # ------------------------------------------------------------------

    # Ref: tsresidualplot.m:98 — figure('Position',[100 100 800 600])
    # MATLAB Position is [left bottom width height] in pixels.
    # figsize=(8, 6) inches at 100 dpi ≈ 800×600 px.
    fig: Figure = plt.figure(figsize=(8, 6))

    # ------------------------------------------------------------------
    # Subplot 1: Data and Fit  (Ref: tsresidualplot.m:99-114)
    # ------------------------------------------------------------------
    # Ref: tsresidualplot.m:99 — subplot(2,1,1)
    ax1 = fig.add_subplot(2, 1, 1)

    # Ref: tsresidualplot.m:100 — plot(dates,[y yhat])
    # Ref: tsresidualplot.m:112 — set(h1,'LineWidth',2)
    ax1.plot(dates_flat, y_flat, linewidth=2)
    ax1.plot(dates_flat, yhat_flat, linewidth=2)

    # Ref: tsresidualplot.m:101-106 — tight axis with 10% vertical margin
    ax1.autoscale(tight=True)
    ymin, ymax = ax1.get_ylim()
    spread = ymax - ymin
    # Ref: tsresidualplot.m:104 — ax(3) = ax(3) - .1*spread
    # Ref: tsresidualplot.m:105 — ax(4) = ax(4) + .1*spread
    ax1.set_ylim(ymin - 0.1 * spread, ymax + 0.1 * spread)

    # Ref: tsresidualplot.m:107 — legend('Data','Fit')
    ax1.legend(["Data", "Fit"])

    # Ref: tsresidualplot.m:108-110 — datetick('x','keeplimits')
    if dateflag:
        locator = mdates.AutoDateLocator()
        formatter = mdates.AutoDateFormatter(locator)
        ax1.xaxis.set_major_locator(locator)
        ax1.xaxis.set_major_formatter(formatter)

    # Ref: tsresidualplot.m:111,114 — title with bold 12pt font
    ax1.set_title("Data and Fit", fontsize=12, fontweight="bold")

    # Ref: tsresidualplot.m:113 — set(s1,'LineWidth',2,'FontSize',12,'FontWeight','Bold')
    ax1.tick_params(labelsize=12)
    for spine in ax1.spines.values():
        spine.set_linewidth(2)

    # ------------------------------------------------------------------
    # Subplot 2: Residuals  (Ref: tsresidualplot.m:116-131)
    # ------------------------------------------------------------------
    # Ref: tsresidualplot.m:116 — subplot(2,1,2)
    ax2 = fig.add_subplot(2, 1, 2)

    # Ref: tsresidualplot.m:117 — plot(dates,errors)
    # Ref: tsresidualplot.m:129 — set(h2,'LineWidth',2)
    ax2.plot(dates_flat, errors_flat, linewidth=2)

    # Ref: tsresidualplot.m:118-123 — tight axis with 10% vertical margin
    ax2.autoscale(tight=True)
    ymin, ymax = ax2.get_ylim()
    spread = ymax - ymin
    # Ref: tsresidualplot.m:121 — ax(3) = ax(3) - .1*spread
    # Ref: tsresidualplot.m:122 — ax(4) = ax(4) + .1*spread
    ax2.set_ylim(ymin - 0.1 * spread, ymax + 0.1 * spread)

    # Ref: tsresidualplot.m:124 — legend('Residual')
    ax2.legend(["Residual"])

    # Ref: tsresidualplot.m:125-127 — datetick('x','keeplimits')
    if dateflag:
        locator2 = mdates.AutoDateLocator()
        formatter2 = mdates.AutoDateFormatter(locator2)
        ax2.xaxis.set_major_locator(locator2)
        ax2.xaxis.set_major_formatter(formatter2)

    # Ref: tsresidualplot.m:128,131 — title with bold 12pt font
    ax2.set_title("Residual", fontsize=12, fontweight="bold")

    # Ref: tsresidualplot.m:130 — set(s2,'LineWidth',2,'FontSize',12,'FontWeight','Bold')
    ax2.tick_params(labelsize=12)
    for spine in ax2.spines.values():
        spine.set_linewidth(2)

    # Ensure subplots don't overlap
    fig.tight_layout()

    # ------------------------------------------------------------------
    # Return values  (Ref: tsresidualplot.m:133-142)
    # MATLAB uses varargout for 0/1/2 outputs; Python always returns both.
    # ------------------------------------------------------------------
    return [ax1, ax2], fig
