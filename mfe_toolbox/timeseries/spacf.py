"""
Sample Partial Autocorrelation Function (SPACF).

Computes sample partial autocorrelations and standard deviations using either
heteroskedasticity-robust standard errors or classic (homoskedastic) standard
errors, with optional bar chart visualization.

Migrated from timeseries/spacf.m — MFE Toolbox v4.0
Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 3.0.1    Date: 1/1/2007
"""

import numpy as np
import matplotlib.pyplot as plt

from mfe_toolbox.utility.newlagmatrix import newlagmatrix

__all__ = ['spacf']


def spacf(data, lags, robust=True, graph=True):
    """
    Compute sample partial autocorrelations and standard deviations.

    Uses OLS regression at each lag order via ``newlagmatrix`` to extract the
    partial autocorrelation coefficient. Standard errors are computed using
    either a heteroskedasticity-robust White sandwich estimator (default) or
    classic 1/sqrt(T) standard errors.

    Parameters
    ----------
    data : array_like
        A T-element vector of data. Must be a 1-D array or column vector.
    lags : int
        The number of partial autocorrelations to compute. Must be a positive
        integer strictly less than the length of ``data``.
    robust : bool or int, optional
        If True (default) or 1, use heteroskedasticity-robust standard errors
        (White sandwich estimator). If False or 0, use classic (homoskedastic)
        standard errors (1/sqrt(T)).
    graph : bool or int, optional
        If True (default) or 1, produce a bar plot of the sample partial
        autocorrelations with ±2 standard error confidence bands.
        If False or 0, suppress plotting.

    Returns
    -------
    pac : numpy.ndarray
        A (lags,) vector of sample partial autocorrelations.
    pacstd : numpy.ndarray
        A (lags,) vector of standard deviations for each partial
        autocorrelation.
    fig : matplotlib.figure.Figure or None
        Figure handle to the bar plot if ``graph=True``, else ``None``.

    Raises
    ------
    ValueError
        If ``data`` is not a 1-D vector or column vector, ``lags`` is not a
        positive integer, ``len(data) <= lags``, or ``robust`` is not in
        {0, 1, True, False}.

    Notes
    -----
    Sample partial autocorrelations are computed from regressions that use the
    maximum number of observations for each lag. For example, if ``data`` has
    100 observations, the first partial autocorrelation is computed using 99
    data points, the second with 98 data points, and so on.

    At each lag order ``i``, the function fits the OLS regression:

        y = β₀ + β₁·y_{t-1} + ··· + βᵢ·y_{t-i} + ε

    and extracts βᵢ as the i-th partial autocorrelation (``pac[i-1]``).

    The robust standard error uses the restricted-model residuals (regression
    on constant + lags 1 through i-1 only) inside a White-type sandwich
    estimator applied to the full regressors.

    Ref: timeseries/spacf.m (MATLAB MFE Toolbox v4.0)

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> data = np.cumsum(rng.standard_normal(200))
    >>> pac, pacstd, fig = spacf(data, 10, robust=True, graph=False)
    """
    # ------------------------------------------------------------------
    # Input validation
    # Ref: spacf.m:33-67 — Input checking block
    # ------------------------------------------------------------------

    # Convert input to numpy float64 array
    data = np.asarray(data, dtype=np.float64)

    # Ref: spacf.m:48-53 — Ensure data is 1-D or column vector;
    # Transpose row vector, error on matrix.
    if data.ndim == 2:
        if data.shape[0] == 1:
            # Row vector: flatten to 1-D
            data = data.ravel()
        elif data.shape[1] == 1:
            # Column vector: flatten to 1-D
            data = data.ravel()
        else:
            raise ValueError('DATA must be a column vector')
    elif data.ndim > 2:
        raise ValueError('DATA must be a column vector')
    # 1-D is acceptable as-is

    # Ref: spacf.m:54-56 — ~isscalar(lags) || lags<0 || floor(lags)~=lags
    # Validate lags: must be a positive integer scalar.
    # MATLAB error message says "positive integer"; instruction says > 0.
    if not np.isscalar(lags):
        raise ValueError('LAGS must be a positive integer')
    try:
        lags_int = int(lags)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('LAGS must be a positive integer')
    if lags_int != lags or lags_int < 1:
        # Ref: spacf.m:55 — lags<0 || floor(lags)~=lags
        # Using < 1 to enforce "positive" as stated in error message
        raise ValueError('LAGS must be a positive integer')
    lags = lags_int

    # Ref: spacf.m:44-46 — T=length(data); if T<=lags ...
    T = len(data)
    if T <= lags:
        raise ValueError('Length of data must be larger than LAGS')

    # Ref: spacf.m:57-64 — Validate robust parameter
    if robust is None:
        robust = True
    else:
        if robust not in (True, False, 0, 1):
            raise ValueError('ROBUST must be either 0 or 1')
        robust = bool(robust)

    # Coerce graph to bool for consistent boolean checks below
    graph = bool(graph)

    # ------------------------------------------------------------------
    # Compute partial autocorrelations and standard errors
    # Ref: spacf.m:68-85 — Main computation loop
    # ------------------------------------------------------------------
    pac = np.zeros(lags)
    pacstd = np.zeros(lags)

    for i in range(1, lags + 1):
        # Ref: spacf.m:71 — [y, x] = newlagmatrix(data,i,1);
        # newlagmatrix with include_constant=1 produces:
        #   y_trimmed: (T-i, 1),  x: (T-i, i+1) = [ones, lag1, ..., lagi]
        y_trimmed, x = newlagmatrix(data, i, 1)

        # Flatten y_trimmed to 1-D for consistent scalar extraction;
        # newlagmatrix returns (T-i, 1) from a 1-D input reshaped internally.
        y_flat = y_trimmed.ravel()

        # Ref: spacf.m:72 — t=length(y);
        t = len(y_flat)

        # Ref: spacf.m:73 — rho = x\y; (OLS regression via MATLAB backslash)
        # Python equivalent: numpy.linalg.lstsq replaces MATLAB left-divide
        rho = np.linalg.lstsq(x, y_flat, rcond=None)[0]

        # Ref: spacf.m:74 — pac(i) = rho(i+1);
        # MATLAB 1-indexed rho(i+1) is the i-th lag coefficient.
        # Python 0-indexed rho[i] is the same element because x columns are
        # [constant(0), lag1(1), ..., lagi(i)].
        pac[i - 1] = rho[i]

        # Ref: spacf.m:75-84 — Robust vs non-robust standard errors
        if robust:
            # Ref: spacf.m:76 — e = y-x(:,1:i)*(x(:,1:i)\y);
            # MATLAB x(:,1:i) selects columns 1..i (1-indexed inclusive),
            # which is [constant, lag1, ..., lag(i-1)] — the RESTRICTED model.
            # Python x[:, :i] selects columns 0..i-1 (0-indexed exclusive),
            # giving the identical set of columns.
            rho_restricted = np.linalg.lstsq(x[:, :i], y_flat, rcond=None)[0]
            e = y_flat - x[:, :i] @ rho_restricted

            # Ref: spacf.m:77 — XpXi = ((x'*x)/t)^(-1);
            # Inverse of average outer product of FULL regressors
            XpXi = np.linalg.inv(x.T @ x / t)

            # Ref: spacf.m:78 — eX = repmat(e,1,i+1).*x;
            # Element-wise product of restricted residuals (tiled) with full x.
            # numpy.tile replicates (t,1) vector into (t, i+1) then multiplies.
            eX = np.tile(e.reshape(-1, 1), (1, i + 1)) * x

            # Ref: spacf.m:79 — XEX = (eX'*eX)/t;
            XEX = eX.T @ eX / t

            # Ref: spacf.m:80 — V = XpXi*XEX*XpXi/t;
            V = XpXi @ XEX @ XpXi / t

            # Ref: spacf.m:81 — pacstd(i) = sqrt(V(i+1,i+1));
            # MATLAB V(i+1,i+1) with 1-indexing → Python V[i,i] with 0-indexing
            # This extracts the variance of the i-th lag coefficient.
            pacstd[i - 1] = np.sqrt(V[i, i])
        else:
            # Ref: spacf.m:83 — pacstd=ones(size(pac))/sqrt(length(data));
            # Classic non-robust SE: 1/sqrt(T) for all lags.
            # In MATLAB this overwrites every iteration (same result).
            pacstd = np.ones(lags) / np.sqrt(T)

    # ------------------------------------------------------------------
    # Optional plotting
    # Ref: spacf.m:87-114 — Bar plot with confidence bands
    # ------------------------------------------------------------------
    fig = None
    if graph:
        # Ref: spacf.m:88-89 — hfig=figure; set(hfig,'Position',[100 100 800 600])
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Ref: spacf.m:91 — h = bar(pac);
        # MATLAB bar(pac) plots bars at x=1..lags (1-indexed default).
        lag_indices = np.arange(1, lags + 1)
        ax.bar(lag_indices, pac, color=[0.5, 0.5, 1.0])

        # Ref: spacf.m:93 — Confidence bands plotted from x=0 to x=lags+1
        # h2 = plot((0:lags+1)', [2*[pacstd(1);pacstd;pacstd(lags)]
        #                         -2*[pacstd(1);pacstd;pacstd(lags)]]);
        # Band x-axis: 0, 1, 2, ..., lags, lags+1 (lags+2 elements)
        band_x = np.arange(0, lags + 2)

        # Band values: padded at front with pacstd[0] and at back with pacstd[-1]
        # to extend the confidence lines to the axis boundaries.
        padded_std = np.concatenate(([pacstd[0]], pacstd, [pacstd[-1]]))
        upper_band = 2.0 * padded_std
        lower_band = -2.0 * padded_std

        # Ref: spacf.m:109-110 — set(h2,'LineWidth',2,'LineStyle',':','Color',[0 0 0])
        ax.plot(band_x, upper_band, linewidth=2, linestyle=':', color=[0, 0, 0])
        ax.plot(band_x, lower_band, linewidth=2, linestyle=':', color=[0, 0, 0])

        # Ref: spacf.m:95-102 — axis tight, then expand y-range by 20% spread
        # Compute tight y-bounds from all plotted data.
        all_y_vals = np.concatenate([pac, upper_band, lower_band])
        ymin = float(np.min(all_y_vals))
        ymax = float(np.max(all_y_vals))
        spread = 0.2 * (ymax - ymin)

        # Ref: spacf.m:98-101 — ax(1)=0; ax(2)=lags+1; ax(3)-=spread; ax(4)+=spread
        ax.set_xlim(0, lags + 1)
        ax.set_ylim(ymin - spread, ymax + spread)

        # Ref: spacf.m:103-108 — Title depends on robust setting
        if robust:
            ax.set_title(
                'Sample Partial Autocorrelations and Robust Standard Errors',
                fontsize=14
            )
        else:
            ax.set_title(
                'Sample Partial Autocorrelations and Non-robust Standard Errors',
                fontsize=14
            )

    # Ref: spacf.m:1 — function [pac, pacstd, hfig] = spacf(data, lags, robust, graph)
    return pac, pacstd, fig
