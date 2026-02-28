"""
Sample Autocorrelation Function (SACF).

Computes sample autocorrelations and standard deviations using either
heteroskedasticity-robust standard errors or classic (homoskedastic) standard
errors, with optional bar chart visualization.

Migrated from timeseries/sacf.m — MFE Toolbox v4.0
Copyright: Kevin Sheppard
kevin.sheppard@economics.ox.ac.uk
Revision: 3.0.1    Date: 1/1/2007
"""

import numpy as np
import matplotlib.pyplot as plt


def sacf(data, lags, robust=True, graph=True):
    """
    Compute sample autocorrelations and standard deviations using either
    heteroskedasticity-robust standard errors or classic (homoskedastic)
    standard errors.

    Parameters
    ----------
    data : array_like
        A T-element vector of data. Must be a 1-D array or column vector.
    lags : int
        The number of autocorrelations to compute. Must be a positive integer
        strictly less than the length of data.
    robust : bool or int, optional
        If True (default) or 1, use heteroskedasticity-robust standard errors
        (White sandwich estimator). If False or 0, use classic (homoskedastic)
        standard errors (1/sqrt(T-L)).
    graph : bool or int, optional
        If True (default) or 1, produce a bar plot of the sample
        autocorrelations with ±2 standard error confidence bands.
        If False or 0, suppress plotting.

    Returns
    -------
    ac : numpy.ndarray
        A (lags,) vector of sample autocorrelations.
    acstd : numpy.ndarray
        A (lags,) vector of standard deviations for each autocorrelation.
    fig : matplotlib.figure.Figure or None
        Figure handle to the bar plot if graph=True, else None.

    Notes
    -----
    Sample autocorrelations are computed using the maximum number of
    observations for each lag. For example, if data has 100 observations, the
    first autocorrelation is computed using 99 data points, the second with 98
    data points, and so on.

    The autocorrelation at each lag L is the OLS slope coefficient from the
    regression of data[L:] on [1, data[:T-L]]. The robust standard error uses
    a White sandwich estimator applied to this regression.

    Ref: timeseries/sacf.m (MATLAB MFE Toolbox v4.0)

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> data = np.cumsum(rng.standard_normal(200))
    >>> ac, acstd, fig = sacf(data, 10, robust=True, graph=False)
    """
    # -----------------------------------------------------------------------
    # Input validation
    # Ref: sacf.m:38-70 — Input checking block
    # -----------------------------------------------------------------------

    # Convert input to numpy float64 array
    data = np.asarray(data, dtype=np.float64)

    # Ensure data is 1-D or column vector
    # Ref: sacf.m:50-55 — Transpose row vector, error on matrix
    if data.ndim == 2:
        if data.shape[0] == 1:
            # Row vector: transpose to column
            data = data.ravel()
        elif data.shape[1] == 1:
            # Column vector: flatten
            data = data.ravel()
        else:
            raise ValueError('DATA must be a column vector')
    elif data.ndim > 2:
        raise ValueError('DATA must be a column vector')
    # 1-D is acceptable as-is

    T = len(data)

    # Validate lags: must be a positive integer scalar
    # Ref: sacf.m:56-58 — ~isscalar(lags) || lags<=0 || floor(lags)~=lags
    if not np.isscalar(lags):
        raise ValueError('LAGS must be a positive integer')
    if lags <= 0 or np.floor(lags) != lags:
        raise ValueError('LAGS must be a positive integer')
    lags = int(lags)

    # Validate data length vs lags
    # Ref: sacf.m:47-49
    if T <= lags:
        raise ValueError('Length of data must be larger than LAGS')

    # Validate robust parameter
    # Ref: sacf.m:59-66
    if robust is None:
        robust = True
    else:
        if robust not in (True, False, 0, 1):
            raise ValueError('ROBUST must be either 0 or 1')
        robust = bool(robust)

    # Validate graph parameter
    # Ref: sacf.m:68-70
    if not np.isscalar(graph) or graph not in (True, False, 0, 1):
        raise ValueError('GRAPH must be a scalar, either 1 or 0.')
    graph = bool(graph)

    # -----------------------------------------------------------------------
    # Compute autocorrelations and standard errors
    # Ref: sacf.m:74-91 — Main computation loop
    # -----------------------------------------------------------------------
    ac = np.zeros(lags)
    acstd = np.zeros(lags)

    for L in range(1, lags + 1):
        # Ref: sacf.m:77-78 — MATLAB 1-based: y=data(L+1:T), x=[ones(T-L,1) data(1:T-L)]
        # Python 0-based: y=data[L:T], x[:,1]=data[0:T-L]
        y = data[L:T]
        n_obs = T - L
        x = np.ones((n_obs, 2))
        x[:, 1] = data[0:T - L]
        t = len(y)

        # Ref: sacf.m:80 — OLS regression: phi = x\y (MATLAB backslash)
        # Python: np.linalg.lstsq replaces MATLAB left-divide
        phi, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
        # Ref: sacf.m:81 — ac(L) = phi(2); MATLAB 1-indexed phi(2) = Python phi[1]
        ac[L - 1] = phi[1]

        if robust:
            # Ref: sacf.m:83 — Ainv = (x'*x/t)^(-1)
            # Robust sandwich standard error (White estimator)
            Ainv = np.linalg.inv(x.T @ x / t)

            # Ref: sacf.m:84 — s = x.*repmat(y-mean(y),1,2)
            # Element-wise product of regressors and demeaned dependent variable
            demeaned_y = (y - np.mean(y)).reshape(-1, 1)
            s = x * np.tile(demeaned_y, (1, 2))

            # Ref: sacf.m:85 — B = (s'*s)/t
            B = (s.T @ s) / t

            # Ref: sacf.m:86 — vcv = Ainv*B*Ainv/t
            vcv = Ainv @ B @ Ainv / t

            # Ref: sacf.m:87 — acstd(L) = sqrt(vcv(2,2))
            # MATLAB vcv(2,2) = Python vcv[1,1] (0-indexed)
            acstd[L - 1] = np.sqrt(vcv[1, 1])
        else:
            # Ref: sacf.m:89 — acstd(L) = sqrt(1/t)
            # Classic (homoskedastic) standard error
            acstd[L - 1] = np.sqrt(1.0 / t)

    # -----------------------------------------------------------------------
    # Optional plotting
    # Ref: sacf.m:93-121 — Bar plot with confidence bands
    # -----------------------------------------------------------------------
    fig = None
    if graph:
        # Ref: sacf.m:94-95 — hfig=figure; set(hfig,'Position',[100 100 800 600])
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Ref: sacf.m:98 — h = bar(ac)
        # MATLAB bar(ac) plots bars at x=1..lags (1-indexed default)
        lag_indices = np.arange(1, lags + 1)
        ax.bar(lag_indices, ac, color=[0.5, 0.5, 1.0])

        # Ref: sacf.m:100 — Confidence bands plotted from x=0 to x=lags+1
        # h2 = plot((0:lags+1)', [2*[acstd(1);acstd;acstd(lags)]
        #                         -2*[acstd(1);acstd;acstd(lags)]]);
        # Band x-axis: 0, 1, 2, ..., lags, lags+1 (lags+2 elements)
        band_x = np.arange(0, lags + 2)
        # Band values: padded at front with acstd[0] and at back with acstd[-1]
        upper_band = 2.0 * np.concatenate(([acstd[0]], acstd, [acstd[-1]]))
        lower_band = -2.0 * np.concatenate(([acstd[0]], acstd, [acstd[-1]]))

        # Ref: sacf.m:116-117 — set(h2(1),'LineWidth',2,'LineStyle',':','Color',[0 0 0])
        ax.plot(band_x, upper_band, linewidth=2, linestyle=':', color=[0, 0, 0])
        ax.plot(band_x, lower_band, linewidth=2, linestyle=':', color=[0, 0, 0])

        # Ref: sacf.m:101-109 — axis tight, then expand by 20% spread
        # Compute tight bounds from all plotted data
        all_y_vals = np.concatenate([ac, upper_band, lower_band])
        ymin = float(np.min(all_y_vals))
        ymax = float(np.max(all_y_vals))
        spread = 0.2 * (ymax - ymin)

        # Ref: sacf.m:105-109 — ax(1)=0; ax(2)=lags+1; ax(3)-=spread; ax(4)+=spread
        ax.set_xlim(0, lags + 1)
        ax.set_ylim(ymin - spread, ymax + spread)

        # Ref: sacf.m:110-115 — Title depends on robust setting
        if robust:
            ax.set_title(
                'Sample Autocorrelations and Robust Standard Errors',
                fontsize=14
            )
        else:
            ax.set_title(
                'Sample Autocorrelations and Non-robust Standard Errors',
                fontsize=14
            )

    return ac, acstd, fig
