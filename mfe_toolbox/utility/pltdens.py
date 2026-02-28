"""
Nonparametric kernel density estimation with optional plotting.

Migrated from utility/pltdens.m (originally by Anders Holtsberg, 18-Nov-1993).
Provides kernel density estimation using Gaussian, Epanechnikov, Biweight, or
Triangular kernels with FFT-based convolution for computational efficiency.
The grid extends beyond the data range by one-third on each side to reduce
boundary effects, and an optional reflection correction ensures non-negative
density estimates for positive-valued data.
"""

import numpy as np
import matplotlib.pyplot as plt


def pltdens(x, h=None, positive=False, kernel=1, plot=False):
    """
    Nonparametric kernel density estimation with optional visualization.

    Estimates a density function using kernel smoothing with FFT-based
    convolution over a grid of 256 equally-spaced evaluation points.
    Optionally displays the density curve with jittered data points below.

    Parameters
    ----------
    x : array_like
        Data vector for density estimation. Will be flattened to 1-D.
    h : float or None, optional
        Kernel bandwidth. If None (default), uses Silverman's rule of thumb:
        ``h = 1.06 * std(x) * n**(-1/5)``.
        See Silverman (1986), p. 45.
    positive : bool, optional
        If True, constrains the density to be zero for negative values using
        a reflection boundary correction. All elements of *x* must be
        non-negative when this flag is set. Default is False.
    kernel : int, optional
        Kernel function selection:

        * 1 — Gaussian (default)
        * 2 — Epanechnikov
        * 3 — Biweight
        * 4 — Triangular
    plot : bool, optional
        If True, produces a matplotlib plot of the density curve with
        jittered data points below the curve. Mirrors the MATLAB
        ``nargout == 0`` plotting branch. Default is False.

    Returns
    -------
    h : float
        Bandwidth used for the density estimate.
    f : numpy.ndarray
        Estimated density values, shape ``(256,)``.
    xx : numpy.ndarray
        Evaluation grid (domain of support), shape ``(256,)``.

    Raises
    ------
    ValueError
        If *positive* is True and any element of *x* is negative.
    ValueError
        If *kernel* is not in ``{1, 2, 3, 4}``.

    Notes
    -----
    The estimation uses linear binning and FFT-based convolution for
    computational efficiency.  The grid extends beyond the data range
    by one-third of the data range on each side to reduce boundary effects.

    When *positive* is True, a reflection boundary correction is applied:
    density mass below zero is folded onto the positive axis, ensuring the
    estimated density integrates to 1 over ``[0, inf)``.

    References
    ----------
    Silverman, B.W. (1986). *Density Estimation for Statistics and Data
    Analysis*. Chapman and Hall, London. Page 45.

    Originally by Anders Holtsberg, 18-Nov-1993.
    Migrated from MATLAB ``pltdens.m``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> data = rng.standard_normal(1000)
    >>> bandwidth, density, grid = pltdens(data)
    >>> bandwidth > 0
    True
    >>> density.shape
    (256,)
    >>> grid.shape
    (256,)
    """
    # ------------------------------------------------------------------
    # Input Processing
    # Ref: pltdens.m:31 — x = x(:); n = length(x)
    # ------------------------------------------------------------------
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)

    # Validate kernel type before any computation
    if kernel not in (1, 2, 3, 4):
        raise ValueError(
            f"Invalid kernel type {kernel}. Must be 1 (Gaussian), "
            "2 (Epanechnikov), 3 (Biweight), or 4 (Triangular)."
        )

    # ------------------------------------------------------------------
    # Default Bandwidth — Silverman Rule
    # Ref: pltdens.m:35-36 — h = 1.06 * std(x) * n^(-1/5)
    # Using ddof=1 to match MATLAB std() which uses N-1 normalization
    # ------------------------------------------------------------------
    if h is None:
        h = 1.06 * np.std(x, ddof=1) * n ** (-1.0 / 5.0)

    # Ensure h is a plain Python float for consistent return type
    h = float(h)

    # ------------------------------------------------------------------
    # Positive Constraint Validation
    # Ref: pltdens.m:38-39 — error('There is a negative element in X')
    # ------------------------------------------------------------------
    if positive and np.any(x < 0):
        raise ValueError("There is a negative element in X")

    # ------------------------------------------------------------------
    # Grid Setup
    # Ref: pltdens.m:41-46
    # ------------------------------------------------------------------
    mn1 = x.min()  # Ref: pltdens.m:41 — min(x)
    mx1 = x.max()  # Ref: pltdens.m:41 — max(x)
    data_range = mx1 - mn1

    # Ref: pltdens.m:42-43 — Extend grid by 1/3 of data range on each side
    mn = mn1 - data_range / 3.0
    mx = mx1 + data_range / 3.0

    gridsize = 256  # Ref: pltdens.m:44

    # Ref: pltdens.m:45 — xx = linspace(mn,mx,gridsize)'
    xx = np.linspace(mn, mx, gridsize)

    # Ref: pltdens.m:46 — d = xx(2) - xx(1)  (grid spacing)
    d = xx[1] - xx[0]

    # ------------------------------------------------------------------
    # Linear Binning
    # Ref: pltdens.m:47-53
    # ------------------------------------------------------------------
    xh = np.zeros(gridsize)

    # Ref: pltdens.m:48 — Map data onto grid coordinates [0, gridsize]
    xa = (x - mn) / (mx - mn) * gridsize

    for i in range(n):
        # Ref: pltdens.m:50 — il = floor(xa(i))
        il = int(np.floor(xa[i]))
        # Ref: pltdens.m:51 — a = xa(i) - il  (fractional position)
        a = xa[i] - il
        # Ref: pltdens.m:52 — MATLAB uses 1-indexed il+[1 2]; Python 0-indexed il and il+1
        idx0 = min(max(il, 0), gridsize - 1)
        idx1 = min(max(il + 1, 0), gridsize - 1)
        xh[idx0] += (1.0 - a)
        xh[idx1] += a

    # ------------------------------------------------------------------
    # Kernel Construction
    # Ref: pltdens.m:54 — xk = [-gridsize:gridsize-1]' * d
    # ------------------------------------------------------------------
    xk = np.arange(-gridsize, gridsize) * d

    if kernel == 1:
        # Ref: pltdens.m:56 — Gaussian kernel: exp(-0.5*(xk/h).^2)
        K = np.exp(-0.5 * (xk / h) ** 2)
    elif kernel == 2:
        # Ref: pltdens.m:58 — Epanechnikov kernel: max(0, 1-(xk/h).^2/5)
        K = np.maximum(0.0, 1.0 - (xk / h) ** 2 / 5.0)
    elif kernel == 3:
        # Ref: pltdens.m:60-61 — Biweight kernel with scaled support
        c = np.sqrt(1.0 / 7.0)
        K = ((1.0 - (xk / h * c) ** 2) ** 2
             * ((1.0 - np.abs(xk / h * c)) > 0).astype(float))
    elif kernel == 4:
        # Ref: pltdens.m:63-64 — Triangular kernel with scaled support
        c = np.sqrt(1.0 / 6.0)
        K = np.maximum(0.0, 1.0 - np.abs(xk / h * c))

    # ------------------------------------------------------------------
    # FFT Convolution
    # Ref: pltdens.m:66-68
    # ------------------------------------------------------------------
    # Ref: pltdens.m:66 — Normalize: K = K / (sum(K)*d*n)
    K = K / (np.sum(K) * d * n)

    # Ref: pltdens.m:67 — f = ifft(fft(fftshift(K)) .* fft([xh; zeros(...)]))
    f = np.fft.ifft(
        np.fft.fft(np.fft.fftshift(K))
        * np.fft.fft(np.concatenate([xh, np.zeros(gridsize)]))
    )

    # Ref: pltdens.m:68 — f = real(f(1:gridsize))
    f = np.real(f[:gridsize])

    # ------------------------------------------------------------------
    # Positive Boundary Correction
    # Ref: pltdens.m:69-74
    # ------------------------------------------------------------------
    if positive:
        # Ref: pltdens.m:70 — m = sum(xx<0)  (count of negative grid points)
        m = int(np.sum(xx < 0))
        if m > 0:
            # Ref: pltdens.m:71 — MATLAB f(m+(1:m)) += f(m:-1:1)
            # Python 0-indexed: f[m:2*m] += f[m-1::-1]
            # Folds density from negative region onto the mirror-image positive region
            f[m:2 * m] = f[m:2 * m] + f[m - 1::-1]

            # Ref: pltdens.m:72 — f(1:m) = zeros(...)  (zero out negative region)
            f[:m] = 0.0

            # Ref: pltdens.m:73 — MATLAB xx(m+[0 1]) = [0 0]
            # Python 0-indexed: xx[m-1] and xx[m] mark the zero boundary
            xx[m - 1] = 0.0
            xx[m] = 0.0

    # ------------------------------------------------------------------
    # Plotting  (mirrors MATLAB nargout==0 branch)
    # Ref: pltdens.m:75-85
    # ------------------------------------------------------------------
    if plot:
        # Create figure and axes — Ref: pltdens.m:76
        _fig, _ax = plt.subplots()

        # Main density curve — Ref: pltdens.m:76 — plot(xx,f)
        _ax.plot(xx, f)

        # Ref: pltdens.m:79 — d = diff(get(get(gcf,'CurrentAxes'),'Ylim'))/100
        fig = plt.gcf()  # noqa: F841 — access current figure (matches MATLAB gcf)
        ax = plt.gca()   # access current axes (matches MATLAB gca)
        d_plot = np.diff(ax.get_ylim())[0] / 100.0

        # Ref: pltdens.m:80 — plot(x, (-rand(size(x))*6-1)*d, '.')
        rng = np.random.default_rng()
        ax.plot(x, (-rng.random(len(x)) * 6.0 - 1.0) * d_plot, ".")

        # Ref: pltdens.m:81 — plot([mn mx],[0 0])  (zero baseline)
        plt.plot([mn, mx], [0, 0])

        # Ref: pltdens.m:82 — axis([mn mx -0.2*max(f) max(f)*1.2])
        ax.set_xlim(mn, mx)
        ax.set_ylim(-0.2 * np.max(f), np.max(f) * 1.2)

    return h, f, xx
