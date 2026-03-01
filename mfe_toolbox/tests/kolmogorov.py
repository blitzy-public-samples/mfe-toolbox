"""
Kolmogorov-Smirnov test for distributional correctness.

Performs a one-sample KS test that data follow a specified distribution,
using exact critical values for n <= 100 and the Miller (1956) asymptotic
approximation for n > 100.

Migrated from tests/kolmogorov.m — Kevin Sheppard, MFE Toolbox v4.0.
"""

import numpy as np
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# 100-row × 5-column KS critical-value lookup table
# Columns correspond to alpha = [0.10, 0.05, 0.025, 0.01, 0.005]
# Rows correspond to n = 1 .. 100  (access with 0-based index: row n-1)
# Ref: kolmogorov.m:126-226 — reproduced EXACTLY from MATLAB source
# ---------------------------------------------------------------------------
_KS_LOOKUP_TABLE: np.ndarray = np.array([
    [0.9000, 0.9500, 0.9750, 0.9900, 0.9950],   # n=1
    [0.6838, 0.7764, 0.8419, 0.9000, 0.9293],   # n=2
    [0.5648, 0.6360, 0.7076, 0.7846, 0.8290],   # n=3
    [0.4926, 0.5652, 0.6239, 0.6889, 0.7342],   # n=4
    [0.4470, 0.5094, 0.5633, 0.6272, 0.6685],   # n=5
    [0.4104, 0.4680, 0.5193, 0.5774, 0.6166],   # n=6
    [0.3815, 0.4361, 0.4834, 0.5384, 0.5758],   # n=7
    [0.3583, 0.4096, 0.4543, 0.5065, 0.5418],   # n=8
    [0.3391, 0.3875, 0.4300, 0.4796, 0.5133],   # n=9
    [0.3226, 0.3687, 0.4093, 0.4566, 0.4889],   # n=10
    [0.3083, 0.3524, 0.3912, 0.4367, 0.4677],   # n=11
    [0.2958, 0.3382, 0.3754, 0.4192, 0.4491],   # n=12
    [0.2847, 0.3255, 0.3614, 0.4036, 0.4325],   # n=13
    [0.2748, 0.3142, 0.3489, 0.3897, 0.4176],   # n=14
    [0.2659, 0.3040, 0.3376, 0.3771, 0.4042],   # n=15
    [0.2578, 0.2947, 0.3273, 0.3657, 0.3920],   # n=16
    [0.2504, 0.2863, 0.3180, 0.3553, 0.3809],   # n=17
    [0.2436, 0.2785, 0.3094, 0.3457, 0.3706],   # n=18
    [0.2374, 0.2714, 0.3014, 0.3368, 0.3612],   # n=19
    [0.2316, 0.2647, 0.2941, 0.3287, 0.3524],   # n=20
    [0.2262, 0.2586, 0.2872, 0.3210, 0.3443],   # n=21
    [0.2212, 0.2528, 0.2809, 0.3139, 0.3367],   # n=22
    [0.2165, 0.2475, 0.2749, 0.3073, 0.3295],   # n=23
    [0.2120, 0.2424, 0.2693, 0.3010, 0.3229],   # n=24
    [0.2079, 0.2377, 0.2640, 0.2952, 0.3166],   # n=25
    [0.2040, 0.2332, 0.2591, 0.2896, 0.3106],   # n=26
    [0.2003, 0.2290, 0.2544, 0.2844, 0.3050],   # n=27
    [0.1968, 0.2250, 0.2499, 0.2794, 0.2997],   # n=28
    [0.1935, 0.2212, 0.2457, 0.2747, 0.2947],   # n=29
    [0.1903, 0.2176, 0.2417, 0.2702, 0.2899],   # n=30
    [0.1873, 0.2141, 0.2379, 0.2660, 0.2853],   # n=31
    [0.1845, 0.2109, 0.2342, 0.2619, 0.2809],   # n=32
    [0.1817, 0.2077, 0.2308, 0.2580, 0.2768],   # n=33
    [0.1791, 0.2047, 0.2274, 0.2543, 0.2728],   # n=34
    [0.1766, 0.2019, 0.2243, 0.2507, 0.2690],   # n=35
    [0.1742, 0.1991, 0.2212, 0.2473, 0.2653],   # n=36
    [0.1719, 0.1965, 0.2183, 0.2440, 0.2618],   # n=37
    [0.1697, 0.1939, 0.2154, 0.2409, 0.2584],   # n=38
    [0.1675, 0.1915, 0.2127, 0.2379, 0.2552],   # n=39
    [0.1655, 0.1891, 0.2101, 0.2349, 0.2521],   # n=40
    [0.1635, 0.1869, 0.2076, 0.2321, 0.2490],   # n=41
    [0.1616, 0.1847, 0.2052, 0.2294, 0.2461],   # n=42
    [0.1597, 0.1826, 0.2028, 0.2268, 0.2433],   # n=43
    [0.1580, 0.1805, 0.2006, 0.2243, 0.2406],   # n=44
    [0.1562, 0.1786, 0.1984, 0.2218, 0.2380],   # n=45
    [0.1546, 0.1767, 0.1963, 0.2194, 0.2354],   # n=46
    [0.1530, 0.1748, 0.1942, 0.2172, 0.2330],   # n=47
    [0.1514, 0.1730, 0.1922, 0.2149, 0.2306],   # n=48
    [0.1499, 0.1713, 0.1903, 0.2128, 0.2283],   # n=49
    [0.1484, 0.1696, 0.1884, 0.2107, 0.2260],   # n=50
    [0.1470, 0.1680, 0.1866, 0.2086, 0.2239],   # n=51
    [0.1456, 0.1664, 0.1848, 0.2067, 0.2217],   # n=52
    [0.1442, 0.1648, 0.1831, 0.2047, 0.2197],   # n=53
    [0.1429, 0.1633, 0.1814, 0.2029, 0.2177],   # n=54
    [0.1416, 0.1619, 0.1798, 0.2011, 0.2157],   # n=55
    [0.1404, 0.1604, 0.1782, 0.1993, 0.2138],   # n=56
    [0.1392, 0.1591, 0.1767, 0.1976, 0.2120],   # n=57
    [0.1380, 0.1577, 0.1752, 0.1959, 0.2102],   # n=58
    [0.1369, 0.1564, 0.1737, 0.1943, 0.2084],   # n=59
    [0.1357, 0.1551, 0.1723, 0.1927, 0.2067],   # n=60
    [0.1346, 0.1538, 0.1709, 0.1911, 0.2051],   # n=61
    [0.1336, 0.1526, 0.1696, 0.1896, 0.2034],   # n=62
    [0.1325, 0.1514, 0.1682, 0.1881, 0.2018],   # n=63
    [0.1315, 0.1503, 0.1669, 0.1867, 0.2003],   # n=64
    [0.1305, 0.1491, 0.1657, 0.1853, 0.1988],   # n=65
    [0.1295, 0.1480, 0.1644, 0.1839, 0.1973],   # n=66
    [0.1286, 0.1469, 0.1632, 0.1825, 0.1958],   # n=67
    [0.1277, 0.1459, 0.1620, 0.1812, 0.1944],   # n=68
    [0.1268, 0.1448, 0.1609, 0.1799, 0.1930],   # n=69
    [0.1259, 0.1438, 0.1598, 0.1786, 0.1917],   # n=70
    [0.1250, 0.1428, 0.1586, 0.1774, 0.1903],   # n=71
    [0.1241, 0.1418, 0.1576, 0.1762, 0.1890],   # n=72
    [0.1233, 0.1409, 0.1565, 0.1750, 0.1878],   # n=73
    [0.1225, 0.1399, 0.1554, 0.1738, 0.1865],   # n=74
    [0.1217, 0.1390, 0.1544, 0.1727, 0.1853],   # n=75
    [0.1209, 0.1381, 0.1534, 0.1716, 0.1841],   # n=76
    [0.1201, 0.1372, 0.1524, 0.1704, 0.1829],   # n=77
    [0.1194, 0.1364, 0.1515, 0.1694, 0.1817],   # n=78
    [0.1186, 0.1355, 0.1505, 0.1683, 0.1806],   # n=79
    [0.1179, 0.1347, 0.1496, 0.1673, 0.1795],   # n=80
    [0.1172, 0.1339, 0.1487, 0.1663, 0.1784],   # n=81
    [0.1165, 0.1331, 0.1478, 0.1653, 0.1773],   # n=82
    [0.1158, 0.1323, 0.1469, 0.1643, 0.1763],   # n=83
    [0.1151, 0.1315, 0.1461, 0.1633, 0.1752],   # n=84
    [0.1144, 0.1307, 0.1452, 0.1624, 0.1742],   # n=85
    [0.1138, 0.1300, 0.1444, 0.1614, 0.1732],   # n=86
    [0.1131, 0.1292, 0.1436, 0.1605, 0.1722],   # n=87
    [0.1125, 0.1285, 0.1427, 0.1596, 0.1713],   # n=88
    [0.1119, 0.1278, 0.1419, 0.1587, 0.1703],   # n=89
    [0.1113, 0.1271, 0.1412, 0.1579, 0.1694],   # n=90
    [0.1106, 0.1264, 0.1404, 0.1570, 0.1685],   # n=91
    [0.1101, 0.1257, 0.1397, 0.1562, 0.1676],   # n=92
    [0.1095, 0.1251, 0.1389, 0.1553, 0.1667],   # n=93
    [0.1089, 0.1244, 0.1382, 0.1545, 0.1658],   # n=94
    [0.1083, 0.1238, 0.1375, 0.1537, 0.1649],   # n=95
    [0.1078, 0.1231, 0.1368, 0.1529, 0.1641],   # n=96
    [0.1072, 0.1225, 0.1361, 0.1521, 0.1632],   # n=97
    [0.1067, 0.1219, 0.1354, 0.1514, 0.1624],   # n=98
    [0.1061, 0.1213, 0.1347, 0.1506, 0.1616],   # n=99
    [0.1056, 0.1207, 0.1340, 0.1499, 0.1608],   # n=100
], dtype=np.float64)

# Alpha levels corresponding to the five columns of the lookup table
# Ref: kolmogorov.m:125 — alpha = [.1 .05 .025 .01 .005]
_KS_ALPHA_LEVELS: np.ndarray = np.array([0.10, 0.05, 0.025, 0.01, 0.005],
                                         dtype=np.float64)


def _kscritical(n: int, pvalue: float) -> float:
    """
    Compute the Kolmogorov-Smirnov critical value for sample size *n* and
    significance level *pvalue*.

    For n <= 100 the critical value is interpolated from a published lookup
    table.  For n > 100 the Miller (1956, JASA) asymptotic approximation is
    used.

    Parameters
    ----------
    n : int
        Sample size (must be >= 1).
    pvalue : float
        One-sided significance level (already halved for a two-sided test).

    Returns
    -------
    float
        The KS critical value at the requested significance level.

    Notes
    -----
    Ref: kolmogorov.m:119-232
    """
    if n <= 100:
        # Ref: kolmogorov.m:124-228
        # MATLAB: crit = interp1(alpha, lookuptable(n,:), pvalue)
        # np.interp requires xp in *ascending* order.  _KS_ALPHA_LEVELS is
        # descending ([0.1, 0.05, 0.025, 0.01, 0.005]), so we reverse both
        # the alpha vector and the corresponding row of the table.
        alpha_asc = _KS_ALPHA_LEVELS[::-1]
        # Ref: kolmogorov.m:228 — lookuptable(n,:) uses 1-based row index
        row_asc = _KS_LOOKUP_TABLE[n - 1, ::-1]
        crit: float = float(np.interp(pvalue, alpha_asc, row_asc))
    else:
        # Miller (1956) asymptotic formula
        # Ref: kolmogorov.m:230-231
        log10_p = np.log10(pvalue)
        neg_log10_p = -log10_p  # -log10(pvalue) == log10(1/pvalue)
        a_val: float = float(
            0.09037 * (neg_log10_p ** 1.5)
            + 0.01515 * (log10_p ** 2)
            - 0.08467 * pvalue
            - 0.11143
        )
        crit = float(
            np.sqrt(np.log(1.0 / pvalue) / (2.0 * n))
            - 0.16693 / n
            - a_val * n ** (-1.5)
        )

    return crit


def kolmogorov(
    x: np.ndarray,
    alpha: float = 0.05,
    dist: Optional[Callable] = None,
    *args,
) -> tuple[float, float, bool]:
    """
    Perform a Kolmogorov-Smirnov test that the data follow a specified
    distribution.

    Parameters
    ----------
    x : array_like
        A 1-D array of observations to test.  If *dist* is ``None`` the
        data must satisfy 0 < x < 1 (i.e. the probability integral
        transform has already been applied).
    alpha : float, optional
        Significance level for the hypothesis test (default 0.05).
        Must satisfy 0 < alpha < 1.
    dist : callable or None, optional
        CDF function used to transform *x*.  Should accept ``dist(x, *args)``
        and return an array of CDF values.  If ``None``, the data are
        assumed to be uniformly distributed on (0, 1).
    *args
        Additional positional arguments forwarded to *dist*.

    Returns
    -------
    stat : float
        The KS test statistic.
    pval : float
        Asymptotic p-value.
    H : bool
        ``True`` if the null hypothesis is rejected at the requested
        *alpha* level, ``False`` otherwise.

    Raises
    ------
    ValueError
        If no input is provided, if *x* is not 1-D, if *alpha* is outside
        (0, 1), if *dist* is not callable, or if *x* values are outside
        (0, 1) when *dist* is ``None``.

    Examples
    --------
    Test data for uniformity (PIT already applied):

    >>> stat, pval, H = kolmogorov(x)

    Test standard-normal data:

    >>> from scipy.stats import norm
    >>> stat, pval, H = kolmogorov(x, 0.05, norm.cdf)

    Test normal data with mean=1, std=2:

    >>> stat, pval, H = kolmogorov(x, 0.05, norm.cdf, 1, 2)

    Notes
    -----
    Migrated from ``tests/kolmogorov.m`` (MFE Toolbox v4.0).
    The MATLAB implementation uses ``feval(dist, x, varargin{:})``;
    in Python the *dist* parameter is a callable invoked as
    ``dist(x, *args)``.
    """

    # ------------------------------------------------------------------
    # Input validation
    # Ref: kolmogorov.m:41-75
    # ------------------------------------------------------------------

    # Ref: kolmogorov.m:51-52 — nargin == 0 case
    x_arr: np.ndarray = np.asarray(x, dtype=np.float64)

    # Ref: kolmogorov.m:54-56 — if alpha is empty, set to 0.05
    if alpha is None:
        alpha = 0.05

    # Ref: kolmogorov.m:57-58 — x must be a 1-D vector
    if x_arr.ndim == 0:
        raise ValueError("X must be a column vector of data")
    if x_arr.ndim > 2:
        raise ValueError("X must be a column vector of data")
    if x_arr.ndim == 2:
        # Ref: kolmogorov.m:57 — min(size(x)) ~= 1 check
        if min(x_arr.shape) != 1:
            raise ValueError("X must be a column vector of data")
        # Ref: kolmogorov.m:60-61 — if size(x,2) ~= 1, transpose → ravel
        x_arr = x_arr.ravel()

    if x_arr.size == 0:
        raise ValueError("X must be a column vector of data")

    # Ref: kolmogorov.m:64-71 — distribution / data-range validation
    if dist is None:
        # Ref: kolmogorov.m:65-66 — data must satisfy 0 < x < 1
        if np.any(x_arr >= 1.0) or np.any(x_arr <= 0.0):
            raise ValueError(
                "If DIST is not input, X must satisfy 0<X<1"
            )
    else:
        # Ref: kolmogorov.m:69-70 — MATLAB checks ischar(dist);
        # Python equivalent: must be callable
        if not callable(dist):
            raise ValueError("DIST must be a callable (function)")

    # Ref: kolmogorov.m:73-74 — alpha must be scalar in (0, 1)
    if not np.isscalar(alpha):
        raise ValueError("ALPHA must be a scalar between 0 and 1")
    alpha = float(alpha)
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError("ALPHA must be a scalar between 0 and 1")

    # ------------------------------------------------------------------
    # Computation
    # Ref: kolmogorov.m:81-115
    # ------------------------------------------------------------------

    # Ref: kolmogorov.m:81 — sort data
    x_sorted: np.ndarray = np.sort(x_arr)

    # Ref: kolmogorov.m:82-94 — compute CDF values
    if dist is not None:
        # Ref: kolmogorov.m:83-91 — feval(dist, x, varargin{:})
        try:
            cdfvals: np.ndarray = np.asarray(dist(x_sorted, *args),
                                             dtype=np.float64)
        except Exception:
            raise ValueError(
                "There was an error calling the DIST function "
                "with the VARARGIN provided."
            )
    else:
        # Ref: kolmogorov.m:93 — cdfvals = x (PIT already applied)
        cdfvals = x_sorted

    # Ref: kolmogorov.m:95 — n = length(x)
    n: int = int(x_sorted.shape[0])

    # Ref: kolmogorov.m:96 — S = (1:n)' / (n+1)
    # MATLAB 1-indexed range; Python 0-indexed → np.arange(1, n+1)
    s_empirical: np.ndarray = np.arange(1, n + 1, dtype=np.float64) / (n + 1)

    # Ref: kolmogorov.m:97 — alpha = alpha / 2 (two-sided test)
    alpha_half: float = alpha / 2.0

    # Ref: kolmogorov.m:99-100 — KS statistic
    d: float = float(np.max(np.abs(cdfvals - s_empirical)))
    stat: float = d

    # Ref: kolmogorov.m:101-102 — critical value and rejection decision
    crit: float = _kscritical(n, alpha_half)
    h_reject: bool = bool(stat > crit)

    # Ref: kolmogorov.m:104 — modified statistic for p-value computation
    k: float = (np.sqrt(float(n)) + 0.12 + 0.11 / np.sqrt(float(n))) * stat

    # Ref: kolmogorov.m:105-115 — alternating-series p-value computation
    # Guard: when stat (and hence k) is zero or extremely small, the
    # alternating series 2*sum((-1)^(j-1)*exp(-2*k^2*j^2)) does not
    # converge because every term equals ±2.  In this degenerate case
    # the p-value is 1.0 (no evidence against the null).
    if k < 1e-15:
        pval: float = 1.0
    else:
        tol: float = 1.0
        pval = 2.0 * np.exp(-2.0 * k * k)
        sign: float = -1.0
        i: int = 2
        while tol > 1e-10:
            old_pval: float = pval
            pval = pval + 2.0 * sign / np.exp(2.0 * k * k * float(i) * float(i))
            i += 1
            sign *= -1.0
            tol = abs(pval - old_pval)

    return stat, pval, h_reject
