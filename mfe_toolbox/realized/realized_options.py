"""
Default options factory for realized kernel and related estimators.

Migrated from realized/realized_options.m — MFE Toolbox (Kevin Sheppard).

Returns a dict of default option values for the specified realized volatility
estimator.  The MATLAB source returns a struct; Python returns a dict with
identical field names and default values.

These values have been calibrated for use with liquid NYSE TAQ data.  It may
be necessary to modify these values if computing realized kernels on illiquid
data or on data which trades over intervals significantly different from
6.5 hours (such as 24-hour markets or where the market is closed for part
of the day).

Supported Estimator Types
-------------------------
* ``'kernel'`` — Realized kernel estimation (``realized_kernel``)
* ``'multivariate kernel'`` — Multivariate realized kernel
  (``realized_multivariate_kernel``)
* ``'optimal sampling'`` — Bandi-Russell optimal sampling
  (``realized_variance_optimal_sampling``)
* ``'twoscale'`` — Two-scale realized variance
* ``'multiscale'`` — Multiscale realized variance
* ``'qmle'`` — QMLE estimation of QV
* ``'preaveraging'`` — Pre-averaged realized variance and related estimators

Notes
-----
Relevant options for each estimator function (marked with ``x``):

.. code-block:: text

                                  | Kernel | MV Kernel | Opt.Samp | TS/MS | QMLE | PreAvg
   kernel                         |   x    |     x     |          |       |      |
   endTreatment                   |   x    |     x     |          |       |      |
   jitterLags                     |   x    |     x     |          |       |      |
   maxBandwidthPerc               |   x    |     x     |          |   x   |      |
   maxBandwidth                   |   x    |     x     |          |   x   |      |
   bandwidth                      |   x    |     x     |          |   x   |      |
   useDebiasedNoise               |   x    |     x     |     x    |   x   |      |
   useAdjustedNoiseCount          |   x    |     x     |     x    |   x   |      |
   medFrequencySamplingType       |   x    |     x     |     x    |   x   |  x   |   x
   medFrequencySamplingInterval   |   x    |     x     |     x    |   x   |  x   |   x
   medFrequencyKernel             |   x    |     x     |     x    |   x   |      |   x
   medFrequencyBandwidth          |   x    |     x     |     x    |   x   |      |   x
   noiseVarianceSamplingType      |   x    |     x     |     x    |   x   |  x   |   x
   noiseVarianceSamplingInterval  |   x    |     x     |     x    |   x   |  x   |   x
   IQEstimationSamplingType       |   x    |     x     |     x    |   x   |      |   x
   IQEstimationSamplingInterval   |   x    |     x     |     x    |   x   |      |   x
   theta                          |        |           |          |       |      |   x

See Also
--------
realized_kernel, realized_variance_optimal_sampling,
realized_twoscale_variance, realized_multiscale_variance,
realized_qmle_variance, realized_multivariate_kernel
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Valid estimator identifiers (lower-cased for comparison)
# Ref: realized_options.m:138
# ---------------------------------------------------------------------------
_VALID_FUNCTIONS: frozenset[str] = frozenset({
    'kernel',
    'multivariate kernel',
    'optimal sampling',
    'twoscale',
    'multiscale',
    'qmle',
    'preaveraging',
})

# ---------------------------------------------------------------------------
# Field lists per estimator group — controls which keys appear in the
# returned dict.  Mirrors the MATLAB switch/case block (lines 184-211).
# ---------------------------------------------------------------------------

# Ref: realized_options.m:186-190
_KERNEL_FIELDS: tuple[str, ...] = (
    'kernel',
    'endTreatment',
    'jitterLags',
    'maxBandwidthPerc',
    'maxBandwidth',
    'bandwidth',
    'useDebiasedNoise',
    'useAdjustedNoiseCount',
    'medFrequencySamplingType',
    'medFrequencySamplingInterval',
    'medFrequencyKernel',
    'medFrequencyBandwidth',
    'noiseVarianceSamplingType',
    'noiseVarianceSamplingInterval',
    'IQEstimationSamplingType',
    'IQEstimationSamplingInterval',
)

# Ref: realized_options.m:192-195
_SAMPLING_FIELDS: tuple[str, ...] = (
    'bandwidth',
    'useDebiasedNoise',
    'useAdjustedNoiseCount',
    'medFrequencySamplingType',
    'medFrequencySamplingInterval',
    'medFrequencyKernel',
    'medFrequencyBandwidth',
    'noiseVarianceSamplingType',
    'noiseVarianceSamplingInterval',
    'IQEstimationSamplingType',
    'IQEstimationSamplingInterval',
)

# Ref: realized_options.m:200-201
_QMLE_FIELDS: tuple[str, ...] = (
    'medFrequencySamplingType',
    'medFrequencySamplingInterval',
    'noiseVarianceSamplingType',
    'noiseVarianceSamplingInterval',
)

# Ref: realized_options.m:205-208
_PREAVERAGING_FIELDS: tuple[str, ...] = (
    'theta',
    'medFrequencySamplingType',
    'medFrequencySamplingInterval',
    'medFrequencyKernel',
    'medFrequencyBandwidth',
    'noiseVarianceSamplingType',
    'noiseVarianceSamplingInterval',
    'IQEstimationSamplingType',
    'IQEstimationSamplingInterval',
    'useAdjustedNoiseCount',
)


def realized_options(realized_function: str = 'kernel') -> dict:
    """Return a dict of default option values for a realized volatility estimator.

    Parameters
    ----------
    realized_function : str, optional
        Case-insensitive name of the estimator.  Supported values:

        * ``'kernel'`` — Realized kernel estimation
        * ``'multivariate kernel'`` — Multivariate realized kernel
        * ``'optimal sampling'`` — Bandi-Russell optimal sampling
        * ``'twoscale'`` — Two-scale realized variance
        * ``'multiscale'`` — Multiscale realized variance
        * ``'qmle'`` — QMLE estimation of QV
        * ``'preaveraging'`` — Pre-averaged realized variance

        Default is ``'kernel'``.

    Returns
    -------
    dict
        Dictionary of option name → default value pairs.  Only keys relevant
        to the selected estimator are included.

    Raises
    ------
    ValueError
        If *realized_function* is not one of the supported estimator names.

    Examples
    --------
    >>> opts = realized_options('kernel')
    >>> opts['kernel']
    'nonflatparzen'
    >>> opts['endTreatment']
    'jitter'
    """
    # ------------------------------------------------------------------
    # Input validation
    # Ref: realized_options.m:131-140
    # ------------------------------------------------------------------
    # Case-insensitive matching — Ref: realized_options.m:137
    realized_function = realized_function.lower()

    if realized_function not in _VALID_FUNCTIONS:
        raise ValueError(
            f"realized_function must be one of {sorted(_VALID_FUNCTIONS)}, "
            f"got '{realized_function}'."
        )

    # ------------------------------------------------------------------
    # Construct full default options dict
    # Ref: realized_options.m:146-180
    # ------------------------------------------------------------------
    options: dict = {
        # Ref: realized_options.m:147 — kernel type
        'kernel': 'nonflatparzen',
        # Ref: realized_options.m:149 — end-point treatment
        'endTreatment': 'jitter',
        # Ref: realized_options.m:151 — maximum bandwidth as % of obs
        'maxBandwidthPerc': 0.25,
        # Ref: realized_options.m:153 — absolute max bandwidth (empty → auto)
        'maxBandwidth': None,
        # Ref: realized_options.m:155 — number of jitter lags
        'jitterLags': 2,
        # Ref: realized_options.m:157 — bandwidth (empty → auto plug-in)
        'bandwidth': None,
        # Ref: realized_options.m:160 — use debiased noise estimate
        'useDebiasedNoise': False,
        # Ref: realized_options.m:162 — adjusted non-zero return count
        'useAdjustedNoiseCount': True,
        # Ref: realized_options.m:164 — medium-frequency sampling type
        'medFrequencySamplingType': 'BusinessUniform',
        # Ref: realized_options.m:166 — medium-frequency sampling interval
        'medFrequencySamplingInterval': 390,
        # Ref: realized_options.m:168 — medium-frequency kernel
        'medFrequencyKernel': 'parzen',
        # Ref: realized_options.m:170 — medium-frequency bandwidth
        'medFrequencyBandwidth': 5,
        # Ref: realized_options.m:172 — noise variance sampling type
        'noiseVarianceSamplingType': 'BusinessUniform',
        # Ref: realized_options.m:174 — noise variance sampling interval
        'noiseVarianceSamplingInterval': 120,
        # Ref: realized_options.m:176 — IQ estimation sampling type
        'IQEstimationSamplingType': 'BusinessUniform',
        # Ref: realized_options.m:178 — IQ estimation sampling interval
        'IQEstimationSamplingInterval': 39,
        # Ref: realized_options.m:180 — theta for pre-averaging estimators
        'theta': 1,
    }

    # ------------------------------------------------------------------
    # Select field list and apply per-estimator overrides
    # Ref: realized_options.m:184-211
    # ------------------------------------------------------------------
    if realized_function in ('kernel', 'multivariate kernel'):
        # Ref: realized_options.m:185-190 — keep kernel-specific fields
        field_list: tuple[str, ...] = _KERNEL_FIELDS

    elif realized_function in ('optimal sampling', 'twoscale', 'multiscale'):
        # Ref: realized_options.m:191-198 — sampling estimators
        field_list = _SAMPLING_FIELDS
        # Overrides — Ref: realized_options.m:196-198
        options['noiseVarianceSamplingType'] = 'BusinessTime'
        options['noiseVarianceSamplingInterval'] = 1
        options['useAdjustedNoiseCount'] = False

    elif realized_function == 'qmle':
        # Ref: realized_options.m:199-203 — QMLE estimator
        field_list = _QMLE_FIELDS
        # Overrides — Ref: realized_options.m:202-203
        options['noiseVarianceSamplingType'] = 'BusinessTime'
        options['noiseVarianceSamplingInterval'] = 1

    elif realized_function == 'preaveraging':
        # Ref: realized_options.m:204-210 — pre-averaging estimators
        field_list = _PREAVERAGING_FIELDS
        # Overrides — Ref: realized_options.m:209-210
        options['noiseVarianceSamplingType'] = 'BusinessTime'
        options['noiseVarianceSamplingInterval'] = 1

    else:
        # Defensive — should never reach here due to validation above
        raise ValueError(
            f"Unhandled realized_function '{realized_function}'."
        )  # pragma: no cover

    # ------------------------------------------------------------------
    # Filter options dict to keep only relevant fields
    # Ref: realized_options.m:214-219
    # ------------------------------------------------------------------
    field_set = frozenset(field_list)
    filtered_options: dict = {
        key: value for key, value in options.items() if key in field_set
    }

    return filtered_options
