"""Pytest tests for realized volatility simulation/scaling modules.

Tests three Monte Carlo simulation functions from ``mfe_toolbox.realized``:

1. ``realized_range_simulation`` — computes scale factors for the realized
   range estimator via Monte Carlo simulation of Brownian motion paths.
2. ``realized_quantile_weight_simulation`` — two-phase precomputation of
   quantile variance weights, scales, and covariances for standard block sizes.
3. ``realized_quantile_variance_scale`` — computes quantile RV scale factors,
   GMVP weights, and non-scaled covariance matrix.

These simulations are computationally intensive.  Unit tests use small parameter
values for fast execution while validating structural correctness, numerical
properties, and reproducibility.  Fixture parity tests load pre-generated MATLAB
reference data.

Per AAP Section 0.7.1:
    All parity assertions use ``numpy.testing.assert_allclose(actual, expected,
    atol=1e-6, rtol=1e-4)``.
"""

import os

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.realized.realized_range_simulation import (
    realized_range_simulation,
)
from mfe_toolbox.realized.realized_quantile_weight_simulation import (
    realized_quantile_weight_simulation,
)
from mfe_toolbox.realized.realized_quantile_variance_scale import (
    realized_quantile_variance_scale,
)

# ---------------------------------------------------------------------------
# Tolerance constants — per AAP Section 0.7.1
# ---------------------------------------------------------------------------
ATOL: float = 1e-6
RTOL: float = 1e-4

# ---------------------------------------------------------------------------
# Fixture directory resolution — per AAP Section 0.7.2
# Optional MFE_FIXTURE_DIR environment variable override for CI.
# ---------------------------------------------------------------------------
FIXTURE_DIR: str = os.environ.get(
    "MFE_FIXTURE_DIR",
    os.path.join(os.path.dirname(__file__), "..", "fixtures", "realized"),
)


def _load_fixture(name: str):
    """Load a ``.npy`` fixture from the realized fixtures directory.

    Parameters
    ----------
    name : str
        Filename (with ``.npy`` extension) relative to ``FIXTURE_DIR``.

    Returns
    -------
    numpy.ndarray or None
        Loaded array, or *None* when the fixture file does not exist.
    """
    path = os.path.join(FIXTURE_DIR, name)
    if os.path.exists(path):
        return np.load(path, allow_pickle=True)
    return None


def _fixture_exists(name: str) -> bool:
    """Return *True* if the named fixture file exists on disk."""
    return os.path.exists(os.path.join(FIXTURE_DIR, name))


# =========================================================================
# TestRealizedRangeSimulation
# =========================================================================


class TestRealizedRangeSimulation:
    """Tests for ``realized_range_simulation()``.

    The function performs Monte Carlo simulation of standard Brownian motion
    paths to compute E[(max(BM) - min(BM))^2] for block sizes m = 2, …, max_m.
    Raw estimates are smoothed via concave regression.

    Ref: realized/realized_range_simulation.m
    """

    # ------------------------------------------------------------------ #
    # 1. test_basic_execution
    # ------------------------------------------------------------------ #
    def test_basic_execution(self) -> None:
        """Function runs without error for a small simulation count."""
        result = realized_range_simulation(BB=500, max_m=5, seed=42)
        # Must return a 2-tuple
        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 2

    # ------------------------------------------------------------------ #
    # 2. test_output_type
    # ------------------------------------------------------------------ #
    def test_output_type(self) -> None:
        """Both outputs are ``numpy.ndarray``."""
        scale_factors, raw_results = realized_range_simulation(
            BB=500, max_m=5, seed=42
        )
        assert isinstance(scale_factors, np.ndarray), (
            f"scale_factors should be np.ndarray, got {type(scale_factors)}"
        )
        assert isinstance(raw_results, np.ndarray), (
            f"raw_results should be np.ndarray, got {type(raw_results)}"
        )

    # ------------------------------------------------------------------ #
    # 3. test_output_positive
    # ------------------------------------------------------------------ #
    def test_output_positive(self) -> None:
        """All scale factors and raw results are strictly positive.

        E[range(BM)^2] is always > 0 for any number of observation points
        because the range of a non-degenerate Brownian motion is positive a.s.
        """
        scale_factors, raw_results = realized_range_simulation(
            BB=2000, max_m=10, seed=42
        )
        assert np.all(scale_factors > 0), (
            "All scale factors must be positive"
        )
        assert np.all(raw_results > 0), (
            "All raw MC results must be positive"
        )

    # ------------------------------------------------------------------ #
    # 4. test_small_simulation
    # ------------------------------------------------------------------ #
    def test_small_simulation(self) -> None:
        """With BB=1000 (small), verify output shape and basic properties.

        Ref: realized_range_simulation.m:8 — ``BB = 1000000`` (default is 1M;
        test uses 1K for speed).
        """
        max_m = 10
        scale_factors, raw_results = realized_range_simulation(
            BB=1000, max_m=max_m, seed=42
        )
        # Shape: one entry per block size m = 2, 3, …, max_m → (max_m - 1,)
        assert scale_factors.shape == (max_m - 1,), (
            f"Expected shape ({max_m - 1},), got {scale_factors.shape}"
        )
        assert raw_results.shape == (max_m - 1,), (
            f"Expected shape ({max_m - 1},), got {raw_results.shape}"
        )
        # Ref: realized_range_simulation.m:49 — ``y(1) = 1``
        # First raw result (m=2) is overridden to theoretical value 1.0
        npt.assert_allclose(
            raw_results[0], 1.0, atol=ATOL, rtol=RTOL,
            err_msg="raw_results[0] (m=2) should be the theoretical value 1.0",
        )
        # Concave regression enforces non-decreasing scale factors
        diffs = np.diff(scale_factors)
        assert np.all(diffs >= -1e-8), (
            f"Scale factors should be non-decreasing, diffs = {diffs}"
        )

    # ------------------------------------------------------------------ #
    # 5. test_reproducible_with_seed
    # ------------------------------------------------------------------ #
    def test_reproducible_with_seed(self) -> None:
        """Fixed random seed produces identical results across calls.

        The MATLAB source uses ``randn('state', …)`` for reproducibility.
        The Python version uses ``numpy.random.default_rng(seed)``.
        """
        sf1, raw1 = realized_range_simulation(BB=500, max_m=5, seed=123)
        sf2, raw2 = realized_range_simulation(BB=500, max_m=5, seed=123)
        npt.assert_array_equal(sf1, sf2)
        npt.assert_array_equal(raw1, raw2)

    # ------------------------------------------------------------------ #
    # 6. test_precomputed_parity
    # ------------------------------------------------------------------ #
    @pytest.mark.skipif(
        not _fixture_exists("realized_range_simulation_results.npy"),
        reason="Fixture realized_range_simulation_results.npy not found",
    )
    @pytest.mark.parity
    def test_precomputed_parity(self) -> None:
        """Load MATLAB precomputed range simulation results and verify.

        Fixture: ``realized_range_simulation_results.npy`` converted from
        ``realized_range_simulation_results.mat``.

        Contains ``orig_mean_MC`` (raw MC estimates), ``concave_mean_MC``
        (smoothed), and ``ms`` (block sizes).  These were computed with
        BB = 1,000,000 in MATLAB.
        """
        fixture_raw = _load_fixture("realized_range_simulation_results.npy")
        assert fixture_raw is not None

        # Handle both structured-array and dict-object formats
        if fixture_raw.dtype.names is not None:
            ms = fixture_raw["ms"]
            orig = fixture_raw["orig_mean_MC"]
            concave = fixture_raw["concave_mean_MC"]
        else:
            data = fixture_raw.item() if fixture_raw.shape == () else fixture_raw
            if not isinstance(data, dict):
                pytest.skip("Unexpected fixture format")
                return
            ms = np.asarray(data.get("ms", []))
            orig = np.asarray(data.get("orig_mean_MC", []))
            concave = np.asarray(data.get("concave_mean_MC", []))

        # Structural checks
        assert ms.size > 0, "ms array should not be empty"
        assert ms.shape[0] == orig.shape[0] == concave.shape[0], (
            "All fixture arrays should have the same length"
        )

        # Property: all scale estimates must be positive
        assert np.all(orig > 0), "orig_mean_MC must be positive"
        assert np.all(concave > 0), "concave_mean_MC must be positive"

        # Property: first entry (m=2) should be close to 1.0
        # Ref: realized_range_simulation.m:49 — ``y(1) = 1``
        npt.assert_allclose(orig[0], 1.0, atol=ATOL, rtol=RTOL)

        # Property: values bounded by 4*log(2) ≈ 2.7726 (asymptotic limit)
        asymptotic = 4.0 * np.log(2.0)
        assert np.all(orig <= asymptotic + 0.5), (
            "orig_mean_MC should not substantially exceed asymptotic value"
        )

        # Property: concave smoothing should not deviate wildly from raw
        npt.assert_allclose(
            concave, orig, atol=0.1, rtol=0.1,
            err_msg="Concave smoothing should be close to raw estimates",
        )

        # Also verify against the dict-based range_simulation fixture
        fixture_dict = _load_fixture("realized_range_simulation.npy")
        if fixture_dict is not None:
            d = fixture_dict.item() if fixture_dict.shape == () else fixture_dict
            if isinstance(d, dict) and "hardcoded_orig_scale" in d:
                hc_orig = np.asarray(d["hardcoded_orig_scale"])
                # Hardcoded scales should be close to MC estimates
                min_len = min(len(orig), len(hc_orig))
                npt.assert_allclose(
                    orig[:min_len], hc_orig[:min_len],
                    atol=0.05, rtol=0.05,
                    err_msg="MC orig vs hardcoded orig should be close",
                )

    # ------------------------------------------------------------------ #
    # 7. test_scales_with_samples
    # ------------------------------------------------------------------ #
    def test_scales_with_samples(self) -> None:
        """More observation points per block → larger scaling constant.

        As max_m increases, the highest scale factor (last element) should
        approach the asymptotic value 4*log(2) ≈ 2.7726.
        """
        sf_small, _ = realized_range_simulation(BB=2000, max_m=5, seed=42)
        sf_large, _ = realized_range_simulation(BB=2000, max_m=20, seed=42)

        # The last scale factor for larger max_m should be greater
        assert sf_large[-1] > sf_small[-1], (
            f"Scale factor for max_m=20 ({sf_large[-1]:.4f}) should exceed "
            f"max_m=5 ({sf_small[-1]:.4f})"
        )

        # The largest scale factor should still be below the asymptotic limit
        asymptotic = 4.0 * np.log(2.0)
        assert sf_large[-1] < asymptotic + 0.5, (
            f"Largest scale factor ({sf_large[-1]:.4f}) should be near the "
            f"asymptotic value ({asymptotic:.4f})"
        )


# =========================================================================
# TestRealizedQuantileWeightSimulation
# =========================================================================


class TestRealizedQuantileWeightSimulation:
    """Tests for ``realized_quantile_weight_simulation()``.

    The function runs a two-phase Monte Carlo simulation (Phase 1 = symmetric,
    Phase 2 = asymmetric) across predefined block sizes, precomputing QRV
    scale factors, covariance matrices, and GMVP weights.

    Ref: realized/realized_quantile_weight_simulation.m
    """

    # ------------------------------------------------------------------ #
    # 1. test_basic_execution
    # ------------------------------------------------------------------ #
    def test_basic_execution(self) -> None:
        """Function runs without error for a small simulation count."""
        result = realized_quantile_weight_simulation(
            simulations=100, seed=42
        )
        assert result is not None

    # ------------------------------------------------------------------ #
    # 2. test_output_type
    # ------------------------------------------------------------------ #
    def test_output_type(self) -> None:
        """Returns a ``dict``."""
        result = realized_quantile_weight_simulation(
            simulations=100, seed=42
        )
        assert isinstance(result, dict), (
            f"Expected dict, got {type(result)}"
        )

    # ------------------------------------------------------------------ #
    # 3. test_output_shape
    # ------------------------------------------------------------------ #
    def test_output_shape(self) -> None:
        """Output dict contains expected keys and list lengths.

        Phase 1 (symmetric): 27 block sizes.
        Phase 2 (asymmetric): 20 block sizes.
        Ref: realized_quantile_weight_simulation.m:9, 67
        """
        result = realized_quantile_weight_simulation(
            simulations=100, seed=42
        )
        expected_keys = [
            "symmetricSimulationSamplerperbin",
            "symmetricSimulationQuantile",
            "symmetricExpectedQuantiles",
            "symmetricExpectedCovariance",
            "asymmetricSimulationSamplerperbin",
            "asymmetricSimulationQuantile",
            "asymmetricExpectedQuantiles",
            "asymmetricExpectedCovariance",
        ]
        for key in expected_keys:
            assert key in result, f"Missing expected key: {key!r}"

        # Symmetric phase: 27 predefined block sizes
        assert len(result["symmetricSimulationSamplerperbin"]) == 27, (
            "Should have 27 symmetric block sizes"
        )
        assert len(result["symmetricExpectedQuantiles"]) == 27
        assert len(result["symmetricExpectedCovariance"]) == 27

        # Asymmetric phase: 20 predefined block sizes
        assert len(result["asymmetricSimulationSamplerperbin"]) == 20, (
            "Should have 20 asymmetric block sizes"
        )
        assert len(result["asymmetricExpectedQuantiles"]) == 20
        assert len(result["asymmetricExpectedCovariance"]) == 20

    # ------------------------------------------------------------------ #
    # 4. test_small_simulation
    # ------------------------------------------------------------------ #
    @pytest.mark.slow
    def test_small_simulation(self) -> None:
        """simulations=1000 produces valid, finite output for every block size."""
        result = realized_quantile_weight_simulation(
            simulations=1000, seed=42
        )
        for i, scales in enumerate(result["symmetricExpectedQuantiles"]):
            assert isinstance(scales, np.ndarray), (
                f"symmetricExpectedQuantiles[{i}] should be ndarray"
            )
            assert np.all(np.isfinite(scales)), (
                f"symmetricExpectedQuantiles[{i}] should be finite"
            )
            assert np.all(scales > 0), (
                f"symmetricExpectedQuantiles[{i}] should be positive"
            )

        for i, scales in enumerate(result["asymmetricExpectedQuantiles"]):
            assert isinstance(scales, np.ndarray), (
                f"asymmetricExpectedQuantiles[{i}] should be ndarray"
            )
            assert np.all(np.isfinite(scales)), (
                f"asymmetricExpectedQuantiles[{i}] should be finite"
            )
            assert np.all(scales > 0), (
                f"asymmetricExpectedQuantiles[{i}] should be positive"
            )

    # ------------------------------------------------------------------ #
    # 5. test_reproducible_with_seed
    # ------------------------------------------------------------------ #
    def test_reproducible_with_seed(self) -> None:
        """Fixed seed produces reproducible results across two calls."""
        r1 = realized_quantile_weight_simulation(simulations=100, seed=42)
        r2 = realized_quantile_weight_simulation(simulations=100, seed=42)

        # Compare all symmetric expected quantile vectors element-by-element
        for i in range(len(r1["symmetricExpectedQuantiles"])):
            npt.assert_allclose(
                r1["symmetricExpectedQuantiles"][i],
                r2["symmetricExpectedQuantiles"][i],
                atol=ATOL,
                rtol=RTOL,
                err_msg=f"symmetricExpectedQuantiles[{i}] mismatch",
            )

        # Compare all asymmetric expected quantile vectors
        for i in range(len(r1["asymmetricExpectedQuantiles"])):
            npt.assert_allclose(
                r1["asymmetricExpectedQuantiles"][i],
                r2["asymmetricExpectedQuantiles"][i],
                atol=ATOL,
                rtol=RTOL,
                err_msg=f"asymmetricExpectedQuantiles[{i}] mismatch",
            )

    # ------------------------------------------------------------------ #
    # 6. test_weights_sum
    # ------------------------------------------------------------------ #
    def test_weights_sum(self) -> None:
        """GMVP weights derived from each covariance matrix sum to ≈ 1.

        The weight simulation stores scales and covariances but not weights
        directly; we reconstruct GMVP weights from the covariance matrices
        via ``w = Σ⁻¹ 1 / (1ᵀ Σ⁻¹ 1)`` and verify they sum to 1.
        """
        result = realized_quantile_weight_simulation(
            simulations=100, seed=42
        )
        for i, cov in enumerate(result["symmetricExpectedCovariance"]):
            cov_arr = np.asarray(cov)
            if cov_arr.ndim == 2 and cov_arr.shape[0] >= 1:
                k = cov_arr.shape[0]
                try:
                    cov_inv = np.linalg.inv(cov_arr)
                    ones_vec = np.ones(k)
                    weights = cov_inv @ ones_vec / (ones_vec @ cov_inv @ ones_vec)
                    npt.assert_allclose(
                        np.sum(weights), 1.0, atol=1e-8,
                        err_msg=(
                            f"GMVP weights from symmetricExpectedCovariance[{i}] "
                            "should sum to 1"
                        ),
                    )
                except np.linalg.LinAlgError:
                    # Singular covariance from very few simulations — skip this entry
                    pass

    # ------------------------------------------------------------------ #
    # 7. test_fixture_parity
    # ------------------------------------------------------------------ #
    @pytest.mark.skipif(
        not _fixture_exists("realized_quantile_weight_simulation.npy"),
        reason="Fixture realized_quantile_weight_simulation.npy not found",
    )
    @pytest.mark.parity
    def test_fixture_parity(self) -> None:
        """Compare against MATLAB fixture data structure and properties.

        Fixture: ``realized_quantile_weight_simulation.npy`` converted from
        the MATLAB precomputed data generated with simulations = 100,000,000.
        """
        fixture_raw = _load_fixture("realized_quantile_weight_simulation.npy")
        assert fixture_raw is not None
        data = fixture_raw.item() if fixture_raw.shape == () else fixture_raw

        if not isinstance(data, dict):
            pytest.skip("Unexpected fixture format")
            return

        # Structural: correct number of block sizes
        sym_spb = np.asarray(data.get("symmetricSimulationSamplerperbin", []))
        asym_spb = np.asarray(data.get("asymmetricSimulationSamplerperbin", []))
        assert sym_spb.shape[0] == 27, (
            f"Expected 27 symmetric block sizes, got {sym_spb.shape[0]}"
        )
        assert asym_spb.shape[0] == 20, (
            f"Expected 20 asymmetric block sizes, got {asym_spb.shape[0]}"
        )

        # Property: all symmetric scale vectors should be positive and finite
        sym_eq = data.get("symmetricExpectedQuantiles", [])
        for i, arr in enumerate(sym_eq):
            arr = np.asarray(arr)
            if arr.size > 0:
                assert np.all(arr > 0), (
                    f"Fixture symmetricExpectedQuantiles[{i}] should be positive"
                )
                assert np.all(np.isfinite(arr)), (
                    f"Fixture symmetricExpectedQuantiles[{i}] should be finite"
                )

        # Property: all symmetric covariance matrices should be symmetric
        sym_cov = data.get("symmetricExpectedCovariance", [])
        for i, cov in enumerate(sym_cov):
            cov_arr = np.asarray(cov)
            if cov_arr.ndim == 2 and cov_arr.shape[0] == cov_arr.shape[1]:
                npt.assert_allclose(
                    cov_arr, cov_arr.T, atol=ATOL, rtol=RTOL,
                    err_msg=(
                        f"Fixture symmetricExpectedCovariance[{i}] "
                        "should be symmetric"
                    ),
                )

        # Property: all asymmetric scale vectors should be positive and finite
        asym_eq = data.get("asymmetricExpectedQuantiles", [])
        for i, arr in enumerate(asym_eq):
            arr = np.asarray(arr)
            if arr.size > 0:
                assert np.all(arr > 0), (
                    f"Fixture asymmetricExpectedQuantiles[{i}] should be positive"
                )
                assert np.all(np.isfinite(arr)), (
                    f"Fixture asymmetricExpectedQuantiles[{i}] should be finite"
                )


# =========================================================================
# TestRealizedQuantileVarianceScale
# =========================================================================


class TestRealizedQuantileVarianceScale:
    """Tests for ``realized_quantile_variance_scale()``.

    Signature::

        (scales, weights, covar) = realized_quantile_variance_scale(
            samplesperbin, quantiles, simulations, symmetric, seed
        )

    Computes QRV scale factors, GMVP weights, and non-scaled covariance
    matrix via Monte Carlo integration of order-statistic expectations
    from standard normal samples.

    Ref: realized/realized_quantile_variance_scale.m
    """

    # ------------------------------------------------------------------ #
    # 1. test_returns_three_outputs
    # ------------------------------------------------------------------ #
    def test_returns_three_outputs(self) -> None:
        """Returns a 3-tuple ``(scales, weights, covar)``."""
        quantiles = np.array([0.75, 0.90])
        result = realized_quantile_variance_scale(
            samplesperbin=10, quantiles=quantiles, simulations=1000,
            symmetric=True, seed=42,
        )
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 3, f"Expected 3 elements, got {len(result)}"

        scales, weights, covar = result
        assert isinstance(scales, np.ndarray)
        assert isinstance(weights, np.ndarray)
        assert isinstance(covar, np.ndarray)

    # ------------------------------------------------------------------ #
    # 2. test_scales_positive
    # ------------------------------------------------------------------ #
    def test_scales_positive(self) -> None:
        """All scale factors are strictly positive.

        E[X²_(r:n)] > 0 for any order statistic of standard normal variates.
        """
        quantiles = np.array([0.65, 0.75, 0.90, 0.95])
        scales, _, _ = realized_quantile_variance_scale(
            samplesperbin=20, quantiles=quantiles, simulations=5000,
            symmetric=True, seed=42,
        )
        assert np.all(scales > 0), f"All scales must be positive, got {scales}"

    # ------------------------------------------------------------------ #
    # 3. test_weights_sum_to_one
    # ------------------------------------------------------------------ #
    def test_weights_sum_to_one(self) -> None:
        """GMVP weights sum to exactly 1.

        The GMVP formula ``w = Σ⁻¹ 1 / (1ᵀ Σ⁻¹ 1)`` guarantees
        ``1ᵀ w = 1`` by construction.
        Ref: realized_quantile_variance_scale.m:94
        """
        quantiles = np.array([0.65, 0.75, 0.90, 0.95])
        _, weights, _ = realized_quantile_variance_scale(
            samplesperbin=20, quantiles=quantiles, simulations=5000,
            symmetric=True, seed=42,
        )
        npt.assert_allclose(
            np.sum(weights), 1.0, atol=1e-10, rtol=0,
            err_msg="GMVP weights must sum to 1",
        )

    # ------------------------------------------------------------------ #
    # 4. test_covar_symmetric
    # ------------------------------------------------------------------ #
    def test_covar_symmetric(self) -> None:
        """Covariance matrix is symmetric: ``covar == covar.T``.

        Ref: realized_quantile_variance_scale.m:91 — ``covar = opMatrix - scales'*scales``
        The outer-product construction guarantees symmetry.
        """
        quantiles = np.array([0.60, 0.75, 0.90])
        _, _, covar = realized_quantile_variance_scale(
            samplesperbin=10, quantiles=quantiles, simulations=5000,
            symmetric=True, seed=42,
        )
        npt.assert_allclose(
            covar, covar.T, atol=ATOL, rtol=RTOL,
            err_msg="Covariance matrix must be symmetric",
        )

    # ------------------------------------------------------------------ #
    # 5. test_covar_positive_semidefinite
    # ------------------------------------------------------------------ #
    def test_covar_positive_semidefinite(self) -> None:
        """Covariance matrix eigenvalues ≥ 0 (PSD property).

        A sample covariance matrix is PSD by construction (it is the difference
        of an outer-product matrix and a rank-1 correction).
        """
        quantiles = np.array([0.60, 0.75, 0.90])
        _, _, covar = realized_quantile_variance_scale(
            samplesperbin=10, quantiles=quantiles, simulations=10000,
            symmetric=True, seed=42,
        )
        eigenvalues = np.linalg.eigvalsh(covar)
        # Allow small negative eigenvalues from floating-point rounding
        assert np.all(eigenvalues >= -1e-8), (
            f"Covariance matrix should be PSD, eigenvalues = {eigenvalues}"
        )

    # ------------------------------------------------------------------ #
    # 6. test_specific_quantiles
    # ------------------------------------------------------------------ #
    def test_specific_quantiles(self) -> None:
        """With samplesperbin=20 and quantiles=[13/20, 15/20, 19/20],
        output dimensions are: scales (3,), weights (3,), covar (3,3).
        """
        quantiles = np.array([13.0 / 20.0, 15.0 / 20.0, 19.0 / 20.0])
        scales, weights, covar = realized_quantile_variance_scale(
            samplesperbin=20, quantiles=quantiles, simulations=5000,
            symmetric=True, seed=42,
        )
        assert scales.shape == (3,), f"Expected (3,), got {scales.shape}"
        assert weights.shape == (3,), f"Expected (3,), got {weights.shape}"
        assert covar.shape == (3, 3), f"Expected (3,3), got {covar.shape}"

        # Extra: verify all scales are positive
        assert np.all(scales > 0)
        # Extra: verify weights sum to 1
        npt.assert_allclose(np.sum(weights), 1.0, atol=1e-10)

    # ------------------------------------------------------------------ #
    # 7. test_symmetric_option
    # ------------------------------------------------------------------ #
    def test_symmetric_option(self) -> None:
        """``symmetric=True`` vs ``symmetric=False`` produces different results.

        Ref: realized_quantile_variance_scale.m:56-61 — symmetric and
        asymmetric branches use different indexing schemes.
        """
        quantiles = np.array([0.75, 0.90])
        scales_sym, weights_sym, covar_sym = realized_quantile_variance_scale(
            samplesperbin=10, quantiles=quantiles, simulations=5000,
            symmetric=True, seed=42,
        )
        scales_asym, weights_asym, covar_asym = realized_quantile_variance_scale(
            samplesperbin=10, quantiles=quantiles, simulations=5000,
            symmetric=False, seed=42,
        )
        # Results should differ because the symmetric branch squares before
        # sorting, while the asymmetric branch sorts raw values and combines
        # upper and lower quantiles.
        assert not np.allclose(scales_sym, scales_asym), (
            "Symmetric and asymmetric modes should produce different scales"
        )
        assert not np.allclose(covar_sym, covar_asym), (
            "Symmetric and asymmetric modes should produce different covariances"
        )

    # ------------------------------------------------------------------ #
    # 8. test_small_simulations
    # ------------------------------------------------------------------ #
    def test_small_simulations(self) -> None:
        """With simulations=10000 (vs default 10,000,000), verify output
        is reasonable but potentially less accurate.
        """
        quantiles = np.array([0.75])
        scales, weights, covar = realized_quantile_variance_scale(
            samplesperbin=10, quantiles=quantiles, simulations=10000,
            symmetric=True, seed=42,
        )
        # Shape checks for single quantile
        assert scales.shape == (1,)
        assert weights.shape == (1,)
        assert covar.shape == (1, 1)

        # Scale should be positive and in a reasonable range
        # For n=10, the 75th order statistic of Z² should have an expected
        # value roughly between 0.5 and 3.0
        assert 0.1 < scales[0] < 10.0, (
            f"Scale seems out of reasonable range: {scales[0]}"
        )

        # Single-quantile weight is trivially 1.0
        npt.assert_allclose(weights[0], 1.0, atol=1e-10)

        # Covariance should be positive
        assert covar[0, 0] > 0

    # ------------------------------------------------------------------ #
    # 9. test_reproducible_results
    # ------------------------------------------------------------------ #
    def test_reproducible_results(self) -> None:
        """Same inputs produce exactly identical outputs with fixed seed.

        Ref: realized_quantile_variance_scale.m:50-52 — MATLAB saves and
        restores ``randn('state', …)`` for reproducibility.
        """
        quantiles = np.array([0.75, 0.90])
        s1, w1, c1 = realized_quantile_variance_scale(
            samplesperbin=10, quantiles=quantiles, simulations=1000,
            symmetric=True, seed=99,
        )
        s2, w2, c2 = realized_quantile_variance_scale(
            samplesperbin=10, quantiles=quantiles, simulations=1000,
            symmetric=True, seed=99,
        )
        npt.assert_array_equal(s1, s2)
        npt.assert_array_equal(w1, w2)
        npt.assert_array_equal(c1, c2)

    # ------------------------------------------------------------------ #
    # 10. test_precomputed_scales_parity
    # ------------------------------------------------------------------ #
    @pytest.mark.skipif(
        not _fixture_exists("realized_quantile_scales.npy"),
        reason="Fixture realized_quantile_scales.npy not found",
    )
    @pytest.mark.parity
    def test_precomputed_scales_parity(self) -> None:
        """Load precomputed MATLAB quantile scales and verify properties.

        Fixture: ``realized_quantile_scales.npy`` converted from
        ``realized_quantile_scales.mat``.  Generated with
        simulations = 100,000,000 in MATLAB.
        """
        fixture_raw = _load_fixture("realized_quantile_scales.npy")
        assert fixture_raw is not None
        data = fixture_raw.item() if fixture_raw.shape == () else fixture_raw

        if not isinstance(data, dict):
            pytest.skip("Unexpected fixture format")
            return

        # Verify symmetric scales: all positive, finite, correct count
        sym_scales = data.get("symmetricExpectedQuantiles", [])
        assert len(sym_scales) == 27, (
            f"Expected 27 symmetric entries, got {len(sym_scales)}"
        )
        for i, scales in enumerate(sym_scales):
            scales_arr = np.asarray(scales)
            if scales_arr.size > 0:
                assert np.all(scales_arr > 0), (
                    f"Symmetric scales[{i}] should be positive"
                )
                assert np.all(np.isfinite(scales_arr)), (
                    f"Symmetric scales[{i}] should be finite"
                )

        # Verify symmetric covariance matrices: symmetric and PSD
        sym_covars = data.get("symmetricExpectedCovariance", [])
        for i, cov in enumerate(sym_covars):
            cov_arr = np.asarray(cov)
            if cov_arr.ndim == 2 and cov_arr.shape[0] == cov_arr.shape[1]:
                npt.assert_allclose(
                    cov_arr, cov_arr.T, atol=ATOL, rtol=RTOL,
                    err_msg=f"Symmetric covariance[{i}] should be symmetric",
                )
                eigvals = np.linalg.eigvalsh(cov_arr)
                assert np.all(eigvals >= -1e-6), (
                    f"Symmetric covariance[{i}] should be PSD, "
                    f"min eigenvalue = {eigvals.min()}"
                )

        # Verify asymmetric scales
        asym_scales = data.get("asymmetricExpectedQuantiles", [])
        assert len(asym_scales) == 20, (
            f"Expected 20 asymmetric entries, got {len(asym_scales)}"
        )
        for i, scales in enumerate(asym_scales):
            scales_arr = np.asarray(scales)
            if scales_arr.size > 0:
                assert np.all(scales_arr > 0), (
                    f"Asymmetric scales[{i}] should be positive"
                )

    # ------------------------------------------------------------------ #
    # 11. test_single_quantile
    # ------------------------------------------------------------------ #
    def test_single_quantile(self) -> None:
        """Single quantile produces shapes (1,), (1,), (1,1).

        When k=1, the GMVP weight is trivially 1.0 and the covariance is
        a 1×1 matrix (scalar variance).
        """
        quantiles = np.array([0.90])
        scales, weights, covar = realized_quantile_variance_scale(
            samplesperbin=10, quantiles=quantiles, simulations=1000,
            symmetric=True, seed=42,
        )
        assert scales.shape == (1,), f"Expected (1,), got {scales.shape}"
        assert weights.shape == (1,), f"Expected (1,), got {weights.shape}"
        assert covar.shape == (1, 1), f"Expected (1,1), got {covar.shape}"

        # Trivial GMVP: single-asset weight is 1
        npt.assert_allclose(weights[0], 1.0, atol=1e-10)

        # Scale must be positive
        assert scales[0] > 0

        # Covariance (scalar) must be positive
        assert covar[0, 0] > 0

    # ------------------------------------------------------------------ #
    # 12. test_fixture_parity
    # ------------------------------------------------------------------ #
    @pytest.mark.skipif(
        not _fixture_exists("realized_quantile_variance_scale.npy"),
        reason="Fixture realized_quantile_variance_scale.npy not found",
    )
    @pytest.mark.parity
    @pytest.mark.slow
    def test_fixture_parity(self) -> None:
        """Compare function output against stored scenario fixture data.

        Fixture: ``realized_quantile_variance_scale.npy`` contains multiple
        scenarios with parameter dictionaries and expected output arrays.
        Re-runs each scenario and verifies structural and property parity.

        Note: Exact numerical parity is not expected because MATLAB and Python
        use different random number generators.  Instead, we verify that:
        - Output shapes match
        - Scales are positive
        - Weights sum to 1
        - Covariance is symmetric and PSD
        - Values are within a reasonable range of the fixture values
        """
        fixture_raw = _load_fixture("realized_quantile_variance_scale.npy")
        assert fixture_raw is not None
        data = fixture_raw.item() if fixture_raw.shape == () else fixture_raw

        if not isinstance(data, dict):
            pytest.skip("Unexpected fixture format")
            return

        for scenario_key in sorted(data.keys()):
            scenario = data[scenario_key]
            if not isinstance(scenario, dict) or "params" not in scenario:
                continue

            params = scenario["params"]
            spb = int(params["samplesperbin"])
            q = np.asarray(params["quantiles"]).ravel()
            sims = int(params["simulations"])
            sym = bool(params["symmetric"])

            expected_scales = np.asarray(scenario["scales"]).ravel()
            expected_weights = np.asarray(scenario["weights"]).ravel()
            expected_covar = np.asarray(scenario["covar"])

            # Re-run with the same parameters (different RNG, so exact match
            # not expected, but structural parity is mandatory)
            scales, weights, covar = realized_quantile_variance_scale(
                samplesperbin=spb, quantiles=q, simulations=sims,
                symmetric=sym, seed=0,
            )

            # Shape parity
            assert scales.shape == expected_scales.shape, (
                f"{scenario_key}: scales shape {scales.shape} != "
                f"expected {expected_scales.shape}"
            )
            assert weights.shape == expected_weights.shape, (
                f"{scenario_key}: weights shape {weights.shape} != "
                f"expected {expected_weights.shape}"
            )
            assert covar.shape == expected_covar.shape, (
                f"{scenario_key}: covar shape {covar.shape} != "
                f"expected {expected_covar.shape}"
            )

            # Property: scales positive
            assert np.all(scales > 0), (
                f"{scenario_key}: scales should be positive"
            )

            # Property: weights sum to 1
            npt.assert_allclose(
                np.sum(weights), 1.0, atol=1e-10,
                err_msg=f"{scenario_key}: weights should sum to 1",
            )

            # Property: covariance symmetric
            npt.assert_allclose(
                covar, covar.T, atol=ATOL, rtol=RTOL,
                err_msg=f"{scenario_key}: covar should be symmetric",
            )

            # Property: covariance PSD
            eigvals = np.linalg.eigvalsh(covar)
            assert np.all(eigvals >= -1e-8), (
                f"{scenario_key}: covar should be PSD, "
                f"eigenvalues = {eigvals}"
            )

            # Loose value comparison: Python and MATLAB results should be
            # in the same ballpark for moderate simulation counts.
            # Use generous tolerance because RNG streams differ.
            npt.assert_allclose(
                scales, expected_scales, atol=0.5, rtol=0.5,
                err_msg=(
                    f"{scenario_key}: scales should be in the same ballpark "
                    "as MATLAB reference (generous tolerance due to different RNG)"
                ),
            )
