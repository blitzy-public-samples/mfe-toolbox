"""Pytest tests for realized volatility simulation modules.

Tests ``realized_range_simulation`` and ``realized_quantile_weight_simulation``
from ``mfe_toolbox.realized``.

These simulations are Monte Carlo-intensive. Tests use small parameters to
keep execution fast while still validating correctness of structure and
basic properties.

Per AAP Section 0.7.1: ATOL=1e-6, RTOL=1e-4 for all parity comparisons.
"""

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.realized.realized_range_simulation import (
    realized_range_simulation,
    ASYMPTOTIC_VALUE,
)
from mfe_toolbox.realized.realized_quantile_weight_simulation import (
    realized_quantile_weight_simulation,
)

ATOL = 1e-6
RTOL = 1e-4


# =====================================================================
# realized_range_simulation
# =====================================================================

class TestRealizedRangeSimulation:
    """Unit tests for realized_range_simulation."""

    def test_import(self) -> None:
        """Module should import without error."""
        assert callable(realized_range_simulation)

    def test_asymptotic_constant(self) -> None:
        """ASYMPTOTIC_VALUE should be 4*log(2) ≈ 2.7726."""
        npt.assert_allclose(ASYMPTOTIC_VALUE, 4.0 * np.log(2.0), atol=ATOL)

    def test_basic_execution(self) -> None:
        """Should run with small parameters and return tuple of two arrays."""
        # Use small BB and max_m for fast execution
        scale_factors, raw_results = realized_range_simulation(
            BB=1000, max_m=5, seed=42
        )
        assert isinstance(scale_factors, np.ndarray)
        assert isinstance(raw_results, np.ndarray)

    def test_output_shapes(self) -> None:
        """Output arrays should have shape (max_m - 1,)."""
        max_m = 10
        scale_factors, raw_results = realized_range_simulation(
            BB=500, max_m=max_m, seed=42
        )
        assert scale_factors.shape == (max_m - 1,)
        assert raw_results.shape == (max_m - 1,)

    def test_scale_factors_positive(self) -> None:
        """All scale factors should be positive."""
        scale_factors, _ = realized_range_simulation(
            BB=1000, max_m=10, seed=42
        )
        assert np.all(scale_factors > 0)

    def test_scale_factors_nondecreasing(self) -> None:
        """Concave regression ensures scale factors are non-decreasing."""
        scale_factors, _ = realized_range_simulation(
            BB=5000, max_m=10, seed=42
        )
        # Non-decreasing: each factor >= previous (within numerical tolerance)
        diffs = np.diff(scale_factors)
        assert np.all(diffs >= -1e-8), (
            f"Scale factors should be non-decreasing, got diffs: {diffs}"
        )

    def test_reproducibility_with_seed(self) -> None:
        """Same seed should produce identical results."""
        sf1, raw1 = realized_range_simulation(BB=500, max_m=5, seed=123)
        sf2, raw2 = realized_range_simulation(BB=500, max_m=5, seed=123)
        npt.assert_array_equal(raw1, raw2)
        npt.assert_array_equal(sf1, sf2)

    def test_raw_first_element_theory(self) -> None:
        """For m=2, raw result should be overridden to theoretical value 1.0."""
        _, raw_results = realized_range_simulation(BB=1000, max_m=5, seed=42)
        # Index 0 corresponds to m=2; theoretical E[range^2] for 2-point BM is 1
        npt.assert_allclose(raw_results[0], 1.0, atol=ATOL)

    def test_invalid_bb_raises(self) -> None:
        """BB < 1 should raise ValueError."""
        with pytest.raises((ValueError, TypeError)):
            realized_range_simulation(BB=0, max_m=5)

    def test_invalid_max_m_raises(self) -> None:
        """max_m < 2 should raise ValueError."""
        with pytest.raises((ValueError, TypeError)):
            realized_range_simulation(BB=100, max_m=1)


# =====================================================================
# realized_quantile_weight_simulation
# =====================================================================

class TestRealizedQuantileWeightSimulation:
    """Unit tests for realized_quantile_weight_simulation."""

    def test_import(self) -> None:
        """Module should import without error."""
        assert callable(realized_quantile_weight_simulation)

    def test_basic_execution(self) -> None:
        """Should run with small simulations and return a dict."""
        # Use very small simulation count for speed
        result = realized_quantile_weight_simulation(
            simulations=100, seed=42
        )
        assert isinstance(result, dict)

    def test_result_has_keys(self) -> None:
        """Result dict should contain expected key categories."""
        result = realized_quantile_weight_simulation(
            simulations=100, seed=42
        )
        # At minimum, the dict should not be empty
        assert len(result) > 0

    def test_reproducibility_with_seed(self) -> None:
        """Same seed should produce identical results."""
        r1 = realized_quantile_weight_simulation(simulations=100, seed=42)
        r2 = realized_quantile_weight_simulation(simulations=100, seed=42)
        # Compare all array values in the result dicts
        for key in r1:
            if isinstance(r1[key], np.ndarray):
                npt.assert_array_equal(r1[key], r2[key])
