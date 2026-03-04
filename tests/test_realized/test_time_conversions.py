"""Pytest tests for the realized volatility time conversion utilities.

Tests seconds2unit, seconds2wall, unit2seconds, unit2wall, wall2seconds,
wall2unit from ``mfe_toolbox.realized``.

Per AAP Section 0.7.1: ATOL=1e-6, RTOL=1e-4 for all parity comparisons.
"""

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.realized.seconds2unit import seconds2unit
from mfe_toolbox.realized.seconds2wall import seconds2wall
from mfe_toolbox.realized.unit2seconds import unit2seconds
from mfe_toolbox.realized.unit2wall import unit2wall
from mfe_toolbox.realized.wall2seconds import wall2seconds
from mfe_toolbox.realized.wall2unit import wall2unit

ATOL = 1e-6
RTOL = 1e-4


# =====================================================================
# seconds2unit
# =====================================================================

class TestSeconds2Unit:
    """Unit tests for seconds2unit."""

    def test_import(self) -> None:
        """Module should import without error."""
        assert callable(seconds2unit)

    def test_boundary_zero(self) -> None:
        """Start-of-window maps to 0."""
        # seconds0=34200 (9:30 AM), seconds1=57600 (4:00 PM)
        seconds0 = 34200.0
        seconds1 = 57600.0
        result = seconds2unit(np.array([seconds0, seconds1]), seconds0, seconds1)
        npt.assert_allclose(result[0], 0.0, atol=ATOL)
        npt.assert_allclose(result[-1], 1.0, atol=ATOL)

    def test_midpoint(self) -> None:
        """Midpoint of window maps to 0.5."""
        seconds0 = 34200.0
        seconds1 = 57600.0
        mid = (seconds0 + seconds1) / 2.0
        result = seconds2unit(np.array([seconds0, mid, seconds1]), seconds0, seconds1)
        npt.assert_allclose(result[1], 0.5, atol=ATOL)

    def test_linearity(self) -> None:
        """Conversion should be linear: equally spaced seconds → equally spaced unit."""
        seconds0 = 34200.0
        seconds1 = 57600.0
        times = np.linspace(seconds0, seconds1, 11)
        result = seconds2unit(times, seconds0, seconds1)
        expected = np.linspace(0.0, 1.0, 11)
        npt.assert_allclose(result, expected, atol=ATOL)

    def test_return_type_ndarray(self) -> None:
        """Return type should be numpy ndarray."""
        result = seconds2unit(np.array([34200.0, 57600.0]), 34200.0, 57600.0)
        assert isinstance(result, np.ndarray)


# =====================================================================
# seconds2wall
# =====================================================================

class TestSeconds2Wall:
    """Unit tests for seconds2wall."""

    def test_import(self) -> None:
        """Module should import without error."""
        assert callable(seconds2wall)

    def test_midnight(self) -> None:
        """0 seconds past midnight → 0 HHMMSS."""
        result = seconds2wall(np.array([0.0, 3600.0]))
        npt.assert_allclose(result[0], 0.0, atol=ATOL)

    def test_nine_thirty(self) -> None:
        """34200 seconds (9:30 AM) → 93000.0 HHMMSS."""
        result = seconds2wall(np.array([34200.0, 36000.0]))
        npt.assert_allclose(result[0], 93000.0, atol=0.01)

    def test_noon(self) -> None:
        """43200 seconds (12:00 PM) → 120000.0 HHMMSS."""
        result = seconds2wall(np.array([43200.0, 43201.0]))
        npt.assert_allclose(result[0], 120000.0, atol=0.01)

    def test_return_type_ndarray(self) -> None:
        """Return type should be numpy ndarray."""
        result = seconds2wall(np.array([34200.0, 57600.0]))
        assert isinstance(result, np.ndarray)


# =====================================================================
# unit2seconds
# =====================================================================

class TestUnit2Seconds:
    """Unit tests for unit2seconds."""

    def test_import(self) -> None:
        """Module should import without error."""
        assert callable(unit2seconds)

    def test_boundary_values(self) -> None:
        """Unit 0 → seconds0, unit 1 → seconds1."""
        seconds0 = 34200.0
        seconds1 = 57600.0
        result = unit2seconds(np.array([0.0, 1.0]), seconds0, seconds1)
        npt.assert_allclose(result[0], seconds0, atol=ATOL)
        npt.assert_allclose(result[-1], seconds1, atol=ATOL)

    def test_midpoint(self) -> None:
        """Unit 0.5 → midpoint of [seconds0, seconds1]."""
        seconds0 = 34200.0
        seconds1 = 57600.0
        mid = (seconds0 + seconds1) / 2.0
        result = unit2seconds(np.array([0.0, 0.5, 1.0]), seconds0, seconds1)
        npt.assert_allclose(result[1], mid, atol=ATOL)


# =====================================================================
# wall2seconds
# =====================================================================

class TestWall2Seconds:
    """Unit tests for wall2seconds."""

    def test_import(self) -> None:
        """Module should import without error."""
        assert callable(wall2seconds)

    def test_nine_thirty(self) -> None:
        """93000.0 HHMMSS → 34200 seconds."""
        result = wall2seconds(np.array([93000.0, 120000.0]))
        npt.assert_allclose(result[0], 34200.0, atol=0.01)

    def test_midnight(self) -> None:
        """0 HHMMSS → 0 seconds."""
        result = wall2seconds(np.array([0.0, 10000.0]))
        npt.assert_allclose(result[0], 0.0, atol=ATOL)

    def test_noon(self) -> None:
        """120000.0 HHMMSS → 43200 seconds."""
        result = wall2seconds(np.array([120000.0, 160000.0]))
        npt.assert_allclose(result[0], 43200.0, atol=0.01)

    def test_four_pm(self) -> None:
        """160000.0 HHMMSS → 57600 seconds."""
        result = wall2seconds(np.array([160000.0, 160001.0]))
        npt.assert_allclose(result[0], 57600.0, atol=0.01)


# =====================================================================
# wall2unit
# =====================================================================

class TestWall2Unit:
    """Unit tests for wall2unit."""

    def test_import(self) -> None:
        """Module should import without error."""
        assert callable(wall2unit)

    def test_boundary_values(self) -> None:
        """Start wall → 0, end wall → 1."""
        wall0 = 93000.0   # 9:30 AM
        wall1 = 160000.0  # 4:00 PM
        result = wall2unit(np.array([wall0, wall1]), wall0, wall1)
        npt.assert_allclose(result[0], 0.0, atol=ATOL)
        npt.assert_allclose(result[-1], 1.0, atol=ATOL)


# =====================================================================
# unit2wall
# =====================================================================

class TestUnit2Wall:
    """Unit tests for unit2wall."""

    def test_import(self) -> None:
        """Module should import without error."""
        assert callable(unit2wall)

    def test_boundary_values(self) -> None:
        """Unit 0 → wall0, unit 1 → wall1."""
        wall0 = 93000.0   # 9:30 AM
        wall1 = 160000.0  # 4:00 PM
        result = unit2wall(np.array([0.0, 1.0]), wall0, wall1)
        npt.assert_allclose(result[0], wall0, atol=0.01)
        npt.assert_allclose(result[-1], wall1, atol=0.01)


# =====================================================================
# Roundtrip tests
# =====================================================================

class TestTimeConversionRoundtrips:
    """Roundtrip tests for time conversion functions."""

    def test_seconds_unit_roundtrip(self) -> None:
        """seconds → unit → seconds roundtrip should recover original."""
        seconds0 = 34200.0  # 9:30 AM
        seconds1 = 57600.0  # 4:00 PM
        seconds_orig = np.linspace(seconds0, seconds1, 20)
        unit_time = seconds2unit(seconds_orig, seconds0, seconds1)
        seconds_recovered = unit2seconds(unit_time, seconds0, seconds1)
        npt.assert_allclose(seconds_recovered, seconds_orig, atol=ATOL)

    def test_wall_seconds_roundtrip(self) -> None:
        """wall → seconds → wall roundtrip should recover original."""
        wall_orig = np.array([93000.0, 120000.0, 160000.0])
        seconds = wall2seconds(wall_orig)
        wall_recovered = seconds2wall(seconds)
        npt.assert_allclose(wall_recovered, wall_orig, atol=0.01)

    def test_wall_unit_roundtrip(self) -> None:
        """wall → unit → wall roundtrip should recover original."""
        wall0 = 93000.0   # 9:30 AM
        wall1 = 160000.0  # 4:00 PM
        wall_orig = np.array([93000.0, 120000.0, 160000.0])
        unit_time = wall2unit(wall_orig, wall0, wall1)
        wall_recovered = unit2wall(unit_time, wall0, wall1)
        npt.assert_allclose(wall_recovered, wall_orig, atol=0.01)
