"""Parametrized pytest parity tests for the 6 time conversion utility functions
migrated from MATLAB's realized volatility toolbox.

Tests cover:
- ``wall2seconds`` — wall clock (HHMMSS) → seconds past midnight
- ``seconds2wall`` — seconds past midnight → wall clock (HHMMSS)
- ``seconds2unit`` — seconds → unit-normalized [0,1] time
- ``unit2seconds`` — unit-normalized → seconds
- ``wall2unit``    — wall clock → unit-normalized
- ``unit2wall``    — unit-normalized → wall clock

Per AAP Section 0.7.1:
    All numerical assertions use
    ``numpy.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-4)``.

Per AAP Section 0.7.3 MATLAB Translation Conventions:
    MATLAB wall time format HHMMSS:
        hr = floor(wall / 10000)
        mm = floor(wall / 100) - hr * 100
        ss = rem(wall, 100)
    seconds2unit formula:
        unit = (seconds - seconds0) / (seconds1 - seconds0)
    unit2seconds formula:
        seconds = seconds0 + (seconds1 - seconds0) * unit
"""

import os
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.realized.wall2seconds import wall2seconds
from mfe_toolbox.realized.seconds2wall import seconds2wall
from mfe_toolbox.realized.seconds2unit import seconds2unit
from mfe_toolbox.realized.unit2seconds import unit2seconds
from mfe_toolbox.realized.wall2unit import wall2unit
from mfe_toolbox.realized.unit2wall import unit2wall

# Import shared tolerance constants and fixture helpers from conftest.
# Per schema: members_accessed = ['ATOL', 'RTOL', 'realized_fixture_dir', 'load_fixture_npy']
from tests.conftest import ATOL, RTOL, load_fixture_npy

# ---------------------------------------------------------------------------
# Module-level Fixture Directory Resolution
# ---------------------------------------------------------------------------
# Per AAP Section 0.7.2: Optional MFE_FIXTURE_DIR environment variable
# override for fixture path during CI.  Used in @pytest.mark.skipif
# decorators which evaluate at collection time (before fixtures are injected).
FIXTURE_DIR: str = os.environ.get(
    'MFE_FIXTURE_DIR',
    os.path.join(os.path.dirname(__file__), '..', 'fixtures', 'realized'),
)


def _fixture_exists(name: str) -> bool:
    """Check whether a ``.npy`` fixture file exists in the realized directory.

    Used for ``@pytest.mark.skipif`` decorators that need module-level
    evaluation before pytest session fixtures are available.
    """
    return os.path.exists(os.path.join(FIXTURE_DIR, f'{name}.npy'))


# ---------------------------------------------------------------------------
# Pytest Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_wall_times() -> np.ndarray:
    """Representative HHMMSS values covering NYSE market hours.

    Ref: wall2seconds.m — wall times encode hours, minutes, seconds as
    ``HH * 10000 + MM * 100 + SS`` (e.g., 93000 = 09:30:00).
    """
    return np.array([93000.0, 100000.0, 120000.0, 133000.0, 150000.0, 160000.0])


@pytest.fixture
def sample_seconds() -> np.ndarray:
    """Seconds past midnight corresponding to *sample_wall_times*.

    Manually computed:
        93000  → 9*3600 + 30*60 = 34200
        100000 → 10*3600        = 36000
        120000 → 12*3600        = 43200
        133000 → 13*3600 + 30*60 = 48600
        150000 → 15*3600        = 54000
        160000 → 16*3600        = 57600
    """
    return np.array([34200.0, 36000.0, 43200.0, 48600.0, 54000.0, 57600.0])


@pytest.fixture
def unit_times_01() -> np.ndarray:
    """Unit-normalized times [0, 1] for standard market hours.

    Based on seconds0=34200 (9:30 AM) and seconds1=57600 (4:00 PM).
    ``unit = (seconds - 34200) / (57600 - 34200)``
    """
    seconds = np.array([34200.0, 36000.0, 43200.0, 48600.0, 54000.0, 57600.0])
    seconds0, seconds1 = 34200.0, 57600.0
    return (seconds - seconds0) / (seconds1 - seconds0)


# =====================================================================
# TestWall2Seconds
# =====================================================================

class TestWall2Seconds:
    """Tests for ``wall2seconds()`` — converts HHMMSS wall clock to seconds
    past midnight.

    Ref: wall2seconds.m — MATLAB formula:
        hr = floor(wall / 10000)
        mm = floor(wall / 100) - hr * 100
        ss = rem(wall, 100)
        seconds = 3600 * hr + 60 * mm + ss
    """

    def test_basic_conversion(self) -> None:
        """Convert known wall times and verify against manually computed seconds.

        93000  → 9*3600 + 30*60 + 0 = 34200
        120000 → 12*3600 + 0 + 0    = 43200
        160000 → 16*3600 + 0 + 0    = 57600
        """
        wall = np.array([93000.0, 120000.0, 160000.0])
        expected = np.array([34200.0, 43200.0, 57600.0])
        result = wall2seconds(wall)
        npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)

    def test_midnight(self) -> None:
        """wall=0 → seconds=0 (midnight).

        Ref: wall2seconds.m:48 — seconds = 3600*0 + 60*0 + 0 = 0
        """
        result = wall2seconds(np.array([0.0]))
        npt.assert_allclose(result, np.array([0.0]), atol=ATOL, rtol=RTOL)

    def test_single_value(self) -> None:
        """Scalar-like (single-element array) input produces correct output."""
        result = wall2seconds(np.array([93000.0]))
        assert isinstance(result, np.ndarray)
        npt.assert_allclose(result[0], 34200.0, atol=ATOL, rtol=RTOL)

    def test_roundtrip_with_seconds2wall(self, sample_wall_times: np.ndarray) -> None:
        """wall → seconds → wall should reproduce original values.

        Ref: seconds2wall.m is the algebraic inverse of wall2seconds.m.
        """
        seconds = wall2seconds(sample_wall_times)
        recovered = seconds2wall(seconds)
        npt.assert_allclose(recovered, sample_wall_times, atol=ATOL, rtol=RTOL)

    def test_invalid_wall_negative(self) -> None:
        """Negative wall times should raise ``ValueError``.

        Ref: wall2seconds.m:30 — ``if any(wall<0) → error``
        """
        with pytest.raises(ValueError):
            wall2seconds(np.array([-1.0]))

    def test_invalid_wall_too_large(self) -> None:
        """wall >= 240000 should raise ``ValueError``.

        Ref: wall2seconds.m:30 — ``if any(wall>=240000) → error``
        """
        with pytest.raises(ValueError):
            wall2seconds(np.array([240000.0]))

    def test_invalid_minutes_over_60(self) -> None:
        """Wall times with mm > 60 (e.g. 127000) should raise ``ValueError``.

        127000 → hr=12, mm=floor(127000/100)-12*100=1270-1200=70 → 70>60 → error
        Ref: wall2seconds.m:40-41
        """
        with pytest.raises(ValueError):
            wall2seconds(np.array([127000.0]))

    def test_invalid_seconds_over_60(self) -> None:
        """Wall times with ss > 60 (e.g. 120070) should raise ``ValueError``.

        120070 → hr=12, mm=0, ss=rem(120070,100)=70 → 70>60 → error
        Ref: wall2seconds.m:40-41
        """
        with pytest.raises(ValueError):
            wall2seconds(np.array([120070.0]))

    @pytest.mark.parametrize("wall, expected_seconds", [
        (np.array([10000.0]),  np.array([3600.0])),    # 01:00:00
        (np.array([12345.0]),  np.array([5025.0])),    # 01:23:45
        (np.array([101534.0]), np.array([36934.0])),   # 10:15:34
        (np.array([235959.0]), np.array([86399.0])),   # 23:59:59
    ])
    def test_parametrized_conversions(self, wall: np.ndarray,
                                      expected_seconds: np.ndarray) -> None:
        """Parametrized test for various wall time → seconds conversions."""
        result = wall2seconds(wall)
        npt.assert_allclose(result, expected_seconds, atol=ATOL, rtol=RTOL)

    def test_sub_second_precision(self) -> None:
        """Fractional seconds preserved through ``fmod``.

        93000.5  → 9*3600 + 30*60 + 0.5  = 34200.5
        160000.75 → 16*3600 + 0 + 0.75   = 57600.75
        Ref: wall2seconds.m:38 — ``ss = rem(wall, 100)`` preserves fraction.
        """
        wall = np.array([93000.5, 160000.75])
        expected = np.array([34200.5, 57600.75])
        result = wall2seconds(wall)
        npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)

    def test_return_type(self) -> None:
        """Return type must be ``numpy.ndarray``."""
        result = wall2seconds(np.array([93000.0]))
        assert isinstance(result, np.ndarray)

    @pytest.mark.skipif(not _fixture_exists('wall2seconds'),
                        reason='Fixture wall2seconds.npy not found')
    def test_fixture_parity_all(self, realized_fixture_dir: Path) -> None:
        """All wall → seconds parity against MATLAB reference outputs."""
        data = load_fixture_npy(realized_fixture_dir, 'wall2seconds').item()
        result = wall2seconds(data['all_wall'])
        npt.assert_allclose(result, data['all_seconds'], atol=ATOL, rtol=RTOL,
                            err_msg='wall2seconds all fixture parity failed')

    @pytest.mark.skipif(not _fixture_exists('wall2seconds'),
                        reason='Fixture wall2seconds.npy not found')
    def test_fixture_parity_market_range(self, realized_fixture_dir: Path) -> None:
        """Market-range wall → seconds parity against MATLAB reference."""
        data = load_fixture_npy(realized_fixture_dir, 'wall2seconds').item()
        result = wall2seconds(data['market_range_wall'])
        npt.assert_allclose(result, data['market_range_seconds'], atol=ATOL,
                            rtol=RTOL,
                            err_msg='wall2seconds market_range parity failed')

    @pytest.mark.skipif(not _fixture_exists('wall2seconds'),
                        reason='Fixture wall2seconds.npy not found')
    def test_fixture_parity_sub_second(self, realized_fixture_dir: Path) -> None:
        """Sub-second wall → seconds parity against MATLAB reference."""
        data = load_fixture_npy(realized_fixture_dir, 'wall2seconds').item()
        result = wall2seconds(data['sub_second_wall'])
        npt.assert_allclose(result, data['sub_second_seconds'], atol=ATOL,
                            rtol=RTOL,
                            err_msg='wall2seconds sub_second parity failed')


# =====================================================================
# TestSeconds2Wall
# =====================================================================

class TestSeconds2Wall:
    """Tests for ``seconds2wall()`` — converts seconds past midnight to
    HHMMSS wall clock.

    Ref: seconds2wall.m — MATLAB formula:
        hr = floor(seconds / 3600)
        mm = floor((seconds - hr*3600) / 60)
        ss = rem(seconds, 60)
        wall = hr*10000 + mm*100 + ss
    """

    def test_basic_conversion(self) -> None:
        """Known seconds → wall: 34200→93000, 43200→120000, 57600→160000."""
        seconds = np.array([34200.0, 43200.0, 57600.0])
        expected = np.array([93000.0, 120000.0, 160000.0])
        result = seconds2wall(seconds)
        npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)

    def test_zero_seconds(self) -> None:
        """0 seconds → 0 (midnight).

        Ref: seconds2wall.m:38-41 — hr=0, mm=0, ss=0 → wall=0
        """
        result = seconds2wall(np.array([0.0]))
        npt.assert_allclose(result, np.array([0.0]), atol=ATOL, rtol=RTOL)

    def test_roundtrip_with_wall2seconds(self, sample_seconds: np.ndarray) -> None:
        """seconds → wall → seconds should reproduce original values.

        Ref: wall2seconds.m is the algebraic inverse of seconds2wall.m.
        """
        wall = seconds2wall(sample_seconds)
        recovered = wall2seconds(wall)
        npt.assert_allclose(recovered, sample_seconds, atol=ATOL, rtol=RTOL)

    def test_fractional_seconds(self) -> None:
        """Verify handling of fractional seconds (e.g. 34200.5 → 93000.5).

        Ref: seconds2wall.m:40 — ``ss = rem(seconds, 60)`` preserves fraction.
        """
        seconds = np.array([34200.5])
        expected = np.array([93000.5])
        result = seconds2wall(seconds)
        npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)

    def test_invalid_negative(self) -> None:
        """Negative seconds should raise ``ValueError``.

        Ref: seconds2wall.m:29 — ``if any(seconds<0) → error``
        """
        with pytest.raises(ValueError):
            seconds2wall(np.array([-1.0]))

    def test_invalid_too_large(self) -> None:
        """Seconds >= 86400 should raise ``ValueError`` (24 hours).

        Ref: seconds2wall.m:29 — ``if any(seconds>=(24*3600)) → error``
        """
        with pytest.raises(ValueError):
            seconds2wall(np.array([86400.0]))

    @pytest.mark.parametrize("seconds, expected_wall", [
        (np.array([3600.0]),  np.array([10000.0])),    # 01:00:00
        (np.array([5025.0]),  np.array([12345.0])),    # 01:23:45
        (np.array([36934.0]), np.array([101534.0])),   # 10:15:34
        (np.array([86399.0]), np.array([235959.0])),   # 23:59:59
    ])
    def test_parametrized_conversions(self, seconds: np.ndarray,
                                      expected_wall: np.ndarray) -> None:
        """Parametrized test for various seconds → wall conversions."""
        result = seconds2wall(seconds)
        npt.assert_allclose(result, expected_wall, atol=ATOL, rtol=RTOL)

    def test_return_type(self) -> None:
        """Return type must be ``numpy.ndarray``."""
        result = seconds2wall(np.array([0.0]))
        assert isinstance(result, np.ndarray)

    @pytest.mark.skipif(not _fixture_exists('seconds2wall'),
                        reason='Fixture seconds2wall.npy not found')
    def test_fixture_parity_all(self, realized_fixture_dir: Path) -> None:
        """All seconds → wall parity against MATLAB reference outputs."""
        data = load_fixture_npy(realized_fixture_dir, 'seconds2wall').item()
        result = seconds2wall(data['all_seconds'])
        npt.assert_allclose(result, data['all_wall'], atol=ATOL, rtol=RTOL,
                            err_msg='seconds2wall all fixture parity failed')

    @pytest.mark.skipif(not _fixture_exists('seconds2wall'),
                        reason='Fixture seconds2wall.npy not found')
    def test_fixture_parity_market_range(self, realized_fixture_dir: Path) -> None:
        """Market-range seconds → wall parity against MATLAB reference."""
        data = load_fixture_npy(realized_fixture_dir, 'seconds2wall').item()
        result = seconds2wall(data['market_range_seconds'])
        npt.assert_allclose(result, data['market_range_wall'], atol=ATOL,
                            rtol=RTOL,
                            err_msg='seconds2wall market_range parity failed')


# =====================================================================
# TestSeconds2Unit
# =====================================================================

class TestSeconds2Unit:
    """Tests for ``seconds2unit()`` — converts seconds past midnight to
    unit [0, 1] interval.

    Key formula from MATLAB source (seconds2unit.m:65):
        ``unit = (seconds - seconds0) / (seconds1 - seconds0)``
    """

    def test_basic_conversion(self) -> None:
        """Seconds [34200, 43200, 57600] with seconds0=34200, seconds1=57600.

        unit = (seconds - 34200) / (57600 - 34200) = (s - 34200) / 23400
        → [0.0, 9000/23400 ≈ 0.384615, 1.0]
        """
        seconds = np.array([34200.0, 43200.0, 57600.0])
        expected = np.array([0.0, 9000.0 / 23400.0, 1.0])
        result = seconds2unit(seconds, 34200.0, 57600.0)
        npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)

    def test_full_range(self) -> None:
        """Seconds spanning the full window give unit values in [0, 1]."""
        seconds0, seconds1 = 34200.0, 57600.0
        seconds = np.linspace(seconds0, seconds1, 20)
        result = seconds2unit(seconds, seconds0, seconds1)
        # Ref: seconds2unit.m:65 — linear map guarantees [0, 1] for in-range inputs
        assert np.all(result >= -ATOL), 'Unit values should be >= 0'
        assert np.all(result <= 1.0 + ATOL), 'Unit values should be <= 1'
        npt.assert_allclose(result[0], 0.0, atol=ATOL)
        npt.assert_allclose(result[-1], 1.0, atol=ATOL)

    def test_linear_spacing(self) -> None:
        """Linearly spaced seconds produce linearly spaced units.

        Ref: seconds2unit.m:65 — linear formula preserves equi-spacing.
        """
        seconds0, seconds1 = 34200.0, 57600.0
        seconds = np.linspace(seconds0, seconds1, 11)
        expected = np.linspace(0.0, 1.0, 11)
        result = seconds2unit(seconds, seconds0, seconds1)
        npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)

    def test_minimum_two_elements(self) -> None:
        """Single element should raise ``ValueError`` (MATLAB requires ≥2).

        Ref: seconds2unit.m:59-61 — ``if length(seconds)<2 → error``
        """
        with pytest.raises(ValueError, match='at least 2 elements'):
            seconds2unit(np.array([34200.0]), 34200.0, 57600.0)

    def test_invalid_seconds0_gte_seconds1(self) -> None:
        """seconds0 >= seconds1 should raise ``ValueError``.

        Ref: seconds2unit.m:44 — ``(seconds1-seconds0)<eps → error``
        """
        with pytest.raises(ValueError):
            seconds2unit(np.array([34200.0, 57600.0]), 57600.0, 34200.0)

    def test_invalid_seconds0_equals_seconds1(self) -> None:
        """seconds0 == seconds1 should raise ``ValueError``."""
        with pytest.raises(ValueError):
            seconds2unit(np.array([34200.0, 57600.0]), 34200.0, 34200.0)

    def test_column_vector(self) -> None:
        """Result should be a 1-D ``numpy.ndarray``.

        Ref: seconds2unit.m:48-50 — MATLAB transposes row to column.
        Python equivalent flattens to 1-D via ``flatten()``.
        """
        result = seconds2unit(np.array([34200.0, 57600.0]), 34200.0, 57600.0)
        assert result.ndim == 1, 'Result must be 1-D'
        assert isinstance(result, np.ndarray)

    def test_negative_seconds_raises(self) -> None:
        """Negative seconds should raise ``ValueError``.

        Ref: seconds2unit.m:34-35
        """
        with pytest.raises(ValueError):
            seconds2unit(np.array([-1.0, 100.0]), 0.0, 57600.0)

    def test_negative_seconds0_raises(self) -> None:
        """Negative seconds0 should raise ``ValueError``.

        Ref: seconds2unit.m:37-38
        """
        with pytest.raises(ValueError):
            seconds2unit(np.array([34200.0, 57600.0]), -1.0, 57600.0)

    def test_negative_seconds1_raises(self) -> None:
        """Negative seconds1 should raise ``ValueError``.

        Ref: seconds2unit.m:40-41
        """
        with pytest.raises(ValueError):
            seconds2unit(np.array([34200.0, 57600.0]), 34200.0, -1.0)

    def test_return_type(self) -> None:
        """Return type must be ``numpy.ndarray``."""
        result = seconds2unit(np.array([34200.0, 57600.0]), 34200.0, 57600.0)
        assert isinstance(result, np.ndarray)

    @pytest.mark.skipif(not _fixture_exists('seconds2unit'),
                        reason='Fixture seconds2unit.npy not found')
    @pytest.mark.parametrize("scenario", [
        'standard_market_hours', 'edge_values', 'midpoint', 'extended_hours',
    ])
    def test_fixture_parity(self, realized_fixture_dir: Path,
                            scenario: str) -> None:
        """Compare against MATLAB reference outputs for ``seconds2unit``.

        Each scenario contains seconds_input, seconds0, seconds1, and the
        expected unit array generated by Octave.
        """
        data = load_fixture_npy(realized_fixture_dir, 'seconds2unit').item()
        case = data[scenario]
        result = seconds2unit(
            case['seconds_input'],
            float(case['seconds0']),
            float(case['seconds1']),
        )
        npt.assert_allclose(
            result, case['unit'], atol=ATOL, rtol=RTOL,
            err_msg=f'seconds2unit {scenario} fixture parity failed',
        )


# =====================================================================
# TestUnit2Seconds
# =====================================================================

class TestUnit2Seconds:
    """Tests for ``unit2seconds()`` — inverse of ``seconds2unit``.

    Formula from MATLAB source (unit2seconds.m:63):
        ``seconds = seconds0 + (seconds1 - seconds0) * unit``
    """

    def test_basic_conversion(self) -> None:
        """Known unit values → expected seconds.

        unit=0.0 → 34200,  unit=0.5 → 45900,  unit=1.0 → 57600
        """
        unit = np.array([0.0, 0.5, 1.0])
        expected = np.array([34200.0, 45900.0, 57600.0])
        result = unit2seconds(unit, 34200.0, 57600.0)
        npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)

    def test_roundtrip_with_seconds2unit(self) -> None:
        """unit → seconds → unit produces original values.

        Ref: unit2seconds.m and seconds2unit.m are algebraic inverses.
        """
        seconds0, seconds1 = 34200.0, 57600.0
        unit_orig = np.linspace(0.0, 1.0, 20)
        seconds = unit2seconds(unit_orig, seconds0, seconds1)
        unit_recovered = seconds2unit(seconds, seconds0, seconds1)
        npt.assert_allclose(unit_recovered, unit_orig, atol=ATOL, rtol=RTOL)

    def test_boundary_values(self) -> None:
        """unit=0 → seconds0,  unit=1 → seconds1.

        Ref: unit2seconds.m:63 — linear mapping boundary conditions.
        """
        seconds0, seconds1 = 34200.0, 57600.0
        result = unit2seconds(np.array([0.0, 1.0]), seconds0, seconds1)
        npt.assert_allclose(result[0], seconds0, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(result[-1], seconds1, atol=ATOL, rtol=RTOL)

    def test_linearity(self) -> None:
        """Linearly spaced units produce linearly spaced seconds."""
        seconds0, seconds1 = 34200.0, 57600.0
        unit = np.linspace(0.0, 1.0, 11)
        expected = np.linspace(seconds0, seconds1, 11)
        result = unit2seconds(unit, seconds0, seconds1)
        npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)

    def test_invalid_seconds0_negative(self) -> None:
        """Negative seconds0 should raise ``ValueError``.

        Ref: unit2seconds.m:34-35
        """
        with pytest.raises(ValueError):
            unit2seconds(np.array([0.0, 1.0]), -1.0, 57600.0)

    def test_invalid_seconds1_negative(self) -> None:
        """Negative seconds1 should raise ``ValueError``.

        Ref: unit2seconds.m:37-38
        """
        with pytest.raises(ValueError):
            unit2seconds(np.array([0.0, 1.0]), 34200.0, -1.0)

    def test_invalid_seconds0_gte_seconds1(self) -> None:
        """seconds0 >= seconds1 should raise ``ValueError``.

        Ref: unit2seconds.m:41-43
        """
        with pytest.raises(ValueError):
            unit2seconds(np.array([0.0, 1.0]), 57600.0, 34200.0)

    def test_return_type(self) -> None:
        """Return type must be ``numpy.ndarray``."""
        result = unit2seconds(np.array([0.0, 1.0]), 34200.0, 57600.0)
        assert isinstance(result, np.ndarray)

    def test_extrapolation_beyond_unit_range(self) -> None:
        """Unit values outside [0, 1] extrapolate linearly.

        Ref: unit2seconds.m — MATLAB does NOT validate unit range,
        so values <0 or >1 produce valid (extrapolated) seconds.
        This test documents MATLAB-compatible behavior.
        """
        seconds0, seconds1 = 34200.0, 57600.0
        # unit = -0.1 → seconds0 + (seconds1-seconds0)*(-0.1)
        #             = 34200 - 2340 = 31860
        # unit =  1.1 → seconds0 + (seconds1-seconds0)*(1.1)
        #             = 34200 + 25740 = 59940
        unit = np.array([-0.1, 1.1])
        expected = np.array([31860.0, 59940.0])
        result = unit2seconds(unit, seconds0, seconds1)
        npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)

    @pytest.mark.skipif(not _fixture_exists('unit2seconds'),
                        reason='Fixture unit2seconds.npy not found')
    @pytest.mark.parametrize("scenario", [
        'standard_market_hours', 'edge_values', 'midpoint', 'extended_hours',
    ])
    def test_fixture_parity(self, realized_fixture_dir: Path,
                            scenario: str) -> None:
        """Compare against MATLAB reference outputs for ``unit2seconds``."""
        data = load_fixture_npy(realized_fixture_dir, 'unit2seconds').item()
        case = data[scenario]
        result = unit2seconds(
            case['unit_input'],
            float(case['seconds0']),
            float(case['seconds1']),
        )
        npt.assert_allclose(
            result, case['seconds'], atol=ATOL, rtol=RTOL,
            err_msg=f'unit2seconds {scenario} fixture parity failed',
        )


# =====================================================================
# TestWall2Unit
# =====================================================================

class TestWall2Unit:
    """Tests for ``wall2unit()`` — composition: wall → seconds → unit.

    Ref: wall2unit.m:73-76:
        wall0 = wall2seconds(wall0)
        wall1 = wall2seconds(wall1)
        seconds = wall2seconds(wall)
        unit = (seconds - wall0) / (wall1 - wall0)
    """

    def test_basic_conversion(self) -> None:
        """Wall times → unit-normalized values.

        93000  → 0.0
        120000 → (43200-34200) / (57600-34200) = 9000/23400 ≈ 0.384615
        160000 → 1.0
        """
        wall = np.array([93000.0, 120000.0, 160000.0])
        expected = np.array([0.0, 9000.0 / 23400.0, 1.0])
        result = wall2unit(wall, 93000.0, 160000.0)
        npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)

    def test_roundtrip_with_unit2wall(self) -> None:
        """wall → unit → wall produces original values.

        Ref: unit2wall.m is the inverse of wall2unit.m.
        Note: unit2wall applies rounding — ``round(100000*x)/100000`` — which
        may introduce up to 5e-6 discrepancy in fractional-second wall times.
        Whole-second wall times like 93000, 120000, etc. are reproduced exactly.
        """
        wall0, wall1 = 93000.0, 160000.0
        wall_orig = np.array([93000.0, 100000.0, 120000.0,
                              133000.0, 150000.0, 160000.0])
        unit = wall2unit(wall_orig, wall0, wall1)
        wall_recovered = unit2wall(unit, wall0, wall1)
        # Ref: unit2wall.m:66 — rounding may introduce small differences
        npt.assert_allclose(wall_recovered, wall_orig, atol=1e-3, rtol=RTOL)

    def test_market_hours(self) -> None:
        """Standard NYSE 9:30–16:00 wall times → unit [0, 1].

        Ref: wall2unit.m:73-76 — standard NYSE market hours are the canonical
        test scenario for this helper function.
        """
        wall0, wall1 = 93000.0, 160000.0
        wall = np.array([93000.0, 160000.0])
        result = wall2unit(wall, wall0, wall1)
        npt.assert_allclose(result[0], 0.0, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(result[-1], 1.0, atol=ATOL, rtol=RTOL)

    def test_midpoint(self) -> None:
        """Midpoint wall time maps to unit 0.5.

        wall0=93000 (34200 s), wall1=160000 (57600 s)
        midpoint seconds = (34200 + 57600) / 2 = 45900
        45900 s → 12h 45m 0s → wall = 124500
        """
        wall0, wall1 = 93000.0, 160000.0
        wall = np.array([93000.0, 124500.0, 160000.0])
        result = wall2unit(wall, wall0, wall1)
        npt.assert_allclose(result[1], 0.5, atol=ATOL, rtol=RTOL)

    def test_minimum_two_elements(self) -> None:
        """Single element should raise ``ValueError``.

        Ref: wall2unit.m:46-48 — ``if length(wall)<2 → error``
        """
        with pytest.raises(ValueError, match='at least 2 elements'):
            wall2unit(np.array([93000.0]), 93000.0, 160000.0)

    def test_invalid_wall_negative(self) -> None:
        """Negative wall times should raise ``ValueError``.

        Ref: wall2unit.m:34 — ``if any(wall<0) → error``
        """
        with pytest.raises(ValueError):
            wall2unit(np.array([-1.0, 93000.0]), 93000.0, 160000.0)

    def test_invalid_wall_too_large(self) -> None:
        """Wall times >= 240000 should raise ``ValueError``.

        Ref: wall2unit.m:34 — ``if any(wall>=240000) → error``
        """
        with pytest.raises(ValueError):
            wall2unit(np.array([93000.0, 250000.0]), 93000.0, 160000.0)

    def test_invalid_wall0(self) -> None:
        """Invalid wall0 (e.g. mm >= 60) should raise ``ValueError``.

        Ref: wall2unit.m:61-62
        """
        with pytest.raises(ValueError):
            wall2unit(np.array([93000.0, 160000.0]), 127000.0, 160000.0)

    def test_invalid_wall1(self) -> None:
        """Invalid wall1 (e.g. mm >= 60) should raise ``ValueError``.

        Ref: wall2unit.m:65-66
        """
        with pytest.raises(ValueError):
            wall2unit(np.array([93000.0, 160000.0]), 93000.0, 167000.0)

    def test_return_type(self) -> None:
        """Return type must be ``numpy.ndarray``."""
        result = wall2unit(np.array([93000.0, 160000.0]), 93000.0, 160000.0)
        assert isinstance(result, np.ndarray)

    @pytest.mark.skipif(not _fixture_exists('wall2unit'),
                        reason='Fixture wall2unit.npy not found')
    @pytest.mark.parametrize("scenario", [
        'standard_market_hours', 'edge_values', 'midpoint', 'extended_hours',
    ])
    def test_fixture_parity(self, realized_fixture_dir: Path,
                            scenario: str) -> None:
        """Compare against MATLAB reference outputs for ``wall2unit``."""
        data = load_fixture_npy(realized_fixture_dir, 'wall2unit').item()
        case = data[scenario]
        result = wall2unit(
            case['wall_input'],
            float(case['wall0']),
            float(case['wall1']),
        )
        npt.assert_allclose(
            result, case['unit'], atol=ATOL, rtol=RTOL,
            err_msg=f'wall2unit {scenario} fixture parity failed',
        )


# =====================================================================
# TestUnit2Wall
# =====================================================================

class TestUnit2Wall:
    """Tests for ``unit2wall()`` — inverse of ``wall2unit``.

    Ref: unit2wall.m:62-66:
        wall0 = wall2seconds(wall0)
        wall1 = wall2seconds(wall1)
        seconds = wall0 + (wall1 - wall0) * unit
        wall = round(100000 * seconds2wall(seconds)) / 100000
    """

    def test_basic_conversion(self) -> None:
        """Unit values → wall times.

        unit=0.0 → 93000,  unit=0.5 → 124500,  unit=1.0 → 160000
        """
        unit = np.array([0.0, 0.5, 1.0])
        expected = np.array([93000.0, 124500.0, 160000.0])
        result = unit2wall(unit, 93000.0, 160000.0)
        npt.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)

    def test_roundtrip_with_wall2unit(self) -> None:
        """unit → wall → unit produces original values.

        Ref: wall2unit.m is the inverse of unit2wall.m.
        """
        wall0, wall1 = 93000.0, 160000.0
        unit_orig = np.linspace(0.0, 1.0, 14)
        wall = unit2wall(unit_orig, wall0, wall1)
        unit_recovered = wall2unit(wall, wall0, wall1)
        # Roundtrip tolerance is slightly relaxed due to unit2wall's 5-dp
        # rounding — Ref: unit2wall.m:66 round(100000*x)/100000
        npt.assert_allclose(unit_recovered, unit_orig, atol=1e-4, rtol=RTOL)

    def test_boundary_values(self) -> None:
        """unit=0 → wall0, unit=1 → wall1.

        Ref: unit2wall.m:65-66 — linear mapping boundary conditions.
        """
        wall0, wall1 = 93000.0, 160000.0
        result = unit2wall(np.array([0.0, 1.0]), wall0, wall1)
        npt.assert_allclose(result[0], wall0, atol=ATOL, rtol=RTOL)
        npt.assert_allclose(result[-1], wall1, atol=ATOL, rtol=RTOL)

    def test_minimum_two_elements(self) -> None:
        """Single element should raise ``ValueError``.

        Ref: unit2wall.m:42-44 — ``if length(unit)<2 → error``
        """
        with pytest.raises(ValueError, match='at least 2 elements'):
            unit2wall(np.array([0.5]), 93000.0, 160000.0)

    def test_invalid_wall0(self) -> None:
        """Invalid wall0 (e.g. mm >= 60) should raise ``ValueError``.

        Ref: unit2wall.m:50-52
        """
        with pytest.raises(ValueError):
            unit2wall(np.array([0.0, 1.0]), 127000.0, 160000.0)

    def test_invalid_wall1(self) -> None:
        """Invalid wall1 (negative) should raise ``ValueError``.

        Ref: unit2wall.m:54-56
        """
        with pytest.raises(ValueError):
            unit2wall(np.array([0.0, 1.0]), 93000.0, -1.0)

    def test_rounding_behavior(self) -> None:
        """Verify MATLAB rounding pattern: ``round(100000*x)/100000``.

        Ref: unit2wall.m:66 — explicit rounding to 5 decimal places prevents
        floating-point noise from producing spurious microsecond digits.
        """
        unit = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
        result = unit2wall(unit, 93000.0, 160000.0)
        # Verify every output element is properly rounded to 5 decimal places
        rounded = np.round(result * 100000.0) / 100000.0
        npt.assert_allclose(result, rounded, atol=1e-10,
                            err_msg='unit2wall result not rounded to 5 dp')

    def test_return_type(self) -> None:
        """Return type must be ``numpy.ndarray``."""
        result = unit2wall(np.array([0.0, 1.0]), 93000.0, 160000.0)
        assert isinstance(result, np.ndarray)

    @pytest.mark.skipif(not _fixture_exists('unit2wall'),
                        reason='Fixture unit2wall.npy not found')
    @pytest.mark.parametrize("scenario", [
        'standard_market_hours', 'edge_values', 'midpoint', 'extended_hours',
    ])
    def test_fixture_parity(self, realized_fixture_dir: Path,
                            scenario: str) -> None:
        """Compare against MATLAB reference outputs for ``unit2wall``."""
        data = load_fixture_npy(realized_fixture_dir, 'unit2wall').item()
        case = data[scenario]
        result = unit2wall(
            case['unit_input'],
            float(case['wall0']),
            float(case['wall1']),
        )
        npt.assert_allclose(
            result, case['wall'], atol=ATOL, rtol=RTOL,
            err_msg=f'unit2wall {scenario} fixture parity failed',
        )


# =====================================================================
# Cross-Function Roundtrip Integration Tests
# =====================================================================

class TestTimeConversionRoundtrips:
    """Full-chain roundtrip integration tests verifying invertibility
    of the time conversion function pairs.
    """

    def test_seconds_unit_roundtrip_linspace(self) -> None:
        """seconds → unit → seconds roundtrip with linearly spaced data."""
        seconds0, seconds1 = 34200.0, 57600.0
        seconds_orig = np.linspace(seconds0, seconds1, 50)
        unit_time = seconds2unit(seconds_orig, seconds0, seconds1)
        seconds_recovered = unit2seconds(unit_time, seconds0, seconds1)
        npt.assert_allclose(seconds_recovered, seconds_orig, atol=ATOL, rtol=RTOL)

    def test_wall_seconds_roundtrip_market(self, sample_wall_times: np.ndarray,
                                           sample_seconds: np.ndarray) -> None:
        """wall → seconds → wall roundtrip with market-hours data."""
        seconds = wall2seconds(sample_wall_times)
        npt.assert_allclose(seconds, sample_seconds, atol=ATOL, rtol=RTOL)
        wall_recovered = seconds2wall(seconds)
        npt.assert_allclose(wall_recovered, sample_wall_times, atol=ATOL,
                            rtol=RTOL)

    def test_wall_unit_roundtrip_market(self) -> None:
        """wall → unit → wall roundtrip with standard market hours."""
        wall0, wall1 = 93000.0, 160000.0
        wall_orig = np.array([93000.0, 100000.0, 110000.0,
                              120000.0, 130000.0, 140000.0,
                              150000.0, 160000.0])
        unit_time = wall2unit(wall_orig, wall0, wall1)
        wall_recovered = unit2wall(unit_time, wall0, wall1)
        # Slightly relaxed due to unit2wall's 5-decimal-place rounding
        npt.assert_allclose(wall_recovered, wall_orig, atol=1e-3, rtol=RTOL)

    def test_full_chain_wall_seconds_unit_roundtrip(self) -> None:
        """Full chain: wall → seconds → unit → seconds → wall."""
        wall0, wall1 = 93000.0, 160000.0
        seconds0, seconds1 = 34200.0, 57600.0
        wall_orig = np.array([93000.0, 120000.0, 160000.0])

        # Forward chain: wall → seconds → unit
        seconds = wall2seconds(wall_orig)
        unit = seconds2unit(seconds, seconds0, seconds1)

        # Verify intermediate values
        npt.assert_allclose(unit[0], 0.0, atol=ATOL)
        npt.assert_allclose(unit[-1], 1.0, atol=ATOL)

        # Reverse chain: unit → seconds → wall
        seconds_back = unit2seconds(unit, seconds0, seconds1)
        wall_back = seconds2wall(seconds_back)
        npt.assert_allclose(wall_back, wall_orig, atol=ATOL, rtol=RTOL)

    def test_consistency_wall2unit_vs_manual_chain(self) -> None:
        """wall2unit matches manual wall → seconds → unit chain.

        wall2unit is documented as a composition of wall2seconds + seconds2unit.
        Verify the composed result matches the direct call.
        """
        wall0, wall1 = 93000.0, 160000.0
        wall = np.array([93000.0, 110000.0, 130000.0, 160000.0])

        # Direct call
        unit_direct = wall2unit(wall, wall0, wall1)

        # Manual chain
        seconds0 = wall2seconds(np.array([wall0]))[0]
        seconds1 = wall2seconds(np.array([wall1]))[0]
        seconds = wall2seconds(wall)
        unit_manual = seconds2unit(seconds, seconds0, seconds1)

        npt.assert_allclose(unit_direct, unit_manual, atol=ATOL, rtol=RTOL)

    def test_consistency_unit2wall_vs_manual_chain(self) -> None:
        """unit2wall matches manual unit → seconds → wall chain.

        unit2wall is documented as a composition of wall2seconds + unit2seconds
        + seconds2wall + rounding.
        """
        wall0, wall1 = 93000.0, 160000.0
        unit = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

        # Direct call
        wall_direct = unit2wall(unit, wall0, wall1)

        # Manual chain
        seconds0 = wall2seconds(np.array([wall0]))[0]
        seconds1 = wall2seconds(np.array([wall1]))[0]
        seconds = unit2seconds(unit, seconds0, seconds1)
        wall_manual = seconds2wall(seconds)
        # Ref: unit2wall.m:66 — apply the same rounding
        wall_manual = np.round(wall_manual * 100000.0) / 100000.0

        npt.assert_allclose(wall_direct, wall_manual, atol=ATOL, rtol=RTOL)
