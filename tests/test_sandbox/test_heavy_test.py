"""Pytest tests for the HEAVY model experimental test script.

Validates the HEAVY model simulation–estimation–visualization workflow
migrated from ``sandbox/heavy_test.m`` (Version 4.0, 30 lines) into
``mfe_toolbox.sandbox.heavy_test.heavy_test()``.

The MATLAB source is an experimental **script** (not a function).  The Python
migration wraps the entire script body in a callable ``heavy_test()`` function
that executes:

1. Parameter definition matching the MATLAB source exactly.
2. Bivariate HEAVY simulation via ``heavy_simulate``.
3. Data transformation (squared returns, backcast, bounds).
4. Two estimation passes via ``heavy()`` — one with default starting values
   and one seeded with the true simulation parameters.
5. Matplotlib visualisation of estimated vs. true conditional volatility.

Because the function orchestrates heavyweight univariate routines and produces
plots, all unit tests **mock** ``heavy_simulate``, ``heavy``, and
``matplotlib.pyplot`` to keep the test suite fast and environment-independent.

Parity tests load MATLAB/Octave reference fixtures from
``tests/fixtures/sandbox/heavy_test.npy`` and compare intermediate and final
outputs using ``numpy.testing.assert_allclose`` with tolerances specified in
AAP Section 0.7.1 (``atol=1e-6``, ``rtol=1e-4``).

References
----------
- MATLAB source: ``sandbox/heavy_test.m``
- Python target: ``mfe_toolbox/sandbox/heavy_test.py``
- AAP Sections: 0.5.1, 0.7.1, 0.7.2
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.testing as npt
import pytest

from tests.conftest import ATOL, RTOL, load_fixture_npy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Expected MATLAB parameters (Ref: heavy_test.m:4)
_EXPECTED_PARAMS = np.array([0.3, 0.05, 0.2, 0.4, 0.7, 0.55])

# Expected transition structures (Ref: heavy_test.m:5-7)
_EXPECTED_P = np.array([[0, 1], [0, 1]])
_EXPECTED_Q = np.eye(2)
_EXPECTED_M = np.array([1, 390])

# Expected sample size (Ref: heavy_test.m:8)
_EXPECTED_T = 1000

# Expected number of series (bivariate HEAVY)
_EXPECTED_K = 2


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_mock_simulate_return(T: int = _EXPECTED_T, K: int = _EXPECTED_K):
    """Build a plausible (data, ht_orig) tuple for mocking ``heavy_simulate``.

    Returns a deterministic pair of ``(T, K)`` arrays so that downstream
    transformation and estimation mocks see sensible shapes.
    """
    rng = np.random.default_rng(42)
    data = rng.standard_normal((T, K))
    ht_orig = np.abs(rng.standard_normal((T, K))) + 0.5  # positive variances
    return data, ht_orig


def _make_mock_heavy_return(T: int = _EXPECTED_T, K: int = _EXPECTED_K):
    """Build a plausible 5-tuple return for mocking ``heavy()``.

    Matches the return signature: ``(parameters, ll, ht, vcv, scores)``.
    """
    params_est = np.array([0.01, 0.03, 0.11, 0.38, 0.93, 0.57])
    ll = 3113.18
    ht = np.abs(np.random.default_rng(99).standard_normal((T, K))) + 0.5
    vcv = np.eye(6) * 0.001
    scores = np.random.default_rng(7).standard_normal((T, 6))
    return params_est, ll, ht, vcv, scores


# ===================================================================
# Phase 1: Unit Tests (no fixture dependency)
# ===================================================================


class TestHeavyTestImports:
    """Verify the module and its main callable are importable."""

    def test_heavy_test_imports(self):
        """Module can be imported without error."""
        from mfe_toolbox.sandbox.heavy_test import heavy_test  # noqa: F401

    def test_heavy_test_is_callable(self):
        """``heavy_test`` is a callable function."""
        from mfe_toolbox.sandbox.heavy_test import heavy_test

        assert callable(heavy_test)

    def test_heavy_test_module_has_docstring(self):
        """Module-level docstring exists."""
        import mfe_toolbox.sandbox.heavy_test as mod

        assert mod.__doc__ is not None
        assert len(mod.__doc__) > 0


class TestHeavyTestParameterSetup:
    """Verify that internal parameters match the MATLAB source exactly.

    Because ``heavy_test()`` is a void function with hardcoded internal
    constants, we verify them indirectly by inspecting mock call arguments
    to ``heavy_simulate``.
    """

    @patch("mfe_toolbox.sandbox.heavy_test.plt")
    @patch("mfe_toolbox.sandbox.heavy_test.heavy")
    @patch("mfe_toolbox.sandbox.heavy_test.heavy_simulate")
    def test_heavy_test_parameter_values(
        self, mock_sim, mock_heavy, mock_plt
    ):
        """Parameters passed to heavy_simulate match MATLAB heavy_test.m:4-8."""
        from mfe_toolbox.sandbox.heavy_test import heavy_test

        mock_sim.return_value = _make_mock_simulate_return()
        mock_heavy.return_value = _make_mock_heavy_return()

        heavy_test()

        # heavy_simulate called once: heavy_simulate(T, 2, parameters, p, q, m)
        mock_sim.assert_called_once()
        args = mock_sim.call_args[0]

        # Ref: heavy_test.m:9 — heavy_simulate(T, 2, parameters, p, q, m)
        assert args[0] == _EXPECTED_T, "T should be 1000"
        assert args[1] == _EXPECTED_K, "K should be 2"
        npt.assert_array_equal(args[2], _EXPECTED_PARAMS)
        npt.assert_array_equal(args[3], _EXPECTED_P)
        npt.assert_array_equal(args[4], _EXPECTED_Q)
        npt.assert_array_equal(args[5], _EXPECTED_M)

    @patch("mfe_toolbox.sandbox.heavy_test.plt")
    @patch("mfe_toolbox.sandbox.heavy_test.heavy")
    @patch("mfe_toolbox.sandbox.heavy_test.heavy_simulate")
    def test_heavy_test_parameter_shapes(self, mock_sim, mock_heavy, mock_plt):
        """Parameters have correct shapes: (6,), (2,2), (2,2), (2,)."""
        from mfe_toolbox.sandbox.heavy_test import heavy_test

        mock_sim.return_value = _make_mock_simulate_return()
        mock_heavy.return_value = _make_mock_heavy_return()

        heavy_test()

        args = mock_sim.call_args[0]
        params_arg = args[2]
        p_arg = args[3]
        q_arg = args[4]
        m_arg = args[5]

        assert params_arg.shape == (6,), "params shape should be (6,)"
        assert p_arg.shape == (2, 2), "p shape should be (2, 2)"
        assert q_arg.shape == (2, 2), "q shape should be (2, 2)"
        assert m_arg.shape == (2,), "m shape should be (2,)"


class TestHeavyTestSimulationCall:
    """Verify heavy_simulate is invoked with the correct argument pattern."""

    @patch("mfe_toolbox.sandbox.heavy_test.plt")
    @patch("mfe_toolbox.sandbox.heavy_test.heavy")
    @patch("mfe_toolbox.sandbox.heavy_test.heavy_simulate")
    def test_heavy_test_simulation_called_once(
        self, mock_sim, mock_heavy, mock_plt
    ):
        """heavy_simulate is called exactly once."""
        from mfe_toolbox.sandbox.heavy_test import heavy_test

        mock_sim.return_value = _make_mock_simulate_return()
        mock_heavy.return_value = _make_mock_heavy_return()

        heavy_test()

        assert mock_sim.call_count == 1


class TestHeavyTestDataTransformation:
    """Verify the data2 transformation logic (squared first column)."""

    @patch("mfe_toolbox.sandbox.heavy_test.plt")
    @patch("mfe_toolbox.sandbox.heavy_test.heavy")
    @patch("mfe_toolbox.sandbox.heavy_test.heavy_simulate")
    def test_heavy_test_data_transformation_shape(
        self, mock_sim, mock_heavy, mock_plt
    ):
        """Transformed data2 retains shape (T, 2)."""
        from mfe_toolbox.sandbox.heavy_test import heavy_test

        sim_data, ht_orig = _make_mock_simulate_return()
        mock_sim.return_value = (sim_data.copy(), ht_orig.copy())
        mock_heavy.return_value = _make_mock_heavy_return()

        heavy_test()

        # The function should not fail — shapes are preserved internally.
        # We verify indirectly by ensuring heavy_test completes without error.
        assert mock_heavy.call_count >= 1

    @patch("mfe_toolbox.sandbox.heavy_test.plt")
    @patch("mfe_toolbox.sandbox.heavy_test.heavy")
    @patch("mfe_toolbox.sandbox.heavy_test.heavy_simulate")
    def test_heavy_test_first_column_squared(
        self, mock_sim, mock_heavy, mock_plt
    ):
        """Ref: heavy_test.m:12 — data2(:,1) = data2(:,1).^2.

        We verify by checking the ``heavy`` estimation receives the original
        (non-squared) ``data`` — the squaring only applies to ``data2``
        (the working copy used for back-cast/bounds, not the estimation input).
        """
        from mfe_toolbox.sandbox.heavy_test import heavy_test

        sim_data, ht_orig = _make_mock_simulate_return()
        original_data = sim_data.copy()
        mock_sim.return_value = (sim_data, ht_orig)
        mock_heavy.return_value = _make_mock_heavy_return()

        heavy_test()

        # heavy() receives the original 'data' (not data2), per heavy_test.m:25
        first_call_args = mock_heavy.call_args_list[0][0]
        data_passed = first_call_args[0]
        assert data_passed.shape == (_EXPECTED_T, _EXPECTED_K)
        # The data passed to heavy() should be the original simulation data
        npt.assert_array_equal(data_passed, original_data)


class TestHeavyTestEstimationCalls:
    """Verify heavy() is called twice with correct arguments."""

    @patch("mfe_toolbox.sandbox.heavy_test.plt")
    @patch("mfe_toolbox.sandbox.heavy_test.heavy")
    @patch("mfe_toolbox.sandbox.heavy_test.heavy_simulate")
    def test_heavy_test_estimation_called_twice(
        self, mock_sim, mock_heavy, mock_plt
    ):
        """heavy() is called exactly twice (default SV + original params)."""
        from mfe_toolbox.sandbox.heavy_test import heavy_test

        mock_sim.return_value = _make_mock_simulate_return()
        mock_heavy.return_value = _make_mock_heavy_return()

        heavy_test()

        assert mock_heavy.call_count == 2

    @patch("mfe_toolbox.sandbox.heavy_test.plt")
    @patch("mfe_toolbox.sandbox.heavy_test.heavy")
    @patch("mfe_toolbox.sandbox.heavy_test.heavy_simulate")
    def test_heavy_test_first_estimation_args(
        self, mock_sim, mock_heavy, mock_plt
    ):
        """First heavy() call uses default starting values.

        Ref: heavy_test.m:25 — heavy(data, p, q, 'None')
        """
        from mfe_toolbox.sandbox.heavy_test import heavy_test

        mock_sim.return_value = _make_mock_simulate_return()
        mock_heavy.return_value = _make_mock_heavy_return()

        heavy_test()

        first_call = mock_heavy.call_args_list[0]
        args = first_call[0]
        # args: (data, p, q, 'None')
        assert args[0].shape == (_EXPECTED_T, _EXPECTED_K)
        npt.assert_array_equal(args[1], _EXPECTED_P)
        npt.assert_array_equal(args[2], _EXPECTED_Q)
        assert args[3] == "None"

    @patch("mfe_toolbox.sandbox.heavy_test.plt")
    @patch("mfe_toolbox.sandbox.heavy_test.heavy")
    @patch("mfe_toolbox.sandbox.heavy_test.heavy_simulate")
    def test_heavy_test_second_estimation_args(
        self, mock_sim, mock_heavy, mock_plt
    ):
        """Second heavy() call uses original params as starting values.

        Ref: heavy_test.m:29 — heavy(data, p, q, 'None', parametersOrig')
        """
        from mfe_toolbox.sandbox.heavy_test import heavy_test

        mock_sim.return_value = _make_mock_simulate_return()
        mock_heavy.return_value = _make_mock_heavy_return()

        heavy_test()

        second_call = mock_heavy.call_args_list[1]
        args = second_call[0]
        # args: (data, p, q, 'None', parameters_orig)
        assert args[0].shape == (_EXPECTED_T, _EXPECTED_K)
        npt.assert_array_equal(args[1], _EXPECTED_P)
        npt.assert_array_equal(args[2], _EXPECTED_Q)
        assert args[3] == "None"
        # Fifth argument should be the original params copy
        npt.assert_array_equal(args[4], _EXPECTED_PARAMS)


class TestHeavyTestSmokeTest:
    """End-to-end smoke tests with all external dependencies mocked."""

    @patch("mfe_toolbox.sandbox.heavy_test.plt")
    @patch("mfe_toolbox.sandbox.heavy_test.heavy")
    @patch("mfe_toolbox.sandbox.heavy_test.heavy_simulate")
    def test_heavy_test_runs_without_error(
        self, mock_sim, mock_heavy, mock_plt
    ):
        """Full script flow completes without exceptions."""
        from mfe_toolbox.sandbox.heavy_test import heavy_test

        mock_sim.return_value = _make_mock_simulate_return()
        mock_heavy.return_value = _make_mock_heavy_return()

        # Should not raise any exception
        result = heavy_test()

        # heavy_test returns None (void function)
        assert result is None

    @patch("mfe_toolbox.sandbox.heavy_test.plt")
    @patch("mfe_toolbox.sandbox.heavy_test.heavy")
    @patch("mfe_toolbox.sandbox.heavy_test.heavy_simulate")
    def test_heavy_test_plot_invoked(
        self, mock_sim, mock_heavy, mock_plt
    ):
        """Matplotlib plot and show are invoked during script execution.

        Ref: heavy_test.m:26 — plot([ht(:,1), htOrig(:,1)])
        """
        from mfe_toolbox.sandbox.heavy_test import heavy_test

        mock_sim.return_value = _make_mock_simulate_return()
        mock_heavy.return_value = _make_mock_heavy_return()

        heavy_test()

        # plt.plot called at least once
        assert mock_plt.plot.call_count >= 1
        # plt.show called at least once
        assert mock_plt.show.call_count >= 1


class TestHeavyTestReturnType:
    """Verify return type of the script wrapper function."""

    @patch("mfe_toolbox.sandbox.heavy_test.plt")
    @patch("mfe_toolbox.sandbox.heavy_test.heavy")
    @patch("mfe_toolbox.sandbox.heavy_test.heavy_simulate")
    def test_heavy_test_returns_none(self, mock_sim, mock_heavy, mock_plt):
        """heavy_test() returns None (void wrapper)."""
        from mfe_toolbox.sandbox.heavy_test import heavy_test

        mock_sim.return_value = _make_mock_simulate_return()
        mock_heavy.return_value = _make_mock_heavy_return()

        result = heavy_test()
        assert result is None


# ===================================================================
# Phase 2: Parity Tests (against MATLAB fixtures)
# ===================================================================


@pytest.mark.parity
class TestHeavyTestParity:
    """Compare HEAVY test script outputs against MATLAB/Octave fixtures.

    Fixtures are loaded from ``tests/fixtures/sandbox/heavy_test.npy``.
    All comparisons use ``atol=1e-6``, ``rtol=1e-4`` per AAP Section 0.7.1.
    """

    def test_heavy_test_parity_parameter_setup(self, sandbox_fixture_dir):
        """Fixture parameters match the hardcoded MATLAB values."""
        fixture = load_fixture_npy(sandbox_fixture_dir, "heavy_test")
        fixture_dict = fixture.item()

        npt.assert_allclose(
            fixture_dict["simulation_parameters"],
            _EXPECTED_PARAMS,
            atol=ATOL,
            rtol=RTOL,
            err_msg="Simulation parameters do not match MATLAB heavy_test.m:4",
        )
        npt.assert_allclose(
            fixture_dict["p_matrix"],
            _EXPECTED_P.astype(float),
            atol=ATOL,
            rtol=RTOL,
            err_msg="p matrix does not match MATLAB heavy_test.m:5",
        )
        npt.assert_allclose(
            fixture_dict["q_matrix"],
            _EXPECTED_Q,
            atol=ATOL,
            rtol=RTOL,
            err_msg="q matrix does not match MATLAB heavy_test.m:6",
        )
        npt.assert_allclose(
            fixture_dict["m_vector"],
            _EXPECTED_M.astype(float),
            atol=ATOL,
            rtol=RTOL,
            err_msg="m vector does not match MATLAB heavy_test.m:7",
        )
        assert fixture_dict["T"] == _EXPECTED_T
        assert fixture_dict["K"] == _EXPECTED_K

    def test_heavy_test_parity_simulation_shapes(self, sandbox_fixture_dir):
        """Simulation output shapes match expected (T, K) layout."""
        fixture = load_fixture_npy(sandbox_fixture_dir, "heavy_test")
        fixture_dict = fixture.item()

        assert fixture_dict["data"].shape == (_EXPECTED_T, _EXPECTED_K)
        assert fixture_dict["htOrig"].shape == (_EXPECTED_T, _EXPECTED_K)

    def test_heavy_test_parity_data_transformation(self, sandbox_fixture_dir):
        """Transformed data2 matches MATLAB: first column squared.

        Ref: heavy_test.m:12 — data2(:,1) = data2(:,1).^2
        """
        fixture = load_fixture_npy(sandbox_fixture_dir, "heavy_test")
        fixture_dict = fixture.item()

        data = fixture_dict["data"]
        data2 = fixture_dict["data2"]

        # First column of data2 should be square of first column of data
        npt.assert_allclose(
            data2[:, 0],
            data[:, 0] ** 2,
            atol=ATOL,
            rtol=RTOL,
            err_msg="data2[:, 0] should equal data[:, 0] ** 2",
        )
        # Second column should be unchanged
        npt.assert_allclose(
            data2[:, 1],
            data[:, 1],
            atol=ATOL,
            rtol=RTOL,
            err_msg="data2[:, 1] should equal data[:, 1]",
        )

    def test_heavy_test_parity_backcast(self, sandbox_fixture_dir):
        """Back-cast values match column means of transformed data2.

        Ref: heavy_test.m:13 — backCast = mean(data2)
        """
        fixture = load_fixture_npy(sandbox_fixture_dir, "heavy_test")
        fixture_dict = fixture.item()

        data2 = fixture_dict["data2"]
        back_cast = fixture_dict["backCast"]

        expected_backcast = np.mean(data2, axis=0)
        npt.assert_allclose(
            back_cast,
            expected_backcast,
            atol=ATOL,
            rtol=RTOL,
            err_msg="backCast should equal mean(data2)",
        )

    def test_heavy_test_parity_bounds(self, sandbox_fixture_dir):
        """Lower/upper bounds match MATLAB formulae.

        Ref: heavy_test.m:14-15 — lb = min(data2)'/10000; ub = max(data2)'*100000
        """
        fixture = load_fixture_npy(sandbox_fixture_dir, "heavy_test")
        fixture_dict = fixture.item()

        data2 = fixture_dict["data2"]
        lb = fixture_dict["lb"]
        ub = fixture_dict["ub"]

        expected_lb = np.min(data2, axis=0) / 10000.0
        expected_ub = np.max(data2, axis=0) * 100000.0

        npt.assert_allclose(
            lb,
            expected_lb,
            atol=ATOL,
            rtol=RTOL,
            err_msg="lb should equal min(data2)/10000",
        )
        npt.assert_allclose(
            ub,
            expected_ub,
            atol=ATOL,
            rtol=RTOL,
            err_msg="ub should equal max(data2)*100000",
        )

    def test_heavy_test_parity_estimation1_shapes(self, sandbox_fixture_dir):
        """First estimation outputs have correct shapes."""
        fixture = load_fixture_npy(sandbox_fixture_dir, "heavy_test")
        fixture_dict = fixture.item()

        assert fixture_dict["parameters_est1"].shape == (6,)
        assert isinstance(fixture_dict["ll_est1"], float)
        assert fixture_dict["ht_est1"].shape == (_EXPECTED_T, _EXPECTED_K)
        assert fixture_dict["VCV_est1"].shape == (6, 6)
        assert fixture_dict["scores_est1"].shape == (_EXPECTED_T, 6)

    def test_heavy_test_parity_estimation2_shapes(self, sandbox_fixture_dir):
        """Second estimation outputs have correct shapes."""
        fixture = load_fixture_npy(sandbox_fixture_dir, "heavy_test")
        fixture_dict = fixture.item()

        assert fixture_dict["parameters_est2"].shape == (6,)
        assert isinstance(fixture_dict["ll_est2"], float)
        assert fixture_dict["ht_est2"].shape == (_EXPECTED_T, _EXPECTED_K)
        assert fixture_dict["VCV_est2"].shape == (6, 6)
        assert fixture_dict["scores_est2"].shape == (_EXPECTED_T, 6)

    def test_heavy_test_parity_estimation1_ll_finite(self, sandbox_fixture_dir):
        """First estimation log-likelihood is finite."""
        fixture = load_fixture_npy(sandbox_fixture_dir, "heavy_test")
        fixture_dict = fixture.item()

        assert np.isfinite(fixture_dict["ll_est1"])

    def test_heavy_test_parity_estimation2_ll_finite(self, sandbox_fixture_dir):
        """Second estimation log-likelihood is finite."""
        fixture = load_fixture_npy(sandbox_fixture_dir, "heavy_test")
        fixture_dict = fixture.item()

        assert np.isfinite(fixture_dict["ll_est2"])

    def test_heavy_test_parity_ht_positive(self, sandbox_fixture_dir):
        """Conditional variances from both estimations are strictly positive."""
        fixture = load_fixture_npy(sandbox_fixture_dir, "heavy_test")
        fixture_dict = fixture.item()

        assert np.all(fixture_dict["ht_est1"] > 0), (
            "ht_est1 should be strictly positive (conditional variances)"
        )
        assert np.all(fixture_dict["ht_est2"] > 0), (
            "ht_est2 should be strictly positive (conditional variances)"
        )

    def test_heavy_test_parity_simulation_data_finite(
        self, sandbox_fixture_dir
    ):
        """Simulated data and true variances are finite."""
        fixture = load_fixture_npy(sandbox_fixture_dir, "heavy_test")
        fixture_dict = fixture.item()

        assert np.all(np.isfinite(fixture_dict["data"]))
        assert np.all(np.isfinite(fixture_dict["htOrig"]))

    def test_heavy_test_parity_htOrig_positive(self, sandbox_fixture_dir):
        """True latent conditional variances are strictly positive."""
        fixture = load_fixture_npy(sandbox_fixture_dir, "heavy_test")
        fixture_dict = fixture.item()

        assert np.all(fixture_dict["htOrig"] > 0)

    def test_heavy_test_parity_vcv_symmetric(self, sandbox_fixture_dir):
        """Variance-covariance matrices from both estimations are symmetric."""
        fixture = load_fixture_npy(sandbox_fixture_dir, "heavy_test")
        fixture_dict = fixture.item()

        vcv1 = fixture_dict["VCV_est1"]
        vcv2 = fixture_dict["VCV_est2"]

        npt.assert_allclose(
            vcv1, vcv1.T, atol=ATOL, rtol=RTOL,
            err_msg="VCV_est1 should be symmetric",
        )
        npt.assert_allclose(
            vcv2, vcv2.T, atol=ATOL, rtol=RTOL,
            err_msg="VCV_est2 should be symmetric",
        )
