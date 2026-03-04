"""Pytest tests for the SARIMAX likelihood computation module.

Tests sarimax_likelihood from ``mfe_toolbox.sandbox``.
"""

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.sandbox.sarimax_likelihood import sarimax_likelihood


class TestSarimaxLikelihoodUnit:
    """Unit tests for sarimax_likelihood()."""

    def test_sarimax_likelihood_import(self) -> None:
        """Module should import without error."""
        assert callable(sarimax_likelihood)

    def test_sarimax_likelihood_basic_execution(self) -> None:
        """Should compute negative log-likelihood for simple AR(1) model."""
        rng = np.random.default_rng(42)
        T = 100
        y = rng.standard_normal(T)
        # AR(1) model: parameters = [constant_coeff, ar1_coeff]
        parameters = np.array([0.0, 0.5])
        p = np.array([1])       # AR lag index 1
        q = np.array([], dtype=int)  # No MA
        constant = 1            # Include constant
        seasonal = np.empty((0, 4))  # No seasonal component
        x = np.ones((T, 1))    # Constant column
        sigma = np.ones(T)     # Unit conditional std
        result = sarimax_likelihood(parameters, p, q, constant, seasonal, y, x, sigma)
        assert result is not None
        # Returns tuple: (LLF, likelihoods, errors)
        assert isinstance(result, tuple)
        assert len(result) == 3
        llf, likelihoods, errors = result
        assert np.isfinite(llf), f"LLF should be finite, got {llf}"

    def test_sarimax_likelihood_returns_scalar_or_tuple(self) -> None:
        """Result should be a 3-tuple (float, ndarray, ndarray)."""
        rng = np.random.default_rng(42)
        T = 50
        y = rng.standard_normal(T)
        parameters = np.array([0.1, 0.3])
        p = np.array([1])
        q = np.array([], dtype=int)
        constant = 1
        seasonal = np.empty((0, 4))
        x = np.ones((T, 1))
        sigma = np.ones(T)
        result = sarimax_likelihood(parameters, p, q, constant, seasonal, y, x, sigma)
        assert isinstance(result, tuple)
        llf, likelihoods, errors = result
        assert isinstance(llf, (float, np.floating))
        assert isinstance(likelihoods, np.ndarray)
        assert isinstance(errors, np.ndarray)
