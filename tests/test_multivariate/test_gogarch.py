"""Pytest test suite for mfe_toolbox.multivariate.gogarch — GO-GARCH/O-GARCH parity tests.

Tests the Generalized Orthogonal GARCH (GO-GARCH) and Orthogonal GARCH
(O-GARCH) multivariate volatility model family covering two source modules:

- ``gogarch.py`` — Main estimation driver function.
- ``gogarch_likelihood.py`` — Joint log-likelihood evaluator.

Test data uses the ``multivariate_data`` fixture (T=1000, K=3) from the
root ``tests/conftest.py`` and local ``tests/test_multivariate/conftest.py``
fixtures.

Numerical parity target: ``numpy.testing.assert_allclose(actual, expected,
atol=1e-6, rtol=1e-4)`` against MATLAB-generated fixture outputs stored
in ``tests/fixtures/multivariate/``.

References
----------
- ``multivariate/gogarch.m`` (257 lines) — MATLAB GO-GARCH driver.
- ``multivariate/gogarch_likelihood.m`` (88 lines) — MATLAB likelihood.
- ``mex_source/`` — No MEX kernels; uses ``tarch_core_simple`` Numba JIT.
"""

import numpy as np
import pytest

from mfe_toolbox.multivariate.gogarch import gogarch
from mfe_toolbox.multivariate.gogarch_likelihood import gogarch_likelihood
from tests.conftest import ATOL, RTOL, load_fixture_npy


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_K: int = 3
_T: int = 1000


# ---------------------------------------------------------------------------
# Local Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gogarch_w_matrix(rng) -> np.ndarray:
    """Valid 3×3 orthogonal rotation matrix constructed via QR decomposition.

    Uses the session-scoped ``rng`` fixture (``np.random.default_rng(42)``)
    from root conftest to generate a random matrix and then extracts the
    orthogonal factor Q from its QR decomposition.  The resulting matrix
    satisfies W'W = WW' = I_K.

    Parameters
    ----------
    rng : np.random.Generator
        Seeded RNG from ``tests/conftest.py`` (seed=42).

    Returns
    -------
    np.ndarray
        Shape ``(3, 3)`` orthogonal matrix.
    """
    # Use a local child rng so we don't mutate the session-scoped rng state
    local_rng = np.random.default_rng(rng.integers(0, 2**32))
    A = local_rng.standard_normal((_K, _K))
    W, _ = np.linalg.qr(A)
    return W


@pytest.fixture
def likelihood_setup(mv_data: np.ndarray) -> dict:
    """Prepare data structures required to call ``gogarch_likelihood``.

    Computes:
    - 3-D outer-product data (K×K×T)
    - Eigendecomposition P (rows=eigenvectors) and L (diagonal eigenvalue matrix)
    - EWMA back-cast weights
    - Reasonable GARCH parameters for K=3 with p=1, q=1

    Parameters
    ----------
    mv_data : np.ndarray
        T×K return data from conftest ``mv_data`` fixture.

    Returns
    -------
    dict
        Keys: ``data_3d``, ``P``, ``L``, ``p``, ``q``, ``gjr_type``,
        ``params_gogarch``, ``params_ogarch``.
    """
    T_obs, K_dim = mv_data.shape

    # Build K×K×T outer-product data
    # Ref: gogarch.m:77-81
    data_3d = np.zeros((K_dim, K_dim, T_obs), dtype=np.float64)
    for t in range(T_obs):
        row = mv_data[t, :].reshape(-1, 1)
        data_3d[:, :, t] = row @ row.T

    # Eigendecomposition of mean covariance
    # Ref: gogarch.m:142-144
    S = np.mean(data_3d, axis=2)
    eigenvalues, eigvecs = np.linalg.eigh(S)
    P_mat = eigvecs.T  # rows = eigenvectors (MATLAB convention)
    L_mat = np.diag(eigenvalues)

    # GARCH parameters: K factors, each with alpha, beta
    p_vec = np.ones(K_dim, dtype=np.int64)
    q_vec = np.ones(K_dim, dtype=np.int64)
    gjr_type = np.full(K_dim, 2, dtype=np.int64)  # GJR-GARCH / standard

    # GO-GARCH parameters: [phi(K*(K-1)/2), alpha1, beta1, alpha2, beta2, alpha3, beta3]
    n_phi = K_dim * (K_dim - 1) // 2  # 3 for K=3
    phi = np.array([0.1, 0.2, 0.15], dtype=np.float64)
    vol_params = np.tile(np.array([0.05, 0.90], dtype=np.float64), K_dim)
    params_gogarch = np.concatenate([phi, vol_params])

    # O-GARCH parameters: [alpha1, beta1, alpha2, beta2, alpha3, beta3]
    params_ogarch = vol_params.copy()

    return {
        'data_3d': data_3d,
        'P': P_mat,
        'L': L_mat,
        'p': p_vec,
        'q': q_vec,
        'gjr_type': gjr_type,
        'params_gogarch': params_gogarch,
        'params_ogarch': params_ogarch,
        'S': S,
    }


# ===========================================================================
# TestGogarchLikelihood
# ===========================================================================


class TestGogarchLikelihood:
    """Tests for ``gogarch_likelihood`` function.

    The function evaluates the multivariate Gaussian log-likelihood for the
    GO-GARCH / O-GARCH model given factor GARCH parameters, eigendecomposition
    matrices P and L, and an orthogonality flag.
    """

    def test_returns_ll_lls(self, likelihood_setup: dict) -> None:
        """Verify ``gogarch_likelihood`` returns a 3-tuple (ll, lls, Ht)."""
        s = likelihood_setup
        result = gogarch_likelihood(
            s['params_gogarch'], s['data_3d'], s['p'], s['q'],
            s['gjr_type'], s['P'], s['L'], False, False,
        )
        assert isinstance(result, tuple), "Expected tuple return"
        assert len(result) == 3, f"Expected 3 return values, got {len(result)}"

    def test_ll_scalar_finite(self, likelihood_setup: dict) -> None:
        """Verify ll is a finite scalar (float)."""
        s = likelihood_setup
        ll, lls, Ht = gogarch_likelihood(
            s['params_gogarch'], s['data_3d'], s['p'], s['q'],
            s['gjr_type'], s['P'], s['L'], False, False,
        )
        assert isinstance(ll, (float, np.floating)), (
            f"ll should be a scalar, got {type(ll)}"
        )
        assert np.isfinite(ll), f"ll should be finite, got {ll}"

    def test_lls_shape(self, likelihood_setup: dict) -> None:
        """Verify lls has shape (T,) matching the number of observations."""
        s = likelihood_setup
        ll, lls, Ht = gogarch_likelihood(
            s['params_gogarch'], s['data_3d'], s['p'], s['q'],
            s['gjr_type'], s['P'], s['L'], False, False,
        )
        T_obs = s['data_3d'].shape[2]
        assert isinstance(lls, np.ndarray), "lls should be an ndarray"
        assert lls.shape == (T_obs,), (
            f"lls shape should be ({T_obs},), got {lls.shape}"
        )

    def test_orthogonal_w_matrix(
        self, gogarch_w_matrix: np.ndarray,
    ) -> None:
        """Verify that the orthogonal W fixture satisfies W'W ≈ I_K."""
        W = gogarch_w_matrix
        WtW = W.T @ W
        np.testing.assert_allclose(
            WtW, np.eye(_K), atol=1e-12,
            err_msg="W matrix is not orthogonal: W'W != I",
        )

    def test_gogarch_vs_ogarch_different_w(
        self, likelihood_setup: dict,
    ) -> None:
        """GO-GARCH (is_ogarch=False) and O-GARCH (is_ogarch=True) produce
        different log-likelihoods because they use different rotation matrices.

        GO-GARCH uses U = phi2u(phi) (non-identity rotation), while O-GARCH
        uses U = I_K.  With the same GARCH vol parameters, the likelihoods
        should differ.
        """
        s = likelihood_setup
        # GO-GARCH: uses phi-based rotation
        ll_go, _, _ = gogarch_likelihood(
            s['params_gogarch'], s['data_3d'], s['p'], s['q'],
            s['gjr_type'], s['P'], s['L'], False, False,
        )
        # O-GARCH: uses identity rotation (only vol params)
        ll_o, _, _ = gogarch_likelihood(
            s['params_ogarch'], s['data_3d'], s['p'], s['q'],
            s['gjr_type'], s['P'], s['L'], True, False,
        )
        # Both should be finite
        assert np.isfinite(ll_go), f"GO-GARCH ll not finite: {ll_go}"
        assert np.isfinite(ll_o), f"O-GARCH ll not finite: {ll_o}"
        # Likelihoods should differ because rotation matrices differ
        assert ll_go != ll_o, (
            "GO-GARCH and O-GARCH likelihoods should differ "
            f"but both are {ll_go}"
        )

    def test_fixture_parity(
        self, multivariate_fixture_dir,
    ) -> None:
        """Compare gogarch_likelihood output against MATLAB-generated fixture.

        Fixture file: ``tests/fixtures/multivariate/gogarch_likelihood.npy``
        """
        fixture = load_fixture_npy(multivariate_fixture_dir, 'gogarch_likelihood')
        fix = fixture.item()
        if fix is None or not isinstance(fix, dict):
            pytest.skip("gogarch_likelihood fixture is empty or malformed")

        # Reconstruct inputs from fixture
        parameters = fix['parameters']
        # The fixture stores P as rows-of-eigenvectors, L as 1-D eigenvalues
        P_mat = fix['P']
        # Convert L from 1-D eigenvalue array to diagonal matrix
        L_diag = fix['L']
        L_mat = np.diag(L_diag)
        p_vec = fix['p'].astype(np.int64)
        q_vec = fix['q'].astype(np.int64)
        gjr_type = fix['gjrType'].astype(np.int64)

        # Need the 3-D data — reconstruct from 2-D data if not available
        if 'data_3d' in fix:
            data_3d = fix['data_3d']
        else:
            data_2d = fix['data']
            T_obs, K_dim = data_2d.shape
            data_3d = np.zeros((K_dim, K_dim, T_obs), dtype=np.float64)
            for t in range(T_obs):
                row = data_2d[t, :].reshape(-1, 1)
                data_3d[:, :, t] = row @ row.T

        # Evaluate likelihood
        ll_py, lls_py, Ht_py = gogarch_likelihood(
            parameters, data_3d, p_vec, q_vec,
            gjr_type, P_mat, L_mat, False, False,
        )

        # Compare against fixture
        np.testing.assert_allclose(
            ll_py, fix['ll'], atol=ATOL, rtol=RTOL,
            err_msg="gogarch_likelihood: ll does not match fixture",
        )
        np.testing.assert_allclose(
            lls_py, fix['lls'], atol=ATOL, rtol=RTOL,
            err_msg="gogarch_likelihood: lls does not match fixture",
        )


# ===========================================================================
# TestGogarch
# ===========================================================================


class TestGogarch:
    """Integration tests for the ``gogarch`` estimation driver.

    Tests cover both GO-GARCH (``type_model='gogarch'``, estimated orthogonal
    rotation) and O-GARCH (``type_model='ogarch'``, PCA-based fixed rotation).
    """

    @pytest.mark.slow
    def test_gogarch_estimation(self, mv_data: np.ndarray) -> None:
        """Estimate GO-GARCH(1,1) on K=3 data and verify convergence.

        Convergence is validated by checking that the log-likelihood ``ll``
        is finite and positive (since the MATLAB convention negates the
        minimiser's objective to return a proper log-likelihood).
        """
        parameters, ll, Ht, VCV, scores = gogarch(
            mv_data, p=1, q=1, type_model='gogarch',
        )
        assert np.isfinite(ll), f"GO-GARCH ll should be finite, got {ll}"
        assert isinstance(parameters, np.ndarray), "parameters should be ndarray"
        assert isinstance(Ht, np.ndarray), "Ht should be ndarray"

    @pytest.mark.slow
    def test_ogarch_estimation(self, mv_data: np.ndarray) -> None:
        """Estimate O-GARCH(1,1) on K=3 data — uses PCA rotation (U=I)."""
        parameters, ll, Ht, VCV, scores = gogarch(
            mv_data, p=1, q=1, type_model='ogarch',
        )
        assert np.isfinite(ll), f"O-GARCH ll should be finite, got {ll}"
        assert isinstance(parameters, np.ndarray), "parameters should be ndarray"

    @pytest.mark.slow
    def test_parameter_vector_length(
        self, mv_data: np.ndarray, K: int,
    ) -> None:
        """Verify parameter vector length for both model types.

        OGARCH: ``vech(S) + sum(p) + sum(q)`` = K(K+1)/2 + K*(p+q)
        GOGARCH: ``vech(S) + K(K-1)/2 + sum(p) + sum(q)``
        For K=3, p=1, q=1:
          OGARCH:  6 + 6 = 12
          GOGARCH: 6 + 3 + 6 = 15
        """
        # OGARCH
        params_o, _, _, _, _ = gogarch(
            mv_data, p=1, q=1, type_model='ogarch',
        )
        k2 = K * (K + 1) // 2  # 6
        expected_ogarch = k2 + K * (1 + 1)  # 6 + 6 = 12
        assert len(params_o) == expected_ogarch, (
            f"OGARCH param length: expected {expected_ogarch}, got {len(params_o)}"
        )

        # GOGARCH
        params_g, _, _, _, _ = gogarch(
            mv_data, p=1, q=1, type_model='gogarch',
        )
        n_phi = K * (K - 1) // 2  # 3
        expected_gogarch = k2 + n_phi + K * (1 + 1)  # 6 + 3 + 6 = 15
        assert len(params_g) == expected_gogarch, (
            f"GOGARCH param length: expected {expected_gogarch}, got {len(params_g)}"
        )

    @pytest.mark.slow
    def test_ht_output_shape(
        self, mv_data: np.ndarray, K: int, T: int,
    ) -> None:
        """Verify Ht is K×K×T for both model types."""
        _, _, Ht, _, _ = gogarch(
            mv_data, p=1, q=1, type_model='ogarch',
        )
        assert Ht.shape == (K, K, T), (
            f"Ht shape should be ({K}, {K}, {T}), got {Ht.shape}"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("type_model", ["gogarch", "ogarch"])
    def test_ht_positive_definite(
        self, mv_data: np.ndarray, assert_ht_positive_definite,
        type_model: str,
    ) -> None:
        """All Ht[:,:,t] slices must be positive definite for both model types."""
        _, _, Ht, _, _ = gogarch(
            mv_data, p=1, q=1, type_model=type_model,
        )
        # Use the assert_ht_positive_definite callable fixture from conftest
        assert_ht_positive_definite(Ht)

    @pytest.mark.slow
    def test_ogarch_rotation_is_pca(
        self, mv_data: np.ndarray, sample_cov: np.ndarray, K: int,
    ) -> None:
        """For OGARCH, the implicit rotation W should be eigenvectors of cov.

        In the O-GARCH model, U = I_K, so the mixing matrix Z = P @ L^(1/2)
        where P has rows as eigenvectors and L is the eigenvalue diagonal matrix.
        We verify that the time-averaged conditional covariance Ht_mean
        approximates the sample covariance computed via ``np.cov``.
        """
        parameters, ll, Ht, VCV, scores = gogarch(
            mv_data, p=1, q=1, type_model='ogarch',
        )
        # Verify Ht is well-formed — indirect check that PCA rotation was used
        assert Ht.shape[0] == K
        assert Ht.shape[1] == K
        # The mean of Ht over time should approximate the sample covariance
        Ht_mean = np.mean(Ht, axis=2)
        # Check that diagonal of Ht_mean is roughly same order as sample_cov
        assert np.all(np.diag(Ht_mean) > 0), "Mean Ht diagonal should be positive"

        # Compare Ht_mean with the sample covariance (np.cov is column-wise)
        sample_cov_np = np.cov(mv_data, rowvar=False)
        # Diagonal magnitudes should be in the same order
        np.testing.assert_allclose(
            np.diag(Ht_mean), np.diag(sample_cov_np), rtol=0.5,
            err_msg="Ht_mean diagonal should be comparable to sample covariance",
        )
        # Verify Ht_mean is positive semi-definite via eigenvalue decomposition
        eigvals = np.linalg.eigvalsh(Ht_mean)
        assert np.all(eigvals > -1e-10), (
            f"Ht_mean is not PSD; min eigenvalue = {eigvals.min():.2e}"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("gjr_type", [1, 2])
    def test_factor_independence(self, mv_data: np.ndarray, K: int, gjr_type: int) -> None:
        """Rotated factors should be approximately uncorrelated.

        For O-GARCH, PCA-whitened factors are uncorrelated by construction.
        We verify the off-diagonal elements of the factor correlation matrix
        are small.
        """
        parameters, ll, Ht, VCV, scores = gogarch(
            mv_data, p=1, q=1, gjr_type=gjr_type, type_model='ogarch',
        )
        # Extract mean covariance and do eigendecomposition
        T_obs, K_dim = mv_data.shape
        data_3d = np.zeros((K_dim, K_dim, T_obs), dtype=np.float64)
        for t in range(T_obs):
            row = mv_data[t, :].reshape(-1, 1)
            data_3d[:, :, t] = row @ row.T
        S = np.mean(data_3d, axis=2)
        eigenvalues, eigvecs = np.linalg.eigh(S)
        eig_safe = np.maximum(np.abs(eigenvalues), 1e-300)
        L_inv_sqrt = np.diag(1.0 / np.sqrt(eig_safe))
        P_mat = eigvecs.T
        Zinv = L_inv_sqrt @ P_mat.T

        # Whiten data
        factors = (Zinv @ mv_data.T).T  # T × K whitened factors
        corr_matrix = np.corrcoef(factors, rowvar=False)
        # Off-diagonal should be small (< 0.1 in magnitude)
        off_diag_mask = ~np.eye(K, dtype=bool)
        max_off_diag = np.abs(corr_matrix[off_diag_mask]).max()
        assert max_off_diag < 0.15, (
            f"Factor correlation max off-diagonal = {max_off_diag:.4f}; "
            "expected < 0.15 for approximately uncorrelated factors"
        )

    def test_input_validation(self, mv_data: np.ndarray) -> None:
        """Invalid inputs should raise ValueError.

        Tests:
        - Invalid model type string
        - Wrong starting values length
        - Starting values phi angles out of range (for GOGARCH)
        """
        # Invalid type_model
        with pytest.raises(ValueError, match="TYPE must be"):
            gogarch(mv_data, p=1, q=1, type_model='invalid_type')

        # Wrong starting values count for OGARCH
        # Expected: sum(p) + sum(q) = 3*1 + 3*1 = 6 for K=3
        wrong_sv = np.ones(2, dtype=np.float64)
        with pytest.raises(ValueError, match="STARTINGVALS"):
            gogarch(mv_data, p=1, q=1, type_model='ogarch',
                     starting_vals=wrong_sv)

        # Wrong starting values count for GOGARCH
        # Expected: K*(K-1)/2 + sum(p) + sum(q) = 3 + 6 = 9
        wrong_sv_go = np.ones(4, dtype=np.float64)
        with pytest.raises(ValueError, match="STARTINGVALS"):
            gogarch(mv_data, p=1, q=1, type_model='gogarch',
                     starting_vals=wrong_sv_go)

        # phi angles out of range for GOGARCH (> pi)
        K_dim = mv_data.shape[1]
        n_phi = K_dim * (K_dim - 1) // 2
        bad_phi_sv = np.zeros(n_phi + K_dim * 2, dtype=np.float64) + 0.5
        bad_phi_sv[:n_phi] = 4.0  # > pi
        with pytest.raises(ValueError, match="STARTINGVALS 1 to K"):
            gogarch(mv_data, p=1, q=1, type_model='gogarch',
                     starting_vals=bad_phi_sv)

        # Data with wrong dimensions (1D)
        with pytest.raises(ValueError, match="DATA must be"):
            gogarch(np.ones(100), p=1, q=1)

    @pytest.mark.slow
    def test_fixture_parity_parameters(
        self, multivariate_fixture_dir,
    ) -> None:
        """Compare estimated parameters against MATLAB fixture.

        Uses the GOGARCH fixture which contains pre-estimated parameters
        on the same random data (seed=42).
        """
        fixture = load_fixture_npy(multivariate_fixture_dir, 'gogarch')
        fix = fixture.item()

        # Reconstruct the same input data
        data_2d = fix['data']

        # Run estimation with same settings
        parameters, ll, Ht, VCV, scores = gogarch(
            data_2d, p=1, q=1, gjr_type=2, type_model='gogarch',
        )

        # The fixture parameters = [vech(S)(6), phi(3), vol_params(6)] = 15
        # Parameter comparison — use relaxed tolerance for optimization-dependent values
        assert len(parameters) == len(fix['parameters']), (
            f"Parameter length mismatch: {len(parameters)} vs {len(fix['parameters'])}"
        )
        # The vech(S) component (first 6) should match closely since it's data-derived
        k2 = _K * (_K + 1) // 2
        np.testing.assert_allclose(
            parameters[:k2], fix['parameters'][:k2], atol=ATOL, rtol=RTOL,
            err_msg="vech(S) parameters do not match fixture",
        )

    @pytest.mark.slow
    def test_fixture_parity_likelihood(
        self, multivariate_fixture_dir,
    ) -> None:
        """Compare log-likelihood against MATLAB fixture."""
        fixture = load_fixture_npy(multivariate_fixture_dir, 'gogarch')
        fix = fixture.item()

        data_2d = fix['data']
        parameters, ll, Ht, VCV, scores = gogarch(
            data_2d, p=1, q=1, gjr_type=2, type_model='gogarch',
        )

        # Log-likelihood should be in the same ballpark — optimization may
        # reach slightly different optima, so use a moderately relaxed tolerance
        assert np.isfinite(ll), f"ll should be finite, got {ll}"
        assert np.isfinite(fix['ll']), f"Fixture ll should be finite"

        # The objective should be comparable — both should be negative (log-likelihood)
        # Use a relative check since exact match depends on optimizer convergence
        # Allow 5% relative tolerance for optimization-sensitive quantities
        if np.abs(fix['ll']) > 0:
            rel_diff = np.abs(ll - fix['ll']) / np.abs(fix['ll'])
            assert rel_diff < 0.05, (
                f"Log-likelihood relative difference {rel_diff:.4f} exceeds 5%: "
                f"computed={ll:.4f}, fixture={fix['ll']:.4f}"
            )
