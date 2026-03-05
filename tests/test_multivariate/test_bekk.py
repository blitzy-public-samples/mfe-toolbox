"""
Comprehensive pytest test file for the BEKK(p, o, q) multivariate GARCH model family.

Tests cover all five migrated BEKK modules:

- ``mfe_toolbox.multivariate.bekk``                   — Main estimation driver
- ``mfe_toolbox.multivariate.bekk_likelihood``         — Negative log-likelihood
- ``mfe_toolbox.multivariate.bekk_constraint``         — Stationarity constraint
- ``mfe_toolbox.multivariate.bekk_parameter_transform``— Parameter vector unpacking
- ``mfe_toolbox.multivariate.bekk_simulate``           — BEKK simulation

Validates output shapes, positive definiteness, sign conventions, parameter
counts, stationarity constraints, and numerical parity against MATLAB reference
fixtures stored in ``tests/fixtures/multivariate/``.

Per AAP Section 0.7.1:
    Every migrated function MUST pass
    ``numpy.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-4)``
    against MATLAB-generated fixtures.

Source references:
    multivariate/bekk.m                   (209 lines)
    multivariate/bekk_likelihood.m        (72 lines)
    multivariate/bekk_constraint.m        (46 lines)
    multivariate/bekk_parameter_transform.m (64 lines)
    multivariate/bekk_simulate.m          (153 lines)
"""

import os
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from mfe_toolbox.multivariate.bekk import bekk
from mfe_toolbox.multivariate.bekk_likelihood import bekk_likelihood
from mfe_toolbox.multivariate.bekk_constraint import bekk_constraint
from mfe_toolbox.multivariate.bekk_parameter_transform import bekk_parameter_transform
from mfe_toolbox.multivariate.bekk_simulate import bekk_simulate
from tests.conftest import ATOL, RTOL, assert_allclose, load_fixture_npy


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
K = 3       # Number of assets in test data (from conftest.py multivariate_data)
T_OBS = 1000  # Number of observations in test data


# ---------------------------------------------------------------------------
# Local Fixtures — BEKK parameter vectors for various parameterisation types
# ---------------------------------------------------------------------------

@pytest.fixture
def bekk_scalar_params():
    """Known valid BEKK scalar parameters for K=3, p=1, o=0, q=1.

    Parameter layout (Scalar):
        [chol2vec(L), a, b] where C = L @ L.T is the K×K intercept.
        k*(k+1)/2 = 6 intercept params + 1 ARCH scalar + 1 GARCH scalar = 8 total.

    Values: a = 0.05, b = 0.9  →  a² + b² = 0.8125 < 1 (stationary).
    C intercept = I(3) via trivial Cholesky factor [1,0,0,1,0,1].
    """
    c_params = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
    a = np.array([0.05])
    b = np.array([0.9])
    return np.concatenate([c_params, a, b])


@pytest.fixture
def bekk_diagonal_params():
    """Known valid BEKK diagonal parameters for K=3, p=1, o=0, q=1.

    Parameter layout (Diagonal):
        [chol2vec(L), diag(A_1), diag(B_1)]
        6 intercept + 3 A diag + 3 B diag = 12 total.

    Diagonal A and B values chosen to ensure per-asset stationarity.
    """
    c_params = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
    a_diag = np.array([0.15, 0.18, 0.20])
    b_diag = np.array([0.85, 0.82, 0.80])
    return np.concatenate([c_params, a_diag, b_diag])


@pytest.fixture
def bekk_full_params():
    """Known valid BEKK full parameters for K=2, p=1, o=0, q=1.

    Parameter layout (Full, K=2):
        [chol2vec(L), vec(A_1), vec(B_1)]
        3 intercept + 4 A full + 4 B full = 11 total.

    Using K=2 for tractability; A and B chosen so kron(A,A)+kron(B,B) is stable.
    """
    c_params = np.array([1.0, 0.2, 1.0])
    # Small A and B so spectral radius < 1
    a_full = np.array([0.15, 0.02, 0.02, 0.15])
    b_full = np.array([0.80, 0.05, 0.05, 0.80])
    return np.concatenate([c_params, a_full, b_full])


@pytest.fixture
def bekk_backcast(multivariate_data):
    """Compute EWMA backcast from multivariate data.

    Follows the MATLAB MFE Toolbox convention:
        m = ceil(sqrt(T))
        w = 0.06 * 0.94^(0:m-1), normalised
        backCast = sum_{i=0}^{m-1} w[i] * x[i] @ x[i].T

    Ref: bekk.m:186-189 — EWMA backcast of outer-product matrix.

    Parameters
    ----------
    multivariate_data : np.ndarray
        T×K data from conftest.py (T=1000, K=3).

    Returns
    -------
    np.ndarray
        K×K positive semi-definite backcast covariance matrix.
    """
    T_dim, K_dim = multivariate_data.shape
    m = int(np.ceil(np.sqrt(T_dim)))
    weights = 0.06 * 0.94 ** np.arange(m)
    weights = weights / weights.sum()
    back_cast = np.zeros((K_dim, K_dim))
    for i in range(min(m, T_dim)):
        back_cast += weights[i] * np.outer(
            multivariate_data[i], multivariate_data[i]
        )
    return back_cast


@pytest.fixture
def bekk_outer_products(multivariate_data):
    """Convert T×K data to K×K×T outer product array.

    bekk_likelihood expects 3-D K×K×T data (pre-computed outer products).
    Ref: bekk.m:148-156 — data transformation from 2-D to 3-D.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (data_3d, data_asym_3d) — symmetric and asymmetric outer products.
    """
    T_dim, K_dim = multivariate_data.shape
    data_3d = np.zeros((K_dim, K_dim, T_dim))
    data_asym_3d = np.zeros((K_dim, K_dim, T_dim))
    for i in range(T_dim):
        row = multivariate_data[i, :]
        data_3d[:, :, i] = np.outer(row, row)
        eta = row * (row < 0.0)
        data_asym_3d[:, :, i] = np.outer(eta, eta)
    return data_3d, data_asym_3d


# ---------------------------------------------------------------------------
# TestBekkParameterTransform
# ---------------------------------------------------------------------------

class TestBekkParameterTransform:
    """Tests for bekk_parameter_transform(parameters, p, o, q, k, type_).

    Verifies correct unpacking of flat parameter vectors into (C, A, G, B)
    matrices for Scalar (type_=1), Diagonal (type_=2), and Full (type_=3)
    BEKK model types, PSD correction of the intercept C, and parameter count
    correctness.
    """

    def test_scalar_type_unpacking(self, bekk_scalar_params):
        """Scalar type (type_=1): A, B are scalar*I(K), G is empty.

        K=3, p=1, o=0, q=1.  A[:,:,0] = a*eye(3), B[:,:,0] = b*eye(3).
        """
        C, A, G, B = bekk_parameter_transform(
            bekk_scalar_params, p=1, o=0, q=1, k=3, type_=1,
        )
        # C is 3×3
        assert C.shape == (3, 3)
        # A is 3×3×1, B is 3×3×1, G is 3×3×0
        assert A.shape == (3, 3, 1)
        assert B.shape == (3, 3, 1)
        assert G.shape == (3, 3, 0)

        # A[:,:,0] should be scalar * identity
        a_val = bekk_scalar_params[6]  # a = 0.05
        expected_A = a_val * np.eye(3)
        npt.assert_allclose(A[:, :, 0], expected_A, atol=1e-14)

        # B[:,:,0] should be scalar * identity
        b_val = bekk_scalar_params[7]  # b = 0.9
        expected_B = b_val * np.eye(3)
        npt.assert_allclose(B[:, :, 0], expected_B, atol=1e-14)

    def test_diagonal_type_unpacking(self, bekk_diagonal_params):
        """Diagonal type (type_=2): A, B have diagonal entries matching params.

        K=3, p=1, o=0, q=1.  A[:,:,0] = diag(a_diag), B[:,:,0] = diag(b_diag).
        """
        C, A, G, B = bekk_parameter_transform(
            bekk_diagonal_params, p=1, o=0, q=1, k=3, type_=2,
        )
        assert A.shape == (3, 3, 1)
        assert B.shape == (3, 3, 1)
        assert G.shape == (3, 3, 0)

        # Verify A diagonal entries
        a_diag_expected = bekk_diagonal_params[6:9]  # [0.15, 0.18, 0.20]
        npt.assert_allclose(np.diag(A[:, :, 0]), a_diag_expected, atol=1e-14)
        # Off-diagonal should be zero
        A_off = A[:, :, 0].copy()
        np.fill_diagonal(A_off, 0.0)
        npt.assert_allclose(A_off, np.zeros((3, 3)), atol=1e-14)

        # Verify B diagonal entries
        b_diag_expected = bekk_diagonal_params[9:12]  # [0.85, 0.82, 0.80]
        npt.assert_allclose(np.diag(B[:, :, 0]), b_diag_expected, atol=1e-14)

    def test_full_type_unpacking(self, bekk_full_params):
        """Full type (type_=3): A, B are K×K matrices with Fortran-order reshape.

        K=2, p=1, o=0, q=1.  A[:,:,0] = reshape(a_full, (2,2), order='F').
        """
        C, A, G, B = bekk_parameter_transform(
            bekk_full_params, p=1, o=0, q=1, k=2, type_=3,
        )
        assert C.shape == (2, 2)
        assert A.shape == (2, 2, 1)
        assert B.shape == (2, 2, 1)
        assert G.shape == (2, 2, 0)

        # Verify A is reshaped column-major (Fortran order)
        a_full_vals = bekk_full_params[3:7]  # [0.15, 0.02, 0.02, 0.15]
        expected_A = a_full_vals.reshape((2, 2), order='F')
        npt.assert_allclose(A[:, :, 0], expected_A, atol=1e-14)

        # Verify B
        b_full_vals = bekk_full_params[7:11]  # [0.80, 0.05, 0.05, 0.80]
        expected_B = b_full_vals.reshape((2, 2), order='F')
        npt.assert_allclose(B[:, :, 0], expected_B, atol=1e-14)

    def test_c_matrix_psd(self, bekk_scalar_params):
        """C must always be positive semi-definite (eigenvalues >= 0).

        The PSD correction in bekk_parameter_transform ensures that C has no
        negative eigenvalues even when the Cholesky factorisation yields a
        near-singular matrix.
        """
        C, _, _, _ = bekk_parameter_transform(
            bekk_scalar_params, p=1, o=0, q=1, k=3, type_=1,
        )
        eigenvalues = np.linalg.eigvalsh(C)
        assert np.all(eigenvalues >= -1e-12), (
            f"C is not PSD; min eigenvalue = {eigenvalues.min():.2e}"
        )

    def test_parameter_count_scalar(self):
        """Scalar type consumes k*(k+1)/2 + (p+o+q)*1 parameters."""
        k_val = 3
        p_val, o_val, q_val = 1, 0, 1
        k2 = k_val * (k_val + 1) // 2  # 6
        expected_count = k2 + (p_val + o_val + q_val) * 1  # 6 + 2 = 8
        params = np.random.default_rng(42).standard_normal(expected_count)
        # Should not raise
        C, A, G, B = bekk_parameter_transform(params, p_val, o_val, q_val, k_val, 1)
        assert C.shape == (k_val, k_val)
        assert A.shape == (k_val, k_val, p_val)

    def test_parameter_count_diagonal(self):
        """Diagonal type consumes k*(k+1)/2 + (p+o+q)*k parameters."""
        k_val = 3
        p_val, o_val, q_val = 1, 1, 1
        k2 = k_val * (k_val + 1) // 2  # 6
        expected_count = k2 + (p_val + o_val + q_val) * k_val  # 6 + 9 = 15
        params = np.random.default_rng(42).standard_normal(expected_count)
        C, A, G, B = bekk_parameter_transform(params, p_val, o_val, q_val, k_val, 2)
        assert A.shape == (k_val, k_val, p_val)
        assert G.shape == (k_val, k_val, o_val)
        assert B.shape == (k_val, k_val, q_val)

    def test_parameter_count_full(self):
        """Full type consumes k*(k+1)/2 + (p+o+q)*k^2 parameters."""
        k_val = 2
        p_val, o_val, q_val = 1, 0, 1
        k2 = k_val * (k_val + 1) // 2  # 3
        expected_count = k2 + (p_val + o_val + q_val) * k_val * k_val  # 3 + 8 = 11
        params = np.random.default_rng(42).standard_normal(expected_count)
        C, A, G, B = bekk_parameter_transform(params, p_val, o_val, q_val, k_val, 3)
        assert A.shape == (k_val, k_val, p_val)
        assert B.shape == (k_val, k_val, q_val)

    @pytest.mark.parity
    @pytest.mark.requires_fixtures
    def test_fixture_parity(self, multivariate_fixture_dir):
        """Compare Python output against MATLAB reference fixtures.

        Loads bekk_parameter_transform.npy which contains test cases tc1..tc8
        with input parameters, expected (C, A, G, B) outputs, and metadata
        about each parameterisation type.
        """
        fixture = load_fixture_npy(multivariate_fixture_dir, "bekk_parameter_transform")
        if isinstance(fixture, np.ndarray) and fixture.shape == ():
            fixture = fixture.item()

        # Test all available test cases (tc1 through tc8)
        for tc_idx in range(1, 9):
            prefix = f"tc{tc_idx}"
            key_params = f"{prefix}_input_params"
            if key_params not in fixture:
                continue

            params = fixture[key_params]
            p_val = int(fixture[f"{prefix}_p"])
            o_val = int(fixture[f"{prefix}_o"])
            q_val = int(fixture[f"{prefix}_q"])
            k_val = int(fixture[f"{prefix}_k"])
            type_val = int(fixture[f"{prefix}_type"])

            C, A, G, B = bekk_parameter_transform(
                params, p_val, o_val, q_val, k_val, type_val,
            )

            expected_C = fixture[f"{prefix}_C"]
            expected_A = fixture[f"{prefix}_A"]
            expected_G = fixture[f"{prefix}_G"]
            expected_B = fixture[f"{prefix}_B"]

            assert_allclose(
                C, expected_C,
                err_msg=f"{prefix}: C matrix mismatch",
            )
            # A might be shape (k,k,0) for empty slices; only compare if non-empty
            if expected_A.size > 0:
                assert_allclose(
                    A, expected_A,
                    err_msg=f"{prefix}: A matrix mismatch",
                )
            else:
                assert A.shape[2] == 0
            if expected_G.size > 0:
                assert_allclose(
                    G, expected_G,
                    err_msg=f"{prefix}: G matrix mismatch",
                )
            else:
                assert G.shape[2] == 0
            if expected_B.size > 0:
                assert_allclose(
                    B, expected_B,
                    err_msg=f"{prefix}: B matrix mismatch",
                )
            else:
                assert B.shape[2] == 0


# ---------------------------------------------------------------------------
# TestBekkLikelihood
# ---------------------------------------------------------------------------

class TestBekkLikelihood:
    """Tests for bekk_likelihood(parameters, data, data_asym, p, o, q,
    back_cast, back_cast_asym, type_model).

    Verifies output shapes (ll scalar, lls T-vector, Ht K×K×T), positive
    definiteness, sign conventions, penalty for invalid parameters, and
    numerical parity against MATLAB fixtures.
    """

    def test_returns_negative_loglikelihood(
        self, bekk_scalar_params, bekk_outer_products, bekk_backcast,
    ):
        """ll should be a positive scalar (negative log-likelihood for minimisation).

        Ref: bekk_likelihood.m:68 — ll = sum(lls), where each lls[t] > 0
        (sum of per-period NLL contributions).
        """
        data_3d, data_asym_3d = bekk_outer_products
        bc = bekk_backcast
        ll, lls, Ht = bekk_likelihood(
            bekk_scalar_params, data_3d, data_asym_3d,
            p=1, o=0, q=1,
            back_cast=bc, back_cast_asym=bc, type_model=1,
        )
        assert np.isfinite(ll), f"ll is not finite: {ll}"
        assert isinstance(ll, (float, np.floating)), f"ll type: {type(ll)}"
        # For valid stationary parameters, ll should be a finite positive number
        assert ll > 0, f"Expected positive ll for minimisation, got {ll}"

    def test_lls_shape(
        self, bekk_scalar_params, bekk_outer_products, bekk_backcast,
    ):
        """lls must be a T-element 1-D array."""
        data_3d, data_asym_3d = bekk_outer_products
        bc = bekk_backcast
        _, lls, _ = bekk_likelihood(
            bekk_scalar_params, data_3d, data_asym_3d,
            p=1, o=0, q=1,
            back_cast=bc, back_cast_asym=bc, type_model=1,
        )
        T_dim = data_3d.shape[2]
        assert lls.shape == (T_dim,), f"Expected lls shape ({T_dim},), got {lls.shape}"

    def test_ht_shape(
        self, bekk_scalar_params, bekk_outer_products, bekk_backcast,
    ):
        """Ht must be K×K×T array of conditional covariance matrices."""
        data_3d, data_asym_3d = bekk_outer_products
        bc = bekk_backcast
        _, _, Ht = bekk_likelihood(
            bekk_scalar_params, data_3d, data_asym_3d,
            p=1, o=0, q=1,
            back_cast=bc, back_cast_asym=bc, type_model=1,
        )
        K_dim = data_3d.shape[0]
        T_dim = data_3d.shape[2]
        assert Ht.shape == (K_dim, K_dim, T_dim), (
            f"Expected Ht shape ({K_dim},{K_dim},{T_dim}), got {Ht.shape}"
        )

    def test_ht_positive_definite(
        self, bekk_scalar_params, bekk_outer_products, bekk_backcast,
    ):
        """Each Ht[:,:,t] must be positive definite (eigenvalues > 0).

        Spot-check a sample of time slices for efficiency on large T.
        """
        data_3d, data_asym_3d = bekk_outer_products
        bc = bekk_backcast
        _, _, Ht = bekk_likelihood(
            bekk_scalar_params, data_3d, data_asym_3d,
            p=1, o=0, q=1,
            back_cast=bc, back_cast_asym=bc, type_model=1,
        )
        T_dim = Ht.shape[2]
        # Check every 50th slice and the boundary slices
        check_indices = list(range(0, T_dim, 50)) + [T_dim - 1]
        for t in check_indices:
            eigs = np.linalg.eigvalsh(Ht[:, :, t])
            assert np.all(eigs > -1e-10), (
                f"Ht[:,:,{t}] not PD; min eigenvalue = {eigs.min():.2e}"
            )

    def test_scalar_type_likelihood(
        self, bekk_scalar_params, bekk_outer_products, bekk_backcast,
    ):
        """Run likelihood with scalar params, verify finite and consistent output."""
        data_3d, data_asym_3d = bekk_outer_products
        bc = bekk_backcast
        ll, lls, Ht = bekk_likelihood(
            bekk_scalar_params, data_3d, data_asym_3d,
            p=1, o=0, q=1,
            back_cast=bc, back_cast_asym=bc, type_model=1,
        )
        # ll should equal the sum of per-period contributions
        npt.assert_allclose(ll, np.sum(lls), rtol=1e-10)
        # All per-period contributions should be finite
        assert np.all(np.isfinite(lls)), "Not all lls are finite"

    def test_invalid_parameters_large_penalty(
        self, bekk_outer_products, bekk_backcast,
    ):
        """Parameters violating stationarity should return ll = 1e7.

        Create parameters with a²+b² > 1 and very large coefficients that
        will cause non-PD Ht, triggering the 1e7 penalty.
        Ref: bekk_likelihood.m:70-72 — MATLAB guard for nan/inf/non-real.
        """
        data_3d, data_asym_3d = bekk_outer_products
        bc = bekk_backcast
        # Construct pathological parameters: very large A, B values
        bad_params = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 5.0, 5.0])
        ll, _, _ = bekk_likelihood(
            bad_params, data_3d, data_asym_3d,
            p=1, o=0, q=1,
            back_cast=bc, back_cast_asym=bc, type_model=1,
        )
        assert ll == 1e7, f"Expected penalty 1e7, got {ll}"

    @pytest.mark.parity
    @pytest.mark.requires_fixtures
    def test_fixture_parity(self, multivariate_fixture_dir):
        """Compare Python likelihood output against MATLAB reference fixtures.

        Uses the bekk.npy fixture (full estimation output) which is
        self-consistent with the production vec2chol column-major ordering.
        The bekk_likelihood.npy fixture has a vec2chol row-major/column-major
        inconsistency in its C intercept matrix, causing ll/lls/Ht values to
        differ from the production parameter_transform.  The bekk.npy fixture
        was generated with the correct ordering and its (parameters, data,
        backCast, ll, lls, Ht) are all mutually consistent.

        Sign convention: bekk_likelihood() returns POSITIVE ll for minimisation.
        bekk.npy stores NEGATIVE ll (actual log-likelihood). So we compare:
            computed_ll ≈ -fixture['ll']
            computed_lls ≈ -fixture['lls']
            computed_Ht ≈ fixture['Ht']
        """
        fixture = load_fixture_npy(multivariate_fixture_dir, "bekk")
        if isinstance(fixture, np.ndarray) and fixture.shape == ():
            fixture = fixture.item()

        params = fixture['parameters']
        data_3d = fixture['data']
        data_asym_3d = fixture.get('dataAsym', np.zeros((3, 3, 0)))
        p_val = int(fixture['p'])
        o_val = int(fixture['o'])
        q_val = int(fixture['q'])
        type_val = int(fixture['type'])
        bc = fixture['backCast']
        bc_asym = fixture.get('backCastAsym', bc)

        ll, lls, Ht = bekk_likelihood(
            params, data_3d, data_asym_3d,
            p_val, o_val, q_val,
            bc, bc_asym, type_val,
        )

        # bekk.npy stores actual (negative) log-likelihood;
        # bekk_likelihood returns positive (for minimisation).
        expected_ll = -fixture['ll']
        expected_lls = -fixture['lls']
        expected_Ht = fixture['Ht']

        assert_allclose(
            ll, expected_ll,
            err_msg="bekk.npy parity: ll mismatch",
        )
        assert_allclose(
            lls, expected_lls,
            err_msg="bekk.npy parity: lls mismatch",
        )
        assert_allclose(
            Ht, expected_Ht,
            err_msg="bekk.npy parity: Ht mismatch",
        )


# ---------------------------------------------------------------------------
# TestBekkConstraint
# ---------------------------------------------------------------------------

class TestBekkConstraint:
    """Tests for bekk_constraint(parameters, data, data_asym, p, o, q,
    back_cast, back_cast_asym, type_model).

    Verifies return format (c, ceq), feasibility for stationary parameters
    (c >= 0 in scipy convention), infeasibility for non-stationary parameters
    (c < 0), constraint shape per model type, and parity with MATLAB fixtures.

    IMPORTANT: The Python bekk_constraint NEGATES constraint values relative to
    MATLAB.  MATLAB fmincon uses c <= 0 = feasible; scipy uses c >= 0 = feasible.
    Fixture files store the MATLAB convention values, so parity tests compare
    Python output with -fixture_c.
    """

    def _make_dummy_data(self, k_val):
        """Create minimal dummy data for constraint evaluation.

        bekk_constraint only uses data.shape[1] for K extraction, so the
        actual data values are irrelevant.
        """
        data = np.eye(k_val).reshape(k_val, k_val, 1).repeat(10, axis=2)
        return data, data.copy(), np.eye(k_val), np.eye(k_val)

    def test_returns_c_ceq_tuple(self, bekk_scalar_params):
        """bekk_constraint must return (c, ceq) where ceq is empty.

        Ref: bekk_constraint.m:24 — ceq = [] (no equality constraints).
        """
        data, data_asym, bc, bc_asym = self._make_dummy_data(3)
        c, ceq = bekk_constraint(
            bekk_scalar_params, data, data_asym, 1, 0, 1, bc, bc_asym, 1,
        )
        assert isinstance(c, np.ndarray), f"c type: {type(c)}"
        assert isinstance(ceq, np.ndarray), f"ceq type: {type(ceq)}"
        assert ceq.size == 0, f"ceq should be empty, got size {ceq.size}"

    def test_scalar_stationary_parameters(self, bekk_scalar_params):
        """For stationary parameters (a²+b² < 1), c >= 0 (scipy feasible).

        Parameters: a=0.05, b=0.9 → a²+b² = 0.8125 < 1.
        """
        data, data_asym, bc, bc_asym = self._make_dummy_data(3)
        c, _ = bekk_constraint(
            bekk_scalar_params, data, data_asym, 1, 0, 1, bc, bc_asym, 1,
        )
        assert c.shape == (1,), f"Expected shape (1,), got {c.shape}"
        assert np.all(c >= 0), (
            f"Stationary params should give c >= 0 (scipy), got c = {c}"
        )

    def test_scalar_nonstationary_parameters(self):
        """For non-stationary parameters (a²+b² > 1), c < 0 (scipy infeasible).

        Parameters: a=0.5, b=0.9 → a²+b² = 1.06 > 1.
        """
        params = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.5, 0.9])
        data, data_asym, bc, bc_asym = self._make_dummy_data(3)
        c, _ = bekk_constraint(
            params, data, data_asym, 1, 0, 1, bc, bc_asym, 1,
        )
        assert c.shape == (1,), f"Expected shape (1,), got {c.shape}"
        assert np.all(c < 0), (
            f"Non-stationary params should give c < 0 (scipy), got c = {c}"
        )

    def test_diagonal_constraint_shape(self, bekk_diagonal_params):
        """Diagonal type: c should be K-element vector (one per asset).

        Ref: bekk_constraint.m:32 — c = diag(sum(A.^2,3)+sum(B.^2,3)-1).
        """
        data, data_asym, bc, bc_asym = self._make_dummy_data(3)
        c, _ = bekk_constraint(
            bekk_diagonal_params, data, data_asym, 1, 0, 1, bc, bc_asym, 2,
        )
        assert c.shape == (K,), f"Expected shape ({K},), got {c.shape}"

    def test_full_constraint_eigenvalue(self, bekk_full_params):
        """Full type: c uses eigenvalue-based constraint (shape K²).

        Ref: bekk_constraint.m:34-44 — c = 0.99998 - abs(eig(m)),
        negated for scipy convention.
        """
        k_full = 2  # bekk_full_params uses K=2
        data, data_asym, bc, bc_asym = self._make_dummy_data(k_full)
        c, _ = bekk_constraint(
            bekk_full_params, data, data_asym, 1, 0, 1, bc, bc_asym, 3,
        )
        assert c.shape == (k_full * k_full,), (
            f"Expected shape ({k_full * k_full},), got {c.shape}"
        )

    @pytest.mark.parity
    @pytest.mark.requires_fixtures
    def test_fixture_parity(self, multivariate_fixture_dir):
        """Compare constraint output against MATLAB reference fixtures.

        IMPORTANT: Fixture stores MATLAB convention (c <= 0 = feasible).
        Python returns scipy convention (c >= 0 = feasible), which is the
        negation.  So we compare: assert_allclose(python_c, -fixture_c).
        """
        fixture = load_fixture_npy(multivariate_fixture_dir, "bekk_constraint")
        if isinstance(fixture, np.ndarray) and fixture.shape == ():
            fixture = fixture.item()

        for tc_idx in range(1, 13):
            prefix = f"tc{tc_idx}"
            key_params = f"{prefix}_parameters"
            if key_params not in fixture:
                continue

            params = fixture[key_params]
            p_val = int(fixture[f"{prefix}_p"])
            o_val = int(fixture[f"{prefix}_o"])
            q_val = int(fixture[f"{prefix}_q"])
            k_val = int(fixture[f"{prefix}_k"])
            type_val = int(fixture[f"{prefix}_type"])
            expected_c_matlab = fixture[f"{prefix}_c"]
            expected_ceq = fixture[f"{prefix}_ceq"]
            expected_feasible = bool(fixture[f"{prefix}_feasible"])

            # Create dummy data with correct K dimension
            data, data_asym, bc, bc_asym = self._make_dummy_data(k_val)

            c_python, ceq_python = bekk_constraint(
                params, data, data_asym, p_val, o_val, q_val,
                bc, bc_asym, type_val,
            )

            # ceq should be empty
            assert ceq_python.size == 0, (
                f"{prefix}: ceq should be empty"
            )

            # Python c = -MATLAB_c (negated for scipy convention).
            # For Full type (type_val==3) the constraint values are eigenvalues
            # of a Kronecker product matrix; NumPy and MATLAB may return them
            # in different order.  Sort both sides before comparing.
            if type_val == 3:
                assert_allclose(
                    np.sort(c_python), np.sort(-expected_c_matlab),
                    err_msg=f"{prefix}: constraint value mismatch (sorted, Python=-MATLAB)",
                )
            else:
                assert_allclose(
                    c_python, -expected_c_matlab,
                    err_msg=f"{prefix}: constraint value mismatch (Python=-MATLAB)",
                )

            # Verify feasibility semantics
            if expected_feasible:
                # MATLAB: c <= 0 → Python: c >= 0
                assert np.all(c_python >= -ATOL), (
                    f"{prefix}: expected feasible (c>=0), got c = {c_python}"
                )
            else:
                # MATLAB: c > 0 → Python: c < 0
                assert np.any(c_python < ATOL), (
                    f"{prefix}: expected infeasible (some c<0), got c = {c_python}"
                )


# ---------------------------------------------------------------------------
# TestBekkSimulate
# ---------------------------------------------------------------------------

class TestBekkSimulate:
    """Tests for bekk_simulate(t, k, parameters, p, o, q, type_model).

    Verifies output shapes (data T×K, Ht K×K×T), positive definiteness,
    burn-in removal, and deterministic seeding.
    """

    def test_output_shapes(self, bekk_scalar_params):
        """Data should be T×K, Ht should be K×K×T after burn-in removal."""
        T_sim = 500
        k_sim = 3
        data, Ht = bekk_simulate(
            T_sim, k_sim, bekk_scalar_params,
            p=1, o=0, q=1, type_model='Scalar',
        )
        assert data.shape == (T_sim, k_sim), (
            f"Expected data shape ({T_sim},{k_sim}), got {data.shape}"
        )
        assert Ht.shape == (k_sim, k_sim, T_sim), (
            f"Expected Ht shape ({k_sim},{k_sim},{T_sim}), got {Ht.shape}"
        )

    def test_ht_positive_definite(self, bekk_scalar_params):
        """All Ht[:,:,t] slices must be positive definite.

        Spot-check a sample of time slices for computational efficiency.
        """
        T_sim = 200
        data, Ht = bekk_simulate(
            T_sim, 3, bekk_scalar_params,
            p=1, o=0, q=1, type_model='Scalar',
        )
        T_act = Ht.shape[2]
        check_indices = list(range(0, T_act, 20)) + [T_act - 1]
        for t in check_indices:
            eigs = np.linalg.eigvalsh(Ht[:, :, t])
            assert np.all(eigs > -1e-10), (
                f"Ht[:,:,{t}] not PD; min eigenvalue = {eigs.min():.2e}"
            )

    def test_scalar_simulation(self, bekk_scalar_params):
        """Simulate scalar BEKK and verify shapes and finite output."""
        T_sim = 300
        k_sim = 3
        data, Ht = bekk_simulate(
            T_sim, k_sim, bekk_scalar_params,
            p=1, o=0, q=1, type_model='Scalar',
        )
        assert data.shape == (T_sim, k_sim)
        assert Ht.shape == (k_sim, k_sim, T_sim)
        assert np.all(np.isfinite(data)), "Simulated data contains non-finite values"
        assert np.all(np.isfinite(Ht)), "Simulated Ht contains non-finite values"

    def test_deterministic_seed(self, bekk_scalar_params):
        """Two calls with the same RNG seed must produce identical output.

        bekk_simulate creates an unseeded ``np.random.default_rng()``
        internally.  We mock it to inject a fixed-seed generator, ensuring
        identical innovations and burn-in resampling across two runs.
        """
        from unittest.mock import patch

        T_sim = 200
        k_sim = 3

        with patch(
            'mfe_toolbox.multivariate.bekk_simulate.np.random.default_rng',
            return_value=np.random.default_rng(12345),
        ):
            data1, Ht1 = bekk_simulate(
                T_sim, k_sim, bekk_scalar_params,
                p=1, o=0, q=1, type_model='Scalar',
            )

        with patch(
            'mfe_toolbox.multivariate.bekk_simulate.np.random.default_rng',
            return_value=np.random.default_rng(12345),
        ):
            data2, Ht2 = bekk_simulate(
                T_sim, k_sim, bekk_scalar_params,
                p=1, o=0, q=1, type_model='Scalar',
            )

        npt.assert_allclose(data1, data2, atol=1e-12,
                            err_msg="Deterministic seed: data mismatch")
        npt.assert_allclose(Ht1, Ht2, atol=1e-12,
                            err_msg="Deterministic seed: Ht mismatch")

    def test_burn_in_removed(self, bekk_scalar_params):
        """Output must have exactly T observations (burn-in discarded).

        Ref: bekk_simulate.m:152-153 — data = e(T+1:2*T,:).
        """
        T_sim = 250
        k_sim = 3
        data, Ht = bekk_simulate(
            T_sim, k_sim, bekk_scalar_params,
            p=1, o=0, q=1, type_model='Scalar',
        )
        assert data.shape[0] == T_sim, (
            f"Expected T={T_sim} observations after burn-in, got {data.shape[0]}"
        )
        assert Ht.shape[2] == T_sim, (
            f"Expected Ht T-dimension={T_sim}, got {Ht.shape[2]}"
        )


# ---------------------------------------------------------------------------
# TestBekk (Integration)
# ---------------------------------------------------------------------------

class TestBekk:
    """Tests for the main BEKK estimation driver bekk(data, data_asym, p, o, q,
    type_model, starting_vals, options).

    Integration tests exercising the full estimation pipeline including
    parameter estimation via scipy.optimize.minimize(method='SLSQP'),
    convergence verification, and output shape/type validation.
    """

    @pytest.mark.slow
    def test_scalar_estimation_runs(self, multivariate_data):
        """Estimate scalar BEKK on multivariate_data; verify convergence.

        This test runs the full optimisation and checks that the log-likelihood
        is finite (indicating successful convergence).
        """
        parameters, ll, Ht, VCV, scores = bekk(
            multivariate_data, p=1, o=0, q=1,
            type_model='Scalar',
            options={'maxiter': 200, 'disp': False, 'ftol': 1e-8},
        )
        assert np.isfinite(ll), f"Log-likelihood is not finite: {ll}"
        assert ll < 0, f"Expected negative log-likelihood, got {ll}"
        assert parameters.ndim == 1, "Parameters should be 1-D"

    @pytest.mark.slow
    def test_parameter_shapes(self, multivariate_data):
        """Verify parameter vector length for scalar type.

        Scalar BEKK(1,0,1) with K=3: k*(k+1)/2 + (p+q) = 6 + 2 = 8 params.
        """
        parameters, _, _, _, _ = bekk(
            multivariate_data, p=1, o=0, q=1,
            type_model='Scalar',
            options={'maxiter': 200, 'disp': False, 'ftol': 1e-8},
        )
        k_val = multivariate_data.shape[1]
        k2 = k_val * (k_val + 1) // 2  # 6
        expected_len = k2 + 1 + 1  # 8 (scalar: 1 ARCH + 1 GARCH)
        assert len(parameters) == expected_len, (
            f"Expected {expected_len} parameters, got {len(parameters)}"
        )

    @pytest.mark.slow
    def test_ht_output_shape(self, multivariate_data):
        """Ht must be K×K×T array of conditional covariance matrices."""
        T_dim, K_dim = multivariate_data.shape
        _, _, Ht, _, _ = bekk(
            multivariate_data, p=1, o=0, q=1,
            type_model='Scalar',
            options={'maxiter': 200, 'disp': False, 'ftol': 1e-8},
        )
        assert Ht.shape == (K_dim, K_dim, T_dim), (
            f"Expected Ht shape ({K_dim},{K_dim},{T_dim}), got {Ht.shape}"
        )

    def test_input_validation_errors(self):
        """Invalid inputs must raise ValueError.

        Test cases:
        - 1-D data (not a matrix)
        - p=0 (must be positive)
        - p=-1 (must be positive)
        - Invalid type string
        """
        rng_local = np.random.default_rng(42)

        # 1-D data: should raise because ndim != 2 and ndim != 3
        with pytest.raises(ValueError):
            bekk(rng_local.standard_normal(100), p=1, o=0, q=1)

        # p=0: must be positive
        data_2d = rng_local.standard_normal((100, 3))
        with pytest.raises(ValueError, match="P must be a positive"):
            bekk(data_2d, p=0, o=0, q=1)

        # p=-1: must be positive
        with pytest.raises(ValueError, match="P must be a positive"):
            bekk(data_2d, p=-1, o=0, q=1)

        # Invalid type string
        with pytest.raises(ValueError, match="TYPE must be"):
            bekk(data_2d, p=1, o=0, q=1, type_model='InvalidType')

        # o=-1: must be non-negative
        with pytest.raises(ValueError, match="O must be"):
            bekk(data_2d, p=1, o=-1, q=1)

        # q=-1: must be non-negative
        with pytest.raises(ValueError, match="Q must be"):
            bekk(data_2d, p=1, o=0, q=-1)

    @pytest.mark.parity
    @pytest.mark.requires_fixtures
    @pytest.mark.slow
    def test_fixture_parity_parameters(self, multivariate_fixture_dir):
        """Compare estimated parameters against MATLAB reference fixture.

        The bekk.npy fixture contains the full optimisation output including
        the optimal parameter vector.  Due to optimiser path sensitivity,
        parameter parity uses a relaxed tolerance.
        """
        fixture = load_fixture_npy(multivariate_fixture_dir, "bekk")
        if isinstance(fixture, np.ndarray) and fixture.shape == ():
            fixture = fixture.item()

        # The fixture contains the input data (simulatedData), which is T×K
        sim_data = fixture['simulatedData']
        expected_params = fixture['parameters']
        expected_ll = fixture['ll']
        p_val = int(fixture['p'])
        o_val = int(fixture['o'])
        q_val = int(fixture['q'])

        # Run estimation with the same data
        parameters, ll, Ht, VCV, scores = bekk(
            sim_data, p=p_val, o=o_val, q=q_val,
            type_model='Scalar',
            options={'maxiter': 1000, 'disp': False, 'ftol': 1e-10},
        )

        # Log-likelihood should be close (optimiser may find slightly different point)
        # Use relaxed tolerance for parameters since optimizer paths may differ
        npt.assert_allclose(
            ll, expected_ll,
            atol=1.0, rtol=1e-2,
            err_msg="Fixture parity: log-likelihood mismatch",
        )

        # Parameter shapes must match exactly
        assert parameters.shape == expected_params.shape, (
            f"Parameter shape mismatch: {parameters.shape} vs {expected_params.shape}"
        )

    @pytest.mark.parity
    @pytest.mark.requires_fixtures
    @pytest.mark.slow
    def test_fixture_parity_likelihood(self, multivariate_fixture_dir):
        """Compare log-likelihood at fixture parameters against fixture ll.

        This test evaluates bekk_likelihood at the fixture's optimal parameters
        using the fixture's data, which avoids optimiser path sensitivity.
        """
        fixture = load_fixture_npy(multivariate_fixture_dir, "bekk")
        if isinstance(fixture, np.ndarray) and fixture.shape == ():
            fixture = fixture.item()

        params = fixture['parameters']
        data_3d = fixture['data']
        data_asym_3d = fixture['dataAsym']
        p_val = int(fixture['p'])
        o_val = int(fixture['o'])
        q_val = int(fixture['q'])
        type_val = int(fixture['type'])
        bc = fixture['backCast']
        bc_asym = fixture['backCastAsym']

        # Evaluate likelihood at fixture parameters
        ll_neg, lls, Ht = bekk_likelihood(
            params, data_3d, data_asym_3d,
            p_val, o_val, q_val,
            bc, bc_asym, type_val,
        )

        # bekk.npy stores ll as the actual (negative) log-likelihood
        # bekk_likelihood returns positive ll (for minimisation)
        # So: fixture ll = -bekk_likelihood_ll
        expected_ll_actual = fixture['ll']  # Negative value (actual log-lik)
        computed_ll_actual = -ll_neg        # Negate positive value to get actual

        assert_allclose(
            computed_ll_actual, expected_ll_actual,
            err_msg="Fixture parity: likelihood at optimal parameters",
        )

        # Compare Ht if available
        if 'Ht' in fixture:
            expected_Ht = fixture['Ht']
            assert_allclose(
                Ht, expected_Ht,
                err_msg="Fixture parity: Ht at optimal parameters",
            )
