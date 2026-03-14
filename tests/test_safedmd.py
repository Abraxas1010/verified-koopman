from __future__ import annotations

import numpy as np
import pytest

import verified_koopman as vk
from verified_koopman.verification.safedmd import (
    EDMDErrorBound,
    compute_edmd_matrices,
    compute_safedmd_error_bound,
    synthesize_controller,
    verify_certificate,
)


def test_generate_system_data_shapes() -> None:
    (train, test, dt, meta) = vk.generate_system_data("toy2d", n_traj=16, time=21, seed=0)
    assert train[0].shape == train[1].shape
    assert test[0].shape == test[1].shape
    assert train[0].shape[1] == 2
    assert meta["state_dim"] == 2
    assert dt > 0.0


def test_safedmd_error_bound_is_psd() -> None:
    rng = np.random.default_rng(0)
    psi_x = rng.normal(size=(128, 4))
    psi_y = psi_x @ (0.8 * np.eye(4)) + 0.01 * rng.normal(size=(128, 4))
    k_hat, g_matrix, a_matrix = compute_edmd_matrices(psi_x, psi_y, psi_x, psi_y)
    bound = compute_safedmd_error_bound(k_hat, g_matrix, a_matrix, psi_x, psi_y)
    eigvals = np.linalg.eigvalsh(bound.E_matrix)
    assert eigvals.min() >= -1e-10
    assert bound.spectral_norm >= 0.0
    assert bound.dictionary_size == 4


def test_controller_smoke() -> None:
    pytest.importorskip("cvxpy")
    m = 4
    k_hat = 0.5 * np.eye(m)
    b_input = np.eye(m)
    bound = EDMDErrorBound(
        E_matrix=0.01 * np.eye(m),
        spectral_norm=0.01,
        data_points_used=100,
        dictionary_size=m,
        condition_number=1.0,
        nucleus_applied=False,
    )
    cert = synthesize_controller(k_hat, b_input, bound, input_dim=m, decay_target=0.95, verbose=False)
    assert cert is not None
    assert cert.sdp_status in {"optimal", "optimal_inaccurate"}
    check = verify_certificate(cert, k_hat, b_input, bound, n_samples=1024)
    assert float(check["satisfaction_rate"]) > 0.99
