"""SafEDMD-inspired residual envelopes for EDMD Koopman approximations.

This module adapts the SafEDMD perspective to the NBA setting with a
residual-covariance-based PSD error envelope. The resulting matrix

    E = (G + λI)^(-1) R (G + λI)^(-1)

is an empirical construction tied directly to the learned residuals; it is not
a direct transcription of the paper's scalar proportional-error theorem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np
import torch


Array = np.ndarray


@dataclass(frozen=True)
class EDMDErrorBound:
    """Residual-covariance PSD error envelope inspired by SafEDMD."""

    E_matrix: Array
    spectral_norm: float
    data_points_used: int
    dictionary_size: int
    condition_number: float
    nucleus_applied: bool

    def to_jsonable(self) -> dict[str, float | int | bool | list[list[float]]]:
        return {
            "E_matrix": np.asarray(self.E_matrix, dtype=np.float64).tolist(),
            "spectral_norm": float(self.spectral_norm),
            "data_points_used": int(self.data_points_used),
            "dictionary_size": int(self.dictionary_size),
            "condition_number": float(self.condition_number),
            "nucleus_applied": bool(self.nucleus_applied),
        }


def _as_2d(name: str, arr: Array) -> Array:
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {out.shape}")
    return out


def _symmetrize(mat: Array) -> Array:
    return 0.5 * (mat + mat.T)


def _project_psd(mat: Array, *, floor: float = 0.0) -> Tuple[Array, Array]:
    eigvals, eigvecs = np.linalg.eigh(_symmetrize(mat))
    clipped = np.maximum(eigvals, floor)
    psd = (eigvecs * clipped) @ eigvecs.T
    return _symmetrize(psd), clipped


def active_dictionary_mask(psi_x: Array, psi_y: Array, *, tol: float = 1e-8) -> Array:
    energy = np.mean(np.square(psi_x) + np.square(psi_y), axis=0)
    max_energy = float(np.max(energy, initial=0.0))
    if max_energy <= tol:
        return np.ones(psi_x.shape[1], dtype=bool)
    mask = energy > (tol * max_energy)
    if not np.any(mask):
        return np.ones(psi_x.shape[1], dtype=bool)
    return mask


def _restrict_active_dictionary(psi_x: Array, psi_y: Array, k_hat: Array, *, tol: float = 1e-8) -> Tuple[Array, Array, Array]:
    mask = active_dictionary_mask(psi_x, psi_y, tol=tol)
    idx = np.flatnonzero(mask)
    return psi_x[:, idx], psi_y[:, idx], k_hat[np.ix_(idx, idx)]


def compute_edmd_matrices(
    X_data: Array,
    Y_data: Array,
    psi_X: Array,
    psi_Y: Array,
) -> Tuple[Array, Array, Array]:
    """Compute EDMD matrices in row-batch form.

    `psi_X` and `psi_Y` are shaped `(N, M)` where rows are samples.
    The returned `K_hat` is a column-operator `(M, M)` satisfying
    `psi_{t+1} ≈ (K_hat @ psi_tᵀ)ᵀ`, equivalently `psi_Y ≈ psi_X @ K_hatᵀ`.
    """

    x = _as_2d("X_data", X_data)
    y = _as_2d("Y_data", Y_data)
    px = _as_2d("psi_X", psi_X)
    py = _as_2d("psi_Y", psi_Y)
    if x.shape[0] != y.shape[0] or px.shape[0] != py.shape[0] or x.shape[0] != px.shape[0]:
        raise ValueError("X_data, Y_data, psi_X, and psi_Y must share sample count")
    if x.shape[0] == 0:
        raise ValueError("at least one sample is required")

    n = float(px.shape[0])
    g_matrix = (px.T @ px) / n
    a_matrix = (px.T @ py) / n
    k_hat = np.linalg.pinv(g_matrix) @ a_matrix
    return k_hat, _symmetrize(g_matrix), a_matrix


def compute_safedmd_error_bound(
    K_hat: Array,
    G_matrix: Array,
    A_matrix: Array,
    psi_X: Array,
    psi_Y: Array,
    regularization: float = 1e-6,
    *,
    nucleus_applied: bool = False,
) -> EDMDErrorBound:
    """Compute a PSD residual envelope from EDMD data.

    This routine uses a SafEDMD-inspired, residual-covariance construction in
    lifted coordinates:

      residual_i = ψ(y_i) - K_hat ψ(x_i)
      R = (1/N) Σ residual_i residual_iᵀ
      E = (G + λI)^{-1} R (G + λI)^{-1}

    The paper's theorem is a scalar proportional bound with coefficients on the
    state and input mismatch terms. Here we retain the same motivation, but
    construct a machine-checkable PSD matrix envelope from data residuals for
    the NBA/EDMD comparison.

    All batch inputs use row-vector convention `(N, M)`. `K_hat` is a column
    operator, so residuals are computed as `psi_Y - psi_X @ K_hat.T`.
    """

    k_hat = _as_2d("K_hat", K_hat)
    g_matrix = _as_2d("G_matrix", G_matrix)
    a_matrix = _as_2d("A_matrix", A_matrix)
    px = _as_2d("psi_X", psi_X)
    py = _as_2d("psi_Y", psi_Y)
    if px.shape != py.shape:
        raise ValueError(f"psi_X and psi_Y must match, got {px.shape} vs {py.shape}")
    if k_hat.shape != (px.shape[1], px.shape[1]):
        raise ValueError(f"K_hat must be square with dictionary size {px.shape[1]}, got {k_hat.shape}")
    if g_matrix.shape != k_hat.shape or a_matrix.shape != k_hat.shape:
        raise ValueError("G_matrix and A_matrix must match K_hat shape")

    reg = float(regularization)
    if reg < 0.0:
        raise ValueError("regularization must be nonnegative")

    n = float(px.shape[0])
    residuals = py - px @ k_hat.T
    residual_cov = _symmetrize((residuals.T @ residuals) / n)
    g_reg = _symmetrize(g_matrix + reg * np.eye(g_matrix.shape[0], dtype=np.float64))
    g_inv = np.linalg.pinv(g_reg)
    e_matrix, eigvals = _project_psd(g_inv @ residual_cov @ g_inv)
    spectral = float(eigvals.max(initial=0.0))
    condition = float(np.linalg.cond(g_reg))

    return EDMDErrorBound(
        E_matrix=e_matrix,
        spectral_norm=spectral,
        data_points_used=int(px.shape[0]),
        dictionary_size=int(px.shape[1]),
        condition_number=condition,
        nucleus_applied=bool(nucleus_applied),
    )


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@torch.no_grad()
def compute_nucleus_aware_error_bound(
    model: Any,
    X_data: torch.Tensor,
    Y_data: torch.Tensor,
    regularization: float = 1e-6,
    device: str = "cpu",
) -> EDMDErrorBound:
    """Compute a SafEDMD-inspired residual envelope from a trained model.

    Boundary convention:
    - `model.generator.step(z, dt)` uses row vectors: `z_next = z @ M`
    - this routine converts to a column operator `K_hat = Mᵀ`
    - lifted residuals are then `psi_Y - psi_X @ K_hatᵀ`, which matches the
      row-batch form used by the rest of the SafEDMD code

    In the nucleus-aware case the lifted dictionary is first restricted to the
    active subspace induced by the projection, so the bound reflects the actual
    effective dictionary rather than dormant latent coordinates.
    """

    target = torch.device(device)
    current = _model_device(model)
    model = model.to(target)
    was_training = bool(model.training)
    model.eval()

    x = X_data.to(device=target, dtype=torch.float32)
    y = Y_data.to(device=target, dtype=torch.float32)

    psi_x_t = model.encode(x)
    psi_y_t = model.encode(y)
    generator_matrix = model.generator.matrix().detach()
    step_matrix_row = torch.matrix_exp(generator_matrix * float(model.cfg.dt))
    k_hat_col = step_matrix_row.transpose(0, 1).cpu().numpy()

    psi_x = psi_x_t.detach().cpu().numpy().astype(np.float64)
    psi_y = psi_y_t.detach().cpu().numpy().astype(np.float64)
    nucleus_applied = bool(hasattr(model, "nucleus"))
    if nucleus_applied:
        psi_x, psi_y, k_hat_col = _restrict_active_dictionary(psi_x, psi_y, k_hat_col)
    x_np = x.detach().cpu().numpy().astype(np.float64)
    y_np = y.detach().cpu().numpy().astype(np.float64)

    _k_hat_edmd, g_matrix, a_matrix = compute_edmd_matrices(x_np, y_np, psi_x, psi_y)
    bound = compute_safedmd_error_bound(
        k_hat_col,
        g_matrix,
        a_matrix,
        psi_x,
        psi_y,
        regularization=float(regularization),
        nucleus_applied=nucleus_applied,
    )

    model = model.to(current)
    if was_training:
        model.train()
    return bound
