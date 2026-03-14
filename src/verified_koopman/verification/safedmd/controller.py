"""Autonomous-system controller synthesis from SafEDMD-style error envelopes.

The full SafEDMD paper studies a richer control-affine setting with a larger
block LMI. This module implements the simplified autonomous-system variant used
for the Koopman NBA experiments, where the learned lifted dynamics are
stabilized against an additive PSD error envelope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from verified_koopman.verification.safedmd.error_bounds import EDMDErrorBound

try:  # pragma: no cover - exercised in integration environments
    import cvxpy as cp
except Exception:  # pragma: no cover
    cp = None


Array = np.ndarray


@dataclass(frozen=True)
class ControllerCertificate:
    gain_L: Array
    lyapunov_P: Array
    decay_rate: float
    certified_region_radius: float
    sdp_status: str
    error_bound_used: float
    nucleus_aware: bool

    def to_jsonable(self) -> dict[str, float | bool | str | list[list[float]]]:
        return {
            "gain_L": np.asarray(self.gain_L, dtype=np.float64).tolist(),
            "lyapunov_P": np.asarray(self.lyapunov_P, dtype=np.float64).tolist(),
            "decay_rate": float(self.decay_rate),
            "certified_region_radius": float(self.certified_region_radius),
            "sdp_status": str(self.sdp_status),
            "error_bound_used": float(self.error_bound_used),
            "nucleus_aware": bool(self.nucleus_aware),
        }


def _as_square(name: str, arr: Array) -> Array:
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim != 2 or out.shape[0] != out.shape[1]:
        raise ValueError(f"{name} must be square, got {out.shape}")
    return out


def _symmetrize(mat: Array) -> Array:
    return 0.5 * (mat + mat.T)


def _require_cvxpy() -> None:
    if cp is None:
        raise RuntimeError("cvxpy is required for SafEDMD controller synthesis")


def _solve_problem(prob: "cp.Problem", verbose: bool) -> str:
    solvers = []
    if hasattr(cp, "SCS"):
        solvers.append((cp.SCS, {"verbose": verbose, "max_iters": 5000, "eps": 1e-5}))
    if hasattr(cp, "MOSEK"):
        solvers.append((cp.MOSEK, {"verbose": verbose}))
    if hasattr(cp, "SDPA"):
        solvers.append((cp.SDPA, {"verbose": verbose}))

    last_status = "solver_not_run"
    for solver, kwargs in solvers:
        try:
            prob.solve(solver=solver, **kwargs)
        except Exception:
            continue
        last_status = str(prob.status)
        if prob.status in {"optimal", "optimal_inaccurate"}:
            return last_status
    return last_status


def synthesize_controller(
    K_hat: Array,
    B_input: Array,
    error_bound: EDMDErrorBound,
    input_dim: int,
    decay_target: float = 0.95,
    region_radius: float = 1.0,
    verbose: bool = True,
) -> Optional[ControllerCertificate]:
    """Synthesize a robust linear feedback gain in lifted coordinates.

    The LMI is solved in `(Q, Y)` coordinates where `Q = P⁻¹` and `L = Y Q⁻¹`.
    We require the Schur block

      [ α²Q - ρE   (KQ + BY) ]
      [ (KQ + BY)ᵀ     Q     ]  ≽ εI

    with `ρ = region_radius`. This is a conservative robustification, but it
    gives an honest convex synthesis problem tied directly to the SafEDMD
    envelope actually computed from data.

    This is a simplified autonomous-system LMI inspired by SafEDMD Section 4.
    The paper's full equation (17) handles bilinear control-affine structure;
    the NBA experiment only needs the reduced lifted linear setting.
    """

    _require_cvxpy()

    k_hat = _as_square("K_hat", K_hat)
    b_input = np.asarray(B_input, dtype=np.float64)
    if b_input.ndim != 2 or b_input.shape[0] != k_hat.shape[0]:
        raise ValueError(f"B_input must have shape ({k_hat.shape[0]}, m), got {b_input.shape}")
    if int(input_dim) != int(b_input.shape[1]):
        raise ValueError(f"input_dim={input_dim} does not match B_input width {b_input.shape[1]}")
    if not (0.0 < float(decay_target) < 1.0):
        raise ValueError("decay_target must lie in (0, 1)")
    if float(region_radius) <= 0.0:
        raise ValueError("region_radius must be positive")

    n = int(k_hat.shape[0])
    m = int(b_input.shape[1])
    eps = 1e-6
    e_matrix = _symmetrize(np.asarray(error_bound.E_matrix, dtype=np.float64))

    q = cp.Variable((n, n), symmetric=True)
    y = cp.Variable((m, n))
    acl_q = k_hat @ q + b_input @ y
    robust_weight = max(float(region_radius), 1e-6)
    alpha_sq = float(decay_target) ** 2

    block = cp.bmat(
        [
            [alpha_sq * q - robust_weight * e_matrix, acl_q],
            [acl_q.T, q],
        ]
    )
    constraints = [
        q >> eps * np.eye(n),
        block >> eps * np.eye(2 * n),
    ]
    objective = cp.Minimize(cp.trace(q))
    prob = cp.Problem(objective, constraints)
    status = _solve_problem(prob, verbose=bool(verbose))
    if status not in {"optimal", "optimal_inaccurate"} or q.value is None or y.value is None:
        return None

    q_val = _symmetrize(np.asarray(q.value, dtype=np.float64))
    y_val = np.asarray(y.value, dtype=np.float64)
    p_val = _symmetrize(np.linalg.inv(q_val))
    gain = y_val @ np.linalg.inv(q_val)

    return ControllerCertificate(
        gain_L=gain,
        lyapunov_P=p_val,
        decay_rate=float(decay_target),
        certified_region_radius=float(region_radius),
        sdp_status=str(status),
        error_bound_used=float(error_bound.spectral_norm),
        nucleus_aware=bool(error_bound.nucleus_applied),
    )


def verify_certificate(
    cert: ControllerCertificate,
    K_hat: Array,
    B_input: Array,
    error_bound: EDMDErrorBound,
    n_samples: int = 10000,
) -> Dict[str, float | bool | int]:
    """Monte Carlo sanity check for a synthesized certificate."""

    k_hat = _as_square("K_hat", K_hat)
    b_input = np.asarray(B_input, dtype=np.float64)
    p = _symmetrize(_as_square("lyapunov_P", cert.lyapunov_P))
    gain = np.asarray(cert.gain_L, dtype=np.float64)
    if b_input.shape[1] != gain.shape[0]:
        raise ValueError("B_input and gain_L have incompatible shapes")

    acl = k_hat + b_input @ gain
    e_matrix = _symmetrize(np.asarray(error_bound.E_matrix, dtype=np.float64))
    rng = np.random.default_rng(0)
    successes = 0
    max_ratio = 0.0
    max_target_ratio = 0.0
    tol = 1e-8

    for _ in range(int(n_samples)):
        z = rng.normal(size=(k_hat.shape[0],))
        norm = float(np.linalg.norm(z))
        if norm == 0.0:
            continue
        radius = float(cert.certified_region_radius) * float(rng.random() ** (1.0 / max(1, k_hat.shape[0])))
        z = radius * z / norm
        z_next = acl @ z
        v = float(z @ p @ z)
        if v <= tol:
            continue
        v_next = float(z_next @ p @ z_next)
        robust_target = float(cert.decay_rate) ** 2 * v + float(z @ e_matrix @ z)
        max_ratio = max(max_ratio, v_next / v)
        max_target_ratio = max(max_target_ratio, robust_target / v)
        if v_next <= robust_target + tol:
            successes += 1

    used = max(1, int(n_samples))
    rate = float(successes / used)
    return {
        "satisfaction_rate": rate,
        "max_ratio": float(max_ratio),
        "target_ratio": float(max_target_ratio),
        "n_samples": int(n_samples),
        "all_satisfied": bool(rate >= 1.0 - 1.0 / used),
    }
