from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch

from verified_koopman.models.learnable_heyting import ParametricHeytingOps


@dataclass(frozen=True)
class HeytingStats:
    steps: int
    latent_dim: int
    boundary_max_abs: float
    regularity_gap_mean_l1: float
    double_neg_change_frac: float


def check_himp_adjoint(hey: ParametricHeytingOps, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    lhs = (hey.meet(a, c) <= b).all(dim=-1)
    rhs = (c <= hey.himp_hard(a, b)).all(dim=-1)
    return lhs == rhs


def stats_for_traj(hey: ParametricHeytingOps, z: torch.Tensor, *, eps: float = 1e-9) -> HeytingStats:
    if z.ndim != 2:
        raise ValueError(f"expected z shape (steps,k), got {tuple(z.shape)}")

    boundary = hey.meet(z, hey.hnot_hard(z))
    bot = hey.bot_like(boundary)
    boundary_err = (boundary - bot).abs()

    dn = hey.double_neg_hard(z)
    gap = (dn - z)
    changed = gap.abs() > float(eps)

    return HeytingStats(
        steps=int(z.shape[0]),
        latent_dim=int(z.shape[1]),
        boundary_max_abs=float(boundary_err.amax().item()) if boundary_err.numel() else 0.0,
        regularity_gap_mean_l1=float(gap.abs().sum(dim=-1).mean().item()) if gap.numel() else 0.0,
        double_neg_change_frac=float(changed.float().mean().item()) if changed.numel() else 0.0,
    )


def pick_top_from_data(z_closed: torch.Tensor, *, quantile: float = 0.99, min_top: float = 1.0) -> float:
    if z_closed.numel() == 0:
        return float(min_top)
    vals = z_closed.reshape(-1)
    vals = vals[torch.isfinite(vals)]
    if vals.numel() == 0:
        return float(min_top)
    q = float(torch.quantile(vals, torch.tensor(float(quantile), device=z_closed.device)).item())
    return float(max(float(min_top), q))


def to_jsonable(stats: HeytingStats) -> Dict[str, float]:
    return {
        "steps": float(stats.steps),
        "latent_dim": float(stats.latent_dim),
        "boundary_max_abs": float(stats.boundary_max_abs),
        "regularity_gap_mean_l1": float(stats.regularity_gap_mean_l1),
        "double_neg_change_frac": float(stats.double_neg_change_frac),
    }

