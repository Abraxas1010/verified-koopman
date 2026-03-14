from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F

from verified_koopman.models.learnable_heyting import LearnableBounds, LearnableThresholdNucleus, ParametricHeytingOps


@dataclass(frozen=True)
class HeytingLossOut:
    total: torch.Tensor
    losses: Dict[str, torch.Tensor]
    metrics: Dict[str, float]


def compute_heyting_losses(
    *,
    epoch: int,
    z: torch.Tensor,  # projected
    z_raw: torch.Tensor,  # raw encoder
    z_nuc: torch.Tensor,  # after nucleus, before clamp
    bounds: LearnableBounds,
    nucleus: LearnableThresholdNucleus,
    heyting: ParametricHeytingOps,
    lambda_regularity: float,
    lambda_tightness: float,
    lambda_internalization: float,
) -> HeytingLossOut:
    _ = epoch
    losses: Dict[str, torch.Tensor] = {}
    metrics: Dict[str, float] = {}

    # Regularity proxy: E[||¬¬z - z||_1] using soft ops for gradients.
    gap = (heyting.double_neg_soft(z) - z).abs().sum(dim=-1).mean()
    losses["regularity"] = float(lambda_regularity) * gap
    metrics["gap_l1"] = float(gap.detach().cpu().item())

    # Tightness: keep bounds near the latent range.
    if lambda_tightness > 0:
        zmin = z_nuc.amin(dim=0).detach()
        zmax = z_nuc.amax(dim=0).detach()
        lo = bounds.lo.to(device=z.device, dtype=z.dtype)
        hi = bounds.hi.to(device=z.device, dtype=z.dtype)
        lo_violation = F.relu(lo - zmin).mean()
        hi_violation = F.relu(zmax - hi).mean()
        data_range = (zmax - zmin).clamp(min=0.1)
        excess_width = F.relu((hi - lo) - data_range).mean()
        tight = lo_violation + hi_violation + 0.1 * excess_width
        losses["tightness"] = float(lambda_tightness) * tight
        metrics["tightness"] = float(tight.detach().cpu().item())
    else:
        metrics["tightness"] = 0.0

    # Internalization: penalize changes made by nucleus/clamp.
    if lambda_internalization > 0:
        thr_delta = (nucleus(z_raw) - z_raw).abs().mean()
        clamp_delta = (bounds.clamp(z_nuc) - z_nuc).abs().mean()
        internal = thr_delta + clamp_delta
        losses["internalization"] = float(lambda_internalization) * internal
        metrics["internalization_delta"] = float(internal.detach().cpu().item())
    else:
        metrics["internalization_delta"] = 0.0

    with torch.no_grad():
        metrics["internalization_rate"] = float(nucleus.internalization_rate(z_raw).detach().cpu().item())
        metrics.update(bounds.stats())
        metrics.update(nucleus.stats())
        metrics["boundary_violation_max"] = float(heyting.boundary_violation_hard(z).detach().cpu().item())

    total = sum(losses.values()) if losses else torch.zeros((), device=z.device)
    return HeytingLossOut(total=total, losses=losses, metrics=metrics)

