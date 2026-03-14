from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from verified_koopman.models.learnable_heyting import LearnableBounds, LearnableThresholdNucleus, ParametricHeytingOps


class CurriculumScheduler:
    def __init__(self, *, target: float, schedule: str, stage1_epochs: int, warmup_epochs: int):
        self.target = float(target)
        self.schedule = str(schedule)
        self.stage1_epochs = int(stage1_epochs)
        self.warmup_epochs = int(warmup_epochs)

    def value(self, epoch: int) -> float:
        e = int(epoch)
        if self.schedule == "constant":
            return self.target
        if self.schedule == "linear_warmup":
            if self.warmup_epochs <= 0:
                return self.target
            return self.target * min(1.0, float(e) / float(self.warmup_epochs))
        if self.schedule == "cosine_warmup":
            if self.warmup_epochs <= 0:
                return self.target
            t = min(1.0, float(e) / float(self.warmup_epochs))
            return self.target * (1.0 - math.cos(math.pi * t)) / 2.0
        if self.schedule == "staged":
            if e < self.stage1_epochs:
                return 0.0
            e2 = e - self.stage1_epochs
            if self.warmup_epochs <= 0:
                return self.target
            return self.target * min(1.0, float(e2) / float(self.warmup_epochs))
        raise ValueError(f"unknown schedule={self.schedule!r}")


@dataclass(frozen=True)
class CurriculumHeytingOut:
    total: torch.Tensor
    losses: Dict[str, torch.Tensor]
    metrics: Dict[str, float]


class CurriculumHeytingLosses(nn.Module):
    def __init__(
        self,
        *,
        bounds: LearnableBounds,
        nucleus: LearnableThresholdNucleus,
        heyting: ParametricHeytingOps,
        schedule: str,
        stage1_epochs: int,
        warmup_epochs: int,
        lambdas: Dict[str, float],
        relative_regularity: bool = True,
        min_width: float = 0.5,
        target_internalization: float = 0.8,
        eps_width: float = 1e-6,
    ):
        super().__init__()
        self.bounds = bounds
        self.nucleus = nucleus
        self.heyting = heyting
        self.relative_regularity = bool(relative_regularity)
        self.min_width = float(min_width)
        self.target_internalization = float(target_internalization)
        self.eps_width = float(eps_width)

        self.schedulers = {
            k: CurriculumScheduler(
                target=float(v), schedule=str(schedule), stage1_epochs=int(stage1_epochs), warmup_epochs=int(warmup_epochs)
            )
            for k, v in lambdas.items()
        }

    def lam(self, name: str, epoch: int) -> float:
        if name not in self.schedulers:
            return 0.0
        return float(self.schedulers[name].value(epoch))

    def compute(self, *, epoch: int, z: torch.Tensor, z_raw: torch.Tensor, z_nuc: torch.Tensor) -> CurriculumHeytingOut:
        losses: Dict[str, torch.Tensor] = {}
        metrics: Dict[str, float] = {}

        # Relative regularity.
        gap = (self.heyting.double_neg_soft(z) - z).abs()
        if self.relative_regularity:
            w = self.bounds.width.to(device=z.device, dtype=z.dtype)[None, :]
            reg = (gap / (w + float(self.eps_width))).sum(dim=-1).mean()
        else:
            reg = gap.sum(dim=-1).mean()
        losses["regularity"] = self.lam("regularity", epoch) * reg
        metrics["regularity_raw"] = float(reg.detach().cpu().item())

        # Tightness.
        zmin = z_nuc.amin(dim=0).detach()
        zmax = z_nuc.amax(dim=0).detach()
        lo = self.bounds.lo.to(device=z.device, dtype=z.dtype)
        hi = self.bounds.hi.to(device=z.device, dtype=z.dtype)
        tight = F.relu(lo - zmin).mean() + F.relu(zmax - hi).mean()
        losses["tightness"] = self.lam("tightness", epoch) * tight
        metrics["tightness_raw"] = float(tight.detach().cpu().item())

        # Decoupled internalization (nucleus only).
        intern = self.nucleus.internalization_rate(z_raw)
        internal = F.relu(float(self.target_internalization) - intern) ** 2
        losses["internalization"] = self.lam("internalization", epoch) * internal
        metrics["internalization_rate"] = float(intern.detach().cpu().item())

        # Threshold utility: encourage non-trivial usage.
        thr = self.nucleus.threshold.to(device=z_raw.device, dtype=z_raw.dtype)[None, :]
        changed = (z_raw < thr).float().mean()
        util = (F.relu(0.1 - changed) ** 2) + (F.relu(changed - 0.4) ** 2)
        losses["threshold_utility"] = self.lam("threshold_utility", epoch) * util
        metrics["threshold_changed_frac"] = float(changed.detach().cpu().item())

        # Width stability.
        wstab = F.relu(float(self.min_width) - self.bounds.width).mean()
        losses["width_stability"] = self.lam("width_stability", epoch) * wstab
        metrics["width_stability_raw"] = float(wstab.detach().cpu().item())

        total = sum(losses.values()) if losses else torch.zeros((), device=z.device)
        with torch.no_grad():
            metrics.update(self.bounds.stats())
            metrics.update(self.nucleus.stats())
            metrics["boundary_violation_max"] = float(self.heyting.boundary_violation_hard(z).detach().cpu().item())
        return CurriculumHeytingOut(total=total, losses=losses, metrics=metrics)

