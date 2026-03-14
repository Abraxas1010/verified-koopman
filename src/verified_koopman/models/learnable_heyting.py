from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableBounds(nn.Module):
    """
    Learn per-coordinate bounds `[lo, hi]^k` with constraints by construction:
    - lo = softplus(lo_raw) (so lo ≥ 0)
    - hi = lo + min_gap + softplus(delta_raw) (so hi > lo)
    """

    def __init__(
        self,
        dim: int,
        *,
        init_lo: float = 0.0,
        init_hi: float = 2.0,
        min_gap: float = 0.1,
        learnable: bool = True,
    ):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = int(dim)
        self.min_gap = float(min_gap)

        init_lo_t = torch.full((dim,), float(init_lo), dtype=torch.float32)
        init_hi_t = torch.full((dim,), float(init_hi), dtype=torch.float32)
        init_delta_t = (init_hi_t - init_lo_t - float(min_gap)).clamp(min=0.1)

        lo_raw0 = self._inv_softplus(init_lo_t + 1e-6)
        delta_raw0 = self._inv_softplus(init_delta_t)

        if learnable:
            self.lo_raw = nn.Parameter(lo_raw0)
            self.delta_raw = nn.Parameter(delta_raw0)
        else:
            self.register_buffer("lo_raw", lo_raw0)
            self.register_buffer("delta_raw", delta_raw0)

    @staticmethod
    def _inv_softplus(y: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.expm1(y).clamp(min=1e-12))

    @property
    def lo(self) -> torch.Tensor:
        return F.softplus(self.lo_raw)

    @property
    def hi(self) -> torch.Tensor:
        return self.lo + float(self.min_gap) + F.softplus(self.delta_raw)

    @property
    def width(self) -> torch.Tensor:
        return self.hi - self.lo

    def clamp(self, z: torch.Tensor) -> torch.Tensor:
        lo = self.lo.to(device=z.device, dtype=z.dtype)[None, :]
        hi = self.hi.to(device=z.device, dtype=z.dtype)[None, :]
        return torch.clamp(z, min=lo, max=hi)

    def stats(self) -> Dict[str, float]:
        with torch.no_grad():
            lo = self.lo.detach()
            hi = self.hi.detach()
            w = hi - lo
            return {
                "lo_mean": float(lo.mean().item()),
                "hi_mean": float(hi.mean().item()),
                "width_mean": float(w.mean().item()),
                "lo_min": float(lo.min().item()),
                "hi_max": float(hi.max().item()),
            }


class LearnableThresholdNucleus(nn.Module):
    """
    Per-coordinate threshold nucleus: R(z) = max(z, threshold), with threshold ≥ 0 by construction.
    """

    def __init__(self, dim: int, *, init_threshold: float = 0.0, learnable: bool = True):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = int(dim)

        raw0 = LearnableBounds._inv_softplus(torch.tensor(float(init_threshold) + 1e-6, dtype=torch.float32))
        raw0 = raw0.repeat(self.dim)
        if learnable:
            self.threshold_raw = nn.Parameter(raw0)
        else:
            self.register_buffer("threshold_raw", raw0)

    @property
    def threshold(self) -> torch.Tensor:
        return F.softplus(self.threshold_raw)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        thr = self.threshold.to(device=z.device, dtype=z.dtype)[None, :]
        return torch.maximum(z, thr)

    def internalization_rate(self, z_raw: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
        thr = self.threshold.to(device=z_raw.device, dtype=z_raw.dtype)[None, :]
        return (z_raw + float(eps) >= thr).float().mean()

    def stats(self) -> Dict[str, float]:
        with torch.no_grad():
            t = self.threshold.detach()
            return {"threshold_mean": float(t.mean().item()), "threshold_min": float(t.min().item())}


class ParametricHeytingOps(nn.Module):
    """
    Bounded Heyting ops on `[lo,hi]^k`, matching the Lean definition:

      (a ↣ b)_i = if a_i ≤ b_i then hi_i else b_i
      ¬a = a ↣ lo
    """

    def __init__(self, bounds: LearnableBounds, *, temperature: float = 0.05):
        super().__init__()
        self.bounds = bounds
        self.register_buffer("_temperature", torch.tensor(float(temperature), dtype=torch.float32))

    @property
    def lo(self) -> torch.Tensor:
        return self.bounds.lo

    @property
    def hi(self) -> torch.Tensor:
        return self.bounds.hi

    @property
    def temperature(self) -> torch.Tensor:
        return self._temperature

    def project(self, z: torch.Tensor) -> torch.Tensor:
        return self.bounds.clamp(z)

    def meet(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.minimum(a, b)

    def join(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.maximum(a, b)

    def bot_like(self, x: torch.Tensor) -> torch.Tensor:
        lo = self.lo.to(device=x.device, dtype=x.dtype)[None, :]
        return lo.expand_as(x)

    def himp_hard(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a2, b2 = torch.broadcast_tensors(a, b)
        hi = self.hi.to(device=a2.device, dtype=a2.dtype)[None, :].expand_as(a2)
        return torch.where(a2 <= b2, hi, b2)

    def hnot_hard(self, a: torch.Tensor) -> torch.Tensor:
        return self.himp_hard(a, self.bot_like(a))

    def double_neg_hard(self, a: torch.Tensor) -> torch.Tensor:
        return self.hnot_hard(self.hnot_hard(a))

    def himp_soft(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a2, b2 = torch.broadcast_tensors(a, b)
        hi = self.hi.to(device=a2.device, dtype=a2.dtype)[None, :].expand_as(a2)
        t = self.temperature.to(device=a2.device, dtype=a2.dtype).clamp(min=1e-4)
        w = torch.sigmoid((b2 - a2) / t)
        return w * hi + (1.0 - w) * b2

    def hnot_soft(self, a: torch.Tensor) -> torch.Tensor:
        return self.himp_soft(a, self.bot_like(a))

    def double_neg_soft(self, a: torch.Tensor) -> torch.Tensor:
        return self.hnot_soft(self.hnot_soft(a))

    def boundary_violation_hard(self, z: torch.Tensor) -> torch.Tensor:
        boundary = self.meet(z, self.hnot_hard(z))
        bot = self.bot_like(boundary)
        return (boundary - bot).abs().amax()
