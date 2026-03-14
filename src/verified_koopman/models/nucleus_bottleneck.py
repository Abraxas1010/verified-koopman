from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from verified_koopman.models.koopman_ae import MLP, StableGenerator
from verified_koopman.models.learnable_heyting import LearnableBounds, LearnableThresholdNucleus, ParametricHeytingOps


@dataclass(frozen=True)
class NucleusStats:
    mean_delta_l1: float
    max_delta: float
    frac_fixed: float


def nucleus_stats(nucleus: nn.Module, x: torch.Tensor, *, eps: float = 1e-6) -> NucleusStats:
    with torch.no_grad():
        y = nucleus(x)
        delta = y - x
        mean_delta_l1 = float(delta.abs().mean().item())
        max_delta = float(delta.abs().amax().item()) if delta.numel() else 0.0
        fixed = delta.abs().amax(dim=-1) <= float(eps)
        frac_fixed = float(fixed.float().mean().item()) if fixed.numel() else 1.0
    return NucleusStats(mean_delta_l1=mean_delta_l1, max_delta=max_delta, frac_fixed=frac_fixed)


class ReLUNucleus(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=0.0)


class ThresholdNucleus(nn.Module):
    def __init__(self, threshold: float):
        super().__init__()
        self.register_buffer("_a", torch.tensor(float(threshold)))

    @property
    def a(self) -> float:
        return float(self._a.item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.maximum(x, self._a.to(dtype=x.dtype, device=x.device))


@dataclass(frozen=True)
class KoopmanNBAConfig:
    in_dim: int
    state_dim: int
    latent_dim: int
    hidden_dim: int = 128
    depth: int = 3
    dt: float = 0.02
    nucleus_type: str = "relu"  # relu | threshold
    nucleus_threshold: float = 0.0


class KoopmanNBA(nn.Module):
    """
    Nucleus-bottleneck Koopman AE:

      z_raw = E(x)
      z     = R(z_raw)
      x̂_t  = D(z)
      z'    = exp(G·dt) z
      x̂_{t+1} = D(z')
    """

    def __init__(self, cfg: KoopmanNBAConfig, *, nucleus: Optional[nn.Module] = None):
        super().__init__()
        self.cfg = cfg
        self.encoder = MLP(cfg.in_dim, cfg.hidden_dim, cfg.latent_dim, depth=cfg.depth)
        self.decoder = MLP(cfg.latent_dim, cfg.hidden_dim, cfg.state_dim, depth=cfg.depth)
        self.generator = StableGenerator(cfg.latent_dim)
        self.nucleus = nucleus if nucleus is not None else self._make_nucleus(cfg)

    @staticmethod
    def _make_nucleus(cfg: KoopmanNBAConfig) -> nn.Module:
        t = str(cfg.nucleus_type).lower().strip()
        if t == "relu":
            return ReLUNucleus()
        if t == "threshold":
            return ThresholdNucleus(float(cfg.nucleus_threshold))
        raise ValueError(f"unknown nucleus_type={cfg.nucleus_type!r}")

    def encode_raw(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.nucleus(self.encode_raw(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, x_t: torch.Tensor, x_t1: torch.Tensor, *, x_t_embed: Optional[torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        x_in = x_t if x_t_embed is None else x_t_embed
        z_raw = self.encode_raw(x_in)
        z = self.nucleus(z_raw)
        x_hat_t = self.decode(z)
        z_t1_pred = self.generator.step(z, dt=float(self.cfg.dt))
        x_hat_t1 = self.decode(z_t1_pred)
        return {
            "x_in": x_in,
            "z_raw": z_raw,
            "z": z,
            "x_hat_t": x_hat_t,
            "z_t1_pred": z_t1_pred,
            "x_hat_t1": x_hat_t1,
        }


class NucleusBottleneckAE(KoopmanNBA):
    """
    Convenience wrapper matching the paper-friendly name "NBA" with a simpler constructor.

    For research scripts, `KoopmanNBA` + `KoopmanNBAConfig` are also available.
    """

    def __init__(
        self,
        *,
        state_dim: int,
        latent_dim: int = 8,
        hidden_dim: int = 128,
        depth: int = 3,
        dt: float = 0.02,
        nucleus_type: str = "relu",
        nucleus_threshold: float = 0.0,
    ):
        cfg = KoopmanNBAConfig(
            in_dim=int(state_dim),
            state_dim=int(state_dim),
            latent_dim=int(latent_dim),
            hidden_dim=int(hidden_dim),
            depth=int(depth),
            dt=float(dt),
            nucleus_type=str(nucleus_type),
            nucleus_threshold=float(nucleus_threshold),
        )
        super().__init__(cfg)


class E2EHeytingNBA(nn.Module):
    """
    NBA variant with *learnable* threshold nucleus and *learnable* Heyting bounds.

    This is a minimal end-to-end model used for the curriculum experiments:
      z_raw = E(x)
      z_nuc = max(z_raw, threshold)
      z = clamp(z_nuc, lo, hi)
      x̂_t = D(z)
      z' = exp(G·dt) z
      z'_proj = clamp(max(z', threshold), lo, hi)
      x̂_{t+1} = D(z'_proj)
    """

    def __init__(
        self,
        *,
        in_dim: int,
        state_dim: int,
        latent_dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
        dt: float = 0.02,
        init_threshold: float = 0.0,
        init_lo: float = 0.0,
        init_hi: float = 2.0,
        min_gap: float = 0.1,
        heyting_temperature: float = 0.05,
    ):
        super().__init__()
        self.dt = float(dt)
        self.encoder = MLP(int(in_dim), int(hidden_dim), int(latent_dim), depth=int(depth))
        self.decoder = MLP(int(latent_dim), int(hidden_dim), int(state_dim), depth=int(depth))
        self.generator = StableGenerator(int(latent_dim))

        self.nucleus = LearnableThresholdNucleus(int(latent_dim), init_threshold=float(init_threshold), learnable=True)
        self.bounds = LearnableBounds(
            int(latent_dim), init_lo=float(init_lo), init_hi=float(init_hi), min_gap=float(min_gap), learnable=True
        )
        self.heyting = ParametricHeytingOps(self.bounds, temperature=float(heyting_temperature))

    def encode(self, x_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_raw = self.encoder(x_in)
        z_nuc = self.nucleus(z_raw)
        z = self.heyting.project(z_nuc)
        return z, z_raw, z_nuc

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def evolve(self, z: torch.Tensor) -> torch.Tensor:
        z1 = self.generator.step(z, dt=self.dt)
        z1 = self.nucleus(z1)
        return self.heyting.project(z1)

    def forward(
        self, x_t: torch.Tensor, x_t1: torch.Tensor, *, x_t_embed: Optional[torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        x_in = x_t if x_t_embed is None else x_t_embed
        z, z_raw, z_nuc = self.encode(x_in)
        x_hat_t = self.decode(z)
        z_t1 = self.evolve(z)
        x_hat_t1 = self.decode(z_t1)
        return {
            "x_in": x_in,
            "z_raw": z_raw,
            "z_nuc": z_nuc,
            "z": z,
            "x_hat_t": x_hat_t,
            "z_t1_pred": z_t1,
            "x_hat_t1": x_hat_t1,
        }
