from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, depth: int = 3):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        layers: list[nn.Module] = []
        if depth == 1:
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            for _ in range(depth - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.Tanh())
            layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StableGenerator(nn.Module):
    """
    Stable continuous-time generator:

      G = -(AᵀA) + (B - Bᵀ)

    so Re(λ(G)) ≤ 0.
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.A = nn.Parameter(0.01 * torch.randn(latent_dim, latent_dim))
        self.B = nn.Parameter(0.01 * torch.randn(latent_dim, latent_dim))

    def matrix(self) -> torch.Tensor:
        dissip = -(self.A.transpose(0, 1) @ self.A)
        skew = self.B - self.B.transpose(0, 1)
        return dissip + skew

    def step(self, z: torch.Tensor, *, dt: float) -> torch.Tensor:
        g = self.matrix()
        m = torch.matrix_exp(g * float(dt))
        return z @ m


@dataclass(frozen=True)
class KoopmanAEConfig:
    in_dim: int
    state_dim: int
    latent_dim: int
    hidden_dim: int = 128
    depth: int = 3
    dt: float = 0.02


class KoopmanAE(nn.Module):
    def __init__(self, cfg: KoopmanAEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = MLP(cfg.in_dim, cfg.hidden_dim, cfg.latent_dim, depth=cfg.depth)
        self.decoder = MLP(cfg.latent_dim, cfg.hidden_dim, cfg.state_dim, depth=cfg.depth)
        self.generator = StableGenerator(cfg.latent_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, x_t: torch.Tensor, x_t1: torch.Tensor, *, x_t_embed: Optional[torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        x_in = x_t if x_t_embed is None else x_t_embed
        z_t = self.encode(x_in)
        x_hat_t = self.decode(z_t)
        z_t1_pred = self.generator.step(z_t, dt=float(self.cfg.dt))
        x_hat_t1 = self.decode(z_t1_pred)
        return {
            "x_in": x_in,
            "z_raw": z_t,
            "z": z_t,
            "x_hat_t": x_hat_t,
            "z_t1_pred": z_t1_pred,
            "x_hat_t1": x_hat_t1,
        }

