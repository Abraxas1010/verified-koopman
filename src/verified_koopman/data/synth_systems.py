from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, Tuple

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class SynthConfig:
    system: str
    n_traj: int = 256
    time: int = 201
    dt: float = 0.02
    seed: int = 0
    noise_std: float = 0.0


_SYSTEM_DT: Dict[str, float] = {
    "toy2d": 0.02,
    "vdp": 0.02,
    "lorenz": 0.01,
    "duffing": 0.02,
}


def system_dt(system: str) -> float:
    key = str(system).lower().strip()
    if key not in _SYSTEM_DT:
        raise ValueError(f"unknown system={system!r}")
    return float(_SYSTEM_DT[key])


def _rk4_step(rhs: Callable[[Array], Array], state: Array, dt: float) -> Array:
    k1 = rhs(state)
    k2 = rhs(state + 0.5 * dt * k1)
    k3 = rhs(state + 0.5 * dt * k2)
    k4 = rhs(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _toy2d_step(state: Array, dt: float) -> Array:
    angle = 1.4 * dt
    decay = np.exp(-0.35 * dt)
    c = np.cos(angle)
    s = np.sin(angle)
    a = decay * np.array([[c, -s], [s, c]], dtype=np.float64)
    return state @ a.T


def _vdp_rhs(state: Array) -> Array:
    x = state[..., 0]
    y = state[..., 1]
    mu = 1.2
    dx = y
    dy = mu * (1.0 - x**2) * y - x
    return np.stack([dx, dy], axis=-1)


def _lorenz_rhs(state: Array) -> Array:
    x = state[..., 0]
    y = state[..., 1]
    z = state[..., 2]
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.stack([dx, dy, dz], axis=-1)


def _duffing_rhs(state: Array) -> Array:
    x = state[..., 0]
    y = state[..., 1]
    delta = 0.2
    alpha = -1.0
    beta = 1.0
    dx = y
    dy = -delta * y - alpha * x - beta * x**3
    return np.stack([dx, dy], axis=-1)


def _sample_initial_conditions(system: str, n_traj: int, rng: np.random.Generator) -> Array:
    if system == "toy2d":
        return rng.uniform(low=-1.0, high=1.0, size=(n_traj, 2))
    if system == "vdp":
        return rng.uniform(low=-2.0, high=2.0, size=(n_traj, 2))
    if system == "lorenz":
        centers = np.array([[-8.0, 8.0, 27.0], [8.0, -8.0, 27.0]], dtype=np.float64)
        base = centers[rng.integers(0, len(centers), size=n_traj)]
        return base + rng.normal(scale=1.5, size=(n_traj, 3))
    if system == "duffing":
        return rng.uniform(low=-1.5, high=1.5, size=(n_traj, 2))
    raise ValueError(f"unknown system={system!r}")


def _integrate(system: str, init: Array, dt: float, time: int, noise_std: float, rng: np.random.Generator) -> Array:
    traj = np.zeros((init.shape[0], int(time), init.shape[1]), dtype=np.float64)
    traj[:, 0, :] = init

    for t in range(1, int(time)):
        prev = traj[:, t - 1, :]
        if system == "toy2d":
            nxt = _toy2d_step(prev, dt)
        elif system == "vdp":
            nxt = _rk4_step(_vdp_rhs, prev, dt)
        elif system == "lorenz":
            nxt = _rk4_step(_lorenz_rhs, prev, dt)
        elif system == "duffing":
            nxt = _rk4_step(_duffing_rhs, prev, dt)
        else:
            raise ValueError(f"unknown system={system!r}")
        if noise_std > 0.0:
            nxt = nxt + rng.normal(scale=float(noise_std), size=nxt.shape)
        traj[:, t, :] = nxt
    return traj


def generate(cfg: SynthConfig) -> Tuple[Array, Dict[str, float | int | str]]:
    system = str(cfg.system).lower().strip()
    dt = float(cfg.dt if cfg.dt is not None else system_dt(system))
    rng = np.random.default_rng(int(cfg.seed))
    init = _sample_initial_conditions(system, int(cfg.n_traj), rng)
    traj = _integrate(system, init, dt, int(cfg.time), float(cfg.noise_std), rng)
    meta: Dict[str, float | int | str] = {
        "system": system,
        "n_traj": int(cfg.n_traj),
        "time": int(cfg.time),
        "dt": float(dt),
        "seed": int(cfg.seed),
        "noise_std": float(cfg.noise_std),
        "state_dim": int(traj.shape[-1]),
    }
    return traj.astype(np.float32), meta


def generate_system_data(
    system_name: str,
    *,
    n_traj: int,
    time: int,
    dt: float | None = None,
    seed: int = 0,
    train_frac: float = 0.8,
    noise_std: float = 0.0,
) -> Tuple[Tuple[Array, Array], Tuple[Array, Array], float, Dict[str, float | int | str]]:
    system = str(system_name).lower().strip()
    cfg = SynthConfig(
        system=system,
        n_traj=int(n_traj),
        time=int(time),
        dt=float(system_dt(system) if dt is None else dt),
        seed=int(seed),
        noise_std=float(noise_std),
    )
    traj, meta = generate(cfg)

    split_idx = min(max(1, int(round(float(train_frac) * traj.shape[0]))), traj.shape[0] - 1)
    train_traj = traj[:split_idx]
    test_traj = traj[split_idx:]

    train = (
        train_traj[:, :-1, :].reshape(-1, traj.shape[-1]).astype(np.float32),
        train_traj[:, 1:, :].reshape(-1, traj.shape[-1]).astype(np.float32),
    )
    test = (
        test_traj[:, :-1, :].reshape(-1, traj.shape[-1]).astype(np.float32),
        test_traj[:, 1:, :].reshape(-1, traj.shape[-1]).astype(np.float32),
    )

    meta_out: Dict[str, float | int | str] = dict(meta)
    meta_out["train_frac"] = float(train_frac)
    meta_out["train_pairs"] = int(train[0].shape[0])
    meta_out["test_pairs"] = int(test[0].shape[0])
    meta_out["config"] = str(asdict(cfg))
    return train, test, float(cfg.dt), meta_out
