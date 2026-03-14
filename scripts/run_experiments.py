#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

import verified_koopman as vk
from verified_koopman.analysis.lyapunov import lyapunov_certificate_from_generator
from verified_koopman.models.nucleus_bottleneck import E2EHeytingNBA, KoopmanNBA, NucleusBottleneckAE, nucleus_stats
from verified_koopman.models.learnable_heyting import ParametricHeytingOps
from verified_koopman.losses.curriculum import CurriculumHeytingLosses
from verified_koopman.utils.training import TrainConfig, train_model


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def _quick_cfg(*, epochs: int) -> TrainConfig:
    return TrainConfig(epochs=int(epochs), batch_size=1024, lr=3e-4, recon_weight=1.0, pred_weight=1.0, log_every=1)


def run_capability(root: Path, *, systems: List[str], epochs: int, seed: int, device: str) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for sys in systems:
        (train, test, dt, meta) = vk.generate_system_data(sys, n_traj=128, time=201, seed=seed)
        state_dim = int(train[0].shape[-1])

        for mt, model in [
            ("baseline", vk.KoopmanAE(vk.KoopmanAEConfig(in_dim=state_dim, state_dim=state_dim, latent_dim=8, dt=dt))),
            ("nba", NucleusBottleneckAE(state_dim=state_dim, latent_dim=8, dt=dt, nucleus_type="relu")),
        ]:
            out_dir = root / "capability" / f"{sys}_{mt}"
            out = train_model(
                model=model,  # type: ignore[arg-type]
                train=train,
                test=test,
                dt=dt,
                output_dir=out_dir,
                cfg=_quick_cfg(epochs=epochs),
                device=device,
                seed=seed,
                extra_meta={"experiment": "capability", "system": sys, "model_type": mt, **meta},
            )
            rows.append({"system": sys, "model": mt, "best_val_loss": float(out.best_val_loss), "out_dir": str(out.output_dir)})
    return {"experiment": "capability", "rows": rows}


def run_lyapunov(root: Path, *, system: str, epochs: int, seed: int, device: str) -> Dict[str, Any]:
    (train, test, dt, meta) = vk.generate_system_data(system, n_traj=256, time=201, seed=seed)
    state_dim = int(train[0].shape[-1])
    model = NucleusBottleneckAE(state_dim=state_dim, latent_dim=8, dt=dt, nucleus_type="relu")

    out_dir = root / "lyapunov" / system
    out = train_model(
        model=model,  # type: ignore[arg-type]
        train=train,
        test=test,
        dt=dt,
        output_dir=out_dir,
        cfg=_quick_cfg(epochs=epochs),
        device=device,
        seed=seed,
        extra_meta={"experiment": "lyapunov", "system": system, **meta},
    )

    G = model.generator.matrix().detach().cpu().numpy()
    cert = lyapunov_certificate_from_generator(G)
    return {
        "experiment": "lyapunov",
        "system": system,
        "best_val_loss": float(out.best_val_loss),
        "lyapunov": cert.to_jsonable(),
        "out_dir": str(out_dir),
    }


def run_heyting(root: Path, *, system: str, epochs: int, seed: int, device: str) -> Dict[str, Any]:
    (train, test, dt, meta) = vk.generate_system_data(system, n_traj=256, time=201, seed=seed)
    state_dim = int(train[0].shape[-1])
    model = E2EHeytingNBA(in_dim=state_dim, state_dim=state_dim, latent_dim=8, dt=dt, init_hi=2.0)
    curriculum = None  # keep this experiment minimal; see `curriculum` for scheduled losses.

    out_dir = root / "heyting" / system
    out = train_model(
        model=model,  # type: ignore[arg-type]
        train=train,
        test=test,
        dt=dt,
        output_dir=out_dir,
        cfg=_quick_cfg(epochs=epochs),
        device=device,
        seed=seed,
        curriculum=curriculum,
        extra_meta={"experiment": "heyting", "system": system, **meta},
    )

    device_t = next(model.parameters()).device
    x = torch.from_numpy(np.asarray(test[0][:1024])).to(dtype=torch.float32, device=device_t)
    z, z_raw, z_nuc = model.encode(x)
    z_proj = model.heyting.project(z)
    hs = nucleus_stats(model.nucleus, z_raw)
    return {
        "experiment": "heyting",
        "system": system,
        "best_val_loss": float(out.best_val_loss),
        "nucleus_stats": hs.__dict__,
        "bounds": model.bounds.stats(),
        "threshold": model.nucleus.stats(),
        "out_dir": str(out_dir),
    }


def run_curriculum(root: Path, *, system: str, epochs: int, seed: int, device: str) -> Dict[str, Any]:
    (train, test, dt, meta) = vk.generate_system_data(system, n_traj=256, time=201, seed=seed)
    state_dim = int(train[0].shape[-1])
    model = E2EHeytingNBA(in_dim=state_dim, state_dim=state_dim, latent_dim=8, dt=dt, init_hi=2.0)
    curriculum = CurriculumHeytingLosses(
        bounds=model.bounds,
        nucleus=model.nucleus,
        heyting=model.heyting,
        schedule="staged",
        stage1_epochs=max(1, int(epochs) // 4),
        warmup_epochs=max(1, int(epochs) // 2),
        lambdas={
            "regularity": 1.0,
            "tightness": 0.2,
            "internalization": 0.5,
            "threshold_utility": 0.1,
            "width_stability": 0.1,
        },
        relative_regularity=True,
    )

    out_dir = root / "curriculum" / system
    out = train_model(
        model=model,  # type: ignore[arg-type]
        train=train,
        test=test,
        dt=dt,
        output_dir=out_dir,
        cfg=_quick_cfg(epochs=epochs),
        device=device,
        seed=seed,
        curriculum=curriculum,
        extra_meta={"experiment": "curriculum", "system": system, **meta},
    )

    return {
        "experiment": "curriculum",
        "system": system,
        "best_val_loss": float(out.best_val_loss),
        "bounds": model.bounds.stats(),
        "threshold": model.nucleus.stats(),
        "out_dir": str(out_dir),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, choices=["capability", "lyapunov", "heyting", "curriculum"])
    ap.add_argument("--output", default="outputs/experiments")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=20, help="Quick-run epochs for smoke reproduction")
    args = ap.parse_args()

    out_root = Path(args.output).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if args.experiment == "capability":
        res = run_capability(out_root, systems=["toy2d", "vdp"], epochs=args.epochs, seed=args.seed, device=args.device)
    elif args.experiment == "lyapunov":
        res = run_lyapunov(out_root, system="toy2d", epochs=args.epochs, seed=args.seed, device=args.device)
    elif args.experiment == "heyting":
        res = run_heyting(out_root, system="toy2d", epochs=args.epochs, seed=args.seed, device=args.device)
    else:
        res = run_curriculum(out_root, system="toy2d", epochs=args.epochs, seed=args.seed, device=args.device)

    out_path = out_root / f"{args.experiment}_result.json"
    _write_json(out_path, res)
    print(f"OK: wrote {out_path}")


if __name__ == "__main__":
    main()
