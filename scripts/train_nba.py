#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

import verified_koopman as vk
from verified_koopman.losses.curriculum import CurriculumHeytingLosses
from verified_koopman.models.learnable_heyting import ParametricHeytingOps
from verified_koopman.utils.training import TrainConfig, train_model


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    with path.open("r", encoding="utf-8") as f:
        return dict(yaml.safe_load(f) or {})


def _merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def build_model(
    *,
    model_type: str,
    state_dim: int,
    cfg: Dict[str, Any],
) -> Tuple[object, Dict[str, Any], object | None]:
    mt = str(model_type).lower().strip()

    if mt == "baseline":
        mcfg = cfg.get("model", {})
        model = vk.KoopmanAE(
            vk.KoopmanAEConfig(
                in_dim=int(state_dim),
                state_dim=int(state_dim),
                latent_dim=int(mcfg.get("latent_dim", 8)),
                hidden_dim=int(mcfg.get("hidden_dim", 128)),
                depth=int(mcfg.get("depth", 3)),
                dt=float(mcfg.get("dt", cfg.get("data", {}).get("dt", 0.02))),
            )
        )
        ctor = {"model_type": "baseline", "state_dim": int(state_dim), "model": mcfg}
        return model, ctor, None

    if mt in {"nba", "koopman_nba"}:
        mcfg = cfg.get("model", {})
        model = vk.NucleusBottleneckAE(
            state_dim=int(state_dim),
            latent_dim=int(mcfg.get("latent_dim", 8)),
            hidden_dim=int(mcfg.get("hidden_dim", 128)),
            depth=int(mcfg.get("depth", 3)),
            dt=float(mcfg.get("dt", cfg.get("data", {}).get("dt", 0.02))),
            nucleus_type=str(mcfg.get("nucleus_type", "relu")),
            nucleus_threshold=float(mcfg.get("nucleus_threshold", 0.0)),
        )
        ctor = {"model_type": "nba", "state_dim": int(state_dim), "model": mcfg}
        return model, ctor, None

    if mt in {"e2e", "e2e_heyting"}:
        ecfg = cfg.get("e2e", {})
        model = vk.E2EHeytingNBA(
            in_dim=int(state_dim),
            state_dim=int(state_dim),
            latent_dim=int(ecfg.get("latent_dim", cfg.get("model", {}).get("latent_dim", 8))),
            hidden_dim=int(ecfg.get("hidden_dim", cfg.get("model", {}).get("hidden_dim", 128))),
            depth=int(ecfg.get("depth", cfg.get("model", {}).get("depth", 3))),
            dt=float(ecfg.get("dt", cfg.get("data", {}).get("dt", 0.02))),
            init_threshold=float(ecfg.get("init_threshold", 0.0)),
            init_lo=float(ecfg.get("init_lo", 0.0)),
            init_hi=float(ecfg.get("init_hi", 2.0)),
            min_gap=float(ecfg.get("min_gap", 0.1)),
            heyting_temperature=float(ecfg.get("heyting_temperature", 0.05)),
        )
        ctor = {"model_type": "e2e", "state_dim": int(state_dim), "e2e": ecfg}

        ccfg = cfg.get("curriculum", {})
        if bool(ccfg.get("enabled", True)):
            schedule = str(ccfg.get("schedule", "staged"))
            stage1_epochs = int(ccfg.get("stage1_epochs", 50))
            warmup_epochs = int(ccfg.get("warmup_epochs", 100))
            lambdas = dict(ccfg.get("lambdas", {}))
            if not lambdas:
                lambdas = {
                    "regularity": 1.0,
                    "tightness": 0.2,
                    "internalization": 0.5,
                    "threshold_utility": 0.1,
                    "width_stability": 0.1,
                }
            curriculum = CurriculumHeytingLosses(
                bounds=model.bounds,
                nucleus=model.nucleus,
                heyting=model.heyting,
                schedule=schedule,
                stage1_epochs=stage1_epochs,
                warmup_epochs=warmup_epochs,
                lambdas={k: float(v) for k, v in lambdas.items()},
                relative_regularity=bool(ccfg.get("relative_regularity", True)),
            )
        else:
            curriculum = None

        return model, ctor, curriculum

    raise ValueError(f"unknown model_type={model_type!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="toy2d", choices=["toy2d", "vdp", "lorenz", "duffing"])
    ap.add_argument("--model", default="nba", choices=["baseline", "nba", "e2e"])
    ap.add_argument("--config", default=None, help="YAML config path (overrides configs/default.yaml)")
    ap.add_argument("--epochs", type=int, default=None, help="Override train.epochs")
    ap.add_argument("--output", default="outputs", help="Output root directory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = _load_yaml(root / "configs" / "default.yaml")
    cfg = _merge(cfg, _load_yaml(root / "configs" / f"{args.system}.yaml"))
    if args.config:
        cfg = _merge(cfg, _load_yaml(Path(args.config)))
    if args.epochs is not None:
        cfg = _merge(cfg, {"train": {"epochs": int(args.epochs)}})

    data_cfg = cfg.get("data", {})
    dt_arg = float(data_cfg["dt"]) if "dt" in data_cfg else None
    (train, test, dt, meta) = vk.generate_system_data(
        args.system,
        n_traj=int(data_cfg.get("n_traj", 256)),
        time=int(data_cfg.get("time", 201)),
        dt=dt_arg,
        seed=int(data_cfg.get("seed", args.seed)),
        train_frac=float(data_cfg.get("train_frac", 0.8)),
    )
    state_dim = int(train[0].shape[-1])

    model, ctor, curriculum = build_model(model_type=args.model, state_dim=state_dim, cfg=cfg)

    tcfg = cfg.get("train", {})
    train_cfg = TrainConfig(
        epochs=int(tcfg.get("epochs", 200)),
        batch_size=int(tcfg.get("batch_size", 1024)),
        lr=float(tcfg.get("lr", 3e-4)),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
        recon_weight=float(tcfg.get("recon_weight", 1.0)),
        pred_weight=float(tcfg.get("pred_weight", 1.0)),
        grad_clip=float(tcfg.get("grad_clip", 1.0)),
        log_every=int(tcfg.get("log_every", 1)),
    )

    out_dir = Path(args.output) / f"{args.system}_{args.model}_{args.seed}"
    out = train_model(
        model=model,  # type: ignore[arg-type]
        train=train,
        test=test,
        dt=dt,
        output_dir=out_dir,
        cfg=train_cfg,
        device=args.device,
        seed=args.seed,
        curriculum=curriculum,
        extra_meta={"system": str(args.system), "model_ctor": ctor, **meta},
    )

    (out_dir / "result.json").write_text(
        json.dumps(
            {"best_val_loss": float(out.best_val_loss), "best_path": str(out.best_path), "last_path": str(out.last_path)},
            indent=2,
        )
    )
    print(f"OK: best_val_loss={out.best_val_loss:.6g} ({out.best_path})")


if __name__ == "__main__":
    main()
