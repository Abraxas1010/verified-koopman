#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

import verified_koopman as vk
from verified_koopman.models.nucleus_bottleneck import nucleus_stats
from verified_koopman.utils.training import TrainConfig, train_model
from verified_koopman.verification.safedmd import (
    active_dictionary_mask,
    compute_nucleus_aware_error_bound,
    synthesize_controller,
    verify_certificate,
)


def _load_best(model: torch.nn.Module, checkpoint: Path, *, device: str) -> torch.nn.Module:
    ckpt = torch.load(str(checkpoint), map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(torch.device(device))
    model.eval()
    return model


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def _train_cfg(epochs: int) -> TrainConfig:
    return TrainConfig(
        epochs=int(epochs),
        batch_size=1024,
        lr=3e-4,
        weight_decay=0.0,
        recon_weight=1.0,
        pred_weight=1.0,
        grad_clip=1.0,
        log_every=1,
    )


def _generator_consistency(model: torch.nn.Module, *, device: str) -> Dict[str, float]:
    latent_dim = int(model.cfg.latent_dim)
    dt = float(model.cfg.dt)
    z = torch.randn(32, latent_dim, device=device)
    row_next = model.generator.step(z, dt=dt)
    k_hat_col = torch.matrix_exp(model.generator.matrix() * dt).transpose(0, 1)
    col_next = (k_hat_col @ z.transpose(0, 1)).transpose(0, 1)
    err = float((row_next - col_next).abs().amax().detach().cpu().item())
    return {"max_abs_error": err}


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    (train, test, dt, meta) = vk.generate_system_data(
        args.system,
        n_traj=int(args.n_traj),
        time=int(args.time),
        dt=float(args.dt) if args.dt is not None else None,
        seed=int(args.seed),
        train_frac=float(args.train_frac),
    )
    train_x, train_y = train
    state_dim = int(train_x.shape[-1])

    nba = vk.NucleusBottleneckAE(
        state_dim=state_dim,
        latent_dim=int(args.latent_dim),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        dt=float(dt),
        nucleus_type="relu",
    )
    baseline = vk.KoopmanAE(
        vk.KoopmanAEConfig(
            in_dim=state_dim,
            state_dim=state_dim,
            latent_dim=int(args.latent_dim),
            hidden_dim=int(args.hidden_dim),
            depth=int(args.depth),
            dt=float(dt),
        )
    )

    output_root = Path(args.output).resolve() / args.system
    nba_out = train_model(
        model=nba,
        train=train,
        test=test,
        dt=dt,
        output_dir=output_root / "nba",
        cfg=_train_cfg(args.epochs),
        device=args.device,
        seed=args.seed,
        extra_meta={"system": args.system, "model_type": "nba", **meta},
    )
    baseline_out = train_model(
        model=baseline,
        train=train,
        test=test,
        dt=dt,
        output_dir=output_root / "baseline",
        cfg=_train_cfg(args.epochs),
        device=args.device,
        seed=args.seed,
        extra_meta={"system": args.system, "model_type": "baseline", **meta},
    )

    nba = _load_best(nba, nba_out.best_path, device=args.device)
    baseline = _load_best(baseline, baseline_out.best_path, device=args.device)

    train_x_t = torch.from_numpy(np.asarray(train_x)).to(dtype=torch.float32)
    train_y_t = torch.from_numpy(np.asarray(train_y)).to(dtype=torch.float32)
    regularization = float(args.regularization)

    nba_bound = compute_nucleus_aware_error_bound(nba, train_x_t, train_y_t, regularization=regularization, device=args.device)
    baseline_bound = compute_nucleus_aware_error_bound(
        baseline, train_x_t, train_y_t, regularization=regularization, device=args.device
    )

    if train_x.shape != train_y.shape:
        raise RuntimeError("training pairs must align")

    with torch.no_grad():
        psi_x_nba = nba.encode(train_x_t.to(dtype=torch.float32, device=args.device)).detach().cpu().numpy().astype(np.float64)
        psi_y_nba = nba.encode(train_y_t.to(dtype=torch.float32, device=args.device)).detach().cpu().numpy().astype(np.float64)
    active_mask = active_dictionary_mask(psi_x_nba, psi_y_nba) if nba_bound.nucleus_applied else np.ones(
        int(args.latent_dim), dtype=bool
    )
    active_idx = np.flatnonzero(active_mask)
    baseline_active_dims = int(baseline_bound.dictionary_size)
    k_hat_nba_full = torch.matrix_exp(nba.generator.matrix() * float(nba.cfg.dt)).detach().cpu().numpy().T
    k_hat_nba = k_hat_nba_full[np.ix_(active_idx, active_idx)]
    b_input = np.eye(len(active_idx), dtype=np.float64)
    controller = synthesize_controller(
        k_hat_nba,
        b_input,
        nba_bound,
        input_dim=len(active_idx),
        decay_target=float(args.decay_target),
        region_radius=float(args.region_radius),
        verbose=bool(args.verbose),
    )

    controller_json: Dict[str, Any]
    if controller is None:
        controller_json = {
            "synthesized": False,
            "sdp_status": "infeasible",
            "decay_rate": float(args.decay_target),
            "region_radius": float(args.region_radius),
        }
    else:
        verification = verify_certificate(controller, k_hat_nba, b_input, nba_bound, n_samples=int(args.verify_samples))
        controller_json = {
            "synthesized": True,
            **controller.to_jsonable(),
            "numerical_verification": verification,
        }

    with torch.no_grad():
        encoded_raw = nba.encode_raw(train_x_t[:1024].to(dtype=torch.float32, device=args.device))
        stats = nucleus_stats(nba.nucleus, encoded_raw)

    improvement = float(baseline_bound.spectral_norm / max(nba_bound.spectral_norm, 1e-12))
    results: Dict[str, Any] = {
        "system": str(args.system),
        "seed": int(args.seed),
        "regularization": regularization,
        "latent_dim": int(args.latent_dim),
        "hidden_dim": int(args.hidden_dim),
        "depth": int(args.depth),
        "epochs": int(args.epochs),
        "data": meta,
        "nba_error_bound": nba_bound.to_jsonable(),
        "baseline_error_bound": baseline_bound.to_jsonable(),
        "improvement_factor": improvement,
        "condition_number_ratio": float(
            baseline_bound.condition_number / max(nba_bound.condition_number, 1e-12)
        ),
        "generator_consistency": _generator_consistency(nba, device=args.device),
        "active_dictionary_size": int(len(active_idx)),
        "active_dimensions_nba": int(len(active_idx)),
        "active_dimensions_baseline": baseline_active_dims,
        "comparison_context": (
            "Improvement factor compares the nucleus-restricted active NBA dictionary "
            "against the baseline full lifted dictionary."
        ),
        "nucleus_stats": stats.__dict__,
        "controller": controller_json,
        "checkpoints": {
            "nba_best": str(nba_out.best_path),
            "baseline_best": str(baseline_out.best_path),
        },
    }
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="toy2d", choices=["toy2d", "vdp", "lorenz", "duffing"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--latent-dim", type=int, default=8)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--n-traj", type=int, default=256)
    ap.add_argument("--time", type=int, default=201)
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--regularization", type=float, default=1e-6)
    ap.add_argument("--decay-target", type=float, default=0.95)
    ap.add_argument("--region-radius", type=float, default=1.0)
    ap.add_argument("--verify-samples", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", default="outputs/safedmd_runs")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    results = run_experiment(args)
    out_path = Path(args.output).resolve() / args.system / "safedmd_results.json"
    _write_json(out_path, results)
    print(f"OK: wrote {out_path}")
    print(
        "Improvement factor: "
        f"{results['improvement_factor']:.3f}x "
        f"({results['active_dimensions_nba']} active NBA dims vs "
        f"{results['active_dimensions_baseline']} baseline dims)"
    )


if __name__ == "__main__":
    main()
