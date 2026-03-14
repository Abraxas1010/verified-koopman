from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from verified_koopman.losses.curriculum import CurriculumHeytingLosses


def _now_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def set_seed(seed: int) -> None:
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def auto_device(device: str) -> torch.device:
    d = str(device).lower().strip()
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 200
    batch_size: int = 1024
    lr: float = 3e-4
    weight_decay: float = 0.0
    recon_weight: float = 1.0
    pred_weight: float = 1.0
    grad_clip: float = 1.0
    log_every: int = 1


@dataclass(frozen=True)
class TrainOut:
    output_dir: Path
    best_path: Path
    last_path: Path
    best_val_loss: float


def _to_tensor(x: np.ndarray, *, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x)).to(device=device, dtype=torch.float32)


def _make_loaders(
    train: Tuple[np.ndarray, np.ndarray],
    test: Tuple[np.ndarray, np.ndarray],
    *,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    x_t, x_t1 = train
    tx_t, tx_t1 = test
    ds_tr = TensorDataset(torch.from_numpy(x_t), torch.from_numpy(x_t1))
    ds_te = TensorDataset(torch.from_numpy(tx_t), torch.from_numpy(tx_t1))
    dl_tr = DataLoader(ds_tr, batch_size=int(batch_size), shuffle=True, drop_last=False)
    dl_te = DataLoader(ds_te, batch_size=int(batch_size), shuffle=False, drop_last=False)
    return dl_tr, dl_te


def _batch_to_device(batch: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x_t, x_t1 = batch
    return x_t.to(device=device, dtype=torch.float32), x_t1.to(device=device, dtype=torch.float32)


def _model_loss(
    out: Dict[str, torch.Tensor],
    x_t: torch.Tensor,
    x_t1: torch.Tensor,
    *,
    recon_weight: float,
    pred_weight: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    recon = F.mse_loss(out["x_hat_t"], x_t)
    pred = F.mse_loss(out["x_hat_t1"], x_t1)
    total = float(recon_weight) * recon + float(pred_weight) * pred
    return total, {"recon_mse": float(recon.detach().cpu().item()), "pred_mse": float(pred.detach().cpu().item())}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dl: DataLoader,
    *,
    device: torch.device,
    cfg: TrainConfig,
    curriculum: Optional[CurriculumHeytingLosses] = None,
    epoch: int = 0,
) -> Dict[str, float]:
    model.eval()
    total = 0.0
    n = 0
    for batch in dl:
        x_t, x_t1 = _batch_to_device(batch, device)
        out = model(x_t, x_t1)
        loss, metrics = _model_loss(out, x_t, x_t1, recon_weight=cfg.recon_weight, pred_weight=cfg.pred_weight)

        if curriculum is not None:
            h = curriculum.compute(epoch=epoch, z=out["z"], z_raw=out["z_raw"], z_nuc=out.get("z_nuc", out["z_raw"]))
            loss = loss + h.total
            metrics.update({f"h_{k}": float(v.detach().cpu().item()) for k, v in h.losses.items()})

        total += float(loss.detach().cpu().item()) * float(x_t.shape[0])
        n += int(x_t.shape[0])
    return {"loss": float(total / max(1, n))}


def train_model(
    *,
    model: nn.Module,
    train: Tuple[np.ndarray, np.ndarray],
    test: Tuple[np.ndarray, np.ndarray],
    dt: float,
    output_dir: Path,
    cfg: TrainConfig,
    device: str = "auto",
    seed: int = 0,
    curriculum: Optional[CurriculumHeytingLosses] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> TrainOut:
    device_t = auto_device(device)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)

    dl_tr, dl_te = _make_loaders(train, test, batch_size=cfg.batch_size)
    model.to(device_t)
    if curriculum is not None:
        curriculum.to(device_t)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    best_val = float("inf")
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    metrics_path = output_dir / "metrics.jsonl"

    meta = {
        "created": _now_id(),
        "dt": float(dt),
        "seed": int(seed),
        "device": str(device_t),
        "train_cfg": cfg.__dict__,
        "extra": extra_meta or {},
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))

    with metrics_path.open("w", encoding="utf-8") as f:
        for epoch in range(int(cfg.epochs)):
            model.train()
            epoch_loss = 0.0
            epoch_n = 0

            for batch in dl_tr:
                x_t, x_t1 = _batch_to_device(batch, device_t)
                opt.zero_grad(set_to_none=True)
                out = model(x_t, x_t1)
                loss, _ = _model_loss(out, x_t, x_t1, recon_weight=cfg.recon_weight, pred_weight=cfg.pred_weight)

                if curriculum is not None:
                    h = curriculum.compute(epoch=epoch, z=out["z"], z_raw=out["z_raw"], z_nuc=out.get("z_nuc", out["z_raw"]))
                    loss = loss + h.total

                loss.backward()
                if cfg.grad_clip and cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.grad_clip))
                opt.step()

                epoch_loss += float(loss.detach().cpu().item()) * float(x_t.shape[0])
                epoch_n += int(x_t.shape[0])

            train_loss = float(epoch_loss / max(1, epoch_n))
            val = evaluate(model, dl_te, device=device_t, cfg=cfg, curriculum=curriculum, epoch=epoch)

            row: Dict[str, Any] = {"epoch": int(epoch), "train_loss": float(train_loss), "val_loss": float(val["loss"])}
            if (epoch % int(cfg.log_every)) == 0:
                f.write(json.dumps(row) + "\n")
                f.flush()

            ckpt = {
                "model_state": model.state_dict(),
                "meta": meta,
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val["loss"]),
            }
            torch.save(ckpt, last_path)
            if float(val["loss"]) < best_val:
                best_val = float(val["loss"])
                torch.save(ckpt, best_path)

    return TrainOut(output_dir=output_dir, best_path=best_path, last_path=last_path, best_val_loss=float(best_val))

