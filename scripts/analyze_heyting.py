#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from verified_koopman.analysis.heyting_analysis import stats_for_traj, to_jsonable
from verified_koopman.data.synth_systems import SynthConfig, generate, system_dt
from verified_koopman.models.nucleus_bottleneck import E2EHeytingNBA


def _load_ckpt(path: Path) -> Dict[str, Any]:
    return torch.load(str(path), map_location="cpu")


def _build_from_ctor(ctor: Dict[str, Any]) -> torch.nn.Module:
    mt = str(ctor.get("model_type", "")).lower().strip()
    state_dim = int(ctor["state_dim"])
    if mt == "e2e":
        ecfg = dict(ctor.get("e2e", {}))
        return E2EHeytingNBA(
            in_dim=state_dim,
            state_dim=state_dim,
            latent_dim=int(ecfg.get("latent_dim", 8)),
            hidden_dim=int(ecfg.get("hidden_dim", 128)),
            depth=int(ecfg.get("depth", 3)),
            dt=float(ecfg.get("dt", 0.02)),
            init_threshold=float(ecfg.get("init_threshold", 0.0)),
            init_lo=float(ecfg.get("init_lo", 0.0)),
            init_hi=float(ecfg.get("init_hi", 2.0)),
            min_gap=float(ecfg.get("min_gap", 0.1)),
            heyting_temperature=float(ecfg.get("heyting_temperature", 0.05)),
        )
    raise ValueError(f"unsupported model_type for analysis: {mt!r}")


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--system", default="toy2d", choices=["toy2d", "vdp", "lorenz", "duffing"])
    ap.add_argument("--n_traj", type=int, default=8)
    ap.add_argument("--time", type=int, default=201)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint).resolve()
    ckpt = _load_ckpt(ckpt_path)
    meta = dict(ckpt.get("meta", {}))
    ctor = dict(meta.get("extra", {}).get("model_ctor", {}))
    model = _build_from_ctor(ctor)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    if not isinstance(model, E2EHeytingNBA):
        raise SystemExit("Heyting analysis expects an E2E checkpoint (train with `scripts/train_nba.py --model e2e`).")

    dt = system_dt(args.system)
    traj, _meta2 = generate(SynthConfig(system=args.system, n_traj=args.n_traj, time=args.time, dt=dt, seed=args.seed))
    x = torch.from_numpy(np.asarray(traj[0])).to(dtype=torch.float32)
    device = next(model.parameters()).device
    x = x.to(device=device)

    z, _z_raw, _z_nuc = model.encode(x)
    z = model.heyting.project(z)
    s = stats_for_traj(model.heyting, z)
    out = {"checkpoint": str(ckpt_path), "stats": to_jsonable(s)}

    out_path = Path(args.output).resolve() if args.output else ckpt_path.parent / "heyting_analysis.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"OK: wrote {out_path}")


if __name__ == "__main__":
    main()
