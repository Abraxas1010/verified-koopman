# Verified Koopman: Proof-Carrying Neural Architecture for Dynamical Systems

This repository contains a minimal, publication-oriented implementation of a **proof-carrying** nucleus-bottleneck Koopman autoencoder and the accompanying Lean 4 verification artifacts.

## What’s Verified (Lean)

Lean project: `lean/`

Core verified files (no `sorryAx` when built):
- `lean/VerifiedKoopman/NucleusReLU.lean`
- `lean/VerifiedKoopman/NucleusThreshold.lean`
- `lean/VerifiedKoopman/HeytingOps.lean`
- `lean/VerifiedKoopman/ParametricHeyting.lean`

## Quick Start (Python)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Train a small nucleus-bottleneck model on a synthetic system:

```bash
python scripts/train_nba.py --system toy2d --model nba --epochs 50 --device cpu
```

Run a Heyting analysis on a saved checkpoint (E2E model):

```bash
python scripts/train_nba.py --system toy2d --model e2e --epochs 50 --device cpu
python scripts/analyze_heyting.py --system toy2d --checkpoint outputs/toy2d_e2e_0/best.pt
```

Run the SafEDMD comparison and controller synthesis:

```bash
python scripts/run_safedmd_experiment.py --system toy2d --epochs 50 --device cpu --latent-dim 4
```

## Verify Lean Proofs

```bash
cd lean
lake build
```

Or via the helper script:

```bash
bash scripts/verify_lean.sh
```

## Reproducing Results

See `docs/reproducing_paper.md`.

## License

MIT (see `LICENSE`).
