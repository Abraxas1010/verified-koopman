# Reproducing Results (Local)

This release bundle is structured for local reproduction (no cloud CI).

## 1) Verify Lean proofs

```bash
./scripts/verify_lean.sh
```

## 2) Install Python package

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 3) Run smoke reproductions

These are *quick runs* to validate the pipeline end-to-end.

```bash
python scripts/run_experiments.py --experiment capability --epochs 20
python scripts/run_experiments.py --experiment lyapunov --epochs 20
python scripts/run_experiments.py --experiment heyting --epochs 20
python scripts/run_safedmd_experiment.py --system toy2d --epochs 50 --device cpu --latent-dim 4
```

Outputs are written under `outputs/experiments/`.

## 4) Full training runs

Use `configs/*.yaml` and the training script:

```bash
python scripts/train_nba.py --system vdp --model nba --epochs 300
python scripts/train_nba.py --system vdp --model e2e --epochs 300
```

## 5) Heyting analysis

```bash
python scripts/analyze_heyting.py --system vdp --checkpoint outputs/vdp_e2e_0/best.pt
```
