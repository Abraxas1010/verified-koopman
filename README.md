<img src="assets/Apoth3osis.webp" alt="Apoth3osis — Formal Mathematics and Verified Software" width="140"/>

<sub><strong>Our tech stack is ontological:</strong><br>
<strong>Hardware — Physics</strong><br>
<strong>Software — Mathematics</strong><br><br>
<strong>Our engineering workflow is simple:</strong> discover, build, grow, learn & teach</sub>

---

<sub>
<strong>Acknowledgment</strong><br>
We humbly thank the collective intelligence of humanity for providing the technology and culture we cherish. We do our best to properly reference the authors of the works utilized herein, though we may occasionally fall short. Our formalization acts as a reciprocal validation—confirming the structural integrity of their original insights while securing the foundation upon which we build. In truth, all creative work is derivative; we stand on the shoulders of those who came before, and our contributions are simply the next link in an unbroken chain of human ingenuity.
</sub>

---

[![License: Apoth3osis License Stack v1](https://img.shields.io/badge/License-Apoth3osis%20License%20Stack%20v1-blue.svg)](LICENSE.md)

# Verified Koopman — Proof-Carrying Nucleus-Bottleneck Koopman Autoencoders with Lean 4 Verification and SafEDMD Controller Synthesis

A minimal, publication-oriented implementation of **proof-carrying** Koopman autoencoders for data-driven dynamical systems analysis, featuring nucleus-bottleneck architecture with sorry-free Lean 4 proofs, SafEDMD-inspired residual-covariance error bounds, and SDP-based Lyapunov controller synthesis.

## What Is This?

This repository provides a complete pipeline for learning Koopman operator approximations from trajectory data with formal verification guarantees. The nucleus-bottleneck autoencoder (NBA) constrains the latent space via a ReLU nucleus — an idempotent, meet-preserving projection — whose algebraic properties are proved in Lean 4 against Mathlib. SafEDMD-inspired error bounds and SDP controller synthesis close the loop from learned dynamics to certified stability.

## Certified EDMD for Lattice GFF (NEW)

The [`edmd-lattice-gff/`](edmd-lattice-gff/) directory contains a **2,251-line Lean 4 formalization** proving certified error bounds for EDMD estimation on lattice Gaussian Free Field Fourier modes. This work builds on [Douglas, Hoback, Mei & Nissim (arXiv:2603.15770)](https://arxiv.org/abs/2603.15770), who proved all five Osterwalder-Schrader axioms for the massive GFF in d=4.

**88 theorems, zero sorry, zero axioms.** Key results:
- Sub-Gaussian concentration chain for weighted innovation sums
- EDMD quotient identity and conservative runtime radius
- Lattice OU Fourier innovation model with product-measure independence
- Certificate surface bridging formal proofs to runtime diagnostics
- Continuum bridge through restricted Koopman correlations with exponential decay

See [`edmd-lattice-gff/README.md`](edmd-lattice-gff/README.md) for the full theorem inventory and build instructions.

## What's Verified (Lean 4)

All proofs build sorry-free against Mathlib. Run verification locally:

```bash
bash scripts/verify_lean.sh
# or equivalently:
cd lean && lake build
```

### Verified Theorems

| File | What's Proved |
|------|--------------|
| `NucleusReLU.lean` | ReLU is a nucleus: inflationary, idempotent, meet-preserving on `Fin n -> R` |
| `NucleusThreshold.lean` | Threshold operator is a nucleus with analogous properties |
| `HeytingOps.lean` | Bounded Heyting algebra operations (meet, join, implication) on real-valued vectors |
| `ParametricHeyting.lean` | Parametric Heyting algebra with learnable bounds preserves lattice laws |
| `SemanticClosure/MR/` | Closure operator properties, inverse evaluation, and basic semantic closure theory |

These theorems guarantee that the neural architecture's bottleneck layer preserves the algebraic structure required for sound Koopman approximation.

## Quick Start (Python)

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Train a Nucleus-Bottleneck Model

```bash
python scripts/train_nba.py --system toy2d --model nba --epochs 50 --device cpu
```

### Run Heyting Analysis

```bash
python scripts/train_nba.py --system toy2d --model e2e --epochs 50 --device cpu
python scripts/analyze_heyting.py --system toy2d --checkpoint outputs/toy2d_e2e_0/best.pt
```

### Run SafEDMD Comparison and Controller Synthesis

```bash
python scripts/run_safedmd_experiment.py --system toy2d --epochs 50 --device cpu --latent-dim 4
```

This trains both a nucleus-bottleneck autoencoder and a standard baseline on the same data, computes SafEDMD-inspired residual-covariance error bounds for each, restricts the NBA bound to the active dictionary (dimensions not killed by the ReLU nucleus), synthesizes a Lyapunov controller via SDP, and reports the improvement factor with full dimension context.

## Supported Systems

Pre-configured dynamical systems in `configs/`:

| System | Description | Config |
|--------|-------------|--------|
| `toy2d` | 2D linear rotation | `configs/toy2d.yaml` |
| `vdp` | Van der Pol oscillator | `configs/vdp.yaml` |
| `duffing` | Duffing oscillator | `configs/duffing.yaml` |
| `lorenz` | Lorenz attractor | `configs/lorenz.yaml` |

## Architecture

```
Input x(t) --> [Encoder] --> z(t) --> [ReLU Nucleus] --> z'(t) --> [StableGenerator] --> z'(t+1)
                                                                         |
                                                                    [Decoder] --> x̂(t+1)
```

- **Encoder/Decoder**: Parametric maps (MLPs) between state space and latent space
- **ReLU Nucleus**: Idempotent, meet-preserving projection (algebraic properties verified in Lean)
- **StableGenerator**: `G = -(A^T A) + (B - B^T)` — Hurwitz by construction, guaranteeing stable learned dynamics

## Verification Tiers

| Tier | Components | Status |
|------|-----------|--------|
| Bronze | Lean proofs + Monte Carlo sampling | Available |
| Silver | Lean proofs + CROWN interval bounds | Available |
| Gold | Lean proofs + dReal + Lyapunov certificate | Available |
| Platinum | Lean proofs + CROWN + SafEDMD SDP controller | Available |

See `configs/cab_platinum_template.json` for the Platinum tier template.

## Reproducing Results

See [`docs/reproducing_paper.md`](docs/reproducing_paper.md) for full reproduction instructions.

## Project Structure

```
verified-koopman/
  configs/          # YAML configs for each dynamical system + CAB templates
  docs/             # Installation, Lean verification, and reproduction guides
  lean/             # Lean 4 + Mathlib project (VerifiedKoopman library)
  notebooks/        # Quickstart Jupyter notebook
  scripts/          # Training, analysis, experiment, and verification scripts
  src/              # Python package (verified_koopman)
    analysis/       #   Heyting analysis + Lyapunov verification
    data/           #   Synthetic trajectory generation
    losses/         #   Heyting lattice losses + curriculum schedules
    models/         #   Koopman AE, nucleus-bottleneck, learnable Heyting
    utils/          #   Training utilities
    verification/   #   dReal verifier, Lean gate, SafEDMD (error bounds + controller)
  tests/            # Unit tests (nucleus, heyting, safedmd, training)
```

## License

[Apoth3osis License Stack v1](LICENSE.md)
