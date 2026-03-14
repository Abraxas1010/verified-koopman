# Installation

## Python

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

GPU (optional):
- Install a CUDA-enabled PyTorch matching your CUDA driver/toolkit.
- This repository does not vendor CUDA binaries.

## Lean 4

This release pins a Lean toolchain in `lean/lean-toolchain` and a Mathlib revision in `lean/lakefile.lean`.

```bash
cd lean
lake update
lake build
```

