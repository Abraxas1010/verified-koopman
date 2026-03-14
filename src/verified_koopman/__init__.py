"""
Verified Koopman: proof-carrying neural architectures for dynamical systems.

This release bundle contains:
- Lean 4 proofs for nucleus + Heyting operators (see `lean/`)
- A minimal PyTorch implementation for reproducing experiments (see `scripts/`)
"""

from __future__ import annotations

__version__ = "0.1.0"

from verified_koopman.data.synth_systems import SynthConfig, generate, generate_system_data
from verified_koopman.models.koopman_ae import KoopmanAE, KoopmanAEConfig
from verified_koopman.models.nucleus_bottleneck import (
    E2EHeytingNBA,
    KoopmanNBA,
    KoopmanNBAConfig,
    NucleusBottleneckAE,
)

__all__ = [
    "__version__",
    "SynthConfig",
    "generate",
    "generate_system_data",
    "KoopmanAE",
    "KoopmanAEConfig",
    "KoopmanNBA",
    "KoopmanNBAConfig",
    "NucleusBottleneckAE",
    "E2EHeytingNBA",
]
