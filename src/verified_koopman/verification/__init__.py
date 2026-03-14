from __future__ import annotations

from pathlib import Path

__all__ = [
    "EDMDErrorBound",
    "ControllerCertificate",
    "active_dictionary_mask",
    "compute_edmd_matrices",
    "compute_safedmd_error_bound",
    "compute_nucleus_aware_error_bound",
    "synthesize_controller",
    "verify_certificate",
    "run_lean_gate",
]

from verified_koopman.verification.safedmd import (
    ControllerCertificate,
    EDMDErrorBound,
    active_dictionary_mask,
    compute_edmd_matrices,
    compute_nucleus_aware_error_bound,
    compute_safedmd_error_bound,
    synthesize_controller,
    verify_certificate,
)


def run_lean_gate(*, lean_dir: Path) -> None:
    from verified_koopman.verification.lean_gate import run

    run(lean_dir=lean_dir)
