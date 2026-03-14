from __future__ import annotations

"""
Optional dReal verifier shim.

This repository keeps dReal as an optional dependency; many users will reproduce
the core claims (Lean verification + training + Heyting analysis) without SMT.

If you have dReal available via Docker, you can implement project-specific
verification by exporting a small artifact (network + property) and running dReal.
"""

from dataclasses import dataclass
from pathlib import Path
import subprocess
import re
from typing import Optional


@dataclass(frozen=True)
class DRealResult:
    sat: bool
    stdout: str


def run_dreal_docker(*, smt2_path: Path, timeout_s: int = 60, image: str = "dreal/dreal4") -> DRealResult:
    smt2_path = smt2_path.resolve()
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{smt2_path.parent}:/work",
        image,
        "dreal",
        f"/work/{smt2_path.name}",
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=int(timeout_s), check=False)
    out = p.stdout.decode("utf-8", errors="replace")
    low = out.lower()
    if re.search(r"\bunsat\b", low):
        sat = False
    elif re.search(r"\bdelta-sat\b", low) or re.search(r"\bsat\b", low):
        sat = True
    else:
        sat = False
    return DRealResult(sat=sat, stdout=out)
