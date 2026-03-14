from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


AXIOM_SNIPPET = """\
import VerifiedKoopman.NucleusReLU
import VerifiedKoopman.NucleusThreshold
import VerifiedKoopman.HeytingOps
import VerifiedKoopman.ParametricHeyting

#print axioms VerifiedKoopman.reluNucleus_idempotent
#print axioms VerifiedKoopman.reluNucleus_le_apply
#print axioms VerifiedKoopman.reluNucleus_map_inf

#print axioms VerifiedKoopman.thresholdNucleus_idempotent
#print axioms VerifiedKoopman.thresholdNucleus_le_apply
#print axioms VerifiedKoopman.thresholdNucleus_map_inf
#print axioms VerifiedKoopman.thresholdNucleus_fixed_iff

#print axioms VerifiedKoopman.BVec.himp_adjoint
#print axioms VerifiedKoopman.PBVec.himp_adjoint
#print axioms VerifiedKoopman.PBVec.not_not_ne
"""


def _scan_for_sorry(*, lean_dir: Path) -> None:
    root = (lean_dir / "VerifiedKoopman").resolve()
    if not root.exists():
        raise SystemExit(f"FAIL: expected Lean sources at {root}")

    pat = re.compile(r"\bsorry\b")
    hits: list[str] = []
    for path in sorted(root.rglob("*.lean")):
        try:
            txt = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            txt = path.read_text(encoding="utf-8", errors="replace")
        if pat.search(txt):
            rel = path.relative_to(lean_dir)
            hits.append(str(rel))
    if hits:
        joined = "\n".join(hits)
        raise SystemExit(f"FAIL: found `sorry` tokens in Lean sources:\n{joined}")


def run(*, lean_dir: Path) -> None:
    lean_dir = lean_dir.resolve()
    lake = "lake"

    _scan_for_sorry(lean_dir=lean_dir)

    subprocess.run([lake, "build"], cwd=str(lean_dir), check=True)

    p = subprocess.run(
        [lake, "env", "lean", "--stdin", "-E", "warning"],
        cwd=str(lean_dir),
        input=AXIOM_SNIPPET.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    out = p.stdout.decode("utf-8", errors="replace")
    if "sorryAx" in out:
        raise SystemExit("FAIL: axiom audit detected sorryAx")
    print("Lean gate OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lean-dir", default="lean")
    args = ap.parse_args()
    run(lean_dir=Path(args.lean_dir))


if __name__ == "__main__":
    main()
