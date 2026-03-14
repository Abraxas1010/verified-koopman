# Lean Verification (VerifiedKoopman)

This Lean 4 project contains the mechanically-verified nucleus and bounded Heyting algebra facts used by the Python code.

## Build

```bash
lake build
```

## Key theorems

- `VerifiedKoopman.reluNucleus_*` (ReLU nucleus axioms)
- `VerifiedKoopman.thresholdNucleus_*` (threshold nucleus axioms)
- `VerifiedKoopman.BVec.himp_adjoint` (bounded Heyting adjunction on `[0,top]^n`)
- `VerifiedKoopman.PBVec.himp_adjoint` (parameterized adjunction on `[lo,hi]^n`)

