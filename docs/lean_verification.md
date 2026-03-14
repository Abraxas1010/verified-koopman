# Lean Verification

## Build + axiom audit (recommended)

From the release root:

```bash
bash scripts/verify_lean.sh
```

This runs:
- `lake build` in `lean/`
- an axiom audit via `#print axioms ...` and fails if `sorryAx` appears

## Manual build

```bash
cd lean
lake update
lake build
```
