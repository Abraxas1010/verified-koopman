# Certified EDMD Estimation for Lattice Gaussian Free Fields

This directory contains a 2,251-line Lean 4 formalization (plus a 440-line Python runtime certifier) proving that **Extended Dynamic Mode Decomposition (EDMD) estimation on lattice GFF Fourier modes has certified, quantified error bounds**.

## Relationship to Douglas et al.

This work builds on the landmark formalization of the Gaussian Free Field by **Douglas, Hoback, Mei, and Nissim** ([arXiv:2603.15770](https://arxiv.org/abs/2603.15770)), who proved all five Osterwalder-Schrader axioms for the massive GFF in d=4 using Lean 4 (~32,000 lines, zero sorry). Their repository: [mrdouglasny/OSforGFF](https://github.com/mrdouglasny/OSforGFF).

We ask a different question: given the GFF, can we formally verify that EDMD estimation on its lattice discretization has certified, quantified error bounds? Where Douglas et al. proved the *measure exists* (OS axioms), we prove that *numerical estimation on that measure's lattice discretization has bounded error* (concentration inequalities, sub-Gaussian theory, EDMD quotient decomposition).

## What's Proved

**88 theorems**, zero sorry, zero admit, zero user-declared axioms. The proof chain:

1. **Sub-Gaussian concentration** — Hoeffding-type tail bounds for weighted sums of independent centered Gaussians
2. **EDMD quotient identity** — the estimation error decomposes as a weighted innovation sum divided by signal energy
3. **Conservative runtime radius** — a formally derived confidence radius from the noise variance and effective denominator
4. **Lattice mass-gap spectrum** — eigenvalue positivity for the lattice Laplacian plus mass operator
5. **Lattice OU innovation model** — the split real/imaginary coordinates are independent centered Gaussians with exact variance
6. **Certificate surface** — diagnostic structures connecting formal guarantees to runtime-checkable assertions
7. **Continuum bridge** — restricted Koopman correlations factorized through exact Gaussian MGF, with exponential decay from the quantitative mass gap

### Key Capstone Theorems

| Theorem | File | Statement |
|---------|------|-----------|
| `latticeOUInnovationCoordinate_map_eq_gaussianReal` | `LatticeOUModel.lean` | Each split coordinate has the exact centered Gaussian law dictated by the lattice OU mode variance |
| `latticeOUInnovationCoordinate_iIndepFun` | `LatticeOUModel.lean` | Split innovation coordinates are mutually independent under the product measure |
| `latticeOUFourierModeEstimator_error_tail_le_half_delta` | `LatticeOUModel.lean` | Capstone: estimation error is bounded by the conservative runtime radius with probability ≥ 1 − δ/2 |
| `error_tail_le_half_delta_of_latticeOUFourierModel` | `LatticeCertificate.lean` | Certificate-layer binding: the formal tail bound discharges at the diagnostic surface |
| `gaussianFreeField_restrictedKoopmanCorrelationDecay` | `GeneratingObservables.lean` | Restricted Koopman correlations decay exponentially via the quantitative mass gap |
| `freeCovariance_exponential_decay` | `OSforGFFQuantitativeGap.lean` | Quantitative exponential decay of the free covariance from the continuum mass gap |
| `gaussianFreeField_satisfies_all_OS_axioms` | `OSforGFFBridge.lean` | All five OS axioms verified (bridge alias from Douglas et al.) |

### Module Inventory

| File | Lines | Theorems | Description |
|------|-------|----------|-------------|
| `EDMDConcentration.lean` | 230 | 11 | Sub-Gaussian/Hoeffding wrappers for weighted sums |
| `EDMDRatioSpecialization.lean` | 734 | 21 | EDMD quotient identity + fixed-regressor theorem |
| `GeneratingObservables.lean` | 274 | 8 | Restricted Koopman correlations (continuum bridge) |
| `LatticeApprox.lean` | 44 | 1 | Lattice approximation definitions |
| `LatticeCertificate.lean` | 321 | 14 | Certificate surface for runtime diagnostics |
| `LatticeMassGap.lean` | 102 | 6 | Lattice mass-gap eigenvalue spectrum |
| `LatticeOUModel.lean` | 203 | 6 | OU innovation model + capstone tail bound |
| `OSforGFFBridge.lean` | 80 | 7 | Qualitative OS axiom bridge aliases |
| `OSforGFFKoopmanBridge.lean` | 109 | 10 | Koopman-facing bridge (MGF factorization, stationarity) |
| `OSforGFFQuantitativeGap.lean` | 67 | 4 | Quantitative mass-gap export |

## Python Runtime Certifier

`python/certify_lattice_gff_edmd.py` (440 lines) bridges the Lean proofs to numerical diagnostics:

```bash
python certify_lattice_gff_edmd.py \
  --lattice-dim 2 --lattice-size 8 --mass 1.0 --spacing 0.5 \
  --dt 0.01 --num-modes 5 --num-samples 1000 --delta 0.05
```

It computes the conservative runtime radius for each retained Fourier mode, runs Monte Carlo validation, and reports whether the empirical error stays within the formally certified bounds.

## Build / Verification

These files are library modules from the [HeytingLean](https://github.com/Abraxas1010) project, building against Lean 4 + Mathlib v4.24.0 with the [OSforGFF](https://github.com/Abraxas1010/OSforGFF) backport (Lean 4.24 compatible). To verify:

```bash
# In the HeytingLean project root:
cd lean && lake build HeytingLean.Physics.KoopmanGFF.LatticeOUModel
```

## Hostile Audit Trail

This formalization has passed 11 rounds of adversarial audit (automated and manual), checking for:
- Vacuous theorems (proofs that are `exact h`, `rfl`, or `linarith` on trivially-zero hypotheses)
- Definitional tautologies (theorem restates a definition with no mathematical content)
- Name-content drift (theorem name promises more than the proof delivers)
- Boundary gaps (inequality proved in one direction, narrated as equality)

## References

- Douglas, M. R., Hoback, S., Mei, A., & Nissim, R. (2026). *Formalization of QFT*. arXiv:2603.15770. [mrdouglasny/OSforGFF](https://github.com/mrdouglasny/OSforGFF)
- Strässer, R., Berberich, J., & Allgöwer, F. (2024). *Data-Driven Control with Inherent Lyapunov Stability*. arXiv:2402.03145.
- The Mathlib Community. *Mathlib4*. https://github.com/leanprover-community/mathlib4

## License

[Apoth3osis License Stack v1](../LICENSE.md)
