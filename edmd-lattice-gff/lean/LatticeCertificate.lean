import HeytingLean.Physics.KoopmanGFF.EDMDRatioSpecialization
import HeytingLean.Physics.KoopmanGFF.LatticeApprox
import HeytingLean.Physics.KoopmanGFF.LatticeOUModel

/-!
# Lattice Koopman Certificate

Historical filename for the lattice-GFF Koopman diagnostic contract.

This file does not parse runtime artifacts. Instead it defines the structure
that the numerical diagnostic script is expected to populate and proves that
the formal lattice mass-gap surface supplies the required covariance tail
budget. It does not certify that a learned or EDMD-estimated gap matches the
formal lower bound.
-/

namespace HeytingLean.Physics.KoopmanGFF

open MeasureTheory ProbabilityTheory

noncomputable section

/-- Runtime payload expected from the numerical diagnostic script. -/
structure NumericalKoopmanDiagnostic where
  latentDim : ℕ
  empiricalGap : ℝ
  theoreticalGapLower : ℝ
  gapTolerance : ℝ
  predictionMSE : ℝ

/-- Proof-carrying diagnostic contract for the lattice Koopman demo. -/
structure LatticeKoopmanDiagnostic (d N : ℕ) [NeZero N] (a mass : ℝ) where
  report : NumericalKoopmanDiagnostic
  hSpacing : 0 < a
  hMass : 0 < mass
  hTheoreticalGapLower : report.theoreticalGapLower = mass ^ 2
  hGapTolerance : 0 ≤ report.gapTolerance
  hPredictionMSE : 0 ≤ report.predictionMSE
  approximationBudget : LatticeApproximationBudget d N a mass

/-- A numerical report plus the formal inverse-mass witness assembles into the
Lean-side diagnostic contract. -/
def latticeKoopmanDiagnosticOfReport (d N : ℕ) [NeZero N]
    (a mass : ℝ) (report : NumericalKoopmanDiagnostic)
    (ha : 0 < a) (hmass : 0 < mass)
    (hGapLower : report.theoreticalGapLower = mass ^ 2)
    (hGapTolerance : 0 ≤ report.gapTolerance)
    (hPredictionMSE : 0 ≤ report.predictionMSE) :
    LatticeKoopmanDiagnostic d N a mass := by
  refine ⟨report, ha, hmass, hGapLower, hGapTolerance, hPredictionMSE, ?_⟩
  exact inverseMassApproximationBudget d N a mass ha hmass

/-- The diagnostic payload records the formal lattice lower bound exactly. -/
theorem LatticeKoopmanDiagnostic.theoreticalGapLower_eq
    {d N : ℕ} [NeZero N] {a mass : ℝ}
    (diag : LatticeKoopmanDiagnostic d N a mass) :
    diag.report.theoreticalGapLower = mass ^ 2 :=
  diag.hTheoreticalGapLower

/-- The diagnostic payload stores a nonnegative user-supplied tolerance. -/
theorem LatticeKoopmanDiagnostic.gapTolerance_nonneg
    {d N : ℕ} [NeZero N] {a mass : ℝ}
    (diag : LatticeKoopmanDiagnostic d N a mass) :
    0 ≤ diag.report.gapTolerance :=
  diag.hGapTolerance

/-- The diagnostic contract carries a uniform modewise covariance tail bound. -/
theorem LatticeKoopmanDiagnostic.modeTailBound
    {d N : ℕ} [NeZero N] {a mass : ℝ}
    (diag : LatticeKoopmanDiagnostic d N a mass) :
    ∀ mode : ℕ,
      |latticeCovarianceSingularValue d N a mass mode| ≤ diag.approximationBudget.tailBound :=
  diag.approximationBudget.hTailBound

/-- Runtime payload expected from the exact-Fourier EDMD diagnostic script. -/
structure NumericalEDMDDiagnostic where
  retainedModes : ℕ
  samplePairs : ℕ
  estimatedGap : ℝ
  certifiedGapLower : ℝ
  edmdGapErrorBound : ℝ
  theoreticalGapLower : ℝ
  delta : ℝ
  structuralGapSlack : ℝ

/-- Lean-side contract for the exact-Fourier EDMD lattice diagnostic. -/
structure LatticeEDMDDiagnostic (d N : ℕ) [NeZero N] (a mass : ℝ) where
  report : NumericalEDMDDiagnostic
  hSpacing : 0 < a
  hMass : 0 < mass
  hTheoreticalGapLower : report.theoreticalGapLower = mass ^ 2
  hCertifiedGapLower : 0 ≤ report.certifiedGapLower
  hErrorBound : 0 ≤ report.edmdGapErrorBound
  hDelta : 0 < report.delta ∧ report.delta < 1
  hStructuralGapSlack : 0 ≤ report.structuralGapSlack
  approximationBudget : LatticeApproximationBudget d N a mass

/-- A numerical EDMD report plus the formal inverse-mass witness assembles into
the Lean-side EDMD diagnostic contract. -/
def latticeEDMDDiagnosticOfReport (d N : ℕ) [NeZero N]
    (a mass : ℝ) (report : NumericalEDMDDiagnostic)
    (ha : 0 < a) (hmass : 0 < mass)
    (hGapLower : report.theoreticalGapLower = mass ^ 2)
    (hCertifiedGapLower : 0 ≤ report.certifiedGapLower)
    (hErrorBound : 0 ≤ report.edmdGapErrorBound)
    (hDelta : 0 < report.delta ∧ report.delta < 1)
    (hStructuralGapSlack : 0 ≤ report.structuralGapSlack) :
    LatticeEDMDDiagnostic d N a mass := by
  refine ⟨report, ha, hmass, hGapLower, hCertifiedGapLower, hErrorBound, hDelta, hStructuralGapSlack, ?_⟩
  exact inverseMassApproximationBudget d N a mass ha hmass

/-- The EDMD diagnostic payload records the formal lattice lower bound exactly. -/
theorem LatticeEDMDDiagnostic.theoreticalGapLower_eq
    {d N : ℕ} [NeZero N] {a mass : ℝ}
    (diag : LatticeEDMDDiagnostic d N a mass) :
    diag.report.theoreticalGapLower = mass ^ 2 :=
  diag.hTheoreticalGapLower

/-- The EDMD diagnostic payload stores a nonnegative one-sided error radius. -/
theorem LatticeEDMDDiagnostic.errorBound_nonneg
    {d N : ℕ} [NeZero N] {a mass : ℝ}
    (diag : LatticeEDMDDiagnostic d N a mass) :
    0 ≤ diag.report.edmdGapErrorBound :=
  diag.hErrorBound

/-- The EDMD diagnostic payload stores nonnegative structural slack metadata. -/
theorem LatticeEDMDDiagnostic.structuralGapSlack_nonneg
    {d N : ℕ} [NeZero N] {a mass : ℝ}
    (diag : LatticeEDMDDiagnostic d N a mass) :
    0 ≤ diag.report.structuralGapSlack :=
  diag.hStructuralGapSlack

/-- The EDMD diagnostic contract carries the same uniform covariance tail bound
as the NBA-facing diagnostic contract. -/
theorem LatticeEDMDDiagnostic.modeTailBound
    {d N : ℕ} [NeZero N] {a mass : ℝ}
    (diag : LatticeEDMDDiagnostic d N a mass) :
    ∀ mode : ℕ,
      |latticeCovarianceSingularValue d N a mass mode| ≤ diag.approximationBudget.tailBound :=
  diag.approximationBudget.hTailBound

/-- Runtime payload expected from the per-mode exact-Fourier EDMD radius lane. -/
structure NumericalEDMDModeDiagnostic where
  mode : ℕ
  multiplierErrorBound : ℝ
  noiseVarianceExact : ℝ
  effectiveDenom : ℝ
  deltaMode : ℝ

/-- Lean-side contract for a single reported mode in the exact-Fourier EDMD
radius diagnostic. This records the runtime radius inputs explicitly and
identifies the reported error bound with the proved square-root formula. -/
structure LatticeEDMDModeDiagnostic where
  report : NumericalEDMDModeDiagnostic
  hMultiplierErrorBound :
    report.multiplierErrorBound =
      edmdConservativeRuntimeRadius
        report.noiseVarianceExact report.effectiveDenom report.deltaMode
  hNoiseVariance : 0 < report.noiseVarianceExact
  hEffectiveDenom : 0 < report.effectiveDenom
  hDeltaMode : 0 < report.deltaMode ∧ report.deltaMode < 8

/-- Assemble the per-mode runtime payload with the exact runtime-radius
equality witness. -/
def latticeEDMDModeDiagnosticOfReport
    (report : NumericalEDMDModeDiagnostic)
    (hMultiplierErrorBound :
      report.multiplierErrorBound =
        edmdConservativeRuntimeRadius
          report.noiseVarianceExact report.effectiveDenom report.deltaMode)
    (hNoiseVariance : 0 < report.noiseVarianceExact)
    (hEffectiveDenom : 0 < report.effectiveDenom)
    (hDeltaMode : 0 < report.deltaMode ∧ report.deltaMode < 8) :
    LatticeEDMDModeDiagnostic := by
  exact ⟨report, hMultiplierErrorBound, hNoiseVariance, hEffectiveDenom, hDeltaMode⟩

/-- The per-mode runtime payload stores the exact conservative square-root
radius. -/
theorem LatticeEDMDModeDiagnostic.multiplierErrorBound_eq_conservativeRuntimeRadius
    (diag : LatticeEDMDModeDiagnostic) :
    diag.report.multiplierErrorBound =
      edmdConservativeRuntimeRadius diag.report.noiseVarianceExact
        diag.report.effectiveDenom diag.report.deltaMode :=
  diag.hMultiplierErrorBound

/-- The per-mode exact-Fourier EDMD radius is justified by the conservative
fixed-regressor split-Gaussian theorem. The remaining trusted interface is that
the runtime's retained Fourier innovation coordinates match these explicit
independence and Gaussian hypotheses. -/
theorem LatticeEDMDModeDiagnostic.error_tail_le_half_delta_of_splitIndependentCenteredGaussian
    {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {n : ℕ} {a : ℂ}
    (diag : LatticeEDMDModeDiagnostic)
    (ξ : Fin n ⊕ Fin n → Ω → ℝ) (z : Fin n → ℂ) {zNext : Ω → Fin n → ℂ}
    (hProb : IsProbabilityMeasure μ)
    (hstep : ∀ ω i, zNext ω i = a * z i + splitComplexInnovation ξ ω i)
    (h_indep : iIndepFun ξ μ)
    (h_gauss :
      ∀ j : Fin n ⊕ Fin n,
        μ.map (ξ j) =
          gaussianReal (0 : ℝ)
            ((⟨diag.report.noiseVarianceExact / 2, by
                have h := diag.hNoiseVariance
                nlinarith⟩ : NNReal)))
    (henergy : 0 < ∑ i, ‖z i‖ ^ 2)
    (hEffLe : diag.report.effectiveDenom ≤ ∑ i, ‖z i‖ ^ 2) :
    μ.real {ω |
      diag.report.multiplierErrorBound ≤ ‖edmdEstimator z (zNext ω) - a‖} ≤
        diag.report.deltaMode / 2 := by
  letI := hProb
  rw [diag.hMultiplierErrorBound]
  exact edmdEstimator_error_tail_at_conservativeRuntimeRadius_of_splitIndependentCenteredGaussian
    (μ := μ)
    (ξ := ξ)
    (z := z)
    (zNext := zNext)
    (noiseVar := diag.report.noiseVarianceExact)
    (effectiveDenom := diag.report.effectiveDenom)
    (delta := diag.report.deltaMode)
    hstep h_indep diag.hNoiseVariance h_gauss henergy diag.hEffectiveDenom hEffLe diag.hDeltaMode

/-- The explicit split-Gaussian hypotheses used by the per-mode certificate
surface are discharged by the exact finite-product lattice OU Fourier innovation
model. This closes the mathematical trust gap left by the generic
fixed-regressor theorem. -/
theorem LatticeEDMDModeDiagnostic.error_tail_le_half_delta_of_latticeOUFourierModel
    {n d N : ℕ} [NeZero N]
    (diag : LatticeEDMDModeDiagnostic)
    {spacing mass dt : ℝ} (mode : ℕ)
    (hSpacing : 0 < spacing) (hMass : 0 < mass) (hDt : 0 < dt)
    (hNoiseExact :
      diag.report.noiseVarianceExact =
        latticeOUInnovationVariance d N spacing mass dt mode)
    (z : Fin n → ℂ)
    (hEffLe : diag.report.effectiveDenom ≤ ∑ i, ‖z i‖ ^ 2) :
    (latticeOUInnovationMeasure d N spacing mass dt mode n hSpacing hMass hDt).real
      {ω |
        diag.report.multiplierErrorBound ≤
          ‖edmdEstimator z (latticeOUFourierNext d N spacing mass dt mode z ω) -
            (latticeOUFourierMultiplier d N spacing mass dt mode : ℂ)‖} ≤
      diag.report.deltaMode / 2 := by
  rw [diag.hMultiplierErrorBound, hNoiseExact]
  exact latticeOUFourierModeEstimator_error_tail_le_half_delta
    d N spacing mass dt mode hSpacing hMass hDt z diag.hEffectiveDenom hEffLe diag.hDeltaMode

/-- Runtime payload expected from the excited-mode exact-Fourier EDMD script. -/
structure NumericalExcitedEDMDDiagnostic where
  retainedModes : ℕ
  samplePairs : ℕ
  estimatedGap : ℝ
  certifiedGapLower : ℝ
  edmdGapErrorBound : ℝ
  theoreticalGapLower : ℝ
  targetRate : ℝ
  minExcitedRate : ℝ
  delta : ℝ
  structuralGapSlack : ℝ

/-- Lean-side contract for the excited-mode exact-Fourier EDMD diagnostic. -/
structure LatticeExcitedEDMDDiagnostic (d N : ℕ) [NeZero N] (a mass : ℝ) where
  report : NumericalExcitedEDMDDiagnostic
  hSpacing : 0 < a
  hMass : 0 < mass
  hTheoreticalGapLower : report.theoreticalGapLower = mass ^ 2
  hCertifiedGapLower : 0 ≤ report.certifiedGapLower
  hErrorBound : 0 ≤ report.edmdGapErrorBound
  hTargetRate : 0 < report.targetRate
  hTargetBelowExcited : report.targetRate < report.minExcitedRate
  hDelta : 0 < report.delta ∧ report.delta < 1
  hStructuralGapSlack : 0 < report.structuralGapSlack
  approximationBudget : LatticeApproximationBudget d N a mass

/-- Assemble the excited-mode runtime payload with the formal inverse-mass tail
witness. -/
def latticeExcitedEDMDDiagnosticOfReport (d N : ℕ) [NeZero N]
    (a mass : ℝ) (report : NumericalExcitedEDMDDiagnostic)
    (ha : 0 < a) (hmass : 0 < mass)
    (hGapLower : report.theoreticalGapLower = mass ^ 2)
    (hCertifiedGapLower : 0 ≤ report.certifiedGapLower)
    (hErrorBound : 0 ≤ report.edmdGapErrorBound)
    (hTargetRate : 0 < report.targetRate)
    (hTargetBelowExcited : report.targetRate < report.minExcitedRate)
    (hDelta : 0 < report.delta ∧ report.delta < 1)
    (hStructuralGapSlack : 0 < report.structuralGapSlack) :
    LatticeExcitedEDMDDiagnostic d N a mass := by
  refine ⟨report, ha, hmass, hGapLower, hCertifiedGapLower, hErrorBound, hTargetRate,
    hTargetBelowExcited, hDelta, hStructuralGapSlack, ?_⟩
  exact inverseMassApproximationBudget d N a mass ha hmass

/-- The excited-mode diagnostic target is strictly positive. -/
theorem LatticeExcitedEDMDDiagnostic.targetRate_pos
    {d N : ℕ} [NeZero N] {a mass : ℝ}
    (diag : LatticeExcitedEDMDDiagnostic d N a mass) :
    0 < diag.report.targetRate :=
  diag.hTargetRate

/-- The excited-mode target lies strictly below the minimum exact retained rate. -/
theorem LatticeExcitedEDMDDiagnostic.targetBelowExcited
    {d N : ℕ} [NeZero N] {a mass : ℝ}
    (diag : LatticeExcitedEDMDDiagnostic d N a mass) :
    diag.report.targetRate < diag.report.minExcitedRate :=
  diag.hTargetBelowExcited

/-- Any certified excited-mode lower bound that clears the target is nontrivial. -/
theorem LatticeExcitedEDMDDiagnostic.nontrivial_of_target
    {d N : ℕ} [NeZero N] {a mass : ℝ}
    (diag : LatticeExcitedEDMDDiagnostic d N a mass)
    (hcert : diag.report.targetRate ≤ diag.report.certifiedGapLower) :
    0 < diag.report.certifiedGapLower :=
  lt_of_lt_of_le diag.hTargetRate hcert

/-- The excited-mode contract carries the same uniform covariance tail bound. -/
theorem LatticeExcitedEDMDDiagnostic.modeTailBound
    {d N : ℕ} [NeZero N] {a mass : ℝ}
    (diag : LatticeExcitedEDMDDiagnostic d N a mass) :
    ∀ mode : ℕ,
      |latticeCovarianceSingularValue d N a mass mode| ≤ diag.approximationBudget.tailBound :=
  diag.approximationBudget.hTailBound

end

end HeytingLean.Physics.KoopmanGFF
