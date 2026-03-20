import HeytingLean.Physics.KoopmanGFF.EDMDRatioSpecialization
import HeytingLean.Physics.KoopmanGFF.LatticeMassGap
import Mathlib.MeasureTheory.Constructions.Pi

/-!
# Lattice OU Fourier Innovation Model

This module closes the remaining mathematical trust gap in the exact-Fourier
EDMD lane at the fixed-regressor boundary. It defines the one-mode stationary
lattice OU innovation law used by the runtime and proves that its split real
coordinates are independent centered Gaussians with the exact variance dictated
by the lattice mass operator.
-/

open MeasureTheory ProbabilityTheory

namespace HeytingLean.Physics.KoopmanGFF

noncomputable section

/-- The exact one-step Fourier multiplier for a lattice OU mode. -/
def latticeOUFourierMultiplier (d N : ℕ) [NeZero N]
    (spacing mass dt : ℝ) (mode : ℕ) : ℝ :=
  Real.exp (-dt * latticeMassEigenvalue d N spacing mass mode)

/-- The exact one-step innovation variance for a lattice OU mode. This matches
the runtime formula `covariance * (1 - decay^2)`. -/
def latticeOUInnovationVariance (d N : ℕ) [NeZero N]
    (spacing mass dt : ℝ) (mode : ℕ) : ℝ :=
  (1 - Real.exp (-2 * dt * latticeMassEigenvalue d N spacing mass mode)) /
    latticeMassEigenvalue d N spacing mass mode

/-- Split real/imaginary innovation coordinates each carry half of the complex
mode variance. -/
def latticeOUInnovationCoordVariance (d N : ℕ) [NeZero N]
    (spacing mass dt : ℝ) (mode : ℕ) : ℝ :=
  latticeOUInnovationVariance d N spacing mass dt mode / 2

/-- The lattice OU mode innovation variance is strictly positive under the
standard physical positivity assumptions. -/
theorem latticeOUInnovationVariance_pos (d N : ℕ) [NeZero N]
    (spacing mass dt : ℝ) (mode : ℕ)
    (hspacing : 0 < spacing) (hmass : 0 < mass) (hdt : 0 < dt) :
    0 < latticeOUInnovationVariance d N spacing mass dt mode := by
  have hEig : 0 < latticeMassEigenvalue d N spacing mass mode :=
    latticeMassEigenvalue_pos d N spacing mass hspacing hmass mode
  have hneg : -2 * dt * latticeMassEigenvalue d N spacing mass mode < 0 := by
    nlinarith
  have hexp_lt : Real.exp (-2 * dt * latticeMassEigenvalue d N spacing mass mode) < 1 := by
    simpa using (Real.exp_lt_one_iff.mpr hneg)
  have hone_minus : 0 < 1 - Real.exp (-2 * dt * latticeMassEigenvalue d N spacing mass mode) := by
    linarith
  exact div_pos hone_minus hEig

/-- Each split real/imaginary innovation coordinate has strictly positive
variance. -/
theorem latticeOUInnovationCoordVariance_pos (d N : ℕ) [NeZero N]
    (spacing mass dt : ℝ) (mode : ℕ)
    (hspacing : 0 < spacing) (hmass : 0 < mass) (hdt : 0 < dt) :
    0 < latticeOUInnovationCoordVariance d N spacing mass dt mode := by
  unfold latticeOUInnovationCoordVariance
  have h := latticeOUInnovationVariance_pos d N spacing mass dt mode hspacing hmass hdt
  linarith

/-- The `NNReal` variance parameter used by the split-coordinate Gaussian law. -/
def latticeOUInnovationCoordVarianceNNReal (d N : ℕ) [NeZero N]
    (spacing mass dt : ℝ) (mode : ℕ)
    (hspacing : 0 < spacing) (hmass : 0 < mass) (hdt : 0 < dt) : NNReal :=
  ⟨latticeOUInnovationCoordVariance d N spacing mass dt mode,
    le_of_lt (latticeOUInnovationCoordVariance_pos d N spacing mass dt mode hspacing hmass hdt)⟩

/-- Sample space for the split real/imaginary innovation coordinates of a
single retained Fourier mode over `n` sample pairs. -/
abbrev LatticeOUInnovationSample (n : ℕ) := Fin n ⊕ Fin n → ℝ

/-- The exact finite-product Gaussian law for the split innovation coordinates
of a single lattice OU Fourier mode. -/
def latticeOUInnovationMeasure (d N : ℕ) [NeZero N]
    (spacing mass dt : ℝ) (mode n : ℕ)
    (hspacing : 0 < spacing) (hmass : 0 < mass) (hdt : 0 < dt) :
    Measure (LatticeOUInnovationSample n) :=
  Measure.pi (fun _ : Fin n ⊕ Fin n =>
    gaussianReal (0 : ℝ)
      (latticeOUInnovationCoordVarianceNNReal d N spacing mass dt mode hspacing hmass hdt))

instance latticeOUInnovationMeasure_instIsProbabilityMeasure (d N : ℕ) [NeZero N]
    (spacing mass dt : ℝ) (mode n : ℕ)
    (hspacing : 0 < spacing) (hmass : 0 < mass) (hdt : 0 < dt) :
    IsProbabilityMeasure (latticeOUInnovationMeasure d N spacing mass dt mode n hspacing hmass hdt) := by
  unfold latticeOUInnovationMeasure
  infer_instance

/-- Coordinate projection on the split innovation sample space. -/
def latticeOUInnovationCoordinate {n : ℕ}
    (j : Fin n ⊕ Fin n) (ω : LatticeOUInnovationSample n) : ℝ :=
  ω j

/-- Each split innovation coordinate has the exact centered Gaussian law
dictated by the lattice OU mode variance. -/
theorem latticeOUInnovationCoordinate_map_eq_gaussianReal (d N : ℕ) [NeZero N]
    (spacing mass dt : ℝ) (mode n : ℕ)
    (hspacing : 0 < spacing) (hmass : 0 < mass) (hdt : 0 < dt)
    (j : Fin n ⊕ Fin n) :
    (latticeOUInnovationMeasure d N spacing mass dt mode n hspacing hmass hdt).map
        (latticeOUInnovationCoordinate (n := n) j) =
      gaussianReal (0 : ℝ)
        (latticeOUInnovationCoordVarianceNNReal d N spacing mass dt mode hspacing hmass hdt) := by
  change (latticeOUInnovationMeasure d N spacing mass dt mode n hspacing hmass hdt).map
      (Function.eval j) =
    gaussianReal (0 : ℝ)
      (latticeOUInnovationCoordVarianceNNReal d N spacing mass dt mode hspacing hmass hdt)
  simpa [latticeOUInnovationMeasure, latticeOUInnovationCoordinate] using
    ((MeasureTheory.measurePreserving_eval
      (μ := fun _ : Fin n ⊕ Fin n =>
        gaussianReal (0 : ℝ)
          (latticeOUInnovationCoordVarianceNNReal d N spacing mass dt mode hspacing hmass hdt))
      j).map_eq)

/-- The split innovation coordinates are mutually independent under the exact
finite-product lattice OU innovation law. -/
theorem latticeOUInnovationCoordinate_iIndepFun (d N : ℕ) [NeZero N]
    (spacing mass dt : ℝ) (mode n : ℕ)
    (hspacing : 0 < spacing) (hmass : 0 < mass) (hdt : 0 < dt) :
    iIndepFun (fun j => latticeOUInnovationCoordinate (n := n) j)
      (latticeOUInnovationMeasure d N spacing mass dt mode n hspacing hmass hdt) := by
  classical
  refine (iIndepFun_iff_map_fun_eq_pi_map
    (μ := latticeOUInnovationMeasure d N spacing mass dt mode n hspacing hmass hdt)
    (f := fun j => latticeOUInnovationCoordinate (n := n) j)
    (hf := fun j => (measurable_pi_apply j).aemeasurable)).2 ?_
  rw [show (fun ω j => latticeOUInnovationCoordinate (n := n) j ω) = id by rfl, Measure.map_id]
  unfold latticeOUInnovationMeasure
  congr
  funext j
  simpa using (latticeOUInnovationCoordinate_map_eq_gaussianReal
    d N spacing mass dt mode n hspacing hmass hdt j).symm

/-- The exact lattice OU Fourier-mode step used by the runtime for one retained
mode and `n` sample pairs. -/
def latticeOUFourierNext {n : ℕ} (d N : ℕ) [NeZero N]
    (spacing mass dt : ℝ) (mode : ℕ) (z : Fin n → ℂ)
    (ω : LatticeOUInnovationSample n) (i : Fin n) : ℂ :=
  (latticeOUFourierMultiplier d N spacing mass dt mode : ℂ) * z i +
    splitComplexInnovation (fun j ω => latticeOUInnovationCoordinate (n := n) j ω) ω i

/-- The exact OU step formula is definitional on the Lean model. -/
theorem latticeOUFourierNext_eq {n : ℕ} (d N : ℕ) [NeZero N]
    (spacing mass dt : ℝ) (mode : ℕ) (z : Fin n → ℂ)
    (ω : LatticeOUInnovationSample n) (i : Fin n) :
    latticeOUFourierNext d N spacing mass dt mode z ω i =
      (latticeOUFourierMultiplier d N spacing mass dt mode : ℂ) * z i +
        splitComplexInnovation (fun j ω => latticeOUInnovationCoordinate (n := n) j ω) ω i := by
  rfl

/-- The exact lattice OU Fourier innovation law discharges the conservative
fixed-regressor EDMD theorem for a single retained mode. -/
theorem latticeOUFourierModeEstimator_error_tail_le_half_delta
    {n : ℕ} (d N : ℕ) [NeZero N]
    (spacing mass dt : ℝ) (mode : ℕ)
    (hspacing : 0 < spacing) (hmass : 0 < mass) (hdt : 0 < dt)
    (z : Fin n → ℂ) {effectiveDenom delta : ℝ}
    (hEff : 0 < effectiveDenom)
    (hEffLe : effectiveDenom ≤ ∑ i, ‖z i‖ ^ 2)
    (hδ : 0 < delta ∧ delta < 8) :
    let noiseVar := latticeOUInnovationVariance d N spacing mass dt mode
    (latticeOUInnovationMeasure d N spacing mass dt mode n hspacing hmass hdt).real
      {ω |
        edmdConservativeRuntimeRadius noiseVar effectiveDenom delta ≤
          ‖edmdEstimator z (latticeOUFourierNext d N spacing mass dt mode z ω) -
            (latticeOUFourierMultiplier d N spacing mass dt mode : ℂ)‖} ≤
      delta / 2 := by
  dsimp
  have hNoise : 0 < latticeOUInnovationVariance d N spacing mass dt mode :=
    latticeOUInnovationVariance_pos d N spacing mass dt mode hspacing hmass hdt
  have henergy : 0 < ∑ i, ‖z i‖ ^ 2 := lt_of_lt_of_le hEff hEffLe
  exact edmdEstimator_error_tail_at_conservativeRuntimeRadius_of_splitIndependentCenteredGaussian
    (μ := latticeOUInnovationMeasure d N spacing mass dt mode n hspacing hmass hdt)
    (ξ := fun j => latticeOUInnovationCoordinate (n := n) j)
    (z := z)
    (zNext := latticeOUFourierNext d N spacing mass dt mode z)
    (a := (latticeOUFourierMultiplier d N spacing mass dt mode : ℂ))
    (noiseVar := latticeOUInnovationVariance d N spacing mass dt mode)
    (effectiveDenom := effectiveDenom)
    (delta := delta)
    (hstep := by
      intro ω i
      exact latticeOUFourierNext_eq d N spacing mass dt mode z ω i)
    (h_indep := latticeOUInnovationCoordinate_iIndepFun
      d N spacing mass dt mode n hspacing hmass hdt)
    (hNoise := hNoise)
    (h_gauss := by
      intro j
      simpa [latticeOUInnovationCoordVarianceNNReal, latticeOUInnovationCoordVariance]
        using latticeOUInnovationCoordinate_map_eq_gaussianReal
          d N spacing mass dt mode n hspacing hmass hdt j)
    (henergy := henergy)
    (hEff := hEff)
    (hEffLe := hEffLe)
    (hδ := hδ)

end

end HeytingLean.Physics.KoopmanGFF
